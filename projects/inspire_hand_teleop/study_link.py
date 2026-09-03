#!/usr/bin/env python3
"""
study_link.py — link between two XRT user-study managers over the internet
(via Tailscale), so robot 1's laptop is the MASTER: it fuses robot 2's crate
detections into the one table-origin world AND drives synchronized teleop /
recording on both.

Designed for a Tailscale (or any VPN) mesh: each machine has a stable, direct,
NAT-traversed address, so a plain TCP socket "just works" — no relay/TURN, no
WebRTC signaling. We send tiny JSON messages, never video:

  master (robot 1, listens)  <───────────────────  peer (robot 2, connects)
     ├─ recv crate `candidate`s  (peer's head-cam crate pose, world frame)
     ├─ recv `status`            (peer teleop/recording state)
     └─ send `command`s          (start/stop teleop; start/stop recording AT epoch T)

Wire format: 4-byte big-endian length + UTF-8 JSON object with a "type" field.

Qt-safety: incoming `command`s are queued (drain on the GUI thread via a QTimer);
`candidate`s go through the on_candidate callback (the tracker's add_remote_candidate
is thread-safe); latest peer `status` is stored under a lock.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("link closed")
        buf += chunk
    return buf


class StudyLink:
    """Bidirectional JSON link. role='master' listens; role='peer' connects.

    on_candidate(dict) is called on the receive thread for each crate candidate.
    Commands land in ``self.commands`` (a Queue) for the GUI thread to drain.
    ``self.peer_status`` holds the latest status dict from the other side.
    """

    def __init__(self, role, host, port=5070, *, listen=None, robot_id=None,
                 on_candidate=None, retry_secs=3.0):
        assert role in ("master", "peer")
        self.role = role
        self.host = host
        self.port = int(port)
        # Whether THIS side listens (vs connects). Decoupled from master/peer so the
        # machine with the reachable address listens regardless of control role.
        # Default keeps old behaviour (master listens) if not specified.
        self._listen = (role == "master") if listen is None else bool(listen)
        self.robot_id = robot_id
        self._on_candidate = on_candidate
        # Seconds to wait between (re)connect / (re)accept attempts. The receive
        # thread keeps retrying forever, so a machine that boots first just waits
        # here until the other machine comes up.
        self._retry_secs = max(0.5, float(retry_secs))

        self.commands: "queue.Queue" = queue.Queue()
        self.peer_status: dict = {}
        self._status_lock = threading.Lock()

        # NTP-style clock-offset estimate (peer_clock - this_clock), timestamped in the
        # RECEIVE THREAD (sub-ms accurate); the min-RTT sample is kept as the estimate.
        self._clock_lock = threading.Lock()
        self._clock = {"offset": None, "rtt": None, "n": 0, "stamp": 0.0}
        self._clock_seq = 0

        self._sock = None
        self._sock_lock = threading.Lock()
        self._srv = None
        self._stop = threading.Event()
        self._thread = None
        self._connected = threading.Event()
        # IP of the OTHER machine on the live connection — from the accepted socket's
        # peer address when we listen, or the address we dialed when we connect. Lets the
        # peer auto-derive the master's rsync/SSH host without a manual entry.
        self._remote_ip = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"link-{self.role}")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._connected.clear()
        for s in (self._sock, self._srv):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def remote_ip(self):
        """IP of the other machine from the live/last connection, or None if never up."""
        return self._remote_ip

    def get_peer_status(self) -> dict:
        with self._status_lock:
            return dict(self.peer_status)

    # ── clock sync (NTP-style over the link) ─────────────────────────────────
    def ping_clock(self):
        """Send one clock-sync probe; the reply updates the offset estimate. Call this a
        few dozen times over ~1 s, then read clock_estimate()."""
        with self._clock_lock:
            self._clock_seq += 1
            seq = self._clock_seq
        return self._send({"type": "clock_ping", "t0": time.time(), "seq": seq})

    def clock_estimate(self) -> dict:
        """Best (min-RTT) estimate: {'offset': peer_clock - this_clock (s), 'rtt': s,
        'n': samples, 'stamp': when}. offset is None until a probe round-trips."""
        with self._clock_lock:
            return dict(self._clock)

    def reset_clock_estimate(self):
        with self._clock_lock:
            self._clock = {"offset": None, "rtt": None, "n": 0, "stamp": 0.0}

    # ── sending ──────────────────────────────────────────────────────────────
    def _send(self, msg: dict):
        if not self._connected.is_set():
            return False
        blob = json.dumps(msg).encode("utf-8")
        with self._sock_lock:
            try:
                self._sock.sendall(struct.pack(">I", len(blob)) + blob)
                return True
            except Exception:
                self._connected.clear()
                return False

    def send_command(self, action, **kw):
        return self._send({"type": "command", "action": action,
                           "from": self.robot_id, **kw})

    def send_candidate(self, p, q, weight, stamp, n_tags=0):
        return self._send({"type": "candidate", "robot_id": self.robot_id,
                           "stamp": stamp, "p": list(map(float, p)),
                           "q": list(map(float, q)), "weight": float(weight),
                           "n_tags": int(n_tags)})

    def send_status(self, **kw):
        return self._send({"type": "status", "robot_id": self.robot_id, **kw})

    # ── receive loop (+ accept/connect with retry) ───────────────────────────
    def _run(self):
        while not self._stop.is_set():
            try:
                self._sock = self._accept() if self._listen else self._connect()
                # No connection yet (peer not up / not listening). Fall through to the
                # retry sleep below and try again — do NOT busy-loop.
                if self._sock is not None:
                    self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    # Long read timeout for DEAD-connection detection only (both sides
                    # heartbeat ~2 Hz, so a live-but-idle link never trips this). Without
                    # this the connect()'s 5 s timeout would drop an idle link every 5 s.
                    self._sock.settimeout(20.0)
                    try:
                        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    except Exception:
                        pass
                    self._connected.set()
                    self._send({"type": "hello", "robot_id": self.robot_id, "role": self.role})
                    while not self._stop.is_set():
                        (length,) = struct.unpack(">I", _recv_exact(self._sock, 4))
                        msg = json.loads(_recv_exact(self._sock, length).decode("utf-8"))
                        self._dispatch(msg)
            except Exception:
                pass
            finally:
                self._connected.clear()
                try:
                    if self._sock is not None:
                        self._sock.close()
                except Exception:
                    pass
                self._sock = None
            # Wait before the next (re)connect / (re)accept attempt. Reached on a
            # failed connect AND on a dropped live link, so retries are always paced
            # (never a tight CPU spin) and keep going until the peer is reachable.
            if not self._stop.is_set():
                time.sleep(self._retry_secs)

    def _accept(self):
        if self._srv is None:
            self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._srv.bind((self.host, self.port))   # host = "0.0.0.0" for master
            self._srv.listen(1)
            self._srv.settimeout(1.0)
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
                self._remote_ip = addr[0] if addr else None
                return conn
            except socket.timeout:
                continue
            except OSError:
                return None
        return None

    def _connect(self):
        try:
            sock = socket.create_connection((self.host, self.port), timeout=5.0)
            try:
                self._remote_ip = sock.getpeername()[0]
            except Exception:
                self._remote_ip = self.host
            return sock
        except Exception:
            return None

    def _dispatch(self, msg):
        t = msg.get("type")
        if t == "command":
            self.commands.put(msg)
        elif t == "candidate":
            if self._on_candidate is not None:
                try:
                    self._on_candidate(msg)
                except Exception:
                    pass
        elif t == "status":
            with self._status_lock:
                self.peer_status = msg
        elif t == "clock_ping":
            # Reply immediately, stamped here in the receive thread for an accurate offset.
            self._send({"type": "clock_pong", "t0": msg.get("t0"),
                        "t1": time.time(), "seq": msg.get("seq")})
        elif t == "clock_pong":
            t3 = time.time()
            t0, t1 = msg.get("t0"), msg.get("t1")
            if t0 is not None and t1 is not None:
                rtt = t3 - t0
                offset = t1 - (t0 + t3) / 2.0        # peer_clock - this_clock
                with self._clock_lock:
                    # keep the estimate from the SMALLEST round-trip (least jitter)
                    if self._clock["rtt"] is None or rtt < self._clock["rtt"]:
                        self._clock = {"offset": offset, "rtt": rtt,
                                       "n": self._clock["n"] + 1, "stamp": t3}
                    else:
                        self._clock["n"] += 1
        # "hello" is informational
