"""Run the WebRTC teleop server in its OWN process, fronted by a drop-in proxy.

WHY: the aiortc event loop is I/O-bound. When it shares a process with the manager's
CPU-heavy threads (60 Hz IK + SEW filter, bone retargeting, CSV formatting, Qt), the GIL
starves the loop under load — it answers its ICE consent-freshness keepalives late, the
measured RTT balloons to hundreds of ms, and the peer connection eventually drops. Moving
the loop into a subprocess with its OWN GIL makes it impossible for the manager to starve
it: nothing else in that process holds the GIL.

ARCHITECTURE (mirrors AprilTagTrackerProxy / TranscriberProxy):
  * subprocess (`--serve`) hosts the real XRRTCBodyPoseDevice + WebRTCServer, does ALL the
    heavy per-frame work (deserialize → retarget → CSV), and ships only the finished
    action_dict to the parent.
  * two socketpairs:
      - POSE  (subprocess → parent, hot ~90 Hz): pickled action_dict frames. The parent
        keeps only the latest (drop-newest), so the manager's ControlThread reads the
        freshest pose with no queue backlog.
      - CTRL  (bidirectional, low rate): parent→child commands (unity_cmds / haptics /
        motor_stats / record config / cleanup); child→parent status pushes (~10 Hz:
        is_connected / link_health / rtt / video_latency / unity_cmds_ready) and
        unity_state events.
  * WebRTCServerProxy exposes the EXACT method surface the manager + streamers use, so
    swapping it in is a one-line change at the device construction site.

A slow parent can stall the subprocess's bone-proc / pose-send, but NEVER its event loop —
which is exactly the isolation we want.
"""
import argparse
import os
import pickle
import queue as _queue
import socket
import struct
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Length-prefixed pickle framing (pickle, not JSON — the pose carries numpy arrays)
# ---------------------------------------------------------------------------
def _send_frame(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("<I", len(data)) + data)


def _recv_exactly(sock: socket.socket, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_raw(sock: socket.socket):
    """Read one length-prefixed frame, returning its raw payload bytes (or None on EOF)."""
    head = _recv_exactly(sock, 4)
    if head is None:
        return None
    (length,) = struct.unpack("<I", head)
    return _recv_exactly(sock, length)


def _recv_frame(sock: socket.socket):
    """Read one length-prefixed frame and unpickle it (or None on EOF)."""
    raw = _recv_raw(sock)
    if raw is None:
        return None
    return pickle.loads(raw)


# ===========================================================================
# Subprocess side: host the real device + bridge control/status
# ===========================================================================
def serve(ctrl_fd: int, pose_fd: int) -> None:
    # Rebuild the inherited socket ends from their fds.
    ctrl = socket.socket(fileno=ctrl_fd)
    pose = socket.socket(fileno=pose_fd)

    # Import here (heavy deps) so `--help` etc. stay light.
    from xr_robot_teleop_client import XRRTCBodyPoseDevice

    init = _recv_frame(ctrl)
    if not init or init[0] != "init":
        print("[webrtc-serve] bad/missing init frame", file=sys.stderr)
        return
    cfg = init[1]

    dev = XRRTCBodyPoseDevice(
        env=None,
        record_data=bool(cfg.get("record_data", False)),
        output_dir=cfg.get("output_dir", "./recordings"),
        ndigits=int(cfg.get("ndigits", 3)),
        server_data_channels=cfg.get("server_data_channels", []),
    )

    # Hot pose sink: pickle each action_dict and push it to the parent. Dedicated POSE
    # socket, so it never contends with the CTRL traffic. Blocking sendall is fine — it can
    # only stall THIS (bone-proc) thread, never the event loop.
    _pose_lock = threading.Lock()

    def _pose_sink(action_dict, arrival_t, recv_t, seq):
        try:
            data = pickle.dumps((action_dict, arrival_t, recv_t, seq),
                                protocol=pickle.HIGHEST_PROTOCOL)
            with _pose_lock:
                pose.sendall(struct.pack("<I", len(data)) + data)
        except Exception:
            pass

    dev.pose_sink = _pose_sink

    # CTRL send is single-writer (this main loop) so no lock is needed, but the ctrl-rx
    # thread and this loop share the socket in opposite directions (full-duplex → safe).
    def _ctrl_rx():
        while True:
            frame = _recv_frame(ctrl)
            if frame is None:
                os._exit(0)                     # parent closed → we're done
            typ, payload = frame
            try:
                if typ == "unity_cmd":
                    dev.send_unity_command(payload)
                elif typ == "haptics":
                    dev.send_haptics(payload)
                elif typ == "motor_stats":
                    dev.send_motor_stats(payload)
                elif typ == "set_record":
                    dev.output_dir = payload["output_dir"]
                    dev.record_data = bool(payload["record_data"])
                    dev.started = bool(payload["started"])
                elif typ == "cleanup_recording":
                    dev.cleanup_recording()
            except Exception as exc:            # noqa: BLE001
                print(f"[webrtc-serve] ctrl '{typ}' error: {exc}", file=sys.stderr)

    threading.Thread(target=_ctrl_rx, daemon=True, name="webrtc-ctrl-rx").start()

    # Status + unity_state push loop (~10 Hz). Single writer of CTRL.
    while True:
        time.sleep(0.1)
        try:
            for msg in dev.poll_unity_state():
                _send_frame(ctrl, ("unity_state", msg))
            status = {
                "is_connected": bool(dev.is_connected),
                "link_health": dev.link_health(),
                "network_rtt_ms": dev.network_rtt_ms(),
                "video_latency": dev.video_latency(),
                "unity_cmds_ready": dev.unity_cmds_ready(),
            }
            _send_frame(ctrl, ("status", status))
        except (BrokenPipeError, ConnectionResetError, OSError):
            os._exit(0)                          # parent gone
        except Exception as exc:                 # noqa: BLE001
            print(f"[webrtc-serve] status push error: {exc}", file=sys.stderr)


# ===========================================================================
# Parent side: drop-in proxy for XRRTCBodyPoseDevice
# ===========================================================================
class _ProxyStateFactory:
    """Shim so the haptics / motor-stats streamers' readiness check
    (`device.state_factory.instance is not None`) reflects the real connection."""
    def __init__(self, proxy):
        self._proxy = proxy

    @property
    def instance(self):
        return self._proxy if self._proxy.is_connected else None


class WebRTCServerProxy:
    """Drop-in stand-in for XRRTCBodyPoseDevice that runs the real device in its OWN process
    (``xr_webrtc_proxy.py --serve``). Exposes the exact surface the manager + streamers use:
    get_controller_state / link_health / pose_age_s / network_rtt_ms / video_latency /
    is_connected / send_unity_command / unity_cmds_ready / poll_unity_state / record_data /
    output_dir / started / cleanup_recording / send_haptics / send_motor_stats /
    state_factory."""

    def __init__(self, init_msg: dict, log_fn=None):
        self._init = dict(init_msg)
        self._log = log_fn or (lambda *_a: None)
        self._proc = None
        self._ctrl = None
        self._pose = None

        self._pose_lock = threading.Lock()
        self._latest_pose = None
        self._last_pose_t = 0.0                  # monotonic when the PARENT received the pose
        self._latest_arrival_t = 0.0             # child-side WebRTC packet arrival (monotonic)
        self._latest_recv_t = 0.0                # child-side bone-proc finish (monotonic)
        self._latest_seq = 0                     # child-side pose counter

        self._status_lock = threading.Lock()
        self._status = {"is_connected": False, "link_health": None,
                        "network_rtt_ms": None, "video_latency": None,
                        "unity_cmds_ready": False}

        self._unity_state_q: "_queue.SimpleQueue" = _queue.SimpleQueue()
        self._send_lock = threading.Lock()

        # Recording control mirrored to the child (properties funnel to a set_record frame).
        self._record_data = bool(init_msg.get("record_data", False))
        self._output_dir = init_msg.get("output_dir", "./recordings")
        self._started = False

        self.state_factory = _ProxyStateFactory(self)

    # -- lifecycle --
    def start(self, timeout: float = 20.0):
        ctrl_p, ctrl_c = socket.socketpair()
        pose_p, pose_c = socket.socketpair()
        script = os.path.join(_HERE, "xr_webrtc_proxy.py")
        try:
            self._proc = __import__("subprocess").Popen(
                [sys.executable, script, "--serve",
                 "--ctrl-fd", str(ctrl_c.fileno()),
                 "--pose-fd", str(pose_c.fileno())],
                pass_fds=[ctrl_c.fileno(), pose_c.fileno()], cwd=_HERE)
        finally:
            ctrl_c.close()
            pose_c.close()
        self._ctrl = ctrl_p
        self._pose = pose_p
        _send_frame(self._ctrl, ("init", self._init))
        threading.Thread(target=self._ctrl_rx, daemon=True, name="webrtc-proxy-ctrl").start()
        threading.Thread(target=self._pose_rx, daemon=True, name="webrtc-proxy-pose").start()
        return self

    def stop(self):
        try:
            if self._proc is not None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3.0)
                except Exception:
                    self._proc.kill()
        except Exception:
            pass
        for s in (self._ctrl, self._pose):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass

    # -- rx threads --
    def _pose_rx(self):
        while True:
            try:
                raw = _recv_raw(self._pose)
            except Exception:
                break
            if raw is None:
                break
            try:
                action, arrival_t, recv_t, seq = pickle.loads(raw)
            except Exception:
                continue
            with self._pose_lock:
                self._latest_pose = action
                self._latest_arrival_t = arrival_t
                self._latest_recv_t = recv_t
                self._latest_seq = seq
                self._last_pose_t = time.monotonic()

    def _ctrl_rx(self):
        while True:
            try:
                frame = _recv_frame(self._ctrl)
            except Exception:
                break
            if frame is None:
                break
            typ, payload = frame
            if typ == "status":
                with self._status_lock:
                    self._status.update(payload)
            elif typ == "unity_state":
                self._unity_state_q.put(payload)

    def _send(self, frame):
        with self._send_lock:
            try:
                _send_frame(self._ctrl, frame)
            except Exception:
                pass

    # -- HOT path --
    def get_controller_state(self):
        """Latest action_dict received from the child (drop-newest), or None. Each frame is
        a freshly-unpickled object, so returning it directly is safe (caller only reads)."""
        with self._pose_lock:
            return self._latest_pose

    def pose_age_s(self):
        """Seconds since the parent last RECEIVED a pose frame — the freshness signal the
        watchdog uses. Grows on a subprocess-side freeze AND on any IPC stall."""
        with self._pose_lock:
            if self._last_pose_t <= 0.0:
                return None
            return max(0.0, time.monotonic() - self._last_pose_t)

    # The ControlThread reaches into state_factory.instance (which is THIS proxy) for pose-
    # rate + staleness tracking. Mirror RobosuiteTeleopState's two accessors. Child-side
    # monotonic timestamps are valid here because CLOCK_MONOTONIC is system-wide on Linux.
    def get_pose_seq(self):
        """(seq, child-side bone-proc-finish monotonic time) — matches RobosuiteTeleopState."""
        with self._pose_lock:
            return self._latest_seq, self._latest_recv_t

    def get_arrival_t(self) -> float:
        """Child-side WebRTC packet-arrival monotonic time of the latest pose."""
        with self._pose_lock:
            return self._latest_arrival_t

    # -- status (cached from the child's ~10 Hz push; non-blocking) --
    def link_health(self, min_window_s: float = 3.0):
        with self._status_lock:
            lh = self._status.get("link_health")
            return dict(lh) if lh else None

    def network_rtt_ms(self):
        with self._status_lock:
            return self._status.get("network_rtt_ms")

    def video_latency(self):
        with self._status_lock:
            v = self._status.get("video_latency")
            return dict(v) if v else None

    def unity_cmds_ready(self) -> bool:
        with self._status_lock:
            return bool(self._status.get("unity_cmds_ready"))

    @property
    def is_connected(self) -> bool:
        with self._status_lock:
            return bool(self._status.get("is_connected"))

    def poll_unity_state(self) -> list:
        out = []
        while True:
            try:
                out.append(self._unity_state_q.get_nowait())
            except _queue.Empty:
                break
        return out

    # -- commands to the child --
    def send_unity_command(self, payload: str) -> None:
        self._send(("unity_cmd", payload))

    def send_haptics(self, payload: str) -> None:
        self._send(("haptics", payload))

    def send_motor_stats(self, payload: str) -> None:
        self._send(("motor_stats", payload))

    def cleanup_recording(self) -> None:
        self._send(("cleanup_recording", None))

    # -- recording control (mirrored to the child) --
    def _push_record(self):
        self._send(("set_record", {"record_data": self._record_data,
                                    "output_dir": self._output_dir,
                                    "started": self._started}))

    @property
    def record_data(self):
        return self._record_data

    @record_data.setter
    def record_data(self, v):
        self._record_data = bool(v)
        self._push_record()

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, v):
        self._output_dir = v
        self._push_record()

    @property
    def started(self):
        return self._started

    @started.setter
    def started(self, v):
        self._started = bool(v)
        self._push_record()


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--ctrl-fd", type=int)
    ap.add_argument("--pose-fd", type=int)
    args = ap.parse_args()
    if args.serve:
        serve(args.ctrl_fd, args.pose_fd)


if __name__ == "__main__":
    _main()
