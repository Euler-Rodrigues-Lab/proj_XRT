#!/usr/bin/env python3
"""
head_cam_localize.py — localize a G1's base in the TABLE frame using its own head
RealSense D435 looking at the table tag (id 0).

No manual touch points: the head camera sees the table tag, and the camera's pose
relative to the robot base is known from the URDF + the recorded waist joints:

  d435_joint (FIXED to torso_link, from the URDF):
      xyz = 0.0576235, 0.01753, 0.42987   rpy = 0, 0.8307767, 0   (~47.6° down)

Chain (T_A_B reads "pose of B in A"):
  T_world_base = inv(T_optical_table) · inv( T_pelvis_torso(q_waist)
                                             · T_torso_d435 · T_d435_optical )
where
  T_optical_table       — AprilTag detection of tag 0 in the camera optical frame
  T_pelvis_torso(q)     — FK through waist_yaw/roll/pitch (recorded), via MuJoCo
  T_torso_d435          — the fixed URDF transform above
  T_d435_optical        — d435_link -> color optical frame (see --optical below)

The world frame is the table tag (tag 0). Each robot self-localizes from its own
head camera, so the camera-less robot 2 localizes too — no external rig needed.

USAGE — standalone (also your camera-access walkthrough + sanity check):
  python head_cam_localize.py --list                 # find the head camera's serial
  python head_cam_localize.py --serial <SN>          # live: detect tag 0, show T_world_base
      keys:  o  toggle d435->optical convention (ros <-> identity)
             s  save robot_base_from_headcam.json
             q  quit
  python head_cam_localize.py --serial <SN> --waist 0 0 0 --table-tag-size 0.12

Run `python head_cam_localize.py --selftest` to verify the chain math.

Deps: pip install pupil-apriltags pyrealsense2   (mujoco + scipy + opencv present)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python required (already in the env).")
try:
    from scipy.spatial.transform import Rotation
except ImportError:
    sys.exit("scipy required (already in the env).")
try:
    from pupil_apriltags import Detector
except ImportError:
    sys.exit("pupil-apriltags not found: pip install pupil-apriltags")

try:
    from apriltag_tracker import inv_T, fuse_poses
    from apriltag_probe import draw_detection, silence_native_logs
    from apriltag_crate_calibrate import is_proper_rotation, make_T, detect_poses
except Exception as exc:
    sys.exit(f"Could not import the apriltag helper modules (keep them alongside): {exc}")

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(_HERE, "g1_with_inspire_hand",
                             "g1_29dof_rev_1_0_with_inspire_hand_FTP.xml")

# Fixed d435_link pose in torso_link, straight from the URDF d435_joint.
D435_XYZ = (0.0576235, 0.01753, 0.42987)
D435_RPY = (0.0, 0.8307767239493009, 0.0)

# Standard ROS body(x-fwd,y-left,z-up) -> optical(x-right,y-down,z-fwd) rotation.
_OPTICAL_ROS = np.array([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])

WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]


# ── pure transforms (testable without hardware) ─────────────────────────────

def rpy_to_mat(r, p, y):
    """URDF fixed-axis rpy -> rotation matrix (R = Rz(y) Ry(p) Rx(r))."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T_torso_d435() -> np.ndarray:
    return make_T(rpy_to_mat(*D435_RPY), D435_XYZ)


def T_d435_optical(convention="ros", color_offset=(0.0, 0.0, 0.0)) -> np.ndarray:
    """d435_link -> color optical frame. 'ros' applies the optical rotation
    (d435_link is the camera body); 'identity' assumes d435_link already IS the
    optical frame. color_offset (m, in d435_link) accounts for the body->color
    sensor translation if you measure it (default 0)."""
    R = _OPTICAL_ROS if convention == "ros" else np.eye(3)
    return make_T(R, color_offset)


def compute_T_world_base(T_optical_table, T_pelvis_torso, convention="ros",
                         T_world_table=None, color_offset=(0.0, 0.0, 0.0)):
    """Localize the base (pelvis) in the world(table) frame from one detection."""
    T_world_table = np.eye(4) if T_world_table is None else np.asarray(T_world_table)
    T_pelvis_optical = (T_pelvis_torso @ T_torso_d435()
                        @ T_d435_optical(convention, color_offset))
    T_optical_world = T_optical_table @ inv_T(T_world_table)
    return inv_T(T_optical_world) @ inv_T(T_pelvis_optical)


def upright_score(T_world_base) -> float:
    """Dot of the base z-axis with world +z. ~1 when the robot is upright and the
    table tag's z points up — a quick check that the optical convention is right."""
    return float(np.dot(T_world_base[:3, 2], np.array([0.0, 0.0, 1.0])))


def world_from_table(z_down=False):
    """T_world_table mapping the detected tag-0 frame to a Z-UP world frame.

    If the table tag's z-axis points DOWN into the table (the usual case — verified
    on this rig: optical=ros gives a clean upright=-1), pass z_down=True to flip it
    180° about x so world-z points up. MUST match the crate tracker's setting so
    the robot and crate share one world."""
    if z_down:
        return make_T(np.diag([1.0, -1.0, -1.0]), (0.0, 0.0, 0.0))   # 180° about x
    return np.eye(4)


# ── MuJoCo waist FK ─────────────────────────────────────────────────────────

class HeadCamFK:
    """T_pelvis_torso(q_waist) via MuJoCo forward kinematics (waist_yaw/roll/pitch)."""

    def __init__(self, model_path=DEFAULT_MODEL):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._waist_adr = [int(self.model.joint(n).qposadr[0]) for n in WAIST_JOINTS]
        self._torso_bid = self.model.body("torso_link").id
        try:
            self._pelvis_bid = self.model.body("pelvis").id
        except Exception:
            self._pelvis_bid = None

    def T_pelvis_torso(self, waist_qpos) -> np.ndarray:
        q = np.asarray(waist_qpos, dtype=np.float64).ravel()
        self.data.qpos[:] = self.model.qpos0
        for adr, v in zip(self._waist_adr, q):
            self.data.qpos[adr] = v
        self._mj.mj_forward(self.model, self.data)
        T_w_torso = make_T(self.data.xmat[self._torso_bid].reshape(3, 3),
                           self.data.xpos[self._torso_bid])
        if self._pelvis_bid is not None:
            T_w_pelvis = make_T(self.data.xmat[self._pelvis_bid].reshape(3, 3),
                                self.data.xpos[self._pelvis_bid])
            return inv_T(T_w_pelvis) @ T_w_torso
        return T_w_torso


def _pq(T):
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


# ── Network camera source (frames from the Jetson) ──────────────────────────

class _NetCameraWorker(threading.Thread):
    """Receives JPEG frames + factory intrinsics from the Jetson head-cam server
    (g1_headcam_server.py) and runs AprilTag detection locally — exposing the same
    interface the localizer needs (latest()/K/dist/cam_params/started_ok/error/
    stop()). Auto-reconnects if the server isn't up yet or drops."""

    def __init__(self, host, port, *, family="tag36h11", tag_size=0.065,
                 dm_thresh=30.0, quad_decimate=1.0, nthreads=2, keep_image=False,
                 detect_hz=None):
        super().__init__(daemon=True, name=f"headcam-net-{host}:{port}")
        self.host = str(host)
        self.port = int(port)
        self.serial = f"{self.host}:{self.port}"
        self._cfg = dict(family=family, tag_size=tag_size, dm_thresh=dm_thresh,
                         quad_decimate=quad_decimate, nthreads=nthreads)
        self._keep_image = keep_image
        # Detection-rate cap. The Jetson streams ~30 fps, but decoding + AprilTag-detecting
        # EVERY frame is heavy — and when this worker runs inside the MANAGER process it
        # contends the GIL with the 60 Hz teleop IK thread (the classic ~10 s teleop/camera
        # lag: 30 full-res detects/s in-process is the exact load that caused it). Excess
        # frames are still received (the socket must be drained) but dropped BEFORE the
        # JPEG decode, which is nearly free. None = detect every frame (old behavior).
        self._detect_dt = (1.0 / float(detect_hz)) if detect_hz else 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self.K = self.dist = self.cam_params = None
        self.started_ok = False
        self.error = None

    @staticmethod
    def _recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("head-cam server closed the connection")
            buf += chunk
        return buf

    def _recv_msg(self, sock):
        (length,) = struct.unpack(">I", self._recv_exact(sock, 4))
        return self._recv_exact(sock, length)

    def run(self):
        detector = Detector(families=self._cfg["family"], nthreads=self._cfg["nthreads"],
                            quad_decimate=self._cfg["quad_decimate"],
                            refine_edges=True, decode_sharpening=0.25)
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                intr = json.loads(self._recv_msg(sock).decode("utf-8"))
                self.K = np.array([[intr["fx"], 0.0, intr["ppx"]],
                                   [0.0, intr["fy"], intr["ppy"]],
                                   [0.0, 0.0, 1.0]], dtype=np.float64)
                self.dist = np.array(intr.get("coeffs") or [0, 0, 0, 0, 0], dtype=np.float64)
                self.cam_params = (intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])
                self.error = None
                self.started_ok = True
                last_detect = 0.0
                while not self._stop.is_set():
                    payload = self._recv_msg(sock)
                    # Rate cap: drop this frame (pre-decode, nearly free) if the next
                    # detection isn't due yet. The recv above still drains the socket.
                    now = time.monotonic()
                    if self._detect_dt and (now - last_detect) < self._detect_dt:
                        continue
                    last_detect = now
                    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    poses, dets = detect_poses(detector, gray, self.cam_params,
                                               self._cfg["tag_size"], self._cfg["dm_thresh"])
                    snap = {"stamp": time.time(), "poses": poses}
                    if self._keep_image:
                        snap["image"] = img
                        snap["dets"] = dets
                    with self._lock:
                        self._latest = snap
            except Exception as exc:
                self.error = str(exc)
                if self._stop.is_set():
                    break
                time.sleep(1.0)   # retry the connection
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass

    def latest(self):
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def stop(self):
        self._stop.set()


# ── Detached camera worker (own process, own GIL) ────────────────────────────
# Running _NetCameraWorker inside the MANAGER process makes its per-frame JPEG decode
# + AprilTag detection contend the GIL with the 60 Hz teleop IK thread — the recurring
# multi-second teleop/camera lag (the manager idles near GIL saturation, so ANY per-
# frame vision work in-process tips the operator-pose queue into backlog). Mirror of
# the crate tracker's fix: run the worker in a subprocess and stream only the tiny
# detection results (tag id → pose) over a socketpair. Length-prefixed JSON.

class _DetLite:
    """Pickle-free stand-in for a pupil_apriltags detection on the manager side —
    carries the fields the localizer actually consumes."""
    __slots__ = ("tag_id", "decision_margin", "pose_err")

    def __init__(self, tag_id, decision_margin, pose_err):
        self.tag_id = int(tag_id)
        self.decision_margin = float(decision_margin)
        self.pose_err = float(pose_err)


def _ipc_default(o):
    """numpy → JSON (a raw numpy value must NEVER kill the IPC send — see the crate
    tracker's identical 'anchor timed out' bug)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _ipc_send(sock, obj):
    blob = json.dumps(obj, default=_ipc_default).encode("utf-8")
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def _ipc_recv(sock):
    buf = b""
    while len(buf) < 4:
        chunk = sock.recv(4 - len(buf))
        if not chunk:
            raise ConnectionError("camera IPC closed")
        buf += chunk
    (length,) = struct.unpack(">I", buf)
    buf = b""
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk:
            raise ConnectionError("camera IPC closed")
        buf += chunk
    return json.loads(buf.decode("utf-8"))


def _serve_camera(fd):
    """Subprocess entry: run a _NetCameraWorker here and push detection results (and,
    on demand, an annotated JPEG for the debug view) to the manager over ``fd``."""
    import base64
    sock = socket.socket(fileno=fd)
    try:
        kw = _ipc_recv(sock)["init"]
    except Exception:
        return
    worker = _NetCameraWorker(
        kw["host"], kw["port"], family=kw.get("family", "tag36h11"),
        tag_size=kw.get("tag_size", 0.065), dm_thresh=kw.get("dm_thresh", 30.0),
        quad_decimate=kw.get("quad_decimate", 1.0), nthreads=int(kw.get("nthreads", 2)),
        keep_image=True,                       # needed for the on-demand debug JPEG
        detect_hz=kw.get("detect_hz"))
    worker.start()

    state = {"jpeg_until": 0.0, "stop": False}

    def _rx():
        while not state["stop"]:
            try:
                msg = _ipc_recv(sock)
            except Exception:
                state["stop"] = True
                return
            if msg.get("cmd") == "shutdown":
                state["stop"] = True
            elif "want_jpeg" in msg:           # debug view open → stream frames briefly
                state["jpeg_until"] = time.time() + float(msg["want_jpeg"])
    threading.Thread(target=_rx, daemon=True).start()

    sent_intr = False
    last_stamp = None
    last_hb = last_jpeg = 0.0
    try:
        while not state["stop"]:
            now = time.time()
            snap = worker.latest()
            msg = None
            if snap is not None and snap["stamp"] != last_stamp:
                last_stamp = snap["stamp"]
                msg = {"stamp": snap["stamp"],
                       "poses": {int(tid): {"T": np.asarray(T).tolist(),
                                            "dm": float(getattr(d, "decision_margin", 0.0)),
                                            "pe": float(getattr(d, "pose_err", 0.0))}
                                 for tid, (T, d) in snap["poses"].items()},
                       "started_ok": worker.started_ok, "error": worker.error}
                if now < state["jpeg_until"] and now - last_jpeg > 0.25 \
                        and "image" in snap:
                    img = snap["image"].copy()
                    for d in snap.get("dets", []):
                        try:
                            draw_detection(img, d, worker.K, worker.dist,
                                           kw.get("tag_size", 0.065) * 0.5)
                        except Exception:
                            pass
                    ok, jpg = cv2.imencode(".jpg", img,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        msg["jpeg"] = base64.b64encode(jpg.tobytes()).decode("ascii")
                        last_jpeg = now
            elif now - last_hb > 0.5:          # heartbeat while connecting/retrying
                msg = {"hb": True, "started_ok": worker.started_ok, "error": worker.error}
            if msg is not None:
                last_hb = now
                if not sent_intr and worker.cam_params is not None:
                    msg["intr"] = {"K": worker.K.tolist(), "dist": worker.dist.tolist(),
                                   "cam_params": list(worker.cam_params)}
                    sent_intr = True
                try:
                    _ipc_send(sock, msg)
                except Exception:
                    break
            time.sleep(0.01)
    finally:
        worker.stop()
        try:
            sock.close()
        except Exception:
            pass


class _DetachedNetCamera:
    """Drop-in for _NetCameraWorker that runs the real worker in a SUBPROCESS (own GIL)
    and mirrors its surface: latest()/K/dist/cam_params/started_ok/error/serial/
    start()/stop()/join(). latest() returns the same {'stamp', 'poses': {id: (T, det)}}
    shape (no 'image'/'dets' — the debug view uses annotated_frame() instead)."""

    def __init__(self, host, port, *, family="tag36h11", tag_size=0.065,
                 dm_thresh=30.0, quad_decimate=1.0, nthreads=2, keep_image=False,
                 detect_hz=None):
        self.host, self.port = str(host), int(port)
        self.serial = f"{self.host}:{self.port}"
        self._init_kw = dict(host=self.host, port=self.port, family=family,
                             tag_size=tag_size, dm_thresh=dm_thresh,
                             quad_decimate=quad_decimate, nthreads=nthreads,
                             detect_hz=detect_hz)
        self.K = self.dist = self.cam_params = None
        self.started_ok = False
        self.error = None
        self._lock = threading.Lock()
        self._latest = None
        self._jpeg = None
        self._sock = None
        self._proc = None

    def start(self):
        import subprocess as _sp
        parent, child = socket.socketpair()
        try:
            self._proc = _sp.Popen(
                [sys.executable, os.path.abspath(__file__), "--serve-cam",
                 "--fd", str(child.fileno())],
                pass_fds=[child.fileno()],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        finally:
            child.close()
        self._sock = parent
        _ipc_send(self._sock, {"init": self._init_kw})
        threading.Thread(target=self._rx_loop, daemon=True,
                         name=f"headcam-proxy-{self.serial}").start()

    def _rx_loop(self):
        import base64
        try:
            while True:
                msg = _ipc_recv(self._sock)
                self.started_ok = bool(msg.get("started_ok", self.started_ok))
                self.error = msg.get("error")
                if "intr" in msg:
                    self.K = np.array(msg["intr"]["K"], dtype=np.float64)
                    self.dist = np.array(msg["intr"]["dist"], dtype=np.float64)
                    self.cam_params = tuple(msg["intr"]["cam_params"])
                if "stamp" in msg:
                    poses = {int(tid): (np.array(v["T"], dtype=np.float64),
                                        _DetLite(tid, v["dm"], v["pe"]))
                             for tid, v in msg.get("poses", {}).items()}
                    with self._lock:
                        self._latest = {"stamp": float(msg["stamp"]), "poses": poses}
                if "jpeg" in msg:
                    with self._lock:
                        self._jpeg = base64.b64decode(msg["jpeg"])
        except Exception:
            if self.error is None:
                self.error = "camera process died"

    def latest(self):
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def request_jpeg(self, secs=3.0):
        """Ask the subprocess to stream annotated JPEGs for ``secs`` (debug view open)."""
        try:
            _ipc_send(self._sock, {"want_jpeg": float(secs)})
        except Exception:
            pass

    def annotated_frame(self):
        """Latest annotated frame (BGR) from the subprocess, or None. Decode happens
        here (manager side) but ONLY while the debug view window is open."""
        self.request_jpeg()
        with self._lock:
            jpg = self._jpeg
        if jpg is None:
            return None
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

    def stop(self):
        try:
            if self._sock is not None:
                _ipc_send(self._sock, {"cmd": "shutdown"})
                self._sock.close()
        except Exception:
            pass
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.wait(timeout=3.0)
            except Exception:
                try:
                    p.terminate()
                except Exception:
                    pass
        self._proc = None

    def join(self, timeout=None):
        return                                  # interface parity with threading.Thread


# ── Localizer (manager-embeddable, mirrors the crate tracker) ───────────────

class HeadCamLocalizer:
    """Self-localizes the robot base from the head camera in its own thread.

    waist_getter() must return the live 3-vector of waist joints (yaw, roll,
    pitch) in radians, or None if unavailable (then a zero waist is assumed)."""

    def __init__(self, host, port=5005, *, model_path=DEFAULT_MODEL, waist_getter=None,
                 table_tag_id=0, tag_size=0.065, optical_convention="ros",
                 color_offset=(0.0, 0.0, 0.0), world_from_table=None,
                 family="tag36h11", dm_thresh=30.0,
                 quad_decimate=1.0, nthreads=2, rate_hz=15.0, debug_view=False,
                 out_of_process=False):
        self.host = str(host)
        self.port = int(port)
        self.serial = f"{self.host}:{self.port}"   # for logging / config_dict
        self.fk = HeadCamFK(model_path)
        self._waist_getter = waist_getter
        self.table_tag_id = int(table_tag_id)
        self.tag_size = float(tag_size)
        self.optical_convention = optical_convention
        self.color_offset = tuple(color_offset)
        # T_world_table: maps the detected tag-0 frame to the Z-up world. Must match
        # the crate tracker's so robot + crate share one world (see world_from_table).
        self.world_from_table = (np.eye(4) if world_from_table is None
                                 else np.asarray(world_from_table, dtype=np.float64))
        self.model_path = model_path
        self._rate_hz = float(rate_hz)
        # Frames come from the Jetson over TCP (the head D435 isn't on the laptop).
        # Detection is capped at rate_hz — the solve loop can't consume results any
        # faster. out_of_process=True (the manager) runs the decode+detect worker in a
        # SUBPROCESS with its own GIL: per-frame vision work inside the manager process
        # contends the GIL with the 60 Hz teleop IK thread → the multi-second teleop lag.
        cam_cls = _DetachedNetCamera if out_of_process else _NetCameraWorker
        self._cam = cam_cls(
            self.host, self.port, family=family, tag_size=self.tag_size,
            dm_thresh=dm_thresh, quad_decimate=quad_decimate, nthreads=nthreads,
            keep_image=debug_view, detect_hz=self._rate_hz)
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._latest = None
        self._rec_lock = threading.Lock()
        self._recording = False
        self._file = self._writer = None
        self._last_flush = 0.0
        # ── Crate contribution (peer → master fusion) ─────────────────────────
        # When enabled, this localizer ALSO turns the head cam's crate-tag detections
        # into world-frame crate-pose candidates and exposes them via latest_crate(),
        # so the manager can stream them to the master's crate tracker. Because the head
        # D435 rides the (teleoperated) torso, the camera's LIVE world pose is used each
        # frame: T_world_optical(t) = T_world_base_locked · T_pelvis_optical(waist(t)).
        # T_world_base is captured once with anchor_base() (robot base is fixed), so it
        # stays valid even when tag 0 is later occluded by the crate.
        self._crate_lock = threading.Lock()
        self._crate_enabled = False
        self._bundle = None          # {tag_id: T_crate_tag 4x4}
        self._bundle_inv = None      # {tag_id: inv(T_crate_tag)}
        self._base_locked = None     # fixed T_world_base captured by anchor_base()
        self._crate_latest = None

    # lifecycle
    def start(self):
        self._cam.start()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="headcam-localize")
        self._thread.start()

    def wait_ready(self, timeout=8.0):
        end = time.time() + timeout
        while time.time() < end and not (self._cam.started_ok or self._cam.error):
            time.sleep(0.05)
        if self._cam.error:
            return False, f"{self.serial}: ERROR {self._cam.error}"
        if not self._cam.started_ok:
            return False, f"{self.serial}: no frames (timeout)"
        return True, f"{self.serial}: ready"

    def _waist(self):
        if self._waist_getter is None:
            return np.zeros(3)
        try:
            q = self._waist_getter()
            return np.zeros(3) if q is None else np.asarray(q, float).ravel()[:3]
        except Exception:
            return np.zeros(3)

    def _localize_from_snap(self, snap):
        """Base pose from tag 0 in ``snap`` (or ok=False)."""
        if not snap or self.table_tag_id not in snap["poses"]:
            return {"stamp": time.time(), "ok": False}
        T_optical_table, det = snap["poses"][self.table_tag_id]
        if not is_proper_rotation(T_optical_table[:3, :3]):
            return {"stamp": time.time(), "ok": False}
        waist = self._waist()
        T_pt = self.fk.T_pelvis_torso(waist)
        T_world_base = compute_T_world_base(
            T_optical_table, T_pt, self.optical_convention,
            T_world_table=self.world_from_table, color_offset=self.color_offset)
        p, q = _pq(T_world_base)
        return {"stamp": time.time(), "ok": True, "T": T_world_base, "p": p, "q": q,
                "base_z": float(p[2]), "upright": upright_score(T_world_base),
                "waist": waist, "T_optical_table": T_optical_table,
                "dm": float(getattr(det, "decision_margin", 0.0))}

    def localize_once(self):
        return self._localize_from_snap(self._cam.latest())

    # ── Crate contribution ────────────────────────────────────────────────────
    def enable_crate_contribution(self, bundle_path):
        """Load a crate bundle so this localizer also emits world-frame crate candidates
        from the head cam (streamed to the master). Returns (ok, message)."""
        try:
            with open(bundle_path) as f:
                b = json.load(f)
            bundle = {int(k): np.array(v["T_crate_tag"], float)
                      for k, v in b["tags"].items()}
        except Exception as exc:                             # noqa: BLE001
            return False, f"crate bundle load failed: {exc}"
        with self._crate_lock:
            self._bundle = bundle
            self._bundle_inv = {k: inv_T(T) for k, T in bundle.items()}
            self._crate_enabled = True
            self._crate_latest = None
        return True, f"crate contribution ready ({len(bundle)} bundled tags)"

    def disable_crate_contribution(self):
        with self._crate_lock:
            self._crate_enabled = False
            self._crate_latest = None

    def anchor_base(self, duration_s=1.5):
        """Capture and LOCK T_world_base by averaging the LOOP thread's recent tag-0
        localizations (the robot base is fixed, so this pose stays valid even once tag 0
        is occluded). Reads via latest() — it must NOT call the FK itself, or it would
        race the loop thread on the shared MuJoCo MjData and crash. Returns (ok, n_obs)."""
        end = time.time() + float(duration_s)
        Ts, seen = [], set()
        while time.time() < end:
            r = self.latest()                            # thread-safe (loop-thread output)
            if r and r.get("ok") and abs(r.get("upright", 0.0)) > 0.5:
                st = r.get("stamp")
                if st not in seen:                       # one sample per loop iteration
                    seen.add(st)
                    Ts.append(np.asarray(r["T"], dtype=np.float64))
            time.sleep(0.03)
        if len(Ts) < 3:
            return False, len(Ts)
        Rs = np.array([T[:3, :3] for T in Ts])
        ts = np.array([T[:3, 3] for T in Ts])
        Rm = Rotation.from_matrix(Rs).mean().as_matrix()
        with self._crate_lock:
            self._base_locked = make_T(Rm, ts.mean(axis=0))
        return True, len(Ts)

    @property
    def base_anchored(self) -> bool:
        return self._base_locked is not None

    @property
    def crate_contributing(self) -> bool:
        return self._crate_enabled

    def _T_pelvis_optical(self, waist):
        """Head-cam optical frame in the pelvis(base) frame, from live waist FK."""
        return (self.fk.T_pelvis_torso(waist) @ T_torso_d435()
                @ T_d435_optical(self.optical_convention, self.color_offset))

    def _crate_from_snap(self, snap):
        """Fuse the head cam's crate-tag detections into ONE world-frame crate pose,
        using the LIVE camera pose (locked base · live waist FK). ok=False if the base
        isn't anchored or no bundled tag is visible."""
        if self._bundle_inv is None or self._base_locked is None or not snap:
            return {"stamp": time.time(), "ok": False}
        waist = self._waist()
        T_world_optical = self._base_locked @ self._T_pelvis_optical(waist)
        cands = []
        for tid, (T_opt_tag, det) in snap["poses"].items():
            inv_b = self._bundle_inv.get(int(tid))
            if inv_b is None or not is_proper_rotation(T_opt_tag[:3, :3]):
                continue
            T_world_crate = T_world_optical @ T_opt_tag @ inv_b
            if not is_proper_rotation(T_world_crate[:3, :3]):
                continue
            cands.append((T_world_crate, float(getattr(det, "decision_margin", 1.0))))
        if not cands:
            return {"stamp": time.time(), "ok": False}
        T, rot_resid, trans_resid = fuse_poses(cands)
        if T is None:
            return {"stamp": time.time(), "ok": False}
        p, q = _pq(T)
        return {"stamp": time.time(), "ok": True, "p": p, "q": q,
                "n_tags": len(cands), "tag_ids": sorted(int(t) for t in snap["poses"]
                                                        if int(t) in self._bundle_inv),
                "rot_resid_deg": rot_resid, "trans_resid_mm": trans_resid}

    def latest_crate(self):
        with self._crate_lock:
            return None if self._crate_latest is None else dict(self._crate_latest)

    def _loop(self):
        dt = 1.0 / max(1.0, self._rate_hz)
        while not self._stop.is_set():
            t0 = time.time()
            snap = self._cam.latest()
            try:
                res = self._localize_from_snap(snap)
            except Exception as exc:
                res = {"stamp": time.time(), "ok": False, "error": str(exc)}
            with self._lock:
                self._latest = res
            if res.get("ok"):
                self._write(res)
            # Crate candidates from the same frame (once the base is anchored).
            if self._crate_enabled and self._base_locked is not None:
                try:
                    crate = self._crate_from_snap(snap)
                except Exception as exc:                     # noqa: BLE001
                    crate = {"stamp": time.time(), "ok": False, "error": str(exc)}
                with self._crate_lock:
                    self._crate_latest = crate
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def latest(self):
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def debug_frame(self):
        """Latest head-camera frame (BGR) with the AprilTag overlay + a 'table tag
        in frame?' banner — for the manager's debug view. Returns None until a
        frame arrives (needs debug_view=True so the worker keeps images)."""
        snap = self._cam.latest()
        if hasattr(self._cam, "annotated_frame"):
            # Detached (out-of-process) camera: the overlay is drawn + JPEG'd in the
            # subprocess on demand; only the decode happens here, and only while the
            # debug view window is actually open.
            img = self._cam.annotated_frame()
            if img is None:
                return None
            tag0 = bool(snap and self.table_tag_id in snap.get("poses", {}))
        else:
            if not snap or "image" not in snap:
                return None
            img = snap["image"].copy()
            tag0 = False
            for d in snap.get("dets", []):
                try:
                    draw_detection(img, d, self._cam.K, self._cam.dist,
                                   self.tag_size * 0.5)
                except Exception:
                    pass
                if int(d.tag_id) == self.table_tag_id:
                    tag0 = True
        res = self.latest()
        if tag0 and res and res.get("ok"):
            up = res.get("upright", 0.0)
            banner = f"table tag {self.table_tag_id} IN FRAME   up={up:+.2f}"
            col = (0, 220, 0) if up > 0.8 else (0, 200, 255)
        elif tag0:
            banner, col = f"table tag {self.table_tag_id} in frame", (0, 220, 0)
        else:
            banner = f"table tag {self.table_tag_id} NOT in frame — aim the head at it"
            col = (0, 0, 255)
        cv2.rectangle(img, (0, 0), (img.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(img, banner, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        return img

    # recording
    _FIELDS = ["timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw",
               "waist_yaw", "waist_roll", "waist_pitch",
               "tag_px", "tag_py", "tag_pz", "tag_qx", "tag_qy", "tag_qz", "tag_qw",
               "decision_margin", "upright"]

    def start_recording(self, session_dir, label=""):
        import csv
        os.makedirs(session_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""
        path = os.path.abspath(os.path.join(session_dir, f"head_base{suffix}_{ts}.csv"))
        with self._rec_lock:
            self._file = open(path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self._FIELDS)
            self._recording = True
            self._last_flush = time.time()
        return path

    def _write(self, res):
        with self._rec_lock:
            if not self._recording:
                return
            p, q = res["p"], res["q"]
            tp, tq = _pq(res["T_optical_table"])
            w = res["waist"]
            self._writer.writerow([
                round(res["stamp"], 4), *np.round(p, 6), *np.round(q, 6),
                *np.round(w[:3], 5), *np.round(tp, 6), *np.round(tq, 6),
                round(res["dm"], 2), round(res["upright"], 4)])
            if time.time() - self._last_flush > 1.0:
                self._file.flush()
                self._last_flush = time.time()

    def stop_recording(self):
        with self._rec_lock:
            self._recording = False
            if self._file is not None:
                try:
                    self._file.flush(); self._file.close()
                except Exception:
                    pass
            self._file = self._writer = None

    def stop(self):
        self.stop_recording()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cam.stop()
        self._cam.join(timeout=2.0)

    def config_dict(self):
        return {
            "headcam_host": self.host,
            "headcam_port": self.port,
            "headcam_table_tag_id": self.table_tag_id,
            "headcam_table_tag_size": self.tag_size,
            "headcam_optical_convention": self.optical_convention,
            "headcam_color_offset": list(self.color_offset),
            "headcam_world_from_table": self.world_from_table.tolist(),
            "headcam_T_torso_d435": T_torso_d435().tolist(),
            "headcam_model_path": self.model_path,
        }


# ── selftest ────────────────────────────────────────────────────────────────

def _selftest():
    rng = np.random.default_rng(0)

    def rand_T():
        T = np.eye(4)
        T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        T[:3, 3] = rng.uniform(-1, 1, 3)
        return T

    print("=== chain math: forward-simulate a detection, recover T_world_base ===")
    for conv in ("ros", "identity"):
        T_world_base_gt = rand_T()
        T_pelvis_torso = rand_T()
        T_pelvis_optical = (T_pelvis_torso @ T_torso_d435() @ T_d435_optical(conv))
        T_world_optical = T_world_base_gt @ T_pelvis_optical
        T_optical_table = inv_T(T_world_optical)             # world = table = identity
        T_rec = compute_T_world_base(T_optical_table, T_pelvis_torso, conv)
        err = np.abs(T_rec - T_world_base_gt).max()
        print(f"  convention={conv:8s} recovery error = {err:.2e}")
        assert err < 1e-9
    print("  PASS")

    print("=== optical rotation is a proper rotation ===")
    print(f"  det(_OPTICAL_ROS) = {np.linalg.det(_OPTICAL_ROS):+.6f}")
    assert abs(np.linalg.det(_OPTICAL_ROS) - 1.0) < 1e-9
    print("  PASS")

    print("=== MuJoCo waist FK loads + torso sits above pelvis ===")
    if os.path.exists(DEFAULT_MODEL):
        try:
            fk = HeadCamFK(DEFAULT_MODEL)
            T = fk.T_pelvis_torso([0.0, 0.0, 0.0])
            print(f"  torso origin in pelvis frame: {np.round(T[:3,3],3).tolist()} "
                  f"(z should be positive, ~0.2-0.5m)")
            assert T[2, 3] > 0.0
            print("  PASS")
        except Exception as exc:
            print(f"  (skipped — model/mujoco issue: {exc})")
    else:
        print(f"  (skipped — model not found at {DEFAULT_MODEL})")

    print("\nALL SELFTESTS PASSED")


# ── standalone: camera-access walkthrough + live verification ───────────────

def _run_live(args):
    fk = HeadCamFK(args.model)
    cam = _NetCameraWorker(args.host, args.port, family="tag36h11",
                           tag_size=args.table_tag_size, dm_thresh=args.decision_margin,
                           quad_decimate=1.0, nthreads=4, keep_image=True)
    cam.start()
    end = time.time() + 10.0
    while time.time() < end and not (cam.started_ok or cam.error):
        time.sleep(0.05)
    if not cam.started_ok:
        sys.exit(f"Could not connect to the head-cam server at {args.host}:{args.port} "
                 f"({cam.error}).\nIs g1_headcam_server.py running on the Jetson?")
    print(f"Connected to head-cam server {args.host}:{args.port}. "
          f"Point the head at the table tag (id {args.table_id}).")
    print("keys:  o toggle optical convention   f toggle table-z-down flip   s save   q quit")
    waist = np.asarray(args.waist, float)
    convention = args.optical
    z_down = args.table_z_down
    win = "Head-cam localize (o=optical, f=z-flip, s=save, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    K = cam.K
    last = None
    try:
        while True:
            snap = cam.latest()
            img = snap.get("image") if snap else None
            disp = img.copy() if img is not None else np.full((360, 640, 3), 40, np.uint8)
            if snap and img is not None:
                for d in snap.get("dets", []):
                    draw_detection(disp, d, cam.K, cam.dist, args.table_tag_size * 0.5)
            txt2, col = "table tag not seen", (0, 165, 255)
            if snap and args.table_id in snap.get("poses", {}):
                T_ot, det = snap["poses"][args.table_id]
                if is_proper_rotation(T_ot[:3, :3]):
                    T_wb = compute_T_world_base(T_ot, fk.T_pelvis_torso(waist), convention,
                                                T_world_table=world_from_table(z_down))
                    p = T_wb[:3, 3]
                    up = upright_score(T_wb)
                    last = T_wb
                    verdict = "UPRIGHT ok" if up > 0.8 else "tilted? try o / f"
                    txt2 = (f"base xyz=[{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}]m  "
                            f"up={up:+.2f} ({verdict})")
                    col = (0, 255, 0) if up > 0.8 else (0, 200, 255)
            cv2.putText(disp, f"optical={convention}  table_z_down={z_down}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(disp, txt2, (12, disp.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("o"):
                convention = "identity" if convention == "ros" else "ros"
                print(f"optical convention -> {convention}")
            elif key == ord("f"):
                z_down = not z_down
                print(f"table_z_down -> {z_down}")
            elif key == ord("s") and last is not None:
                out = {"robot_id": args.robot_id, "frame": "table_apriltag_id0",
                       "T_world_base": np.round(last, 6).tolist(),
                       "optical_convention": convention, "table_z_down": z_down,
                       "waist": list(map(float, waist)),
                       "upright": round(upright_score(last), 4),
                       "created": datetime.now().isoformat(timespec="seconds")}
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"saved {args.out}")
    finally:
        cam.stop()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(
        description="Head-camera base localization (frames relayed from the Jetson).")
    p.add_argument("--selftest", action="store_true", help="verify the math, no hardware")
    p.add_argument("--serve-cam", dest="serve_cam", action="store_true",
                   help="internal: run the detached camera worker over --fd (spawned "
                        "by _DetachedNetCamera; not for manual use).")
    p.add_argument("--fd", type=int, default=None,
                   help="internal: inherited socketpair fd for --serve-cam.")
    p.add_argument("--host", type=str, default=None,
                   help="Jetson IP running g1_headcam_server.py (e.g. 192.168.123.164)")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--table-id", dest="table_id", type=int, default=0)
    p.add_argument("--table-tag-size", dest="table_tag_size", type=float, default=0.065)
    p.add_argument("--optical", choices=("ros", "identity"), default="ros")
    p.add_argument("--table-z-down", dest="table_z_down", action="store_true", default=True,
                   help="flip the world 180deg so Z is up (tag 0's z points into the "
                        "table — verified true on this rig; this is the default).")
    p.add_argument("--no-table-z-down", dest="table_z_down", action="store_false",
                   help="disable the flip (use if tag 0's z already points up).")
    p.add_argument("--waist", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("YAW", "ROLL", "PITCH"))
    p.add_argument("--robot-id", dest="robot_id", type=int, default=None)
    p.add_argument("--out", type=str, default="robot_base_from_headcam.json")
    p.add_argument("--decision-margin", dest="decision_margin", type=float, default=30.0)
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.serve_cam:
        silence_native_logs()
        _serve_camera(args.fd)
        return

    silence_native_logs()
    if not args.host:
        sys.exit("Pass --host <jetson-ip> (the Jetson running g1_headcam_server.py).\n"
                 "Run with --selftest to check the math without hardware.")
    _run_live(args)


if __name__ == "__main__":
    main()
