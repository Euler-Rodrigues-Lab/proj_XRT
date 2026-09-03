#!/usr/bin/env python3
"""
apriltag_tracker.py — runtime crate-pose tracker for the user study.

Drives one or two Intel RealSense D435 cameras, detects ``tag36h11`` markers,
and fuses every visible crate face into a single 6-DoF crate pose expressed in
the TABLE frame (tag id 0 = world origin). Designed to run both:

  * standalone  — for bring-up, anchoring, live debug, and recording, and
  * embedded    — imported by XRT_user_study_manager.py, started/stopped with a
                  recording (mirrors ZEDRecorder / SessionRecorder).

Pipeline (all source-keyed by camera, so more sources can be added later):
  per camera (own thread, since RealSense wait_for_frames blocks):
      frame -> detect_poses() -> {tag_id: T_cam_tag}      (reflected poses dropped)
  one-time anchor:
      see table tag 0 -> T_world_cam = T_world_table @ inv(T_cam_table)
  fusion (own thread, ~30 Hz):
      for each visible crate tag i on camera c:
          T_world_crate = T_world_cam[c] @ T_cam_tag_i @ inv(T_crate_tag_i)
      robust mean over all candidates (scipy Rotation.mean + weighted centroid)
      + inter-camera agreement as a live health check

Recording (into the SAME session dir as the robot/ZED data, time.time() stamps):
  crate_pose_<label>_<ts>.csv      fused world crate pose + fusion stats / frame
  crate_tags_raw_<label>_<ts>.csv  every tag's world pose per camera (re-fusable;
                                   also captures table re-checks + future robot tags)

Cameras are assumed STATIC: anchor once at session start (per the project plan).
The crate body frame is whatever crate_bundle.json defines (anchor tag id 10 =
inside-centre-base by default).

Controls (standalone --debug-view):  q quit   a re-anchor cameras

Deps:  pip install pupil-apriltags pyrealsense2   (scipy + opencv already present)
"""

from __future__ import annotations

import argparse
import csv
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
    # OpenCV defaults its internal thread pool to ALL cores, so a single cvtColor can
    # briefly fan out across the box and thrash the scheduler. In the manager this shares
    # a process (and GIL) with the teleop IK loop, so cap it to one thread — cvtColor on a
    # 720p frame is fast single-threaded, and this keeps AprilTag frame prep from starving
    # IK. Harmless when the tracker runs standalone.
    cv2.setNumThreads(1)
except ImportError:
    sys.exit("opencv-python required (already in the env): pip install 'opencv-python>=4.5,<5'")

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    sys.exit("scipy required (already in the env): pip install scipy")

try:
    from pupil_apriltags import Detector
except ImportError:
    sys.exit("pupil-apriltags not found: pip install pupil-apriltags")

# Reuse helpers already written + verified (keep these files alongside).
try:
    from apriltag_probe import (list_devices, start_pipeline, draw_detection,
                                silence_native_logs)
    from apriltag_crate_calibrate import is_proper_rotation, make_T, detect_poses
except Exception as exc:
    sys.exit(f"Could not import apriltag_probe.py / apriltag_crate_calibrate.py "
             f"(keep them in the same folder): {exc}")


# ── SE(3) + fusion helpers ───────────────────────────────────────────────────

def inv_T(T: np.ndarray) -> np.ndarray:
    """Exact rigid-transform inverse (assumes T[:3,:3] orthonormal)."""
    R = T[:3, :3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def T_to_pq(T: np.ndarray):
    """4x4 -> (translation (3,), quaternion [x,y,z,w])."""
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


def fuse_poses(candidates):
    """Robustly fuse [(T 4x4, weight), ...] -> (T_mean, rot_resid_deg, trans_resid_mm).

    Reflected/degenerate candidates are dropped first so scipy never sees a
    det<=0 matrix (same guard as the calibrator)."""
    good = [(T, w) for (T, w) in candidates if is_proper_rotation(T[:3, :3])]
    if not good:
        return None, None, None
    Rs = np.array([T[:3, :3] for T, _ in good])
    ts = np.array([T[:3, 3] for T, _ in good])
    w = np.array([w for _, w in good], dtype=float)
    w = np.ones(len(good)) if w.sum() <= 0 else w / w.sum()
    rot = Rotation.from_matrix(Rs)
    rmean = rot.mean(weights=w)
    tmean = np.average(ts, axis=0, weights=w)
    ang = np.degrees((rmean.inv() * rot).magnitude())
    rot_resid = float(np.sqrt(np.mean(ang ** 2)))
    trans_resid = float(np.sqrt(np.mean(np.sum((ts - tmean) ** 2, axis=1))) * 1000.0)
    return make_T(rmean.as_matrix(), tmean), rot_resid, trans_resid


# ── Per-camera worker thread ─────────────────────────────────────────────────

class _CameraWorker(threading.Thread):
    """Owns one RealSense pipeline; publishes the latest per-tag camera-frame
    poses. RealSense wait_for_frames blocks, so this MUST be its own thread."""

    def __init__(self, serial, *, width, height, fps, family, tag_size,
                 dm_thresh, quad_decimate, nthreads, keep_image):
        super().__init__(daemon=True, name=f"apriltag-cam-{serial}")
        self.serial = str(serial)
        self._cfg = dict(width=width, height=height, fps=fps, family=family,
                         tag_size=tag_size, dm_thresh=dm_thresh,
                         quad_decimate=quad_decimate, nthreads=nthreads)
        self._keep_image = keep_image
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self.K = self.dist = self.cam_params = None
        self.started_ok = False
        self.error = None

    def run(self):
        try:
            pipeline, K, dist, intr = start_pipeline(
                self.serial, self._cfg["width"], self._cfg["height"], self._cfg["fps"])
        except Exception as exc:
            self.error = f"{exc}"
            return
        self.K, self.dist = K, dist
        self.cam_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        detector = Detector(families=self._cfg["family"],
                            nthreads=self._cfg["nthreads"],
                            quad_decimate=self._cfg["quad_decimate"],
                            refine_edges=True, decode_sharpening=0.25)
        self.started_ok = True
        try:
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
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
            self.error = f"{exc}"
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    def latest(self):
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def stop(self):
        self._stop.set()


# ── Tracker ──────────────────────────────────────────────────────────────────

class AprilTagTracker:
    """Multi-camera crate-pose tracker. Thread-safe ``latest()`` for the GUI;
    autonomous fusion + recording threads off the teleop hot path."""

    def __init__(self, serials, bundle_path, *, table_id=0, world_from_table=None,
                 tag_size=None, width=1280, height=720, fps=30, dm_thresh=30.0,
                 quad_decimate=1.0, nthreads=2, rate_hz=30.0, debug_view=False,
                 extra_cams=None):
        with open(bundle_path) as f:
            b = json.load(f)
        self.bundle_path = os.path.abspath(bundle_path)
        self.family = b.get("family", "tag36h11")
        self.tag_size = float(tag_size if tag_size is not None else b["tag_size"])
        self.anchor_id = b.get("anchor_id")
        self.bundle = {int(k): np.array(v["T_crate_tag"], float)
                       for k, v in b["tags"].items()}
        self.bundle_inv = {k: inv_T(T) for k, T in self.bundle.items()}

        self.table_id = int(table_id)
        # World = table tag frame by default. Once the table tag's axes are
        # confirmed (probe 'p'), pass a fixed offset here to force a true Z-up.
        self.T_world_table = (np.eye(4) if world_from_table is None
                              else np.array(world_from_table, float))
        self._rate_hz = float(rate_hz)
        self._debug_view = debug_view
        # Local USB cameras (the external rig) + any pre-built extra cameras (e.g. a
        # network worker for the G1 head cam). Both kinds expose the same interface
        # (latest()/serial/K/dist/started_ok/error/stop), so anchor + fusion treat
        # them identically — every camera that sees tag 0 anchors into the SAME world.
        self.cams = [_CameraWorker(
            s, width=width, height=height, fps=fps, family=self.family,
            tag_size=self.tag_size, dm_thresh=dm_thresh, quad_decimate=quad_decimate,
            nthreads=nthreads, keep_image=debug_view) for s in serials]
        if extra_cams:
            self.cams.extend(extra_cams)
        self.T_world_cam = {c.serial: None for c in self.cams}

        self._stop = threading.Event()
        self._fusion_thread = None
        self._latest = None
        self._latest_lock = threading.Lock()

        # Remote crate candidates (e.g. robot 2's head cam over the study link),
        # fused into the SAME table-origin world. (T 4x4, weight, stamp).
        self._remote_lock = threading.Lock()
        self._remote_candidates = []
        self._remote_ttl = 0.5   # seconds; drop stale remote candidates
        # The peer's head cam is a self-anchored (world-frame) contributor — not a local
        # _CameraWorker. Track when it last sent so it can be reported as a 4th, already-
        # anchored crate camera (so calibration reads 4/4, not 3/3). Longer window than the
        # fusion TTL so the "camera present" status stays steady between candidate bursts.
        self._remote_last_stamp = 0.0
        self._remote_window = 3.0
        self._remote_serial = "peer:headcam"

        self._rec_lock = threading.Lock()
        self._recording = False
        self._pose_file = self._raw_file = None
        self._pose_writer = self._raw_writer = None
        self._last_flush = 0.0

    # -- lifecycle --
    def start(self):
        for c in self.cams:
            c.start()
        self._fusion_thread = threading.Thread(target=self._fusion_loop,
                                               daemon=True, name="apriltag-fuse")
        self._fusion_thread.start()

    def wait_ready(self, timeout=8.0):
        """Block until every camera has produced frames (or errored/timed out).
        Returns (ok, messages)."""
        end = time.time() + timeout
        while time.time() < end:
            if all(c.started_ok or c.error for c in self.cams):
                break
            time.sleep(0.05)
        msgs = []
        ok = True
        for c in self.cams:
            if c.error:
                ok = False
                msgs.append(f"{c.serial}: ERROR {c.error}")
            elif not c.started_ok:
                ok = False
                msgs.append(f"{c.serial}: no frames (timeout)")
            else:
                msgs.append(f"{c.serial}: ready")
        return ok, msgs

    def anchor_cameras(self, duration_s=1.5, serials=None, keep_on_fail=False):
        """One-time extrinsic capture: average the table-tag pose per camera and set
        T_world_cam. Returns {serial: (status, n_obs)} with status ok / kept / no_table.

        serials      : restrict to these cameras (None = all). Use to re-anchor ONE
                       bumped camera without disturbing the others' anchors.
        keep_on_fail : if a (selected) camera can't see the table tag, KEEP its existing
                       anchor instead of clearing it — so a mid-session re-anchor of the
                       bumped camera never drops a good camera that momentarily can't see
                       tag 0 (e.g. occluded by the crate)."""
        cams = [c for c in self.cams if serials is None or c.serial in serials]
        end = time.time() + duration_s
        buf = {c.serial: [] for c in cams}
        last = {c.serial: None for c in cams}
        while time.time() < end:
            for c in cams:
                snap = c.latest()
                if snap and snap["stamp"] != last[c.serial]:
                    last[c.serial] = snap["stamp"]
                    if self.table_id in snap["poses"]:
                        buf[c.serial].append(snap["poses"][self.table_id][0])
            time.sleep(0.005)

        results = {}
        for c in cams:
            Ts = buf[c.serial]
            if len(Ts) >= 3:
                Rs = np.array([T[:3, :3] for T in Ts])
                ts = np.array([T[:3, 3] for T in Ts])
                Rm = Rotation.from_matrix(Rs).mean()
                T_cam_table = make_T(Rm.as_matrix(), ts.mean(axis=0))
                self.T_world_cam[c.serial] = self.T_world_table @ inv_T(T_cam_table)
                results[c.serial] = ("ok", len(Ts))
            elif keep_on_fail and self.T_world_cam.get(c.serial) is not None:
                results[c.serial] = ("kept", len(Ts))     # no tag 0 now → keep old anchor
            else:
                self.T_world_cam[c.serial] = None
                results[c.serial] = ("no_table", len(Ts))
        return results

    @property
    def is_anchored(self) -> bool:
        return any(v is not None for v in self.T_world_cam.values())

    def remote_active(self) -> bool:
        """True while the peer's head cam is actively contributing (a candidate within the
        window). Its poses are already world-frame, so it counts as an anchored camera."""
        return (time.time() - self._remote_last_stamp) < self._remote_window

    def add_remote_candidate(self, p, q, weight=1.0, stamp=None):
        """Inject a crate-pose candidate from a remote source (robot 2's head cam
        over the study link), already in the table-origin world. Thread-safe."""
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(q).as_matrix()
        T[:3, 3] = np.asarray(p, dtype=np.float64)
        st = stamp or time.time()
        with self._remote_lock:
            self._remote_candidates.append((T, float(weight), st))
            cutoff = time.time() - self._remote_ttl
            self._remote_candidates = [c for c in self._remote_candidates if c[2] >= cutoff]
        self._remote_last_stamp = max(self._remote_last_stamp, st)

    # -- fusion --
    def fuse_once(self):
        candidates, raw, per_cam = [], [], {}
        cams_used, tag_ids = set(), set()
        for c in self.cams:
            Twc = self.T_world_cam.get(c.serial)
            if Twc is None:
                continue
            snap = c.latest()
            if not snap:
                continue
            for tid, (T_cam_tag, det) in snap["poses"].items():
                T_world_tag = Twc @ T_cam_tag
                raw.append((c.serial, snap["stamp"], tid, T_world_tag, det))
                inv_bundle = self.bundle_inv.get(tid)
                if inv_bundle is None:          # not a crate tag (table / robot / unknown)
                    continue
                T_world_crate = T_world_tag @ inv_bundle
                if not is_proper_rotation(T_world_crate[:3, :3]):
                    continue
                w = float(getattr(det, "decision_margin", 1.0))
                candidates.append((T_world_crate, w))
                per_cam.setdefault(c.serial, []).append((T_world_crate, w))
                cams_used.add(c.serial)
                tag_ids.add(tid)

        # Remote crate candidates (robot 2's head cam over the link) — same world.
        with self._remote_lock:
            now = time.time()
            for T_rc, w_rc, st_rc in self._remote_candidates:
                if now - st_rc <= self._remote_ttl and is_proper_rotation(T_rc[:3, :3]):
                    candidates.append((T_rc, w_rc))
                    per_cam.setdefault("remote", []).append((T_rc, w_rc))
                    cams_used.add("remote")

        stamp = time.time()
        if not candidates:
            return {"stamp": stamp, "ok": False, "raw": raw}

        T, rot_resid, trans_resid = fuse_poses(candidates)
        if T is None:
            return {"stamp": stamp, "ok": False, "raw": raw}

        # inter-camera agreement (health check for a bumped mount)
        intercam_t = intercam_r = None
        if len(per_cam) == 2:
            means = [fuse_poses(cl)[0] for cl in per_cam.values()]
            if all(m is not None for m in means):
                Ta, Tb = means
                intercam_t = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000.0)
                Rrel = Ta[:3, :3].T @ Tb[:3, :3]
                if is_proper_rotation(Rrel):
                    intercam_r = float(np.degrees(Rotation.from_matrix(Rrel).magnitude()))

        p, q = T_to_pq(T)
        return {"stamp": stamp, "ok": True, "T": T, "p": p, "q": q,
                "n_tags": len(tag_ids), "tag_ids": sorted(tag_ids),
                "cams": sorted(cams_used), "n_cams": len(cams_used),
                "rot_resid_deg": rot_resid, "trans_resid_mm": trans_resid,
                "intercam_trans_mm": intercam_t, "intercam_rot_deg": intercam_r,
                "raw": raw}

    def _fusion_loop(self):
        dt = 1.0 / max(1.0, self._rate_hz)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                res = self.fuse_once()
            except Exception as exc:
                res = {"stamp": time.time(), "ok": False, "error": str(exc), "raw": []}
            with self._latest_lock:
                self._latest = res
            if res is not None:
                self._write_rows(res)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def latest(self):
        """Thread-safe snapshot for the GUI (without the bulky raw list)."""
        with self._latest_lock:
            if self._latest is None:
                return None
            r = dict(self._latest)
        r.pop("raw", None)
        return r

    # -- recording --
    _POSE_FIELDS = ["timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw",
                    "n_tags", "n_cams", "cams", "tag_ids",
                    "rot_resid_deg", "trans_resid_mm",
                    "intercam_trans_mm", "intercam_rot_deg"]
    _RAW_FIELDS = ["timestamp", "cam_serial", "cam_stamp", "tag_id",
                   "px", "py", "pz", "qx", "qy", "qz", "qw",
                   "decision_margin", "pose_err"]

    def start_recording(self, session_dir, label=""):
        os.makedirs(session_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""
        pose_path = os.path.abspath(os.path.join(session_dir, f"crate_pose{suffix}_{ts}.csv"))
        raw_path = os.path.abspath(os.path.join(session_dir, f"crate_tags_raw{suffix}_{ts}.csv"))
        with self._rec_lock:
            self._pose_file = open(pose_path, "w", newline="")
            self._pose_writer = csv.writer(self._pose_file)
            self._pose_writer.writerow(self._POSE_FIELDS)
            self._raw_file = open(raw_path, "w", newline="")
            self._raw_writer = csv.writer(self._raw_file)
            self._raw_writer.writerow(self._RAW_FIELDS)
            self._recording = True
            self._last_flush = time.time()
        return pose_path, raw_path

    def _write_rows(self, res):
        with self._rec_lock:
            if not self._recording:
                return
            stamp = round(res["stamp"], 4)
            if res.get("ok"):
                p, q = res["p"], res["q"]
                self._pose_writer.writerow([
                    stamp, *np.round(p, 6), *np.round(q, 6),
                    res["n_tags"], res["n_cams"], "|".join(res["cams"]),
                    " ".join(map(str, res["tag_ids"])),
                    round(res["rot_resid_deg"], 4), round(res["trans_resid_mm"], 3),
                    None if res["intercam_trans_mm"] is None else round(res["intercam_trans_mm"], 3),
                    None if res["intercam_rot_deg"] is None else round(res["intercam_rot_deg"], 4),
                ])
            for serial, cam_stamp, tid, T_world_tag, det in res.get("raw", []):
                p, q = T_to_pq(T_world_tag)
                self._raw_writer.writerow([
                    stamp, serial, round(cam_stamp, 4), tid,
                    *np.round(p, 6), *np.round(q, 6),
                    round(float(getattr(det, "decision_margin", 0.0)), 2),
                    round(float(getattr(det, "pose_err", float("nan"))), 6),
                ])
            if time.time() - self._last_flush > 1.0:
                self._pose_file.flush()
                self._raw_file.flush()
                self._last_flush = time.time()

    def stop_recording(self):
        with self._rec_lock:
            self._recording = False
            for f in (self._pose_file, self._raw_file):
                if f is not None:
                    try:
                        f.flush()
                        f.close()
                    except Exception:
                        pass
            self._pose_file = self._raw_file = None
            self._pose_writer = self._raw_writer = None

    def stop(self):
        self.stop_recording()
        self._stop.set()
        if self._fusion_thread is not None:
            self._fusion_thread.join(timeout=2.0)
        for c in self.cams:
            c.stop()
        for c in self.cams:
            c.join(timeout=2.0)

    def config_dict(self):
        """For folding into the session meta.json so the recording is self-describing."""
        return {
            "apriltag_serials": [c.serial for c in self.cams],
            "apriltag_bundle": self.bundle_path,
            "apriltag_family": self.family,
            "apriltag_tag_size": self.tag_size,
            "apriltag_anchor_id": self.anchor_id,
            "apriltag_table_id": self.table_id,
            "apriltag_world_from_table": self.T_world_table.tolist(),
            "apriltag_T_world_cam": {
                s: (T.tolist() if T is not None else None)
                for s, T in self.T_world_cam.items()},
        }


# ── Standalone bring-up / debug / record ─────────────────────────────────────

def _debug_compose(cams, axis_len):
    panels = []
    for c in cams:
        snap = c.latest()
        if snap is None or "image" not in snap:
            panel = np.full((360, 480, 3), 40, np.uint8)
            cv2.putText(panel, f"{c.serial}: no frame", (12, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)
        else:
            panel = snap["image"].copy()
            for d in snap.get("dets", []):
                draw_detection(panel, d, c.K, c.dist, axis_len)
            cv2.putText(panel, c.serial, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)
        h = 360
        panel = cv2.resize(panel, (int(panel.shape[1] * h / panel.shape[0]), h))
        panels.append(panel)
    return cv2.hconcat(panels) if panels else None


# ── Separate-process server (own GIL, no teleop contention) ──────────────────
# The manager runs the tracker as its OWN process and talks to it over a socketpair
# using length-prefixed JSON. This keeps the heavy AprilTag detection off the manager's
# GIL so the single teleop IK thread is never starved. Protocol:
#   manager → tracker : one {"init": {...}} then {"cmd": ...} messages
#   tracker → manager : {"hello"|"error": ...} once, then {"status": ...} @ ~8 Hz +
#                       {"reply": id, ...} for each command that expects one.

def _json_default(o):
    """Make numpy types JSON-serializable. A fused crate pose carries numpy arrays/scalars
    (p, q from T_to_pq); without this, json.dumps raises the moment a pose exists, which
    (through the command loop's except) silently kills the tracker and hangs the manager's
    next RPC — the "anchor timed out" bug. Last-resort stringify instead of raising."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _send_msg(sock, obj):
    blob = json.dumps(obj, default=_json_default).encode("utf-8")
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("tracker IPC socket closed")
        buf += chunk
    return buf


def _recv_msg(sock):
    (length,) = struct.unpack(">I", _recv_exact(sock, 4))
    return json.loads(_recv_exact(sock, length).decode("utf-8"))


def _status_dict(tracker):
    """Lightweight status for the manager's proxy: per-camera health + a trimmed fused
    pose (no heavy T/raw arrays — the subprocess records those itself)."""
    cams = []
    for c in tracker.cams:
        try:
            snap = c.latest()
        except Exception:
            snap = None
        cams.append({
            "serial": c.serial,
            "started_ok": bool(getattr(c, "started_ok", False)),
            "error": getattr(c, "error", None),
            "stamp": float(snap.get("stamp", 0.0)) if snap else 0.0,
            "anchored": tracker.T_world_cam.get(c.serial) is not None,
        })
    # The peer's head cam (over the study link) is a self-anchored, world-frame
    # contributor — report it as a 4th already-anchored crate camera while it's active,
    # so the anchor/calibration count reads 4/4. Its ":" serial keeps it out of the
    # "external RealSense" status lights (same as the local head-cam extra).
    if tracker.remote_active():
        cams.append({
            "serial": tracker._remote_serial,
            "started_ok": True, "error": None,
            "stamp": float(tracker._remote_last_stamp), "anchored": True,
        })
    lt = tracker.latest()
    latest = None
    if lt:
        latest = {k: lt.get(k) for k in
                  ("stamp", "ok", "p", "q", "n_cams", "n_tags", "intercam_trans_mm")}
        # p/q come straight from T_to_pq → numpy arrays; make them plain lists so the JSON
        # send can never raise (which would kill the command loop and hang the next RPC).
        for k in ("p", "q"):
            v = latest.get(k)
            if v is not None:
                latest[k] = np.asarray(v).tolist()
    return {"anchored": bool(tracker.is_anchored), "cams": cams, "latest": latest}


def serve(fd):
    """Run the tracker in this process, driven over the inherited socket ``fd``."""
    silence_native_logs()
    sock = socket.socket(fileno=fd)
    tracker = None
    try:
        init = _recv_msg(sock)["init"]
        extra_cams = None
        hc = init.get("headcam")
        if hc:
            try:
                from head_cam_localize import _NetCameraWorker
                # The head cam gets its OWN decimation (default 1.0 = full res), NOT the
                # RealSense cameras' quad_decimate: the RealSense crate cams sit close so 2.0
                # is fine, but the head cam sees table tag 0 from farther / a steeper angle,
                # where 2.0 downsamples it below detection (it then anchors "NO TABLE TAG"
                # while the head-cam LOCALIZER, which runs at 1.0, sees it). Match the
                # localizer so the head cam anchors as the 3rd crate camera.
                extra_cams = [_NetCameraWorker(
                    hc["host"], hc["port"], family="tag36h11",
                    tag_size=hc.get("tag_size"), dm_thresh=30.0,
                    quad_decimate=float(hc.get("quad_decimate", 1.0)),
                    nthreads=2, keep_image=False,
                    # Cap detection at the fusion rate — the Jetson streams ~30 fps but
                    # fusion only consumes rate_hz; detecting the excess is wasted CPU.
                    detect_hz=float(init.get("rate_hz", 15.0)))]
            except Exception as exc:                      # head cam optional
                _send_msg(sock, {"log": f"[crate-proc] head cam not added: {exc}"})
        tracker = AprilTagTracker(
            init["serials"], init["bundle"],
            table_id=int(init.get("table_id", 0)),
            world_from_table=init.get("world_from_table"),
            tag_size=init.get("tag_size"),
            width=int(init.get("width", 1280)), height=int(init.get("height", 720)),
            fps=int(init.get("fps", 15)), dm_thresh=float(init.get("dm_thresh", 30.0)),
            quad_decimate=float(init.get("quad_decimate", 2.0)),
            rate_hz=float(init.get("rate_hz", 15.0)), extra_cams=extra_cams)
        tracker.start()
    except Exception as exc:
        try:
            _send_msg(sock, {"error": f"{exc}"})
        except Exception:
            pass
        return
    _send_msg(sock, {"hello": {
        "tag_size": tracker.tag_size, "anchor_id": tracker.anchor_id,
        "serials": [c.serial for c in tracker.cams], "config": tracker.config_dict()}})

    # Status streamer (~8 Hz). A dedicated send lock keeps status + replies from
    # interleaving on the wire.
    send_lock = threading.Lock()
    stop = threading.Event()

    def _safe_send(obj):
        with send_lock:
            _send_msg(sock, obj)

    def _stream():
        while not stop.is_set():
            try:
                _safe_send({"status": _status_dict(tracker)})
            except Exception:
                # Never permanently kill the status feed on a transient hiccup — a truly
                # dead socket is handled by the command loop exiting (→ stop.set()). The
                # 0.12 s pace below keeps this from spinning while the link tears down.
                pass
            stop.wait(0.12)
    threading.Thread(target=_stream, daemon=True, name="crate-status").start()

    try:
        while True:
            msg = _recv_msg(sock)        # connection death raises → break out to cleanup
            cmd = msg.get("cmd")
            if cmd == "shutdown":
                break
            # A single command handler must NEVER kill the whole tracker. On error, reply
            # with the failure (so the manager's RPC fails fast instead of hanging its full
            # timeout — the "anchor timed out" symptom) and keep serving.
            try:
                if cmd == "anchor":
                    res = tracker.anchor_cameras(
                        duration_s=float(msg.get("duration_s", 1.5)),
                        serials=msg.get("serials"),
                        keep_on_fail=bool(msg.get("keep_on_fail", False)))
                    # Push the post-anchor status BEFORE the reply so the manager's cached
                    # T_world_cam is up to date the instant anchor_cameras() returns (the
                    # socket is ordered, so the proxy applies this status before unblocking).
                    _safe_send({"status": _status_dict(tracker)})
                    _safe_send({"reply": msg["id"], "ok": True,
                                "results": {s: [st, n] for s, (st, n) in res.items()}})
                elif cmd == "start_recording":
                    paths = tracker.start_recording(msg["dir"], msg.get("label", ""))
                    _safe_send({"reply": msg["id"], "ok": True, "paths": list(paths)})
                elif cmd == "stop_recording":
                    tracker.stop_recording()
                    _safe_send({"reply": msg["id"], "ok": True})
                elif cmd == "remote_candidate":
                    tracker.add_remote_candidate(
                        msg["p"], msg["q"], msg.get("weight", 1.0), msg.get("stamp"))
            except Exception as exc:
                rid = msg.get("id")
                if rid is not None:
                    try:
                        _safe_send({"reply": rid, "ok": False, "error": str(exc)})
                    except Exception:
                        pass
                try:
                    _safe_send({"log": f"[crate-proc] command {cmd!r} failed: {exc}"})
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        stop.set()
        try:
            tracker.stop()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass


def build_argparser():
    p = argparse.ArgumentParser(description="Runtime crate-pose tracker.")
    p.add_argument("--serve", action="store_true",
                   help="Run as the manager's subprocess tracker (IPC over --fd).")
    p.add_argument("--fd", type=int, default=None,
                   help="Inherited socket fd for --serve mode.")
    p.add_argument("--list", action="store_true", help="List RealSense cameras and exit.")
    p.add_argument("--serials", nargs="*", default=None,
                   help="RealSense serials to use (default: first two found).")
    p.add_argument("--bundle", type=str, default="crate_bundle.json",
                   help="Crate bundle from apriltag_crate_calibrate.py.")
    p.add_argument("--tag-size", "--tag_size", dest="tag_size", type=float, default=None,
                   help="Override tag size (default: read from bundle).")
    p.add_argument("--table-id", dest="table_id", type=int, default=0)
    p.add_argument("--rate", type=float, default=30.0, help="Fusion rate Hz.")
    p.add_argument("--anchor-duration", dest="anchor_duration", type=float, default=1.5)
    p.add_argument("--no-anchor", dest="no_anchor", action="store_true",
                   help="Skip anchoring (poses will be in camera frame; for debug only).")
    p.add_argument("--debug-view", dest="debug_view", action="store_true")
    p.add_argument("--record", type=str, default=None,
                   help="Directory to record crate_pose/raw CSVs into.")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--quad-decimate", dest="quad_decimate", type=float, default=1.0)
    p.add_argument("--decision-margin", dest="dm", type=float, default=30.0)
    return p


def main():
    args = build_argparser().parse_args()
    if args.serve:
        if args.fd is None:
            sys.exit("--serve requires --fd")
        serve(args.fd)
        return
    silence_native_logs()   # mute the native "more than one minima" spam

    devices = list_devices()
    if args.list or not devices:
        print("No RealSense cameras found." if not devices else "Connected RealSense cameras:")
        for name, serial in devices:
            print(f"  {name}  serial={serial}")
        return

    serials = args.serials or [s for _, s in devices][:2]
    if not os.path.exists(args.bundle):
        sys.exit(f"Bundle not found: {args.bundle} — run apriltag_crate_calibrate.py first.")

    tracker = AprilTagTracker(
        serials, args.bundle, table_id=args.table_id, tag_size=args.tag_size,
        width=args.width, height=args.height, fps=args.fps, dm_thresh=args.dm,
        quad_decimate=args.quad_decimate, rate_hz=args.rate, debug_view=args.debug_view)
    print(f"Cameras: {serials}   bundle: {args.bundle}   "
          f"tag_size={tracker.tag_size * 1000:.0f}mm   anchor_id={tracker.anchor_id}")
    tracker.start()

    ok, msgs = tracker.wait_ready()
    for m in msgs:
        print("  " + m)
    if not ok:
        print("Some cameras failed — continuing with whatever started.")

    if not args.no_anchor:
        print("Anchoring (both cameras must see the table tag)...")
        for serial, (status, n) in tracker.anchor_cameras(args.anchor_duration).items():
            flag = "OK" if status == "ok" else "** NO TABLE TAG **"
            print(f"  {serial}: {flag} ({n} obs)")
        if not tracker.is_anchored:
            print("  No camera anchored — crate pose cannot be put in the table frame.")

    rec_paths = None
    if args.record:
        rec_paths = tracker.start_recording(args.record, args.label)
        print(f"Recording -> {rec_paths[0]}\n            {rec_paths[1]}")

    win = "AprilTag tracker (q quit, a re-anchor)"
    if args.debug_view:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    axis_len = tracker.tag_size * 0.5
    last_print = 0.0
    try:
        while True:
            res = tracker.latest()
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                if res and res.get("ok"):
                    p = res["p"]
                    ic = ("" if res["intercam_trans_mm"] is None
                          else f"  intercam {res['intercam_trans_mm']:.1f}mm/"
                               f"{res['intercam_rot_deg']:.2f}deg")
                    print(f"\rcrate p=[{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]  "
                          f"{res['n_tags']} tag(s) {res['cams']}  "
                          f"resid {res['rot_resid_deg']:.2f}deg/"
                          f"{res['trans_resid_mm']:.1f}mm{ic}      ", end="", flush=True)
                else:
                    print("\rcrate: LOST (no crate tags visible)            ",
                          end="", flush=True)

            if args.debug_view:
                frame = _debug_compose(tracker.cams, axis_len)
                if frame is not None:
                    if res and res.get("ok"):
                        txt = (f"crate {res['n_tags']}tag {res['cams']}  "
                               f"resid {res['rot_resid_deg']:.1f}deg/{res['trans_resid_mm']:.0f}mm")
                        col = (0, 255, 0)
                    else:
                        txt, col = "crate LOST", (0, 0, 255)
                    cv2.putText(frame, txt, (12, frame.shape[0] - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("a"):
                    print("\nre-anchoring...")
                    print("  " + str(tracker.anchor_cameras(args.anchor_duration)))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        tracker.stop()
        if args.debug_view:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
