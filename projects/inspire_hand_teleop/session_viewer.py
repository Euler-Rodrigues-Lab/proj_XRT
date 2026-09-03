#!/usr/bin/env python3
"""Synced session viewer for G1 + Inspire-hand user-study recordings.

Given a session directory recorded by ``XRT_user_study_manager.py`` (the unified
``Start Recording``), this plays back **everything raw, in sync**:

  * a MuJoCo passive viewer showing the robot (legs + torso + arms + fingers)
    posed from the recorded joint angles, with
      - end-effector **wrench** force arrows at each palm (estimated from the
        recorded joint torques, quasi-static — see wrench_estimation.py), and
      - the **tactile** palm/fingertip force overlay (markers + arrows), exactly
        like demo_g1_inspire_sensor_vis_webrtc_haptics_hw.py; and
  * an OpenCV window showing the recorded **ZED left/right videos** side by side,
    time-aligned to the robot playback.

Controls (focus the OpenCV window):
  space = play/pause   a / d = scrub -1s / +1s   , / . = slower / faster
  r = restart          q or ESC = quit

A session folder contains:  robot_raw_*.csv (+ .meta.json), tactile_*.bin,
zed_left_*.mp4, zed_right_*.mp4, body_pose_data_*.csv.

Examples
--------
  python session_viewer.py /path/to/XRT_recordings_20260624_1742
  python session_viewer.py session_dir --no-video          # MuJoCo only
  python session_viewer.py session_dir --known-mass 500     # accuracy report
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wrench_estimation as we  # noqa: E402

try:
    import cv2
except Exception:                       # noqa: BLE001
    cv2 = None


# --------------------------------------------------------------------------- #
# Tactile overlay configuration — mirrors demo_g1_inspire_sensor_vis_webrtc_haptics_hw.py
# --------------------------------------------------------------------------- #
def _plus_x_override() -> dict:
    return {
        "axis": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "frame": np.eye(3, dtype=np.float64),
    }


def _sensor_direction_overrides() -> dict:
    overrides: dict = {}
    for hand in ("left", "right"):
        overrides[f"{hand}_palm_force_sensor"] = _plus_x_override()
        overrides[f"{hand}_thumb_force_sensor_1"] = _plus_x_override()
        overrides[f"{hand}_thumb_force_sensor_3"] = _plus_x_override()
        for finger in ("little", "ring", "middle", "index"):
            overrides[f"{hand}_{finger}_force_sensor_1"] = _plus_x_override()
            overrides[f"{hand}_{finger}_force_sensor_2"] = _plus_x_override()
    return overrides


_LIVE_ARROW_CFG = {
    "normalization_mode": "running",
    "smoothing_alpha": 0.35,
    "deadband": 0.02,
    "fixed_max": 1024.0,
    "running_decay": 0.995,
}


# --------------------------------------------------------------------------- #
# Session loading
# --------------------------------------------------------------------------- #
def find_session(path: str) -> dict:
    """Resolve a session dir (or a robot_raw CSV) into its component files."""
    p = Path(path)
    if p.is_dir():
        csvs = sorted(p.glob("robot_raw*.csv"))
        if not csvs:
            csvs = sorted(p.glob("*.csv"))
            csvs = [c for c in csvs if "body_pose" not in c.name]
        if not csvs:
            raise FileNotFoundError(f"no robot_raw*.csv found in {p}")
        csv_path = csvs[-1]
        session_dir = p
    else:
        csv_path = p
        session_dir = p.parent

    meta = we.load_capture_meta(str(csv_path))
    cap = we.read_capture_csv(str(csv_path))

    # Tactile sidecar
    tactile = None
    tac_name = meta.get("tactile_file")
    tac_path = session_dir / tac_name if tac_name else None
    if tac_path and tac_path.exists() and meta.get("tactile_width"):
        tt, vv = we.read_tactile_bin(str(tac_path), int(meta["tactile_width"]))
        if tt.size:
            tactile = {"t": tt, "v": vv, "layout": meta.get("tactile_layout", [])}

    # ZED videos
    def _find_video(meta_key, glob_pat):
        name = meta.get(meta_key)
        if name and (session_dir / name).exists():
            return str(session_dir / name)
        hits = sorted(glob.glob(str(session_dir / glob_pat)))
        return hits[-1] if hits else None

    left_video = _find_video("zed_left", "zed_left*.mp4")
    right_video = _find_video("zed_right", "zed_right*.mp4")
    # Current recordings are ONE side-by-side stereo file (zed_stereo_*.mp4); the
    # separate left/right files are the legacy format.
    stereo_video = _find_video("zed_stereo", "zed_stereo*.mp4")

    # AprilTag crate pose (world/table frame), if this session recorded it.
    crate_csvs = sorted(session_dir.glob("crate_pose*.csv"))
    crate_csv = str(crate_csvs[-1]) if crate_csvs else None

    return {
        "csv_path": str(csv_path),
        "session_dir": str(session_dir),
        "meta": meta,
        "cap": cap,
        "tactile": tactile,
        "left_video": left_video,
        "right_video": right_video,
        "stereo_video": stereo_video,
        "crate_csv": crate_csv,
    }


# --------------------------------------------------------------------------- #
# Wrench computation (per frame, on demand)
# --------------------------------------------------------------------------- #
class WrenchSource:
    """Lazily estimates per-frame EE wrench for both hands from a loaded cap."""

    def __init__(self, cap: dict, meta: dict, model_path: str):
        self._cap = cap
        loff = np.asarray(meta["left_offset"]) if meta.get("left_offset") else None
        roff = np.asarray(meta["right_offset"]) if meta.get("right_offset") else None
        self._est = we.WrenchEstimator(model_path, left_offset=loff, right_offset=roff)
        base = meta.get("wrench_baseline") or {}
        self._baseline = {
            s: np.asarray(base.get(s, np.zeros(6)), dtype=np.float64) for s in ("left", "right")
        }
        self._have = {s: (s in cap and "q" in cap[s] and "tau" in cap[s])
                      for s in ("left", "right")}
        self._q_torso = cap["torso"]["q"] if ("torso" in cap and "q" in cap["torso"]) else None

    def at(self, k: int, side: str):
        """Return (wrench (6,) baseline-subtracted, palm point (3,)) or (None,None)."""
        if not self._have.get(side):
            return None, None
        cap = self._cap
        other = "right" if side == "left" else "left"
        q_other = cap[other]["q"][k] if self._have.get(other) else None
        q_torso = self._q_torso[k] if self._q_torso is not None else None
        r = self._est.estimate(side, cap[side]["q"][k], cap[side]["tau"][k],
                               q_torso=q_torso, q_other_arm=q_other)
        return r.wrench - self._baseline[side], r.point


def accuracy_report(wrench_src: WrenchSource, cap: dict, meta: dict):
    mass_g = meta.get("known_mass_g")
    if not mass_g:
        return
    side = meta.get("wrench_side", "both")
    sides = ["left", "right"] if side in (None, "both") else [side]
    N = cap["t"].size
    print(f"\n── known-mass accuracy ({mass_g:.0f} g) ──")
    for s in sides:
        if not wrench_src._have.get(s):
            continue
        # still frames only (low arm speed) if dq present
        if "dq" in cap.get(s, {}):
            speed = np.max(np.abs(cap[s]["dq"]), axis=1)
            idx = np.where(speed < 0.05)[0]
        else:
            idx = np.arange(N)
        if idx.size == 0:
            idx = np.arange(N)
        forces = np.array([wrench_src.at(k, s)[0][:3] for k in idx])
        rep = we.accuracy_report(forces, mass_g)
        print(f"  {s:>5}: |F| {rep['mean_mag_N']:.3f} ± {rep['std_mag_N']:.3f} N  "
              f"(expected {rep['expected_mag_N']:.3f})  "
              f"err {rep['mag_error_pct']:+.1f}%  dir {rep['direction_error_deg']:.1f}°  "
              f"[{idx.size} still frames]")


# --------------------------------------------------------------------------- #
# Crate (AprilTag) pose source
# --------------------------------------------------------------------------- #
class CrateSource:
    """Loads crate_pose_*.csv (world/table-frame crate pose) and returns the
    pose nearest a given absolute epoch. Only frames the tracker fused are in the
    file, so a gap larger than ``max_gap_s`` means the crate was untracked then."""

    def __init__(self, csv_path, max_gap_s=0.3):
        self._t = np.empty(0)
        self._p = np.empty((0, 3))
        self._q = np.empty((0, 4))
        self._max_gap = float(max_gap_s)
        self.n = 0
        if not csv_path or not os.path.exists(csv_path):
            return
        try:
            # cols 0..7 = timestamp, px,py,pz, qx,qy,qz,qw  (later cols are strings)
            arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1,
                                usecols=range(8), dtype=float)
        except Exception:
            return
        if arr.ndim == 1:                    # single data row
            arr = arr.reshape(1, -1) if arr.size >= 8 else arr.reshape(0, 8)
        if arr.shape[0] == 0:
            return
        arr = arr[~np.any(np.isnan(arr), axis=1)]
        if arr.shape[0] == 0:
            return
        self._t, self._p, self._q = arr[:, 0], arr[:, 1:4], arr[:, 4:8]
        self.n = int(arr.shape[0])

    @property
    def available(self) -> bool:
        return self._t.size > 0

    def at(self, epoch):
        """Return (pos (3,), quat [x,y,z,w]) nearest ``epoch``, or None if the
        crate was untracked around then."""
        if self._t.size == 0:
            return None
        i = int(np.searchsorted(self._t, epoch))
        i = min(max(i, 0), self._t.size - 1)
        if i > 0 and abs(self._t[i - 1] - epoch) < abs(self._t[i] - epoch):
            i -= 1
        if abs(self._t[i] - epoch) > self._max_gap:
            return None
        return self._p[i], self._q[i]


# --------------------------------------------------------------------------- #
# MuJoCo scene helpers
# --------------------------------------------------------------------------- #
def _quat_to_mat(q):
    """Quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([[1 - (yy + zz), xy - wz,       xz + wy],
                     [xy + wz,       1 - (xx + zz), yz - wx],
                     [xz - wy,       yz + wx,       1 - (xx + yy)]])


def _add_arrow(scene, p_from, p_to, width, rgba):
    import mujoco
    if scene.ngeom + 1 >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_ARROW, size=np.zeros(3),
                        pos=np.zeros(3), mat=np.eye(3).flatten(),
                        rgba=np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(p_from, dtype=np.float64),
                         np.asarray(p_to, dtype=np.float64))
    scene.ngeom += 1


def _add_sphere(scene, p, r, rgba):
    import mujoco
    if scene.ngeom + 1 >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([r, r, r]), pos=np.asarray(p, dtype=np.float64),
                        mat=np.eye(3).flatten(), rgba=np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def _add_box(scene, p, mat, half, rgba):
    import mujoco
    if scene.ngeom + 1 >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=np.asarray(half, dtype=np.float64),
                        pos=np.asarray(p, dtype=np.float64),
                        mat=np.asarray(mat, dtype=np.float64).flatten(),
                        rgba=np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def _add_triad(scene, p, mat, length, width):
    """Draw a 3-axis triad (X=red, Y=green, Z=blue) for pose (p, mat)."""
    p = np.asarray(p, dtype=np.float64)
    mat = np.asarray(mat, dtype=np.float64)
    for col, rgba in ((0, (1.0, 0.25, 0.25, 1.0)),
                      (1, (0.25, 1.0, 0.25, 1.0)),
                      (2, (0.3, 0.5, 1.0, 1.0))):
        _add_arrow(scene, p, p + mat[:, col] * length, width, rgba)


# --------------------------------------------------------------------------- #
# Video sync
# --------------------------------------------------------------------------- #
class VideoPair:
    """ZED capture(s), composited and seekable by epoch time. Either the legacy separate
    left/right files, or — the current format — ONE side-by-side stereo recording
    (``stereo_path``), whose frames already contain L|R and need no compositing."""

    def __init__(self, left_path, right_path, start_epoch, stereo_path=None):
        self.start_epoch = float(start_epoch) if start_epoch else None
        self.caps = []
        self.labels = []
        self.stereo = False
        if stereo_path and os.path.exists(stereo_path):
            cap = cv2.VideoCapture(stereo_path)
            self.caps.append(cap if cap.isOpened() else None)
            self.labels.append("ZED L|R")
            self.stereo = True
        else:
            for path, lbl in ((left_path, "LEFT"), (right_path, "RIGHT")):
                cap = cv2.VideoCapture(path) if (path and os.path.exists(path)) else None
                self.caps.append(cap)
                self.labels.append(lbl)
        self.available = any(c is not None and c.isOpened() for c in self.caps)

    def frame_at(self, epoch):
        """Composite frame for absolute time ``epoch``; None if no video yet."""
        if self.start_epoch is None:
            tsec = None
        else:
            tsec = epoch - self.start_epoch
        imgs = []
        for cap in self.caps:
            if cap is None or not cap.isOpened():
                imgs.append(None)
                continue
            if tsec is not None and tsec >= 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000.0)
            ok, fr = cap.read()
            imgs.append(fr if ok else None)
        labels = self.labels
        # The ZED stream-type=2 composite is stacked TOP-BOTTOM (left eye above right —
        # e.g. 1280x1440 for HD720). Split a tall stereo frame and show the eyes SIDE BY
        # SIDE (LEFT | RIGHT), like the legacy two-file layout.
        if self.stereo and imgs and imgs[0] is not None:
            fr = imgs[0]
            if fr.shape[0] > fr.shape[1]:
                h2 = fr.shape[0] // 2
                imgs = [fr[:h2], fr[h2:]]
                labels = ["LEFT", "RIGHT"]
        return self._compose(imgs, tsec, labels)

    def _compose(self, imgs, tsec, labels=None):
        h = 480
        panels = []
        for img, lbl in zip(imgs, labels if labels is not None else self.labels):
            if img is None:
                panel = np.zeros((h, h * 16 // 9, 3), dtype=np.uint8)
                msg = "waiting…" if (tsec is not None and tsec < 0) else "no video"
                cv2.putText(panel, f"{lbl} {msg}", (20, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
            else:
                scale = h / img.shape[0]
                panel = cv2.resize(img, (int(img.shape[1] * scale), h))
                cv2.putText(panel, lbl, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (40, 220, 40), 2)
            panels.append(panel)
        return np.hstack(panels) if panels else None

    def release(self):
        for cap in self.caps:
            if cap is not None:
                cap.release()


# --------------------------------------------------------------------------- #
# Main playback
# --------------------------------------------------------------------------- #
def run(session: dict, args):
    import mujoco
    import mujoco.viewer
    from inspire_hand_vis_utils import (
        create_mujoco_sensor_overlay_runtime,
        build_mujoco_sensor_overlay_values_from_snapshot,
        prepare_live_scalar_arrow_values,
        render_mujoco_sensor_overlay,
    )

    cap = session["cap"]
    meta = session["meta"]
    tactile = session["tactile"]
    model_path = args.model or meta.get("model_path") or we.DEFAULT_MODEL_PATH

    t_rel = cap["t"]
    t_epoch = cap.get("t_epoch", t_rel)
    N = t_rel.size
    duration = float(t_rel[-1]) if N else 0.0
    print(f"Loaded {N} frames ({duration:.1f}s) from "
          f"{Path(session['csv_path']).name}.  model={Path(model_path).name}")
    print(f"  tactile: {'yes' if tactile else 'no'}   "
          f"video: stereo={'yes' if session.get('stereo_video') else 'no'} "
          f"L={'yes' if session['left_video'] else 'no'} "
          f"R={'yes' if session['right_video'] else 'no'}")

    wrench_src = WrenchSource(cap, meta, model_path)
    accuracy_report(wrench_src, cap, meta)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    def addrs(names):
        return [int(model.joint(n).qposadr[0]) for n in names]

    # Legs are recorded down to the knees (hip+knee); ankles stay at default.
    seg_addr = {}
    seg_names = {"legs": we.LEG_JOINTS, "torso": we.TORSO_JOINTS,
                 "left": we.LEFT_ARM_JOINTS, "right": we.RIGHT_ARM_JOINTS}
    for seg, names in seg_names.items():
        if seg in cap and "q" in cap[seg]:
            try:
                seg_addr[seg] = addrs(names)
            except Exception:
                pass
    finger_addr = we.hand_qpos_addrs(model)
    has_fingers = "finger_left" in cap and "finger_right" in cap

    # tactile overlay runtime + per-csv-frame tactile index
    overlay = None
    tac_idx = None
    if tactile is not None:
        try:
            overlay = create_mujoco_sensor_overlay_runtime(
                model, sensor_direction_overrides=_sensor_direction_overrides())
            overlay["running_max"] = 0.0
            overlay["smoothed_values"] = None
            tac_idx = np.clip(np.searchsorted(tactile["t"], t_epoch), 0,
                              tactile["v"].shape[0] - 1)
            print(f"  tactile overlay: {overlay['count']} sensor points")
        except Exception as exc:                         # noqa: BLE001
            print(f"  tactile overlay unavailable: {exc}")
            overlay = None

    # video
    video = None
    if not args.no_video and cv2 is not None:
        video = VideoPair(session["left_video"], session["right_video"],
                          meta.get("zed_record_start_epoch"),
                          stereo_path=session.get("stereo_video"))
        if not video.available:
            print("  (no ZED video opened — robot playback only)")
    win = "G1 session — ZED video + EE wrench/tactile (space=play a/d=scrub ,/.=speed q=quit)"
    if cv2 is not None:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Crate (AprilTag) overlay
    crate_src = None
    if not args.no_crate:
        crate_src = CrateSource(session.get("crate_csv"))
        if crate_src.available:
            print(f"  crate: {crate_src.n} tracked frames "
                  f"({Path(session['crate_csv']).name})")
        else:
            crate_src = None
            print("  crate: none (no crate_pose CSV in this session)")
    # The crate pose is recorded in the TABLE frame; the robot here is in its own
    # base frame. Until the robot<->table transform is solved (digital-twin stage)
    # this stays identity, so the crate is drawn in the table frame (its trajectory
    # is correct; its placement relative to the robot is not yet). Plug the
    # T_mujocoworld_table transform in here later.
    T_align = np.eye(4)
    crate_half = np.asarray(args.crate_size, dtype=np.float64) * 0.5

    fs = args.force_scale

    def render_frame(k):
        data.qpos[:] = model.qpos0
        for seg, ad in seg_addr.items():
            q = cap[seg]["q"][k]
            for i, a in enumerate(ad):
                if i < len(q):
                    data.qpos[a] = q[i]
        if has_fingers:
            data.qpos[finger_addr["finger_left"]] = cap["finger_left"][k]
            data.qpos[finger_addr["finger_right"]] = cap["finger_right"][k]
        mujoco.mj_forward(model, data)

        # tactile overlay first (it resets user_scn.ngeom = 0), then wrench arrows
        if overlay is not None:
            snap = we.reconstruct_touch_snapshot(tactile["v"][tac_idx[k]], tactile["layout"])
            raw = build_mujoco_sensor_overlay_values_from_snapshot(
                snap, overlay["sensor_points"], normalize=False)
            vals = prepare_live_scalar_arrow_values(raw, overlay, **_LIVE_ARROW_CFG)
            render_mujoco_sensor_overlay(viewer, data, overlay["sensor_points"], vals)
        else:
            viewer.user_scn.ngeom = 0

        fmags = {}
        for side in ("left", "right"):
            w, p = wrench_src.at(k, side)
            if w is None:
                continue
            F = w[:3]
            fmag = float(np.linalg.norm(F))
            fmags[side] = fmag
            a = min(1.0, fmag / max(1e-6, args.force_full))
            _add_sphere(viewer.user_scn, p, 0.012, (0.2, 0.4, 1.0, 0.9))
            if fmag > 1e-3:
                _add_arrow(viewer.user_scn, p, p + F * fs, 0.012,
                           (a, 1.0 - a, 0.1, 0.9))

        # Crate (AprilTag) pose in the table frame, synced by absolute epoch.
        crate_tracked = False
        if crate_src is not None:
            R0, o0 = T_align[:3, :3], T_align[:3, 3]
            _add_triad(viewer.user_scn, o0, R0, 0.15, 0.006)   # table/world origin
            pose = crate_src.at(t_epoch[k])
            if pose is not None:
                p_c, q_c = pose
                R_c = R0 @ _quat_to_mat(q_c)
                p_v = R0 @ p_c + o0
                _add_box(viewer.user_scn, p_v, R_c, crate_half, (0.95, 0.6, 0.1, 0.33))
                _add_triad(viewer.user_scn, p_v, R_c, 0.10, 0.006)
                crate_tracked = True
        return fmags, crate_tracked

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.2, 0.0, 0.95]

        play = True
        speed = float(args.speed)
        tp = 0.0
        last = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            dt = now - last
            last = now
            if play:
                tp += dt * speed
                if tp >= duration:
                    tp = duration if not args.loop else 0.0
            tp = min(max(tp, 0.0), duration)
            k = int(np.searchsorted(t_rel, tp, side="right")) - 1
            k = min(max(k, 0), N - 1)

            fmags, crate_tracked = render_frame(k)
            viewer.sync()

            # video / HUD window
            if cv2 is not None:
                frame = video.frame_at(t_epoch[k]) if (video and video.available) else None
                if frame is None:
                    frame = np.zeros((200, 900, 3), dtype=np.uint8)
                hud = (f"t={tp:6.2f}/{duration:.1f}s  frame {k+1}/{N}  "
                       f"{'PLAY' if play else 'PAUSE'} x{speed:.2f}  "
                       f"|F| L={fmags.get('left', 0):.1f} R={fmags.get('right', 0):.1f} N")
                if crate_src is not None:
                    hud += f"  crate:{'ok' if crate_tracked else '—'}"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
                cv2.putText(frame, hud, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (230, 230, 230), 1)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord(' '):
                    play = not play
                elif key == ord('a'):
                    tp = max(0.0, tp - 1.0); play = False
                elif key == ord('d'):
                    tp = min(duration, tp + 1.0); play = False
                elif key == ord(','):
                    speed = max(0.1, speed / 1.5)
                elif key == ord('.'):
                    speed = min(8.0, speed * 1.5)
                elif key == ord('r'):
                    tp = 0.0
                # window closed?
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                time.sleep(max(0.0, 1.0 / 60.0 - (time.monotonic() - now)))

        if video is not None:
            video.release()
        if cv2 is not None:
            cv2.destroyAllWindows()


def main(argv=None):
    p = argparse.ArgumentParser(description="Synced G1 session viewer "
                                "(ZED video + MuJoCo robot + EE wrench + tactile).")
    p.add_argument("session", help="session directory (or a robot_raw*.csv path)")
    p.add_argument("--model", default=None, help="override MuJoCo XML path")
    p.add_argument("--known-mass", type=float, default=None,
                   help="known mass (g) for the accuracy report (else from meta)")
    p.add_argument("--no-video", action="store_true", help="skip ZED video playback")
    p.add_argument("--force-scale", type=float, default=0.02,
                   help="wrench arrow length (m) per Newton")
    p.add_argument("--force-full", type=float, default=10.0,
                   help="force (N) mapped to a fully-red arrow")
    p.add_argument("--speed", type=float, default=1.0, help="initial playback speed")
    p.add_argument("--loop", action="store_true", help="loop playback")
    p.add_argument("--no-crate", action="store_true",
                   help="skip the AprilTag crate overlay")
    p.add_argument("--crate-size", type=float, nargs=3, default=(0.3, 0.3, 0.3),
                   metavar=("LX", "LY", "LZ"),
                   help="crate box full extents (m) for the overlay (default 0.3 0.3 0.3)")
    args = p.parse_args(argv)

    if not os.path.exists(args.session):
        p.error(f"not found: {args.session}")

    session = find_session(args.session)
    if args.known_mass is not None:
        session["meta"]["known_mass_g"] = args.known_mass
    if cv2 is None and not args.no_video:
        print("[viewer] OpenCV (cv2) unavailable — running MuJoCo only.")
        args.no_video = True
    run(session, args)


if __name__ == "__main__":
    main()
