#!/usr/bin/env python3
"""Localize the ZED cameras in the TABLE (world) frame from the recorded video.

The ZED sits on the moving neck, so its pose changes every frame — but the table
AprilTag (id 0) is fixed, so wherever tag 0 is visible in the ZED image we can recover
the ZED's pose directly:

    T_world_zed = world_from_table · inv(T_zed_tag0)

CAMERA AIMING (for recording): keep each ZED angled DOWNWARD ~60° so the table tag (id 0)
— and ideally the crate tags — sit clearly in its field of view. The ZED is on the neck,
so tag 0 will drop out of frame whenever the neck looks away; the more it's tilted down at
the workspace, the more frames get a pose (gaps are interpolated by the dataset builder).

This runs OFFLINE on the recorded ZED video (the crate + table are already tracked live
by the fixed RealSense cameras — this only adds the ZED's own extrinsic). It uses the
LEFT or RIGHT camera per the session meta's ``zed_localize_side`` (the peer records
'right', since the table tag sits to its right), or ``--camera``. It writes
``zed_pose_<side>_*.csv`` next to the video with, per frame where tag 0 is seen:

    timestamp, px, py, pz, qx, qy, qz, qw, decision_margin

(the ZED-left optical frame pose in the world/table frame; quaternion xyzw). Frames
where tag 0 is out of view get no row — the dataset builder interpolates across gaps.

You must supply the ZED-left intrinsics at the RECORDED resolution (fx, fy, cx, cy),
either directly or from the ZED factory conf (SN<serial>.conf):

  python zed_localize.py <session_dir> --fx 700 --fy 700 --cx 640 --cy 360
  python zed_localize.py <session_dir> --zed-conf ~/zed/SN12345.conf   # auto-picks res
  python zed_localize.py <video.mp4> --fx .. --fy .. --cx .. --cy .. --start-epoch 1720...

Get the numbers from the ZED SDK once:
  cam_info.camera_configuration.calibration_parameters.left_cam .fx/.fy/.cx/.cy
"""

from __future__ import annotations

import argparse
import configparser
import glob
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    import cv2
except Exception:                       # noqa: BLE001
    sys.exit("OpenCV (cv2) is required: pip install opencv-python")
try:
    from pupil_apriltags import Detector
except Exception:                       # noqa: BLE001
    sys.exit("pupil-apriltags is required: pip install pupil-apriltags")
from apriltag_crate_calibrate import detect_poses, make_T, is_proper_rotation  # noqa: E402
from apriltag_tracker import inv_T                                             # noqa: E402
from apriltag_probe import silence_native_logs                                 # noqa: E402

try:
    from scipy.spatial.transform import Rotation
    def _mat_to_quat_xyzw(R):
        return Rotation.from_matrix(R).as_quat()
except Exception:                       # noqa: BLE001
    def _mat_to_quat_xyzw(R):
        R = np.asarray(R, float); t = np.trace(R)
        if t > 0:
            s = np.sqrt(t + 1) * 2; w = .25 * s
            x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = .25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = .25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = .25 * s
        q = np.array([x, y, z, w]); return q / np.linalg.norm(q)


def _world_from_table(z_down):
    """Match the rig's convention (crate/base recorded with this flip). Z-down tag 0 →
    Z-up world = 180° about X."""
    return np.diag([1.0, -1.0, -1.0, 1.0]) if z_down else np.eye(4)


def _zed_conf_intrinsics(conf_path, width, height, side="left"):
    """Read fx,fy,cx,cy for the LEFT/RIGHT cam from a ZED SN<serial>.conf, picking the
    section that matches the recorded resolution."""
    res_by_wh = {(2208, 1242): "2K", (1920, 1080): "FHD", (1280, 720): "HD",
                 (672, 376): "VGA"}
    tag = res_by_wh.get((width, height))
    cam = "RIGHT_CAM" if side == "right" else "LEFT_CAM"
    cp = configparser.ConfigParser()
    cp.read(conf_path)
    sec = f"{cam}_{tag}" if tag else None
    if sec is None or sec not in cp:
        cand = [s for s in cp.sections() if s.startswith(f"{cam}_")]
        if not cand:
            raise ValueError(f"no {cam}_* section in {conf_path}")
        sec = cand[0]
        print(f"[zed] resolution {width}x{height} not matched; using {sec}")
    g = cp[sec]
    return (float(g["fx"]), float(g["fy"]), float(g["cx"]), float(g["cy"]))


def _find_zed_videos(path):
    """Resolve a session dir / rec dir / mp4 to (left_mp4|None, right_mp4|None,
    stereo_mp4|None, dir, meta_dict). The manager records the low-latency SBS stereo as
    ``zed_stereo_*.mp4`` (one file, both eyes); older recordings have separate eyes."""
    from wrench_estimation import load_capture_meta
    p = Path(path)
    if p.is_file() and p.suffix == ".mp4":
        d = p.parent
        stereo = str(p) if "zed_stereo" in p.stem else None
        left = None if stereo else str(p)
        right = None
    else:
        d = p
        def _pick(pat):
            return sorted(glob.glob(str(d / pat)))
        stereos, lefts, rights = _pick("zed_stereo*.mp4"), _pick("zed_left*.mp4"), _pick("zed_right*.mp4")
        if not (stereos or lefts or rights):         # search one level down
            hit = (sorted(d.rglob("zed_stereo*.mp4")) or sorted(d.rglob("zed_left*.mp4"))
                   or sorted(d.rglob("zed_right*.mp4")))
            if hit:
                d = hit[-1].parent
                stereos, lefts, rights = _pick("zed_stereo*.mp4"), _pick("zed_left*.mp4"), _pick("zed_right*.mp4")
        if not (stereos or lefts or rights):
            raise FileNotFoundError(f"no zed_stereo/left/right*.mp4 under {path}")
        stereo = stereos[-1] if stereos else None
        left = lefts[-1] if lefts else None
        right = rights[-1] if rights else None
    # meta (start epoch, tag size, world_from_table) — from the robot_raw sidecar
    meta = {}
    csvs = sorted(d.glob("robot_raw*.csv"))
    if csvs:
        try:
            meta = load_capture_meta(str(csvs[-1]))
        except Exception:
            meta = {}
    return left, right, stereo, d, meta


def localize_video(video, out_csv, cam_params, tag_size, wft, start_epoch,
                   dm_thresh, family="tag36h11", table_id=0, stride=1,
                   crop_x0=0, crop_w=None, crop_y0=0, crop_h=None):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[zed] cannot open {video}")
        return 0
    det = Detector(families=family, nthreads=4, quad_decimate=1.0,
                   refine_edges=True, decode_sharpening=0.25)
    n_seen = n_frames = 0
    with open(out_csv, "w") as f:
        f.write("timestamp,px,py,pz,qx,qy,qz,qw,decision_margin\n")
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if stride > 1 and (idx % stride):
                continue
            n_frames += 1
            # frame epoch from its PTS (robust to fps drift), else index/fps
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            epoch = start_epoch + (msec / 1000.0 if msec and msec > 0 else 0.0)
            if crop_w or crop_h:                 # stereo composite → one eye (SBS or TB)
                frame = frame[crop_y0:(crop_y0 + crop_h) if crop_h else None,
                              crop_x0:(crop_x0 + crop_w) if crop_w else None]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            poses, _ = detect_poses(det, gray, cam_params, tag_size, dm_thresh)
            if table_id not in poses:
                continue
            T_cam_tag, d = poses[table_id]
            T_world_zed = wft @ inv_T(T_cam_tag)
            if not is_proper_rotation(T_world_zed[:3, :3]):
                continue
            p = T_world_zed[:3, 3]
            q = _mat_to_quat_xyzw(T_world_zed[:3, :3])
            f.write(f"{epoch:.4f},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},"
                    f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f},"
                    f"{float(getattr(d, 'decision_margin', 0)):.1f}\n")
            n_seen += 1
    cap.release()
    frac = (n_seen / n_frames) if n_frames else 0.0
    print(f"[zed] {Path(video).name}: tag 0 in {n_seen}/{n_frames} frames "
          f"({frac * 100:.0f}%) → {Path(out_csv).name}")
    if n_frames and frac < 0.5:
        print("[zed] LOW tag-0 visibility — for the next recording tilt the ZED DOWN "
              "~60° so table tag 0 stays in view (fewer pose gaps).")
    return n_seen


def main(argv=None):
    p = argparse.ArgumentParser(description="Localize the ZED camera(s) in the table "
                                "frame from recorded video via the table AprilTag (id 0).")
    p.add_argument("session", help="session dir / rec dir / a zed_left*.mp4")
    p.add_argument("--fx", type=float); p.add_argument("--fy", type=float)
    p.add_argument("--cx", type=float); p.add_argument("--cy", type=float)
    p.add_argument("--zed-conf", default=None, help="ZED SN<serial>.conf for intrinsics")
    p.add_argument("--tag-size", type=float, default=0.065,
                   help="BASE tag edge (m). Tag 0 (table, 100 mm) is auto-scaled via "
                        "TAG_SIZE_OVERRIDES, so keep this at the regular 0.065.")
    p.add_argument("--table-id", type=int, default=0)
    p.add_argument("--dm-thresh", type=float, default=30.0)
    p.add_argument("--start-epoch", type=float, default=None,
                   help="ZED record start epoch (else from meta zed_record_start_epoch)")
    p.add_argument("--table-z-down", dest="z_down", action="store_true", default=True)
    p.add_argument("--no-table-z-down", dest="z_down", action="store_false")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--camera", choices=["auto", "left", "right"], default="auto",
                   help="which ZED camera to localize. auto = the meta's zed_localize_side "
                        "(peer records 'right'). Right sees the table tag better on the peer.")
    args = p.parse_args(argv)

    silence_native_logs()
    left, right, stereo, d, meta = _find_zed_videos(args.session)
    side = args.camera if args.camera != "auto" else meta.get("zed_localize_side", "left")
    start_epoch = args.start_epoch or meta.get("zed_record_start_epoch")
    if not start_epoch:
        sys.exit("no start epoch (pass --start-epoch or record with zed_record_start_epoch)")
    tag_size = args.tag_size or float(meta.get("table_tag_size", 0.065))
    # world_from_table: prefer the exact one recorded in meta so it matches crate/base.
    wft = np.asarray(meta["headcam_world_from_table"]) if meta.get("headcam_world_from_table") \
        else _world_from_table(args.z_down)

    # video: prefer the SBS stereo mp4 (what the manager records); else a separate
    # zed_<side>_*.mp4; else whatever single eye exists.
    video = stereo or (right if (side == "right" and right) else left)
    if video is None:
        sys.exit("no ZED video to localize")

    # intrinsics: CLI > --zed-conf > session meta (this side's, written at record time)
    intr_w = intr_h = None
    if None not in (args.fx, args.fy, args.cx, args.cy):
        cam = (args.fx, args.fy, args.cx, args.cy)
    elif args.zed_conf:
        vc = cv2.VideoCapture(video)
        w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vc.release()
        # A stereo composite is either 2 eyes WIDE (legacy SBS) or 2 eyes TALL (the
        # zedsrc stream-type=2 recording is stacked TOP-BOTTOM, e.g. 1280x1440).
        if stereo and h > w:
            eye_w, eye_h = w, h // 2                  # top-bottom
        elif stereo:
            eye_w, eye_h = w // 2, h                  # side-by-side
        else:
            eye_w, eye_h = w, h
        cam = _zed_conf_intrinsics(args.zed_conf, eye_w, eye_h, side)
        if stereo:
            intr_w, intr_h = eye_w, eye_h             # → stereo crop below
        print(f"[zed] intrinsics from {Path(args.zed_conf).name} @ {eye_w}x{eye_h} ({side})")
    else:
        zi = (meta.get(f"zed_{side}_intrinsics") or meta.get("zed_intrinsics")
              or meta.get("zed_left_intrinsics"))
        if not zi:
            sys.exit("no intrinsics — pass --fx/... or --zed-conf, or record with the "
                     "manager (it writes zed_<side>_intrinsics into the session meta).")
        cam = (zi["fx"], zi["fy"], zi["cx"], zi["cy"])
        intr_w, intr_h = zi.get("width"), zi.get("height")
        print(f"[zed] {side.upper()}-cam intrinsics from meta ({zi.get('conf', '?')} "
              f"{zi.get('resolution', '?')}): fx={cam[0]:.1f} cx={cam[2]:.1f}")

    # Stereo composite? Crop this camera's eye out. Legacy SBS = 2× the calibrated
    # WIDTH (LEFT = columns [0,w)); the current zedsrc stream-type=2 recording is
    # TOP-BOTTOM = 2× the calibrated HEIGHT (LEFT = the TOP half).
    crop_x0, crop_w = 0, None
    crop_y0, crop_h = 0, None
    if intr_w:
        vc = cv2.VideoCapture(video)
        vw = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vc.release()
        if vw == 2 * intr_w:
            crop_w = intr_w
            crop_x0 = intr_w if side == "right" else 0
            print(f"[zed] side-by-side stereo mp4 ({vw}px) — using {side.upper()} half "
                  f"[{crop_x0},{crop_x0 + crop_w})")
        elif intr_h and vh == 2 * intr_h and vw == intr_w:
            crop_h = intr_h
            crop_y0 = intr_h if side == "right" else 0
            print(f"[zed] top-bottom stereo mp4 ({vw}x{vh}) — using {side.upper()} "
                  f"({'bottom' if side == 'right' else 'top'}) half rows "
                  f"[{crop_y0},{crop_y0 + crop_h})")

    ts = (Path(video).stem.replace("zed_stereo", "").replace("zed_left", "")
          .replace("zed_right", ""))
    localize_video(video, str(d / f"zed_pose_{side}{ts}.csv"), cam, tag_size, wft,
                   float(start_epoch), args.dm_thresh, table_id=args.table_id,
                   stride=args.stride, crop_x0=crop_x0, crop_w=crop_w,
                   crop_y0=crop_y0, crop_h=crop_h)


if __name__ == "__main__":
    main()
