#!/usr/bin/env python3
"""
apriltag_probe.py — Phase 2 sanity check for the AprilTag crate-tracking system.

Opens ONE Intel RealSense D435 (color stream only), detects ``tag36h11`` markers,
estimates each tag's 6-DoF pose (tag size default 65 mm), and draws the tag id +
a 3D axis triad (X=red, Y=green, Z=blue) on the live image.

This is a throwaway diagnostic — it touches no other project code and writes
nothing except snapshots you explicitly ask for. Run it BEFORE any integration
to confirm:

  * every printed tag (ids 0-10) detects reliably at the working distance and
    your lighting (walk the crate around; the HUD shows which expected ids are
    still missing),
  * the printed size is correct — the reported distance should match a tape
    measure to the camera; if it's off by a constant %, your print is mis-scaled,
  * **which way the tag frame's X/Y/Z point** — you need to know tag 0's axes to
    define the world frame in Phase 3 (table tag = world origin). Press 'p' with
    tag 0 in view to dump its pose to the console.

Controls:
  q / ESC   quit
  s         save the current annotated frame to a PNG
  p         print every visible tag's pose to the console (id 0 highlighted)

Examples:
  python apriltag_probe.py --list                 # list connected RealSense cameras
  python apriltag_probe.py                         # first camera, 65 mm tags
  python apriltag_probe.py --serial 123456789      # pick a camera by serial
  python apriltag_probe.py --tag-size 0.065 --width 1280 --height 720 --fps 30

Deps (additive, won't disturb the teleop stack):
  pip install pupil-apriltags pyrealsense2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

# --- OpenCV (already a project dependency) ---------------------------------
try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required (it's already in the project env): "
             "pip install 'opencv-python>=4.5,<5'")

# --- RealSense + AprilTag (the two new, additive deps) ---------------------
try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not found. Install with:  pip install pyrealsense2")

try:
    from pupil_apriltags import Detector
except ImportError:
    sys.exit("pupil-apriltags not found. Install with:  pip install pupil-apriltags")


def silence_native_logs():
    """Send the C-level stdout/stderr (fds 1 & 2) to /dev/null so the AprilTag
    pose solver's repeated "Error, more than one new minima found." messages —
    printed by the native library, not catchable with Python logging — don't
    flood the console. Python's own print()/tracebacks are preserved by
    repointing sys.stdout/sys.stderr at dups of the original fds. Set
    APRILTAG_VERBOSE=1 to keep the native output. Safe to call repeatedly."""
    if os.environ.get("APRILTAG_VERBOSE") or getattr(silence_native_logs, "_done", False):
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_out, saved_err = os.dup(1), os.dup(2)
        sys.stdout = os.fdopen(saved_out, "w", buffering=1)
        sys.stderr = os.fdopen(saved_err, "w", buffering=1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        silence_native_logs._done = True
    except Exception:
        pass


def list_devices() -> list[tuple[str, str]]:
    """Return [(name, serial), ...] for every connected RealSense device."""
    out = []
    for d in rs.context().query_devices():
        out.append((d.get_info(rs.camera_info.name),
                    d.get_info(rs.camera_info.serial_number)))
    return out


def start_pipeline(serial: str | None, width: int, height: int, fps: int):
    """Start a color-only RealSense pipeline; return (pipeline, K, dist, intr)."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)

    color = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color.get_intrinsics()
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    # pupil-apriltags estimates pose with a pinhole model (no distortion). We draw
    # the axes with zero distortion to stay CONSISTENT with that pose; the real
    # D435 color distortion is tiny (printed below so you can judge).
    dist = np.zeros(5, dtype=np.float64)
    return pipeline, K, dist, intr


def draw_detection(img, det, K, dist, axis_len) -> float | None:
    """Draw one detection (outline, center, id, axes). Returns distance in m."""
    corners = det.corners.astype(int)
    for i in range(4):
        cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
    center = tuple(det.center.astype(int))
    cv2.circle(img, center, 4, (0, 0, 255), -1)

    dist_m = None
    pose_R = getattr(det, "pose_R", None)
    if pose_R is not None:
        rvec, _ = cv2.Rodrigues(np.asarray(pose_R, dtype=np.float64))
        tvec = np.asarray(det.pose_t, dtype=np.float64).reshape(3, 1)
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, axis_len, 2)
        dist_m = float(np.linalg.norm(tvec))
        label = f"id {det.tag_id}  {dist_m * 100:.1f}cm  dm{det.decision_margin:.0f}"
    else:
        label = f"id {det.tag_id}"
    cv2.putText(img, label, (center[0] - 45, center[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return dist_m


def print_poses(dets):
    """Dump every visible tag's pose to the console (id 0 highlighted)."""
    if not dets:
        print("  (no tags visible)")
        return
    print(f"  ── poses @ {datetime.now().strftime('%H:%M:%S')} "
          f"({len(dets)} tag(s)) ──")
    for det in sorted(dets, key=lambda d: d.tag_id):
        pose_R = getattr(det, "pose_R", None)
        if pose_R is None:
            print(f"  id {det.tag_id}: (no pose)")
            continue
        t = np.asarray(det.pose_t, dtype=np.float64).ravel()
        R = np.asarray(pose_R, dtype=np.float64)
        tag = "  >> TAG 0 (world-origin candidate)" if det.tag_id == 0 else ""
        print(f"  id {det.tag_id}: t=[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] m"
              f"  dist={np.linalg.norm(t) * 100:.1f}cm  dm={det.decision_margin:.0f}{tag}")
        if det.tag_id == 0:
            print("     R (cols = tag X,Y,Z in camera frame):")
            for row in R:
                print(f"       [{row[0]:+.3f} {row[1]:+.3f} {row[2]:+.3f}]")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-camera AprilTag pose sanity check (Phase 2).")
    p.add_argument("--list", action="store_true",
                   help="List connected RealSense cameras and exit.")
    p.add_argument("--serial", type=str, default=None,
                   help="RealSense serial to open (default: first found).")
    p.add_argument("--tag-size", "--tag_size", dest="tag_size", type=float,
                   default=0.065, help="Tag black-square edge length, metres (default 0.065).")
    p.add_argument("--family", type=str, default="tag36h11",
                   help="AprilTag family (default tag36h11).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--quad-decimate", "--quad_decimate", dest="quad_decimate",
                   type=float, default=1.0,
                   help="Detector decimation (1.0 = full res / most accurate; "
                        "2.0 = faster, less range). Default 1.0.")
    p.add_argument("--nthreads", type=int, default=4)
    p.add_argument("--expect-ids", "--expect_ids", dest="expect_ids",
                   type=str, default="0-10",
                   help="Ids you expect to see, for the 'missing' HUD readout "
                        "(e.g. '0-10' or '0,1,2'). Default 0-10.")
    return p


def parse_id_set(spec: str) -> set[int]:
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return ids


def main():
    args = build_argparser().parse_args()
    silence_native_logs()   # mute the native "more than one minima" spam

    devices = list_devices()
    if args.list or not devices:
        if not devices:
            print("No RealSense cameras found.")
        else:
            print("Connected RealSense cameras:")
            for name, serial in devices:
                print(f"  {name}  serial={serial}")
        return

    expect = parse_id_set(args.expect_ids)
    pipeline, K, dist, intr = start_pipeline(
        args.serial, args.width, args.height, args.fps)

    serial_used = args.serial or devices[0][1]
    print(f"Opened RealSense {serial_used}  @ {args.width}x{args.height}/{args.fps}")
    print(f"Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
    print(f"Distortion model={intr.model}  coeffs={np.round(intr.coeffs, 5).tolist()}")
    print(f"Family={args.family}  tag_size={args.tag_size * 1000:.0f}mm")
    print("Controls:  q/ESC quit   s save snapshot   p print poses\n")

    detector = Detector(
        families=args.family,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        refine_edges=True,
        decode_sharpening=0.25,
    )

    win = "AprilTag probe — X=red Y=green Z=blue (q quit, s snap, p print)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cam_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    axis_len = args.tag_size * 0.5

    seen: set[int] = set()
    fps_ema = 0.0
    last_t = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            dets = detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=cam_params, tag_size=args.tag_size)

            for det in dets:
                draw_detection(img, det, K, dist, axis_len)
                seen.add(int(det.tag_id))

            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_t)
            last_t = now
            fps_ema = inst_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst_fps

            missing = sorted(expect - seen)
            cv2.putText(img, f"{fps_ema:4.1f} fps   {len(dets)} tag(s) now",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            seen_txt = ("ALL expected ids seen ✓" if not missing
                        else f"missing ids: {missing}")
            cv2.putText(img, seen_txt, (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if not missing else (0, 200, 255), 2)

            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):           # q or ESC
                break
            elif key == ord("s"):
                fn = f"apriltag_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fn, img)
                print(f"Saved {fn}")
            elif key == ord("p"):
                print_poses(dets)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nDone. Ids seen this run: {sorted(seen)}")
        miss = sorted(expect - seen)
        if miss:
            print(f"Never saw expected ids: {miss} — re-check those tags.")


if __name__ == "__main__":
    main()
