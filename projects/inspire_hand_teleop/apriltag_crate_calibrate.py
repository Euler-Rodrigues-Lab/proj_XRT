#!/usr/bin/env python3
"""
apriltag_crate_calibrate.py — Phase 4: learn the crate's tag bundle.

The crate carries one ``tag36h11`` marker per face (default ids 1-10). To track
the crate as a single rigid body we need each tag's fixed pose in a common
*crate body frame*: ``T_crate_tag_i`` for every face. This tool learns that
geometry from observation (no CAD, no hand measurement) and writes it to
``crate_bundle.json`` — the single source of truth later read by the tracker and
the digital-twin viewer.

How it works
------------
Point one RealSense D435 at the crate and rotate it SLOWLY so that, over the
capture, every face shares the view with at least one neighbour. For each frame
we compute every visible tag's pose ``T_cam_tag`` and, for each co-visible pair
(i, j), the relative transform ``T_tag_i_tag_j = inv(T_cam_tag_i) @ T_cam_tag_j``.
Observations are accumulated per edge (motion-gated so a still crate adds
nothing — no dwell bias), averaged with scipy's robust rotation mean, then a
max-reliability spanning tree is propagated from an anchor tag to place every
face in the crate frame. Per-edge spread (deg / mm) is reported so a wobbly
mount is obvious before you trust the result.

The crate body frame defaults to the anchor tag's frame (deterministic). Pass
``--center-on-centroid`` to move the origin to the centroid of all tags (keeps
the anchor's orientation) — nicer for a centred box mesh in the twin.

Controls
--------
  SPACE   start / stop capturing observations
  s       solve the bundle and write the JSON now (preview any time)
  r       reset all accumulated observations
  q / ESC quit  (auto-solves + saves if there are unsaved observations)

Examples
--------
  python apriltag_crate_calibrate.py --list
  python apriltag_crate_calibrate.py --serials 123456789 987654321   # both externals, ids 1-10
  python apriltag_crate_calibrate.py --ids 1-10 --anchor-id 10 --out crate_bundle.json
  python apriltag_crate_calibrate.py --center-on-centroid

Deps:  pip install pupil-apriltags pyrealsense2   (scipy + opencv already present)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

try:
    import cv2
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

# Reuse the probe's camera + parsing helpers (keep apriltag_probe.py alongside).
try:
    from apriltag_probe import (list_devices, start_pipeline, parse_id_set,
                                draw_detection, silence_native_logs)
except Exception as exc:
    sys.exit(f"Could not import apriltag_probe.py (keep it in the same folder): {exc}")


# ── SE(3) helpers ───────────────────────────────────────────────────────────

def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def is_proper_rotation(R, tol: float = 1e-3) -> bool:
    """True only for a right-handed orthonormal matrix (a real rotation).

    The AprilTag pose solver can return a *reflection* (orthonormal but
    determinant = -1) when a tag's pose is ambiguous/degenerate — the same
    condition behind the "more than one new minima found" warning. Those are
    not real poses; we drop them rather than feed them downstream (scipy's
    Rotation.from_matrix rejects det<=0 and would crash the capture)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return False
    if np.linalg.det(R) <= 0:                      # reflection / left-handed
        return False
    return np.allclose(R.T @ R, np.eye(3), atol=tol)   # orthonormal


# Per-tag PHYSICAL sizes (m) for tags that are NOT the regular size. Tag 0 (table anchor)
# and tag 10 (crate inside-base anchor) are printed at 100 mm for a stronger, farther-range
# pose; every other tag uses the base ``tag_size`` passed to detect_poses (65 mm crate
# faces). Shared by ALL AprilTag features (crate calibrate + tracker, head-cam, ZED).
TAG_SIZE_OVERRIDES = {0: 0.100, 10: 0.100}


def detect_poses(detector, gray, cam_params, tag_size, dm_thresh, size_overrides=None):
    """Return ({tag_id: (T_cam_tag, det)}, raw_dets) for tags above the margin.

    ``tag_size`` is the size assumed for tags NOT in ``size_overrides`` (default
    TAG_SIZE_OVERRIDES: ids 0 and 10 = 100 mm). Tags with a different true size are
    corrected by RESCALING their translation by real/tag_size — the PnP translation is
    linear in the tag's physical size while the rotation is scale-invariant, so one detect
    call handles a mix of tag sizes exactly. Degenerate poses (non-finite, reflected, or
    non-orthonormal pose_R) are discarded here so no bad rotation reaches the bundle math."""
    overrides = TAG_SIZE_OVERRIDES if size_overrides is None else size_overrides
    dets = detector.detect(gray, estimate_tag_pose=True,
                           camera_params=cam_params, tag_size=tag_size)
    poses = {}
    for d in dets:
        R = getattr(d, "pose_R", None)
        if R is None or d.decision_margin < dm_thresh:
            continue
        R = np.asarray(R, dtype=np.float64)
        if not is_proper_rotation(R):
            continue
        tid = int(d.tag_id)
        t = np.asarray(d.pose_t, dtype=np.float64).reshape(3)
        real = overrides.get(tid)
        if real and real != tag_size:
            t = t * (real / tag_size)          # translation ∝ physical tag size
        poses[tid] = (make_T(R, t), d)
    return poses, dets


# ── Bundle solve ────────────────────────────────────────────────────────────

def solve_edges(Rmats, tvecs, wts):
    """Average each edge's relative pose; return {(i,j): {R,t,n,spreads}}."""
    edges = {}
    for key in Rmats:
        try:
            Rs = np.array(Rmats[key])
            ts = np.array(tvecs[key])
            w = np.array(wts[key], dtype=float)
            w = np.ones(len(Rs)) if w.sum() <= 0 else w / w.sum()
            rot = Rotation.from_matrix(Rs)
            rmean = rot.mean(weights=w)
            tmean = np.average(ts, axis=0, weights=w)
            ang = np.degrees((rmean.inv() * rot).magnitude())
            rot_spread = float(np.sqrt(np.mean(ang ** 2)))
            trans_spread = float(np.sqrt(np.mean(np.sum((ts - tmean) ** 2, axis=1))) * 1000.0)
            edges[key] = dict(R=rmean.as_matrix(), t=tmean, n=len(Rs),
                              rot_spread_deg=rot_spread, trans_spread_mm=trans_spread)
        except Exception as exc:
            print(f"  skipping edge {key} (bad observations: {exc})")
    return edges


def spanning_tree(edges):
    """Max-reliability spanning forest (Kruskal). Prefers edges with many
    observations and low rotational spread, so propagation never chains through
    a noisy pair when a better path exists."""
    def score(e):
        return e["n"] / (1.0 + e["rot_spread_deg"])
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for (i, j) in sorted(edges, key=lambda k: score(edges[k]), reverse=True):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((i, j))
    return tree


def propagate(edges, tree, anchor):
    """Place every reachable tag in the anchor frame via the tree edges."""
    adj = defaultdict(list)
    for (i, j) in tree:
        adj[i].append(j)
        adj[j].append(i)
    T = {anchor: np.eye(4)}
    via = {anchor: None}
    queue = [anchor]
    while queue:
        i = queue.pop(0)
        for j in adj[i]:
            if j in T:
                continue
            if (i, j) in edges:
                e = edges[(i, j)]
                Tij = make_T(e["R"], e["t"])
            else:
                e = edges[(j, i)]
                Tij = np.linalg.inv(make_T(e["R"], e["t"]))
            T[j] = T[i] @ Tij
            via[j] = dict(via_tag=i, n_obs=e["n"],
                          rot_spread_deg=round(e["rot_spread_deg"], 3),
                          trans_spread_mm=round(e["trans_spread_mm"], 2))
            queue.append(j)
    return T, via


def reachable(Rmats, anchor, min_obs):
    """Tags currently reachable from anchor given edges with >= min_obs obs
    (used for the live connectivity HUD)."""
    adj = defaultdict(set)
    for (i, j), lst in Rmats.items():
        if len(lst) >= min_obs:
            adj[i].add(j)
            adj[j].add(i)
    if anchor is None:
        return set()
    seen = {anchor}
    queue = [anchor]
    while queue:
        i = queue.pop(0)
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                queue.append(j)
    return seen


def solve_and_save(Rmats, tvecs, wts, seen, args) -> bool:
    """Solve the bundle and write the JSON. Returns True on success."""
    seen_crate = sorted(seen)
    if not seen_crate:
        print("No crate tags observed — nothing to solve.")
        return False

    if args.anchor_id is not None and args.anchor_id in seen_crate:
        anchor = args.anchor_id
    else:
        if args.anchor_id is not None:
            print(f"  anchor id {args.anchor_id} was never seen; "
                  f"using lowest seen id {seen_crate[0]} instead.")
        anchor = seen_crate[0]

    edges = solve_edges(Rmats, tvecs, wts)
    if not edges:
        print("No co-visible tag pairs captured — rotate the crate so neighbouring "
              "faces overlap in view, then capture again.")
        return False

    tree = spanning_tree(edges)
    T, via = propagate(edges, tree, anchor)

    frame_origin = "anchor"
    if args.center_on_centroid and len(T) > 0:
        centroid = np.mean([Ti[:3, 3] for Ti in T.values()], axis=0)
        shift = np.eye(4)
        shift[:3, 3] = -centroid          # move origin to centroid, keep orientation
        T = {i: shift @ Ti for i, Ti in T.items()}
        frame_origin = "centroid"

    disconnected = sorted(set(seen_crate) - set(T))
    n_obs_total = int(sum(len(v) for v in Rmats.values()))

    tags = {}
    for tid in sorted(T):
        entry = {"T_crate_tag": np.round(T[tid], 6).tolist()}
        if tid == anchor:
            entry["anchor"] = True
        else:
            entry.update(via[tid])
        tags[str(tid)] = entry

    bundle = {
        "family": args.family,
        "tag_size": args.tag_size,
        "anchor_id": int(anchor),
        "frame_origin": frame_origin,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_frames_observed_pairs": n_obs_total,
        "tags": tags,
        "disconnected_ids": disconnected,
        "note": ("T_crate_tag maps a point in the tag frame into the crate body "
                 "frame. tag_size is the black-square edge length in metres."),
    }

    out = os.path.abspath(args.out)
    with open(out, "w") as f:
        json.dump(bundle, f, indent=2)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\nBundle solved — anchor id {anchor}, origin = {frame_origin}, "
          f"{len(T)}/{len(seen_crate)} tags placed, {n_obs_total} pair-obs.")
    print(f"  {'id':>3}  {'via':>4}  {'n_obs':>6}  {'rot_spread':>10}  {'trans_spread':>12}")
    for tid in sorted(T):
        if tid == anchor:
            print(f"  {tid:>3}  {'--':>4}  {'anchor':>6}  {'--':>10}  {'--':>12}")
        else:
            v = via[tid]
            warn = "  ⚠" if (v["rot_spread_deg"] > 2.0 or v["trans_spread_mm"] > 5.0) else ""
            print(f"  {tid:>3}  {v['via_tag']:>4}  {v['n_obs']:>6}  "
                  f"{v['rot_spread_deg']:>8.2f}°  {v['trans_spread_mm']:>9.2f}mm{warn}")
    if disconnected:
        print(f"  ⚠ DISCONNECTED (never co-visible with the bundle): {disconnected}")
        print("    → rotate so those faces overlap a placed neighbour, then re-solve.")
    print(f"\nWrote {out}")
    return True


# ── Main ────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Learn the crate's AprilTag bundle (Phase 4).")
    p.add_argument("--list", action="store_true",
                   help="List connected RealSense cameras and exit.")
    p.add_argument("--serials", nargs="*", default=None,
                   help="RealSense serial(s) to open (default: first two found). "
                        "Using both external cameras gives faster, multi-angle "
                        "coverage of the crate faces.")
    p.add_argument("--ids", type=str, default="1-10",
                   help="Tag ids that belong to the crate (default '1-10'). Other "
                        "ids in view (e.g. 0 = table) are ignored.")
    p.add_argument("--anchor-id", "--anchor_id", dest="anchor_id", type=int, default=10,
                   help="Tag id whose frame becomes the crate body frame / box "
                        "centre (default 10 = the tag on the inside centre base). "
                        "Falls back to the lowest seen id if id 10 was never "
                        "observed. Do NOT combine with --center-on-centroid, which "
                        "would move the origin away from this tag.")
    p.add_argument("--tag-size", "--tag_size", dest="tag_size", type=float, default=0.065,
                   help="BASE tag edge, metres (default 0.065 for the crate faces). Tags 0 "
                        "and 10 (100 mm) are auto-handled by TAG_SIZE_OVERRIDES; keep this "
                        "at the regular-face size.")
    p.add_argument("--family", type=str, default="tag36h11")
    p.add_argument("--out", type=str, default="crate_bundle.json")
    p.add_argument("--center-on-centroid", "--center_on_centroid",
                   dest="center_on_centroid", action="store_true",
                   help="Put the crate origin at the centroid of all tags "
                        "(default: anchor tag frame).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--decision-margin", "--decision_margin", dest="decision_margin",
                   type=float, default=30.0,
                   help="Reject detections weaker than this (default 30).")
    p.add_argument("--rot-gate-deg", dest="rot_gate_deg", type=float, default=2.0,
                   help="Min rotation change to log a new observation of an edge.")
    p.add_argument("--trans-gate-mm", dest="trans_gate_mm", type=float, default=3.0,
                   help="Min translation change to log a new observation of an edge.")
    p.add_argument("--min-obs", dest="min_obs", type=int, default=3,
                   help="Edge obs needed before it counts toward live connectivity.")
    p.add_argument("--nthreads", type=int, default=4)
    return p


def main():
    args = build_argparser().parse_args()
    silence_native_logs()   # mute the native "more than one minima" spam

    devices = list_devices()
    if args.list or not devices:
        print("No RealSense cameras found." if not devices else "Connected RealSense cameras:")
        for name, serial in devices:
            print(f"  {name}  serial={serial}")
        return

    crate_ids = parse_id_set(args.ids)
    axis_len = args.tag_size * 0.5
    serials = args.serials or [s for _, s in devices][:2]

    # Open every camera (both externals by default) — more angles see more
    # co-visible face pairs at once, so the bundle solves faster and more robustly.
    cams = []
    for s in serials:
        try:
            pipeline, K, dist, intr = start_pipeline(s, args.width, args.height, args.fps)
            cams.append({"serial": str(s), "pipeline": pipeline, "K": K, "dist": dist,
                         "cam_params": (intr.fx, intr.fy, intr.ppx, intr.ppy), "img": None})
        except Exception as exc:
            print(f"  could not open camera {s}: {exc}")
    if not cams:
        sys.exit("No cameras opened.")

    print(f"Opened {len(cams)} camera(s): {[c['serial'] for c in cams]}  "
          f"@ {args.width}x{args.height}/{args.fps}")
    print(f"Crate ids: {sorted(crate_ids)}   family={args.family}   "
          f"tag_size={args.tag_size * 1000:.0f}mm")
    print("Rotate the crate SLOWLY so every face overlaps a neighbour.")
    print("Controls:  SPACE capture on/off   s solve+save   r reset   q/ESC quit\n")

    detector = Detector(families=args.family, nthreads=args.nthreads,
                        quad_decimate=1.0, refine_edges=True, decode_sharpening=0.25)

    Rmats: dict = defaultdict(list)
    tvecs: dict = defaultdict(list)
    wts: dict = defaultdict(list)
    last_edge: dict = {}            # keyed (serial, i, j) so each camera contributes
    seen: set = set()

    capturing = False
    saved_obs_count = -1            # obs count at last save, to detect "unsaved"
    win = "Crate bundle calibration (SPACE capture, s save, r reset, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    hud_reach: set = set()
    frame_i = 0

    try:
        while True:
            for cam in cams:
                frames = cam["pipeline"].poll_for_frames()   # non-blocking per camera
                if not frames:
                    continue
                color = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data()).copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                poses, dets = detect_poses(detector, gray, cam["cam_params"],
                                           args.tag_size, args.decision_margin)
                for d in dets:
                    draw_detection(img, d, cam["K"], cam["dist"], axis_len)
                cam["img"] = img

                crate_now = [i for i in poses if i in crate_ids]
                for i in crate_now:
                    seen.add(i)
                if not capturing:
                    continue
                for a in range(len(crate_now)):
                    for b in range(a + 1, len(crate_now)):
                        i, j = sorted((crate_now[a], crate_now[b]))
                        try:
                            Tij = np.linalg.inv(poses[i][0]) @ poses[j][0]
                            R, t = Tij[:3, :3], Tij[:3, 3]
                            if not is_proper_rotation(R):       # belt-and-suspenders
                                continue
                            ekey = (cam["serial"], i, j)        # per-camera motion gate
                            if ekey in last_edge:
                                R0, t0 = last_edge[ekey]
                                dang = np.degrees((Rotation.from_matrix(R0).inv()
                                                   * Rotation.from_matrix(R)).magnitude())
                                if (dang < args.rot_gate_deg
                                        and np.linalg.norm(t - t0) * 1000 < args.trans_gate_mm):
                                    continue
                            last_edge[ekey] = (R, t)
                            Rmats[(i, j)].append(R)
                            tvecs[(i, j)].append(t)
                            wts[(i, j)].append(min(poses[i][1].decision_margin,
                                                   poses[j][1].decision_margin))
                        except Exception:
                            continue          # never let one bad frame kill the capture

            # connectivity HUD (refresh a few times/sec, not every frame)
            frame_i += 1
            if frame_i % 10 == 0:
                anchor = (args.anchor_id if (args.anchor_id in seen)
                          else (min(seen) if seen else None))
                hud_reach = reachable(Rmats, anchor, args.min_obs)

            n_obs = int(sum(len(v) for v in Rmats.values()))
            missing = sorted(crate_ids - seen)
            unreached = sorted((seen & crate_ids) - hud_reach)

            # composite all camera views side by side
            panels = []
            for cam in cams:
                im = cam["img"]
                if im is None:
                    im = np.full((args.height, args.width, 3), 40, np.uint8)
                h = 480
                im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
                cv2.putText(im, cam["serial"], (12, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                panels.append(im)
            disp = cv2.hconcat(panels) if len(panels) > 1 else panels[0]

            cv2.putText(disp, f"CAPTURE {'ON' if capturing else 'off'}   "
                              f"{len(cams)} cam   pairs:{len(Rmats)}  obs:{n_obs}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if capturing else (0, 165, 255), 2)
            cv2.putText(disp, f"seen:{sorted(seen)}  missing:{missing}", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(disp, f"connected to anchor: {sorted(hud_reach)}  "
                              f"not-yet-linked:{unreached}", (12, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if not unreached else (0, 200, 255), 2)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                if n_obs > 0 and n_obs != saved_obs_count:
                    print("\nAuto-solving on quit (unsaved observations)...")
                    solve_and_save(Rmats, tvecs, wts, seen & crate_ids, args)
                break
            elif key == ord(" "):
                capturing = not capturing
                print(f"Capture {'ON' if capturing else 'OFF'}.")
            elif key == ord("s"):
                if solve_and_save(Rmats, tvecs, wts, seen & crate_ids, args):
                    saved_obs_count = n_obs
            elif key == ord("r"):
                Rmats.clear(); tvecs.clear(); wts.clear(); last_edge.clear(); seen.clear()
                saved_obs_count = -1
                print("Reset all observations.")
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cams:
            try:
                cam["pipeline"].stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
