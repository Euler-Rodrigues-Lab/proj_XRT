#!/usr/bin/env python3
"""Assemble a training-ready, time-aligned dataset from raw study sessions.

Turns the raw recordings (robot_raw CSVs + crate/base/ZED pose CSVs + ZED videos, for
BOTH robots merged under {session}/{cond}/attempt{K}/robot{1,2}/) into per-episode HDF5
with everything resampled onto ONE common clock + rate — the (observation, action) form a
centralized Diffusion Policy trains on.

Key alignment steps:
  * everything is put on the MASTER's clock: robot 2's timestamps are shifted by the
    master↔peer offset recorded in robot 1's meta (clock_sync.link_peer_minus_local_offset_s);
  * each modality is nearest-neighbour resampled onto a common timeline at --rate Hz,
    with a per-modality max-gap so stale samples become NaN + a valid mask.

Per episode (one attempt) it writes, on the common timeline:
  action  : r1_qgoal (25) + r2_qgoal (25) + r1_fingers (24) + r2_fingers (24)  [commands]
  obs     : r{N}_q/dq/tau (25 each), r{N}_fingers (24), r{N}_base (7), r{N}_zed_pose (7),
            crate_pose (7, object in the table frame), and optionally r{N}_image (ZED-left).
Joint order per 25-vector: legs(8)+torso(3)+left_arm(7)+right_arm(7).

  python build_dataset.py <session_dir> -o dataset.hdf5              # no images (fast)
  python build_dataset.py <session_dir> -o dataset.hdf5 --images --image-size 224
  python build_dataset.py <session_dir> -o dataset.hdf5 --rate 15 --only-good
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wrench_estimation as we  # noqa: E402

try:
    import h5py
except Exception:                       # noqa: BLE001
    sys.exit("h5py required: pip install h5py")
try:
    import cv2
except Exception:                       # noqa: BLE001
    cv2 = None

_ROBOT_LEAF = re.compile(r"^robot(\d+)$")
_JSEGS = ("legs", "torso", "left", "right")     # → 8+3+7+7 = 25


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def _nearest(t_src, v_src, t_q, max_gap):
    """Nearest-neighbour resample v_src(t_src) onto t_q; samples farther than max_gap
    become NaN. Returns (values (len(t_q), ...), valid mask)."""
    v_src = np.asarray(v_src, dtype=np.float64)
    out = np.full((len(t_q),) + v_src.shape[1:], np.nan)
    valid = np.zeros(len(t_q), dtype=bool)
    t_src = np.asarray(t_src, dtype=np.float64)
    if t_src.size == 0:
        return out, valid
    order = np.argsort(t_src)
    t_src, v_src = t_src[order], v_src[order]
    j = np.searchsorted(t_src, t_q)
    j = np.clip(j, 0, t_src.size - 1)
    for k in range(len(t_q)):
        i = j[k]
        if i > 0 and abs(t_src[i - 1] - t_q[k]) < abs(t_src[i] - t_q[k]):
            i -= 1
        if abs(t_src[i] - t_q[k]) <= max_gap:
            out[k] = v_src[i]
            valid[k] = True
    return out, valid


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _joints(cap, field):
    return np.concatenate([cap[s][field] for s in _JSEGS if s in cap], axis=1)


def _load_pose_csv(leaf, patterns):
    """First matching pose CSV (timestamp,px,py,pz,qx,qy,qz,qw). Returns (t(N,), pose(N,7))."""
    for pat in patterns:
        hits = sorted(Path(leaf).glob(pat))
        if hits:
            try:
                arr = np.genfromtxt(str(hits[-1]), delimiter=",", skip_header=1,
                                    usecols=range(8), dtype=float)
            except Exception:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = arr[~np.any(np.isnan(arr), axis=1)]
            if arr.shape[0]:
                return arr[:, 0], arr[:, 1:8]
    return np.empty(0), np.empty((0, 7))


def _zed_video_for(leaf, side):
    """The ZED video to use for this leaf. Prefers the SBS ``zed_stereo_*.mp4`` (the
    low-latency stereo stream we record), then the localization eye's separate file, then
    whatever single eye exists."""
    leaf = Path(leaf)
    stereo = sorted(leaf.glob("zed_stereo*.mp4"))
    if stereo:
        return stereo[-1]
    rights = sorted(leaf.glob("zed_right*.mp4"))
    lefts = sorted(leaf.glob("zed_left*.mp4"))
    if side == "right" and rights:
        return rights[-1]
    if lefts:
        return lefts[-1]
    return rights[-1] if rights else None


def _zed_side_crop(meta, video):
    """Crop rect (x0, y0, w, h) selecting the localization eye (master=left, peer=right)
    out of a stereo frame; (0, 0, None, None) when already a single eye. Handles BOTH
    stereo layouts: legacy side-by-side (video 2× the eye WIDTH) and the current zedsrc
    stream-type=2 composite, which is stacked TOP-BOTTOM — left eye on TOP (e.g.
    1280x1440 for HD720). Without the TB case, localization/images silently ran on the
    full double-height frame with single-eye intrinsics."""
    side = meta.get("zed_localize_side", "left")
    zi = (meta.get(f"zed_{side}_intrinsics") or meta.get("zed_intrinsics")
          or meta.get("zed_left_intrinsics") or {})
    w, h = zi.get("width"), zi.get("height")
    if not (w and cv2 is not None and video):
        return 0, 0, None, None
    cap = cv2.VideoCapture(str(video))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if vw == 2 * w:                                   # side-by-side → horizontal half
        return (w if side == "right" else 0), 0, w, None
    if h and vh == 2 * h and vw == w:                 # top-bottom → vertical half
        return 0, (h if side == "right" else 0), None, h
    return 0, 0, None, None


def _ensure_zed_pose(leaf, meta, dm_thresh=30.0):
    """--localize-zed: produce zed_pose_<side>_*.csv for this leaf from its ZED video +
    the meta intrinsics for the localization side (peer=right). Skips if it already exists
    or data/intrinsics are missing; handles a side-by-side stereo mp4 by cropping the half."""
    leaf = Path(leaf)
    side = meta.get("zed_localize_side", "left")
    if sorted(leaf.glob("zed_pose_*.csv")):
        return
    video = _zed_video_for(leaf, side)
    zi = (meta.get(f"zed_{side}_intrinsics") or meta.get("zed_intrinsics")
          or meta.get("zed_left_intrinsics"))
    start = meta.get("zed_record_start_epoch")
    if not (video and zi and start):
        return
    try:
        import zed_localize as zl
    except Exception as exc:                         # noqa: BLE001
        print(f"[ds] zed localize unavailable ({exc})")
        return
    if cv2 is None:
        print("[ds] cv2 unavailable — cannot localize ZED")
        return
    cam = (zi["fx"], zi["fy"], zi["cx"], zi["cy"])
    tag_size = float(meta.get("table_tag_size", 0.065))
    wft = (np.asarray(meta["headcam_world_from_table"])
           if meta.get("headcam_world_from_table") else zl._world_from_table(True))
    crop_x0, crop_y0, crop_w, crop_h = _zed_side_crop(meta, video)
    ts = (video.stem.replace("zed_stereo", "").replace("zed_left", "")
          .replace("zed_right", ""))
    out = str(leaf / f"zed_pose_{side}{ts}.csv")
    zl.localize_video(str(video), out, cam, tag_size, wft, float(start), dm_thresh,
                      crop_x0=crop_x0, crop_w=crop_w, crop_y0=crop_y0, crop_h=crop_h)


def _load_robot(leaf, localize_zed=False):
    csvs = sorted(Path(leaf).glob("robot_raw*.csv"))
    if not csvs:
        return None
    cap = we.read_capture_csv(str(csvs[-1]))
    meta = we.load_capture_meta(str(csvs[-1]))
    if localize_zed:
        _ensure_zed_pose(leaf, meta)
    side = meta.get("zed_localize_side", "left")
    zed_video = _zed_video_for(leaf, side)            # SBS stereo preferred
    return {
        "leaf": str(leaf), "meta": meta,
        "t_epoch": np.asarray(cap["t_epoch"], float),
        "q": _joints(cap, "q"), "qgoal": _joints(cap, "qgoal"),
        "dq": _joints(cap, "dq"), "tau": _joints(cap, "tau"),
        "fingers": np.concatenate([cap["finger_left"], cap["finger_right"]], axis=1),
        "base": _load_pose_csv(leaf, ["head_base*.csv"]),
        "zed_pose": _load_pose_csv(leaf, ["zed_pose_*.csv"]),
        "crate": _load_pose_csv(leaf, ["crate_pose*.csv"]),
        "zed_video": str(zed_video) if zed_video else None,
        # Crop the localization eye out of an SBS stereo frame so the image observation
        # is ONE eye (master=left, peer=right), not the squashed full stereo frame.
        "zed_crop": _zed_side_crop(meta, zed_video),
        "zed_start_epoch": meta.get("zed_record_start_epoch"),
    }


def _find_attempts(session_dir):
    """Yield (cond, attempt, {robot_id: leaf_path}) for every attempt with ≥1 robot leaf."""
    root = Path(session_dir)
    leaves = sorted(root.rglob("robot_raw*.csv"))
    attempts = {}
    for csv in leaves:
        leaf = csv.parent
        m = _ROBOT_LEAF.match(leaf.name)
        if not m:                                   # legacy flat layout → treat as robot 1
            rid, adir = 1, leaf
        else:
            rid, adir = int(m.group(1)), leaf.parent
        key = str(adir)
        attempts.setdefault(key, {"dir": adir, "robots": {}})["robots"][rid] = leaf
    for a in attempts.values():
        cond = a["dir"].parent.name
        att = a["dir"].name
        yield cond, att, a["robots"]


# --------------------------------------------------------------------------- #
# Episode assembly
# --------------------------------------------------------------------------- #
def _decode_images(video, start_epoch, t_epoch_q, size, crop=(0, 0, None, None)):
    """Decode the ZED video into (N, size, size, 3) RGB frames at the query times. ``crop``
    = (x0, y0, w, h) selects one eye out of a stereo frame (side-by-side OR top-bottom)
    before the square resize; (0, 0, None, None) keeps the full frame (already
    single-eye)."""
    if cv2 is None or not video or not start_epoch:
        return None
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None
    x0, y0, w, h = crop
    frames = np.zeros((len(t_epoch_q), size, size, 3), dtype=np.uint8)
    for k, te in enumerate(t_epoch_q):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (te - start_epoch) * 1000.0))
        ok, fr = cap.read()
        if ok:
            if w or h:                               # crop the localization eye out
                fr = fr[y0:(y0 + h) if h else None, x0:(x0 + w) if w else None]
            frames[k] = cv2.cvtColor(cv2.resize(fr, (size, size)), cv2.COLOR_BGR2RGB)
    cap.release()
    return frames


def build_episode(robots, rate, want_images, image_size, localize_zed=False):
    """Assemble one aligned episode dict (arrays on the common timeline) or None."""
    if 1 not in robots:                              # need at least the master
        return None
    r1 = _load_robot(robots[1], localize_zed)
    if r1 is None:
        return None
    r2 = _load_robot(robots[2], localize_zed) if 2 in robots else None

    # Put robot 2 on the MASTER clock using the offset in robot 1's meta.
    cs = (r1["meta"].get("clock_sync") or {})
    offset = float(cs.get("link_peer_minus_local_offset_s") or 0.0)   # peer - master
    t1 = r1["t_epoch"]
    spans = [(t1[0], t1[-1])]
    if r2 is not None:
        t2 = r2["t_epoch"] - offset                  # → master clock
        spans.append((t2[0], t2[-1]))
    t0 = max(s[0] for s in spans)
    te = min(s[1] for s in spans)
    if te - t0 < 0.3:                                # <0.3 s overlap → skip
        return None
    tq = np.arange(t0, te, 1.0 / rate)               # common timeline (master clock)
    jgap = 1.5 / rate                                # joints are dense
    pgap = 3.0 / rate                                # poses can be gappy

    ep = {"t_epoch": tq, "t": tq - tq[0], "clock_offset_s": offset,
          "rate_hz": float(rate)}

    def add_robot(tag, R, shift):
        te_r = R["t_epoch"] - shift
        for f in ("q", "qgoal", "dq", "tau", "fingers"):
            v, _ = _nearest(te_r, R[f], tq, jgap)
            ep[f"{tag}_{f}"] = v
        for name, (tp, pv) in (("base", R["base"]), ("zed_pose", R["zed_pose"])):
            v, valid = _nearest(tp - shift, pv, tq, pgap)
            ep[f"{tag}_{name}"] = v
            ep[f"{tag}_{name}_valid"] = valid
        if want_images and R["zed_video"] and R["zed_start_epoch"]:
            imgs = _decode_images(R["zed_video"], R["zed_start_epoch"] - shift, tq,
                                  image_size, R.get("zed_crop", (0, 0, None, None)))
            if imgs is not None:
                ep[f"{tag}_image"] = imgs

    add_robot("r1", r1, 0.0)
    if r2 is not None:
        add_robot("r2", r2, offset)

    # Object (crate) — tracked on the master → in the master clock already.
    ct, cp = r1["crate"]
    if ct.size == 0 and r2 is not None:
        ct, cp = r2["crate"]; ct = ct - offset
    cv, cvalid = _nearest(ct, cp, tq, pgap)
    ep["crate_pose"] = cv
    ep["crate_pose_valid"] = cvalid

    # Convenience: the action vector = both robots' joint + finger COMMANDS.
    act = [ep["r1_qgoal"]]
    if "r2_qgoal" in ep:
        act.append(ep["r2_qgoal"])
    act.append(ep["r1_fingers"])
    if "r2_fingers" in ep:
        act.append(ep["r2_fingers"])
    ep["action"] = np.concatenate(act, axis=1)
    ep["n_steps"] = len(tq)
    ep["quality"] = str(r1["meta"].get("attempt_quality", "unrated"))
    ep["ntp"] = json.dumps(cs.get("ntp", {}))
    return ep


def main(argv=None):
    p = argparse.ArgumentParser(description="Build an aligned HDF5 dataset from raw study "
                                "sessions for centralized-policy training.")
    p.add_argument("sessions", nargs="+", help="session dir(s) (with robot{1,2}/ leaves)")
    p.add_argument("-o", "--out", default="dataset.hdf5")
    p.add_argument("--rate", type=float, default=15.0, help="common sampling rate (Hz)")
    p.add_argument("--images", action="store_true", help="decode ZED-left frames (slow)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--only-good", action="store_true",
                   help="skip attempts whose meta attempt_quality == 'bad'")
    p.add_argument("--localize-zed", action="store_true",
                   help="first localize each ZED video (zed_localize) using the meta "
                        "intrinsics, so zed_pose is available for the extrinsics")
    args = p.parse_args(argv)

    manifest, n = [], 0
    with h5py.File(args.out, "w") as h5:
        h5.attrs["rate_hz"] = args.rate
        for session in args.sessions:
            for cond, att, robots in _find_attempts(session):
                ep = build_episode(robots, args.rate, args.images, args.image_size,
                                   localize_zed=args.localize_zed)
                if ep is None:
                    print(f"[ds] skip {session}:{cond}/{att} (no overlap / no master)")
                    continue
                if args.only_good and ep["quality"] == "bad":
                    print(f"[ds] skip {cond}/{att} (marked bad)")
                    continue
                g = h5.create_group(f"episode_{n:04d}")
                for k, v in ep.items():
                    if isinstance(v, np.ndarray):
                        g.create_dataset(k, data=v, compression="gzip")
                    else:
                        g.attrs[k] = v
                g.attrs["condition"] = cond
                g.attrs["attempt"] = att
                manifest.append({"episode": n, "session": str(session), "condition": cond,
                                 "attempt": att, "n_steps": ep["n_steps"],
                                 "quality": ep["quality"],
                                 "clock_offset_s": ep["clock_offset_s"],
                                 "robots": sorted(robots)})
                print(f"[ds] episode_{n:04d}  {cond}/{att}  {ep['n_steps']} steps  "
                      f"robots={sorted(robots)}  quality={ep['quality']}")
                n += 1
        h5.attrs["n_episodes"] = n
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(
        {"rate_hz": args.rate, "n_episodes": n, "episodes": manifest}, indent=2))
    print(f"[ds] wrote {n} episodes → {args.out} (+ .manifest.json)")


if __name__ == "__main__":
    main()
