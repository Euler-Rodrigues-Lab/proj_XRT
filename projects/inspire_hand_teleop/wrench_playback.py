#!/usr/bin/env python3
"""Offline playback + end-effector wrench visualisation for G1 arm captures.

Reads a wrench-capture CSV (written by the "Wrench Capture" mode in
``XRT_user_study_manager.py``), re-estimates the external wrench at the hand for
every recorded frame using :mod:`wrench_estimation`, and:

  * replays the recorded arm/torso motion in a MuJoCo passive viewer, drawing a
    force arrow (and optional torque arrow) at the palm, and
  * prints a known-mass accuracy report (mean |F| vs m·g, direction error), and
  * optionally saves a wrench-vs-time plot.

The estimate is quasi-static (see wrench_estimation.py): it is most accurate on
slow / still frames, so the accuracy report is computed over "still" frames only
(joint speed below --still-thresh) when a known mass is given.

Examples
--------
  # Visualise a capture, both hands:
  python wrench_playback.py capture.csv

  # Validate a 500 g mass hung in the right hand, headless, with a plot:
  python wrench_playback.py capture.csv --side right --known-mass 500 \
         --tare auto --no-view --plot wrench.png

The known-mass test: with the arm holding still and empty, run a short capture
and use it (or --tare auto on the same file's still frames) as the baseline.
Then place a known mass in the palm, hold still, capture, and run with
--known-mass <grams>. |F| should read ≈ m·g (e.g. 500 g → 4.9 N).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running both as `python wrench_playback.py` from this dir and as a module.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wrench_estimation as we  # noqa: E402


# --------------------------------------------------------------------------- #
# Wrench computation over a whole capture
# --------------------------------------------------------------------------- #
def compute_wrench_series(est: we.WrenchEstimator, cap: dict, side: str,
                          tau_sign=None) -> dict:
    """Estimate the wrench for every frame of one side. Returns dict with
    'wrench' (N,6), 'point' (N,3), 'speed' (N,) max|dq|, 'tau_gravity' (N,7)."""
    N = cap["t"].size
    q_arm = cap[side]["q"]
    dq_arm = cap[side]["dq"]
    tau_arm = cap[side]["tau"]
    q_torso = cap["torso"]["q"]
    other = "right" if side == "left" else "left"
    q_other = cap[other]["q"]

    wrench = np.zeros((N, 6))
    point = np.zeros((N, 3))
    tau_g = np.zeros((N, 7))
    for i in range(N):
        r = est.estimate(side, q_arm[i], tau_arm[i], q_torso=q_torso[i],
                         q_other_arm=q_other[i], tau_sign=tau_sign)
        wrench[i] = r.wrench
        point[i] = r.point
        tau_g[i] = r.tau_gravity
    speed = np.max(np.abs(dq_arm), axis=1)
    return {"wrench": wrench, "point": point, "speed": speed, "tau_gravity": tau_g}


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Zero-phase-ish moving average along axis 0 (window in samples)."""
    if window <= 1:
        return x
    k = np.ones(window) / window
    out = np.empty_like(x)
    pad = window // 2
    for c in range(x.shape[1]):
        padded = np.pad(x[:, c], pad, mode="edge")
        out[:, c] = np.convolve(padded, k, mode="same")[pad:pad + x.shape[0]]
    return out


def resolve_baseline(series: dict, mode: str, meta: dict, side: str,
                     still_thresh: float, tare_frames: int):
    """Return ``(baseline (6,), used_idx)`` per the --tare mode.

    ``used_idx`` is the set of leading frames consumed to build an auto baseline;
    they are excluded from the known-mass accuracy report so the empty-arm frames
    don't dilute the loaded average (empty=0, returned otherwise)."""
    if mode == "none":
        return np.zeros(6), np.array([], dtype=int)
    if mode == "meta":
        b = (meta.get("baseline") or {}).get(side)
        if b is None:
            print(f"[tare] no baseline for '{side}' in meta.json; using zero.")
            return np.zeros(6), np.array([], dtype=int)
        return np.asarray(b, dtype=np.float64), np.array([], dtype=int)
    if mode == "auto":
        w = series["wrench"]
        still = series["speed"] < still_thresh
        idx = np.where(still)[0]
        if idx.size == 0:
            idx = np.arange(min(tare_frames, w.shape[0]))
        else:
            idx = idx[:tare_frames] if tare_frames > 0 else idx
        base = w[idx].mean(axis=0)
        print(f"[tare] auto baseline from {idx.size} leading still frames: "
              f"F={np.round(base[:3], 3)} N (excluded from the report)")
        return base, idx
    return np.zeros(6), np.array([], dtype=int)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(side: str, wrench: np.ndarray, speed: np.ndarray,
                 mass_g: float | None, still_thresh: float,
                 exclude_idx=None):
    f = wrench[:, :3]
    mag = np.linalg.norm(f, axis=1)
    print(f"\n── {side.upper()} hand wrench summary ──")
    print(f"  frames: {wrench.shape[0]}   |F| mean {mag.mean():.2f} N  "
          f"max {mag.max():.2f} N")
    print(f"  mean force  F = [{f[:,0].mean():+.2f} {f[:,1].mean():+.2f} "
          f"{f[:,2].mean():+.2f}] N")
    print(f"  mean moment M = [{wrench[:,3].mean():+.3f} {wrench[:,4].mean():+.3f} "
          f"{wrench[:,5].mean():+.3f}] N·m")
    if mass_g is not None:
        keep = speed < still_thresh
        if exclude_idx is not None and len(exclude_idx):
            keep[np.asarray(exclude_idx, dtype=int)] = False
        sub = f[keep] if keep.any() else f
        rep = we.accuracy_report(sub, mass_g)
        print(f"\n  ── known-mass check ({mass_g:.0f} g, "
              f"{rep['n_frames']} still frames) ──")
        print(f"     expected |F| = {rep['expected_mag_N']:.3f} N "
              f"(straight down)")
        print(f"     measured |F| = {rep['mean_mag_N']:.3f} ± "
              f"{rep['std_mag_N']:.3f} N")
        print(f"     error        = {rep['mag_error_N']:+.3f} N "
              f"({rep['mag_error_pct']:+.1f} %)")
        print(f"     direction    = {rep['direction_error_deg']:.1f}° off "
              f"vertical")


def save_plot(out_path: str, results: dict, mass_g: float | None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable ({exc}); skipping plot.")
        return
    sides = list(results.keys())
    fig, axes = plt.subplots(len(sides), 1, figsize=(10, 4 * len(sides)),
                             squeeze=False)
    for ax, side in zip(axes[:, 0], sides):
        t = results[side]["t"]
        w = results[side]["wrench"]
        mag = np.linalg.norm(w[:, :3], axis=1)
        ax.plot(t, w[:, 0], label="Fx", lw=1)
        ax.plot(t, w[:, 1], label="Fy", lw=1)
        ax.plot(t, w[:, 2], label="Fz", lw=1)
        ax.plot(t, mag, "k--", label="|F|", lw=1.5)
        if mass_g is not None:
            ax.axhline(we.expected_force_mag(mass_g), color="r", ls=":",
                       label=f"expected |F| ({mass_g:.0f} g)")
        ax.set_title(f"{side.upper()} hand — estimated EE force")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("force (N)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] saved {out_path}")


# --------------------------------------------------------------------------- #
# Viewer playback
# --------------------------------------------------------------------------- #
def _add_arrow(scene, p_from, p_to, width, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, type=mujoco.mjtGeom.mjGEOM_ARROW, size=np.zeros(3),
        pos=np.zeros(3), mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(p_from, dtype=np.float64),
                         np.asarray(p_to, dtype=np.float64))
    scene.ngeom += 1


def _add_sphere(scene, p, r, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([r, r, r]),
        pos=np.asarray(p, dtype=np.float64), mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def play_in_viewer(model_path: str, cap: dict, results: dict, args):
    import mujoco
    import mujoco.viewer
    from wrench_estimation import (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
                                   TORSO_JOINTS, hand_qpos_addrs)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    def qpos_addrs(names):
        return [int(model.joint(n).qposadr[0]) for n in names]

    left_addr = qpos_addrs(LEFT_ARM_JOINTS)
    right_addr = qpos_addrs(RIGHT_ARM_JOINTS)
    torso_addr = qpos_addrs(TORSO_JOINTS)

    # Finger replay (only if the capture recorded finger positions).
    finger_addr = hand_qpos_addrs(model)
    has_fingers = "finger_left" in cap and "finger_right" in cap
    if not has_fingers:
        print("[playback] no finger data in this capture — hands shown static "
              "(re-capture to record finger motion).")

    N = cap["t"].size
    t = cap["t"]
    fs = args.force_scale
    ts = args.torque_scale

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.2, 0.0, 0.9]

        while viewer.is_running():
            for i in range(N):
                if not viewer.is_running():
                    break
                data.qpos[:] = model.qpos0
                data.qpos[torso_addr] = cap["torso"]["q"][i]
                data.qpos[left_addr] = cap["left"]["q"][i]
                data.qpos[right_addr] = cap["right"]["q"][i]
                if has_fingers:
                    data.qpos[finger_addr["finger_left"]] = cap["finger_left"][i]
                    data.qpos[finger_addr["finger_right"]] = cap["finger_right"][i]
                mujoco.mj_forward(model, data)

                viewer.user_scn.ngeom = 0
                line = f"frame {i+1}/{N}  t={t[i]:6.2f}s"
                for side, res in results.items():
                    w = res["wrench"][i]
                    p = res["point"][i]
                    F, M = w[:3], w[3:]
                    fmag = float(np.linalg.norm(F))
                    # colour: green (light) → red (heavy)
                    a = min(1.0, fmag / max(1e-6, args.force_full))
                    rgba = (a, 1.0 - a, 0.1, 0.9)
                    _add_sphere(viewer.user_scn, p, 0.012, (0.2, 0.4, 1.0, 0.9))
                    if fmag > 1e-3:
                        _add_arrow(viewer.user_scn, p, p + F * fs, 0.012, rgba)
                    tmag = float(np.linalg.norm(M))
                    if ts > 0 and tmag > 1e-3:
                        _add_arrow(viewer.user_scn, p, p + M * ts, 0.006,
                                   (0.6, 0.2, 0.9, 0.8))
                    line += (f"   {side[0].upper()}:|F|={fmag:4.1f}N "
                             f"[{F[0]:+.1f} {F[1]:+.1f} {F[2]:+.1f}]")
                viewer.sync()
                print("\r" + line + "   ", end="", flush=True)
                time.sleep(max(0.0, args.dt / max(1e-6, args.speed)))
            print()
            if not args.loop:
                break


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Offline G1 end-effector wrench playback / validation.")
    p.add_argument("csv", help="wrench-capture CSV path")
    p.add_argument("--side", choices=["left", "right", "both"], default=None,
                   help="hand(s) to analyse (default: from meta, else both)")
    p.add_argument("--model", default=None,
                   help="MuJoCo XML (default: from meta, else the bundled model)")
    p.add_argument("--known-mass", type=float, default=None,
                   help="known mass in grams hung in the hand (enables the "
                        "accuracy report; default: from meta if present)")
    p.add_argument("--tare", choices=["none", "auto", "meta"], default="none",
                   help="baseline subtraction: none, auto (mean of still frames), "
                        "or meta (baseline saved by the GUI Tare button)")
    p.add_argument("--tare-frames", type=int, default=30,
                   help="auto-tare: number of leading still frames to average")
    p.add_argument("--still-thresh", type=float, default=0.05,
                   help="max joint speed (rad/s) for a frame to count as 'still'")
    p.add_argument("--smooth", type=int, default=1,
                   help="moving-average window (samples) for display/plot; 1=off")
    p.add_argument("--tau-sign", default=None,
                   help="comma-separated 7 values of +1/-1 to flip joint torque "
                        "signs (debug; default all +1)")
    p.add_argument("--force-scale", type=float, default=0.02,
                   help="arrow length in metres per Newton of force")
    p.add_argument("--torque-scale", type=float, default=0.0,
                   help="arrow length per N·m of moment (0 disables torque arrow)")
    p.add_argument("--force-full", type=float, default=10.0,
                   help="force (N) mapped to fully-red arrow colour")
    p.add_argument("--dt", type=float, default=0.05,
                   help="nominal seconds between playback frames")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed x")
    p.add_argument("--loop", action="store_true", help="loop playback")
    p.add_argument("--no-view", action="store_true",
                   help="skip the viewer; just print the report (+ plot)")
    p.add_argument("--plot", default=None, help="save a wrench-vs-time PNG here")
    args = p.parse_args(argv)

    csv_path = args.csv
    if not Path(csv_path).exists():
        p.error(f"CSV not found: {csv_path}")

    meta = we.load_capture_meta(csv_path)
    cap = we.read_capture_csv(csv_path)

    model_path = args.model or meta.get("model_path") or we.DEFAULT_MODEL_PATH
    side = args.side or meta.get("side") or "both"
    sides = ["left", "right"] if side == "both" else [side]
    mass_g = args.known_mass if args.known_mass is not None else meta.get("known_mass_g")

    tau_sign = None
    if args.tau_sign:
        tau_sign = np.array([float(x) for x in args.tau_sign.split(",")])

    left_off = np.asarray(meta["left_offset"]) if meta.get("left_offset") else None
    right_off = np.asarray(meta["right_offset"]) if meta.get("right_offset") else None
    est = we.WrenchEstimator(model_path, left_offset=left_off, right_offset=right_off)

    print(f"Loaded {cap['t'].size} frames from {Path(csv_path).name} "
          f"({cap['t'][-1]:.1f}s).  model={Path(model_path).name}  sides={sides}"
          + (f"  known_mass={mass_g:.0f} g" if mass_g is not None else ""))

    results = {}
    for s in sides:
        series = compute_wrench_series(est, cap, s, tau_sign=tau_sign)
        base, used_idx = resolve_baseline(series, args.tare, meta, s,
                                          args.still_thresh, args.tare_frames)
        series["wrench"] = series["wrench"] - base
        if args.smooth > 1:
            series["wrench"] = moving_average(series["wrench"], args.smooth)
        series["t"] = cap["t"]
        results[s] = series
        print_report(s, series["wrench"], series["speed"], mass_g,
                     args.still_thresh, exclude_idx=used_idx)

    if args.plot:
        save_plot(args.plot, results, mass_g)

    if not args.no_view:
        try:
            play_in_viewer(model_path, cap, results, args)
        except Exception as exc:
            print(f"[viewer] could not launch ({exc}). "
                  f"Re-run with --no-view for the report only.")


if __name__ == "__main__":
    main()
