#!/usr/bin/env python3
"""
robot_base_calibration.py — localize a robot's floating base in the TABLE frame
by touching known points (rigid point-set registration, a.k.a. Kabsch/Umeyama).

Why: the digital-twin fusion needs each robot's base pose ``T_world_base`` in the
table (tag-0) frame, but the recorded joints alone don't give it (the G1's onboard
sensors can't supply an absolute world pose). This needs no cameras, so it also
localizes the camera-less robot 2.

Procedure
---------
1. Lay out >=3 (use 4-6) physical targets whose coordinates in the TABLE frame are
   known — e.g. crosshairs on a board registered to tag 0, or features measured
   relative to tag 0. Spread them out; don't make them collinear; if you can,
   include some height variation (a known-height block) for a better-conditioned
   solve.
2. Teleop the robot so a known reference point (a stylus tip rigidly held by the
   hand, with a known tip offset — far more precise than a soft fingertip) touches
   each target. When still, capture: record the joints; FK gives the reference
   point ``a_i`` in the robot's BASE (pelvis) frame. Pair it with the known world
   point ``b_i``.
3. Solve ``T_world_base`` with :func:`kabsch`, check the RMS residual, save.

The capture (which joints, which reference site) is host-specific and done by the
manager; this module is the math + accumulation + save/load, fully testable
offline. Run ``python robot_base_calibration.py --selftest`` to verify.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np


def kabsch(src, dst, weights=None):
    """Best-fit rigid transform mapping src -> dst (no scaling).

    Args:
        src: (N,3) points in the source frame (e.g. robot base frame).
        dst: (N,3) corresponding points in the target frame (e.g. table frame).
        weights: optional (N,) non-negative weights.

    Returns:
        (T 4x4, rms, per_point_residual (N,)) where T maps src into dst. The
        determinant correction guarantees a proper rotation (never a reflection).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must be matching (N,3) arrays")
    if src.shape[0] < 3:
        raise ValueError("need at least 3 point correspondences")

    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    w = w / w.sum()

    cs = (w[:, None] * src).sum(axis=0)
    cd = (w[:, None] * dst).sum(axis=0)
    A = src - cs
    B = dst - cd
    H = (A * w[:, None]).T @ B            # weighted cross-covariance (3x3)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T   # proper rotation (no reflection)
    t = cd - R @ cs

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    resid = (R @ src.T).T + t - dst
    per_point = np.linalg.norm(resid, axis=1)
    rms = float(np.sqrt(np.mean(per_point ** 2)))
    return T, rms, per_point


class TouchCalibration:
    """Accumulates touch correspondences and solves for ``T_world_base``.

    The host adds (base_point, world_point) pairs — base_point is the reference
    point expressed in the robot BASE frame (from FK at the captured joints);
    world_point is the known target in the TABLE frame.
    """

    def __init__(self):
        self.base_pts: list = []
        self.world_pts: list = []
        self.labels: list = []

    def add(self, base_pt, world_pt, label: str = ""):
        self.base_pts.append(np.asarray(base_pt, dtype=np.float64).reshape(3))
        self.world_pts.append(np.asarray(world_pt, dtype=np.float64).reshape(3))
        self.labels.append(label or f"P{len(self.base_pts)}")

    def clear(self):
        self.base_pts.clear()
        self.world_pts.clear()
        self.labels.clear()

    @property
    def n(self) -> int:
        return len(self.base_pts)

    def solve(self):
        """Return (T_world_base 4x4, rms, per_point (N,)). Raises if <3 points."""
        return kabsch(np.array(self.base_pts), np.array(self.world_pts))

    def save(self, path: str, robot_id=None, extra: dict | None = None) -> dict:
        T, rms, per = self.solve()
        out = {
            "robot_id": robot_id,
            "frame": "table_apriltag_id0",
            "T_world_base": np.round(T, 6).tolist(),
            "rms_mm": round(rms * 1000.0, 3),
            "n_points": self.n,
            "points": [
                {"label": self.labels[i],
                 "base": np.round(self.base_pts[i], 6).tolist(),
                 "world": np.round(self.world_pts[i], 6).tolist(),
                 "resid_mm": round(float(per[i]) * 1000.0, 3)}
                for i in range(self.n)
            ],
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        if extra:
            out.update(extra)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return out


def load_base_calib(path: str) -> np.ndarray:
    """Load a saved calibration and return T_world_base (4x4)."""
    with open(path) as f:
        return np.array(json.load(f)["T_world_base"], dtype=np.float64)


def _selftest():
    from numpy.random import default_rng
    rng = default_rng(0)

    def rand_R():
        # random rotation via QR (proper)
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q

    print("=== exact recovery ===")
    R_gt, t_gt = rand_R(), rng.uniform(-1, 1, 3)
    src = rng.uniform(-0.5, 0.5, (6, 3))
    dst = (R_gt @ src.T).T + t_gt
    T, rms, per = kabsch(src, dst)
    print(f"  R err {np.abs(T[:3,:3]-R_gt).max():.2e}  t err {np.abs(T[:3,3]-t_gt).max():.2e}  rms {rms:.2e}")
    assert np.allclose(T[:3, :3], R_gt, atol=1e-9) and np.allclose(T[:3, 3], t_gt, atol=1e-9)
    assert np.linalg.det(T[:3, :3]) > 0
    print("  PASS")

    print("=== noisy (2 mm/pt), 6 points ===")
    dst_n = dst + rng.normal(0, 0.002, dst.shape)
    T2, rms2, per2 = kabsch(src, dst_n)
    rerr = np.degrees(np.arccos(np.clip((np.trace(T2[:3, :3].T @ R_gt) - 1) / 2, -1, 1)))
    print(f"  rms {rms2*1000:.2f} mm  rot err {rerr:.3f} deg  t err {np.linalg.norm(T2[:3,3]-t_gt)*1000:.2f} mm")
    assert rms2 < 0.004 and rerr < 1.0
    print("  PASS")

    print("=== reflection guard (degenerate/mirrored data) ===")
    src_planar = rng.uniform(-0.5, 0.5, (5, 3)); src_planar[:, 2] = 0.0
    dst_planar = (R_gt @ src_planar.T).T + t_gt
    T3, _, _ = kabsch(src_planar, dst_planar)
    print(f"  coplanar src: det(R) = {np.linalg.det(T3[:3,:3]):+.6f} (must be +1)")
    assert np.linalg.det(T3[:3, :3]) > 0.99
    print("  PASS")

    print("=== TouchCalibration round-trip ===")
    tc = TouchCalibration()
    for i in range(5):
        tc.add(src[i], dst[i], f"P{i+1}")
    Tt, rmst, _ = tc.solve()
    print(f"  solved {tc.n} pts, rms {rmst*1000:.3f} mm")
    assert rmst < 1e-6
    print("  PASS")
    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robot base touch calibration (Kabsch).")
    p.add_argument("--selftest", action="store_true", help="run synthetic checks")
    args = p.parse_args()
    if args.selftest:
        _selftest()
    else:
        p.print_help()
