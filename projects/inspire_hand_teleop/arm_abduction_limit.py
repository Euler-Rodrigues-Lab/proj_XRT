"""
Maximum arm-abduction limit for the G1 shoulders.

"Abduction" is raising the upper arm outward, away from the torso — driven by
the shoulder-roll joint (index 1 of each 7-DOF arm). With shoulder heat sinks
installed, abducting too far makes the heat sink collide with the body, so this
module stores a per-arm maximum abduction angle that clamps the commanded
shoulder roll during teleop. The trim persists to JSON so it is reused every
session and can be tuned right up to the physical limit.

Sign convention (from the URDF g1_29dof_rev_1_0 shoulder_roll_joint, axis +X):
    left  arm: abduction is POSITIVE roll  (joint range [-1.5882, 2.2515])
    right arm: abduction is NEGATIVE roll  (joint range [-2.2515, 1.5882])
So the limits below are stored as positive magnitudes and applied as:
    q_left[1]  <= +left_max          (clamp the upper bound)
    q_right[1] >= -right_max         (clamp the lower bound)
Adduction (inward / lowering) is never restricted — only outward raising is.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

# Index of the shoulder-roll DOF within each 7-DOF arm
# [Shoulder Pitch, Shoulder Roll, Shoulder Yaw, Elbow, Wrist Roll/Pitch/Yaw].
SHOULDER_ROLL_IDX: int = 1

# Physical abduction limit (radians) from the URDF shoulder_roll joint range.
# Used as the default (no extra restriction) and as the UI's upper bound.
JOINT_ABDUCTION_MAX_RAD: float = 2.2515

# Canonical on-disk location so the study manager can auto-load it at start.
DEFAULT_LIMIT_PATH: Path = Path(__file__).with_name("arm_abduction_limit.json")


class ArmAbductionLimit:
    """Per-arm maximum abduction angle (radians, positive magnitude)."""

    def __init__(self) -> None:
        self.created: str = datetime.now().isoformat(timespec="seconds")
        # Default to the physical joint limit => no additional restriction until
        # the operator tunes it down to clear the heat sinks.
        self.left_max: float = JOINT_ABDUCTION_MAX_RAD
        self.right_max: float = JOINT_ABDUCTION_MAX_RAD

    # ── access ────────────────────────────────────────────────────────────────

    def set_left(self, value_rad: float) -> None:
        self.left_max = float(np.clip(value_rad, 0.0, JOINT_ABDUCTION_MAX_RAD))

    def set_right(self, value_rad: float) -> None:
        self.right_max = float(np.clip(value_rad, 0.0, JOINT_ABDUCTION_MAX_RAD))

    def is_restrictive(self) -> bool:
        """True if either limit is tighter than the physical joint range."""
        eps = 1e-6
        return (self.left_max < JOINT_ABDUCTION_MAX_RAD - eps
                or self.right_max < JOINT_ABDUCTION_MAX_RAD - eps)

    # ── application ─────────────────────────────────────────────────────────────

    def clamp_arms(self, q_left, q_right):
        """Return abduction-clamped copies of the two 7-DOF arm goal vectors."""
        ql = np.asarray(q_left, dtype=np.float64).copy() if q_left is not None else None
        qr = np.asarray(q_right, dtype=np.float64).copy() if q_right is not None else None
        if ql is not None and ql.size > SHOULDER_ROLL_IDX and ql[SHOULDER_ROLL_IDX] > self.left_max:
            ql[SHOULDER_ROLL_IDX] = self.left_max
        if qr is not None and qr.size > SHOULDER_ROLL_IDX and qr[SHOULDER_ROLL_IDX] < -self.right_max:
            qr[SHOULDER_ROLL_IDX] = -self.right_max
        return ql, qr

    @staticmethod
    def abduction_deg(side: str, shoulder_roll_rad: float) -> float:
        """Convert a measured shoulder-roll angle to abduction degrees
        (positive = raised outward), accounting for the per-arm sign."""
        roll = float(shoulder_roll_rad)
        return float(np.rad2deg(roll if side == "left" else -roll))

    # ── persistence ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"created": self.created, "version": 1,
                "left_max_rad": self.left_max, "right_max_rad": self.right_max}

    def save(self, path: str | Path = DEFAULT_LIMIT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.created = datetime.now().isoformat(timespec="seconds")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LIMIT_PATH) -> "ArmAbductionLimit":
        obj = cls()
        path = Path(path)
        if not path.exists():
            return obj
        try:
            with open(path) as f:
                raw = json.load(f)
            obj.created = raw.get("created", obj.created)
            obj.set_left(raw.get("left_max_rad", obj.left_max))
            obj.set_right(raw.get("right_max_rad", obj.right_max))
        except Exception as exc:  # noqa: BLE001 - calibration is best-effort
            print(f"[ArmAbductionLimit] failed to load {path}: {exc}")
        return obj
