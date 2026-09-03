"""
Kneeling-pose calibration for the G1 lower body.

When the robot teleoperates while kneeling, the legs are locked at a fixed pose
and the waist/torso is held at a fixed angle. The nominal pose keeps one robot
perfectly upright, but unit-to-unit differences (joint zero offsets, backlash,
mass distribution) can leave another robot sagging — e.g. leaning backward. This
module stores small per-robot trims to the locked leg and waist angles so each
robot can be nudged to a true upright kneel, and persists them to JSON so the
trim is reused every run.

Offsets (radians) are added to the nominal locked pose. The leg offsets are
applied symmetrically to both legs:

    hip_pitch   += hip_pitch_offset      (tilts the whole upper body fore/aft)
    knee        += knee_offset
    ankle_pitch += ankle_pitch_offset

and the waist is held at:

    [waist_yaw_offset, waist_roll_offset, waist_pitch_offset]

All offsets default to zero (i.e. the nominal demo pose) so an uncalibrated
robot behaves exactly as before.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

# Canonical on-disk location so the study manager can auto-load it at start.
DEFAULT_CALIBRATION_PATH: Path = Path(__file__).with_name(
    "kneeling_pose_calibration.json"
)

# Indices into the 12-element leg array (left[6] + right[6]) for each tunable
# joint. Leg layout per side: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch,
# ankle_roll]. Offsets are applied to both legs symmetrically.
_LEG_PARAM_INDICES: dict[str, tuple[int, ...]] = {
    "hip_pitch":   (0, 6),
    "knee":        (3, 9),
    "ankle_pitch": (4, 10),
}

# Indices into the 3-element waist/torso array [yaw, roll, pitch].
_TORSO_PARAM_INDEX: dict[str, int] = {
    "waist_yaw":   0,
    "waist_roll":  1,
    "waist_pitch": 2,
}

# Ordered tunable parameters for the calibration UI: (key, label, kind).
# Waist pitch and hip pitch are the primary trims for a fore/aft sag; waist roll
# corrects a left/right lean; knee/ankle pitch fine-tune posture and height.
PARAMS: list[tuple[str, str, str]] = [
    ("waist_pitch", "Waist Pitch", "torso"),
    ("waist_roll",  "Waist Roll",  "torso"),
    ("hip_pitch",   "Hip Pitch",   "leg"),
    ("knee",        "Knee",        "leg"),
    ("ankle_pitch", "Ankle Pitch", "leg"),
]

_PARAM_KEYS: tuple[str, ...] = tuple(key for key, _, _ in PARAMS)


class KneelingCalibration:
    """Per-robot trims (radians) to the locked kneeling leg/waist angles."""

    def __init__(self) -> None:
        self.created: str = datetime.now().isoformat(timespec="seconds")
        self.offsets: dict[str, float] = {key: 0.0 for key in _PARAM_KEYS}

    # ── access ────────────────────────────────────────────────────────────────

    def get(self, key: str) -> float:
        return float(self.offsets.get(key, 0.0))

    def set(self, key: str, value: float) -> None:
        if key in self.offsets:
            self.offsets[key] = float(value)

    def reset(self) -> None:
        for key in self.offsets:
            self.offsets[key] = 0.0

    def has_any_offset(self) -> bool:
        return any(abs(v) > 1e-9 for v in self.offsets.values())

    # ── derived vectors ─────────────────────────────────────────────────────────

    def leg_offset_vector(self) -> np.ndarray:
        """12-element offset to add to the nominal kneeling leg angles."""
        v = np.zeros(12)
        for key, idxs in _LEG_PARAM_INDICES.items():
            off = self.get(key)
            for i in idxs:
                v[i] += off
        return v

    def torso_offset_vector(self) -> np.ndarray:
        """3-element waist hold angle [yaw, roll, pitch]."""
        v = np.zeros(3)
        for key, i in _TORSO_PARAM_INDEX.items():
            v[i] += self.get(key)
        return v

    def apply_to_legs(self, base_legs) -> np.ndarray:
        """Return the calibrated 12-element kneeling leg angles."""
        return np.asarray(base_legs, dtype=np.float64) + self.leg_offset_vector()

    # ── persistence ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"created": self.created, "version": 1, "offsets": self.offsets}

    def save(self, path: str | Path = DEFAULT_CALIBRATION_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.created = datetime.now().isoformat(timespec="seconds")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CALIBRATION_PATH) -> "KneelingCalibration":
        obj = cls()
        path = Path(path)
        if not path.exists():
            return obj
        try:
            with open(path) as f:
                raw = json.load(f)
            obj.created = raw.get("created", obj.created)
            for key, val in raw.get("offsets", {}).items():
                if key in obj.offsets:
                    obj.offsets[key] = float(val)
        except Exception as exc:  # noqa: BLE001 - calibration is best-effort
            print(f"[KneelingCalibration] failed to load {path}: {exc}")
        return obj
