"""
Fingertip / palm tactile calibration for the Inspire RH56DFTP hand.

The hand's tactile arrays report raw int16 ADC counts. The haptics use that RAW
signal directly (see ``region_raw`` — peak tip/nail for fingers, mean for the
palm — mapped to the 0..1 intensity scale Unity thresholds on). The ONLY
calibration is per-(hand, finger, sub-region) baseline + idle NOISE FLOOR:

Per-sub-region calibration
---------------------------
A finger has tactile patches — ``tip`` (3x3) and ``nail`` (12x8, the main contact
pad) — measured independently, each with its own baseline and noise floor. The
``noise_floor`` (counts above baseline observed while idle) is used as a deadband
to suppress phantom triggers; the raw intensity above it is unchanged.

(The weight→grams force calibration was removed — the haptics never needed a
physical force value; the raw sensor reading drives them directly.)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Region layout (must match InspireHandHapticsStreamer ordering / naming)
# ---------------------------------------------------------------------------

# Inspire finger register key -> short region label, in display order.
FINGER_MAP: list[tuple[str, str]] = [
    ("fingerfive",  "thumb"),
    ("fingerfour",  "index"),
    ("fingerthree", "middle"),
    ("fingertwo",   "ring"),
    ("fingerone",   "little"),
    ("fingerpalm",  "palm"),
]
REGION_KEYS: list[str] = [ik for ik, _ in FINGER_MAP]
_LABEL: dict[str, str] = {ik: hf for ik, hf in FINGER_MAP}

# Sub-regions per finger; the palm is a single region.
# Internal keys come from the device map ('tip', manufacturer 'top', 'pulp'); the
# code historically renamed 'top'->'nail' and 'pulp'->'pad'. EMPIRICALLY the 12x8
# 'nail'/'top' patch is the MAIN contact pad (it spikes when a weight rests on the
# finger palm-up, and the haptics metric already uses max(tip, nail)); the 10x8
# 'pad'/'pulp' patch is a secondary area that barely loads palm-up — it is omitted
# from the calibration targets below (kept only for back-compat / diagnostics).
FINGER_SUBREGIONS: tuple[str, ...] = ("tip", "nail")
PALM_SUBREGION: str = "palm"

# Friendly display names (the raw 'nail'/'pad' keys are misleading — see above).
SUBREGION_LABEL: dict[str, str] = {
    "tip":  "tip (3x3)",
    "nail": "pad (12x8)",
    "pad":  "pad-pulp (10x8)",   # retained for diagnostics only; not a target
    "palm": "palm",
}


def subregion_label(sub: str) -> str:
    return SUBREGION_LABEL.get(sub, sub)

# Tip / nail / pad slice bounds within each finger's *full-mode* flattened touch
# array.  Layout is fixed by the device firmware:
#   fingers 1-4: [tip(3x3=9), nail(12x8=96), pad(10x8=80)]                 -> 185
#   thumb (5)  : [tip(3x3=9), nail(12x8=96), mid(3x3=9), pad(12x8=96)]     -> 210
_FINGER_SLICES: dict[str, slice] = {
    "tip":  slice(0, 9),
    "nail": slice(9, 105),
    "pad":  slice(105, 185),
}
_THUMB_SLICES: dict[str, slice] = {
    "tip":  slice(0, 9),
    "nail": slice(9, 105),
    "pad":  slice(114, 210),   # skips the thumb's 3x3 "mid" patch (105:114)
}

# Canonical on-disk location so the haptics streamer can auto-load it at start.
DEFAULT_CALIBRATION_PATH: Path = Path(__file__).with_name(
    "fingertip_force_calibration.json"
)


def subregions_for(finger_key: str) -> tuple[str, ...]:
    return (PALM_SUBREGION,) if finger_key == "fingerpalm" else FINGER_SUBREGIONS


def label_for(finger_key: str) -> str:
    return _LABEL.get(finger_key, finger_key)


def _finger_slice(finger_key: str, sub: str) -> slice | None:
    table = _THUMB_SLICES if finger_key == "fingerfive" else _FINGER_SLICES
    return table.get(sub)


def subregion_raw(arr, finger_key: str, sub: str) -> float:
    """Peak (non-negative) taxel value for one sub-region, or mean for the palm.

    Returns 0.0 if the array is too short to contain the sub-region (e.g. the
    pad is absent in the latency-optimised tip+nail ``haptics_only`` read).
    """
    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return 0.0
    arr = np.asarray(arr)
    if finger_key == "fingerpalm":
        return max(0.0, float(np.mean(arr)))
    sl = _finger_slice(finger_key, sub)
    if sl is None or arr.size <= sl.start:
        return 0.0
    seg = arr[sl]
    if seg.size == 0:
        return 0.0
    return max(0.0, float(np.max(seg)))


def subregion_segment(arr, finger_key: str, sub: str):
    """Return the raw (signed, *unclamped*) taxel values for a sub-region, or
    ``None`` if absent. For diagnostics: distinguishes a dead patch (all ~0) from
    a negative-offset patch (which ``subregion_raw`` clamps to 0) and from genuine
    no-contact."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    if finger_key == "fingerpalm":
        return arr
    sl = _finger_slice(finger_key, sub)
    if sl is None or arr.size <= sl.start:
        return None
    seg = arr[sl]
    return seg if seg.size else None


def subregion_present(arr, finger_key: str, sub: str) -> bool:
    """True if ``arr`` actually contains this sub-region.  The pad lives past the
    end of the latency-optimised tip+nail (``haptics_only``) read, so this lets
    the UI distinguish 'no contact' (0) from 'not being read' (n/a)."""
    if arr is None:
        return False
    arr = np.asarray(arr)
    if finger_key == "fingerpalm":
        return arr.size > 0
    sl = _finger_slice(finger_key, sub)
    return sl is not None and arr.size > sl.start


def region_raw(arr, finger_key: str) -> float:
    """Live haptics per-region metric — UNCHANGED behaviour (peak of tip/nail for
    fingers, mean for the palm).  Source of truth shared with the streamer so the
    0..1 intensity scale and Unity thresholds stay stable."""
    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return 0.0
    if finger_key == "fingerpalm":
        return max(0.0, float(np.mean(np.asarray(arr))))
    return max(subregion_raw(arr, finger_key, "tip"),
               subregion_raw(arr, finger_key, "nail"))


def region_raw_from_snapshot(snapshot: dict, side: str, finger_key: str) -> float:
    arr = snapshot.get("touch", {}).get(side, {}).get(finger_key)
    return region_raw(arr, finger_key)


def subregion_raw_from_snapshot(snapshot: dict, side: str, finger_key: str, sub: str) -> float:
    arr = snapshot.get("touch", {}).get(side, {}).get(finger_key)
    return subregion_raw(arr, finger_key, sub)


def _new_entry() -> dict:
    return {"baseline": 0.0, "noise_floor": 0.0}


class ForceCalibration:
    """Per-(hand, finger, sub-region) baseline + idle noise floor for the fingertip haptics.

    The haptics use the RAW force-sensor readings directly; the only calibration is the noise
    floor (used as a deadband to suppress phantom triggers). There is no weight/grams fit.
    """

    def __init__(self) -> None:
        self.created: str = datetime.now().isoformat(timespec="seconds")
        self.data: dict[str, dict[str, dict[str, dict]]] = {
            side: {
                ik: {sub: _new_entry() for sub in subregions_for(ik)}
                for ik in REGION_KEYS
            }
            for side in ("Left", "Right")
        }

    # ── mutation ────────────────────────────────────────────────────────────

    def set_baseline(self, side: str, finger_key: str, sub: str,
                     baseline: float, noise_floor: float = 0.0) -> None:
        e = self.data[side][finger_key][sub]
        e["baseline"] = float(baseline)
        e["noise_floor"] = float(noise_floor)

    # ── query ─────────────────────────────────────────────────────────────────

    def live_noise_floor(self, side: str, finger_key: str) -> float:
        """Deadband (counts) for the *live* metric (``region_raw``).  The live
        metric is max(tip, nail) for fingers, so its idle noise floor is the max
        of those sub-regions'; the palm uses its own."""
        d = self.data.get(side, {}).get(finger_key, {})
        if finger_key == "fingerpalm":
            return float(d.get(PALM_SUBREGION, {}).get("noise_floor", 0.0))
        return max(float(d.get("tip", {}).get("noise_floor", 0.0)),
                   float(d.get("nail", {}).get("noise_floor", 0.0)))

    def has_any_noise_floor(self) -> bool:
        return any(
            self.data[side][ik][sub].get("noise_floor", 0.0) > 0.0
            for side in ("Left", "Right") for ik in REGION_KEYS
            for sub in subregions_for(ik)
        )

    # ── persistence ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"created": self.created, "version": 3,
                "metric": "haptics_raw_per_subregion", "data": self.data}

    def save(self, path: str | Path = DEFAULT_CALIBRATION_PATH,
             csv_path: str | Path | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        if csv_path is not None:
            self.save_csv(csv_path)
        return path

    def save_csv(self, csv_path: str | Path) -> Path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hand", "finger", "subregion", "baseline", "noise_floor"])
            for side in ("Left", "Right"):
                for ik in REGION_KEYS:
                    for sub in subregions_for(ik):
                        e = self.data[side][ik][sub]
                        w.writerow([side, label_for(ik), sub,
                                    e.get("baseline", 0.0), e.get("noise_floor", 0.0)])
        return csv_path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CALIBRATION_PATH) -> "ForceCalibration":
        obj = cls()
        path = Path(path)
        if not path.exists():
            return obj
        try:
            with open(path) as f:
                raw = json.load(f)
            obj.created = raw.get("created", obj.created)
            loaded = raw.get("data", {})
            for side in ("Left", "Right"):
                for ik in REGION_KEYS:
                    for sub in subregions_for(ik):
                        src = loaded.get(side, {}).get(ik, {}).get(sub)
                        if isinstance(src, dict):
                            # Copy ONLY the keys we still use, so an old JSON's leftover
                            # slope/r2/points load without error and are never re-persisted.
                            for k in ("baseline", "noise_floor"):
                                if k in src:
                                    obj.data[side][ik][sub][k] = float(src[k])
        except Exception as exc:  # noqa: BLE001 - calibration is best-effort
            print(f"[ForceCalibration] failed to load {path}: {exc}")
        return obj
