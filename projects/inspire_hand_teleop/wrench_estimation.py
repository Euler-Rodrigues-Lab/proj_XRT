"""End-effector wrench estimation for the G1 arms via quasi-static inverse dynamics.

Given the measured joint torques (``tau_est``) of a 7-DOF G1 arm together with the
arm + torso joint configuration, estimate the external **wrench** (force + moment)
acting on the hand, expressed in the world frame at a chosen palm reference point.

Physics — quasi-static residual method
--------------------------------------
The robot's equation of motion, with an external wrench ``w`` applied at the hand::

    M(q) q̈ + qfrc_bias(q, q̇) = tau_motor + J(q)ᵀ w

where ``qfrc_bias`` is MuJoCo's Coriolis + centrifugal + gravity generalized force
and ``J`` is the 6×N spatial Jacobian of the palm point. Holding the arm slowly /
still (q̇ ≈ 0, q̈ ≈ 0) collapses this to::

    qfrc_bias(q, 0) = tau_motor + J(q)ᵀ w
    ⇒ J(q)ᵀ w = qfrc_bias(q, 0) − tau_motor                 (the "residual")
    ⇒ w = pinv(J(q)ᵀ) · (qfrc_bias − tau_motor)             (7 eqns, 6 unknowns → LS)

With the arm unloaded ``w ≈ 0``. Hang a known mass ``m`` at the palm and the world
force component should read ``w_z ≈ −m·g`` (downward), ``|F| ≈ m·g``. This is the
validation test the offline tool reports on.

Assumptions / caveats
----------------------
* **Quasi-static.** Joint accelerations and velocity-product (Coriolis) terms are
  ignored, so the estimate is most accurate when the arm is slow or still. The
  capture records ``dq``/``ddq`` so a full ``M q̈ + C q̇`` correction can be added
  offline later without re-collecting data.
* **Sign convention.** ``tau_est`` is assumed to follow the same per-joint axis
  sign as the URDF/MJCF (it does on the G1, since both come from the same model).
  Validate with the known-mass test; ``tau_sign`` lets you flip a joint if needed.
* **Model fidelity.** Gravity comp uses the URDF link masses baked into the MJCF.
  A small residual offset on the unloaded arm is expected (model error + joint
  friction, ``frictionloss`` in the model). Capture a baseline (tare) with the arm
  empty and subtract it — :class:`WrenchEstimator` is stateless, so baseline
  handling lives in the caller (live panel / playback tool).
* **Friction.** The arms have static friction (``frictionloss`` ≈ 0.8 Nm in the
  model). This shows up as a dead-band / noise floor of a few N on |F|; it bounds
  the achievable accuracy of the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Model / joint layout (matches g1_with_inspire_hand_mujoco_controller.py and
# the hardware index ordering in g1_upper_body_teleop/unitree_api_utils.py).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
DEFAULT_MODEL_PATH = str(
    _HERE / "g1_with_inspire_hand" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.xml"
)

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
TORSO_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

# Legs recorded "down to the knees": hip (pitch/roll/yaw) + knee per leg, 8 total
# (ankles are not recorded). Order matches the recorder's leg index selection
# (left hip+knee, then right hip+knee).
LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",
]
LEG_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS  # 8

# Full Inspire-hand joint set (12 per hand, incl. the coupled distal joints).
# Recorded purely for *visual* playback — finger mass is negligible and the
# joints are distal to the wrist where the wrench is measured, so they do not
# affect the estimate. Recording the full coupled set (rather than the 6 driven
# joints) lets the playback tool write qpos verbatim without re-deriving the
# joint-equality coupling.
LEFT_HAND_JOINTS = [
    "left_thumb_1_joint", "left_thumb_2_joint", "left_thumb_3_joint", "left_thumb_4_joint",
    "left_index_1_joint", "left_index_2_joint", "left_middle_1_joint", "left_middle_2_joint",
    "left_ring_1_joint", "left_ring_2_joint", "left_little_1_joint", "left_little_2_joint",
]
RIGHT_HAND_JOINTS = [
    "right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint",
    "right_index_1_joint", "right_index_2_joint", "right_middle_1_joint", "right_middle_2_joint",
    "right_ring_1_joint", "right_ring_2_joint", "right_little_1_joint", "right_little_2_joint",
]
FINGER_DOF = 12

LEFT_EE_BODY = "left_wrist_yaw_link"
RIGHT_EE_BODY = "right_wrist_yaw_link"

# Palm reference point in the wrist-yaw link frame. Taken from the palm
# force-sensor geom positions in the MJCF — the rough centre of the palm, which
# is where a grasped / hung mass sits. Choosing the point here makes the reported
# *moment* meaningful (≈ 0 for a mass centred on the palm); the reported *force*
# is independent of the reference point either way.
LEFT_EE_OFFSET = np.array([0.1611, -0.012296, 0.00036819])
RIGHT_EE_OFFSET = np.array([0.1611, 0.012296, 0.00036819])

GRAVITY = 9.81  # m/s^2, magnitude (the model's gravity is (0, 0, -9.81)).


@dataclass
class WrenchResult:
    """Result of one wrench estimate, world frame, at the palm reference point."""
    side: str                       # "left" | "right"
    wrench: np.ndarray              # (6,) [Fx, Fy, Fz, Mx, My, Mz]
    point: np.ndarray               # (3,) palm reference point, world coords
    tau_gravity: np.ndarray         # (7,) model gravity-comp torque
    tau_meas: np.ndarray            # (7,) measured joint torque (sign-corrected)
    residual: np.ndarray            # (7,) tau_gravity - tau_meas = Jᵀ w
    jacobian: np.ndarray = field(repr=False, default=None)  # (6, 7)

    @property
    def force(self) -> np.ndarray:
        return self.wrench[:3]

    @property
    def torque(self) -> np.ndarray:
        return self.wrench[3:]

    @property
    def force_mag(self) -> float:
        return float(np.linalg.norm(self.wrench[:3]))

    @property
    def torque_mag(self) -> float:
        return float(np.linalg.norm(self.wrench[3:]))


class WrenchEstimator:
    """Estimate the external wrench at a G1 hand from arm torques (quasi-static).

    Owns a *private* MjModel/MjData so it never touches any sim/viewer state the
    teleop loop is using — safe to call from a background thread. One forward
    pass + one Jacobian per estimate (sub-millisecond on the 53-DOF model).
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 left_offset: np.ndarray | None = None,
                 right_offset: np.ndarray | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.model_path = str(model_path)

        def _dofs(names):
            return [int(self.model.joint(n).dofadr[0]) for n in names]

        def _qpos(names):
            return [int(self.model.joint(n).qposadr[0]) for n in names]

        self._sides = {
            "left": {
                "joints": LEFT_ARM_JOINTS,
                "dof": np.array(_dofs(LEFT_ARM_JOINTS)),
                "qpos": np.array(_qpos(LEFT_ARM_JOINTS)),
                "body": int(self.model.body(LEFT_EE_BODY).id),
                "offset": np.asarray(left_offset if left_offset is not None
                                     else LEFT_EE_OFFSET, dtype=np.float64),
            },
            "right": {
                "joints": RIGHT_ARM_JOINTS,
                "dof": np.array(_dofs(RIGHT_ARM_JOINTS)),
                "qpos": np.array(_qpos(RIGHT_ARM_JOINTS)),
                "body": int(self.model.body(RIGHT_EE_BODY).id),
                "offset": np.asarray(right_offset if right_offset is not None
                                     else RIGHT_EE_OFFSET, dtype=np.float64),
            },
        }
        self._torso_qpos = np.array(_qpos(TORSO_JOINTS))
        # Scratch Jacobian buffers (reused to avoid per-call allocation).
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    # ------------------------------------------------------------------ #
    def palm_offset(self, side: str) -> np.ndarray:
        return self._sides[side]["offset"].copy()

    def estimate(self, side: str, q_arm, tau_arm, q_torso=None,
                 q_other_arm=None, tau_sign=None) -> WrenchResult:
        """Estimate the external wrench at ``side`` hand.

        Args:
            side:        "left" or "right".
            q_arm:       (7,) measured arm joint angles, in the joint order of
                         LEFT_ARM_JOINTS / RIGHT_ARM_JOINTS.
            tau_arm:     (7,) measured joint torques (``tau_est``), same order.
            q_torso:     (3,) waist [yaw, roll, pitch] angles, or None to leave
                         the torso upright. The torso orientation matters because
                         it sets the gravity direction at the shoulder mount.
            q_other_arm: optional (7,) angles for the *other* arm (cosmetic — does
                         not affect this arm's result; useful for visual parity).
            tau_sign:    optional (7,) of ±1 to flip a joint whose hardware torque
                         sign disagrees with the model. Defaults to all +1.

        Returns:
            WrenchResult in the world frame about the palm reference point.
        """
        if side not in self._sides:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        cfg = self._sides[side]
        q_arm = np.asarray(q_arm, dtype=np.float64).reshape(-1)
        tau_arm = np.asarray(tau_arm, dtype=np.float64).reshape(-1)
        if q_arm.size != 7 or tau_arm.size != 7:
            raise ValueError("q_arm and tau_arm must each have 7 elements")

        m, d = self.model, self.data
        # Reset to the model default so untouched joints (legs / fingers / other
        # arm) sit at a defined pose. Legs are below the fixed pelvis and finger
        # mass is negligible, so neither materially affects the arm gravity comp.
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0.0          # quasi-static: qfrc_bias becomes pure gravity
        d.qacc[:] = 0.0

        if q_torso is not None:
            q_torso = np.asarray(q_torso, dtype=np.float64).reshape(-1)
            n = min(self._torso_qpos.size, q_torso.size)
            d.qpos[self._torso_qpos[:n]] = q_torso[:n]
        d.qpos[cfg["qpos"]] = q_arm
        if q_other_arm is not None:
            other = self._sides["right" if side == "left" else "left"]
            q_other_arm = np.asarray(q_other_arm, dtype=np.float64).reshape(-1)
            if q_other_arm.size == 7:
                d.qpos[other["qpos"]] = q_other_arm

        mujoco.mj_forward(m, d)

        dof = cfg["dof"]
        tau_gravity = np.array(d.qfrc_bias[dof], dtype=np.float64)

        if tau_sign is not None:
            tau_sign = np.asarray(tau_sign, dtype=np.float64).reshape(-1)
            tau_meas = tau_arm * tau_sign
        else:
            tau_meas = tau_arm.copy()

        residual = tau_gravity - tau_meas      # == Jᵀ w (static balance)

        # Palm point in world coordinates, rigidly attached to the wrist body.
        bid = cfg["body"]
        R = d.xmat[bid].reshape(3, 3)
        point = d.xpos[bid] + R @ cfg["offset"]

        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        mujoco.mj_jac(m, d, self._jacp, self._jacr, point, bid)
        J = np.vstack((self._jacp[:, dof], self._jacr[:, dof]))   # (6, 7)

        # w = (Jᵀ)⁺ residual. lstsq on Jᵀ (7×6) is the robust least-squares solve
        # of the overdetermined system Jᵀ w = residual.
        wrench, *_ = np.linalg.lstsq(J.T, residual, rcond=None)

        return WrenchResult(
            side=side, wrench=wrench, point=point.copy(),
            tau_gravity=tau_gravity, tau_meas=tau_meas, residual=residual,
            jacobian=J,
        )


# --------------------------------------------------------------------------- #
# Small helpers used by the live panel and the offline validation report.
# --------------------------------------------------------------------------- #
def expected_force_for_mass(mass_g: float, g: float = GRAVITY) -> np.ndarray:
    """World-frame force a mass of ``mass_g`` grams exerts on the hand (downward)."""
    return np.array([0.0, 0.0, -(mass_g / 1000.0) * g])


def expected_force_mag(mass_g: float, g: float = GRAVITY) -> float:
    return (mass_g / 1000.0) * g


def accuracy_report(forces: np.ndarray, mass_g: float,
                    g: float = GRAVITY) -> dict:
    """Compare a stack of estimated forces (N×3) against a known hanging mass.

    Returns a dict of summary stats: measured mean force, magnitude, expected
    magnitude, percent error, and the angular deviation of the mean force from
    straight-down. (Apply any baseline subtraction *before* calling this.)
    """
    forces = np.atleast_2d(np.asarray(forces, dtype=np.float64))
    mean_f = forces.mean(axis=0)
    mags = np.linalg.norm(forces, axis=1)
    expected_mag = expected_force_mag(mass_g, g)

    down = np.array([0.0, 0.0, -1.0])
    mean_norm = np.linalg.norm(mean_f)
    if mean_norm > 1e-9:
        cos = float(np.clip(np.dot(mean_f, down) / mean_norm, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos)))
    else:
        angle_deg = float("nan")

    return {
        "n_frames": int(forces.shape[0]),
        "mass_g": float(mass_g),
        "expected_mag_N": float(expected_mag),
        "expected_force_N": expected_force_for_mass(mass_g, g),
        "mean_force_N": mean_f,
        "mean_mag_N": float(mags.mean()),
        "std_mag_N": float(mags.std()),
        "mag_error_N": float(mags.mean() - expected_mag),
        "mag_error_pct": (float(100.0 * (mags.mean() - expected_mag) / expected_mag)
                          if expected_mag > 1e-9 else float("nan")),
        "direction_error_deg": angle_deg,
    }


# --------------------------------------------------------------------------- #
# Capture / session file I/O — shared schema so the recorder
# (XRT_user_study_manager.py) and the viewers (wrench_playback.py,
# session_viewer.py) can never drift apart.
#
# Layout: one CSV row per sample. Columns are `timestamp` followed by, for each
# joint segment and field, one column per joint:  {field}_{segment}_{joint};
# then optional finger qpos and Inspire-hand register columns. A sidecar
# `<name>.meta.json` carries model path, palm offsets, known test mass, side,
# tare baseline, kneeling state, the tactile-binary layout, and recording start
# times for video sync.
#
# Reading is schema-driven and *graceful*: only columns actually present are
# loaded, so this reads both old wrench-only captures and new full-session CSVs.
# --------------------------------------------------------------------------- #

# Wrench-relevant subset (kept for backward references / fabricated test files).
CAPTURE_SEGMENTS = (("torso", 3), ("left", 7), ("right", 7))
CAPTURE_FIELDS = ("q", "dq", "ddq", "tau")

# Session joint schema: legs-down-to-the-knees (8: hip+knee per leg, no ankles) +
# torso + both arms, with measured + commanded data.
SESSION_JOINT_SEGMENTS = (("legs", 8), ("torso", 3), ("left", 7), ("right", 7))
SESSION_JOINT_FIELDS = ("q", "dq", "ddq", "tau", "temp", "qgoal")

# Finger qpos (full coupled 12-joint set per hand) — for visual playback.
FINGER_SEGMENTS = (("finger_left", FINGER_DOF), ("finger_right", FINGER_DOF))

# Inspire-hand register snapshot, per hand, 6 fingers each.
HAND_SIDES = ("left", "right")
HAND_REGISTER_FIELDS = ("pos_act", "angle_act", "force_act",
                        "current", "error", "status", "temp")
HAND_REGISTER_DOF = 6
# Map our snake-case CSV field -> the snapshot['states'] register key.
HAND_REGISTER_KEY = {
    "pos_act": "POS_ACT", "angle_act": "ANGLE_ACT", "force_act": "FORCE_ACT",
    "current": "CURRENT", "error": "ERROR", "status": "STATUS", "temp": "TEMP",
}
# snapshot['states'] keys are "Left"/"Right".
HAND_SIDE_STATE_KEY = {"left": "Left", "right": "Right"}


def finger_fieldnames() -> list:
    cols = []
    for seg, n in FINGER_SEGMENTS:
        for j in range(n):
            cols.append(f"q_{seg}_{j}")
    return cols


def hand_register_fieldnames() -> list:
    cols = []
    for side in HAND_SIDES:
        for fld in HAND_REGISTER_FIELDS:
            for j in range(HAND_REGISTER_DOF):
                cols.append(f"hand_{side}_{fld}_{j}")
    return cols


def capture_fieldnames() -> list:
    """Wrench-only column set (legacy; SessionRecorder uses session_fieldnames)."""
    cols = ["timestamp"]
    for seg, n in CAPTURE_SEGMENTS:
        for fld in CAPTURE_FIELDS:
            for j in range(n):
                cols.append(f"{fld}_{seg}_{j}")
    cols += finger_fieldnames()
    return cols


def session_fieldnames() -> list:
    """Comprehensive raw-session column set written by the SessionRecorder."""
    cols = ["timestamp"]
    for seg, n in SESSION_JOINT_SEGMENTS:
        for fld in SESSION_JOINT_FIELDS:
            for j in range(n):
                cols.append(f"{fld}_{seg}_{j}")
    cols += finger_fieldnames()
    cols += hand_register_fieldnames()
    return cols


def read_capture_csv(path: str) -> dict:
    """Read a capture / session CSV into numpy arrays (schema-driven, graceful).

    Always returns ``"t"`` (seconds from first sample). For each joint segment
    present (``legs``/``torso``/``left``/``right``) returns ``out[seg]`` as a
    dict of whichever fields were recorded (``q``/``dq``/``ddq``/``tau``/
    ``temp``/``qgoal``), each ``(N, n_joints)``. Finger qpos, when present, come
    back as ``out["finger_left"]``/``out["finger_right"]`` ``(N,12)``. Inspire
    register snapshots, when present, as ``out["hand_left"]``/``out["hand_right"]``
    dicts of ``(N,6)`` arrays. Absent columns are simply omitted, so this reads
    both old wrench-only captures and new full-session CSVs.
    """
    import csv as _csv
    rows = []
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        header = set(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"no data rows in {path}")

    def col(name):
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    t_epoch = col("timestamp")
    t = t_epoch - t_epoch[0]
    out = {"t": t, "t_epoch": t_epoch}

    for seg, n in SESSION_JOINT_SEGMENTS:
        seg_out = {}
        for fld in SESSION_JOINT_FIELDS:
            if f"{fld}_{seg}_0" in header:
                seg_out[fld] = np.column_stack([col(f"{fld}_{seg}_{j}") for j in range(n)])
        if seg_out:
            out[seg] = seg_out

    for seg, n in FINGER_SEGMENTS:
        if f"q_{seg}_0" in header:
            out[seg] = np.column_stack([col(f"q_{seg}_{j}") for j in range(n)])

    for side in HAND_SIDES:
        hand_out = {}
        for fld in HAND_REGISTER_FIELDS:
            if f"hand_{side}_{fld}_0" in header:
                hand_out[fld] = np.column_stack(
                    [col(f"hand_{side}_{fld}_{j}") for j in range(HAND_REGISTER_DOF)])
        if hand_out:
            out[f"hand_{side}"] = hand_out

    return out


# --------------------------------------------------------------------------- #
# Tactile (touch) binary I/O. The full per-fingertip / palm touch arrays are
# large (~1300 ints/frame), so they live in a streamed binary sidecar rather
# than the CSV. Each record is a float64 timestamp + an int32 vector of the
# flattened touch arrays in a fixed ``layout`` order (stored in meta.json).
# --------------------------------------------------------------------------- #
TACTILE_VALUE_DTYPE = "<i4"   # int32 — touch registers are unsigned 16-bit ADC


def tactile_record_dtype(width: int) -> np.dtype:
    return np.dtype([("t", "<f8"), ("v", TACTILE_VALUE_DTYPE, (int(width),))])


# In the latency-optimised "haptics_only" read used during teleop, each finger
# region is the coalesced tip + nail block (105 regs): tip = 3×3 (9), and the
# 12×8 "nail" patch (96) which is EMPIRICALLY the main contact PAD (see
# inspire_fingertip_force_calibration.py). The palm is a single region. So a
# 105-long finger region decomposes into fingertip + pad; anything else is left
# undecomposed. This is descriptive metadata only (the raw arrays are recorded
# whole) so a recording self-documents as "fingertip + pad + palm".
def describe_tactile_layout(layout: list) -> dict:
    """Annotate each tactile region with its sub-region breakdown (for meta)."""
    out = {}
    for seg in layout:
        region, size = seg["region"], int(seg["size"])
        side = seg["side"]
        key = f"{side}/{region}"
        if region == "fingerpalm":
            out[key] = {"palm": [0, size]}
        elif size >= 105:           # haptics_only tip+nail("pad") block
            out[key] = {"fingertip": [0, 9], "pad": [9, 105]}
        else:
            out[key] = {"raw": [0, size]}
    return out


def build_tactile_layout(touch: dict) -> tuple:
    """From a snapshot['touch'] dict (``{side: {region: array}}``) build a
    deterministic flatten layout and total width.

    Returns ``(layout, width)`` where ``layout`` is a list of
    ``{"side", "region", "size"}`` in flatten order (sides Left→Right, regions
    sorted by name)."""
    layout = []
    width = 0
    for side in ("Left", "Right"):
        regions = touch.get(side, {})
        for region in sorted(regions.keys()):
            size = int(np.asarray(regions[region]).size)
            layout.append({"side": side, "region": region, "size": size})
            width += size
    return layout, width


def flatten_touch(touch: dict, layout: list) -> np.ndarray:
    """Flatten a snapshot['touch'] dict into one int32 vector per ``layout``."""
    parts = []
    for seg in layout:
        arr = np.asarray(touch.get(seg["side"], {}).get(seg["region"], []),
                         dtype=np.int64).reshape(-1)
        if arr.size < seg["size"]:
            arr = np.concatenate([arr, np.zeros(seg["size"] - arr.size, dtype=np.int64)])
        parts.append(arr[:seg["size"]])
    if not parts:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate(parts).astype(np.int32)


def reconstruct_touch_snapshot(frame_vec: np.ndarray, layout: list) -> dict:
    """Inverse of :func:`flatten_touch`: rebuild a ``{'touch': {...}}`` snapshot
    from one flat frame vector, suitable for
    :func:`inspire_hand_vis_utils.build_mujoco_sensor_overlay_values_from_snapshot`."""
    frame_vec = np.asarray(frame_vec).reshape(-1)
    touch: dict = {}
    off = 0
    for seg in layout:
        size = int(seg["size"])
        touch.setdefault(seg["side"], {})[seg["region"]] = \
            np.asarray(frame_vec[off:off + size], dtype=int)
        off += size
    return {"touch": touch}


def read_tactile_bin(path: str, width: int) -> tuple:
    """Read a tactile binary into ``(t (N,), values (N, width))``."""
    arr = np.fromfile(path, dtype=tactile_record_dtype(width))
    if arr.size == 0:
        return np.zeros(0), np.zeros((0, int(width)), dtype=np.int32)
    return arr["t"].astype(np.float64), arr["v"].astype(np.int32)


def hand_qpos_addrs(model) -> dict:
    """Return ``{"finger_left": [12 qpos addrs], "finger_right": [...]}``.

    Used by both the recorder (to read finger qpos from the sim) and the
    playback tool (to write them) so the joint order can never disagree.
    """
    return {
        "finger_left":  [int(model.joint(n).qposadr[0]) for n in LEFT_HAND_JOINTS],
        "finger_right": [int(model.joint(n).qposadr[0]) for n in RIGHT_HAND_JOINTS],
    }


def meta_path_for(csv_path: str) -> str:
    p = Path(csv_path)
    return str(p.with_suffix("")) + ".meta.json"


def save_capture_meta(csv_path: str, meta: dict) -> str:
    import json as _json
    mp = meta_path_for(csv_path)
    with open(mp, "w") as f:
        _json.dump(meta, f, indent=2)
    return mp


def load_capture_meta(csv_path: str) -> dict:
    import json as _json
    mp = meta_path_for(csv_path)
    if not Path(mp).exists():
        return {}
    with open(mp) as f:
        return _json.load(f)
