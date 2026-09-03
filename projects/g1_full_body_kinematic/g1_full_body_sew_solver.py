"""Legacy XRT solver facade backed entirely by the ``g1_teleop`` solver API.

The active study applications historically used a two-stage
``preprocess_human_action`` / ``solve`` API and mutable dictionaries.  The extracted
G1 package uses the typed ``RetargetFrame`` / ``RetargetOutput`` contract instead.
This module only translates those contracts; all retargeting, hand IK, torso IK, and
collision filtering remain in the pinned G1 dependency.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from geo_kin_core.session import resolve_session
from geo_kin_core.types import RetargetOutput
from g1_teleop import XML_INSPIRE_MOUNTED
from g1_teleop.input import action_to_retarget_frame
from g1_teleop.mjcf import load_mjcf

_COMPAT_XML = (
    Path(__file__).resolve().parents[1]
    / "inspire_hand_teleop"
    / "g1_with_inspire_hand"
    / "g1_29dof_rev_1_0_with_inspire_hand_FTP.xml"
)
_DEFAULT_GEOMETRY_CONFIG = Path(__file__).parent / "config" / "g1_geometry.yaml"


class _PreparedAction(dict):
    """Dict-compatible carrier for a typed frame and the legacy ``q_head`` key."""

    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        if frame.head_rotation is not None:
            euler = Rotation.from_matrix(frame.head_rotation).as_euler("ZYX")
            self["q_head"] = np.asarray(euler[:2], dtype=float)


def _output_dict(output: RetargetOutput | Any) -> dict[str, Any]:
    names = [item.name for item in fields(RetargetOutput)]
    return {name: getattr(output, name, None) for name in names}


class G1FullBodySEWSolver:
    """Compatibility facade for legacy XRT call sites.

    New code should use ``geo_kin_core.session.resolve_session`` directly.
    """

    def __init__(
        self,
        urdf_path=None,
        control_rate_hz=1000.0,
        elbow_filter_cutoff_hz=10.0,
        debug=False,
        timing_enabled=False,
        collision_avoidance_enabled=True,
        hand_type="psyonic",
        base_alignment_mode="mocap",
        geometry_config_path=_DEFAULT_GEOMETRY_CONFIG,
    ):
        # ``debug`` and ``timing_enabled`` belonged to the monolith wrapper and
        # are intentionally not forwarded: they are not part of the stable
        # geo_kin_core session contract (notably, the reference backend rejects
        # them).  ``urdf_path`` is retained in the signature for launch-script
        # compatibility; g1_teleop owns and selects the signed robot assets.
        del urdf_path, timing_enabled, debug
        self._session_config = {
            "control_rate_hz": float(control_rate_hz),
            "elbow_filter_cutoff_hz": float(elbow_filter_cutoff_hz),
            "collision_avoidance": bool(collision_avoidance_enabled),
            "torso": True,
            "base_alignment_mode": base_alignment_mode,
            "preprocess": {
                "shoulder_width_scale": 0.8,
                "hip_width_scale": 1.3,
                "mocap_cartesian_scale": (0.85, 0.85, 0.65),
                "mocap_offset": (0.0, 0.0, 0.4),
            },
        }
        if geometry_config_path is not None:
            self._session_config["geometry_config_path"] = geometry_config_path
        self.hand_type = hand_type
        self.geometry_source = None
        self._session = self._resolve_session()
        self.last_filtered_sew = None
        self._fk_model = None
        self._fk_data = None

    def _resolve_session(self):
        config = dict(self._session_config)
        while True:
            try:
                session = resolve_session(robot="g1", hand=self.hand_type, **config)
                if self.geometry_source is None:
                    self.geometry_source = (
                        "yaml" if config.get("geometry_config_path") is not None else "backend-default"
                    )
                return session
            except ValueError as exc:
                message = str(exc)
                if "geometry_config_path is not supported by the geo_kin wheel" in message:
                    # Rust production wheels compile safety geometry in. Keep the
                    # YAML for the reference backend, but do not prevent the licensed
                    # backend from being selected when it is installed.
                    config.pop("geometry_config_path", None)
                    self.geometry_source = "compiled"
                    continue
                if "needs the robot MJCF" in message and "model_xml" not in config:
                    # The public Mink fallback is the only backend that needs the
                    # model path. It accepts the other compatibility options as
                    # ignored kwargs, so one retry is sufficient.
                    config["model_xml"] = _COMPAT_XML
                    continue
                raise

    def preprocess_human_action(
        self,
        action,
        shoulder_width_scale=0.8,
        hip_width_scale=1.3,
        mocap_cartesian_scale=(0.85, 0.85, 0.65),
        mocap_offset=(0.0, 0.0, 0.4),
        R_mjworld_human=None,
    ):
        requested = {
            "shoulder_width_scale": float(shoulder_width_scale),
            "hip_width_scale": float(hip_width_scale),
            "mocap_cartesian_scale": tuple(np.asarray(mocap_cartesian_scale, dtype=float)),
            "mocap_offset": tuple(np.asarray(mocap_offset, dtype=float)),
        }
        if R_mjworld_human is not None:
            requested["R_mjworld_human"] = np.asarray(R_mjworld_human, dtype=float)

        current = self._session_config["preprocess"]
        changed = any(
            key not in current or not np.allclose(current[key], value)
            for key, value in requested.items()
        )
        if changed:
            self._session_config["preprocess"] = requested
            self._session = self._resolve_session()

        return _PreparedAction(action_to_retarget_frame(action))

    def solve(self, action, engaged=True, q_current_right=None, q_current_left=None):
        frame = action.frame if isinstance(action, _PreparedAction) else action_to_retarget_frame(action)
        output = self._session.solve(
            frame,
            engaged=engaged,
            q_current_right=q_current_right,
            q_current_left=q_current_left,
        )
        diag = getattr(output, "diag", None)
        self.last_filtered_sew = getattr(diag, "filtered_sew", None)
        return _output_dict(output)

    def reset(self, q_init_right=None, q_init_left=None, **_unused):
        return self._session.reset(q_init_right=q_init_right, q_init_left=q_init_left)

    def sew_capsules(self):
        method = getattr(self._session, "sew_capsules", None)
        return method() if method is not None else []

    @property
    def R_mocap_world(self):
        return getattr(self._session, "R_mocap_world", np.eye(3))

    @property
    def p_mocap_world(self):
        return getattr(self._session, "p_mocap_world", np.zeros(3))

    def arm_wrist_position(self, arm_side, q_arm):
        """Pure MuJoCo FK used by XRT's optional behind-the-head safety gate.

        This replaces the monolith's Pinocchio call in the 500 Hz control thread.
        A private model/data pair keeps the live visualization state untouched.
        """
        import mujoco

        if arm_side not in ("left", "right"):
            raise ValueError("arm_side must be 'left' or 'right'")
        if self._fk_model is None:
            self._fk_model = load_mjcf(XML_INSPIRE_MOUNTED)
            self._fk_data = mujoco.MjData(self._fk_model)

        model, data = self._fk_model, self._fk_data
        data.qpos[:] = model.qpos0
        names = (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        )
        for value, name in zip(np.asarray(q_arm, dtype=float), names):
            data.qpos[model.joint(f"{arm_side}_{name}_joint").qposadr[0]] = value
        mujoco.mj_forward(model, data)

        pelvis = model.body("pelvis").id
        wrist = model.body(f"{arm_side}_wrist_yaw_link").id
        offset = np.array([0.1611, -0.012296 if arm_side == "left" else 0.012296, 0.00036819])
        palm_world = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ offset
        return data.xmat[pelvis].reshape(3, 3).T @ (palm_world - data.xpos[pelvis])
