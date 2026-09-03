from __future__ import annotations

from pathlib import Path
import builtins
import importlib
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def _xr_action():
    wrist = Rotation.from_euler("ZYX", [0.3, 0.1, -0.2]).as_matrix()
    return {
        "R_world_upper_body": np.eye(3),
        "p_world_upper_body": np.array([0.0, 0.0, 1.2]),
        "left_sew": np.concatenate(
            [[0.0, 0.2, 1.3], [0.1, 0.3, 1.1], [0.3, 0.3, 1.1], np.eye(3).ravel()]
        ),
        "right_sew": np.concatenate(
            [[0.0, -0.2, 1.3], [0.1, -0.3, 1.1], [0.3, -0.3, 1.1], wrist.ravel()]
        ),
        "head_rotation": Rotation.from_euler("ZYX", [0.2, -0.1, 0.0]).as_matrix(),
        "R_lower_upper": np.eye(3),
        "body_center": np.array([0.0, 0.0, 0.9]),
        "ankle_to_body": np.array([0.02, 0.0, 0.9]),
    }


def test_legacy_model_paths_resolve_into_submodule():
    model_path = (
        ROOT
        / "projects/inspire_hand_teleop/g1_with_inspire_hand"
        / "g1_29dof_rev_1_0_with_inspire_hand_FTP.xml"
    )
    assert model_path.is_symlink()
    assert model_path.resolve().is_relative_to(ROOT / "external/g1_teleop")

    # Several XRT tools call MuJoCo directly instead of g1_teleop.load_mjcf;
    # the compatibility mesh symlink must keep that legacy path loadable.
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.nq == 53
    assert model.nu == 41


def test_controller_shim_uses_g1_teleop_implementation():
    from g1_teleop.control import G1InspireHandMuJoCoController as BaseController
    from projects.inspire_hand_teleop.g1_with_inspire_hand_mujoco_controller import (
        G1InspireHandMuJoCoController,
    )

    assert G1InspireHandMuJoCoController is BaseController


def test_g1_hardware_module_is_lazy_without_unitree_sdk(monkeypatch):
    real_import = builtins.__import__

    def reject_unitree(name, *args, **kwargs):
        if name.startswith("unitree_sdk2py"):
            raise ImportError("simulated missing hardware SDK")
        return real_import(name, *args, **kwargs)

    for name in tuple(sys.modules):
        if name.startswith("projects.g1_upper_body_teleop"):
            sys.modules.pop(name)
    monkeypatch.setattr(builtins, "__import__", reject_unitree)
    module = importlib.import_module("projects.g1_upper_body_teleop.unitree_api_utils")
    assert module.Controller is not None


def test_solver_facade_runs_submodule_retargeting_and_safety_fk():
    from projects.g1_full_body_kinematic.g1_full_body_sew_solver import G1FullBodySEWSolver

    solver = G1FullBodySEWSolver(
        hand_type="inspire",
        collision_avoidance_enabled=False,
        control_rate_hz=60.0,
    )
    assert solver.geometry_source in {"yaml", "compiled"}
    prepared = solver.preprocess_human_action(_xr_action())
    assert np.allclose(prepared["q_head"], [0.2, -0.1])

    goals = solver.solve(prepared)
    assert goals["q_goal_right"].shape == (7,)
    assert goals["q_goal_left"].shape == (7,)
    assert goals["q_goal_torso"].shape == (3,)

    palm = solver.arm_wrist_position("left", np.zeros(7))
    assert palm.shape == (3,)
    assert np.isfinite(palm).all()
