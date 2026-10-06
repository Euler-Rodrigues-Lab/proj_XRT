from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from ostrich_neck.conversion import convert_neck_input_to_encoder_steps

NECK = Path(__file__).resolve().parents[1] / "ostrich_neck"


def test_neck_centre_maps_to_mid_encoder_range():
    yaw, pitch = convert_neck_input_to_encoder_steps((0.0, 0.0))
    assert abs(yaw - 2048) <= 1
    assert 0 <= pitch <= 4095


def test_neck_input_is_clamped_to_encoder_limits():
    yaw, _pitch = convert_neck_input_to_encoder_steps((np.pi, 0.0))
    assert 0 <= yaw <= 4095


def test_vendored_driver_keeps_reboot_used_by_the_manager():
    # Parsed, not imported: the driver needs dynamixel_sdk (hardware extra only).
    for name in ("controller.py", "hardware.py"):
        tree = ast.parse((NECK / name).read_text())
        methods = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "reboot" in methods, name
