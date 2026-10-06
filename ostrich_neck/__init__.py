"""Ostrich Neck hardware interface package.

Driver for the robot's 2-DOF Dynamixel neck. Vendored from
https://github.com/kczttm/ostrich-neck (commit 0aca7ff, by Advaith Nidumukkala and
Chuizheng Kong) plus the servo ``reboot()`` recovery that XRT's neck worker uses.
"""

from .conversion import clamp_to_encoder_steps, convert_neck_input_to_encoder_steps


def __getattr__(name):
    # The servo classes import dynamixel_sdk (hardware extra); load them only on use.
    if name == "DynamixelNeckController":
        from .controller import DynamixelNeckController
        return DynamixelNeckController
    if name == "OstrichNeckHardwareInterface":
        from .hardware import OstrichNeckHardwareInterface
        return OstrichNeckHardwareInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "clamp_to_encoder_steps",
    "convert_neck_input_to_encoder_steps",
    "DynamixelNeckController",
    "OstrichNeckHardwareInterface",
]
