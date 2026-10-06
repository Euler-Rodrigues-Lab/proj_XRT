"""Utilities for mapping neck inputs to Dynamixel encoder steps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class EncoderLimits:
    """Configuration parameters for a single Dynamixel axis."""

    encoder_min: int
    encoder_max: int
    angle_min: float
    angle_max: float

    @property
    def encoder_range(self) -> int:
        return self.encoder_max - self.encoder_min

    @property
    def angle_range(self) -> float:
        return self.angle_max - self.angle_min


DEFAULT_YAW_LIMITS = EncoderLimits(
    encoder_min=0,
    encoder_max=4095,
    angle_min=-1.5707963267948966,  # -pi/2
    angle_max=1.5707963267948966,   # pi/2
)

DEFAULT_PITCH_LIMITS = EncoderLimits(
    encoder_min=1024,
    encoder_max=3072,
    angle_min=-1.0471975511965976,  # -pi/3
    angle_max=1.0471975511965976,   # pi/3
)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def clamp_to_encoder_steps(angle: float, limits: EncoderLimits) -> int:
    """Clamp ``angle`` to ``limits`` and convert to encoder steps.

    Args:
        angle: Input angle in radians.
        limits: Encoder configuration describing valid angle / step ranges.

    Returns:
        Integer encoder steps within ``[limits.encoder_min, limits.encoder_max]``.
    """
    if limits.angle_range == 0:
        raise ValueError("EncoderLimits angle_range cannot be zero")

    clamped_angle = _clamp(angle, limits.angle_min, limits.angle_max)
    normalized = (clamped_angle - limits.angle_min) / limits.angle_range
    encoder_value = limits.encoder_min + normalized * limits.encoder_range
    return int(round(_clamp(encoder_value, limits.encoder_min, limits.encoder_max)))


def convert_neck_input_to_encoder_steps(
    neck_input: Iterable[float],
    yaw_limits: EncoderLimits = DEFAULT_YAW_LIMITS,
    pitch_limits: EncoderLimits = DEFAULT_PITCH_LIMITS,
) -> Tuple[int, int]:
    """Convert ``neck_input`` (yaw, pitch) to Dynamixel encoder targets.

    The mapping is linear between the radian limits provided in ``yaw_limits`` /
    ``pitch_limits`` and the encoder counts for each axis. Inputs outside the
    allowed angle range are saturated so the hardware remains within safe bounds.

    Args:
        neck_input: Iterable with two elements ``(yaw, pitch)`` in radians.
        yaw_limits: Encoder / angle limits for the yaw axis.
        pitch_limits: Encoder / angle limits for the pitch axis.

    Returns:
        Tuple ``(yaw_steps, pitch_steps)`` suitable for ``DynamixelNeckController.move``.
    """
    try:
        yaw_angle, pitch_angle = neck_input
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive programming
        raise ValueError("neck_input must be an iterable with two floats: (yaw, pitch)") from exc

    yaw_steps = clamp_to_encoder_steps(float(yaw_angle), yaw_limits)
    pitch_steps = clamp_to_encoder_steps(float(pitch_angle), pitch_limits)
    return yaw_steps, pitch_steps
