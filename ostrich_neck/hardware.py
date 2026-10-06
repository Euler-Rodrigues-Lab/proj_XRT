"""High-level hardware interface for the Ostrich neck module."""
from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

from .conversion import (
    DEFAULT_PITCH_LIMITS,
    DEFAULT_YAW_LIMITS,
    convert_neck_input_to_encoder_steps,
)
from .controller import DynamixelNeckController, DynamixelConnectionError


class OstrichNeckHardwareInterface:
    """Convenience wrapper for controlling the 2-DOF neck hardware."""

    def __init__(
        self,
        controller: Optional[DynamixelNeckController] = None,
        *,
        yaw_home: float = 0.0,
        pitch_home: float = 0.0,
        settle_time: float = 0.2,
    ) -> None:
        self._controller = controller or DynamixelNeckController()
        self._yaw_home = yaw_home
        self._pitch_home = pitch_home
        self._settle_time = settle_time
        self._connected = False

    @property
    def controller(self) -> DynamixelNeckController:
        return self._controller

    def connect(self, *, move_home: bool = True) -> None:
        """Connect to the motors and optionally move to the home configuration."""
        if self._connected:
            return
        self._controller.connect()
        self._connected = True

        if move_home:
            self.move_from_angles((self._yaw_home, self._pitch_home))
            time.sleep(self._settle_time)

    def move_from_angles(
        self,
        neck_input: Iterable[float],
        *,
        yaw_limits=DEFAULT_YAW_LIMITS,
        pitch_limits=DEFAULT_PITCH_LIMITS,
    ) -> Tuple[int, int]:
        """Convert ``neck_input`` in radians and send to the motors."""
        if not self._connected:
            raise DynamixelConnectionError("connect() must be called before move_from_angles()")
        yaw_steps, pitch_steps = convert_neck_input_to_encoder_steps(
            neck_input,
            yaw_limits=yaw_limits,
            pitch_limits=pitch_limits,
        )
        self._controller.move(yaw_steps, pitch_steps)
        return yaw_steps, pitch_steps

    def move_to_encoder_steps(self, yaw_steps: int, pitch_steps: int) -> None:
        """Directly send encoder targets to the motors."""
        if not self._connected:
            raise DynamixelConnectionError("connect() must be called before move_to_encoder_steps()")
        self._controller.move(yaw_steps, pitch_steps)

    def shutdown(self) -> None:
        """Disable torque and close the serial port."""
        if not self._connected:
            return
        self._controller.cleanup()
        self._connected = False

    def reboot(self) -> None:
        """Reboot the neck servos + re-establish control (clears a latched servo error and
        recovers a wedged serial link) WITHOUT restarting the process. After this,
        move_from_angles() works again. Raises DynamixelConnectionError on failure.

        Returns the neck to its front-facing HOME pose, so pressing "Reboot Neck" lands the
        same forward-looking pose as a fresh manager boot (connect(move_home=True)) instead
        of leaving the servos wherever they happened to stop."""
        self._controller.reboot()
        self._connected = True
        self.move_from_angles((self._yaw_home, self._pitch_home))
        time.sleep(self._settle_time)

    def __enter__(self) -> "OstrichNeckHardwareInterface":
        self.connect(move_home=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
