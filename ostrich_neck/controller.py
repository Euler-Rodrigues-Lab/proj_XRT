"""Low-level Dynamixel controller utilities for the Ostrich Neck hardware."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from dynamixel_sdk import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)


@dataclass(frozen=True)
class MotorIds:
    yaw: int = 11
    pitch: int = 12


class DynamixelConnectionError(RuntimeError):
    """Raised when the Dynamixel controller cannot connect to the motors."""


class DynamixelNeckController:
    """Thin wrapper around ``dynamixel_sdk`` for two DOF neck control."""

    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    LEN_GOAL_POSITION = 4
    LEN_PRESENT_POSITION = 4
    PROTOCOL_VERSION = 2.0

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 4_000_000,
        motor_ids: MotorIds | Iterable[int] = MotorIds(),
    ) -> None:
        if isinstance(motor_ids, MotorIds):
            self._motor_ids = motor_ids
        else:
            try:
                yaw_id, pitch_id = motor_ids  # type: ignore[misc]
            except (TypeError, ValueError) as exc:
                raise ValueError("motor_ids must be MotorIds or iterable of length 2") from exc
            self._motor_ids = MotorIds(yaw=yaw_id, pitch=pitch_id)

        self._port_name = port
        self._baudrate = baudrate
        self._port_handler: Optional[PortHandler] = None
        self._packet_handler: Optional[PacketHandler] = None
        self._group_sync_write: Optional[GroupSyncWrite] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Open the serial port, set the baudrate, and enable torque."""
        if self._connected:
            return

        port_handler = PortHandler(self._port_name)
        if not port_handler.openPort():
            raise DynamixelConnectionError(f"Failed to open port {self._port_name}")
        if not port_handler.setBaudRate(self._baudrate):
            port_handler.closePort()
            raise DynamixelConnectionError(f"Failed to set baudrate {self._baudrate}")

        packet_handler = PacketHandler(self.PROTOCOL_VERSION)

        for motor_id in (self._motor_ids.yaw, self._motor_ids.pitch):
            dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
                port_handler,
                motor_id,
                self.ADDR_TORQUE_ENABLE,
                1,
            )
            if dxl_comm_result != COMM_SUCCESS:
                port_handler.closePort()
                raise DynamixelConnectionError(
                    f"Error enabling torque for ID {motor_id}: "
                    f"{packet_handler.getTxRxResult(dxl_comm_result)}"
                )
            if dxl_error != 0:  # pragma: no cover - hardware specific
                port_handler.closePort()
                raise DynamixelConnectionError(
                    f"Torque enable error for ID {motor_id}: "
                    f"{packet_handler.getRxPacketError(dxl_error)}"
                )

        group_sync_write = GroupSyncWrite(
            port_handler,
            packet_handler,
            self.ADDR_GOAL_POSITION,
            self.LEN_GOAL_POSITION,
        )

        self._port_handler = port_handler
        self._packet_handler = packet_handler
        self._group_sync_write = group_sync_write
        self._connected = True

    def cleanup(self) -> None:
        """Disable torque and close the port safely."""
        if not self._connected or self._packet_handler is None or self._port_handler is None:
            return

        for motor_id in (self._motor_ids.yaw, self._motor_ids.pitch):
            self._packet_handler.write1ByteTxRx(
                self._port_handler,
                motor_id,
                self.ADDR_TORQUE_ENABLE,
                0,
            )

        self._port_handler.closePort()
        self._connected = False

    def reboot(self) -> None:
        """Recover a neck that stopped responding WITHOUT restarting the process.

        Sends the Dynamixel REBOOT instruction to both servos (which clears a latched
        hardware error — overload/overheat — that a write-only GroupSyncWrite can't surface,
        and resets the servo), waits for them to come back, then re-enables torque and
        rebuilds the sync writer. If the serial bus itself is wedged (the REBOOT can't even
        be sent), falls back to a full port reconnect (close + reopen + torque-enable).
        Raises DynamixelConnectionError if it cannot recover.
        """
        # Never connected (or the port object is gone) → just (re)connect from scratch.
        if self._port_handler is None or self._packet_handler is None:
            self._connected = False
            self.connect()
            return

        ph, pk = self._port_handler, self._packet_handler
        bus_ok = True
        for motor_id in (self._motor_ids.yaw, self._motor_ids.pitch):
            try:
                dxl_comm_result, _dxl_error = pk.reboot(ph, motor_id)
                if dxl_comm_result != COMM_SUCCESS:
                    bus_ok = False
            except Exception:  # noqa: BLE001 - any serial fault → fall back to reconnect
                bus_ok = False
        time.sleep(0.5)  # servos take ~0.2-0.5 s to reboot and re-appear on the bus

        if not bus_ok:
            # Bus/port wedged — REBOOT never reached the servos. Reopen the port fully.
            try:
                ph.closePort()
            except Exception:  # noqa: BLE001
                pass
            self._connected = False
            self._port_handler = None
            self._packet_handler = None
            self._group_sync_write = None
            self.connect()
            return

        # Reboot drops torque — re-enable it and rebuild the goal-position sync writer.
        for motor_id in (self._motor_ids.yaw, self._motor_ids.pitch):
            res, err = pk.write1ByteTxRx(ph, motor_id, self.ADDR_TORQUE_ENABLE, 1)
            if res != COMM_SUCCESS or err != 0:
                raise DynamixelConnectionError(
                    f"Neck reboot: torque re-enable failed for ID {motor_id} "
                    f"(result={res}, error={err})"
                )
        self._group_sync_write = GroupSyncWrite(
            ph, pk, self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION,
        )
        self._connected = True

    def __enter__(self) -> "DynamixelNeckController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # Motion helpers
    # ------------------------------------------------------------------
    def move(self, yaw_steps: int, pitch_steps: int) -> None:
        """Send goal positions to the yaw and pitch motors."""
        if not self._connected:
            raise DynamixelConnectionError("DynamixelNeckController is not connected")
        if self._group_sync_write is None:
            raise DynamixelConnectionError("GroupSyncWrite is not initialized")
        if self._packet_handler is None:
            raise DynamixelConnectionError("PacketHandler is not initialized")

        params_yaw = [
            DXL_LOBYTE(DXL_LOWORD(yaw_steps)),
            DXL_HIBYTE(DXL_LOWORD(yaw_steps)),
            DXL_LOBYTE(DXL_HIWORD(yaw_steps)),
            DXL_HIBYTE(DXL_HIWORD(yaw_steps)),
        ]
        params_pitch = [
            DXL_LOBYTE(DXL_LOWORD(pitch_steps)),
            DXL_HIBYTE(DXL_LOWORD(pitch_steps)),
            DXL_LOBYTE(DXL_HIWORD(pitch_steps)),
            DXL_HIBYTE(DXL_HIWORD(pitch_steps)),
        ]

        self._group_sync_write.addParam(self._motor_ids.yaw, params_yaw)
        self._group_sync_write.addParam(self._motor_ids.pitch, params_pitch)
        dxl_comm_result = self._group_sync_write.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            error = self._packet_handler.getTxRxResult(dxl_comm_result)
            raise DynamixelConnectionError(f"Error sending sync write: {error}")
        self._group_sync_write.clearParam()

    def move_from_angles(self, neck_input: Iterable[float], converter) -> Tuple[int, int]:
        """Convert ``neck_input`` using ``converter`` and send the command."""
        yaw_steps, pitch_steps = converter(neck_input)
        self.move(yaw_steps, pitch_steps)
        return yaw_steps, pitch_steps
