"""
High-level controller interface for Inspire robotic hand.

This module provides a clean, intuitive API for controlling the Inspire hand
without needing to deal with DDS messages and binary mode flags directly.

Author: Roman Mineyev
"""

from __future__ import annotations

import numpy as np
import os
import sys
import threading
import time
import logging
import signal
from typing import Any, Dict
import ctypes

from dataclasses import dataclass
from projects.inspire_hand_teleop.inspire_hand_vis_utils import (
    build_mujoco_sensor_overlay_values_from_snapshot,
    create_mujoco_sensor_overlay_runtime,
    create_dual_hand_plot_context,
    prepare_live_scalar_arrow_values,
    refresh_dual_hand_plot_context,
    render_mujoco_sensor_overlay,
)


_SIZEOF_C_SHORT: int = ctypes.sizeof(ctypes.c_short) # in Bytes
_MAX_HAND_SPEED: list[int] = [1000] * 6  # Max speed per finger (range 0-1000)

# Lost-link quieting (see InspireHandController): go quiet after this many consecutive Modbus
# failures, and slow the retries to this interval while quiet, so a powered-off robot doesn't
# spam the console.
_IO_QUIET_AFTER = 3
_IO_QUIET_RETRY_S = 1.0


def _modbus_client_class():
    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError as exc:
        raise ImportError(
            "Inspire hand hardware requires pymodbus; install proj-XRT[hardware]."
        ) from exc
    return ModbusTcpClient

@dataclass
class InspireHandActStates:
    """
    Data class for Inspire Hand Actuation States
    
    Reference: https://en.inspire-robots.com/wp-content/uploads/2025/01/INSPIRE-ROBOTS-The-Dexterous-Hand-RH56DFTP-User-Manual-V1.0.0.pdf
    
    Modes:
        - Angle control mode: 0b0001
        - Position control mode: 0b0010
        - Force control mode: 0b0100
        - Velocity (speed) control mode: 0b1000
    
    Combination of modes can be achieved by bitOR operation
    
    """
    IDLE: int = 0b0000
    ANGLE: int = 0b0001
    POSITION: int = 0b0010
    FORCE: int = 0b0100
    VELOCITY: int = 0b1000
    # TODO: Add brake mode if there is one
    

    
@dataclass
class InspireHandRegAddr:
    """
    Data class for Inspire Hand Register Addresses
    
    Reference: https://en.inspire-robots.com/wp-content/uploads/2025/01/INSPIRE-ROBOTS-The-Dexterous-Hand-RH56DFTP-User-Manual-V1.0.0.pdf
    
    """
    # W/R Registers
    ## 1 Byte Config
    HAND_ID:int = 1000
    REDU_RATIO:int = 1002
    CLEAR_ERROR:int = 1004
    SAVE: int = 1005
    RESET_PARA: int = 1006
    GESTURE_FORCE_CLB: int = 1009
    ## 12 Byte Config
    DEFAULT_SPEED_SET: int = 1032
    DEFAULT_FORCE_SET: int = 1044
    POS_SET: int = 1474
    ANGLE_SET: int = 1486
    FORCE_SET: int = 1498
    SPEED_SET: int = 1522
    
    # R-Only Joint Sensors
    ## 12 Byte Config
    POS_ACT: int = 1534
    ANGLE_ACT: int = 1546
    FORCE_ACT: int = 1582
    CURRENT: int = 1594
    ## 6 Byte Config
    ERROR: int = 1606
    STATUS: int = 1612
    TEMP: int = 1618
    
    # IP setting is W/R
    ## All 1 Byte Config
    IP_PART1: int = 1700
    IP_PART2: int = 1701
    IP_PART3: int = 1702
    IP_PART4: int = 1703
    
    # R-Only Finger Sensors
    ## All 370 Byte Config
    ### 18 (3, 3) from tip, 192 (12, 8) from top, 160 (10, 8) from pulp
    FINGERONE_TOUCH: int = 3000 # Pinky
    FINGERTWO_TOUCH: int = 3370 # Ring
    FINGERTHE_TOUCH: int = 3740 # Middle
    FINGERFOR_TOUCH: int = 4110 # Index
    
    ### 18 (3, 3) from tip, 192 (12, 8) from top, 18 (3, 3) from middle, 192 (12, 8) from pulp
    FINGERFIV_TOUCH: int = 4480 # Thumb
    ### 224 (14, 8) from palm
    FINGERPALM_TOUCH: int = 4900 # Palm


class InspireHandController:
    """
    High-level controller for Inspire robotic hand.
    
    This class provides a simplified interface for controlling the hand through
    DDS messaging, abstracting away the low-level details of mode flags and
    message construction.
    
    Example:
        >>> hand = InspireHandController('l')  # Left hand
        >>> hand.set_angle([100, 200, 300, 400, 500, 600])
        >>> hand.set_velocity([50, 50, 50, 50, 50, 50])
    """
    
    def __init__(self,
                 hand_side='b',
                 network=None,
                 initialize_dds=True,
                 RegAddr=InspireHandRegAddr,
                 ActStates=InspireHandActStates,
                 haptics_only=False,
                 ):
        """
        Initialize controller for left or right hand.
        
        Args:
            hand_side (str): 'l' for left hand, 'r' for right hand. Defaults to 'l'.
            network (str, optional): Network interface name for DDS. Defaults to None.
            initialize_dds (bool): Whether to initialize DDS factory. Set to False if
                                  already initialized elsewhere. Defaults to True.
        """
        
        # data classes for register addresses and actuation states
        self.RegAddr: InspireHandRegAddr = RegAddr
        self.ActStates: InspireHandActStates = ActStates
        # When True, tip+nail are fetched in one coalesced read per finger (no pad/middle),
        # cutting touch Modbus round trips from 17 → 6 per hand scan.
        self.haptics_only: bool = bool(haptics_only)
        self.states: dict[str, dict[str, list]] = {
            'Left': {
                'POS_ACT': [0]*6,
                'ANGLE_ACT': [0]*6,
                'FORCE_ACT': [0]*6,
                'CURRENT': [0]*6,
                'ERROR': [0]*6,
                'STATUS': [0]*6,
                'TEMP': [0]*6
            },
            'Right': {
                'POS_ACT': [0]*6,
                'ANGLE_ACT': [0]*6,
                'FORCE_ACT': [0]*6,
                'CURRENT': [0]*6,
                'ERROR': [0]*6,
                'STATUS': [0]*6,
                'TEMP': [0]*6, 
            }   
        }
        _touch_sizes = {
            key: sum(count for _, count in segs)
            for key, segs in self._get_touch_segments(RegAddr, self.haptics_only).items()
        }
        self.touch: dict[str, dict[str, np.ndarray]] = {
            side: {key: np.zeros(size, dtype=int) for key, size in _touch_sizes.items()}
            for side in ('Left', 'Right')
        }

        # Protect shared state/touch dictionaries from concurrent read/write races.
        self._data_lock = threading.Lock()
        
        if hand_side not in ['l', 'r', 'b']:
            raise ValueError("hand_side must be 'l' (left), 'r' (right), or 'b' (both)")

        self.hand_side = hand_side
        if (self.hand_side == 'l'):
            print("Using Left Hand")
        elif (self.hand_side == 'r'):
            print("Using Right Hand")
        else:
            print("Using Both Hands")
        
        # Initialize DDS if requested
        if initialize_dds:
            pass # We no longer use DDS initialization here for modbus direct client

        print("Creating Modbus TCP connections directly...")
        ModbusTcpClient = _modbus_client_class()
        
        # Connect to Modbus directly to bypass DDS bottlenecks
        self.device_id = 1
        
        if self.hand_side == 'b':
            self.client_left = ModbusTcpClient('192.168.123.210', port=6000)
            self.client_left.connect()
            self.client_right = ModbusTcpClient('192.168.123.211', port=6000)
            self.client_right.connect()
            self.client = None

            # Serialize Modbus I/O per client across actuator and sensor threads.
            self._io_lock_left = threading.Lock()
            self._io_lock_right = threading.Lock()
        else:
            ip = '192.168.123.210' if self.hand_side == 'l' else '192.168.123.211'
            self.client = ModbusTcpClient(ip, port=6000)
            self.client.connect()

            # Single client in left-only or right-only mode.
            self._io_lock = threading.Lock()

        print("Modbus clients initialized")

        # MuJoCo joint ranges (min, max) in radians
        # Order in goals array: [little, ring, middle, index, thumb_1, thumb_2]
        self.joint_ranges = {
            'pinky': [0.0, 1.4381],
            'ring': [0.0, 1.4381],
            'middle': [0.0, 1.4381],
            'index': [0.0, 1.4381],
            'thumb_1': [0.0, 1.1641],
            'thumb_2': [0.0, 0.5864]
        }
        
        # Convert to list for easier access in the same order as goals
        self.range_list = [
            self.joint_ranges['pinky'][1], # 1 is max value (since min is 0 we can ignore)
            self.joint_ranges['ring'][1],
            self.joint_ranges['middle'][1],
            self.joint_ranges['index'][1],
            self.joint_ranges['thumb_2'][1], # thumb_2 is bend
            self.joint_ranges['thumb_1'][1]  # thumb_1 is rotate
        ]

        # Other stuff
        # self._update_current_positions()

        # Initialize default goals to avoid startup races before first external goal set.
        self.q_goal_left_hand = [0] * 6
        self.q_goal_right_hand = [0] * 6

        print("Joint goals and ranges initialized")

        # Start modbus worker threads to avoid blocking main loop
        self._is_engaged = False
        # Max finger speed (SPEED_SET) is written ONCE per engage, not every command — see
        # _actuator_worker. This latches whether it's been applied for the current engage.
        self._speed_applied = False
        self._running = True
        # When set, BOTH worker threads idle (no Modbus writes or reads): the hands are held
        # off + the bus goes quiet. Used when the robot is power-cycled so the hands aren't
        # commanded the instant they come back online. Cleared by reboot().
        self._suppressed = threading.Event()
        # Lost-link quieting: when the robot is powered off, every Modbus op fails. Instead of
        # spamming one error line per op (ours) plus pymodbus's own "Connection to … failed"
        # log, count consecutive failures and after _IO_QUIET_AFTER go QUIET — print ONE notice,
        # silence pymodbus's connection logging, and slow the retries — until a read/write
        # succeeds again (robot back). Auto-detects the drop in well under a second.
        self._io_fail_count = 0
        self._io_quiet = False
        self._io_state_lock = threading.Lock()
        self._pymodbus_log_level = logging.getLogger("pymodbus").level

        # Actuation thread: sends position commands at 50 Hz.
        self._actuator_thread = threading.Thread(target=self._actuator_worker, daemon=True)
        self._actuator_thread.start()

        # Sensor threads run independently so touch reads cannot block actuation.
        # Both hands are polled in parallel (each has its own Modbus client and lock).
        if self.hand_side == 'b':
            self._sensor_thread_left = threading.Thread(
                target=self._sensor_worker_single,
                args=(self.client_left, 'Left'),
                daemon=True,
            )
            self._sensor_thread_right = threading.Thread(
                target=self._sensor_worker_single,
                args=(self.client_right, 'Right'),
                daemon=True,
            )
            self._sensor_thread_left.start()
            self._sensor_thread_right.start()
        else:
            self._sensor_thread = threading.Thread(target=self._sensor_worker, daemon=True)
            self._sensor_thread.start()

    @staticmethod
    def _decode_int16_registers(registers: list[int]) -> list[int]:
        """Decode unsigned 16-bit register words to signed int16 values."""
        decoded: list[int] = []
        for reg in registers:
            reg16 = reg & 0xFFFF
            decoded.append(reg16 - 0x10000 if reg16 & 0x8000 else reg16)
        return decoded

    @staticmethod
    def _unpack_registers_to_bytes(registers: list[int]) -> list[int]:
        """Split each 16-bit register into [high_byte, low_byte] uint8 values."""
        byte_list: list[int] = []
        for reg in registers:
            reg16 = reg & 0xFFFF
            byte_list.append((reg16 >> 8) & 0xFF)
            byte_list.append(reg16 & 0xFF)
        return byte_list

    def _get_io_lock(self, client: ModbusTcpClient):
        """Return the lock guarding the given Modbus client connection."""
        if self.hand_side == 'b':
            if client is self.client_left:
                return self._io_lock_left
            if client is self.client_right:
                return self._io_lock_right
            raise ValueError("unknown Modbus client for lock lookup")
        return self._io_lock

    def _read_input_registers(self, client: ModbusTcpClient, address: int, count: int) -> list[int] | None:
        """Read input registers and return None on Modbus exception responses."""
        with self._get_io_lock(client):
            response = client.read_holding_registers(address, count=count, device_id=self.device_id)
        if response.isError():
            print(f"Modbus read error at address={address}, count={count}: {response}")
            return None
        return response.registers

    def _read_touch_field(self, client: ModbusTcpClient, segments: list[tuple[int, int]]) -> np.ndarray | None:
        """Read a touch field from multiple register segments and concatenate."""
        values: list[int] = []
        for addr, count in segments:
            registers = self._read_input_registers(client, addr, count)
            if registers is None:
                return None
            values.extend(registers)
        return np.array(values, dtype=int)

    @staticmethod
    def _get_touch_segments(
        RegAddr: InspireHandRegAddr,
        haptics_only: bool = False,
    ) -> dict[str, list[tuple[int, int]]]:
        """Return touch register segments for a full hand snapshot.

        haptics_only=True coalesces tip (3×3=9 regs) + nail (12×8=96 regs) into
        one 105-register read per finger, skipping pad and middle entirely.
        Cuts Modbus round trips 17 → 6 per hand scan (1 read/finger + 1 palm).

        haptics_only=False (default) reads all sections for full visualisation.
        """
        _TIP  = 18  // _SIZEOF_C_SHORT   #  9 regs — 3×3 tip
        _NAIL = 192 // _SIZEOF_C_SHORT   # 96 regs — 12×8 nail
        _PAD4 = 160 // _SIZEOF_C_SHORT   # 80 regs — 10×8 pad  (fingers 1-4)
        _MID5 = 18  // _SIZEOF_C_SHORT   #  9 regs — 3×3 middle section (thumb only)
        _PAD5 = 192 // _SIZEOF_C_SHORT   # 96 regs — 12×8 pad  (thumb)
        _PALM = 224 // _SIZEOF_C_SHORT   # 112 regs — full palm

        if haptics_only:
            # Tip (3×3=9 regs) and nail (12×8=96 regs) are contiguous in device
            # memory, so they can be fetched in one 105-register transaction each.
            # Total: 6 reads/hand (was 17).  Array layout: tip[0:9], nail[9:105].
            _TIP_NAIL = _TIP + _NAIL  # 105 registers
            return {
                'fingerone':   [(RegAddr.FINGERONE_TOUCH, _TIP_NAIL)],
                'fingertwo':   [(RegAddr.FINGERTWO_TOUCH, _TIP_NAIL)],
                'fingerthree': [(RegAddr.FINGERTHE_TOUCH, _TIP_NAIL)],
                'fingerfour':  [(RegAddr.FINGERFOR_TOUCH, _TIP_NAIL)],
                'fingerfive':  [(RegAddr.FINGERFIV_TOUCH, _TIP_NAIL)],
                'fingerpalm':  [(RegAddr.FINGERPALM_TOUCH, _PALM)],
            }

        return {
            'fingerone': [
                (RegAddr.FINGERONE_TOUCH,             _TIP),
                (RegAddr.FINGERONE_TOUCH + 18,        _NAIL),
                (RegAddr.FINGERONE_TOUCH + 18 + 192,  _PAD4),
            ],
            'fingertwo': [
                (RegAddr.FINGERTWO_TOUCH,             _TIP),
                (RegAddr.FINGERTWO_TOUCH + 18,        _NAIL),
                (RegAddr.FINGERTWO_TOUCH + 18 + 192,  _PAD4),
            ],
            'fingerthree': [
                (RegAddr.FINGERTHE_TOUCH,             _TIP),
                (RegAddr.FINGERTHE_TOUCH + 18,        _NAIL),
                (RegAddr.FINGERTHE_TOUCH + 18 + 192,  _PAD4),
            ],
            'fingerfour': [
                (RegAddr.FINGERFOR_TOUCH,             _TIP),
                (RegAddr.FINGERFOR_TOUCH + 18,        _NAIL),
                (RegAddr.FINGERFOR_TOUCH + 18 + 192,  _PAD4),
            ],
            'fingerfive': [
                (RegAddr.FINGERFIV_TOUCH,                    _TIP),
                (RegAddr.FINGERFIV_TOUCH + 18,               _NAIL),
                (RegAddr.FINGERFIV_TOUCH + 18 + 192,         _MID5),
                (RegAddr.FINGERFIV_TOUCH + 18 + 192 + 18,    _PAD5),
            ],
            'fingerpalm': [
                (RegAddr.FINGERPALM_TOUCH, _PALM),
            ],
        }

    def _read_joint_states_once(self, client: ModbusTcpClient) -> dict[str, list[int]]:
        """Read all joint-related state registers from the Inspire hand.

        POS_ACT/ANGLE_ACT are coalesced into one block (their gap reads back
        fine). FORCE_ACT, CURRENT, ERROR, STATUS and TEMP are each read in their
        own request — coalescing those across the unmapped register gaps makes
        the firmware return garbage for the tail field (observed: TEMP=255/0,
        CURRENT=0/514). This matches how the official Inspire SDK reads them.
        """
        ra = self.RegAddr
        state_updates: dict[str, list[int]] = {}

        b1 = self._read_input_registers(client, ra.POS_ACT, ra.ANGLE_ACT - ra.POS_ACT + 6)
        if b1 is not None:
            state_updates['POS_ACT']   = self._decode_int16_registers(b1[:6])
            off = ra.ANGLE_ACT - ra.POS_ACT
            state_updates['ANGLE_ACT'] = self._decode_int16_registers(b1[off:off + 6])

        # FORCE_ACT and CURRENT are 6 int16 registers each, separated by an
        # unmapped gap (1588-1593). Coalescing them into one block that spans the
        # gap makes the firmware return garbage for the CURRENT tail (symptom:
        # current stuck at 0/514 mA). The official Inspire SDK reads every field
        # in its own request — do the same here.
        fb = self._read_input_registers(client, ra.FORCE_ACT, 6)
        if fb is not None:
            state_updates['FORCE_ACT'] = self._decode_int16_registers(fb[:6])
        cb = self._read_input_registers(client, ra.CURRENT, 6)
        if cb is not None:
            state_updates['CURRENT'] = self._decode_int16_registers(cb[:6])

        # ERROR / STATUS / TEMP are byte-packed (6 bytes = 3 registers each).
        # Reading them as one block that runs past the end of the TEMP field
        # makes some Inspire firmware return 0xFF for the tail (symptom: temps
        # read 255, with a lone 0 for the ring slot). Read each field in its own
        # 3-register request so none of them runs into unmapped registers.
        for key, addr in (('ERROR', ra.ERROR), ('STATUS', ra.STATUS), ('TEMP', ra.TEMP)):
            regs = self._read_input_registers(client, addr, 3)
            if regs is not None:
                state_updates[key] = self._unpack_registers_to_bytes(regs)

        return state_updates

    def _read_touch_once(self, client: ModbusTcpClient) -> dict[str, np.ndarray]:
        """Read all touch fields for one hand."""
        touch_updates: dict[str, np.ndarray] = {}
        for touch_key, segments in self._get_touch_segments(self.RegAddr, self.haptics_only).items():
            values = self._read_touch_field(client, segments)
            if values is not None:
                touch_updates[touch_key] = values
        return touch_updates

    def _refresh_sensors_once(self):
        """Read both state and touch once, synchronized to the control loop."""
        if self.hand_side == 'b':
            left_state = self._read_joint_states_once(self.client_left)
            left_touch = self._read_touch_once(self.client_left)
            right_state = self._read_joint_states_once(self.client_right)
            right_touch = self._read_touch_once(self.client_right)

            with self._data_lock:
                if left_state:
                    self.states['Left'].update(left_state)
                if left_touch:
                    self.touch['Left'].update(left_touch)
                if right_state:
                    self.states['Right'].update(right_state)
                if right_touch:
                    self.touch['Right'].update(right_touch)
        elif self.hand_side == 'l':
            state = self._read_joint_states_once(self.client)
            touch = self._read_touch_once(self.client)
            with self._data_lock:
                if state:
                    self.states['Left'].update(state)
                if touch:
                    self.touch['Left'].update(touch)
        else:
            state = self._read_joint_states_once(self.client)
            touch = self._read_touch_once(self.client)
            with self._data_lock:
                if state:
                    self.states['Right'].update(state)
                if touch:
                    self.touch['Right'].update(touch)

    def read(self) -> Dict:
        """Return a thread-safe snapshot of state and touch data."""
        with self._data_lock:
            return {
                'states': {
                    hand: {key: value.copy() for key, value in hand_states.items()}
                    for hand, hand_states in self.states.items()
                },
                'touch': {
                    hand: {key: value.copy() for key, value in hand_touch.items()}
                    for hand, hand_touch in self.touch.items()
                }
            }
    
    def _actuator_worker(self):
        """Background thread that sends position commands at 50 Hz.

        Sensor reads have been moved to dedicated sensor threads so they cannot
        delay position commands.
        """
        send_interval = 0.02  # target 50 Hz

        while self._running:
            t0 = time.monotonic()
            if self._suppressed.is_set():          # hands held off (e.g. robot power-cycled)
                time.sleep(0.05)
                continue
            try:
                if self._is_engaged:
                    # Set the max finger speed ONCE per engage. SPEED_SET (1522) is a
                    # persistent holding register, so re-writing it on every command (the old
                    # POSITION|VELOCITY combo) just doubled the Modbus round-trips per cycle
                    # for no benefit — that extra write time raised command latency and made
                    # the fingers feel sluggish. Now only POS_SET goes out each frame; the
                    # speed persists. Retried next cycle if the one-time write fails.
                    if not self._speed_applied and self._apply_max_speed():
                        self._speed_applied = True
                    _mode = self.ActStates.POSITION
                    if self.hand_side == 'b':
                        self._send_command(_mode, hand='l', pos_set=self.q_goal_left_hand)
                        self._send_command(_mode, hand='r', pos_set=self.q_goal_right_hand)
                    elif self.hand_side == 'l':
                        self._send_command(_mode, pos_set=self.q_goal_left_hand)
                    else:  # 'r'
                        self._send_command(_mode, pos_set=self.q_goal_right_hand)
                else:
                    self._speed_applied = False        # re-apply max speed on the next engage
            except Exception as e:
                if self._note_io_failure():          # _send_command already gates its own log
                    print(f"Error in actuator loop: {e} — hand unreachable (robot powered off?).")

            if self._io_quiet:
                time.sleep(_IO_QUIET_RETRY_S)          # slow, silent retry while the link is down
            # Sleep only for the remainder of the period so the Modbus write time (one POS_SET
            # write per hand) doesn't stack on top of the full interval — keeps the true
            # command rate at ~50 Hz instead of drifting slower, which makes fingers feel laggy.
            remaining = send_interval - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def _apply_max_speed(self) -> bool:
        """Write the max finger speed (SPEED_SET) once — returns True on success. Because
        SPEED_SET is a persistent holding register, a single write at engage keeps every finger
        at full speed without re-sending it each command (which doubled the per-cycle Modbus
        traffic and added command latency). Re-applied automatically on each re-engage and after
        a reboot (a power-cycle reloads the slower DEFAULT_SPEED_SET at 1032)."""
        mode = self.ActStates.VELOCITY
        if self.hand_side == 'b':
            ok_l = self._send_command(mode, hand='l', speed_set=_MAX_HAND_SPEED)
            ok_r = self._send_command(mode, hand='r', speed_set=_MAX_HAND_SPEED)
            return bool(ok_l and ok_r)
        return bool(self._send_command(mode, speed_set=_MAX_HAND_SPEED))

    def _sensor_worker_single(self, client: ModbusTcpClient, hand_key: str) -> None:
        """Continuously poll joint state and touch for one hand as fast as Modbus allows."""
        while self._running:
            if self._suppressed.is_set():          # bus held quiet while hands are off
                time.sleep(0.05)
                continue
            try:
                state = self._read_joint_states_once(client)
                touch = self._read_touch_once(client)
                with self._data_lock:
                    if state:
                        self.states[hand_key].update(state)
                    if touch:
                        self.touch[hand_key].update(touch)
                self._note_io_success()
            except Exception as e:
                if self._note_io_failure():
                    print(f"Sensor read error ({hand_key}): {e} — hand unreachable (robot "
                          "powered off?); suppressing further errors + slowing retries until "
                          "the link returns.")
            if self._io_quiet:
                time.sleep(_IO_QUIET_RETRY_S)          # slow, silent retry while the link is down

    def _sensor_worker(self) -> None:
        """Continuously poll joint state and touch for single-hand mode as fast as Modbus allows."""
        hand_key = 'Left' if self.hand_side == 'l' else 'Right'
        while self._running:
            if self._suppressed.is_set():          # bus held quiet while hands are off
                time.sleep(0.05)
                continue
            try:
                state = self._read_joint_states_once(self.client)
                touch = self._read_touch_once(self.client)
                with self._data_lock:
                    if state:
                        self.states[hand_key].update(state)
                    if touch:
                        self.touch[hand_key].update(touch)
                self._note_io_success()
            except Exception as e:
                if self._note_io_failure():
                    print(f"Sensor read error: {e} — hand unreachable (robot powered off?); "
                          "suppressing further errors + slowing retries until the link returns.")
            if self._io_quiet:
                time.sleep(_IO_QUIET_RETRY_S)          # slow, silent retry while the link is down

    def _convert_mediapipe_goals_to_hw_goals(self, finger_goals: list):
        """
        Assuming goals comes from solver where each hand comes in this order:
        [pinky, ring, middle, index, thumb1, thumb2]
        """

        # 1) Adjust values to match hw expectations (0-2000 range for position control)
        hw_goals = [int(np.clip(finger_goals[i] / self.range_list[i] * 2000, 0, 2000)) for i in range(6)]
        
        return hw_goals

    def set_joint_goals(self, goals: Dict):
        """
        Set joint goals for the hand.
        
        Args:
            joint_goals (list): List of 6 joint goals for the hand.
        
        Returns:
            bool: True if command was sent successfully
        """
        if not goals: return
        
        if goals.get("q_goal_right_hand") is not None:
            # Convert from mediapipe rad to hw pos:

            self.q_goal_right_hand = self._convert_mediapipe_goals_to_hw_goals(goals["q_goal_right_hand"])

        if goals.get("q_goal_left_hand") is not None:
            self.q_goal_left_hand = self._convert_mediapipe_goals_to_hw_goals(goals["q_goal_left_hand"])

        # print("Left joint goals: ", self.q_goal_left_hand)
        # print("Right joint goals: ", self.q_goal_right_hand)

    def apply_control(self, engaged=True):
        """
        Apply control to the hand.
        
        Args:
            engaged (bool): Whether to apply control (True) or stop (False).
        
        Returns:
            bool: True if command was sent successfully
        """
        self._is_engaged = engaged
        if not engaged:
            pass

        return True

    def _update_current_positions(self):
        """
        Update current positions of the hand.
        """
        pass
        # TODO: implement this
    
    def _send_command(self, mode, hand=None, **kwargs):
        """
        Internal method to correctly route command down via direct Modbus writes
        Bypassing the original _send_command DDS mechanism and directly applying 
        it exactly as the ModbusDataHandler class did.
        """
        client = None
        if self.hand_side == 'b':
            if hand == 'l':
                client = self.client_left
            elif hand == 'r':
                client = self.client_right
            else:
                raise ValueError("hand parameter must be 'l' or 'r' when hand_side is 'b'")
        else:
            client = self.client
            
        try:
            with self._get_io_lock(client):
                if mode & self.ActStates.ANGLE:  # Angle mode
                    client.write_registers(self.RegAddr.ANGLE_SET, kwargs.get('angle_set', [0]*6), device_id=self.device_id)
                if mode & self.ActStates.POSITION:  # Position mode
                    client.write_registers(self.RegAddr.POS_SET, kwargs.get('pos_set', [0]*6), device_id=self.device_id)
                if mode & self.ActStates.FORCE:  # Force mode
                    client.write_registers(self.RegAddr.FORCE_SET, kwargs.get('force_set', [0]*6), device_id=self.device_id)
                if mode & self.ActStates.VELOCITY:  # Velocity (speed) mode
                    client.write_registers(self.RegAddr.SPEED_SET, kwargs.get('speed_set', [0]*6), device_id=self.device_id)
            self._note_io_success()
            return True
        except Exception as e:
            if self._note_io_failure():
                print(f"Modbus write exception: {e} — hand unreachable (robot powered off?); "
                      "suppressing further errors until the link returns.")
            return False
    
    def read_touch_full(self, side: str) -> dict:
        """Directly read the FULL touch map (tip+nail+pad per finger, plus palm)
        for one hand, independent of the ``haptics_only`` setting.

        Returns ``{finger_key: np.ndarray}`` (keys ``fingerone``..``fingerpalm``);
        a finger is omitted if its segmented read failed (a Modbus error is then
        printed with the offending address). Used by the force-calibration window
        so the pad is always available without flipping the live haptics read mode
        and waiting on the background sensor threads. Thread-safe via the per-client
        IO lock, so it serialises with the sensor threads on the same connection.
        """
        if self.hand_side == 'b':
            client = self.client_left if side == 'Left' else self.client_right
        else:
            client = self.client
        out: dict[str, Any] = {}
        segments_by_key = self._get_touch_segments(self.RegAddr, haptics_only=False)
        for touch_key, segments in segments_by_key.items():
            values = self._read_touch_field(client, segments)
            if values is not None:
                out[touch_key] = values
        return out

    def calibrate_force_sensors(self, hand=None):
        """Trigger the hand's built-in force-sensor calibration (GESTURE_FORCE_CLB,
        reg 1009) by writing 1.

        Per the RH56DFTP manual this re-zeros the force sensing and MUST be run
        with the hand fully open and unloaded (no contact on any fingertip or the
        palm). Optional: the fingertip-force calibration window relies primarily
        on a software baseline, but this lets the operator also reset the hand's
        own reference if desired.

        Args:
            hand (str, optional): 'l' or 'r' when in two-hand mode; ignored
                                  otherwise. None calibrates whichever client(s)
                                  are active.

        Returns:
            bool: True if the write(s) were acknowledged.
        """
        def _zero(client) -> bool:
            try:
                with self._get_io_lock(client):
                    client.write_registers(self.RegAddr.GESTURE_FORCE_CLB, [1], device_id=self.device_id)
                return True
            except Exception as e:
                print(f"Force-sensor calibration write failed: {e}")
                return False

        if self.hand_side == 'b':
            if hand == 'l':
                return _zero(self.client_left)
            if hand == 'r':
                return _zero(self.client_right)
            ok_l = _zero(self.client_left)
            ok_r = _zero(self.client_right)
            return ok_l and ok_r
        return _zero(self.client)

    def set_angle(self, angles):
        """
        Set joint angles.
        
        Args:
            angles (list): List of 6 angle values for joints in order:
                          [pinky, ring, middle, index, thumb-bend, thumb-rotation]
        
        Returns:
            bool: True if command was sent successfully
        
        Example:
            >>> hand.set_angle([0, 0, 0, 0, 1000, 1000])
        """
        if len(angles) != 6:
            raise ValueError("angles must be a list of 6 values")
        return self._send_command(self.ActStates.ANGLE, angle_set=angles)
    
    def set_position(self, positions, hand=None):
        """
        Set joint positions.
        
        Args:
            positions (list): List of 6 position values for joints in order:
                             [pinky, ring, middle, index, thumb-bend, thumb-rotation]
            hand (str, optional): Which hand to send to ('l', 'r', or None for current hand)
        
        Returns:
            bool: True if command was sent successfully
        
        Example:
            >>> hand.set_position([500, 500, 500, 500, 500, 500])
        """
        
        return self._send_command(
            mode=self.ActStates.POSITION,
            hand=hand,
            pos_set=positions
        )
    
    def set_force(self, forces):
        """
        Set force control values.
        
        Args:
            forces (list): List of 6 force values for joints in order:
                          [pinky, ring, middle, index, thumb-bend, thumb-rotation]
        
        Returns:
            bool: True if command was sent successfully
        
        Example:
            >>> hand.set_force([100, 100, 100, 100, 100, 100])
        """
        if len(forces) != 6:
            raise ValueError("forces must be a list of 6 values")
        return self._send_command(
            mode=self.ActStates.FORCE, 
            force_set=forces
        )
    
    def set_velocity(self, velocities):
        """
        Set joint velocities (speed).
        
        Args:
            velocities (list): List of 6 velocity values for joints in order:
                              [pinky, ring, middle, index, thumb-bend, thumb-rotation]
        
        Returns:
            bool: True if command was sent successfully
        
        Example:
            >>> hand.set_velocity([50, 50, 50, 50, 50, 50])
        """
        if len(velocities) != 6:
            raise ValueError("velocities must be a list of 6 values")
        return self._send_command(
            mode=self.ActStates.VELOCITY, 
            speed_set=velocities
        )

    def set_angle_and_position(self, angles, positions):
        """
        Set both angles and positions simultaneously.
        
        Args:
            angles (list): List of 6 angle values
            positions (list): List of 6 position values
        
        Returns:
            bool: True if command was sent successfully
        """
        if len(angles) != 6 or len(positions) != 6:
            raise ValueError("angles and positions must each be lists of 6 values")
        return self._send_command(
            mode=self.ActStates.ANGLE | self.ActStates.POSITION, 
            angle_set=angles, 
            pos_set=positions
        )
    
    def set_angle_and_velocity(self, angles, velocities):
        """
        Set both angles and velocities simultaneously.
        
        Args:
            angles (list): List of 6 angle values
            velocities (list): List of 6 velocity values
        
        Returns:
            bool: True if command was sent successfully
        """
        if len(angles) != 6 or len(velocities) != 6:
            raise ValueError("angles and velocities must each be lists of 6 values")
        return self._send_command(
            mode=self.ActStates.ANGLE | self.ActStates.VELOCITY, 
            angle_set=angles, 
            speed_set=velocities
        )
    
    def connected_sides(self) -> list[str]:
        """Hand sides ('l' / 'r') whose Modbus TCP link connected."""
        if self.hand_side == 'b':
            return [side for side, client in (('l', self.client_left), ('r', self.client_right))
                    if client.connected]
        return [self.hand_side] if self.client.connected else []

    def close(self):
        """Stop the worker threads and close the Modbus links without sending a command
        (for hands that never connected, where stop() would only time out)."""
        self._is_engaged = False
        self._running = False
        for _attr in ('_actuator_thread', '_sensor_thread', '_sensor_thread_left', '_sensor_thread_right'):
            _t = getattr(self, _attr, None)
            if _t is not None and _t.is_alive():
                _t.join(timeout=0.5)
        for client in (self.client, getattr(self, 'client_left', None), getattr(self, 'client_right', None)):
            if client is not None:
                client.close()

    def stop(self):
        """
        Send a no-operation command (mode 0) and stop thread.
        
        Returns:
            bool: True if command was sent successfully
        """
        self._is_engaged = False
        self._running = False
        if hasattr(self, '_actuator_thread') and self._actuator_thread is not None and self._actuator_thread.is_alive():
            self._actuator_thread.join(timeout=0.1)
        for _attr in ('_sensor_thread', '_sensor_thread_left', '_sensor_thread_right'):
            _t = getattr(self, _attr, None)
            if _t is not None and _t.is_alive():
                _t.join(timeout=0.5)
        
        if self.hand_side == 'b':
            # Send stop to both hands
            success_left = self._send_command(mode=self.ActStates.IDLE, hand='l')
            success_right = self._send_command(mode=self.ActStates.IDLE, hand='r')
            return success_left and success_right
        else:
            return self._send_command(mode=self.ActStates.IDLE)

    def _note_io_failure(self) -> bool:
        """Record a Modbus I/O failure. Returns True EXACTLY ONCE — when the link is first judged
        lost (≥ _IO_QUIET_AFTER consecutive failures) — so the caller prints a single notice;
        every failure after that returns False (silent). On entering the quiet state it also
        silences pymodbus's own 'Connection to … failed' logging so a powered-off robot goes
        fully quiet."""
        with self._io_state_lock:
            self._io_fail_count += 1
            if self._io_fail_count >= _IO_QUIET_AFTER and not self._io_quiet:
                self._io_quiet = True
                logging.getLogger("pymodbus").setLevel(logging.CRITICAL)
                return True
            return False

    def _note_io_success(self) -> None:
        """Record a successful Modbus op. If we were quiet (link had dropped), announce the
        reconnect and restore normal pymodbus logging."""
        with self._io_state_lock:
            was_quiet = self._io_quiet
            self._io_fail_count = 0
            self._io_quiet = False
        if was_quiet:
            logging.getLogger("pymodbus").setLevel(self._pymodbus_log_level)
            print("[InspireHand] Modbus link back — resuming normal hand I/O + error logging.")

    def suppress_output(self, on: bool) -> None:
        """Hold the hands OFF (on=True): both worker threads idle — no Modbus writes (no hand
        commands) and no reads (the bus goes quiet) — until released. Used when the robot is
        power-cycled so the hands aren't driven the instant they reconnect. reboot() clears it."""
        if on:
            self._suppressed.set()
        else:
            self._suppressed.clear()

    def output_suppressed(self) -> bool:
        return self._suppressed.is_set()

    def reboot(self) -> bool:
        """Re-run the __init__ hardware bring-up IN PLACE — reconnect the Modbus link(s) and
        restart the actuator + sensor worker threads — so the hands and their force/tactile
        sensors start fresh WITHOUT recreating the controller (every reference held by the
        ControlThread / haptics streamer / recorder / monitor stays valid). Recovers hands or
        force sensors that stopped responding without restarting the manager.

        Blocks ~1-2 s (thread joins + Modbus reconnect); call it OFF the teleop path. read()
        keeps returning the last cached snapshot throughout, and apply_control() writes are
        exception-guarded, so the brief reconnect window is safe. Returns True if the Modbus
        link(s) reconnected."""
        # 1. Stop the worker threads (they loop on _running).
        self._running = False
        _t = getattr(self, '_actuator_thread', None)
        if _t is not None and _t.is_alive():
            _t.join(timeout=1.0)
        for _attr in ('_sensor_thread', '_sensor_thread_left', '_sensor_thread_right'):
            _t = getattr(self, _attr, None)
            if _t is not None and _t.is_alive():
                _t.join(timeout=1.0)

        # 2. Close the old Modbus client(s).
        for _attr in ('client', 'client_left', 'client_right'):
            _c = getattr(self, _attr, None)
            if _c is not None:
                try:
                    _c.close()
                except Exception:
                    pass

        # 3. Reconnect — same targets as __init__.
        ok = True
        if self.hand_side == 'b':
            self.client_left = ModbusTcpClient('192.168.123.210', port=6000)
            ok = bool(self.client_left.connect()) and ok
            self.client_right = ModbusTcpClient('192.168.123.211', port=6000)
            ok = bool(self.client_right.connect()) and ok
            self.client = None
        else:
            ip = '192.168.123.210' if self.hand_side == 'l' else '192.168.123.211'
            self.client = ModbusTcpClient(ip, port=6000)
            ok = bool(self.client.connect()) and ok

        # 4. Restart the worker threads (mirrors the tail of __init__). A reboot always brings
        # the hands back live, so release any hold-off set during a robot power-cycle and clear
        # the lost-link quiet state (restoring pymodbus logging).
        self._suppressed.clear()
        # A power-cycle reloads the slower DEFAULT_SPEED_SET, so force max speed to be
        # re-written on the next engage rather than trusting the stale latch.
        self._speed_applied = False
        with self._io_state_lock:
            self._io_fail_count = 0
            self._io_quiet = False
        logging.getLogger("pymodbus").setLevel(self._pymodbus_log_level)
        self._running = True
        self._actuator_thread = threading.Thread(target=self._actuator_worker, daemon=True)
        self._actuator_thread.start()
        if self.hand_side == 'b':
            self._sensor_thread_left = threading.Thread(
                target=self._sensor_worker_single, args=(self.client_left, 'Left'), daemon=True)
            self._sensor_thread_right = threading.Thread(
                target=self._sensor_worker_single, args=(self.client_right, 'Right'), daemon=True)
            self._sensor_thread_left.start()
            self._sensor_thread_right.start()
        else:
            self._sensor_thread = threading.Thread(target=self._sensor_worker, daemon=True)
            self._sensor_thread.start()

        print("[InspireHand] reboot complete — Modbus reconnected + workers restarted."
              if ok else
              "[InspireHand] reboot: Modbus reconnect FAILED — check hand power / cables / IPs.")
        return ok
   


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    finger_keys = ['fingerone', 'fingertwo', 'fingerthree', 'fingerfour', 'fingerfive', 'fingerpalm']
    
    hand_controller = InspireHandController(hand_side='b', initialize_dds=True)

    # Constant step command for all 6 joints (in radians) on both hands.
    step_ratio = 0.35
    step_goal = [step_ratio * r for r in hand_controller.range_list]
    cli_period_sec = 0.1

    try:
        hand_controller.set_joint_goals({
            'q_goal_left_hand': step_goal.copy(),
            'q_goal_right_hand': step_goal.copy(),
        })
        hand_controller.apply_control(engaged=True)

        logging.info(
            "Started minimal constant-step test | step_ratio=%.2f q_goal(rad)=%s",
            step_ratio,
            np.array2string(np.asarray(step_goal), precision=3, separator=','),
        )
        logging.info(
            "Sensor refresh runs concurrently in controller worker; printing snapshots every %.2fs",
            cli_period_sec,
        )

        while True:
            # step for update sensor signal
            snapshot = hand_controller.read()
            
            for hand_label in ('Left', 'Right'):
                
                # snapshot contains 'states' and 'touch' dicts for each hand; extract relevant data for logging
                states = snapshot.get('states', {}).get(hand_label, {})
                touch = snapshot.get('touch', {}).get(hand_label, {})

                pos_act = np.asarray(states.get('POS_ACT', []), dtype=float)
                pos_preview = np.array2string(pos_act[:6], precision=0, separator=',') if pos_act.size else '[]'

                touch_means: dict[str, float] = {}
                for key in finger_keys:
                    values = np.asarray(touch.get(key, []), dtype=float)
                    # simplification: average all the grid readings
                    touch_means[key] = float(np.mean(values)) if values.size else 0.0

                logging.info(
                    "%s | POS_ACT=%s | touch: palm=%.1f f1=%.1f f2=%.1f f3=%.1f f4=%.1f f5=%.1f",
                    hand_label,
                    pos_preview,
                    touch_means['fingerpalm'],
                    touch_means['fingerone'],
                    touch_means['fingertwo'],
                    touch_means['fingerthree'],
                    touch_means['fingerfour'],
                    touch_means['fingerfive'],
                )
            time.sleep(cli_period_sec)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received; stopping minimal test.")
    finally:
        try:
            hand_controller.apply_control(engaged=False)
        except Exception:
            pass
        hand_controller.stop()
