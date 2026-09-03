"""
MotorStatsStreamer
=================
Streams G1 arm and Inspire finger actuator telemetry (torque, current,
temperature) over a dedicated WebRTC data channel ("motor_stats") to Unity.

Both data sources are zero-cost in terms of additional I/O:
- Arm data is read directly from the Unitree DDS LowState, which the
  Controller already subscribes to in order to send motion commands.
- Finger data is pulled from the InspireHandController's cached sensor
  snapshot, which the sensor-worker threads update continuously.

This keeps haptics latency completely unaffected.

Usage (background thread, analogous to InspireHandHapticsStreamer)
------------------------------------------------------------------
    streamer = MotorStatsStreamer(hw_controller, hand_controller, device)
    streamer.start()
    # ... main loop ...
    streamer.stop()

Usage (inline, from a loop that already has a hand snapshot)
-------------------------------------------------------------
    snapshot = hand_controller.read()
    streamer.send_once(hand_snapshot=snapshot)

JSON payload schema
-------------------
{
  "type":      "motor_stats",
  "timestamp": <float>,          # Unix epoch seconds
  "arms": {
    "torso": {
      "motor_ids": [int, ...],   # Unitree motor indices (3 joints)
      "tau_est_Nm": [float, ...],
      "temp_c":     [float, ...]
    },
    "left":  { ... },            # 7 joints
    "right": { ... }             # 7 joints
  },
  "fingers": {
    "left": {
      "force_act": [int, ...],   # Actual force per DOF (0.1 N/unit)
      "current":   [int, ...],   # Actuator current per DOF (mA)
      "temp_c":    [int, ...]    # Actuator temperature per DOF (°C)
    },
    "right": { ... }
  }
}

Finger DOF order (matches InspireHandRegAddr / g1.yaml):
  [pinky, ring, middle, index, thumb_bend, thumb_rotate]

"arms" is omitted when hw_controller is None (hand-only or dry-run mode).
"fingers" is omitted when hand_controller is None.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class MotorStatsStreamer:
    """Streams arm + finger motor telemetry over the 'motor_stats' WebRTC channel.

    Parameters
    ----------
    hw_controller:
        Unitree ``Controller`` instance whose ``low_state.motor_state[i]``
        exposes ``.tau_est`` (Nm) and ``.temperature`` (°C) per motor.
        Pass ``None`` in hand-only or dry-run mode — arm stats are omitted.
    hand_controller:
        ``InspireHandController`` with a thread-safe ``read()`` method.
        Pass ``None`` to omit finger stats.
    xr_device:
        Object with a ``send_motor_stats(payload: str)`` method, typically
        an ``XRRTCBodyPoseDevice``.
    rate_hz:
        Background-thread send rate (default 10 Hz).  Thermal monitoring
        does not need haptic-level latency.
    """

    def __init__(
        self,
        hw_controller: Any,
        hand_controller: Any,
        xr_device: Any,
        rate_hz: float = 10.0,
    ) -> None:
        self._hw = hw_controller
        self._hand = hand_controller
        self._device = xr_device
        self._interval = 1.0 / rate_hz

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> "MotorStatsStreamer":
        """Start the background streaming thread; returns *self* for chaining."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="motor-stats"
        )
        self._thread.start()
        logger.info(
            "[MotorStatsStreamer] Background thread started at %.0f Hz.",
            1.0 / self._interval,
        )
        return self

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("[MotorStatsStreamer] Stopped.")

    def send_once(self, hand_snapshot: dict | None = None) -> None:
        """Send one motor-stats payload immediately (call from your own loop).

        Pass *hand_snapshot* if you already called ``hand.read()`` this tick
        to avoid a redundant read.  All errors are swallowed so the caller's
        loop is never interrupted.
        """
        try:
            if not self._is_peer_connected():
                return
            payload = self._build_payload(hand_snapshot)
            self._device.send_motor_stats(payload)
        except Exception as exc:
            logger.warning("[MotorStatsStreamer] send_once error: %s", exc)

    def __enter__(self) -> "MotorStatsStreamer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                if self._is_peer_connected():
                    payload = self._build_payload()
                    self._device.send_motor_stats(payload)
            except Exception as exc:
                logger.warning("[MotorStatsStreamer] loop error: %s", exc)
            elapsed = time.monotonic() - t0
            remaining = self._interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _is_peer_connected(self) -> bool:
        try:
            return self._device.state_factory.instance is not None
        except AttributeError:
            return True

    def _build_payload(self, hand_snapshot: dict | None = None) -> str:
        payload: dict[str, Any] = {
            "type": "motor_stats",
            "timestamp": time.time(),
        }

        # ── Arm telemetry (Unitree DDS LowState — no extra I/O) ──────────────
        if self._hw is not None:
            try:
                ls = self._hw.low_state

                def _temp_c(t) -> float:
                    # G1 (HG) MotorState.temperature is int16[2] (motor, driver
                    # sensor); report the hotter one. Go firmware may give a scalar.
                    try:
                        return float(max(t))
                    except TypeError:
                        return float(t)

                def _arm_stats(indices: list[int]) -> dict:
                    return {
                        "motor_ids":  list(indices),
                        "tau_est_Nm": [round(float(ls.motor_state[i].tau_est),    4) for i in indices],
                        "temp_c":     [round(_temp_c(ls.motor_state[i].temperature), 1) for i in indices],
                    }

                payload["arms"] = {
                    "torso": _arm_stats(self._hw._torso_idx),
                    "left":  _arm_stats(self._hw._left_idx),
                    "right": _arm_stats(self._hw._right_idx),
                }
            except Exception as exc:
                logger.debug("[MotorStatsStreamer] arm read error: %s", exc)
                payload["arms"] = None

        # ── Finger telemetry (cached InspireHand snapshot — no extra Modbus) ─
        if self._hand is not None:
            try:
                snap = hand_snapshot if hand_snapshot is not None else self._hand.read()
                states = snap.get("states", {})

                def _finger_stats(side: str) -> dict:
                    s = states.get(side, {})
                    return {
                        "force_act": s.get("FORCE_ACT", []),
                        "current":   s.get("CURRENT",   []),
                        "temp_c":    s.get("TEMP",      []),
                    }

                payload["fingers"] = {
                    "left":  _finger_stats("Left"),
                    "right": _finger_stats("Right"),
                }
            except Exception as exc:
                logger.debug("[MotorStatsStreamer] finger read error: %s", exc)
                payload["fingers"] = None

        return json.dumps(payload)
