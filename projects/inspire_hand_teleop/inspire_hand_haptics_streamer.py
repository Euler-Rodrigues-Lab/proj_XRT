"""
InspireHandHapticsStreamer
==========================
Reads Inspire RH56DFTP touch-sensor data and streams normalised per-finger /
palm haptic intensities over WebRTC.

Designed to drop into any script that already holds an InspireHandController
and an XRRTCBodyPoseDevice (or any object with a ``send_haptics(str)`` method).

Two usage modes
---------------
Background thread — haptics run autonomously at ``rate_hz``, decoupled from
your main loop.  Suitable for teleop scripts where the loop is busy::

    streamer = InspireHandHapticsStreamer(hand_controller, device, rate_hz=60)
    streamer.start()
    # … your main loop, no haptics code needed …
    streamer.stop()

Manual / per-tick — you drive the timing and supply a snapshot you already
read for another purpose (avoids a second ``hand.read()`` call and any
thread-safety concern)::

    snapshot = hand.read()
    streamer.send_once(snapshot=snapshot)   # safe to call from the main thread

Context-manager (background thread)::

    with InspireHandHapticsStreamer(hand_controller, device) as streamer:
        ...

Thread safety
-------------
When running as a background thread, ``hand_controller.read()`` is called from
the streamer thread while the main thread may call ``apply_control()`` or its
own ``read()``.  If your hand controller is not thread-safe, either use
``send_once()`` from a single thread, or pass a shared ``threading.Lock`` via
the ``hand_lock`` parameter::

    lock = threading.Lock()
    streamer = InspireHandHapticsStreamer(hand, device, hand_lock=lock)

    # In your main loop, acquire the same lock before touching the controller:
    with lock:
        hand.apply_control(engaged=True)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import numpy as np

from projects.inspire_hand_teleop.inspire_fingertip_force_calibration import (
    ForceCalibration,
    region_raw,
)

logger = logging.getLogger(__name__)

# Inspire RH56DFTP ADC full-scale (signed int16; negatives = no contact → clamped to 0)
_SENSOR_MAX_DEFAULT: float = 1024.0

# The per-region raw metric (peak tip/nail taxel for fingers, mean for the palm)
# lives in inspire_fingertip_force_calibration.region_raw so the haptics signal
# and the force calibration can never diverge — see that module for the layout.

# Inspire finger register key → WebRTC haptics JSON field, in order
_FINGER_MAP: list[tuple[str, str]] = [
    ("fingerfive",  "thumb"),
    ("fingerfour",  "index"),
    ("fingerthree", "middle"),
    ("fingertwo",   "ring"),
    ("fingerone",   "little"),
    ("fingerpalm",  "palm"),
]


class InspireHandHapticsStreamer:
    """Streams Inspire touch-sensor data as WebRTC haptics payloads.

    Parameters
    ----------
    hand_controller:
        Object with a ``read() -> dict`` method returning a snapshot that
        contains ``snapshot['touch']['Left']`` and ``snapshot['touch']['Right']``
        keyed by Inspire finger register names.
    xr_device:
        Object with a ``send_haptics(payload: str)`` method.  Typically an
        ``XRRTCBodyPoseDevice``.  Peer-connection awareness is detected
        automatically if the object exposes ``state_factory.instance``.
    rate_hz:
        Background-thread send rate in Hz.  Only used when ``start()`` is
        called; has no effect when driving manually via ``send_once()``.
    sensor_max:
        ADC value that maps to intensity 1.0 (default 1024).
    hand_lock:
        Optional ``threading.Lock`` acquired before every ``hand.read()``
        call inside the background thread.  Pass the same lock wherever
        else you access the hand controller.
    debug:
        When True, log haptic values at INFO level every ~2 s and on the
        first non-zero contact detection.
    calibration_secs:
        Seconds of baseline collection on first use (default 3.0).  During
        this window no haptics are sent and the resting sensor values are
        averaged per finger.  That offset is subtracted from all subsequent
        readings so idle noise does not produce false vibrations.
    """

    def __init__(
        self,
        hand_controller: Any,
        xr_device: Any,
        rate_hz: float = 60.0,
        sensor_max: float = _SENSOR_MAX_DEFAULT,
        hand_lock: threading.Lock | None = None,
        debug: bool = False,
        calibration_secs: float = 3.0,
        default_deadband: float = 0.0,
    ) -> None:
        self._hand = hand_controller
        self._device = xr_device
        self._interval = 1.0 / rate_hz
        self._sensor_max = float(sensor_max)
        self._hand_lock = hand_lock
        self._debug = debug

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Test mode: when set, the streamer holds ALL real haptics sends so a synthetic
        # "Test Haptics" sequence can drive the channel directly without interleaving live
        # (near-zero) frames on top of the ramp. Set/cleared by the manager around a test.
        self._output_suppressed = threading.Event()

        self._last_log_t: float = 0.0
        self._ever_nonzero: bool = False

        # Fingertip/palm calibration is NOISE-FLOOR only: the per-region idle noise floors are
        # used as a deadband to suppress phantom triggers (the raw sensor intensities are
        # streamed unchanged). ``default_deadband`` applies the same gate (counts above
        # baseline) when no calibration is loaded. (Weight→grams calibration was removed —
        # the haptics use the raw force-sensor readings directly.)
        self._default_deadband: float = float(default_deadband)
        # Palm channel gate. The palm IS a first-class haptics region (read in
        # haptics_only mode, streamed as left/right "palm") — this flag lets the
        # manager mute JUST the palm at the source, which silences BOTH the
        # vibrotactile and the audio haptics Unity derives from the intensities.
        self.palm_enabled: bool = True
        # Per-region deadband cache, precomputed on calibration load so the
        # per-payload hot path (which runs inside the IK ControlThread) does a
        # single dict lookup instead of recomputing the noise floor each time.
        self._deadbands: dict[str, dict[str, float]] | None = None

        # Calibration: collect baseline for `calibration_secs` seconds on first
        # use (assuming no contact), then subtract from all future readings.
        self._cal_secs: float = float(calibration_secs)
        self._cal_start: float | None = None   # set on first _build_payload call
        self._calibrating: bool = True
        self._cal_samples: dict[str, dict[str, list[float]]] = {
            "Left":  {hf: [] for _, hf in _FINGER_MAP},
            "Right": {hf: [] for _, hf in _FINGER_MAP},
        }
        self._baseline: dict[str, dict[str, float]] = {
            "Left":  {hf: 0.0 for _, hf in _FINGER_MAP},
            "Right": {hf: 0.0 for _, hf in _FINGER_MAP},
        }

        # Verification snapshot for the Motor Monitor (read-only). The hot path (below)
        # only STORES references to the touch snapshot + the signals it already computed —
        # a few-microsecond lock+assign that does NOT add a hardware read or slow the IK/
        # haptics loop. latest_forces() does the per-sub-region math LATER, on the monitor's
        # own ~5 Hz UI thread, so nothing extra ever runs on the teleop path.
        self._forces_lock = threading.Lock()
        self._latest_touch: dict | None = None
        self._latest_left: dict | None = None
        self._latest_right: dict | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> "InspireHandHapticsStreamer":
        """Start the background streaming thread; returns *self* for chaining."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="inspire-haptics"
        )
        self._thread.start()
        logger.info(
            "[InspireHandHapticsStreamer] Background thread started at %.0f Hz.",
            1.0 / self._interval,
        )
        return self

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("[InspireHandHapticsStreamer] Stopped.")

    def load_force_calibration(self, path=None) -> bool:
        """Load the fingertip/palm calibration's per-region NOISE FLOORS and apply them as the
        live deadband (suppresses phantom triggers). The raw sensor intensities are streamed
        unchanged — there is no weight/grams calibration. Returns True if any noise floor was
        loaded. Safe to call at startup (auto-apply) or after a fresh auto-calibration.
        """
        from projects.inspire_hand_teleop.inspire_fingertip_force_calibration import (
            DEFAULT_CALIBRATION_PATH,
        )
        cal = ForceCalibration.load(path or DEFAULT_CALIBRATION_PATH)
        if cal.has_any_noise_floor():
            # Precompute the per-region deadband once so the IK-loop payload path
            # only does a dict lookup (no per-payload noise-floor computation).
            self._deadbands = {
                side: {ik: cal.live_noise_floor(side, ik) for ik, _ in _FINGER_MAP}
                for side in ("Left", "Right")
            }
            logger.info("[haptics] Fingertip calibration loaded (noise-gate applied).")
            return True
        logger.info("[haptics] No fingertip calibration found; noise-gate disabled.")
        return False

    def suppress_output(self, on: bool) -> None:
        """Pause (on=True) or resume (on=False) the streamer's own haptics sends. Used while a
        synthetic Test-Haptics sequence drives the 'haptics' channel directly, so live
        near-zero frames don't interleave with (and stutter) the test ramp."""
        if on:
            self._output_suppressed.set()
        else:
            self._output_suppressed.clear()

    def output_suppressed(self) -> bool:
        return self._output_suppressed.is_set()

    def send_once(self, snapshot: dict | None = None) -> None:
        """Send one haptics payload immediately (call from your own loop).

        If *snapshot* is supplied it is used directly (no extra ``read()``
        call).  All errors are logged and swallowed so the caller's loop is
        never interrupted.
        """
        try:
            if self._output_suppressed.is_set():
                return
            if not self._is_peer_connected():
                return
            if snapshot is None:
                if self._hand_lock is not None:
                    with self._hand_lock:
                        snapshot = self._hand.read()
                else:
                    snapshot = self._hand.read()
            payload = self._build_payload(snapshot)
            self._device.send_haptics(payload)
            if self._debug:
                self._maybe_log(payload)
        except Exception as exc:
            logger.warning("[InspireHandHapticsStreamer] send_once error: %s", exc)

    # ── context-manager support ───────────────────────────────────────────────

    def __enter__(self) -> "InspireHandHapticsStreamer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                if not self._output_suppressed.is_set() and self._is_peer_connected():
                    if self._hand_lock is not None:
                        with self._hand_lock:
                            snapshot = self._hand.read()
                    else:
                        snapshot = self._hand.read()
                    payload = self._build_payload(snapshot)
                    self._device.send_haptics(payload)
                    if self._debug:
                        self._maybe_log(payload)
            except Exception as exc:
                logger.warning("[InspireHandHapticsStreamer] loop error: %s", exc)
            elapsed = time.monotonic() - t0
            remaining = self._interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _is_peer_connected(self) -> bool:
        """Return True if a WebRTC peer is connected (or check is unavailable)."""
        try:
            return self._device.state_factory.instance is not None
        except AttributeError:
            return True

    def _build_payload(self, snapshot: dict) -> str:
        touch = snapshot.get("touch", {})
        now = time.monotonic()

        def _raw(hand_key: str, finger_key: str) -> float:
            """Raw sensor intensity, non-negative, before baseline subtraction.

            Delegates to the shared metric so the live signal and the force
            calibration stay identical.
            """
            return region_raw(touch.get(hand_key, {}).get(finger_key), finger_key)

        # ── Calibration phase ─────────────────────────────────────────────────
        if self._calibrating:
            if self._cal_start is None:
                self._cal_start = now
                logger.info(
                    "[haptics] Calibration started — keep fingers clear for %.1f s.",
                    self._cal_secs,
                )
            if now - self._cal_start < self._cal_secs:
                for ik, hf in _FINGER_MAP:
                    self._cal_samples["Left"][hf].append(_raw("Left", ik))
                    self._cal_samples["Right"][hf].append(_raw("Right", ik))
                zero = {hf: 0.0 for _, hf in _FINGER_MAP}
                return json.dumps({"type": "haptics", "timestamp": time.time(),
                                   "left": zero, "right": zero})
            # Calibration window elapsed — lock in baselines.
            for hand_key in ("Left", "Right"):
                for _, hf in _FINGER_MAP:
                    samps = self._cal_samples[hand_key][hf]
                    self._baseline[hand_key][hf] = float(np.mean(samps)) if samps else 0.0
            self._calibrating = False
            logger.info(
                "[haptics] Calibration complete.  "
                "L: %s  R: %s",
                {hf: f"{v:.1f}" for hf, v in self._baseline["Left"].items()},
                {hf: f"{v:.1f}" for hf, v in self._baseline["Right"].items()},
            )

        # ── Normal operation (baseline-corrected, noise-gated) ────────────────
        def _deadband(hand_key: str, finger_key: str) -> float:
            if self._deadbands is not None:
                return self._deadbands[hand_key][finger_key]
            return self._default_deadband

        def _intensity(hand_key: str, finger_key: str, hf: str) -> float:
            raw = _raw(hand_key, finger_key)
            delta = raw - self._baseline[hand_key][hf]
            # Gate idle noise: anything at/below the measured noise floor is zero.
            # Above it the scale is unchanged, so Unity thresholds set on the
            # existing 0..1 scale keep mapping cleanly (keep the floor below them).
            if delta <= _deadband(hand_key, finger_key):
                return 0.0
            return float(np.clip(delta / self._sensor_max, 0.0, 1.0))

        left  = {hf: _intensity("Left",  ik, hf) for ik, hf in _FINGER_MAP}
        right = {hf: _intensity("Right", ik, hf) for ik, hf in _FINGER_MAP}
        if not self.palm_enabled:                 # palm muted at the source
            left["palm"] = 0.0
            right["palm"] = 0.0

        payload = {
            "type": "haptics",
            "timestamp": time.time(),
            "left": left,
            "right": right,
        }

        # The payload is intensity-only (raw force-sensor readings, noise-gated) — Unity derives
        # both audio and vibrotactile from these intensities. No grams/weight signalling.

        # Cache references for the Motor Monitor's verification readout. Cheap: no copy, no
        # read — just publish the objects we already built. All display math happens in
        # latest_forces() on the monitor thread.
        with self._forces_lock:
            self._latest_touch = touch
            self._latest_left = left
            self._latest_right = right

        return json.dumps(payload)

    def latest_forces(self) -> dict | None:
        """Verification snapshot of the per-region force signals for the Motor Monitor.
        Reuses the LAST touch snapshot the haptics loop already read (NO new hardware read),
        and computes the per-sub-region raw metrics on the CALLER's thread — so the force
        sensor read rate and the teleop loop are completely untouched.

        Returns {'left': {region: {...}}, 'right': {...}} or None if nothing read yet. Per
        finger: 'intensity' (the 0..1 value that actually drives the haptics) + 'tip'/'pad' raw
        sub-region metrics. Palm: 'intensity' + 'raw'."""
        from projects.inspire_hand_teleop.inspire_fingertip_force_calibration import (
            region_raw, subregion_raw,
        )
        with self._forces_lock:
            touch = self._latest_touch
            left_i = dict(self._latest_left or {})
            right_i = dict(self._latest_right or {})
        if touch is None:
            return None
        out: dict = {}
        for side, key, intens in (("left", "Left", left_i), ("right", "Right", right_i)):
            htouch = touch.get(key, {})
            hand_out: dict = {}
            for ik, hf in _FINGER_MAP:
                arr = htouch.get(ik)
                if hf == "palm":
                    hand_out[hf] = {"intensity": round(float(intens.get(hf, 0.0)), 3),
                                    "raw": round(float(region_raw(arr, ik)), 1)}
                else:
                    hand_out[hf] = {"intensity": round(float(intens.get(hf, 0.0)), 3),
                                    "tip": round(float(subregion_raw(arr, ik, "tip")), 1),
                                    "pad": round(float(subregion_raw(arr, ik, "nail")), 1)}
            out[side] = hand_out
        return out

    def _maybe_log(self, payload: str) -> None:
        now = time.monotonic()
        data = json.loads(payload)
        left, right = data["left"], data["right"]
        any_contact = any(v > 0.01 for v in list(left.values()) + list(right.values()))
        should_log = (
            (any_contact and not self._ever_nonzero)
            or (now - self._last_log_t > 2.0)
        )
        if should_log:
            if any_contact:
                self._ever_nonzero = True
            logger.info(
                "[haptics→WebRTC]  "
                "L: th=%.2f ix=%.2f md=%.2f rg=%.2f lt=%.2f pa=%.2f  |  "
                "R: th=%.2f ix=%.2f md=%.2f rg=%.2f lt=%.2f pa=%.2f",
                left["thumb"],  left["index"],  left["middle"],
                left["ring"],   left["little"], left["palm"],
                right["thumb"], right["index"], right["middle"],
                right["ring"],  right["little"],right["palm"],
            )
            self._last_log_t = now
