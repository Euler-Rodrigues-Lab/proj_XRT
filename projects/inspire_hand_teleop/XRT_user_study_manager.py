"""
User Study Manager for G1 + Inspire Hand Teleoperation.

PyQt5-based GUI that wraps demo_g1_inspire_xr_robot_teleop_full_body_hw.py to make
conducting user studies easier. Provides buttons for:
  - Starting/stopping data collection (robot CSV + ZED video + body-pose CSV)
  - Starting/stopping teleop (stop lowers arms to sides)
  - Sending commands to Unity (audio haptics, vibrotactile haptics, viewport UI)

Threading model:
  Main thread    : Qt event loop + sim/viewer updates via 20 Hz QTimer
  ik-control     : IK solve + hardware sends at ik_rate Hz (tight loop, no Qt jitter)
  hw-cmd thread  : sends G1 hardware targets at 500 Hz
  csv thread     : writes robot-state CSV rows from a queue
  ZED recording  : managed as subprocesses (ffmpeg from RTSP)

The IK control pipeline runs entirely in the `ControlThread`, matching the raw demo's
architecture. This decouples IK latency from Qt event-loop overhead and raises the
effective IK rate from ~50 Hz (QTimer-limited) to the full ik_rate setting (default 60 Hz).

Usage:
    python XRT_user_study_manager.py [--net enp132s0] [--dry_run] [--no_haptics] [--kneeling]
    (all demo_g1_inspire_xr_robot_teleop_full_body_hw.py flags are supported)
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import os
import queue
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

# Use MuJoCo's standard GUI backend. Avoid forcing GLX vendor overrides here,
# because that can break on some desktop sessions even when rendering works.
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.pop("__NV_PRIME_RENDER_OFFLOAD", None)
os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)

import mujoco
import mujoco.viewer
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QProcess
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPolygonF, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QSpinBox, QTextEdit,
    QSplitter, QFrame, QStatusBar, QGridLayout, QScrollArea,
    QComboBox, QRadioButton, QButtonGroup, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QListWidget, QListWidgetItem, QAbstractItemView, QMessageBox, QInputDialog,
    QSizePolicy, QDialog, QDialogButtonBox,
)

# ---------------------------------------------------------------------------
# Path setup (mirrors the demo script)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent


def _nice_child():
    """subprocess preexec_fn: de-prioritise a heavy DATA-COLLECTION child (crate
    AprilTag tracking, room transcription) so it yields CPU cores to THIS process.
    The manager's WebRTC asyncio loop is I/O-bound and, under a busy machine, is the
    first thread the OS starves — a starved loop can't drain the reliable body_pose SCTP
    channel, which shows up as growing teleop lag right after Prepare Study / Start
    Session spin these subprocesses up. nice(10) keeps them from competing head-to-head
    with the teleop path. Best-effort: never let a nice() failure block the launch."""
    try:
        os.nice(10)
    except Exception:
        pass


_XML = _HERE / "g1_with_inspire_hand" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.xml"
_URDF = _HERE / "g1_with_inspire_hand" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
_REPO_ROOT = _HERE.parent.parent
_MEDIAMTX = _REPO_ROOT / "workspaces" / "mediamtx" / "mediamtx"
_MEDIAMTX_CFG = _REPO_ROOT / "workspaces" / "mediamtx" / "mediamtx.yml"
from xrt_devices.video import zed_stream_script

_ZED_SCRIPT = zed_stream_script()

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "g1_upper_body_teleop"))  # exposes common.common.*

# End-effector wrench estimation (shared with the offline wrench_playback.py tool).
import wrench_estimation as we

# ---------------------------------------------------------------------------
# Kneeling pose constants (identical to the demo)
# ---------------------------------------------------------------------------
_HIP_PITCH_KNEEL = -0.2
_ANKLE_PITCH_KNEEL = 0.4
_KNEELING_ANGLES = np.array([
    _HIP_PITCH_KNEEL, 0.0, 0.0, np.pi / 2 - _HIP_PITCH_KNEEL, _ANKLE_PITCH_KNEEL, 0.0,
    _HIP_PITCH_KNEEL, 0.0, 0.0, np.pi / 2 - _HIP_PITCH_KNEEL, _ANKLE_PITCH_KNEEL, 0.0,
])

# Indices into the 12-element leg ordering (leg_joint2motor_idx / _KNEELING_ANGLES,
# per leg [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]) selecting
# the joints recorded "down to the knees": hip(3) + knee per leg, 8 total (no ankles).
_LEG_KNEE_SEL = [0, 1, 2, 3, 6, 7, 8, 9]

_ARMS_AT_SIDE_LEFT  = np.array([0.0, 0.0, 0.0, np.pi / 2 - np.deg2rad(25), 0.0, 0.0, 0.0])
_ARMS_AT_SIDE_RIGHT = np.array([0.0, 0.0, 0.0, np.pi / 2 - np.deg2rad(25), 0.0, 0.0, 0.0])

# Staged startup arm choreography. To raise the arms to the ready pose without
# the hands sweeping through the crate or table in front, the arms move BACK,
# fold UP into a "chicken wing", then swing FORWARD and inward:
#   1. LATERAL RAISE — abduct the shoulders out to the side (still down)
#   2. BACK          — swing the abducted arms backwards (behind the robot)
#   3. CHICKEN WING  — fold the elbows UP while back (hands tuck up and back)
#   4. FORWARD       — swing the arms forward to the ready pose: held OUTWARD
#                      (shoulders abducted, hands apart) with the FOREARMS
#                      PARALLEL TO THE FLOOR, staying high/clear of the crate/table
#
# Elbow folds in the NEGATIVE direction (forearm rotates UP, toward the head) —
# folding up keeps the hands high while the arms come around. The elbow joint
# range is [-60, +120] deg, so the chicken-wing fold is capped a touch above the
# -60 limit. Before FORWARD the hands never go forward of their start; the FORWARD
# sweep stays high. FK-verified.
#
# Arm DOF indices: 0 shoulder pitch, 1 shoulder roll, 2 shoulder yaw, 3 elbow.
# +pitch swings the upper arm behind (FK: +X is forward — the foot toe is at +X
# — and +pitch moves the hand toward -X / backward). Abduction is +roll on the
# LEFT arm, -roll on the RIGHT (shoulder yaw is left at 0 — internal rotation
# would turn the hands inward and make them touch).
_ARM_RETRACT_ABDUCT = np.deg2rad(30)   # lateral raise — shoulder roll out
_ARM_RETRACT_PITCH  = np.deg2rad(45)   # how far back the arms swing
_ELBOW_REST    = float(_ARMS_AT_SIDE_LEFT[3])   # elbow during raise+back (rest bend ~65 deg)
_ELBOW_CHICKEN = np.deg2rad(-55)       # chicken-wing fold (UP; -60 is the joint limit)
# Final ready pose: held outward, forearms parallel to the floor (hands apart).
_FRONT_PITCH  = np.deg2rad(15)         # shoulder pitch for a roughly level forearm
_FRONT_ABDUCT = np.deg2rad(58)         # shoulder roll out (wide ~23in gap between the hands)
_FRONT_ELBOW  = np.deg2rad(-12)        # slight forearm extension to widen the reach (still ~level)


def _stage_arm(pitch: float, roll: float, yaw: float, elbow: float) -> np.ndarray:
    """Build a 7-DOF staged-choreography arm pose (wrists left at 0)."""
    a = np.zeros(7)
    a[0] = pitch
    a[1] = roll
    a[2] = yaw
    a[3] = elbow
    return a


def _smoothstep(t: float) -> float:
    """Ease in/out: 0 velocity at the endpoints. t clamped to [0, 1]."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)


def _path_pose(waypoints, s: float):
    """Piecewise-LINEAR interpolation through a list of pose vectors at the
    normalized parameter s in [0, 1]. Pair with a smoothstep-eased s to get a
    motion that eases in/out at the ends but flows through the intermediate
    waypoints without stopping — and never overshoots the waypoint path (so it
    stays collision-safe), unlike a spline."""
    n = len(waypoints)
    if n == 1:
        return np.array(waypoints[0], dtype=np.float64).copy()
    s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
    u = s * (n - 1)
    i = int(u)
    if i >= n - 1:
        return np.array(waypoints[-1], dtype=np.float64).copy()
    t = u - i
    return (1.0 - t) * np.asarray(waypoints[i], dtype=np.float64) + t * np.asarray(waypoints[i + 1], dtype=np.float64)


_ABD = _ARM_RETRACT_ABDUCT
# Lateral-raise (abducted, down) and back poses keep the elbow at the rest bend.
_ARMS_ABDUCT_LEFT   = _stage_arm(0.0, +_ABD, 0.0, _ELBOW_REST)
_ARMS_ABDUCT_RIGHT  = _stage_arm(0.0, -_ABD, 0.0, _ELBOW_REST)
_ARMS_BEHIND_LEFT   = _stage_arm(_ARM_RETRACT_PITCH, +_ABD, 0.0, _ELBOW_REST)
_ARMS_BEHIND_RIGHT  = _stage_arm(_ARM_RETRACT_PITCH, -_ABD, 0.0, _ELBOW_REST)
# Chicken-wing: still back/abducted, elbows folded UP.
_ARMS_CHICKEN_LEFT  = _stage_arm(_ARM_RETRACT_PITCH, +_ABD, 0.0, _ELBOW_CHICKEN)
_ARMS_CHICKEN_RIGHT = _stage_arm(_ARM_RETRACT_PITCH, -_ABD, 0.0, _ELBOW_CHICKEN)
# Held-outward, forearms-parallel pose (hands out in front, slightly apart).
# Retained for reference; not currently used — the manager bootup now just raises
# 45 deg and damps, and the teleop entry sweeps to the operator's LIVE pose.
_ARMS_FRONT_LEFT    = _stage_arm(_FRONT_PITCH, +_FRONT_ABDUCT, 0.0, _FRONT_ELBOW)
_ARMS_FRONT_RIGHT   = _stage_arm(_FRONT_PITCH, -_FRONT_ABDUCT, 0.0, _FRONT_ELBOW)

# Manager-bootup pose: arms laterally raised outward (shoulders abducted) and
# nothing else — the robot raises to this, then drops into damp. Separate from
# the teleop-entry abduct so bootup can be wider without changing the choreography.
_BOOTUP_ABDUCT = np.deg2rad(45)
_ARMS_BOOTUP_LEFT   = _stage_arm(0.0, +_BOOTUP_ABDUCT, 0.0, _ELBOW_REST)
_ARMS_BOOTUP_RIGHT  = _stage_arm(0.0, -_BOOTUP_ABDUCT, 0.0, _ELBOW_REST)

# Safe-Shutdown 'A-pose' hold: arms raised OUT to the sides, DROOPED at the shoulders
# (abducted ~55 deg, below a horizontal T-pose), with the elbows FULLY EXTENDED so the arms
# are STRAIGHT — a true A-pose, no elbow bend — plus a slight backward lean. Straight arms
# keep the hands out to the sides and clear of the table as the torso bows forward during
# the hip-lowering descent (any elbow bend let the forearm/hand hang down-and-in toward the
# table). NOTE: on the G1 the elbow joint's ZERO is bent ~76 deg — the arm is actually
# STRAIGHT (collinear, max reach) at elbow_q ≈ +76 deg, NOT 0, so that's the value used.
# FK-verified on the g1+inspire model: the straight-arm hands land at x≈-0.03 (BEHIND the
# shoulder plane, well off the front table), y≈±0.44 (~0.9 m apart, clear of the chest),
# z≈0.88, and stay behind/out through the ~25 deg torso bow. Reached in ONE direct move
# (never more forward than the start), held through the descent, then dropped.
_SD_A_PITCH  = np.deg2rad(15)    # slight backward lean (compensates the forward torso bow
                                 # during the descent so the hands stay off the front table)
_SD_A_ABDUCT = np.deg2rad(55)    # drooped abduction — the 'A' droop, below the horizontal T
_SD_A_ELBOW  = np.deg2rad(76)    # FULLY EXTENDED / STRAIGHT arm (the G1 elbow is straight
                                 # at ~+76 deg, not 0 — q=0 is a ~76 deg bend). Raise
                                 # _SD_A_ABDUCT toward horizontal to lift the hands higher.
_ARMS_SHUTDOWN_A_LEFT  = _stage_arm(_SD_A_PITCH, +_SD_A_ABDUCT, 0.0, _SD_A_ELBOW)
_ARMS_SHUTDOWN_A_RIGHT = _stage_arm(_SD_A_PITCH, -_SD_A_ABDUCT, 0.0, _SD_A_ELBOW)

# Fraction of the teleop-entry transition spent on the staged retract (lateral
# raise -> back -> chicken wing); the remainder sweeps from the wing into the
# operator's live pose.
_RETRACT_FRAC = 0.55


# ============================================================================
# LatencyTracker — thread-safe circular buffer for latency samples
# ============================================================================

# ── End-to-end latency: estimated constants for stages not measurable in software ──────────
# The variable, dominant stages (network, jitter buffer, decode, manager pose-wait + IK, and robot
# actuation) are MEASURED. These constants fill the fixed endpoints so the reported totals are
# complete; tune them to your hardware. Override any at launch with the matching --lat-* CLI flag.
_LAT_EST_CAPTURE_ENCODE_MS = 12.0   # ZED capture + NVENC low-latency encode (sender side)
_LAT_EST_DISPLAY_PHOTON_MS = 14.0   # Quest decode→photons: display refresh + pixel persistence
_LAT_EST_TRACKING_MS       = 10.0   # body-tracking sensor → pose sample on the Quest
_LAT_EST_ACTUATION_MS      = 30.0   # fallback robot actuation if the motor-feedback estimate is idle


class ActuationLatencyEstimator:
    """Estimate robot actuation latency (ms) from G1 motor feedback.

    During motion the actual joint trajectory lags the commanded one by the control+mechanical
    latency. We buffer commanded vs actual arm-joint vectors over a sliding window and, for the
    most-active joint, find the time-shift (0..max_lag) that maximises the normalised correlation
    between actual(t) and commanded(t-shift). That shift is the actuation latency. Returns None
    until there is enough motion and a confident correlation (so a still robot doesn't report 0)."""

    def __init__(self, dt: float, window_s: float = 2.0, max_lag_s: float = 0.3,
                 min_motion_rad: float = 0.03, min_corr: float = 0.5):
        self._dt = dt
        self._n = max(8, int(window_s / dt))
        self._max_lag = max(1, int(max_lag_s / dt))
        self._min_motion = min_motion_rad
        self._min_corr = min_corr
        self._cmd: deque = deque(maxlen=self._n)
        self._act: deque = deque(maxlen=self._n)
        self._lock = threading.Lock()

    def add(self, cmd_vec, act_vec) -> None:
        with self._lock:
            self._cmd.append(np.asarray(cmd_vec, dtype=np.float64))
            self._act.append(np.asarray(act_vec, dtype=np.float64))

    def estimate_ms(self):
        with self._lock:
            if len(self._cmd) < self._n:
                return None
            cmd = np.array(self._cmd)   # (N, J)
            act = np.array(self._act)
        if cmd.ndim != 2 or cmd.shape != act.shape:
            return None
        span = cmd.max(axis=0) - cmd.min(axis=0)
        j = int(np.argmax(span))
        if span[j] < self._min_motion:
            return None   # robot/operator essentially still — can't measure a lag
        c = cmd[:, j] - cmd[:, j].mean()
        a = act[:, j] - act[:, j].mean()
        best_lag, best_corr = 0, -np.inf
        for lag in range(0, self._max_lag + 1):
            av = a[lag:]
            cv = c[:len(c) - lag] if lag > 0 else c
            denom = np.linalg.norm(av) * np.linalg.norm(cv)
            if denom < 1e-9:
                continue
            corr = float(np.dot(av, cv) / denom)   # actual(t) vs commanded(t-lag)
            if corr > best_corr:
                best_corr, best_lag = corr, lag
        if best_corr < self._min_corr:
            return None   # weak alignment — don't report a noisy number
        return best_lag * self._dt * 1000.0


class EndToEndLatency:
    """Thread-safe accumulator of assembled end-to-end latency samples for live logging and
    per-trial reporting. Holds video (glass-to-glass) and teleop (motion-to-motion) totals plus
    the most recent per-stage breakdown for each."""

    RETAIN = 600.0   # seconds of samples kept (covers a long trial for the summary)

    def __init__(self):
        self._lock = threading.Lock()
        self._video: deque = deque()    # (t, total_ms)
        self._teleop: deque = deque()   # (t, total_ms)
        self._video_bd = None           # latest breakdown dict
        self._teleop_bd = None

    def add_video(self, total_ms: float, breakdown: dict) -> None:
        self._push(self._video, total_ms)
        with self._lock:
            self._video_bd = breakdown

    def add_teleop(self, total_ms: float, breakdown: dict) -> None:
        self._push(self._teleop, total_ms)
        with self._lock:
            self._teleop_bd = breakdown

    def _push(self, q: deque, total_ms: float) -> None:
        now = time.monotonic()
        with self._lock:
            q.append((now, total_ms))
            while q and q[0][0] < now - self.RETAIN:
                q.popleft()

    def latest(self):
        """Return (video_total_ms, video_bd, teleop_total_ms, teleop_bd); each may be None."""
        with self._lock:
            v = self._video[-1][1] if self._video else None
            t = self._teleop[-1][1] if self._teleop else None
            return v, dict(self._video_bd) if self._video_bd else None, \
                   t, dict(self._teleop_bd) if self._teleop_bd else None

    def summary(self, window_s: float | None = None) -> dict:
        """Mean / p95 / max / count of both totals over the last ``window_s`` (default all retained)."""
        now = time.monotonic()
        cutoff = now - (self.RETAIN if window_s is None else window_s)

        def _stats(q):
            vals = [v for t, v in q if t >= cutoff]
            if not vals:
                return None
            arr = np.array(vals)
            return {
                "mean_ms": float(arr.mean()),
                "p95_ms": float(np.percentile(arr, 95)),
                "max_ms": float(arr.max()),
                "n": int(arr.size),
            }

        with self._lock:
            return {
                "video_glass_to_glass": _stats(self._video),
                "teleop_motion_to_motion": _stats(self._teleop),
                "video_breakdown_last": dict(self._video_bd) if self._video_bd else None,
                "teleop_breakdown_last": dict(self._teleop_bd) if self._teleop_bd else None,
            }


class LatencyTracker:
    """Thread-safe sliding-window store for latency measurements.

    Three series are tracked:
      ik_period   — wall time of each ControlThread tick (IK + hardware sends), ms
      ik_compute  — time from get_controller_state() to hw_cmd_thread.set_goals(), ms
      pose_interval — time between successive body-pose packets from the Quest, ms
    """

    WINDOW = 10.0    # seconds shown in the scrolling graph
    HISTORY = 30.0   # seconds of samples retained (for the 30 s peak hold)

    def __init__(self):
        self._lock = threading.Lock()
        self._ik_period: deque   = deque()
        self._ik_compute: deque  = deque()
        self._pose_ivl: deque    = deque()

    # ── public write API (called from ControlThread) ─────────────────────────

    def add_ik_period(self, ms: float):
        self._push(self._ik_period, ms)

    def add_ik_compute(self, ms: float):
        self._push(self._ik_compute, ms)

    def add_pose_interval(self, ms: float):
        self._push(self._pose_ivl, ms)

    # ── public read API (called from GUI thread) ──────────────────────────────

    def get_all(self):
        """Return (ik_period_samples, ik_compute_samples, pose_interval_samples, now_monotonic)."""
        now = time.monotonic()
        cutoff = now - self.WINDOW
        with self._lock:
            return (
                [(t, v) for t, v in self._ik_period  if t >= cutoff],
                [(t, v) for t, v in self._ik_compute  if t >= cutoff],
                [(t, v) for t, v in self._pose_ivl    if t >= cutoff],
                now,
            )

    def latest(self):
        """Return (latest_ik_period_ms, latest_ik_compute_ms, latest_pose_ivl_ms)."""
        with self._lock:
            p = self._ik_period[-1][1]  if self._ik_period  else None
            c = self._ik_compute[-1][1] if self._ik_compute  else None
            i = self._pose_ivl[-1][1]   if self._pose_ivl    else None
        return p, c, i

    def peaks(self, window: float | None = None):
        """Return (peak_ik_period, peak_ik_compute, peak_pose_ivl) over the last
        ``window`` seconds (default = full retained HISTORY). Each is the max ms
        in that span, or None if no samples. Used by the 30 s UI peak-hold and
        the 10 s recording summary."""
        now = time.monotonic()
        cutoff = now - (self.HISTORY if window is None else window)
        with self._lock:
            def _pk(q):
                vals = [v for t, v in q if t >= cutoff]
                return max(vals) if vals else None
            return _pk(self._ik_period), _pk(self._ik_compute), _pk(self._pose_ivl)

    # ── internal ─────────────────────────────────────────────────────────────

    def _push(self, q: deque, ms: float):
        now = time.monotonic()
        cutoff = now - self.HISTORY
        with self._lock:
            q.append((now, ms))
            while q and q[0][0] < cutoff:
                q.popleft()


# ============================================================================
# LatencyGraphWidget — SteamVR-style scrolling frame-timing display
# ============================================================================

class LatencyGraphWidget(QWidget):
    """Scrolling line graph of IK pipeline latency.

    Three series:
      Blue   — IK tick period (total wall time per ControlThread iteration)
      Red    — IK compute time (from pose read → hw goals set)
      Orange — Body-pose packet receive interval (Quest → PC pipeline health)

    A dashed green line marks the target IK period (1000 / ik_rate ms).
    """

    _SERIES = [
        ("IK period",   QColor(64,  144, 255)),   # blue
        ("IK compute",  QColor(255,  80,  80)),   # red
        ("Pose interval", QColor(255, 160,   0)), # orange
    ]

    def __init__(self, tracker: LatencyTracker, ik_rate: int = 60, parent=None):
        super().__init__(parent)
        self._tracker = tracker
        self._ik_rate = ik_rate
        self.setMinimumHeight(140)
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self.update)
        self._update_timer.start(80)  # 12.5 Hz repaint

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H = self.width(), self.height()
        ML, MR, MT, MB = 44, 8, 8, 22
        PW = max(W - ML - MR, 1)
        PH = max(H - MT - MB, 1)

        # Background
        p.fillRect(0, 0, W, H, QColor(18, 18, 18))
        p.fillRect(ML, MT, PW, PH, QColor(26, 26, 26))

        ik_periods, ik_computes, pose_ivls, now = self._tracker.get_all()
        window = LatencyTracker.WINDOW

        # Auto y_max: pick a round ceiling that fits all data
        all_vals = [v for _, v in ik_periods + ik_computes + pose_ivls]
        if all_vals:
            raw_max = max(all_vals)
            y_max = max(50.0, min(200.0, raw_max * 1.15))
        else:
            y_max = 50.0

        target_ms = 1000.0 / max(1, self._ik_rate)

        def yt(ms):
            """Y pixel for a latency value."""
            return MT + PH - (min(ms, y_max) / y_max) * PH

        def xt(t):
            """X pixel for a monotonic timestamp."""
            return ML + (1.0 - (now - t) / window) * PW

        # Grid lines
        grid_pen = QPen(QColor(42, 42, 42), 1)
        p.setPen(grid_pen)
        for ms in [10, 20, 33, 50, 100]:
            if ms > y_max:
                break
            y = yt(ms)
            p.drawLine(ML, int(y), ML + PW, int(y))

        # Target period line (dashed green)
        ty = yt(target_ms)
        p.setPen(QPen(QColor(60, 160, 60), 1, Qt.DashLine))
        p.drawLine(ML, int(ty), ML + PW, int(ty))

        # Data series
        all_series = [ik_periods, ik_computes, pose_ivls]
        for samples, (_, color) in zip(all_series, self._SERIES):
            if len(samples) < 2:
                continue
            pen = QPen(color, 1.5)
            p.setPen(pen)
            pts = [QPointF(xt(t), yt(ms)) for t, ms in samples]
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])

        # 30 s peak-hold: a faint dashed line per series at its peak over the
        # retained history, so the worst-case spike stays visible after it
        # scrolls off. Peak values are also printed in the legend below.
        peaks = self._tracker.peaks(LatencyTracker.HISTORY)
        for pk, (_, color) in zip(peaks, self._SERIES):
            if pk is None:
                continue
            py = yt(pk)
            pen = QPen(color, 1, Qt.DotLine)
            p.setPen(pen)
            p.drawLine(ML, int(py), ML + PW, int(py))

        # Y-axis labels
        lbl_font = QFont("Monospace", 8)
        p.setFont(lbl_font)
        p.setPen(QColor(160, 160, 160))
        for ms in [0, 10, 20, 33]:
            if ms > y_max:
                break
            y = yt(ms)
            p.drawText(2, int(y) + 4, f"{ms}ms")

        # Legend (upper-right area, inside plot) — each entry shows the 30 s peak.
        p.setFont(QFont("Monospace", 8))
        lx = ML + PW - 165
        ly = MT + 12
        for (label, color), pk in zip(self._SERIES, peaks):
            p.setPen(color)
            pk_str = f"pk {pk:.0f}ms" if pk is not None else "pk —"
            p.drawText(lx, ly, f"▬ {label}  ({pk_str}/30s)")
            ly += 13

        # Current-value readout (bottom strip)
        lat_p, lat_c, lat_i = self._tracker.latest()
        p_str = f"{lat_p:.1f}" if lat_p is not None else "—"
        c_str = f"{lat_c:.1f}" if lat_c is not None else "—"
        i_str = f"{lat_i:.1f}" if lat_i is not None else "—"
        readout = f"Period {p_str}ms  Compute {c_str}ms  Pose recv {i_str}ms   (target {target_ms:.1f}ms @ {self._ik_rate}Hz)"
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Monospace", 8))
        p.drawText(ML + 4, MT + PH + MB - 4, readout)

        p.end()


# ============================================================================
# Motor thermal / torque monitor (arms, legs, fingers)
# ============================================================================

# Short joint names in the order the Unitree controller exposes them.
# arm_waist_joint2motor_idx → torso[3], left arm[7], right arm[7].
_TORSO_JOINT_NAMES = ["Waist Yaw", "Waist Roll", "Waist Pitch"]
_ARM_JOINT_NAMES = [
    "Shoulder Pitch", "Shoulder Roll", "Shoulder Yaw",
    "Elbow",
    "Wrist Roll", "Wrist Pitch", "Wrist Yaw",
]
# leg_joint2motor_idx → left leg[6], right leg[6].
_LEG_JOINT_NAMES = ["Hip Pitch", "Hip Roll", "Hip Yaw", "Knee", "Ankle Pitch", "Ankle Roll"]
# Inspire hand DOF order (matches InspireHand snapshot lists).
_FINGER_NAMES = ["Pinky", "Ring", "Middle", "Index", "Thumb Bend", "Thumb Rot"]

# Human-readable labels for the six per-hand actuator DOFs (SAME index order as _FINGER_NAMES),
# used by the Hand DOFs enable/disable dialog. Index 5 = thumb rotation = abduction/adduction.
_HAND_DOF_LABELS = [
    "Pinky (flexion)",
    "Ring (flexion)",
    "Middle (flexion)",
    "Index (flexion)",
    "Thumb bend (flexion / extension)",
    "Thumb rotation (abduction / adduction)",
]


def _hg_temp(t) -> float:
    """Coerce a Unitree HG MotorState.temperature (int16[2]: motor, driver) to
    the hotter scalar. Go/older firmware or finger snapshots give a scalar."""
    try:
        return float(max(t))
    except TypeError:
        return float(t)


def _lbl_text(lbl, text):
    """setText ONLY when the text changed (cache on the widget). The 5 Hz status
    refresh calls setText/setStyleSheet on dozens of labels with mostly-unchanged
    values; unconditional Qt updates on the GUI thread hold the GIL and were the
    profiled root cause of the multi-second teleop lag."""
    if getattr(lbl, "_last_text", None) != text:
        lbl._last_text = text
        lbl.setText(text)


def _lbl_style(lbl, style):
    """setStyleSheet ONLY when the style changed (see _lbl_text). setStyleSheet is
    the expensive one: every call forces a full style recompute + repaint."""
    if getattr(lbl, "_last_style", None) != style:
        lbl._last_style = style
        lbl.setStyleSheet(style)


class MotorBox(QFrame):
    """A single actuator tile: name, a large main readout and a small secondary
    readout. The background fades from white toward a target color as a chosen
    metric rises (temperature→red for body motors, current→blue for fingers),
    so a hot or hard-working actuator is obvious at a glance.
    """

    def __init__(self, name: str, motor_id: int | None = None):
        super().__init__()
        self._base_name = name
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self.setFixedSize(118, 60)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(0)

        label = name if motor_id is None else f"{name}  (#{motor_id})"
        self._name_lbl = QLabel(label)
        self._name_lbl.setStyleSheet("background: transparent; color: #222; font-size: 9px; font-weight: bold;")
        self._name_lbl.setAlignment(Qt.AlignCenter)

        self._main_lbl = QLabel("—")
        self._main_lbl.setStyleSheet("background: transparent; color: #000; font-size: 13px; font-weight: bold;")
        self._main_lbl.setAlignment(Qt.AlignCenter)

        self._sec_lbl = QLabel("—")
        self._sec_lbl.setStyleSheet("background: transparent; color: #333; font-size: 10px;")
        self._sec_lbl.setAlignment(Qt.AlignCenter)

        lay.addWidget(self._name_lbl)
        lay.addWidget(self._main_lbl)
        lay.addWidget(self._sec_lbl)
        self._set_bg(QColor(255, 255, 255))

    def set_id(self, motor_id: int):
        self._name_lbl.setText(f"{self._base_name}  (#{motor_id})")

    def _set_bg(self, color: QColor):
        # setStyleSheet is EXPENSIVE (full style recompute + repaint on the GUI thread,
        # which holds the GIL) — skip when the color didn't change. With ~40 tiles at
        # 5 Hz the unconditional version alone was ~200 restyles/s; profiled as the #1
        # GIL consumer during the teleop-lag incident.
        name = color.name()
        if getattr(self, "_last_bg", None) == name:
            return
        self._last_bg = name
        self.setStyleSheet(
            f"MotorBox {{ background-color: {name}; border: 1px solid #888; border-radius: 3px; }}"
        )

    def update_box(self, main_text: str, sec_text: str, frac: float,
                   target=(255, 0, 0), warn: bool = False):
        """Refresh the tile — but ONLY touch Qt for parts whose rendered value actually
        changed (change-detection: an idle tile costs zero; see _set_bg note).

        frac   : 0..1 — how far the colored metric is toward its limit.
        target : RGB the background fades to at frac=1 (white at frac=0).
        warn   : if True, render the secondary value in red (e.g. near a limit).
        """
        frac = max(0.0, min(1.0, frac))
        # Quantize the fade so imperceptible frac jitter can't force a restyle.
        frac = round(frac * 50.0) / 50.0
        r = int(round(255 + (target[0] - 255) * frac))
        g = int(round(255 + (target[1] - 255) * frac))
        b = int(round(255 + (target[2] - 255) * frac))
        self._set_bg(QColor(r, g, b))

        dark = frac > 0.55  # background dark enough to need light text
        if (dark, warn) != getattr(self, "_last_style_state", None):
            self._last_style_state = (dark, warn)
            self._main_lbl.setStyleSheet(
                "background: transparent; font-size: 13px; font-weight: bold; "
                f"color: {'#fff' if dark else '#000'};"
            )
            self._name_lbl.setStyleSheet(
                "background: transparent; font-size: 9px; font-weight: bold; "
                f"color: {'#fff' if dark else '#222'};"
            )
            if dark:
                sec_color = "#fff"
            elif warn:
                sec_color = "#c0392b"
            else:
                sec_color = "#333"
            self._sec_lbl.setStyleSheet(
                f"background: transparent; font-size: 10px; color: {sec_color};"
                f"{' font-weight: bold;' if warn else ''}"
            )
        if main_text != getattr(self, "_last_main_text", None):
            self._last_main_text = main_text
            self._main_lbl.setText(main_text)
        if sec_text != getattr(self, "_last_sec_text", None):
            self._last_sec_text = sec_text
            self._sec_lbl.setText(sec_text)


class MotorMonitor(QWidget):
    """Separate window laying out the G1 actuators so an operator can watch
    temperature/load per joint and stop before anything overheats. Arms hang
    down each side, the Inspire-hand fingers sit in the open center, and the
    legs run along the bottom. Every box fades white→red with temperature."""

    def __init__(self, temp_min: float, temp_max: float, torque_max: float,
                 finger_temp_max: float, finger_current_max: float):
        super().__init__()
        self._temp_min = temp_min
        self._temp_max = temp_max
        self._torque_max = torque_max
        self._finger_temp_max = finger_temp_max
        self._finger_current_max = finger_current_max
        self.setWindowTitle("G1 Motor Monitor — Temp / Torque")

        root = QVBoxLayout(self)

        legend = QLabel(
            f"Body motors: box white ≤ {temp_min:.0f} °C (room) → red = {temp_max:.0f} °C; "
            f"torque flags red above {0.85 * torque_max:.0f} Nm (limit {torque_max:.0f}).  "
            f"Fingers: box white = 0 mA → blue = {finger_current_max:.0f} mA (current); "
            f"temperature text flags red above {0.85 * finger_temp_max:.0f} °C."
        )
        legend.setStyleSheet("color: #444; font-size: 11px;")
        legend.setAlignment(Qt.AlignCenter)
        legend.setWordWrap(True)
        root.addWidget(legend)

        grid = QGridLayout()
        grid.setSpacing(6)
        root.addLayout(grid)

        # Column headers + torso row across the top, arms hanging down each side.
        left_hdr = QLabel("LEFT ARM")
        left_hdr.setStyleSheet("font-weight: bold; color: #2c3e50;")
        left_hdr.setAlignment(Qt.AlignCenter)
        right_hdr = QLabel("RIGHT ARM")
        right_hdr.setStyleSheet("font-weight: bold; color: #2c3e50;")
        right_hdr.setAlignment(Qt.AlignCenter)
        grid.addWidget(left_hdr, 0, 0)
        grid.addWidget(right_hdr, 0, 2)

        torso_box = QGroupBox("Waist / Torso")
        torso_row = QHBoxLayout(torso_box)
        torso_row.setSpacing(6)
        self._torso_boxes = []
        for name in _TORSO_JOINT_NAMES:
            b = MotorBox(name)
            self._torso_boxes.append(b)
            torso_row.addWidget(b)
        grid.addWidget(torso_box, 0, 1)

        self._left_boxes = []
        self._right_boxes = []
        for row, name in enumerate(_ARM_JOINT_NAMES):
            lb = MotorBox(name)
            rb = MotorBox(name)
            self._left_boxes.append(lb)
            self._right_boxes.append(rb)
            grid.addWidget(lb, row + 1, 0)
            grid.addWidget(rb, row + 1, 2)

        # Fingers in the open center area, beneath the torso row.
        fingers_box = QGroupBox("Fingers (Inspire Hands)")
        fingers_row = QHBoxLayout(fingers_box)
        fingers_row.setSpacing(10)
        self._finger_left_boxes = self._build_finger_column(fingers_row, "LEFT HAND")
        self._finger_right_boxes = self._build_finger_column(fingers_row, "RIGHT HAND")
        grid.addWidget(fingers_box, 1, 1, len(_ARM_JOINT_NAMES), 1)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        # Legs underneath everything — stacked (left leg above right leg) so the
        # window stays narrow instead of spreading 12 tiles across one row.
        legs_box = QGroupBox("Legs")
        legs_col = QVBoxLayout(legs_box)
        legs_col.setSpacing(6)
        self._leg_left_boxes = self._build_leg_group(legs_col, "LEFT LEG")
        self._leg_right_boxes = self._build_leg_group(legs_col, "RIGHT LEG")
        root.addWidget(legs_box)

        # Fingertip / fingerpad / palm FORCE sensors — the same signals that drive the
        # haptics, shown here purely to VERIFY the sensors are working. Fed from the haptics
        # streamer's cached snapshot (no extra hardware read), so this never slows the sensor
        # read rate or the teleop loop. Monospace so columns line up; hidden until data.
        force_box = QGroupBox("Hand Force Sensors  (haptics verification — tip / pad / palm)")
        fcol = QVBoxLayout(force_box)
        fhdr = QLabel("Per finger: tip &amp; pad raw + haptics intensity (0–1). "
                      "A resting finger should read ~0 and rise when you press it.")
        fhdr.setStyleSheet("color: #444; font-size: 10px;")
        fhdr.setWordWrap(True)
        fcol.addWidget(fhdr)
        self._force_lbl_left = QLabel("LEFT: (waiting for haptics stream…)")
        self._force_lbl_right = QLabel("RIGHT: (waiting for haptics stream…)")
        for _l in (self._force_lbl_left, self._force_lbl_right):
            _l.setStyleSheet("font-family: monospace; font-size: 11px;")
            _l.setTextFormat(Qt.PlainText)
            fcol.addWidget(_l)
        root.addWidget(force_box)

        root.addStretch()
        self._ids_assigned = False

    @staticmethod
    def _build_finger_column(parent_row: QHBoxLayout, title: str) -> list:
        col = QVBoxLayout()
        hdr = QLabel(title)
        hdr.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10px;")
        hdr.setAlignment(Qt.AlignCenter)
        col.addWidget(hdr)
        boxes = []
        for name in _FINGER_NAMES:
            b = MotorBox(name)
            boxes.append(b)
            col.addWidget(b)
        parent_row.addLayout(col)
        return boxes

    @staticmethod
    def _build_leg_group(parent_layout, title: str) -> list:
        grp = QGroupBox(title)
        row = QHBoxLayout(grp)
        row.setSpacing(6)
        boxes = []
        for name in _LEG_JOINT_NAMES:
            b = MotorBox(name)
            boxes.append(b)
            row.addWidget(b)
        parent_layout.addWidget(grp)
        return boxes

    def _assign_ids(self, torso_idx, left_idx, right_idx, leg_left_idx, leg_right_idx):
        """Relabel body-motor boxes with their real Unitree motor indices (once)."""
        for boxes, idxs in (
            (self._torso_boxes, torso_idx),
            (self._left_boxes, left_idx),
            (self._right_boxes, right_idx),
            (self._leg_left_boxes, leg_left_idx),
            (self._leg_right_boxes, leg_right_idx),
        ):
            for b, mid in zip(boxes, idxs):
                b.set_id(int(mid))
        self._ids_assigned = True

    def update_from_controller(self, hw_controller, hand_controller=None, forces=None):
        """Refresh all tiles. Body data comes from the Unitree LowState (no extra
        I/O — the controller already subscribes to it); finger data from the
        InspireHand's cached snapshot. ``forces`` is the haptics streamer's cached
        verification snapshot (streamer.latest_forces()); None leaves the readout as-is."""
        if hw_controller is not None:
            self._update_body(hw_controller)
        if hand_controller is not None:
            self._update_fingers(hand_controller)
        if forces is not None:
            self._update_forces(forces)

    _FORCE_ORDER = ("thumb", "index", "middle", "ring", "little")

    def _update_forces(self, forces):
        """Render the fingertip/fingerpad/palm force verification readout. Pure formatting
        of the streamer's cached values — no hardware access."""
        def _fmt(hand: dict) -> str:
            parts = []
            for hf in self._FORCE_ORDER:
                d = hand.get(hf)
                if not d:
                    continue
                parts.append(f"{hf[:3]} tip{d.get('tip', 0):>5.0f}/pad{d.get('pad', 0):>5.0f}"
                             f" i{d.get('intensity', 0):.2f}")
            p = hand.get("palm")
            if p:
                parts.append(f"palm{p.get('raw', 0):>5.0f} i{p.get('intensity', 0):.2f}")
            return "   ".join(parts)
        try:
            self._force_lbl_left.setText("LEFT   " + _fmt(forces.get("left", {})))
            self._force_lbl_right.setText("RIGHT  " + _fmt(forces.get("right", {})))
        except Exception:
            pass

    def _update_body(self, hw_controller):
        try:
            ls = hw_controller.low_state
            torso_idx = list(hw_controller._torso_idx)
            left_idx = list(hw_controller._left_idx)
            right_idx = list(hw_controller._right_idx)
            leg_idx = list(hw_controller.config.leg_joint2motor_idx)
        except Exception:
            return
        leg_left_idx, leg_right_idx = leg_idx[:6], leg_idx[6:12]
        if not self._ids_assigned:
            self._assign_ids(torso_idx, left_idx, right_idx, leg_left_idx, leg_right_idx)

        span = self._temp_max - self._temp_min
        def _apply(boxes, indices):
            for b, i in zip(boxes, indices):
                ms = ls.motor_state[i]
                temp = _hg_temp(ms.temperature)
                tau = float(ms.tau_est)
                # Box color: white at room temp -> red at temp_max.
                frac = 0.0 if span <= 0 else (temp - self._temp_min) / span
                warn = self._torque_max > 0 and abs(tau) / self._torque_max >= 0.85
                b.update_box(f"{temp:.1f} °C", f"{tau:+.1f} Nm", frac,
                             target=(255, 0, 0), warn=warn)
        try:
            _apply(self._torso_boxes, torso_idx)
            _apply(self._left_boxes, left_idx)
            _apply(self._right_boxes, right_idx)
            _apply(self._leg_left_boxes, leg_left_idx)
            _apply(self._leg_right_boxes, leg_right_idx)
        except Exception:
            pass

    def _update_fingers(self, hand_controller):
        try:
            snap = hand_controller.read()
            states = snap.get("states", {})
        except Exception:
            return

        def _apply(boxes, side_key):
            s = states.get(side_key, {})
            temps = s.get("TEMP", [])
            currents = s.get("CURRENT", [])
            for j, b in enumerate(boxes):
                t = float(temps[j]) if j < len(temps) else 0.0
                c = float(currents[j]) if j < len(currents) else 0.0
                # Box color: white at idle (0 mA) -> blue at max current.
                frac = 0.0 if self._finger_current_max <= 0 else c / self._finger_current_max
                # Secondary temperature still flags red if the finger gets hot.
                warn = self._finger_temp_max > 0 and t / self._finger_temp_max >= 0.85
                b.update_box(f"{c:.0f} mA", f"{t:.0f} °C", frac,
                             target=(30, 90, 255), warn=warn)
        try:
            _apply(self._finger_left_boxes, "Left")
            _apply(self._finger_right_boxes, "Right")
        except Exception:
            pass


# ============================================================================
# SessionRecorder — single raw-data recorder for a whole user-study session
# ============================================================================

class SessionRecorder:
    """One recorder that captures EVERYTHING raw, in one place, at the Qt tick
    rate (~50 Hz). Nothing here is derived/computed — the end-effector wrench is
    reconstructed from this raw data offline (session_viewer.py / wrench_playback.py).

    Per frame it records:
      * all G1 joints (legs/torso/arms): q, dq, ddq, tau_est, temperature, and
        the commanded goal (qgoal) — into one comprehensive CSV whose schema is
        :func:`wrench_estimation.session_fieldnames`.
      * the full coupled finger qpos (12/hand) from the sim, for visual playback.
      * the Inspire-hand register snapshot (pos/angle/force/current/error/status/temp).
      * the full tactile/touch arrays — streamed to a binary sidecar (.bin),
        which is too wide (~1300 ints/frame) for the CSV.

    A sidecar ``<csv>.meta.json`` makes the session self-describing: model path,
    palm offsets, kneeling state, known test mass, tare baseline, tactile layout,
    and record start/stop epochs (for aligning the ZED videos in the viewer).

    A background consumer thread does all file writes so the Qt tick never blocks.
    """

    def __init__(self, hw_controller, hand_controller):
        self._hw = hw_controller
        self._hand = hand_controller
        self._queue: queue.Queue = queue.Queue(maxsize=4000)
        self._file = None
        self._writer = None
        self._tactile_file = None
        self._tactile_dtype = None
        self._tactile_layout: list = []
        self._tactile_width = 0
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._active = False
        self._csv_path: str | None = None
        self._tactile_path: str | None = None
        self._meta: dict = {}
        self._leg_idx: list = []

    def start(self, session_dir: str, session_label: str = "",
              meta: dict | None = None, record_tactile: bool = True) -> str:
        os.makedirs(session_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{session_label}" if session_label else ""
        # Absolute paths so the meta sidecar and the viewer subprocess (different
        # cwd) resolve correctly.
        self._csv_path = os.path.abspath(
            os.path.join(session_dir, f"robot_raw{suffix}_{ts}.csv"))
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=we.session_fieldnames())
        self._writer.writeheader()
        self._file.flush()

        # Leg motor indices "down to the knees" (hip+knee per leg, 8; no ankles),
        # selected from the G1 config's 12-element leg ordering.
        try:
            leg_all = list(self._hw.config.leg_joint2motor_idx)
            self._leg_idx = [leg_all[i] for i in _LEG_KNEE_SEL]
        except Exception:
            self._leg_idx = []

        self._meta = dict(meta or {})
        self._meta.setdefault("created", datetime.now().isoformat(timespec="seconds"))
        self._meta.setdefault("model_path", we.DEFAULT_MODEL_PATH)
        self._meta["record_start_epoch"] = time.time()
        self._meta["csv_file"] = os.path.basename(self._csv_path)

        # Tactile binary sidecar (only if the hand exposes touch data).
        # Tactile: we record exactly the hand's LIVE touch snapshot — i.e. the
        # latency-optimised "haptics_only" reduced read used during teleop
        # (per-finger tip 3×3 + the 12×8 contact "pad", plus the palm). We never
        # trigger a fuller read here, so the recording matches what the operator
        # actually felt, and stays small.
        self._tactile_file = None
        self._tactile_layout, self._tactile_width = [], 0
        if record_tactile and self._hand is not None:
            try:
                touch = (self._hand.read() or {}).get("touch", {})
                layout, width = we.build_tactile_layout(touch)
                if width > 0:
                    self._tactile_layout, self._tactile_width = layout, width
                    self._tactile_dtype = we.tactile_record_dtype(width)
                    self._tactile_path = os.path.abspath(
                        os.path.join(session_dir, f"tactile{suffix}_{ts}.bin"))
                    self._tactile_file = open(self._tactile_path, "wb")
                    self._meta["tactile_file"] = os.path.basename(self._tactile_path)
                    self._meta["tactile_width"] = int(width)
                    self._meta["tactile_layout"] = layout
                    self._meta["tactile_dtype"] = we.TACTILE_VALUE_DTYPE
                    self._meta["tactile_subregions"] = we.describe_tactile_layout(layout)
                    self._meta["tactile_note"] = (
                        "haptics_only reduced read: per-finger fingertip (3x3) + "
                        "pad (12x8) + palm — the low-latency set used in teleop.")
            except Exception as exc:
                print(f"[SessionRecorder] tactile init failed: {exc}")

        we.save_capture_meta(self._csv_path, self._meta)

        self._stop_evt.clear()
        self._active = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="session-rec")
        self._thread.start()
        return self._csv_path

    def update_meta(self, **kw):
        """Merge keys into the sidecar meta (e.g. final known_mass / baseline)."""
        self._meta.update(kw)
        if self._csv_path is not None:
            try:
                we.save_capture_meta(self._csv_path, self._meta)
            except Exception:
                pass

    def stop(self) -> str | None:
        self._active = False
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
        if self._tactile_file is not None:
            self._tactile_file.flush()
            self._tactile_file.close()
            self._tactile_file = None
        self._meta["record_stop_epoch"] = time.time()
        if self._csv_path is not None:
            try:
                we.save_capture_meta(self._csv_path, self._meta)
            except Exception:
                pass
        return self._csv_path

    def capture(self, q_goal_torso=None, q_goal_left=None, q_goal_right=None,
                q_goal_legs=None, finger_left=None, finger_right=None,
                snapshot=None):
        """Queue one raw frame. All args optional; missing data is written as 0.

        Records legs-down-to-the-knees + torso + both arms. ``q_goal_legs`` is the
        8-element hip+knee goal; ``finger_left``/``finger_right`` are the sim's
        coupled finger qpos (12); ``snapshot`` is ``hand_controller.read()``."""
        if not self._active or self._hw is None:
            return
        try:
            ms = self._hw.low_state.motor_state

            def _temp(i):
                t = ms[i].temperature
                try:
                    return float(max(t))   # HG int16[2]: hotter of motor/driver
                except TypeError:
                    return float(t)         # Go scalar

            idx_map = {
                "legs":  self._leg_idx,
                "torso": list(self._hw._torso_idx),
                "left":  list(self._hw._left_idx),
                "right": list(self._hw._right_idx),
            }
        except Exception:
            return

        goal_map = {"legs": q_goal_legs, "torso": q_goal_torso,
                    "left": q_goal_left, "right": q_goal_right}

        row = {"timestamp": round(time.time(), 4)}
        for seg, n in we.SESSION_JOINT_SEGMENTS:
            idxs = idx_map.get(seg, [])
            goal = goal_map.get(seg)
            for j in range(n):
                if j < len(idxs):
                    i = idxs[j]
                    row[f"q_{seg}_{j}"]    = round(float(ms[i].q), 5)
                    row[f"dq_{seg}_{j}"]   = round(float(ms[i].dq), 5)
                    row[f"ddq_{seg}_{j}"]  = round(float(ms[i].ddq), 5)
                    row[f"tau_{seg}_{j}"]  = round(float(ms[i].tau_est), 5)
                    row[f"temp_{seg}_{j}"] = round(_temp(i), 2)
                else:
                    for fld in ("q", "dq", "ddq", "tau", "temp"):
                        row[f"{fld}_{seg}_{j}"] = 0.0
                row[f"qgoal_{seg}_{j}"] = (round(float(goal[j]), 5)
                                          if goal is not None and j < len(goal) else 0.0)

        finger_map = {"finger_left": finger_left, "finger_right": finger_right}
        for seg, n in we.FINGER_SEGMENTS:
            arr = finger_map.get(seg)
            for j in range(n):
                row[f"q_{seg}_{j}"] = (round(float(arr[j]), 5)
                                       if arr is not None and j < len(arr) else 0.0)

        states = snapshot.get("states", {}) if snapshot else {}
        for side in we.HAND_SIDES:
            sdict = states.get(we.HAND_SIDE_STATE_KEY[side], {})
            for fld in we.HAND_REGISTER_FIELDS:
                arr = sdict.get(we.HAND_REGISTER_KEY[fld], [])
                for j in range(we.HAND_REGISTER_DOF):
                    val = arr[j] if (arr is not None and j < len(arr)) else 0
                    row[f"hand_{side}_{fld}_{j}"] = (
                        int(val) if isinstance(val, (int, np.integer))
                        else round(float(val), 3))

        # Tactile frame travels in the same queue item so it stays frame-aligned.
        tac = None
        if self._tactile_width > 0 and snapshot is not None:
            try:
                tac = (row["timestamp"],
                       we.flatten_touch(snapshot.get("touch", {}), self._tactile_layout))
            except Exception:
                tac = None

        try:
            self._queue.put_nowait((row, tac))
        except queue.Full:
            pass

    def _run(self):
        while not self._stop_evt.is_set() or not self._queue.empty():
            try:
                row, tac = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._writer is not None:
                    self._writer.writerow(row)
                if tac is not None and self._tactile_file is not None:
                    rec = np.zeros((), dtype=self._tactile_dtype)
                    rec["t"] = tac[0]
                    rec["v"] = tac[1]
                    rec.tofile(self._tactile_file)
                if self._queue.empty():
                    if self._file is not None:
                        self._file.flush()
                    if self._tactile_file is not None:
                        self._tactile_file.flush()
            except Exception as exc:
                print(f"[SessionRecorder] write error: {exc}")


# ============================================================================
# WrenchLiveWorker — live end-effector wrench readout (off the hot path)
# ============================================================================

class WrenchLiveWorker:
    """Computes the live end-effector wrench for one or both hands in a dedicated
    daemon thread, completely off the teleop hot path.

    It owns its own :class:`wrench_estimation.WrenchEstimator` (private MjModel /
    MjData), so it never touches the sim/viewer state or the ControlThread, and
    it only *reads* the hardware ``low_state`` — never any hardware write or
    blocking serial I/O. Disabled by default; the operator enables it for the
    known-mass test to watch the estimate update as a load is applied.

    Per the project latency rule, this never runs inside the IK loop: it samples
    at a modest ``rate_hz`` and publishes the latest result for the 5 Hz GUI
    status refresh to read.
    """

    def __init__(self, hw_controller, estimator: "we.WrenchEstimator",
                 sides=("left", "right"), rate_hz: float = 10.0):
        self._hw = hw_controller
        self._est = estimator
        self._sides = tuple(sides)
        self._dt = 1.0 / max(1.0, rate_hz)

        self._enabled = threading.Event()
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._raw: dict = {s: None for s in self._sides}        # latest raw wrench (6,)
        self._point: dict = {s: None for s in self._sides}
        self._baseline: dict = {s: np.zeros(6) for s in self._sides}

        # Tare request state (serviced inside the worker thread to keep the
        # estimator's MjData single-threaded).
        self._tare_remaining = 0
        self._tare_acc: dict = {s: [] for s in self._sides}

        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="wrench-live")
        self._thread.start()

    # ── control ───────────────────────────────────────────────────────────
    def set_enabled(self, enabled: bool):
        if enabled:
            self._enabled.set()
        else:
            self._enabled.clear()
            with self._lock:
                self._raw = {s: None for s in self._sides}
                self._point = {s: None for s in self._sides}

    def is_enabled(self) -> bool:
        return self._enabled.is_set()

    def request_tare(self, n_samples: int = 15):
        """Average the next ``n_samples`` readings (per side) into the baseline.

        Call with the arm held empty and still; the live readout then reports the
        load relative to this baseline (cancels model gravity-comp error + a
        static-friction bias). Auto-enables sampling."""
        self._enabled.set()
        with self._lock:
            self._tare_acc = {s: [] for s in self._sides}
            self._tare_remaining = max(1, int(n_samples))

    def clear_baseline(self):
        with self._lock:
            self._baseline = {s: np.zeros(6) for s in self._sides}

    def baseline_dict(self) -> dict:
        with self._lock:
            return {s: self._baseline[s].tolist() for s in self._sides}

    def get(self, side: str):
        """Return ``(wrench_corrected (6,), point (3,))`` or ``(None, None)``."""
        with self._lock:
            raw = self._raw.get(side)
            pt = self._point.get(side)
            base = self._baseline.get(side)
        if raw is None:
            return None, None
        return raw - base, pt

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=1.0)

    # ── worker loop ─────────────────────────────────────────────────────────
    def _run(self):
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            if self._enabled.is_set() and self._hw is not None:
                try:
                    self._sample_once()
                except Exception:
                    pass
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._dt - elapsed))

    def _sample_once(self):
        ls = self._hw.low_state
        ms = ls.motor_state
        q_torso = np.array([ms[i].q for i in self._hw._torso_idx], dtype=np.float64)
        q_left  = np.array([ms[i].q for i in self._hw._left_idx],  dtype=np.float64)
        q_right = np.array([ms[i].q for i in self._hw._right_idx], dtype=np.float64)
        tau = {
            "left":  np.array([ms[i].tau_est for i in self._hw._left_idx],  dtype=np.float64),
            "right": np.array([ms[i].tau_est for i in self._hw._right_idx], dtype=np.float64),
        }
        q_arm = {"left": q_left, "right": q_right}
        q_other = {"left": q_right, "right": q_left}

        results = {}
        for s in self._sides:
            r = self._est.estimate(s, q_arm[s], tau[s], q_torso=q_torso,
                                   q_other_arm=q_other[s])
            results[s] = (r.wrench, r.point)

        with self._lock:
            for s in self._sides:
                self._raw[s], self._point[s] = results[s]
            if self._tare_remaining > 0:
                for s in self._sides:
                    self._tare_acc[s].append(results[s][0])
                self._tare_remaining -= 1
                if self._tare_remaining == 0:
                    for s in self._sides:
                        acc = self._tare_acc[s]
                        if acc:
                            self._baseline[s] = np.mean(acc, axis=0)
                    self._tare_acc = {s: [] for s in self._sides}


# ============================================================================
# ZEDRecorder
# ============================================================================

class ZEDRecorder:
    """Manages ZED streaming (mediamtx + GStreamer) and recording (ffmpeg)."""

    RTSP_LEFT   = "rtsp://localhost:8554/zed/left"
    RTSP_RIGHT  = "rtsp://localhost:8554/zed/right"
    # SBS (default): the ZED pipeline publishes ONE stitched [left|right] frame here (this
    # is the low-latency single-stream mode the headset uses). We record THIS with -c copy
    # (no re-encode → zero extra GPU load → no teleop-latency impact); the dataset builder
    # crops the correct single eye per robot (master=left half, peer=right half) offline.
    RTSP_STEREO = "rtsp://localhost:8554/zed_stereo"

    def __init__(self, output_dir: str = "./xr_recordings"):
        self._output_dir = output_dir
        self._mediamtx_proc: subprocess.Popen | None = None
        self._zed_proc: subprocess.Popen | None = None
        # Active recording ffmpeg procs as (label, proc, path). One entry for the SBS
        # stereo stream, or one/two for the separate-path (SBS=0) mode.
        self._ffmpegs: list[tuple[str, subprocess.Popen, str]] = []
        self._stream_start_time: float = 0.0
        # Match the ZED script's SBS default ("${SBS:-1}"): SBS=1 streams one stitched
        # /zed_stereo frame (what we record), SBS=0 streams separate /zed/left+/zed/right.
        self.sbs: bool = os.environ.get("SBS", "1").strip() != "0"

    def start_streaming(self) -> bool:
        if not self.is_streaming:
            ok = self._start_mediamtx()
            if not ok:
                return False
            time.sleep(0.5)
        ok = self._start_zed_pipeline()
        if ok:
            self._stream_start_time = time.monotonic()
        return ok

    def stop_streaming(self):
        self._kill(self._zed_proc, "ZED pipeline")
        self._zed_proc = None
        self._kill(self._mediamtx_proc, "mediamtx")
        self._mediamtx_proc = None
        self._stream_start_time = 0.0

    def reboot_streaming(self) -> bool:
        """FULL ZED video-pipeline reboot — the fresh bring-up a manager restart gives, but
        MORE thorough: tears down the ZED publisher AND mediamtx — including an ADOPTED /
        orphaned mediamtx that stop_streaming() spares — plus any stray publishers, waits for
        the RTSP port to free, then spawns a genuinely fresh mediamtx + publisher and reopens
        the camera. Blocks ~2-4 s (call off the UI thread). Returns True if it came back up."""
        # 1. Tear the whole pipeline down.
        self._kill(self._zed_proc, "ZED pipeline")
        self._zed_proc = None
        self._kill_orphan_publishers()                 # stray zed_to_mediamtx.sh / gst zedsrc
        self._kill(self._mediamtx_proc, "mediamtx")    # our own, if we spawned it
        self._mediamtx_proc = None
        self._kill_all_mediamtx()                      # + any adopted/orphaned mediamtx
        self._mediamtx_external_until = 0.0            # forget the 'adopt' cache
        self._stream_start_time = 0.0
        # 2. Wait for the RTSP port to actually free, so _start_mediamtx spawns fresh instead
        #    of re-adopting a dying server (TIME_WAIT) — up to ~3 s.
        for _ in range(30):
            if not self._mediamtx_port_open():
                break
            time.sleep(0.1)
        # 3. Bring it back up fresh (mediamtx, then the ZED publisher).
        if not self._start_mediamtx():
            return False
        time.sleep(0.5)
        ok = self._start_zed_pipeline()
        if ok:
            self._stream_start_time = time.monotonic()
        return ok

    @staticmethod
    def _kill_all_mediamtx():
        """Kill EVERY mediamtx serving our RTSP port (ours + any adopted/orphaned one) so the
        reboot brings up a genuinely fresh mediamtx. Matches our specific binary path, so it
        can't hit an unrelated 'mediamtx' elsewhere."""
        try:
            subprocess.run(["pkill", "-f", str(_MEDIAMTX)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5.0)
        except Exception:
            pass
        time.sleep(0.3)

    def start_recording(self, session_label: str = "", side: str | None = None) -> dict | None:
        """Record the ZED video for one take. In SBS mode records the single stitched
        /zed_stereo stream → ``zed_stereo_*.mp4`` (one -c copy, no re-encode). In the
        separate-path mode records only the localization eye (``side`` = 'left'/'right'),
        or both eyes when ``side`` is None. Returns {label: path} or None.

        All ffmpeg here is stream-copy (``-c copy``), so recording adds NO encode load and
        cannot affect teleop latency — the frames are already encoded by the GStreamer
        pipeline that feeds the headset."""
        if not self.is_streaming:
            print("[ZEDRecorder] Cannot record: streaming not active.")
            return None
        os.makedirs(self._output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{session_label}" if session_label else ""

        def _p(name):
            return os.path.join(self._output_dir, f"{name}{suffix}_{ts}.mp4")

        targets: list[tuple[str, str, str]] = []   # (label, rtsp_url, path)
        if self.sbs:
            targets.append(("stereo", self.RTSP_STEREO, _p("zed_stereo")))
        else:
            if side in (None, "left"):
                targets.append(("left", self.RTSP_LEFT, _p("zed_left")))
            if side in (None, "right"):
                targets.append(("right", self.RTSP_RIGHT, _p("zed_right")))

        self._ffmpegs = [(lbl, self._launch_ffmpeg(url, path), path)
                         for (lbl, url, path) in targets]
        return {lbl: path for (lbl, _proc, path) in self._ffmpegs}

    def stop_recording(self):
        for label, proc, _path in self._ffmpegs:
            if proc is not None and proc.poll() is None:
                try:
                    proc.stdin.write(b"q")
                    proc.stdin.flush()
                    proc.wait(timeout=5.0)
                except Exception:
                    self._kill(proc, f"ffmpeg-{label}")
        self._ffmpegs = []

    _RTSP_PORT = 8554

    def _mediamtx_port_open(self) -> bool:
        """Is SOME mediamtx serving the RTSP port (ours or one left over from an earlier
        session)? Cheap TCP probe."""
        try:
            with socket.create_connection(("127.0.0.1", self._RTSP_PORT), timeout=0.3):
                return True
        except Exception:
            return False

    def _mediamtx_ok(self) -> bool:
        """mediamtx is available: our own child is alive, OR an EXTERNAL mediamtx (e.g.
        surviving from a previous manager run) is serving the port. Without the external
        case, a leftover mediamtx makes our own spawn die on the busy port → is_streaming
        reads False forever → red ZED light + 'Cannot record: streaming not active' at
        every take, even though video is actually flowing. Probe result cached ~5 s."""
        if self._mediamtx_proc is not None and self._mediamtx_proc.poll() is None:
            return True
        now = time.monotonic()
        if now < getattr(self, "_mediamtx_external_until", 0.0):
            return True
        if self._mediamtx_port_open():
            self._mediamtx_external_until = now + 5.0
            return True
        return False

    @property
    def is_streaming(self) -> bool:
        return (self._mediamtx_ok()
                and self._zed_proc is not None and self._zed_proc.poll() is None)

    @property
    def is_recording(self) -> bool:
        return any(proc is not None and proc.poll() is None
                   for _label, proc, _path in self._ffmpegs)

    @property
    def stream_uptime_s(self) -> float:
        if self._stream_start_time > 0 and self.is_streaming:
            return time.monotonic() - self._stream_start_time
        return 0.0

    def _start_mediamtx(self) -> bool:
        # ADOPT an already-running mediamtx (ours from a previous run that outlived the
        # manager, or any other) instead of spawning a doomed duplicate — a second
        # instance dies instantly on the busy port, leaving is_streaming falsely False.
        if self._mediamtx_port_open():
            print("[ZEDRecorder] adopting already-running mediamtx (RTSP port "
                  f"{self._RTSP_PORT} is serving).")
            self._mediamtx_external_until = time.monotonic() + 5.0
            return True
        if not _MEDIAMTX.exists():
            print(f"[ZEDRecorder] mediamtx binary not found at {_MEDIAMTX}. Skipping.")
            return False
        cmd = [str(_MEDIAMTX)]
        if _MEDIAMTX_CFG.exists():
            cmd.append(str(_MEDIAMTX_CFG))
        self._mediamtx_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True

    def _start_zed_pipeline(self) -> bool:
        if not _ZED_SCRIPT.exists():
            print(f"[ZEDRecorder] ZED script not found at {_ZED_SCRIPT}.")
            return False
        # Clear any ORPHANED publisher first. A previous manager run that crashed or was
        # killed before stop_streaming() leaves its zed_to_mediamtx.sh / gst-launch zedsrc
        # re-parented to systemd, still HOLDING the ZED camera open but usually STALLED
        # (0 fps). THIS run's fresh publisher then can't open the busy camera and exits, so
        # /zed_stereo stays 404 and the XR app — which needs that video — tears down and
        # retries the whole peer connection forever. Kill leftovers so the camera is free.
        self._kill_orphan_publishers()
        self._zed_proc = subprocess.Popen(
            ["bash", str(_ZED_SCRIPT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True

    @staticmethod
    def _kill_orphan_publishers():
        """Kill any leftover ZED publisher (bash launcher + its gst-launch zedsrc) so the
        camera is free before we open it. Only our own pipeline matches these patterns, and
        there is exactly one ZED per host, so this can't hit an unrelated process."""
        for pat in ("zed_to_mediamtx.sh", "gst-launch-1.0.*zedsrc"):
            try:
                subprocess.run(["pkill", "-f", pat], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=5.0)
            except Exception:
                pass
        time.sleep(1.0)     # let the camera handle release before we re-open it

    def rtsp_path_live(self, path: str = "zed_stereo", timeout: float = 1.0) -> bool:
        """True if mediamtx is actually serving frames on the given path (RTSP DESCRIBE →
        200). A 404 means the publisher isn't publishing — camera wedged / not producing
        frames. Cheap; used to surface a silently-failed ZED start instead of the manager
        reporting phantom 'streaming started' while the XR app can't get any video."""
        try:
            with socket.create_connection(("127.0.0.1", self._RTSP_PORT), timeout=timeout) as s:
                s.settimeout(timeout)
                s.sendall(
                    (f"DESCRIBE rtsp://127.0.0.1:{self._RTSP_PORT}/{path} RTSP/1.0\r\n"
                     f"CSeq: 1\r\nAccept: application/sdp\r\n\r\n").encode())
                resp = s.recv(128).decode("latin-1", "replace")
            return resp.startswith("RTSP/1.0 200")
        except Exception:
            return False

    @staticmethod
    def _launch_ffmpeg(rtsp_url: str, output_path: str) -> subprocess.Popen:
        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-c", "copy",
            "-f", "mp4",
            output_path,
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def _kill(proc: subprocess.Popen | None, label: str):
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


# ============================================================================
# AprilTagTrackerProxy — the crate tracker in a SEPARATE PROCESS (own GIL)
# ============================================================================

class _CamView:
    """Read-only view of one tracker camera's health, from the status stream. Mirrors the
    in-process _CameraWorker surface the manager's _cam_online() reads."""
    def __init__(self, d):
        self.serial = d.get("serial", "")
        self.started_ok = bool(d.get("started_ok", False))
        self.error = d.get("error")
        self._stamp = float(d.get("stamp", 0.0))

    def latest(self):
        return {"stamp": self._stamp} if self._stamp else None


class AprilTagTrackerProxy:
    """Drop-in stand-in for AprilTagTracker that runs the real tracker in its OWN process
    (``apriltag_tracker.py --serve``) and talks to it over a socketpair with length-prefixed
    JSON. All the heavy AprilTag detection then runs on a different GIL, so it can never
    starve the manager's single teleop IK thread. Exposes the same methods/attributes the
    manager uses: start/stop, latest, is_anchored, T_world_cam, cams, anchor_cameras,
    start_recording/stop_recording, config_dict, add_remote_candidate, tag_size, anchor_id."""

    def __init__(self, init_msg, log_fn=None):
        self._init = init_msg
        self._log = log_fn or (lambda *_a: None)
        self._proc = None
        self._sock = None
        self._rx = None
        self._lock = threading.Lock()          # guards _status
        self._status = {"anchored": False, "cams": [], "latest": None}
        self._pending = {}                     # rpc id -> [Event, reply]
        self._id_lock = threading.Lock()
        self._send_lock = threading.Lock()     # serialize sends (GUI + link-recv threads)
        self._next_id = 0
        self._hello = threading.Event()
        self._start_err = None
        self.tag_size = init_msg.get("tag_size")
        self.anchor_id = None
        self._config = {}

    # -- lifecycle --
    def start(self, timeout=20.0):
        parent, child = socket.socketpair()
        script = str(_HERE / "apriltag_tracker.py")
        try:
            self._proc = subprocess.Popen(
                [sys.executable, script, "--serve", "--fd", str(child.fileno())],
                pass_fds=[child.fileno()], cwd=str(_HERE),
                preexec_fn=_nice_child,   # yield CPU to the teleop path
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            child.close()
        self._sock = parent
        self._send({"init": self._init})
        self._rx = threading.Thread(target=self._rx_loop, daemon=True, name="crate-proxy-rx")
        self._rx.start()
        if not self._hello.wait(timeout):
            self.stop()
            raise TimeoutError("crate tracker subprocess did not come up in time")
        if self._start_err:
            self.stop()
            raise RuntimeError(self._start_err)

    def stop(self):
        try:
            self._send({"cmd": "shutdown"})
        except Exception:
            pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.wait(timeout=4.0)            # let it release the RealSense cameras
            except Exception:
                try:
                    p.terminate(); p.wait(timeout=3.0)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        self._proc = None

    # -- wire --
    def _send(self, obj):
        s = self._sock
        if s is None:
            raise ConnectionError("tracker process not connected")
        blob = json.dumps(obj).encode("utf-8")
        frame = struct.pack(">I", len(blob)) + blob
        with self._send_lock:                  # one atomic frame at a time on the wire
            s.sendall(frame)

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("tracker IPC socket closed")
            buf += chunk
        return buf

    def _rx_loop(self):
        try:
            while True:
                (length,) = struct.unpack(">I", self._recv_exact(4))
                msg = json.loads(self._recv_exact(length).decode("utf-8"))
                if "status" in msg:
                    with self._lock:
                        self._status = msg["status"]
                elif "hello" in msg:
                    h = msg["hello"]
                    self.tag_size = h.get("tag_size", self.tag_size)
                    self.anchor_id = h.get("anchor_id")
                    self._config = h.get("config", {})
                    self._hello.set()
                elif "error" in msg:
                    self._start_err = msg["error"]
                    self._hello.set()
                elif "reply" in msg:
                    with self._id_lock:
                        slot = self._pending.get(msg["reply"])
                    if slot is not None:
                        slot[1] = msg
                        slot[0].set()
                elif "log" in msg:
                    self._log(msg["log"])
        except Exception:
            pass

    def _rpc(self, cmd, timeout=10.0, **kw):
        with self._id_lock:
            rid = self._next_id
            self._next_id += 1
            evt = threading.Event()
            self._pending[rid] = [evt, None]
        self._send({"cmd": cmd, "id": rid, **kw})
        ok = evt.wait(timeout)
        with self._id_lock:
            slot = self._pending.pop(rid, None)
        if not ok or slot is None or slot[1] is None:
            raise TimeoutError(f"crate tracker '{cmd}' timed out")
        reply = slot[1]
        if reply.get("ok") is False:
            raise RuntimeError(reply.get("error") or f"crate tracker '{cmd}' failed")
        return reply

    # -- interface the manager uses --
    def anchor_cameras(self, duration_s=1.5, serials=None, keep_on_fail=False):
        r = self._rpc("anchor", timeout=max(12.0, duration_s + 8.0),
                      duration_s=duration_s, serials=serials, keep_on_fail=keep_on_fail)
        return {s: tuple(v) for s, v in r.get("results", {}).items()}

    def start_recording(self, session_dir, label=""):
        r = self._rpc("start_recording", timeout=8.0,
                      dir=os.path.abspath(session_dir), label=label)
        return tuple(r.get("paths", (None, None)))

    def stop_recording(self):
        try:
            self._rpc("stop_recording", timeout=5.0)
        except Exception:
            pass

    def config_dict(self):
        return dict(self._config)

    def add_remote_candidate(self, p, q, weight=1.0, stamp=None):
        try:
            self._send({"cmd": "remote_candidate", "p": list(p), "q": list(q),
                        "weight": float(weight), "stamp": stamp})
        except Exception:
            pass

    def latest(self):
        with self._lock:
            return self._status.get("latest")

    @property
    def is_anchored(self):
        with self._lock:
            return bool(self._status.get("anchored"))

    @property
    def T_world_cam(self):
        # The manager only counts non-None entries; expose True/None per external camera.
        with self._lock:
            return {c["serial"]: (True if c.get("anchored") else None)
                    for c in self._status.get("cams", [])}

    @property
    def cams(self):
        with self._lock:
            return [_CamView(c) for c in self._status.get("cams", [])]


# ============================================================================
# TranscriberProxy — live speech transcription in its OWN process
# ============================================================================

class TranscriberProxy:
    """Runs study_transcriber.py --serve in a subprocess (own GIL: whisper/VAD/
    embeddings can never touch teleop) and mirrors the small interface the manager
    needs. Fire-and-forget for session/take plumbing; blocking RPC (with a Qt event
    pump) only for voice enrollment."""

    def __init__(self, init_msg, log_fn=None):
        self._init = dict(init_msg)
        self._log = log_fn or (lambda m: None)
        self._sock = None
        self._proc = None
        self._send_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._pending = {}
        self._next_id = 1
        self._enroll_rid = None      # id of an in-progress open-ended enrollment
        self.on_transcript = None    # set by the manager: called per finished transcript row
        self._status = {}
        self._lock = threading.Lock()
        self.ready = False
        self.failed = None

    def start(self):
        import subprocess as _sp
        script = _HERE / "study_transcriber.py"
        parent, child = socket.socketpair()
        try:
            self._proc = _sp.Popen(
                [sys.executable, str(script), "--serve", "--fd", str(child.fileno())],
                pass_fds=[child.fileno()], cwd=str(_HERE),
                preexec_fn=_nice_child,   # yield CPU to the teleop path
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        finally:
            child.close()
        self._sock = parent
        self._send({"init": self._init})
        threading.Thread(target=self._rx_loop, daemon=True,
                         name="transcribe-proxy-rx").start()

    def stop(self):
        try:
            self._send({"cmd": "shutdown"})
        except Exception:
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.wait(timeout=3.0)
            except Exception:
                try:
                    p.terminate()
                except Exception:
                    pass
        self._proc = None

    # -- wire --
    def _send(self, obj):
        s = self._sock
        if s is None:
            raise ConnectionError("transcriber not connected")
        blob = json.dumps(obj).encode("utf-8")
        with self._send_lock:
            s.sendall(struct.pack(">I", len(blob)) + blob)

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("transcriber IPC closed")
            buf += chunk
        return buf

    def _rx_loop(self):
        try:
            while True:
                (length,) = struct.unpack(">I", self._recv_exact(4))
                msg = json.loads(self._recv_exact(length).decode("utf-8"))
                if "status" in msg:
                    with self._lock:
                        self._status = msg["status"]
                elif "hello" in msg:
                    h = msg["hello"]
                    self.ready = bool(h.get("ok"))
                    self.failed = h.get("error")
                    if self.ready:
                        self._log(f"[transcribe] listening on {h.get('device')}")
                    else:
                        self._log(f"[transcribe] NOT started: {h.get('error')}")
                elif "reply" in msg:
                    with self._id_lock:
                        slot = self._pending.get(msg["reply"])
                    if slot is not None:
                        slot[1] = msg
                        slot[0].set()
                elif "transcript" in msg:
                    cb = self.on_transcript
                    if cb is not None:
                        try:
                            cb(msg["transcript"])
                        except Exception:
                            pass
                elif "log" in msg:
                    self._log(msg["log"])
        except Exception:
            pass

    def _fire(self, **kw):
        try:
            self._send(kw)
        except Exception:
            pass

    # -- interface --
    def status(self):
        with self._lock:
            return dict(self._status)

    def start_session(self, session_dir, session_id):
        self._fire(cmd="start_session", dir=os.path.abspath(session_dir),
                   session_id=str(session_id), id=0)

    def set_take(self, label):
        self._fire(cmd="set_take", label=str(label))

    def set_recording(self, on):
        """Gate saving: only speech while an attempt is recording is transcribed + saved."""
        self._fire(cmd="set_recording", on=bool(on))

    def stop_session(self):
        self._fire(cmd="stop_session", id=0)

    def enroll(self, name, seconds=12.0, timeout=40.0):
        """Blocking (Qt-pumped) voice enrollment; returns (ok, message)."""
        with self._id_lock:
            rid = self._next_id
            self._next_id += 1
            evt = threading.Event()
            self._pending[rid] = [evt, None]
        try:
            self._send({"cmd": "enroll", "id": rid, "name": str(name),
                        "seconds": float(seconds)})
        except Exception as exc:
            return False, f"send failed: {exc}"
        end = time.monotonic() + timeout
        while time.monotonic() < end and not evt.is_set():
            QApplication.processEvents()
            time.sleep(0.05)
        with self._id_lock:
            slot = self._pending.pop(rid, None)
        if slot is None or slot[1] is None:
            return False, "enrollment timed out"
        return bool(slot[1].get("ok")), str(slot[1].get("msg", slot[1].get("error", "")))

    def enroll_start(self, name):
        """Begin OPEN-ENDED voice enrollment: the subprocess records until enroll_stop().
        Non-blocking; the embedding result is delivered by the matching enroll_stop().
        Returns True if the start was sent."""
        with self._id_lock:
            rid = self._next_id
            self._next_id += 1
            self._pending[rid] = [threading.Event(), None]
            self._enroll_rid = rid
        try:
            self._send({"cmd": "enroll_start", "id": rid, "name": str(name)})
            return True
        except Exception as exc:
            with self._id_lock:
                self._pending.pop(rid, None)
                self._enroll_rid = None
            self._log(f"[transcribe] enroll start failed: {exc}")
            return False

    def enroll_stop(self, timeout=40.0):
        """Signal STOP for the in-progress open-ended enrollment and wait (Qt-pumped) for
        the embedding result. Returns (ok, message)."""
        rid = self._enroll_rid
        if rid is None:
            return False, "no enrollment in progress"
        try:
            self._send({"cmd": "enroll_stop", "id": 0})
        except Exception as exc:
            return False, f"stop send failed: {exc}"
        with self._id_lock:
            slot = self._pending.get(rid)
        evt = slot[0] if slot else None
        end = time.monotonic() + timeout
        while evt is not None and time.monotonic() < end and not evt.is_set():
            QApplication.processEvents()
            time.sleep(0.05)
        with self._id_lock:
            slot = self._pending.pop(rid, None)
            self._enroll_rid = None
        if slot is None or slot[1] is None:
            return False, "enrollment timed out"
        return bool(slot[1].get("ok")), str(slot[1].get("msg", slot[1].get("error", "")))

    def enroll_clear(self):
        self._fire(cmd="enroll_clear", id=0)


# ============================================================================
# _NeckWorker — fire-and-forget Dynamixel neck command thread
# ============================================================================

# ONE lock shared by EVERY _NeckWorker (and the reset), so the Dynamixel serial is NEVER
# touched by two workers at once. A neck HARD RESET builds a fresh interface + worker; if the
# OLD worker isn't fully dead yet (e.g. it was mid serial op when stopped), the old and new
# workers could otherwise issue GroupSyncWrite txPackets concurrently on the same /dev/ttyUSB
# → a C-level use-after/​double-access in the dynamixel_sdk/pyserial layer → SEGFAULT (seen as
# a manager crash on the first teleop-engage AFTER a neck reset). Serializing every serial op
# through this module-level lock makes that race impossible regardless of worker lifecycle.
_NECK_SERIAL_LOCK = threading.Lock()


class _NeckWorker:
    """Sends neck commands from a dedicated daemon thread so the ControlThread
    is never blocked by synchronous serial I/O (pyserial / dynamixel_sdk).

    Uses the same drop-newest pattern as _BoneProcessor: if a new command
    arrives while the previous one is still in-flight, the older one is
    silently replaced.  At 60 Hz IK rate and ~1 ms serial write time the
    neck tracks almost as tightly as in the demo while keeping the IK loop
    latency-free.
    """

    def __init__(self, neck_interface):
        self._neck = neck_interface
        self._lock = threading.Lock()
        self._pending = None          # latest q_head to send, or None
        self._wakeup = threading.Event()
        self._faulted = False         # serial/servo error seen → drop cmds until recovered
        self._reboot_req = False      # manual reboot requested (manager 'Reboot Neck' button)
        self._err_logged = False      # rate-limit the fault log
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="neck-worker")
        self._thread.start()

    def send(self, q_head) -> None:
        """Queue a neck command; callable from any thread."""
        with self._lock:
            self._pending = q_head
        self._wakeup.set()

    def stop(self) -> None:
        """Terminate the worker thread (used when the ControlThread is torn down / re-init'd
        so a stale worker can't keep touching the serial alongside a fresh one)."""
        self._stop.set()
        self._wakeup.set()

    def request_reboot(self) -> None:
        """Ask the worker to REBOOT the neck servos + reconnect. Done on the worker thread
        (the only one that touches the Dynamixel serial), so it's safe to call from the Qt
        button handler. Also clears a silent servo error (overload/overheat) that never
        raised an exception — so it recovers the neck even when the worker looks healthy."""
        with self._lock:
            self._reboot_req = True
        self._wakeup.set()

    @property
    def faulted(self) -> bool:
        with self._lock:
            return self._faulted

    def _do_reboot(self) -> bool:
        if self._stop.is_set():            # torn down → never touch the serial (avoids racing a
            return False                   # fresh interface being built on the same /dev/ttyUSB)
        try:
            with _NECK_SERIAL_LOCK:        # never overlap another worker's serial op
                self._neck.reboot()        # also returns the neck to its front-facing home
            with self._lock:
                self._faulted = False
                # Drop any stale pre-fault head command so the freshly-homed neck stays
                # front-facing until LIVE tracking sends a new pose (mirrors a fresh boot).
                self._pending = None
            self._err_logged = False
            print("[NeckWorker] neck rebooted + reconnected (homed front) — tracking resumed.")
            return True
        except Exception as exc:                       # noqa: BLE001
            with self._lock:
                self._faulted = True
            print(f"[NeckWorker] neck reboot attempt failed: {exc}")
            return False

    def _run(self) -> None:
        # The thread NEVER exits on error anymore (that used to leave the neck dead until a
        # manager restart). On a serial/servo fault it drops commands, AUTO-recovers on a
        # capped exponential backoff, and honors a manual reboot request immediately.
        backoff = 2.0
        next_auto = None                               # monotonic time of next auto-retry
        while True:
            self._wakeup.wait(timeout=1.0)             # periodic tick drives auto-recovery
            self._wakeup.clear()
            if self._stop.is_set():
                return
            now = time.monotonic()

            with self._lock:
                do_reboot = self._reboot_req
                self._reboot_req = False
                faulted = self._faulted
            # Auto-recovery: while faulted, retry on the backoff schedule.
            if faulted and not do_reboot and next_auto is not None and now >= next_auto:
                do_reboot = True
                backoff = min(backoff * 2.0, 30.0)
                next_auto = now + backoff
            if self._stop.is_set():                    # stopped between the tick and here
                return
            if do_reboot:
                if self._do_reboot():
                    backoff, next_auto = 2.0, None     # recovered → reset schedule
                elif next_auto is None:
                    next_auto = now + backoff

            # Drain to the newest command and send it (dropped while faulted).
            while True:
                with self._lock:
                    q = self._pending
                    self._pending = None
                    faulted = self._faulted
                if q is None or faulted:
                    break
                if self._stop.is_set():                # torn down mid-tick → don't touch serial
                    return
                try:
                    with _NECK_SERIAL_LOCK:            # never overlap another worker's serial op
                        self._neck.move_from_angles(q)
                except Exception as exc:               # noqa: BLE001
                    with self._lock:
                        self._faulted = True
                        self._pending = None
                    if not self._err_logged:
                        print(f"[NeckWorker] serial error: {exc}; neck disabled — "
                              "auto-recovering (or press 'Reboot Neck').")
                        self._err_logged = True
                    next_auto = time.monotonic() + backoff
                    break


# ============================================================================
# HardwareCommandThread — 500 Hz G1 arm command sender
# ============================================================================

class HardwareCommandThread(threading.Thread):
    """DEPRECATED — no longer instantiated.

    The 500 Hz hardware send was merged into ControlThread so that the IK solve
    and the rt/lowcmd send run in a single loop (mirroring the demo). Running the
    send in a separate high-frequency thread made it compete with the IK thread
    for the Python GIL and starved the solve, which was the teleop latency source.
    Kept only for reference; safe to delete.
    """

    def __init__(self, hw_controller, args):
        super().__init__(daemon=True, name="hw-cmd")
        self._hw = hw_controller
        self._args = args
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

        self._q_torso = np.zeros(3)
        self._q_left  = _ARMS_AT_SIDE_LEFT.copy()
        self._q_right = _ARMS_AT_SIDE_RIGHT.copy()
        self._legs    = _KNEELING_ANGLES.copy() if args.kneeling else None

    def set_goals(self, q_torso, q_left, q_right):
        with self._lock:
            self._q_torso = np.asarray(q_torso, dtype=np.float64).copy()
            self._q_left  = np.asarray(q_left,  dtype=np.float64).copy()
            self._q_right = np.asarray(q_right, dtype=np.float64).copy()

    def get_goals(self):
        with self._lock:
            return self._q_torso.copy(), self._q_left.copy(), self._q_right.copy()

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1.0)

    def run(self):
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                with self._lock:
                    q_t = self._q_torso.copy()
                    q_l = self._q_left.copy()
                    q_r = self._q_right.copy()
                    legs = self._legs.copy() if self._legs is not None else None
                self._hw.send_upper_body_targets(
                    q_goal_torso=q_t,
                    q_goal_left=q_l,
                    q_goal_right=q_r,
                    q_goal_legs=legs,
                    kp_scale=self._args.hw_kp_scale,
                    kd_scale=self._args.hw_kd_scale,
                    leg_kp_scale=self._args.leg_kp_scale,
                    leg_kd_scale=self._args.leg_kd_scale,
                    ankle_kp_scale=0.0,
                    ankle_kd_scale=1.0,
                )
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            remaining = 0.002 - elapsed
            if remaining > 0:
                time.sleep(remaining)


# ============================================================================
# ControlThread — IK pipeline decoupled from the Qt event loop
# ============================================================================

class ControlThread(threading.Thread):
    """Single teleop loop that mirrors demo_g1_inspire_xr_robot_teleop_full_body_hw.py.

    The demo runs IK *and* the hardware send inside ONE tight loop: it sends
    rt/lowcmd to the G1 every iteration (~max_fr Hz, watchdog-safe) and runs the
    IK solve gated at ik_rate Hz. Keeping both in the same thread is what makes
    the demo low-latency — there is no second high-frequency thread fighting the
    IK solve for the Python GIL, so the IK step runs deterministically every
    16 ms instead of being starved for hundreds of ms.

    This class reproduces that exactly:
      * every loop iteration  -> send_upper_body_targets() (hold pose or IK goals)
      * gated at ik_rate Hz    -> get pose, solve IK, blend, update goal buffer
    The Qt main thread only reads the goal buffer for MuJoCo visualisation.
    """

    def __init__(self, device, solver, controller, hand_controller,
                 neck_interface, hw_controller, haptics_streamer,
                 args, latency_tracker: LatencyTracker):
        super().__init__(daemon=True, name="ik-control")
        self._device = device
        self._solver = solver
        self._controller = controller
        self._hand_controller = hand_controller
        # Per-hand actuator DOFs to STOP commanding (held at rest): for a robot with a broken hand
        # actuator (e.g. a sheared thumb-rotation shaft), freezing it keeps the sim, the recording,
        # and the digital twin matching what the hardware physically does. Index order per hand is
        # [pinky, ring, middle, index, thumb-bend, thumb-rotation]. Replaced atomically from the
        # manager (set_disabled_hand_dofs); read once per solve in _do_ik.
        self._hand_dof_mask = (frozenset(), frozenset())   # (left disabled, right disabled)
        self._hand_dof_hold = 0.0                            # radian a disabled DOF is held at (rest)
        self._neck_interface = neck_interface
        # Offload Dynamixel serial writes to a dedicated thread so the IK loop
        # is never stalled by synchronous pyserial I/O.
        self._neck_worker = _NeckWorker(neck_interface) if neck_interface is not None else None
        self._hw_controller = hw_controller
        # Teleop-latency telemetry. Manager processing (pose arrival → goals committed) is recorded
        # in _do_ik (a single subtraction). The actuation-latency estimate — motor-feedback reads +
        # cross-correlation — runs on a SEPARATE daemon thread (_actuation_loop) so it NEVER adds
        # work to the IK / 500 Hz hardware-send hot path. Both are read by the GUI for the budget.
        self._last_manager_proc_ms = None
        self._last_actuation_ms = None
        self._actuation_sample_hz = 50.0
        self._actuation_est = (ActuationLatencyEstimator(dt=1.0 / self._actuation_sample_hz)
                               if hw_controller is not None else None)
        self._haptics_streamer = haptics_streamer
        self._args = args
        self._tracker = latency_tracker
        self._legs = _KNEELING_ANGLES.copy() if getattr(args, 'kneeling', False) else None
        # Waist angle held while kneeling (trimmed by the kneeling-pose
        # calibration); zero = nominal upright waist. Replaces the previous
        # hard-coded zero torso goal so a sagging robot can be brought upright.
        self._torso_hold = np.zeros(3)

        # Maximum outward abduction (shoulder roll) per arm, in radians as a
        # positive magnitude. Left abducts with +roll, right with -roll, so the
        # commanded shoulder roll is clamped to <=+left and >=-right. inf = no
        # limit (default until the arm-abduction calibration is applied). Guards
        # the shoulder heat sinks against self-collision during teleop.
        self._abd_left_max = np.inf
        self._abd_right_max = np.inf

        # ── "No hands behind the head" keep-out (OPT-IN, default OFF) ─────────
        # Operators sometimes reach up to adjust their headset while teleop is live; that maps
        # the robot's hands up and BEHIND its own head, where they can get stuck. When enabled,
        # reject any IK arm target whose resolved hand (FK, pelvis frame) lands in the keep-out
        # slab: BEHIND the head plane (x < x_back) AND at head height (z > z_head); the arm then
        # holds its last safe pose until the hand comes back forward.
        # DEFAULT OFF: the check runs a pinocchio FK inside the 500 Hz control loop — a native
        # crash there can't be caught and would drop the robot, and a false trigger at teleop
        # start can hold the arms back and tip a kneeling robot. Must be validated on hardware
        # (and ideally re-implemented off the hot path) before it's trusted enough to default on.
        self._head_keepout_on = bool(getattr(args, "head_keepout", False))
        self._head_keepout_x = float(getattr(args, "head_keepout_x", 0.02))   # m, pelvis +x fwd
        self._head_keepout_z = float(getattr(args, "head_keepout_z", 0.33))   # m, pelvis +z up
        self._head_keepout_warn_t = 0.0

        self._stop_evt = threading.Event()
        self._teleop_active = threading.Event()
        # When set (but teleop is NOT active), the loop runs a lightweight neck-only tick:
        # it reads the operator's head pose and drives the Ostrich neck WITHOUT solving full-body
        # IK or moving the arms. Driven by the XR camera streaming feed, so the operator can look
        # around with the live feed before pressing Play. See _do_neck_only().
        self._neck_active = threading.Event()
        # When set, the arm joints are streamed in pure damping mode (kp=0,
        # kd>0) so they slowly lower under gravity without pushing on objects;
        # the torso/legs stay rigid. Used both before teleop starts (bootup) and
        # after it is stopped — start_teleop() clears it. Default ON so the arms
        # idle limp rather than being held rigidly at their rest pose.
        self._arm_damp = threading.Event()
        self._arm_damp.set()

        # ── eco mode ────────────────────────────────────────────────────────
        # Off by default. When enabled, the arms shed shoulder-pitch holding
        # heat in two ways that do NOT touch active-teleop fidelity:
        #   * idle auto-relax: when the operator's commanded arm goals stay
        #     static for eco_idle_secs, the arms ramp into PARTIAL damping
        #     (kp shrinks toward arm_damp_kd) so they settle to a lower-effort
        #     posture; the first real motion ramps stiffness back fast.
        #   * thermal governor: if a shoulder-pitch motor nears its limit, a
        #     capped backoff is applied even while teleoperating, so the joint
        #     stays below the firmware thermal-fault temperature instead of
        #     collapsing. Only the arms are affected; torso/legs stay rigid.
        # The actual stiffness backoff is expressed as a single arm_damp_blend
        # in [0, 1] (0 = full stiffness, 1 = pure damping) ramped smoothly here.
        self._eco_enabled = threading.Event()
        if getattr(args, "eco_mode", False):
            self._eco_enabled.set()
        self._eco_blend = 0.0           # currently applied backoff (0..1)
        self._eco_last_motion_t = 0.0   # last time the arm goals moved
        self._eco_prev_goal = None      # last (left,right) goal for motion calc
        self._eco_last_temp = None      # cached hottest shoulder-pitch temp
        self._eco_last_temp_t = 0.0
        self._eco_relax_logged = False  # rate-limit idle-relax log
        self._eco_gov_logged = False    # rate-limit governor log
        # Shoulder-pitch motor indices (first joint of each arm). Used by the
        # thermal governor to watch the joints that carry the gravity load.
        if hw_controller is not None:
            self._eco_sp_idx = [hw_controller._left_idx[0], hw_controller._right_idx[0]]
        else:
            self._eco_sp_idx = []

        # ── load-bearing-joint thermal governor + position-error clamp ────────
        # The kneeling hip/knee/waist-pitch hold the whole body against gravity for the
        # entire session with NO thermal protection (the eco governor above is arms-only).
        # A 30-min static hold overheats them → firmware torque derating → the joint sags
        # → and because the command is a FIXED angle at high kp, the accumulated position
        # error snaps back VIOLENTLY the instant torque returns (this broke the neck). Two
        # defenses, both applied in _send_hardware:
        #   1. POSITION-ERROR CLAMP (hard guarantee): the leg/torso command may never lead
        #      the MEASURED joint by more than a few degrees, so the corrective torque is
        #      bounded (kp·Δ) and a violent snap is physically impossible — recovery is a
        #      smooth ramp, not an impulse. A no-op in normal (small-error) operation.
        #   2. THERMAL GOVERNOR (mitigation): as the hip-pitch/waist-pitch heat up, trim
        #      THOSE joints' kp (capped, knee kept firm) so they yield gently instead of
        #      fighting to the derate point; paired with the clamp so any yield is smooth.
        self._pos_err_clamp_on = not bool(getattr(args, "no_pos_err_clamp", False))
        self._leg_gov_on = bool(getattr(args, "leg_thermal_governor", True))
        self._leg_thermal_blend = 0.0
        self._leg_gov_logged = False
        self._load_last_temp = None
        self._load_last_temp_t = 0.0
        # Waist-lock mode: a mechanical lock holds the waist, so the waist MOTORS should not
        # be driven — commanding them to a trim they can't reach just stalls them against the
        # lock (wasted current → heat + stress on the lock/gearbox). When locked, idle them
        # (kp→waist_locked_kp_scale, kd damping only) so the lock does the holding, and take
        # the waist OUT of the leg thermal governor (it's mechanically handled now — the
        # governor focuses on the leg hip-pitch, the joint that actually carries + can snap).
        self._waist_locked = bool(getattr(args, "waist_locked", True))
        self._waist_locked_kp_scale = float(getattr(args, "waist_locked_kp_scale", 0.0))
        # The mechanical waist lock holds the load-bearing PITCH (and the lateral ROLL trim)
        # for the kneel, but NOT the yaw. Keep the yaw MOTOR DRIVEN so the upper body is
        # actively held facing forward (yaw goal = 0): idling it (the old all-3 behaviour)
        # left the waist free to rotate side-to-side during teleop and never corrected a
        # slight boot misalignment. Use --no-waist-lock-drive-yaw for a lock that rigidly
        # holds the yaw as well (restores the old idle-all-3 behaviour).
        self._waist_locked_drive_yaw = bool(getattr(args, "waist_lock_drive_yaw", True))
        if hw_controller is not None:
            _leg = hw_controller.config.leg_joint2motor_idx      # [L:hp,hr,hy,kn,ap,ar, R:…]
            self._hip_pitch_idx = [_leg[0], _leg[6]]
            self._knee_idx = [_leg[3], _leg[9]]
            _torso = list(hw_controller._torso_idx)                # order [yaw, roll, pitch]
            self._waist_pitch_idx = [_torso[2]]
            # Motors IDLED against the lock: roll+pitch (the locked axes) with the yaw kept
            # driven, or all three for a full-waist lock (--no-waist-lock-drive-yaw).
            self._waist_idx = ([_torso[1], _torso[2]] if self._waist_locked_drive_yaw
                               else list(_torso))
            # The yaw is force-held forward only when it is a DRIVEN (non-idled) axis.
            self._waist_yaw_driven = (self._waist_locked and self._waist_locked_drive_yaw)
            if self._waist_locked:
                # Waist is mechanically held → don't watch/soften it; focus on the leg hip.
                self._load_watch_idx = self._hip_pitch_idx + self._knee_idx
                self._load_relax_idx = self._hip_pitch_idx
            else:
                # Joints WATCHED for heat (warn + governor input): the pitch joints that carry
                # the gravity load, plus the knees (for the warning only).
                self._load_watch_idx = self._hip_pitch_idx + self._knee_idx + self._waist_pitch_idx
                # Joints the governor SOFTENS — hip-pitch + waist-pitch. The knee is left
                # firm: softening it risks the kneel collapsing.
                self._load_relax_idx = self._hip_pitch_idx + self._waist_pitch_idx
        else:
            self._hip_pitch_idx = self._knee_idx = self._waist_pitch_idx = []
            self._waist_idx = []
            self._waist_yaw_driven = False
            self._load_watch_idx = self._load_relax_idx = []

        # ── current-triggered arm compliance ──────────────────────────────────
        # For collaborative manipulation (two robots on one crate): when a partner
        # forces the crate and an arm joint's |tau_est| (a current proxy) climbs
        # toward the overcurrent limit, soften THAT joint (kp shrinks, damping
        # grows) so it yields with the crate instead of fighting and tripping the
        # overcurrent fault. When the spike passes, the joint re-stiffens and the
        # position target pulls it back to the operator's matched pose. Per-joint,
        # so only the loaded joints give. Off unless enabled.
        self._comp_enabled = threading.Event()
        if getattr(args, "arm_compliance", False):
            self._comp_enabled.set()
        if hw_controller is not None:
            self._comp_arm_idx = list(hw_controller._left_idx) + list(hw_controller._right_idx)
        else:
            self._comp_arm_idx = []
        self._comp_blend = {i: 0.0 for i in self._comp_arm_idx}  # motor_idx -> blend
        self._comp_active_logged = False

        # Transition blend state — written from main thread, read here
        self._blend_lock = threading.Lock()
        self._teleop_start_time = 0.0
        self._q_arm_start_torso = np.zeros(3)
        self._q_arm_start_left  = _ARMS_AT_SIDE_LEFT.copy()
        self._q_arm_start_right = _ARMS_AT_SIDE_RIGHT.copy()
        # Whether the opening transition runs the full staged choreography (lateral
        # raise -> back -> chicken wing -> operator). True for an initial start;
        # False on RESUME from pause, where the arms are already up — then it just
        # blends directly from the held pose to the operator's live pose.
        self._teleop_staged = True
        self._teleop_blend_dur = 0.0

        # Latest goal buffer — sent to hardware every loop iteration AND read by
        # the Qt thread for sim visualisation. Initialised to the rest pose so
        # the watchdog-safe send holds arms-at-side while idle.
        self._goals_lock = threading.Lock()
        self._q_goal_torso = np.zeros(3)
        self._q_goal_left  = _ARMS_AT_SIDE_LEFT.copy()
        self._q_goal_right = _ARMS_AT_SIDE_RIGHT.copy()

        # Persistent last-good arm IK result. Read straight from the solver
        # output (not from the shared controller, which the Qt viz thread also
        # writes) so the blend can't race; holds the previous pose when a frame
        # has no IK solution — same persistence the demo gets from the controller.
        self._last_arm_left  = _ARMS_AT_SIDE_LEFT.copy()
        self._last_arm_right = _ARMS_AT_SIDE_RIGHT.copy()

        # Pose-arrival tracking for latency measurement
        self._last_pose_seq = 0
        self._last_pose_recv_t = 0.0
        # Stale-pose failsafe state (see _stale_pose_guard).
        self._pose_stale = False

    # ── public API ────────────────────────────────────────────────────────────

    def start_teleop(self, q_arm_start_torso, q_arm_start_left, q_arm_start_right,
                     staged=True):
        """Enable IK ticking and set the blend-from pose for the transition ramp.

        staged=True  -> initial start: full lateral-raise/back/chicken-wing/forward
                        choreography over transition_duration.
        staged=False -> resume from pause: a short DIRECT blend from the held pose
                        straight to the operator's live pose (no back-swing).
        """
        with self._blend_lock:
            self._q_arm_start_torso = np.asarray(q_arm_start_torso).copy()
            self._q_arm_start_left  = np.asarray(q_arm_start_left).copy()
            self._q_arm_start_right = np.asarray(q_arm_start_right).copy()
        self._teleop_staged = bool(staged)
        _td = float(self._args.transition_duration)
        # Resume blends quickly and directly; initial start uses the full duration.
        self._teleop_blend_dur = _td if staged else min(_td, 1.5)
        self._teleop_start_time = time.time()
        # Treat teleop start as fresh motion so eco does not relax the arms
        # during the opening transition blend.
        self._eco_last_motion_t = time.monotonic()
        self._eco_prev_goal = None
        self._arm_damp.clear()
        self._teleop_active.set()

    def stop_teleop(self):
        """Pause IK ticking (thread keeps running and keeps sending the hold pose)."""
        self._teleop_active.clear()

    def enter_arm_damping(self):
        """Stop teleop and put the arms into damping mode.

        Pauses IK and streams the arms with kp=0/kd>0 so they slowly lower
        under gravity without applying force against any object, while the
        torso and legs stay rigid (normal position gains). Equivalent to
        damp mode for the arms only.
        """
        self._teleop_active.clear()
        self._arm_damp.set()

    def exit_arm_damping(self):
        """Leave arm-damping mode (arms resume normal position control)."""
        self._arm_damp.clear()

    def set_neck_active(self, active: bool):
        """Enable/disable neck-only head tracking (independent of teleop).

        When enabled, the loop drives the Ostrich neck from the operator's head pose even
        while teleop is paused/idle, so the operator can look around with the camera feed
        before starting teleop. Has no effect on the arms. Tied to the XR streaming feed."""
        if active:
            self._neck_active.set()
        else:
            self._neck_active.clear()

    def set_eco_enabled(self, enabled: bool):
        """Enable/disable eco mode (idle auto-relax + thermal governor).

        Affects only the arms while teleoperating; it is a no-op whenever the
        operator is actively moving and a motor is below the derate temperature.
        """
        if enabled:
            # Reset motion timing so toggling on mid-hold does not instantly
            # relax: give the operator the full idle window first.
            self._eco_last_motion_t = time.monotonic()
            self._eco_enabled.set()
        else:
            self._eco_enabled.clear()

    def eco_status(self) -> tuple:
        """Return (enabled, applied_blend, hottest_shoulder_temp_or_None)."""
        return (self._eco_enabled.is_set(), float(self._eco_blend),
                self._eco_last_temp)

    def set_compliance_enabled(self, enabled: bool):
        """Enable/disable current-triggered arm compliance (collaborative manip)."""
        if enabled:
            self._comp_enabled.set()
        else:
            self._comp_enabled.clear()

    def compliance_status(self) -> tuple:
        """Return (enabled, max_applied_blend across arm joints)."""
        mb = max(self._comp_blend.values()) if self._comp_blend else 0.0
        return (self._comp_enabled.is_set(), float(mb))

    def set_hold_goals(self, q_torso, q_left, q_right):
        """Set the goal buffer directly (used by the lowering ramp / idle hold).

        The loop keeps streaming these to hardware at ~max_fr Hz so the firmware
        watchdog stays satisfied even when IK is not running.
        """
        with self._goals_lock:
            self._q_goal_torso = np.asarray(q_torso, dtype=np.float64).copy()
            self._q_goal_left  = np.asarray(q_left,  dtype=np.float64).copy()
            self._q_goal_right = np.asarray(q_right, dtype=np.float64).copy()

    def set_kneeling_pose(self, legs, torso_hold):
        """Update the locked kneeling leg angles and the held waist angle.

        Called live by the kneeling-pose calibration window. ``legs`` is the
        full 12-element leg array (or None when not kneeling); ``torso_hold`` is
        the 3-element waist hold [yaw, roll, pitch]. The waist hold is pushed
        into the goal buffer too so the trim takes effect immediately, even when
        idle (not actively teleoperating).
        """
        self._legs = None if legs is None else np.asarray(legs, dtype=np.float64).copy()
        th = np.asarray(torso_hold, dtype=np.float64).copy()
        with self._goals_lock:
            self._torso_hold = th
            self._q_goal_torso = th.copy()

    def set_abduction_limits(self, left_max_rad, right_max_rad):
        """Set the per-arm maximum outward abduction (shoulder roll) in radians
        (positive magnitude). Applied live to clamp teleop. None/inf = no limit."""
        self._abd_left_max  = np.inf if left_max_rad  is None else float(left_max_rad)
        self._abd_right_max = np.inf if right_max_rad is None else float(right_max_rad)

    def _clamp_abduction(self, q_l, q_r):
        """Clamp shoulder-roll (DOF index 1) so the arms cannot abduct past the
        configured limit. Left abducts with +roll, right with -roll. Modifies the
        arrays in place and returns them. Adduction is never restricted."""
        if q_l is not None and len(q_l) > 1 and q_l[1] > self._abd_left_max:
            q_l[1] = self._abd_left_max
        if q_r is not None and len(q_r) > 1 and q_r[1] < -self._abd_right_max:
            q_r[1] = -self._abd_right_max
        return q_l, q_r

    def _arm_target_behind_head(self, side, q_arm):
        """True if the arm config ``q_arm`` (7-DOF, solver order) would place the robot hand in
        the behind-the-head keep-out: FK the hand into the pelvis frame and test the slab
        (x < x_back AND z > z_head). FK failures never block (returns False)."""
        if not self._head_keepout_on or q_arm is None or self._solver is None:
            return False
        try:
            p = self._solver.arm_wrist_position(side, np.asarray(q_arm, dtype=np.float64))
        except Exception:                                 # noqa: BLE001 — never block on FK error
            return False
        return bool(p[0] < self._head_keepout_x and p[2] > self._head_keepout_z)

    def get_goals(self) -> tuple:
        """Return latest (q_torso, q_left, q_right) for sim visualisation."""
        with self._goals_lock:
            return (self._q_goal_torso.copy(),
                    self._q_goal_left.copy(),
                    self._q_goal_right.copy())

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1.5)

    # ── main loop ─────────────────────────────────────────────────────────────

    def _actuation_loop(self):
        """Estimate robot actuation latency OFF the teleop hot path. Samples commanded vs actual arm
        joints at a fixed rate and refreshes the cross-correlation estimate ~2 Hz. Runs on its own
        daemon thread so the motor reads + numpy never stall the IK / 500 Hz hardware-send loop."""
        dt = 1.0 / self._actuation_sample_hz
        calc_every = max(1, int(self._actuation_sample_hz / 2.0))   # ~2 Hz estimate refresh
        n = 0
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                with self._goals_lock:
                    cmd = np.concatenate((self._q_goal_left, self._q_goal_right))
                ms = self._hw_controller.low_state.motor_state
                act = np.array([ms[i].q for i in self._hw_controller._left_idx]
                               + [ms[i].q for i in self._hw_controller._right_idx], dtype=np.float64)
                self._actuation_est.add(cmd, act)
                n += 1
                if n >= calc_every:
                    n = 0
                    est = self._actuation_est.estimate_ms()
                    if est is not None:
                        self._last_actuation_ms = est
            except Exception:
                pass
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    def run(self):
        # Off-thread actuation-latency sampler — kept entirely off this hot loop so the latency
        # telemetry can never add IK / hardware-send latency.
        if self._actuation_est is not None and self._hw_controller is not None:
            threading.Thread(target=self._actuation_loop, daemon=True,
                             name="actuation-latency").start()

        # Mirror the demo's single teleop loop: send rt/lowcmd every iteration,
        # solve IK gated at ik_rate Hz. The send loop runs at 500 Hz — the proven
        # watchdog-safe rate (same as the previous dedicated 500 Hz sender) — which
        # leaves ample GIL headroom for the gated IK solve in the same thread.
        send_dt = 1.0 / 500.0
        ik_dt   = 1.0 / max(1, self._args.ik_rate)
        last_ik = 0.0
        last_eco = time.monotonic()

        while not self._stop_evt.is_set():
            t_start = time.monotonic()

            if self._teleop_active.is_set():
                now = time.monotonic()
                if now - last_ik >= ik_dt:
                    self._do_ik()
                    last_ik = now
            elif self._neck_active.is_set():
                # Camera feed up but teleop not started: track the head on the neck only
                # (no full-body IK, arms stay in damping) so the operator can look around.
                now = time.monotonic()
                if now - last_ik >= ik_dt:
                    self._do_neck_only()
                    last_ik = now

            # Advance the eco backoff (idle relax + thermal governor) every loop
            # so it ramps smoothly at the 500 Hz send rate, independent of IK.
            now = time.monotonic()
            self._update_eco_blend(now, now - last_eco)
            # Advance the current-triggered arm compliance at the same rate.
            self._update_compliance(now, now - last_eco)
            # Advance the load-bearing-joint (hip/waist pitch) thermal governor.
            self._update_leg_thermal(now, now - last_eco)
            last_eco = now

            # Always stream the latest goal buffer to hardware (watchdog-safe),
            # whether idle (hold pose), lowering, or teleop (IK goals).
            self._send_hardware()

            tick_ms = (time.monotonic() - t_start) * 1000
            self._tracker.add_ik_period(tick_ms)

            remaining = send_dt - tick_ms / 1000
            if remaining > 0:
                time.sleep(remaining)

    def _send_hardware(self):
        """Send the current goal buffer to the G1. Runs every loop iteration."""
        if self._hw_controller is None:
            return
        with self._goals_lock:
            q_t = self._q_goal_torso.copy()
            q_l = self._q_goal_left.copy()
            q_r = self._q_goal_right.copy()
        # Hard safety clamp on outward abduction at the single point every arm
        # command passes through, regardless of source (IK, hold, blend).
        self._clamp_abduction(q_l, q_r)

        # Waist-lock: force the DRIVEN yaw goal to face forward (0) so the upper body is held
        # square regardless of how the robot was aligned at boot. Applied BEFORE the torso
        # clamp below, so a boot yaw offset is walked out as a smooth ramp (≤ clamp/tick),
        # never a snap. (Roll/pitch stay at their held trim; those motors are idle-locked.)
        if getattr(self, "_waist_yaw_driven", False):
            q_t[0] = 0.0

        # ── SAFETY: bound the position error on the load-bearing legs + torso ──
        # The command may never lead the MEASURED joint by more than a few degrees, so the
        # corrective torque stays bounded (≈kp·Δ) and a thermally-sagged joint can NEVER
        # snap back violently — recovery is a smooth ramp. This is a no-op in normal
        # operation (error ≪ Δ), so teleop tracking is unaffected. Also builds the thermal
        # kp trim map for the hip/waist pitch. Both read the live measured pose once.
        legs_cmd = self._legs
        kp_map = None
        if self._hw_controller is not None:
            try:
                ms = self._hw_controller.low_state.motor_state
                if self._pos_err_clamp_on:
                    d_tor = float(getattr(self._args, "torso_pos_err_clamp", 0.2))
                    if d_tor > 0.0:
                        t_idx = self._hw_controller._torso_idx
                        qt_meas = np.array([ms[i].q for i in t_idx], dtype=np.float64)
                        q_t = qt_meas + np.clip(q_t - qt_meas, -d_tor, d_tor)
                    d_leg = float(getattr(self._args, "leg_pos_err_clamp", 0.2))
                    if self._legs is not None and d_leg > 0.0:
                        leg_idx = self._hw_controller.config.leg_joint2motor_idx
                        ql_meas = np.array([ms[i].q for i in leg_idx], dtype=np.float64)
                        legs_cmd = ql_meas + np.clip(np.asarray(self._legs) - ql_meas,
                                                     -d_leg, d_leg)
                kp_map = self._build_kp_scale_map()
            except Exception:
                legs_cmd, kp_map = self._legs, None
        # Resolve the arm stiffness backoff. A full-damp request (pre/post
        # teleop) always wins; otherwise eco's partial backoff (idle relax or
        # thermal governor) applies. blend=0 => normal position control.
        if self._arm_damp.is_set():
            arm_damping, arm_damp_blend = True, 1.0
        elif self._eco_blend > 1e-3:
            arm_damping, arm_damp_blend = True, float(self._eco_blend)
        else:
            arm_damping, arm_damp_blend = False, 0.0
        try:
            self._hw_controller.send_upper_body_targets(
                q_goal_torso=q_t,
                q_goal_left=q_l,
                q_goal_right=q_r,
                q_goal_legs=legs_cmd,
                kp_scale=self._args.hw_kp_scale,
                kd_scale=self._args.hw_kd_scale,
                leg_kp_scale=self._args.leg_kp_scale,
                leg_kd_scale=self._args.leg_kd_scale,
                ankle_kp_scale=0.0,
                ankle_kd_scale=1.0,
                arm_damping=arm_damping,
                arm_damp_kd=self._args.arm_damp_kd,
                arm_damp_blend=arm_damp_blend,
                # Per-joint current-triggered compliance (0 for every joint unless
                # engaged); combined with the uniform blend by max inside the send.
                arm_damp_blend_map=self._comp_blend,
                # Per-joint kp trim from the load-bearing thermal governor (hip/waist pitch).
                kp_scale_map=kp_map,
            )
        except Exception:
            pass

    def _read_shoulder_temp(self, now):
        """Hottest shoulder-pitch motor temperature (°C), throttled to ~5 Hz.

        Returns the cached value between reads, or None if unavailable."""
        if not self._eco_sp_idx or self._hw_controller is None:
            return None
        if now - self._eco_last_temp_t < 0.2:
            return self._eco_last_temp
        try:
            ms = self._hw_controller.low_state.motor_state
            self._eco_last_temp = max(_hg_temp(ms[i].temperature) for i in self._eco_sp_idx)
        except Exception:
            return self._eco_last_temp
        self._eco_last_temp_t = now
        return self._eco_last_temp

    def _update_eco_blend(self, now, dt):
        """Compute and ramp the arm stiffness backoff for eco mode.

        Two independent demands, combined by taking the larger:
          * idle relax: full eco_relax_blend once the arm goals have been
            static for eco_idle_secs.
          * thermal governor: a temperature-proportional backoff (capped at
            eco_temp_max_blend) as a shoulder-pitch motor crosses eco_temp_derate
            heading toward eco_temp_full.
        The backoff ramps in slowly (eco_relax_in_secs) so the arms settle
        gently, and ramps out fast (eco_relax_out_secs) so the first real motion
        re-stiffens the arms almost immediately. While disabled, or not actively
        teleoperating, the target is 0 and the arms return to full stiffness.
        """
        dt = max(0.0, min(dt, 0.1))  # clamp to ignore scheduler hiccups
        a = self._args
        target = 0.0
        relax_active = False
        gov_temp = None
        if (self._eco_enabled.is_set() and self._teleop_active.is_set()
                and not self._arm_damp.is_set()):
            # Idle auto-relax.
            if now - self._eco_last_motion_t > a.eco_idle_secs:
                target = max(target, float(a.eco_relax_blend))
                relax_active = True
            # Thermal governor.
            t = self._read_shoulder_temp(now)
            if t is not None and a.eco_temp_full > a.eco_temp_derate:
                frac = (t - a.eco_temp_derate) / (a.eco_temp_full - a.eco_temp_derate)
                frac = min(1.0, max(0.0, frac))
                gov_blend = frac * float(a.eco_temp_max_blend)
                if gov_blend > target:
                    target, gov_temp = gov_blend, t

        # Asymmetric ramp: slow into backoff, fast back to full stiffness.
        if target > self._eco_blend:
            rate = 1.0 / max(1e-3, a.eco_relax_in_secs)
        else:
            rate = 1.0 / max(1e-3, a.eco_relax_out_secs)
        step = rate * dt
        self._eco_blend += max(-step, min(step, target - self._eco_blend))
        self._eco_blend = min(1.0, max(0.0, self._eco_blend))

        # Rate-limited operator feedback (thread-side, like other ControlThread logs).
        if relax_active and not self._eco_relax_logged:
            print("[eco] arms idle — relaxing to save shoulder-pitch heat "
                  "(will re-engage on motion)")
            self._eco_relax_logged = True
        elif not relax_active and self._eco_relax_logged:
            self._eco_relax_logged = False
        if gov_temp is not None and not self._eco_gov_logged:
            print(f"[eco] THERMAL GOVERNOR engaged: shoulder pitch {gov_temp:.0f} °C "
                  f"— softening arms to stay below the {a.eco_temp_full:.0f} °C fault")
            self._eco_gov_logged = True
        elif gov_temp is None and self._eco_gov_logged:
            print("[eco] thermal governor released (shoulder pitch cooled)")
            self._eco_gov_logged = False

    def _build_kp_scale_map(self):
        """Per-motor kp trim for send_upper_body_targets. Idles the waist motors when a
        mechanical lock is installed (so they don't stall against it) and applies the leg
        thermal governor's backoff to the hip-pitch. Returns None when there's nothing to
        trim (so the send path stays on its normal uniform-gain path)."""
        kp_map = {}
        if self._waist_locked and self._waist_idx:
            for i in self._waist_idx:
                kp_map[i] = self._waist_locked_kp_scale
        if self._leg_thermal_blend > 1e-3 and self._load_relax_idx:
            s = 1.0 - float(self._leg_thermal_blend)
            for i in self._load_relax_idx:
                kp_map[i] = s
        return kp_map or None

    def _read_load_temp(self, now):
        """Hottest load-bearing-joint (hip-pitch/knee/waist-pitch) temperature (°C),
        throttled to ~5 Hz. Returns the cached value between reads, or None."""
        if not self._load_watch_idx or self._hw_controller is None:
            return None
        if now - self._load_last_temp_t < 0.2:
            return self._load_last_temp
        try:
            ms = self._hw_controller.low_state.motor_state
            self._load_last_temp = max(_hg_temp(ms[i].temperature) for i in self._load_watch_idx)
        except Exception:
            return self._load_last_temp
        self._load_last_temp_t = now
        return self._load_last_temp

    def _update_leg_thermal(self, now, dt):
        """Temperature-proportional, CAPPED kp backoff for the hip-pitch/waist-pitch as the
        kneeling load joints heat up — the leg/torso analogue of the arm eco governor. Only
        active while teleoperating. Ramps smoothly (shared eco ramp rates) and never exceeds
        leg_temp_max_blend (kept small — over-softening a load joint risks collapse; the knee
        is never touched). The position-error clamp in _send_hardware keeps any resulting
        yield smooth. Warns the operator, escalating near the fault temperature."""
        dt = max(0.0, min(dt, 0.1))
        # PAUSED but still holding the kneel (teleop stopped, NOT lowering/damping): the load
        # joints (hip/waist pitch) carry the identical static load as during teleop, so releasing
        # the softening HERE re-stiffens the thermally-sagged joints and snaps the whole posture
        # up — the "knees lose power, fall back, then violently jump up" seen when pausing on a
        # stopped attempt. FREEZE the current derate instead; it is re-evaluated smoothly (under
        # the position-error clamp) once teleop resumes. Lowering (arm damping) still releases it
        # via the normal path below, because the robot is coming down out of the kneel then.
        if not self._teleop_active.is_set() and not self._arm_damp.is_set():
            return
        a = self._args
        target = 0.0
        gov_temp = None
        if (self._leg_gov_on and self._teleop_active.is_set() and self._load_relax_idx):
            t = self._read_load_temp(now)
            derate = float(getattr(a, "leg_temp_derate", 60.0))
            full = float(getattr(a, "leg_temp_full", 75.0))
            if t is not None and full > derate:
                frac = min(1.0, max(0.0, (t - derate) / (full - derate)))
                target = frac * float(getattr(a, "leg_temp_max_blend", 0.25))
                if target > 1e-3:
                    gov_temp = t
        # Same asymmetric ramp as the arm eco governor (slow in, fast out).
        if target > self._leg_thermal_blend:
            rate = 1.0 / max(1e-3, float(getattr(a, "eco_relax_in_secs", 1.2)))
        else:
            rate = 1.0 / max(1e-3, float(getattr(a, "eco_relax_out_secs", 0.3)))
        step = rate * dt
        self._leg_thermal_blend += max(-step, min(step, target - self._leg_thermal_blend))
        self._leg_thermal_blend = min(1.0, max(0.0, self._leg_thermal_blend))
        if gov_temp is not None and not self._leg_gov_logged:
            print(f"[leg-thermal] GOVERNOR engaged: load joint {gov_temp:.0f} °C — softening "
                  f"hip/waist pitch (capped) to stay below the {float(getattr(a,'leg_temp_full',75.0)):.0f} °C "
                  "derate. STOP soon and let it cool — a long static kneel overheats these joints.")
            self._leg_gov_logged = True
        elif gov_temp is None and self._leg_gov_logged:
            print("[leg-thermal] governor released (load joints cooled).")
            self._leg_gov_logged = False

    def _update_compliance(self, now, dt):
        """Ramp the per-arm-joint compliance blend from each joint's measured
        |tau_est| (a current proxy). Above comp_tau_start a joint softens
        proportionally, reaching comp_max_blend by comp_tau_full, so it yields to
        an external force before hitting the overcurrent limit; below the band it
        re-stiffens and the position target returns it to the operator's pose.
        Ramps in fast (yield before a trip) and out slower (return smoothly).
        Only while actively teleoperating and not already in full arm-damp."""
        dt = max(0.0, min(dt, 0.1))
        a = self._args
        active = (self._comp_enabled.is_set() and self._teleop_active.is_set()
                  and self._hw_controller is not None and not self._arm_damp.is_set())
        taus = None
        if active:
            try:
                ms = self._hw_controller.low_state.motor_state
                taus = {i: abs(float(ms[i].tau_est)) for i in self._comp_arm_idx}
            except Exception:
                taus = None
        span = max(1e-3, float(a.comp_tau_full) - float(a.comp_tau_start))
        engaged = False
        for i in self._comp_arm_idx:
            if taus is not None:
                frac = (taus[i] - float(a.comp_tau_start)) / span
                frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
                target = frac * float(a.comp_max_blend)
            else:
                target = 0.0
            cur = self._comp_blend[i]
            rate = 1.0 / max(1e-3, a.comp_in_secs if target > cur else a.comp_out_secs)
            step = rate * dt
            cur += max(-step, min(step, target - cur))
            cur = 0.0 if cur < 0.0 else (1.0 if cur > 1.0 else cur)
            self._comp_blend[i] = cur
            if cur > 0.05:
                engaged = True
        # Rate-limited operator feedback: edge-triggered AND at most one line / 5 s —
        # oscillating around the threshold (e.g. holding a heavy crate, or arms pinned
        # on a stale pose) used to print hundreds of identical lines per minute.
        if engaged and not self._comp_active_logged:
            if now - getattr(self, "_comp_log_t", 0.0) > 5.0:
                self._comp_log_t = now
                print("[compliance] arm forced near overcurrent — softening to yield "
                      "(re-stiffens and returns when the force passes)")
            self._comp_active_logged = True
        elif not engaged and self._comp_active_logged:
            self._comp_active_logged = False

    def _stale_pose_guard(self, recv_t, now):
        """Operator-pose staleness FAILSAFE. Returns True when _do_ik must SKIP applying
        goals this tick (the 500 Hz sender then keeps the last commanded pose = HOLD).

        Without this, a frozen pose stream (Quest hiccup / WebRTC stall / WiFi drop) makes
        the arms chase a stale target indefinitely — mid-lift that reads as 'lost control'
        (constant compliance/overcurrent as the operator moves but the robot stays pinned)
        — and when the stream returns the arms SNAP to wherever the operator now is. Here:
        stale > --stale-pose-secs → hold + log; on recovery → re-enter through the same
        short DIRECT blend as a resume-from-pause, so the arms sweep (not snap) to the
        live pose."""
        if recv_t <= 0:
            return False                              # no pose ever received yet
        thresh = float(getattr(self._args, "stale_pose_secs", 0.5))
        age = now - recv_t
        if age > thresh:
            if not self._pose_stale:
                self._pose_stale = True
                print(f"[safety] operator pose STALE ({age:.2f}s > {thresh:.2f}s) — "
                      "HOLDING pose until the stream recovers (arms will blend, not "
                      "snap, when it does).")
            return True
        if self._pose_stale:
            self._pose_stale = False
            with self._goals_lock:
                cur_t = self._q_goal_torso.copy()
                cur_l = self._q_goal_left.copy()
                cur_r = self._q_goal_right.copy()
            # Same machinery as resume-from-pause: short direct blend held → live.
            self.start_teleop(cur_t, cur_l, cur_r, staged=False)
            print("[safety] operator pose stream RECOVERED — blending back to the "
                  "live pose.")
        return False

    def set_disabled_hand_dofs(self, left_disabled, right_disabled):
        """Set which hand actuator indices to STOP commanding (held at rest). Atomic replace so
        _do_ik reads a consistent mask. Index order per hand: [pinky, ring, middle, index,
        thumb-bend, thumb-rotation]."""
        self._hand_dof_mask = (frozenset(int(i) for i in left_disabled),
                               frozenset(int(i) for i in right_disabled))

    def _apply_hand_dof_mask(self, goals):
        """Freeze any DISABLED hand DOFs at the hold position, in place on the solver goals, BEFORE
        they fan out to the sim controller (→ recorded finger qpos → digital twin) and the hand
        hardware. So a broken actuator is never driven and the twin matches the real hand."""
        left_dis, right_dis = self._hand_dof_mask
        if left_dis and goals.get("q_goal_left_hand") is not None:
            g = goals["q_goal_left_hand"]
            for i in left_dis:
                if 0 <= i < len(g):
                    g[i] = self._hand_dof_hold
        if right_dis and goals.get("q_goal_right_hand") is not None:
            g = goals["q_goal_right_hand"]
            for i in right_dis:
                if 0 <= i < len(g):
                    g[i] = self._hand_dof_hold

    def _do_ik(self):
        """One IK solve + goal update. Gated at ik_rate Hz. Mirrors the demo."""
        t0 = time.monotonic()

        action = self._device.get_controller_state()
        t1 = time.monotonic()
        if action is None:
            return

        # Track pose-packet arrival interval and check for staleness.
        arrival_t = 0.0
        state_inst = self._device.state_factory.instance
        if state_inst is not None:
            seq, recv_t = state_inst.get_pose_seq()
            arrival_t = state_inst.get_arrival_t()

            # Full pipeline age: from WebRTC packet arrival to now.
            now = time.monotonic()
            pipeline_ms = (now - arrival_t) * 1000 if arrival_t > 0 else 0.0
            # BoneProcessor lag: how long bone processing took.
            bone_proc_ms = (recv_t - arrival_t) * 1000 if arrival_t > 0 and recv_t >= arrival_t else 0.0
            # Staleness since bone processing finished.
            stale_ms = (now - recv_t) * 1000

            if pipeline_ms > 300 and arrival_t > 0:
                if now - getattr(self, '_last_stale_warn', 0) > 2.0:
                    print(
                        f"[ControlThread] LATENCY: pipeline={pipeline_ms:.0f} ms  "
                        f"(bone_proc={bone_proc_ms:.0f} ms  stale_after={stale_ms:.0f} ms)"
                    )
                    self._last_stale_warn = now

            if seq != self._last_pose_seq:
                if self._last_pose_seq > 0 and self._last_pose_recv_t > 0:
                    ivl_ms = (recv_t - self._last_pose_recv_t) * 1000
                    if 1.0 < ivl_ms < 500.0:   # discard outliers / first sample
                        self._tracker.add_pose_interval(ivl_ms)
                self._last_pose_seq = seq
                self._last_pose_recv_t = recv_t

            # FAILSAFE: frozen pose stream → hold (skip the solve entirely); on
            # recovery re-enter through a short direct blend instead of snapping.
            if self._stale_pose_guard(recv_t, now):
                return

        action['right_hka'] = None
        action['left_hka']  = None
        action['R_lower_upper'] = np.eye(3)

        solver_input = self._solver.preprocess_human_action(
            action,
            shoulder_width_scale=0.8,
            hip_width_scale=1.3,
            mocap_cartesian_scale=np.array([0.85, 0.85, 0.65]),
            mocap_offset=np.array([0, 0, 0.4]),
            R_mjworld_human=None,
        )
        t2 = time.monotonic()
        goals = self._solver.solve(solver_input, engaged=True,
                                   q_current_left=None, q_current_right=None)
        t3 = time.monotonic()
        # Freeze any operator-disabled hand DOFs (e.g. a sheared thumb-rotation shaft) BEFORE the
        # goals fan out — so both the sim/recording/twin AND the hand hardware stop moving it.
        self._apply_hand_dof_mask(goals)
        # Push full goals (arms + fingers) to the sim controller for visualisation.
        self._controller.set_joint_goals(goals)

        # Hands follow the IK target immediately (no blend) — matches the demo.
        if self._hand_controller is not None:
            self._hand_controller.set_joint_goals(goals)
            self._hand_controller.apply_control(engaged=True)

        q_head = solver_input.get('q_head')
        if self._neck_worker is not None and q_head is not None:
            self._neck_worker.send(q_head)   # non-blocking; serial I/O on NeckWorker thread

        # Hardware arm command: read straight from the solver output and hold the
        # last good pose on a frame with no solution (race-free vs. the Qt thread).
        # SAFETY keep-out: if a target would drive the hand behind the head, DON'T adopt it —
        # hold the last safe pose so the arm stops at the boundary instead of getting stuck back
        # there (e.g. an operator reaching up to adjust their headset mid-teleop).
        behind = False
        cand_l = np.asarray(goals["q_goal_left"], dtype=np.float64) \
            if goals.get("q_goal_left") is not None else None
        cand_r = np.asarray(goals["q_goal_right"], dtype=np.float64) \
            if goals.get("q_goal_right") is not None else None
        if cand_l is not None:
            if self._arm_target_behind_head('left', cand_l):
                behind = True
            else:
                self._last_arm_left = cand_l
        if cand_r is not None:
            if self._arm_target_behind_head('right', cand_r):
                behind = True
            else:
                self._last_arm_right = cand_r
        if behind and (t0 - self._head_keepout_warn_t) > 2.0:
            self._head_keepout_warn_t = t0
            print("[ControlThread] keep-out: hand target behind the head — holding at the "
                  "boundary (bring the hands forward/down to resume).")
        q_g_l = self._last_arm_left.copy()
        q_g_r = self._last_arm_right.copy()

        # Eco idle-relax: track operator motion from the live IK arm targets.
        # If any arm joint moves faster than eco_motion_thresh (rad/s), stamp the
        # time; the eco loop relaxes the arms only after eco_idle_secs of no
        # motion. Uses the raw IK output (not the transition-blended goal) so it
        # reflects operator intent directly.
        if self._eco_enabled.is_set():
            cur = np.concatenate((q_g_l, q_g_r))
            if self._eco_prev_goal is not None and cur.shape == self._eco_prev_goal.shape:
                dt_ik = max(1e-3, t0 - getattr(self, "_eco_prev_goal_t", t0))
                vel = float(np.max(np.abs(cur - self._eco_prev_goal))) / dt_ik
                if vel > self._args.eco_motion_thresh:
                    self._eco_last_motion_t = time.monotonic()
            self._eco_prev_goal = cur
            self._eco_prev_goal_t = t0

        with self._goals_lock:
            q_g_t = self._torso_hold.copy()   # waist held at the (trimmed) kneel angle

        # Opening transition so the robot does not jerk to the operator's pose.
        #   INITIAL start (staged): the full choreography — a continuous smooth pass
        #     through start -> lateral raise -> back -> chicken wing for the first
        #     _RETRACT_FRAC of the duration, then a sweep from the wing into the
        #     operator's LIVE pose. Hands come up from behind, clear of the table.
        #   RESUME from pause (not staged): the arms are already up, so just blend
        #     DIRECTLY from the held pose to the operator's live pose — no back-swing.
        # The waist eases straight to its hold over the whole transition.
        td = self._teleop_blend_dur
        if td > 0:
            t_el = time.time() - self._teleop_start_time
            if t_el < td:
                with self._blend_lock:
                    start_t = self._q_arm_start_torso
                    start_l = self._q_arm_start_left
                    start_r = self._q_arm_start_right
                if not self._teleop_staged:
                    # Resume: direct smooth blend held-pose -> operator's live pose.
                    a = _smoothstep(t_el / td)
                    q_g_l = (1.0 - a) * start_l + a * q_g_l
                    q_g_r = (1.0 - a) * start_r + a * q_g_r
                else:
                    t_retract = td * _RETRACT_FRAC
                    if t_el < t_retract:
                        s = _smoothstep(t_el / t_retract)
                        q_g_l = _path_pose([start_l, _ARMS_ABDUCT_LEFT,  _ARMS_BEHIND_LEFT,  _ARMS_CHICKEN_LEFT],  s)
                        q_g_r = _path_pose([start_r, _ARMS_ABDUCT_RIGHT, _ARMS_BEHIND_RIGHT, _ARMS_CHICKEN_RIGHT], s)
                    else:
                        a = _smoothstep((t_el - t_retract) / max(1e-6, td - t_retract))
                        q_g_l = (1.0 - a) * _ARMS_CHICKEN_LEFT  + a * q_g_l
                        q_g_r = (1.0 - a) * _ARMS_CHICKEN_RIGHT + a * q_g_r
                q_g_t = (1.0 - _smoothstep(t_el / td)) * start_t + _smoothstep(t_el / td) * q_g_t

        # Clamp outward abduction so the shoulder heat sinks can't self-collide.
        # Done on the final (post-blend) goal so the buffer the sim viz and the
        # recorder read matches what is sent to hardware.
        self._clamp_abduction(q_g_l, q_g_r)

        with self._goals_lock:
            self._q_goal_torso = q_g_t
            self._q_goal_left  = q_g_l
            self._q_goal_right = q_g_r

        if self._haptics_streamer is not None:
            self._haptics_streamer.send_once()
        t4 = time.monotonic()

        self._tracker.add_ik_compute((t4 - t0) * 1000)

        # ── Teleop-latency telemetry (HOT PATH — keep trivial) ───────────────
        # Record ONLY manager processing = pose arrival → final goals committed (one subtraction).
        # The actuation estimate (motor reads + cross-correlation) is done on _actuation_loop, a
        # separate daemon thread, so it adds nothing to the IK / 500 Hz hardware-send loop.
        if arrival_t > 0:
            self._last_manager_proc_ms = (t4 - arrival_t) * 1000.0

        # Per-step timing: warn when any single step is unexpectedly slow.
        steps = {
            "get_state":  (t1 - t0) * 1000,
            "preprocess": (t2 - t1) * 1000,
            "solve_ik":   (t3 - t2) * 1000,
            "hand_neck":  (t4 - t3) * 1000,
        }
        for name, ms in steps.items():
            if ms > 100:
                _warn_t = getattr(self, f'_last_step_warn_{name}', 0)
                if t4 - _warn_t > 2.0:
                    print(f"[ControlThread] SLOW step '{name}': {ms:.0f} ms")
                    setattr(self, f'_last_step_warn_{name}', t4)

    def _do_neck_only(self):
        """Drive the Ostrich neck from the operator's head pose WITHOUT full-body IK.

        Runs while the camera feed is up but teleop has not started (look-around). Reads
        the head pose and computes q_head through the SAME preprocess path as _do_ik, then
        sends it to the neck worker — but skips the expensive IK solve and does NOT touch the
        arm/torso goal buffer, so the arms stay limp in damping. Mirrors _do_ik's head path."""
        if self._neck_worker is None:
            return
        action = self._device.get_controller_state()
        if action is None:
            return
        action['right_hka'] = None
        action['left_hka']  = None
        action['R_lower_upper'] = np.eye(3)
        solver_input = self._solver.preprocess_human_action(
            action,
            shoulder_width_scale=0.8,
            hip_width_scale=1.3,
            mocap_cartesian_scale=np.array([0.85, 0.85, 0.65]),
            mocap_offset=np.array([0, 0, 0.4]),
            R_mjworld_human=None,
        )
        q_head = solver_input.get('q_head')
        if q_head is not None:
            self._neck_worker.send(q_head)   # non-blocking; serial I/O on NeckWorker thread


# ============================================================================
# ForceCalibrationWindow — fingertip / palm force-sensor calibration
# ============================================================================

class _CalibTactileReader(threading.Thread):
    """Reads the full tip+nail+pad tactile map for the selected hand on a BACKGROUND
    thread and caches the latest snapshot. The calibration window's live grid + captures
    read the cache instead of doing the slow synchronous Modbus ``read_touch_full`` on the
    Qt GUI thread — which was blocking the GUI (~5 Hz in the live refresh, ~20 Hz during a
    capture) and making the whole app laggy. Shares the hand's io-lock with the live
    haptics reads, so the bus stays coherent."""

    def __init__(self, hand, side_getter, hz=10.0):
        super().__init__(daemon=True, name="calib-tactile")
        self._hand = hand
        self._side_getter = side_getter
        self._dt = 1.0 / max(1.0, float(hz))
        # NB: named _stop_evt, not _stop — threading.Thread has an internal _stop() method
        # that join()/shutdown call; shadowing it with an Event breaks thread cleanup.
        self._stop_evt = threading.Event()
        self._latest: dict = {}
        self._lock = threading.Lock()

    def run(self):
        while not self._stop_evt.is_set():
            side = self._side_getter()
            try:
                t = self._hand.read_touch_full(side)
            except Exception:
                t = None
            if t:
                with self._lock:
                    self._latest = t
            self._stop_evt.wait(self._dt)

    def latest(self) -> dict:
        with self._lock:
            return self._latest

    def clear(self):
        with self._lock:
            self._latest = {}

    def stop(self):
        self._stop_evt.set()


class ForceCalibrationWindow(QWidget):
    """Separate window for calibrating the Inspire tactile fingertip/palm force
    readings against known weights.

    The haptics use the RAW force-sensor readings directly. The only calibration is
    each region's baseline + idle NOISE FLOOR (used as a deadband to suppress phantom
    triggers) — there is no weight/grams calibration. One-click **Auto Calibrate** does
    it for both hands.

    Workflow:
      1. (Optional) raise the arms palms-up with teleop and *Freeze arms & open hands*.
      2. With nothing touching the sensors, press *Auto Calibrate* (HW-zero + baseline +
         noise floor for both hands, then Save & Apply) — or *Set baseline (unloaded)*
         for the selected hand only.

    Closing the window restores the low-latency tip+nail read mode and releases
    the arms to damping.
    """

    def __init__(self, hand_controller, control_thread, hw_controller,
                 log_fn, output_dir_getter, apply_fn):
        super().__init__()
        from projects.inspire_hand_teleop import inspire_fingertip_force_calibration as ifc

        self._ifc = ifc
        self._hand = hand_controller
        self._control_thread = control_thread
        self._hw = hw_controller
        self._log = log_fn
        self._output_dir_getter = output_dir_getter
        self._apply_fn = apply_fn
        self._side = "Left"
        self._arms_frozen = False   # True only after the user clicks Freeze
        # Load any existing calibration so points/baselines persist across runs.
        self._cal = ifc.ForceCalibration.load()
        # The window reads the pad via direct full-finger reads (read_touch_full),
        # so it never changes the live haptics_only mode or its latency.

        # Per-region rows shown in the live readout: each finger sub-region (tip/pad) + palm.
        self._targets: list[tuple[str, str, str]] = []
        for ik, hf in ifc.FINGER_MAP:
            if ik == "fingerpalm":
                self._targets.append((ik, ifc.PALM_SUBREGION, "Palm"))
            else:
                for sub in ifc.FINGER_SUBREGIONS:
                    self._targets.append(
                        (ik, sub, f"{hf.capitalize()} – {ifc.subregion_label(sub)}"))

        self._auto_running = False

        # Background tactile reader: keeps the slow synchronous Modbus read OFF the GUI
        # thread so the live grid + captures don't make the app laggy (see class docstring).
        self._reader = None
        if self._hand is not None:
            self._reader = _CalibTactileReader(self._hand, lambda: self._side, hz=10.0)
            self._reader.start()

        self.setWindowTitle("Inspire Hand — Fingertip / Palm Calibration")
        self.setMinimumWidth(680)
        self._build_ui()
        self._refresh_status_lbl()

        # Live readout refresh (~5 Hz) — reads the cached snapshot, not the Modbus bus.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_live)
        self._timer.start(200)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── One-click auto calibration (no weights) — top of the window ──────────
        self._btn_auto = QPushButton("⚡  Auto Calibrate (both hands — HW zero + baseline)")
        self._btn_auto.setStyleSheet("QPushButton { background: #2ecc71; color: white; "
                                     "font-weight: bold; padding: 10px; font-size: 13px; }")
        self._btn_auto.setToolTip(
            "One click, both hands, correct order: HW-zero (reg 1009) then set the baseline "
            "+ noise floor for each hand, then Save & Apply. Gives consistent, phantom-free "
            "haptics from the raw sensor readings. Keep BOTH hands OPEN and fully UNLOADED.")
        self._btn_auto.clicked.connect(self._on_auto_calibrate)
        root.addWidget(self._btn_auto)

        instr = QLabel(
            "Auto Calibrate (above) is all you need: with both hands open and unloaded it "
            "records each region's zero + idle noise floor and applies it, so the raw "
            "fingertip readings drive the haptics with phantom triggers gated out. The live "
            "readout below lets you confirm a region rises when you press it."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #444; font-size: 11px;")
        root.addWidget(instr)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #2c3e50; font-size: 11px; font-weight: bold;")
        root.addWidget(self._status_lbl)

        if self._hand is None:
            warn = QLabel("No hand hardware available (dry-run). Calibration disabled.")
            warn.setStyleSheet("color: #c0392b; font-weight: bold;")
            root.addWidget(warn)

        # ── Prepare / hand-hold row ─────────────────────────────────────────────
        prep_box = QGroupBox("Pose & hands")
        prep_row = QHBoxLayout(prep_box)
        self._btn_prepare = QPushButton("✋  Freeze arms & open hands")
        self._btn_prepare.clicked.connect(self._on_prepare)
        prep_row.addWidget(self._btn_prepare)
        self._btn_resume = QPushButton("↩  Release arms (damping)")
        self._btn_resume.clicked.connect(self._on_resume)
        prep_row.addWidget(self._btn_resume)
        root.addWidget(prep_box)

        # ── Hand selector + baseline row ────────────────────────────────────────
        sel_box = QGroupBox("Hand & baseline")
        sel_row = QHBoxLayout(sel_box)
        sel_row.addWidget(QLabel("Hand:"))
        self._rb_left = QRadioButton("Left")
        self._rb_right = QRadioButton("Right")
        self._rb_left.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._rb_left)
        grp.addButton(self._rb_right)
        self._rb_left.toggled.connect(self._on_side_changed)
        sel_row.addWidget(self._rb_left)
        sel_row.addWidget(self._rb_right)
        sel_row.addStretch()
        self._btn_baseline = QPushButton("⊘  Set baseline (unloaded)")
        self._btn_baseline.setToolTip("Average ~1 s of readings with nothing touching "
                                      "the sensors; subtracted from all captures.")
        self._btn_baseline.clicked.connect(self._on_baseline)
        sel_row.addWidget(self._btn_baseline)
        self._btn_hw_zero = QPushButton("HW zero (reg 1009)")
        self._btn_hw_zero.setToolTip("Optional: trigger the hand's built-in force "
                                     "calibration. Run only with the hand open and "
                                     "fully unloaded.")
        self._btn_hw_zero.clicked.connect(self._on_hw_zero)
        sel_row.addWidget(self._btn_hw_zero)
        self._btn_diagnose = QPushButton("🔬 Diagnose pad reads")
        self._btn_diagnose.setToolTip("Log per-finger array size and tip/nail/pad "
                                      "peaks for the selected hand. Press a pad while "
                                      "clicking to confirm it responds.")
        self._btn_diagnose.clicked.connect(self._on_diagnose)
        sel_row.addWidget(self._btn_diagnose)
        root.addWidget(sel_box)

        # ── Live readout grid ───────────────────────────────────────────────────
        live_box = QGroupBox("Live readout (selected hand)")
        live_grid = QGridLayout(live_box)
        live_grid.setSpacing(4)
        for col, head in enumerate(["Region", "Raw ADC", "Baseline", "Noise"]):
            lbl = QLabel(head)
            lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
            live_grid.addWidget(lbl, 0, col)
        self._live_labels: dict[tuple[str, str], dict[str, QLabel]] = {}
        for row, (ik, sub, disp) in enumerate(self._targets, start=1):
            name = QLabel(disp)
            raw = QLabel("—")
            base = QLabel("—")
            noise = QLabel("—")
            for c, w in enumerate([name, raw, base, noise]):
                w.setStyleSheet("font-size: 11px;")
                live_grid.addWidget(w, row, c)
            self._live_labels[(ik, sub)] = {"raw": raw, "base": base, "noise": noise}
        root.addWidget(live_box)

        # ── Save row ────────────────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._fit_lbl = QLabel("")
        self._fit_lbl.setStyleSheet("color: #2c3e50; font-size: 11px;")
        bottom.addWidget(self._fit_lbl)
        self._btn_fit = QPushButton("💾  Save & Apply")
        self._btn_fit.setToolTip("Persist the baseline + noise floors and apply them live.")
        self._btn_fit.setStyleSheet("QPushButton { background: #2ecc71; color: white; font-weight: bold; padding: 6px; }")
        self._btn_fit.clicked.connect(self._on_fit_save_apply)
        bottom.addWidget(self._btn_fit)
        root.addLayout(bottom)

        # Disable everything when there's no hand to read.
        if self._hand is None:
            for w in (self._btn_auto, self._btn_prepare, self._btn_resume, self._btn_baseline,
                      self._btn_hw_zero, self._btn_diagnose, self._btn_fit):
                w.setEnabled(False)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _side_letter(self) -> str:
        return "l" if self._side == "Left" else "r"

    def _read_full_touch(self) -> dict:
        """Latest full tip+nail+pad touch map for the selected hand — from the background
        reader's CACHE (no Modbus on the GUI thread). Returns ``{finger_key: np.ndarray}``
        (empty until the first read lands / on failure). Falls back to a direct read only
        if the reader isn't running."""
        if self._reader is not None:
            return self._reader.latest()
        if self._hand is None:
            return {}
        try:
            return self._hand.read_touch_full(self._side)
        except Exception:
            return {}

    def _on_diagnose(self):
        """Log per-finger array size and tip/nail/pad peaks so 'pad not reading'
        can be pinned to: not in the read (size ~105 / READ FAILED) vs. read but
        unresponsive (pad stays 0 while you press it)."""
        if self._hand is None:
            return
        touch = self._read_full_touch()
        self._log(f"[calib] --- pad read diagnose ({self._side}) ---")
        for ik, hf in self._ifc.FINGER_MAP:
            arr = touch.get(ik)
            if arr is None:
                self._log(f"[calib]   {hf}: READ FAILED (check console for a "
                          f"'Modbus read error at address=…' line).")
                continue
            size = int(np.asarray(arr).size)
            if ik == "fingerpalm":
                self._log(f"[calib]   palm: size={size} max={float(np.max(arr)):.0f}")
            else:
                # Raw (unclamped) tip range tells dead (all 0) vs negative-offset
                # (min<0, clamped to 0) vs just-not-contacted.
                tseg = self._ifc.subregion_segment(arr, ik, "tip")
                if tseg is not None and tseg.size:
                    traw = f"tip_raw[min={int(np.min(tseg))},max={int(np.max(tseg))}]"
                else:
                    traw = "tip_raw[absent]"
                self._log(f"[calib]   {hf}: size={size}  "
                          f"tip={self._ifc.subregion_raw(arr, ik, 'tip'):.0f} {traw}  "
                          f"pad(12x8)={self._ifc.subregion_raw(arr, ik, 'nail'):.0f}")

    def _collect(self, dur: float = 1.0, hz: float = 20.0) -> dict[tuple[str, str], dict[str, float]]:
        """Sample every (finger, sub-region) for the selected hand over `dur`s and
        return per-target mean + max (max feeds the noise floor). Uses direct
        full-finger reads. Keeps the GUI responsive via processEvents()."""
        samples: dict[tuple[str, str], list[float]] = {}
        for ik in self._ifc.REGION_KEYS:
            for sub in self._ifc.subregions_for(ik):
                samples[(ik, sub)] = []
        n = max(1, int(dur * hz))
        for _ in range(n):
            touch = self._read_full_touch()
            if touch:
                for (ik, sub) in samples:
                    samples[(ik, sub)].append(
                        self._ifc.subregion_raw(touch.get(ik), ik, sub)
                    )
            QApplication.processEvents()
            time.sleep(1.0 / hz)
        return {
            key: {"mean": float(np.mean(v)) if v else 0.0,
                  "max": float(np.max(v)) if v else 0.0}
            for key, v in samples.items()
        }

    def _refresh_status_lbl(self):
        if self._cal.has_any_noise_floor():
            self._status_lbl.setText(
                f"Loaded calibration from {self._cal.created} (noise-gate applied).")
        else:
            self._status_lbl.setText("No saved calibration yet — press Auto Calibrate.")

    # ── button handlers ─────────────────────────────────────────────────────────

    def _on_side_changed(self):
        self._side = "Left" if self._rb_left.isChecked() else "Right"
        # Drop the cached snapshot so the grid doesn't briefly show the other hand's data
        # until the reader's next read for the new side lands.
        if self._reader is not None:
            self._reader.clear()

    def _on_prepare(self):
        if self._hand is None:
            return
        # Freeze the arms rigidly at their current pose so a weight can't make
        # them sag, then open the fingers and hold them open.
        if self._control_thread is not None and self._hw is not None:
            try:
                q_t, q_l, q_r = self._hw.get_current_upper_body()
                self._control_thread.stop_teleop()
                self._control_thread.set_hold_goals(q_t, q_l, q_r)
                self._control_thread.exit_arm_damping()
            except Exception as exc:
                self._log(f"[calib] could not freeze arms: {exc}")
        try:
            self._hand.set_joint_goals({
                "q_goal_left_hand": [0.0] * 6,
                "q_goal_right_hand": [0.0] * 6,
            })
            self._hand.apply_control(engaged=True)
        except Exception as exc:
            self._log(f"[calib] could not open hands: {exc}")
        self._arms_frozen = True
        self._log("[calib] Arms held rigid at current pose; hands open. Ready to calibrate.")

    def _on_resume(self):
        if self._control_thread is not None:
            self._control_thread.enter_arm_damping()
        self._arms_frozen = False
        self._log("[calib] Arms released to damping (will lower gently). Remove weights first.")

    def _on_baseline(self):
        if self._hand is None:
            return
        self._log(f"[calib] Collecting {self._side} baseline + noise floor "
                  f"(keep all sensors unloaded)…")
        stats = self._collect(dur=2.0)
        # baseline = idle mean; noise floor = worst idle spike above it (×1.25
        # margin) so the live deadband sits just above observed noise.
        for ik in self._ifc.REGION_KEYS:
            for sub in self._ifc.subregions_for(ik):
                s = stats[(ik, sub)]
                nf = max(0.0, s["max"] - s["mean"]) * 1.25
                self._cal.set_baseline(self._side, ik, sub, s["mean"], nf)
        self._log("[calib] Baseline/noise set for " + self._side +
                  ". Live noise floors: " +
                  ", ".join(f"{self._ifc.label_for(ik)}={self._cal.live_noise_floor(self._side, ik):.1f}"
                            for ik in self._ifc.REGION_KEYS))

    def _busy_wait(self, secs: float):
        """Block ``secs`` while keeping the GUI responsive (repaints + processes events)."""
        end = time.monotonic() + max(0.0, secs)
        while time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.02)

    def _select_side(self, side: str):
        """Switch the calibration to `side` (updates the radio, which sets self._side and
        clears the reader cache), and let the background reader repopulate for it."""
        (self._rb_left if side == "Left" else self._rb_right).setChecked(True)
        QApplication.processEvents()
        self._busy_wait(0.8)          # let the reader read a fresh snapshot for this hand

    def _on_auto_calibrate(self):
        """One click, BOTH hands, correct order: for each hand HW-zero (reg 1009) THEN set
        the baseline + noise floor on the freshly-zeroed sensor, then Save & Apply. This is
        just the baseline zero + noise-floor deadband that gives consistent, phantom-free
        haptics from the raw sensor readings. Both hands must be OPEN and fully UNLOADED."""
        if self._hand is None or getattr(self, "_auto_running", False):
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Auto Calibrate")
        box.setText("Auto Calibrate will HW-zero then set the baseline + noise floor for "
                    "BOTH hands (correct order), then save & apply — no weights.\n\nMake "
                    "sure BOTH hands are OPEN and fully UNLOADED (nothing touching any "
                    "fingertip or palm), then continue.")
        go = box.addButton("Calibrate", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(go)
        box.exec_()
        if box.clickedButton() is not go:
            self._log("[calib] Auto Calibrate cancelled.")
            return
        self._auto_running = True
        prev_side = self._side
        self._btn_auto.setEnabled(False)
        self._btn_auto.setText("⚡  Auto-calibrating… (keep hands unloaded)")
        try:
            self._log("[calib] ===== AUTO CALIBRATE (both hands, HW zero → baseline) =====")
            for side in ("Left", "Right"):
                self._select_side(side)
                self._log(f"[calib] {side}: HW zero (reg 1009)…")
                self._on_hw_zero()
                self._busy_wait(1.5)          # let the sensor re-zero…
                if self._reader is not None:
                    self._reader.clear()      # …and drop pre-zero cached reads
                self._busy_wait(0.4)
                self._log(f"[calib] {side}: baseline + noise floor (unloaded)…")
                self._on_baseline()
            self._on_fit_save_apply()          # saves + applies BOTH hands (deadband only)
            self._log("[calib] ===== AUTO CALIBRATE done — consistent haptics applied "
                      "(baseline + noise-gate). =====")
        except Exception as exc:
            self._log(f"[calib] Auto Calibrate failed: {exc}")
        finally:
            self._select_side(prev_side)       # restore the operator's selected hand
            self._auto_running = False
            self._btn_auto.setEnabled(True)
            self._btn_auto.setText("⚡  Auto Calibrate (both hands — HW zero + baseline)")

    def _on_hw_zero(self):
        if self._hand is None:
            return
        try:
            ok = self._hand.calibrate_force_sensors(hand=self._side_letter())
            self._log(f"[calib] HW force calibration ({self._side}) "
                      f"{'sent' if ok else 'FAILED'} — ensure the hand was unloaded.")
        except Exception as exc:
            self._log(f"[calib] HW calibration error: {exc}")

    def _on_fit_save_apply(self):
        """Persist the baseline + noise floors and apply them to the live haptics."""
        json_path = self._ifc.DEFAULT_CALIBRATION_PATH
        out_dir = self._output_dir_getter() or "."
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"fingertip_calibration_{ts}.csv")
        try:
            self._cal.save(json_path, csv_path=csv_path)
        except Exception as exc:
            self._log(f"[calib] save failed: {exc}")
            return
        self._fit_lbl.setText(f"Saved → {Path(json_path).name}")
        self._log(f"[calib] Saved {json_path} and {Path(csv_path).name}")
        self._refresh_status_lbl()
        # Auto-apply to the live haptics streamer (reloads the noise-floor deadband).
        if self._apply_fn is not None:
            try:
                applied = self._apply_fn(json_path)
                self._log(f"[calib] Live haptics calibration {'applied' if applied else 'not applied'}.")
            except Exception as exc:
                self._log(f"[calib] apply failed: {exc}")

    # ── live refresh ─────────────────────────────────────────────────────────────

    def _refresh_live(self):
        if self._hand is None or not self.isVisible():
            return
        touch_side = self._read_full_touch()
        if not touch_side:
            return
        for (ik, sub), labels in self._live_labels.items():
            arr = touch_side.get(ik)
            present = self._ifc.subregion_present(arr, ik, sub)
            raw = self._ifc.subregion_raw(arr, ik, sub)
            e = self._cal.data[self._side][ik][sub]
            labels["raw"].setText(f"{raw:.1f}" if present else "n/a")
            labels["base"].setText(f"{e['baseline']:.1f}")
            labels["noise"].setText(f"{e['noise_floor']:.1f}")

    # ── window lifecycle ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Stop the background tactile reader first so it releases the Modbus bus.
        if self._reader is not None:
            try:
                self._reader.stop()
            except Exception:
                pass
            self._reader = None
        # Only if calibration actually froze the arms, release them to damping so
        # the rigid hold can never outlive the calibration session. (Touch reads
        # are direct, so there is no live haptics mode to restore.)
        if self._arms_frozen and self._control_thread is not None:
            try:
                self._control_thread.enter_arm_damping()
            except Exception:
                pass
            self._arms_frozen = False
            self._log("[calib] Calibration window closed — arms released to damping.")
        else:
            self._log("[calib] Calibration window closed.")
        super().closeEvent(event)


# ============================================================================
# KneelingPoseCalibrationWindow — trim the locked kneeling leg / waist angles
# ============================================================================

class KneelingPoseCalibrationWindow(QWidget):
    """Separate window for trimming the robot's locked kneeling pose.

    While kneeling, the legs are held at a fixed pose and the waist at a fixed
    angle. The nominal pose keeps one robot perfectly upright but can leave
    another sagging (e.g. leaning backward). This window exposes small per-joint
    trims (in degrees) that are applied live to the running control loop, so the
    operator can nudge the robot to a true upright kneel and watch the result,
    then save the trim. Saved trims auto-load on the next run.

    The trims are applied symmetrically to both legs (hip pitch, knee, ankle
    pitch) plus the waist (pitch, roll); waist pitch and hip pitch are the
    primary knobs for a fore/aft sag.
    """

    def __init__(self, calibration, base_kneeling_angles, log_fn, apply_fn,
                 enabled: bool = True):
        super().__init__()
        from projects.inspire_hand_teleop import kneeling_pose_calibration as kpc

        self._kpc = kpc
        self._cal = calibration
        self._base = np.asarray(base_kneeling_angles, dtype=np.float64)
        self._log = log_fn
        self._apply_fn = apply_fn
        self._enabled = enabled
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._applying = False   # guard against re-entrant valueChanged

        self.setWindowTitle("G1 — Kneeling Pose Calibration")
        self.setMinimumWidth(420)
        self._build_ui()
        self._sync_spins_from_cal()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        instr = QLabel(
            "Trim the locked kneeling pose so the robot sits perfectly upright. "
            "Adjustments apply live while kneeling — nudge a value, watch the "
            "robot, then Save. Waist Pitch / Hip Pitch correct a fore/aft sag "
            "(increase to pitch the body forward); Waist Roll corrects a side "
            "lean. Values are in degrees, added to the nominal pose."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #444; font-size: 11px;")
        root.addWidget(instr)

        if not self._enabled:
            warn = QLabel("Kneeling mode is OFF (run with --kneeling). "
                          "Trims are saved but have no live effect.")
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #c0392b; font-size: 11px; font-weight: bold;")
            root.addWidget(warn)

        grid = QGridLayout()
        grid.setSpacing(6)
        grid.addWidget(self._hdr("Joint"),  0, 0)
        grid.addWidget(self._hdr("Trim (deg)"), 0, 1)
        grid.addWidget(self._hdr(""), 0, 2)
        for row, (key, label, kind) in enumerate(self._kpc.PARAMS, start=1):
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 12px;")
            grid.addWidget(lbl, row, 0)

            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setRange(-30.0, 30.0)
            spin.setSuffix(" °")
            spin.setMinimumWidth(110)
            spin.valueChanged.connect(self._on_value_changed)
            self._spins[key] = spin
            grid.addWidget(spin, row, 1)

            zero_btn = QPushButton("0")
            zero_btn.setMaximumWidth(32)
            zero_btn.clicked.connect(lambda _=False, k=key: self._zero_one(k))
            grid.addWidget(zero_btn, row, 2)
        root.addLayout(grid)

        btn_row = QHBoxLayout()
        self._btn_reset = QPushButton("Reset All to 0")
        self._btn_reset.clicked.connect(self._on_reset_all)
        btn_row.addWidget(self._btn_reset)
        self._btn_save = QPushButton("💾  Save")
        self._btn_save.setStyleSheet(
            "QPushButton { background: #2ecc71; color: white; font-weight: bold; padding: 6px; }")
        self._btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(self._btn_save)
        root.addLayout(btn_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #555; font-size: 11px;")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)
        self._refresh_status_lbl()

    @staticmethod
    def _hdr(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 11px;")
        return lbl

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _sync_spins_from_cal(self):
        """Push the calibration's stored offsets (radians) into the spin boxes
        (degrees) without triggering a re-apply."""
        self._applying = True
        for key, spin in self._spins.items():
            spin.setValue(np.rad2deg(self._cal.get(key)))
        self._applying = False

    def _on_value_changed(self, *_):
        if self._applying:
            return
        for key, spin in self._spins.items():
            self._cal.set(key, np.deg2rad(spin.value()))
        self._apply_live()

    def _zero_one(self, key: str):
        self._spins[key].setValue(0.0)

    def _on_reset_all(self):
        self._applying = True
        for spin in self._spins.values():
            spin.setValue(0.0)
        self._applying = False
        self._cal.reset()
        self._apply_live()
        self._log("[kneel] Reset all kneeling trims to 0.")

    def _apply_live(self):
        if self._apply_fn is None:
            return
        try:
            self._apply_fn(self._cal)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[kneel] apply failed: {exc}")

    def _on_save(self):
        try:
            path = self._cal.save()
        except Exception as exc:  # noqa: BLE001
            self._log(f"[kneel] save failed: {exc}")
            return
        self._log(f"[kneel] Saved kneeling calibration → {path.name} "
                  + ", ".join(f"{lbl}={np.rad2deg(self._cal.get(k)):+.2f}°"
                              for k, lbl, _ in self._kpc.PARAMS))
        self._refresh_status_lbl()

    def _refresh_status_lbl(self):
        if self._cal.has_any_offset():
            self._status_lbl.setText(
                f"Active trims (loaded {self._cal.created}). "
                "Adjust and Save to update.")
        else:
            self._status_lbl.setText(
                "No trims set — nominal kneeling pose. Adjust a value to begin.")


# ============================================================================
# ArmAbductionLimitWindow — cap how far the arms may raise outward
# ============================================================================

class ArmAbductionLimitWindow(QWidget):
    """Separate window for setting the maximum arm abduction (outward raise).

    Abduction is the shoulder-roll motion that lifts the upper arm away from the
    torso. With heat sinks on the shoulders, abducting too far collides them with
    the body, so this caps the commanded shoulder roll during teleop. The cap is
    applied live (clamps the arms immediately) and saved per robot so it reloads
    next session.

    Tuning workflow (to get as close to the physical limit as possible):
      1. Open this window and watch the live "Current" abduction readout.
      2. In teleop, slowly raise an arm outward; the value rises toward the cap.
      3. Lower the cap until the arm stops just short of the heat-sink collision,
         then Save. The live readout turns red once the arm reaches the cap.
    """

    def __init__(self, limit, hw_controller, log_fn, apply_fn):
        super().__init__()
        from projects.inspire_hand_teleop import arm_abduction_limit as aal

        self._aal = aal
        self._limit = limit
        self._hw = hw_controller
        self._log = log_fn
        self._apply_fn = apply_fn
        self._applying = False
        self._max_deg = float(np.rad2deg(aal.JOINT_ABDUCTION_MAX_RAD))

        self.setWindowTitle("G1 — Max Arm Abduction Limit")
        self.setMinimumWidth(440)
        self._build_ui()
        self._sync_spins_from_limit()
        self._apply_live()   # ensure the loaded limit is active

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_live)
        self._timer.start(150)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        instr = QLabel(
            "Cap how far each arm may raise outward (shoulder abduction) to keep "
            "the shoulder heat sinks from colliding. The cap clamps teleop live "
            f"and is saved per robot. Physical max is {self._max_deg:.0f}°; lower "
            "the cap until the arm stops just short of contact."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #444; font-size: 11px;")
        root.addWidget(instr)

        grid = QGridLayout()
        grid.setSpacing(6)
        grid.addWidget(self._hdr("Arm"),       0, 0)
        grid.addWidget(self._hdr("Max (deg)"), 0, 1)
        grid.addWidget(self._hdr("Current"),   0, 2)

        self._spins: dict[str, QDoubleSpinBox] = {}
        self._cur_lbls: dict[str, QLabel] = {}
        for row, side in enumerate(("left", "right"), start=1):
            lbl = QLabel(side.capitalize())
            lbl.setStyleSheet("font-size: 12px;")
            grid.addWidget(lbl, row, 0)

            spin = QDoubleSpinBox()
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setRange(0.0, self._max_deg)
            spin.setSuffix(" °")
            spin.setMinimumWidth(110)
            spin.valueChanged.connect(self._on_value_changed)
            self._spins[side] = spin
            grid.addWidget(spin, row, 1)

            cur = QLabel("—")
            cur.setStyleSheet("font-size: 12px; font-family: monospace;")
            self._cur_lbls[side] = cur
            grid.addWidget(cur, row, 2)
        root.addLayout(grid)

        btn_row = QHBoxLayout()
        self._btn_max = QPushButton("Reset to Physical Max")
        self._btn_max.clicked.connect(self._on_reset_max)
        btn_row.addWidget(self._btn_max)
        self._btn_save = QPushButton("💾  Save")
        self._btn_save.setStyleSheet(
            "QPushButton { background: #2ecc71; color: white; font-weight: bold; padding: 6px; }")
        self._btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(self._btn_save)
        root.addLayout(btn_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #555; font-size: 11px;")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)
        self._refresh_status_lbl()

    @staticmethod
    def _hdr(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 11px;")
        return lbl

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _sync_spins_from_limit(self):
        self._applying = True
        self._spins["left"].setValue(np.rad2deg(self._limit.left_max))
        self._spins["right"].setValue(np.rad2deg(self._limit.right_max))
        self._applying = False

    def _on_value_changed(self, *_):
        if self._applying:
            return
        self._limit.set_left(np.deg2rad(self._spins["left"].value()))
        self._limit.set_right(np.deg2rad(self._spins["right"].value()))
        self._apply_live()

    def _on_reset_max(self):
        self._applying = True
        self._spins["left"].setValue(self._max_deg)
        self._spins["right"].setValue(self._max_deg)
        self._applying = False
        self._limit.set_left(self._aal.JOINT_ABDUCTION_MAX_RAD)
        self._limit.set_right(self._aal.JOINT_ABDUCTION_MAX_RAD)
        self._apply_live()
        self._log("[abduction] Reset abduction limits to physical max.")

    def _apply_live(self):
        if self._apply_fn is None:
            return
        try:
            self._apply_fn(self._limit)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[abduction] apply failed: {exc}")

    def _on_save(self):
        try:
            path = self._limit.save()
        except Exception as exc:  # noqa: BLE001
            self._log(f"[abduction] save failed: {exc}")
            return
        self._log(f"[abduction] Saved → {path.name}  "
                  f"left={np.rad2deg(self._limit.left_max):.1f}°, "
                  f"right={np.rad2deg(self._limit.right_max):.1f}°")
        self._refresh_status_lbl()

    def _refresh_live(self):
        """Show each arm's current abduction angle, flagged red at/over the cap."""
        if self._hw is None or not self.isVisible():
            for lbl in self._cur_lbls.values():
                lbl.setText("n/a")
            return
        try:
            _q_t, q_l, q_r = self._hw.get_current_upper_body()
        except Exception:
            return
        idx = self._aal.SHOULDER_ROLL_IDX
        for side, q_arm in (("left", q_l), ("right", q_r)):
            if q_arm is None or len(q_arm) <= idx:
                continue
            abd = self._aal.ArmAbductionLimit.abduction_deg(side, q_arm[idx])
            cap = np.rad2deg(self._limit.left_max if side == "left" else self._limit.right_max)
            at_cap = abd >= cap - 1.0
            self._cur_lbls[side].setText(f"{abd:+.1f}°")
            self._cur_lbls[side].setStyleSheet(
                "font-size: 12px; font-family: monospace; "
                + ("color: #c0392b; font-weight: bold;" if at_cap else "color: #222;"))

    def _refresh_status_lbl(self):
        if self._limit.is_restrictive():
            self._status_lbl.setText(
                f"Active limit (loaded {self._limit.created}). Adjust and Save to update.")
        else:
            self._status_lbl.setText(
                "No restriction — arms may abduct to the physical max. Lower a value to cap.")


# ============================================================================
# StudyManagerApp
# ============================================================================

_S_IDLE      = "IDLE"
_S_WAITING   = "WAITING"
_S_TELEOP    = "TELEOP"
_S_INIT      = "INITIALIZING"

# Build marker — BUMP the date/tags whenever a user-facing fix lands. It's printed at the top of
# every startup so a machine's log reveals exactly which fixes it's running. The peer runs a
# SEPARATE proj_XRT checkout; if its startup log doesn't show
# THIS date/tags, it's a STALE copy → sync it. Ends the "is the peer up to date?" guessing that
# let the neck-reset segfault keep recurring on an un-synced peer.
_MANAGER_BUILD = ("2026-08-23 · hand-dof-enable-disable · thread-safe-log (widget writes marshalled "
                  "to the GUI thread — the post-neck-reset segfault) · faulthandler-armed · "
                  "neck-reset: serial-lock + deferred-reset + locked-reconnect · "
                  "stop-attempt-pauses-both · resume-on-boot")

# User-study trial conditions (the recording order differs per participant group, so
# they are picked manually from the dropdown). Each condition has _ATTEMPTS_PER_TRIAL
# attempts. Recording lays these out as subfolders:
#   {output_dir}/{pid}_robot{N}_{date}_{time}/{condition}/attempt{K}/…
_TRIAL_CONDITIONS = ["V", "A+V", "H+V", "H+A+V"]
_ATTEMPTS_PER_TRIAL = 2

# ── Test Haptics sequence ─────────────────────────────────────────────────────────────────
# Per-region triangular ramp: 0 → max over UP_S, max → 0 over DOWN_S, then GAP_S of silence
# before the next region. Regions fire in this order. "max" is 1.0 on the 0..1 intensity scale
# Unity maps to audio/vibrotactile. Regions are the same keys the live haptics payload uses.
_HAPTICS_TEST_ORDER = ("palm", "thumb", "index", "middle", "ring", "little")
_HAPTICS_TEST_UP_S = 1.0
_HAPTICS_TEST_DOWN_S = 1.0
_HAPTICS_TEST_GAP_S = 0.75
_HAPTICS_TEST_SEND_HZ = 50.0            # payload rate during the ramp (smooth, low overhead)
_HAPTICS_TEST_REGIONS = ("thumb", "index", "middle", "ring", "little", "palm")  # full key set

# Which Unity feedback modalities each trial condition enables, applied automatically at
# each take start (updates the manager toggle buttons + sends the same Unity commands the
# buttons do). V(ideo)=FPV viewport (always on), A(udio)=Audio Haptics, H(aptics)=Vibrotactile.
_CONDITION_FEEDBACK = {
    "V":     {"viewport": True, "audio_haptics": False, "vibrotactile": False},
    "A+V":   {"viewport": True, "audio_haptics": True,  "vibrotactile": False},
    "H+V":   {"viewport": True, "audio_haptics": False, "vibrotactile": True},
    "H+A+V": {"viewport": True, "audio_haptics": True,  "vibrotactile": True},
}


class _RankListWidget(QListWidget):
    """QListWidget that emits ``reordered`` after a drag-drop reorder. Needed because
    QListWidget's InternalMove drag is implemented as a REMOVE + INSERT internally, so
    it never emits the model's rowsMoved signal — a plain rowsMoved.connect misses drags
    (only the ↑/↓ buttons fire). Overriding dropEvent catches every reorder."""
    reordered = QtCore.pyqtSignal()

    def dropEvent(self, event):
        super().dropEvent(event)
        self.reordered.emit()


class HeadCamViewWindow(QWidget):
    """Live head-camera feed with the AprilTag overlay, so the operator can check
    the table tag (id 0) is in frame before/while localizing. Pulls annotated BGR
    frames from ``frame_getter`` (HeadCamLocalizer.debug_frame) at ~15 Hz."""

    def __init__(self, frame_getter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Head Camera — table-tag view")
        self.setMinimumSize(640, 420)
        self._get = frame_getter
        self._buf = None  # keep the last frame buffer alive for the QImage
        lay = QVBoxLayout(self)
        self._label = QLabel("Head camera not running.\nClick 'Start Head-Cam Localize' first.")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background:#202020; color:#aaaaaa; font-size:13px;")
        lay.addWidget(self._label)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _tick(self):
        try:
            frame = self._get()
        except Exception:
            frame = None
        if frame is None:
            return
        # BGR -> RGB without cv2; keep the array alive while QImage references it.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        self._buf = rgb
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._label.width(), self._label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._label.setPixmap(pix)

    def showEvent(self, e):
        self._timer.start(66)   # ~15 Hz
        super().showEvent(e)

    def closeEvent(self, e):
        self._timer.stop()
        super().closeEvent(e)


class ChecklistWindow(QWidget):
    """Interactive, EDITABLE pre-flight checklist. Every step is a clickable checkbox;
    press ``✏ Edit`` to add / remove / rename / reorder steps and sections. The WHOLE
    structure — sections, step text, AND checked state — persists to checklist_state.json,
    so both edits and checkmarks survive shutdown/reboot. Two steps carry an inline ACTION
    button (voice calibration, the pre-robot PRACTICE round); the button is anchored by the
    step's ``action`` tag, so it follows the step when it's moved or renamed.

    On first run (or via "Restore Default Checklist") the structure is seeded from
    _DEFAULT_SECTIONS; an older flat {key: checked} state file is migrated by carrying its
    checkmarks onto the defaults."""

    _STATE_PATH = _HERE / "checklist_state.json"

    # Built-in default checklist, used to SEED the persisted structure (first run /
    # Restore Defaults). Each entry is (key, text) or (key, text, action) where action in
    # {"voices", "practice"} embeds that step's inline button. ``key`` is only used to
    # migrate checkmarks from an older flat state file onto these defaults.
    _DEFAULT_SECTIONS = [
        ("A. Once per study (persist across days)", [
            ("a_jetson", "Jetson head-cam service active on BOTH robots (multi-client "
                         "build): `systemctl status g1-headcam.service` on each Jetson."),
            ("a_fingertip", "Fingertip force calibration on EACH robot — \U0001f90f window "
                            "\u2192 \u26a1 Auto Calibrate (hands OPEN + unloaded). "
                            "Persists; redo only after a hand swap."),
            ("a_kneel", "Kneeling Pose (\U0001f9ce) + Arm Abduction Limit (\U0001f4aa) "
                        "saved on each robot (persist to JSON)."),
            ("a_bundle", "Crate bundle calibrated on the MASTER — \U0001f527 Recalibrate "
                         "Crate (2 external cams). Auto-pushed to the peer."),
        ]),
        ("B. Start of day / participant group", [
            ("b_power", "Robots + PCs on. Table tag 0 + crate tags placed; tag 0 clearly "
                        "visible to EVERY camera. (Tag 0 + crate base = 100 mm, faces = "
                        "65 mm — handled in software, don't change --tag-size.)"),
            ("b_launch", "Managers launched on BOTH machines (roles persist; check the "
                         "robot number). Link light green: \u201cpeer CONNECTED\u201d."),
            ("b_boot", "\U0001f916 Bootup Robot on EACH machine — AREA CLEAR, robot "
                       "rises into the kneel. (After a Safe Shutdown/E-STOP the same "
                       "button reads \u267b Reboot.)"),
            ("b_zed", "Video streaming ON both; each ZED aimed ~60\u00b0 DOWN at the "
                      "workspace (tag 0 in view = ZED localization quality)."),
            ("b_prepare", "\U0001f680 Prepare Study on the MASTER \u2014 wait for all "
                          "green: base localized both robots, crate cams anchored "
                          "(3/3 master + peer contribution = 4/4), bundle pushed, "
                          "clock sync logged."),
            ("b_zedcal", "\u201cZED calib\u201d light GREEN on both robots in System "
                         "Status (SN conf present \u2192 intrinsics auto-recorded)."),
            ("b_quest", "Both Quests connected: log shows \u201cUnity command channel "
                        "OPEN\u201d on each station (feedback conditions depend on it)."),
            ("b_order", "Trial order drag-ranked for this group (shared to the peer)."),
        ]),
        ("C. Per participant group", [
            ("c_pid", "Pair ID + both 16-digit operator codes (master + peer) entered on the "
                      "MASTER \u2014 synced to the peer at Start Session."),
            ("c_voices", "\U0001f5e3 Calibrate Voices — 1-MIC (voice-ID) MODE ONLY; auto-hidden with a 2-mic Lark setup (speakers separated by channel). OPERATOR 1 then OPERATOR 2 each read the CALIBRATION SCRIPT aloud near the conference mic while everyone else stays SILENT — press Start, then Stop when they finish (no fixed timer). SCRIPT (say your own number): \u201cI am operator one/two. Ok, you hold the crate steady from the bottom while I lift this side. Loosen your grip on the corner so I can pivot. Rotate it on the count of three. Bring the box closer to you. I have a firm grasp of the crate now. Now that we\u2019ve flipped it, lets set it down.\u201d The research coordinator is NOT enrolled \u2014 their rare remarks are labeled \u2018unknown\u2019. Transcript status should then show 2/2 voices.", "voices"),
            ("c_session", "\U0001f3ac Start Session — gates run, folder created, ARMED "
                          "(nothing recording yet). The room TRANSCRIPT IS NOW LIVE."),
            ("c_practice", "PRACTICE ROUND (no robots): press \u25b6 Start Practice (no robot) "
                           "in the main window FIRST, then participants lift/flip the box "
                           "BY HAND to practice communication. Recorded: ONLY the transcript "
                           "rows labeled \u2018practice\u2019 (no robot data \u2014 the robot isn\u2019t "
                           "used). Press \u23f9 End Practice when done."),
            ("c_teleop", "\u25b6 Start Teleop (BOTH) — operators in headsets, robots "
                         "following, feel check (~0.1 s)."),
            ("c_takes", f"Takes \u00d7{len(_TRIAL_CONDITIONS) * _ATTEMPTS_PER_TRIAL}: "
                        "\u25cf Start Attempt \u2192 task \u2192 \U0001f504 redo if needed "
                        "\u2192 \u23f9 End Attempt \u2192 \u23ed Continue to "
                        "<cond> Run <n> (confirms at trial changes)."),
            ("c_stop", "\u25a0 Stop Session (confirmed) \u2192 WAIT for the peer\u2019s "
                       "\u201cSent to master \u2713\u201d before closing anything."),
        ]),
        ("D. End of day", [
            ("d_shutdown", "\u23fb Quit Manager (runs the Safe Shutdown descent "
                           "automatically if the robot is still up), or \U0001f6e1 "
                           "Safe Shutdown first to keep the app open for data work."),
            ("d_verify", "Spot-check the data: open the Digital Twin on today\u2019s "
                         "last session (both robots + crate + video present)."),
        ]),
    ]

    def __init__(self, manager):
        super().__init__()
        self._mgr = manager
        self.setWindowTitle("Study Pre-Flight Checklist")
        self.resize(760, 820)
        self._edit_mode = False
        self._btn_practice = None
        # Live structure: [{"title": str, "items": [{"text","checked","action"}, ...]}, ...]
        self._sections = self._load()

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self._progress = QLabel("")
        self._progress.setStyleSheet("font-weight: bold;")
        top.addWidget(self._progress)
        top.addStretch()
        self._btn_edit = QPushButton("\u270f  Edit")
        self._btn_edit.setCheckable(True)
        self._btn_edit.setToolTip("Add, remove, rename, or reorder steps and sections. "
                                  "Every change saves automatically.")
        self._btn_edit.toggled.connect(self._on_edit_toggled)
        top.addWidget(self._btn_edit)
        reset = QPushButton("\u21ba  Reset All")
        reset.setToolTip("Uncheck every step (new day / new group). Keeps your edits.")
        reset.clicked.connect(self._reset)
        top.addWidget(reset)
        root.addLayout(top)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        root.addWidget(self._scroll)
        self._rebuild()

    # -- build ----------------------------------------------------------------
    def _rebuild(self):
        """Render the whole checklist from ``self._sections``. Rebuilt after every edit so
        the view always matches the persisted structure (simpler + safer than surgically
        patching widgets). Setting a fresh inner widget frees the previous one's boxes."""
        self._btn_practice = None
        inner = QWidget()
        col = QVBoxLayout(inner)
        for s_idx, sec in enumerate(self._sections):
            hdr_row = QHBoxLayout()
            hdr = QLabel(sec.get("title", ""))
            hdr.setStyleSheet("font-weight: bold; font-size: 13px; color: #2980b9; "
                              "margin-top: 8px;")
            hdr.setWordWrap(True)
            hdr_row.addWidget(hdr, 1)
            if self._edit_mode:
                hdr_row.addWidget(self._mini_btn(
                    "\u270e", "Rename section", lambda _, s=s_idx: self._rename_section(s)))
                hdr_row.addWidget(self._mini_btn(
                    "\u25b2", "Move section up", lambda _, s=s_idx: self._move_section(s, -1)))
                hdr_row.addWidget(self._mini_btn(
                    "\u25bc", "Move section down", lambda _, s=s_idx: self._move_section(s, +1)))
                hdr_row.addWidget(self._mini_btn(
                    "\u2715", "Delete section", lambda _, s=s_idx: self._delete_section(s)))
            col.addLayout(hdr_row)

            for i_idx, item in enumerate(sec.get("items", [])):
                # Mode-irrelevant steps are hidden in the normal view (still shown/editable
                # in Edit mode): e.g. voice calibration makes no sense with per-operator mics.
                if self._item_hidden(item) and not self._edit_mode:
                    continue
                row = QHBoxLayout()
                cb = QCheckBox(item.get("text", ""))
                cb.setChecked(bool(item.get("checked")))
                cb.setStyleSheet("QCheckBox { margin-left: 12px; }")
                cb.setToolTip(item.get("text", ""))
                cb.setMinimumHeight(28)
                cb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
                cb.toggled.connect(lambda v, it=item: self._on_toggle(it, v))
                row.addWidget(cb, 1)
                if self._edit_mode:
                    row.addWidget(self._mini_btn(
                        "\u270e", "Edit text", lambda _, s=s_idx, i=i_idx: self._edit_item(s, i)))
                    row.addWidget(self._mini_btn(
                        "\u25b2", "Move up", lambda _, s=s_idx, i=i_idx: self._move_item(s, i, -1)))
                    row.addWidget(self._mini_btn(
                        "\u25bc", "Move down", lambda _, s=s_idx, i=i_idx: self._move_item(s, i, +1)))
                    row.addWidget(self._mini_btn(
                        "\u2715", "Delete step", lambda _, s=s_idx, i=i_idx: self._delete_item(s, i)))
                col.addLayout(row)

                # Inline action button for a step (only "voices" now \u2014 the checklist's
                # Start Practice button was removed; use "Start Practice (no robot)" below
                # Start Session in the main window instead).
                if item.get("action") == "voices":
                    b = QPushButton("\U0001f5e3  Calibrate Voices now")
                    b.setMaximumWidth(260)
                    b.clicked.connect(self._mgr._on_calibrate_voices)
                    col.addWidget(b)

            if self._edit_mode:
                add = QPushButton("\uff0b  Add step")
                add.setMaximumWidth(160)
                add.clicked.connect(lambda _, s=s_idx: self._add_item(s))
                col.addWidget(add)

        if self._edit_mode:
            add_sec = QPushButton("\uff0b  Add section")
            add_sec.clicked.connect(lambda _: self._add_section())
            col.addWidget(add_sec)
            restore = QPushButton("\u21ba  Restore Default Checklist")
            restore.setToolTip("Replace the whole checklist with the built-in default "
                               "(clears your edits and every checkmark).")
            restore.clicked.connect(lambda _: self._restore_defaults())
            col.addWidget(restore)

        col.addStretch()
        self._scroll.setWidget(inner)
        self._update_progress()
        self.refresh_practice()

    def _mini_btn(self, glyph, tip, on_click):
        b = QPushButton(glyph)
        b.setFixedWidth(30)
        b.setToolTip(tip)
        b.clicked.connect(on_click)
        return b

    # -- structure load / seed / save ----------------------------------------
    @classmethod
    def _seed_from_defaults(cls, checked_by_key=None):
        checked_by_key = checked_by_key or {}
        sections = []
        for title, items in cls._DEFAULT_SECTIONS:
            out = []
            for entry in items:
                key, text = entry[0], entry[1]
                action = entry[2] if len(entry) > 2 else None
                out.append({"text": text, "action": action,
                            "checked": bool(checked_by_key.get(key, False))})
            sections.append({"title": title, "items": out})
        return sections

    def _load(self):
        """Load the persisted structure. New format is {"sections": [...]}; an older flat
        {key: checked} file (or a missing file) seeds the defaults, carrying its checks."""
        try:
            with open(self._STATE_PATH) as f:
                data = json.load(f)
        except Exception:
            data = None
        if isinstance(data, dict) and isinstance(data.get("sections"), list):
            secs = []
            for s in data["sections"]:
                if not isinstance(s, dict):
                    continue
                items = []
                for it in (s.get("items") or []):
                    if not isinstance(it, dict):
                        continue
                    items.append({"text": str(it.get("text", "")),
                                  "checked": bool(it.get("checked", False)),
                                  "action": it.get("action") or None})
                secs.append({"title": str(s.get("title", "")), "items": items})
            if secs:
                return secs
        checked = data if isinstance(data, dict) else {}
        return self._seed_from_defaults(checked)

    def _save(self):
        try:
            with open(self._STATE_PATH, "w") as f:
                json.dump({"sections": self._sections}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # -- checkbox state -------------------------------------------------------
    def _on_toggle(self, item, val):
        item["checked"] = bool(val)
        self._save()
        self._update_progress()

    def _reset(self):
        for sec in self._sections:
            for it in sec.get("items", []):
                it["checked"] = False
        self._save()
        self._rebuild()

    def _item_hidden(self, item):
        """A step that doesn't apply to the current mic mode → hidden from the normal view
        and excluded from the progress count. Voice calibration (action='voices') is
        meaningless in 2-mic stereo mode, where speakers are separated by channel."""
        stereo = bool((getattr(self._mgr, "_cfg", None) or {}).get("transcribe_stereo", False))
        return stereo and item.get("action") == "voices"

    def _update_progress(self):
        items = [it for sec in self._sections for it in sec.get("items", [])
                 if not self._item_hidden(it)]
        done = sum(1 for it in items if it.get("checked"))
        self._progress.setText(f"{done} / {len(items)} steps checked")

    # -- edit mode ------------------------------------------------------------
    def _on_edit_toggled(self, on):
        self._edit_mode = bool(on)
        self._btn_edit.setText("\u2713  Done" if on else "\u270f  Edit")
        self._rebuild()

    def _confirm(self, question, detail=""):
        msg = question + (("\n\n" + detail) if detail else "")
        return QMessageBox.question(self, question, msg,
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    # -- step edits -----------------------------------------------------------
    def _add_item(self, s):
        text, ok = QInputDialog.getMultiLineText(self, "Add step", "New step text:", "")
        if ok and text.strip():
            self._sections[s]["items"].append(
                {"text": text.strip(), "checked": False, "action": None})
            self._save()
            self._rebuild()

    def _edit_item(self, s, i):
        item = self._sections[s]["items"][i]
        text, ok = QInputDialog.getMultiLineText(self, "Edit step", "Step text:",
                                                 item.get("text", ""))
        if ok and text.strip():
            item["text"] = text.strip()
            self._save()
            self._rebuild()

    def _delete_item(self, s, i):
        item = self._sections[s]["items"][i]
        if not self._confirm("Delete this step?", item.get("text", "")):
            return
        del self._sections[s]["items"][i]
        self._save()
        self._rebuild()

    def _move_item(self, s, i, delta):
        """Move a step up/down; at a section's ends it hops into the neighbouring section."""
        items = self._sections[s]["items"]
        j = i + delta
        if 0 <= j < len(items):
            items[i], items[j] = items[j], items[i]
        elif j < 0 and s > 0:
            self._sections[s - 1]["items"].append(items.pop(i))
        elif j >= len(items) and s < len(self._sections) - 1:
            self._sections[s + 1]["items"].insert(0, items.pop(i))
        else:
            return
        self._save()
        self._rebuild()

    # -- section edits --------------------------------------------------------
    def _add_section(self):
        title, ok = QInputDialog.getText(self, "Add section", "Section title:")
        if ok and title.strip():
            self._sections.append({"title": title.strip(), "items": []})
            self._save()
            self._rebuild()

    def _rename_section(self, s):
        title, ok = QInputDialog.getText(self, "Rename section", "Section title:",
                                         text=self._sections[s].get("title", ""))
        if ok and title.strip():
            self._sections[s]["title"] = title.strip()
            self._save()
            self._rebuild()

    def _move_section(self, s, delta):
        j = s + delta
        if 0 <= j < len(self._sections):
            self._sections[s], self._sections[j] = self._sections[j], self._sections[s]
            self._save()
            self._rebuild()

    def _delete_section(self, s):
        sec = self._sections[s]
        n = len(sec.get("items", []))
        if n and not self._confirm(f"Delete this section and its {n} step(s)?",
                                   sec.get("title", "")):
            return
        del self._sections[s]
        self._save()
        self._rebuild()

    def _restore_defaults(self):
        if not self._confirm("Restore the default checklist?",
                             "This replaces ALL your edits with the built-in checklist "
                             "and clears every checkmark."):
            return
        self._sections = self._seed_from_defaults()
        self._save()
        self._rebuild()

    # -- misc -----------------------------------------------------------------
    def refresh_practice(self):
        if getattr(self, "_btn_practice", None) is None:
            return
        active = bool(getattr(self._mgr, "_practice_active", False))
        self._btn_practice.setText(
            "\u23f9  End Practice (RECORDING\u2026)" if active
            else "\u25b6  Start Practice")
        self._btn_practice.setStyleSheet(
            "QPushButton { background: %s; color: white; font-weight: bold; "
            "padding: 6px; }" % ("#c0392b" if active else "#8e44ad"))

    def closeEvent(self, e):
        self._save()
        super().closeEvent(e)


class TranscriptWindow(QWidget):
    """Live room-transcription view: one line per finished utterance (clock time · take ·
    speaker · text), appended as whisper produces them — so it keeps updating even when an
    accuracy-first model runs behind realtime. A header shows the model, mic state, and the
    processing BACKLOG (utterances still queued). No audio is ever saved; the authoritative
    record is transcript_<session>.csv in the session folder."""

    _SPK_COLOR = {"operator1": "#2980b9", "operator2": "#27ae60"}

    def __init__(self, manager):
        super().__init__()
        self._mgr = manager
        self.setWindowTitle("Live Transcription")
        self.resize(680, 620)
        root = QVBoxLayout(self)
        self._hdr = QLabel("")
        self._hdr.setStyleSheet("font-weight: bold; color: #2c3e50;")
        self._hdr.setWordWrap(True)
        root.addWidget(self._hdr)
        self._view = QTextEdit()
        self._view.setReadOnly(True)
        self._view.setStyleSheet("QTextEdit { font-size: 13px; }")
        root.addWidget(self._view, 1)
        row = QHBoxLayout()
        self._autoscroll = QCheckBox("Auto-scroll")
        self._autoscroll.setChecked(True)
        row.addWidget(self._autoscroll)
        row.addStretch()
        clr = QPushButton("Clear view")
        clr.setToolTip("Clear this window only — the transcript CSV on disk is untouched.")
        clr.clicked.connect(self._view.clear)
        row.addWidget(clr)
        root.addLayout(row)
        # Replay rows captured before the window was (re)opened.
        for r in list(getattr(manager, "_transcript_rows", [])):
            self.append_row(r, scroll=False)
        if self._autoscroll.isChecked():
            sb = self._view.verticalScrollBar()
            sb.setValue(sb.maximum())
        # 1 Hz header refresh (model / mic / backlog).
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_header)
        self._timer.start(1000)
        self._refresh_header()

    @staticmethod
    def _esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def _refresh_header(self):
        tr = getattr(self._mgr, "_transcriber", None)
        st = {}
        if tr is not None:
            try:
                st = tr.status() or {}
            except Exception:
                st = {}
        if not st:
            self._hdr.setText("Transcription is off.")
            return
        model = st.get("model", "?")
        backlog = int(st.get("backlog", 0) or 0)
        mic = "listening" if st.get("listening") else "not listening"
        lag = (f" · <span style='color:#e67e22'>{backlog} queued "
               f"(behind realtime)</span>" if backlog else " · caught up")
        err = st.get("error")
        txt = f"whisper <b>{self._esc(model)}</b> · {mic}{lag}"
        if err:
            txt += f" · <span style='color:#c0392b'>⚠ {self._esc(err)}</span>"
        self._hdr.setText(txt)

    def append_row(self, row, scroll=True):
        try:
            # START → END at millisecond precision (HH:MM:SS.mmm) so each phrase can be
            # matched to the digital-twin playback exactly, with its duration. t0/t1 are the
            # audio-derived speech onset/offset (wall-clock epoch); %f is µs → trim to ms.
            def _fmt(t):
                return datetime.fromtimestamp(float(t)).strftime("%H:%M:%S.%f")[:-3]
            t0 = float(row.get("t0", 0.0))
            t1 = float(row.get("t1", t0))
            ts = f"{_fmt(t0)}→{_fmt(t1)} ({max(0.0, t1 - t0):.2f}s)"
        except Exception:
            ts = "--:--:--.---"
        spk = str(row.get("speaker", "unknown"))
        take = str(row.get("take", "between"))
        saved = bool(row.get("saved", True))
        color = self._SPK_COLOR.get(spk, "#7f8c8d")
        practice = (take == "practice")
        text = self._esc(row.get("text", ""))
        if practice:
            # PRACTICE rows stand out: orange badge + orange italic text, so they're never
            # mistaken for an actual attempt take while scanning the live feed.
            take_tag = (" <span style='background:#e67e22; color:#ffffff; "
                        "font-weight:bold;'>&nbsp;PRACTICE&nbsp;</span>")
            text = f"<i style='color:#e67e22'>{text}</i>"
        elif take in ("between", "", "None"):
            take_tag = ""
        else:
            take_tag = f" <span style='color:#95a5a6'>[{self._esc(take)}]</span>"
        # Rows spoken outside a recorded take are SHOWN but not written to the session file —
        # mark them so the operator knows what will (and won't) be in the saved transcript.
        live_tag = ("" if saved else
                    " <span style='color:#7f8c8d; font-style:italic;'>· not saved</span>")
        line = (f"<div><span style='color:#95a5a6'>{ts}</span>{take_tag}{live_tag} "
                f"<b style='color:{color}'>{self._esc(spk)}:</b> {text}</div>")
        self._view.append(line)
        if scroll and self._autoscroll.isChecked():
            sb = self._view.verticalScrollBar()
            sb.setValue(sb.maximum())


class StudyManagerApp(QMainWindow):
    """Main GUI window for user study management."""

    # Emitted (from the transcriber proxy's rx thread) for each finished transcript row;
    # a queued connection marshals it onto the GUI thread for the Live Transcription window.
    _transcript_sig = QtCore.pyqtSignal(dict)

    # Every log line goes through this signal so _log() is safe to call from ANY thread.
    # QTextEdit / QTextDocument are GUI-thread-only: appending from a worker thread while the
    # GUI thread appends + lays out the same document corrupts the document's internals at the
    # C++ level, and the process dies LATER inside Qt's paint/layout path with a bare SIGSEGV
    # (no Python traceback — faulthandler installs its alt-stack only for the main thread).
    # That was the post-neck-reset crash: _reconnect_neck_from_scratch (a worker thread) logged
    # its result directly into the widget, then the manager segfaulted ~30 s later on the next
    # burst of GUI-thread logging. The connection is AutoConnection: GUI-thread emits stay a
    # direct call (unchanged behaviour), worker-thread emits get queued onto the GUI thread.
    _log_sig = QtCore.pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        # Connect FIRST — before anything can log — so no line can reach the widget off-thread.
        self._log_sig.connect(self._append_log)
        self._args = args
        self._state = _S_INIT
        self._recording_active = False
        # Auto resume-on-boot: after a crash/restart, prompt ONCE to resume a study that's still
        # in progress (this machine's saved state, or the OTHER machine's live session over the
        # link). Delayed a few seconds so the peer link can connect first (prefer its live state).
        self._resume_prompt_shown = False
        self._boot_monotonic = time.monotonic()
        # Two-phase startup: init connects everything read-only; the operator's
        # 🤖 Bootup Robot button runs the physical bootup. Until it has run, Quit
        # must NOT damp (the un-released FSM still owns the robot).
        self._init_done = False
        self._hw_booted = False
        self._exit_damping_done = True
        # One-press-per-slot guards: takes already started this session (Start Attempt
        # greys out for them), and an open-session request in flight (Start Session greys
        # immediately on click, before the synced epoch fires).
        self._recorded_takes = set()          # {(condition, attempt), ...}
        self._session_opening = False
        # Robot-comms watchdog: True while the G1's DDS link is stale (bumped ethernet).
        self._robot_comms_lost = False
        self._comms_lost_since = None       # monotonic time the current loss began
        # True once a SUSTAINED loss (robot power-cycled off) has escalated the manager to the
        # UNBOOTED state, so the robot is never streamed commands the instant it reconnects.
        self._robot_powered_down = False
        # Per-take feedback-event log (audio/haptic/viewport on-off, timestamped).
        self._feedback_file = None
        self._feedback_writer = None
        self._audio_haptics_on = False
        self._vibrotactile_on = False
        # Default the streaming feed ON to match Unity's auto-started stream, so the
        # manager-priority push on connect doesn't tear down a live feed by default.
        self._viewport_on = True
        self._was_connected = False  # rising-edge tracker for the connect-time Unity state push
        # Live-link (pose-freshness) rising-edge tracker: is_connected LATCHES True forever,
        # so it can't detect a headset off/on reconnect. pose_age_s resets when body_pose
        # frames resume, so this edge fires on every real (re)connect and re-asserts the
        # Unity haptics/FPV toggles (fixes haptics going silent after a headset off/on).
        self._link_live_last = False
        # True only when the manager's Start Teleop was pressed BEFORE a client was
        # connected — auto-engage once it connects. NOT set on pause, so pausing
        # holds the pose instead of being instantly re-engaged by _refresh_status.
        self._engage_when_connected = False
        # True while teleop is paused (or the client dropped mid-teleop). The next
        # engage is then a RESUME — it blends directly to the operator's pose
        # instead of replaying the full staged entry choreography.
        self._teleop_paused = False
        # WHY teleop last paused, for the state label: "headset" (Quest removed / operator
        # paused in VR), "comms" (robot link lost), "linked" (other station), "manual",
        # "link" (Quest data-link freeze — set by _check_link_watchdog).
        self._pause_reason = None
        # Quest data-link watchdog debounce timers (monotonic instants, or None).
        self._link_bad_since = None
        self._link_ok_since = None

        # Robot identity (1 or 2). Two identical G1s sit on opposite sides of the
        # table, each running its own manager. ``robot_id`` is the ONLY thing that
        # distinguishes the two machines' recordings; the participant_id+trial
        # (``session_label``) is kept identical on both robots and is the
        # cross-machine *pairing key* used to match a robot-1 trial with the
        # corresponding robot-2 trial at fusion time. Resolution order:
        # --robot-id  >  persisted file (so each machine remembers its robot)  >  1.
        # Per-machine config (persisted), so a bare `--net …` launch "just works":
        # robot id, head-cam Jetson IP, whether to fuse the head cam into crate
        # tracking, and the study-link role/host. Everything is editable in the UI;
        # CLI flags only override when explicitly passed.
        self._cfg_path = os.path.expanduser("~/.xrt_study_config.json")
        self._cfg = self._load_cfg(args)
        self._robot_id = self._cfg["robot_id"]
        # Eco / arm-compliance modes are restored from the persisted config (a CLI
        # flag, if passed, forced them ON in _load_cfg). Toggling them live re-saves.
        self._eco_on  = bool(self._cfg["eco_mode"])
        self._comp_on = bool(self._cfg["arm_compliance"])

        # Hardware handles (populated during init)
        self._device            = None
        self._solver            = None
        self._controller        = None
        self._hw_controller     = None
        self._hand_controller   = None
        self._haptics_streamer  = None
        self._motor_stats_streamer = None
        # Test Haptics sequence state (QTimer-driven ramp; see _on_test_haptics).
        self._test_haptics_timer = None
        self._test_haptics_sides = ()
        self._test_haptics_t0 = None
        self._neck_interface    = None
        self._model = None
        self._data  = None
        self._viewer_ctx = None
        self._viewer     = None

        # Control state (for recording)
        self._q_goal_torso  = np.zeros(3)
        self._q_goal_left   = np.zeros(7)
        self._q_goal_right  = np.zeros(7)

        self._last_viz_time = 0.0

        # Background helpers
        self._hw_cmd_thread: HardwareCommandThread | None = None
        self._control_thread: ControlThread | None = None
        self._latency_tracker = LatencyTracker()
        # End-to-end latency budgets (video glass-to-glass, teleop motion-to-motion), assembled
        # each status tick from measured stages + estimated endpoint constants; shown live in the
        # Latency Monitor panel and saved per-trial to latency_summary.json.
        self._latency_e2e = EndToEndLatency()
        self._motor_monitor = MotorMonitor(
            temp_min=args.motor_temp_min,
            temp_max=args.motor_temp_max, torque_max=args.motor_torque_max,
            finger_temp_max=args.finger_temp_max,
            finger_current_max=args.finger_current_max,
        )
        # Single unified raw-data recorder (robot CSV + tactile bin + meta), plus
        # ZED video — all written into one session directory.
        self._session_recorder = SessionRecorder(None, None)
        self._zed_recorder   = ZEDRecorder(args.output_dir)
        self._session_label  = ""
        self._last_session_dir: str | None = None
        # Continuous-recording session state: the session folder (pid_robotN_date_time)
        # created at Start, plus the participant id in use. Trial condition + attempt
        # live in the UI selectors and pick the current subfolder; switching them (Next
        # Trial / Next Attempt, or the dropdown/spinbox) saves + rotates to a new folder
        # without stopping the streaming pipeline. _suppress_rotate guards programmatic
        # selector changes (e.g. a synced switch from the peer) from re-triggering a rotate.
        self._session_root: str | None = None
        # Shared session id ({pid}_{date}_{time}) — SAME on both robots (the start
        # initiator generates it and sends it over the link) so each robot's tree merges
        # into one folder on the master, with a robot{N}/ leaf per attempt.
        self._session_id: str | None = None
        # Last session folder started (kept after Stop for the "Send Data to Master" button).
        self._last_session_root: str | None = None
        self._rec_pid = "unknown"                # pair id of the open session
        # Per-operator 16-digit codes for the open session (both set from the master, synced).
        self._rec_master_code = ""
        self._rec_peer_code = ""
        # Clock-sync quality (master↔peer offset + NTP status), stamped into each meta.
        self._clock_sync_info = {}
        # ZED-left factory intrinsics (read once from SN<serial>.conf), stamped into meta.
        self._zed_intrinsics_meta = None
        # Per-attempt demo quality label ("good"/"bad"/"unrated") for filtering demos.
        self._attempt_quality = "unrated"
        # Running-attempt/practice timer: monotonic start of the current take (None when idle),
        # plus which mode it's timing ("Attempt"/"Practice") so a mode change restarts the clock.
        self._attempt_start_t = None
        self._timer_mode = None
        # UI mode: User Study (default) hides the Good/Bad attempt-quality marking; Data
        # Collection Mode exposes it. Persisted; everything else is identical between modes.
        self._study_mode = bool(self._cfg.get("study_mode", True))
        self._suppress_rotate = False
        # Guards a peer-applied trial-order update from echoing back over the link.
        self._suppress_order_broadcast = False
        self._send_proc = None   # background rsync process handle (peer → master send)
        self._pull_proc = None   # background rsync process handle (master ← peer pull)
        self._dataset_proc = None   # background build_dataset process handle

        # Crate AprilTag tracker — started ON DEMAND from the UI (Start Crate
        # Tracking button), never auto-started, so the camera-less robot's machine
        # is unaffected unless the operator opts in. Stays None until started.
        self._apriltag_tracker = None
        self._apriltag_starting = False
        self._calib_proc: "subprocess.Popen | None" = None  # recalibration window

        # Head-camera base self-localization — runs on EVERY robot's machine (uses
        # this robot's own head D435 looking at the table tag). Started from the UI.
        self._headcam_localizer = None
        self._headcam_starting = False
        self._headcam_view = None
        # Live room transcription (subprocess proxy; spawned post-init when enabled).
        self._transcriber = None
        self._transcribe_on = None   # resolved from cfg at init
        # Live-transcription window + a rolling buffer of recent rows (replayed when the
        # window is (re)opened). Rows arrive on the proxy rx thread and are marshalled onto
        # the GUI thread via _transcript_sig (queued connection).
        self._transcript_window = None
        self._transcript_rows = deque(maxlen=2000)
        self._transcript_sig.connect(self._append_transcript_row)
        # Mic self-test (record → play back → delete). In stereo it runs per-mic in
        # sequence (Mic 1 = LEFT, then Mic 2 = RIGHT). State: running QProcess, temp wavs,
        # capture device/env (PipeWire-aware), and the phase list.
        self._mic_test_proc = None
        self._mic_test_wav = None          # the raw recording for the current phase
        self._mic_test_playwav = None      # the isolated-channel mono wav played back
        self._mic_test_nch = 1
        self._mic_test_capdev = None
        self._mic_test_capenv = None
        self._mic_test_phases = []         # [(label, channel_to_isolate_or_None), …]
        self._mic_test_phase = 0
        # True only while the voice-calibration dialogs are up — suppresses the E-STOP
        # cursor auto-park, which would otherwise raise the main window over the modal and
        # steal focus while an operator reads the script (mouse idle > 15 s).
        self._calibrating_voices = False
        # Pre-robot PRACTICE round (participants hand-lift the box): records ONLY the
        # transcript (take='practice') — no robot data, since the robot isn't used.
        self._practice_active = False
        self._checklist_window = None
        self._auto_contrib_busy = False   # re-entry guard for auto crate-contribution

        # Cross-machine study link (over Tailscale). role 'none'|'master'|'peer'.
        # master = robot 1 (also fuses robot 2's crate candidates); peer = robot 2.
        # Teleop/recording control is now SYMMETRIC — either machine can drive both
        # robots from the embedded "Both Robots" panel, and commands flow either way.
        self._link = None
        self._link_tick_n = 0
        self._link_was_connected = False   # for connect/disconnect transition logs

        # Low-rate (10 s) session summary: peak latencies, written while
        # recording for quick later review. (Motor temps are not logged here —
        # they're already in the per-frame raw CSV.)
        self._summary_file = None
        self._summary_writer = None
        self._summary_start_mono = 0.0

        # Wrench (end-effector force) mode — live readout only (verification).
        self._wrench_estimator: "we.WrenchEstimator | None" = None
        self._wrench_worker: WrenchLiveWorker | None = None
        self._finger_qpos_addrs = None
        self._calib_window: ForceCalibrationWindow | None = None
        self._kneeling_window: KneelingPoseCalibrationWindow | None = None
        self._abduction_window: ArmAbductionLimitWindow | None = None

        # Load any saved per-robot kneeling-pose trim and pre-compute the
        # calibrated locked leg angles + held waist angle. These are used in
        # place of the nominal _KNEELING_ANGLES everywhere the kneel pose is
        # commanded, so a sagging robot can be brought upright.
        from projects.inspire_hand_teleop import kneeling_pose_calibration as kpc
        self._kneeling_cal = kpc.KneelingCalibration.load()
        self._kneeling_angles = self._kneeling_cal.apply_to_legs(_KNEELING_ANGLES)
        self._kneeling_torso = (self._kneeling_cal.torso_offset_vector()
                                if args.kneeling else np.zeros(3))

        # Load any saved max arm-abduction limit (shoulder-roll clamp) that keeps
        # the shoulder heat sinks from self-colliding when the arms raise outward.
        from projects.inspire_hand_teleop import arm_abduction_limit as aal
        self._abduction_limit = aal.ArmAbductionLimit.load()

        self._setup_ui()
        self._setup_timers()

        QTimer.singleShot(100, self._initialize_system)
        # Video streaming is ON by default — launch mediamtx + zed_to_mediamtx
        # shortly after the UI is up (independent of the robot init).
        QTimer.singleShot(300, self._start_video_streaming_default)
        if self._cfg["link_role"] != "none":
            QTimer.singleShot(600, self._init_link)
        # Flag a missing ZED calibration conf right at startup (esp. on the peer).
        QTimer.singleShot(900, self._check_zed_calibration)

    # =========================================================================
    # UI construction
    # =========================================================================

    def _setup_ui(self):
        self.setWindowTitle(f"G1 Inspire Hand — User Study Manager  (Robot {self._robot_id})")
        # Taller minimum + initial size so the Data Collection column's stack of
        # buttons opens at full height instead of being vertically compressed until
        # the window is hand-stretched. (_init_link grows it further when the extra
        # "Both Robots" column appears.)
        # Minimum is modest (so the window fits smaller screens); the scroll area
        # below handles any overflow. Open tall by default so the top control area
        # (incl. the whole Data Collection column) is visible without scrolling.
        self.setMinimumSize(880, 640)
        # Open WIDE enough that the far-right Safety / E-Stop column is fully in view on
        # load (the System Status panel is wide, so the main row needs the room), and the
        # right-hand column now also holds the Digital Twin panel to keep the layout
        # horizontal rather than vertically stacked.
        self.resize(1760, 900)

        # The full panel is taller than some screens. Host it in a vertical scroll
        # area so every column keeps its natural height — the Data Collection buttons
        # no longer compress; the lower monitor sections scroll instead. SetMinimumSize
        # keeps the inner panel at full height (scrollbar appears) rather than letting
        # the layout shrink the buttons; setWidgetResizable tracks the viewport width.
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(8)
        root_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        # ── Session bar ───────────────────────────────────────────────────────
        session_bar = QHBoxLayout()
        session_bar.addWidget(QLabel("Pair ID:"))
        self._pid_edit = QLineEdit()           # holds the PAIR id (e.g. P01), shared by both robots
        self._pid_edit.setPlaceholderText("e.g. P01")
        self._pid_edit.setMaximumWidth(70)
        self._pid_edit.setToolTip("Participant PAIR id (e.g. P01) — the shared session folder "
                                  "name, synced to the peer.")
        session_bar.addWidget(self._pid_edit)
        # Per-operator 16-digit codes, BOTH assigned from the master (synced to the peer at
        # Start Session). Each robot records its own operator's code in that take's meta.
        session_bar.addWidget(QLabel("Master code:"))
        self._master_code_edit = QLineEdit()
        self._master_code_edit.setPlaceholderText("16-digit master-operator code")
        self._master_code_edit.setMaxLength(16)
        self._master_code_edit.setMaximumWidth(150)
        self._master_code_edit.setToolTip("16-digit code for the operator on THIS (master) "
                                          "robot. Recorded in the master robot's meta.")
        session_bar.addWidget(self._master_code_edit)
        session_bar.addWidget(QLabel("Peer code:"))
        self._peer_code_edit = QLineEdit()
        self._peer_code_edit.setPlaceholderText("16-digit peer-operator code")
        self._peer_code_edit.setMaxLength(16)
        self._peer_code_edit.setMaximumWidth(150)
        self._peer_code_edit.setToolTip("16-digit code for the peer robot's operator — set "
                                        "here on the master; pushed to the peer at Start "
                                        "Session and recorded in the peer robot's meta.")
        session_bar.addWidget(self._peer_code_edit)
        session_bar.addWidget(QLabel("Trial:"))
        self._cond_combo = QComboBox()
        self._cond_combo.addItems(_TRIAL_CONDITIONS)
        self._cond_combo.setToolTip(
            "Trial condition. The recording ORDER differs per participant group, so "
            "pick each one manually. While recording, changing this saves the current "
            "data and switches to attempt 1 of the new condition (a new subfolder). "
            "Synced to the peer when linked.")
        self._cond_combo.currentIndexChanged.connect(self._on_condition_changed)
        session_bar.addWidget(self._cond_combo)
        session_bar.addWidget(QLabel("Attempt:"))
        self._attempt_spin = QSpinBox()
        self._attempt_spin.setMinimum(1)
        self._attempt_spin.setMaximum(_ATTEMPTS_PER_TRIAL)
        self._attempt_spin.setMaximumWidth(55)
        self._attempt_spin.setToolTip(
            f"Attempt within the trial (1..{_ATTEMPTS_PER_TRIAL}). While recording, "
            "changing this saves the current data and switches to that attempt's "
            "subfolder. Synced to the peer when linked.")
        self._attempt_spin.valueChanged.connect(self._on_attempt_changed)
        session_bar.addWidget(self._attempt_spin)
        # Robot selector — which physical G1 this machine is. Locked while
        # recording so the identity can never change mid-session. Persisted, so
        # each machine reopens on its own robot.
        session_bar.addWidget(QLabel("Robot:"))
        self._robot_combo = QComboBox()
        self._robot_combo.addItems(["1", "2"])
        self._robot_combo.setCurrentIndex(self._robot_id - 1)
        self._robot_combo.setMaximumWidth(60)
        self._robot_combo.setToolTip(
            "Which physical G1 this machine controls (1 or 2). Stamped into every "
            "recording's meta so the two robots can be fused into one digital twin "
            "later. Keep Participant ID + Trial identical on both robots — that is "
            "the key that pairs a robot-1 trial with its robot-2 counterpart.")
        self._robot_combo.currentIndexChanged.connect(self._on_robot_id_changed)
        session_bar.addWidget(self._robot_combo)
        session_bar.addWidget(QLabel("Output dir:"))
        self._outdir_edit = QLineEdit(self._args.output_dir)
        session_bar.addWidget(self._outdir_edit, 1)
        root_layout.addLayout(session_bar)

        # ── Setup bar (per-machine; persisted; no CLI flags needed) ────────────
        setup_bar = QHBoxLayout()
        # (The head-cam Jetson IP box and the "fuse head cam into crate" checkbox used to
        # live here — removed: the IP is fixed per robot (config headcam_host, default
        # 192.168.123.164) and the head cam is ALWAYS fused as a crate camera now.)
        # Video streaming toggle (mediamtx + zed_to_mediamtx.sh). On by default;
        # the actual start is kicked off shortly after launch in __init__.
        self._btn_stream_toggle = QPushButton("📹  Video Streaming: ON")
        self._btn_stream_toggle.clicked.connect(self._on_toggle_streaming)
        self._btn_stream_toggle.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; padding: 6px; }")
        self._btn_stream_toggle.setToolTip(
            "Start/stop the ZED video streaming pipeline (mediamtx + zed_to_mediamtx.sh). "
            "On by default.")
        setup_bar.addWidget(self._btn_stream_toggle)
        self._btn_reboot_zed = QPushButton("⟳  Reboot ZED Streaming")
        self._btn_reboot_zed.clicked.connect(self._on_reboot_zed_streaming)
        self._btn_reboot_zed.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_reboot_zed.setToolTip(
            "FULL reboot of the ZED video pipeline WITHOUT restarting the manager — a fresh "
            "mediamtx (even an adopted/orphaned one) + a fresh publisher + camera reopen, the "
            "equivalent of what a manager restart does to the stream (and more thorough than "
            "the toggle, which spares an adopted mediamtx). Disabled WHILE an attempt records "
            "(it would break the take); fine with a session open between takes.")
        setup_bar.addWidget(self._btn_reboot_zed)
        self._btn_reboot_webrtc = QPushButton("⟳  Reboot WebRTC / Haptics")
        self._btn_reboot_webrtc.clicked.connect(self._on_reboot_webrtc)
        self._btn_reboot_webrtc.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_reboot_webrtc.setToolTip(
            "Reboot the WebRTC server WITHOUT restarting the manager — recovers a wedged "
            "connection or a crashed out-of-process WebRTC subprocess, and REBUILDS the haptics "
            "data channel (the fix when vibrotactile stops reaching the gloves). The Quest "
            "reconnects in ~5-10 s and the toggles re-assert automatically. The bHaptics gloves "
            "connect to the HEADSET over Bluetooth, so if they still don't buzz after this, "
            "re-pair them on the headset. Disabled mid-take (it drops the Quest link).")
        setup_bar.addWidget(self._btn_reboot_webrtc)
        setup_bar.addWidget(QLabel("Link:"))
        self._link_role_combo = QComboBox()
        self._link_role_combo.addItems(["None", "Master", "Peer"])
        self._link_role_combo.setCurrentText(self._cfg["link_role"].capitalize())
        self._link_role_combo.setToolTip(
            "Master = robot 1 (control board + fuses robot 2); Peer = robot 2 "
            "(obeys master). None = standalone.")
        self._link_role_combo.currentIndexChanged.connect(self._on_setup_changed)
        setup_bar.addWidget(self._link_role_combo)
        setup_bar.addWidget(QLabel("Link IP:"))
        self._link_host_edit = QLineEdit(self._cfg["link_host"])
        self._link_host_edit.setMaximumWidth(150)
        self._link_host_edit.setPlaceholderText("blank = this PC listens")
        self._link_host_edit.setToolTip(
            "Address of the OTHER manager. Leave BLANK on the machine that LISTENS "
            "(the one with the reachable/open IP); on the other machine enter that "
            "listener's IP to connect. Independent of Master/Peer role.")
        self._link_host_edit.editingFinished.connect(self._on_setup_changed)
        setup_bar.addWidget(self._link_host_edit)

        self._btn_link_connect = QPushButton("🔗  Connect Link")
        self._btn_link_connect.clicked.connect(self._on_connect_link)
        self._btn_link_connect.setStyleSheet("QPushButton { padding: 4px; }")
        setup_bar.addWidget(self._btn_link_connect)
        self._btn_link_disconnect = QPushButton("✂  Disconnect Link")
        self._btn_link_disconnect.clicked.connect(self._on_disconnect_link)
        self._btn_link_disconnect.setStyleSheet("QPushButton { padding: 4px; }")
        self._btn_link_disconnect.setToolTip(
            "Tear down the peer link and stop reconnect attempts. Use Connect Link to "
            "re-establish it. When linked, the Data Collection + Teleop buttons drive "
            "BOTH robots.")
        setup_bar.addWidget(self._btn_link_disconnect)
        # Link status light + short peer label (right next to Connect/Disconnect). Grey =
        # no link role, red = role set but peer down, green = peer connected.
        self._link_light = QLabel("●")
        self._link_light.setStyleSheet(f"color: {self._STATE_UNKNOWN}; font-size: 15px;")
        self._link_light.setToolTip("Peer link: grey = off, red = not connected, "
                                    "green = connected (buttons drive BOTH robots).")
        setup_bar.addWidget(self._link_light)
        self._link_peer_lbl = QLabel("Link: off")
        self._link_peer_lbl.setStyleSheet("font-weight: bold; color: #7f8c8d;")
        setup_bar.addWidget(self._link_peer_lbl)
        setup_bar.addSpacing(10)
        # PEER side: the master's rsync destination is AUTO-identified from the live link
        # (master IP from the socket + username/recordings dir from the master's status) —
        # no manual entry. This read-only label shows what was resolved.
        self._master_dest_lbl = QLabel("Master dest: (link down)")
        self._master_dest_lbl.setStyleSheet("font-size: 11px; color: #7f8c8d;")
        self._master_dest_lbl.setToolTip(
            "Auto-identified destination the peer rsyncs its data to, derived from the "
            "peer link. Requires the link connected + SSH to the master working.")
        setup_bar.addWidget(self._master_dest_lbl)
        self._btn_test_dest = QPushButton("🔗  Verify Master Link")
        self._btn_test_dest.setStyleSheet("QPushButton { padding: 4px; }")
        self._btn_test_dest.setToolTip(
            "PEER side: SSH to the auto-identified master destination now and confirm it's "
            "reachable + writable, so the auto-send at Stop will succeed. Runs "
            "automatically on connect; click to re-check.")
        self._btn_test_dest.clicked.connect(self._on_test_master_link)
        setup_bar.addWidget(self._btn_test_dest)
        setup_bar.addStretch()
        root_layout.addLayout(setup_bar)

        # ── System-status lights (both robots + all cameras, shared over the link) ──
        # Built here but ADDED later as a column to the right of the Linked Study panel
        # (see below) instead of a full-width band, so the window reads horizontally
        # rather than stacking tall enough to need scrolling.
        status_box = self._build_status_panel()

        # ── Main panel (columns) ───────────────────────────────────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(8)
        root_layout.addLayout(main_row)

        # -- Column 1: Teleop control --
        teleop_box = QGroupBox("Teleop Control")
        teleop_col = QVBoxLayout(teleop_box)
        self._teleop_state_lbl = QLabel("State: INITIALIZING")
        self._teleop_state_lbl.setAlignment(Qt.AlignCenter)
        font = self._teleop_state_lbl.font()
        font.setBold(True)
        self._teleop_state_lbl.setFont(font)
        teleop_col.addWidget(self._teleop_state_lbl)
        self._btn_start_teleop = QPushButton("▶  Start Teleop")
        self._btn_start_teleop.setEnabled(False)
        self._btn_start_teleop.clicked.connect(self._on_start_teleop)
        self._btn_start_teleop.setStyleSheet("QPushButton { background: #2ecc71; color: white; font-weight: bold; padding: 8px; }")
        teleop_col.addWidget(self._btn_start_teleop)
        self._btn_stop_teleop = QPushButton("■  Stop & Lower Arms")
        self._btn_stop_teleop.setEnabled(False)
        self._btn_stop_teleop.clicked.connect(self._on_stop_teleop_clicked)
        self._btn_stop_teleop.setStyleSheet("QPushButton { background: #e74c3c; color: white; font-weight: bold; padding: 8px; }")
        teleop_col.addWidget(self._btn_stop_teleop)
        # Individual "this robot only" controls — shown ONLY when linked (the buttons above
        # drive BOTH robots then). Lets you start/pause/stop THIS robot without touching the
        # peer (e.g. resume just this robot after its operator's headset auto-paused it).
        self._solo_teleop_lbl = QLabel("— this robot only —")
        self._solo_teleop_lbl.setAlignment(Qt.AlignCenter)
        self._solo_teleop_lbl.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        teleop_col.addWidget(self._solo_teleop_lbl)
        self._btn_start_teleop_solo = QPushButton("▶  Start (this robot)")
        self._btn_start_teleop_solo.clicked.connect(self._on_start_teleop_solo)
        self._btn_start_teleop_solo.setStyleSheet("QPushButton { background: #2ecc71; color: white; padding: 6px; }")
        teleop_col.addWidget(self._btn_start_teleop_solo)
        self._btn_stop_teleop_solo = QPushButton("■  Stop (this robot)")
        self._btn_stop_teleop_solo.clicked.connect(self._on_stop_teleop_solo)
        self._btn_stop_teleop_solo.setStyleSheet("QPushButton { background: #e74c3c; color: white; padding: 6px; }")
        teleop_col.addWidget(self._btn_stop_teleop_solo)
        for _w in (self._solo_teleop_lbl, self._btn_start_teleop_solo, self._btn_stop_teleop_solo):
            _w.setVisible(False)                   # revealed only when linked (see _update_buttons)
        # ── PEER robot teleop controls (MASTER only, when linked) ──────────────
        # Individually start/pause/stop the OTHER robot's teleop from here — mirrors the
        # "this robot only" controls above. Sends link commands the peer runs locally; it
        # never overrides more than the peer's own Start/Pause button would.
        self._peer_teleop_lbl = QLabel("— peer robot only —")
        self._peer_teleop_lbl.setAlignment(Qt.AlignCenter)
        self._peer_teleop_lbl.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        teleop_col.addWidget(self._peer_teleop_lbl)
        self._btn_start_teleop_peer = QPushButton("▶  Start (peer)")
        self._btn_start_teleop_peer.clicked.connect(self._on_start_teleop_peer)
        self._btn_start_teleop_peer.setStyleSheet("QPushButton { background: #2ecc71; color: white; padding: 6px; }")
        teleop_col.addWidget(self._btn_start_teleop_peer)
        self._btn_stop_teleop_peer = QPushButton("■  Stop (peer)")
        self._btn_stop_teleop_peer.clicked.connect(self._on_stop_teleop_peer)
        self._btn_stop_teleop_peer.setStyleSheet("QPushButton { background: #e74c3c; color: white; padding: 6px; }")
        teleop_col.addWidget(self._btn_stop_teleop_peer)
        for _w in (self._peer_teleop_lbl, self._btn_start_teleop_peer, self._btn_stop_teleop_peer):
            _w.setVisible(False)                   # revealed only on the master when linked
        # ── divider: teleop controls (above) │ hardware / reboot / settings (below) ──
        teleop_col.addSpacing(8)
        _hw_sep = QFrame()
        _hw_sep.setFrameShape(QFrame.HLine)
        _hw_sep.setFrameShadow(QFrame.Sunken)
        _hw_sep.setStyleSheet("color: #b0b0b0;")
        teleop_col.addWidget(_hw_sep)
        _hw_hdr = QLabel("Hardware · Reboot · Calibration · Settings")
        _hw_hdr.setAlignment(Qt.AlignCenter)
        _hw_hdr.setStyleSheet("color: #7f8c8d; font-size: 10px; font-weight: bold;")
        teleop_col.addWidget(_hw_hdr)
        teleop_col.addSpacing(4)
        self._btn_motor_monitor = QPushButton("🌡  Motor Monitor")
        self._btn_motor_monitor.clicked.connect(self._on_show_motor_monitor)
        self._btn_motor_monitor.setStyleSheet("QPushButton { padding: 8px; }")
        teleop_col.addWidget(self._btn_motor_monitor)
        self._btn_reboot_neck = QPushButton("⟳  Reboot Neck")
        self._btn_reboot_neck.clicked.connect(self._on_reboot_neck)
        self._btn_reboot_neck.setStyleSheet("QPushButton { padding: 8px; }")
        self._btn_reboot_neck.setToolTip(
            "Reboot the Ostrich-neck Dynamixel servos + reconnect the serial link, WITHOUT "
            "restarting the manager. Fixes the neck when it stops responding to head motion "
            "(a latched servo overload/overheat error, or a wedged serial link — common after "
            "taking the headset off and back on). Runs off the UI thread; watch the Log.")
        teleop_col.addWidget(self._btn_reboot_neck)
        self._btn_reboot_hands = QPushButton("⟳  Reboot Hands")
        self._btn_reboot_hands.clicked.connect(self._on_reboot_hands)
        self._btn_reboot_hands.setStyleSheet("QPushButton { padding: 8px; }")
        self._btn_reboot_hands.setToolTip(
            "Reboot the Inspire hands + their force/tactile sensors WITHOUT restarting the "
            "manager — the same Modbus bring-up done at manager bootup (reconnect + restart "
            "the actuator/sensor worker threads). Use when a hand or its force sensors stop "
            "responding. Fingers hold their pose during the ~1-2 s reconnect. Runs off the UI "
            "thread; watch the Log / terminal.")
        teleop_col.addWidget(self._btn_reboot_hands)
        self._btn_hand_dofs = QPushButton("🖐  Hand DOFs (Enable / Disable)")
        self._btn_hand_dofs.clicked.connect(self._on_hand_dofs)
        self._btn_hand_dofs.setStyleSheet("QPushButton { padding: 8px; }")
        self._btn_hand_dofs.setToolTip(
            "Enable/disable individual hand actuators on THIS robot. A disabled DOF is no longer "
            "commanded — it's frozen at rest — so a broken actuator (e.g. a sheared thumb-rotation "
            "shaft) is never driven and the recording + digital twin match what the hand actually "
            "does. Per-machine, saved between sessions. All DOFs enabled by default.")
        teleop_col.addWidget(self._btn_hand_dofs)
        self._btn_eco = QPushButton(f"🌱  Eco Mode: {'ON' if self._eco_on else 'OFF'}")
        self._btn_eco.setToolTip(
            "Relax the arms toward damping while the operator is idle (re-engaging "
            "instantly on motion) and back off near a shoulder-pitch motor's thermal "
            "limit, to keep the shoulder-pitch motors from overheating. Only affects "
            "the arms during teleop; no effect while actively moving below the "
            "derate temperature.")
        self._btn_eco.clicked.connect(self._on_toggle_eco)
        eco_on_style  = "QPushButton { background: #27ae60; color: white; padding: 8px; }"
        eco_off_style = "QPushButton { padding: 8px; }"
        self._btn_eco.setStyleSheet(eco_on_style if self._eco_on else eco_off_style)
        teleop_col.addWidget(self._btn_eco)
        self._btn_compliance = QPushButton(
            f"🤝  Arm Compliance: {'ON' if self._comp_on else 'OFF'}")
        self._btn_compliance.setToolTip(
            "For collaborative manipulation: when an arm joint's torque nears the "
            "overcurrent limit (e.g. a partner robot forcing a shared crate), soften "
            "that joint so it yields with the crate instead of tripping, then "
            "re-stiffen and return to the matched pose when the force passes. "
            "Tune with --comp_tau_start / --comp_tau_full.")
        self._btn_compliance.clicked.connect(self._on_toggle_compliance)
        self._btn_compliance.setStyleSheet(eco_on_style if self._comp_on else eco_off_style)
        teleop_col.addWidget(self._btn_compliance)
        self._btn_force_calib = QPushButton("🤏  Calibration")
        self._btn_force_calib.clicked.connect(self._on_show_force_calibration)
        self._btn_force_calib.setStyleSheet("QPushButton { padding: 8px; }")
        teleop_col.addWidget(self._btn_force_calib)
        self._btn_kneel_calib = QPushButton("🧎  Kneeling Pose Calibration")
        self._btn_kneel_calib.clicked.connect(self._on_show_kneeling_calibration)
        self._btn_kneel_calib.setStyleSheet("QPushButton { padding: 8px; }")
        teleop_col.addWidget(self._btn_kneel_calib)
        self._btn_abduction = QPushButton("💪  Arm Abduction Limit")
        self._btn_abduction.clicked.connect(self._on_show_abduction_limit)
        self._btn_abduction.setStyleSheet("QPushButton { padding: 8px; }")
        teleop_col.addWidget(self._btn_abduction)
        # Palm haptics gate: the palm IS a live haptics region (read + streamed like the
        # five fingers); this mutes/unmutes JUST the palm channel at the source, which
        # silences both the vibrotactile AND the audio haptics Unity derives from it.
        self._palm_haptics_on = bool(self._cfg.get("palm_haptics", True))
        self._btn_palm_haptics = QPushButton(
            f"🖐  Palm Haptics: {'ON' if self._palm_haptics_on else 'OFF'}")
        self._btn_palm_haptics.clicked.connect(self._on_toggle_palm_haptics)
        self._btn_palm_haptics.setStyleSheet(
            "QPushButton { background: %s; color: white; padding: 8px; }"
            % ("#27ae60" if self._palm_haptics_on else "#7f8c8d"))
        self._btn_palm_haptics.setToolTip(
            "Include the PALM touch sensor in the haptics stream (vibrotactile + audio "
            "in the headset), alongside the five fingertips. OFF mutes only the palm "
            "channel — fingertip haptics are unaffected. Persisted per machine.")
        teleop_col.addWidget(self._btn_palm_haptics)
        teleop_col.addStretch()
        self._webrtc_lbl = self._status_label("WebRTC: Disconnected", "red")
        teleop_col.addWidget(self._webrtc_lbl)
        main_row.addWidget(teleop_box)

        # -- Column 2: Data collection --
        data_box = QGroupBox("Data Collection")
        data_col = QVBoxLayout(data_box)
        # ── Conduct-study flow, stacked in the order things are used ──────────────
        # 1) checklist  2) trial ordering  3) Prepare Study  4) Start Session
        # 5) Start/Stop Attempt  6) Continue-to-next  7) quality + redo  8) Stop Session.
        # 1. Pre-flight checklist.
        self._btn_checklist = QPushButton("📋  Pre-Flight Checklist")
        self._btn_checklist.clicked.connect(self._on_open_checklist)
        self._btn_checklist.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_checklist.setToolTip(
            "Open the study operating checklist (setup order + per-participant steps + "
            "recovery tips) in your text editor.")
        data_col.addWidget(self._btn_checklist)
        # (Voice-calibration button now lives above the Transcription toggle — see
        # _btn_calib_voices_tr — so it sits with the other transcription controls.)
        # 2. Trial-order ranking: DRAG the conditions (or use ↑/↓) to set the recording
        # sequence for this participant group. The Continue button follows this order;
        # the current condition's row is highlighted while a session is open.
        data_col.addWidget(QLabel("Trial order (drag to rank):"))
        order_row = QHBoxLayout()
        self._order_list = _RankListWidget()
        for c in _TRIAL_CONDITIONS:
            self._order_list.addItem(QListWidgetItem(c))
        self._order_list.setDragDropMode(QAbstractItemView.InternalMove)   # drag to reorder
        self._order_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._order_list.setFixedHeight(4 * 22 + 8)   # ~4 rows, compact
        self._order_list.setToolTip(
            "Recording order of the trial conditions for this participant group. Drag a "
            "row (or select it and use ↑/↓) to rank them. The Continue button steps "
            "through this order; you can also jump to any condition with the Trial "
            "dropdown.")
        self._order_list.reordered.connect(self._on_order_changed)   # fires on drag reorder
        order_row.addWidget(self._order_list, 1)
        # ↑ / ↓ move buttons (explicit alternative to dragging).
        move_col = QVBoxLayout()
        self._btn_order_up = QPushButton("↑")
        self._btn_order_up.setMaximumWidth(30)
        self._btn_order_up.setToolTip("Move the selected trial up in the order.")
        self._btn_order_up.clicked.connect(lambda: self._move_order_item(-1))
        move_col.addWidget(self._btn_order_up)
        self._btn_order_down = QPushButton("↓")
        self._btn_order_down.setMaximumWidth(30)
        self._btn_order_down.setToolTip("Move the selected trial down in the order.")
        self._btn_order_down.clicked.connect(lambda: self._move_order_item(+1))
        move_col.addWidget(self._btn_order_down)
        move_col.addStretch()
        order_row.addLayout(move_col)
        data_col.addLayout(order_row)
        self._order_lbl = QLabel("")
        self._order_lbl.setStyleSheet("font-size: 11px; color: #e67e22;")
        self._order_lbl.setWordWrap(True)
        data_col.addWidget(self._order_lbl)
        # Mode toggle: User Study (default; no Good/Bad attempt marking) vs Data Collection
        # (exposes the demo-quality buttons). Everything else is identical between modes.
        self._btn_mode = QPushButton()
        self._btn_mode.clicked.connect(self._on_toggle_mode)
        self._btn_mode.setToolTip(
            "Switch between User Study Mode (default) and Data Collection Mode. The ONLY "
            "difference: Data Collection Mode shows the Mark Good / Mark Bad attempt-quality "
            "buttons for curating demos; User Study Mode hides them. Persisted.")
        data_col.addWidget(self._btn_mode)
        # 3. One-click session prep: calibrate everything that needs calibrating before a
        # participant group (both robots' head cams + the crate cameras + the peer's
        # crate contribution, and push the bundle to the peer). Run once per group.
        self._btn_prepare_study = QPushButton("🚀  Prepare Study (Calibrate All)")
        self._btn_prepare_study.clicked.connect(self._on_prepare_study)
        self._btn_prepare_study.setStyleSheet(
            "QPushButton { background: #2980b9; color: white; font-weight: bold; padding: 8px; }")
        self._btn_prepare_study.setToolTip(
            "ONE-CLICK prep before a participant group: anchors every camera to table "
            "tag 0 on BOTH robots (head-cam base localization + external crate cameras + "
            "the peer's head-cam crate contribution) and pushes the crate bundle to the "
            "peer. Watch the Log + status lines; then just Start Session per participant.")
        data_col.addWidget(self._btn_prepare_study)
        # Peer (other robot) live status — shown only while linked. When a link is up the
        # buttons below drive BOTH robots (labels gain "(BOTH)"); this line reports the
        # peer's teleop/recording/calibration so you can confirm they're in step.
        self._link_peer_status_lbl = QLabel("(no peer status yet)")
        self._link_peer_status_lbl.setStyleSheet("font-size: 11px; color: #2980b9;")
        self._link_peer_status_lbl.setWordWrap(True)
        self._link_peer_status_lbl.setVisible(False)
        data_col.addWidget(self._link_peer_status_lbl)
        data_col.addSpacing(4)
        # 4-6. Session/attempt model: "Start Session" OPENS the session (folder,
        # calibration gates, subsystems — everything ready to record) but records NOTHING
        # until "Start Attempt" is pressed; that button toggles to "Stop Attempt" which
        # ends the take (downtime between attempts is never recorded). The Continue button
        # then advances to the next run — rolling into the next trial after the last
        # attempt. "Stop Session" (confirmation-guarded) finalizes + saves everything.
        self._btn_start_rec = QPushButton("🎬  Start Session")
        self._btn_start_rec.setEnabled(False)
        self._btn_start_rec.clicked.connect(self._on_session_toggle)
        self._btn_start_rec.setStyleSheet("QPushButton { background: #e67e22; color: white; font-weight: bold; padding: 8px; }")
        data_col.addWidget(self._btn_start_rec)
        # Practice round (humans move the box BY HAND, no robots) — records into the
        # transcript only. Needs an open session; toggles to End Practice.
        self._btn_practice_main = QPushButton("▶  Start Practice (no robot)")
        self._btn_practice_main.clicked.connect(self._on_practice_toggle)
        self._btn_practice_main.setStyleSheet(
            "QPushButton { background: #8e44ad; color: white; font-weight: bold; padding: 8px; }")
        self._btn_practice_main.setToolTip(
            "Start/stop the practice round (no robots): participants lift/flip the box "
            "by hand while ONLY the transcript (labeled 'practice') records — no robot "
            "data (no ZED video or crate pose). Open a session first.")
        data_col.addWidget(self._btn_practice_main)
        self._btn_attempt = QPushButton("●  Start Attempt")
        self._btn_attempt.setEnabled(False)
        self._btn_attempt.clicked.connect(self._on_attempt_toggle)
        self._btn_attempt.setStyleSheet("QPushButton { background: #c0392b; color: white; font-weight: bold; padding: 8px; }")
        self._btn_attempt.setToolTip(
            "Start recording the CURRENT (condition, attempt); the SAME button becomes "
            "'End Attempt' while recording (classify / redo BEFORE ending — they act on "
            "the recording take). One take per slot: advance with Continue for the next "
            "run. Synced to the peer when linked.")
        data_col.addWidget(self._btn_attempt)
        # Running-attempt timer (elapsed while a take records; red past 3 min).
        self._attempt_timer_lbl = self._status_label("Attempt: —", "gray")
        self._attempt_timer_lbl.setStyleSheet("font-weight: bold;")
        data_col.addWidget(self._attempt_timer_lbl)
        # 6. ONE Continue button (replaces Next Trial + Next Attempt): steps to the next
        # attempt, and after the trial's final attempt rolls into the NEXT trial in the
        # ranked order (with a confirmation at the trial boundary). Disabled while a take
        # is recording — press Stop Attempt first.
        self._btn_next = QPushButton("⏭  Continue")
        self._btn_next.setEnabled(False)
        self._btn_next.clicked.connect(self._on_next_button)
        self._btn_next.setStyleSheet("QPushButton { padding: 6px; font-weight: bold; }")
        self._btn_next.setToolTip(
            "Advance to the next run: the next attempt of this trial, or — after the "
            f"trial's final attempt ({_ATTEMPTS_PER_TRIAL}) — the next trial in the "
            "ranked order (asks for confirmation at a trial change). Disabled while "
            "recording: Stop Attempt first. Synced to the peer when linked.")
        data_col.addWidget(self._btn_next)
        # 7. Redo: DISCARD the current attempt/trial's data and re-record it from scratch —
        # synced so BOTH robots redo together. For when something goes wrong mid-trial.
        redo_row = QHBoxLayout()
        self._btn_redo_attempt = QPushButton("🔄  Redo Attempt")
        self._btn_redo_attempt.setEnabled(False)
        self._btn_redo_attempt.clicked.connect(self._on_redo_attempt)
        self._btn_redo_attempt.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; padding: 6px; }")
        self._btn_redo_attempt.setToolTip(
            "DISCARD this attempt's recorded data and re-record the SAME attempt from "
            "scratch (both robots, in sync). Use when a take goes wrong. Only while "
            "recording.")
        redo_row.addWidget(self._btn_redo_attempt)
        self._btn_redo_trial = QPushButton("🔄  Redo Trial")
        self._btn_redo_trial.setEnabled(False)
        self._btn_redo_trial.clicked.connect(self._on_redo_trial)
        self._btn_redo_trial.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; padding: 6px; }")
        self._btn_redo_trial.setToolTip(
            "DISCARD ALL attempts of the CURRENT condition and restart it from attempt 1 "
            "(both robots, in sync). Heavier than Redo Attempt. Only while recording.")
        redo_row.addWidget(self._btn_redo_trial)
        # Per-attempt demo quality — label the CURRENT take good/bad (synced to both
        # robots) so you can keep only clean demos when training. Written to meta.json.
        # (Added to the column BEFORE the redo row: classify first, redo if needed.)
        qual_row = QHBoxLayout()
        self._btn_mark_good = QPushButton("✓  Mark Good")
        self._btn_mark_good.setEnabled(False)
        self._btn_mark_good.clicked.connect(lambda: self._on_mark_attempt("good"))
        self._btn_mark_good.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; padding: 6px; }")
        self._btn_mark_good.setToolTip(
            "Label the CURRENT attempt a GOOD demo (both robots) — written to meta.json "
            "(attempt_quality) so the dataset builder can keep only clean takes.")
        qual_row.addWidget(self._btn_mark_good)
        self._btn_mark_bad = QPushButton("✗  Mark Bad")
        self._btn_mark_bad.setEnabled(False)
        self._btn_mark_bad.clicked.connect(lambda: self._on_mark_attempt("bad"))
        self._btn_mark_bad.setStyleSheet(
            "QPushButton { background: #95a5a6; color: white; padding: 6px; }")
        self._btn_mark_bad.setToolTip(
            "Label the CURRENT attempt a BAD demo (both robots). Prefer Redo Attempt to "
            "re-record; use this to keep-but-flag a take.")
        qual_row.addWidget(self._btn_mark_bad)
        self._attempt_quality_lbl = self._status_label("Quality: unrated", "gray")
        qual_row.addWidget(self._attempt_quality_lbl)
        data_col.addLayout(qual_row)     # classify the take…
        data_col.addLayout(redo_row)     # …then redo it if it was bad
        # (No separate Stop buttons — Start Attempt / Start Session are single toggles that
        # become End Attempt / End Session while active. Classify / redo act on the take
        # WHILE it records; do those BEFORE End Attempt.)
        data_col.addSpacing(4)
        # Mid-study save/resume: survive a manager relaunch (either machine or both)
        # without losing the session — auto-saved at every state change too.
        resume_row = QHBoxLayout()
        self._btn_save_study = QPushButton("💾  Save State")
        self._btn_save_study.setEnabled(False)
        self._btn_save_study.clicked.connect(self._on_save_study)
        self._btn_save_study.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_save_study.setToolTip(
            "Snapshot the mid-study state on BOTH machines (session id, participant, "
            "condition/attempt pointer, recorded-takes ledger, trial order) to "
            "study_resume.json beside the recordings. Also auto-saved at every "
            "session state change — this button just forces it.")
        resume_row.addWidget(self._btn_save_study)
        self._btn_resume_study = QPushButton("📥  Resume Study")
        self._btn_resume_study.setEnabled(False)
        self._btn_resume_study.clicked.connect(self._on_resume_study)
        self._btn_resume_study.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_resume_study.setToolTip(
            "After a manager relaunch mid-study: restore the saved session (ARMED at "
            "the exact condition/attempt, with completed takes still locked out). "
            "Synced — the payload carries the full state, so one click resumes BOTH "
            "machines, even if the other side was also restarted. Then: Bootup Robot "
            "→ Start Teleop → Start Attempt to continue.")
        resume_row.addWidget(self._btn_resume_study)
        data_col.addLayout(resume_row)
        # (Calibrate Voices moved up under the Pre-Flight Checklist button; the
        # Transcription toggle + transcript status moved under Robot Localization.)
        data_col.addSpacing(4)
        # (The video-streaming toggle moved to the top setup bar.)
        self._rec_body_pose_lbl  = self._status_label("Body Pose CSV: Off", "gray")
        self._rec_robot_lbl      = self._status_label("Robot State CSV: Off", "gray")
        self._rec_zed_stream_lbl = self._status_label("ZED Streaming: Off", "gray")
        self._rec_zed_rec_lbl    = self._status_label("ZED Recording: Off", "gray")
        self._rec_crate_lbl      = self._status_label("Crate: off", "gray")
        for lbl in [self._rec_body_pose_lbl, self._rec_robot_lbl,
                    self._rec_zed_stream_lbl, self._rec_zed_rec_lbl,
                    self._rec_crate_lbl]:
            data_col.addWidget(lbl)
        # ── Cameras & Calibration — its own group, stacked BELOW the conduct-study
        # flow: everything here is setup/recovery, not per-participant interaction.
        calib_box = QGroupBox("Cameras & Calibration")
        calib_col = QVBoxLayout(calib_box)
        # Pre-flight camera calibration gate: before recording, verify/auto-anchor
        # every camera to the table tag (id 0) — this robot's head cam (its base pose)
        # AND the external RealSense crate cameras — so all the digital-twin data lands
        # in ONE shared table frame. Off = skip (bench testing without cameras).
        self._require_calib_chk = QCheckBox("Calibrate cameras to table tag before recording")
        self._require_calib_chk.setChecked(bool(self._cfg.get("require_calibration", True)))
        self._require_calib_chk.setToolTip(
            "Before recording, make sure every camera is anchored to the table "
            "AprilTag (id 0): this robot's head camera (base localization) and the 2 "
            "external RealSense crate cameras (auto-anchored). When linked, the other "
            "robot's head camera is checked too. If a camera can't see tag 0 you'll be "
            "warned before any data is recorded. Uncheck to skip.")
        self._require_calib_chk.stateChanged.connect(self._on_setup_changed)
        calib_col.addWidget(self._require_calib_chk)
        # Manual backup to the on-start auto-calibration: anchor every camera to tag 0
        # now (fresh crate anchor + head-cam lock check), any time.
        self._btn_calibrate_now = QPushButton("📐  Calibrate Cameras Now")
        # Drives BOTH robots when linked (local + remote calibrate), this robot only when
        # not — _on_calibrate_both already branches on the link state.
        self._btn_calibrate_now.clicked.connect(self._on_calibrate_both)
        self._btn_calibrate_now.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_calibrate_now.setToolTip(
            "Anchor every camera to the table AprilTag (id 0) right now: this robot's "
            "head camera (base pose) and the external RealSense crate cameras (a fresh "
            "one-time anchor). After anchoring, tag 0 may be obscured during "
            "manipulation — tracking stays in the table frame as long as the cameras "
            "don't move. Backup to the automatic check at Start Session.")
        calib_col.addWidget(self._btn_calibrate_now)
        # ── Crate (AprilTag) tracking controls ──
        calib_col.addSpacing(6)
        self._crate_hdr = QLabel("Crate Tracking")
        self._crate_hdr.setStyleSheet("font-weight: bold;")
        calib_col.addWidget(self._crate_hdr)
        self._btn_crate_track = QPushButton("🎯  Start Crate Tracking")
        self._btn_crate_track.clicked.connect(self._on_toggle_crate_tracking)
        self._btn_crate_track.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_crate_track.setToolTip(
            "Open the RealSense cameras and start tracking the crate. Start this "
            "(then Anchor) BEFORE recording, so crate pose is captured in sync with "
            "the rest of the data. No effect on the camera-less robot's machine.")
        calib_col.addWidget(self._btn_crate_track)
        self._btn_anchor_cameras = QPushButton("📐  Anchor Cameras")
        self._btn_anchor_cameras.setEnabled(False)
        self._btn_anchor_cameras.clicked.connect(self._on_anchor_cameras)
        self._btn_anchor_cameras.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_anchor_cameras.setToolTip(
            "One-time: capture each camera's pose from the table tag (id 0), with "
            "the table tag visible to both cameras, BEFORE recording. Static "
            "cameras only.")
        calib_col.addWidget(self._btn_anchor_cameras)
        self._btn_reanchor_bump = QPushButton("🔧  Re-Anchor Bumped Cam")
        self._btn_reanchor_bump.setEnabled(False)
        self._btn_reanchor_bump.clicked.connect(self._on_reanchor_bumped_camera)
        self._btn_reanchor_bump.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_reanchor_bump.setToolTip(
            "A crate camera got bumped? Point it back at the table tag (id 0) and "
            "re-capture JUST that camera's pose — the others keep their anchors. Works "
            "DURING recording (tag 0 is the fixed world origin, so the frame doesn't "
            "move); crate frames recorded before the re-anchor may be off.")
        calib_col.addWidget(self._btn_reanchor_bump)
        self._btn_recalib_crate = QPushButton("🔧  Recalibrate Crate")
        self._btn_recalib_crate.clicked.connect(self._on_recalibrate_crate)
        self._btn_recalib_crate.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_recalib_crate.setToolTip(
            "Re-learn the crate's tag bundle: stops tracking to free the cameras, "
            "then opens the calibration tool in a new window. When it finishes, "
            "click Start Crate Tracking to reload the new crate_bundle.json.")
        calib_col.addWidget(self._btn_recalib_crate)

        # ── Crate Contribution (PEER only): stream this robot's head-cam crate
        # observations to the MASTER so its fusion has an extra viewpoint. Uses the
        # head-cam localizer's live (waist-compensated) camera pose — no external cameras.
        self._contrib_hdr = QLabel("Crate Contribution (head cam → master)")
        self._contrib_hdr.setStyleSheet("font-weight: bold;")
        calib_col.addWidget(self._contrib_hdr)
        self._btn_crate_contribute = QPushButton("🛰  Contribute Crate Tracking")
        self._btn_crate_contribute.clicked.connect(self._on_toggle_crate_contribution)
        self._btn_crate_contribute.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_crate_contribute.setToolTip(
            "PEER: use this robot's head camera as an EXTRA crate viewpoint for the "
            "master. ON BY DEFAULT — engages automatically once the head-cam localizer is "
            "up and the master has sent the crate bundle (this button then just toggles it "
            "off/on). Enabling loads the bundle, anchors the base to table tag 0, and "
            "streams live crate poses to the master's fusion.")
        calib_col.addWidget(self._btn_crate_contribute)
        self._btn_contrib_anchor = QPushButton("📐  Re-Anchor Base to Table")
        self._btn_contrib_anchor.setEnabled(False)
        self._btn_contrib_anchor.clicked.connect(self._on_anchor_contribution_base)
        self._btn_contrib_anchor.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_contrib_anchor.setToolTip(
            "Re-capture this robot's base pose from table tag 0 (aim the head at tag 0). "
            "The base is fixed, so this is one-time — redo it only if the robot's base "
            "was moved.")
        calib_col.addWidget(self._btn_contrib_anchor)
        self._rec_contrib_lbl = self._status_label("Crate contribution: off", "gray")
        calib_col.addWidget(self._rec_contrib_lbl)

        # ── Robot base self-localization (this robot's head camera) ──
        calib_col.addSpacing(6)
        _base_hdr = QLabel("Robot Localization")
        _base_hdr.setStyleSheet("font-weight: bold;")
        calib_col.addWidget(_base_hdr)
        self._btn_headcam = QPushButton("📍  Start Head-Cam Localize")
        self._btn_headcam.clicked.connect(self._on_toggle_headcam)
        self._btn_headcam.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_headcam.setToolTip(
            "Use THIS robot's head D435 to find its base pose in the table frame "
            "(looks at table tag id 0). Runs on every robot — no external rig "
            "needed. Start before recording so base pose is logged in sync.")
        calib_col.addWidget(self._btn_headcam)
        self._btn_headcam_view = QPushButton("👁  Head Cam View")
        self._btn_headcam_view.clicked.connect(self._on_show_headcam_view)
        self._btn_headcam_view.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_headcam_view.setToolTip(
            "Open a live view of the head camera with the AprilTag overlay, to "
            "check the table tag (id 0) is in frame. Start Head-Cam Localize first.")
        calib_col.addWidget(self._btn_headcam_view)
        self._rec_base_lbl = self._status_label("Base: off", "gray")
        calib_col.addWidget(self._rec_base_lbl)

        # ── Room transcription (conference mic → whisper, own process → per-session
        # transcript CSV with speaker labels; no audio is ever saved) ──
        calib_col.addSpacing(6)
        # Order: Transcription ON/OFF (master switch) → Calibrate Voices → Test Microphone
        # → Open Transcript.
        self._btn_transcribe = QPushButton("🎙  Transcription: …")
        self._btn_transcribe.clicked.connect(self._on_toggle_transcribe)
        self._btn_transcribe.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_transcribe.setToolTip(
            "Live transcription of the room's conference mic (whisper on the GPU, in "
            "its own process — zero teleop impact). Writes transcript_<session>.csv "
            "into the session folder: epoch times, speaker label, take, text. RAW "
            "AUDIO IS NEVER SAVED. Calibrate Voices maps who's speaking. Turning it OFF "
            "asks for confirmation.")
        calib_col.addWidget(self._btn_transcribe)
        # Mic mode toggle: 1 conference mic (voice-ID) ⇄ 2-mic STEREO (Lark M1). Switching
        # shows/hides the mode-specific controls below and (if running) respawns the
        # transcriber in the new mode.
        self._btn_mic_mode = QPushButton("🎚  Mic Mode: …")
        self._btn_mic_mode.clicked.connect(self._on_toggle_mic_mode)
        self._btn_mic_mode.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_mic_mode.setToolTip(
            "Switch between 1 conference mic (speakers separated by a trained voice-ID model "
            "— needs Calibrate Voices) and 2-mic STEREO (a dual-channel device like the "
            "Hollyland Lark M1: LEFT = operator 1, RIGHT = operator 2, attributed by channel "
            "— no calibration needed, and talking over each other is fine).")
        calib_col.addWidget(self._btn_mic_mode)
        # Voice calibration — only shown in 1-mic (voice-ID) mode. Same action as the
        # button that used to sit under the Pre-Flight Checklist.
        self._btn_calib_voices_tr = QPushButton("🗣  Calibrate Voices")
        self._btn_calib_voices_tr.setEnabled(False)
        self._btn_calib_voices_tr.clicked.connect(self._on_calibrate_voices)
        self._btn_calib_voices_tr.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_calib_voices_tr.setToolTip(
            "Enroll operator 1 and operator 2's voices so transcript rows are labeled by "
            "speaker: press Start, then Stop when they finish the script (no fixed timer). "
            "Only voice embeddings are stored — never audio. Requires transcription ON.")
        calib_col.addWidget(self._btn_calib_voices_tr)
        # Shown only in 2-mic stereo mode: which channel is which operator.
        self._stereo_hint_lbl = self._status_label(
            "2-mic stereo: LEFT = Operator 1  ·  RIGHT = Operator 2", "#16a085")
        calib_col.addWidget(self._stereo_hint_lbl)
        # Mic self-test: record 5 s from the mic, play it back, then delete it.
        self._btn_test_mic = QPushButton("🎧  Test Microphone (5 s)")
        self._btn_test_mic.clicked.connect(self._on_test_mic)
        self._btn_test_mic.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_test_mic.setToolTip(
            "Record 5 seconds from the conference mic and play it straight back so you can "
            "hear what it captures, then the recording is deleted. Turn Transcription OFF "
            "first — only one thing can use the mic at a time. Nothing is kept on disk.")
        calib_col.addWidget(self._btn_test_mic)
        self._btn_open_transcript = QPushButton("🗒  Open Transcript Window")
        self._btn_open_transcript.clicked.connect(self._on_open_transcript)
        self._btn_open_transcript.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_open_transcript.setToolTip(
            "Open the Live Transcription window — each utterance appears as whisper finishes "
            "it, with a backlog indicator when the model runs behind realtime. The session "
            "CSV is the saved record; no audio is kept.")
        calib_col.addWidget(self._btn_open_transcript)
        self._rec_transcript_lbl = self._status_label("Transcript: off", "gray")
        calib_col.addWidget(self._rec_transcript_lbl)
        self._refresh_mic_mode_ui()      # set the mode label + show/hide mode-specific rows

        # (Cameras & Calibration moved under the Unity Commands column.)

        # NOTE: the Digital Twin panel (which absorbed the old Playback buttons) used to
        # live here; it now sits under the Wrench panel in the right-hand column (built
        # near the System Status / Wrench
        # column below) to keep the Data Collection column from stacking so tall.

        data_col.addStretch()
        main_row.addWidget(data_box)

        # -- Column 3: Unity commands --
        unity_box = QGroupBox("Unity Commands")
        unity_col = QVBoxLayout(unity_box)
        self._btn_audio_haptics = QPushButton("🔊  Audio Haptics: OFF")
        self._btn_audio_haptics.setEnabled(False)
        self._btn_audio_haptics.clicked.connect(self._on_toggle_audio_haptics)
        self._btn_audio_haptics.setStyleSheet("QPushButton { padding: 8px; }")
        unity_col.addWidget(self._btn_audio_haptics)
        self._btn_vibrotactile = QPushButton("📳  Vibrotactile: OFF")
        self._btn_vibrotactile.setEnabled(False)
        self._btn_vibrotactile.clicked.connect(self._on_toggle_vibrotactile)
        self._btn_vibrotactile.setStyleSheet("QPushButton { padding: 8px; }")
        unity_col.addWidget(self._btn_vibrotactile)
        self._btn_viewport = QPushButton("🖥️  FPV Display: ON")
        self._btn_viewport.setEnabled(False)
        self._btn_viewport.clicked.connect(self._on_show_viewport)
        self._btn_viewport.setStyleSheet(self._TOGGLE_STYLE_ON)
        unity_col.addWidget(self._btn_viewport)
        # ── Test Haptics: synthetic per-region ramp so the operator can verify the headset
        # actually plays haptics (audio or vibrotactile, whichever is enabled) before a study.
        self._test_haptics_lbl = QLabel("— test haptics (ramp per finger/palm) —")
        self._test_haptics_lbl.setAlignment(Qt.AlignCenter)
        self._test_haptics_lbl.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        unity_col.addWidget(self._test_haptics_lbl)
        self._btn_test_haptics_left = QPushButton("🖐  Test Haptics (Left)")
        self._btn_test_haptics_left.clicked.connect(lambda: self._on_test_haptics(("left",)))
        self._btn_test_haptics_left.setStyleSheet("QPushButton { padding: 8px; }")
        unity_col.addWidget(self._btn_test_haptics_left)
        self._btn_test_haptics_right = QPushButton("🖐  Test Haptics (Right)")
        self._btn_test_haptics_right.clicked.connect(lambda: self._on_test_haptics(("right",)))
        self._btn_test_haptics_right.setStyleSheet("QPushButton { padding: 8px; }")
        unity_col.addWidget(self._btn_test_haptics_right)
        self._btn_test_haptics_both = QPushButton("🖐  Test Haptics (Both)")
        self._btn_test_haptics_both.clicked.connect(
            lambda: self._on_test_haptics(("left", "right")))
        self._btn_test_haptics_both.setStyleSheet("QPushButton { padding: 8px; }")
        unity_col.addWidget(self._btn_test_haptics_both)
        # ── PEER Unity toggles (MASTER only, when linked) ──────────────────────
        # Flip the PEER's own haptics / audio / FPV toggles from here. These send a link
        # command the peer runs as if its own button was pressed (a TOGGLE) — the master
        # never forces a specific state, so it can't override anything the peer can't.
        self._peer_unity_lbl = QLabel("— peer robot toggles —")
        self._peer_unity_lbl.setAlignment(Qt.AlignCenter)
        self._peer_unity_lbl.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        unity_col.addWidget(self._peer_unity_lbl)
        self._btn_peer_audio_haptics = QPushButton("🔊  Peer Audio Haptics")
        self._btn_peer_audio_haptics.clicked.connect(self._on_peer_toggle_audio_haptics)
        self._btn_peer_audio_haptics.setStyleSheet(self._TOGGLE_STYLE_OFF)
        unity_col.addWidget(self._btn_peer_audio_haptics)
        self._btn_peer_vibrotactile = QPushButton("📳  Peer Vibrotactile")
        self._btn_peer_vibrotactile.clicked.connect(self._on_peer_toggle_vibrotactile)
        self._btn_peer_vibrotactile.setStyleSheet(self._TOGGLE_STYLE_OFF)
        unity_col.addWidget(self._btn_peer_vibrotactile)
        self._btn_peer_viewport = QPushButton("🖥️  Peer FPV Display")
        self._btn_peer_viewport.clicked.connect(self._on_peer_toggle_viewport)
        self._btn_peer_viewport.setStyleSheet(self._TOGGLE_STYLE_OFF)
        unity_col.addWidget(self._btn_peer_viewport)
        for _w in (self._peer_unity_lbl, self._btn_peer_audio_haptics,
                   self._btn_peer_vibrotactile, self._btn_peer_viewport):
            _w.setVisible(False)                   # revealed only on the master when linked
        # Cameras & Calibration lives under the Unity commands (setup/recovery controls,
        # out of the per-participant Data Collection flow). Widen the column so the
        # longer calibration button labels fit without eliding.
        unity_col.addWidget(calib_box)
        unity_col.addStretch()
        unity_box.setMinimumWidth(320)
        main_row.addWidget(unity_box)

        # The old separate "Linked Study — Both Robots" column has been folded into the
        # Data Collection + Teleop panels: when a link is established, those buttons drive
        # BOTH robots and are relabelled "(BOTH)"; unlinked, they act on this robot only.
        # See _both_suffix() + _update_buttons() for the dynamic labels and _update_link_
        # status() for the top-bar link light + peer status.

        # -- Column 4: Safety / E-Stop --
        safety_box = QGroupBox("Safety / E-Stop")
        safety_col = QVBoxLayout(safety_box)
        # 🤖 Bootup Robot — the manager never boots the robot automatically at launch;
        # this (confirmation-guarded) button runs the physical bootup sequence. Lives
        # right above the E-STOP so power-up and kill are in one place.
        self._btn_bootup = QPushButton("🤖  Bootup Robot")
        self._btn_bootup.setEnabled(False)      # enabled once init (phase 1) completes
        self._btn_bootup.clicked.connect(self._on_bootup_button)
        self._btn_bootup.setStyleSheet("QPushButton { background: #2ecc71; color: white; "
                                       "font-weight: bold; padding: 10px; }")
        self._btn_bootup.setToolTip(
            "ONE button, three states. Bootup Robot: run the physical bootup (release "
            "FSM, close hands, arms out + back, rise into the kneel, damp). Robot "
            "Booted: running — nothing to press. ♻ Reboot (after Safe Shutdown / "
            "E-STOP): boot up again WITHOUT restarting the manager — session, anchors, "
            "calibration and link are preserved so the study resumes; pose + control "
            "reset only (firmware faults need a G1 power cycle). Both modes confirm "
            "first — the robot moves.")
        safety_col.addWidget(self._btn_bootup)
        # Big, always-enabled emergency stop: instantly damps the ROBOT — the manager app
        # keeps running (see _on_estop_kill; Quit Manager below closes the app). Requires TWO
        # clicks within 1 s (with a "click again" prompt between them) so a stray single
        # click can't fire it, while staying fast/forgiving in an emergency. The cursor auto-parks over
        # this button after 15 s of mouse inactivity — tracked live from the button's rect, so it
        # follows the window if moved/scaled.
        self._ESTOP_LABEL = "🛑  KILL / E-STOP\n(click twice)"
        self._ESTOP_LABEL_DEFAULT = self._ESTOP_LABEL
        self._estop_armed = False        # first click arms; second within 1 s fires
        self._estop_armed_at = 0.0
        self._estop_arm_token = 0        # so a stale disarm timer can't reset a fresh arm
        self._btn_estop = QPushButton(self._ESTOP_LABEL)
        self._btn_estop.clicked.connect(self._on_estop_clicked)
        self._btn_estop.setStyleSheet("QPushButton { background: #c0392b; color: white; "
                                      "font-weight: bold; font-size: 15px; padding: 16px; }")
        safety_col.addWidget(self._btn_estop)
        # Safe Shutdown (ROBOT only): NOT instant — smoothly tilts the robot forward at the
        # hips so it topples FORWARD and damps it limp. The manager app stays open.
        self._btn_safe_shutdown = QPushButton("🛡  Safe Shutdown (robot)")
        self._btn_safe_shutdown.clicked.connect(self._on_safe_shutdown)
        self._btn_safe_shutdown.setStyleSheet("QPushButton { background: #e67e22; color: white; "
                                              "font-weight: bold; padding: 10px; }")
        self._btn_safe_shutdown.setToolTip(
            "ROBOT ONLY — the manager app stays open. First raises the arms OUT to the "
            "sides into a drooped 'A-pose' (extended elbows, ~0.6 m apart — no bent-elbow "
            "boxing) so they lift off the table and away from the chest, THEN tilts forward "
            "and SLOWLY folds the knees (~15 s) into a deeper kneel while leaning the pelvis "
            "+ bowing the torso progressively FORWARD so the head stays out front (it drops "
            "head-forward when damped). The arms stay out to the sides until damp mode, then "
            "everything drops limp. The E-STOP above is the instant no-tilt stop; Quit "
            "Manager below closes the app. --shutdown_wing_secs 0 skips the retract.")
        safety_col.addWidget(self._btn_safe_shutdown)
        # Quit Manager: the ONLY control that closes the APP (confirmation-guarded).
        # E-STOP / Safe Shutdown act on the robot alone, so data transfer / playback /
        # the peer link keep running after the robot is down.
        self._btn_quit = QPushButton("⏻  Quit Manager")
        self._btn_quit.clicked.connect(self._on_quit_manager)
        self._btn_quit.setStyleSheet("QPushButton { background: #34495e; color: white; "
                                     "font-weight: bold; padding: 10px; }")
        self._btn_quit.setToolTip(
            "Close the manager APP gracefully: if the robot is booted and hasn't been "
            "e-stopped / safe-shut-down, it runs the SAFE SHUTDOWN first (controlled "
            "descent), then finalizes any open session and stops cameras/streaming/"
            "link. Asks for confirmation.")
        safety_col.addWidget(self._btn_quit)
        safety_col.addStretch()
        main_row.addWidget(safety_box)

        # Cursor auto-park state: last cursor position + when it last moved, and a one-shot guard so
        # the window is raised / cursor parked only ONCE per idle period (re-armed when the mouse moves).
        self._last_seen_cursor = None
        self._last_cursor_move_time = time.monotonic()
        self._idle_parked = False

        # ── Wrench (end-effector force) verification ──────────────────────────
        # The wrench is NOT stored — only the raw data it needs is recorded by
        # the unified Start Recording. This panel is the live readout used to
        # verify the recording is sane, plus the known-mass test inputs (written
        # into the session meta) and a button to open the offline viewer.
        wrench_box = QGroupBox("Wrench (EE Force) — live verification")
        # Never grow taller than the content needs — without this the column layout
        # stretched the box and left dead space under the controls.
        wrench_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        wrench_row = QHBoxLayout(wrench_box)
        wrench_row.setSpacing(12)

        # Left sub-column: controls
        wctl = QVBoxLayout()
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Side:"))
        self._wrench_side_combo = QComboBox()
        self._wrench_side_combo.addItems(["Both", "Left", "Right"])
        opts.addWidget(self._wrench_side_combo)
        opts.addSpacing(8)
        opts.addWidget(QLabel("Known mass (g):"))
        self._wrench_mass_spin = QDoubleSpinBox()
        self._wrench_mass_spin.setRange(0.0, 5000.0)
        self._wrench_mass_spin.setDecimals(0)
        self._wrench_mass_spin.setSingleStep(50.0)
        self._wrench_mass_spin.setMaximumWidth(90)
        self._wrench_mass_spin.setToolTip(
            "Mass hung in the hand for the accuracy test. Drives the live "
            "expected-force hint and is written into the session meta.json so "
            "the viewer reports |F| vs m·g.")
        opts.addWidget(self._wrench_mass_spin)
        opts.addStretch()
        wctl.addLayout(opts)

        live_row = QHBoxLayout()
        self._btn_wrench_live = QPushButton("🔬  Live: OFF")
        self._btn_wrench_live.setEnabled(False)
        self._btn_wrench_live.clicked.connect(self._on_toggle_wrench_live)
        self._btn_wrench_live.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_wrench_live.setToolTip(
            "Compute the live end-effector wrench off the teleop hot path "
            "(own MjData, ~10 Hz). Verification only — not recorded.")
        live_row.addWidget(self._btn_wrench_live)
        self._btn_wrench_tare = QPushButton("⊘  Tare (empty)")
        self._btn_wrench_tare.setEnabled(False)
        self._btn_wrench_tare.clicked.connect(self._on_wrench_tare)
        self._btn_wrench_tare.setStyleSheet("QPushButton { padding: 6px; }")
        self._btn_wrench_tare.setToolTip(
            "Hold the arm empty & still, then click: averages the current "
            "reading into a baseline that cancels model gravity-comp error. "
            "The baseline is saved into the session meta.")
        live_row.addWidget(self._btn_wrench_tare)
        self._btn_wrench_clear_tare = QPushButton("↺")
        self._btn_wrench_clear_tare.setEnabled(False)
        self._btn_wrench_clear_tare.setMaximumWidth(36)
        self._btn_wrench_clear_tare.clicked.connect(self._on_wrench_clear_tare)
        self._btn_wrench_clear_tare.setToolTip("Clear the tare baseline.")
        live_row.addWidget(self._btn_wrench_clear_tare)
        wctl.addLayout(live_row)

        # (View Last / Open Digital Twin — the playback viewers — live in the Digital Twin
        # panel just below this Wrench panel in the same right-hand column.)
        wctl.addStretch()
        wrench_row.addLayout(wctl, 3)

        # Right sub-column: live readout
        wread = QVBoxLayout()
        wread.addWidget(QLabel("<b>Live EE force</b>"))
        self._wrench_left_lbl = self._status_label("L: —", "gray")
        self._wrench_right_lbl = self._status_label("R: —", "gray")
        self._wrench_expected_lbl = self._status_label("expected: —", "gray")
        for lbl in (self._wrench_left_lbl, self._wrench_right_lbl,
                    self._wrench_expected_lbl):
            f = lbl.font(); f.setFamily("Courier"); lbl.setFont(f)
            wread.addWidget(lbl)
        wread.addStretch()
        wrench_row.addLayout(wread, 2)

        # ── Digital Twin panel (the single playback + reconstruction viewer) ──
        # One scene for everything: table + crate + BOTH robots, EE wrench arrows, the
        # tactile fingertip/palm overlay, and synced ZED video. It gracefully shows just
        # THIS robot until the peer's data is merged in, so it also serves as the
        # single-take "view last" viewer — the old standalone Playback panel was folded in
        # here. PEER: push this robot's session to the master so both robots' data sit in
        # one place. MASTER: open the twin once both are present.
        twin_box = QGroupBox("Digital Twin")
        dt_col = QVBoxLayout(twin_box)
        # LIVE viewer toggle (the in-process MuJoCo window of THIS robot's real-time sim) —
        # distinct from the playback twin below. Closed by default so its 50 Hz render
        # doesn't compete with the WebRTC teleop loop; open it only to watch the live sim.
        self._btn_live_viewer = QPushButton("🖥  Show Live 3D Viewer")
        self._btn_live_viewer.clicked.connect(self._on_toggle_live_viewer)
        self._btn_live_viewer.setStyleSheet("QPushButton { padding: 8px; }")
        self._btn_live_viewer.setToolTip(
            "Open/close the LIVE MuJoCo viewer (this robot's real-time sim). Closed by "
            "default during studies because its 50 Hz render competes for CPU/GIL with the "
            "WebRTC teleop loop — open it only when you want to watch the sim. This is the "
            "live robot, NOT the recorded playback Digital Twin below.")
        dt_col.addWidget(self._btn_live_viewer)
        self._btn_send_master = QPushButton("📤  Send Data to Master")
        self._btn_send_master.clicked.connect(self._on_send_data_to_master)
        self._btn_send_master.setStyleSheet(
            "QPushButton { background: #16a085; color: white; font-weight: bold; padding: 8px; }")
        self._btn_send_master.setToolTip(
            "PEER: send THIS run's finished session to the master (Master data dest, above). "
            "It MERGES into the master's matching session so each attempt folder holds BOTH "
            "robots (robot1/ + robot2/). For OLD sessions from previous runs, use 'Send Past "
            "Session…' below. Background; watch the Log. Disabled while recording.")
        dt_col.addWidget(self._btn_send_master)
        # Send an ARBITRARY past session (or all) — back-fills previously-unsynced peer data
        # without recording a new session first. Peer-only, like Send Data to Master.
        self._btn_send_past = QPushButton("📤  Send Past Session…")
        self._btn_send_past.clicked.connect(self._on_send_past_session)
        self._btn_send_past.setStyleSheet("QPushButton { padding: 8px; }")
        self._btn_send_past.setToolTip(
            "PEER: pick ANY past session folder (or the recordings root to send ALL) and "
            "rsync it to the master, merging like Send Data to Master. Use this to back-fill "
            "OLD recordings you never synced — no need to record a new session first.")
        dt_col.addWidget(self._btn_send_past)
        # MASTER counterpart of Send Data to Master: PULL the peer's recordings over SSH,
        # merging into the master. Works WITHOUT the peer manager running (peer machine +
        # sshd only) — so old unsynced data can be back-filled. Master/standalone only.
        self._btn_pull_peer = QPushButton("📥  Sync Peer Data")
        self._btn_pull_peer.clicked.connect(self._on_pull_peer_data)
        self._btn_pull_peer.setStyleSheet(
            "QPushButton { background: #16a085; color: white; font-weight: bold; padding: 8px; }")
        self._btn_pull_peer.setToolTip(
            "MASTER: rsync-PULL the peer's recordings straight into this master's, merging "
            "each shared session (the peer's robot2/ lands beside robot1/). Works even when "
            "the peer MANAGER isn't running — only the peer machine + sshd need to be up — so "
            "you can back-fill OLD unsynced data. Incremental; runs in the background. The "
            "peer's IP is remembered from the link, or set peer_host in the config.")
        dt_col.addWidget(self._btn_pull_peer)
        self._btn_view_recording = QPushButton("👁  View Last")
        self._btn_view_recording.setEnabled(False)
        self._btn_view_recording.clicked.connect(self._on_view_last_recording)
        self._btn_view_recording.setStyleSheet(
            "QPushButton { background: #8e44ad; color: white; font-weight: bold; padding: 8px; }")
        self._btn_view_recording.setToolTip(
            "Open the MOST RECENT recording in the digital twin: table + crate + both "
            "robots, EE wrench + fingertip/palm forces, synced to the ZED video. Shows "
            "just this robot until the peer's data is merged in.")
        dt_col.addWidget(self._btn_view_recording)
        self._btn_open_twin = QPushButton("🧊  Open Digital Twin…")
        self._btn_open_twin.clicked.connect(self._on_open_digital_twin)
        self._btn_open_twin.setStyleSheet("QPushButton { padding: 8px; }")
        self._btn_open_twin.setToolTip(
            "Same scene as View Last, but you PICK the session (either robot's folder; its "
            "robot1/robot2 pair is auto-found). Shows just one robot until both robots' "
            "data is present (use Send Data to Master on the peer first).")
        dt_col.addWidget(self._btn_open_twin)
        self._wrench_csv_lbl = self._status_label("Last recording: —", "gray")
        dt_col.addWidget(self._wrench_csv_lbl)
        self._btn_build_dataset = QPushButton("🧱  Build Dataset")
        self._btn_build_dataset.clicked.connect(self._on_build_dataset)
        self._btn_build_dataset.setStyleSheet("QPushButton { padding: 8px; }")
        self._btn_build_dataset.setToolTip(
            "Turn a finished session into a training-ready HDF5 dataset: localizes the "
            "ZED cameras (AprilTag), aligns both robots on the master clock, resamples "
            "to 15 Hz, decodes images, and writes dataset.hdf5 + manifest. Pick the "
            "merged session folder on the master. Runs in the background; watch the Log.")
        dt_col.addWidget(self._btn_build_dataset)
        self._send_status_lbl = self._status_label("", "gray")
        dt_col.addWidget(self._send_status_lbl)
        self._dataset_status_lbl = self._status_label("", "gray")
        dt_col.addWidget(self._dataset_status_lbl)

        # ── Right-of-Data column: System Status + Wrench + Digital Twin ──
        # These used to be full-width bands / part of the tall Data Collection column,
        # which forced vertical scrolling. Stacking them in one column to the right keeps
        # the layout wide + short. The Safety / E-Stop panel stays at the far-right edge.
        # (The standalone Playback panel was folded into the Digital Twin panel — the twin
        # is now the single viewer, so View Last opens it too.)
        status_wrench_col = QWidget()
        sw_col = QVBoxLayout(status_wrench_col)
        sw_col.setContentsMargins(0, 0, 0, 0)
        sw_col.setSpacing(8)
        sw_col.addWidget(status_box)
        sw_col.addWidget(wrench_box)
        sw_col.addWidget(twin_box)
        sw_col.addStretch()
        main_row.insertWidget(main_row.indexOf(safety_box), status_wrench_col)

        # ── Log area ──────────────────────────────────────────────────────────
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(150)
        self._log_text.setFont(QtGui.QFont("Courier", 9))
        log_layout.addWidget(self._log_text)
        root_layout.addWidget(log_box)

        # ── Latency monitor ───────────────────────────────────────────────────
        latency_box = QGroupBox("Latency Monitor")
        latency_layout = QHBoxLayout(latency_box)
        latency_layout.setSpacing(8)

        # Scrolling graph (takes most of the width)
        self._latency_graph = LatencyGraphWidget(
            self._latency_tracker, ik_rate=self._args.ik_rate
        )
        latency_layout.addWidget(self._latency_graph, stretch=4)

        # ZED status panel (right column)
        zed_panel = QWidget()
        zed_col = QVBoxLayout(zed_panel)
        zed_col.setSpacing(4)
        zed_col.setContentsMargins(4, 4, 4, 4)
        zed_title = QLabel("ZED Camera")
        zed_title.setStyleSheet("font-weight: bold;")
        zed_col.addWidget(zed_title)
        self._zed_status_lbl   = self._status_label("Stream: Off", "gray")
        self._zed_uptime_lbl   = self._status_label("Uptime: —", "gray")
        self._zed_rec_lbl2     = self._status_label("Rec: Off", "gray")
        self._pose_rate_lbl    = self._status_label("Pose rate: — Hz", "gray")
        for lbl in [self._zed_status_lbl, self._zed_uptime_lbl,
                    self._zed_rec_lbl2, self._pose_rate_lbl]:
            zed_col.addWidget(lbl)
        note = QLabel("RTSP latency ~100-300ms\n(GStreamer → mediamtx)")
        note.setStyleSheet("color: #666; font-size: 10px;")
        zed_col.addWidget(note)
        zed_col.addStretch()
        latency_layout.addWidget(zed_panel, stretch=1)

        # End-to-end latency panel (right column): live assembled totals + breakdowns.
        e2e_panel = QWidget()
        e2e_col = QVBoxLayout(e2e_panel)
        e2e_col.setSpacing(4)
        e2e_col.setContentsMargins(4, 4, 4, 4)
        e2e_title = QLabel("End-to-End Latency")
        e2e_title.setStyleSheet("font-weight: bold;")
        e2e_col.addWidget(e2e_title)
        self._lat_video_lbl     = self._status_label("Video (cam→eyes): —", "gray")
        self._lat_video_bd_lbl  = QLabel("")
        self._lat_video_bd_lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._lat_teleop_lbl    = self._status_label("Teleop (move→robot): —", "gray")
        self._lat_teleop_bd_lbl = QLabel("")
        self._lat_teleop_bd_lbl.setStyleSheet("color: #888; font-size: 10px;")
        # Quest data-link health: inbound body_pose delivery rate + STALLS (a stall = the
        # reliable/ordered channel waiting on a WiFi-loss retransmit → teleop lag + flaky
        # commands). Should read ~0 stalls once body_pose is unreliable on the Quest side.
        # Prominent at-a-glance link status (OK / DEGRADED / FROZEN) driven by the SAME
        # pose_age + RTT signals the teleop watchdog trips on — so you can see the link
        # heading toward a pause BEFORE it happens.
        self._lat_link_status_lbl = QLabel("Quest link: —")
        self._lat_link_status_lbl.setStyleSheet(
            "color: gray; font-weight: bold; font-size: 12px;")
        self._lat_link_lbl = self._status_label("Link (Quest data): —", "gray")
        for w in [self._lat_video_lbl, self._lat_video_bd_lbl,
                  self._lat_teleop_lbl, self._lat_teleop_bd_lbl,
                  self._lat_link_status_lbl, self._lat_link_lbl]:
            e2e_col.addWidget(w)
        e2e_note = QLabel("measured: net/jit/dec/proc/act\nestimated: cap/disp/track")
        e2e_note.setStyleSheet("color: #666; font-size: 9px;")
        e2e_col.addWidget(e2e_note)
        e2e_col.addStretch()
        latency_layout.addWidget(e2e_panel, stretch=1)

        root_layout.addWidget(latency_box)

        self.setStatusBar(QStatusBar())

        # Hide UI that doesn't apply to this manager's role (peer hides the external-camera
        # crate controls; master/standalone hides the peer's 'Send Data to Master').
        self._apply_role_visibility()
        # Apply the User Study / Data Collection mode (hides Good/Bad marking in study mode).
        self._apply_mode_visibility()

    @staticmethod
    def _status_label(text: str, color: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
        return lbl

    @staticmethod
    def _set_latency_label(lbl: QLabel, name: str, total_ms: float, ok_ms: float, warn_ms: float):
        """Update a latency total label, colour-coded green/orange/red by the thresholds."""
        color = "green" if total_ms < ok_ms else ("orange" if total_ms < warn_ms else "red")
        lbl.setText(f"{name}: {total_ms:.0f} ms")
        lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")

    def _log(self, msg: str):
        """Log a line to the terminal + the Log pane. Safe to call from ANY thread: the widget
        write is marshalled onto the GUI thread through _log_sig (see its declaration)."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        try:
            self._log_sig.emit(line)
        except RuntimeError:
            pass                      # window already destroyed (shutdown) — terminal line stands

    def _append_log(self, line: str):
        """GUI-thread slot (via _log_sig): append one line to the Log pane and keep it scrolled
        to the bottom. NEVER call this directly from a worker thread — use _log()."""
        w = getattr(self, "_log_text", None)
        if w is None:
            return                    # logged before the UI was built — terminal only
        w.append(line)
        w.verticalScrollBar().setValue(w.verticalScrollBar().maximum())

    # =========================================================================
    # Timers
    # =========================================================================

    def _setup_timers(self):
        # Sim + viewer update — only handles MuJoCo and state machine, not IK
        self._control_timer = QTimer(self)
        self._control_timer.timeout.connect(self._on_tick)
        self._control_timer.start(20)  # 50 Hz is plenty for sim/viewer

        # Slower status refresh for labels and ZED info
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(200)  # 5 Hz

        # 10 s session summary (peak latency while recording)
        self._summary_timer = QTimer(self)
        self._summary_timer.timeout.connect(self._on_summary_tick)
        self._summary_timer.start(10000)  # 0.1 Hz

        # Cross-machine link: drain master commands (peer) + send candidates/status.
        self._link_timer = QTimer(self)
        self._link_timer.timeout.connect(self._on_link_tick)
        self._link_timer.start(100)  # 10 Hz

        # Cursor auto-park (e-stop aid): if the mouse is idle for 15 s, move the cursor over the
        # E-STOP button so it's a single click away. Always active.
        self._cursor_timer = QTimer(self)
        self._cursor_timer.timeout.connect(self._on_cursor_idle_tick)
        self._cursor_timer.start(1000)  # 1 Hz idle check (15 s threshold)

    # =========================================================================
    # System initialization
    # =========================================================================

    def _initialize_system(self):
        # Print the build marker FIRST so every startup log shows which fixes this machine runs.
        # If the peer's log doesn't show this exact date/tags, it's a stale checkout → sync it.
        self._log(f"[build] XRT manager — {_MANAGER_BUILD}")
        self._log("Initializing system...")
        try:
            self._init_core()
        except Exception as exc:
            self._log(f"FATAL init error: {exc}")
            traceback.print_exc()
            return
        self._init_done = True
        # Room transcription (own process; models load there — nothing blocks here). MASTER /
        # standalone only — the peer never auto-starts it (transcription + mic are master-only).
        self._transcribe_on = bool(self._cfg.get("transcribe_enabled", True))
        if self._transcribe_on and self._transcription_allowed():
            QTimer.singleShot(2500, self._init_transcriber)
        self._refresh_transcribe_button()
        if self._args.dry_run:
            # No hardware to boot — go straight to ready.
            self._finish_bootup()
        else:
            # HARDWARE IS NOT TOUCHED YET: the robot boots only when the operator
            # presses 🤖 Bootup Robot (top right). Until then teleop/session controls
            # stay disabled (state INIT = not ready).
            self._btn_bootup.setEnabled(True)
            self._log("System initialized — hardware connected (read-only). Press "
                      "🤖 Bootup Robot (top right) to run the robot bootup sequence.")
        self._update_buttons()

    def _init_core(self):
        from xrt_devices.study import StudyXRDevice as XRRTCBodyPoseDevice
        from projects.inspire_hand_teleop.inspire_hand_haptics_streamer import InspireHandHapticsStreamer
        from projects.inspire_hand_teleop.motor_stats_streamer import MotorStatsStreamer
        from proj_xrt.g1_adapter import G1FullBodySEWSolver
        from projects.inspire_hand_teleop.g1_with_inspire_hand_mujoco_controller import G1InspireHandMuJoCoController
        from projects.inspire_hand_teleop.inspire_hand_hw_controller import InspireHandController
        from projects.g1_upper_body_teleop.config import Config

        args = self._args
        config_path = str(_HERE.parent / "g1_upper_body_teleop" / "g1.yaml")
        config = Config(config_path)

        if not args.dry_run:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(0, args.net)

        if not _XML.exists():
            raise FileNotFoundError(f"XML not found: {_XML}")

        self._log(f"Loading MuJoCo model from {_XML.name}...")
        self._model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self._data  = mujoco.MjData(self._model)
        self._log("Model loaded.")

        # End-effector wrench estimator (own MjModel/MjData; safe off-thread).
        try:
            self._wrench_estimator = we.WrenchEstimator(_XML.as_posix())
        except Exception as exc:
            self._log(f"Wrench estimator unavailable: {exc}")
            self._wrench_estimator = None

        # Finger qpos addresses, for recording the live finger pose into the
        # wrench capture so the playback tool can animate the hands.
        try:
            self._finger_qpos_addrs = we.hand_qpos_addrs(self._model)
        except Exception:
            self._finger_qpos_addrs = None

        # Neck
        try:
            from ostrich_neck.hardware import OstrichNeckHardwareInterface
            from ostrich_neck.controller import DynamixelConnectionError as _DCError
            self._neck_interface = OstrichNeckHardwareInterface()
            self._neck_interface.connect(move_home=True)
            self._log("Ostrich neck connected.")
        except Exception as exc:
            self._log(f"Neck unavailable: {exc}")
            self._neck_interface = None

        # WebRTC device.
        # NOTE: the manager->XR control channels (unity_cmds, haptics, motor_stats) are now
        # CLIENT-CREATED by the Quest (WebRTCController.StartWebRTC), NOT server-created here.
        # aiortc-as-answerer-created data channels do not reliably surface on Unity's
        # libwebrtc-as-offerer (that asymmetry is why manager->XR toggles/haptics/motor_stats
        # never applied), whereas offerer(Quest)-created channels do — the same proven path the
        # working body_pose/unity_state channels use. The manager still SENDS on these via its
        # existing send_to_datachannel sender loops; the server stores client-created channels in
        # the same per-peer channel map, so no send-side change is needed. (--no-haptics /
        # --no-motor-stats still suppress the corresponding sender loops; the unused channel is
        # harmless.) So create NO server channels.
        _server_channels = []

        _dev_init = dict(
            record_data=False,
            output_dir=self._outdir_edit.text() or args.output_dir,
            ndigits=args.ndigits,
            server_data_channels=_server_channels,
        )
        # Kept so the "Reboot WebRTC" button can rebuild the server without a manager restart.
        self._webrtc_dev_init = _dev_init
        self._webrtc_subprocess = bool(getattr(args, "webrtc_subprocess", False))
        self._webrtc_reboot_inflight = False     # guards concurrent reboots (button + watchdog)
        self._webrtc_last_autoreboot = 0.0       # rate-limits the auto-restart watchdog
        if bool(getattr(args, "webrtc_subprocess", False)):
            # Host the aiortc event loop in its OWN process so no amount of manager-side GIL
            # load (60 Hz IK+SEW, retargeting, CSV, Qt) can starve it → no RTT-balloon drops.
            # The proxy exposes the same surface the device does, so nothing else changes.
            from xrt_devices.study_process import WebRTCServerProxy
            self._device = WebRTCServerProxy(_dev_init, log_fn=self._log)
            try:
                self._device.start()
                self._log("WebRTC server started on port 8080 (OUT-OF-PROCESS — isolated "
                          "from manager GIL load).")
            except Exception as exc:
                self._log(f"Out-of-process WebRTC failed to start ({exc}); falling back to "
                          "in-process.")
                self._webrtc_subprocess = False
                self._device = XRRTCBodyPoseDevice(env=None, **_dev_init).start()
                self._log("WebRTC server started on port 8080.")
        else:
            self._device = XRRTCBodyPoseDevice(env=None, **_dev_init).start()
            self._log("WebRTC server started on port 8080.")

        # Solver
        self._solver = G1FullBodySEWSolver(
            urdf_path=str(_URDF),
            hand_type="inspire",
            debug=False,
            timing_enabled=False,
            control_rate_hz=float(args.ik_rate),
            elbow_filter_cutoff_hz=2.0,
            collision_avoidance_enabled=args.safety_filter,
        )

        # MuJoCo controller
        self._controller = G1InspireHandMuJoCoController(self._model, self._data, debug=False)
        if args.kneeling:
            self._controller.q_goal_left_leg  = self._kneeling_angles[:6].copy()
            self._controller.q_goal_right_leg = self._kneeling_angles[6:].copy()

        # MuJoCo viewer — CLOSED by default so its 50 Hz render never steals GIL/CPU from the
        # WebRTC teleop loop during a study. Open on demand with the "Show Live 3D Viewer"
        # button, or pass --show-viewer to launch it at startup.
        if bool(getattr(args, "show_viewer", False)):
            self._open_viewer()

        # Hardware
        if not args.dry_run:
            self._log("Connecting to G1 hardware...")
            from projects.g1_upper_body_teleop.unitree_api_utils import Controller as UnitreeController
            self._hw_controller = UnitreeController(config)
            self._log("G1 hardware connected.")

            self._log("Connecting to Inspire hands...")
            self._hand_controller = InspireHandController(
                hand_side='b', initialize_dds=False, haptics_only=True
            )

            if not args.no_haptics:
                self._haptics_streamer = InspireHandHapticsStreamer(
                    self._hand_controller, self._device, rate_hz=120, debug=False,
                    default_deadband=args.haptic_deadband,
                )
                # Auto-apply a previously-saved fingertip calibration so the measured
                # noise floors gate phantom triggers from the first run (raw readings,
                # no grams).
                self._haptics_streamer.load_force_calibration()
                # Apply the persisted palm-channel gate (Palm Haptics toggle).
                self._haptics_streamer.palm_enabled = bool(
                    self._cfg.get("palm_haptics", True))
            if not args.no_motor_stats:
                self._motor_stats_streamer = MotorStatsStreamer(
                    self._hw_controller, self._hand_controller, self._device, rate_hz=10
                ).start()

            # Sync sim to hardware
            q_torso, q_left, q_right = self._hw_controller.get_current_upper_body()
            self._sync_sim(q_right, q_left, q_torso)
            mujoco.mj_forward(self._model, self._data)

            # Physical bootup (release FSM, hand fist, arm choreography, rise, damp)
            # NO LONGER runs here — the operator triggers it with the 🤖 Bootup Robot
            # button (_on_bootup_robot -> _bootup_robot_hw). Until then the robot is
            # untouched: hardware is connected read-only (state cache, monitors).
        else:
            self._log("Dry-run mode: hardware commands disabled.")

        # The ControlThread (IK + 500 Hz sender) is created in _finish_bootup(),
        # AFTER the operator-triggered hardware bootup — starting it here would
        # immediately stream hold-pose commands to an un-booted robot.

    # ── Operator-triggered hardware bootup ──────────────────────────────────
    # ── Room transcription (conference mic → whisper subprocess) ─────────────
    def _transcription_allowed(self):
        """Transcription + the conference mic are a MASTER-ONLY (or standalone) feature: there
        is ONE shared room mic, transcribed once, on the master. A PEER manager never runs the
        transcriber or touches a mic — so it can't fight the master for the device or spin up a
        second whisper subprocess. Gated on the Link role."""
        return self._link_role() != "peer"

    def _shutdown_transcriber(self):
        """Stop + drop the transcriber subprocess (idempotent). Used when transcription is
        turned off and when the Link role switches to peer."""
        if self._transcriber is not None:
            try:
                self._transcriber.stop_session()
                self._transcriber.stop()
            except Exception:                            # noqa: BLE001
                pass
            self._transcriber = None

    def _init_transcriber(self):
        """Spawn the transcriber subprocess (idempotent). Model loading happens in
        the subprocess; the manager never blocks on it."""
        if self._transcriber is not None or not self._transcribe_on:
            return
        if not self._transcription_allowed():
            self._log("[transcribe] disabled on the PEER — transcription and the mic run on "
                      "the master only (one shared room mic).")
            return
        try:
            model = self._cfg.get("transcribe_model") or "large-v3-turbo"
            proxy = TranscriberProxy(
                {"device": "auto", "model": model,
                 "enroll_path": str(_HERE / "voice_profiles.json"),
                 "min_speaker_sim": float(self._cfg.get("transcribe_min_sim", 0.45)),
                 "min_speaker_margin": float(self._cfg.get("transcribe_min_margin", 0.06)),
                 "stereo": bool(self._cfg.get("transcribe_stereo", False))},
                log_fn=self._log)
            # Stream finished rows to the Live Transcription window (rx thread → GUI via a
            # queued signal so widget updates stay on the GUI thread).
            proxy.on_transcript = lambda row: self._transcript_sig.emit(row)
            self._log(f"[transcribe] using whisper model '{model}'.")
            proxy.start()
            self._transcriber = proxy
            self._log("[transcribe] starting (models load in the subprocess)…")
        except Exception as exc:                     # noqa: BLE001
            self._log(f"[transcribe] failed to start: {exc}")
            self._transcriber = None
        # Surface the live-transcription window (opens once; operator can close/reopen it).
        # Outside the try so a UI hiccup can never null a working transcriber subprocess.
        if self._transcriber is not None:
            try:
                if self._transcript_window is None:
                    self._transcript_window = TranscriptWindow(self)
                self._transcript_window.show()
            except Exception as exc:                 # noqa: BLE001
                self._log(f"[transcribe] transcript window failed to open: {exc}")
        self._refresh_transcribe_button()
        # If a session is ALREADY open (toggled on mid-study), attach to it.
        if self._transcriber is not None and self._session_open():
            QTimer.singleShot(4000, lambda: (
                self._transcriber.start_session(self._session_root, self._session_id)
                if (self._transcriber is not None and self._session_open()) else None))

    def _append_transcript_row(self, row):
        """GUI-thread slot (via _transcript_sig): buffer a finished transcript row and append
        it to the Live Transcription window if it exists (even while hidden, so reopening
        shows the full run). Before the window is first created, rows live in the buffer and
        are replayed by TranscriptWindow.__init__."""
        self._transcript_rows.append(row)
        if self._transcript_window is not None:
            self._transcript_window.append_row(row)

    def _on_open_transcript(self):
        """Open (or raise) the Live Transcription window."""
        if self._transcript_window is None:
            self._transcript_window = TranscriptWindow(self)
        self._transcript_window.show()
        self._transcript_window.raise_()
        self._transcript_window.activateWindow()

    # ── Microphone self-test (record 5 s → play back → delete) ────────────────
    def _mic_test_device(self):
        """The ALSA capture device to test: what a running transcriber reports, else the
        same auto-detected USB mic the transcriber would pick."""
        tr = self._transcriber
        if tr is not None:
            dev = (tr.status() or {}).get("device")
            if dev:
                return dev
        try:
            from study_transcriber import find_mic_device
            return find_mic_device("auto")
        except Exception as exc:                         # noqa: BLE001
            self._log(f"[mic-test] could not detect the mic: {exc}")
            return None

    def _on_test_mic(self):
        """Record and play back the mic so you can hear it, then delete the recording.
        In 2-mic stereo it runs SEQUENTIALLY — Mic 1 (Operator 1 / LEFT) then Mic 2
        (Operator 2 / RIGHT), isolating each channel. Captures THROUGH PipeWire when it owns
        the device (raw hw access would be 'busy'). QProcess keeps the GUI responsive."""
        if not self._transcription_allowed():
            self._log("[mic] the microphone test is master-only — the peer has no room mic.")
            return
        if self._mic_test_proc is not None:             # a test is already running
            return
        detected = self._mic_test_device()
        if not detected:
            QMessageBox.warning(self, "Test Microphone",
                                "No microphone detected. Plug it in and check that "
                                "`arecord -l` lists it.")
            return
        try:
            from study_transcriber import resolve_capture
            self._mic_test_capdev, self._mic_test_capenv = resolve_capture(detected)
        except Exception:
            self._mic_test_capdev, self._mic_test_capenv = detected, None
        stereo = bool(self._cfg.get("transcribe_stereo", False))
        self._mic_test_nch = 2 if stereo else 1
        # Phases: (label, channel-to-isolate). Stereo → Mic 1 (LEFT) then Mic 2 (RIGHT).
        self._mic_test_phases = ([("Mic 1 · Operator 1 (LEFT)", 0),
                                  ("Mic 2 · Operator 2 (RIGHT)", 1)]
                                 if stereo else [("microphone", None)])
        self._mic_test_phase = 0
        self._btn_test_mic.setEnabled(False)
        self._start_mic_phase()

    def _start_mic_phase(self):
        """Record 5 s for the current phase (2 ch in stereo so a channel can be isolated)."""
        label = self._mic_test_phases[self._mic_test_phase][0]
        import tempfile
        try:
            fd, path = tempfile.mkstemp(prefix="mic_test_", suffix=".wav")
            os.close(fd)
        except Exception as exc:                         # noqa: BLE001
            self._log(f"[mic-test] could not create a temp file: {exc}")
            self._finish_mic_test()
            return
        self._mic_test_wav = path
        self._btn_test_mic.setText(f"🎤  Recording {label} (5 s)…")
        self._log(f"[mic-test] 🎤 SPEAK INTO {label} — recording 5 s …")
        p = QProcess(self)
        self._mic_test_proc = p
        if self._mic_test_capenv:                        # PipeWire node targeting
            qenv = QtCore.QProcessEnvironment.systemEnvironment()
            for k, v in self._mic_test_capenv.items():
                qenv.insert(k, v)
            p.setProcessEnvironment(qenv)
        p.finished.connect(self._on_mic_record_done)
        p.start("arecord", ["-D", self._mic_test_capdev, "-q", "-f", "S16_LE", "-r", "44100",
                            "-c", str(self._mic_test_nch), "-d", "5", "-t", "wav", path])

    def _on_mic_record_done(self, code, _status):
        p = self._mic_test_proc
        err = (bytes(p.readAllStandardError()).decode("utf-8", "ignore").strip()
               if p is not None else "")
        self._mic_test_proc = None
        if code != 0:
            if "busy" in err.lower():
                self._log("[mic-test] mic is BUSY — another app holds it. Turn Transcription "
                          "OFF (and close other recorders), then test again.")
            else:
                self._log(f"[mic-test] recording failed: {err or 'arecord error'}")
            self._finish_mic_test()
            return
        label, ch = self._mic_test_phases[self._mic_test_phase]
        play_path = self._mic_test_wav
        if ch is not None:                               # stereo: isolate this mic's channel
            play_path, rms = self._extract_channel(self._mic_test_wav, ch)
            if play_path is None:
                self._log(f"[mic-test] {label}: could not read the recording.")
                self._finish_mic_test()
                return
            tag = ("OK" if rms >= 3e-3 else "weak — move closer / raise gain"
                   if rms >= 5e-4 else "SILENT — this mic isn't sending audio")
            self._log(f"[mic-test] {label}: rms {rms:.4f} ({tag})")
        self._mic_test_playwav = play_path
        self._btn_test_mic.setText(f"🔊  Playing {label}…")
        self._log(f"[mic-test] 🔊 playing back {label} …")
        p2 = QProcess(self)
        self._mic_test_proc = p2
        p2.finished.connect(self._on_mic_play_done)
        p2.start("aplay", ["-q", self._mic_test_playwav])

    def _on_mic_play_done(self, code, _status):
        p = self._mic_test_proc
        err = (bytes(p.readAllStandardError()).decode("utf-8", "ignore").strip()
               if p is not None else "")
        self._mic_test_proc = None
        if code != 0:
            self._log(f"[mic-test] playback issue: {err or 'aplay error'} — is an output "
                      "device (speakers/headphones) connected?")
        self._cleanup_mic_phase_files()
        self._mic_test_phase += 1
        if self._mic_test_phase < len(self._mic_test_phases):
            self._start_mic_phase()                      # on to the next mic
        else:
            self._log("[mic-test] done — recordings deleted.")
            self._finish_mic_test()

    def _cleanup_mic_phase_files(self):
        for attr in ("_mic_test_wav", "_mic_test_playwav"):
            f = getattr(self, attr, None)
            if f:
                try:
                    os.remove(f)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _finish_mic_test(self):
        """Delete any temp recordings and restore the button/state."""
        self._cleanup_mic_phase_files()
        self._mic_test_phases = []
        self._mic_test_phase = 0
        self._btn_test_mic.setEnabled(True)
        self._btn_test_mic.setText("🎧  Test Microphone (5 s)")

    @staticmethod
    def _extract_channel(wav_path, ch):
        """Isolate channel ``ch`` of a multi-channel WAV into a NEW mono WAV. Returns
        (mono_wav_path, rms), or (None, 0.0) on error."""
        import wave
        import tempfile
        try:
            with wave.open(wav_path, "rb") as w:
                nch, sw, fr = w.getnchannels(), w.getsampwidth(), w.getframerate()
                a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            if nch > 1:
                a = a.reshape(-1, nch)[:, min(ch, nch - 1)]
            mono = np.ascontiguousarray(a)
            rms = (float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2)))
                   if mono.size else 0.0)
            fd, out = tempfile.mkstemp(prefix="mic_test_ch_", suffix=".wav")
            os.close(fd)
            with wave.open(out, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(sw)
                w.setframerate(fr)
                w.writeframes(mono.tobytes())
            return out, rms
        except Exception:
            return None, 0.0

    def _on_toggle_transcribe(self):
        # Turning transcription OFF is confirmation-guarded (an accidental click would
        # silently stop capturing speech for the session).
        if self._transcribe_on:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Turn off transcription?")
            box.setText("Stop live transcription?\n\nNo more speech will be logged to the "
                        "session transcript until you turn it back on. (No audio is ever "
                        "saved either way.)")
            off = box.addButton("Turn OFF", QMessageBox.AcceptRole)
            box.addButton("Keep ON", QMessageBox.RejectRole)
            box.setDefaultButton(box.buttons()[-1])
            box.exec_()
            if box.clickedButton() is not off:
                return
        self._transcribe_on = not bool(self._transcribe_on)
        self._cfg["transcribe_enabled"] = self._transcribe_on
        self._save_cfg()
        if self._transcribe_on:
            self._init_transcriber()
        else:
            if self._transcriber is not None:
                try:
                    self._transcriber.stop_session()
                    self._transcriber.stop()
                except Exception:
                    pass
                self._transcriber = None
            self._log("[transcribe] OFF.")
        self._refresh_transcribe_button()

    def _refresh_transcribe_button(self):
        on = bool(self._transcribe_on)
        if hasattr(self, "_btn_transcribe"):
            self._btn_transcribe.setText(f"🎙  Transcription: {'ON' if on else 'OFF'}")
            _lbl_style(self._btn_transcribe,
                       "QPushButton { background: %s; color: white; padding: 6px; }"
                       % ("#27ae60" if on else "#7f8c8d"))
        if hasattr(self, "_btn_calib_voices_tr"):
            self._btn_calib_voices_tr.setEnabled(on and self._transcriber is not None)

    def _refresh_mic_mode_ui(self):
        """Set the Mic Mode button label and show/hide the mode-specific rows: Calibrate
        Voices only in 1-mic (voice-ID) mode; the L/R operator hint only in 2-mic stereo."""
        stereo = bool(self._cfg.get("transcribe_stereo", False))
        allowed = self._transcription_allowed()      # hidden entirely on the peer
        if hasattr(self, "_btn_mic_mode"):
            self._btn_mic_mode.setText("🎚  Mic Mode: 2 mics (stereo L/R)" if stereo
                                       else "🎚  Mic Mode: 1 mic (voice-ID)")
        if hasattr(self, "_btn_calib_voices_tr"):
            self._btn_calib_voices_tr.setVisible(allowed and not stereo)
        if hasattr(self, "_stereo_hint_lbl"):
            self._stereo_hint_lbl.setVisible(allowed and stereo)

    def _on_toggle_mic_mode(self):
        """Flip 1-mic ⇄ 2-mic stereo. Persists to config, re-lays out the mode-specific
        buttons, and — if transcription is running — respawns the subprocess in the new mode
        (2 mics needs a 2-channel capture, so it can't switch in place)."""
        stereo = not bool(self._cfg.get("transcribe_stereo", False))
        self._cfg["transcribe_stereo"] = stereo
        self._save_cfg()
        self._log("[transcribe] mic mode → "
                  + ("2 mics (stereo: L=Operator 1, R=Operator 2)" if stereo
                     else "1 mic (voice-ID)") + ".")
        self._refresh_mic_mode_ui()
        # Re-render an open checklist so the voice-calibration step shows/hides for the mode.
        if getattr(self, "_checklist_window", None) is not None:
            try:
                self._checklist_window._rebuild()
            except Exception:
                pass
        # Apply live: respawn the transcriber if it's running (mode is fixed at spawn time).
        if self._transcriber is not None:
            try:
                self._transcriber.stop_session()
                self._transcriber.stop()
            except Exception:
                pass
            self._transcriber = None
            self._init_transcriber()

    def _on_calibrate_voices(self):
        """Voice enrollment: each speaker reads the script for as long as they like — you
        press Start, then Stop when they finish (no fixed timer). Embeddings (never audio)
        are stored so transcript rows carry speaker labels. Re-run per group."""
        if not self._transcription_allowed():
            self._log("[transcribe] voice calibration is master-only (the peer has no mic).")
            return
        if self._transcriber is None:
            self._log("[transcribe] transcription is off — enable it first.")
            return
        # Only the two operators are enrolled. The research coordinator stays
        # UNENROLLED on purpose: they rarely speak, and their remarks simply come out
        # as speaker="unknown" (below the similarity threshold) — no calibration cost.
        roles = [("operator1", "Operator 1"),
                 ("operator2", "Operator 2")]
        self._log("[transcribe] ===== VOICE CALIBRATION (2 operators) =====")
        # Fresh slate each group: stale profiles from a previous group would
        # mis-attribute this group's speech.
        self._transcriber.enroll_clear()
        # Suppress the E-STOP cursor auto-park for the whole calibration (see the flag's
        # comment): otherwise the mouse goes idle while the operator reads the script, the
        # auto-park raises the main window OVER the modal and steals focus, and the next
        # operator's dialog opens behind it — which looked like it "closed out".
        self._calibrating_voices = True
        try:
            for key, label in roles:
                try:
                    self._calibrate_one_voice(key, label)
                except Exception as exc:             # one operator must never abort the other
                    self._log(f"[transcribe] {label}: calibration error: {exc}")
        finally:
            self._calibrating_voices = False
        self._log("[transcribe] ===== voice calibration done =====")

    def _calibrate_one_voice(self, key, label):
        """Prompt ONE operator, record OPEN-ENDED, and enroll on Stop. Split out of
        _on_calibrate_voices so a Skip / error for one operator returns cleanly without
        skipping the next one."""
        n = label[-1]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(f"Calibrate voice — {label}")
        box.setText(f"{label}: stand near the conference mic, click Start, and "
                    "read the CALIBRATION SCRIPT aloud at your normal task voice. "
                    "Others stay quiet. Press Stop when they finish — there is no "
                    "time limit.\n\nSCRIPT:\n"
                    f"“I am operator {n}. Ok, you hold the crate steady from the "
                    "bottom while I lift this side. Loosen your grip on the corner "
                    "so I can pivot. Rotate it on the count of three. Bring the box "
                    "closer to you. I have a firm grasp of the crate now. Now that "
                    "we've flipped it, lets set it down.”\n\nOnly a voice "
                    "fingerprint (embedding) is stored — never the audio.")
        rec = box.addButton(f"● Start {label}", QMessageBox.AcceptRole)
        box.addButton("Skip", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is not rec:
            self._log(f"[transcribe] {label}: skipped.")
            return
        if not self._transcriber.enroll_start(key):
            QMessageBox.warning(self, "Enrollment failed",
                                f"{label}: could not start recording.")
            return
        self._log(f"[transcribe] recording {label} — reading now; press Stop when "
                  "they finish…")
        # Modal Stop: capture runs in the transcriber subprocess while this is up.
        rbox = QMessageBox(self)
        rbox.setIcon(QMessageBox.Information)
        rbox.setWindowTitle(f"Recording {label}…")
        rbox.setText(f"● Recording {label}.\n\nLet them read the whole script, then "
                     "click Stop Recording.")
        rbox.addButton("■  Stop Recording", QMessageBox.AcceptRole)
        rbox.exec_()                       # the only button → returns on Stop
        ok, msg = self._transcriber.enroll_stop()
        self._log(f"[transcribe] {label}: {msg}")
        if not ok:
            QMessageBox.warning(self, "Enrollment failed",
                                f"{label}: {msg}\n\nTry again from Calibrate Voices.")

    def _on_bootup_button(self):
        """The single Bootup/Reboot button: first boot when the robot has never
        booted; ♻ Reboot when it is down after a Safe Shutdown / E-STOP."""
        if self._hw_booted and getattr(self, "_exit_damping_done", False):
            self._on_reboot_robot()
        else:
            self._on_bootup_robot()

    def _on_bootup_robot(self):
        """🤖 Bootup Robot (confirmation-guarded): run the PHYSICAL bootup sequence —
        release the high-level FSM, close the hands into the fist, spread + raise the
        arms clear of the table, rise into the kneel, drop the arms into damping — then
        start the teleop control loop. Runs once; the manager launches with the robot
        untouched until this is pressed."""
        if not self._init_done or self._hw_booted:
            return
        if self._hw_controller is None:
            self._log("[bootup] no hardware connected (dry-run?) — nothing to boot.")
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Bootup Robot")
        box.setText("Boot up the ROBOT hardware?\n\nThe robot WILL MOVE: hands close "
                    "into a fist, the arms spread out + lift behind it, and it rises "
                    "into the kneeling pose (~10 s), then the arms drop limp.\n\nMake "
                    "sure the area around the robot and the table in front of it are "
                    "clear.")
        go = box.addButton("Boot up", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])   # default = Cancel
        box.exec_()
        if box.clickedButton() is not go:
            self._log("[bootup] cancelled.")
            return
        self._btn_bootup.setEnabled(False)
        self._btn_bootup.setText("🤖  Booting…")
        QApplication.processEvents()
        # EVERYTHING below is contained: an exception escaping this Qt slot would make
        # PyQt5 abort() the whole app — killing the 500 Hz command stream and dropping
        # the risen robot (firmware watchdog → backward fall on the kneel rig).
        try:
            self._bootup_robot_hw()
        except Exception as exc:
            self._bootup_failed_standby("bootup choreography", exc)
            return
        try:
            self._finish_bootup()
        except Exception as exc:
            self._bootup_failed_standby("post-bootup start", exc)
            return

    def _bootup_failed_standby(self, where, exc):
        """Bootup failure containment: put the ROBOT into the gentlest available
        standby and keep the APP alive. If the control thread came up, it already
        streams the kneel hold (true standby). Otherwise stream a bounded gentle damp
        so the robot settles onto the rig instead of free-falling when the firmware
        watchdog trips."""
        self._log(f"[bootup] FAILED during {where}: {exc}")
        traceback.print_exc()
        ct = self._control_thread
        if ct is not None and ct.is_alive():   # ControlThread subclasses threading.Thread
            self._log("[bootup] control loop IS running — robot is holding standby. "
                      "Fix the error and restart the manager when convenient.")
            return
        try:
            if self._hw_controller is not None:
                self._log("[bootup] no control loop — easing the robot down with a "
                          "gentle damp (it will settle onto the rig).")
                self._hw_controller.enter_damping(duration_s=2.0)
        except Exception as exc2:                    # noqa: BLE001
            self._log(f"[bootup] damping also failed: {exc2}")
        self._btn_bootup.setText("🤖  Bootup Robot (retry)")
        self._btn_bootup.setEnabled(True)

    def _on_reboot_robot(self):
        """♻ Reboot (confirmation-guarded): after Safe Shutdown / E-STOP,
        re-run the physical bootup and restart the control loop — WITHOUT restarting
        the manager, so session/calibration/anchor/link state survives and the study
        can resume mid-participant. Pose + control reset only (firmware faults need a
        power cycle)."""
        if not self._init_done or self._hw_controller is None:
            return
        if not (self._hw_booted and getattr(self, "_exit_damping_done", False)):
            self._log("[reboot] robot isn't in a shut-down state — nothing to re-boot.")
            return
        if self._recording_active:
            self._log("[reboot] a take is RECORDING — Stop Attempt (or Redo) first, "
                      "then re-boot; the session stays open through the re-boot.")
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Reboot Robot")
        box.setText("Boot the robot up AGAIN?\n\nThe robot WILL MOVE: from its current "
                    "slumped pose it closes the hands, spreads + raises the arms, and "
                    "rises back into the kneel (~10 s).\n\nEverything else is "
                    "PRESERVED — open session, camera anchors, calibration, peer link — "
                    "so the study resumes where it left off.\n\nNOTE: this resets pose "
                    "+ control only. For robot firmware faults, power-cycle the G1 "
                    "instead. Clear the area around the robot.")
        go = box.addButton("Reboot", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])   # default = Cancel
        box.exec_()
        if box.clickedButton() is not go:
            self._log("[reboot] cancelled.")
            return
        self._btn_bootup.setEnabled(False)
        self._btn_bootup.setText("♻  Booting…")
        QApplication.processEvents()
        self._prepare_reboot()
        # Same contained sequence as the first boot (PyQt5 aborts on slot exceptions).
        try:
            self._bootup_robot_hw()
        except Exception as exc:
            self._bootup_failed_standby("re-boot choreography", exc)
            return
        try:
            self._finish_bootup()
        except Exception as exc:
            self._bootup_failed_standby("post-re-boot start", exc)
            return
        self._log("[reboot] robot is back up — resume the study (Start Teleop, then "
                  "continue the session where it left off).")

    def _prepare_reboot(self):
        """Retire the stale per-boot objects so the bootup path can run again
        cleanly: the dead control thread (+ its idle neck worker), the old wrench
        worker (a fresh one is created during bootup), and the one-shot flags. The
        session/calibration/link state is deliberately untouched."""
        if self._wrench_worker is not None:
            try:
                self._wrench_worker.stop()
            except Exception:
                pass
            self._wrench_worker = None
        ct = self._control_thread
        if ct is not None:
            try:
                if ct.is_alive():
                    ct.stop()
            except Exception:
                pass
            try:
                if ct._neck_worker is not None:
                    ct._neck_worker.stop()      # idle daemon; frees the serial queue
            except Exception:
                pass
            self._control_thread = None
        # Re-arm the one-shot bootup machinery: with _hw_booted False, the whole
        # first-boot path (guards + failure handler + retry button) applies as-is.
        self._hw_booted = False

    def _bootup_robot_hw(self):
        """The physical bootup choreography (moved verbatim from _init_core; blocking,
        ~10 s, animates the sim viewer in lock-step)."""
        args = self._args
        # Release high-level FSM if kneeling
        if args.kneeling:
            self._log("Releasing high-level motion mode...")
            from projects.inspire_hand_teleop.demo_g1_inspire_xr_robot_teleop_full_body_hw import release_motion_mode
            release_motion_mode()
            time.sleep(0.5)

        # Hand gesture before any arm motion: first a THUMBS-UP, then curl the
        # thumb in to complete a full FIST. DOF order is [pinky, ring, middle,
        # index, thumb-bend, thumb-rotate] with range_list = fully closed and
        # 0 = open. The worker streams the goal at 50 Hz once engaged; we pause
        # after each step so the gesture is fully formed BEFORE the arms move.
        # The fist holds through the startup choreography; teleop takes the
        # hands over when it starts.
        if self._hand_controller is not None:
            # 1) Thumbs-up: four fingers closed, thumb extended (bend+rotate open).
            self._log("Closing hands into a thumbs-up...")
            _thumbs_up = list(self._hand_controller.range_list)
            _thumbs_up[4] = 0.0   # thumb bend   -> extended (not curled into the fist)
            _thumbs_up[5] = 0.0   # thumb rotate -> out (not opposed across the palm)
            self._hand_controller.set_joint_goals(
                {"q_goal_left_hand": _thumbs_up, "q_goal_right_hand": _thumbs_up})
            self._hand_controller.apply_control(engaged=True)
            time.sleep(1.0)
            # 2) Bend the thumb down, but keep it rotated OUT to the side (not
            #    opposed across the palm) — i.e. the thumbs-up with only the
            #    thumb bend closed. So everything closes EXCEPT thumb rotate.
            self._log("Bending thumb to the side...")
            _fist = list(self._hand_controller.range_list)  # fingers + thumb bend closed
            _fist[5] = 0.0   # thumb rotate -> stays out to the side (as in the thumbs-up)
            self._hand_controller.set_joint_goals(
                {"q_goal_left_hand": _fist, "q_goal_right_hand": _fist})
            self._hand_controller.apply_control(engaged=True)
            time.sleep(0.8)

        # Manager bootup arm choreography. The robot starts from a bent-over /
        # collapsed pose, so the arms are spread out and lifted behind the robot
        # BEFORE it rises — otherwise the hands sweep into the table in front while
        # the body straightens. Sequence (body held low for A+B, then rises in C):
        #   A. ABDUCT OUT — spread the arms out to the sides (45 deg abduction).
        #   B. RAISE BACK — swing those out-held arms back at the shoulders, so the
        #      hands lift up and behind, clear of the front.
        #   C. RISE       — straighten to the kneel/upright pose with the arms HELD
        #      out-and-back, so nothing sweeps into the table.
        # Then the arms drop into damp (below). (The full lateral-raise -> back ->
        # chicken-wing -> forward choreography runs only at TELEOP ENGAGE.)
        _ramp_dt = 0.01
        _kneel = self._kneeling_angles if args.kneeling else None
        _back = np.deg2rad(float(getattr(args, "bootup_arm_back_deg", 20.0)))
        # Abduct-out pose (down, out 45 deg) and the final "A pose" (same abduction,
        # shoulders pitched back) the bootup holds before damping. Safe Shutdown
        # reuses _a_pose_arms() so it settles the arms into this exact pose too.
        _abd_L, _abd_R = _ARMS_BOOTUP_LEFT, _ARMS_BOOTUP_RIGHT
        _back_L, _back_R = self._a_pose_arms()

        # Animate the sim in lock step with these moves. Without it the viewer
        # freezes on the pose synced from hardware (the robot's current pose).
        def _mtp_sync(q_torso, q_left, q_right):
            self._sync_sim(q_right, q_left, q_torso)
            mujoco.mj_forward(self._model, self._data)
            if self._viewer is not None:            # viewer is optional / closed by default
                self._viewer.sync()

        # Use the SAME (boosted) hip/knee gains the ongoing sends use, so the legs
        # reach and hold their target firmly during the move instead of sagging
        # under the weaker default gain and then snapping into pose.
        _lks, _lds = args.leg_kp_scale, args.leg_kd_scale

        # Current (bent) pose to HOLD the body at during A and B (only the arms move).
        _cur_t, _cur_l, _cur_r = self._hw_controller.get_current_upper_body()
        _cur_legs = np.array(
            [self._hw_controller.low_state.motor_state[i].q
             for i in self._hw_controller.config.leg_joint2motor_idx], dtype=np.float64)

        # A) Abduct the arms out to the sides (body held low).
        self._log("Bootup: abducting arms outward (45 deg)...")
        self._hw_controller.move_to_pose(
            q_goal_legs=_cur_legs, total_time=1.5,
            q_goal_upper=np.concatenate((_cur_t, _abd_L, _abd_R)),
            on_step=_mtp_sync, leg_kp_scale=_lks, leg_kd_scale=_lds)

        # B) Raise those out-held arms back at the shoulders (body still held low),
        #    so the hands lift clear of the table before rising.
        self._log(f"Bootup: raising arms back {np.rad2deg(_back):.0f} deg (clear of the table)...")
        self._hw_controller.move_to_pose(
            q_goal_legs=_cur_legs, total_time=1.5,
            q_goal_upper=np.concatenate((_cur_t, _back_L, _back_R)),
            on_step=_mtp_sync, leg_kp_scale=_lks, leg_kd_scale=_lds)

        # C) Rise to the kneel/upright pose with the arms HELD out-and-back.
        self._log("Bootup: rising with arms held out-and-back (no table collision)...")
        self._hw_controller.move_to_pose(
            q_goal_legs=_kneel, total_time=3.5,
            q_goal_upper=np.concatenate((np.zeros(3), _back_L, _back_R)),
            on_step=_mtp_sync, leg_kp_scale=_lks, leg_kd_scale=_lds)

        # Ease the arms from the rigid 45-deg-raised hold into damping so they
        # drop gently. The control loop idles the arms limp (kp=0, kd=arm_damp_kd);
        # switching in one step would drop the holding torque and jerk them down.
        # Blend arm stiffness 0->full-damping (smoothstep). Ends at blend=1,
        # matching the loop's idle damping. The waist eases to its kneel trim.
        self._log("Bootup: dropping arms into damp mode...")
        _damp_dur = 1.0
        _damp_steps = int(_damp_dur / _ramp_dt)
        for _i in range(_damp_steps + 1):
            _frac = _i / _damp_steps
            _beta = _frac * _frac * (3.0 - 2.0 * _frac)
            self._hw_controller.send_upper_body_targets(
                q_goal_torso=self._kneeling_torso,
                q_goal_left=_back_L, q_goal_right=_back_R,
                q_goal_legs=_kneel,
                kp_scale=args.hw_kp_scale, kd_scale=args.hw_kd_scale,
                leg_kp_scale=args.leg_kp_scale, leg_kd_scale=args.leg_kd_scale,
                ankle_kp_scale=0.0, ankle_kd_scale=1.0,
                arm_damping=True, arm_damp_kd=args.arm_damp_kd,
                arm_damp_blend=_beta,
            )
            self._sync_sim(_back_R, _back_L, self._kneeling_torso)
            mujoco.mj_forward(self._model, self._data)
            if self._viewer is not None:            # viewer is optional / closed by default
                self._viewer.sync()
            time.sleep(_ramp_dt)

        # Give the unified raw recorder its hardware + hand handles.
        self._session_recorder._hw   = self._hw_controller
        self._session_recorder._hand = self._hand_controller

        # Wrench live readout worker (disabled until the operator enables it).
        if self._wrench_estimator is not None:
            self._wrench_worker = WrenchLiveWorker(
                self._hw_controller, self._wrench_estimator,
                rate_hz=args.wrench_live_rate,
            )

        # Pop up the motor thermal/torque monitor so the operator can
        # watch for overheating joints during the study.
        self._motor_monitor.update_from_controller(
            self._hw_controller, self._hand_controller,
            forces=self._latest_hand_forces()
        )
        self._motor_monitor.show()

    def _finish_bootup(self):
        """Create + start the teleop control loop and mark the system ready. Runs after
        the operator-triggered hardware bootup (or immediately in dry-run)."""
        args = self._args
        # Single teleop loop (mirrors the demo): runs the IK solve gated at
        # ik_rate Hz AND streams rt/lowcmd to the G1 every iteration at max_fr Hz.
        # It begins holding the rest pose immediately (watchdog-safe) and starts
        # following the operator once start_teleop() is called.
        self._control_thread = ControlThread(
            device=self._device,
            solver=self._solver,
            controller=self._controller,
            hand_controller=self._hand_controller,
            neck_interface=self._neck_interface,
            hw_controller=self._hw_controller,
            haptics_streamer=self._haptics_streamer,
            args=args,
            latency_tracker=self._latency_tracker,
        )
        # Push the loaded kneeling-pose trim BEFORE starting the loop so its very
        # first send already holds the calibrated leg/waist angles. Setting it
        # after start() would briefly send the nominal pose (torso=0, untrimmed
        # legs) the ramp just moved away from — a small startup jerk.
        if args.kneeling:
            self._control_thread.set_kneeling_pose(
                self._kneeling_angles, self._kneeling_torso)
        # Apply the saved arm-abduction limit so the shoulder heat sinks are
        # protected from the very first teleop command.
        self._control_thread.set_abduction_limits(
            self._abduction_limit.left_max, self._abduction_limit.right_max)
        self._control_thread.start()
        # Sync the GUI eco toggle to the freshly-created control thread (in case
        # it was flipped before the thread existed).
        self._control_thread.set_eco_enabled(self._eco_on)
        self._control_thread.set_compliance_enabled(self._comp_on)
        # Apply any saved per-hand disabled DOFs (e.g. a sheared thumb-rotation shaft) so they're
        # never commanded — the recording + digital twin then match the real hand.
        self._apply_hand_dofs_to_control_thread()
        # Sync neck head-tracking to the current streaming-feed state, so if the camera feed
        # is already up when the robot connects, the neck starts tracking the head right away.
        self._control_thread.set_neck_active(self._viewport_on)
        td = args.transition_duration
        td_note = "instant" if td == 0 else f"{td:.1f}s ramp"
        self._log(f"Teleop loop started (IK {args.ik_rate} Hz, send 500 Hz, "
                  f"transition={td_note}). Arms idle in damp mode until Start Teleop."
                  + (f" Eco mode ON." if self._eco_on else ""))
        self._hw_booted = True
        self._exit_damping_done = False    # from here on, Quit must damp on exit
        # Clear the comms/power-down watchdog state so a fresh boot starts clean (a re-boot
        # after a power-cycle re-arms it).
        self._robot_comms_lost = False
        self._robot_powered_down = False
        self._comms_lost_since = None
        # If the hands were held off by a power-cycle, reconnect them now (fresh Modbus +
        # workers, which also clears the hold-off) so they come back live with the robot. Runs
        # off the boot thread; no-op on a first boot (freshly-created hands aren't suppressed).
        if (self._hand_controller is not None
                and getattr(self._hand_controller, "output_suppressed", lambda: False)()):
            self._log("[hands] reconnecting the held-off hands after the robot power-cycle…")
            threading.Thread(target=self._reboot_hands_bg, daemon=True).start()
            QTimer.singleShot(2500, self._log_hands_reboot_result)
        if hasattr(self, "_btn_bootup"):
            self._btn_bootup.setText("✅  Robot Booted")
            self._btn_bootup.setEnabled(False)
        # Restore the robot-down controls (no-ops on the very first boot).
        if hasattr(self, "_btn_safe_shutdown"):
            self._btn_safe_shutdown.setText("🛡  Safe Shutdown (robot)")
            self._btn_safe_shutdown.setEnabled(True)
        if hasattr(self, "_ESTOP_LABEL_DEFAULT"):
            self._ESTOP_LABEL = self._ESTOP_LABEL_DEFAULT
            self._btn_estop.setText(self._ESTOP_LABEL)
        self._state = _S_IDLE
        self._log("System ready. Press Start Teleop to begin.")
        self._update_buttons()

    # =========================================================================
    # Control tick — sim/viewer only, no IK (that lives in ControlThread)
    # =========================================================================

    # ── Live MuJoCo viewer (optional; closed by default to spare the WebRTC loop) ──
    def _open_viewer(self):
        """Launch the in-process live MuJoCo viewer on demand. Must run on the Qt main
        thread (GLFW). Idempotent — a no-op if it's already open."""
        if self._viewer is not None:
            try:
                if self._viewer.is_running():
                    return
            except Exception:
                pass
            self._close_viewer()
        try:
            self._viewer_ctx = mujoco.viewer.launch_passive(
                model=self._model, data=self._data,
                show_left_ui=False, show_right_ui=True,
            )
            self._viewer = self._viewer_ctx.__enter__()
            self._viewer.cam.distance = 2.0
            self._viewer.cam.azimuth  = 135
            self._viewer.cam.elevation = -10
            self._viewer.cam.lookat[:] = [0, 0, 0.8]
            self._log("Live 3D viewer opened.")
        except Exception as exc:                     # noqa: BLE001
            self._viewer = None
            self._viewer_ctx = None
            self._log(f"Live 3D viewer failed to open: {exc}")

    def _close_viewer(self):
        """Tear down the live MuJoCo viewer (safe if already closed). Leaves the manager
        running headless — the sim/recording keep working without it."""
        ctx = self._viewer_ctx
        self._viewer = None
        self._viewer_ctx = None
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass

    def _on_toggle_live_viewer(self):
        """Show/hide the live MuJoCo viewer from the manager button."""
        if self._viewer is not None:
            self._close_viewer()
            self._log("Live 3D viewer closed.")
        else:
            self._open_viewer()
        self._update_buttons()

    def _on_tick(self):
        if self._state == _S_INIT:
            return
        # Robot-comms watchdog (bumped ethernet / lost DDS link) — runs every tick,
        # independent of the viewer, so a lost robot is caught even headless.
        self._check_robot_comms()
        # WebRTC subprocess watchdog — auto-restart it if it died (e.g. an aiortc crash).
        self._check_webrtc_alive()
        # The viewer is now OPTIONAL (closed by default to spare the WebRTC loop its 50 Hz
        # render). If the user closed the window, drop the handle and keep running HEADLESS —
        # closing it no longer quits the manager. Everything below (sim sync, recording
        # capture, control apply) runs regardless; only render/visualization is gated on it.
        if self._viewer is not None:
            try:
                running = self._viewer.is_running()
            except Exception:
                running = False
            if not running:
                self._close_viewer()
                self._log("Live 3D viewer window closed — running headless "
                          "(reopen with 'Show Live 3D Viewer').")
        have_viewer = self._viewer is not None

        # Sync sim from hardware feedback (read-only from hardware)
        if self._hw_controller is not None:
            try:
                q_t, q_l, q_r = self._hw_controller.get_current_upper_body()
                self._sync_sim(q_r, q_l, q_t)
            except Exception:
                pass

        # Unified raw recording — capture EVERYTHING in ANY state (the known-mass
        # test holds the arm still, which may be while idle, not teleoperating).
        # Reads the hand snapshot (hand registers + tactile) and the live finger
        # qpos from the sim (set each tick by the controller's kinematic
        # apply_control) so playback animates the hands. Runs before the
        # state-specific apply_control, so finger qpos lags one tick (~20 ms).
        if self._recording_active and self._session_recorder._active:
            fl = fr = None
            if self._finger_qpos_addrs is not None and self._data is not None:
                try:
                    fl = self._data.qpos[self._finger_qpos_addrs["finger_left"]].copy()
                    fr = self._data.qpos[self._finger_qpos_addrs["finger_right"]].copy()
                except Exception:
                    fl = fr = None
            snap = None
            if self._hand_controller is not None:
                try:
                    snap = self._hand_controller.read()
                except Exception:
                    snap = None
            # Leg goal "down to the knees" (8: hip+knee per leg) from the
            # calibrated kneeling angles, or None when not kneeling.
            q_goal_legs = (self._kneeling_angles[_LEG_KNEE_SEL]
                           if self._args.kneeling else None)
            self._session_recorder.capture(
                q_goal_torso=self._q_goal_torso,
                q_goal_left=self._q_goal_left,
                q_goal_right=self._q_goal_right,
                q_goal_legs=q_goal_legs,
                finger_left=fl, finger_right=fr, snapshot=snap,
            )

        if self._state in (_S_IDLE, _S_WAITING):
            if hasattr(self._controller, 'q_goal_left'):
                self._controller.q_goal_left  = _ARMS_AT_SIDE_LEFT.copy()
                self._controller.q_goal_right = _ARMS_AT_SIDE_RIGHT.copy()
            self._controller.apply_control(engaged=True, kinematic_mode=True)
            mujoco.mj_forward(self._model, self._data)
            if have_viewer:
                self._viewer.sync()
            return

        # ── TELEOP: apply goals from ControlThread to sim (visualization) ────
        if self._control_thread is not None:
            q_g_t, q_g_l, q_g_r = self._control_thread.get_goals()
            # Write to controller so apply_control() uses the latest IK output
            self._controller.q_goal_torso = q_g_t
            self._controller.q_goal_left  = q_g_l
            self._controller.q_goal_right = q_g_r
            # Cache for recording
            self._q_goal_torso = q_g_t
            self._q_goal_left  = q_g_l
            self._q_goal_right = q_g_r

        # Apply control to sim (writes data.ctrl, no hardware I/O)
        self._controller.apply_control(engaged=True, kinematic_mode=not self._args.dynamics)

        # Visualization capsules (30 Hz gated) — only when the window is shown.
        if have_viewer:
            now = time.time()
            if now - self._last_viz_time >= 1.0 / 30:
                self._tick_viz()
                self._last_viz_time = now

        if self._args.dynamics:
            mujoco.mj_step(self._model, self._data)
        else:
            mujoco.mj_forward(self._model, self._data)
        if have_viewer:
            self._viewer.sync()

    def _tick_viz(self):
        try:
            self._viewer.user_scn.ngeom = 0
            from geo_kin_core.viz import draw_filtered_sew
            draw_filtered_sew(
                self._viewer,
                self._solver,
                to_world=self._controller.get_sew_transform(),
            )
        except Exception:
            pass

    # =========================================================================
    # System-status panel (both robots + all cameras; shared over the link)
    # =========================================================================
    _STATE_COLORS = {"green": "#2ecc71", "blue": "#3498db", "red": "#e74c3c"}
    _STATE_UNKNOWN = "#7f8c8d"   # grey: no data (e.g. peer not linked yet)

    def _build_status_panel(self):
        """A row of status lights for BOTH robots (robot + neck + head cam + ZED) plus the
        2 external crate cameras. green=online, blue=recording/active, red=offline. Filled
        from THIS machine's state + the peer's (over the link), so it's the same on both."""
        box = QGroupBox("System Status")
        row = QHBoxLayout(box)
        row.setContentsMargins(8, 2, 8, 2)
        self._lights = {}
        self._rec_dots = {}            # per-robot flashing REC indicator (id -> QLabel)
        comps = [("robot", "Robot"), ("neck", "Neck"),
                 ("headcam", "Head cam (D435)"), ("zed", "ZED"),
                 ("zedcal", "ZED calib")]

        def _dot():
            d = QLabel("●")
            d.setStyleSheet(f"color: {self._STATE_UNKNOWN}; font-size: 15px;")
            return d

        for rid in (1, 2):
            g = QGroupBox(f"Robot {rid}" + (" (Master)" if rid == 1 else " (Peer)"))
            grid = QGridLayout(g)
            grid.setContentsMargins(6, 2, 6, 2)
            grid.setVerticalSpacing(1)
            # Flashing red REC indicator across the top — shows at a glance which robots
            # are actually recording (flashes only while recording; grey ○ idle otherwise).
            rec = QLabel("○  idle")
            rec.setStyleSheet("color: #7f8c8d; font-size: 12px; font-weight: bold;")
            self._rec_dots[rid] = rec
            grid.addWidget(rec, 0, 0, 1, 2)
            for r, (key, label) in enumerate(comps):
                dot = _dot()
                self._lights[(rid, key)] = dot
                grid.addWidget(dot, r + 1, 0)
                lb = QLabel(label)
                lb.setStyleSheet("font-size: 11px;")
                grid.addWidget(lb, r + 1, 1)
            row.addWidget(g)

        cg = QGroupBox("Crate cams (2× external)")
        cgrid = QGridLayout(cg)
        cgrid.setContentsMargins(6, 2, 6, 2)
        cgrid.setVerticalSpacing(1)
        for i in range(2):
            dot = _dot()
            self._lights[("crate", i)] = dot
            cgrid.addWidget(dot, i, 0)
            lb = QLabel(f"Cam {chr(65 + i)}")
            lb.setStyleSheet("font-size: 11px;")
            cgrid.addWidget(lb, i, 1)
        row.addWidget(cg)

        # Legend STACKED vertically — side-by-side entries made this the widest
        # element of the whole System Status column.
        legend = QLabel(
            f'<span style="color:{self._STATE_COLORS["green"]}">●</span> online<br>'
            f'<span style="color:{self._STATE_COLORS["blue"]}">●</span> recording/active<br>'
            f'<span style="color:{self._STATE_COLORS["red"]}">●</span> offline<br>'
            f'<span style="color:{self._STATE_UNKNOWN}">●</span> no data')
        legend.setStyleSheet("font-size: 10px; color: #888;")
        row.addWidget(legend)
        row.addStretch()
        return box

    def _cam_online(self, cam):
        """A camera worker is 'online' if it started, has no error, and produced a frame
        in the last few seconds."""
        try:
            if cam is None or not getattr(cam, "started_ok", False) \
                    or getattr(cam, "error", None):
                return False
            snap = cam.latest()
            return snap is not None and (time.time() - snap.get("stamp", 0.0)) < 3.0
        except Exception:
            return False

    def _crate_states(self):
        """Per-external-camera state for the crate rig (skips the head-cam extra which has
        a host:port serial). Empty when there's no tracker on this machine."""
        tr = self._apriltag_tracker
        if tr is None:
            return []
        out = []
        for c in getattr(tr, "cams", []):
            if ":" in str(getattr(c, "serial", "")):
                continue                                   # head-cam extra, not external
            if not self._cam_online(c):
                out.append("red")
            elif self._recording_active:
                out.append("blue")
            else:
                out.append("green")
        return out

    def _status_snapshot(self):
        """THIS machine's component states {name: green|blue|red}, sent over the link so
        both managers show both robots + all cameras identically."""
        rec = bool(self._recording_active)
        teleop = (self._state == _S_TELEOP)
        # Robot itself — RED if the DDS link to the G1 is stale (bumped ethernet), so the
        # light no longer falsely reads healthy while the robot is actually gone.
        if self._hw_controller is None or self._state == _S_INIT or self._robot_comms_lost:
            robot = "red"
        elif rec or teleop:
            robot = "blue"
        else:
            robot = "green"
        # Neck (Dynamixel) — driven by the ControlThread's neck worker
        ct = getattr(self, "_control_thread", None)
        nw = getattr(ct, "_neck_worker", None) if ct is not None else None
        nt = getattr(nw, "_thread", None) if nw is not None else None
        if nt is None or not nt.is_alive():
            neck = "red"                                   # no neck / worker gone
        elif getattr(nw, "faulted", False):
            neck = "red"                                   # serial/servo fault → press Reboot Neck
        elif teleop:
            neck = "blue"
        else:
            neck = "green"
        # Head cam (internal RealSense, over the Jetson stream)
        loc = self._headcam_localizer
        if loc is None or not self._cam_online(getattr(loc, "_cam", None)):
            headcam = "red"
        elif rec or getattr(loc, "crate_contributing", False):
            headcam = "blue"
        else:
            headcam = "green"
        # ZED (on top of the robot)
        if not self._zed_recorder.is_streaming:
            zed = "red"
        elif self._zed_recorder.is_recording:
            zed = "blue"
        else:
            zed = "green"
        # ZED factory-calibration conf present? (needed to auto-write intrinsics into meta)
        if self._zed_intrinsics_meta is None:            # compute once, then cached
            self._zed_intrinsics_meta = self._zed_intrinsics(self._zed_localize_side())
        zedcal = "green" if self._zed_intrinsics_meta else "red"
        # SSH identity so the PEER can auto-derive the master's rsync destination
        # (master_user@master_ip:master_rec_dir) with no manual entry — the IP comes from
        # the live link socket, the user + recordings dir from here.
        # Digital-twin data readiness — is THIS robot capturing everything the twin needs
        # (robot_raw + ZED video + tactile)? Broadcast so the OTHER manager can gate its
        # Start Attempt on both robots being complete (no extra round-trip).
        twin_ok, twin_missing = self._twin_readiness()
        return {"robot_id": self._robot_id, "robot": robot, "neck": neck,
                "headcam": headcam, "zed": zed, "zedcal": zedcal,
                "crate": self._crate_states(),
                "twin_ready": twin_ok, "twin_missing": twin_missing,
                # This robot's LIVE Unity-toggle state, so the OTHER manager's peer
                # buttons mirror what this manager actually shows (not a stale OFF).
                "unity": {"audio_haptics": bool(self._audio_haptics_on),
                          "vibrotactile": bool(self._vibrotactile_on),
                          "viewport": bool(self._viewport_on)},
                "ssh_user": os.environ.get("USER", ""),
                "rec_dir": os.path.abspath(self._outdir_edit.text()
                                           or self._args.output_dir),
                # This machine's live STUDY state, so the OTHER manager — after a crash/restart —
                # can detect that a study is still in progress here and offer to resume it in
                # sync (session id + pointer + recorded-take ledger + trial order).
                "session": ({"open": True, "session_id": self._session_id,
                             "pid": self._rec_pid,
                             "master_code": getattr(self, "_rec_master_code", ""),
                             "peer_code": getattr(self, "_rec_peer_code", ""),
                             "condition": self._cond_combo.currentText(),
                             "attempt": int(self._attempt_spin.value()),
                             "recorded_takes": sorted([list(t) for t in self._recorded_takes]),
                             "trial_order": self._trial_order()}
                            if self._session_open() else {"open": False})}

    def _set_light(self, key, state):
        lbl = self._lights.get(key)
        if lbl is None:
            return
        color = self._STATE_COLORS.get(state, self._STATE_UNKNOWN)
        # setStyleSheet only when the color actually changes — this runs for every light
        # at 5 Hz and unconditional restyles were a top GIL consumer (teleop-lag incident).
        if getattr(lbl, "_last_style", None) == color:
            return
        lbl._last_style = color
        lbl.setStyleSheet(f"color: {color}; font-size: 15px;")

    def _refresh_status_panel(self):
        """Update all lights from this machine's snapshot + the peer's (over the link)."""
        if not hasattr(self, "_lights"):
            return
        local = self._status_snapshot()
        peer = {}
        if self._link is not None:
            st = self._link.get_peer_status()
            peer = (st.get("components") or {}) if st else {}
        by_id = {int(local["robot_id"]): local}
        if peer.get("robot_id") is not None:
            by_id[int(peer["robot_id"])] = peer
        for rid in (1, 2):
            snap = by_id.get(rid)
            for comp in ("robot", "neck", "headcam", "zed", "zedcal"):
                self._set_light((rid, comp), snap.get(comp) if snap else None)
        crate = (local.get("crate") or peer.get("crate") or [])
        for i in range(2):
            self._set_light(("crate", i), crate[i] if i < len(crate) else None)

    # =========================================================================
    # Status refresh (5 Hz)
    # =========================================================================

    def _update_rec_indicators(self):
        """Flash a red ● REC on each robot that is ACTIVELY recording (this machine from its
        own state, the other robot from its status broadcast) so the operator sees exactly
        which robots are capturing data. ~1.25 Hz blink driven off the 5 Hz status cadence."""
        if not hasattr(self, "_rec_dots"):
            return
        self._rec_blink_ctr = getattr(self, "_rec_blink_ctr", 0) + 1
        on = (self._rec_blink_ctr // 2) % 2 == 0          # toggle every ~400 ms
        my_id = int(self._robot_id)
        rec_by_id = {my_id: bool(self._recording_active)}
        st = self._link.get_peer_status() if self._link is not None else {}
        comps = (st or {}).get("components") or {}
        if comps.get("robot_id") is not None:
            rec_by_id[int(comps["robot_id"])] = bool(st.get("recording"))
        for rid, lbl in self._rec_dots.items():
            if rec_by_id.get(rid, False):
                want = ("#e74c3c" if on else "#7a2018", "●  REC")   # pulse bright↔dark red
            else:
                want = ("#7f8c8d", "○  idle")
            if getattr(lbl, "_rec_style", None) != want:          # restyle only on change
                lbl._rec_style = want
                lbl.setText(want[1])
                lbl.setStyleSheet(f"color: {want[0]}; font-size: 12px; font-weight: bold;")

    def _refresh_status(self):
        # System-status lights for both robots + all cameras (local + peer over the link).
        self._refresh_status_panel()
        # Flashing red REC indicator per robot (what's actually being captured).
        self._update_rec_indicators()
        # Mirror the peer's live Unity-toggle state onto the master's peer buttons.
        self._update_peer_unity_buttons()

        # After a crash/restart, offer (once) to resume a study still in progress here or on the
        # other machine, so the operator can't forget to and leave the two robots out of sync.
        self._maybe_prompt_resume_on_boot()

        # WebRTC connection
        connected = self._device is not None and self._device.is_connected
        _lbl_text(self._webrtc_lbl, f"WebRTC: {'Connected' if connected else 'Disconnected'}")
        _lbl_style(self._webrtc_lbl, f"color: {'green' if connected else 'red'}; font-size: 11px;")

        # Re-assert the manager's authoritative Unity state on EVERY (re)connection (see
        # _maybe_reassert_unity_on_reconnect) so a headset off/on can't leave the headset's
        # haptics toggle un-reasserted → silent haptics while teleop still runs.
        self._maybe_reassert_unity_on_reconnect()
        self._was_connected = connected

        # Mirror any in-VR toggle changes the operator made back onto our buttons.
        self._apply_unity_state_updates()

        # Assemble + log the end-to-end latency budgets (video + teleop).
        self._update_latency_budget()

        # Park the arms if the Quest DATA link freezes (WiFi/WebRTC drop or a starved loop),
        # and blend back in when it recovers — runs at the 5 Hz status cadence.
        self._check_link_watchdog()

        # Running-attempt elapsed timer (red past 3 min).
        self._update_attempt_timer()

        # Teleop NEVER auto-starts on connect — it engages only on an explicit press of the in-VR
        # Play button or the manager's green Start Teleop button. Refresh button states each tick so
        # the green button shows Start (green) vs Pause (yellow) as the teleop state changes.
        # NOTE: teleop is intentionally NOT auto-paused when the client drops — it runs standalone,
        # so a missing/unreliable client connection must not force it to pause. Use the Stop or
        # Pause buttons to end teleop.
        self._update_buttons()

        # ZED status
        streaming = self._zed_recorder.is_streaming
        recording = self._zed_recorder.is_recording
        uptime_s  = self._zed_recorder.stream_uptime_s

        _lbl_text(self._rec_zed_stream_lbl, f"ZED Streaming: {'On' if streaming else 'Off'}")
        _lbl_style(self._rec_zed_stream_lbl, f"color: {'green' if streaming else 'gray'}; font-size: 11px;")
        self._update_stream_button()  # keep the toggle button in sync with the actual state
        _lbl_text(self._rec_zed_rec_lbl, f"ZED Recording: {'On' if recording else 'Off'}")
        _lbl_style(self._rec_zed_rec_lbl, f"color: {'green' if recording else 'gray'}; font-size: 11px;")

        # Crate AprilTag tracking status
        if self._calib_proc is not None and self._calib_proc.poll() is None:
            crate_txt, crate_col = "Crate: calibrating…", "blue"
        elif self._apriltag_tracker is not None:
            if not self._apriltag_tracker.is_anchored:
                crate_txt, crate_col = "Crate: not anchored", "orange"
            else:
                res = self._apriltag_tracker.latest()
                if res and res.get("ok"):
                    ic = res.get("intercam_trans_mm")
                    ic_txt = f", Δcam {ic:.0f}mm" if ic is not None else ""
                    crate_txt = (f"Crate: tracked ({res['n_tags']} tag/"
                                 f"{res['n_cams']} cam{ic_txt})")
                    crate_col = "green"
                else:
                    crate_txt, crate_col = "Crate: lost (no tags)", "red"
        elif self._apriltag_starting:
            crate_txt, crate_col = "Crate: starting…", "gray"
        else:
            crate_txt, crate_col = "Crate: off", "gray"
        _lbl_text(self._rec_crate_lbl, crate_txt)
        _lbl_style(self._rec_crate_lbl, f"color: {crate_col}; font-size: 11px;")

        # Robot base self-localization status (head camera)
        if self._headcam_localizer is not None:
            res = self._headcam_localizer.latest()
            if res and res.get("ok"):
                base_txt = (f"Base: localized (z={res['base_z']:.2f}m, "
                            f"up={res['upright']:+.2f})")
                base_col = "green" if res["upright"] > 0.8 else "orange"
            else:
                base_txt, base_col = "Base: searching (no tag 0)", "orange"
        elif self._headcam_starting:
            base_txt, base_col = "Base: starting…", "gray"
        else:
            base_txt, base_col = "Base: off", "gray"
        _lbl_text(self._rec_base_lbl, base_txt)
        _lbl_style(self._rec_base_lbl, f"color: {base_col}; font-size: 11px;")

        # Crate contribution status (peer feeding the master's fusion)
        loc = self._headcam_localizer
        if loc is not None and getattr(loc, "crate_contributing", False):
            if not loc.base_anchored:
                c_txt, c_col = "Crate contribution: base NOT anchored", "orange"
            else:
                c = loc.latest_crate()
                if c and c.get("ok"):
                    c_txt = f"Crate contribution: sending ({c['n_tags']} tags → master)"
                    c_col = "green"
                else:
                    c_txt, c_col = "Crate contribution: on (no crate tags in view)", "gray"
        else:
            c_txt, c_col = "Crate contribution: off", "gray"
        _lbl_text(self._rec_contrib_lbl, c_txt)
        _lbl_style(self._rec_contrib_lbl, f"color: {c_col}; font-size: 11px;")

        # Latency panel ZED info
        _lbl_text(self._zed_status_lbl, f"Stream: {'On' if streaming else 'Off'}")
        _lbl_style(self._zed_status_lbl, f"color: {'green' if streaming else 'gray'}; font-size: 11px;")
        _lbl_text(self._zed_rec_lbl2, f"Rec: {'On' if recording else 'Off'}")
        _lbl_style(self._zed_rec_lbl2, f"color: {'green' if recording else 'gray'}; font-size: 11px;")
        if streaming:
            mins, secs = divmod(int(uptime_s), 60)
            _lbl_text(self._zed_uptime_lbl, f"Uptime: {mins}m {secs:02d}s")
            _lbl_style(self._zed_uptime_lbl, "color: green; font-size: 11px;")
        else:
            _lbl_text(self._zed_uptime_lbl, "Uptime: —")
            _lbl_style(self._zed_uptime_lbl, "color: gray; font-size: 11px;")

        # Pose receive rate derived from tracker
        _, _, pose_ivls, now = self._latency_tracker.get_all()
        if len(pose_ivls) >= 3:
            recent = [ms for t, ms in pose_ivls if now - t < 2.0]
            if recent:
                avg_ivl = sum(recent) / len(recent)
                rate_hz = 1000.0 / avg_ivl if avg_ivl > 0 else 0
                _lbl_text(self._pose_rate_lbl, f"Pose rate: {rate_hz:.1f} Hz")
                color = "green" if rate_hz > 50 else ("orange" if rate_hz > 20 else "red")
                _lbl_style(self._pose_rate_lbl, f"color: {color}; font-size: 11px;")
            else:
                _lbl_text(self._pose_rate_lbl, "Pose rate: — Hz")
                _lbl_style(self._pose_rate_lbl, "color: gray; font-size: 11px;")
        else:
            _lbl_text(self._pose_rate_lbl, "Pose rate: — Hz")
            _lbl_style(self._pose_rate_lbl, "color: gray; font-size: 11px;")

        # Room transcription status
        if hasattr(self, "_rec_transcript_lbl"):
            if self._transcriber is None:
                _lbl_text(self._rec_transcript_lbl, "Transcript: off")
                _lbl_style(self._rec_transcript_lbl, "color: gray; font-size: 11px;")
            else:
                ts = self._transcriber.status()
                if ts.get("error"):
                    _lbl_text(self._rec_transcript_lbl,
                              f"Transcript: ERR {str(ts['error'])[:40]}")
                    _lbl_style(self._rec_transcript_lbl, "color: red; font-size: 11px;")
                elif not ts.get("listening"):
                    _lbl_text(self._rec_transcript_lbl, "Transcript: starting…")
                    _lbl_style(self._rec_transcript_lbl, "color: #7f8c8d; font-size: 11px;")
                else:
                    n = ts.get("n_segments", 0)
                    who = ts.get("last_speaker") or "—"
                    enr = len(ts.get("enrolled", []))
                    _lbl_text(self._rec_transcript_lbl,
                              f"Transcript: listening ({n} segs, {enr}/2 voices, "
                              f"last: {who})")
                    _lbl_style(self._rec_transcript_lbl,
                               "color: green; font-size: 11px;"
                               if ts.get("session_open") else
                               "color: #16a085; font-size: 11px;")
            if hasattr(self, "_btn_calib_voices_tr"):
                self._btn_calib_voices_tr.setEnabled(
                    self._transcriber is not None and bool(self._transcribe_on))
        # Body pose CSV
        bp_on = self._device is not None and getattr(self._device, 'record_data', False)
        _lbl_text(self._rec_body_pose_lbl, f"Body Pose CSV: {'On' if bp_on else 'Off'}")
        _lbl_style(self._rec_body_pose_lbl, f"color: {'green' if bp_on else 'gray'}; font-size: 11px;")

        # Raw robot CSV (+ tactile)
        rb_on = self._session_recorder._active
        _lbl_text(self._rec_robot_lbl, f"Raw Robot CSV: {'On' if rb_on else 'Off'}")
        _lbl_style(self._rec_robot_lbl, f"color: {'green' if rb_on else 'gray'}; font-size: 11px;")

        # State label — clear, color-coded, with the pause REASON (headset removed, etc.).
        self._refresh_teleop_state_label()

        # Motor thermal / torque monitor (white→red per joint). Refreshed every OTHER
        # status tick (2.5 Hz): it's ~40 tiles and temps/torques don't need 5 Hz — this
        # halves the biggest remaining Qt cost of the status refresh (GIL, GUI thread).
        self._motor_tick = getattr(self, "_motor_tick", 0) + 1
        if (self._motor_monitor is not None and self._motor_monitor.isVisible()
                and self._motor_tick % 2 == 0):
            self._motor_monitor.update_from_controller(
                self._hw_controller, self._hand_controller,
                forces=self._latest_hand_forces()
            )

        # Wrench mode live readout + status
        self._refresh_wrench_labels()

    def _refresh_wrench_labels(self):
        if self._last_session_dir:
            self._wrench_csv_lbl.setText(
                f"Last recording: {Path(self._last_session_dir).name}")
            self._wrench_csv_lbl.setStyleSheet("color: #8e44ad; font-size: 11px;")
        else:
            self._wrench_csv_lbl.setText("Last recording: —")
            self._wrench_csv_lbl.setStyleSheet("color: gray; font-size: 11px;")

        worker = self._wrench_worker
        live = worker is not None and worker.is_enabled()

        def _fmt(side, lbl):
            if not live:
                lbl.setText(f"{side[0].upper()}: —")
                lbl.setStyleSheet("color: gray; font-size: 11px;")
                return
            w, _pt = worker.get(side)
            if w is None:
                lbl.setText(f"{side[0].upper()}: …")
                lbl.setStyleSheet("color: gray; font-size: 11px;")
                return
            F = w[:3]
            mag = float(np.linalg.norm(F))
            lbl.setText(f"{side[0].upper()}: |F| {mag:5.2f} N  "
                        f"[{F[0]:+5.1f} {F[1]:+5.1f} {F[2]:+5.1f}]")
            a = min(1.0, mag / max(1e-6, self._args.motor_torque_max))
            col = "#c0392b" if mag > 3.0 else ("#27ae60" if live else "gray")
            lbl.setStyleSheet(f"color: {col}; font-size: 11px;")

        _fmt("left", self._wrench_left_lbl)
        _fmt("right", self._wrench_right_lbl)

        mass_g = self._wrench_mass_spin.value()
        if mass_g > 0:
            self._wrench_expected_lbl.setText(
                f"expected: {we.expected_force_mag(mass_g):5.2f} N "
                f"({mass_g:.0f} g)")
            self._wrench_expected_lbl.setStyleSheet("color: #2980b9; font-size: 11px;")
        else:
            self._wrench_expected_lbl.setText("expected: —")
            self._wrench_expected_lbl.setStyleSheet("color: gray; font-size: 11px;")

    # =========================================================================
    # Button handlers
    # =========================================================================

    def _on_start_teleop(self):
        # Single engage/pause toggle: GREEN "Start Teleop" -> engage; YELLOW "Pause Teleop" -> pause.
        # Works WITHOUT a client connected — teleop no longer requires the XR client. (Only the
        # Linked Study "Both Robots" buttons depend on a connection, namely the peer link.) The XR
        # center Play/Pause button is kept in sync via the teleop_active command either way.
        # When linked, this drives BOTH robots (relabelled "(BOTH)"); unlinked, just this
        # robot. The peer engage/pause is idempotent so both converge.
        linked = self._linked()
        if self._state == _S_TELEOP:
            if linked:
                self._peer_cmd("pause_teleop")
            self._pause_teleop("manual")
            self._send_unity_cmd("teleop_active", {"enabled": False})
        else:
            if linked:
                self._peer_cmd("engage_teleop")
            self._start_teleop_session()   # engages + sends teleop_active{true}
        self._update_buttons()

    def _on_toggle_eco(self):
        """Toggle eco mode (idle arm relax + thermal governor) live."""
        self._eco_on = not self._eco_on
        label = "ON" if self._eco_on else "OFF"
        self._btn_eco.setText(f"🌱  Eco Mode: {label}")
        on_style  = "QPushButton { background: #27ae60; color: white; padding: 8px; }"
        off_style = "QPushButton { padding: 8px; }"
        self._btn_eco.setStyleSheet(on_style if self._eco_on else off_style)
        if self._control_thread is not None:
            self._control_thread.set_eco_enabled(self._eco_on)
        self._cfg["eco_mode"] = self._eco_on   # persist across reboots
        self._save_cfg()
        self._log(
            f"Eco mode {'enabled' if self._eco_on else 'disabled'} — arms "
            f"{'relax when idle and back off near the shoulder thermal limit' if self._eco_on else 'always held at full stiffness'}.")

    def _on_toggle_compliance(self):
        """Toggle current-triggered arm compliance live."""
        self._comp_on = not self._comp_on
        self._btn_compliance.setText(
            f"🤝  Arm Compliance: {'ON' if self._comp_on else 'OFF'}")
        on_style  = "QPushButton { background: #27ae60; color: white; padding: 8px; }"
        off_style = "QPushButton { padding: 8px; }"
        self._btn_compliance.setStyleSheet(on_style if self._comp_on else off_style)
        if self._control_thread is not None:
            self._control_thread.set_compliance_enabled(self._comp_on)
        self._cfg["arm_compliance"] = self._comp_on   # persist across reboots
        self._save_cfg()
        self._log(
            f"Arm compliance {'enabled' if self._comp_on else 'disabled'} — arms "
            f"{'yield near the overcurrent limit and return to pose' if self._comp_on else 'always held at full stiffness'}.")

    def _engage_teleop(self):
        """Engage teleop — start the IK pipeline with a fresh blend from the robot's CURRENT
        pose so the arms ramp smoothly instead of snapping. Does NOT require a connected
        client (works standalone); auto-arms from IDLE/WAITING. Called by BOTH the XR Play
        button and the manager's green Start Teleop button.
        """
        if self._state == _S_TELEOP:
            return
        q_t, q_l, q_r = np.zeros(3), np.zeros(7), np.zeros(7)
        if self._hw_controller is not None:
            try:
                q_t, q_l, q_r = self._hw_controller.get_current_upper_body()
            except Exception:
                pass
        # Resume (after a pause/drop) blends straight to the operator; an initial
        # start runs the full staged entry choreography.
        staged = not self._teleop_paused
        self._teleop_paused = False
        self._pause_reason = None
        if self._control_thread is not None:
            self._control_thread.start_teleop(q_t, q_l, q_r, staged=staged)
        self._state = _S_TELEOP
        self._log("Teleop engaged." if staged else "Teleop resumed (direct blend).")
        self._update_buttons()

    def _start_teleop_session(self):
        """Engage teleop AND tell the XR app to flip its in-VR Play button to Pause
        (`teleop_active` over the unity_cmds channel). Used by the manager's green Start
        Teleop button so a manager-driven start mirrors into the headset UI."""
        self._engage_teleop()
        if self._state == _S_TELEOP:
            self._send_unity_cmd("teleop_active", {"enabled": True})

    def _pause_teleop(self, reason="manual"):
        """Pause teleop — freeze in place. ``reason`` (headset / comms / linked / manual)
        is surfaced on the state label so the operator can see WHY it paused.

        Stops IK ticking so the robot holds its current pose (the goal buffer keeps
        streaming the last target). Does NOT lower the arms — that's the manager's
        explicit Stop & Lower button.
        """
        if self._state != _S_TELEOP:
            return
        if self._control_thread is not None:
            self._control_thread.stop_teleop()
        self._teleop_paused = True   # next engage is a RESUME (direct blend)
        self._pause_reason = reason
        self._state = _S_WAITING
        self._log({
            "headset": "Teleop paused — Quest headset removed / operator paused (holding pose).",
            "comms":   "Teleop paused — robot link lost (holding pose).",
            "linked":  "Teleop paused by the other station (holding pose).",
            "link":    "Teleop paused — Quest DATA link degraded/frozen (holding pose).",
        }.get(reason, "Teleop paused — holding pose."))
        self._update_buttons()

    def _refresh_teleop_state_label(self):
        """Set THIS robot's Teleop state label — clear, color-coded, and (when paused) WHY:
        headset removed / robot link lost / other station / manual."""
        st = self._state
        if st == _S_INIT:
            text, color = "INITIALIZING", "#7f8c8d"
        elif st == _S_TELEOP:
            text, color = "● TELEOP ACTIVE", "#27ae60"
        elif st == _S_WAITING and self._teleop_paused:
            text, color = {
                "headset": ("⏸ PAUSED — headset removed", "#e67e22"),
                "comms":   ("⏸ PAUSED — robot link lost", "#c0392b"),
                "linked":  ("⏸ PAUSED — by other station", "#f39c12"),
                "link":    ("⏸ PAUSED — Quest data link", "#c0392b"),
            }.get(self._pause_reason, ("⏸ PAUSED", "#f39c12"))
        elif st == _S_WAITING:
            text, color = "READY", "#2980b9"
        else:                                    # _S_IDLE
            text, color = "IDLE / STOPPED", "#7f8c8d"
        _lbl_text(self._teleop_state_lbl, f"State: {text}")
        _lbl_style(self._teleop_state_lbl, f"color: {color}; font-weight: bold;")

    def _on_start_teleop_solo(self):
        """Start/Pause THIS robot only — never touches the peer, even while linked. (Unlinked,
        the main Start/Pause button already acts on this robot alone, so these stay hidden.)"""
        if self._state == _S_TELEOP:
            self._pause_teleop("manual")
            self._send_unity_cmd("teleop_active", {"enabled": False})
        else:
            self._start_teleop_session()
        self._update_buttons()

    def _on_stop_teleop_solo(self):
        """Stop & lower THIS robot only — never touches the peer, even while linked."""
        self._on_stop_teleop()

    # ── PEER-only controls, driven from the master (link commands the peer runs) ──
    def _on_start_teleop_peer(self):
        """Start/Pause the PEER's teleop only (master button). Sends engage/pause the peer
        runs locally — same effect as the peer pressing its own Start/Pause; no override."""
        peer_teleop, _rec = self._peer_state()
        if peer_teleop == _S_TELEOP:
            self._peer_cmd("pause_teleop")
        else:
            self._peer_cmd("engage_teleop")

    def _on_stop_teleop_peer(self):
        """Stop & lower the PEER's arms only (master button)."""
        self._peer_cmd("stop_teleop")

    def _on_peer_toggle_audio_haptics(self):
        """Flip the PEER's Audio Haptics toggle (it runs its own toggle handler)."""
        self._peer_cmd("toggle_audio_haptics")

    def _on_peer_toggle_vibrotactile(self):
        self._peer_cmd("toggle_vibrotactile")

    def _on_peer_toggle_viewport(self):
        self._peer_cmd("toggle_viewport")

    # ── Test Haptics: synthetic per-region ramp to the Quest ────────────────────────
    def _on_test_haptics(self, sides):
        """Start (or stop, if one is already running) a synthetic haptics test on the given
        side(s). Each region ramps 0 → max → 0 while every other region stays 0, so the
        operator can confirm the headset plays haptics — audio or vibrotactile, whichever
        Unity toggle is enabled — and identify which region is which. sides is a tuple like
        ('left',), ('right',) or ('left','right'). Any Test button acts as Stop while running."""
        if self._test_haptics_timer is not None:      # already running → this press stops it
            self._stop_test_haptics("stopped by operator")
            return
        if self._recording_active:
            self._log("[haptics-test] finish the current attempt (End Attempt) before testing haptics.")
            return
        if self._device is None or not self._device.is_connected:
            self._log("[haptics-test] connect the Quest headset first — no client connected.")
            return
        if not (self._audio_haptics_on or self._vibrotactile_on):
            self._log("[haptics-test] NOTE: neither Audio Haptics nor Vibrotactile is enabled, so "
                      "the headset won't play anything. Enable one and re-run to feel/hear it.")
        # Hold the live streamer's own sends so its near-zero frames can't interleave (stutter)
        # the synthetic ramp on the shared 'haptics' channel.
        if self._haptics_streamer is not None:
            self._haptics_streamer.suppress_output(True)
        self._test_haptics_sides = tuple(sides)
        self._test_haptics_t0 = time.monotonic()
        self._test_haptics_timer = QTimer(self)
        self._test_haptics_timer.timeout.connect(self._test_haptics_tick)
        self._test_haptics_timer.start(max(1, int(round(1000.0 / _HAPTICS_TEST_SEND_HZ))))
        who = "+".join(s.capitalize() for s in self._test_haptics_sides)
        self._log(f"[haptics-test] {who}: ramping " + " → ".join(_HAPTICS_TEST_ORDER)
                  + f" (1 s up / 1 s down, {_HAPTICS_TEST_GAP_S:g} s gap). Press any Test button to stop.")
        self._update_test_haptics_buttons()

    def _test_haptics_region_state(self, elapsed):
        """(region, intensity) at ``elapsed`` seconds into the sequence, or (None, 0.0) once it
        is complete. Each region owns UP+DOWN+GAP seconds: a triangular 0→max→0 ramp (max = 1.0
        on the 0..1 intensity scale) followed by GAP seconds of silence before the next region."""
        cycle = _HAPTICS_TEST_UP_S + _HAPTICS_TEST_DOWN_S + _HAPTICS_TEST_GAP_S
        total = cycle * len(_HAPTICS_TEST_ORDER)
        if elapsed >= total:
            return None, 0.0
        k = int(elapsed / cycle)
        t = elapsed - k * cycle
        if t < _HAPTICS_TEST_UP_S:
            val = t / _HAPTICS_TEST_UP_S
        elif t < _HAPTICS_TEST_UP_S + _HAPTICS_TEST_DOWN_S:
            val = 1.0 - (t - _HAPTICS_TEST_UP_S) / _HAPTICS_TEST_DOWN_S
        else:
            val = 0.0                                 # inter-region gap
        return _HAPTICS_TEST_ORDER[k], float(min(1.0, max(0.0, val)))

    def _test_haptics_tick(self):
        """One synthetic frame during the ramp; finishes + cleans up when the sequence ends."""
        if self._test_haptics_t0 is None:
            return
        region, val = self._test_haptics_region_state(time.monotonic() - self._test_haptics_t0)
        if region is None:
            self._stop_test_haptics("complete")
            return
        zero = {r: 0.0 for r in _HAPTICS_TEST_REGIONS}
        active = dict(zero); active[region] = val
        left = active if "left" in self._test_haptics_sides else dict(zero)
        right = active if "right" in self._test_haptics_sides else dict(zero)
        self._send_test_haptics(left, right)

    def _send_test_haptics(self, left, right):
        """Send ONE synthetic haptics payload — identical schema to the live streamer's — so the
        Quest triggers audio/vibrotactile exactly as it does for real contact."""
        if self._device is None:
            return
        payload = json.dumps({"type": "haptics", "timestamp": time.time(),
                              "left": left, "right": right})
        try:
            self._device.send_haptics(payload)
        except Exception as exc:                       # noqa: BLE001
            self._log(f"[haptics-test] send failed: {exc}")

    def _stop_test_haptics(self, reason=""):
        """Stop the sequence: kill the timer, send one all-zero frame (nothing left buzzing),
        and resume the live haptics streamer."""
        if self._test_haptics_timer is not None:
            self._test_haptics_timer.stop()
            self._test_haptics_timer = None
        self._test_haptics_t0 = None
        zero = {r: 0.0 for r in _HAPTICS_TEST_REGIONS}
        self._send_test_haptics(dict(zero), dict(zero))
        if self._haptics_streamer is not None:
            self._haptics_streamer.suppress_output(False)
        if reason:
            self._log(f"[haptics-test] {reason}.")
        self._update_test_haptics_buttons()

    def _update_test_haptics_buttons(self):
        """Reflect running/idle on the three Test Haptics buttons; while a sequence runs, the
        active button becomes Stop and the other two are disabled (one sequence at a time)."""
        if not hasattr(self, "_btn_test_haptics_left"):
            return
        running = self._test_haptics_timer is not None
        # A test can only START with the XR client connected and no attempt recording; while
        # one runs, only the active button (its Stop) stays live.
        connected = self._device is not None and self._device.is_connected
        can_start = connected and not self._recording_active
        specs = ((self._btn_test_haptics_left, "Left", ("left",)),
                 (self._btn_test_haptics_right, "Right", ("right",)),
                 (self._btn_test_haptics_both, "Both", ("left", "right")))
        for btn, label, sides in specs:
            is_active = running and tuple(self._test_haptics_sides) == sides
            if is_active:
                btn.setText(f"⏹  Stop Test ({label})")
                btn.setStyleSheet(self._TOGGLE_STYLE_ON)
            else:
                btn.setText(f"🖐  Test Haptics ({label})")
                btn.setStyleSheet("QPushButton { padding: 8px; }")
            btn.setEnabled(is_active if running else can_start)

    def _update_peer_unity_buttons(self):
        """Mirror the PEER's live Unity-toggle state (audio-haptics / vibrotactile / FPV)
        onto the master's peer buttons, read from the peer's status broadcast — so they show
        ON/OFF matching what the peer manager actually shows, instead of a stale OFF. Shows
        '?' (grey) when the peer isn't reporting (unlinked or an older peer build). Restyles
        only on change (this runs at the 5 Hz status cadence)."""
        if not hasattr(self, "_btn_peer_audio_haptics"):
            return
        st = self._link.get_peer_status() if self._link is not None else {}
        unity = ((st or {}).get("components") or {}).get("unity")
        specs = ((self._btn_peer_audio_haptics, "🔊  Peer Audio Haptics", "audio_haptics"),
                 (self._btn_peer_vibrotactile,  "📳  Peer Vibrotactile",  "vibrotactile"),
                 (self._btn_peer_viewport,      "🖥️  Peer FPV Display",   "viewport"))
        for btn, label, key in specs:
            val = unity.get(key) if isinstance(unity, dict) else None
            if val is None:
                text, style = f"{label}: ?", self._TOGGLE_STYLE_OFF
            else:
                text = f"{label}: {'ON' if val else 'OFF'}"
                style = self._TOGGLE_STYLE_ON if val else self._TOGGLE_STYLE_OFF
            if getattr(btn, "_peer_unity_shown", None) != (text, style):
                btn._peer_unity_shown = (text, style)
                btn.setText(text)
                btn.setStyleSheet(style)

    # ── Robot-comms watchdog (bumped ethernet / lost DDS link) ────────────────
    def _check_robot_comms(self):
        """Detect a lost robot DDS link (e.g. a bumped ethernet cable): the LowState stops
        arriving so ``state_age()`` grows. On loss: warn loudly, flag the robot light red,
        pause teleop, and DAMP the arms so the streamed command is limp — so when the link
        returns the robot doesn't SNAP back to the stale position. On recovery: reclaim
        low-level control (release the FSM) + re-sync, but do NOT auto-resume teleop.

        IMPORTANT: this cannot PREVENT the robot's onboard controller from taking over
        (it can stand up) during the outage — the manager has no path to the robot over a
        dead link. It makes the loss visible and the reconnect safe."""
        hw = self._hw_controller
        if hw is None or self._state == _S_INIT:
            return
        try:
            age = hw.state_age()
        except Exception:
            return
        if age == float("inf"):          # never received state yet — startup handles that
            return
        timeout = float(getattr(self._args, "robot_comms_timeout", 0.3))
        lost = age > timeout
        now = time.monotonic()
        if lost:
            if not self._robot_comms_lost:
                self._robot_comms_lost = True
                self._comms_lost_since = now
                self._on_robot_comms_lost(age)
            # A SUSTAINED loss (state gone for > robot_powerdown_s) is a hardware power-cycle,
            # not a momentary ethernet bump (which reseats in well under a second and a robot
            # boot keeps DDS down for tens of seconds). Escalate to the UNBOOTED state so the
            # robot is never streamed commands the instant it comes back on the network.
            powerdown_s = float(getattr(self._args, "robot_powerdown_s", 4.0))
            if (not self._robot_powered_down and self._comms_lost_since is not None
                    and now - self._comms_lost_since >= powerdown_s):
                self._on_robot_powered_down()
        elif self._robot_comms_lost:
            self._robot_comms_lost = False
            self._comms_lost_since = None
            self._on_robot_comms_recovered()

    def _on_robot_comms_lost(self, age):
        self._log(f"*** ROBOT LINK LOST — no state from the G1 for {age * 1000:.0f} ms. "
                  "CHECK THE ETHERNET CABLE. The robot's onboard controller may take over "
                  "(it can STAND UP) — the manager cannot command it over a dead link. "
                  "Teleop paused; the robot is FROZEN at its current pose (held, NOT damped) "
                  "so it neither sags nor snaps when the link returns. A sustained loss "
                  "escalates to UNBOOTED. ***")
        # FREEZE, don't damp. The old behavior damped the arms on any blip — but that made the
        # arms (and, via the leg thermal governor releasing when arm-damp is set, the load
        # joints) go LIMP and sag during the outage, then spring back violently on reconnect.
        # Instead just STOP computing new teleop targets and keep streaming the CURRENT held
        # pose at normal stiffness: a receive-only hiccup (sends still land) never sags at all,
        # and a full outage recovers under the position-error clamp — either way, no snap. A
        # SUSTAINED loss still escalates to UNBOOTED (_on_robot_powered_down) after
        # robot_powerdown_s, which stops commanding the robot entirely.
        if self._state == _S_TELEOP:
            self._pause_teleop("comms")
        self._refresh_status()          # robot light → red immediately

    def _on_robot_comms_recovered(self):
        self._log("*** ROBOT LINK RESTORED. Reclaiming low-level control (releasing the "
                  "high-level FSM) + re-syncing to the robot's actual pose. Teleop is NOT "
                  "auto-resumed — re-engage manually (staged) so nothing snaps. If the "
                  "robot stood up during the outage, bring it back to the kneel first. ***")
        # If a locomotion FSM grabbed control during the outage (the stand-up), release it
        # again so low-level rt/lowcmd regains exclusive ownership.
        if getattr(self._args, "kneeling", False):
            try:
                from projects.inspire_hand_teleop.demo_g1_inspire_xr_robot_teleop_full_body_hw \
                    import release_motion_mode
                release_motion_mode()
            except Exception as exc:
                self._log(f"[watchdog] FSM re-release failed: {exc}")
        # Re-sync the sim/viewer to the robot's real current pose (it was frozen/stale).
        try:
            q_t, q_l, q_r = self._hw_controller.get_current_upper_body()
            self._sync_sim(q_r, q_l, q_t)
            mujoco.mj_forward(self._model, self._data)
        except Exception:
            pass
        self._update_buttons()

    def _on_robot_powered_down(self):
        """The G1's state has been gone for a SUSTAINED period (> robot_powerdown_s) — treat it
        as a hardware power-cycle (power button / lost power) and switch the manager to the
        UNBOOTED state, so it never streams the stale goal buffer to the robot the instant it
        reconnects. Unlike Safe Shutdown we CANNOT gently lower the arms (the robot is off), so
        we just stop the control loop and flip the flags. The operator presses Bootup (which
        now reads 'Reboot') to re-sync + reclaim control once the robot has finished booting."""
        self._log("*** ROBOT POWERED OFF — no state for > "
                  f"{float(getattr(self._args, 'robot_powerdown_s', 4.0)):.0f}s (power button / "
                  "lost power). Switching to UNBOOTED so the robot is NOT commanded when it "
                  "reconnects. Once it has finished rebooting, press Bootup (Reboot) to "
                  "re-establish control — teleop stays disabled until then. ***")
        # Stop the control loop so it can't stream the (now stale) goal buffer to the robot the
        # moment it reappears on the network. Same teardown Safe Shutdown does, minus the
        # arm-lowering choreography (there is no robot to command).
        try:
            if self._control_thread is not None:
                self._control_thread.stop_teleop()
                self._control_thread.stop()
        except Exception:
            pass
        # Hold the Inspire hands off too — idle their Modbus workers so the grippers aren't
        # driven (and the bus goes quiet, ending the timeout spam) until the reboot reconnects
        # them. Released by the hands' reboot() at re-boot.
        if self._hand_controller is not None:
            try:
                self._hand_controller.suppress_output(True)
            except Exception:
                pass
        # Robot-down / unbooted flags — identical to the post-Safe-Shutdown state, so teleop +
        # attempt buttons grey out (ready = state != _S_INIT) and the Bootup button becomes an
        # enabled 'Reboot' (via _update_conditional_buttons). _hw_booted stays True so the boot
        # button routes to the reboot path, not a cold first boot.
        self._robot_powered_down = True
        self._exit_damping_done = True
        self._state = _S_INIT
        self._refresh_status()
        self._update_buttons()

    # ── Quest DATA-link watchdog (WiFi/WebRTC freeze — NOT the robot DDS link) ──────
    def _check_link_watchdog(self):
        """Auto-pause teleop (and DAMP the arms) when the Quest DATA link freezes, then
        auto-resume once it's healthy again. Distinct from _check_robot_comms (that guards
        the G1's ethernet/DDS link); THIS guards the operator→robot WebRTC pose stream.

        Why it's needed: on a loaded host the aiortc event loop can be starved hard enough
        that ICE consent-freshness fails and the whole peer connection drops — body_pose
        delivery freezes for seconds (the operator's teleop 'freeze + disconnect'). Rather
        than rely on someone noticing and hitting Stop, park the arms limp the instant the
        stream goes stale, show WHY, and blend back in when it recovers.

        Primary signal is pose_age_s() (monotonic time since the last body_pose frame) — it
        grows on BOTH a connection drop and a starved loop. RTT is a secondary early trip.
        This escalates ABOVE the ControlThread's 0.5 s stale-pose hold: that briefly holds
        the arms on a blip; this pauses + damps + notifies on a real freeze."""
        dev = self._device
        if dev is None:
            return
        a = self._args
        # Guard only while teleop is running, or while WE paused it for the link (so we can
        # auto-resume). Any OTHER pause reason (headset/comms/linked/manual) is left alone.
        active = self._state == _S_TELEOP
        ours_paused = (self._state == _S_WAITING and self._pause_reason == "link")
        if not (active or ours_paused):
            self._link_bad_since = None
            self._link_ok_since = None
            return
        try:
            age = dev.pose_age_s()
            rtt = dev.network_rtt_ms()
        except Exception:
            return
        now = time.monotonic()
        freeze_s = float(getattr(a, "link_freeze_s", 1.0))
        rtt_bad_ms = float(getattr(a, "link_pause_rtt_ms", 400.0))
        # FREEZE (no fresh frames for freeze_s) is the reliable danger signal — it's when the
        # arms would chase a stale target / snap on reconnect. RTT is NOT reliable on a loaded
        # host: a starved event loop inflates the MEASURED rtt to hundreds of ms WHILE frames
        # keep flowing at full rate (teleop is merely laggy, not frozen), so pausing on it
        # would block teleop entirely. RTT-based pausing is therefore OPT-IN
        # (--link-pause-on-rtt); the default trips on freeze only. RTT still drives the
        # DEGRADED indicator so you keep the early-warning visibility.
        pause_on_rtt = bool(getattr(a, "link_pause_on_rtt", False))
        frozen = age is not None and age >= freeze_s
        rtt_bad = pause_on_rtt and rtt is not None and rtt >= rtt_bad_ms
        bad = frozen or rtt_bad

        if active:
            if bad:
                self._link_bad_since = self._link_bad_since or now
                debounce = float(getattr(a, "link_pause_debounce_s", 0.3))
                if now - self._link_bad_since >= debounce:
                    why = (f"stream frozen {age:.1f}s" if frozen
                           else f"RTT {rtt:.0f} ms ≥ {rtt_bad_ms:.0f}")
                    self._log(f"[link-watchdog] Quest data link degraded ({why}) — pausing "
                              "teleop + damping arms so they won't chase a stale target or "
                              "snap on reconnect.")
                    self._pause_teleop("link")
                    if self._control_thread is not None:
                        try:
                            self._control_thread.enter_arm_damping()
                        except Exception:
                            pass
                    self._link_bad_since = None
                    self._link_ok_since = None
            else:
                self._link_bad_since = None
        elif ours_paused:
            if not getattr(a, "link_auto_resume", True):
                return                                   # manual re-engage only
            resume_rtt_ms = float(getattr(a, "link_resume_rtt_ms", 120.0))
            # Healthy = fresh frames flowing again. Only gate resume on RTT when RTT-pausing
            # is enabled — otherwise a chronically starved-loop RTT (which never drops to the
            # resume threshold) would keep teleop paused forever after a freeze recovers.
            fresh = age is not None and age <= freeze_s * 0.5
            rtt_ok = (not pause_on_rtt) or (rtt is None or rtt <= resume_rtt_ms)
            healthy = fresh and rtt_ok
            if healthy:
                self._link_ok_since = self._link_ok_since or now
                if now - self._link_ok_since >= float(getattr(a, "link_resume_s", 2.0)):
                    self._log("[link-watchdog] Quest data link recovered — resuming teleop "
                              "(direct blend from the robot's current pose).")
                    self._start_teleop_session()         # resume + mirror the in-VR Play/Pause UI
                    self._link_ok_since = None
                    self._link_bad_since = None
            else:
                self._link_ok_since = None

    def _latest_hand_forces(self):
        """Haptics streamer's cached fingertip/pad/palm verification snapshot (or None).
        No hardware read — reuses what the haptics loop already captured; safe to call from
        the Qt thread only when the monitor is visible."""
        hs = getattr(self, "_haptics_streamer", None)
        if hs is None:
            return None
        try:
            return hs.latest_forces()
        except Exception:
            return None

    def _on_show_motor_monitor(self):
        """Show / raise the arm motor thermal-torque monitor window."""
        if self._motor_monitor is None:
            return
        self._motor_monitor.show()
        self._motor_monitor.raise_()
        self._motor_monitor.activateWindow()

    def _update_conditional_buttons(self):
        """Grey out every button that is a no-op under the current state, so it can't be clicked
        until its precondition is met — each gate mirrors that handler's own early-return guard
        (robot not booted, session not open, recording in progress, not connected, wrong link
        role, etc.). Runs at the 5 Hz status cadence; buttons already gated inline in
        _update_buttons (Start Session/Attempt, teleop, reboots, twin/dataset, crate…) are not
        repeated here."""
        session_open = self._session_open()
        # Prepare Study can't run mid-take.
        self._btn_prepare_study.setEnabled(not self._recording_active)
        # Camera (re)calibration shifts the shared world frame → refused while a session is open.
        self._btn_calibrate_now.setEnabled(not session_open)
        # Practice needs an OPEN session and no take recording.
        if hasattr(self, "_btn_practice_main"):
            self._btn_practice_main.setEnabled(session_open and not self._recording_active)
        # Mic test: master/standalone only (peer has no room mic), and not while one is running.
        if hasattr(self, "_btn_test_mic"):
            mic_busy = getattr(self, "_mic_test_proc", None) is not None
            self._btn_test_mic.setEnabled(self._transcription_allowed() and not mic_busy)
        # Head-cam view needs the localizer running.
        if hasattr(self, "_btn_headcam_view"):
            self._btn_headcam_view.setEnabled(self._headcam_localizer is not None)
        # Safe Shutdown: only when the robot is actually up (booted, not already down/e-stopped).
        if hasattr(self, "_btn_safe_shutdown"):
            self._btn_safe_shutdown.setEnabled(self._hw_booted and not self._exit_damping_done)
        # ZED streaming toggle is only blocked from STOPPING mid-record (else always actionable).
        self._btn_stream_toggle.setEnabled(
            not (self._zed_recorder.is_streaming and self._zed_recorder.is_recording))
        # Link: Connect needs a role selected; Disconnect needs a live link.
        if hasattr(self, "_btn_link_connect"):
            self._btn_link_connect.setEnabled(self._link_role() != "none")
        if hasattr(self, "_btn_link_disconnect"):
            self._btn_link_disconnect.setEnabled(self._link is not None)
        # Peer's "Test link to master": peer role only, and not while a check is already running.
        if hasattr(self, "_btn_test_dest"):
            ssh_busy = bool(getattr(self, "_ssh_check_thread", None) is not None
                            and self._ssh_check_thread.is_alive())
            self._btn_test_dest.setEnabled(self._link_role() == "peer" and not ssh_busy)
        # Bootup button: after a Safe-Shutdown / E-STOP the (previously booted) robot is down →
        # offer Reboot, blocked mid-take. Before/during/after a boot the button is owned by the
        # boot flow itself (enabled at rest, disabled while booting + once booted), so leave it.
        if self._hw_booted and self._exit_damping_done:
            self._btn_bootup.setText("♻  Reboot Robot")
            self._btn_bootup.setEnabled(not self._recording_active)
        # Per-hand Test Haptics buttons (connection + recording + running-state aware).
        self._update_test_haptics_buttons()

    def _update_attempt_timer(self):
        """Show how long the current attempt (or practice round) has been recording (mm:ss);
        the label turns RED once 3 minutes have elapsed — same behaviour for both. Practice and
        attempts are mutually exclusive, so they share this one label. Idle text otherwise."""
        if not hasattr(self, "_attempt_timer_lbl"):
            return
        mode = ("Attempt" if self._recording_active
                else "Practice" if self._practice_active else None)
        if mode is None:
            self._attempt_start_t = None
            self._timer_mode = None
            self._attempt_timer_lbl.setText("Attempt: —")
            self._attempt_timer_lbl.setStyleSheet("font-weight: bold; color: gray;")
            return
        # (Re)start the clock on first entry OR when the active mode changes (practice→attempt),
        # so an attempt never inherits the practice round's elapsed time.
        if self._attempt_start_t is None or getattr(self, "_timer_mode", None) != mode:
            self._attempt_start_t = time.monotonic()
            self._timer_mode = mode
        el = time.monotonic() - self._attempt_start_t
        mm, ss = divmod(int(el), 60)
        over = el >= 180.0
        self._attempt_timer_lbl.setText(
            f"{mode}: {mm:d}:{ss:02d}" + ("   ⚠ over 3 min" if over else ""))
        self._attempt_timer_lbl.setStyleSheet(
            "font-weight: bold; color: %s;" % ("#c0392b" if over else "#27ae60"))

    def _on_reboot_webrtc(self):
        """Reboot the WebRTC server WITHOUT restarting the manager — recovers a wedged aiortc /
        data-channel state (e.g. haptics stops reaching the gloves, or the out-of-process WebRTC
        subprocess crashed) and gives the Quest a fresh connection. The bHaptics gloves live on
        the HEADSET (Bluetooth), so the manager can't reboot them directly — the most it can do is
        rebuild the haptics data channel + re-assert the haptics toggles, which kicks the XR app to
        re-init its haptics; if they still don't buzz, re-pair the gloves on the headset. Runs off
        the UI thread; blocked mid-take (it drops the Quest link)."""
        if self._recording_active:
            self._log("[webrtc] finish (End Attempt) the current take before rebooting WebRTC "
                      "— it drops the Quest link.")
            return
        if getattr(self, "_device", None) is None or not getattr(self, "_webrtc_dev_init", None):
            self._log("[webrtc] no WebRTC server to reboot.")
            return
        if not getattr(self, "_webrtc_subprocess", False):
            self._log("[webrtc] in-process WebRTC (no --webrtc-subprocess) can't be hot-rebooted "
                      "— restart the manager. Run with --webrtc-subprocess for a reboot button.")
            return
        self._log("[webrtc] rebooting the WebRTC server + haptics channel — the Quest will "
                  "reconnect (~5-10 s); toggles re-assert on connect. If the gloves still don't "
                  "buzz afterward, re-pair them on the headset.")
        threading.Thread(target=self._reboot_webrtc_bg, daemon=True).start()

    def _check_webrtc_alive(self):
        """Watchdog for the out-of-process WebRTC subprocess. The manager SURVIVES a subprocess
        crash (every proxy-client handler catches + continues), but the Quest link + haptics go
        dead. If the subprocess has EXITED, auto-restart it once (rate-limited) so teleop/haptics
        recover without a manual Reboot WebRTC or a manager restart. Skips mid-take."""
        if not getattr(self, "_webrtc_subprocess", False):
            return
        dev = getattr(self, "_device", None)
        if dev is None or getattr(dev, "worker_alive", True):
            return
        now = time.monotonic()
        if (self._recording_active or self._webrtc_reboot_inflight
                or now - self._webrtc_last_autoreboot < 10.0):
            return
        self._webrtc_reboot_inflight = True
        self._webrtc_last_autoreboot = now
        self._log("[webrtc] the WebRTC server subprocess DIED (aiortc crash?) — auto-restarting "
                  "it; the manager is unaffected and the Quest reconnects in ~5-10 s.")
        threading.Thread(target=self._reboot_webrtc_bg, daemon=True).start()

    def _reboot_webrtc_bg(self):
        """Release the listening port, start a new server, and replace consumer references."""
        try:
            from xrt_devices.study_process import WebRTCServerProxy
            old = self._device
            if old is not None:
                old.stop()
            new = WebRTCServerProxy(self._webrtc_dev_init, log_fn=self._log)
            new.start()
        except Exception as exc:                          # noqa: BLE001
            self._webrtc_reboot_inflight = False
            self._log(f"[webrtc] reboot failed ({exc}). Restart the manager if WebRTC stays unavailable.")
            return
        # Each consumer sees either the stopped device or the new one.
        # The stopped device reports no fresh pose.
        self._device = new
        if self._control_thread is not None:
            self._control_thread._device = new
        if self._haptics_streamer is not None:
            self._haptics_streamer._device = new
        if self._motor_stats_streamer is not None:
            self._motor_stats_streamer._device = new
        self._webrtc_reboot_inflight = False
        self._log("[webrtc] WebRTC server rebooted — fresh haptics/pose channels; waiting for the "
                  "Quest to reconnect.")

    def _on_reboot_zed_streaming(self):
        """Full ZED video-pipeline reboot (fresh mediamtx + publisher + camera reopen) —
        the equivalent of what a manager restart does to the stream, without touching
        teleop / recording / calibration / hardware. Blocked while an attempt records."""
        if self._recording_active:
            self._log("[zed] finish (End Attempt) the current take before rebooting "
                      "streaming — it would kill the video being recorded.")
            return
        self._log("[zed] rebooting ZED streaming pipeline — fresh mediamtx + publisher + "
                  "camera reopen (~2-4 s). Teleop / session state are untouched.")
        self._zed_reboot_result = None
        threading.Thread(target=self._reboot_zed_bg, daemon=True).start()
        QTimer.singleShot(6000, self._log_zed_reboot_result)

    def _reboot_zed_bg(self):
        """Background: perform the blocking full streaming reboot (prints to the terminal)."""
        try:
            self._zed_reboot_result = bool(self._zed_recorder.reboot_streaming())
        except Exception as exc:                       # noqa: BLE001
            self._zed_reboot_result = False
            print(f"[zed] streaming reboot error: {exc}")

    def _log_zed_reboot_result(self):
        res = getattr(self, "_zed_reboot_result", None)
        if res is None:
            self._log("[zed] streaming reboot still in progress — check the terminal.")
            return
        if res:
            self._log("[zed] streaming pipeline rebooted — verifying frames…")
            QTimer.singleShot(7000, self._verify_zed_publishing)   # /zed_stereo live?
        else:
            self._log("[zed] streaming reboot FAILED to come back up — check the camera / "
                      "mediamtx / GStreamer.")
        self._update_stream_button()

    def _apply_hand_dofs_to_control_thread(self):
        """Push the saved per-hand disabled-DOF config to the running ControlThread so it stops
        commanding those actuators (held at rest). No-op if the thread isn't up yet — the boot
        path calls this again once it is."""
        cfg = self._cfg.get("disabled_hand_dofs") or {"left": [], "right": []}
        left = [int(i) for i in cfg.get("left", []) if 0 <= int(i) < 6]
        right = [int(i) for i in cfg.get("right", []) if 0 <= int(i) < 6]
        ct = getattr(self, "_control_thread", None)
        if ct is not None:
            ct.set_disabled_hand_dofs(left, right)

    def _on_hand_dofs(self):
        """Per-hand DOF enable/disable menu. Each of the 6 actuators per hand gets a checkbox
        (checked = enabled). Unchecking one STOPS commanding it — it's frozen at rest — so a broken
        actuator (e.g. a sheared thumb-rotation shaft) is never driven and the recording + digital
        twin match the real hand. Saved to THIS machine's config (persists across sessions) and
        applied live to the running ControlThread. All DOFs enabled by default."""
        cur = self._cfg.get("disabled_hand_dofs") or {"left": [], "right": []}
        cur_dis = {"left": set(int(i) for i in cur.get("left", [])),
                   "right": set(int(i) for i in cur.get("right", []))}

        dlg = QDialog(self)
        dlg.setWindowTitle("Hand DOFs — Enable / Disable")
        root = QVBoxLayout(dlg)
        info = QLabel("Uncheck a DOF to STOP commanding it (frozen at rest). Use it for a broken "
                      "actuator so the digital twin matches the real hand. Per-machine; saved "
                      "between sessions.")
        info.setWordWrap(True)
        root.addWidget(info)

        checks = {"left": [], "right": []}
        cols = QHBoxLayout()
        for side in ("left", "right"):
            box = QGroupBox(f"{side.capitalize()} Hand")
            col = QVBoxLayout(box)
            for idx, label in enumerate(_HAND_DOF_LABELS):
                cb = QCheckBox(label)
                cb.setChecked(idx not in cur_dis[side])      # checked = enabled
                col.addWidget(cb)
                checks[side].append(cb)
            cols.addWidget(box)
        root.addLayout(cols)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        root.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        disabled = {side: [i for i, cb in enumerate(checks[side]) if not cb.isChecked()]
                    for side in ("left", "right")}
        self._cfg["disabled_hand_dofs"] = disabled
        self._save_cfg()
        self._apply_hand_dofs_to_control_thread()

        def _fmt(side):
            d = disabled[side]
            return ("all enabled" if not d
                    else ", ".join(_FINGER_NAMES[i] for i in d) + " DISABLED")
        self._log(f"[hand-dofs] left: {_fmt('left')} | right: {_fmt('right')} — saved + applied. "
                  "Disabled DOFs are held at rest (not commanded), so the recording + digital "
                  "twin match the real hand.")

    def _on_reboot_hands(self):
        """Reboot the Inspire hands + force sensors (Modbus reconnect + worker restart, the
        same bring-up as manager bootup) WITHOUT restarting the manager. The Modbus work
        blocks ~1-2 s, so it runs on a background thread (the controller keeps serving its
        cached read() throughout); the outcome is logged from the UI thread shortly after.
        Blocked while an attempt records (it would disrupt the take)."""
        if self._recording_active:
            self._log("[hands] finish (End Attempt) the current take before rebooting the "
                      "hands.")
            return
        if self._hand_controller is None:
            self._log("[hands] no hand controller connected — nothing to reboot.")
            return
        self._log("[hands] rebooting Inspire hands + force sensors (Modbus reconnect + "
                  "worker restart)… fingers hold their pose for ~1-2 s.")
        self._hands_reboot_result = None
        threading.Thread(target=self._reboot_hands_bg, daemon=True).start()
        QTimer.singleShot(8000, self._log_hands_reboot_result)

    def _reboot_hands_bg(self):
        """Background: perform the blocking hand reboot. Stores the result for the UI-thread
        follow-up; the controller also prints the outcome to the terminal."""
        try:
            self._hands_reboot_result = bool(self._hand_controller.reboot())
        except Exception as exc:                       # noqa: BLE001
            self._hands_reboot_result = False
            print(f"[hands] reboot error: {exc}")

    def _log_hands_reboot_result(self):
        res = getattr(self, "_hands_reboot_result", None)
        if res is None:
            self._log("[hands] reboot still in progress — check the terminal for the result.")
        elif res:
            self._log("[hands] reboot OK — hands + force sensors reconnected and reading.")
        else:
            self._log("[hands] reboot FAILED — Modbus didn't reconnect. Check hand power / "
                      "cables / IPs (192.168.123.210/211).")

    def _on_reboot_neck(self):
        """HARD-reset the Ostrich neck: tear down its serial link and rebuild the interface +
        worker FROM SCRATCH — the same clean slate the neck gets on a manager restart, but
        without restarting the manager. An in-place Dynamixel REBOOT reuses the existing serial
        port, so it cannot recover a neck whose USB link actually dropped (e.g. jolted loose by
        a violent robot movement) — the port handle is stale. This closes that port (releasing
        the USB device), waits for it to settle, opens a brand-new serial connection, and
        restarts the neck worker (facing forward). Runs off the UI thread; blocked mid-take."""
        if self._recording_active:
            self._log("[neck] finish (End Attempt) the current take before resetting the neck.")
            return
        ct = getattr(self, "_control_thread", None)
        if ct is None:
            self._log("[neck] boot the robot first — no control thread yet.")
            return
        self._log("[neck] HARD RESET — closing the serial port + rebuilding the neck from "
                  "scratch (same as a manager restart)…")
        threading.Thread(target=self._reconnect_neck_from_scratch, daemon=True).start()

    def _teardown_neck_iface(self, iface):
        """Release an old neck interface's serial port / USB device (best-effort), so a fresh
        connect can reopen it — mirrors the OS releasing the device when the manager restarts.
        Force-closes the underlying port even if the interface's _connected bookkeeping went
        stale after a fault (in which case shutdown() alone would no-op and leak the fd)."""
        if iface is None:
            return
        try:
            iface.shutdown()                       # torque off + closePort (when 'connected')
        except Exception:
            pass
        try:
            ph = getattr(getattr(iface, "controller", None), "_port_handler", None)
            if ph is not None:
                ph.closePort()
        except Exception:
            pass

    def _reconnect_neck_from_scratch(self):
        """Full HARD RESET of the neck subsystem, in-process (background thread). Stops the old
        worker + waits for its thread to exit, CLOSES the old serial port (releases the USB
        device), then builds a BRAND-NEW interface + serial connection and a fresh worker —
        exactly the clean slate a manager restart gives. Serial I/O here is safe: the old
        worker is fully stopped first and the new one is created only after the fresh connect,
        so two threads never touch the serial at once (see [[feedback_neck_threading]])."""
        ct = getattr(self, "_control_thread", None)
        old_worker = getattr(ct, "_neck_worker", None) if ct is not None else None
        old_iface = self._neck_interface
        # 1. Stop the old worker and WAIT for its thread to exit so it can't be mid-serial-op
        #    when we close the port (stop() only signals; it doesn't join).
        if old_worker is not None:
            try:
                old_worker.stop()
                t = getattr(old_worker, "_thread", None)
                if t is not None:
                    # Wait GENEROUSLY: the neck is usually rebooted BECAUSE it's faulted, so the
                    # old worker is often mid-servo-REBOOT — a multi-second serial op. If we
                    # closed the port / opened a NEW interface on the same /dev/ttyUSB while the
                    # old worker still held it, the two would clash at the C level → SEGFAULT
                    # (the crash you saw). Join long enough to outlast a reboot; the worker is
                    # now gated so it does no NEW serial ops once stopped.
                    t.join(timeout=6.0)
                    if t.is_alive():
                        # Still finishing a serial op — do NOT build a second interface on the
                        # same device (that's the segfault). Defer; it will exit shortly.
                        self._log("[neck] HARD RESET deferred — the old neck worker is still "
                                  "finishing a serial operation. Wait a few seconds and press "
                                  "Reboot Neck again.")
                        return
            except Exception:
                pass
        # 2-4. Close the old port, let the device settle, then open + home a BRAND-NEW interface —
        #      ALL under _NECK_SERIAL_LOCK so NO worker serial op (even one lingering past the join)
        #      can touch the /dev/ttyUSB while it's being closed and reopened. Two accesses to one
        #      USB serial device at the C level is the segfault; the lock makes it impossible. The
        #      old worker is already stopped+joined and the new one isn't created until after, so
        #      the lock is uncontended here — a hard guarantee, not a bottleneck. (send() on the
        #      old stopped worker during this window only sets _pending; it does no serial I/O.)
        iface = None
        try:
            with _NECK_SERIAL_LOCK:
                self._teardown_neck_iface(old_iface)
                self._neck_interface = None
                time.sleep(0.6)                    # let a dropped USB device re-enumerate
                from ostrich_neck.hardware import OstrichNeckHardwareInterface
                last_exc = None
                for _ in range(2):                 # one retry for a slow re-enumerate
                    try:
                        cand = OstrichNeckHardwareInterface()
                        cand.connect(move_home=True)
                        iface = cand
                        break
                    except Exception as exc:       # noqa: BLE001
                        last_exc = exc
                        time.sleep(0.8)
                if iface is None:
                    raise last_exc or RuntimeError("neck serial did not open")
                self._neck_interface = iface
            # 5. Fresh worker — created AFTER the lock is released so its own per-op serial locking
            #    behaves normally. Atomic replace on the ControlThread (no None window; the old
            #    stopped worker's .send() is a harmless no-op until this reassigns it).
            if ct is not None:
                ct._neck_worker = _NeckWorker(iface)
            self._log("[neck] HARD RESET complete — fresh serial link + worker; the neck should "
                      "track again (faced forward).")
        except Exception as exc:                   # noqa: BLE001
            self._log(f"[neck] HARD RESET FAILED: {exc} — check the USB cable / servo power, "
                      "then press Reboot Neck again.")

    def _on_show_force_calibration(self):
        """Open (or raise) the fingertip/palm force-sensor calibration window."""
        if self._calib_window is None:
            self._calib_window = ForceCalibrationWindow(
                hand_controller=self._hand_controller,
                control_thread=self._control_thread,
                hw_controller=self._hw_controller,
                log_fn=self._log,
                output_dir_getter=lambda: self._outdir_edit.text() or self._args.output_dir,
                apply_fn=self._apply_force_calibration,
            )
        self._calib_window.show()
        self._calib_window.raise_()
        self._calib_window.activateWindow()

    def _apply_force_calibration(self, path) -> bool:
        """Load a freshly-saved calibration into the running haptics streamer."""
        if self._haptics_streamer is None:
            return False
        return self._haptics_streamer.load_force_calibration(path)

    def _on_show_kneeling_calibration(self):
        """Open (or raise) the kneeling-pose trim window."""
        if self._kneeling_window is None:
            self._kneeling_window = KneelingPoseCalibrationWindow(
                calibration=self._kneeling_cal,
                base_kneeling_angles=_KNEELING_ANGLES,
                log_fn=self._log,
                apply_fn=self._apply_kneeling_calibration,
                enabled=bool(self._args.kneeling),
            )
        self._kneeling_window.show()
        self._kneeling_window.raise_()
        self._kneeling_window.activateWindow()

    def _apply_kneeling_calibration(self, cal):
        """Apply a kneeling-pose trim live to the control loop and sim viz."""
        self._kneeling_cal = cal
        self._kneeling_angles = cal.apply_to_legs(_KNEELING_ANGLES)
        self._kneeling_torso = (cal.torso_offset_vector()
                                if self._args.kneeling else np.zeros(3))
        if self._args.kneeling and self._controller is not None:
            self._controller.q_goal_left_leg  = self._kneeling_angles[:6].copy()
            self._controller.q_goal_right_leg = self._kneeling_angles[6:].copy()
        if self._args.kneeling and self._control_thread is not None:
            self._control_thread.set_kneeling_pose(
                self._kneeling_angles, self._kneeling_torso)

    def _on_toggle_palm_haptics(self):
        """Toggle the palm channel in the live haptics stream (source-side gate in the
        streamer — affects both vibrotactile and audio haptics; fingers unaffected)."""
        self._palm_haptics_on = not self._palm_haptics_on
        if self._haptics_streamer is not None:
            self._haptics_streamer.palm_enabled = self._palm_haptics_on
        self._btn_palm_haptics.setText(
            f"🖐  Palm Haptics: {'ON' if self._palm_haptics_on else 'OFF'}")
        self._btn_palm_haptics.setStyleSheet(
            "QPushButton { background: %s; color: white; padding: 8px; }"
            % ("#27ae60" if self._palm_haptics_on else "#7f8c8d"))
        self._cfg["palm_haptics"] = self._palm_haptics_on
        self._save_cfg()
        self._log(f"[haptics] palm channel {'ENABLED' if self._palm_haptics_on else 'MUTED'} "
                  "(fingertips unaffected).")

    def _on_show_abduction_limit(self):
        """Open (or raise) the max arm-abduction limit window."""
        if self._abduction_window is None:
            self._abduction_window = ArmAbductionLimitWindow(
                limit=self._abduction_limit,
                hw_controller=self._hw_controller,
                log_fn=self._log,
                apply_fn=self._apply_abduction_limit,
            )
        self._abduction_window.show()
        self._abduction_window.raise_()
        self._abduction_window.activateWindow()

    def _apply_abduction_limit(self, limit):
        """Apply an arm-abduction limit live to the control loop."""
        self._abduction_limit = limit
        if self._control_thread is not None:
            self._control_thread.set_abduction_limits(limit.left_max, limit.right_max)

    # ── Wrench (end-effector force) mode ──────────────────────────────────────

    def _wrench_meta_side(self) -> str:
        return self._wrench_side_combo.currentText().lower()  # both | left | right

    def _on_toggle_wrench_live(self):
        if self._wrench_worker is None:
            self._log("Wrench live readout unavailable (no hardware / estimator).")
            return
        on = not self._wrench_worker.is_enabled()
        self._wrench_worker.set_enabled(on)
        self._btn_wrench_live.setText(f"🔬  Live: {'ON' if on else 'OFF'}")
        self._btn_wrench_live.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; padding: 6px; }"
            if on else "QPushButton { padding: 6px; }")
        self._log(f"Live wrench readout {'enabled' if on else 'disabled'}.")

    def _on_wrench_tare(self):
        if self._wrench_worker is None:
            return
        self._wrench_worker.request_tare(n_samples=15)
        # Reflect the auto-enable in the Live button.
        self._btn_wrench_live.setText("🔬  Live: ON")
        self._btn_wrench_live.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; padding: 6px; }")
        self._log("Taring wrench baseline — hold the arm EMPTY and still…")
        # If a recording is running, fold the new baseline into the session meta
        # shortly after the samples have been collected.
        if self._recording_active:
            QTimer.singleShot(2000, lambda: self._session_recorder.update_meta(
                wrench_baseline=self._wrench_worker.baseline_dict()))

    def _on_wrench_clear_tare(self):
        if self._wrench_worker is not None:
            self._wrench_worker.clear_baseline()
            self._log("Wrench tare baseline cleared.")
            if self._recording_active:
                self._session_recorder.update_meta(
                    wrench_baseline=self._wrench_worker.baseline_dict())

    def _on_view_last_recording(self):
        """Open the MOST RECENT recording in the digital-twin viewer (table + crate +
        both robots + EE wrench + tactile fingertip/palm overlay + synced ZED video).
        The twin gracefully shows just this robot until the peer's data is merged in."""
        if not self._last_session_dir or not os.path.isdir(self._last_session_dir):
            self._log("No recording to view yet — record a session first.")
            return
        self._launch_digital_twin(self._last_session_dir)

    # ── Digital twin: send peer data to master + open the reconstruction ───────
    def _master_rsync_dest(self):
        """PEER's rsync destination on the master, resolved AUTOMATICALLY (no manual entry):
          • master username + recordings dir from the master's live status (sent nested
            under 'components'), falling back to the baked-in master_ssh_user / master_rec_dir
            defaults so a freshly git-pulled peer works before the master's first status;
          • master IP from the peer's Link-IP field / live socket, falling back to master_host.
        A non-blank ``master_rsync_dest`` in the config overrides all of it. Returns '' only if
        neither a host nor a recordings path can be determined."""
        override = (self._cfg.get("master_rsync_dest") or "").strip()
        if override:
            return override
        link = self._link
        st = (link.get_peer_status() if link is not None else None) or {}
        # The master ships its status snapshot nested under 'components' (see _peer_send_status)
        # — the ssh_user / rec_dir live THERE, not at the top level.
        comp = st.get("components") or {}
        user = (comp.get("ssh_user") or st.get("ssh_user")
                or self._cfg.get("master_ssh_user") or "").strip()
        rec_dir = (comp.get("rec_dir") or st.get("rec_dir")
                   or self._cfg.get("master_rec_dir") or "").strip()
        # Master IP: the address the peer dials (Link-IP field) or the live socket's peer IP,
        # else the configured fallback host. Live/configured Link-IP wins so a DHCP address
        # change is picked up automatically without editing the default.
        ip = (self._link_host()
              or (getattr(link, "remote_ip", None) if link is not None else None)
              or self._cfg.get("master_host") or "").strip()
        if not (ip and rec_dir):
            return ""
        return f"{(user + '@') if user else ''}{ip}:{rec_dir}"

    def _peer_rsync_src(self):
        """MASTER's rsync SOURCE on the PEER machine — where the master PULLS the peer's
        recordings from, over SSH, so it works even when the peer MANAGER isn't running (the
        peer MACHINE just needs to be on + reachable via sshd). Resolved from:
          • the live link's peer IP when the peer manager happens to be connected, else the
            configured/auto-remembered peer_host;
          • peer_ssh_user + peer_rec_dir from config (rec_dir defaults to this machine's own
            recordings path, symmetric deploy).
        A non-blank ``peer_rsync_src`` config overrides all of it. '' if no host/path."""
        override = (self._cfg.get("peer_rsync_src") or "").strip()
        if override:
            return override
        user = (self._cfg.get("peer_ssh_user") or "").strip()
        rec_dir = (self._cfg.get("peer_rec_dir") or self._output_dir_path() or "").strip()
        link = self._link
        ip = ((getattr(link, "remote_ip", None) if link is not None else None)
              or self._cfg.get("peer_host") or "").strip()
        if not (ip and rec_dir):
            return ""
        return f"{(user + '@') if user else ''}{ip}:{rec_dir}"

    @staticmethod
    def _parse_rsync_remote(dest):
        """Split an rsync dest into ('user@host', 'path') for the remote 'user@host:path'
        form, or (None, path) for a local path. (None, None) if empty."""
        if not dest:
            return None, None
        # Remote form: a colon that isn't a local absolute/relative path root.
        if ":" in dest and not dest.startswith(("/", ".", "~")):
            host, _, path = dest.partition(":")
            return host, path
        return None, dest

    def _check_master_ssh_dest(self, timeout=8.0, dest=None):
        """Verify THIS peer can deliver data to the master before a session: SSH to the
        rsync host and confirm the destination folder exists + is writable. Returns
        (state, message): state True=ok, False=blocked, None=not applicable (local dest /
        not a peer). Non-interactive SSH (BatchMode) so a missing key fails fast instead
        of hanging on a password prompt. Pass ``dest`` (resolved on the GUI thread) when
        calling from a worker thread — _master_rsync_dest() reads Qt widgets."""
        if dest is None:
            dest = self._master_rsync_dest()
        if not dest:
            return False, ("Master data dest not set — the peer can't send its data to "
                           "the master (set it in the setup bar).")
        host, path = self._parse_rsync_remote(dest)
        if host is None:
            # Local / mounted path — just check it's a writable dir (no SSH needed).
            ok = os.path.isdir(path) and os.access(path, os.W_OK)
            return (True if ok else None), (
                f"Master dest is a local path ({path}) — "
                + ("writable, no SSH needed." if ok else "not a writable dir yet; it will "
                   "be created on first send."))
        remote = (f"mkdir -p {shlex.quote(path)} && test -w {shlex.quote(path)} && echo OK")
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
               "-o", "StrictHostKeyChecking=accept-new", host, remote]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, (f"Master SSH timed out ({host}) — check the network / that "
                           "sshd is running on the master.")
        except FileNotFoundError:
            return False, "ssh client not found on this machine."
        except Exception as exc:                     # noqa: BLE001
            return False, f"Master SSH check error: {exc}"
        if r.returncode == 0 and "OK" in (r.stdout or ""):
            return True, f"master reachable + dest writable ({host}:{path})."
        tail_lines = (r.stderr or r.stdout or "").strip().splitlines()
        tail = tail_lines[-1] if tail_lines else f"ssh exit {r.returncode}"
        low = tail.lower()
        hint = ("  → set up a passwordless SSH key: run  ssh-copy-id "
                f"{host}  on this peer." if ("denied" in low or "publickey" in low
                                             or "password" in low) else "")
        return False, f"Master SSH failed ({host}): {tail}{hint}"

    def _master_link_gate_ok(self):
        """Peer-only pre-session gate: confirm the master's rsync dest is reachable so the
        auto-send at Stop will succeed. Returns True to proceed (blocking dialog on
        failure lets the operator fix it or record anyway). No-op on the master / unlinked."""
        if self._link_role() != "peer" or not self._linked():
            return True
        state, msg = self._check_master_ssh_dest()
        if state is True:
            self._log(f"[twin] master link OK — {msg}")
            return True
        if state is None:
            self._log(f"[twin] {msg}")
            return True
        self._log(f"[twin] PRE-FLIGHT: {msg}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Master link not reachable")
        box.setText("The peer can't reach the master's data destination:\n\n  • " + msg
                    + "\n\nRecording now is fine, but the auto-send to the master at Stop "
                      "will fail until this is fixed (you can still send it later with "
                      "'Send Data to Master').\n\nRecord anyway?")
        fix = box.addButton("Fix first", QMessageBox.RejectRole)
        rec = box.addButton("Record anyway", QMessageBox.AcceptRole)
        box.setDefaultButton(fix)
        box.exec_()
        return box.clickedButton() is rec

    def _on_test_master_link(self):
        """'Verify Master Link' button: re-run the auto SSH check now."""
        self._verify_master_link_async()

    def _verify_master_link_async(self):
        """Run the master-link SSH check OFF the GUI thread (it can block up to ~8 s) and
        show the result inline. Auto-run on link connect + on demand. Peer only."""
        if self._link_role() != "peer":
            return
        th = getattr(self, "_ssh_check_thread", None)
        if th is not None and th.is_alive():
            return                                   # a check is already running
        self._ssh_check_result = None
        self._send_status_lbl.setText("Checking master link…")
        self._send_status_lbl.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        # Resolve the destination HERE (GUI thread) — it reads Qt widgets — then hand the
        # plain string to the worker thread for the blocking SSH probe.
        dest = self._master_rsync_dest()

        def _work():
            try:
                self._ssh_check_result = self._check_master_ssh_dest(dest=dest)
            except Exception as exc:                 # noqa: BLE001
                self._ssh_check_result = (False, f"SSH check error: {exc}")
        self._ssh_check_thread = threading.Thread(target=_work, daemon=True)
        self._ssh_check_thread.start()
        QTimer.singleShot(200, self._poll_ssh_check)

    def _poll_ssh_check(self):
        """Poll the background SSH check; show the result once it finishes."""
        if getattr(self, "_ssh_check_result", None) is None:
            QTimer.singleShot(200, self._poll_ssh_check)
            return
        state, msg = self._ssh_check_result
        self._ssh_check_result = None
        if state is True:
            self._log(f"[twin] master link OK — {msg}")
            self._send_status_lbl.setText("Master link OK ✓")
            self._send_status_lbl.setStyleSheet("color: green; font-size: 11px;")
        elif state is None:
            self._log(f"[twin] {msg}")
            self._send_status_lbl.setText("Local dest (no SSH)")
            self._send_status_lbl.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        else:
            self._log(f"[twin] master link FAILED — {msg}")
            self._send_status_lbl.setText("Master link FAILED ✗ (see Log)")
            self._send_status_lbl.setStyleSheet("color: red; font-size: 11px;")

    def _on_send_data_to_master(self):
        """PEER: rsync THIS run's finished session onto the master's, merging into the same
        {session_id}/{cond}/attempt{K}/ folders (this robot's data under robot{N}/). To push
        an OLDER, previously-unsynced session instead, use 'Send Past Session…'."""
        if self._session_open():
            self._log("[twin] finish recording before sending — the session isn't "
                      "complete yet.")
            return
        src = self._last_session_root
        if not src or not os.path.isdir(src):
            self._log("No finished session from THIS run to send — record + stop one, or use "
                      "'Send Past Session…' to push an older session.")
            return
        self._start_master_send(src)

    def _on_send_past_session(self):
        """PEER: pick ANY past session folder (or the recordings ROOT to send ALL) and rsync
        it to the master — back-fills old, previously-unsynced recordings WITHOUT having to
        record a new session first."""
        if self._session_open():
            self._log("[twin] finish the current session before sending past data.")
            return
        start_dir = self._output_dir_path()
        if not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")
        picked = QFileDialog.getExistingDirectory(
            self, "Pick a past session folder to send to the master "
                  "(or the recordings root to send ALL sessions)", start_dir)
        if picked:
            self._start_master_send(picked)

    def _start_master_send(self, src):
        """Shared rsync-PUSH of a folder to the master's recordings over SSH, merging. If
        ``src`` is the recordings ROOT its CONTENTS merge (all sessions); a single session
        folder is copied INTO the master's recordings so it merges with the master's same-
        named session (robot leaves keep the two robots distinct). Background; BatchMode SSH
        so a missing key fails FAST instead of hanging on a password prompt."""
        if self._send_proc is not None and self._send_proc.poll() is None:
            self._log("A data transfer is already running — wait for it to finish.")
            return
        src_abs = os.path.abspath(src).rstrip("/")
        if not os.path.isdir(src_abs):
            self._log(f"[twin] not a folder: {src_abs}")
            return
        dest = self._master_rsync_dest()
        if not dest:
            self._log("[twin] can't send — the master link isn't up yet (the destination "
                      "is auto-identified from the link). Connect the link + wait for the "
                      "peer light to go green, then retry.")
            return
        rec_root = os.path.abspath(self._output_dir_path()).rstrip("/")
        src_arg = src_abs + "/" if src_abs == rec_root else src_abs   # root → merge contents
        dest = dest if dest.endswith("/") else dest + "/"
        host, _p = self._parse_rsync_remote(dest)
        cmd = ["rsync", "-az", "--partial"]
        if host is not None:                                          # remote → non-interactive ssh
            cmd += ["-e", "ssh -o BatchMode=yes -o ConnectTimeout=5 "
                          "-o StrictHostKeyChecking=accept-new"]
        cmd += [src_arg, dest]
        what = "all recordings" if src_arg.endswith("/") else Path(src_abs).name
        self._log(f"[twin] sending {what} → {dest} (merges into the master's matching "
                  "session folder[s]) …")
        self._send_status_lbl.setText(f"Sending {what} …")
        self._send_status_lbl.setStyleSheet("color: #16a085; font-size: 11px;")
        try:
            self._send_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except FileNotFoundError:
            self._log("rsync not found — install rsync (or transfer manually).")
            self._send_status_lbl.setText("rsync not found")
            return
        except Exception as exc:
            self._log(f"[twin] transfer failed to start: {exc}")
            return
        self._send_dest = dest
        QTimer.singleShot(400, self._poll_send)

    def _poll_send(self):
        """Poll the background rsync; log the outcome when it finishes."""
        proc = self._send_proc
        if proc is None:
            return
        if proc.poll() is None:
            QTimer.singleShot(400, self._poll_send)
            return
        out = ""
        try:
            out = (proc.stdout.read() or "").strip() if proc.stdout else ""
        except Exception:
            pass
        if proc.returncode == 0:
            self._log(f"[twin] data sent to master ({getattr(self, '_send_dest', '')}).")
            self._send_status_lbl.setText("Sent to master ✓")
            self._send_status_lbl.setStyleSheet("color: green; font-size: 11px;")
        else:
            tail = out.splitlines()[-1] if out else ""
            self._log(f"[twin] transfer FAILED (rsync exit {proc.returncode}). {tail}")
            self._send_status_lbl.setText(f"Transfer failed (exit {proc.returncode})")
            self._send_status_lbl.setStyleSheet("color: red; font-size: 11px;")
        self._send_proc = None

    def _on_pull_peer_data(self):
        """MASTER: rsync-PULL the PEER's recordings over SSH straight into this master's
        recordings, MERGING each shared session folder (the peer's robot{N}/ lands beside the
        master's robot1/). Works WITHOUT the peer manager running — only the peer MACHINE +
        sshd need to be up — so you can back-fill old, previously-unsynced data. Incremental
        (rsync skips already-synced files). Background; logs progress/result."""
        if self._session_open():
            self._log("[sync] finish the current session before syncing peer data.")
            return
        if (self._pull_proc is not None and self._pull_proc.poll() is None) or \
           (self._send_proc is not None and self._send_proc.poll() is None):
            self._log("[sync] a data transfer is already running — wait for it to finish.")
            return
        src = self._peer_rsync_src()
        if not src:
            self._log("[sync] peer source unknown — link to the peer once (its IP is then "
                      "remembered), or set peer_host + peer_rec_dir (or peer_rsync_src) in "
                      "~/.xrt_study_config.json.")
            return
        host, _path = self._parse_rsync_remote(src)
        src = src if src.endswith("/") else src + "/"          # contents → merge, not nest
        dest = self._output_dir_path()
        os.makedirs(dest, exist_ok=True)
        dest = dest if dest.endswith("/") else dest + "/"
        cmd = ["rsync", "-az", "--partial",
               "-e", "ssh -o BatchMode=yes -o ConnectTimeout=5 "
                     "-o StrictHostKeyChecking=accept-new",
               src, dest]
        self._log(f"[sync] pulling peer recordings {src} → local {dest} "
                  "(merges into the master's matching sessions; peer manager NOT required) …")
        self._send_status_lbl.setText("Pulling peer data …")
        self._send_status_lbl.setStyleSheet("color: #16a085; font-size: 11px;")
        try:
            self._pull_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except FileNotFoundError:
            self._log("[sync] rsync not found — install rsync.")
            return
        except Exception as exc:                           # noqa: BLE001
            self._log(f"[sync] pull failed to start: {exc}")
            return
        self._pull_host = host
        QTimer.singleShot(400, self._poll_pull)
        self._update_buttons()

    def _poll_pull(self):
        """Poll the background peer-pull rsync; log the outcome when it finishes."""
        proc = self._pull_proc
        if proc is None:
            return
        if proc.poll() is None:
            QTimer.singleShot(400, self._poll_pull)
            return
        out = ""
        try:
            out = (proc.stdout.read() or "").strip() if proc.stdout else ""
        except Exception:
            pass
        if proc.returncode == 0:
            self._log(f"[sync] peer data pulled into the master "
                      f"({getattr(self, '_pull_host', '') or 'peer'}). Merged sessions are "
                      "ready — open them in the Digital Twin.")
            self._send_status_lbl.setText("Peer data synced ✓")
            self._send_status_lbl.setStyleSheet("color: green; font-size: 11px;")
        else:
            tail = out.splitlines()[-1] if out else ""
            hint = ""
            if "denied" in out.lower() or "publickey" in out.lower() or "password" in out.lower():
                hint = (f"  → set up a passwordless key: ssh-copy-id "
                        f"{getattr(self, '_pull_host', '') or 'PEER'}")
            elif "no route" in out.lower() or "timed out" in out.lower() or "refused" in out.lower():
                hint = "  → check the peer machine is on + reachable (peer_host)."
            self._log(f"[sync] peer pull FAILED (rsync exit {proc.returncode}). {tail}{hint}")
            self._send_status_lbl.setText(f"Peer sync failed (exit {proc.returncode})")
            self._send_status_lbl.setStyleSheet("color: red; font-size: 11px;")
        self._pull_proc = None
        self._update_buttons()

    def _launch_digital_twin(self, path):
        """Launch digital_twin.py on a session folder / robot_raw CSV (ABSOLUTE path).
        Reconstructs the table + crate + BOTH robots (auto-finding the robot1/robot2 pair)
        in one MuJoCo scene with EE wrench arrows, the tactile fingertip/palm overlay, and
        synced ZED video. Runs in a subprocess with cwd=_HERE."""
        path = os.path.abspath(path)
        if not os.path.exists(path):
            self._log(f"[twin] recording not found: {path}")
            return
        script = _HERE / "digital_twin.py"
        if not script.exists():
            self._log("digital_twin.py not found next to the manager.")
            return
        try:
            subprocess.Popen([sys.executable, str(script), path], cwd=str(_HERE))
            self._log(f"[twin] launching digital twin for {Path(path).name} "
                      f"(MuJoCo + ZED video windows).")
        except Exception as exc:
            self._log(f"[twin] could not launch digital twin: {exc}")

    def _on_open_digital_twin(self):
        """Reconstruct table + crate + BOTH robots in one MuJoCo scene. Pick the TRIAL
        PARENT folder (holding the V / A+V / H+V / H+A+V conditions) to tab across every
        condition + attempt in one window — or a single attempt folder (either robot's)."""
        start_dir = (self._last_session_root or self._last_session_dir
                     or self._outdir_edit.text() or self._args.output_dir)
        start_dir = os.path.abspath(start_dir) if start_dir else os.getcwd()
        path = QFileDialog.getExistingDirectory(
            self, "Select the TRIAL PARENT folder (V / A+V / H+V / H+A+V) to tab across "
                  "— or a single attempt (either robot's; the pair is auto-found)",
            start_dir)
        if not path:
            return
        self._launch_digital_twin(path)

    def _on_build_dataset(self):
        """One click: turn a finished session into a training-ready HDF5 dataset. Picks a
        session folder, then runs build_dataset.py (localize ZED + align + resample +
        decode images) in the BACKGROUND, logging progress + the result."""
        if self._dataset_proc is not None and self._dataset_proc.poll() is None:
            self._log("[dataset] a build is already running — wait for it to finish.")
            return
        start_dir = (self._last_session_root or self._outdir_edit.text()
                     or self._args.output_dir)
        start_dir = os.path.abspath(start_dir) if start_dir else os.getcwd()
        session = QFileDialog.getExistingDirectory(
            self, "Select the (merged) session folder to build a dataset from", start_dir)
        if not session:
            return
        script = _HERE / "build_dataset.py"
        if not script.exists():
            self._log("build_dataset.py not found next to the manager.")
            return
        out = os.path.join(os.path.abspath(session), "dataset.hdf5")
        cmd = [sys.executable, str(script), os.path.abspath(session), "-o", out,
               "--localize-zed", "--images", "--only-good", "--rate", "15"]
        self._log(f"[dataset] building {Path(out).name}: localizing ZED + aligning both "
                  "robots + resampling to 15 Hz + decoding images. Background job — watch "
                  "the Log (can take a while for image decode).")
        self._dataset_status_lbl.setText("Building dataset…")
        self._dataset_status_lbl.setStyleSheet("color: #2980b9; font-size: 11px;")
        try:
            self._dataset_proc = subprocess.Popen(
                cmd, cwd=str(_HERE), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
        except Exception as exc:                     # noqa: BLE001
            self._log(f"[dataset] failed to start: {exc}")
            self._dataset_status_lbl.setText("Build failed to start")
            return
        self._dataset_out = out
        QTimer.singleShot(700, self._poll_build_dataset)

    def _poll_build_dataset(self):
        """Poll the background dataset build; log the outcome + open the folder when done."""
        proc = self._dataset_proc
        if proc is None:
            return
        if proc.poll() is None:
            QTimer.singleShot(700, self._poll_build_dataset)
            return
        out = ""
        try:
            out = (proc.stdout.read() or "").strip() if proc.stdout else ""
        except Exception:
            pass
        if proc.returncode == 0:
            tail = "\n  ".join(out.splitlines()[-4:]) if out else ""
            self._log(f"[dataset] done → {self._dataset_out}\n  {tail}")
            self._dataset_status_lbl.setText("Dataset built ✓")
            self._dataset_status_lbl.setStyleSheet("color: green; font-size: 11px;")
            self._open_in_file_manager(os.path.dirname(self._dataset_out))
        else:
            tail = out.splitlines()[-1] if out else ""
            self._log(f"[dataset] build FAILED (exit {proc.returncode}). {tail}")
            self._dataset_status_lbl.setText(f"Build failed (exit {proc.returncode})")
            self._dataset_status_lbl.setStyleSheet("color: red; font-size: 11px;")
        self._dataset_proc = None

    def _on_stop_teleop_clicked(self):
        """Stop & Lower button: when linked, stop BOTH robots (broadcast + local); unlinked,
        just this robot. Separate from _on_stop_teleop so the REMOTE stop_teleop handler
        (which calls _on_stop_teleop directly) never echoes the broadcast back."""
        if self._linked():
            self._peer_cmd("stop_teleop")
        self._on_stop_teleop()

    def _on_stop_teleop(self):
        if self._state not in (_S_WAITING, _S_TELEOP):
            return
        # Cancel any pending "engage once connected" arm so Stop fully stops.
        self._engage_when_connected = False
        # A full Stop lowers the arms, so the next start is a fresh staged entry.
        self._teleop_paused = False
        self._pause_reason = None
        self._log("Stopping teleop — arms damping (slow lower)...")

        # Pause IK and put the arms into damping mode: kp=0/kd>0 so they
        # slowly lower under gravity without pushing on any object, while the
        # torso and legs stay rigid. The ControlThread keeps streaming at
        # ~500 Hz so the firmware watchdog stays satisfied.
        if self._control_thread is not None:
            self._control_thread.enter_arm_damping()

        # Flip the XR app's Play/Pause button back to Play.
        self._send_unity_cmd("teleop_active", {"enabled": False})

        self._state = _S_IDLE
        self._log("Arms in damp mode. Teleop stopped.")
        self._update_buttons()

    # ── Robot identity ────────────────────────────────────────────────────────

    _CFG_DEFAULTS = {
        "robot_id": 1, "headcam_host": "192.168.123.164", "headcam_port": 5005,
        "include_headcam": True, "link_role": "none", "link_host": "",
        # Arm heat/compliance modes persist across reboots (toggleable live).
        "eco_mode": False, "arm_compliance": False,
        # Hand actuator DOFs to STOP commanding on THIS robot (per-machine hardware quirk, e.g. a
        # sheared thumb-rotation shaft): freezing a broken DOF keeps the recording + digital twin
        # matching the real hand. Lists of actuator indices per hand — [pinky, ring, middle, index,
        # thumb-bend, thumb-rotation]. Empty = every DOF enabled (default).
        "disabled_hand_dofs": {"left": [], "right": []},
        # rsync destination on the MASTER for "Send Data to Master" (peer side).
        # Normally left blank and auto-derived: IP from the peer's Link-IP / live socket,
        # user + recordings dir from the master's live status. Set master_rsync_dest to a
        # full 'user@host:/path' only to FORCE a fixed destination (overrides everything).
        "master_rsync_dest": "",
        # Baked-in fallbacks describing THIS study's master machine (MEL07649L), used by
        # the peer when the master's live status hasn't arrived yet — so a freshly git-
        # pulled peer resolves the destination with no manual setup. The live Link-IP /
        # socket IP still WINS over master_host when the link is up (survives DHCP changes).
        "master_ssh_user": "agraves36",
        "master_rec_dir": str(_REPO_ROOT / "xr_recordings"),
        "master_host": "192.168.7.226",
        # Symmetric fallbacks describing the PEER machine, used by the MASTER's "Sync Peer
        # Data" button to rsync-PULL the peer's recordings over SSH — so old unsynced data
        # can be pulled WITHOUT the peer manager running. peer_host is auto-remembered from
        # the live link whenever the peer manager connects; set it here for pure-offline
        # pulls. peer_rec_dir defaults to this machine's own recordings path (symmetric
        # deploy). peer_rsync_src ('user@host:/path') overrides all of it.
        "peer_ssh_user": "agraves36",
        "peer_rec_dir": str(_REPO_ROOT / "xr_recordings"),
        "peer_host": "",
        "peer_rsync_src": "",
        # Gate Start Recording on every camera being anchored to table tag 0 (this
        # robot's head cam + the external crate cameras), so the digital-twin data
        # shares one table frame. Uncheck for bench testing without cameras.
        "require_calibration": True,
        # Peer auto-contributes its head-cam crate observations to the master (its head
        # cam becomes the 4th crate camera) with NO manual "Contribute Crate Tracking"
        # click — engaged automatically once the head-cam localizer is up and the crate
        # bundle has arrived. Master-side this is ignored (contribution is a peer→master
        # feed). Set False to go back to manual.
        "crate_contribute_default": True,
        # UI mode: True = User Study Mode (default, no Good/Bad attempt marking), False =
        # Data Collection Mode (exposes the Mark Good / Mark Bad demo-quality buttons).
        "study_mode": True,
        # Include the palm touch sensor in the live haptics stream (vibro + audio).
        "palm_haptics": True,
        # Live speech transcription of the room (conference mic → whisper → per-session
        # transcript CSV; no audio saved). Runs only where a USB mic is found.
        "transcribe_enabled": True,
        # faster-whisper model id for the room transcription. Finished utterances buffer in
        # the transcriber and are processed in order (see the backlog readout).
        #   large-v3-turbo (near-large accuracy, ~8x faster — DEFAULT, chosen to keep host
        #     load low so it doesn't starve the WebRTC teleop loop)  •  large-v3 (most
        #     accurate, heavier)  •  distil-large-v3.5 (fast, robust)  •  medium.en /
        #     small.en (lighter, English-only)
        # If CUDA is unavailable it falls back to CPU int8 — use a small model there.
        "transcribe_model": "large-v3-turbo",
        # Speaker-ID thresholds (see study_transcriber._identify). A finished utterance is
        # labeled with the nearest enrolled operator only if its cosine similarity clears
        # transcribe_min_sim (absolute floor) AND beats the other operator by
        # transcribe_min_margin (relative separation). Far-field / short-utterance
        # embeddings sit ~0.4-0.6, so the OLD single 0.70 threshold rejected everyone; lower
        # the floor and/or raise the margin to trade coverage vs mis-labels. Watch the
        # "[speaker 0.NN]" scores in the transcript log to tune these to your room.
        "transcribe_min_sim": 0.45,
        "transcribe_min_margin": 0.06,
        # Microphone mode. False = 1 conference mic → voice-ID (needs Calibrate Voices).
        # True = 2-mic STEREO: a dual-channel device (e.g. Hollyland Lark M1 in stereo mode)
        # where LEFT = operator1, RIGHT = operator2; each channel is transcribed + attributed
        # BY CHANNEL, so overlap and speaker-ID are non-issues (no calibration needed).
        "transcribe_stereo": False,
    }

    def _load_cfg(self, args):
        """Load per-machine config: defaults < persisted file (new, or migrated
        from the old robot-id file) < any explicitly-passed CLI flags."""
        cfg = dict(self._CFG_DEFAULTS)
        # new combined file
        try:
            with open(self._cfg_path) as f:
                cfg.update({k: v for k, v in json.load(f).items() if k in cfg})
        except Exception:
            # migrate the old robot-id-only file if present
            try:
                with open(os.path.expanduser("~/.xrt_study_robot_id.json")) as f:
                    rid = int(json.load(f).get("robot_id"))
                    if rid in (1, 2):
                        cfg["robot_id"] = rid
            except Exception:
                pass
        # One-time model migration: large-v3-turbo replaced large-v3 as the default (near-
        # large accuracy at ~8x less host load, so the transcriber can't starve the WebRTC
        # teleop loop). Bump a persisted value still on the OLD default; an intentionally
        # different choice (distil / medium.en / …) is left untouched. An explicit
        # --transcribe_model below still wins.
        if cfg.get("transcribe_model") == "large-v3":
            cfg["transcribe_model"] = "large-v3-turbo"
        # CLI overrides (only when explicitly passed)
        if getattr(args, "robot_id", None) in (1, 2):
            cfg["robot_id"] = int(args.robot_id)
        for k in ("headcam_host", "headcam_port", "link_role", "link_host",
                  "transcribe_model", "transcribe_min_sim", "transcribe_min_margin"):
            v = getattr(args, k, None)
            if v is not None:
                cfg[k] = v
        if getattr(args, "apriltag_include_headcam", False):
            cfg["include_headcam"] = True
        # Eco / compliance use BooleanOptionalAction with default=None: None means
        # neither --flag nor --no-flag was passed → keep the persisted value; a real
        # bool means it was explicitly forced ON (--flag) or OFF (--no-flag) at boot.
        for k in ("eco_mode", "arm_compliance", "transcribe_stereo"):
            v = getattr(args, k, None)
            if v is not None:
                cfg[k] = bool(v)
        if cfg["robot_id"] not in (1, 2):
            cfg["robot_id"] = 1
        return cfg

    def _save_cfg(self):
        try:
            with open(self._cfg_path, "w") as f:
                json.dump(self._cfg, f, indent=2)
        except Exception as exc:
            self._log(f"Could not persist config: {exc}")

    # -- UI-backed getters (the UI widgets are the source of truth at run time) --
    def _headcam_host(self):
        # Fixed per robot (UI box removed): the config value — default 192.168.123.164,
        # still overridable via --headcam_host / the per-machine config file.
        return self._cfg["headcam_host"]

    def _headcam_port(self):
        return int(self._cfg.get("headcam_port", 5005))

    def _include_headcam(self):
        # ALWAYS fuse the head cam as a crate camera (checkbox removed by design).
        # Hard True — not the config — so a stale persisted False can't disable it.
        return True

    def _link_role(self):
        return (self._link_role_combo.currentText().lower()
                if hasattr(self, "_link_role_combo") else self._cfg["link_role"])

    def _link_host(self):
        return (self._link_host_edit.text().strip()
                if hasattr(self, "_link_host_edit") else self._cfg["link_host"])

    def _on_setup_changed(self, *_):
        """Persist any Setup-bar change to the per-machine config."""
        self._cfg.update(
            headcam_host=self._headcam_host(), include_headcam=self._include_headcam(),
            link_role=self._link_role(), link_host=self._link_host(),
            require_calibration=(self._require_calib_chk.isChecked()
                                 if hasattr(self, "_require_calib_chk")
                                 else self._cfg.get("require_calibration", True)))
        self._save_cfg()
        self._apply_role_visibility()   # role may have changed → show/hide role-specific UI

    def _apply_role_visibility(self):
        """Show/hide role-specific UI. The external RealSense crate cameras are connected
        to the MASTER only, so a PEER manager hides everything that drives them — the whole
        Crate Tracking section (start/anchor/re-anchor/recalibrate + status) and the
        head-cam→crate fuse option. Conversely 'Send Data to Master' (+ its destination
        field) is the peer's job — the master is the destination — so those show only on
        the peer. Called at startup and whenever the Link role changes."""
        if not hasattr(self, "_send_status_lbl"):
            return                                   # UI not fully built yet
        is_peer = (self._link_role() == "peer")
        # Master/standalone-ONLY controls — hidden on the peer because they do nothing (or
        # the wrong thing) there:
        #   * Crate tracking/calibration — external RealSense crate cameras are on the master.
        #   * Prepare Study — the MASTER orchestrates it (it pushes the bundle + drives the
        #     peer's prep via the link); a peer button would run backwards.
        #   * Build Dataset — the master merges BOTH robots' data; the peer only ships its
        #     own up ('Send Data to Master').
        #   * View Last / Open Digital Twin — the twin is the master-side review of the
        #     merged both-robot data; the peer sends its data up instead of viewing here.
        # (The head-cam 'Calibrate Cameras Now' stays on the peer — it anchors the peer's
        # OWN head-cam base, which the peer needs.)
        master_only = [self._crate_hdr, self._btn_crate_track, self._btn_anchor_cameras,
                       self._btn_reanchor_bump, self._btn_recalib_crate, self._rec_crate_lbl]
        for _name in ("_btn_prepare_study", "_btn_build_dataset",
                      "_btn_view_recording", "_btn_open_twin", "_btn_pull_peer",
                      # Room transcription + conference mic — one shared mic, transcribed on the
                      # master only; the peer never runs them (the mode-specific Calibrate Voices
                      # / stereo-hint rows are governed by _refresh_mic_mode_ui below).
                      "_btn_transcribe", "_btn_mic_mode", "_btn_test_mic", "_btn_open_transcript"):
            _w = getattr(self, _name, None)
            if _w is not None:
                master_only.append(_w)
        for w in master_only:
            w.setVisible(not is_peer)
        # Switched to peer with a transcriber already running → tear it down (frees the mic).
        if is_peer and self._transcriber is not None:
            self._shutdown_transcriber()
            self._log("[transcribe] stopped — transcription/mic are master-only (peer role).")
        # Refresh the mic-mode rows (Calibrate Voices / stereo hint), which also honour the
        # peer gate, so they hide/show correctly after a role change.
        if hasattr(self, "_btn_mic_mode"):
            self._refresh_mic_mode_ui()
        # Sending this robot's data up to the master — peer only.
        for w in (self._master_dest_lbl, self._btn_test_dest, self._btn_send_master,
                  self._btn_send_past, self._send_status_lbl):
            w.setVisible(is_peer)
        # Crate contribution (head-cam viewpoint fed to the master's fusion) — peer only.
        for w in (self._contrib_hdr, self._btn_crate_contribute, self._btn_contrib_anchor,
                  self._rec_contrib_lbl):
            w.setVisible(is_peer)
        # Operator codes are ASSIGNED on the master (synced to the peer at Start Session), so
        # they're read-only on the peer.
        if hasattr(self, "_master_code_edit"):
            self._master_code_edit.setReadOnly(is_peer)
            self._peer_code_edit.setReadOnly(is_peer)

    def _apply_mode_visibility(self):
        """User Study Mode hides the Good/Bad attempt-quality marking (buttons + label);
        Data Collection Mode shows it. Also refreshes the mode button's own label. Called at
        startup and whenever the mode toggles."""
        if not hasattr(self, "_btn_mode"):
            return
        study = self._study_mode
        self._btn_mode.setText(
            "🧪  Mode: User Study" if study else "🗃  Mode: Data Collection")
        self._btn_mode.setStyleSheet(
            "QPushButton { background: %s; color: white; font-weight: bold; padding: 6px; }"
            % ("#16a085" if study else "#8e44ad"))
        show_quality = not study
        for w in (self._btn_mark_good, self._btn_mark_bad, self._attempt_quality_lbl):
            if w is not None:
                w.setVisible(show_quality)

    def _on_toggle_mode(self):
        self._study_mode = not self._study_mode
        self._cfg["study_mode"] = self._study_mode
        self._save_cfg()
        self._apply_mode_visibility()
        self._log("Mode: " + ("User Study (Good/Bad marking hidden)." if self._study_mode
                               else "Data Collection (Good/Bad marking available)."))
        self._update_buttons()

    def _on_robot_id_changed(self, idx: int):
        if self._session_open():   # guarded by the disabled combo, belt-and-suspenders
            return
        self._robot_id = idx + 1
        self._cfg["robot_id"] = self._robot_id
        self._save_cfg()
        self.setWindowTitle(
            f"G1 Inspire Hand — User Study Manager  (Robot {self._robot_id})")
        self._log(f"Robot identity set to {self._robot_id}.")

    # ── Recording (continuous; per-trial / per-attempt subfolder rotation) ─────
    # Layout: {output_dir}/{pid}_robot{N}_{date}_{time}/{condition}/attempt{K}/…
    # The streaming pipeline stays up across trials/attempts — switching just
    # finalizes the current files and reopens them in the next subfolder.

    def _output_dir_path(self):
        return self._outdir_edit.text() or self._args.output_dir

    def _new_session_id(self, pid):
        """A fresh shared session id: {pid}_{date}_{time}. NO robot{N} — the two robots
        share ONE session folder and are told apart by a robot{N}/ leaf per attempt, so
        the peer's tree merges cleanly into the master's."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{pid}_{ts}"

    def _session_root_for(self, session_id):
        """The (shared) session folder: {output_dir}/{session_id}. Identical on both
        robots, so the peer's copy lands right on top of the master's."""
        return os.path.abspath(os.path.join(self._output_dir_path(), session_id))

    def _current_rec_dir(self):
        """Current attempt's folder {session_root}/{condition}/attempt{K}/robot{N}, from
        the dropdown + spinbox. The robot{N} leaf lets BOTH robots' data live in the same
        attempt folder once the peer's data is sent over. Requires _session_root (set at
        Start)."""
        cond = self._cond_combo.currentText()
        attempt = self._attempt_spin.value()
        return os.path.join(self._session_root, cond.replace("/", "_"),
                            f"attempt{attempt}", f"robot{self._robot_id}")

    def _session_open(self):
        """True while a participant session is OPEN — from the first Start Recording
        until Stop Recording — including the ARMED gaps between takes (recording ended,
        pointer advanced, not yet recording the next). ``_recording_active`` is the
        narrower 'a take is actively being written' flag."""
        return self._session_root is not None

    def _begin_recording_into(self, rec_dir, pid, cond, attempt):
        """Start ALL file recorders into rec_dir (body pose, robot CSV + tactile + meta,
        summary, ZED recording, crate, base). Keeps ZED STREAMING up (starts it only if
        not already running). Used by both the initial Start and each rotate."""
        os.makedirs(rec_dir, exist_ok=True)
        # A take can never overlap a practice round (they share the transcriber's take label
        # + recording flag; practice would otherwise tag this take's rows 'practice').
        if self._practice_active:
            self._local_practice_stop()
        self._last_session_dir = rec_dir
        self._attempt_quality = "unrated"            # each new take starts unrated
        self._update_attempt_quality_label()
        self._session_label = f"{pid}_{cond.replace('/', '_')}_attempt{attempt}"
        # Auto-set the Unity feedback modalities for THIS trial condition (updates the
        # toggle buttons + sends the commands) BEFORE the feedback-event log snapshots the
        # state, so each take opens with — and records — the correct H/A/V feedback.
        self._apply_feedback_for_condition(cond)

        # Operator XR body pose CSV (the device writes it + csv_start_time.txt).
        if self._device is not None:
            self._device.configure_recording(
                output_dir=rec_dir, record_data=True, started=False)

        mass_g = self._wrench_mass_spin.value()
        wrench_baseline = (self._wrench_worker.baseline_dict()
                           if self._wrench_worker is not None else {})
        meta = {
            "model_path": _XML.as_posix(),
            "session_label": self._session_label,
            "robot_id": self._robot_id,
            "robots_total": 2,
            "participant_id": pid,               # = pair id (back-compat)
            "pair_id": pid,
            # This robot's operator (master robot → master code, peer robot → peer code),
            # plus the full pair mapping so any single robot's meta is self-contained.
            "operator_code": self._this_operator_code(),
            "master_operator_code": self._rec_master_code,
            "peer_operator_code": self._rec_peer_code,
            "trial_condition": cond,
            "attempt": attempt,
            "kneeling": bool(self._args.kneeling),
            "ik_rate": self._args.ik_rate,
            "left_offset": we.LEFT_EE_OFFSET.tolist(),
            "right_offset": we.RIGHT_EE_OFFSET.tolist(),
            "wrench_side": self._wrench_meta_side(),
            "known_mass_g": mass_g if mass_g > 0 else None,
            "wrench_baseline": wrench_baseline,
        }
        record_tactile = not getattr(self._args, "no_tactile", False)
        robot_csv_path = self._session_recorder.start(
            rec_dir, self._session_label, meta=meta, record_tactile=record_tactile)
        self._log(f"Recording {cond} / attempt{attempt} → {Path(robot_csv_path).name}"
                  + ("  (+ tactile .bin)" if self._session_recorder._tactile_width else ""))

        # ── Twin-completeness guard ────────────────────────────────────────────
        # robot_raw / tactile / ZED only capture when this robot's hardware is up.
        # If a robot records with no booted motors, no hand, or no ZED, its files
        # come out header-only / empty — and the digital twin will silently show
        # only the OTHER robot (exactly how a peer ends up "missing" from the twin).
        # Warn LOUDLY here so it's caught before the attempt, not at playback.
        gaps = []
        if self._session_recorder._hw is None:
            gaps.append("robot NOT booted → robot_raw will record 0 frames")
        if self._hand_controller is None:
            gaps.append("no hand controller → tactile/finger data will not record")
        if not self._zed_recorder.is_streaming:
            gaps.append("ZED not streaming yet → no video unless it starts below")
        if gaps:
            self._log("⚠⚠ TWIN DATA INCOMPLETE for "
                      f"robot{self._robot_id}: " + "; ".join(gaps) + ".")
            self._log("⚠⚠ This robot will be ABSENT from the merged digital twin. "
                      "Boot the robot / hand / ZED before recording if you want it "
                      "to appear alongside the other robot.")

        self._open_summary_log(rec_dir, self._session_label)
        self._open_feedback_log(rec_dir, self._session_label)
        if self._transcriber is not None:
            self._transcriber.set_take(self._session_label)
            # Attempt is recording → START saving transcript (only speech from here until the
            # attempt ends is written to the session transcript).
            self._transcriber.set_recording(True)

        self._zed_recorder._output_dir = rec_dir
        if not self._zed_recorder.is_streaming:
            self._log("Starting ZED camera streaming...")
            ok = self._zed_recorder.start_streaming()
            if ok:
                self._log("ZED streaming started. Waiting 3 s for pipeline to stabilize...")
                QTimer.singleShot(3000, self._start_zed_recording)
            else:
                self._log("ZED streaming failed to start (check mediamtx / GStreamer).")
        else:
            self._start_zed_recording()

        # Crate AprilTag pose — separate CSVs in the SAME folder, time.time() stamped.
        if self._apriltag_tracker is not None:
            if not self._apriltag_tracker.is_anchored:
                self._log("[crate] WARNING: cameras not anchored — click 'Anchor "
                          "Cameras' (table tag visible) or crate pose won't record.")
            try:
                cp, cr = self._apriltag_tracker.start_recording(rec_dir, self._session_label)
                self._session_recorder.update_meta(**self._apriltag_tracker.config_dict())
                self._log(f"[crate] recording: {Path(cp).name}, {Path(cr).name}")
            except Exception as exc:
                self._log(f"[crate] failed to start recording: {exc}")

        # Head-cam base self-localization — same folder, synced timestamps.
        if self._headcam_localizer is not None:
            try:
                bp = self._headcam_localizer.start_recording(rec_dir, self._session_label)
                self._session_recorder.update_meta(**self._headcam_localizer.config_dict())
                self._log(f"[base] recording: {Path(bp).name}")
            except Exception as exc:
                self._log(f"[base] failed to start recording: {exc}")

        # Stamp the last-known clock-sync quality + robot id so the dataset builder can
        # time-align the two robots (a fresh measurement is scheduled just after Start).
        # Plus the ZED intrinsics for the localization side (peer=right) so zed_localize
        # needs no manual --fx/... Write BOTH cams' intrinsics + which side to use.
        zed_side = self._zed_localize_side()
        zed_left = self._zed_intrinsics("left")
        zed_right = self._zed_intrinsics("right")
        chosen = zed_right if zed_side == "right" else zed_left
        self._zed_intrinsics_meta = chosen
        if chosen:
            self._log(f"[zed] {zed_side.upper()}-cam intrinsics {chosen['resolution']} "
                      f"fx={chosen['fx']:.1f} cx={chosen['cx']:.1f} (from {chosen['conf']}) "
                      "→ meta; zed_localize will use this camera.")
        else:
            self._log("[zed] no ZED intrinsics found (SN<serial>.conf) — set "
                      "ZED_SETTINGS_PATH or --zed-conf, or pass --fx/... to zed_localize.")
        try:
            self._session_recorder.update_meta(
                robot_id=self._robot_id, robots_total=2,
                clock_sync=self._clock_sync_info,
                attempt_quality=self._attempt_quality,
                zed_localize_side=zed_side,
                zed_intrinsics=(chosen or None),
                zed_left_intrinsics=(zed_left or None),
                zed_right_intrinsics=(zed_right or None))
        except Exception:
            pass

    # -- synced dispatch: run on BOTH machines at a shared epoch when linked --
    def _linked(self):
        return self._link is not None and self._link.connected

    def _sync_or_local(self, action, payload, local_call):
        """When linked, broadcast `action`+payload to the peer AND schedule the local
        action at a shared epoch, so BOTH managers act together; else run local now."""
        if self._linked():
            T = time.time() + float(self._args.link_sync_delay)
            self._peer_cmd(action, at_epoch=T, **payload)
            self._schedule_at(T, local_call)
            self._log(f"[link] {action} on both at epoch {T:.3f}.")
        else:
            local_call()

    def _set_selectors(self, pid, condition, attempt, master_code=None, peer_code=None):
        """Set the pair-id / operator-codes / condition / attempt UI to match a (possibly
        remote) command WITHOUT re-triggering a rotate (guarded by _suppress_rotate)."""
        self._suppress_rotate = True
        try:
            if pid is not None:
                self._pid_edit.setText(str(pid))
            if master_code is not None:
                self._master_code_edit.setText(str(master_code))
                self._rec_master_code = str(master_code)
            if peer_code is not None:
                self._peer_code_edit.setText(str(peer_code))
                self._rec_peer_code = str(peer_code)
            if condition is not None:
                i = self._cond_combo.findText(str(condition))
                if i >= 0:
                    self._cond_combo.setCurrentIndex(i)
            if attempt is not None:
                self._attempt_spin.setValue(int(attempt))
        finally:
            self._suppress_rotate = False

    def _this_operator_code(self) -> str:
        """The 16-digit code of the operator on THIS robot: the peer code on the peer, the
        master code on the master/standalone."""
        return self._rec_peer_code if self._link_role() == "peer" else self._rec_master_code

    # ── Pre-flight: every camera anchored to table tag 0 before recording ──────
    def _ensure_headcam_localized(self, wait_s=10.0):
        """This robot's head camera must currently see the table tag (id 0) so its base
        pose is recorded in the table frame (the digital twin places the robot by it).
        Auto-starts the localizer if it's off. Returns (ok, message). wait_s covers the
        DETACHED detection subprocess's cold start (~3-5 s of imports before the first
        detection can even arrive) plus a few detect cycles — 4 s was reporting 'NOT
        seeing table tag 0' on a fresh start while the subprocess was still booting."""
        if self._headcam_localizer is None:
            self._init_headcam_localizer()       # needs the Jetson head-cam server + IP
        if self._headcam_localizer is None:
            return False, ("Head cam: OFF — base pose won't be recorded (start "
                           "Head-Cam Localize / set the Jetson IP)")
        deadline = time.time() + wait_s
        res = self._headcam_localizer.latest()
        while not (res and res.get("ok")) and time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.1)
            res = self._headcam_localizer.latest()
        if not (res and res.get("ok")):
            return False, "Head cam: NOT seeing table tag 0 — aim the head at the tag"
        up = float(res.get("upright", 0.0))
        if up < 0.8:
            return False, (f"Head cam: tag 0 seen but base pose is weak (up={up:+.2f}) "
                           "— check aim / optical convention")
        return True, (f"Head cam: locked on table tag 0 "
                      f"(up={up:+.2f}, z={float(res.get('base_z', 0.0)):.2f}m)")

    def _wait_for_crate_cameras_ready(self, timeout=6.0, fresh_window=1.5):
        """Before anchoring, wait until EVERY (local) crate camera is streaming FRESH frames.
        The external RealSense cams warm up in a few hundred ms, but the G1 head cam is a
        NETWORK camera (TCP connect + intrinsics handshake + detector init) that lags — so an
        anchor fired ~1 s after a COLD tracker start misses it ("2/3 … NO TABLE TAG"), while a
        warm tracker gets all of them (the exact 2/3-then-4/4 flip seen in the log). Frames are
        time.time()-stamped in the tracker process on THIS machine, so a freshness check is
        valid. Returns as soon as all local cams have a recent frame, or at timeout (then
        anchors anyway). No-op once warm (first poll passes). Peer contribution cameras
        (serial 'peer:…') are self-anchored and excluded."""
        tr = self._apriltag_tracker
        if tr is None:
            return

        def _fresh(c):
            snap = c.latest()
            return (getattr(c, "started_ok", False) and snap is not None
                    and (time.time() - float(snap.get("stamp", 0.0))) < fresh_window)

        deadline = time.time() + timeout
        while True:
            cams = [c for c in tr.cams if not c.serial.startswith("peer:")]
            if cams and all(_fresh(c) for c in cams):
                return
            if time.time() >= deadline:
                cold = [f"{c.serial}"
                        + (f" [{getattr(c, 'error', None)}]" if getattr(c, "error", None)
                           else " [connected, no frames]" if getattr(c, "started_ok", False)
                           else " [never connected — is the head-cam server single-client / "
                                "stale on the Jetson? redeploy g1_headcam_server.py]")
                        for c in cams if not _fresh(c)] or ["<no cameras yet>"]
                self._log(f"[calib] camera warm-up wait timed out ({timeout:g}s) — anchoring "
                          f"anyway; not yet streaming: {', '.join(cold)}.")
                return
            QApplication.processEvents()
            time.sleep(0.1)

    def _ensure_crate_anchored(self, force=False):
        """The external RealSense crate cameras must be anchored to table tag 0. Anchors
        now (one-time ~1.5 s capture) if not yet anchored, or always when ``force`` (the
        manual Calibrate button, e.g. after a camera was bumped). Returns (state, message)
        where state is True/False, or None if there's no tracker on this machine."""
        tr = self._apriltag_tracker
        if tr is None:
            return None, "Crate cams: no tracker on this machine"
        if force or not tr.is_anchored:
            # The network head cam is slow to warm up on a fresh tracker start; wait for all
            # local cameras to be streaming so the anchor doesn't miss one (cold-start race).
            self._wait_for_crate_cameras_ready()
            self._log("[calib] anchoring crate cameras to table tag 0…")
            QApplication.processEvents()
            try:
                results = tr.anchor_cameras()
            except Exception as exc:                       # noqa: BLE001
                return False, f"Crate cams: anchor failed ({exc})"
            bad = [s for s, (st, _) in results.items() if st != "ok"]
            ok = [s for s, (st, _) in results.items() if st == "ok"]
            if bad:
                return False, (f"Crate cams: {len(ok)}/{len(results)} anchored — "
                               f"{len(bad)} camera(s) can't see table tag 0")
        nanch = sum(1 for v in tr.T_world_cam.values() if v is not None)
        ntot = len(tr.T_world_cam)
        if nanch < ntot:
            return False, (f"Crate cams: only {nanch}/{ntot} anchored — reposition so "
                           "tag 0 is visible to ALL, then Anchor Cameras")
        return True, f"Crate cams: anchored to table tag 0 ({nanch} camera(s))"

    def _preflight_calibration(self, force_crate=False, local_only=False):
        """Verify/auto-anchor every camera to table tag 0 before recording. Returns
        (problems, notes): problems is the list to block on, notes is every status line
        for the log. Covers this robot's head cam + crate cams, and (when linked and not
        ``local_only``) the other robot's head cam and crate coverage. ``force_crate``
        re-anchors the crate cameras even if already anchored (manual Calibrate)."""
        problems, notes = [], []

        head_ok, head_msg = self._ensure_headcam_localized()
        notes.append(head_msg)
        if not head_ok:
            problems.append(head_msg)

        crate_state, crate_msg = self._ensure_crate_anchored(force=force_crate)
        notes.append(crate_msg)
        peer_st = (self._link.get_peer_status()
                   if (self._linked() and not local_only) else {})
        peer_crate = peer_st.get("crate_anchored") if peer_st else None
        if crate_state is False:
            problems.append(crate_msg)               # a running tracker failed to anchor
        elif crate_state is None and peer_crate is not True and not local_only:
            # crate_state is None = NO crate tracker process is running on THIS machine (it's
            # not that a camera can't see the tag — nothing is reading the cameras at all).
            # Combined with no anchored crate rig on the peer → crate won't be in the table
            # frame. The fix is to START the tracker, not to reposition the tag.
            problems.append("Crate cams: crate tracker not running on either machine — click "
                            "'Start Crate Tracking' on the machine with the external RealSense "
                            "cameras (aiming cameras at the tag does nothing until it's running)")

        if self._linked() and not local_only:
            base_up = peer_st.get("base_up") if peer_st else None
            if base_up is None:
                problems.append("Peer robot head cam: NOT localized to table tag 0")
                notes.append("Peer head cam: not seeing tag 0")
            elif float(base_up) < 0.8:
                problems.append(f"Peer robot head cam: base pose weak (up={base_up:+.2f})")
                notes.append(f"Peer head cam: weak (up={base_up:+.2f})")
            else:
                notes.append(f"Peer head cam: locked on tag 0 (up={base_up:+.2f})")
            if peer_crate is True:
                notes.append("Peer crate cams: anchored")

        return problems, notes

    def _ensure_recording_subsystems(self):
        """Before recording, START (if down) the data-producing subsystems whose output is
        ONLY captured while they run — so a take can't silently miss them. Start-if-down
        ONLY: it never re-anchors an already-anchored crate rig (the pre-flight anchors what
        needs it without shifting a world frame that's already set), so it's safe to call on
        every take — already-running subsystems are a no-op. Best-effort + self-logging."""
        # Head-cam base localizer — records THIS robot's base pose CSV (both robots). On a
        # peer, starting it also auto-engages crate contribution (on by default).
        if self._headcam_localizer is None and not self._headcam_starting:
            self._log("[rec] head-cam localizer down — starting it so the base pose is "
                      "recorded…")
            self._init_headcam_localizer()
            QApplication.processEvents()
        # External crate tracker — records the crate pose CSVs. The peer feeds the crate via
        # its head cam (contribution) instead, so only auto-start the tracker where it lives.
        if (self._link_role() != "peer" and self._apriltag_tracker is None
                and not self._apriltag_starting):
            self._log("[rec] crate tracker down — starting it so the crate pose is "
                      "recorded…")
            self._init_apriltag_tracker()
            QApplication.processEvents()

    def _calibration_gate_ok(self):
        """Run the pre-flight calibration before recording. Returns True to proceed.
        If a camera isn't anchored, warn and let the operator fix it or record anyway."""
        if not (hasattr(self, "_require_calib_chk") and self._require_calib_chk.isChecked()):
            return True
        self._ensure_recording_subsystems()   # auto-start crate tracker + head-cam localizer
        self._log("[calib] pre-flight: checking every camera is anchored to table tag 0…")
        problems, notes = self._preflight_calibration()
        for n in notes:
            self._log(f"[calib]   {n}")
        if not problems:
            self._log("[calib] all cameras anchored to table tag 0 — recording is "
                      "spatially calibrated.")
            return True
        msg = ("Some cameras are NOT calibrated to the table tag (id 0):\n\n"
               + "\n".join(f"  •  {p}" for p in problems)
               + "\n\nRecording now means this data may not reconstruct into one shared "
                 "table frame for the digital twin.\n\nFix the cameras first, or record "
                 "anyway?")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Camera calibration incomplete")
        box.setText(msg)
        fix_btn = box.addButton("Fix first", QMessageBox.RejectRole)
        rec_btn = box.addButton("Record anyway", QMessageBox.AcceptRole)
        box.setDefaultButton(fix_btn)
        box.exec_()
        if box.clickedButton() is rec_btn:
            self._log("[calib] operator chose to record ANYWAY despite calibration "
                      "warnings.")
            return True
        self._log("[calib] recording cancelled — fix camera calibration first.")
        return False

    # ---- Digital-twin data-completeness gate -------------------------------------
    def _twin_readiness(self):
        """Whether THIS robot is set up to capture the FULL digital-twin dataset this
        instant: the robot_raw joint stream (booted robot), ZED video, and tactile
        (hand). Returns (ok: bool, missing: [str]). Mirrors exactly the conditions that
        produced the empty robot2 recording (booted robot → robot_raw, ZED stream →
        zed_stereo.mp4, hand → tactile.bin)."""
        missing = []
        if self._hw_controller is None or self._state == _S_INIT or self._robot_comms_lost:
            missing.append("robot not booted → no robot_raw joint stream (robot won't "
                           "appear in the twin scene)")
        if not self._zed_recorder.is_streaming:
            missing.append("ZED not streaming → no zed_stereo video (no camera view)")
        if self._hand_controller is None:
            missing.append("no hand controller → no tactile/finger data")
        return (len(missing) == 0), missing

    def _twin_gate_ok(self):
        """HARD pre-attempt gate: BOTH robots must self-confirm they are collecting every
        piece of data the two-robot digital twin needs. Blocks the attempt with a per-robot
        list of what's missing (this machine checks itself directly; the other robot's
        readiness comes from its status broadcast over the link). A deliberate, non-default
        override is offered so the operator is never permanently stuck, but the easy path is
        blocked — so an attempt can't silently record data that won't reconstruct."""
        my_id = int(self._robot_id)
        other_id = 2 if my_id == 1 else 1
        problems = {}

        ok_local, miss_local = self._twin_readiness()
        if miss_local:
            problems[my_id] = miss_local

        # The OTHER robot's readiness rides in on its status broadcast (twin_ready /
        # twin_missing added to _status_snapshot). No link ⇒ there is no second robot.
        role_active, connected = self._link_state()
        if not role_active:
            problems[other_id] = ["not linked — no second robot connected, so only one "
                                  "robot's data would be recorded (link to the peer)"]
        elif not connected:
            problems[other_id] = ["link DOWN — can't confirm the other robot is recording "
                                  "(reconnect the link)"]
        else:
            comps = (self._link.get_peer_status() or {}).get("components") or {}
            if comps.get("twin_ready") is None:
                problems[other_id] = ["not reporting twin readiness yet — update/restart "
                                      "the other manager, or wait for its next status"]
            elif not comps.get("twin_ready"):
                problems[other_id] = list(comps.get("twin_missing")
                                          or ["missing required twin data"])

        if not problems:
            self._log("[twin] pre-attempt check: both robots collecting full twin data "
                      "(robot_raw + ZED + tactile). OK to record.")
            return True

        def _flat():
            return "; ".join(f"robot{r}: {', '.join(v)}" for r, v in sorted(problems.items()))
        lines = []
        for rid in sorted(problems):
            who = "Master" if rid == 1 else "Peer"
            lines.append(f"Robot {rid} ({who}):")
            lines += [f"    •  {m}" for m in problems[rid]]
        msg = ("Not all data required for two-robot digital-twin playback is being "
               "collected — this attempt would NOT reconstruct into a merged twin:\n\n"
               + "\n".join(lines)
               + "\n\nFix the items above, then press Start Attempt again.")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Twin data incomplete — attempt blocked")
        box.setText(msg)
        fix_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        over_btn = box.addButton("Start anyway (incomplete)", QMessageBox.AcceptRole)
        box.setDefaultButton(fix_btn)
        box.exec_()
        if box.clickedButton() is over_btn:
            self._log("[twin] ⚠ operator OVERRODE the twin-data gate — recording with "
                      "INCOMPLETE twin data: " + _flat())
            return True
        self._log("[twin] attempt BLOCKED — required twin data missing: " + _flat())
        return False

    def _on_calibrate_cameras_now(self):
        """Manual backup to the auto pre-flight: anchor EVERY camera to table tag 0 now
        (FRESH crate anchor + confirm head-cam lock). Use it to (re)calibrate before a
        session or after a camera was bumped. Refused while recording (re-anchoring would
        move the world frame mid-session)."""
        if self._session_open():
            self._log("[calib] stop recording before re-calibrating cameras.")
            QMessageBox.information(self, "Recording in progress",
                                    "Stop recording before re-calibrating — re-anchoring "
                                    "would shift the world frame mid-session.")
            return
        # "Calibrate Cameras" should bring the crate cameras UP, not just anchor an already-
        # running tracker — auto-start it here if it isn't running (no-op if the crate rig
        # lives on the other machine / there are no external RealSense cameras on this one;
        # _init_apriltag_tracker logs the specific reason and leaves tracking off if so).
        if self._apriltag_tracker is None and not self._apriltag_starting:
            self._log("[calib] crate tracker not running — starting it before calibration…")
            self._init_apriltag_tracker()
            QApplication.processEvents()
        self._log("[calib] manual calibration — anchoring all cameras to table tag 0…")
        problems, notes = self._preflight_calibration(force_crate=True)
        for n in notes:
            self._log(f"[calib]   {n}")
        if not problems:
            self._log("[calib] all cameras anchored to table tag 0.")
            QMessageBox.information(
                self, "Cameras calibrated",
                "All cameras are anchored to the table AprilTag (id 0):\n\n"
                + "\n".join(f"  •  {n}" for n in notes)
                + "\n\nThe table tag can now be obscured during manipulation — the "
                  "one-time anchor keeps crate tracking in the table frame (just don't "
                  "move the cameras).")
        else:
            QMessageBox.warning(
                self, "Calibration incomplete",
                "Some cameras are NOT calibrated to the table tag (id 0):\n\n"
                + "\n".join(f"  •  {p}" for p in problems)
                + "\n\nReposition so tag 0 is clearly visible to every camera, then "
                  "Calibrate Cameras Now again.")

    def _on_calibrate_both(self):
        """Calibrate BOTH robots' cameras from either machine: send a remote calibrate
        request to the peer (runs in parallel) AND calibrate locally. The peer's result
        is logged here when it reports back."""
        if self._linked():
            if self._peer_cmd("calibrate_cameras"):
                self._log("[calib] asked the peer to calibrate its cameras (result will "
                          "appear here).")
        else:
            self._log("[calib] no peer linked — calibrating this machine only.")
        self._on_calibrate_cameras_now()   # local, with dialog

    def _remote_calibrate(self):
        """Execute a remote 'calibrate cameras' request from the other machine: force a
        fresh anchor of THIS robot's cameras to table tag 0 (non-interactive, no dialog),
        then report the result back over the link. Refused while recording."""
        if self._session_open():
            self._log("[calib] remote calibrate ignored — recording in progress.")
            self._peer_cmd("calib_result", robot=self._robot_id,
                           text="ignored (recording in progress)")
            return
        self._log("[calib] remote request — calibrating this robot's cameras…")
        # Bring this machine's crate tracker up first if it isn't running, so a Calibrate
        # triggered from the OTHER machine still opens the local external cameras.
        if self._apriltag_tracker is None and not self._apriltag_starting:
            self._log("[calib] crate tracker not running — starting it before calibration…")
            self._init_apriltag_tracker()
            QApplication.processEvents()
        problems, notes = self._preflight_calibration(force_crate=True, local_only=True)
        for n in notes:
            self._log(f"[calib]   {n}")
        summary = ("calibration INCOMPLETE: " + "; ".join(problems) if problems
                   else "cameras anchored to table tag 0")
        self._log(f"[calib] {summary}")
        self._peer_cmd("calib_result", robot=self._robot_id, text=summary)

    def _peer_preflight_on_start(self):
        """Peer side: when the OTHER machine commands a recording start, auto-calibrate
        THIS robot's own cameras to table tag 0 first. Non-interactive (no dialog) so it
        can't stall / desync the synchronized start — it logs any issue and proceeds."""
        # Master-link reachability is checked regardless of the camera-calib toggle (the
        # master initiated, so this peer never ran the blocking start gate). Warn only —
        # a synced start must not be blocked; the data can still be sent later.
        if self._link_role() == "peer" and self._linked():
            state, msg = self._check_master_ssh_dest()
            if state is False:
                self._log(f"[twin] WARNING: master link not reachable — {msg} "
                          "(data can still be sent manually later).")
            elif state is True:
                self._log(f"[twin] master link OK — {msg}")
        if not (hasattr(self, "_require_calib_chk") and self._require_calib_chk.isChecked()):
            return
        self._ensure_recording_subsystems()   # bring this robot's head-cam localizer (+ crate) up
        self._log("[calib] remote start — running this robot's camera pre-flight…")
        problems, notes = self._preflight_calibration(local_only=True)
        for n in notes:
            self._log(f"[calib]   {n}")
        if problems:
            self._log("[calib] WARNING: starting despite calibration issues: "
                      + "; ".join(problems))
        else:
            self._log("[calib] this robot's cameras are anchored to table tag 0.")

    # ── Per-attempt demo quality label (synced to both robots) ────────────────
    def _on_mark_attempt(self, quality):
        if not self._recording_active:
            return
        self._sync_or_local("set_attempt_quality", {"quality": quality},
                            lambda: self._local_mark_attempt(quality))

    def _local_mark_attempt(self, quality):
        if not self._recording_active:
            return
        self._attempt_quality = quality
        try:
            self._session_recorder.update_meta(attempt_quality=quality)
        except Exception:
            pass
        self._log(f"[quality] this attempt marked {quality.upper()}.")
        self._update_attempt_quality_label()

    def _update_attempt_quality_label(self):
        col = {"good": "green", "bad": "#e67e22"}.get(self._attempt_quality, "gray")
        self._attempt_quality_lbl.setText(f"Quality: {self._attempt_quality}")
        self._attempt_quality_lbl.setStyleSheet(f"color: {col}; font-size: 11px;")

    # ── One-click study prep ──────────────────────────────────────────────────
    def _on_prepare_study(self):
        """One button, run once per participant GROUP: calibrate everything that needs
        calibrating before recording — on BOTH robots. Master anchors its crate cameras +
        pushes the bundle; the peer sets up its head-cam crate contribution; both ensure
        head-cam base localization. After this, per participant you only Start Recording."""
        if self._recording_active:
            self._log("[prep] stop recording before preparing a new group.")
            return
        self._log("[prep] ===== Preparing study group — calibrating all cameras =====")
        # Push the bundle first so the peer has it before it tries to contribute.
        if self._link_role() != "peer":
            self._send_bundle_to_peer()
        if self._linked():
            self._peer_cmd("prepare_study")          # the other robot preps in parallel
        self._prepare_local()
        self._log("[prep] ===== Prep done — check the status lines are green, then "
                  "Start Session per participant. =====")

    def _prepare_local(self):
        """Role-appropriate local calibration for _on_prepare_study (also run on receipt
        of a remote prepare_study). Non-interactive; logs each step."""
        role = self._link_role()
        head_ok, head_msg = self._ensure_headcam_localized()
        self._log(f"[prep]   {head_msg}")
        if role == "peer":
            self._prepare_peer_contribution()
        else:
            if self._apriltag_tracker is None:
                self._init_apriltag_tracker()        # open the external cameras
            if self._apriltag_tracker is not None:
                _, msg = self._ensure_crate_anchored(force=True)
                self._log(f"[prep]   {msg}")
            else:
                self._log("[prep]   crate cameras: none on this machine")
            self._send_bundle_to_peer()
        # Measure + log the master↔peer clock offset and NTP quality for this group.
        self._clock_sync_info = self._measure_clock_sync()
        ntp = self._clock_sync_info.get("ntp", {})
        off = self._clock_sync_info.get("link_peer_minus_local_offset_s")
        self._log(f"[prep]   clock: NTP synced={ntp.get('synced')}" +
                  (f", peer offset {off * 1000:+.1f} ms (rtt "
                   f"{self._clock_sync_info.get('link_rtt_s', 0) * 1000:.1f} ms)"
                   if off is not None else " (no peer link)"))
        # Verify this machine's ZED calibration conf so a missing one is caught before
        # recording (each robot checks its OWN — the peer flags its own here).
        self._check_zed_calibration()

    def _prepare_peer_contribution(self):
        """PEER prep: enable the head-cam crate contribution (load bundle + anchor base)."""
        loc = self._headcam_localizer
        if loc is None:
            self._log("[prep]   crate contribution: head cam not available")
            return
        bundle = self._args.apriltag_bundle
        if not os.path.exists(bundle):
            self._log("[prep]   crate contribution: no bundle yet (waiting for the "
                      "master to send it) — re-run Prepare once linked.")
            return
        ok, msg = loc.enable_crate_contribution(bundle)
        self._log(f"[prep]   crate contribution: {msg}")
        if not ok:
            return
        okb, n = loc.anchor_base()
        self._log(f"[prep]   base anchor: {'ok' if okb else 'FAILED'} ({n} tag-0 views)")
        self._update_buttons()

    def _auto_enable_crate_contribution(self, reason=""):
        """PEER: engage head-cam crate contribution automatically (it's on by default via
        the crate_contribute_default config) so the operator never has to click 'Contribute
        Crate Tracking'. Idempotent + best-effort: it needs the head-cam localizer up and the
        crate bundle present; if a prerequisite isn't ready it logs and bails so a later hook
        (bundle arrival / localizer start) can retry. The re-entry guard makes it safe to call
        from _init_headcam_localizer (which this may itself invoke to bring the head cam up)."""
        if not self._cfg.get("crate_contribute_default", True):
            return
        if self._link_role() != "peer":
            return                                   # contribution is a peer→master feed
        if getattr(self, "_auto_contrib_busy", False):
            return                                   # re-entry from _init_headcam_localizer
        loc = self._headcam_localizer
        if loc is not None and getattr(loc, "crate_contributing", False):
            return                                   # already contributing — nothing to do
        bundle = self._args.apriltag_bundle
        if not os.path.exists(bundle):
            self._log("[crate] auto-contribute pending: no crate bundle yet "
                      "(the master shares it on connect).")
            return
        self._auto_contrib_busy = True
        try:
            if loc is None:
                self._init_headcam_localizer()       # contribution rides the localizer
                loc = self._headcam_localizer
                if loc is None:
                    self._log("[crate] auto-contribute pending: head cam not up "
                              "(set the Jetson IP / start Head-Cam Localize).")
                    return
                if getattr(loc, "crate_contributing", False):
                    return                           # engaged during a nested retry
            ok, msg = loc.enable_crate_contribution(bundle)
            self._log(f"[crate] auto-contribute{f' ({reason})' if reason else ''}: {msg}")
            if not ok:
                return
            try:
                okb, n = loc.anchor_base()
                self._log("[crate] auto-contribute base anchor: "
                          + ("ok" if okb else "FAILED — aim the head at table tag 0")
                          + f" ({n} tag-0 views)")
            except Exception as exc:                 # noqa: BLE001
                self._log(f"[crate] auto-contribute base anchor error: {exc}")
        finally:
            self._auto_contrib_busy = False
        self._update_buttons()

    def _on_open_checklist(self):
        """Open the INTERACTIVE pre-flight checklist (clickable steps, state persists;
        includes the voice-calibration script and the pre-robot practice recorder)."""
        if self._checklist_window is None:
            self._checklist_window = ChecklistWindow(self)
        self._checklist_window.show()
        self._checklist_window.raise_()
        self._checklist_window.activateWindow()

    # ── Pre-robot PRACTICE round (hand-lift the box; records ONLY the comms transcript) ──
    def _on_practice_toggle(self):
        """Start/stop the practice recording (synced to both machines): records ONLY the
        audio transcript rows labeled 'practice'. No robot data (ZED video / crate pose) —
        the robot isn't used during practice. Needs an open session (for the transcript)."""
        if not self._session_open():
            self._log("[practice] Start Session first — the practice transcript needs "
                      "the open session.")
            return
        if self._recording_active:
            self._log("[practice] a take is recording — not during takes.")
            return
        if not self._practice_active:
            self._sync_or_local("practice_start", {}, self._local_practice_start)
        else:
            self._sync_or_local("practice_stop", {}, self._local_practice_stop)

    def _local_practice_start(self):
        if (not self._session_open() or self._recording_active
                or self._practice_active):
            return
        self._practice_active = True
        # PRACTICE round uses NO robot — record ONLY the audio transcript (rows tagged
        # 'practice'). No robot data (ZED video / crate pose) is captured, since the robot
        # isn't in use while participants move the box by hand.
        if self._transcriber is not None:
            self._transcriber.set_take("practice")
            self._transcriber.set_recording(True)
        self._log("[practice] ▶ RECORDING the practice round — transcript only (rows "
                  "labeled 'practice'; no robot data). Participants lift/flip the box "
                  "BY HAND; press End Practice when done.")
        self._refresh_practice_button()
        if self._checklist_window is not None:
            self._checklist_window.refresh_practice()

    def _local_practice_stop(self):
        if not self._practice_active:
            return
        self._practice_active = False
        if self._transcriber is not None:
            self._transcriber.set_recording(False)   # practice ended → stop saving
            self._transcriber.set_take("between")
        self._log("[practice] ⏹ practice transcript saved (rows labeled 'practice'; "
                  "no robot data recorded).")
        self._refresh_practice_button()
        if self._checklist_window is not None:
            self._checklist_window.refresh_practice()

    def _refresh_practice_button(self):
        """Update the Data-Collection Start/End Practice button to match practice state."""
        if not hasattr(self, "_btn_practice_main"):
            return
        active = bool(self._practice_active)
        self._btn_practice_main.setText(
            "⏹  End Practice (RECORDING…)" if active else "▶  Start Practice (no robot)")
        self._btn_practice_main.setStyleSheet(
            "QPushButton { background: %s; color: white; font-weight: bold; padding: 8px; }"
            % ("#c0392b" if active else "#8e44ad"))

    def _on_session_toggle(self):
        """Single Start Session ⇄ End Session button: open the session, or (when one is
        open) end it — replaces the old separate Stop Session button."""
        if self._session_open():
            self._on_stop_recording()
        else:
            self._on_start_button()

    def _on_attempt_toggle(self):
        """Single Start Attempt ⇄ End Attempt button: start the take, or (while recording)
        end it — replaces the old separate Stop Attempt button."""
        if self._recording_active:
            self._on_stop_attempt()
        else:
            self._on_start_attempt_button()

    def _on_start_button(self):
        """Start Session: OPEN the session — run the calibration/SSH gates, start every
        recording subsystem, create the folder, set the pointer to the TOP of the ranked
        trial order (attempt 1) — but record NOTHING yet. Recording is per-attempt via
        the Start/End Attempt toggle, so downtime between attempts is never captured.
        Synced when linked."""
        if self._session_open() or self._recording_active or self._session_opening:
            return
        if not self._calibration_gate_ok():
            return
        # Peer only: verify it can deliver its data to the master before committing to a
        # whole session (the auto-send happens at Stop Session).
        if not self._master_link_gate_ok():
            return
        pid = self._pid_edit.text().strip() or "unknown"
        master_code = self._master_code_edit.text().strip()
        peer_code = self._peer_code_edit.text().strip()
        cond = self._trial_order()[0]   # first condition in the ranked order
        attempt = 1
        # One-press guard: grey the button NOW — a double-click during the synced-epoch
        # delay must not queue a second open_session. Cleared when the session opens
        # (or by the failsafe if the synced open never lands).
        self._session_opening = True
        self._update_buttons()
        QTimer.singleShot(6000, self._clear_session_opening)
        # Generate the shared session id HERE (the initiator) and send it to the peer so
        # both robots write into the identically-named session folder. The operator codes go
        # with it so the peer records its operator's code (both are assigned here on master).
        session_id = self._new_session_id(pid)
        self._sync_or_local(
            "open_session",
            {"pid": pid, "condition": cond, "attempt": attempt, "session_id": session_id,
             "master_code": master_code, "peer_code": peer_code},
            lambda: self._local_open_session(pid, cond, attempt, session_id,
                                             master_code, peer_code))

    def _clear_session_opening(self):
        """Failsafe: re-enable Start Session if an open request never landed."""
        if self._session_opening and not self._session_open():
            self._session_opening = False
            self._log("[session] open request timed out — Start Session re-enabled.")
            self._update_buttons()

    def _local_open_session(self, pid, condition, attempt, session_id=None,
                            master_code=None, peer_code=None):
        """Open the session locally (no peer broadcast): folder + pointer + everything
        READY — armed, not recording. Called by Start Session and by a remote
        open_session command."""
        if self._session_open() or self._recording_active:
            return
        self._set_selectors(pid, condition, attempt, master_code, peer_code)
        self._rec_pid = pid
        self._session_id = session_id or self._new_session_id(pid)
        self._session_root = self._session_root_for(self._session_id)
        self._last_session_root = self._session_root
        self._attempt_quality = "unrated"
        self._recorded_takes = set()      # fresh one-press slate for the new session
        self._session_opening = False     # the open landed — guard released
        os.makedirs(self._session_root, exist_ok=True)
        # Bring the ZED streaming pipeline up NOW (normally already running) so the first
        # Start Attempt doesn't pay the ~3 s stabilization wait.
        if not self._zed_recorder.is_streaming:
            self._log("Starting ZED camera streaming...")
            self._zed_recorder.start_streaming()
        if self._transcriber is not None:
            self._transcriber.start_session(self._session_root, self._session_id)
        self._log(f"[session] OPEN — {self._session_id} armed at “{condition}” attempt "
                  f"{attempt}. Nothing is recording: press Start Attempt to record the "
                  "first take.")
        self._update_buttons()
        self._update_order_label()
        self._autosave_study()

    def _on_start_recording(self):
        """LEGACY alias (old flow: open + record first take immediately). Kept only for
        compatibility with a remote 'start_recording' from an un-updated peer."""
        if self._recording_active:
            return
        if not self._calibration_gate_ok():
            return
        if not self._master_link_gate_ok():
            return
        pid = self._pid_edit.text().strip() or "unknown"
        master_code = self._master_code_edit.text().strip()
        peer_code = self._peer_code_edit.text().strip()
        cond = self._trial_order()[0]
        attempt = 1
        session_id = self._new_session_id(pid)
        self._sync_or_local(
            "start_recording",
            {"pid": pid, "condition": cond, "attempt": attempt, "session_id": session_id,
             "master_code": master_code, "peer_code": peer_code},
            lambda: self._local_start_recording(pid, cond, attempt, session_id,
                                                master_code, peer_code))

    def _on_start_attempt_button(self):
        """Start Attempt (one press per attempt): starts recording the current
        (condition, attempt) and immediately greys itself out for this slot — a second
        press on the same attempt is impossible. Re-enables when Continue advances the
        pointer to a fresh run. Stop Attempt is its own button below classify/redo."""
        if not self._session_open():
            self._log("[session] open a session first (Start Session).")
            return
        if self._recording_active:
            return
        cur = (self._cond_combo.currentText(), self._attempt_spin.value())
        if cur in self._recorded_takes:
            self._log(f"[session] “{cur[0]}” attempt {cur[1]} was already started — "
                      "advance with Continue (or Redo while recording to re-take).")
            return
        self._on_start_attempt()

    def _on_stop_attempt(self):
        """Stop Attempt: END + save the current take, stay ARMED at the same
        (condition, attempt) — advance with Next Attempt / Next Trial. Synced."""
        if not self._session_open() or not self._recording_active:
            return
        self._sync_or_local("stop_attempt", {}, self._local_stop_attempt)

    def _local_stop_attempt(self):
        """Finalize the current take locally (no peer broadcast); session stays open.
        PAUSES teleop first (freeze in place) so the robot holds its pose before the take is
        ended + saved rather than still teleoperating as recording stops. Runs on BOTH robots —
        each side runs _local_stop_attempt (master directly, peer via the synced stop_attempt
        command) so both pause in step. A no-op if teleop is already paused, so a stop from a
        paused state just saves like normal."""
        if not self._recording_active:
            return
        self._pause_local_teleop()   # hold pose on this robot before finalizing (no-op if paused)
        self._finalize_recording()
        self._recording_active = False
        self._log("[session] attempt saved — ARMED (not recording). Advance with Next "
                  "Attempt / Next Trial, then Start Attempt.")
        self._update_buttons()
        self._update_order_label()
        self._autosave_study()

    def _local_start_recording(self, pid, condition, attempt, session_id=None,
                               master_code=None, peer_code=None):
        """Begin recording locally (no peer broadcast). Called by the button scheduler
        and by a remote start_recording command. ``session_id`` is the SHARED folder name;
        when omitted (unlinked / legacy) a fresh one is generated."""
        if self._recording_active:
            return
        self._set_selectors(pid, condition, attempt, master_code, peer_code)
        self._rec_pid = pid
        self._session_id = session_id or self._new_session_id(pid)
        self._session_root = self._session_root_for(self._session_id)
        self._last_session_root = self._session_root
        self._attempt_quality = "unrated"            # fresh label for the new attempt
        self._begin_recording_into(self._current_rec_dir(), pid, condition, attempt)
        self._recording_active = True
        self._update_buttons()
        # Re-measure the clock sync ~1.5 s AFTER the synced start so its ~0.6 s block
        # can't skew the shared-epoch recording start (then re-stamp it into the meta).
        QTimer.singleShot(1500, self._refresh_clock_sync_meta)
        self._autosave_study()

    def _on_start_attempt(self):
        """Start Attempt: begin recording the CURRENT (condition, attempt) inside the
        already-open session. This is what you press after Next Trial/Attempt (which END
        the previous take and advance the pointer). Synced when linked. No-op if there is
        no open session or a take is already recording."""
        if not self._session_open() or self._recording_active:
            return
        if not self._calibration_gate_ok():
            return
        # Both robots must confirm they're capturing the full twin dataset before we commit
        # (and before the peer is told to start) — runs only on the initiator.
        if not self._twin_gate_ok():
            return
        cond = self._cond_combo.currentText()
        attempt = self._attempt_spin.value()
        # One-press guard: mark this slot as started AT CLICK TIME so the button greys
        # instantly — a double-click can't queue a second synced start for the same take.
        self._recorded_takes.add((cond, attempt))
        self._update_buttons()
        self._sync_or_local(
            "start_attempt", {"condition": cond, "attempt": attempt},
            lambda: self._local_start_attempt(cond, attempt))

    def _local_start_attempt(self, condition, attempt):
        """Begin the next take locally (no peer broadcast) inside the open session. Called
        by the Start Attempt button scheduler and by a remote start_attempt command."""
        if not self._session_open() or self._recording_active:
            return
        self._set_selectors(None, condition, attempt)
        self._recorded_takes.add((condition, attempt))   # both sides track one-press slots
        self._attempt_quality = "unrated"            # fresh label for the new take
        self._begin_recording_into(self._current_rec_dir(), self._rec_pid, condition, attempt)
        self._recording_active = True
        self._update_buttons()
        QTimer.singleShot(1500, self._refresh_clock_sync_meta)
        self._autosave_study()

    # ── Mid-study save / resume (survives manager relaunches) ───────────────
    def _study_state_path(self):
        """Per-machine resume file, next to the recordings (NOT inside the session
        folder, so it survives even if the session dir is being rsynced)."""
        return os.path.join(self._output_dir_path(), "study_resume.json")

    def _local_save_study(self, quiet=False):
        """Write this machine's mid-study state to disk: open session id, participant,
        (condition, attempt) pointer, the one-press recorded-takes ledger, and the trial
        ranking. Auto-saved at every session state change; the 💾 button forces it.
        Resume (this machine or a synced peer) restores an ARMED session from it."""
        if not self._session_open():
            if not quiet:
                self._log("[resume] no open session to save.")
            return
        st = {
            "session_id": self._session_id,
            "pid": self._rec_pid,
            "master_code": self._rec_master_code,
            "peer_code": self._rec_peer_code,
            "condition": self._cond_combo.currentText(),
            "attempt": int(self._attempt_spin.value()),
            "recorded_takes": sorted([list(t) for t in self._recorded_takes]),
            "trial_order": self._trial_order(),
            "robot_id": self._robot_id,
            "saved_at": time.time(),
        }
        try:
            with open(self._study_state_path(), "w") as f:
                json.dump(st, f, indent=2)
            if not quiet:
                self._log(f"[resume] study state saved → "
                          f"{os.path.basename(self._study_state_path())} "
                          f"({st['condition']} attempt {st['attempt']}, "
                          f"{len(st['recorded_takes'])} takes recorded).")
        except Exception as exc:                     # noqa: BLE001
            self._log(f"[resume] save failed: {exc}")

    def _autosave_study(self):
        """Silent auto-save at every session state change, so a manager crash/restart
        can always resume — even if 💾 was never pressed."""
        try:
            self._local_save_study(quiet=True)
        except Exception:
            pass

    def _on_save_study(self):
        """💾 Save State: snapshot the mid-study state on BOTH machines (synced)."""
        if not self._session_open():
            self._log("[resume] no open session to save.")
            return
        self._sync_or_local("save_study", {}, self._local_save_study)

    def _on_resume_study(self):
        """📥 Resume Study: restore the saved mid-study state — synced, and the
        payload carries the full state, so ONE click resumes BOTH machines even when
        the other side lost its own file (or both were restarted)."""
        if self._session_open():
            self._log("[resume] a session is already open — nothing to resume.")
            return
        try:
            with open(self._study_state_path()) as f:
                st = json.load(f)
        except Exception as exc:                     # noqa: BLE001
            self._log(f"[resume] no saved study state ({exc}).")
            return
        self._sync_or_local("resume_study", {"state": st},
                            lambda: self._local_resume_study(st))

    def _maybe_prompt_resume_on_boot(self):
        """Once, a few seconds after startup, offer to resume a study that's still in progress —
        either saved locally (THIS machine crashed mid-session) or live on the OTHER machine (its
        manager kept running the study). Prevents the common footgun where, after a crash + manager
        restart, the operator forgets to press 📥 Resume Study and the two robots drift out of sync.
        Prefers the peer's LIVE state when linked (authoritative + current)."""
        if self._resume_prompt_shown:
            return
        # Already in a session (armed or opening) → nothing to resume; stop checking.
        if self._session_open() or getattr(self, "_session_opening", False):
            self._resume_prompt_shown = True
            return
        # Let the peer link connect first so we can prefer its live session over a (possibly
        # older) local save.
        if (time.monotonic() - self._boot_monotonic) < 6.0:
            return

        # Source 1 (preferred): the OTHER machine is running a study right now.
        peer_sess = None
        if self._linked():
            s = (self._link.get_peer_status() or {}).get("session") or {}
            if s.get("open") and s.get("session_id"):
                peer_sess = s
        # Source 2: this machine's own saved state (survives its own crash).
        local_sess = None
        if peer_sess is None:
            try:
                with open(self._study_state_path()) as f:
                    local_sess = json.load(f)
            except Exception:                             # noqa: BLE001
                local_sess = None

        st = peer_sess or local_sess
        if st is None:
            return   # nothing resumable yet — keep checking (the peer may still connect)

        self._resume_prompt_shown = True                  # ask exactly once
        where = ("the OTHER machine (still running)" if peer_sess
                 else "your last session on THIS machine")
        n_done = len(st.get("recorded_takes", []))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Resume study in progress?")
        box.setText(
            "A study appears to still be in progress:\n\n"
            f"    Source:       {where}\n"
            f"    Participant:  {st.get('pid', '?')}\n"
            f"    At:           “{st.get('condition', '?')}”  "
            f"attempt {st.get('attempt', '?')}\n"
            f"    Recorded:     {n_done} attempt(s) so far\n\n"
            "Resume it here so both robots stay in sync and you continue right where you "
            "left off?")
        yes = box.addButton("Resume study", QMessageBox.AcceptRole)
        box.addButton("No, start fresh", QMessageBox.RejectRole)
        box.setDefaultButton(yes)
        box.exec_()
        if box.clickedButton() is not yes:
            self._log("[resume] boot resume declined — press 📥 Resume Study later if you "
                      "change your mind.")
            return
        self._resume_study_from(st, from_peer=peer_sess is not None)

    def _resume_study_from(self, st, from_peer):
        """Restore the armed session from ``st``, then run a quick fresh calibration so the
        resumed session lands in the same table frame as the pre-crash data. When ``st`` is the
        peer's LIVE session, restore locally only (the peer is already running it — a synced
        resume would just echo back); when it's a local file, use the synced resume so both
        machines converge."""
        self._log(f"[resume] resuming study from "
                  f"{'the other machine' if from_peer else 'the local save'}…")
        if from_peer:
            self._local_resume_study(st)
        else:
            self._sync_or_local("resume_study", {"state": st},
                                lambda: self._local_resume_study(st))
        self._resume_calibration_if_needed(from_peer=from_peer)

    def _resume_calibration_if_needed(self, from_peer=False):
        """A quick FRESH camera calibration on resume so the resumed data lands in the same table
        frame as the pre-crash data (cameras re-anchored to tag 0, base re-localized). Only when
        the calibration gate is on AND it's actually needed — an already-anchored rig is left
        alone so we don't shift a world frame that's already set. Best-effort: the cameras/robot
        must be up, so if they're not ready yet the pre-flight gate at the next Start Attempt
        still enforces it."""
        if not (hasattr(self, "_require_calib_chk") and self._require_calib_chk.isChecked()):
            self._log("[resume] calibration gate off — skipping the on-resume re-anchor.")
            return
        try:
            self._ensure_recording_subsystems()          # start head-cam localizer / crate tracker
            problems, _notes = self._preflight_calibration(local_only=True)
        except Exception as exc:                          # noqa: BLE001
            self._log(f"[resume] calibration check unavailable now ({exc}); it will run at the "
                      "next Start Attempt.")
            return
        if not problems:
            self._log("[resume] cameras already anchored to table tag 0 — no re-calibration "
                      "needed; continuing in sync.")
            return
        if from_peer:
            # The peer's session is the authority and it's still anchored/running — re-anchor only
            # THIS machine (the one that restarted) so we don't disturb the peer's world frame.
            self._log("[resume] re-anchoring THIS robot's cameras to table tag 0 for a clean, "
                      "in-sync resume…")
            self._on_calibrate_cameras_now()
        else:
            # Both machines were restarted — re-anchor both.
            self._log("[resume] re-anchoring BOTH robots' cameras to table tag 0 for a clean, "
                      "in-sync resume…")
            self._on_calibrate_both()

    def _local_resume_study(self, st):
        """Restore an ARMED session from a saved state dict. Idempotent for a side
        whose session is still open (same id → just re-syncs the pointer + ledger, so
        resuming after only ONE machine restarted is safe on both)."""
        sid = (st or {}).get("session_id")
        if not sid:
            self._log("[resume] invalid saved state (no session id).")
            return
        if self._recording_active:
            self._log("[resume] a take is RECORDING — refuse (stop it first).")
            return
        if self._session_open():
            if self._session_id == sid:
                self._set_selectors(st.get("pid"), st.get("condition"),
                                    st.get("attempt"), st.get("master_code", ""),
                                    st.get("peer_code", ""))
                self._recorded_takes = {tuple(t) for t in st.get("recorded_takes", [])}
                if st.get("trial_order"):
                    self._apply_trial_order(st["trial_order"])
                self._log("[resume] session already open — pointer + take ledger "
                          "re-synced from the save.")
                self._update_buttons()
                self._update_order_label()
            else:
                self._log(f"[resume] REFUSED: a DIFFERENT session is open "
                          f"({self._session_id} vs saved {sid}).")
            return
        self._rec_pid = st.get("pid") or "unknown"
        self._session_id = sid
        self._session_root = self._session_root_for(sid)
        self._last_session_root = self._session_root
        os.makedirs(self._session_root, exist_ok=True)
        self._recorded_takes = {tuple(t) for t in st.get("recorded_takes", [])}
        self._session_opening = False
        self._attempt_quality = "unrated"
        if st.get("trial_order"):
            self._apply_trial_order(st["trial_order"])
        self._set_selectors(st.get("pid"), st.get("condition"), st.get("attempt"),
                            st.get("master_code", ""), st.get("peer_code", ""))
        if not self._zed_recorder.is_streaming:
            self._log("Starting ZED camera streaming...")
            self._zed_recorder.start_streaming()
        done = len(self._recorded_takes)
        self._log(f"[resume] session {sid} RESUMED (armed) at "
                  f"“{st.get('condition')}” attempt {st.get('attempt')} — {done} "
                  "take(s) already recorded. Boot the robot + Start Teleop if needed, "
                  "then Start Attempt to continue.")
        self._update_buttons()
        self._update_order_label()

    def _switch_to(self, condition, attempt):
        """Advance the (condition, attempt) pointer. PER-TAKE model: if a take is
        recording, advancing ENDS it (finalizes the files) and leaves the session ARMED —
        it does NOT roll straight into the next recording (press Start Attempt for that).
        Synced when linked so both robots' pointers move together. When no session is open
        yet, just moves the local selectors."""
        if not self._session_open():
            self._set_selectors(None, condition, attempt)
            return
        self._sync_or_local(
            "switch_recording", {"condition": condition, "attempt": attempt},
            lambda: self._local_switch_recording(condition, attempt))

    def _local_switch_recording(self, condition, attempt):
        """Finalize the current take (if one is recording) and move the pointer to
        (condition, attempt), staying ARMED. Does NOT begin the next take — that's Start
        Attempt. (Formerly this rolled straight into the next recording, so takes ran
        back-to-back with the between-take reset time captured; now each take is clean.)"""
        if not self._session_open():
            self._set_selectors(None, condition, attempt)
            return
        if self._recording_active:
            self._finalize_recording()
            self._recording_active = False
        self._set_selectors(None, condition, attempt)
        self._update_buttons()
        self._update_order_label()
        self._autosave_study()

    # ── Redo: discard the current take + re-record (synced to BOTH robots) ─────
    def _discard_dir(self, path):
        """Delete a recording folder — only if it's INSIDE this session's root (guards
        against ever deleting the wrong path)."""
        root, ap = self._session_root, (os.path.abspath(path) if path else "")
        if not (root and ap):
            return False
        ar = os.path.abspath(root)
        if not (ap == ar or ap.startswith(ar + os.sep)):
            self._log(f"[redo] refusing to delete outside the session: {path}")
            return False
        try:
            shutil.rmtree(ap, ignore_errors=True)
            return True
        except Exception as exc:                     # noqa: BLE001
            self._log(f"[redo] delete failed ({path}): {exc}")
            return False

    def _confirm_redo(self, question):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Redo — discard recorded data")
        box.setText(question + "\n\nThis PERMANENTLY deletes that data on this robot"
                    + (" AND the peer." if self._linked() else "."))
        yes = box.addButton("Discard & redo", QMessageBox.AcceptRole)
        no = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(no)                     # default = safe (Cancel)
        box.exec_()
        return box.clickedButton() is yes

    def _on_redo_attempt(self):
        """Discard THIS attempt's data and re-record it. Two modes, both synced:
        • MID-TAKE (recording): discard + immediately re-record — as always.
        • ARMED (after Stop Attempt) — the guarded post-stop ESCAPE HATCH: discard the
          SAVED take and un-mark the one-press slot, so Start Attempt re-enables for it
          (recording starts only when you press it — reposition first)."""
        cond, att = self._cond_combo.currentText(), self._attempt_spin.value()
        if self._recording_active:
            if not self._confirm_redo(f"Discard and re-record “{cond}” attempt {att}?"):
                return
            self._sync_or_local("redo_attempt", {"condition": cond, "attempt": att},
                                lambda: self._local_redo_attempt(cond, att))
            return
        # Armed escape hatch: only for a slot that was actually recorded + saved.
        if not self._session_open() or (cond, att) not in self._recorded_takes:
            return
        if not self._confirm_redo(
                f"“{cond}” attempt {att} is already SAVED. Discard that take so it can "
                "be re-recorded?\n(Start Attempt re-enables for it — recording begins "
                "only when you press it.)"):
            return
        self._sync_or_local("redo_attempt_armed", {"condition": cond, "attempt": att},
                            lambda: self._local_redo_attempt_armed(cond, att))

    def _local_redo_attempt_armed(self, condition, attempt):
        """Armed-state redo (no take recording): delete the saved attempt folder and
        re-open its one-press slot. Local side of the synced escape hatch."""
        if self._recording_active or not self._session_open():
            return
        cond = condition or self._cond_combo.currentText()
        att = int(attempt) if attempt is not None else self._attempt_spin.value()
        self._set_selectors(None, cond, att)
        self._discard_dir(self._current_rec_dir())   # delete THIS robot's saved take
        self._recorded_takes.discard((cond, att))
        self._log(f"[redo] discarded saved “{cond}” attempt {att} — Start Attempt "
                  "re-enabled for it; press it when ready.")
        self._update_buttons()
        self._update_order_label()

    def _local_redo_attempt(self, condition=None, attempt=None):
        if not self._recording_active:
            return
        cond = condition or self._cond_combo.currentText()
        att = int(attempt) if attempt is not None else self._attempt_spin.value()
        self._finalize_recording()                   # close the discarded take's files
        self._set_selectors(None, cond, att)
        self._discard_dir(self._current_rec_dir())   # delete THIS robot's attempt leaf
        self._log(f"[redo] discarded “{cond}” attempt {att} — re-recording from scratch.")
        self._begin_recording_into(self._current_rec_dir(), self._rec_pid, cond, att)
        self._update_buttons()

    def _on_redo_trial(self):
        """Discard ALL attempts of the current condition + restart at attempt 1. Works BOTH
        mid-take AND armed — so after you've moved on you can go BACK to any previous trial
        (via the Trial dropdown) and re-do the whole thing. Synced."""
        cond = self._cond_combo.currentText()
        recorded_here = any(c == cond for (c, _a) in self._recorded_takes)
        if not self._recording_active and not (self._session_open() and recorded_here):
            return
        if not self._confirm_redo(f"Discard ALL attempts of “{cond}” and restart it "
                                  "from attempt 1?"):
            return
        self._sync_or_local("redo_trial", {"condition": cond},
                            lambda: self._local_redo_trial(cond))

    def _local_redo_trial(self, condition=None):
        if not self._session_open():
            return
        cond = condition or self._cond_combo.currentText()
        if self._recording_active:
            # Mid-take: finalize, discard the whole condition, re-record attempt 1 immediately.
            self._finalize_recording()
            self._discard_dir(os.path.join(self._session_root, cond.replace("/", "_")))
            self._recorded_takes = {(c, a) for (c, a) in self._recorded_takes
                                    if c != cond or a == 1}
            self._log(f"[redo] discarded ALL attempts of “{cond}” — restarting at attempt 1.")
            self._set_selectors(None, cond, 1)
            self._begin_recording_into(self._current_rec_dir(), self._rec_pid, cond, 1)
            self._update_buttons()
            return
        # ARMED (not recording): discard every attempt of this trial, re-open ALL of its slots,
        # and stay ARMED at attempt 1 — the escape hatch for re-doing a PREVIOUS completed trial
        # you've already moved past. Recording starts only when you press Start Attempt.
        self._discard_dir(os.path.join(self._session_root, cond.replace("/", "_")))
        self._recorded_takes = {(c, a) for (c, a) in self._recorded_takes if c != cond}
        self._log(f"[redo] discarded ALL attempts of “{cond}” — armed at attempt 1 (press "
                  "Start Attempt to re-record).")
        self._set_selectors(None, cond, 1)
        self._update_buttons()
        self._update_order_label()
        self._autosave_study()

    def _trial_order(self):
        """The trial sequence, read top-to-bottom from the drag-to-rank list. Falls back
        to the default order before the list exists. This is what 'Next Trial' follows."""
        if not hasattr(self, "_order_list"):
            return list(_TRIAL_CONDITIONS)
        order = [self._order_list.item(i).text() for i in range(self._order_list.count())]
        return order or list(_TRIAL_CONDITIONS)

    def _move_order_item(self, delta):
        """Move the selected trial up (delta=-1) or down (delta=+1) in the ranking list."""
        row = self._order_list.currentRow()
        new = row + delta
        if row < 0 or not (0 <= new < self._order_list.count()):
            return
        it = self._order_list.takeItem(row)
        self._order_list.insertItem(new, it)
        self._order_list.setCurrentRow(new)
        self._on_order_changed()

    def _on_order_changed(self, *_):
        """The trial ranking changed here (drag or ↑/↓). Refresh the cue and SHARE the
        new order with the peer so both managers' rankings match."""
        self._update_order_label()
        if not self._suppress_order_broadcast and self._linked():
            self._peer_cmd("set_trial_order", order=self._trial_order())

    def _apply_trial_order(self, order):
        """Set this manager's ranking list to `order` received from the peer, WITHOUT
        echoing it back. Coerces to a full permutation of the valid conditions."""
        valid = [c for c in (order or []) if c in _TRIAL_CONDITIONS]
        full = valid + [c for c in _TRIAL_CONDITIONS if c not in valid]
        if full == self._trial_order():
            return
        self._suppress_order_broadcast = True
        try:
            self._order_list.clear()
            for c in full:
                self._order_list.addItem(QListWidgetItem(c))
        finally:
            self._suppress_order_broadcast = False
        self._update_order_label()
        self._log(f"[link] trial order synced from peer: {' → '.join(full)}")

    def _apply_attempt(self, attempt):
        """Set the attempt spinbox to `attempt` from a peer, without re-broadcasting."""
        self._suppress_rotate = True
        try:
            self._attempt_spin.setValue(int(attempt))
        finally:
            self._suppress_rotate = False
        self._update_buttons()

    def _update_order_label(self, *_):
        """Highlight the CURRENT condition's row in the ranking list (bold + colour, so
        the item TEXT stays a pure condition name) and show what's next below it."""
        if not hasattr(self, "_order_lbl"):
            return
        cur = self._cond_combo.currentText() if hasattr(self, "_cond_combo") else None
        active = self._session_open()
        for i in range(self._order_list.count()):
            it = self._order_list.item(i)
            is_cur = bool(active and it.text() == cur)
            f = it.font(); f.setBold(is_cur); it.setFont(f)
            it.setForeground(QColor("#e67e22") if is_cur else QBrush())
        order = self._trial_order()
        if active and cur in order:
            j = order.index(cur)
            nxt = order[j + 1] if j + 1 < len(order) else None
            verb = "Recording" if self._recording_active else "Ready —"
            tail = (f"next → {nxt}" if nxt else "last trial in the order")
            self._order_lbl.setText(f"{verb} {cur} — {tail}")
        else:
            self._order_lbl.setText("")

    def _next_target(self):
        """Where the ONE Continue button goes from the current (condition, attempt):
        the next attempt of this trial, or — after the trial's final attempt — attempt 1
        of the NEXT trial in the ranked order. None when the whole session is complete
        (last trial's last attempt)."""
        cond = self._cond_combo.currentText()
        att = self._attempt_spin.value()
        if att < _ATTEMPTS_PER_TRIAL:
            return cond, att + 1
        order = self._trial_order()
        i = order.index(cond) if cond in order else -1
        if 0 <= i + 1 < len(order):
            return order[i + 1], 1
        return None

    def _on_next_button(self):
        """The combined Continue button: advance to _next_target(). Confirmation-guarded
        at a TRIAL boundary (so a stray click can't hop conditions unnoticed). Disabled
        while recording (Stop Attempt first), so it never cuts a take."""
        if not self._session_open() or self._recording_active:
            return
        tgt = self._next_target()
        if tgt is None:
            self._log(f"All trials complete ({_ATTEMPTS_PER_TRIAL} attempts each) — "
                      "press Stop Session to finish.")
            return
        cond, att = tgt
        cur_cond = self._cond_combo.currentText()
        if cond != cur_cond:
            # Trial boundary: confirm so Next-Trial can't be pressed by accident.
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Continue to the next trial")
            box.setText(f"“{cur_cond}” is complete (all {_ATTEMPTS_PER_TRIAL} attempts "
                        f"recorded).\n\nContinue to trial “{cond}”, Run 1?\n\nThe "
                        f"feedback modalities switch to {cond} at the next Start Attempt.")
            go = box.addButton(f"Continue to {cond}", QMessageBox.AcceptRole)
            cancel = box.addButton("Cancel", QMessageBox.RejectRole)
            box.setDefaultButton(cancel)      # a stray double-click lands on the safe side
            box.exec_()
            if box.clickedButton() is not go:
                self._log("[session] trial advance cancelled.")
                return
        self._set_selectors(None, cond, att)   # reflect in UI immediately
        self._switch_to(cond, att)

    def _on_condition_changed(self):
        """User picked a trial condition. While a session is open, end the current take
        (if recording) + move to attempt 1 of it (ARMED). Programmatic changes are
        suppressed."""
        if self._suppress_rotate or not self._session_open():
            return
        self._switch_to(self._cond_combo.currentText(), 1)

    def _on_attempt_changed(self):
        """User picked an attempt. While a session is open, end the current take (if
        recording) + move the pointer to it (synced to the peer via switch_recording). No
        open session: still SHARE the attempt with the peer so both spinboxes agree."""
        if self._suppress_rotate:
            return
        if self._session_open():
            self._switch_to(self._cond_combo.currentText(), self._attempt_spin.value())
        elif self._linked():
            self._peer_cmd("set_attempt", attempt=self._attempt_spin.value())

    # ── Video streaming (mediamtx + zed_to_mediamtx) ──────────────────────────

    def _verify_zed_publishing(self):
        """Scheduled ~7 s after a streaming start: confirm the ZED is actually publishing
        FRAMES, not just that the pipeline launched. The pipeline can 'start' fine while
        zedsrc produces nothing (wedged camera / stale holder) — then mediamtx serves a 404
        on the video path and the XR app can never connect its camera, so it retries the
        whole peer connection forever. Surface it loudly instead of a phantom 'started'."""
        if self._zed_recorder is None or not self._zed_recorder.is_streaming:
            return
        path = "zed_stereo" if getattr(self._zed_recorder, "sbs", True) else "zed/left"
        if self._zed_recorder.rtsp_path_live(path):
            self._log(f"[zed] publishing OK — /{path} is live (XR app can get video).")
            return
        self._log(f"⚠ [zed] NOT publishing video — no frames on /{path}. The camera is likely "
                  "WEDGED (a stale publisher held it, or it needs a physical replug). The XR "
                  "app will keep failing on the camera until fixed. First try: Video Streaming "
                  "OFF then ON (now clears leftover publishers + reopens the camera).")

    def _start_video_streaming_default(self):
        """Auto-start the ZED video streaming pipeline at launch (streaming is ON
        by default). Launches ./mediamtx then ./zed_to_mediamtx.sh."""
        if not self._zed_recorder.is_streaming:
            self._log("Starting video streaming (mediamtx + zed_to_mediamtx)...")
            ok = self._zed_recorder.start_streaming()
            if not ok:
                self._log("Video streaming failed to start (check mediamtx / GStreamer / ZED).")
            else:
                QTimer.singleShot(7000, self._verify_zed_publishing)
        self._update_stream_button()

    def _on_toggle_streaming(self):
        """Start/stop the ZED video streaming pipeline from the UI button."""
        if self._zed_recorder.is_streaming:
            if self._zed_recorder.is_recording:
                self._log("Cannot stop streaming while recording — stop recording first.")
                return
            self._log("Stopping video streaming...")
            self._zed_recorder.stop_streaming()
        else:
            self._log("Starting video streaming (mediamtx + zed_to_mediamtx)...")
            ok = self._zed_recorder.start_streaming()
            if not ok:
                self._log("Video streaming failed to start (check mediamtx / GStreamer / ZED).")
            else:
                QTimer.singleShot(7000, self._verify_zed_publishing)
        self._update_stream_button()

    def _update_stream_button(self):
        """Sync the streaming toggle button to the pipeline's actual state."""
        on = self._zed_recorder.is_streaming
        self._btn_stream_toggle.setText(f"📹  Video Streaming: {'ON' if on else 'OFF'}")
        self._btn_stream_toggle.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; padding: 6px; }" if on
            else "QPushButton { padding: 6px; }")

    def _start_zed_recording(self):
        side = self._zed_localize_side()
        paths = self._zed_recorder.start_recording(self._session_label, side=side)
        if paths:
            # Record the video start epoch + filenames + mode so the viewer / dataset
            # builder can align the video and crop the correct single eye (master=left,
            # peer=right) from the SBS stereo frame.
            meta = {"zed_record_start_epoch": time.time(),
                    "zed_recording_sbs": bool(self._zed_recorder.sbs),
                    "zed_localize_side": side}
            for lbl, p in paths.items():
                meta[f"zed_{lbl}"] = os.path.basename(p)   # zed_stereo / zed_left / zed_right
            self._session_recorder.update_meta(**meta)
            names = ", ".join(Path(p).name for p in paths.values())
            mode = ("SBS stereo — dataset crops the "
                    f"{side.upper()} half" if self._zed_recorder.sbs
                    else f"{side.upper()} eye only")
            self._log(f"ZED recording ({mode}): {names}")
        else:
            self._log("ZED recording could not start (pipeline not ready).")

    # ── 10 s session summary (peak latency) ───────────────────────────────────
    # Motor temperatures are NOT logged here — they are already recorded
    # per-frame in the raw CSV (temp_torso/left/right + the Inspire-hand TEMP
    # registers), so this stays a peak-latency-only log.
    _SUMMARY_FIELDS = [
        "timestamp", "t_elapsed_s",
        "peak_ik_period_ms", "peak_ik_compute_ms", "peak_pose_interval_ms",
    ]

    def _open_summary_log(self, session_dir: str, label: str):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{label}" if label else ""
            path = os.path.join(session_dir, f"session_summary{suffix}_{ts}.csv")
            self._summary_file = open(path, "w", newline="")
            self._summary_writer = csv.DictWriter(self._summary_file,
                                                  fieldnames=self._SUMMARY_FIELDS)
            self._summary_writer.writeheader()
            self._summary_file.flush()
            self._summary_start_mono = time.monotonic()
            self._log(f"Session summary: {Path(path).name} "
                      f"(peak latency every 10 s)")
        except Exception as exc:
            self._log(f"Could not open session summary log: {exc}")
            self._summary_file = None
            self._summary_writer = None

    def _close_summary_log(self):
        if self._summary_file is not None:
            try:
                self._on_summary_tick()   # capture a final row
                self._summary_file.flush()
                self._summary_file.close()
            except Exception:
                pass
        self._summary_file = None
        self._summary_writer = None

    def _on_summary_tick(self):
        """Write one summary row: peak latency over the last 10 s. No-op unless a
        recording is active."""
        if not self._recording_active or self._summary_writer is None:
            return
        pk = self._latency_tracker.peaks(10.0)

        def _r(v, n=2):
            return round(v, n) if v is not None else ""

        row = {
            "timestamp": round(time.time(), 3),
            "t_elapsed_s": round(time.monotonic() - self._summary_start_mono, 1),
            "peak_ik_period_ms":     _r(pk[0]),
            "peak_ik_compute_ms":    _r(pk[1]),
            "peak_pose_interval_ms": _r(pk[2]),
        }
        try:
            self._summary_writer.writerow(row)
            self._summary_file.flush()
        except Exception:
            pass

    # ── Per-take feedback-event log (condition delivery, provable from data) ───
    # A tiny CSV that timestamps every audio/haptic/viewport modality change so you can
    # prove which feedback condition was active during each take WITHOUT recording the
    # Quest headset (which would cost teleop framerate). The discrete in-headset buzzes
    # themselves are reconstructable offline from the recorded tactile .bin + thresholds;
    # this log captures the modality state (which is the manipulated study variable).
    _FEEDBACK_FIELDS = ["timestamp", "t_take_s", "event", "enabled", "source"]

    def _open_feedback_log(self, session_dir: str, label: str):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{label}" if label else ""
            path = os.path.join(session_dir, f"feedback_events{suffix}_{ts}.csv")
            self._feedback_file = open(path, "w", newline="")
            self._feedback_writer = csv.DictWriter(self._feedback_file,
                                                   fieldnames=self._FEEDBACK_FIELDS)
            self._feedback_writer.writeheader()
            self._feedback_start_mono = time.monotonic()
            # Snapshot the active modalities at take start so the log is self-contained.
            for event, enabled in (("audio_haptics", self._audio_haptics_on),
                                   ("vibrotactile", self._vibrotactile_on),
                                   ("viewport", self._viewport_on)):
                self._log_feedback_event(event, enabled, "take_start")
            self._feedback_file.flush()
            self._log(f"Feedback events: {Path(path).name} "
                      "(audio/haptic/viewport on-off, timestamped)")
        except Exception as exc:
            self._log(f"Could not open feedback-event log: {exc}")
            self._feedback_file = None
            self._feedback_writer = None

    def _close_feedback_log(self):
        if getattr(self, "_feedback_file", None) is not None:
            try:
                self._feedback_file.flush()
                self._feedback_file.close()
            except Exception:
                pass
        self._feedback_file = None
        self._feedback_writer = None

    def _log_feedback_event(self, event: str, enabled: bool, source: str):
        """Append one feedback-modality change to the current take's log (no-op when not
        recording a take). Called from the _apply_*_ui choke points, so it captures changes
        from BOTH the manager button (source='manager') and the in-VR operator toggle
        (source='operator_vr'); the take-start snapshot uses source='take_start'."""
        w = getattr(self, "_feedback_writer", None)
        if w is None:
            return
        try:
            w.writerow({
                "timestamp": round(time.time(), 3),
                "t_take_s": round(time.monotonic() - self._feedback_start_mono, 3),
                "event": event,
                "enabled": int(bool(enabled)),
                "source": source,
            })
            self._feedback_file.flush()
        except Exception:
            pass

    def _finalize_recording(self):
        """Finalize + save the CURRENT attempt's files (body pose, robot CSV + tactile,
        latency summary, summary log, ZED recording, crate, base). Leaves the ZED
        STREAMING pipeline UP so recording can be reopened in the next subfolder without
        a camera restart. Shared by Stop and by each trial/attempt rotate."""
        if self._device is not None and getattr(self._device, 'record_data', False):
            self._device.configure_recording(
                output_dir=self._device.output_dir, record_data=False, started=False)

        # Persist the final known mass + tare baseline, then finalize the raw CSV.
        mass_g = self._wrench_mass_spin.value()
        self._session_recorder.update_meta(
            known_mass_g=mass_g if mass_g > 0 else None,
            wrench_baseline=(self._wrench_worker.baseline_dict()
                             if self._wrench_worker is not None else {}),
        )
        self._session_recorder.stop()

        # Persist the end-to-end latency summary for this attempt (into its folder).
        try:
            summ = self._latency_e2e.summary()
            summ["estimated_constants_ms"] = {
                "capture_encode": self._args.lat_capture_encode_ms,
                "display_photon": self._args.lat_display_ms,
                "tracking": self._args.lat_tracking_ms,
                "actuation_fallback": self._args.lat_actuation_ms,
            }
            out = os.path.join(self._last_session_dir, "latency_summary.json")
            with open(out, "w") as f:
                json.dump(summ, f, indent=2)
        except Exception as exc:
            self._log(f"Latency summary save error: {exc}")

        self._close_summary_log()
        self._close_feedback_log()

        if self._zed_recorder.is_recording:
            self._zed_recorder.stop_recording()   # recording only — streaming stays up

        if self._apriltag_tracker is not None:
            try:
                self._apriltag_tracker.stop_recording()
            except Exception as exc:
                self._log(f"[crate] stop_recording error: {exc}")

        if self._headcam_localizer is not None:
            try:
                self._headcam_localizer.stop_recording()
            except Exception as exc:
                self._log(f"[base] stop_recording error: {exc}")
        if self._transcriber is not None:
            # Attempt ended → STOP saving transcript (setup/between-attempt speech isn't saved).
            self._transcriber.set_recording(False)
            self._transcriber.set_take("between")
        self._log(f"Saved {self._session_label}.")

    def _on_stop_recording(self):
        """Stop Session: finalize the current take (if any), SAVE everything not already
        saved, and END the session. Works whether a take is recording or the session is
        merely ARMED between takes. Confirmation-guarded (ending a session mid-participant
        by accident would fragment their data); the remote side executes without a second
        dialog (synced). When linked, stops BOTH robots' sessions."""
        if not self._session_open():
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Stop Session")
        box.setText("End this participant's session?\n\nThis finalizes and saves "
                    + ("the take currently recording and " if self._recording_active else "")
                    + "all session files, closes the session"
                    + (" on BOTH robots, and the peer auto-sends its data to the master"
                       if self._linked() else "")
                    + ".\n\nYou can't add more attempts to it afterwards.")
        end = box.addButton("End session && save", QMessageBox.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)   # accidental double-click / Enter lands on the safe side
        box.exec_()
        if box.clickedButton() is not end:
            self._log("[session] stop cancelled.")
            return
        self._sync_or_local("stop_recording", {}, self._local_stop_recording)

    def _local_stop_recording(self):
        """Finalize + end the recording session locally (no peer broadcast), then open
        the session folder in the file manager so the data is right there. On the PEER,
        automatically ship the finished session to the master (no manual button). Safe to
        call from the ARMED state (no take open) — it just closes the session."""
        if not self._session_open():
            return
        root = self._session_root
        if self._recording_active:
            self._finalize_recording()   # synchronous — all files (incl. ZED mp4s) complete
            self._recording_active = False
        if self._practice_active:
            self._local_practice_stop()
        if self._transcriber is not None:
            self._transcriber.stop_session()
        self._session_root = None
        self._session_id = None
        # Clean end: nothing to resume any more.
        try:
            os.remove(self._study_state_path())
        except Exception:
            pass
        self._update_buttons()
        self._open_in_file_manager(root)
        # Peer: auto-send the just-finished (fully finalized) session to the master.
        if self._link_role() == "peer":
            self._log("[twin] recording finished — auto-sending this session to the master…")
            self._on_send_data_to_master()

    def _open_in_file_manager(self, path):
        """Open `path` in the OS file manager (best-effort; logs on failure)."""
        if not path or not os.path.isdir(path):
            return
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(path)))
            self._log(f"Opened recordings folder: {path}")
        except Exception as exc:
            self._log(f"Could not open recordings folder ({path}): {exc}")

    # ── Crate AprilTag tracker ────────────────────────────────────────────────

    def _on_toggle_crate_tracking(self):
        """Start the crate tracker (open the cameras) or stop it (release them).
        Driven by the Start/Stop Crate Tracking button — no CLI flag needed."""
        if self._apriltag_tracker is None:
            self._init_apriltag_tracker()
        else:
            self._log("[crate] stopping tracker (cameras released).")
            try:
                self._apriltag_tracker.stop()
            except Exception:
                pass
            self._apriltag_tracker = None
            self._update_buttons()

    def _ensure_native_logs_silenced(self):
        """Mute the AprilTag C pose-solver's 'more than one new minima found.' spam on
        fd 1/2 (see apriltag_probe.silence_native_logs). Called ONCE, BEFORE any detector
        starts (head-cam localizer or crate tracker), so the console never floods. Safe:
        Python print/tracebacks are preserved; only native fd writes go to /dev/null."""
        if getattr(self, "_native_silenced", False):
            return
        try:
            from apriltag_probe import silence_native_logs
            silence_native_logs()
            self._native_silenced = True
        except Exception:
            pass

    def _init_apriltag_tracker(self):
        """Lazy-construct + start the crate tracker (Start Crate Tracking button).
        Fully guarded: a missing dep / bundle / camera just logs and leaves crate
        tracking off — it never prevents the manager from running."""
        self._apriltag_starting = True
        self._log("[crate] starting tracker (separate process) — opening cameras...")
        QApplication.processEvents()
        self._ensure_native_logs_silenced()   # mute the native "more than one minima" spam
        try:
            from apriltag_tracker import list_devices
        except Exception as exc:
            self._log(f"[crate] tracking unavailable (import failed): {exc}")
            self._apriltag_starting = False
            self._update_buttons()
            return
        bundle = self._args.apriltag_bundle
        if not os.path.exists(bundle):
            self._log(f"[crate] bundle not found: {bundle} — run Recalibrate Crate "
                      "first. Crate tracking off.")
            self._apriltag_starting = False
            self._update_buttons()
            return
        try:
            serials = self._args.apriltag_serials or [s for _, s in list_devices()][:2]
            include_head = self._include_headcam()
            if not serials and not include_head:
                self._log("[crate] no RealSense cameras found — crate tracking off.")
                self._apriltag_starting = False
                self._update_buttons()
                return
            # The tracker runs in its OWN process (own GIL) so its AprilTag detection can
            # never starve the teleop IK thread. It opens the RealSense cameras and, when
            # requested, THIS robot's head cam (over the Jetson stream) as a 3rd crate
            # camera — all inside that subprocess. We just hand it the config + world frame.
            headcam = None
            if include_head:
                # Full-res detection for the head cam (its own knob, default 1.0) — it views
                # table tag 0 from farther than the close-in RealSense crate cams, so the
                # shared 2.0 decimation loses the tag. Matches the head-cam localizer (1.0).
                hc_qd = float(getattr(self._args, "apriltag_headcam_quad_decimate", 1.0))
                headcam = {"host": self._headcam_host(), "port": self._headcam_port(),
                           "tag_size": self._args.table_tag_size, "quad_decimate": hc_qd}
                self._log(f"[crate] including G1 head cam ({headcam['host']}:"
                          f"{headcam['port']}) as a 3rd crate camera "
                          f"(detect quad_decimate={hc_qd:g}).")
            init = {
                "serials": list(serials), "bundle": os.path.abspath(bundle),
                "world_from_table": self._world_from_table().tolist(),
                "fps": int(getattr(self._args, "apriltag_fps", 15)),
                "quad_decimate": float(getattr(self._args, "apriltag_quad_decimate", 2.0)),
                "rate_hz": float(getattr(self._args, "apriltag_rate", 15.0)),
                "tag_size": None, "headcam": headcam,
            }
            proxy = AprilTagTrackerProxy(init, log_fn=self._log)
            proxy.start()          # launches the subprocess + waits for its 'hello'
            self._apriltag_tracker = proxy
            ncams = len(serials) + (1 if headcam else 0)
            self._log(f"[crate] tracker started (SEPARATE PROCESS) on {ncams} camera(s) "
                      f"(tag {proxy.tag_size * 1000:.0f}mm, anchor id "
                      f"{proxy.anchor_id}). Anchoring to table tag 0 happens "
                      "automatically (Prepare Study / Start Recording) — keep the tag "
                      "visible to ALL cameras; 'Anchor Cameras' is only the manual redo.")
        except Exception as exc:
            self._log(f"[crate] tracker failed to start: {exc}")
            self._apriltag_tracker = None
        self._apriltag_starting = False
        self._update_buttons()

    def _on_anchor_cameras(self):
        """One-time table-tag anchor: capture each camera's world pose. Blocks the
        GUI briefly (~1.5 s) by design — a deliberate one-shot calibration."""
        if self._apriltag_tracker is None:
            self._log("[crate] tracker not running.")
            return
        self._log("[crate] anchoring — keep the table tag (id 0) visible to BOTH "
                  "cameras...")
        self._btn_anchor_cameras.setEnabled(False)
        QApplication.processEvents()   # repaint before the blocking capture
        try:
            results = self._apriltag_tracker.anchor_cameras()
        except Exception as exc:
            self._log(f"[crate] anchor failed: {exc}")
            self._update_buttons()
            return
        for serial, (status, n) in results.items():
            if status == "ok":
                self._log(f"[crate]   {serial}: anchored ({n} table obs).")
            else:
                self._log(f"[crate]   {serial}: NO TABLE TAG — reposition so id 0 "
                          "is visible, then re-anchor.")
        if self._apriltag_tracker.is_anchored:
            self._log("[crate] ready — crate pose will record in the table frame.")
        else:
            self._log("[crate] NOT anchored — crate pose can't be placed in the "
                      "world frame yet.")
        self._update_buttons()

    def _on_reanchor_bumped_camera(self):
        """Re-anchor ONE bumped crate camera to table tag 0 without disturbing the
        others. Safe DURING recording — tag 0 is the fixed world origin, so re-anchoring
        corrects the moved camera's extrinsic without shifting the world frame."""
        tr = self._apriltag_tracker
        if tr is None:
            self._log("[crate] start crate tracking first — no cameras to re-anchor.")
            return
        serials = [c.serial for c in tr.cams]
        if not serials:
            self._log("[crate] no crate cameras to re-anchor.")
            return
        # Pick the bumped camera (or all) when there's more than one.
        pick = None                                   # None → all cameras
        if len(serials) > 1:
            items = ["All crate cameras"] + serials
            item, ok = QInputDialog.getItem(
                self, "Re-anchor bumped camera",
                "Point the bumped camera back at table tag 0, then pick it "
                "(others keep their anchors):", items, 0, False)
            if not ok:
                return
            if item != "All crate cameras":
                pick = [item]
        if self._session_open():
            self._log("[crate] re-anchoring MID-SESSION — crate frames recorded before "
                      "now may be off (camera had moved); frames after use the corrected "
                      "anchor.")
        self._log("[crate] re-anchoring — keep table tag 0 visible to the chosen "
                  "camera(s)…")
        self._btn_reanchor_bump.setEnabled(False)
        QApplication.processEvents()                  # repaint before the blocking capture
        try:
            results = tr.anchor_cameras(serials=pick, keep_on_fail=True)
        except Exception as exc:                      # noqa: BLE001
            self._log(f"[crate] re-anchor failed: {exc}")
            self._update_buttons()
            return
        for serial, (status, n) in results.items():
            if status == "ok":
                self._log(f"[crate]   {serial}: RE-ANCHORED ({n} table obs).")
            elif status == "kept":
                self._log(f"[crate]   {serial}: no tag 0 seen — kept its previous anchor.")
            else:
                self._log(f"[crate]   {serial}: NO TABLE TAG and no prior anchor — still "
                          "unanchored; reposition so id 0 is visible and retry.")
        self._update_buttons()

    def _on_toggle_crate_contribution(self):
        """PEER: start/stop feeding this robot's head-cam crate observations to the
        master. Turning it on ensures the head-cam localizer is running, loads the crate
        bundle, and anchors the base to table tag 0."""
        loc = self._headcam_localizer
        if loc is None:
            self._init_headcam_localizer()          # crate contribution rides the localizer
            loc = self._headcam_localizer
            if loc is None:
                self._log("[crate] start Head-Cam Localize first (set the Jetson IP) — "
                          "crate contribution uses the head camera.")
                return
        if getattr(loc, "crate_contributing", False):
            loc.disable_crate_contribution()
            self._log("[crate] stopped contributing crate tracking to the master.")
            self._update_buttons()
            return
        bundle = self._args.apriltag_bundle
        if not os.path.exists(bundle):
            self._log(f"[crate] crate bundle not found: {bundle} — scp crate_bundle.json "
                      "from the master so this robot can place the crate tags.")
            return
        ok, msg = loc.enable_crate_contribution(bundle)
        self._log(f"[crate] {msg}")
        if not ok:
            return
        self._on_anchor_contribution_base()         # lock the base pose to tag 0
        self._update_buttons()

    def _on_anchor_contribution_base(self):
        """Capture + lock this robot's base pose from table tag 0 for crate contribution
        (aim the head at tag 0). One-time — the base is fixed."""
        loc = self._headcam_localizer
        if loc is None:
            self._log("[crate] start Head-Cam Localize first.")
            return
        self._log("[crate] anchoring base — keep table tag 0 in the head cam's view…")
        self._btn_contrib_anchor.setEnabled(False)
        QApplication.processEvents()
        try:
            ok, n = loc.anchor_base()
        except Exception as exc:                     # noqa: BLE001
            self._log(f"[crate] base anchor failed: {exc}")
            self._update_buttons()
            return
        if ok:
            self._log(f"[crate] base anchored ({n} tag-0 views) — now streaming crate "
                      "poses to the master whenever crate tags are in view.")
        else:
            self._log(f"[crate] base anchor FAILED ({n} tag-0 views) — aim the head at "
                      "table tag 0 and retry.")
        self._update_buttons()

    def _on_recalibrate_crate(self):
        """Re-learn the crate's tag bundle. Stops the tracker to free the cameras,
        then launches apriltag_crate_calibrate.py in its own window (same pattern
        as the session viewer). When it finishes, the operator clicks Start Crate
        Tracking to reload the new crate_bundle.json."""
        if self._session_open():
            self._log("[crate] stop recording before recalibrating.")
            return
        if self._calib_proc is not None and self._calib_proc.poll() is None:
            self._log("[crate] calibration window already open.")
            return
        # The calibration tool needs exclusive access to the cameras.
        if self._apriltag_tracker is not None:
            self._log("[crate] stopping tracker to free cameras for calibration.")
            try:
                self._apriltag_tracker.stop()
            except Exception:
                pass
            self._apriltag_tracker = None
        script = _HERE / "apriltag_crate_calibrate.py"
        if not script.exists():
            self._log(f"[crate] calibration script not found: {script}")
            self._update_buttons()
            return
        cmd = [sys.executable, str(script), "--out", str(self._args.apriltag_bundle)]
        if self._args.apriltag_serials:
            cmd += ["--serials", *self._args.apriltag_serials]
        try:
            self._calib_proc = subprocess.Popen(cmd, cwd=str(_HERE))
            self._log("[crate] calibration window launched. Rotate the crate "
                      "(SPACE capture, s save, q quit), then click Start Crate "
                      "Tracking to reload the new bundle.")
        except Exception as exc:
            self._log(f"[crate] could not launch calibration: {exc}")
        self._update_buttons()

    def _world_from_table(self):
        """T_world_table: maps the detected tag-0 frame to the Z-up world. A 180°
        flip about x when tag 0's z points into the table (--table-z-down). Shared
        by the crate tracker AND the head-cam localizer so robot + crate land in one
        consistent world."""
        T = np.eye(4)
        if getattr(self._args, "table_z_down", False):
            T[1, 1] = -1.0
            T[2, 2] = -1.0
        return T

    # ── Robot base self-localization (head camera) ────────────────────────────

    def _read_waist_joints(self):
        """Live waist (yaw, roll, pitch) from hardware, for the head-cam FK chain.
        Returns a (3,) array or None — the localizer assumes zero waist if None."""
        hw = self._hw_controller
        if hw is None:
            return None
        try:
            ms = hw.low_state.motor_state
            return np.array([ms[i].q for i in hw._torso_idx], dtype=np.float64)
        except Exception:
            return None

    def _on_toggle_headcam(self):
        """Start head-cam base localization (open head camera) or stop it."""
        if self._headcam_localizer is None:
            self._init_headcam_localizer()
        else:
            self._log("[base] stopping head-cam localization (camera released).")
            try:
                self._headcam_localizer.stop()
            except Exception:
                pass
            self._headcam_localizer = None
            self._update_buttons()

    def _init_headcam_localizer(self):
        """Lazy-construct + start the head-cam localizer. Fully guarded: a missing
        dep / camera just logs and leaves localization off."""
        self._headcam_starting = True
        self._log("[base] starting head-cam localization — opening head camera...")
        QApplication.processEvents()
        self._ensure_native_logs_silenced()   # BEFORE detection starts — no console flood
        try:
            from head_cam_localize import HeadCamLocalizer
        except Exception as exc:
            self._log(f"[base] localization unavailable (import failed): {exc}")
            self._headcam_starting = False
            self._update_buttons()
            return
        host, port = self._headcam_host(), self._headcam_port()
        if not host:
            self._log("[base] set the head-cam (Jetson) IP in the Setup bar. Off.")
            self._headcam_starting = False
            self._update_buttons()
            return
        try:
            loc = HeadCamLocalizer(
                host, port, model_path=str(_XML), waist_getter=self._read_waist_joints,
                table_tag_id=0, tag_size=self._args.table_tag_size,
                optical_convention=self._args.headcam_optical,
                world_from_table=self._world_from_table(),
                rate_hz=float(getattr(self._args, "headcam_rate", 8.0)),
                debug_view=True,        # Head Cam View streams annotated JPEGs on demand
                # Decode+detect in a SUBPROCESS (own GIL) — per-frame vision work inside
                # the manager process starves the teleop IK thread (multi-second lag).
                out_of_process=True)
            loc.start()
            self._headcam_localizer = loc
            self._log(f"[base] head-cam localization started (Jetson {host}:{port}, "
                      f"optical={self._args.headcam_optical}). Make sure "
                      "g1_headcam_server.py is running there, and aim the head at the "
                      "table tag (id 0). 'up' near +1.0 means the convention is right.")
        except Exception as exc:
            self._log(f"[base] head-cam localization failed to start: {exc}")
            self._headcam_localizer = None
        self._headcam_starting = False
        self._update_buttons()
        # If this peer is meant to auto-contribute and the bundle is already here, engage it
        # now that the head cam is up. (Guarded against re-entry when the call chain started
        # from _auto_enable_crate_contribution itself.)
        if self._headcam_localizer is not None:
            self._auto_enable_crate_contribution("head cam started")

    def _on_show_headcam_view(self):
        """Open the live head-camera debug view (AprilTag overlay) so the operator
        can confirm the table tag is in frame."""
        if self._headcam_localizer is None:
            self._log("[base] start Head-Cam Localize first, then open the view.")
            return
        if self._headcam_view is None:
            self._headcam_view = HeadCamViewWindow(
                lambda: (self._headcam_localizer.debug_frame()
                         if self._headcam_localizer is not None else None))
        self._headcam_view.show()
        self._headcam_view.raise_()
        self._headcam_view.activateWindow()

    # ── Cross-machine study link / master control board ───────────────────────

    def _on_connect_link(self):
        """Persist the Setup-bar values and (re)start the link with them."""
        self._on_setup_changed()
        self._init_link()

    def _on_disconnect_link(self):
        """Tear down the peer link (stops the receive/retry thread) and stop syncing. The
        Data Collection + Teleop buttons revert to single-robot. Use Connect Link to
        re-establish it."""
        if self._link is not None:
            try:
                self._link.stop()
            except Exception:
                pass
            self._link = None
        self._link_was_connected = False
        self._update_buttons()          # reverts the (BOTH) labels + link light
        self._log("[link] disconnected (reconnect with Connect Link).")

    def _init_link(self):
        """(Re)start the StudyLink (over Tailscale) from the current UI/config role + host.
        Control is symmetric — once connected, EITHER machine's Data Collection + Teleop
        buttons drive both robots (labels gain "(BOTH)"). Guarded."""
        # tear down any existing link first (supports reconnect)
        if self._link is not None:
            try:
                self._link.stop()
            except Exception:
                pass
            self._link = None

        role = self._link_role()
        if role == "none":
            self._update_buttons()
            self._log("[link] role = None (standalone, no link).")
            return
        try:
            from study_link import StudyLink
        except Exception as exc:
            self._log(f"[link] unavailable (import failed): {exc}")
            return
        # Direction is set by the Link-IP field, independent of master/peer:
        # blank  -> THIS machine listens (it has the reachable/open IP);
        # filled -> connect to that address (the OTHER machine listens).
        link_ip = self._link_host()
        listen = not link_ip
        host = link_ip or "0.0.0.0"
        retry_secs = float(getattr(self._args, "link_retry_secs", 3.0))
        try:
            self._link = StudyLink(role, host, self._args.link_port, listen=listen,
                                   robot_id=self._robot_id,
                                   on_candidate=self._on_remote_candidate,
                                   retry_secs=retry_secs)
            self._link.start()
        except Exception as exc:
            self._log(f"[link] failed to start: {exc}")
            self._link = None
            self._update_buttons()
            return
        # Reset the connection-transition tracker so the first connect logs "connected".
        self._link_was_connected = False
        where = (f"listening on :{self._args.link_port}" if listen
                 else f"connecting to {host}:{self._args.link_port}")
        self._update_buttons()          # applies the (BOTH) labels + link light
        self._log(f"[link] {role.upper()} ({where}). Auto-retrying every {retry_secs:g}s "
                  f"until the peer is up. Once connected, the Data Collection + Teleop "
                  f"buttons drive BOTH robots (labels show '(BOTH)').")

    def _on_remote_candidate(self, msg):
        """Inject robot 2's head-cam crate candidate into our fusion (master side).
        Called on the link's receive thread; add_remote_candidate is thread-safe."""
        tr = self._apriltag_tracker
        if tr is not None:
            try:
                tr.add_remote_candidate(msg["p"], msg["q"],
                                        msg.get("weight", 1.0), msg.get("stamp"))
            except Exception:
                pass

    def _on_link_tick(self):
        """10 Hz on the GUI thread: drain remote commands; push candidates+status;
        and log connect/disconnect transitions (the link retries on its own thread)."""
        if self._link is None:
            return
        while True:
            try:
                cmd = self._link.commands.get_nowait()
            except Exception:
                break
            self._handle_link_command(cmd)
        self._link_tick_n += 1
        # Edge-detect the connection state so the user sees exactly when the peer
        # comes up (after booting the other machine) or drops. Retrying itself is
        # handled continuously by the StudyLink receive thread.
        connected = self._link.connected
        if connected != self._link_was_connected:
            self._link_was_connected = connected
            if connected:
                self._log("[link] peer CONNECTED.")
                # Master is authoritative on connect: push its trial ranking + attempt
                # so both managers start from the same shared state. (After that, EITHER
                # side's changes propagate.) Small delay so the link is fully ready.
                if self._link_role() == "master":
                    QTimer.singleShot(500, lambda: (
                        self._peer_cmd("set_trial_order", order=self._trial_order()),
                        self._peer_cmd("set_attempt", attempt=self._attempt_spin.value()),
                        self._send_bundle_to_peer()))   # auto-share the crate bundle
                # Peer: auto-verify the (now auto-identified) master link so a broken SSH
                # setup is caught at connect, not at the first Send. Delay ~1.8 s so the
                # master's first status (username + rec_dir) has arrived to resolve the dest.
                if self._link_role() == "peer":
                    QTimer.singleShot(1800, self._verify_master_link_async)
            else:
                retry_secs = float(getattr(self._args, "link_retry_secs", 3.0))
                self._log(f"[link] peer connection lost — auto-retrying every "
                          f"{retry_secs:g}s until it's back.")
        if not connected:
            return
        if self._link_role() == "peer":
            self._peer_send_candidate()
        # Heartbeat BOTH ways (~2 Hz) so the link never goes idle (fixes the
        # in/out clipping) and a dead connection is detected within the timeout.
        if self._link_tick_n % 5 == 0:
            self._peer_send_status()

    def _peer_send_candidate(self):
        """Stream this robot's crate observation to the master. Preferred source is the
        head-cam localizer's LIVE-pose crate candidate (waist-compensated, so it's valid
        even as the peer is teleoperated); falls back to a local crate tracker if one is
        running (e.g. a robot with its own external cameras)."""
        res = None
        loc = self._headcam_localizer
        if loc is not None and getattr(loc, "crate_contributing", False):
            try:
                res = loc.latest_crate()
            except Exception:
                res = None
        if res is None and self._apriltag_tracker is not None:
            res = self._apriltag_tracker.latest()
        if res and res.get("ok"):
            self._link.send_candidate(
                res["p"], res["q"],
                weight=max(1.0, res.get("n_tags", 1) * 10.0),
                stamp=res.get("stamp", time.time()), n_tags=res.get("n_tags", 0))

    # ── Clock-sync quality (for tight bimanual time alignment at training time) ──
    def _ntp_sync_info(self):
        """Best-effort NTP sync quality for THIS machine (chrony preferred). Tells you
        whether the two robots' wall clocks can be trusted for alignment."""
        out = {"synced": None}
        try:
            r = subprocess.run(["chronyc", "-n", "tracking"], capture_output=True,
                               text=True, timeout=2.0)
            if r.returncode == 0:
                out["ntp_tool"] = "chrony"
                for line in r.stdout.splitlines():
                    if line.startswith("System time"):
                        v = float(line.split(":", 1)[1].split()[0])
                        out["chrony_system_time_s"] = (-v if "slow" in line else v)
                    elif line.startswith("Leap status"):
                        out["synced"] = ("Normal" in line)
                    elif line.startswith("Reference ID"):
                        out["ntp_source"] = line.split(":", 1)[1].strip()
                return out
        except Exception:
            pass
        try:
            r = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized",
                                "--value"], capture_output=True, text=True, timeout=2.0)
            if r.returncode == 0:
                out["synced"] = (r.stdout.strip() == "yes")
                out["ntp_tool"] = "timedatectl"
        except Exception:
            pass
        return out

    def _measure_clock_sync(self):
        """Measure the master↔peer clock offset over the link (NTP-style, min-RTT) plus
        this machine's NTP quality → a dict for the session meta. Blocks ~0.6 s."""
        info = {"ntp": self._ntp_sync_info(), "measured_epoch": round(time.time(), 3)}
        if self._link is not None and self._link.connected:
            self._link.reset_clock_estimate()
            for _ in range(24):
                self._link.ping_clock()
                time.sleep(0.025)
            est = self._link.clock_estimate()
            if est.get("offset") is not None:
                # to convert the OTHER robot's timestamps to this machine's clock:
                #   t_local = t_peer - link_peer_minus_local_offset_s
                info["link_peer_minus_local_offset_s"] = round(est["offset"], 6)
                info["link_rtt_s"] = round(est["rtt"], 6)
                info["link_samples"] = est["n"]
        return info

    def _refresh_clock_sync_meta(self):
        """Re-measure the clock sync shortly AFTER the synced start (so the measurement's
        ~0.6 s block never delays the shared-epoch recording start) and stamp it in."""
        if not self._recording_active:
            return
        self._clock_sync_info = self._measure_clock_sync()
        try:
            self._session_recorder.update_meta(clock_sync=self._clock_sync_info)
        except Exception:
            pass
        off = self._clock_sync_info.get("link_peer_minus_local_offset_s")
        if off is not None:
            self._log(f"[clock] peer↔local offset {off * 1000:+.1f} ms "
                      f"(rtt {self._clock_sync_info.get('link_rtt_s', 0) * 1000:.1f} ms) — "
                      "logged to meta.")

    def _zed_localize_side(self):
        """Which ZED camera to use for AprilTag localization. --zed-localize-side wins;
        otherwise 'right' for the PEER (it faces the table from the opposite side, so the
        table tag sits to its right and its RIGHT camera sees it more reliably) and 'left'
        otherwise."""
        s = getattr(self._args, "zed_localize_side", "auto")
        if s in ("left", "right"):
            return s
        return "right" if self._link_role() == "peer" else "left"

    def _zed_intrinsics(self, side="left"):
        """ZED factory intrinsics (fx,fy,cx,cy) for camera ``side`` ('left'/'right') at the
        recording resolution, parsed from the ZED SN<serial>.conf — stamped into the meta
        so zed_localize.py needs no manual --fx/... Reads the conf (not the live camera →
        no conflict) via the resolution in zed_to_mediamtx.sh + ZED_SETTINGS_PATH/--zed-conf.
        {} if it can't be resolved."""
        import re as _re, glob as _glob, configparser as _cfgp
        # zed_to_mediamtx.sh RESOLUTION enum → (conf section tag, width, height)
        res_map = {"0": ("2K", 2208, 1242), "1": ("FHD", 1920, 1080),
                   "2": ("FHD", 1920, 1200), "3": ("HD", 1280, 720),
                   "5": ("VGA", 672, 376)}
        res = "3"
        try:
            m = _re.search(r'^RESOLUTION="?(\d+)"?', _ZED_SCRIPT.read_text(), _re.M)
            if m:
                res = m.group(1)
        except Exception:
            pass
        tag, w, h = res_map.get(res, ("HD", 1280, 720))
        cands = []
        if getattr(self._args, "zed_conf", None):
            cands.append(self._args.zed_conf)
        for d in (os.environ.get("ZED_SETTINGS_PATH"), "/usr/local/zed/settings"):
            if d:
                cands += sorted(_glob.glob(os.path.join(os.path.expanduser(d), "SN*.conf")))
        conf = next((c for c in cands if c and os.path.exists(c)), None)
        if conf is None:
            return {}
        cam = "RIGHT_CAM" if side == "right" else "LEFT_CAM"
        try:
            cp = _cfgp.ConfigParser()
            cp.read(conf)
            sec = f"{cam}_{tag}"
            if sec not in cp:
                alt = [s for s in cp.sections() if s.startswith(f"{cam}_")]
                if not alt:
                    return {}
                sec = alt[0]
            g = cp[sec]
            sn = _re.search(r"SN(\d+)", os.path.basename(conf))
            return {"fx": float(g["fx"]), "fy": float(g["fy"]),
                    "cx": float(g["cx"]), "cy": float(g["cy"]),
                    "width": w, "height": h, "resolution": tag, "side": side,
                    "conf": os.path.basename(conf),
                    "serial": sn.group(1) if sn else None}
        except Exception:
            return {}

    def _check_zed_calibration(self):
        """Verify THIS machine's ZED factory-calibration conf is present (needed to auto-
        write intrinsics into the recording meta). Logs a clear result; returns True if
        found. Run at startup + in Prepare Study, so the peer flags a missing conf itself."""
        side = self._zed_localize_side()
        self._zed_intrinsics_meta = self._zed_intrinsics(side)   # fresh read (updates the light)
        zi = self._zed_intrinsics_meta
        if zi:
            self._log(f"[zed] calibration OK — {zi['conf']} (SN{zi.get('serial', '?')}, "
                      f"{zi['resolution']}, {side.upper()} cam); intrinsics → meta.")
            return True
        self._log("[zed] ⚠ NO ZED calibration conf (SN<serial>.conf) found in "
                  "$ZED_SETTINGS_PATH or /usr/local/zed/settings — the ZED intrinsics "
                  "WON'T be written to meta (zed_localize would need --fx/...). Plug this "
                  "ZED in with internet once (the SDK auto-downloads it) or copy its SN conf.")
        return False

    def _peer_send_status(self):
        base = self._headcam_localizer.latest() if self._headcam_localizer else None
        tr = self._apriltag_tracker
        self._link.send_status(
            teleop=self._state, recording=bool(self._recording_active),
            base_up=(round(base["upright"], 2) if (base and base.get("ok")) else None),
            # crate anchor state so the other machine can gate recording on it:
            # True/False when a tracker is running here, None when there's no rig here.
            crate_anchored=(bool(tr.is_anchored) if tr is not None else None),
            # full component status → the other manager's System Status lights.
            components=self._status_snapshot())

    def _handle_link_command(self, cmd):
        """Execute a command the OTHER manager sent us, on the GUI thread. Symmetric:
        either machine (master or peer) can send these to drive the other robot."""
        action = cmd.get("action")
        self._log(f"[link] remote command: {action}")
        # engage/pause are explicit (idempotent) so the two robots converge no matter
        # which side initiated; legacy "start_teleop" is treated as an engage.
        if action in ("engage_teleop", "start_teleop"):
            self._start_teleop_session()
        elif action == "pause_teleop":
            self._pause_local_teleop()
        elif action == "stop_teleop":
            self._on_stop_teleop()
        # Peer-driven Unity toggles: the other station flipped one of OUR toggles. Run our
        # own toggle handler so it updates our button + pushes to OUR Quest (a flip, never a
        # forced state — matches pressing the local button).
        elif action == "toggle_audio_haptics":
            self._on_toggle_audio_haptics()
        elif action == "toggle_vibrotactile":
            self._on_toggle_vibrotactile()
        elif action == "toggle_viewport":
            self._on_show_viewport()
        elif action == "open_session":
            pid = cmd.get("pid", "unknown")
            cond = cmd.get("condition", _TRIAL_CONDITIONS[0])
            attempt = int(cmd.get("attempt", 1))
            session_id = cmd.get("session_id")   # SHARED folder name from the initiator
            mc = cmd.get("master_code", "")      # operator codes assigned on the master
            pc = cmd.get("peer_code", "")
            # Auto-calibrate THIS robot's cameras + bring its recording subsystems up at
            # session open, so the first Start Attempt is instant.
            self._peer_preflight_on_start()
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              lambda: self._local_open_session(pid, cond, attempt,
                                                               session_id, mc, pc))
        elif action == "stop_attempt":
            self._schedule_at(cmd.get("at_epoch", time.time()), self._local_stop_attempt)
        elif action == "practice_start":
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              self._local_practice_start)
        elif action == "practice_stop":
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              self._local_practice_stop)
        elif action == "save_study":
            self._schedule_at(cmd.get("at_epoch", time.time()), self._local_save_study)
        elif action == "resume_study":
            _st = cmd.get("state") or {}
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              lambda: self._local_resume_study(_st))
        elif action == "start_recording":
            pid = cmd.get("pid", "unknown")
            cond = cmd.get("condition", _TRIAL_CONDITIONS[0])
            attempt = int(cmd.get("attempt", 1))
            session_id = cmd.get("session_id")   # SHARED folder name from the initiator
            mc = cmd.get("master_code", "")
            pc = cmd.get("peer_code", "")
            # Auto-calibrate THIS robot's cameras before the synced start (usually instant
            # when already anchored; only a cold start adds a brief delay).
            self._peer_preflight_on_start()
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              lambda: self._local_start_recording(pid, cond, attempt,
                                                                  session_id, mc, pc))
        elif action == "start_attempt":
            cond = cmd.get("condition", _TRIAL_CONDITIONS[0])
            attempt = int(cmd.get("attempt", 1))
            # Re-anchor this robot's cameras just like the initial start (cheap when
            # already anchored) so each take's crate/base pose is in the table frame.
            self._peer_preflight_on_start()
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              lambda: self._local_start_attempt(cond, attempt))
        elif action == "switch_recording":
            cond = cmd.get("condition", _TRIAL_CONDITIONS[0])
            attempt = int(cmd.get("attempt", 1))
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              lambda: self._local_switch_recording(cond, attempt))
        elif action == "stop_recording":
            self._schedule_at(cmd.get("at_epoch", time.time()), self._local_stop_recording)
        elif action == "redo_attempt":
            cond, att = cmd.get("condition"), cmd.get("attempt")
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              lambda: self._local_redo_attempt(cond, att))
        elif action == "redo_attempt_armed":
            cond, att = cmd.get("condition"), cmd.get("attempt")
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              lambda: self._local_redo_attempt_armed(cond, att))
        elif action == "redo_trial":
            cond = cmd.get("condition")
            self._schedule_at(cmd.get("at_epoch", time.time()),
                              lambda: self._local_redo_trial(cond))
        elif action == "set_attempt_quality":
            self._local_mark_attempt(cmd.get("quality", "unrated"))
        elif action == "set_trial_order":
            # Shared trial ranking — apply immediately (not epoch-scheduled).
            self._apply_trial_order(cmd.get("order", list(_TRIAL_CONDITIONS)))
        elif action == "set_attempt":
            self._apply_attempt(int(cmd.get("attempt", 1)))
        elif action == "calibrate_cameras":
            # Remote "Calibrate Cameras (BOTH)" — anchor THIS robot's cameras now.
            self._remote_calibrate()
        elif action == "calib_result":
            # The other robot reporting its calibrate outcome back to us.
            self._log(f"[calib] peer (robot {cmd.get('robot', '?')}): "
                      f"{cmd.get('text', '')}")
        elif action == "crate_bundle":
            # Master pushed the crate bundle — save it so this robot can place crate tags.
            self._receive_crate_bundle(cmd.get("data"))
        elif action == "prepare_study":
            # One-click "Prepare Study" from the other machine — prep THIS robot too.
            self._prepare_local()

    def _send_bundle_to_peer(self):
        """MASTER: push crate_bundle.json to the peer over the link so it can place the
        crate tags for its head-cam contribution (no manual scp)."""
        if not self._linked() or self._link_role() == "peer":
            return
        bundle = self._args.apriltag_bundle
        if not os.path.exists(bundle):
            return
        try:
            with open(bundle) as f:
                data = json.load(f)
        except Exception as exc:                     # noqa: BLE001
            self._log(f"[crate] couldn't read bundle to send: {exc}")
            return
        if self._peer_cmd("crate_bundle", data=data):
            self._log(f"[crate] pushed crate bundle to peer ({len(data.get('tags', {}))} tags).")

    def _receive_crate_bundle(self, data):
        """PEER: save a crate bundle received from the master, and reload it live if this
        robot is already contributing."""
        if not data:
            return
        path = self._args.apriltag_bundle
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:                     # noqa: BLE001
            self._log(f"[crate] couldn't save received bundle: {exc}")
            return
        self._log(f"[crate] received crate bundle from master → {os.path.basename(path)}")
        loc = self._headcam_localizer
        if loc is not None and getattr(loc, "crate_contributing", False):
            loc.enable_crate_contribution(path)      # hot-reload the new bundle
        else:
            # On by default: the bundle just arrived → engage contribution automatically so
            # the peer's head cam becomes the 4th crate camera with no manual step.
            self._auto_enable_crate_contribution("bundle received")

    def _schedule_at(self, epoch, fn):
        QTimer.singleShot(max(0, int((epoch - time.time()) * 1000)), fn)

    # -- dual "Both Robots" panel handlers (drive BOTH robots from either machine) --
    def _peer_cmd(self, action, **kw):
        """Send a command to the OTHER machine and log whether it went out (so a
        dropped link is visible instead of a silent no-op). Works from either side."""
        if self._link is None:
            self._log(f"[link] no link — '{action}' not sent.")
            return False
        ok = self._link.send_command(action, **kw)
        self._log(f"[link] '{action}' -> peer: {'sent' if ok else 'NOT sent (link down)'}")
        return ok

    def _peer_state(self):
        """(teleop_state_str, recording_bool) from the latest peer status."""
        st = self._link.get_peer_status() if self._link is not None else {}
        return st.get("teleop"), bool(st.get("recording"))

    def _pause_local_teleop(self):
        """The pause half of the single-robot Start/Pause toggle (freeze in place +
        flip the XR Play/Pause button), WITHOUT touching the link — so a remote
        'pause_teleop' pauses just this robot without echoing back."""
        if self._state == _S_TELEOP:
            self._pause_teleop("linked")
            self._send_unity_cmd("teleop_active", {"enabled": False})

    def _link_state(self):
        """(role_active, connected): role_active = a Master/Peer role is selected;
        connected = the peer link is actually up."""
        role_active = self._link_role() != "none"
        connected = self._link is not None and self._link.connected
        return role_active, connected

    def _both_suffix(self):
        """'  (BOTH)' when a peer link is up (the Data Collection + Teleop buttons drive
        both robots), else '' (they act on this robot only)."""
        return "  (BOTH)" if self._linked() else ""

    def _update_link_status(self):
        """Refresh the top-bar link light + short label, and the peer's detailed status
        line in the Data Collection panel. Grey = no link role, red = role set but peer
        down, green = connected (buttons drive BOTH robots)."""
        role_active, connected = self._link_state()
        color = (self._STATE_COLORS["green"] if connected
                 else self._STATE_COLORS["red"] if role_active
                 else self._STATE_UNKNOWN)
        self._link_light.setStyleSheet(f"color: {color}; font-size: 15px;")
        if not role_active:
            self._link_peer_lbl.setText("Link: off")
            self._link_peer_lbl.setStyleSheet("font-weight: bold; color: #7f8c8d;")
        elif connected:
            self._link_peer_lbl.setText("Link: BOTH robots")
            self._link_peer_lbl.setStyleSheet("font-weight: bold; color: green;")
        else:
            self._link_peer_lbl.setText("Link: connecting…")
            self._link_peer_lbl.setStyleSheet("font-weight: bold; color: #c0392b;")

        # MASTER: remember the peer's live IP so "Sync Peer Data" can pull LATER, even after
        # the peer manager is shut down. On the master the link listens, so remote_ip is the
        # PEER's address (on the peer it'd be the master's — hence the role guard).
        if connected and self._link_role() != "peer":
            rip = (getattr(self._link, "remote_ip", None) or "").strip()
            if rip and rip != (self._cfg.get("peer_host") or ""):
                self._cfg["peer_host"] = rip
                self._save_cfg()

        # Peer's detailed status line (Data Collection panel) — only while linked.
        self._link_peer_status_lbl.setVisible(connected)
        st = self._link.get_peer_status() if self._link is not None else {}
        if connected and st:
            pc = st.get("crate_anchored")
            crate_txt = "—" if pc is None else ("anchored" if pc else "NOT anchored")
            self._link_peer_status_lbl.setText(
                f"Peer robot {st.get('robot_id', '?')}: teleop={st.get('teleop', '?')}  "
                f"rec={'on' if st.get('recording') else 'off'}  "
                f"base_up={st.get('base_up', '—')}  crate={crate_txt}")
        elif connected:
            self._link_peer_status_lbl.setText("Peer: (no status yet)")

        # PEER: reflect the auto-identified master rsync destination in the top-bar label.
        if hasattr(self, "_master_dest_lbl") and self._link_role() == "peer":
            dest = self._master_rsync_dest()
            if dest:
                self._master_dest_lbl.setText(f"Master dest: {dest}")
                self._master_dest_lbl.setStyleSheet("font-size: 11px; color: #2980b9;")
            else:
                self._master_dest_lbl.setText("Master dest: (waiting for link)")
                self._master_dest_lbl.setStyleSheet("font-size: 11px; color: #7f8c8d;")

    # ── Unity commands ────────────────────────────────────────────────────────

    # Button UI styling shared by all three Unity-command toggles.
    _TOGGLE_STYLE_ON  = "QPushButton { background: #2980b9; color: white; padding: 8px; }"
    _TOGGLE_STYLE_OFF = "QPushButton { padding: 8px; }"

    # Dual "Both Robots" teleop toggle: green Start ↔ yellow Pause, mirroring the
    # single-robot Start/Pause button (same colours) but driving both robots.
    _BOTH_TELEOP_START_STYLE = "QPushButton { background: #2ecc71; color: white; font-weight: bold; padding: 8px; }"
    _BOTH_TELEOP_PAUSE_STYLE = "QPushButton { background: #f1c40f; color: black; font-weight: bold; padding: 8px; }"

    def _apply_audio_haptics_ui(self, val: bool, source: str = "manager"):
        """Reflect `val` on the Audio Haptics button. Pure UI — sends nothing. Logs the
        change to the per-take feedback-event log (source = who flipped it)."""
        self._audio_haptics_on = val
        self._btn_audio_haptics.setText(f"🔊  Audio Haptics: {'ON' if val else 'OFF'}")
        self._btn_audio_haptics.setStyleSheet(self._TOGGLE_STYLE_ON if val else self._TOGGLE_STYLE_OFF)
        self._log_feedback_event("audio_haptics", val, source)

    def _apply_vibrotactile_ui(self, val: bool, source: str = "manager"):
        """Reflect `val` on the Vibrotactile button. Pure UI — sends nothing. Logs to the
        per-take feedback-event log."""
        self._vibrotactile_on = val
        self._btn_vibrotactile.setText(f"📳  Vibrotactile: {'ON' if val else 'OFF'}")
        self._btn_vibrotactile.setStyleSheet(self._TOGGLE_STYLE_ON if val else self._TOGGLE_STYLE_OFF)
        self._log_feedback_event("vibrotactile", val, source)

    def _apply_viewport_ui(self, val: bool, source: str = "manager"):
        """Reflect `val` (FPV video canvas shown in the Quest) on the button and gate neck tracking.

        Sends nothing over the wire, but does drive the Ostrich neck: while the camera canvas
        is shown the neck tracks the operator's head (look-around) even before teleop starts.
        Logs the change to the per-take feedback-event log."""
        self._viewport_on = val
        self._btn_viewport.setText(f"🖥️  FPV Display: {'ON' if val else 'OFF'}")
        self._btn_viewport.setStyleSheet(self._TOGGLE_STYLE_ON if val else self._TOGGLE_STYLE_OFF)
        if self._control_thread is not None:
            self._control_thread.set_neck_active(val)
        self._log_feedback_event("viewport", val, source)

    def _on_toggle_audio_haptics(self):
        self._apply_audio_haptics_ui(not self._audio_haptics_on)
        self._send_unity_cmd("toggle_audio_haptics", {"enabled": self._audio_haptics_on})
        self._log(f"Unity: audio haptics {'enabled' if self._audio_haptics_on else 'disabled'}")

    def _on_toggle_vibrotactile(self):
        self._apply_vibrotactile_ui(not self._vibrotactile_on)
        self._send_unity_cmd("toggle_vibrotactile_haptics", {"enabled": self._vibrotactile_on})
        self._log(f"Unity: vibrotactile haptics {'enabled' if self._vibrotactile_on else 'disabled'}")

    def _on_show_viewport(self):
        # Shows/hides the FPV video canvas in the Quest (display visibility), keeping the WHEP feed
        # connected so re-showing is instant and low-latency. This is robust (no peer teardown), unlike
        # the old connect/disconnect which could leave the canvas stuck on "press to retry".
        self._apply_viewport_ui(not self._viewport_on)
        self._send_unity_cmd("show_streaming_viewport", {"enabled": self._viewport_on})
        self._log(f"Unity: FPV display {'shown' if self._viewport_on else 'hidden'}")

    # ── Per-condition feedback: auto-set the Unity toggles for the trial ──────
    def _set_feedback(self, which: str, on: bool):
        """Force ONE feedback modality to `on` exactly as its toggle button would: update
        the button UI (+ feedback-event log, source='trial') AND send the Unity command.
        Idempotent — sending the current state re-asserts it on the headset."""
        if which == "audio_haptics":
            self._apply_audio_haptics_ui(on, source="trial")
            self._send_unity_cmd("toggle_audio_haptics", {"enabled": on})
        elif which == "vibrotactile":
            self._apply_vibrotactile_ui(on, source="trial")
            self._send_unity_cmd("toggle_vibrotactile_haptics", {"enabled": on})
        elif which == "viewport":
            self._apply_viewport_ui(on, source="trial")
            self._send_unity_cmd("show_streaming_viewport", {"enabled": on})

    def _apply_feedback_for_condition(self, cond: str):
        """At each take start, turn the Unity feedback modalities on/off to match the trial
        condition (H+A+V / H+V / A+V / V) — updates the manager buttons AND sends the same
        commands the buttons send, so the operator's headset feedback matches the trial."""
        fb = _CONDITION_FEEDBACK.get(cond)
        if fb is None:
            return
        for which in ("viewport", "audio_haptics", "vibrotactile"):
            self._set_feedback(which, bool(fb[which]))
        on = [k for k in ("viewport", "audio_haptics", "vibrotactile") if fb[k]]
        self._log(f"[feedback] '{cond}' → "
                  + ", ".join(k.replace("audio_haptics", "audio")
                              .replace("vibrotactile", "haptics")
                              .replace("viewport", "video") for k in on))

    # =========================================================================
    # Safety: default-cursor parking + e-stop kill
    # =========================================================================

    @staticmethod
    def _cursor_pos():
        p = QtGui.QCursor.pos()
        return (p.x(), p.y())

    def _on_cursor_idle_tick(self):
        """After 15 s of no mouse movement: raise the manager window to the front (restoring it if
        minimized) so NOTHING covers the E-STOP, then move the cursor over the E-STOP button so it's a
        double-click away. Both are read LIVE from the button's on-screen rect, so they track the window
        even if it's moved, resized/scaled, or on another monitor. Done once per idle period (re-armed
        when the mouse moves)."""
        btn = getattr(self, "_btn_estop", None)
        if btn is None:
            return
        # Only auto-surface/park while a participant SESSION is open (Start Session → Stop
        # Session) AND the robot is booted — i.e. during live operation, when the E-STOP
        # actually matters. During setup / bootup / calibration (no session open), leave the
        # operator's cursor alone. Keep the idle clock re-armed so it never parks the instant
        # a session starts.
        if not (self._session_open() and getattr(self, "_hw_booted", False)):
            self._last_cursor_move_time = time.monotonic()
            self._idle_parked = False
            return
        # Suppress the auto-surface/park while the fingertip force-calibration window is
        # open — it would steal focus and yank the cursor onto the E-STOP mid-calibration.
        # Keep the idle clock re-armed so it doesn't instantly park the moment it closes.
        cw = getattr(self, "_calib_window", None)
        if cw is not None and cw.isVisible():
            self._last_cursor_move_time = time.monotonic()
            self._idle_parked = False
            return
        # Same suppression while the VOICE-calibration dialogs are up: the operator reads a
        # script for ~15 s with the mouse idle, which would otherwise trip the auto-park,
        # raise the main window OVER the modal, and steal its focus (making the next
        # operator's dialog appear behind the main window — it looked like it "closed out").
        if getattr(self, "_calibrating_voices", False):
            self._last_cursor_move_time = time.monotonic()
            self._idle_parked = False
            return
        cur = self._cursor_pos()
        if self._last_seen_cursor is None or cur != self._last_seen_cursor:
            # Mouse moved → reset the idle timer and re-arm the auto-surface.
            self._last_seen_cursor = cur
            self._last_cursor_move_time = time.monotonic()
            self._idle_parked = False
            return
        if self._idle_parked or (time.monotonic() - self._last_cursor_move_time) < 15.0:
            return
        # 15 s idle → surface the e-stop and park the cursor on it (once).
        self._idle_parked = True
        try:
            if self.isMinimized():
                self.showNormal()
            self.raise_()
            self.activateWindow()
        except Exception:
            pass
        if btn.isVisible():
            gc = btn.mapToGlobal(btn.rect().center())
            QtGui.QCursor.setPos(gc.x(), gc.y())
            self._last_seen_cursor = (gc.x(), gc.y())

    def _on_estop_clicked(self):
        """E-STOP arm/fire: the FIRST click arms it and shows a 'click again' prompt; a
        SECOND click within 1 s fires. If no second click lands in time it disarms and
        resets. A stray single click therefore can't trigger the stop."""
        now = time.monotonic()
        if self._estop_armed and (now - self._estop_armed_at) <= 1.0:
            self._estop_armed = False
            self._on_estop_kill()                # second click in time → FIRE
            return
        # First click → arm + prompt, auto-disarm after 1 s.
        self._estop_armed = True
        self._estop_armed_at = now
        self._estop_arm_token += 1
        token = self._estop_arm_token
        self._btn_estop.setText("🛑  CLICK AGAIN to E-STOP\n(within 1 s)")
        QTimer.singleShot(1000, lambda: self._estop_disarm(token))

    def _estop_disarm(self, token):
        """Reset the E-STOP if it was armed and no second click landed — but only if this
        is the latest arm (token guards against an older timer clobbering a fresh prompt)."""
        if self._estop_armed and token == self._estop_arm_token:
            self._estop_armed = False
            self._btn_estop.setText(self._ESTOP_LABEL)

    def _on_estop_kill(self):
        """E-STOP (DOUBLE-CLICK): instantly stop + damp the ROBOT — the manager app STAYS
        RUNNING (recordings, data, link and UI survive; use Quit Manager to close the app).
        No confirmation dialog — an e-stop must be instant (no tilt; use Safe Shutdown for
        the controlled descent). Teleop needs a manager restart afterwards (the control
        loop is terminated so nothing can ever fight the damping)."""
        try:
            self._log("*** E-STOP — stopping + damping the ROBOT (the manager stays "
                      "running; restart it to teleop again). ***")
        except Exception:
            pass
        # 1) Non-blocking: tell the running control loop to stop teleop and stream damping NOW.
        try:
            if self._control_thread is not None:
                self._control_thread.stop_teleop()
                self._control_thread.enter_arm_damping()
        except Exception:
            pass
        # 2) TERMINATE the control thread — it owns the 500 Hz sender, and any later
        #    command stream would re-stiffen the damped robot. After this the direct
        #    damping below is the sole publisher, and stays authoritative for good.
        try:
            if self._control_thread is not None:
                self._control_thread.stop()
        except Exception:
            pass
        # 3) Authoritative direct full-body damping. Bounded (~0.3 s) so the e-stop can't
        #    hang. Ignores --no_exit_damping: an e-stop must always stop.
        try:
            if self._hw_controller is not None:
                self._hw_controller.enter_damping(duration_s=0.3)
        except Exception:
            pass
        # 4) App STAYS ALIVE: mark the robot down so the UI reflects it — teleop/session
        #    controls disable (state INIT = not ready), a later Quit won't re-damp, and
        #    the E-STOP button itself becomes a status badge (still safe to re-fire).
        self._exit_damping_done = True
        self._state = _S_INIT
        self._ESTOP_LABEL = "🛑  ROBOT E-STOPPED (damped)\n♻ Reboot to recover"
        self._btn_estop.setText(self._ESTOP_LABEL)
        if getattr(self, "_hw_booted", False):
            self._btn_bootup.setText("♻  Reboot")
            self._btn_bootup.setEnabled(True)
        try:
            self._update_buttons()
            self._refresh_status_panel()
        except Exception:
            pass

    def _on_quit_manager(self):
        """Quit Manager (confirmation-guarded): gracefully close the APP. This is the only
        button that exits — E-STOP and Safe Shutdown act on the robot alone. closeEvent
        does the actual teardown: finalizes any open session, stops cameras/streaming/
        trackers/link, and damps the robot on exit unless it was already e-stopped /
        safe-shut-down (_exit_damping_done)."""
        robot_up = (getattr(self, "_hw_booted", False)
                    and self._hw_controller is not None
                    and not getattr(self, "_exit_damping_done", False))
        lower_secs = max(0.0, float(getattr(self._args, "shutdown_lower_secs", 15.0)))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Quit Manager")
        box.setText("Close the manager app?\n\nThis finalizes "
                    + ("the OPEN recording session, " if self._session_open() else "")
                    + "all recordings, stops the cameras/streaming and the peer link"
                    + (f", and first runs the SAFE SHUTDOWN on the robot (arms out, "
                       f"tilt forward, slow ~{lower_secs:g}s knee fold, then limp) — "
                       "it is booted and has not been shut down"
                       if robot_up else "")
                    + ".")
        quit_btn = box.addButton("Shut down robot && Quit" if robot_up else "Quit",
                                 QMessageBox.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)          # stray click/Enter lands on the safe side
        box.exec_()
        if box.clickedButton() is not quit_btn:
            self._log("Quit cancelled.")
            return
        if robot_up:
            # Controlled descent BEFORE teardown (blocking ~15 s; sets
            # _exit_damping_done so closeEvent won't re-damp the folded robot).
            self._log("Quit: running the robot Safe Shutdown first…")
            try:
                self._run_safe_shutdown_robot()
            except Exception as exc:                 # noqa: BLE001
                # Contained: closeEvent's plain exit-damp is the fallback.
                self._log(f"Quit: safe shutdown failed ({exc}) — falling back to the "
                          "plain exit damp.")
                self._exit_damping_done = False
        self._log("Quitting the manager (graceful close)…")
        self.close()

    # ── Safe Shutdown: controlled forward fall (ROBOT only — app stays open) ──
    def _shutdown_hip_tilt_rad(self):
        """Hip_pitch delta (rad) for the Safe-Shutdown forward tilt. --shutdown-tilt-deg
        is the FORWARD lean in degrees (positive = toward the toes); a negative hip_pitch
        delta pitches the pelvis forward (hip_pitch axis is +Y, feet planted). If the
        robot tips BACKWARD on your rig, pass a negative --shutdown-tilt-deg to flip it."""
        deg = float(getattr(self._args, "shutdown_tilt_deg", 25.0))
        return -np.deg2rad(deg)

    def _a_pose_arms(self):
        """The arm 'A pose' the robot holds at the END of bootup, right before it drops
        into damping: arms abducted OUT (45 deg) and pitched BACK by --bootup_arm_back_deg,
        elbow at the rest bend. Returns (left_7dof, right_7dof). Shared by the bootup rise
        and Safe Shutdown so both settle the arms into the exact same pose."""
        back = np.deg2rad(float(getattr(self._args, "bootup_arm_back_deg", 20.0)))
        left  = _stage_arm(back, +_BOOTUP_ABDUCT, 0.0, _ELBOW_REST)
        right = _stage_arm(back, -_BOOTUP_ABDUCT, 0.0, _ELBOW_REST)
        return left, right

    # Leg vector is [L(6), R(6)] with per-leg order
    # [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll].
    _LEG_KNEE_IDX = (3, 9)
    _LEG_HIP_PITCH_IDX = (0, 6)
    _KNEE_LIMIT = 2.8798      # from the model; cap the collapse target just under it

    def _shutdown_collapse_legs(self):
        """12-DOF leg target for the Safe-Shutdown DEEPER collapse: the robot is already
        at its kneeling pose during operation, so this flexes the knees FURTHER so it
        settles LOWER, and leans the PELVIS (hip_pitch) FORWARD by the initial tilt PLUS
        an extra collapse lean. Because waist_pitch is capped at ±30°, the pelvis lean —
        which has a long lever to the head — is what keeps the head out in FRONT as the
        knees fold, so the robot drops head-forward when damping is entered.

        The pelvis lean is in the tilt's forward direction (flip --shutdown_tilt_deg's
        sign to flip forward); the extra lean is --shutdown_collapse_hip_deg. Knee and
        hip targets are clamped inside their joint limits."""
        legs = np.asarray(self._kneeling_angles, dtype=np.float64).copy()
        kdel = np.deg2rad(float(getattr(self._args, "shutdown_collapse_knee_deg", 60.0)))
        tilt_deg = float(getattr(self._args, "shutdown_tilt_deg", 25.0))
        lean_deg = float(getattr(self._args, "shutdown_collapse_hip_deg", 100.0))
        # Total forward pelvis lean = |initial tilt| + extra collapse lean, applied in
        # the tilt's forward direction (forward = -hip_pitch for a positive tilt_deg).
        fwd_dir = -1.0 if tilt_deg >= 0.0 else 1.0
        hip_delta = fwd_dir * np.deg2rad(abs(tilt_deg) + lean_deg)
        for ki in self._LEG_KNEE_IDX:        # more knee flexion (positive) = lower/deeper
            legs[ki] = min(legs[ki] + kdel, self._KNEE_LIMIT - 0.03)
        for hi in self._LEG_HIP_PITCH_IDX:   # lean pelvis forward (tilt + extra collapse lean)
            legs[hi] = float(np.clip(legs[hi] + hip_delta, -2.5, self._KNEE_LIMIT))
        return legs

    def _shutdown_a_pose_arms(self):
        """The Safe-Shutdown 'A-pose' hold: arms raised OUT to the sides but DROOPED at the
        shoulders with the elbows FULLY EXTENDED (straight arms, no elbow bend) — hands out
        to the sides, clear of BOTH the front table and the chest. Held through the descent,
        then dropped. Returns (left_7dof, right_7dof)."""
        return _ARMS_SHUTDOWN_A_LEFT.copy(), _ARMS_SHUTDOWN_A_RIGHT.copy()

    def _shutdown_retract_arms(self, hw):
        """FIRST step of Safe Shutdown, before any tilt: raise the arms straight OUT to the
        sides into a drooped 'A-pose' (FULLY EXTENDED elbows, straight arms) so the hands
        lift off the table and away from the chest. ONE body-held move — the arms only swing
        out to the sides, never more forward than the start (FK-verified) — after which they
        hold that pose through the descent and drop."""
        secs = float(getattr(self._args, "shutdown_wing_secs", 1.5))
        if secs <= 0.0:
            return
        try:
            cur_t, _cur_l, _cur_r = hw.get_current_upper_body()
            cur_legs = np.array(
                [hw.low_state.motor_state[i].q for i in hw.config.leg_joint2motor_idx],
                dtype=np.float64)
        except Exception as exc:
            self._log(f"[shutdown] A-pose retract skipped (no state): {exc}")
            return
        a_l, a_r = self._shutdown_a_pose_arms()
        self._log("[shutdown] raising the arms OUT to the sides (drooped A-pose, clear of "
                  "the table + chest) before tilting…")
        try:
            hw.move_to_pose(
                q_goal_legs=cur_legs, total_time=max(0.6, secs),
                q_goal_upper=np.concatenate((np.asarray(cur_t, dtype=np.float64), a_l, a_r)),
                leg_kp_scale=self._args.leg_kp_scale, leg_kd_scale=self._args.leg_kd_scale)
        except Exception as exc:
            self._log(f"[shutdown] A-pose retract skipped: {exc}")

    def _forward_tilt_then_damp(self, hw):
        """Safe-shutdown motion: (0) raise the arms OUT to the sides into a drooped 'A-pose'
        so the hands lift off the table AND away from the chest, then (1) tilt the torso/hips
        forward so the robot's weight goes forward, then (2) SLOWLY fold the knees so it sinks
        down/back into its kneeling rest pose — the arms stay out to the sides, then drop under
        damping — then (3) go fully limp. The knee fold is POSITION-controlled toward the
        kneeling pose (rate-limited, so the gravity-driven drop at the knees is slowed) and
        stays quiet: the legs keep a position term, so their velocity loop never runs at kp=0
        with a high kd, which is what made the motors buzz. ``hw`` must be the sole publisher.
        --shutdown-tilt-deg 0 skips the tilt; --shutdown_wing_secs 0 skips the arm retract."""
        # (0) Raise the hands OUT to the sides FIRST (drooped A-pose) so the tilt + descent
        # never drag them forward/down through the table or into the chest.
        self._shutdown_retract_arms(hw)
        if abs(float(getattr(self._args, "shutdown_tilt_deg", 25.0))) > 1e-6:
            try:
                hw.tilt_hips_forward(
                    self._shutdown_hip_tilt_rad(),
                    ramp_s=max(0.0, float(getattr(self._args, "shutdown_tilt_ramp_s", 1.2))),
                    hold_s=max(0.0, float(getattr(self._args, "shutdown_tilt_hold_s", 0.3))),
                    kp_scale=self._args.hw_kp_scale, kd_scale=self._args.hw_kd_scale,
                    leg_kp_scale=self._args.leg_kp_scale, leg_kd_scale=self._args.leg_kd_scale,
                    ankle_kp_scale=0.0, ankle_kd_scale=1.0)
            except Exception:
                pass
        # Slowly fold the knees deeper — from the operating kneel into a DEEPER collapse
        # (more knee flexion) — under rate-limited position control on the legs (slow +
        # silent). The torso bows forward, and the arms are HELD out to the sides in the
        # drooped A-pose (already reached by the retract above) the whole way down — they
        # only drop when the final damping is entered. Raise --shutdown_lower_secs to fold
        # more slowly; --shutdown_collapse_knee_deg sets how much deeper it sinks.
        _wing_l, _wing_r = self._shutdown_a_pose_arms()
        try:
            hw.lower_into_kneel(
                self._shutdown_collapse_legs(),
                duration_s=max(0.0, float(getattr(self._args, "shutdown_lower_secs", 15.0))),
                ease_power=float(getattr(self._args, "shutdown_lower_ease_power", 2.0)),
                arm_kd=float(getattr(self._args, "arm_damp_kd", 8.0)),
                # Bow the torso forward (positive waist_pitch) as it lowers so the CoM
                # ends up forward and it topples FORWARD when damped.
                torso_bow_rad=np.deg2rad(float(getattr(self._args, "shutdown_torso_bow_deg", 25.0))),
                # Arms: HOLD the drooped A-pose (hands out to the sides, already reached by
                # the retract — clear of the table AND the chest) the whole way down
                # (repeated waypoint = hold), so they only drop when damping starts.
                arm_left_waypoints=[_wing_l, _wing_l],
                arm_right_waypoints=[_wing_r, _wing_r],
                leg_kp_scale=self._args.leg_kp_scale, leg_kd_scale=self._args.leg_kd_scale,
                ankle_kp_scale=0.0, ankle_kd_scale=1.0,
                torso_kp_scale=self._args.hw_kp_scale, torso_kd_scale=self._args.hw_kd_scale,
                arm_kp_scale=self._args.hw_kp_scale, arm_kd_scale=self._args.hw_kd_scale)
        except Exception:
            pass
        try:
            hw.enter_damping(duration_s=1.0)   # everything limp; legs already at the kneel → minimal motion
        except Exception:
            pass

    def _run_safe_shutdown_robot(self):
        """The PHYSICAL safe-shutdown (no dialog): stop the control loop, run the
        controlled forward-tilt + slow knee fold, go limp, flip the flags + UI.
        Shared by the Safe Shutdown button and Quit Manager."""
        lower_secs = max(0.0, float(getattr(self._args, 'shutdown_lower_secs', 15.0)))
        try:
            self._log(f"*** SAFE SHUTDOWN (robot) — raise the arms out to the sides "
                      f"(drooped A-pose) off the table, tilt forward, fold knees into the "
                      f"kneel slowly over ~{lower_secs:g}s, go limp. The manager stays "
                      "open. ***")
        except Exception:
            pass
        # Stop the teleop control loop so the low-level Controller is the SOLE publisher
        # (otherwise the loop's stream fights the tilt/damp commands), then run the
        # controlled forward fall right away (no long gap that could trip the watchdog).
        try:
            if self._control_thread is not None:
                self._control_thread.stop_teleop()
                self._control_thread.stop()
        except Exception:
            pass
        try:
            if self._hw_controller is not None:
                self._forward_tilt_then_damp(self._hw_controller)
        except Exception:
            pass
        # Robot is down + limp; the APP stays alive. Mark damping done (a later Quit
        # must not re-damp the folded robot) and reflect the state in the UI.
        self._exit_damping_done = True
        self._state = _S_INIT
        self._btn_safe_shutdown.setText("🛡  Robot Shut Down (limp)")
        self._btn_safe_shutdown.setEnabled(False)
        self._ESTOP_LABEL = "🛑  ROBOT SHUT DOWN\n(already damped)"
        self._btn_estop.setText(self._ESTOP_LABEL)
        self._btn_bootup.setText("♻  Reboot")
        self._btn_bootup.setEnabled(True)
        try:
            self._update_buttons()
            self._refresh_status_panel()
        except Exception:
            pass

    def _on_safe_shutdown(self):
        """SAFE SHUTDOWN (ROBOT ONLY): tilt the torso forward, then SLOWLY fold the knees
        so the robot sinks down/back into its kneeling rest pose (arms + torso drop under
        normal damping) and goes fully limp. The MANAGER STAYS OPEN — recordings, data
        transfer, playback and the link keep working; close the app with the separate
        Quit Manager button. Unlike the instant E-STOP, this takes ~10 s+ for the
        controlled slow knee descent. Teleop needs a manager restart afterwards."""
        if not getattr(self, "_hw_booted", False):
            self._log("[shutdown] robot was never booted — nothing to shut down.")
            return
        lower_secs = max(0.0, float(getattr(self._args, "shutdown_lower_secs", 15.0)))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Safe Shutdown (robot)")
        box.setText("Safely shut down the ROBOT?\n\nIt will raise the arms OUT to the sides "
                    "(drooped A-pose, clearing the table and the chest), tilt forward, "
                    f"slowly fold into the kneel over ~{lower_secs:g}s, and go limp.\n\n"
                    "The manager app STAYS OPEN (data / playback / peer link keep "
                    "working) — use Quit Manager to close it. Teleop requires a manager "
                    "restart afterwards.")
        go = box.addButton("Shut down robot", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])   # default = Cancel
        box.exec_()
        if box.clickedButton() is not go:
            self._log("Safe Shutdown cancelled.")
            return
        self._run_safe_shutdown_robot()
        self._log("Robot is down and limp. The manager remains open — Quit Manager "
                  "closes it (recordings/cameras are finalized then).")

    def _apply_unity_state_updates(self):
        """Mirror in-VR toggle changes reported by Unity onto the manager buttons.

        The XR operator can toggle haptics / audio haptics / the FPV display from the
        in-VR menu; Unity reports each change over the 'unity_state' channel and we
        update the matching button here so the manager UI never disagrees with what
        the operator sees. This is a pure UI update — it must NOT re-send a command,
        which would echo straight back to Unity.
        """
        if self._device is None:
            return
        try:
            updates = self._device.poll_unity_state()
        except Exception:
            return
        for upd in updates:
            cmd = upd.get("command")
            val = bool(upd.get("enabled", False))
            if cmd == "unity_ready":
                # The XR app's unity_cmds channel just opened — push our authoritative toggle/
                # streaming state NOW (overwrites whatever the operator set in VR pre-connect and
                # syncs the in-VR UI). This is the reliable trigger; the timer below is a fallback.
                self._push_unity_state()
                continue
            if cmd == "toggle_audio_haptics" and val != self._audio_haptics_on:
                self._apply_audio_haptics_ui(val, source="operator_vr")
                self._log(f"Operator (VR): audio haptics {'enabled' if val else 'disabled'}")
            elif cmd == "toggle_vibrotactile_haptics" and val != self._vibrotactile_on:
                self._apply_vibrotactile_ui(val, source="operator_vr")
                self._log(f"Operator (VR): vibrotactile {'enabled' if val else 'disabled'}")
            elif cmd == "show_streaming_viewport" and val != self._viewport_on:
                self._apply_viewport_ui(val, source="operator_vr")
                self._log(f"Operator (VR): FPV display {'shown' if val else 'hidden'}")
            elif cmd == "teleop_active":
                # XR operator pressed Play (true) / Pause (false) in VR — this is ALSO what
                # fires when the participant takes the Quest headset OFF (auto-pause). Engage/
                # pause only — do NOT echo a unity_cmds command back (the headset knows its state).
                if val:
                    self._engage_teleop()
                else:
                    self._pause_teleop("headset")

    def _update_latency_budget(self):
        """Assemble the end-to-end VIDEO (camera→eyes) and TELEOP (operator move→robot move)
        latency from the measured stages plus the estimated endpoint constants, store the samples
        for the per-trial summary, and print a breakdown to the console ~every 5 s.

        Measured: Quest↔PC network (clock-sync RTT), video jitter-buffer + decode (Quest WebRTC
        stats), manager pose-wait+IK, and robot actuation (motor feedback). Estimated (constants,
        tunable via --lat-* flags): camera capture+encode, headset display→photons, body-tracking
        sensor lag, and an actuation fallback when the robot is too still to measure."""
        dev = self._device
        if dev is None:
            return
        a = self._args

        # ── Video: camera first capture → operator's eyes (glass-to-glass) ──
        try:
            vid = dev.video_latency()
        except Exception:
            vid = None
        if vid is not None:
            cap, disp = a.lat_capture_encode_ms, a.lat_display_ms
            net, jit, dec = vid["net_ms"], vid["jitter_ms"], vid["decode_ms"]
            total = cap + net + jit + dec + disp
            self._latency_e2e.add_video(total, {
                "capture_encode_ms(est)": round(cap, 1),
                "network_ms":             round(net, 1),
                "jitter_buffer_ms":       round(jit, 1),
                "decode_ms":              round(dec, 1),
                "display_photon_ms(est)": round(disp, 1),
                "total_ms":               round(total, 1),
            })
            self._set_latency_label(self._lat_video_lbl, "Video (cam→eyes)", total, 200.0, 400.0)
            self._lat_video_bd_lbl.setText(
                f"net {net:.0f} · jit {jit:.0f} · dec {dec:.0f} · cap+disp {cap + disp:.0f}")

        # ── Teleop: operator first moves → robot moves (motion-to-motion) ──
        try:
            rtt = dev.network_rtt_ms()
        except Exception:
            rtt = None
        ct = self._control_thread
        proc = getattr(ct, "_last_manager_proc_ms", None) if ct is not None else None
        act = getattr(ct, "_last_actuation_ms", None) if ct is not None else None
        if rtt is not None and proc is not None:
            track = a.lat_tracking_ms
            net_one_way = rtt * 0.5
            act_measured = act is not None
            act_ms = act if act_measured else a.lat_actuation_ms
            total = track + net_one_way + proc + act_ms
            self._latency_e2e.add_teleop(total, {
                "tracking_ms(est)":  round(track, 1),
                "network_ms":        round(net_one_way, 1),
                "manager_proc_ms":   round(proc, 1),
                ("actuation_ms" if act_measured else "actuation_ms(est)"): round(act_ms, 1),
                "total_ms":          round(total, 1),
            })
            self._set_latency_label(self._lat_teleop_lbl, "Teleop (move→robot)", total, 120.0, 250.0)
            act_tag = "act" if act_measured else "act(est)"
            self._lat_teleop_bd_lbl.setText(
                f"net {net_one_way:.0f} · proc {proc:.0f} · {act_tag} {act_ms:.0f} · track {track:.0f}")

        # Quest data-link health: inbound body_pose delivery rate + STALLS. A stall is a
        # >50 ms gap = the reliable/ordered body_pose channel waiting on a WiFi-loss
        # retransmit — the cause of the residual teleop lag AND the flaky start/pause command
        # (unity_state/unity_cmds share the same congested SCTP link). Green = clean;
        # orange/red = loss stalling the link → switch body_pose to unreliable on the Quest.
        try:
            lh = dev.link_health()
        except Exception:
            lh = None
        if lh:
            st = lh["stalls_per_s"]
            rtt = lh.get("rtt_ms")
            rtt_txt = f"{rtt:.0f} ms" if rtt is not None else "—"
            color = "green" if st < 0.5 else "#e67e22" if st < 3.0 else "#c0392b"
            _lbl_text(self._lat_link_lbl,
                      f"Link (Quest data): {lh['fps']:.0f} fps · {st:.1f} stalls/s "
                      f"(max gap {lh['max_gap_ms']:.0f} ms) · RTT {rtt_txt}")
            _lbl_style(self._lat_link_lbl, f"color: {color}; font-size: 11px;")
            # Throttled console record (~every 10 s) so a study has a timestamped link-health
            # trail alongside the live panel. Only meaningful once teleop is running.
            now = time.monotonic()
            if now - getattr(self, "_last_link_log_t", 0.0) >= 10.0:
                self._last_link_log_t = now
                tag = "OK" if st < 0.5 else "STALLING" if st < 3.0 else "BAD-LOSS"
                self._log(f"[link-health] {tag}: {lh['fps']:.0f} fps · {st:.1f} stalls/s · "
                          f"max gap {lh['max_gap_ms']:.0f} ms · RTT {rtt_txt}")
        else:
            _lbl_text(self._lat_link_lbl, "Link (Quest data): —")
            _lbl_style(self._lat_link_lbl, "color: gray; font-size: 11px;")

        # At-a-glance OK / DEGRADED / FROZEN — driven by the SAME pose_age + RTT the teleop
        # watchdog trips on, so it visibly heads toward a trip before the arms park. DEGRADED
        # is the warn band (half the trip thresholds, or loss stalls); FROZEN = the trip zone.
        a = self._args
        freeze_s = float(getattr(a, "link_freeze_s", 1.0))
        pause_rtt = float(getattr(a, "link_pause_rtt_ms", 400.0))
        warn_age = freeze_s * 0.5
        warn_rtt = pause_rtt * 0.5
        try:
            age = dev.pose_age_s()
        except Exception:
            age = None
        rtt_now = lh.get("rtt_ms") if lh else None
        stalls = lh.get("stalls_per_s", 0.0) if lh else 0.0
        if age is None and rtt_now is None:
            s_txt, s_col = "Quest link: —", "gray"
        elif (age is not None and age >= freeze_s) or \
             (rtt_now is not None and rtt_now >= pause_rtt):
            drv = (f"no pose {age:.1f}s" if (age is not None and age >= freeze_s)
                   else f"RTT {rtt_now:.0f} ms")
            s_txt, s_col = f"Quest link: ■ FROZEN ({drv})", "#c0392b"
        elif (age is not None and age >= warn_age) or \
             (rtt_now is not None and rtt_now >= warn_rtt) or stalls >= 1.0:
            bits = []
            if age is not None and age >= warn_age:
                bits.append(f"pose {age * 1000:.0f} ms old")
            if rtt_now is not None and rtt_now >= warn_rtt:
                bits.append(f"RTT {rtt_now:.0f} ms")
            if stalls >= 1.0:
                bits.append(f"{stalls:.1f} stalls/s")
            s_txt, s_col = f"Quest link: ▲ DEGRADED ({', '.join(bits)})", "#e67e22"
        else:
            s_txt, s_col = "Quest link: ● OK", "#27ae60"
        _lbl_text(self._lat_link_status_lbl, s_txt)
        _lbl_style(self._lat_link_status_lbl,
                   f"color: {s_col}; font-weight: bold; font-size: 12px;")

        # The Video/Teleop budgets stay in the Latency Monitor panel ONLY; the link-health
        # line above is also logged, throttled to ~10 s, for a per-study record.

    def _maybe_reassert_unity_on_reconnect(self):
        """Re-push the manager's authoritative Unity toggle state (haptics / vibrotactile / FPV,
        via _push_unity_state) on EVERY (re)connection of the operator's pose link — not just the
        first connect.

        Why keyed on pose freshness, not is_connected: the device's is_connected latches True on
        the first body_pose and NEVER resets (so it can't see a headset off/on), which left the old
        `connected and not was_connected` edge firing exactly once. When a reconnect ALSO missed the
        Quest's 'unity_ready', the headset's haptics toggle was never re-asserted → payloads kept
        flowing but the headset stayed SILENT (the intermittent "haptics die after headset off/on"
        bug; teleop, being inbound pose, is unaffected). pose_age_s() resets the instant body_pose
        frames resume, so this trips on the true reconnect edge (and still covers the first connect).

        Staggered singleShots give a just-reopened unity_cmds channel time to open. Idempotent.
        Returns True when it (re)fired (for tests)."""
        try:
            age = self._device.pose_age_s() if self._device is not None else None
        except Exception:
            age = None
        link_live = age is not None and age < float(getattr(self._args, "link_freeze_s", 1.0))
        fired = False
        if link_live and not self._link_live_last:
            for _delay_ms in (300, 1200, 2500):
                QTimer.singleShot(_delay_ms, self._push_unity_state)
            self._log("[xr] pose link (re)connected — re-asserting Unity toggle state "
                      "(haptics / vibrotactile / FPV) so the headset matches the manager.")
            fired = True
        self._link_live_last = link_live
        return fired

    def _push_unity_state(self):
        """Send the manager's current toggle states to Unity in one burst.

        The User Study Manager takes priority on every fresh client connection: it pushes
        ALL of its current toggle states to Unity so the XR app is overwritten to match
        these buttons — eliminating any mismatch if the app and manager disagreed before
        connecting. This includes the streaming feed (the manager's button wins, even over
        Unity's auto-started stream). Relies on the device's unity_cmds sender forwarding
        every queued command (it no longer coalesces to the latest), so all three arrive
        rather than just the last.
        """
        self._send_unity_cmd("toggle_audio_haptics",        {"enabled": self._audio_haptics_on})
        self._send_unity_cmd("toggle_vibrotactile_haptics", {"enabled": self._vibrotactile_on})
        self._send_unity_cmd("show_streaming_viewport",     {"enabled": self._viewport_on})
        self._log("Pushed current toggle states to client on connect (manager takes priority).")

    def _send_unity_cmd(self, command: str, payload: dict):
        if self._device is None:
            return
        msg = json.dumps({"type": "unity_command", "command": command, **payload})
        try:
            self._device.send_unity_command(msg)
        except Exception as exc:
            self._log(f"Unity command send error: {exc}")
            return
        # Surface silently-dropped commands: the 'unity_cmds' channel is created by the
        # XR app (Quest). If it isn't open, send_to_datachannel just drops the command
        # (returns 0) and the toggle never reaches the headset — the usual reason a
        # manager button doesn't reflect in VR. Warn once per state change so it's visible.
        try:
            ready = self._device.unity_cmds_ready()
        except Exception:
            ready = True   # can't tell — don't nag
        if ready != getattr(self, "_unity_cmds_ready_last", None):
            self._unity_cmds_ready_last = ready
            if ready:
                self._log("Unity command channel OPEN — manager toggles are reaching the XR app.")
            else:
                self._log("Unity command channel NOT open — manager toggles are NOT reaching the "
                          "XR app (the XR app must create/open the 'unity_cmds' channel; "
                          "check the XR build and that it's connected).")

    # =========================================================================
    # Button state management
    # =========================================================================

    def _update_buttons(self):
        is_idle     = self._state == _S_IDLE
        is_waiting  = self._state == _S_WAITING
        is_teleop   = self._state == _S_TELEOP
        # Live 3D viewer toggle label reflects whether the window is currently open.
        if hasattr(self, "_btn_live_viewer"):
            self._btn_live_viewer.setText(
                "🖥  Hide Live 3D Viewer" if self._viewer is not None
                else "🖥  Show Live 3D Viewer")
        # Hardware/stream reboot buttons (neck / hands / ZED streaming) — usable any time
        # EXCEPT while an attempt is recording (they'd disrupt the take); fine with a session
        # open and armed between takes.
        can_reboot_hw = not self._recording_active
        for _bname in ("_btn_reboot_neck", "_btn_reboot_hands", "_btn_reboot_zed",
                       "_btn_reboot_webrtc"):
            _w = getattr(self, _bname, None)
            if _w is not None:
                _w.setEnabled(can_reboot_hw)
        ready       = self._state != _S_INIT
        # "(BOTH)" suffix when a peer link is up — the Data Collection + Teleop buttons
        # then drive both robots; unlinked they act on this robot only.
        bs = self._both_suffix()

        # Green Start / Yellow Pause — a single engage/pause toggle. Enabled once the
        # system is initialized; a connected XR client is NOT required (teleop runs
        # standalone). When linked it drives BOTH robots (label gains "(BOTH)").
        self._btn_start_teleop.setEnabled(ready)
        if is_teleop:
            self._btn_start_teleop.setText("⏸  Pause Teleop" + bs)
            self._btn_start_teleop.setStyleSheet("QPushButton { background: #f1c40f; color: black; font-weight: bold; padding: 8px; }")
        else:
            self._btn_start_teleop.setText("▶  Start Teleop" + bs)
            self._btn_start_teleop.setStyleSheet("QPushButton { background: #2ecc71; color: white; font-weight: bold; padding: 8px; }")
        self._btn_stop_teleop.setText("■  Stop & Lower Arms" + bs)
        self._btn_stop_teleop.setEnabled(is_waiting or is_teleop)
        # Individual "this robot only" controls — visible ONLY when linked (unlinked, the
        # buttons above already act on this robot alone). Drive just this robot, no peer echo.
        linked = self._linked()
        for w in (self._solo_teleop_lbl, self._btn_start_teleop_solo, self._btn_stop_teleop_solo):
            w.setVisible(linked)
        if linked:
            if is_teleop:
                self._btn_start_teleop_solo.setText("⏸  Pause (this robot)")
                self._btn_start_teleop_solo.setStyleSheet("QPushButton { background: #f1c40f; color: black; padding: 6px; }")
            else:
                self._btn_start_teleop_solo.setText("▶  Start (this robot)")
                self._btn_start_teleop_solo.setStyleSheet("QPushButton { background: #2ecc71; color: white; padding: 6px; }")
            self._btn_start_teleop_solo.setEnabled(ready)
            self._btn_stop_teleop_solo.setEnabled(is_waiting or is_teleop)
        # PEER controls (teleop + Unity toggles) — MASTER only, when linked. The peer's own
        # manager hides these (it's the target, not the driver).
        master_side = linked and (self._link_role() != "peer")
        peer_widgets = (self._peer_teleop_lbl, self._btn_start_teleop_peer,
                        self._btn_stop_teleop_peer, self._peer_unity_lbl,
                        self._btn_peer_audio_haptics, self._btn_peer_vibrotactile,
                        self._btn_peer_viewport)
        for w in peer_widgets:
            w.setVisible(master_side)
        if master_side:
            peer_teleop, _pr = self._peer_state()
            if peer_teleop == _S_TELEOP:
                self._btn_start_teleop_peer.setText("⏸  Pause (peer)")
                self._btn_start_teleop_peer.setStyleSheet("QPushButton { background: #f1c40f; color: black; padding: 6px; }")
            else:
                self._btn_start_teleop_peer.setText("▶  Start (peer)")
                self._btn_start_teleop_peer.setStyleSheet("QPushButton { background: #2ecc71; color: white; padding: 6px; }")
        self._btn_prepare_study.setText("🚀  Prepare Study (Calibrate All)" + bs)
        self._btn_calibrate_now.setText("📐  Calibrate Cameras Now" + bs)
        # Session/attempt model — SINGLE toggle buttons (no separate stop buttons):
        #   Start Session ⇄ End Session ; Start Attempt ⇄ End Attempt.
        session_open = self._session_open()
        if session_open:
            self._btn_start_rec.setText("■  End Session" + bs)
            self._btn_start_rec.setStyleSheet(
                "QPushButton { background: #7f8c8d; color: white; font-weight: bold; padding: 8px; }")
            self._btn_start_rec.setEnabled(True)
        else:
            self._btn_start_rec.setText("🎬  Start Session" + bs)
            self._btn_start_rec.setStyleSheet(
                "QPushButton { background: #e67e22; color: white; font-weight: bold; padding: 8px; }")
            self._btn_start_rec.setEnabled(ready and not self._session_opening)
        # Start Attempt ⇄ End Attempt. Start is one-press per (condition, attempt) — greyed
        # once this slot has been started (re-enables when Continue moves to a fresh run);
        # while recording the same button is the enabled End Attempt.
        cur_take = (self._cond_combo.currentText(), self._attempt_spin.value())
        if self._recording_active:
            self._btn_attempt.setText("⏹  End Attempt" + bs)
            self._btn_attempt.setStyleSheet(
                "QPushButton { background: #27ae60; color: white; font-weight: bold; padding: 8px; }")
            self._btn_attempt.setEnabled(True)
        else:
            self._btn_attempt.setText("●  Start Attempt" + bs)
            # A custom `background` overrides Qt's automatic disabled shading, so an already-
            # recorded (disabled) Start Attempt would still LOOK live ("unresponsive" click). The
            # :disabled rule greys it out explicitly once this slot is recorded (or teleop's not
            # ready) — re-enables when Continue advances the pointer to a fresh run.
            self._btn_attempt.setStyleSheet(
                "QPushButton { background: #c0392b; color: white; font-weight: bold; padding: 8px; }"
                "QPushButton:disabled { background: #7f8c8d; color: #d0d3d4; }")
            self._btn_attempt.setEnabled(ready and session_open
                                         and cur_take not in self._recorded_takes)
        # Mid-study save/resume.
        self._btn_save_study.setText("💾  Save State" + bs)
        self._btn_save_study.setEnabled(session_open)
        self._btn_resume_study.setText("📥  Resume Study" + bs)
        self._btn_resume_study.setEnabled(
            not session_open and os.path.exists(self._study_state_path()))
        # Trial/attempt rotation (end current take + advance) works while the session is
        # open, recording or armed.
        # The ONE Continue button: shows exactly where it goes next ("Continue to
        # <cond> Run <n>"); rolls into the next trial after the final attempt; disabled
        # while recording (Stop Attempt first) and at the very end of the session.
        tgt = self._next_target() if session_open else None
        if not session_open:
            self._btn_next.setText("⏭  Continue" + bs)
        elif tgt is None:
            self._btn_next.setText("✔  All Trials Complete — Stop Session" + bs)
        else:
            self._btn_next.setText(f"⏭  Continue to {tgt[0]} Run {tgt[1]}" + bs)
        self._btn_next.setEnabled(
            session_open and not self._recording_active and tgt is not None)
        # Redo Attempt: mid-take (discard + re-record) OR the armed post-stop escape
        # hatch (current slot already saved → discard + re-open its one-press slot).
        self._btn_redo_attempt.setText("🔄  Redo Attempt" + bs)
        self._btn_redo_attempt.setEnabled(
            self._recording_active
            or (session_open and cur_take in self._recorded_takes))
        # Redo Trial (discard the whole condition) stays mid-take only.
        self._btn_redo_trial.setText("🔄  Redo Trial" + bs)
        self._btn_redo_trial.setEnabled(
            self._recording_active
            or (session_open and any(c == cur_take[0] for (c, _a) in self._recorded_takes)))
        # Quality labelling is only meaningful while recording the attempt.
        self._btn_mark_good.setText("✓  Mark Good" + bs)
        self._btn_mark_good.setEnabled(self._recording_active)
        self._btn_mark_bad.setText("✗  Mark Bad" + bs)
        self._btn_mark_bad.setEnabled(self._recording_active)
        # These control XR-app features. Keep them actionable without gating on the
        # client-connection flag (which was blocking them): the command is enqueued and
        # sent over the unity_cmds channel when it is open, so pressing them updates the
        # XR app. Enabled once the system is up.
        for btn in (self._btn_audio_haptics, self._btn_vibrotactile, self._btn_viewport):
            btn.setEnabled(ready)

        # Wrench mode (live verification only)
        worker_present = self._wrench_worker is not None
        self._btn_wrench_live.setEnabled(worker_present)
        self._btn_wrench_tare.setEnabled(worker_present)
        self._btn_wrench_clear_tare.setEnabled(worker_present)
        self._btn_view_recording.setEnabled(
            bool(self._last_session_dir) and not self._recording_active)
        # Digital twin: send this robot's session to the master. Locked for the WHOLE
        # open session (incl. armed gaps between takes) so an incomplete session is never
        # shipped; needs a finished session + no transfer already running.
        sending = (self._send_proc is not None and self._send_proc.poll() is None)
        self._btn_send_master.setEnabled(
            bool(self._last_session_root) and not session_open and not sending)
        # Send Past Session…: usable with NO current session (it picks an old folder), just
        # not while a session is open or a transfer runs.
        if hasattr(self, "_btn_send_past"):
            self._btn_send_past.setEnabled(not session_open and not sending)
        # MASTER "Sync Peer Data" pull: usable when NO session is open and no transfer runs
        # (works with the peer manager off — it only needs the peer machine reachable).
        if hasattr(self, "_btn_pull_peer"):
            pulling = (self._pull_proc is not None and self._pull_proc.poll() is None)
            self._btn_pull_peer.setEnabled(not session_open and not sending and not pulling)
        self._btn_open_twin.setEnabled(not self._recording_active)
        building = (self._dataset_proc is not None and self._dataset_proc.poll() is None)
        # Don't build a dataset from a session that's still open (armed or recording).
        self._btn_build_dataset.setEnabled(not session_open and not building)
        # Robot identity is locked for the whole session.
        self._robot_combo.setEnabled(not session_open)
        # Crate tracking controls.
        calib_running = (self._calib_proc is not None and self._calib_proc.poll() is None)
        tracking = self._apriltag_tracker is not None
        self._btn_crate_track.setText(
            "⏹  Stop Crate Tracking" if tracking else "🎯  Start Crate Tracking")
        # Frame-shifting actions (start/stop tracking, anchor, recalibrate) are locked for
        # the WHOLE open session — re-anchoring between takes would move the world frame.
        self._btn_crate_track.setEnabled(
            not session_open and not calib_running and not self._apriltag_starting)
        self._btn_anchor_cameras.setEnabled(tracking and not session_open)
        # Bumped-camera re-anchor is allowed DURING recording (recovery), unlike the
        # setup-time Anchor Cameras — tag 0 is the fixed world origin.
        self._btn_reanchor_bump.setEnabled(tracking)
        self._btn_recalib_crate.setEnabled(not session_open and not calib_running)
        # Crate contribution (peer): toggle text + re-anchor availability.
        contributing = (self._headcam_localizer is not None
                        and getattr(self._headcam_localizer, "crate_contributing", False))
        self._btn_crate_contribute.setText(
            "⏹  Stop Contributing" if contributing else "🛰  Contribute Crate Tracking")
        self._btn_contrib_anchor.setEnabled(contributing)
        # Head-cam base localization toggle.
        self._btn_headcam.setText(
            "📍  Stop Head-Cam Localize" if self._headcam_localizer is not None
            else "📍  Start Head-Cam Localize")
        self._btn_headcam.setEnabled(
            not session_open and not self._headcam_starting)

        # Grey out every button that is a no-op under the current state (extracted for testing).
        self._update_conditional_buttons()

        # Top-bar link light + peer status (buttons above already carry the (BOTH) suffix).
        self._update_link_status()
        # Keep the trial-order cue in sync with the current condition.
        self._update_order_label()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _sync_sim(self, q_right, q_left, q_torso):
        ctrl = self._controller
        if q_right is not None:
            for i, addr in enumerate(ctrl.right_arm_qpos_addrs):
                if i < len(q_right):
                    self._data.qpos[addr] = q_right[i]
        if q_left is not None:
            for i, addr in enumerate(ctrl.left_arm_qpos_addrs):
                if i < len(q_left):
                    self._data.qpos[addr] = q_left[i]
        if q_torso is not None:
            for i, addr in enumerate(ctrl.torso_qpos_addrs):
                if i < len(q_torso):
                    self._data.qpos[addr] = q_torso[i]

    # =========================================================================
    # Shutdown
    # =========================================================================

    def closeEvent(self, event):
        self._log("Shutting down...")
        self._control_timer.stop()
        self._status_timer.stop()

        if self._session_open():
            self._on_stop_recording()

        if self._wrench_worker is not None:
            try:
                self._wrench_worker.stop()
            except Exception:
                pass

        if self._zed_recorder.is_streaming:
            self._zed_recorder.stop_streaming()

        if self._apriltag_tracker is not None:
            try:
                self._apriltag_tracker.stop()
            except Exception:
                pass

        if self._transcriber is not None:
            try:
                self._transcriber.stop_session()
                self._transcriber.stop()
            except Exception:
                pass

        if self._headcam_view is not None:
            try:
                self._headcam_view.close()
            except Exception:
                pass

        if self._headcam_localizer is not None:
            try:
                self._headcam_localizer.stop()
            except Exception:
                pass

        if self._link is not None:
            try:
                self._link.stop()
            except Exception:
                pass

        # Stop IK control thread first so it stops calling hardware
        if self._control_thread is not None:
            self._control_thread.stop_teleop()
            self._control_thread.stop()

        if self._haptics_streamer is not None:
            try:
                self._haptics_streamer.stop()
            except Exception:
                pass
        if self._motor_stats_streamer is not None:
            try:
                self._motor_stats_streamer.stop()
            except Exception:
                pass

        if self._hw_cmd_thread is not None:
            self._hw_cmd_thread.stop()

        if self._neck_interface is not None:
            try:
                self._neck_interface.shutdown()
            except Exception:
                pass

        # Damp on exit so the robot goes limp. A plain damp here (no tilt) — the
        # forward-tilt controlled fall lives on the Safe Shutdown button, which sets
        # _exit_damping_done after it has already tilted + damped, so we don't repeat it.
        if (self._hw_controller is not None and not self._args.no_exit_damping
                and getattr(self, "_hw_booted", False)
                and not getattr(self, "_exit_damping_done", False)):
            try:
                self._hw_controller.enter_damping(duration_s=0.3)
            except Exception:
                pass

        if self._motor_monitor is not None:
            try:
                self._motor_monitor.close()
            except Exception:
                pass

        if self._calib_window is not None:
            try:
                self._calib_window.close()
            except Exception:
                pass

        if self._kneeling_window is not None:
            try:
                self._kneeling_window.close()
            except Exception:
                pass

        if self._abduction_window is not None:
            try:
                self._abduction_window.close()
            except Exception:
                pass

        # Secondary windows that closeEvent previously MISSED. With the default
        # quitOnLastWindowClosed, any still-open top-level window keeps the app alive after
        # the main window closes → Quit hangs. The transcript window now auto-opens with
        # transcription, so it was always left open; close it (and the checklist, whose
        # closeEvent also saves its state) explicitly.
        for w in (getattr(self, "_transcript_window", None),
                  getattr(self, "_checklist_window", None)):
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass

        # Abort an in-flight mic self-test and delete its temp recordings.
        if getattr(self, "_mic_test_proc", None) is not None:
            try:
                self._mic_test_proc.kill()
            except Exception:
                pass
        try:
            self._cleanup_mic_phase_files()
        except Exception:
            pass

        if self._viewer_ctx is not None:
            try:
                self._viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass

        # Tear down the out-of-process WebRTC server (no-op for the in-process device).
        if self._device is not None and hasattr(self._device, "stop"):
            try:
                self._device.stop()
            except Exception:
                pass

        event.accept()
        # Guarantee the Qt event loop returns even if some other top-level window is still
        # open, so Quit can never hang on a stray window (app.exec_() → sys.exit in main()).
        QApplication.quit()


# ============================================================================
# Entry point
# ============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="User Study Manager for G1 + Inspire Hand teleoperation"
    )
    p.add_argument("--max_fr", default=1000, type=int)
    p.add_argument("--ik_rate", default=60, type=int)
    # "No hands behind the head" keep-out (see ControlThread). Rejects an IK arm target whose
    # resolved hand is behind the head plane (x < x_back) AND at head height (z > z_head), in the
    # pelvis frame (+x forward, +z up). Prevents an operator adjusting their headset from driving
    # the robot's hands behind its head. OPT-IN / default OFF: it runs a pinocchio FK inside the
    # 500 Hz control loop and needs hardware validation before it can be trusted on by default.
    p.add_argument("--head_keepout", action="store_true",
                   help="ENABLE the behind-the-head hand keep-out (off by default; experimental "
                        "— validate on hardware, it can hold the arms back if it mistriggers)")
    p.add_argument("--head_keepout_x", type=float, default=0.02,
                   help="keep-out back plane (m, pelvis +x fwd): hand blocked when x is behind "
                        "this AND above --head_keepout_z (default 0.02)")
    p.add_argument("--head_keepout_z", type=float, default=0.33,
                   help="keep-out floor height (m, pelvis +z up): only enforced above this so "
                        "low front/back reaches are unaffected (default 0.33; shoulders ≈0.29)")
    p.add_argument("--stale-pose-secs", dest="stale_pose_secs", type=float, default=0.5,
                   help="Operator-pose staleness failsafe: if no fresh Quest pose for this "
                        "many seconds mid-teleop, HOLD the current pose (stop chasing the "
                        "frozen target) and blend back smoothly when the stream recovers "
                        "— prevents 'lost control' + the recovery snap.")
    p.add_argument("--robot_comms_timeout", dest="robot_comms_timeout", type=float, default=0.3,
                   help="Robot-comms watchdog: if no LowState arrives from the G1 for this "
                        "many seconds (bumped ethernet / dropped DDS link), flag the robot "
                        "light red, warn, pause teleop, and damp the arms so they don't snap "
                        "back when the link returns. The G1 publishes state at ~500 Hz, so "
                        "0.3 s is well clear of normal jitter. NOTE: this makes the loss "
                        "VISIBLE + the reconnect safe; it cannot stop the robot's onboard "
                        "controller from taking over during the outage (secure the cable).")
    p.add_argument("--robot_powerdown_s", "--robot-powerdown-s", dest="robot_powerdown_s",
                   type=float, default=4.0,
                   help="If the G1's state stays gone for this many seconds, treat it as a "
                        "hardware power-cycle (power button / lost power) rather than a "
                        "momentary ethernet bump: switch the manager to the UNBOOTED state so "
                        "it never streams commands to the robot the instant it reconnects. The "
                        "operator re-boots (Bootup→Reboot) to reclaim control. A real power "
                        "cycle keeps DDS down for tens of seconds, so 4 s tolerates brief bumps "
                        "while reliably catching a power-off.")
    p.add_argument("--webrtc-subprocess", dest="webrtc_subprocess", action="store_true",
                   help="Run the WebRTC teleop server in its OWN process (isolated GIL) so "
                        "manager-side CPU load can't starve the event loop → eliminates the "
                        "RTT-balloon / connection-drop under recording load. Recommended on "
                        "the MASTER (heavy data collection). Same behavior otherwise; the "
                        "operator-pose CSV is then written by the subprocess. Default OFF.")
    p.set_defaults(webrtc_subprocess=False)
    p.add_argument("--show-viewer", dest="show_viewer", action="store_true",
                   help="Launch the live MuJoCo 3D viewer at startup. Default OFF — the "
                        "viewer's 50 Hz render competes for CPU/GIL with the WebRTC teleop "
                        "loop, so during a study leave it closed and open it on demand with "
                        "the 'Show Live 3D Viewer' button.")
    p.set_defaults(show_viewer=False)
    # Quest DATA-link watchdog (WiFi/WebRTC pose-stream freeze — NOT the robot DDS link).
    p.add_argument("--link-freeze-secs", dest="link_freeze_s", type=float, default=1.0,
                   help="Quest data-link watchdog: if no body_pose frame arrives for this "
                        "many seconds mid-teleop (WiFi/WebRTC drop or a starved event loop), "
                        "PAUSE teleop and damp the arms, then auto-resume when the stream is "
                        "healthy again. Escalates above --stale-pose-secs (which only HOLDS "
                        "the pose on a brief blip). Set ≥ stale-pose-secs.")
    p.add_argument("--link-pause-on-rtt", dest="link_pause_on_rtt", action="store_true",
                   help="Also pause teleop on high RTT (see --link-pause-rtt-ms), not just on "
                        "a frame freeze. OFF by default: on a loaded host the event loop can be "
                        "starved enough to inflate the MEASURED rtt to hundreds of ms while "
                        "frames still flow at full rate, so RTT-pausing would block otherwise-"
                        "usable (laggy) teleop. Enable only on a host whose loop is NOT starved "
                        "and where you want the pre-drop early trip.")
    p.set_defaults(link_pause_on_rtt=False)
    p.add_argument("--link-pause-rtt-ms", dest="link_pause_rtt_ms", type=float, default=400.0,
                   help="RTT (ms) that trips the watchdog WHEN --link-pause-on-rtt is set. A "
                        "dedicated 6 GHz link idles at single-digit ms, so 400 only fires on a "
                        "real degradation/pre-drop balloon. Ignored unless --link-pause-on-rtt.")
    p.add_argument("--link-pause-debounce-secs", dest="link_pause_debounce_s", type=float,
                   default=0.3, help="How long the bad-link condition must persist before the "
                        "watchdog pauses (debounce against one-off blips).")
    p.add_argument("--link-resume-rtt-ms", dest="link_resume_rtt_ms", type=float, default=120.0,
                   help="Auto-resume only once RTT is back at/below this (ms) — hysteresis "
                        "vs --link-pause-rtt-ms so it doesn't flap.")
    p.add_argument("--link-resume-secs", dest="link_resume_s", type=float, default=2.0,
                   help="The link must stay healthy for this many seconds before the watchdog "
                        "auto-resumes teleop (direct blend).")
    p.add_argument("--no-link-auto-resume", dest="link_auto_resume", action="store_false",
                   help="Don't auto-resume after a data-link pause — require a manual re-engage "
                        "(matches the robot-comms watchdog's manual-resume behavior).")
    p.set_defaults(link_auto_resume=True)
    # End-to-end latency: estimated constants (ms) for the stages not measurable in software.
    p.add_argument("--lat-capture-encode-ms", dest="lat_capture_encode_ms", type=float,
                   default=_LAT_EST_CAPTURE_ENCODE_MS,
                   help="Estimated ZED capture + encode latency (ms) for the video glass-to-glass budget.")
    p.add_argument("--lat-display-ms", dest="lat_display_ms", type=float,
                   default=_LAT_EST_DISPLAY_PHOTON_MS,
                   help="Estimated Quest decode->photons display latency (ms) for the video budget.")
    p.add_argument("--lat-tracking-ms", dest="lat_tracking_ms", type=float,
                   default=_LAT_EST_TRACKING_MS,
                   help="Estimated body-tracking sensor->pose latency (ms) for the teleop budget.")
    p.add_argument("--lat-actuation-ms", dest="lat_actuation_ms", type=float,
                   default=_LAT_EST_ACTUATION_MS,
                   help="Fallback robot actuation latency (ms) when the motor-feedback estimate is idle.")
    p.add_argument("--safety_filter", default=False, action="store_true")
    p.add_argument("--output_dir", type=str, default="./xr_recordings")
    p.add_argument("--ndigits", type=int, default=3)
    p.add_argument("--transcribe_model", "--transcribe-model", dest="transcribe_model",
                   default=None,
                   help="faster-whisper model id for the room transcription (persists to "
                        "config). e.g. large-v3-turbo (default), large-v3, "
                        "distil-large-v3.5, medium.en, small.en.")
    p.add_argument("--transcribe_min_sim", "--transcribe-min-sim", dest="transcribe_min_sim",
                   type=float, default=None,
                   help="Speaker-ID absolute cosine floor (persists to config; default "
                        "0.45). Lower to label more utterances, raise to reduce mis-labels.")
    p.add_argument("--transcribe_min_margin", "--transcribe-min-margin",
                   dest="transcribe_min_margin", type=float, default=None,
                   help="Speaker-ID margin the nearest voice must beat the runner-up by "
                        "(persists to config; default 0.06). Raise to be stricter.")
    p.add_argument("--transcribe_stereo", dest="transcribe_stereo",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Use 2-mic STEREO capture (dual-channel device, L=operator1 / "
                        "R=operator2, attributed by channel — no voice-ID) instead of the "
                        "1-mic voice-ID mode. Persists to config; toggle live in the UI.")
    p.add_argument("--robot-id", "--robot_id", dest="robot_id", type=int,
                   choices=(1, 2), default=None,
                   help="Which physical G1 this machine controls (1 or 2). Two "
                        "identical robots sit on opposite sides of the table; this "
                        "stamps every recording so the two robots' raw poses can be "
                        "fused into one digital twin later. If omitted, the last "
                        "selection is restored from ~/.xrt_study_robot_id.json.")
    # Crate AprilTag tracking is started from the UI (Start Crate Tracking button),
    # NOT a flag — so no flags are needed on either machine. These two are optional
    # config for the machine that has the cameras.
    p.add_argument("--apriltag-serials", "--apriltag_serials", dest="apriltag_serials",
                   nargs="*", default=None,
                   help="RealSense serials for crate tracking (default: first two found).")
    p.add_argument("--apriltag-include-headcam", dest="apriltag_include_headcam",
                   action="store_true", default=False,
                   help="Also use THIS robot's G1 head camera (via the Jetson stream) "
                        "as a 3rd statically-anchored crate camera (close, good angle).")
    # Crate-tracking cost knobs. On the MASTER the tracker's AprilTag detection runs as
    # CPU-bound Python threads that share the ONE process GIL with the single teleop IK
    # thread — full-res 30 Hz detection can starve IK and add seconds of teleop lag. These
    # defaults keep detection cheap (half-res decode + 15 Hz) so IK keeps the GIL, with
    # negligible crate-pose quality loss (edges still refine at full res; dataset is 15 Hz).
    p.add_argument("--apriltag-fps", dest="apriltag_fps", type=int, default=15,
                   help="Crate camera capture fps (lower = less GIL load; default 15).")
    p.add_argument("--apriltag-quad-decimate", dest="apriltag_quad_decimate", type=float,
                   default=2.0, help="AprilTag quad_decimate: detect on 1/N-res then refine "
                        "at full res. 2.0 ≈ 4x cheaper than 1.0 with ~no pose loss for the "
                        "close crate cams. Set 1.0 to restore full-res detection.")
    p.add_argument("--apriltag-headcam-quad-decimate", dest="apriltag_headcam_quad_decimate",
                   type=float, default=1.0,
                   help="Separate quad_decimate for the G1 head cam as a crate camera "
                        "(default 1.0 = full res). It views table tag 0 from farther than "
                        "the close RealSense crate cams, so it needs finer detection than "
                        "them; matches the head-cam localizer. Raise toward 2.0 only if CPU "
                        "is tight and the head cam still anchors.")
    p.add_argument("--apriltag-rate", dest="apriltag_rate", type=float, default=15.0,
                   help="Crate-pose fusion rate Hz (default 15; the dataset is 15 Hz).")
    # Cross-machine study link (over Tailscale) + master control board.
    p.add_argument("--link-role", dest="link_role", choices=("none", "master", "peer"),
                   default=None,
                   help="OPTIONAL override of the UI Setup bar. master = robot 1 "
                        "(control board + fuses robot 2); peer = robot 2 (obeys master).")
    p.add_argument("--link-host", dest="link_host", default=None,
                   help="OPTIONAL override. peer: the master's Tailscale IP.")
    p.add_argument("--link-port", dest="link_port", type=int, default=5070)
    p.add_argument("--link-sync-delay", dest="link_sync_delay", type=float, default=0.8,
                   help="Seconds ahead to schedule synchronized recording start/stop "
                        "(absorbs command latency; needs NTP-synced clocks).")
    p.add_argument("--link-retry-secs", dest="link_retry_secs", type=float, default=3.0,
                   help="Seconds between link (re)connect attempts. The manager auto-"
                        "connects to its saved peer on boot and keeps retrying at this "
                        "interval until the other machine comes up.")
    p.add_argument("--apriltag-bundle", "--apriltag_bundle", dest="apriltag_bundle",
                   type=str, default=str(_HERE / "crate_bundle.json"),
                   help="Crate bundle JSON from apriltag_crate_calibrate.py "
                        "(default: crate_bundle.json next to this script).")
    # Head-camera base self-localization (configured in the UI Setup bar; these
    # flags are OPTIONAL overrides only).
    p.add_argument("--headcam-host", "--headcam_host", dest="headcam_host",
                   type=str, default=None,
                   help="OPTIONAL override of the UI: IP of THIS robot's Jetson "
                        "running g1_headcam_server.py.")
    p.add_argument("--headcam-port", "--headcam_port", dest="headcam_port",
                   type=int, default=None, help="OPTIONAL override of the head-cam port.")
    p.add_argument("--headcam-optical", dest="headcam_optical",
                   choices=("ros", "identity"), default="ros",
                   help="d435_link->optical frame convention; verify once with "
                        "head_cam_localize.py.")
    p.add_argument("--headcam-rate", dest="headcam_rate", type=float, default=8.0,
                   help="Head-cam base-localization rate (Hz, default 8) — now ALSO caps the "
                        "in-manager AprilTag DETECTION on the Jetson stream (frames beyond "
                        "it are dropped pre-decode). Detecting all 30 fps in the manager "
                        "process starves the teleop IK thread (~10 s lag). The base is "
                        "static, so 8 Hz loses nothing; also drives the peer's "
                        "crate-contribution rate.")
    p.add_argument("--table-tag-size", dest="table_tag_size", type=float, default=0.065,
                   help="BASE (regular crate-face) tag edge in metres. Keep at 0.065: tags "
                        "0 and 10 are 100 mm and are handled automatically by "
                        "apriltag_crate_calibrate.TAG_SIZE_OVERRIDES — do NOT raise this to "
                        "0.1 or the 65 mm faces get mis-sized.")
    p.add_argument("--zed-localize-side", dest="zed_localize_side",
                   choices=["auto", "left", "right"], default="auto",
                   help="Which ZED camera to use for AprilTag localization. 'auto' = right "
                        "for the PEER (table tag sits to its right), left otherwise.")
    p.add_argument("--zed-conf", "--zed_conf", dest="zed_conf", type=str, default=None,
                   help="OPTIONAL path to the ZED SN<serial>.conf; its LEFT-cam intrinsics "
                        "are written into the session meta for zed_localize.py. Auto-found "
                        "via ZED_SETTINGS_PATH / /usr/local/zed/settings if omitted.")
    p.add_argument("--table-z-down", dest="table_z_down", action="store_true", default=True,
                   help="Tag 0's z-axis points DOWN into the table (verified true on "
                        "this rig) — flip the world so Z is up. Shared by the crate "
                        "tracker AND head-cam so robot+crate share one world.")
    p.add_argument("--no-table-z-down", dest="table_z_down", action="store_false",
                   help="Use if tag 0's z already points up (disables the flip).")
    p.add_argument("--dynamics", action="store_true", default=False)
    p.add_argument("--net", type=str, default=None,
                   help="Optional Unitree DDS interface name (e.g. eno1/enx00e04c560761). If omitted, CycloneDDS auto-detects the best interface.")
    p.add_argument("--hw_kp_scale", type=float, default=1.0)
    p.add_argument("--hw_kd_scale", type=float, default=1.0)
    p.add_argument("--leg_kp_scale", type=float, default=3.0)
    p.add_argument("--leg_kd_scale", type=float, default=2.0)
    p.add_argument("--dry_run", default=False, action="store_true")
    p.add_argument("--hand_only_hw", action="store_true", default=False)
    p.add_argument("--motor_temp_min", type=float, default=25.0,
                   help="Room/baseline temperature (°C) shown as white in the "
                        "motor monitor — boxes only start reddening above this.")
    p.add_argument("--motor_temp_max", type=float, default=80.0,
                   help="Temperature (°C) treated as the max for the arm motor "
                        "monitor — each joint box fades white→red as it nears this.")
    p.add_argument("--motor_torque_max", type=float, default=25.0,
                   help="Torque (Nm) treated as the limit in the motor "
                        "monitor — the torque value turns red above 85%% of this.")
    p.add_argument("--finger_temp_max", type=float, default=75.0,
                   help="Temperature (°C) treated as the max for Inspire-hand "
                        "finger actuators in the motor monitor (white→red).")
    p.add_argument("--finger_current_max", type=float, default=2000.0,
                   help="Current (mA) treated as the max for Inspire-hand finger "
                        "actuators — boxes fade white→blue toward this. Per the "
                        "Inspire R1 manual each finger's CURRENT register is 0-2000 mA.")
    p.add_argument("--no_exit_damping", action="store_true")
    p.add_argument("--shutdown_wing_secs", dest="shutdown_wing_secs", type=float, default=1.5,
                   help="Seconds for the Safe-Shutdown arm retract that runs BEFORE the "
                        "forward tilt: the arms raise straight OUT to the sides into a "
                        "drooped 'A-pose' (extended elbows, ~0.6 m apart) so they clear the "
                        "table AND the chest, and stay there through the descent. Set 0 to "
                        "skip the retract.")
    p.add_argument("--shutdown_tilt_deg", dest="shutdown_tilt_deg", type=float, default=25.0,
                   help="Forward hip tilt (degrees) applied by the Safe Shutdown button "
                        "before damping, so the robot always topples FORWARD instead of "
                        "backward. Set 0 to disable the tilt (damp-only). If your robot "
                        "leans BACKWARD instead, pass a negative value to flip it.")
    p.add_argument("--shutdown_tilt_ramp_s", dest="shutdown_tilt_ramp_s", type=float, default=1.2,
                   help="Seconds over which the Safe-Shutdown forward tilt is ramped in "
                        "(smoothstep easing, zero start/end velocity) so the lean is "
                        "smooth rather than a jerky step. Increase for a gentler tilt.")
    p.add_argument("--shutdown_tilt_hold_s", dest="shutdown_tilt_hold_s", type=float, default=0.3,
                   help="Seconds to hold the forward hip tilt (at operating stiffness) "
                        "after the ramp, before releasing into damping, so the lean "
                        "carries the robot forward as it goes limp.")
    p.add_argument("--shutdown_lower_secs", dest="shutdown_lower_secs", type=float, default=15.0,
                   help="Seconds over which Safe Shutdown folds the knees into the deeper "
                        "kneel (and holds the arms out). Larger = slower, gentler descent. "
                        "This is the main knob for the descent speed.")
    p.add_argument("--shutdown_lower_ease_power", dest="shutdown_lower_ease_power",
                   type=float, default=2.0,
                   help="Shape of the Safe-Shutdown descent: alpha=1-(1-t)^power (ease-out). "
                        ">1 makes it quicker off the top and decelerate into a softer landing "
                        "at the bottom; 1.0 is a constant-speed descent.")
    p.add_argument("--shutdown_collapse_knee_deg", dest="shutdown_collapse_knee_deg",
                   type=float, default=60.0,
                   help="How much FURTHER to flex the knees on Safe Shutdown, beyond the "
                        "operating kneel, so the robot collapses DEEPER into a lower rest "
                        "pose. Larger = lower collapse (clamped below the knee limit). Set 0 "
                        "to just settle at the current kneel.")
    p.add_argument("--shutdown_collapse_hip_deg", dest="shutdown_collapse_hip_deg",
                   type=float, default=100.0,
                   help="Extra FORWARD pelvis (hip_pitch) lean (deg) during the Safe-Shutdown "
                        "knee fold, on TOP of the initial tilt. This is the main lever that "
                        "keeps the head out in front as it lowers (the waist bow alone is "
                        "capped at 30°). Raise it to push the head further forward. Same "
                        "forward direction as --shutdown_tilt_deg.")
    p.add_argument("--shutdown_torso_bow_deg", dest="shutdown_torso_bow_deg",
                   type=float, default=30.0,
                   help="How far the torso BOWS forward at the waist (deg) as it lowers, on top "
                        "of the pelvis lean, to carry the head further forward. Positive = bow "
                        "forward (clamped to the ±30° waist_pitch joint limit; 30 = maxed). If "
                        "the torso bows BACKWARD instead, pass a negative value to flip it.")
    p.add_argument("--arm_damp_kd", type=float, default=8.0,
                   help="Damping gain (kd, kp=0) applied to the arm joints when "
                        "teleop is stopped, so the arms slowly lower under gravity "
                        "without pushing on objects while the body stays rigid.")
    p.add_argument("--kneeling", action="store_true", default=True)
    p.add_argument("--transition_duration", type=float, default=4.0,
                   help="Seconds for the staged teleop-entry choreography (lateral "
                        "raise -> back -> chicken wing -> sweep into the operator's "
                        "live pose) when teleop is STARTED. A few seconds keeps it "
                        "from snapping. 0 disables the ramp.")
    p.add_argument("--bootup_arm_back_deg", dest="bootup_arm_back_deg", type=float, default=20.0,
                   help="Degrees the arms are raised straight BACK at the shoulders on "
                        "bootup BEFORE the robot rises from its bent-over pose, and held "
                        "there through the rise, so the hands clear the table in front "
                        "instead of colliding. Increase if the hands still hit the table.")
    p.add_argument("--no_haptics", action="store_true", default=False)
    p.add_argument("--no_motor_stats", action="store_true", default=False)
    p.add_argument("--wrench_live_rate", type=float, default=10.0,
                   help="Rate (Hz) of the live end-effector wrench readout worker "
                        "in the Wrench mode. Runs off the teleop hot path on its "
                        "own MjData; default 10 Hz is plenty for a visual readout.")
    p.add_argument("--no_tactile", action="store_true", default=False,
                   help="Do not record the raw tactile/touch arrays to the "
                        "session's tactile .bin (the comprehensive robot CSV is "
                        "still written). Use to shrink session size if the touch "
                        "map is not needed.")
    p.add_argument("--haptic_deadband", type=float, default=0.0,
                   help="Noise-gate (ADC counts above baseline) below which a "
                        "finger/palm haptic intensity is forced to 0, to suppress "
                        "phantom triggers when nothing is touched. Overridden "
                        "per-region by a loaded force calibration's measured noise "
                        "floor. Keep below your Unity trigger threshold.")

    # ── eco mode (shoulder-pitch heat reduction) ────────────────────────────
    # Reduces motor duty cycle without touching active-teleop fidelity: relaxes
    # the arms only while the operator is idle, plus a thermal safety net.
    p.add_argument("--eco_mode", action=argparse.BooleanOptionalAction, default=None,
                   help="Force eco mode ON (--eco_mode) or OFF (--no-eco_mode) at boot: "
                        "eco relaxes the arms toward damping during operator idle "
                        "(re-engaging instantly on motion) and backs off near a "
                        "shoulder-pitch motor's thermal limit, to keep the shoulder-pitch "
                        "motors from overheating. Toggleable live from the GUI; the last "
                        "state persists across reboots. Omit both to restore the persisted "
                        "state. Affects only the arms.")
    p.add_argument("--eco_idle_secs", type=float, default=4.0,
                   help="Eco: seconds of no commanded arm motion before the arms "
                        "relax to the low-effort idle posture.")
    p.add_argument("--eco_motion_thresh", type=float, default=0.05,
                   help="Eco: max per-joint arm speed (rad/s) below which the arms "
                        "count as idle. Above this the idle timer resets and the "
                        "arms stay fully stiff. Set so pose jitter stays below it.")
    p.add_argument("--eco_relax_blend", type=float, default=0.7,
                   help="Eco: arm damping blend (0..1) applied when idle. 0=full "
                        "stiffness, 1=pure damping (arms hang). 0.7 lets the arms "
                        "settle to a lower-effort pose while staying roughly up.")
    p.add_argument("--eco_relax_in_secs", type=float, default=1.2,
                   help="Eco: ramp time (s) into the relaxed hold — kept slow so "
                        "the arms drift down gently rather than dropping.")
    p.add_argument("--eco_relax_out_secs", type=float, default=0.3,
                   help="Eco: ramp time (s) back to full stiffness on motion — "
                        "kept short so renewed teleop re-engages almost instantly.")
    p.add_argument("--eco_temp_derate", type=float, default=65.0,
                   help="Eco thermal governor: shoulder-pitch temperature (°C) at "
                        "which the backoff begins, even while actively teleoperating.")
    p.add_argument("--eco_temp_full", type=float, default=75.0,
                   help="Eco thermal governor: temperature (°C) at which the backoff "
                        "reaches eco_temp_max_blend. Keep below the ~80 °C firmware "
                        "thermal-fault temperature that drops the arms.")
    p.add_argument("--eco_temp_max_blend", type=float, default=0.6,
                   help="Eco thermal governor: max damping blend (0..1) the governor "
                        "may apply, so an overheating arm sags gently rather than "
                        "collapsing. The idle-relax blend can still exceed this.")

    # ── load-bearing-joint safety: position-error clamp + thermal governor ──────
    # Prevents the uncommanded 'slow hip sag → violent snap upright' failure: a fixed
    # high-kp hold on the kneeling hip/knee/waist-pitch (which carry the body against
    # gravity) accumulates a large position error as the motor thermally derates, then
    # corrects it violently when torque returns.
    p.add_argument("--leg_pos_err_clamp", "--leg-pos-err-clamp", dest="leg_pos_err_clamp",
                   type=float, default=0.20,
                   help="Max radians the LEG position command may lead the measured joint "
                        "(hard cap on the corrective torque ≈ kp·Δ, so a thermally-sagged "
                        "leg can never snap back violently — recovery is a smooth ramp). "
                        "No-op in normal operation (error ≪ Δ). Must be large enough that "
                        "kp·Δ still holds the kneel; 0 disables the leg clamp.")
    p.add_argument("--torso_pos_err_clamp", "--torso-pos-err-clamp", dest="torso_pos_err_clamp",
                   type=float, default=0.20,
                   help="Same error clamp for the TORSO/waist joints (rad). 0 disables.")
    p.add_argument("--no-pos-err-clamp", dest="no_pos_err_clamp", action="store_true",
                   help="Disable the leg/torso position-error clamp entirely (NOT "
                        "recommended — this is the guarantee against a violent snap).")
    p.add_argument("--leg-thermal-governor", dest="leg_thermal_governor",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="As the kneeling hip-pitch/waist-pitch heat up, trim THOSE joints' kp "
                        "(capped, knee untouched) so they yield gently instead of holding to "
                        "the firmware derate point. Paired with the clamp so the yield is "
                        "smooth. Warns the operator. --no-leg-thermal-governor to disable.")
    p.add_argument("--leg_temp_derate", type=float, default=60.0,
                   help="Load-joint temperature (°C) where the leg thermal governor begins "
                        "trimming hip/waist-pitch kp.")
    p.add_argument("--leg_temp_full", type=float, default=75.0,
                   help="Load-joint temperature (°C) at which the governor reaches its max "
                        "(capped) backoff — kept below the firmware thermal-fault temp.")
    p.add_argument("--leg_temp_max_blend", type=float, default=0.25,
                   help="Max kp reduction fraction (0..1) the leg governor may apply to the "
                        "hip/waist-pitch. Deliberately small — over-softening a load-bearing "
                        "joint risks the kneel collapsing.")
    p.add_argument("--waist-locked", dest="waist_locked",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="A mechanical waist lock is installed (DEFAULT ON). Idles the waist "
                        "MOTORS (kp→--waist-locked-kp-scale, damping only) so they don't "
                        "stall against the lock (wasted heat + lock/gearbox stress), and "
                        "takes the waist out of the leg thermal governor (mechanically "
                        "handled). ONLY safe with the lock actually engaged — without it the "
                        "torso would sag on damping alone. --no-waist-locked to drive the "
                        "waist normally.")
    p.add_argument("--waist-locked-kp-scale", dest="waist_locked_kp_scale", type=float,
                   default=0.0,
                   help="kp scale applied to the waist motors when --waist-locked (0 = pure "
                        "damping, let the lock hold everything; a small value gently centers "
                        "the motor if the lock has slop).")
    p.add_argument("--waist-lock-drive-yaw", dest="waist_lock_drive_yaw",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="With --waist-locked, keep the waist YAW motor DRIVEN so the upper "
                        "body is actively held facing forward (yaw goal 0) and can't rotate "
                        "side-to-side; only the roll+pitch (the load-bearing axes the lock "
                        "holds) are idled (DEFAULT ON). A boot yaw misalignment is corrected "
                        "smoothly via the torso position-error clamp. Use "
                        "--no-waist-lock-drive-yaw if your lock rigidly holds the yaw too "
                        "(idles all three waist motors, the old behaviour).")

    # ── current-triggered arm compliance (collaborative manipulation) ───────
    # When an arm joint's torque (a current proxy) nears the overcurrent limit —
    # e.g. a partner robot forcing a shared crate — soften that joint so it
    # yields, then re-stiffen and return to the matched pose when the spike passes.
    p.add_argument("--arm_compliance", action=argparse.BooleanOptionalAction, default=None,
                   help="Force current-triggered arm compliance ON (--arm_compliance) or "
                        "OFF (--no-arm_compliance) at boot: a per-joint arm joint softens "
                        "when its |torque| nears the overcurrent limit so it yields to an "
                        "external force instead of tripping, then re-stiffens and returns "
                        "to pose. Toggleable from the GUI; the last state persists across "
                        "reboots. Omit both to restore it.")
    p.add_argument("--comp_tau_start", type=float, default=12.0,
                   help="Compliance: arm-joint |tau_est| (Nm) at which softening "
                        "begins. Set ABOVE the normal holding torque so only external "
                        "pushes (not gravity hold) trigger it.")
    p.add_argument("--comp_tau_full", type=float, default=20.0,
                   help="Compliance: |tau_est| (Nm) at which softening reaches "
                        "comp_max_blend. Set just below the overcurrent/effort limit "
                        "(the G1 arm joints are rated ~25 Nm).")
    p.add_argument("--comp_max_blend", type=float, default=0.8,
                   help="Compliance: max stiffness backoff at comp_tau_full "
                        "(0=no softening, 1=fully limp). 0.8 yields strongly while "
                        "keeping a little pull toward the target.")
    p.add_argument("--comp_in_secs", type=float, default=0.1,
                   help="Compliance: ramp time (s) INTO softening — kept short so a "
                        "joint yields before the current spike trips the fault.")
    p.add_argument("--comp_out_secs", type=float, default=0.5,
                   help="Compliance: ramp time (s) back to full stiffness once the "
                        "torque drops, so the arm returns smoothly to the matched pose.")
    return p


def main():
    args = _build_argparser().parse_args()

    # A C-level crash (SIGSEGV/SIGABRT/SIGBUS/SIGFPE) — e.g. inside a native extension like the
    # Unitree SDK, MuJoCo or pinocchio — kills the process with a bare "Segmentation fault (core
    # dumped)" that no Python excepthook can catch, so we never see WHICH thread / C-call died.
    # faulthandler installs an async-signal-safe handler that dumps ALL thread stacks the instant
    # the fault happens, right before the process dies — turning the mystery peer segfault into an
    # actionable stack. Written to stderr (shows in the terminal you copy) — this is how we finally
    # pin down the post-neck-reset crash. After the NEXT crash, paste the "Fatal Python error" +
    # "Current thread … (most recent call first)" block that now precedes "Segmentation fault".
    faulthandler.enable(all_threads=True)

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName("G1 User Study Manager")

    window = StudyManagerApp(args)

    # SAFETY NET: PyQt5 >= 5.5 calls abort() on any UNCAUGHT Python exception raised
    # inside a Qt slot (button handler / timer tick) — instantly killing the manager
    # and with it the 500 Hz command stream, so a risen robot free-falls when the
    # firmware watchdog trips (this is exactly the post-bootup crash+fall incident).
    # Installing our own excepthook restores log-and-continue: the bug is printed +
    # logged, the app — and the robot's standby stream — keep running.
    def _slot_exception_hook(exc_type, exc, tb):
        import traceback as _tb
        _tb.print_exception(exc_type, exc, tb)
        try:
            window._log(f"[ERROR] uncaught exception (app kept alive): "
                        f"{exc_type.__name__}: {exc}")
        except Exception:
            pass
    sys.excepthook = _slot_exception_hook

    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
