"""
Passive full-body Inspire sensor visualization + WebRTC haptics streaming.

Identical to demo_g1_inspire_sensor_vis_full_body_passive_hw but also runs a
WebRTC server (port 8080) that streams per-finger/palm touch averages to a
connected Unity/Quest app over the 'haptics' data channel.

Per-finger haptic intensity is derived from the tip (3×3) and nail (12×8) sensors
of each Inspire finger: intensity = max(mean(tip), mean(nail)), clamped to [0, 1].
Contact at either the fingertip or the nail surface triggers haptics on the
corresponding vibrotactile glove finger. The palm channel uses its full sensor array.
Negative ADC values (no contact) are clamped to zero before normalisation.

Usage:
    python demo_g1_inspire_sensor_vis_webrtc_haptics_hw.py
    # Connect your Quest app to port 8080 on this machine.
"""

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PyQt5 import QtCore

from projects.inspire_hand_teleop.inspire_hand_haptics_streamer import InspireHandHapticsStreamer
from projects.inspire_hand_teleop.inspire_hand_hw_controller import (
    InspireHandController,
)
from projects.inspire_hand_teleop.inspire_hand_vis_utils import (
    build_mujoco_sensor_overlay_values_from_snapshot,
    create_mujoco_sensor_overlay_runtime,
    create_dual_hand_plot_context,
    prepare_live_scalar_arrow_values,
    refresh_dual_hand_plot_context,
    render_mujoco_sensor_overlay,
)
from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice

# ── Copied helpers from sensor_vis demo (unchanged) ──────────────────────────

def _log_sensor_snapshot(hand_side: str, snapshot: Dict):
    """Log compact state and touch summaries for the selected hand(s)."""
    hand_labels = []
    if hand_side in ['l', 'b']:
        hand_labels.append('Left')
    if hand_side in ['r', 'b']:
        hand_labels.append('Right')

    for hand_label in hand_labels:
        touch = snapshot['touch'][hand_label]
        touch_means = {}
        for finger in ['fingerone', 'fingertwo', 'fingerthree', 'fingerfour', 'fingerfive', 'fingerpalm']:
            touch_means[finger] = float(np.mean(touch[finger])) if touch[finger].size else 0.0
        logging.info(
            "%s | touch: palm=%.1f f1=%.1f f2=%.1f f3=%.1f f4=%.1f f5=%.1f",
            hand_label,
            touch_means['fingerpalm'],
            touch_means['fingerone'],
            touch_means['fingertwo'],
            touch_means['fingerthree'],
            touch_means['fingerfour'],
            touch_means['fingerfive'],
        )


def apply_upper_body_hw_state_to_mujoco(mj_controller: Any, q_right, q_left, q_torso):
    if q_right is not None:
        for i, addr in enumerate(mj_controller.right_arm_qpos_addrs):
            if i < len(q_right):
                mj_controller.data.qpos[addr] = q_right[i]
    if q_left is not None:
        for i, addr in enumerate(mj_controller.left_arm_qpos_addrs):
            if i < len(q_left):
                mj_controller.data.qpos[addr] = q_left[i]
    if q_torso is not None:
        for i, addr in enumerate(mj_controller.torso_qpos_addrs):
            if i < len(q_torso):
                mj_controller.data.qpos[addr] = q_torso[i]


def convert_hand_pos_act_to_mj_radians(pos_act_values, range_list):
    if pos_act_values is None:
        return None
    pos = np.asarray(pos_act_values, dtype=float)
    if pos.size < 6:
        return None
    pos = np.clip(pos[:6], 0.0, 2000.0)
    ranges = np.asarray(range_list[:6], dtype=float)
    return (pos / 2000.0) * ranges


def apply_inspire_snapshot_to_mujoco_hands(mj_controller: Any, snapshot: Dict, hand_controller: InspireHandController):
    if snapshot is None or hand_controller is None:
        return
    states = snapshot.get('states', {})
    left_pos_act = states.get('Left', {}).get('POS_ACT')
    right_pos_act = states.get('Right', {}).get('POS_ACT')
    left_rad = convert_hand_pos_act_to_mj_radians(left_pos_act, hand_controller.range_list)
    right_rad = convert_hand_pos_act_to_mj_radians(right_pos_act, hand_controller.range_list)
    if left_rad is not None:
        for i, addr in enumerate(mj_controller.left_hand_qpos_addrs):
            if i < left_rad.size:
                mj_controller.data.qpos[addr] = left_rad[i]
    if right_rad is not None:
        for i, addr in enumerate(mj_controller.right_hand_qpos_addrs):
            if i < right_rad.size:
                mj_controller.data.qpos[addr] = right_rad[i]


def update_mirror_sim_from_hw_state(
    mj_controller: Any,
    mj_model: Any,
    mj_data: Any,
    hw_controller: Any,
    hand_controller: InspireHandController,
    q_right=None,
    q_left=None,
    q_torso=None,
    snapshot: Dict | None = None,
):
    if hw_controller is not None and (q_right is None or q_left is None or q_torso is None):
        q_torso, q_left, q_right = hw_controller.get_current_upper_body()
    if hw_controller is not None:
        apply_upper_body_hw_state_to_mujoco(mj_controller, q_right, q_left, q_torso)
    if hand_controller is not None and snapshot is None:
        snapshot = hand_controller.read()
    if hand_controller is not None:
        apply_inspire_snapshot_to_mujoco_hands(mj_controller, snapshot, hand_controller)
    import mujoco
    mujoco.mj_forward(mj_model, mj_data)


def _is_hand_related_mujoco_geom(model: Any, geom_id: int, *, include_force_sensor_meshes: bool = False) -> bool:
    import mujoco
    text_parts: list[str] = []
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if geom_name:
        text_parts.append(geom_name.lower())
    body_id = int(model.geom_bodyid[geom_id])
    if body_id >= 0:
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name:
            text_parts.append(body_name.lower())
    if int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            if mesh_name:
                text_parts.append(mesh_name.lower())
    haystack = " ".join(text_parts)
    if not haystack:
        return False
    if not include_force_sensor_meshes and "force_sensor" in haystack:
        return False
    hand_tokens = ("inspire", "hand", "finger", "thumb", "palm", "little", "index", "middle", "ring")
    return any(token in haystack for token in hand_tokens)


def apply_hand_related_geom_transparency(model: Any, alpha: float, *, include_force_sensor_meshes: bool = False) -> int:
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    updated_geoms = 0
    for geom_id in range(model.ngeom):
        if not _is_hand_related_mujoco_geom(model, geom_id, include_force_sensor_meshes=include_force_sensor_meshes):
            continue
        model.geom_rgba[geom_id, 3] = alpha_clamped
        updated_geoms += 1
    return updated_geoms


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    # ── WebRTC server via XRRTCBodyPoseDevice ─────────────────────────────────
    # Using XRRTCBodyPoseDevice registers the body_pose and apriltag_pose
    # channels that the Quest app expects during its connection handshake.
    # env=None because we are not driving a simulation — passive sensor vis only.
    xr_device = XRRTCBodyPoseDevice(env=None, server_data_channels=["haptics"])
    logging.info("[WebRTC] Haptics + body-pose server started on port 8080.")
    logging.info("[WebRTC] Connect your Quest app — waiting for peer connection...")

    # ── Inspire hand ──────────────────────────────────────────────────────────
    hand = InspireHandController(hand_side='b', initialize_dds=False)
    haptics_streamer = InspireHandHapticsStreamer(hand, xr_device, rate_hz=60, debug=True)

    dt = 20                             # Qt timer interval ms (~50 Hz)
    phase_duration = 3.0
    finger_goal = 0.85 * 1.4381
    finger_start = 0.2 * 1.4381
    thumb_goal = 0.0

    TEST_UNITREE_NET_IFACE = "enp0s31f6"
    TEST_G1_INIT_TIMEOUT_SEC = 5.0
    TEST_LIVE_ARROW_NORMALIZATION_MODE = "running"
    TEST_LIVE_ARROW_SMOOTHING_ALPHA = 0.35
    TEST_LIVE_ARROW_DEADBAND = 0.02
    TEST_LIVE_ARROW_FIXED_MAX = 1024.0
    TEST_LIVE_ARROW_RUNNING_DECAY = 0.995
    TEST_HAND_MODEL_ALPHA: float | None = 0.45
    TEST_INCLUDE_FORCE_SENSOR_IN_HAND_ALPHA = False

    def _plus_x_override() -> dict[str, object]:
        return {
            'axis': np.array([1.0, 0.0, 0.0], dtype=np.float64),
            'frame': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        }

    SENSOR_DIRECTION_OVERRIDES: dict[str, dict[str, object]] = {}
    for hand_name in ("left", "right"):
        SENSOR_DIRECTION_OVERRIDES[f"{hand_name}_palm_force_sensor"] = _plus_x_override()
        SENSOR_DIRECTION_OVERRIDES[f"{hand_name}_thumb_force_sensor_1"] = _plus_x_override()
        SENSOR_DIRECTION_OVERRIDES[f"{hand_name}_thumb_force_sensor_3"] = _plus_x_override()
        for finger_name in ("little", "ring", "middle", "index"):
            SENSOR_DIRECTION_OVERRIDES[f"{hand_name}_{finger_name}_force_sensor_1"] = _plus_x_override()
            SENSOR_DIRECTION_OVERRIDES[f"{hand_name}_{finger_name}_force_sensor_2"] = _plus_x_override()

    open_goal_vector = [finger_start, finger_start, finger_start, finger_start, thumb_goal, thumb_goal]
    curl_goal_vector = [finger_goal, finger_goal, finger_goal, finger_goal, thumb_goal, thumb_goal]

    hand.apply_control(engaged=True)
    plot_ctx = create_dual_hand_plot_context(hand.RegAddr, plot_touch=True)
    app = plot_ctx['app']

    mirror_runtime = {
        'ctx': None, 'viewer': None, 'model': None,
        'data': None, 'controller': None, 'sensor_overlay': None,
    }

    class _PassiveG1UpperBodyReader:
        def __init__(self, controller):
            self._controller = controller
        def get_current_upper_body(self):
            return self._controller.get_current_upper_body()

    # ── MuJoCo mirror viewer (optional) ──────────────────────────────────────
    try:
        import mujoco
        import mujoco.viewer as mujoco_viewer
        from projects.inspire_hand_teleop.g1_with_inspire_hand_mujoco_controller import G1InspireHandMuJoCoController

        xml_path = Path(__file__).parent / "g1_with_inspire_hand" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.xml"
        if xml_path.exists():
            mirror_model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
            if TEST_HAND_MODEL_ALPHA is not None:
                updated = apply_hand_related_geom_transparency(
                    mirror_model, TEST_HAND_MODEL_ALPHA,
                    include_force_sensor_meshes=TEST_INCLUDE_FORCE_SENSOR_IN_HAND_ALPHA,
                )
                logging.info(
                    "Applied hand-part transparency alpha=%.2f to %d geoms.",
                    float(TEST_HAND_MODEL_ALPHA), updated,
                )
            mirror_data = mujoco.MjData(mirror_model)
            mirror_controller = G1InspireHandMuJoCoController(mirror_model, mirror_data, debug=False)
            mirror_ctx = mujoco_viewer.launch_passive(
                model=mirror_model, data=mirror_data,
                show_left_ui=False, show_right_ui=True,
            )
            mirror_viewer = mirror_ctx.__enter__()
            mirror_viewer.cam.distance = 2.0
            mirror_viewer.cam.azimuth = 135
            mirror_viewer.cam.elevation = -10
            mirror_viewer.cam.lookat[:] = [0, 0, 0.8]
            mirror_runtime.update({
                'ctx': mirror_ctx, 'viewer': mirror_viewer,
                'model': mirror_model, 'data': mirror_data,
                'controller': mirror_controller,
            })
            try:
                sensor_overlay = create_mujoco_sensor_overlay_runtime(
                    mirror_model, sensor_direction_overrides=SENSOR_DIRECTION_OVERRIDES,
                )
                sensor_overlay['live_arrow_cfg'] = {
                    'normalization_mode': TEST_LIVE_ARROW_NORMALIZATION_MODE,
                    'smoothing_alpha': TEST_LIVE_ARROW_SMOOTHING_ALPHA,
                    'deadband': TEST_LIVE_ARROW_DEADBAND,
                    'fixed_max': TEST_LIVE_ARROW_FIXED_MAX,
                    'running_decay': TEST_LIVE_ARROW_RUNNING_DECAY,
                }
                sensor_overlay['running_max'] = 0.0
                sensor_overlay['smoothed_values'] = None
                mirror_runtime['sensor_overlay'] = sensor_overlay
                logging.info("MuJoCo sensor overlay ready with %d points.", sensor_overlay['count'])
                region_counts = sensor_overlay.get('region_counts', {})
                if region_counts:
                    logging.info(
                        "Sensor point mapping: %s",
                        ", ".join(f"{n}={c}" for n, c in sorted(region_counts.items())),
                    )
                unmapped = sensor_overlay.get('unmapped_meshes', [])
                if unmapped:
                    logging.warning("Unmapped sensor meshes (%d). Sample: %s", len(unmapped), ", ".join(unmapped[:8]))
            except Exception as overlay_err:
                logging.warning("MuJoCo sensor overlay unavailable: %s", overlay_err)
            logging.info("MuJoCo mirror viewer started: %s", xml_path)
        else:
            logging.warning("MuJoCo mirror XML not found, skipping: %s", xml_path)
    except Exception as e:
        logging.warning("Failed to initialize MuJoCo mirror viewer: %s", e)

    # ── Optional Unitree upper-body reader ───────────────────────────────────
    g1_hw_reader = None
    try:
        g1_upper_body_root = (Path(__file__).parent.parent / "g1_upper_body_teleop").as_posix()
        if g1_upper_body_root not in sys.path:
            sys.path.append(g1_upper_body_root)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from projects.g1_upper_body_teleop.config import Config
        from projects.g1_upper_body_teleop.unitree_api_utils import Controller

        net_iface = TEST_UNITREE_NET_IFACE or os.environ.get("UNITREE_NET_IFACE", "enp0s31f6")
        config_path = Path(__file__).parent.parent / "g1_upper_body_teleop" / "g1.yaml"
        init_timeout_sec = float(os.environ.get("G1_INIT_TIMEOUT_SEC", str(TEST_G1_INIT_TIMEOUT_SEC)))
        init_result: dict[str, Any] = {}
        init_error: dict[str, Exception] = {}
        init_done = threading.Event()

        def _init_g1_reader():
            try:
                ChannelFactoryInitialize(0, net_iface)
                init_result['reader'] = _PassiveG1UpperBodyReader(Controller(Config(config_path.as_posix())))
            except Exception as exc:
                init_error['error'] = exc
            finally:
                init_done.set()

        threading.Thread(target=_init_g1_reader, daemon=True).start()
        init_done.wait(init_timeout_sec)
        if not init_done.is_set():
            logging.warning("Unitree upper-body init timed out after %.1fs.", init_timeout_sec)
        elif 'error' in init_error:
            raise init_error['error']
        else:
            g1_hw_reader = init_result['reader']
            logging.info("Unitree upper-body reader enabled on iface=%s.", net_iface)
    except Exception as e:
        logging.warning("Unitree upper-body reader unavailable: %s", e)

    # ── Shared debug state ────────────────────────────────────────────────────
    g1_debug_state = {'last_log_t': 0.0, 'warned_no_reader': False}

    logging.info(
        "Qt timer started: dt=%dms  phase_duration=%.2fs  finger_goal=%.3f  thumb_goal=%.3f",
        dt, phase_duration, finger_goal, thumb_goal,
    )

    cycle_state = {'start': time.monotonic(), 'last_phase': None}
    shutdown_state = {'closing': False, 'cleaned': False}

    # ── Cleanup / signal handling ─────────────────────────────────────────────
    def cleanup():
        if shutdown_state['cleaned']:
            return
        shutdown_state['cleaned'] = True
        shutdown_state['closing'] = True
        try:
            timer.stop()
            timer.timeout.disconnect(on_timer_tick)
        except Exception:
            pass
        try:
            signal_timer.stop()
        except Exception:
            pass
        try:
            hand.apply_control(engaged=False)
        except Exception:
            pass
        try:
            if mirror_runtime['ctx'] is not None:
                mirror_runtime['ctx'].__exit__(None, None, None)
        except Exception:
            pass
        try:
            haptics_streamer.stop()
        except Exception:
            pass
        hand.stop()

    def _handle_sigint(_sig, _frame):
        if shutdown_state['closing']:
            return
        logging.info("SIGINT received, shutting down...")
        shutdown_state['closing'] = True
        QtCore.QTimer.singleShot(0, app.quit)

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Qt timer tick ─────────────────────────────────────────────────────────
    def on_timer_tick():
        if shutdown_state['closing']:
            return
        try:
            elapsed = time.monotonic() - cycle_state['start']
            phase = 'open' if int(elapsed / phase_duration) % 2 == 0 else 'curl'
            active_goal_vector = open_goal_vector if phase == 'open' else curl_goal_vector
            hand.set_joint_goals({
                'q_goal_left_hand': active_goal_vector.copy(),
                'q_goal_right_hand': active_goal_vector.copy(),
            })
            if phase != cycle_state['last_phase']:
                logging.info(
                    "Phase → %s  goals=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                    phase, *active_goal_vector,
                )
                cycle_state['last_phase'] = phase

            snapshot = hand.read()
            refresh_dual_hand_plot_context(plot_ctx, snapshot)

            # ── Send haptics over WebRTC (every tick, ~12 Hz) ─────────────────
            haptics_streamer.send_once(snapshot=snapshot)

            # ── MuJoCo mirror ─────────────────────────────────────────────────
            mirror_viewer = mirror_runtime['viewer']
            if mirror_viewer is not None:
                if mirror_viewer.is_running():
                    q_torso_hw = q_left_hw = q_right_hw = None
                    if g1_hw_reader is not None:
                        try:
                            q_torso_hw, q_left_hw, q_right_hw = g1_hw_reader.get_current_upper_body()
                        except Exception as e:
                            now = time.monotonic()
                            if now - g1_debug_state['last_log_t'] > 2.0:
                                logging.warning("Failed to read Unitree upper-body state: %s", e)
                                g1_debug_state['last_log_t'] = now
                    else:
                        now = time.monotonic()
                        if not g1_debug_state['warned_no_reader'] and now - g1_debug_state['last_log_t'] > 2.0:
                            logging.warning("Unitree upper-body reader unavailable; mirror is hand-only.")
                            g1_debug_state['warned_no_reader'] = True
                            g1_debug_state['last_log_t'] = now

                    update_mirror_sim_from_hw_state(
                        mirror_runtime['controller'], mirror_runtime['model'], mirror_runtime['data'],
                        hw_controller=g1_hw_reader, hand_controller=hand,
                        q_right=q_right_hw, q_left=q_left_hw, q_torso=q_torso_hw,
                        snapshot=snapshot,
                    )

                    sensor_overlay = mirror_runtime.get('sensor_overlay')
                    if sensor_overlay is not None:
                        raw_sensor_values = build_mujoco_sensor_overlay_values_from_snapshot(
                            snapshot, sensor_overlay['sensor_points'], normalize=False,
                        )
                        sensor_values = prepare_live_scalar_arrow_values(
                            raw_sensor_values, sensor_overlay,
                            **sensor_overlay.get('live_arrow_cfg', {}),
                        )
                        render_mujoco_sensor_overlay(
                            mirror_viewer, mirror_runtime['data'],
                            sensor_overlay['sensor_points'], sensor_values,
                        )

                    now = time.monotonic()
                    if q_torso_hw is not None and now - g1_debug_state['last_log_t'] > 2.0:
                        logging.info(
                            "Unitree upper-body | torso=%s left0=%.3f right0=%.3f",
                            np.array2string(np.asarray(q_torso_hw), precision=3),
                            float(q_left_hw[0]) if q_left_hw is not None and len(q_left_hw) > 0 else float('nan'),
                            float(q_right_hw[0]) if q_right_hw is not None and len(q_right_hw) > 0 else float('nan'),
                        )
                        g1_debug_state['last_log_t'] = now
                    mirror_viewer.sync()
                else:
                    logging.info("MuJoCo mirror viewer closed; continuing Qt hand view only.")
                    try:
                        if mirror_runtime['ctx'] is not None:
                            mirror_runtime['ctx'].__exit__(None, None, None)
                    except Exception:
                        pass
                    mirror_runtime.update({'ctx': None, 'viewer': None, 'model': None,
                                           'data': None, 'controller': None, 'sensor_overlay': None})
        except KeyboardInterrupt:
            _handle_sigint(None, None)
        except Exception as e:
            logging.error("Error in Qt timer loop: %s", e)

    timer = QtCore.QTimer()
    timer.timeout.connect(on_timer_tick)
    timer.start(dt)
    app.aboutToQuit.connect(cleanup)

    signal_timer = QtCore.QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(200)

    plot_ctx['window']._timer = timer
    plot_ctx['window']._signal_timer = signal_timer
    plot_ctx['window']._hand = hand

    plot_ctx['window'].show()
    app.exec_()
