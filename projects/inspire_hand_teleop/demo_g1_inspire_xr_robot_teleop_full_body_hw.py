"""
G1 robot teleoperation using WebRTC body pose estimation.
Integrates WebRTC teleop + SEW IK + Unitree low-level DDS control for real G1 + Inspire Hand Control.
Author: Chuizheng Kong, Yunho Cho
Date created: 2025-07-29
Modified for G1 Inspire: 2026-03-24
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

# Use MuJoCo's standard GUI backend. Avoid forcing GLX vendor overrides here,
# because that can break on some desktop sessions even when rendering works.
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.pop("__NV_PRIME_RENDER_OFFLOAD", None)
os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)

import mujoco
import mujoco.viewer
import numpy as np

from xrt_devices.study import StudyXRDevice as XRRTCBodyPoseDevice
from projects.inspire_hand_teleop.inspire_hand_haptics_streamer import InspireHandHapticsStreamer
from projects.inspire_hand_teleop.motor_stats_streamer import MotorStatsStreamer
# Add current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from projects.g1_full_body_kinematic.g1_full_body_sew_solver import G1FullBodySEWSolver
from projects.inspire_hand_teleop.g1_with_inspire_hand_mujoco_controller import G1InspireHandMuJoCoController
from projects.inspire_hand_teleop.inspire_hand_hw_controller import InspireHandController
from geo_kin_core.viz import draw_filtered_sew

import traceback
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g1_upper_body_teleop"))
from projects.g1_upper_body_teleop.config import Config
from projects.g1_upper_body_teleop.unitree_api_utils import Controller
try:
    from unitree_sdk2py.rpc.client import Client as _RpcClient
except ImportError:
    _RpcClient = None

class _G1LocoClient(_RpcClient if _RpcClient is not None else object):
    """Minimal G1 loco client — sets FSM to Damp (state 1) as a belt-and-suspenders measure."""
    def __init__(self):
        super().__init__("sport", False)
    def Init(self):
        self._SetApiVerson("1.0.0.0")
        self._RegistApi(7101, 0)
    def Damp(self):
        import json as _json
        self._Call(7101, _json.dumps({"data": 1}))

class _MotionSwitcherClient(_RpcClient if _RpcClient is not None else object):
    """Minimal motion switcher client — releases the high-level motion mode so low-level
    rt/lowcmd control takes exclusive ownership and the locomotion FSM watchdog cannot fire."""
    def __init__(self):
        super().__init__("motion_switcher", False)
    def Init(self):
        self._SetApiVerson("1.0.0.1")
        self._RegistApi(1001, 0)  # CheckMode
        self._RegistApi(1003, 0)  # ReleaseMode
    def CheckMode(self):
        import json as _json
        code, data = self._Call(1001, _json.dumps({}))
        return code, (_json.loads(data) if code == 0 and data else {})
    def ReleaseMode(self):
        import json as _json
        code, _ = self._Call(1003, _json.dumps({}))
        return code


def release_motion_mode():
    """Release the high-level locomotion FSM so low-level control has exclusive ownership."""
    if _RpcClient is None:
        raise ImportError("release_motion_mode requires unitree_sdk2py")
    try:
        msc = _MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        for attempt in range(10):
            code, result = msc.CheckMode()
            print(f"  [MSC] CheckMode attempt {attempt}: code={code}, result={result}")
            if code != 0:
                print(f"  [MSC] CheckMode RPC failed (code={code}). Motion switcher may be unreachable.")
                break
            name = result.get("name", "")
            if not name:
                print("  [MSC] No active motion mode — low-level control is exclusive.")
                break
            print(f"  [MSC] Active mode='{name}', calling ReleaseMode...")
            rel_code = msc.ReleaseMode()
            print(f"  [MSC] ReleaseMode returned code={rel_code}")
            time.sleep(1.0)
        print("[MSC] release_motion_mode() complete.")
    except Exception as e:
        print(f"[MSC] Exception in release_motion_mode: {e}")
        import traceback; traceback.print_exc()

# Import ostrich neck hardware 
# Optionally import the Ostrich neck hardware interface for physical neck actuation
try:
    from ostrich_neck.hardware import OstrichNeckHardwareInterface
    from ostrich_neck.controller import DynamixelConnectionError
except ImportError:
    OstrichNeckHardwareInterface = None
    DynamixelConnectionError = RuntimeError

_HERE = Path(__file__).parent
_XML = _HERE / "g1_with_inspire_hand" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.xml"
_URDF = _HERE / "g1_with_inspire_hand" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"


def update_sim_from_hardware(controller, model, data, q_right, q_left, q_torso):
    """Update MuJoCo simulation state from hardware feedback."""
    if q_right is not None:
        for i, addr in enumerate(controller.right_arm_qpos_addrs):
            if i < len(q_right): data.qpos[addr] = q_right[i]
    if q_left is not None:
        for i, addr in enumerate(controller.left_arm_qpos_addrs):
            if i < len(q_left): data.qpos[addr] = q_left[i]
    if q_torso is not None:
        for i, addr in enumerate(controller.torso_qpos_addrs):
            if i < len(q_torso): data.qpos[addr] = q_torso[i]

def main():
    """Main function for G1 WebRTC teleoperation on real G1 robot with Inspire Hands."""
    parser = argparse.ArgumentParser(
        description="WebRTC teleoperation demo for real G1 robot with Inspire Hands"
    )

    parser.add_argument("--max_fr", default=1000, type=int, help="Maximum frame rate (Sim Step Rate)")
    parser.add_argument("--ik_rate", default=60, type=int, help="IK update rate (Hz)")
    parser.add_argument("--safety_filter", default=False, action="store_true", help="Enable safety filter for self-collision avoidance")
    parser.add_argument("--record_data", action="store_true", help="Enable CSV recording of body pose and action data")
    parser.add_argument("--output_dir", type=str, default="./xr_recordings", help="Output directory for CSV recordings")
    parser.add_argument("--ndigits", type=int, default=3, help="Number of decimal places for recorded values")
    parser.add_argument("--dynamics", action="store_true", default=False, help="Enable dynamic simulation (mj_step) instead of kinematic (mj_forward)")
    parser.add_argument(
        "--net",
        type=str,
        default="enp132s0",
        help="Network interface used by Unitree DDS (e.g., eno1)",
    )
    parser.add_argument(
        "--hw_kp_scale",
        type=float,
        default=1.0,
        help="Scale for hardware arm/torso joint kp",
    )
    parser.add_argument(
        "--hw_kd_scale",
        type=float,
        default=1.0,
        help="Scale for hardware arm/torso joint kd",
    )
    parser.add_argument(
        "--leg_kp_scale",
        type=float,
        default=3.0,
        help="Scale for leg joint kp during kneeling (default 3x for rigid hold)",
    )
    parser.add_argument(
        "--leg_kd_scale",
        type=float,
        default=2.0,
        help="Scale for leg joint kd during kneeling (default 2x for rigid hold)",
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        action="store_true",
        help="Do not initialize/send hardware commands (sim + IK only)",
    )
    parser.add_argument(
        "--hand_only_hw",
        "--hand_only_hw",
        action="store_true",
        default=False,
        help="When not in dry-run mode, run only Inspire hand hardware and skip G1 arm hardware commands",
    )
    parser.add_argument(
        "--no_exit_damping",
        action="store_true",
        help="Skip sending damping command on exit",
    )
    parser.add_argument(
        "--kneeling",
        action="store_true",
        default=True,
        help="Hold lower body in kneeling pose (upright torso, ~90 deg knee bend) instead of standing defaults",
    )
    parser.add_argument(
        "--transition_duration",
        type=float,
        default=1.5,
        help="Seconds to blend from starting pose to teleop target on client connect (0 to disable)",
    )
    parser.add_argument(
        "--no_haptics",
        action="store_true",
        default=False,
        help="Disable haptics streaming (haptics enabled by default)",
    )
    parser.add_argument(
        "--no_motor_stats",
        action="store_true",
        default=False,
        help="Disable motor stats streaming (motor stats enabled by default)",
    )
    args = parser.parse_args()
    
    # Fixed lower-body kneeling pose: robot on its knees, shins parallel to the ground,
    # torso sitting back over the shin area for stability.
    # Joint order per leg: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
    # Geometry: for horizontal shins, knee = pi/2 - hip_pitch.
    #   hip_pitch < 0  ->  thigh tilts backward (torso shifts over shins, "sits into" kneel)
    #   hip_pitch > 0  ->  thigh tilts forward  (torso shifts over knees)
    # Tune _HIP_PITCH_KNEEL to control how far back the robot leans.
    _HIP_PITCH_KNEEL = -0.2   # ~17 deg backward lean; increase for more recline
    # ankle_pitch limit: -0.87267 to 0.5236 rad. Must stay BELOW 0.5236 or the motor
    # hits its hard joint limit, draws max current, and triggers overcurrent protection
    # after ~15 seconds. 0.4 rad gives a natural kneeling foot angle with margin to spare.
    _ANKLE_PITCH_KNEEL = 0.4
    _KNEELING_ANGLES = np.array([
        _HIP_PITCH_KNEEL, 0.0, 0.0,  np.pi / 2 - _HIP_PITCH_KNEEL, _ANKLE_PITCH_KNEEL, 0.0,   # left leg
        _HIP_PITCH_KNEEL, 0.0, 0.0,  np.pi / 2 - _HIP_PITCH_KNEEL, _ANKLE_PITCH_KNEEL, 0.0,   # right leg
    ])

    # Needs to locate the correct config path. The config logic uses config.yaml, probably in g1_upper_body_teleop
    config_path = os.path.join(_HERE.parent, "g1_upper_body_teleop", "g1.yaml")
    config = Config(config_path)
    if not args.dry_run:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        iface = args.net if args.net else None
        print(f"[DDS] Initializing Unitree DDS with interface: {iface if iface else 'autodetect'}")
        ChannelFactoryInitialize(0, iface)

    if not _XML.exists():
        print(f"Error: XML file not found at {_XML}")
        sys.exit(1)

    try:
        print(f"Loading model from: {_XML}")
        model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        data = mujoco.MjData(model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Initialize neck interface if available. Keep this optional so the demo can
    # run in dry-run / simulation-only mode without blocking on serial permissions.
    neck_interface = None
    if args.dry_run:
        print("Dry-run mode: skipping Ostrich neck hardware initialization.")
    elif OstrichNeckHardwareInterface is None:
        print("Warning: Ostrich neck hardware interface unavailable. Running simulation only.")
    else:
        try:
            neck_interface = OstrichNeckHardwareInterface()
            neck_interface.connect(move_home=True)
            print("Ostrich neck hardware connected and homed.")
        except PermissionError as exc:
            print("Warning: Failed to initialize Ostrich neck hardware: permission denied on /dev/ttyUSB0. "
                  "Run 'sudo usermod -aG dialout $USER' and log out/in to fix this.")
            neck_interface = None
        except DynamixelConnectionError as exc:
            print(f"Warning: Failed to initialize Ostrich neck hardware: {exc}")
            neck_interface = None
        except Exception as exc:  # pragma: no cover - defensive for unexpected hardware errors
            print(f"Warning: Unexpected error initializing Ostrich neck hardware: {exc}")
            neck_interface = None

    print("Initializing teleoperation system...")
    try:
        _server_channels = []
        # Feedback channels are created by the Quest client.
        device = XRRTCBodyPoseDevice(
            env=None,
            record_data=args.record_data,
            output_dir=args.output_dir,
            ndigits=args.ndigits,
            server_data_channels=_server_channels,
        ).start()
        
        # Initialize Solver
        solver_kwargs = {
            "urdf_path": str(_URDF),
            "hand_type": "inspire",
            "debug": False,
            "timing_enabled": False,
            "control_rate_hz": float(args.ik_rate),
            "elbow_filter_cutoff_hz": 2.0,
            "collision_avoidance_enabled": args.safety_filter,
        }
        solver = G1FullBodySEWSolver(**solver_kwargs)

        # Initialize Controller
        controller = G1InspireHandMuJoCoController(
            model,
            data,
            debug=False,
        )

        if args.kneeling:
            controller.q_goal_left_leg = _KNEELING_ANGLES[:6].copy()
            controller.q_goal_right_leg = _KNEELING_ANGLES[6:].copy()
        
        # Setup mocap and scaling parameters
        mocap_cartesian_scale = np.array([0.85, 0.85, 0.65])
        mocap_offset = np.array([0, 0, 0.4])
        shoulder_width_scale = 0.8
        hip_width_scale = 1.3

        print("Teleoperation system initialized successfully!")
    except Exception as e:
        print(f"Error initializing teleoperation system: {e}")
        traceback.print_exc()
        sys.exit(1)

    hw_controller = None
    hand_controller = None
    haptics_streamer = None
    motor_stats_streamer = None
    if not args.dry_run:
        try:
            if not args.hand_only_hw:
                print("Initializing real G1 hardware interface...")
                hw_controller = Controller(config)
                print("Connected to real G1 low-level DDS.")
            else:
                print("Hand-only hardware mode enabled: skipping G1 arm hardware interface.")
            
            # Initialize Hand Hardware Controller
            print("Initializing Inspire Hand hardware interface...")
            # haptics_only=True: tip+nail coalesced into 1 read/finger (6 RTTs vs 17).
            hand_controller = InspireHandController(hand_side='b', initialize_dds=False, haptics_only=True)
            # Haptic intensity = max pixel in tip or nail; send_once() called inline
            # each main-loop iteration so haptics are never blocked by thread scheduling.
            if not args.no_haptics:
                haptics_streamer = InspireHandHapticsStreamer(
                    hand_controller, device, rate_hz=120, debug=True
                )
            else:
                print("Haptics streaming disabled (--no_haptics).")
            if not args.no_motor_stats:
                motor_stats_streamer = MotorStatsStreamer(
                    hw_controller, hand_controller, device, rate_hz=10
                ).start()
            else:
                print("Motor stats streaming disabled (--no_motor_stats).")

            # Sync simulation to current robot state immediately
            if controller is not None and hw_controller is not None:
                q_torso, q_left, q_right = hw_controller.get_current_upper_body()
                update_sim_from_hardware(controller, model, data, q_right, q_left, q_torso)
                mujoco.mj_forward(model, data)
        except Exception as e:
            print(f"Error initializing real hardware: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Dry run enabled: hardware command output is disabled.")

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=True,
    ) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0, 0, 0.8]

        # Move hardware to starting pose once before entering the wait loop.
        if hw_controller is not None:
            if args.kneeling:
                # Release the high-level locomotion FSM so low-level rt/lowcmd control is
                # exclusive. Without this, the sport FSM watchdog overrides low-level commands
                # after ~10 seconds when the robot is in a non-standing pose.
                print("Releasing high-level motion mode for exclusive low-level control...")
                release_motion_mode()
                time.sleep(0.5)

            print("Moving hardware to starting pose...")
            hw_controller.move_to_pose(
                q_goal_legs=_KNEELING_ANGLES if args.kneeling else None
            )

        # Spin up a high-frequency background thread to hold the hardware pose.
        # The G1 firmware watchdog expects rt/lowcmd at ~500Hz; sending only at the
        # visualization rate (~50Hz) causes the watchdog to trigger after ~10 seconds.
        _hold_stop = threading.Event()
        _hold_legs = _KNEELING_ANGLES if args.kneeling else None

        def _hw_hold_loop():
            _last_diag = time.time()
            _cmd_count = 0
            while not _hold_stop.is_set():
                try:
                    hw_controller.send_upper_body_targets(
                        q_goal_torso=np.zeros(3),
                        q_goal_left=np.zeros(7),
                        q_goal_right=np.zeros(7),
                        q_goal_legs=_hold_legs,
                        leg_kp_scale=args.leg_kp_scale,
                        leg_kd_scale=args.leg_kd_scale,
                        ankle_kp_scale=1.0,
                        ankle_kd_scale=1.0,
                    )
                    _cmd_count += 1
                except Exception as e:
                    print(f"[hold] send error: {e}")
                time.sleep(0.002)  # 500 Hz

                now = time.time()
                if now - _last_diag >= 10.0:
                    print(f"[hold] alive: mode_machine={hw_controller.mode_machine_}, cmds={_cmd_count}")
                    _last_diag = now

        if hw_controller is not None:
            _hold_thread = threading.Thread(target=_hw_hold_loop, daemon=True, name="hw-hold")
            _hold_thread.start()

        print("\nWaiting for a WebRTC client to connect...")
        while not device.is_connected and viewer.is_running():
            if hw_controller is not None:
                q_torso_hw, q_left_hw, q_right_hw = hw_controller.get_current_upper_body()
                update_sim_from_hardware(controller, model, data, q_right_hw, q_left_hw, q_torso_hw)
            if args.kneeling and controller is not None:
                controller.q_goal_left_leg = _KNEELING_ANGLES[:6]
                controller.q_goal_right_leg = _KNEELING_ANGLES[6:]
            if controller is not None:
                controller.apply_control(engaged=True, kinematic_mode=not args.dynamics)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)

        if hw_controller is not None:
            _hold_stop.set()
            _hold_thread.join(timeout=1.0)

        if not viewer.is_running():
            print("Viewer closed before client connection.")
            sys.exit(0)

        print("Client connected! Starting simulation.")
        viewer.sync()

        # Snapshot the robot's arm pose at connect time for transition blending.
        teleop_start_time = time.time()
        q_arm_start_torso = np.zeros(3)
        q_arm_start_left = np.zeros(7)
        q_arm_start_right = np.zeros(7)
        if hw_controller is not None:
            try:
                q_arm_start_torso, q_arm_start_left, q_arm_start_right = hw_controller.get_current_upper_body()
            except Exception:
                pass
        
        viz_fps = 30
        viz_interval = 1.0 / viz_fps
        last_viz_time = 0.0

        ik_update_interval = 1.0 / args.ik_rate
        last_ik_update_time = 0.0

        _prev_connected = True  # already connected when we enter this loop

        while viewer.is_running():
            start_time = time.time()

            # Sync simulation with real robot state
            if hw_controller is not None:
                try:
                    q_torso_hw, q_left_hw, q_right_hw = hw_controller.get_current_upper_body()
                    update_sim_from_hardware(controller, model, data, q_right_hw, q_left_hw, q_torso_hw)
                except Exception as e:
                    pass

            # Detect reconnect and re-zero the neck reference.
            _curr_connected = device.is_connected if device is not None else False
            if _curr_connected and not _prev_connected:
                solver.reset()
                print("[reconnect] Client reconnected — head reference re-zeroed.")
            _prev_connected = _curr_connected

            if time.time() - last_ik_update_time >= ik_update_interval:
                if device is not None and controller is not None:
                    action = device.get_controller_state()
                    if action:
                        action['right_hka'] = None
                        action['left_hka'] = None
                        action['R_lower_upper'] = np.eye(3)

                        # Note: we are not removing action['right_fingers'] here so Inspire hands can use it later
                        solver_input = solver.preprocess_human_action(
                            action,
                            shoulder_width_scale=shoulder_width_scale,
                            hip_width_scale=hip_width_scale,
                            mocap_cartesian_scale=mocap_cartesian_scale,
                            mocap_offset=mocap_offset,
                            R_mjworld_human=None
                        )

                        current_goals = solver.solve(solver_input, engaged=True, q_current_left=None, q_current_right=None)
                        controller.set_joint_goals(current_goals)
                        
                        if hand_controller is not None:
                            hand_controller.set_joint_goals(current_goals)
                        
                        q_head = solver_input.get('q_head', None)
                        if neck_interface is not None and q_head is not None:
                            try:
                                neck_interface.move_from_angles(q_head)
                            except DynamixelConnectionError as neck_exc:
                                print(f"Warning: Ostrich neck command failed: {neck_exc}")
                                neck_interface.shutdown()
                                neck_interface = None
                            except Exception as neck_exc:  # pragma: no cover - hardware communication errors
                                print(f"Warning: Unexpected Ostrich neck error: {neck_exc}")
                                neck_interface.shutdown()
                                neck_interface = None
                            
                last_ik_update_time = time.time()
                    
            if controller is not None:
                controller.apply_control(engaged=True, kinematic_mode=not args.dynamics)
                
            if hand_controller is not None:
                hand_controller.apply_control(engaged=True)

            if haptics_streamer is not None:
                haptics_streamer.send_once()

            if hw_controller is not None and controller is not None:
                q_goal_right = controller.q_goal_right if hasattr(controller, 'q_goal_right') else None
                q_goal_left = controller.q_goal_left if hasattr(controller, 'q_goal_left') else None
                q_goal_torso = controller.q_goal_torso if hasattr(controller, 'q_goal_torso') else None
                
                # Zero out torso
                if q_goal_torso is not None:
                    q_goal_torso = np.zeros_like(q_goal_torso)
                
                if q_goal_right is not None and q_goal_left is not None and q_goal_torso is not None:
                    # Blend from snapshot pose to teleop target over transition_duration seconds.
                    if args.transition_duration > 0:
                        alpha = min(1.0, (time.time() - teleop_start_time) / args.transition_duration)
                        if alpha < 1.0:
                            q_goal_torso = (1.0 - alpha) * q_arm_start_torso + alpha * q_goal_torso
                            q_goal_left  = (1.0 - alpha) * q_arm_start_left  + alpha * q_goal_left
                            q_goal_right = (1.0 - alpha) * q_arm_start_right + alpha * q_goal_right

                    hw_controller.send_upper_body_targets(
                         q_goal_torso=q_goal_torso,
                         q_goal_left=q_goal_left,
                         q_goal_right=q_goal_right,
                         q_goal_legs=_KNEELING_ANGLES if args.kneeling else None,
                         kp_scale=args.hw_kp_scale,
                         kd_scale=args.hw_kd_scale,
                         leg_kp_scale=args.leg_kp_scale,
                         leg_kd_scale=args.leg_kd_scale,
                         ankle_kp_scale=1.0,
                         ankle_kd_scale=1.0,
                    )
            
            # Visualize
            if time.time() - last_viz_time >= viz_interval:
                viewer.user_scn.ngeom = 0
                try:
                    draw_filtered_sew(
                        viewer, solver, to_world=controller.get_sew_transform()
                    )
                except Exception:
                    pass
                
                last_viz_time = time.time()                      

            if args.dynamics:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            viewer.sync()

            elapsed = time.time() - start_time
            if elapsed < 1 / args.max_fr:
                time.sleep(1 / args.max_fr - elapsed)

    print("Shutting down...")
    if 'haptics_streamer' in dir() and haptics_streamer is not None:
        haptics_streamer.stop()
    if 'motor_stats_streamer' in dir() and motor_stats_streamer is not None:
        motor_stats_streamer.stop()
    if args.record_data:
        device.close()

    if neck_interface is not None:
        neck_interface.shutdown()
    
    if hw_controller is not None and not args.no_exit_damping:
        hw_controller.enter_damping(duration_s=0.3)


    print("Demo completed.")


if __name__ == "__main__":
    main()
