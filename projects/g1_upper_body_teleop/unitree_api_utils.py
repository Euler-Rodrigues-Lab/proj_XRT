from __future__ import annotations

from typing import Union
import numpy as np
import time

from .common.common.command_helper import (
    MotorMode,
    create_damping_cmd,
    create_zero_cmd,
    init_cmd_go,
    init_cmd_hg,
)
from .common.common.rotation_helper import get_gravity_orientation, transform_imu_data
from .config import Config


def _load_sdk():
    """Load the Unitree SDK only when physical hardware is initialized."""
    try:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__LowCmd_,
            unitree_go_msg_dds__LowState_,
            unitree_go_msg_dds__WirelessController_,
            unitree_hg_msg_dds__LowCmd_,
            unitree_hg_msg_dds__LowState_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as exc:
        raise ImportError(
            "Unitree hardware control requires unitree_sdk2py; install the "
            "proj-XRT hardware dependencies before connecting a robot."
        ) from exc
    return locals()

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        sdk = _load_sdk()
        globals().update({name: value for name, value in sdk.items() if not name.startswith("_")})

        # Initialize the policy network
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        # Monotonic time of the last LowState received from the robot. Stays put when the
        # DDS link drops (e.g. a bumped ethernet cable), so state_age() lets the manager's
        # robot-comms watchdog notice the robot is gone instead of running blind on frozen
        # state. Set in the LowState handlers below.
        self._last_state_mono = 0.0

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0
            self._mode_machine_set = False  # latch: only accept mode_machine from first LowState

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")
        
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.controller_subscriber = ChannelSubscriber(config.controller_topic, WirelessController_)
        # self.controller_subscriber.Init(self.remote_controller.set, 10)

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        # Cached command vectors for fast control updates.
        self._dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        self._dof_kps = self.config.kps + self.config.arm_waist_kps
        self._dof_kds = self.config.kds + self.config.arm_waist_kds
        self._dof_default_pos = np.concatenate(
            (self.config.default_angles, self.config.arm_waist_target), axis=0
        )

        if not (
            len(self._dof_idx)
            == len(self._dof_kps)
            == len(self._dof_kds)
            == len(self._dof_default_pos)
        ):
            raise ValueError("Invalid config: inconsistent DOF index/gain/default lengths")

        self._motor_to_default = {
            motor_idx: float(q) for motor_idx, q in zip(self._dof_idx, self._dof_default_pos)
        }
        self._motor_to_kp = {
            motor_idx: float(kp) for motor_idx, kp in zip(self._dof_idx, self._dof_kps)
        }
        self._motor_to_kd = {
            motor_idx: float(kd) for motor_idx, kd in zip(self._dof_idx, self._dof_kds)
        }

        self._upper_body_idx = self.config.arm_waist_joint2motor_idx
        if len(self._upper_body_idx) < 17:
            raise ValueError(
                f"Expected at least 17 upper-body motor indices, got {len(self._upper_body_idx)}"
            )
        self._torso_idx = self._upper_body_idx[0:3]
        self._left_idx = self._upper_body_idx[3:10]
        self._right_idx = self._upper_body_idx[10:17]

        # Ankle joints: positions 4 & 5 within each 6-DOF leg [hip_p, hip_r, hip_y, knee, ankle_p, ankle_r].
        # Kept separate so ankle gains can be scaled independently from hip/knee.
        leg_idx = self.config.leg_joint2motor_idx
        self._ankle_idx = set(leg_idx[4:6] + leg_idx[10:12])

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self._last_state_mono = time.monotonic()   # robot-comms freshness for the watchdog
        if not self._mode_machine_set:
            self.mode_machine_ = self.low_state.mode_machine
            self._mode_machine_set = True
        # self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self._last_state_mono = time.monotonic()   # robot-comms freshness for the watchdog
        # self.remote_controller.set(self.low_state.wireless_remote)

    def state_age(self) -> float:
        """Seconds since the last LowState arrived from the robot — grows without bound the
        moment the DDS link drops (the handler stops firing). ``inf`` if none ever received.
        The manager's watchdog uses this to detect a lost robot link."""
        return (time.monotonic() - self._last_state_mono) if self._last_state_mono else float("inf")

    # def WirelessControllerHandler(self, msg: WirelessController_):
    #     self.wireless_controller = msg
    #     self.remote_controller.set(msg)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        if self.config.msg_type == "hg":
            cmd.mode_machine = self.mode_machine_
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        # print("self.remote_controller.button[KeyMap.start]:", self.remote_controller.button[KeyMap.start])
        # while self.remote_controller.button[KeyMap.start] != 1:
        #     create_zero_cmd(self.low_cmd)
        #     self.send_cmd(self.low_cmd)
        #     time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        # print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_size = len(self._dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[self._dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = self._dof_idx[j]
                target_pos = self._dof_default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self._dof_kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self._dof_kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_pose(self, q_goal_legs=None, total_time=2.0, q_goal_upper=None,
                     on_step=None, leg_kp_scale=1.0, leg_kd_scale=1.0,
                     ankle_kp_scale=1.0, ankle_kd_scale=1.0):
        """
        Interpolate all DOF from current state to a target pose over total_time seconds.
        Upper body targets arm_waist_target defaults (all zeros).
        q_goal_legs (12,): leg targets [left_leg(6), right_leg(6)]; uses standing defaults if None.
        Legs are the first len(leg_joint2motor_idx) entries in _dof_idx.
        q_goal_upper (17,): upper-body targets [torso(3), left_arm(7), right_arm(7)] in
            arm_waist_joint2motor_idx order; overrides the arm_waist defaults when given.
            Pass the robot's current upper-body pose to hold the arms in place (e.g.
            while only the legs move into a kneel, so the arms don't swing to the
            default pose first).
        on_step: optional callback(q_torso, q_left, q_right) invoked each step with the
            interpolated upper-body target, so a caller can animate a sim/viewer in lock
            step with the hardware (otherwise the sim stays frozen on its last pose for
            the whole move).
        leg_kp_scale/leg_kd_scale: scale the hip/knee gains (ankles excluded).
        ankle_kp_scale/ankle_kd_scale: scale the ankle gains. Pass the SAME leg scales
            used by the ongoing send_upper_body_targets so the legs reach (and hold) the
            target pose during the move at the operating stiffness — otherwise the legs
            can sag under a weaker gain and then snap when the full-gain sends take over.
        """
        num_step = int(total_time / self.config.control_dt)
        dof_size = len(self._dof_idx)

        init_dof_pos = np.array(
            [self.low_state.motor_state[self._dof_idx[i]].q for i in range(dof_size)],
            dtype=np.float32,
        )

        target_dof_pos = self._dof_default_pos.copy().astype(np.float32)
        n_legs = len(self.config.leg_joint2motor_idx)
        if q_goal_legs is not None:
            q_goal_legs = np.asarray(q_goal_legs, dtype=np.float32).reshape(-1)
            target_dof_pos[:n_legs] = q_goal_legs
        if q_goal_upper is not None:
            q_goal_upper = np.asarray(q_goal_upper, dtype=np.float32).reshape(-1)
            target_dof_pos[n_legs:n_legs + q_goal_upper.size] = q_goal_upper

        _leg_set = set(self.config.leg_joint2motor_idx)
        for step in range(num_step):
            t = step / num_step
            # Smoothstep easing: starts and ends at zero velocity so the robot
            # accelerates gently rather than jerking to full speed immediately.
            alpha = t * t * (3.0 - 2.0 * t)
            q_cmd = init_dof_pos * (1 - alpha) + target_dof_pos * alpha
            for j in range(dof_size):
                motor_idx = self._dof_idx[j]
                kp = self._dof_kps[j]
                kd = self._dof_kds[j]
                if motor_idx in self._ankle_idx:
                    kp *= ankle_kp_scale
                    kd *= ankle_kd_scale
                elif motor_idx in _leg_set:
                    kp *= leg_kp_scale
                    kd *= leg_kd_scale
                self.low_cmd.motor_cmd[motor_idx].q = float(q_cmd[j])
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kp
                self.low_cmd.motor_cmd[motor_idx].kd = kd
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            if on_step is not None:
                ub = n_legs
                on_step(q_cmd[ub:ub + 3], q_cmd[ub + 3:ub + 10], q_cmd[ub + 10:ub + 17])
            time.sleep(self.config.control_dt)

    def get_current_upper_body(self):
        """Return current upper-body joint angles as (torso, left_arm, right_arm)."""
        q_torso = np.array([self.low_state.motor_state[i].q for i in self._torso_idx], dtype=np.float64)
        q_left = np.array([self.low_state.motor_state[i].q for i in self._left_idx], dtype=np.float64)
        q_right = np.array([self.low_state.motor_state[i].q for i in self._right_idx], dtype=np.float64)
        return q_torso, q_left, q_right

    def send_upper_body_targets(
        self,
        q_goal_torso,
        q_goal_left,
        q_goal_right,
        q_goal_legs=None,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        leg_kp_scale: float = 1.0,
        leg_kd_scale: float = 1.0,
        ankle_kp_scale: float = 1.0,
        ankle_kd_scale: float = 1.0,
        arm_damping: bool = False,
        arm_damp_kd: float = 8.0,
        arm_damp_blend: float = 1.0,
        arm_damp_blend_map=None,
        kp_scale_map=None,
    ):
        """
        Send torso/left/right upper-body targets while keeping non-commanded joints at defaults.
        Upper-body ordering follows arm_waist_joint2motor_idx in g1.yaml:
        [torso(3), left_arm(7), right_arm(7)].
        Optional q_goal_legs (12,) overrides leg defaults; ordering follows leg_joint2motor_idx:
        [left_leg(6), right_leg(6)] each as [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll].

        If arm_damping is True, the arm joints (left+right) are commanded toward
        damping mode (kp->0, kd->arm_damp_kd) so they slowly lower under gravity
        without pushing against anything. The torso and legs keep their normal
        position gains and stay rigid.

        arm_damp_blend (0..1) controls how far into damping the arms are blended,
        for a jerk-free handoff from a rigid hold:
            beta=0 -> full position hold at q_goal_left/right (normal gains)
            beta=1 -> pure damping (kp=0, kd=arm_damp_kd); the held target is
                      irrelevant since kp=0 (matches the legacy behaviour)
        Ramp beta 0->1 while keeping q_goal_left/right at the held pose to ease
        the arms from a stiff hold into limp damping. The default (1.0) preserves
        the original instantaneous pure-damping behaviour.
        """
        q_goal_torso = np.asarray(q_goal_torso, dtype=np.float64).reshape(-1)
        q_goal_left = np.asarray(q_goal_left, dtype=np.float64).reshape(-1)
        q_goal_right = np.asarray(q_goal_right, dtype=np.float64).reshape(-1)
        if q_goal_torso.size != 3 or q_goal_left.size != 7 or q_goal_right.size != 7:
            raise ValueError(
                f"Expected torso/left/right sizes (3,7,7), got "
                f"({q_goal_torso.size},{q_goal_left.size},{q_goal_right.size})"
            )

        upper_targets = np.concatenate((q_goal_torso, q_goal_left, q_goal_right), axis=0)
        targets_by_motor_idx = {
            motor_idx: float(q) for motor_idx, q in zip(self._upper_body_idx, upper_targets)
        }

        if q_goal_legs is not None:
            q_goal_legs = np.asarray(q_goal_legs, dtype=np.float64).reshape(-1)
            if q_goal_legs.size != 12:
                raise ValueError(f"Expected 12 leg DOF values, got {q_goal_legs.size}")
            targets_by_motor_idx.update(
                {motor_idx: float(q) for motor_idx, q in zip(self.config.leg_joint2motor_idx, q_goal_legs)}
            )

        _leg_set = set(self.config.leg_joint2motor_idx)
        _arm_set = set(self._left_idx) | set(self._right_idx)
        # Per-arm-joint stiffness backoff: the LARGER of the uniform damp/eco blend
        # (arm_damp_blend, applied only when arm_damping) and any per-joint entry in
        # arm_damp_blend_map {motor_idx: 0..1}. The map lets a single joint go
        # compliant (e.g. current-triggered compliance) while the rest stay stiff.
        _blend_map = arm_damp_blend_map or {}
        _base_arm_blend = float(np.clip(arm_damp_blend, 0.0, 1.0)) if arm_damping else 0.0
        for motor_idx in self._dof_idx:
            if motor_idx in _arm_set:
                # beta blends a stiff position hold (0) toward pure damping (1):
                # position stiffness shrinks while velocity damping grows, so the
                # holding torque is released gradually (no jerk) and the joint can
                # yield to an external force. Hold the commanded target throughout;
                # at beta=1 kp=0 makes the target irrelevant (pure damping).
                beta = _base_arm_blend
                _m = _blend_map.get(motor_idx)
                if _m is not None and _m > beta:
                    beta = float(_m)
                beta = 0.0 if beta < 0.0 else (1.0 if beta > 1.0 else beta)
                full_kp = self._motor_to_kp[motor_idx] * kp_scale
                full_kd = self._motor_to_kd[motor_idx] * kd_scale
                target_q = targets_by_motor_idx.get(motor_idx, self._motor_to_default[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].mode = 1
                self.low_cmd.motor_cmd[motor_idx].q = float(target_q)
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = (1.0 - beta) * full_kp
                self.low_cmd.motor_cmd[motor_idx].kd = (1.0 - beta) * full_kd + beta * float(arm_damp_kd)
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
                continue
            target_q = targets_by_motor_idx.get(motor_idx, self._motor_to_default[motor_idx])
            if motor_idx in self._ankle_idx:
                kp = self._motor_to_kp[motor_idx] * ankle_kp_scale
                kd = self._motor_to_kd[motor_idx] * ankle_kd_scale
            elif motor_idx in _leg_set:
                kp = self._motor_to_kp[motor_idx] * leg_kp_scale
                kd = self._motor_to_kd[motor_idx] * leg_kd_scale
            else:
                kp = self._motor_to_kp[motor_idx] * kp_scale
                kd = self._motor_to_kd[motor_idx] * kd_scale
            # Optional per-motor kp trim (leg/torso thermal governor softens specific
            # gravity-load joints — hip-pitch/waist-pitch — without touching the knee).
            if kp_scale_map is not None:
                kp *= float(kp_scale_map.get(motor_idx, 1.0))
            self.low_cmd.motor_cmd[motor_idx].mode = 1  # 1: enable
            self.low_cmd.motor_cmd[motor_idx].q = float(target_q)
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = kp
            self.low_cmd.motor_cmd[motor_idx].kd = kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        self.send_cmd(self.low_cmd)

    def tilt_hips_forward(self, hip_pitch_delta_rad, ramp_s=1.2, hold_s=0.3, *,
                          kp_scale=1.0, kd_scale=1.0,
                          leg_kp_scale=1.0, leg_kd_scale=1.0,
                          ankle_kp_scale=0.0, ankle_kd_scale=1.0):
        """Smoothly pitch the pelvis/torso forward at the hips, then hold it there.

        Freezes the WHOLE body at its current measured pose (so the arms/torso don't
        move) and RAMPS both leg hip_pitch joints from their current angle to
        +``hip_pitch_delta_rad`` over ``ramp_s`` seconds using smoothstep easing (zero
        velocity at the start AND end, so the lean eases in/out instead of jerking to
        the target), then holds the full tilt for ``hold_s``. The ramp starts from the
        current measured pose, so the first command matches the robot exactly — no
        step discontinuity. Streamed at operating gains. Blocking and bounded.

        Used on shutdown so the robot always falls FORWARD (never back onto its heels
        or equipment). The hip_pitch axis is +Y; with the feet planted a NEGATIVE delta
        pitches the pelvis forward toward +X (the caller's tilt-deg flag is
        positive-for-forward and applies the sign). Caller must be the sole publisher —
        stop any control-loop thread first."""
        ms = self.low_state.motor_state
        leg_idx = self.config.leg_joint2motor_idx
        # Base = current measured pose; only hip_pitch (index 0 = left, 6 = right in the
        # 12-DOF leg order [hip_pitch, hip_roll, hip_yaw, knee, ankle_p, ankle_r]) moves.
        legs0 = np.array([ms[i].q for i in leg_idx], dtype=np.float64)
        torso = np.array([ms[i].q for i in self._torso_idx], dtype=np.float64)
        left  = np.array([ms[i].q for i in self._left_idx], dtype=np.float64)
        right = np.array([ms[i].q for i in self._right_idx], dtype=np.float64)
        delta = float(hip_pitch_delta_rad)
        dt = self.config.control_dt

        def _send(frac):
            legs = legs0.copy()
            legs[0] = legs0[0] + frac * delta
            legs[6] = legs0[6] + frac * delta
            self.send_upper_body_targets(
                torso, left, right, q_goal_legs=legs,
                kp_scale=kp_scale, kd_scale=kd_scale,
                leg_kp_scale=leg_kp_scale, leg_kd_scale=leg_kd_scale,
                ankle_kp_scale=ankle_kp_scale, ankle_kd_scale=ankle_kd_scale)

        # Smooth ramp 0 -> full tilt with smoothstep easing (starts/ends at zero speed).
        num_step = max(1, int(max(0.0, float(ramp_s)) / dt))
        for step in range(num_step + 1):
            t = step / num_step
            alpha = t * t * (3.0 - 2.0 * t)
            _send(alpha)
            time.sleep(dt)
        # Hold at the full forward tilt.
        end = time.time() + max(0.0, float(hold_s))
        while time.time() < end:
            _send(1.0)
            time.sleep(dt)

    def enter_damping(self, duration_s=0.3, kd=8.0):
        """Put EVERY motor into pure damping (kp=0, tau=0, velocity gain ``kd``) and
        stream it for ~``duration_s`` so the robot goes limp under gravity without
        pushing on anything. Blocking and bounded.

        NOTE: keep ``kd`` near the default 8. With kp=0 this is a pure velocity loop,
        and a HIGH kd amplifies the motors' noisy velocity estimate into a
        high-frequency torque oscillation — the loud buzzing/whining these actuators
        make when over-damped. To lower the robot SLOWLY, do NOT raise kd; use
        ``lower_into_kneel`` (position control on the legs). Contends with any other
        publisher, so stop the control-loop thread first for it to be authoritative."""
        end = time.time() + max(0.0, float(duration_s))
        size = len(self.low_cmd.motor_cmd)
        while True:
            create_damping_cmd(self.low_cmd)          # kp=0, kd=8, tau=0 for every motor
            if kd != 8.0:
                for i in range(size):
                    self.low_cmd.motor_cmd[i].kd = float(kd)
            self.send_cmd(self.low_cmd)
            if time.time() >= end:
                break
            time.sleep(self.config.control_dt)

    def lower_into_kneel(self, kneel_legs, duration_s=10.0, ease_power=2.0, arm_kd=8.0,
                         torso_bow_rad=0.0, torso_pitch_limit=0.52,
                         arm_left_waypoints=None, arm_right_waypoints=None, *,
                         leg_kp_scale=1.0, leg_kd_scale=1.0,
                         ankle_kp_scale=0.0, ankle_kd_scale=1.0,
                         torso_kp_scale=1.0, torso_kd_scale=1.0,
                         arm_kp_scale=1.0, arm_kd_scale=1.0):
        """Slowly fold the LEGS from their current pose into the kneeling pose so the
        robot sinks DOWN/back into its kneeling rest pose gently and QUIETLY, while the
        torso BOWS forward and the arms move to their resting sides.

        ``kneel_legs`` is the 12-DOF leg target [left(6), right(6)] — including any
        forward hip_pitch tilt, which is kept so the pelvis stays leaning FORWARD the
        whole way down. The legs are eased toward it under POSITION control over
        ~``duration_s`` (ease-out) — the knees bend and the body settles down at a rate
        set by the trajectory, NOT by a gravity free-fall, so the descent is slow and,
        crucially, SILENT (the legs keep a position term, so their velocity loop never
        runs at kp=0 with a high kd — the condition that makes these motors buzz).

        The TORSO is position-controlled (``torso_kp_scale``): waist yaw/roll are held at
        their current angle (no side drift) while waist_pitch is RAMPED forward by
        ``torso_bow_rad`` over the descent (as if bowing), clamped to ±``torso_pitch_limit``.
        This drives the upper-body mass forward so it topples FORWARD when damped.

        ARMS: if ``arm_left_waypoints`` / ``arm_right_waypoints`` are given (each a list
        of 7-DOF poses, e.g. [abduct-out, at-side]), the arms are POSITION-controlled
        (``arm_kp_scale``) from their current pose PATH-interpolated through those
        waypoints across the descent — so they abduct outward first, then lower to the
        sides — synced with the knee fold; the caller then damps them at the end. If no
        waypoints are given the arms are limp (kp=0, kd=``arm_kd``) and just drop.

        The trajectory is EASE-OUT (``alpha = 1 - (1 - t)**ease_power``): a bit quicker
        off the top, decelerating into a gentle settle as it reaches the kneel. Raise
        ``duration_s`` to slow the whole knee descent. Blocking and bounded; caller must
        be the sole publisher — stop any control-loop thread first."""
        leg_idx = list(self.config.leg_joint2motor_idx)
        legs_now = np.array([self.low_state.motor_state[m].q for m in leg_idx], dtype=np.float64)
        legs_goal = np.asarray(kneel_legs, dtype=np.float64).reshape(-1)
        if legs_goal.size != len(leg_idx):
            legs_goal = legs_now.copy()               # bad target → just hold (still drops arms)
        torso_hold = {m: float(self.low_state.motor_state[m].q) for m in self._torso_idx}
        # waist_pitch is the 3rd torso motor ([yaw, roll, pitch]); it bows the torso.
        pitch_m = self._torso_idx[2]
        pitch0 = torso_hold[pitch_m]
        lim = abs(float(torso_pitch_limit))
        pitch_goal = float(np.clip(pitch0 + float(torso_bow_rad), -lim, lim))

        # Arm choreography: build a path [current, *waypoints] per arm (or None = limp).
        def _arm_path(idxs, waypoints):
            if not waypoints:
                return None
            cur = np.array([self.low_state.motor_state[m].q for m in idxs], dtype=np.float64)
            return [cur] + [np.asarray(w, dtype=np.float64).reshape(-1) for w in waypoints]

        def _path_at(wps, s):                          # piecewise-linear through waypoints
            m = len(wps)
            if m == 1:
                return wps[0]
            s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
            u = s * (m - 1); i = int(u)
            if i >= m - 1:
                return wps[-1]
            f = u - i
            return wps[i] * (1.0 - f) + wps[i + 1] * f

        left_path = _arm_path(self._left_idx, arm_left_waypoints)
        right_path = _arm_path(self._right_idx, arm_right_waypoints)
        arm_idx = set(self._left_idx) | set(self._right_idx)
        p = max(0.1, float(ease_power))
        dt = self.config.control_dt
        n = max(1, int(max(0.0, float(duration_s)) / dt))
        for step in range(n + 1):
            t = step / n
            alpha = 1.0 - (1.0 - t) ** p               # ease-out: quicker start, gentle end
            legs_cmd = legs_now * (1.0 - alpha) + legs_goal * alpha
            pitch_cmd = pitch0 * (1.0 - alpha) + pitch_goal * alpha   # bow forward over the descent
            # Legs (incl. knees): position-controlled toward the kneel target — slow, quiet.
            for j, m in enumerate(leg_idx):
                if m in self._ankle_idx:
                    kp = ankle_kp_scale * self._motor_to_kp[m]
                    kd = ankle_kd_scale * self._motor_to_kd[m]
                else:
                    kp = leg_kp_scale * self._motor_to_kp[m]
                    kd = leg_kd_scale * self._motor_to_kd[m]
                mc = self.low_cmd.motor_cmd[m]
                mc.q = float(legs_cmd[j]); mc.qd = 0.0
                mc.kp = kp; mc.kd = kd; mc.tau = 0.0
            # Torso: waist yaw/roll held at current; waist_pitch ramps forward (bow) so
            # the upper-body mass moves forward and it topples FORWARD when damped.
            for m in self._torso_idx:
                mc = self.low_cmd.motor_cmd[m]
                mc.q = pitch_cmd if m == pitch_m else torso_hold[m]
                mc.qd = 0.0
                mc.kp = torso_kp_scale * self._motor_to_kp[m]
                mc.kd = torso_kd_scale * self._motor_to_kd[m]
                mc.tau = 0.0
            # Arms: position-controlled through the abduct→side waypoints (if given),
            # else limp under normal damping (they just drop).
            left_cmd = _path_at(left_path, alpha) if left_path is not None else None
            right_cmd = _path_at(right_path, alpha) if right_path is not None else None
            for side_idxs, cmd in ((self._left_idx, left_cmd), (self._right_idx, right_cmd)):
                for k, m in enumerate(side_idxs):
                    mc = self.low_cmd.motor_cmd[m]
                    if cmd is not None:
                        mc.q = float(cmd[k]); mc.qd = 0.0
                        mc.kp = arm_kp_scale * self._motor_to_kp[m]
                        mc.kd = arm_kd_scale * self._motor_to_kd[m]
                        mc.tau = 0.0
                    else:
                        mc.q = 0.0; mc.qd = 0.0; mc.kp = 0.0; mc.kd = float(arm_kd); mc.tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(dt)

    # def default_pos_state(self):
    #     print("Enter default pos state.")
    #     print("Waiting for the Button A signal...")
    #     print("self.remote_controller.button[KeyMap.A] :", self.remote_controller.button[KeyMap.A])
    #     while self.remote_controller.button[KeyMap.A] != 1:
    #         for i in range(len(self.config.leg_joint2motor_idx)):
    #             motor_idx = self.config.leg_joint2motor_idx[i]
    #             self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
    #             self.low_cmd.motor_cmd[motor_idx].qd = 0
    #             self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
    #             self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
    #             self.low_cmd.motor_cmd[motor_idx].tau = 0
    #         for i in range(len(self.config.arm_waist_joint2motor_idx)):
    #             motor_idx = self.config.arm_waist_joint2motor_idx[i]
    #             self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
    #             self.low_cmd.motor_cmd[motor_idx].qd = 0
    #             self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
    #             self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
    #             self.low_cmd.motor_cmd[motor_idx].tau = 0
    #         self.send_cmd(self.low_cmd)
    #         time.sleep(self.config.control_dt)

        
