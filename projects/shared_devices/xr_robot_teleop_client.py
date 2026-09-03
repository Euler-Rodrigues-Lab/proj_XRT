import asyncio
import csv
import json as _json
import os
import queue as _queue
import threading
import time
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation
from xr_robot_teleop_server import configure_logging
from xr_robot_teleop_server.schemas.body_pose import (
    Bone,
    deserialize_pose_data,
)
from xr_robot_teleop_server.schemas.openxr_skeletons import FullBodyBoneId
from xr_robot_teleop_server.streaming import WebRTCServer

try:
    from robosuite.devices import Device
except ImportError:
    class Device:
        """Fallback base for the standalone G1 ``env=None`` use case."""

        def __init__(self, env=None):
            if env is not None:
                raise ImportError("env-backed devices require the optional robosuite package")
            self.env = None


# Finger bones for hand tracking
simplified_finger_bones = [
    FullBodyBoneId.FullBody_LeftHandThumbMetacarpal,
    FullBodyBoneId.FullBody_LeftHandThumbProximal,
    FullBodyBoneId.FullBody_LeftHandThumbDistal,
    FullBodyBoneId.FullBody_LeftHandThumbTip,
    FullBodyBoneId.FullBody_LeftHandIndexProximal,
    FullBodyBoneId.FullBody_LeftHandIndexIntermediate,
    FullBodyBoneId.FullBody_LeftHandIndexTip,
    FullBodyBoneId.FullBody_LeftHandMiddleProximal,
    FullBodyBoneId.FullBody_LeftHandMiddleIntermediate,
    FullBodyBoneId.FullBody_LeftHandMiddleTip,
    FullBodyBoneId.FullBody_LeftHandRingProximal,
    FullBodyBoneId.FullBody_LeftHandRingIntermediate,
    FullBodyBoneId.FullBody_LeftHandRingTip,
    FullBodyBoneId.FullBody_LeftHandLittleProximal,
    FullBodyBoneId.FullBody_LeftHandLittleIntermediate,
    FullBodyBoneId.FullBody_LeftHandLittleTip,
    FullBodyBoneId.FullBody_RightHandThumbMetacarpal,
    FullBodyBoneId.FullBody_RightHandThumbProximal,
    FullBodyBoneId.FullBody_RightHandThumbDistal,
    FullBodyBoneId.FullBody_RightHandThumbTip,
    FullBodyBoneId.FullBody_RightHandIndexProximal,
    FullBodyBoneId.FullBody_RightHandIndexIntermediate,
    FullBodyBoneId.FullBody_RightHandIndexTip,
    FullBodyBoneId.FullBody_RightHandMiddleProximal,
    FullBodyBoneId.FullBody_RightHandMiddleIntermediate,
    FullBodyBoneId.FullBody_RightHandMiddleTip,
    FullBodyBoneId.FullBody_RightHandRingProximal,
    FullBodyBoneId.FullBody_RightHandRingIntermediate,
    FullBodyBoneId.FullBody_RightHandRingTip,
    FullBodyBoneId.FullBody_RightHandLittleProximal,
    FullBodyBoneId.FullBody_RightHandLittleIntermediate,
    FullBodyBoneId.FullBody_RightHandLittleTip,
]


def _get_body_frame(
    bone_positions: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculates the body-centric coordinate frame (origin and rotation matrix).

    Args:
        bone_positions (dict): A dictionary mapping bone IDs to their positions.

    Returns:
        A tuple containing:
        - np.ndarray: The origin of the body frame (shoulder_center).
        - np.ndarray: The rotation matrix from world to body frame (R_world_body).
        Returns (None, None) if essential bones are missing.
    """
    # Get key body landmarks.
    left_shoulder = bone_positions.get(FullBodyBoneId.FullBody_LeftShoulder)
    right_shoulder = bone_positions.get(FullBodyBoneId.FullBody_RightShoulder)
    left_hip = bone_positions.get(FullBodyBoneId.FullBody_LeftUpperLeg)
    right_hip = bone_positions.get(FullBodyBoneId.FullBody_RightUpperLeg)
    spine_middle = bone_positions.get(FullBodyBoneId.FullBody_SpineMiddle)

    if left_shoulder is None or right_shoulder is None or spine_middle is None:
        return None, None
    if left_hip is None or right_hip is None:
        return None, None

    # Body origin and "up" vector from shoulders — same convention as the original
    # system, so the euler-angle decomposition for neck yaw/pitch stays 1:1.
    shoulder_center = (left_shoulder + right_shoulder) / 2

    # Y-axis (lateral) from HIPS, not shoulders.  Hip positions are completely
    # decoupled from arm kinematics: extending either arm never moves the hips,
    # so the body frame's lateral axis stays fixed and the neck cannot be disturbed.
    # Shoulders (even clavicles) drift slightly with scapular movement.
    y_axis = left_hip - right_hip
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    torso_vector = shoulder_center - spine_middle

    # X-axis: forward direction (cross product)
    x_axis = np.cross(-torso_vector, y_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Z-axis: up direction (cross product)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # Create transformation matrix from world to body-centric frame
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    # normalize the rotation matrix
    U_rot, _, V_rot_T = np.linalg.svd(rotation_matrix)
    R_world_body = U_rot @ V_rot_T

    return shoulder_center, R_world_body


def _get_lower_body_frame(
    bone_positions: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculates the lower body-centric coordinate frame (origin and rotation matrix).

    Args:
        bone_positions (dict): A dictionary mapping bone IDs to their positions.

    Returns:
        A tuple containing:
        - np.ndarray: The origin of the lower body frame (hip_center).
        - np.ndarray: The rotation matrix from world to lower body frame (R_world_lower_body).
        Returns (None, None) if essential bones are missing.
    """
    # Get key lower body landmarks
    left_hip = bone_positions.get(FullBodyBoneId.FullBody_LeftUpperLeg)
    right_hip = bone_positions.get(FullBodyBoneId.FullBody_RightUpperLeg)
    hip_bone = bone_positions.get(FullBodyBoneId.FullBody_Hips)
    # Calculate lower body center (origin of the lower body frame)
    hip_center = (left_hip + right_hip) / 2.0

    spine_middle = bone_positions.get(FullBodyBoneId.FullBody_SpineMiddle)
    torso_vector = -(hip_center - spine_middle)

    if left_hip is None or right_hip is None or spine_middle is None:
        return None, None

    # Create lower body-centric coordinate frame
    # Y-axis: right to left (hip line)
    y_axis = left_hip - right_hip
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    # X-axis: forward direction (cross product)
    x_axis = np.cross(y_axis, torso_vector)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Z-axis: up direction (cross product)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # Create transformation matrix from world to lower body-centric frame
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    # normalize the rotation matrix
    U_rot, _, V_rot_T = np.linalg.svd(rotation_matrix)
    R_world_lower_body = U_rot @ V_rot_T

    return hip_center, R_world_lower_body


def _get_body_centric_coordinates(bones: list[Bone], get_wrist_rot_from_hand=False) -> dict:
    """
    Convert bone positions to a body-centric coordinate system for dual Kinova3 robot.
    """
    bone_positions = {b.id: np.array(b.position) for b in bones}
    bone_rotations = {b.id: np.array(b.rotation) for b in bones}

    shoulder_center, R_world_body = _get_body_frame(bone_positions)

    if shoulder_center is None or R_world_body is None:
        return None

    def transform_to_body_frame(world_pos):
        """Transform a world position to body-centric coordinates."""
        translated = world_pos - shoulder_center
        body_pos = R_world_body.T @ translated
        return body_pos

    # Extract SEW coordinates in body-centric frame
    sew_coordinates = {}

    for side in ["left", "right"]:
        side_key_pascal = side.capitalize()

        ArmUpper_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}ArmUpper")
        shoulder_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}Shoulder")
        elbow_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}ArmLower")
        wrist_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}HandWrist")

        ArmUpper_pos = bone_positions.get(ArmUpper_id)
        shoulder_pos = bone_positions.get(shoulder_id)
        shoulder_pos_act = (ArmUpper_pos + shoulder_pos) / 2.0
        elbow_pos = bone_positions.get(elbow_id)
        wrist_pos = bone_positions.get(wrist_id)

        if shoulder_pos is None or elbow_pos is None or wrist_pos is None:
            sew_coordinates[side] = None
            continue

        # Transform to body-centric coordinates
        S_body = transform_to_body_frame(ArmUpper_pos)
        E_body = transform_to_body_frame(elbow_pos)
        W_body = transform_to_body_frame(wrist_pos)

        # Get wrist rotation
        
        if get_wrist_rot_from_hand:
            wrist_rot = (
            q2R(bone_rotations.get(wrist_id)) if wrist_id in bone_rotations else None
        )
        else:
            wrist_rot = R_world_body.T @ _get_hand_frame(side=side, bone_positions=bone_positions)
        body_frame_wrist_rot = wrist_rot

        sew_coordinates[side] = {
            "S": S_body,
            "E": E_body,
            "W": W_body,
            "wrist_rot": body_frame_wrist_rot.flatten(),
        }

        sew_coordinates['R_world_body'] = R_world_body
        sew_coordinates['shoulder_center'] = shoulder_center

    return sew_coordinates


def _get_lower_body_centric_coordinates(bones: list[Bone]) -> dict:
    """
    Convert bone positions to a lower body-centric coordinate system.
    """
    bone_positions = {b.id: np.array(b.position) for b in bones}
    bone_rotations = {b.id: np.array(b.rotation) for b in bones}

    hip_center, R_world_lower_body = _get_lower_body_frame(bone_positions)

    if hip_center is None or R_world_lower_body is None:
        return None

    def transform_to_lower_body_frame(world_pos):
        """Transform a world position to lower body-centric coordinates."""
        translated = world_pos - hip_center
        lower_body_pos = R_world_lower_body.T @ translated
        return lower_body_pos

    # Extract key lower body coordinates in lower body-centric frame
    lower_body_coordinates = {}

    for side in ["left", "right"]:
        side_key_pascal = side.capitalize()

        hip_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}UpperLeg")
        knee_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}LowerLeg")
        ankle_id = getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}FootAnkle")

        hip_pos = bone_positions.get(hip_id)
        knee_pos = bone_positions.get(knee_id)
        ankle_pos = bone_positions.get(ankle_id)

        if hip_pos is None or knee_pos is None or ankle_pos is None:
            lower_body_coordinates[side] = None
            continue

        # Transform to lower body-centric coordinates
        H_lower_body = transform_to_lower_body_frame(hip_pos)
        K_lower_body = transform_to_lower_body_frame(knee_pos)
        A_lower_body = transform_to_lower_body_frame(ankle_pos)

        # Get Foot Rotation from bone poses
        foot_ball_wd = bone_positions.get(getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}FootBall"))
        foot_trans_wd = bone_positions.get(getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}FootTransverse"))
        foot_sub_wd = bone_positions.get(getattr(FullBodyBoneId, f"FullBody_{side_key_pascal}FootSubtalar"))

        if foot_sub_wd is None:
            foot_heel_wd = ankle_pos
        else:
            foot_heel_wd = foot_sub_wd

        if foot_ball_wd is None or foot_heel_wd is None:
            R_body_foot = None
        else:
            foot_ball_pos = transform_to_lower_body_frame(foot_ball_wd)
            foot_heel_pos = transform_to_lower_body_frame(foot_heel_wd)

            foot_x_axis = foot_ball_pos - foot_heel_pos
            foot_x_axis /= np.linalg.norm(foot_x_axis) + 1e-8

            if foot_trans_wd is not None:
                foot_trans_pos = transform_to_lower_body_frame(foot_trans_wd)
                trans_heel_vec = foot_trans_pos - foot_heel_pos
                foot_y_axis = np.cross(trans_heel_vec, foot_x_axis)
            else:
                if foot_sub_wd is not None:  # ankle and sub are two seperate joint to be used to compute pose
                    foot_SA = A_lower_body - foot_heel_pos
                    foot_y_axis = np.cross(foot_SA, foot_x_axis)
                else:  # use lower leg direction as proxy
                    lower_leg_vec = K_lower_body - A_lower_body
                    foot_y_axis = np.cross(lower_leg_vec, foot_x_axis)

            foot_y_axis /= np.linalg.norm(foot_y_axis) + 1e-8
            foot_z_axis = np.cross(foot_x_axis, foot_y_axis)
            foot_z_axis /= np.linalg.norm(foot_z_axis) + 1e-8

            rotation = np.column_stack([foot_x_axis, foot_y_axis, foot_z_axis])
            U_rot, _, V_rot_T = np.linalg.svd(rotation)
            R_body_foot = U_rot @ V_rot_T

        lower_body_coordinates[side] = {
            "H": H_lower_body,
            "K": K_lower_body,
            "A": A_lower_body,
            "ankle_rot": R_body_foot,
            "A_world": foot_heel_wd,
            "hip_center_world": hip_center,
        }

    lower_body_coordinates['R_world_lower_body'] = R_world_lower_body

    return lower_body_coordinates


def _compute_extruded_tip_position(tip_pos: np.ndarray, tip_rot_q: np.ndarray, side: str) -> np.ndarray:
    """
    Computes real tip position extruded out by 0.01m in +y for left hand and -y for right hand
    from the local tip bone rotation.
    """
    sign = 1.0 if side.lower() == "left" else -1.0
    extrude_vec = np.array([0.0, sign * 0.01, 0.0])
    return tip_pos + q2R(tip_rot_q) @ extrude_vec

def _get_finger_coordinates(bones: list[Bone], coord_type: str, bones_to_return: list[int]) -> dict:
    """
    Get finger bone coordinates in either absolute, body-centric, or hand-centric frame.

    Args:
        bones (list[Bone]): A list of bone objects from the XR device.
        coord_type (str): The coordinate system to use ("absolute", "body_centric", or "hand_centric").
        bones_to_return (list[int]): A list of FullBodyBoneId enums for which to return coordinates.

    Returns:
        dict: A dictionary mapping bone IDs to their coordinates.
    """
    bone_positions = {b.id: np.array(b.position) for b in bones}
    bone_rotations = {b.id: np.array(b.rotation) for b in bones}
    
    # Non-vmd dataset check (has ThumbDistal)
    is_non_vmd = bone_positions.get(FullBodyBoneId.FullBody_LeftHandThumbDistal) is not None or \
                 bone_positions.get(getattr(FullBodyBoneId, "FullBody_LeftHandThumbDistal", None).value if getattr(FullBodyBoneId, "FullBody_LeftHandThumbDistal", None) else None) is not None
                 
    if is_non_vmd:
        for b in bones:
            bone_name = getattr(FullBodyBoneId(b.id), 'name', str(b.id))
            if "Hand" in bone_name and "Tip" in bone_name:
                side = "left" if "Left" in bone_name else "right"
                if b.id in bone_positions and b.id in bone_rotations:
                    original_tip = bone_positions[b.id]
                    computed_tip = _compute_extruded_tip_position(original_tip, bone_rotations[b.id], side)

                    if "Thumb" in bone_name:
                        side_pascal = side.capitalize()
                        distal_enum = getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbDistal", None)
                        if distal_enum is not None:
                            if distal_enum in bone_positions:
                                bone_positions[distal_enum] = original_tip
                            elif hasattr(distal_enum, 'value') and distal_enum.value in bone_positions:
                                bone_positions[distal_enum.value] = original_tip
                    
                    bone_positions[b.id] = computed_tip

    returned_coords = {}

    if coord_type == "absolute":
        for bone_id in bones_to_return:
            if bone_id in bone_positions:
                returned_coords[bone_id] = bone_positions[bone_id]
            else:
                returned_coords[bone_id] = None
        return returned_coords

    elif coord_type == "body_centric":
        shoulder_center, R_world_body = _get_body_frame(bone_positions)

        if shoulder_center is None or R_world_body is None:
            # Cannot compute body frame, return None for all requested bones
            for bone_id in bones_to_return:
                returned_coords[bone_id] = None
            return returned_coords

        def transform_to_body_frame(world_pos):
            """Transform a world position to body-centric coordinates."""
            translated = world_pos - shoulder_center
            body_pos = R_world_body.T @ translated
            return body_pos

        for bone_id in bones_to_return:
            if bone_id in bone_positions:
                world_pos = bone_positions[bone_id]
                returned_coords[bone_id] = transform_to_body_frame(world_pos)
            else:
                returned_coords[bone_id] = None
        return returned_coords

    elif coord_type == "hand_centric":
        # First transform to body-centric coordinates (like other coordinate types)
        shoulder_center, R_world_body = _get_body_frame(bone_positions)

        if shoulder_center is None or R_world_body is None:
            # Cannot compute body frame, return None for all requested bones
            for bone_id in bones_to_return:
                returned_coords[bone_id] = None
            return returned_coords

        def transform_to_body_frame(world_pos):
            """Transform a world position to body-centric coordinates."""
            translated = world_pos - shoulder_center
            body_pos = R_world_body.T @ translated
            return body_pos

        # Get SEW coordinates which contain wrist rotation information in body frame
        sew_coords = _get_body_centric_coordinates(bones)

        if sew_coords is None or sew_coords["left"] is None or sew_coords["right"] is None:
            # Cannot compute hand frame, return None for all requested bones
            for bone_id in bones_to_return:
                returned_coords[bone_id] = None
            return returned_coords

        # Get wrist positions in body frame (from SEW coordinates)
        left_wrist_pos_body = sew_coords["left"]["W"]  # W is wrist position in body frame
        right_wrist_pos_body = sew_coords["right"]["W"]

        # Extract wrist rotation matrices (reshape from flattened form) - already in body frame
        left_wrist_rot = sew_coords["left"]["wrist_rot"].reshape(3, 3) if sew_coords["left"]["wrist_rot"] is not None else None
        right_wrist_rot = sew_coords["right"]["wrist_rot"].reshape(3, 3) if sew_coords["right"]["wrist_rot"] is not None else None

        for bone_id in bones_to_return:
            bone_name = FullBodyBoneId(bone_id).name
            world_pos = bone_positions.get(bone_id)

            if world_pos is None:
                returned_coords[bone_id] = None
                continue

            # First transform world position to body frame
            body_pos = transform_to_body_frame(world_pos)

            if "LeftHand" in bone_name and left_wrist_pos_body is not None and left_wrist_rot is not None:
                # Transform from body-centric to hand-centric coordinates
                translated = body_pos - left_wrist_pos_body
                returned_coords[bone_id] = left_wrist_rot.T @ translated
            elif "RightHand" in bone_name and right_wrist_pos_body is not None and right_wrist_rot is not None:
                # Transform from body-centric to hand-centric coordinates
                translated = body_pos - right_wrist_pos_body
                returned_coords[bone_id] = right_wrist_rot.T @ translated
            else:
                returned_coords[bone_id] = None

        return returned_coords

    else:
        raise ValueError("coord_type must be 'absolute', 'body_centric', or 'hand_centric'")


def _calc_gripper_state(bone_positions: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Heuristically determine gripper state based on thumb and index finger distance.
    """
    # Gripper state based on thumb and index finger distance
    left_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandThumbTip)
    left_index_tip = bone_positions.get(FullBodyBoneId.FullBody_LeftHandIndexTip)
    right_thumb_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandThumbTip)
    right_index_tip = bone_positions.get(FullBodyBoneId.FullBody_RightHandIndexTip)

    left_gripper_dist = (
        np.linalg.norm(left_thumb_tip - left_index_tip)
        if left_thumb_tip is not None and left_index_tip is not None
        else 0.1
    )
    right_gripper_dist = (
        np.linalg.norm(right_thumb_tip - right_index_tip)
        if right_thumb_tip is not None and right_index_tip is not None
        else 0.1
    )

    left_gripper_action = np.array([1]) if left_gripper_dist > 0.05 else np.array([-1])
    right_gripper_action = (
        np.array([1]) if right_gripper_dist > 0.05 else np.array([-1])
    )
    return left_gripper_action, right_gripper_action, left_gripper_dist, right_gripper_dist


def _get_hand_frame(
    side: str, bone_positions: dict
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Calculates the hand-centric coordinate frame for a given side.

    Args:
        side (str): "left" or "right".
        bone_positions (dict): A dictionary mapping bone IDs to their positions.

    Returns:
        A tuple containing:
        - np.ndarray: The origin of the hand frame (wrist position).
        - np.ndarray: The rotation matrix from world to hand frame.
        Returns (None, None) if essential bones are missing.
    """
    # print(f"{bone_positions=}")
    side_pascal = side.capitalize()
    wrist_pos = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandWrist")
    )

    # index_mcp = bone_positions.get(
    #     getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexMetacarpal")
    # )
    # middle_mcp = bone_positions.get(
    #     getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleMetacarpal")
    # )
    # ring_mcp = bone_positions.get(
    #     getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingMetacarpal")
    # )
    # little_mcp = bone_positions.get(
    #     getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleMetacarpal")
    # )
    index_mcp = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexProximal")
    )
    middle_mcp = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleProximal")
    )
    ring_mcp = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingProximal")
    )
    little_mcp = bone_positions.get(
        getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleProximal")
    )

    # X-axis: from wrist towards palm center (average of finger MCPs)
    if middle_mcp is not None and ring_mcp is not None:
        palm_center = (index_mcp + middle_mcp + ring_mcp + little_mcp) / 4
    elif middle_mcp is not None:
        palm_center = (index_mcp + middle_mcp + little_mcp) / 3
    else:
        palm_center = (index_mcp + little_mcp) / 2

    x_axis = (palm_center - wrist_pos)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Collect all available MCP points relative to wrist
    mcp_points = []
    if index_mcp is not None:
        mcp_points.append(index_mcp - wrist_pos)
    if middle_mcp is not None:
        mcp_points.append(middle_mcp - wrist_pos)
    if ring_mcp is not None:
        mcp_points.append(ring_mcp - wrist_pos)
    if little_mcp is not None:
        mcp_points.append(little_mcp - wrist_pos)

    if len(mcp_points) < 2:
        return None  # Need at least 2 MCP points for reliable computation

    # Use least squares to find Y-axis that's perpendicular to the palm plane
    # Stack MCP vectors into matrix A
    A = np.array(mcp_points)  # shape: (n_points, 3)

    # Find the normal to the plane containing MCP points using SVD
    # The normal is the last column of V (smallest singular value)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    palm_normal = Vt[-1, :]  # Last row of V^T = last column of V
    palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)

    # Determine handedness-consistent Y-axis direction
    # Use the palm normal, but ensure correct orientation based on hand side
    wrist_to_index = index_mcp - wrist_pos
    wrist_to_pinky = little_mcp - wrist_pos

    # Cross product gives a reference direction
    reference_y = np.cross(wrist_to_index, wrist_to_pinky)
    reference_y = reference_y / (np.linalg.norm(reference_y) + 1e-8)

    # Choose palm normal direction that aligns with handedness
    if np.dot(palm_normal, reference_y) < 0:
        palm_normal = -palm_normal

    y_axis = palm_normal

        # Z-axis: Z = X × Y (ensuring right-handed coordinate system)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # Create rotation matrix
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    U_rot, _, V_rot_T = np.linalg.svd(rotation_matrix)
    R_world_hand = U_rot @ V_rot_T

    return R_world_hand


def _convert_bones_to_finger_dict(finger_coords: dict, which_side: str, is_vmd=False) -> dict:
    """
    Convert bone coordinate dictionary to finger position format expected by hand controller.
    If no valid finger data is available, returns empty dict (hand controller will use zeros).

    Args:
        finger_coords: Dictionary mapping bone IDs to 3D coordinates
        which_side: "left" or "right"

    Returns:
        Dictionary in format expected by compute_finger_joint_angles_from_mediapipe:
        {
            'thumb': {'thumb_mcp': pos, 'thumb_pip': pos},
            'index': {'index_finger_mcp': pos, 'index_finger_pip': pos},
            'middle': {'middle_finger_mcp': pos, 'middle_finger_pip': pos},
            'ring': {'ring_finger_mcp': pos, 'ring_finger_pip': pos},
            'pinky': {'pinky_mcp': pos, 'pinky_pip': pos}
        }
        Or empty dict if no valid data (controller will use zero joint angles)
    """
    # Check if we have any valid finger coordinate data
    has_any_valid_data = False
    if finger_coords:
        for bone_id, coord in finger_coords.items():
            if coord is not None and np.isfinite(coord).all():
                has_any_valid_data = True
                break

    # If no valid finger data, return empty dict - hand controller will use zeros
    if not has_any_valid_data:
        return {}

    side_pascal = which_side.capitalize()
    finger_dict = {}
    # Define the mapping from bone IDs to finger positions
    if is_vmd:
        # it's probably vmd data set use the vmd mapping
        finger_mapping = {
            'thumb': {
                'thumb_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbMetacarpal", None),
                'thumb_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbProximal", None),
                'thumb_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbTip", None)
                },
            'index': {
                'index_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexIntermediate", None),
                'index_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexTip", None),
                'index_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexTip", None)
            },
            'middle': {
                'middle_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleIntermediate", None),
                'middle_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleTip", None),
                'middle_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleTip", None)
            },
            'ring': {
                'ring_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingIntermediate", None),
                'ring_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingTip", None),
                'ring_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingTip", None)
            },
            'pinky': {
                'pinky_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleIntermediate", None),
                'pinky_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleTip", None),
                'pinky_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleTip", None)
            }
        }
    else:
        finger_mapping = {
        'thumb': {
            # 'thumb_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbMetacarpal", None),  # changed from Proximal to Metacarpal for testing
            'thumb_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbProximal", None),
            'thumb_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbDistal", None),
            'thumb_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandThumbTip", None)
        },
        'index': {
            'index_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexProximal", None),
            'index_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexIntermediate", None),
            'index_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandIndexTip", None)
        },
        'middle': {
            'middle_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleProximal", None),
            'middle_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleIntermediate", None),
            'middle_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandMiddleTip", None)
        },
        'ring': {
            'ring_finger_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingProximal", None),
            'ring_finger_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingIntermediate", None),
            'ring_finger_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandRingTip", None)
        },
        'pinky': {
            'pinky_mcp': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleProximal", None),
            'pinky_pip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleIntermediate", None),
            'pinky_tip': getattr(FullBodyBoneId, f"FullBody_{side_pascal}HandLittleTip", None)
        }
    }



    for finger_name, positions in finger_mapping.items():
        finger_positions = {}
        has_valid_finger_data = False

        for position_name, bone_id in positions.items():
            if bone_id is not None and bone_id in finger_coords:
                coord = finger_coords[bone_id]
                # Check for NaN or Inf values
                if coord is not None and np.isfinite(coord).all():
                    finger_positions[position_name] = np.array(coord)
                    has_valid_finger_data = True

        # Only add finger if we have valid data for it
        if has_valid_finger_data:
            # Fill in missing positions with zeros for this finger
            for position_name, bone_id in positions.items():
                if position_name not in finger_positions:
                    finger_positions[position_name] = np.array([0.0, 0.0, 0.0])
            finger_dict[finger_name] = finger_positions

    return finger_dict


def _compute_finger_tip_centroids(finger_dict: dict) -> np.ndarray | None:
    """
    Compute the centroid (mean position) of the five finger tips from the finger positions.

    Args:
        finger_dict (dict): Dictionary with keys 'thumb', 'index', 'middle', 'ring', 'pinky',
                           each containing joint positions (e.g., 'thumb_tip', 'index_finger_tip', etc.).

    Returns:
        np.ndarray: Mean position (3,) of the five finger tips in hand frame, or None if any tip is missing.
    """
    if not finger_dict:
        return None

    tip_keys = {
        # 'thumb': 'thumb_tip',
        'index': 'index_finger_tip',
        'middle': 'middle_finger_tip',
        'ring': 'ring_finger_tip',
        'pinky': 'pinky_tip'
    }

    tips = []
    for finger, tip_key in tip_keys.items():
        finger_data = finger_dict.get(finger, {})
        tip_pos = finger_data.get(tip_key)
        if tip_pos is None:
            return None  # If any tip is missing, return None
        tips.append(tip_pos)

    tips = np.array(tips)
    return np.mean(tips, axis=0)


def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.

    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [qv;q0] or [x, y, z, w]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix
    """

    I = np.identity(3)
    qhat = hat(q[0:3])
    qhat2 = qhat.dot(qhat)
    return I + 2 * q[-1] * qhat + 2 * qhat2


def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] = k[1]
    khat[1, 0] = k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] = k[0]
    return khat


# --- Robosuite and WebRTC Integration ---


class RobosuiteTeleopState:
    """
    Thread-safe state to share data from the WebRTC server thread
    to the main robosuite simulation thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.pose_action = None
        self.tags = {}  # Store latest tags by ID
        self.is_connected = False
        self._pose_seq: int = 0        # increments each time a new pose arrives
        self._pose_recv_t: float = 0.0 # monotonic timestamp of latest pose
        self._packet_arrival_t: float = 0.0  # time.monotonic() when packet first hit on_body_pose_message

    def update_pose(self, action_dict):
        with self._lock:
            # We want to merge the latest tag data into the action dict before publishing it
            if action_dict is not None:
                action_dict["tags"] = self.tags.copy()
            self.pose_action = action_dict
            self._pose_seq += 1
            self._pose_recv_t = time.monotonic()

    def update_tags(self, tag_dict):
        with self._lock:
            self.tags = tag_dict

    def get_pose(self):
        with self._lock:
            return self.pose_action.copy() if self.pose_action else None

    def get_pose_seq(self) -> tuple:
        """Return (seq, recv_monotonic_time) for pose-arrival rate tracking."""
        with self._lock:
            return self._pose_seq, self._pose_recv_t

    def get_arrival_t(self) -> float:
        """Return the monotonic time when the most recent packet first arrived."""
        return self._packet_arrival_t


class StateFactory:
    """A factory to create and hold a reference to the state object."""

    def __init__(self):
        self.instance = None

    def __call__(self):
        # Called by WebRTCServer to create a state for a new peer.
        # We store the instance so the main thread can access it.
        if self.instance is None:
            self.instance = RobosuiteTeleopState()
        return self.instance


class _BoneProcessor:
    """Single persistent background thread that always processes the *newest* frame.

    Frames that arrive while the previous one is still processing are silently
    replaced so the consumer always sees the freshest pose.  This avoids:

    * asyncio.get_running_loop() RuntimeError when called from non-async callbacks
    * Thread-pool-executor queue backlog (the root cause of multi-second latency)
    * Python GIL contention from spawning new threads per frame
    """

    def __init__(self, device: "XRRTCBodyPoseDevice"):
        self._device = device
        self._lock = threading.Lock()
        self._pending: tuple | None = None   # (bones, state) or None
        self._wakeup = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="bone-proc")
        self._thread.start()

    def put(self, message: bytes, state: "RobosuiteTeleopState") -> None:
        """Drop newest RAW frame in; callable from any thread. Deserialization happens on
        THIS worker thread (not the caller / event loop) so the loop stays free to drain
        inbound frames and only the freshest raw frame is ever parsed."""
        with self._lock:
            self._pending = (message, state)
        self._wakeup.set()  # wake processor immediately

    def _run(self) -> None:
        while True:
            self._wakeup.wait()
            self._wakeup.clear()
            # Drain: keep processing until no more pending frames
            while True:
                with self._lock:
                    item = self._pending
                    self._pending = None
                if item is None:
                    break
                raw, state = item
                try:
                    _bp_t0 = time.monotonic()
                    bones = deserialize_pose_data(raw)      # OFF the event loop now
                    if bones:
                        self._device._process_body_pose(bones, state)
                    _bp_ms = (time.monotonic() - _bp_t0) * 1000
                    if _bp_ms > 100:
                        print(f"[BoneProcessor] SLOW: parse+process took {_bp_ms:.0f} ms")
                except Exception as exc:
                    import traceback
                    print(f"[BoneProcessor] error: {exc}")
                    traceback.print_exc()


class XRRTCBodyPoseDevice(Device):
    """
    A device to control a robot using body pose data from XR Robot Teleop Client.
    """

    def __init__(self, env, process_bones_to_action_fn=None, record_data=False, output_dir="./recordings", ndigits=3, server_data_channels=None, **kwargs):
        if env is not None:
            super().__init__(env)
        else:
            self.env = None
            self.all_robot_arms = []
            self.all_robot_grippers = []
            self.num_robot_arms = 0
            self.num_robot_grippers = 0
        self.state_factory = StateFactory()

        if process_bones_to_action_fn is None:
            self.process_bones_to_action_fn = self._default_process_bones_to_action
        else:
            self.process_bones_to_action_fn = process_bones_to_action_fn

        # Pre-check if function handles tags to avoid inspecting per frame
        import inspect
        sig = inspect.signature(self.process_bones_to_action_fn)
        self._fn_accepts_tags = 'tags' in sig.parameters

        # CSV recording setup — attributes ALWAYS initialized: the study manager
        # constructs this device with record_data=False and toggles it True at each
        # take. Gating these on the constructor flag left ndigits/csv_* undefined, so
        # the FIRST recorded take crashed on every frame (AttributeError: ndigits),
        # spammed tracebacks at frame rate, and wrote a header-only body-pose CSV.
        self.record_data = record_data
        self.output_dir = output_dir
        self.csv_file = None
        self.csv_writer = None
        self.started = False
        self.start_time = None
        self.ndigits = ndigits
        # --- CSV recording runs on a DEDICATED writer thread (mirrors SessionRecorder) ---
        # The per-frame row formatting + flush used to run inline on the bone-processor
        # thread (see _process_body_pose). That hot thread's GIL time is exactly what the
        # WebRTC asyncio loop must contend for to drain the reliable/ordered body_pose SCTP
        # channel; the ~80 writerow() calls + per-frame flush added the moment recording
        # started were enough to starve the loop, so inbound frames queued and the delivered
        # pose got progressively staler — the multi-second, growing teleop lag that appeared
        # right after Start Session. Now the hot path only stamps a timestamp and hands off
        # references; ALL formatting/IO happens on this sink thread, off the teleop path.
        self._csv_q: "_queue.Queue" = _queue.Queue(maxsize=6000)  # ~1 min @ 90 Hz
        self._csv_thread: "threading.Thread | None" = None
        self._csv_stop = threading.Event()
        self._csv_drops = 0
        self._csv_last_drop_log = 0.0
        # Optional sink called with each freshly-computed action_dict (post tag-merge).
        # Used by the out-of-process host (xr_webrtc_proxy) to ship the pose to the parent
        # manager over a socket; None in the normal in-process path. Set externally.
        self.pose_sink = None
        if self.record_data:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

        datachannel_handlers = {
            "body_pose": self.on_body_pose_message,
            "apriltag_pose": self.on_apriltag_message,
            # Manager->XR control channels are CLIENT-created by the Quest now (see the manager's
            # server_data_channels note). They are send-only from the manager's perspective, so the
            # inbound handlers are no-ops — they exist only so the channel registers without a
            # "No handler registered" warning; the server stores the channel for send_to_datachannel
            # regardless. Do NOT route unity_cmds to on_unity_state_message: that is the SEPARATE
            # XR->manager reverse path and would misparse outbound command echoes.
            "haptics": lambda msg, state: None,
            "motor_stats": lambda msg, state: None,
            "unity_cmds": lambda msg, state: None,
            # Reverse UI-state channel: Unity reports in-VR toggle changes here so the
            # study manager can mirror them onto its buttons (client-created channel,
            # same dispatch path as body_pose / apriltag_pose).
            "unity_state": self.on_unity_state_message,
        }

        self._haptics_queue: _queue.SimpleQueue = _queue.SimpleQueue()
        self._motor_stats_queue: _queue.SimpleQueue = _queue.SimpleQueue()
        self._unity_cmds_queue: _queue.SimpleQueue = _queue.SimpleQueue()
        # Inbound toggle-state reports from Unity (Unity → manager). Drained by the
        # study manager's status tick via poll_unity_state().
        self._unity_state_queue: _queue.SimpleQueue = _queue.SimpleQueue()

        # ── Latency telemetry ────────────────────────────────────────────────
        # Clock-sync round-trip on the teleop connection (manager↔Quest) + the Quest's reported
        # video receive-latency breakdown. Both are read by the study manager to assemble the
        # end-to-end video (glass-to-glass) and teleop (motion-to-motion) latency budgets.
        self._lat_lock = threading.Lock()
        self._net_rtt_ms = None        # EMA of the round-trip time (ms) on the teleop datachannel
        self._video_latency = None     # latest {"jitter_ms","decode_ms","net_ms"} from the Quest
        self._lat_ping_id = 0

        # Inbound body_pose delivery-health tracking = the packet-loss / HOL-blocking readout.
        # On the RELIABLE/ordered body_pose channel a lost WiFi packet STALLS delivery until
        # the retransmit, then a burst of queued frames arrives — so a big inter-arrival GAP
        # is the signature of loss. link_health() summarises these for the latency panel; the
        # stall rate should fall to ~0 once body_pose is switched to unreliable on the Quest.
        self._lh_lock = threading.Lock()
        self._lh_prev_t = 0.0          # monotonic time of the previous frame's arrival
        self._lh_frames = 0            # frames since the window opened
        self._lh_stalls = 0            # frames that arrived after a >STALL gap
        self._lh_max_gap_ms = 0.0      # worst inter-arrival gap in the window
        self._lh_window_t0 = 0.0       # window start (monotonic)

        # Drop-newest bone processing: a single dedicated thread always holds the
        # newest frame and discards intermediate ones.  This avoids any asyncio
        # dependency in the hot path (get_running_loop() can raise RuntimeError
        # when called from non-async WebRTC callbacks) and eliminates executor
        # queue backlog that caused multi-second latency.
        self._bone_processor = _BoneProcessor(self)

        # Normalise server_data_channels to dict and configure the haptics
        # channel as unreliable (ordered=False, maxRetransmits=0). Reliable/
        # ordered delivery causes head-of-line blocking on WiFi packet loss,
        # producing 0.5-2 s haptic latency.
        # motor_stats uses the default reliable/ordered channel (10 Hz thermal data).
        if isinstance(server_data_channels, dict):
            _ch = dict(server_data_channels)
        else:
            _ch = {label: {} for label in (server_data_channels or [])}
        if "haptics" in _ch:
            _ch["haptics"] = {"ordered": False, "maxRetransmits": 0}

        configure_logging(level="INFO")  # set xr_robot_teleop's verbosity
        self.server = WebRTCServer(
            datachannel_handlers=datachannel_handlers,
            state_factory=self.state_factory,
            video_track_factory=None,
            server_data_channels=_ch,
        )

        # Register an asyncio lifespan task that drains _haptics_queue from
        # within the server's event loop (thread-safe bridge from the main thread).
        @asynccontextmanager
        async def _haptics_lifespan(app):
            task = asyncio.create_task(self._haptics_sender_loop())
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        self.server.add_lifespan_context(_haptics_lifespan)

        @asynccontextmanager
        async def _motor_stats_lifespan(app):
            task = asyncio.create_task(self._motor_stats_sender_loop())
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        self.server.add_lifespan_context(_motor_stats_lifespan)

        @asynccontextmanager
        async def _unity_cmds_lifespan(app):
            task = asyncio.create_task(self._unity_cmds_sender_loop())
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        self.server.add_lifespan_context(_unity_cmds_lifespan)

        def connection_monitor():
            was_connected = False
            while True:
                time.sleep(0.1)
                try:
                    is_conn = self.is_connected
                    if was_connected and not is_conn:
                        self.on_connection_state_change("disconnected")
                    was_connected = is_conn
                except Exception:
                    pass

        self.monitor_thread = threading.Thread(target=connection_monitor, daemon=True)
        self.monitor_thread.start()

        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()
        print("=" * 80)
        print("Integrated Robosuite WebRTC Teleoperation Server")
        print("The WebRTC server is running in the background.")
        print("Connect your VR client to this machine on port 8080.")
        print("=" * 80)

    @property
    def is_connected(self):
        """
        Returns true if the WebRTC client is connected.

        state.is_connected latches True on the first body_pose packet and is never reset, so it
        alone can NEVER go False again — which left connection_monitor's disconnect edge dead
        (mid-take CSVs were never finalized on a drop), the GUI "WebRTC: Connected" label lying
        after disconnects, and the reconnect state-push unable to re-arm. AND it with the live
        peer set: the server prunes pcs on failed/closed/disconnected, so this now truly drops
        when the client goes away (clean closes promptly; unclean drops at ICE-consent expiry).
        """
        return (
            self.state_factory.instance is not None
            and self.state_factory.instance.is_connected
            and len(self.server.pcs) > 0
        )

    @staticmethod
    def _default_process_bones_to_action(bones: list[Bone], tags: dict = None) -> dict:
        """
        Custom function to process bones into actions for dual Kinova3 robot.
        """
        action_dict = {}

        if tags is not None:
            action_dict["tags"] = tags

        # Get bone positions and rotations
        bone_positions = {b.id: np.array(b.position) for b in bones}
        bone_rotations = {b.id: np.array(b.rotation) for b in bones}

        # Hand Control
        left_gripper_action, right_gripper_action, left_gripper_dist, right_gripper_dist = _calc_gripper_state(bone_positions)


        finger_coords = _get_finger_coordinates(
            bones=bones,
            coord_type="hand_centric",
            bones_to_return=simplified_finger_bones,
        )

        # Arm Control (absolute SEW)
        sew_coords = _get_body_centric_coordinates(bones)

        if sew_coords is None or sew_coords["left"] is None or sew_coords["right"] is None:
            print("Warning: Could not calculate SEW coordinates. Skipping action.")
            return None

        # Head rotation for camera (relative to the upper body frame).
        R_world_upper_body = sew_coords['R_world_body']
        head_rot_q = bone_rotations.get(FullBodyBoneId.FullBody_Head)
        R_world_head = q2R(head_rot_q) @ Rotation.from_euler("ZYX",
                                                             [-np.pi / 2, -np.pi / 2, 0]
                                                             ).as_matrix() if head_rot_q is not None else np.eye(3)
        action_dict["head_rotation"] = R_world_upper_body.T @ R_world_head
        # Get wrist rotations
        left_rot_matrix = sew_coords["left"]["wrist_rot"]
        right_rot_matrix = sew_coords["right"]["wrist_rot"]

        if left_rot_matrix is None or right_rot_matrix is None:
            print("Warning: Could not get wrist rotation. Skipping action.")
            return None

        # Assemble final action
        left_sew_pos = np.concatenate(
            [sew_coords["left"]["S"], sew_coords["left"]["E"], sew_coords["left"]["W"]]
        )
        right_sew_pos = np.concatenate(
            [sew_coords["right"]["S"], sew_coords["right"]["E"], sew_coords["right"]["W"]]
        )

        left_sew = np.concatenate([left_sew_pos, left_rot_matrix])
        right_sew = np.concatenate([right_sew_pos, right_rot_matrix])

        # lower body coordinates
        hka_coords = _get_lower_body_centric_coordinates(bones)

        # compute torso rotation
        R_world_lower_body = hka_coords['R_world_lower_body']
        R_lower_upper = R_world_lower_body.T @ R_world_upper_body

        # Body center for mocap control
        # hip_center_world = hka_coords["left"]["hip_center_world"]
        # body_center_in_world = hip_center_world

        body_center_in_world = bone_positions.get(FullBodyBoneId.FullBody_SpineLower)

        # check if this is vmd data set
        # vmd dataset doesn't have Scapula bone
        left_scapula = bone_positions.get(FullBodyBoneId.FullBody_LeftScapula)
        is_vmd = left_scapula is None

        # compute ankles / body center vector for robot mocap orientation correction
        if hka_coords.get("left") is not None and hka_coords.get("right") is not None and body_center_in_world is not None:
            left_ankle = hka_coords["left"]["A_world"]
            right_ankle = hka_coords["right"]["A_world"]
            ankle_center = (left_ankle + right_ankle) / 2
            action_dict["ankle_to_body"] = body_center_in_world - ankle_center
        else:
            action_dict["ankle_to_body"] = None

        action_dict["left_sew"] = left_sew
        action_dict["right_sew"] = right_sew
        action_dict["left_gripper"] = left_gripper_action
        action_dict["right_gripper"] = right_gripper_action
        action_dict["left_gripper_val"] = left_gripper_dist
        action_dict["right_gripper_val"] = right_gripper_dist
        action_dict["right_fingers"] = _convert_bones_to_finger_dict(finger_coords, "right", is_vmd=is_vmd)
        action_dict["left_fingers"] = _convert_bones_to_finger_dict(finger_coords, "left", is_vmd=is_vmd)
        action_dict["left_finger_tip_centroid"] = _compute_finger_tip_centroids(action_dict["left_fingers"])
        action_dict["right_finger_tip_centroid"] = _compute_finger_tip_centroids(action_dict["right_fingers"])
        action_dict["left_finger_mcp_centroid"] = action_dict["left_fingers"].get('index', {}).get('index_finger_mcp', None)
        action_dict["right_finger_mcp_centroid"] = action_dict["right_fingers"].get('index', {}).get('index_finger_mcp', None)

        # Add lower body HKA coordinates to action_dict
        action_dict["left_hka"] = hka_coords.get("left")
        action_dict["right_hka"] = hka_coords.get("right")
        action_dict["body_center"] = body_center_in_world
        action_dict["R_world_upper_body"] = R_world_upper_body
        action_dict["R_lower_upper"] = R_lower_upper
        action_dict["R_world_upper_body"] = R_world_upper_body
        action_dict["p_world_upper_body"] = sew_coords["shoulder_center"]

        return action_dict

    # Header shared by the hot path (take open) and the writer thread.
    _CSV_FIELDNAMES = [
        "time_elapsed",
        "data_type",  # "bone" or "action"
        "id",         # bone_id for bones, action_key for actions
        "pos_x", "pos_y", "pos_z",
        "rot_x", "rot_y", "rot_z", "rot_w",
    ]

    def _record_to_csv(self, bones: list[Bone], action_dict: dict):
        """HOT PATH (bone-processor thread): stamp the frame and hand it to the CSV
        writer thread. Deliberately does NO row formatting, NO writerow, and NO flush —
        those steal GIL time from the WebRTC event loop and were the cause of the growing
        teleop lag that appeared the instant recording started. All of that now happens on
        _csv_writer_run(). ``bones`` and ``action_dict`` are freshly built every frame
        (deserialize_pose_data / process_bones_to_action_fn), so handing off the references
        is safe — the sink thread only ever reads them."""
        if not self.started:
            self.started = True
            os.makedirs(self.output_dir, exist_ok=True)   # output_dir is set per take
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = os.path.join(self.output_dir, f"body_pose_data_{timestamp_str}.csv")
            self.start_time = time.time()
            start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))
            # Sibling start-time file for cross-sensor alignment (written once, cheaply).
            try:
                start_time_file = os.path.join(os.path.dirname(self.output_file), "csv_start_time.txt")
                with open(start_time_file, "w") as f:
                    f.write(f"Start time (epoch): {self.start_time}\n")
                    f.write(f"Start time (datetime): {start_datetime}\n")
            except Exception as exc:                       # noqa: BLE001
                print(f"[csv] start-time file error: {exc}")
            # Fresh queue + writer thread for this take (the previous take's thread was
            # already joined by cleanup_recording()).
            self._csv_stop.clear()
            self._csv_q = _queue.Queue(maxsize=6000)
            self._csv_drops = 0
            self._csv_thread = threading.Thread(
                target=self._csv_writer_run, args=(self.output_file,),
                daemon=True, name="body-csv")
            self._csv_thread.start()
            print(f"Started recording data to {self.output_file}")

        # Non-blocking hand-off. If the sink can't keep up (e.g. a disk stall) we DROP
        # rather than block the bone-processor thread — a stalled hot path would lag the
        # live teleop, which matters far more than a few missing logged frames.
        try:
            self._csv_q.put_nowait((time.time() - self.start_time, bones, action_dict))
        except _queue.Full:
            self._csv_drops += 1
            now = time.monotonic()
            if now - self._csv_last_drop_log > 5.0:
                self._csv_last_drop_log = now
                print(f"[csv] writer backlog — dropped {self._csv_drops} frames so far "
                      f"(disk too slow?)")

    def _csv_writer_run(self, output_file: str):
        """CSV writer SINK thread. Owns the file handle exclusively; formats rows and
        writes them with a THROTTLED flush (≈2 Hz) instead of once per frame. Drains the
        queue until _csv_stop is set and the queue is empty, so a Stop never loses buffered
        frames."""
        try:
            f = open(output_file, "w", newline="")
        except Exception as exc:                           # noqa: BLE001
            print(f"[csv] open failed for {output_file}: {exc}")
            return
        writer = csv.writer(f)
        writer.writerow(self._CSV_FIELDNAMES)
        nd = self.ndigits
        last_flush = time.monotonic()
        try:
            while True:
                try:
                    item = self._csv_q.get(timeout=0.2)
                except _queue.Empty:
                    if self._csv_stop.is_set():
                        break
                    if time.monotonic() - last_flush > 0.5:
                        f.flush()
                        last_flush = time.monotonic()
                    continue
                time_elapsed, bones, action_dict = item
                te = round(time_elapsed, nd)
                rows = []
                for bone in bones:
                    p = bone.position
                    r = bone.rotation
                    rows.append((
                        te, "bone", bone.id,
                        round(float(p[0]), nd), round(float(p[1]), nd), round(float(p[2]), nd),
                        round(float(r[0]), nd), round(float(r[1]), nd),
                        round(float(r[2]), nd), round(float(r[3]), nd),
                    ))
                if action_dict:
                    for action_key, action_value in action_dict.items():
                        if isinstance(action_value, np.ndarray):
                            flat_value = action_value.flatten()
                            if len(flat_value) >= 3:
                                pos_x = round(float(flat_value[0]), nd)
                                pos_y = round(float(flat_value[1]), nd)
                                pos_z = round(float(flat_value[2]), nd)
                            else:
                                pos_x = pos_y = pos_z = 0.0
                            if len(flat_value) >= 7:  # SEW format with rotation
                                rot_x = round(float(flat_value[9]), nd) if len(flat_value) > 9 else 0.0
                                rot_y = round(float(flat_value[10]), nd) if len(flat_value) > 10 else 0.0
                                rot_z = round(float(flat_value[11]), nd) if len(flat_value) > 11 else 0.0
                                rot_w = 1.0
                            else:
                                rot_x = rot_y = rot_z = 0.0
                                rot_w = 1.0
                            rows.append((te, "action", action_key,
                                         pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w))
                        elif action_key in ("left_gripper_val", "right_gripper_val"):
                            rows.append((te, "action", action_key,
                                         round(float(action_value), nd),
                                         0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
                writer.writerows(rows)
                now = time.monotonic()
                if now - last_flush > 0.5:
                    f.flush()
                    last_flush = now
        except Exception as exc:                           # noqa: BLE001
            print(f"[csv] writer error: {exc}")
        finally:
            try:
                f.flush()
                f.close()
            except Exception:                              # noqa: BLE001
                pass

    def cleanup_recording(self):
        """Stop the CSV writer thread and finalize the file. Signals the sink to drain
        its queue, then joins it (bounded) so all buffered frames land before the next
        take opens a fresh file."""
        if getattr(self, "_csv_thread", None) is not None:
            self._csv_stop.set()
            self._csv_thread.join(timeout=5.0)
            if self._csv_thread.is_alive():
                print("[csv] writer thread still draining after 5 s — some tail frames "
                      "may be lost")
            self._csv_thread = None
            print(f"CSV recording stopped and saved to: {getattr(self, 'output_file', 'unknown')}")
        self.csv_file = None
        self.csv_writer = None
        self.started = False

    def on_body_pose_message(self, message: bytes, state: RobosuiteTeleopState):
        """Callback for the 'body_pose' data channel.

        Deserialises the frame and hands it to the dedicated bone-processor
        thread.  The processor always works on the newest frame, discarding
        any that arrived while the previous one was being processed.
        This function is safe to call from any thread or asyncio context.
        """
        if not state.is_connected:
            state.is_connected = True
            print("\n[WebRTC] Body pose data channel connected and receiving data.")

        try:
            state._packet_arrival_t = time.monotonic()  # stamp before any processing
            # Delivery-health tracking (packet-loss / HOL-blocking readout): a big gap since
            # the previous frame = the reliable body_pose channel waited on a retransmit.
            _now = state._packet_arrival_t
            with self._lh_lock:
                if self._lh_prev_t > 0.0:
                    _gap_ms = (_now - self._lh_prev_t) * 1000.0
                    self._lh_frames += 1
                    if _gap_ms > 50.0:            # >~3-5 dropped frames = a reliable-channel stall
                        self._lh_stalls += 1
                    if _gap_ms > self._lh_max_gap_ms:
                        self._lh_max_gap_ms = _gap_ms
                else:
                    self._lh_window_t0 = _now
                self._lh_prev_t = _now
            # Hand the RAW bytes to the bone-processor thread and return immediately —
            # deserialize_pose_data() (a per-bone struct-unpack + object build) now runs OFF
            # the asyncio event loop. Parsing here made the loop pay that cost for EVERY
            # inbound frame; when the loop fell behind, the reliable/ordered body_pose SCTP
            # channel QUEUED frames and the delivered pose got progressively staler — the
            # multi-second teleop lag that grows over time. Off-loop + drop-newest means the
            # loop drains inbound at line rate and only the FRESHEST frame is ever parsed.
            self._bone_processor.put(message, state)
        except Exception as e:
            import traceback
            print(f"Error in on_body_pose_message: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

    def _process_body_pose(self, bones: list, state: RobosuiteTeleopState) -> None:
        """Heavy bone → action computation; called from _BoneProcessor._run."""
        try:
            tags = state.tags.copy()
            if getattr(self, '_fn_accepts_tags', False):
                action_dict = self.process_bones_to_action_fn(bones, tags=tags)
            else:
                action_dict = self.process_bones_to_action_fn(bones)
            state.update_pose(action_dict)   # merges tags into action_dict in-place
            if self.record_data:
                self._record_to_csv(bones, action_dict)
            # Ship the freshest pose to the out-of-process parent (proxy host), if any. Runs
            # on the bone-proc thread; a slow parent can stall THIS thread but never the
            # event loop (that's the whole point of hosting the loop in this subprocess).
            if self.pose_sink is not None:
                try:
                    # Pass the pose-timing too (monotonic is system-wide on Linux, so these
                    # are valid in the parent process): arrival_t = WebRTC packet arrival,
                    # recv_t = bone-proc finish, seq = pose counter. Read on THIS thread right
                    # after update_pose() wrote them, so they're consistent without the lock.
                    self.pose_sink(action_dict, state._packet_arrival_t,
                                   state._pose_recv_t, state._pose_seq)
                except Exception:
                    pass
        except Exception as e:
            # Rate-limited: this runs per FRAME (~60 Hz) — a persistent error printing a
            # full traceback per frame floods stdout and adds real load mid-teleop.
            now = time.monotonic()
            self._pbp_err_n = getattr(self, "_pbp_err_n", 0) + 1
            if now - getattr(self, "_pbp_err_t", 0.0) > 5.0:
                self._pbp_err_t = now
                import traceback
                print(f"Error in _process_body_pose (x{self._pbp_err_n} since last "
                      f"report): {e}")
                print(f"Traceback:\n{traceback.format_exc()}")
                self._pbp_err_n = 0

    def on_apriltag_message(self, message: bytes, state: RobosuiteTeleopState):
        """
        Callback for the 'apriltag_pose' data channel.
        Processes the incoming byte stream from Unity.
        """
        import struct
        try:
            if len(message) < 4:
                return
            count = struct.unpack_from("<i", message, 0)[0]

            expected_size = 4 + count * 32  # 1 int (4B) + 7 floats (28B) = 32 bytes per tag
            if len(message) < expected_size:
                return

            tags = {}
            offset = 4
            for _ in range(count):
                tag_id, px, py, pz, rx, ry, rz, rw = struct.unpack_from("<i7f", message, offset)
                
                # Convert from Unity's Left-Handed Y-up to Right-Handed Z-up (+X Forward, +Y Left, +Z Up)
                # Position conversion: Unity (x,y,z) -> (z, -x, y)
                z_up_pos = np.array([pz, -px, py])
                
                # Rotation quaternion conversion: Unity (qx,qy,qz,qw) -> (-qz, qx, -qy, qw)
                z_up_quat = np.array([-rz, rx, -ry, rw])
                
                rot_matrix = q2R(z_up_quat)

                tags[tag_id] = {
                    "position": z_up_pos, 
                    "quaternion": z_up_quat,
                    "rotation_matrix": rot_matrix
                }
                offset += 32

            state.update_tags(tags)
        except Exception as e:
            import traceback
            print(f"Error processing apriltag message: {e}")

    def on_connection_state_change(self, state):
        """
        Callback for when the WebRTC connection state changes.
        """
        
        # When connection is lost or disconnected, trigger cleanup
        if state in ["disconnected", "closed", "failed"]:
            if self.record_data and self.started:
                print("\n[WebRTC] Connection lost, saving CSV data...")
                self.cleanup_recording()
                self.started = False  # Allow a new recording to start if reconnected

    async def _haptics_sender_loop(self) -> None:
        """Drain _haptics_queue and forward only the latest payload each cycle.

        Drains the entire queue but discards all but the most recent entry.
        This prevents a burst of stale haptic frames (accumulated while the event
        loop was busy) from being replayed to the Quest with artificial delay.
        """
        while True:
            latest = None
            while True:
                try:
                    latest = self._haptics_queue.get_nowait()
                except _queue.Empty:
                    break
            if latest is not None:
                self.server.send_to_datachannel("haptics", latest)
            await asyncio.sleep(0.001)  # 1 ms — yields without meaningful blocking

    def send_haptics(self, payload: str) -> None:
        """Enqueue a JSON haptics payload to be sent over the 'haptics' WebRTC data channel.

        Thread-safe: can be called from any thread.
        """
        self._haptics_queue.put_nowait(payload)

    async def _motor_stats_sender_loop(self) -> None:
        """Drain _motor_stats_queue and forward only the latest payload each cycle."""
        while True:
            latest = None
            while True:
                try:
                    latest = self._motor_stats_queue.get_nowait()
                except _queue.Empty:
                    break
            if latest is not None:
                self.server.send_to_datachannel("motor_stats", latest)
            await asyncio.sleep(0.05)  # 20 Hz poll — motor stats arrive at 10 Hz

    def send_motor_stats(self, payload: str) -> None:
        """Enqueue a JSON motor-stats payload to be sent over the 'motor_stats' WebRTC data channel.

        Thread-safe: can be called from any thread.
        """
        self._motor_stats_queue.put_nowait(payload)

    async def _unity_cmds_sender_loop(self) -> None:
        """Drain _unity_cmds_queue and forward EVERY queued payload, in order.

        Unlike the haptics / motor_stats streams — where only the freshest sample
        matters, so stale frames are coalesced away — unity_cmds carries discrete
        one-shot UI commands. Coalescing to 'latest' would silently drop a command
        whenever two are enqueued within one cycle (e.g. the connect-time state
        push that sends audio + vibrotactile + viewport back-to-back), so each
        payload is forwarded individually rather than collapsed to the newest.
        """
        cycles = 0
        while True:
            while True:
                try:
                    payload = self._unity_cmds_queue.get_nowait()
                except _queue.Empty:
                    break
                # Harden: a send to a not-yet-open channel must NEVER bubble up and kill this loop
                # (that would silently disable ALL manager->XR commands — toggles, teleop_active,
                # state push — while haptics/motor_stats on their own loops keep working).
                try:
                    self.server.send_to_datachannel("unity_cmds", payload)
                except Exception:
                    pass
            # ~1 Hz latency clock-sync ping (t0 stamped here, at actual send time, for accuracy).
            cycles += 1
            if cycles >= 50:
                cycles = 0
                self._send_latency_ping()
            await asyncio.sleep(0.02)

    def _send_latency_ping(self) -> None:
        """Send a clock-sync ping over unity_cmds. Quest echoes it as a latency_pong from which
        we derive the round-trip time on the teleop connection (see _handle_latency_pong)."""
        if not self.is_connected:
            return
        with self._lat_lock:
            self._lat_ping_id += 1
            pid = self._lat_ping_id
        payload = _json.dumps({
            "type": "unity_command", "command": "latency_ping",
            "ping_id": pid, "t0_ms": time.monotonic() * 1000.0,
        })
        try:
            self.server.send_to_datachannel("unity_cmds", payload)
        except Exception:
            pass

    def _handle_latency_pong(self, msg: dict) -> None:
        """Compute the teleop-connection RTT from a Quest pong. t0/t3 are manager-clock ms (cancel
        in t3-t0); t1/t2 are Quest-clock ms (cancel in t2-t1), so RTT = (t3-t0) - (t2-t1)."""
        try:
            t0 = float(msg["t0_ms"]); t1 = float(msg["t1_ms"]); t2 = float(msg["t2_ms"])
        except (KeyError, TypeError, ValueError):
            return
        t3 = time.monotonic() * 1000.0
        rtt = (t3 - t0) - (t2 - t1)
        if rtt < 0.0 or rtt > 2000.0:   # discard implausible samples (clock hiccup / huge stall)
            return
        with self._lat_lock:
            self._net_rtt_ms = rtt if self._net_rtt_ms is None else (0.7 * self._net_rtt_ms + 0.3 * rtt)

    def network_rtt_ms(self):
        """Latest smoothed teleop-connection round-trip time in ms (or None)."""
        with self._lat_lock:
            return self._net_rtt_ms

    def video_latency(self):
        """Latest Quest-reported video receive-latency breakdown dict (or None)."""
        with self._lat_lock:
            return dict(self._video_latency) if self._video_latency else None

    def pose_age_s(self):
        """Seconds since the last body_pose frame arrived (monotonic), or None if none has
        ever arrived. This is the direct 'is the teleop stream frozen?' signal: it grows
        both when a WiFi/WebRTC drop stops delivery AND when a starved event loop can't
        drain the channel. The manager's link-health watchdog parks the arms on it."""
        with self._lh_lock:
            if self._lh_prev_t <= 0.0:
                return None
            return max(0.0, time.monotonic() - self._lh_prev_t)

    def link_health(self, min_window_s: float = 3.0):
        """Inbound body_pose delivery health = the packet-loss / HOL-blocking readout.

        Returns {fps, stalls_per_s, max_gap_ms, rtt_ms} averaged over a rolling ~min_window_s
        window (or None if no frames yet). A STALL is a >50 ms inter-arrival gap — the
        RELIABLE/ordered body_pose channel waiting on a WiFi-loss retransmit. High stalls/s ⇒
        packet loss is stalling teleop; it should drop to ~0 once body_pose is unreliable.
        """
        with self._lh_lock:
            if self._lh_frames == 0 or self._lh_window_t0 <= 0.0:
                return None
            now = time.monotonic()
            elapsed = max(1e-3, now - self._lh_window_t0)
            out = {"fps": self._lh_frames / elapsed,
                   "stalls_per_s": self._lh_stalls / elapsed,
                   "max_gap_ms": self._lh_max_gap_ms}
            if elapsed >= min_window_s:          # roll the window so the readout stays stable
                self._lh_frames = 0
                self._lh_stalls = 0
                self._lh_max_gap_ms = 0.0
                self._lh_window_t0 = now
        out["rtt_ms"] = self.network_rtt_ms()    # read outside _lh_lock (own lock)
        return out

    def send_unity_command(self, payload: str) -> None:
        """Enqueue a JSON command payload to be sent over the 'unity_cmds' WebRTC data channel.

        Thread-safe: can be called from any thread. Unity receives these on the
        'unity_cmds' channel; payload should be a JSON string with a 'command' key.
        """
        self._unity_cmds_queue.put_nowait(payload)

    def unity_cmds_ready(self) -> bool:
        """True if at least one connected peer has an OPEN 'unity_cmds' data channel —
        i.e. a manager->XR command sent now would actually be delivered.

        The 'unity_cmds' channel is CREATED BY THE QUEST (offerer). If the XR app build
        doesn't open it, sends are silently dropped by send_to_datachannel (it just
        returns 0). The study manager calls this to surface that case instead of the
        command vanishing. Reads the server's per-peer channel map best-effort."""
        try:
            for pc in list(self.server.pcs):
                ctx = self.server._peer_context.get(pc)
                if ctx is None:
                    continue
                ch = ctx["channels"].get("unity_cmds")
                if ch is not None and getattr(ch, "readyState", None) == "open":
                    return True
        except Exception:
            pass
        return False

    def on_unity_state_message(self, message: bytes, state=None) -> None:
        """Callback for the 'unity_state' data channel (Unity → manager).

        Unity reports in-VR toggle changes here as JSON
        ``{"type":"unity_state","command":<cmd>,"enabled":<bool>}``. We just decode
        and enqueue; the study manager drains the queue on its Qt status tick via
        poll_unity_state() and updates its button UI. Runs on the WebRTC event-loop
        thread, so it must not touch Qt directly — hence the thread-safe queue.
        """
        try:
            text = message.decode("utf-8") if isinstance(message, (bytes, bytearray)) else str(message)
            msg = _json.loads(text)
        except Exception:
            return
        # Latency telemetry is consumed HERE (WebRTC thread) for accurate timing — not enqueued for
        # the GUI poll. Everything else is a toggle-state report for the study manager's buttons.
        cmd = msg.get("command")
        if cmd == "latency_pong":
            self._handle_latency_pong(msg)
            return
        if cmd == "video_latency":
            with self._lat_lock:
                self._video_latency = {
                    "jitter_ms": float(msg.get("jitter_ms", 0.0)),
                    "decode_ms": float(msg.get("decode_ms", 0.0)),
                    "net_ms":    float(msg.get("net_ms", 0.0)),
                }
            return
        try:
            self._unity_state_queue.put_nowait(msg)
        except Exception:
            pass

    def poll_unity_state(self) -> list:
        """Drain and return all pending Unity toggle-state reports as parsed dicts.

        Thread-safe; intended to be called from the Qt thread. Returns [] when idle.
        """
        out = []
        while True:
            try:
                out.append(self._unity_state_queue.get_nowait())
            except _queue.Empty:
                break
        return out

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        # No-op for this device, as the server is already running.
        pass

    def get_controller_state(self):
        """
        Returns the current state of the device, a dictionary of pos, orn, grasp, and reset.
        """
        shared_state = self.state_factory.instance
        if not self.is_connected:
            return None

        pose_action = shared_state.get_pose()
        if pose_action is None:
            return None
        return pose_action

    def input2action(self, mirror_actions=False):
        """
        Converts an input from an active device into a valid action sequence that can be fed into an env.step() call
        If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action
        Args:
            mirror_actions (bool): actions corresponding to viewing robot from behind.
                first axis: left/right. second axis: back/forward. third axis: down/up.
        Returns:
            Optional[Dict]: Dictionary of actions to be fed into env.step()
                            if reset is triggered, returns None
        """

        input_ac_dict = self.get_controller_state()

        if input_ac_dict is None:
            return None

        # Create the action vector for robosuite
        if self.env is None:
            return input_ac_dict

        # Create the action vector for robosuite environments
        active_robot = self.env.robots[0]
        action_dict = deepcopy(input_ac_dict)
        for arm in active_robot.arms:
            action_dict[arm] = input_ac_dict.get(f"{arm}_sew")
            action_dict[f"{arm}_gripper"] = input_ac_dict.get(f"{arm}_gripper")

        return action_dict
