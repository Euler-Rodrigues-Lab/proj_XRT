"""Plotting utilities for Inspire hand sensor visualization."""

import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np


LOCAL_STATUS_CODES = {
    0: "Opening",
    1: "Closing",
    2: "Position Stop",
    3: "Force Stop",
    5: "Current Protection Stop",
    6: "Stall Stop",
    7: "Actuator Fault Stop",
    255: "Error",
}

LOCAL_ERROR_DESCRIPTIONS = {
    0: "Stall Fault",
    1: "Overheat Fault",
    2: "Overcurrent Fault",
    3: "Motor Fault",
    4: "Communication Fault",
}


_MUJOCO_IDENTITY_MAT = np.array(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    dtype=np.float64,
)
_DEFAULT_LOCAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_DEFAULT_LOCAL_FRAME = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MujocoSensorPoint:
    geom_id: int
    geom_local_pos: tuple[float, float, float]
    normal_local: tuple[float, float, float]
    mesh_name: str


_FINGER_TOKEN_TO_TOUCH_KEY = {
    "little": "fingerone",
    "ring": "fingertwo",
    "middle": "fingerthree",
    "index": "fingerfour",
    "thumb": "fingerfive",
}

_TOUCH_SEGMENT_LAYOUTS: dict[str, dict[str, tuple[int, int, tuple[int, int]]]] = {
    "fingerone": {
        "tip": (0, 9, (3, 3)),
        "top": (9, 105, (12, 8)),
        "pulp": (105, 185, (10, 8)),
    },
    "fingertwo": {
        "tip": (0, 9, (3, 3)),
        "top": (9, 105, (12, 8)),
        "pulp": (105, 185, (10, 8)),
    },
    "fingerthree": {
        "tip": (0, 9, (3, 3)),
        "top": (9, 105, (12, 8)),
        "pulp": (105, 185, (10, 8)),
    },
    "fingerfour": {
        "tip": (0, 9, (3, 3)),
        "top": (9, 105, (12, 8)),
        "pulp": (105, 185, (10, 8)),
    },
    "fingerfive": {
        "tip": (0, 9, (3, 3)),
        "top": (9, 105, (12, 8)),
        "middle": (105, 114, (3, 3)),
        "pulp": (114, 210, (12, 8)),
    },
    "fingerpalm": {
        "whole": (0, 112, (14, 8)),
    },
}

_NON_THUMB_INDEX_TO_SEGMENT = {
    1: "pulp",
    2: "top",
    3: "tip",
}

_THUMB_INDEX_TO_SEGMENT = {
    1: "pulp",
    2: "middle",
    3: "top",
    4: "tip",
}


def _vec_norm(v: tuple[float, float, float]) -> float:
    return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def _normalize(
    v: tuple[float, float, float],
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    n = _vec_norm(v)
    if n < 1e-12:
        return fallback
    return (v[0] / n, v[1] / n, v[2] / n)


def _mat3_mul_vec(
    mat: tuple[float, ...],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
        mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
        mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2],
    )


def _configured_local_direction(local_axis: np.ndarray, local_frame: np.ndarray) -> tuple[float, float, float]:
    tuned_direction = np.asarray(local_frame, dtype=np.float64) @ np.asarray(local_axis, dtype=np.float64)
    return _normalize(
        (float(tuned_direction[0]), float(tuned_direction[1]), float(tuned_direction[2])),
        (0.0, 0.0, 1.0),
    )


def _resolve_sensor_normal(
    mesh_name: str,
    sensor_direction_overrides: Dict[str, Dict[str, object]] | None,
) -> tuple[float, float, float]:
    override = (sensor_direction_overrides or {}).get(mesh_name, {})
    local_axis = np.asarray(override.get("axis", _DEFAULT_LOCAL_AXIS), dtype=np.float64)
    local_frame = np.asarray(override.get("frame", _DEFAULT_LOCAL_FRAME), dtype=np.float64)
    if local_axis.shape != (3,):
        raise ValueError(f"Override axis for {mesh_name} must have shape (3,), got {local_axis.shape}")
    if local_frame.shape != (3, 3):
        raise ValueError(f"Override frame for {mesh_name} must have shape (3, 3), got {local_frame.shape}")
    return _configured_local_direction(local_axis, local_frame)


def collect_mujoco_sensor_points(
    model: Any,
    *,
    mesh_name_contains: str = "sensor",
    sensor_direction_overrides: Dict[str, Dict[str, object]] | None = None,
) -> list[MujocoSensorPoint]:
    """Collect sensor points from mesh geoms in a MuJoCo model.

    This uses the same mesh-local to geom-local center computation pattern as the
    reference g1_sensor_sync script, so no external CSV/XML point definitions are needed.
    """
    import mujoco

    sensor_points: list[MujocoSensorPoint] = []
    seen_mesh_ids: set[int] = set()
    name_filter = mesh_name_contains.lower().strip()

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue

        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0 or mesh_id in seen_mesh_ids:
            continue

        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if mesh_name is None:
            continue
        if name_filter and name_filter not in mesh_name.lower():
            continue

        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        if vert_num <= 0:
            continue

        mesh_vertices = np.asarray(model.mesh_vert[vert_adr: vert_adr + vert_num], dtype=np.float64)
        mesh_quat = np.asarray(model.mesh_quat[mesh_id], dtype=np.float64)
        mesh_pos = np.asarray(model.mesh_pos[mesh_id], dtype=np.float64)
        mesh_rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mesh_rot, mesh_quat)
        mesh_rot = mesh_rot.reshape(3, 3)

        geom_vertices = mesh_vertices @ mesh_rot.T + mesh_pos
        center_geom = geom_vertices.mean(axis=0)
        chosen_normal = _resolve_sensor_normal(mesh_name, sensor_direction_overrides)

        sensor_points.append(
            MujocoSensorPoint(
                geom_id=geom_id,
                geom_local_pos=(
                    float(center_geom[0]),
                    float(center_geom[1]),
                    float(center_geom[2]),
                ),
                normal_local=chosen_normal,
                mesh_name=mesh_name,
            )
        )
        seen_mesh_ids.add(mesh_id)

    sensor_points.sort(key=lambda point: point.geom_id)
    return sensor_points


def _geom_local_to_world(
    data: Any,
    geom_id: int,
    local_pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    geom_xmat = tuple(float(value) for value in data.geom_xmat[geom_id])
    geom_xpos = tuple(float(value) for value in data.geom_xpos[geom_id])
    rotated = _mat3_mul_vec(geom_xmat, local_pos)
    return (
        geom_xpos[0] + rotated[0],
        geom_xpos[1] + rotated[1],
        geom_xpos[2] + rotated[2],
    )


def _geom_dir_to_world(
    data: Any,
    geom_id: int,
    local_dir: tuple[float, float, float],
) -> tuple[float, float, float]:
    geom_xmat = tuple(float(value) for value in data.geom_xmat[geom_id])
    return _normalize(_mat3_mul_vec(geom_xmat, local_dir), (0.0, 0.0, 1.0))


def create_mujoco_sensor_overlay_runtime(
    model: Any,
    *,
    mesh_name_contains: str = "sensor",
    sensor_direction_overrides: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, Any]:
    """Build one-time runtime state for MuJoCo sensor overlays."""
    sensor_points = collect_mujoco_sensor_points(
        model,
        mesh_name_contains=mesh_name_contains,
        sensor_direction_overrides=sensor_direction_overrides,
    )
    region_counts, unmapped_meshes = _summarize_sensor_point_region_mapping(sensor_points)
    return {
        "sensor_points": sensor_points,
        "count": len(sensor_points),
        "region_counts": region_counts,
        "unmapped_meshes": unmapped_meshes,
    }


def build_mujoco_sensor_overlay_values_from_snapshot(
    snapshot: Dict[str, Any] | None,
    sensor_points: Iterable[MujocoSensorPoint],
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Create overlay values using hard mapping from mesh names to touch regions."""
    points = list(sensor_points)
    if not points:
        return np.zeros(0, dtype=np.float64)
    if not snapshot:
        return np.zeros(len(points), dtype=np.float64)

    point_regions: list[tuple[str, str, str] | None] = []
    required_regions: set[tuple[str, str, str]] = set()
    for point in points:
        region = _infer_touch_region_from_mesh_name(point.mesh_name)
        point_regions.append(region)
        if region is not None:
            required_regions.add(region)

    region_values = _build_touch_region_values(snapshot, required_regions)
    values = np.zeros(len(points), dtype=np.float64)
    for i, region in enumerate(point_regions):
        if region is None:
            continue
        values[i] = float(region_values.get(region, 0.0))

    values = np.clip(values, 0.0, None)
    if not normalize:
        return values

    return _normalize_values_frame(values)


def _normalize_values_frame(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values

    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if v_max > v_min:
        return (values - v_min) / (v_max - v_min)
    elif v_max > 0.0:
        return np.ones_like(values)
    return np.zeros_like(values)


def prepare_live_scalar_arrow_values(
    raw_values: np.ndarray | Iterable[float],
    runtime_state: Dict[str, Any] | None = None,
    *,
    smoothing_alpha: float = 0.35,
    deadband: float = 0.02,
    normalization_mode: str = "running",
    fixed_max: float = 1024.0,
    running_decay: float = 0.995,
) -> np.ndarray:
    """Condition raw per-point scalars into stable [0, 1] live arrow values.

    Supported normalization modes:
    - frame: per-frame min-max normalization
    - fixed: divide by fixed_max and clamp to [0, 1]
    - running: divide by a decaying running maximum in runtime_state
    """
    values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values

    values = np.clip(values, 0.0, None)

    alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
    previous = None if runtime_state is None else runtime_state.get("smoothed_values")
    if isinstance(previous, np.ndarray) and previous.shape == values.shape:
        smoothed = alpha * values + (1.0 - alpha) * previous
    else:
        smoothed = values

    mode = str(normalization_mode).lower()
    if mode == "fixed":
        denom = max(float(fixed_max), 1e-9)
        normalized = np.clip(smoothed / denom, 0.0, 1.0)
    elif mode == "running":
        decay = float(np.clip(running_decay, 0.0, 1.0))
        prev_max = 0.0
        if runtime_state is not None:
            prev_max = float(runtime_state.get("running_max", 0.0))
        current_max = float(np.max(smoothed)) if smoothed.size else 0.0
        running_max = max(current_max, prev_max * decay)
        if running_max <= 1e-9:
            normalized = np.zeros_like(smoothed)
        else:
            normalized = np.clip(smoothed / running_max, 0.0, 1.0)
        if runtime_state is not None:
            runtime_state["running_max"] = running_max
    else:
        normalized = _normalize_values_frame(smoothed)

    threshold = max(0.0, float(deadband))
    if threshold > 0.0:
        normalized = normalized.copy()
        normalized[normalized < threshold] = 0.0

    if runtime_state is not None:
        runtime_state["smoothed_values"] = smoothed
        runtime_state["last_values"] = normalized

    return normalized


def _infer_touch_region_from_mesh_name(mesh_name: str) -> tuple[str, str, str] | None:
    name = mesh_name.lower()
    if name.startswith("left_"):
        hand_label = "Left"
    elif name.startswith("right_"):
        hand_label = "Right"
    else:
        return None

    if "palm_force_sensor" in name and "thumb_force_sensor" not in name:
        return (hand_label, "fingerpalm", "whole")

    for finger_token, touch_key in _FINGER_TOKEN_TO_TOUCH_KEY.items():
        if f"_{finger_token}_force_sensor" not in name:
            continue

        match = re.search(r"_force_sensor_(\d+)$", name)
        sensor_index = int(match.group(1)) if match else 0
        if touch_key == "fingerfive":
            segment = _THUMB_INDEX_TO_SEGMENT.get(sensor_index)
        else:
            segment = _NON_THUMB_INDEX_TO_SEGMENT.get(sensor_index)
        if segment is None:
            return None
        return (hand_label, touch_key, segment)

    return None


def _summarize_sensor_point_region_mapping(
    sensor_points: Iterable[MujocoSensorPoint],
) -> tuple[Dict[str, int], list[str]]:
    region_counts: Dict[str, int] = {}
    unmapped_meshes: list[str] = []
    for point in sensor_points:
        region = _infer_touch_region_from_mesh_name(point.mesh_name)
        if region is None:
            unmapped_meshes.append(point.mesh_name)
            continue
        region_key = f"{region[0]}.{region[1]}.{region[2]}"
        region_counts[region_key] = region_counts.get(region_key, 0) + 1
    return region_counts, unmapped_meshes


def _build_touch_region_values(
    snapshot: Dict[str, Any],
    required_regions: set[tuple[str, str, str]],
) -> Dict[tuple[str, str, str], float]:
    touch = snapshot.get("touch", {})
    region_values: Dict[tuple[str, str, str], float] = {}
    for region in required_regions:
        hand_label, touch_key, segment = region
        hand_touch = touch.get(hand_label, {})
        raw = hand_touch.get(touch_key)
        if raw is None:
            region_values[region] = 0.0
            continue
        segment_cfg = _TOUCH_SEGMENT_LAYOUTS.get(touch_key, {}).get(segment)
        if segment_cfg is None:
            region_values[region] = 0.0
            continue
        start, end, shape = segment_cfg
        segment_grid = _reshape_segment(np.asarray(raw), start, end, shape)
        region_values[region] = average_grid_signal(segment_grid, shape[0], shape[1])
    return region_values


def render_mujoco_sensor_overlay(
    viewer: Any,
    data: Any,
    sensor_points: Iterable[MujocoSensorPoint],
    values: np.ndarray | None = None,
    *,
    marker_radius: float = 0.008,
    marker_rgba: tuple[float, float, float, float] = (1.0, 0.25, 0.1, 0.95),
    arrow_scale: float = 0.25,
    arrow_width: float = 0.004,
) -> None:
    """Render MuJoCo sensor marker/arrow overlays in viewer.user_scn."""
    import mujoco

    points = list(sensor_points)
    if values is None:
        value_arr = np.ones(len(points), dtype=np.float64)
    else:
        value_arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if value_arr.size == 0:
            value_arr = np.zeros(len(points), dtype=np.float64)
        elif value_arr.size < len(points):
            repeats = (len(points) + value_arr.size - 1) // value_arr.size
            value_arr = np.tile(value_arr, repeats)[:len(points)]
        else:
            value_arr = value_arr[:len(points)]

    viewer.user_scn.ngeom = 0
    for idx, sensor_point in enumerate(points):
        if viewer.user_scn.ngeom + 2 >= viewer.user_scn.maxgeom:
            break

        value = float(value_arr[idx])
        value = float(np.clip(value, 0.0, 1.0))
        world_pos = _geom_local_to_world(data, sensor_point.geom_id, sensor_point.geom_local_pos)

        marker_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            marker_geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([marker_radius, marker_radius, marker_radius], dtype=np.float64),
            np.array(world_pos, dtype=np.float64),
            _MUJOCO_IDENTITY_MAT,
            np.array(marker_rgba, dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1

        world_normal = _geom_dir_to_world(data, sensor_point.geom_id, sensor_point.normal_local)
        arrow_len = arrow_scale * value
        arrow_end = (
            world_pos[0] + arrow_len * world_normal[0],
            world_pos[1] + arrow_len * world_normal[1],
            world_pos[2] + arrow_len * world_normal[2],
        )
        arrow_color = (
            0.2 + 0.8 * value,
            0.15 + 0.35 * value,
            1.0 - 0.8 * value,
            0.85,
        )

        arrow_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_connector(
            arrow_geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            arrow_width,
            world_pos,
            arrow_end,
        )
        arrow_geom.rgba[0] = arrow_color[0]
        arrow_geom.rgba[1] = arrow_color[1]
        arrow_geom.rgba[2] = arrow_color[2]
        arrow_geom.rgba[3] = arrow_color[3]
        viewer.user_scn.ngeom += 1


def build_local_touch_plot_layout(reg_addr: Any):
    """Build local touch plot schema aligned with register naming in reg_addr."""
    return [
        ("FINGERONE_TOUCH_TIP", reg_addr.FINGERONE_TOUCH, 18, (3, 3), "FINGERONE_TOUCH_TIP"),
        ("FINGERONE_TOUCH_TOP", reg_addr.FINGERONE_TOUCH + 18, 192, (12, 8), "FINGERONE_TOUCH_TOP"),
        ("FINGERONE_TOUCH_PALM", reg_addr.FINGERONE_TOUCH + 18 + 192, 160, (10, 8), "FINGERONE_TOUCH_PALM"),
        ("FINGERTWO_TOUCH_TIP", reg_addr.FINGERTWO_TOUCH, 18, (3, 3), "FINGERTWO_TOUCH_TIP"),
        ("FINGERTWO_TOUCH_TOP", reg_addr.FINGERTWO_TOUCH + 18, 192, (12, 8), "FINGERTWO_TOUCH_TOP"),
        ("FINGERTWO_TOUCH_PALM", reg_addr.FINGERTWO_TOUCH + 18 + 192, 160, (10, 8), "FINGERTWO_TOUCH_PALM"),
        ("FINGERTHREE_TOUCH_TIP", reg_addr.FINGERTHE_TOUCH, 18, (3, 3), "FINGERTHREE_TOUCH_TIP"),
        ("FINGERTHREE_TOUCH_TOP", reg_addr.FINGERTHE_TOUCH + 18, 192, (12, 8), "FINGERTHREE_TOUCH_TOP"),
        ("FINGERTHREE_TOUCH_PALM", reg_addr.FINGERTHE_TOUCH + 18 + 192, 160, (10, 8), "FINGERTHREE_TOUCH_PALM"),
        ("FINGERFOUR_TOUCH_TIP", reg_addr.FINGERFOR_TOUCH, 18, (3, 3), "FINGERFOUR_TOUCH_TIP"),
        ("FINGERFOUR_TOUCH_TOP", reg_addr.FINGERFOR_TOUCH + 18, 192, (12, 8), "FINGERFOUR_TOUCH_TOP"),
        ("FINGERFOUR_TOUCH_PALM", reg_addr.FINGERFOR_TOUCH + 18 + 192, 160, (10, 8), "FINGERFOUR_TOUCH_PALM"),
        ("FINGERFIVE_TOUCH_TIP", reg_addr.FINGERFIV_TOUCH, 18, (3, 3), "FINGERFIVE_TOUCH_TIP"),
        ("FINGERFIVE_TOUCH_TOP", reg_addr.FINGERFIV_TOUCH + 18, 192, (12, 8), "FINGERFIVE_TOUCH_TOP"),
        ("FINGERFIVE_TOUCH_MIDDLE", reg_addr.FINGERFIV_TOUCH + 18 + 192, 18, (3, 3), "FINGERFIVE_TOUCH_MIDDLE"),
        ("FINGERFIVE_TOUCH_PALM", reg_addr.FINGERFIV_TOUCH + 18 + 192 + 18, 192, (12, 8), "FINGERFIVE_TOUCH_PALM"),
        ("FINGERPALM_TOUCH", reg_addr.FINGERPALM_TOUCH, 224, (14, 8), "FINGERPALM_TOUCH"),
    ]


def _decode_error_reasons(error_value: int) -> list[str]:
    reasons: list[str] = []
    for bit, description in LOCAL_ERROR_DESCRIPTIONS.items():
        if error_value & (1 << bit):
            reasons.append(description)
    return reasons


def _format_error_label(errors: list[int]) -> str:
    summary: list[str] = []
    for error in errors:
        error_int = int(error)
        binary_error = f"{error_int:04b}"
        reasons = _decode_error_reasons(error_int)
        if reasons:
            summary.append(f"ERROR {error_int} ({binary_error}): " + ", ".join(reasons))
        else:
            summary.append(f"ERROR {error_int} ({binary_error}): No Fault")
    return "\t".join(summary)


def _reshape_segment(raw: np.ndarray, start: int, end: int, shape: tuple[int, int]) -> np.ndarray:
    segment = raw[start:end]
    if segment.size != (shape[0] * shape[1]):
        return np.zeros(shape, dtype=int)
    return segment.reshape(shape)


def average_grid_signal(sensor: Any, length: int, width: int) -> float:
    """Return the mean value of a sensor grid as a single scalar.

    The input may already have the expected 2D shape or may be flat. If the
    expected reshape is not possible, the helper falls back to averaging the
    flattened values so callers still get a stable summary reading.
    """
    if length <= 0 or width <= 0:
        return 0.0

    arr = np.asarray(sensor, dtype=np.float64)
    if arr.size == 0:
        return 0.0

    if arr.shape == (length, width):
        return float(np.nanmean(arr))

    if arr.size == length * width:
        return float(np.nanmean(arr.reshape(length, width)))

    return float(np.nanmean(arr.reshape(-1)))


def build_hand_qt_payload(snapshot: Dict, hand_label: str) -> Dict:
    """Adapt a hand snapshot to local register-name-based Qt plotting payload."""
    hand_states = snapshot['states'][hand_label]
    hand_touch = snapshot['touch'][hand_label]

    fingerone = np.asarray(hand_touch['fingerone'], dtype=int)
    fingertwo = np.asarray(hand_touch['fingertwo'], dtype=int)
    fingerthree = np.asarray(hand_touch['fingerthree'], dtype=int)
    fingerfour = np.asarray(hand_touch['fingerfour'], dtype=int)
    fingerfive = np.asarray(hand_touch['fingerfive'], dtype=int)
    fingerpalm = np.asarray(hand_touch['fingerpalm'], dtype=int)

    touch = {
        'FINGERONE_TOUCH_TIP': _reshape_segment(fingerone, 0, 9, (3, 3)),
        'FINGERONE_TOUCH_TOP': _reshape_segment(fingerone, 9, 105, (12, 8)),
        'FINGERONE_TOUCH_PALM': _reshape_segment(fingerone, 105, 185, (10, 8)),
        'FINGERTWO_TOUCH_TIP': _reshape_segment(fingertwo, 0, 9, (3, 3)),
        'FINGERTWO_TOUCH_TOP': _reshape_segment(fingertwo, 9, 105, (12, 8)),
        'FINGERTWO_TOUCH_PALM': _reshape_segment(fingertwo, 105, 185, (10, 8)),
        'FINGERTHREE_TOUCH_TIP': _reshape_segment(fingerthree, 0, 9, (3, 3)),
        'FINGERTHREE_TOUCH_TOP': _reshape_segment(fingerthree, 9, 105, (12, 8)),
        'FINGERTHREE_TOUCH_PALM': _reshape_segment(fingerthree, 105, 185, (10, 8)),
        'FINGERFOUR_TOUCH_TIP': _reshape_segment(fingerfour, 0, 9, (3, 3)),
        'FINGERFOUR_TOUCH_TOP': _reshape_segment(fingerfour, 9, 105, (12, 8)),
        'FINGERFOUR_TOUCH_PALM': _reshape_segment(fingerfour, 105, 185, (10, 8)),
        'FINGERFIVE_TOUCH_TIP': _reshape_segment(fingerfive, 0, 9, (3, 3)),
        'FINGERFIVE_TOUCH_TOP': _reshape_segment(fingerfive, 9, 105, (12, 8)),
        'FINGERFIVE_TOUCH_MIDDLE': _reshape_segment(fingerfive, 105, 114, (3, 3)),
        'FINGERFIVE_TOUCH_PALM': _reshape_segment(fingerfive, 114, 210, (12, 8)),
        'FINGERPALM_TOUCH': _reshape_segment(fingerpalm, 0, 112, (14, 8)),
    }

    return {
        'states': {
            'POS_ACT': list(hand_states['POS_ACT']),
            'ANGLE_ACT': list(hand_states['ANGLE_ACT']),
            'FORCE_ACT': list(hand_states['FORCE_ACT']),
            'CURRENT': list(hand_states['CURRENT']),
            'ERROR': list(hand_states['ERROR']),
            'STATUS': list(hand_states['STATUS']),
            'TEMP': list(hand_states['TEMP']),
        },
        'touch': touch,
    }


def _create_image_tab(tabs, data_sheet, pg, colorcet):
    from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QWidget

    image_tab = QWidget()
    root_layout = QVBoxLayout()
    image_tab.setLayout(root_layout)
    grid_layout = QGridLayout()
    root_layout.addLayout(grid_layout)

    ui = {
        'widget': image_tab,
        'plots': [],
        'color_maps': [],
        'color_bars': [],
        'data_sheet': data_sheet,
    }

    num_cols = 4
    for i, (name, _addr, _length, size, _var) in enumerate(data_sheet):
        row = i // num_cols
        col = i % num_cols

        layout_widget = pg.GraphicsLayoutWidget(show=True)
        plot_item = layout_widget.addPlot(row=0, col=0)
        plot_item.setTitle(name)
        img_item = pg.ImageItem(np.zeros((size[0], size[1]), dtype=float))
        plot_item.addItem(img_item)
        ui['plots'].append(img_item)

        color_map = pg.ColorMap(pos=np.linspace(0, 1, 256), color=colorcet.fire[:256])
        ui['color_maps'].append(color_map)

        color_bar = pg.ColorBarItem(colorMap=color_map, values=(0, 1), width=5, orientation='h')
        ui['color_bars'].append(color_bar)
        layout_widget.addItem(color_bar, row=1, col=0)

        grid_layout.addWidget(layout_widget, row, col)

    tabs.addTab(image_tab, "Images")
    return ui


def _create_curve_tab(tabs, pg, colorcet, history_len=100):
    from PyQt5.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

    curve_tab = QWidget()
    root_layout = QVBoxLayout()
    curve_tab.setLayout(root_layout)
    grid_layout = QGridLayout()
    root_layout.addLayout(grid_layout)

    error_label = QLabel("ERROR: ")
    status_label = QLabel("STATUS: ")
    root_layout.addWidget(error_label)
    root_layout.addWidget(status_label)

    categories = ['POS_ACT', 'ANGLE_ACT', 'FORCE_ACT', 'CURRENT', 'ERROR', 'STATUS', 'TEMP']
    plot_items = {name: pg.PlotWidget() for name in categories}
    for i, (name, plot_widget) in enumerate(plot_items.items()):
        grid_layout.addWidget(plot_widget, i // 2, i % 2)
        plot_widget.setTitle(name)
        plot_widget.setLabel('left', 'Y-axis')
        plot_widget.setLabel('bottom', 'X-axis')
        plot_widget.setBackground((0, 0, 0))
        plot_widget.addLegend()
        plot_widget.showButtons()
        plot_widget.enableAutoRange()
        plot_widget.showGrid(x=True, y=True, alpha=0.2)

    curves = {
        name: [plot_items[name].plot(pen=pg.mkPen(color), name=f'{name} {i + 1}')
               for i, color in enumerate(colorcet.glasbey[:6])]
        for name in categories
    }
    history = {name: [np.zeros(history_len) for _ in range(6)] for name in categories}

    tabs.addTab(curve_tab, "Curves")
    return {
        'widget': curve_tab,
        'categories': categories,
        'error_label': error_label,
        'status_label': status_label,
        'curves': curves,
        'history': history,
    }


def _update_image_tab(image_ui, touch_payload):
    for i, (_name, _addr, _length, _size, var) in enumerate(image_ui['data_sheet']):
        img = touch_payload[var]
        image_ui['plots'][i].setImage(img, autoLevels=True)
        max_val = float(np.max(img)) if img.size else 1.0
        if max_val <= 0:
            max_val = 1.0
        image_ui['plots'][i].setLevels((0, max_val))
        image_ui['color_bars'][i].setLevels((0, max_val))
        image_ui['plots'][i].setColorMap(image_ui['color_maps'][i])


def _update_curve_tab(curve_ui, states_payload):
    for category in curve_ui['categories']:
        datas = states_payload.get(category)
        if datas is None:
            continue
        for i in range(min(6, len(datas))):
            curve_ui['history'][category][i] = np.roll(curve_ui['history'][category][i], -1)
            curve_ui['history'][category][i][-1] = datas[i]
            curve_ui['curves'][category][i].setData(curve_ui['history'][category][i])

    err = _format_error_label(states_payload['ERROR'])
    curve_ui['error_label'].setText(err)
    status_text = ', '.join([
        '%s' % LOCAL_STATUS_CODES.get(int(s), f'UNKNOWN({s})')
        for s in states_payload['STATUS']
    ])
    curve_ui['status_label'].setText("STATUS: " + status_text)


def create_dual_hand_plot_context(
    reg_addr: Any,
    *,
    plot_touch: bool = True,
    window_title: str = "Modbus Subscribe Dual",
    geometry: tuple[int, int, int, int] = (100, 100, 1600, 900),
    history_len: int = 100,
):
    """Create and return dual-hand Qt plotting context (UI objects only)."""
    import pyqtgraph as pg
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
    import colorcet

    app = QApplication(sys.argv)
    data_sheet = build_local_touch_plot_layout(reg_addr)

    window = QMainWindow()
    window.setWindowTitle(window_title)
    window.setGeometry(*geometry)
    root_tabs = QTabWidget()
    window.setCentralWidget(root_tabs)

    left_tabs = QTabWidget()
    right_tabs = QTabWidget()
    root_tabs.addTab(left_tabs, "Left")
    root_tabs.addTab(right_tabs, "Right")

    left_image_ui = _create_image_tab(left_tabs, data_sheet, pg, colorcet) if plot_touch else None
    left_curve_ui = _create_curve_tab(left_tabs, pg, colorcet, history_len=history_len)
    right_image_ui = _create_image_tab(right_tabs, data_sheet, pg, colorcet) if plot_touch else None
    right_curve_ui = _create_curve_tab(right_tabs, pg, colorcet, history_len=history_len)

    ctx = {
        'app': app,
        'window': window,
        'plot_touch': plot_touch,
        'left_image_ui': left_image_ui,
        'left_curve_ui': left_curve_ui,
        'right_image_ui': right_image_ui,
        'right_curve_ui': right_curve_ui,
    }
    return ctx


def refresh_dual_hand_plot_context(ctx: Dict, snapshot: Dict):
    """Refresh both left/right plots from the provided controller snapshot."""
    left_payload = build_hand_qt_payload(snapshot, 'Left')
    right_payload = build_hand_qt_payload(snapshot, 'Right')

    _update_curve_tab(ctx['left_curve_ui'], left_payload['states'])
    _update_curve_tab(ctx['right_curve_ui'], right_payload['states'])
    if ctx['plot_touch'] and ctx['left_image_ui'] is not None:
        _update_image_tab(ctx['left_image_ui'], left_payload['touch'])
    if ctx['plot_touch'] and ctx['right_image_ui'] is not None:
        _update_image_tab(ctx['right_image_ui'], right_payload['touch'])
