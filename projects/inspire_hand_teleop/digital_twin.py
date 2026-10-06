#!/usr/bin/env python3
"""Digital-twin reconstruction: table + crate + BOTH G1 robots in one MuJoCo scene.

This is the two-robot counterpart to ``session_viewer.py``. Where the session viewer
plays back one robot in its own base frame, the digital twin composes the WHOLE
workspace in a single shared world (= the table / AprilTag-0 frame):

  * the **table** (a static reference slab at the world origin),
  * the **crate** (its per-frame AprilTag pose, in the table frame), and
  * **both robots**, each posed from its recorded joint angles AND *placed* at the
    base pose the head-cam localizer solved for it (``head_base_*.csv`` =
    ``T_world_base``, the pelvis pose in the table frame).

Because the model's pelvis is a FIXED base (no free joint), the two robots are
composed with ``mujoco.MjSpec`` — two prefixed copies of the G1 model, each welded
under a frame at its localized base pose (corrected for the model's built-in pelvis
height so the pelvis lands exactly on ``T_world_base``).

The two robots are recorded independently on two machines, so they are re-synced on
the **absolute epoch** timeline (``t_epoch`` / ``zed_record_start_epoch`` /
crate-pose timestamps — all ``time.time()``). This assumes the two machines' clocks
are reasonably in sync (NTP / Tailscale); a constant skew shifts one robot in time.

Data comes from a session recorded by ``XRT_user_study_manager.py`` and gathered in
ONE place by the peer's **Send Data to Master** button, so both robots' session
folders sit side by side::

    {output_dir}/{pid}_robot1_{date}_{time}/{condition}/attempt{K}/…
    {output_dir}/{pid}_robot2_{date}_{time}/{condition}/attempt{K}/…

Pass EITHER robot's attempt folder (or its session root / a robot_raw CSV) — the
matching folder for the other robot is auto-found by participant id + the same
{condition}/attempt{K} subpath.

SOLO takes (the manager's single-operator "Start Recording") play too: ONE robot, no
partner, no head-cam base localization, no crate, no video::

    {output_dir}/solo_{date}_{time}/solo/attempt{K}/robot{N}/…   (meta "solo": true)

The robot stands at the world origin; the table / crate are drawn only if that take
actually holds crate data.

Controls (focus the OpenCV window):
  seek bar (drag/click, YouTube-style)   space = play/pause
  a / d = -5s / +5s    A / D = -30s / +30s    , / . = slower / faster
  r = restart          q or ESC = quit

Examples
--------
  python digital_twin.py /data/xr_recordings/P1_robot1_20260708_1530/V/attempt1
  python digital_twin.py /data/xr_recordings/P1_robot2_20260708_1530  # session root
  python digital_twin.py <attempt_dir> --no-video          # MuJoCo only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wrench_estimation as we           # noqa: E402
import session_viewer as sv              # noqa: E402  (reuse loaders + scene helpers)

try:
    import cv2
except Exception:                        # noqa: BLE001
    cv2 = None


# Drop CAD here to skin the twin: twin_assets/table.stl (origin at table tag 0) and
# twin_assets/crate.stl (origin at crate tag 10). Auto-used if present; override with
# --table-stl / --crate-stl. See twin_assets/README.md.
_ASSETS_DIR = _HERE / "twin_assets"


def _resolve_asset(explicit, default_name):
    """Return the STL path to use: the explicit --*-stl flag if given, else
    twin_assets/<default_name> if it exists, else None (fall back to a box placeholder)."""
    if explicit:
        p = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.exists(p):
            print(f"[twin] STL not found: {p} — using the box placeholder.")
            return None
        return p
    cand = _ASSETS_DIR / default_name
    return str(cand) if cand.exists() else None


# Two on-disk layouts are supported:
#   NEW (merged, 2026-07+): one shared session folder, each robot under a robot{N}/ leaf
#       {out}/{pid}_{YYYYmmdd}_{HHMM}/{condition}/attempt{K}/robot{N}/…
#   OLD (per-robot top-level folders):
#       {out}/{pid}_robot{N}_{YYYYmmdd}_{HHMM}/{condition}/attempt{K}/…
_ROBOT_LEAF_RE = re.compile(r"^robot(?P<n>\d+)$")
_SESSION_ROOT_RE = re.compile(r"^(?P<pid>.+)_robot(?P<n>\d+)_\d{8}_\d{3,6}$")


# --------------------------------------------------------------------------- #
# Small rigid-transform helpers (kept local so this file stands alone)
# --------------------------------------------------------------------------- #
def _make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _inv_T(T):
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def _mat_to_quat_wxyz(R):
    """3x3 rotation -> quaternion [w, x, y, z] (MuJoCo frame convention)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _rpy_to_mat(rpy_deg):
    """Roll-pitch-yaw (degrees, applied X then Y then Z) -> 3x3 rotation. Used for the
    small manual --table-rot / --crate-rot alignment nudges."""
    r, p, y = (np.radians(float(a)) for a in rpy_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# --------------------------------------------------------------------------- #
# Session pairing: given one path, find both robots' attempt folders
# --------------------------------------------------------------------------- #
def _robot_leaf_id(p: Path):
    """Robot id N if p is a 'robot{N}' leaf directory (new layout), else None."""
    m = _ROBOT_LEAF_RE.match(p.name)
    return int(m.group("n")) if m else None


def _has_raw(p: Path) -> bool:
    return p.is_dir() and bool(list(p.glob("robot_raw*.csv")))


def _gather_new_layout(p: Path) -> dict:
    """NEW merged layout: find the attempt dir holding robot{N}/ leaves, return {n: leaf}."""
    attempt = None
    if _robot_leaf_id(p) is not None and _has_raw(p):
        attempt = p.parent                       # given a robot{N}/ leaf
    elif p.is_dir() and any(_robot_leaf_id(s) is not None and _has_raw(s)
                            for s in p.iterdir() if s.is_dir()):
        attempt = p                              # given the attempt dir itself
    elif p.is_dir():
        # given a condition dir / session root — pick the newest robot{N}/ leaf below it
        for csv in sorted(p.rglob("robot_raw*.csv"), key=lambda x: str(x), reverse=True):
            if _robot_leaf_id(csv.parent) is not None:
                attempt = csv.parent.parent
                break
    if attempt is None:
        return {}
    out = {}
    for s in sorted(attempt.iterdir()):
        n = _robot_leaf_id(s)
        if n is not None and _has_raw(s):
            out[n] = s
    return out


def _resolve_rec_dir(path: Path) -> Path:
    """OLD layout: resolve a path to the flat attempt dir holding a robot_raw CSV."""
    p = Path(path)
    if p.is_file():
        return p.parent
    if _has_raw(p):
        return p
    hits = sorted(p.rglob("robot_raw*.csv"), key=lambda x: x.name)
    if hits:
        return hits[-1].parent
    return p


def _split_root(rec_dir: Path):
    """OLD layout: (session_root, pid, robot_n, rel_subpath) for a rec dir, or Nones."""
    for anc in [rec_dir, *rec_dir.parents]:
        m = _SESSION_ROOT_RE.match(anc.name)
        if m:
            return anc, m.group("pid"), int(m.group("n")), rec_dir.relative_to(anc)
    return None, None, None, None


def _gather_old_layout(p: Path) -> dict:
    """OLD layout: separate {pid}_robot{N}_{date}/{cond}/attempt{K} trees, paired by pid
    + the same {condition}/attempt subpath."""
    rec = _resolve_rec_dir(p)
    root, pid, rn, rel = _split_root(rec)
    if rn is None:
        return {}
    out = {rn: rec}
    for d in sorted(root.parent.glob(f"{pid}_robot*")):
        m = _SESSION_ROOT_RE.match(d.name)
        if not m or int(m.group("n")) == rn:
            continue
        cand = d / rel
        if _has_raw(cand):
            out[int(m.group("n"))] = cand
    return out


def _label_for(rec_dir: Path):
    """Best-effort (pid, 'condition/attempt') for logging, from a rec dir of either layout."""
    d = Path(rec_dir)
    attempt = d.parent if _robot_leaf_id(d) is not None else d
    cond = attempt.parent
    root = cond.parent
    mo = _SESSION_ROOT_RE.match(root.name)             # old: {pid}_robotN_{date}
    if mo:
        return mo.group("pid"), f"{cond.name}/{attempt.name}"
    mn = re.match(r"^(?P<pid>.+)_\d{8}_\d{3,6}$", root.name)   # new: {pid}_{date}
    pid = mn.group("pid") if mn else root.name
    return pid, f"{cond.name}/{attempt.name}"


def is_solo(sessions: dict) -> bool:
    """True for a single-operator SOLO take: every robot's meta is marked ``solo`` (written by
    the manager's single-operator Start Recording). Such takes hold one robot and no partner /
    base-localization / crate / video data, so their absence is expected, not an error."""
    return bool(sessions) and all(bool(s.get("meta", {}).get("solo"))
                                  for s in sessions.values())


def gather_sessions(path: str) -> dict:
    """Map robot_id -> session dict (from session_viewer.find_session) for every robot
    whose data is present for this trial/attempt. Handles the merged robot{N}/ layout and
    the legacy per-robot-folder layout, and falls back to a lone robot_raw."""
    p = Path(os.path.abspath(path))
    if p.is_file():
        p = p.parent
    rec_dirs = _gather_new_layout(p)
    if not rec_dirs:
        rec_dirs = _gather_old_layout(p)
    if not rec_dirs:                                   # last resort: any robot_raw below p
        hits = sorted(p.rglob("robot_raw*.csv"), key=lambda x: str(x))
        if hits:
            rec_dirs = {1: hits[-1].parent}
    sessions = {}
    for n, d in sorted(rec_dirs.items()):
        try:
            sessions[n] = sv.find_session(str(d))
        except Exception as exc:                       # noqa: BLE001
            print(f"[twin] robot{n}: skipping {d} ({exc})")
    if not sessions:
        raise FileNotFoundError(f"no robot_raw*.csv found for {path}")
    if is_solo(sessions):
        only = next(iter(sessions))
        print(f"[twin] SOLO take (single-operator recording): robot{only} only — no partner "
              "robot, base localization, crate or video expected.")
    elif len(sessions) == 1:
        only = next(iter(sessions))
        missing = 2 if only == 1 else 1
        attempt = Path(next(iter(rec_dirs.values()))).parent
        present = sorted(d.name for d in attempt.iterdir() if d.is_dir()) \
            if attempt.is_dir() else []
        print(f"[twin] WARNING: only robot{only}'s data found — robot{missing} is "
              f"MISSING from this attempt.\n"
              f"[twin]   attempt dir: {attempt}\n"
              f"[twin]   contains:    {present or '(nothing)'}\n"
              f"[twin]   robot{missing}'s data lands here via the PEER's auto-send at "
              "Stop Session (or its 'Send Data to Master' button). If the peer never "
              "recorded/sent this session, the twin can only show one robot.")
    pid, rel = _label_for(next(iter(rec_dirs.values())))
    return sessions, pid, rel


# --------------------------------------------------------------------------- #
# Base localization: robust pelvis-in-table pose from head_base_*.csv
# --------------------------------------------------------------------------- #
def load_base_pose(rec_dir: str, dm_min=15.0, upright_min=0.7):
    """Return (T_world_base 4x4, n_used) for a robot from its head_base_*.csv, or
    (None, 0) if absent. Averages over the (static) session: median position + a
    sign-aligned mean quaternion, keeping only confident, upright detections."""
    hits = sorted(Path(rec_dir).glob("head_base*.csv"))
    if not hits:
        return None, 0
    # cols: 0=t 1..3=px,py,pz 4..7=qx,qy,qz,qw … 18=decision_margin 19=upright
    try:
        arr = np.genfromtxt(hits[-1], delimiter=",", skip_header=1, dtype=float)
    except Exception:                                # noqa: BLE001
        return None, 0
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0 or arr.shape[1] < 8:
        return None, 0
    P = arr[:, 1:4]
    Q = arr[:, 4:8]                                  # [x, y, z, w]
    ok = np.all(np.isfinite(P), axis=1) & np.all(np.isfinite(Q), axis=1)
    if arr.shape[1] >= 20:
        dm, up = arr[:, 18], arr[:, 19]
        good = ok & (dm >= dm_min) & (np.abs(up) >= upright_min)
        if np.count_nonzero(good) >= 3:
            ok = good                                # else keep all valid rows
    P, Q = P[ok], Q[ok]
    if P.shape[0] == 0:
        return None, 0
    pos = np.median(P, axis=0)
    q0 = Q[0]
    Qa = Q * np.sign(Q @ q0)[:, None]                # resolve q/-q double cover
    qm = Qa.mean(axis=0)
    n = np.linalg.norm(qm)
    qm = qm / n if n > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])
    R = sv._quat_to_mat(qm)                          # xyzw -> 3x3
    return _make_T(R, pos), int(P.shape[0])


# --------------------------------------------------------------------------- #
# Scene composition (two prefixed robots welded at their base poses)
# --------------------------------------------------------------------------- #
def _pelvis_in_childworld(mujoco, model_path):
    """Pelvis pose in the single-robot model's own world frame (captures the built-in
    0.793 m pelvis height). The attach frame is offset by its inverse so the pelvis
    lands exactly on the localized base pose."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = m.body("pelvis").id
    return _make_T(d.xmat[bid].reshape(3, 3).copy(), d.xpos[bid].copy())


def _add_static_mesh(mujoco, parent, name, path, scale, pos, rgba, quat=(1, 0, 0, 0)):
    """Add an STL as a static (welded) visual geom in the worldbody at ``pos`` with
    orientation ``quat`` (wxyz). The STL's own origin maps to ``pos``."""
    parent.add_mesh(name=name, file=os.path.abspath(path), scale=[scale] * 3)
    g = parent.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = name
    g.pos = list(pos)
    g.quat = list(quat)
    g.rgba = rgba
    g.contype = 0          # visual only — never collides with the robots
    g.conaffinity = 0


def build_twin_model(mujoco, model_path, base_T, table_pos, table_half,
                     table_mesh=None, crate_mesh=None, table_scale=1.0,
                     crate_scale=1.0, table_z_down=True,
                     table_offset=(0, 0, 0), table_rot=(0, 0, 0), floor=True,
                     table=True):
    """Compose floor + table + (optional) crate mesh + one prefixed G1 per robot.
    ``table=False`` (solo takes with no crate data) omits the table entirely and puts the
    floor at the robots' own ground level.
    base_T maps robot_id -> T_world_base (or None → unlocalized, placed with a warning).
    If ``table_mesh`` / ``crate_mesh`` (STL paths) are given they replace the box
    placeholders — the STL origin must be at table tag 0 (table) / crate tag 10 (crate).
    ``table_z_down`` matches the rig's world_from_table: the table STL is modelled in the
    tag-0 frame (Z into the table), but the twin world is Z-UP, so the table is flipped
    180° about X to sit right-side-up (must match the --table-z-down used at record time).
    Returns (model, prefixes, frame_T, crate_qadr) where frame_T[id] is the child->world
    attach pose and crate_qadr is the crate free-joint qpos address (None if no crate STL,
    in which case the crate is drawn as a box overlay each frame)."""
    T_cw_pelvis = _pelvis_in_childworld(mujoco, model_path)

    parent = mujoco.MjSpec()
    # NOTE: no floor plane here — z=0 is the TABLE-TOP surface (world origin = table
    # tag 0), so a plane at z=0 slices the scene in half at table height. The real
    # floor is added AFTER the robots are placed, at the height their bases stand on.

    # Table: STL if given (origin at tag 0 → placed at the world origin), else a box slab
    # whose TOP face sits at z=0 (the tag sits on the table surface). The STL is authored
    # in the tag-0 frame (Z into the table); the twin world is Z-up, so flip it 180° about
    # X (world_from_table) when the rig ran with --table-z-down (the default).
    table_pos = np.asarray(table_pos, dtype=np.float64) + np.asarray(table_offset, float)
    if not table:
        pass                                             # solo take: no table in the scene
    elif table_mesh:
        # world-frame nudge: (manual rot) ∘ (Z-down→Z-up flip), about the table origin.
        flip_R = np.diag([1.0, -1.0, -1.0]) if table_z_down else np.eye(3)
        table_quat = _mat_to_quat_wxyz(_rpy_to_mat(table_rot) @ flip_R)
        _add_static_mesh(mujoco, parent, "table_mesh", table_mesh, table_scale,
                         list(table_pos), [0.55, 0.42, 0.30, 1.0], quat=table_quat)
        print(f"[twin] table STL: {Path(table_mesh).name} (origin = table tag 0"
              f"{', flipped to Z-up' if table_z_down else ''}"
              f"{', nudged' if np.any(table_offset) or np.any(table_rot) else ''})")
    else:
        parent.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=list(table_half),
            pos=[table_pos[0], table_pos[1], table_pos[2] - table_half[2]],
            rgba=[0.55, 0.42, 0.30, 1.0])

    # Crate: STL on a FREE-JOINT body (its qpos is set to the tracked pose each frame). The
    # STL origin must be at crate tag 10 = the bundle anchor = the fused crate frame, so
    # the tracked pose places it correctly. Explicit inertial + visual-only geom keep the
    # compile independent of the mesh being watertight (we never step physics).
    crate_qadr = None
    if crate_mesh:
        parent.add_mesh(name="crate_mesh", file=os.path.abspath(crate_mesh),
                        scale=[crate_scale] * 3)
        cb = parent.worldbody.add_body(name="twin_crate")
        cb.pos = [float(table_pos[0]), float(table_pos[1]), float(table_pos[2]) + 0.15]
        try:
            cb.mass = 0.5
            cb.inertia = [1e-3, 1e-3, 1e-3]
            cb.explicitinertial = True
        except Exception:                                # noqa: BLE001
            pass
        cb.add_freejoint()
        cg = cb.add_geom()
        cg.type = mujoco.mjtGeom.mjGEOM_MESH
        cg.meshname = "crate_mesh"
        cg.rgba = [0.95, 0.6, 0.1, 0.85]
        cg.contype = 0
        cg.conaffinity = 0
        # The crate NUDGE (--crate-offset/--crate-rot, and live hotkeys) is applied to the
        # crate BODY pose each frame in run() — NOT to the geom — because a free-joint
        # body's geom_pos is cached at compile time and can't be nudged live.
        print(f"[twin] crate STL: {Path(crate_mesh).name} (origin = crate tag 10)")

    prefixes, frame_T = {}, {}
    fallback_i = 0
    for n in sorted(base_T):
        Twb = base_T[n]
        if Twb is None:
            # Unlocalized: sit at default pelvis height, offset so robots don't overlap.
            Twb = _make_T(np.eye(3), [fallback_i * 1.4, 0.0, float(T_cw_pelvis[2, 3])])
            fallback_i += 1
            if table:
                print(f"[twin] robot{n}: NO base localization — placing at a default "
                      f"offset (its table placement is NOT reconstructed).")
            else:
                print(f"[twin] robot{n}: no table/base data — robot placed at the world "
                      f"origin (pos={np.round(Twb[:3, 3], 3).tolist()}).")
        Tframe = Twb @ _inv_T(T_cw_pelvis)           # pelvis lands on Twb
        child = mujoco.MjSpec.from_file(model_path)
        # The robot XML carries its own INFINITE ground plane ('floor'). Attached under
        # the robot's (rotated/offset) base frame it becomes a huge stray plane slicing
        # through the twin scene — hide every plane the child brings along.
        for g in child.worldbody.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
                g.rgba = [0, 0, 0, 0]
        fr = parent.worldbody.add_frame()
        fr.pos = Tframe[:3, 3].tolist()
        fr.quat = _mat_to_quat_wxyz(Tframe[:3, :3]).tolist()
        parent.attach(child, prefix=f"r{n}_", frame=fr)
        prefixes[n] = f"r{n}_"
        frame_T[n] = Tframe

    # ONE floor, at the height the robots actually stand on: each attach frame's z is
    # where that robot's local ground (z=0 in its model) sits in the twin world — use
    # the lowest. Falls back to 75 cm below the table when nothing is localized.
    if floor:
        floor_z = min([T[2, 3] for T in frame_T.values()] or [-0.75])
        if abs(floor_z) < 1e-6 and table:
            floor_z = -0.75              # unlocalized fallback frames sit at z≈0
        parent.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6.0, 6.0, 0.1],
            pos=[0.0, 0.0, float(floor_z)], rgba=[0.28, 0.30, 0.33, 1.0])
        print(f"[twin] floor at z={floor_z:+.3f} m (table top = z=0)")

    model = parent.compile()
    if crate_mesh:
        bid = int(model.body("twin_crate").id)
        crate_qadr = int(model.jnt_qposadr[model.body_jntadr[bid]])
    return model, prefixes, frame_T, crate_qadr


# --------------------------------------------------------------------------- #
# Crate track clean-up (playback-side): smooth + freeze-when-still + rest-on-table
# --------------------------------------------------------------------------- #
def _read_stl_min_z(path, scale=1.0):
    """Lowest vertex z of an STL in its own (crate/tag-10) frame — i.e. where the
    crate's base plane sits below its origin. Binary STL fast path, ascii fallback."""
    try:
        with open(path, "rb") as f:
            head = f.read(84)
            n = int.from_bytes(head[80:84], "little")
            data = np.fromfile(f, dtype=np.uint8, count=n * 50)
        if n == 0 or data.size < n * 50:
            raise ValueError("not binary")
        rec = data.reshape(n, 50)[:, 12:48].copy()          # 3 verts × 3 f32 per tri
        tri = rec.view("<f4").reshape(n, 9)
        return float(min(tri[:, 2].min(), tri[:, 5].min(), tri[:, 8].min())) * scale
    except Exception:                                        # noqa: BLE001
        try:
            zs = [float(t[3]) for t in (l.split() for l in open(path))
                  if len(t) == 4 and t[0] == "vertex"]
            return (min(zs) * scale) if zs else None
        except Exception:                                    # noqa: BLE001
            return None


def _med_filt(a, k):
    """Sliding median over rows of (n, d) — kills single-frame fusion spikes."""
    if a.shape[0] < 3 or k < 3:
        return a.copy()
    k |= 1
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.median(ap[i:i + k], axis=0) for i in range(a.shape[0])])


def _boxcar(a, k):
    """Zero-phase moving average over rows of (n, d)."""
    if a.shape[0] < 2 or k <= 1:
        return a.copy()
    k |= 1
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    S = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(ap, axis=0)])
    return (S[k:] - S[:-k]) / k


def smooth_crate_track(csrc, mesh_min_z=None, z_bias=None, smooth_s=0.35,
                       still_v=0.006, still_w=2.0, still_min_s=0.4, gap_s=0.5):
    """Offline clean-up of the fused crate track for PLAYBACK (in place on CrateSource).

    The live tracker fuses every frame INDEPENDENTLY, so the pose jitters as tags/
    cameras drop in and out, and camera-anchor bias leaves it hovering. Playback can do
    better — the whole track is known:
      1. quaternion sign-alignment, then a sliding MEDIAN (single-frame fusion spikes)
         + zero-phase moving average over ``smooth_s`` (no lag: symmetric window);
      2. STATIONARY FREEZE: where velocity < ``still_v`` m/s and rotation rate
         < ``still_w`` °/s for ≥ ``still_min_s``, the pose is pinned to the segment's
         median — a crate nobody is touching is perfectly still;
      3. REST-ON-TABLE: with the crate STL's base height (``mesh_min_z``), the LOWEST
         upright stationary segment must be sitting ON the table (z=0) — any constant
         offset there is anchor bias, subtracted from the whole track (auto; pass
         ``z_bias`` to force a value, 0 to disable). Capped at ±8 cm.
    Tracking gaps > ``gap_s`` split the track so nothing smooths across an occlusion.
    Returns a dict of what was done (for logging)."""
    t, p, q = csrc._t, csrc._p, csrc._q
    rep = {"n": int(t.size), "runs": 0, "z_bias": 0.0}
    if t.size < 5:
        return rep
    for i in range(1, q.shape[0]):                 # hemisphere-align for averaging
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    chunks = np.split(np.arange(t.size), np.where(np.diff(t) > gap_s)[0] + 1)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / 15.0
    win = max(1, int(round(smooth_s / max(dt, 1e-3))))
    rest_base_z = []                                # (base_z, n) per upright still run
    for idx in chunks:
        if idx.size < 3:
            continue
        sl = slice(int(idx[0]), int(idx[-1]) + 1)
        tt = t[sl]
        ps = _boxcar(_med_filt(p[sl], 5), win)
        qs = _boxcar(q[sl], win)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        m = ps.shape[0]
        # stationary detection on the smoothed track
        v = np.zeros(m)
        wdeg = np.zeros(m)
        dts = np.maximum(np.diff(tt), 1e-3)
        v[1:] = np.linalg.norm(np.diff(ps, axis=0), axis=1) / dts
        dq = np.abs(np.sum(qs[1:] * qs[:-1], axis=1)).clip(-1.0, 1.0)
        wdeg[1:] = 2.0 * np.degrees(np.arccos(dq)) / dts
        still = (v < still_v) & (wdeg < still_w)
        i = 0
        while i < m:
            if not still[i]:
                i += 1
                continue
            j = i
            while j + 1 < m and still[j + 1]:
                j += 1
            if tt[j] - tt[i] >= still_min_s:
                ps[i:j + 1] = np.median(ps[i:j + 1], axis=0)
                qm = qs[i:j + 1].mean(axis=0)
                qs[i:j + 1] = qm / max(np.linalg.norm(qm), 1e-9)
                rep["runs"] += 1
                R = sv._quat_to_mat(qs[i])
                if mesh_min_z is not None and R[2, 2] > 0.95:   # upright → resting?
                    base_z = ps[i, 2] + (R @ [0.0, 0.0, mesh_min_z])[2]
                    rest_base_z.append((float(base_z), j - i + 1))
            i = j + 1
        p[sl] = ps
        q[sl] = qs
    # Constant z-bias: the lowest upright still segment sits ON the table.
    if z_bias is None and rest_base_z:
        z_bias = min(bz for bz, _n in rest_base_z)
    if z_bias:
        z_bias = float(np.clip(z_bias, -0.08, 0.08))
        p[:, 2] -= z_bias
        rep["z_bias"] = z_bias
    return rep


# --------------------------------------------------------------------------- #
# Per-robot playback state
# --------------------------------------------------------------------------- #
def _parse_pin_specs(specs):
    """Parse --pin-joint 'RID:joint_name=radians' entries into {rid: {joint_name: value}}.
    Used to hold a broken/sheared actuator at a fixed angle in playback so the twin matches the
    real robot. Raises ValueError on a malformed entry."""
    pins = {}
    for spec in (specs or []):
        try:
            rid_s, rest = spec.split(":", 1)
            jname, val = rest.split("=", 1)
            pins.setdefault(int(rid_s), {})[jname.strip()] = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"bad --pin-joint '{spec}' — expected RID:joint_name=radians")
    return pins


class RobotTrack:
    """Everything needed to pose + annotate one robot in the shared scene."""

    def __init__(self, mujoco, rid, session, model, prefix, frame_T, model_path, pins=None):
        self.rid = rid
        self.prefix = prefix
        self.frame_T = frame_T                       # child-world -> shared world
        self.R_frame = frame_T[:3, :3]
        self.cap = session["cap"]
        self.meta = session["meta"]
        # Tactile palm/fingertip force stream ({"t","v","layout"} or None), rendered as
        # the sensor-point overlay in the shared world (see the overlay setup in run()).
        self.tactile = session.get("tactile")
        self.t_epoch = self.cap.get("t_epoch", self.cap["t"])
        self.n = int(self.t_epoch.size)

        def addrs(names):
            return [int(model.joint(prefix + nm).qposadr[0]) for nm in names]

        seg_names = {"legs": we.LEG_JOINTS, "torso": we.TORSO_JOINTS,
                     "left": we.LEFT_ARM_JOINTS, "right": we.RIGHT_ARM_JOINTS}
        self.seg_addr = {}
        for seg, names in seg_names.items():
            if seg in self.cap and "q" in self.cap[seg]:
                try:
                    self.seg_addr[seg] = addrs(names)
                except Exception:                    # noqa: BLE001
                    pass
        self.has_fingers = "finger_left" in self.cap and "finger_right" in self.cap
        if self.has_fingers:
            self.finger_addr = {
                "finger_left": addrs(we.LEFT_HAND_JOINTS),
                "finger_right": addrs(we.RIGHT_HAND_JOINTS),
            }
        # Fixed-angle joint pins for THIS robot: override the recorded value with a constant so a
        # broken/sheared actuator is shown at the position it's physically stuck at (e.g. a dead
        # LEFT thumb abduction/adduction → pin left_thumb_1_joint). Resolved joint name → qpos
        # address ONCE here; applied last in pose_into so it wins over the recorded pose.
        self.pin_overrides = []
        for jname, val in (pins or {}).items():
            try:
                self.pin_overrides.append(
                    (int(model.joint(prefix + jname).qposadr[0]), float(val)))
            except Exception:                        # noqa: BLE001
                print(f"[twin] robot{rid}: cannot pin unknown joint '{jname}' — skipped")
        # Wrench estimator (own single-robot model; points come out in child-world).
        self.wrench = sv.WrenchSource(self.cap, self.meta, model_path)
        try:
            self.pelvis_bid = int(model.body(prefix + "pelvis").id)
        except Exception:                            # noqa: BLE001
            self.pelvis_bid = None

    def index_at(self, epoch, eps=0.25):
        """Frame index nearest absolute ``epoch`` and whether it's within range."""
        te = self.t_epoch
        k = int(np.searchsorted(te, epoch, side="right")) - 1
        k = min(max(k, 0), self.n - 1)
        in_range = (te[0] - eps) <= epoch <= (te[-1] + eps)
        return k, in_range

    def pose_into(self, data, model_qpos0, k):
        for seg, ad in self.seg_addr.items():
            q = self.cap[seg]["q"][k]
            for i, a in enumerate(ad):
                if i < len(q):
                    data.qpos[a] = q[i]
        if self.has_fingers:
            for key, ad in self.finger_addr.items():
                q = self.cap[key][k]
                for i, a in enumerate(ad):
                    if i < len(q):
                        data.qpos[a] = q[i]
        # Fixed pins LAST, so a broken actuator held at its stuck angle overrides the recorded
        # (commanded) motion and the twin matches the real robot.
        for a, v in self.pin_overrides:
            data.qpos[a] = v

    def to_world(self, p_child):
        return (self.frame_T @ np.append(np.asarray(p_child, float), 1.0))[:3]


# --------------------------------------------------------------------------- #
# Video: stack each present robot's ZED LEFT|RIGHT pair, synced by epoch
# --------------------------------------------------------------------------- #
class MultiVideo:
    def __init__(self, sessions):
        self.pairs = {}
        if cv2 is None:
            self.available = False
            return
        for rid, s in sorted(sessions.items()):
            vp = sv.VideoPair(s["left_video"], s["right_video"],
                              s["meta"].get("zed_record_start_epoch"),
                              stereo_path=s.get("stereo_video"))
            if vp.available:
                self.pairs[rid] = vp
        self.available = len(self.pairs) > 0

    def frame_at(self, epoch):
        if not self.available:
            return None
        rows = []
        for rid, vp in self.pairs.items():
            comp = vp.frame_at(epoch)
            if comp is None:
                continue
            bar = np.zeros((26, comp.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, f"ROBOT {rid}", (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (80, 200, 255), 2)
            rows.append(np.vstack([bar, comp]))
        if not rows:
            return None
        w = max(r.shape[1] for r in rows)
        rows = [r if r.shape[1] == w else
                np.hstack([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)])
                for r in rows]
        return np.vstack(rows)

    def release(self):
        for vp in self.pairs.values():
            vp.release()


# --------------------------------------------------------------------------- #
# Trial discovery: open a PARENT folder and tab across conditions + attempts
# --------------------------------------------------------------------------- #
_TRIAL_CONDITION_ORDER = ["V", "A+V", "H+V", "H+A+V"]


def _cond_sort_key(name):
    return (_TRIAL_CONDITION_ORDER.index(name)
            if name in _TRIAL_CONDITION_ORDER else len(_TRIAL_CONDITION_ORDER), name)


def _is_attempt_dir(d) -> bool:
    """True only for an ATTEMPT dir: one that directly holds robot{N}/ leaves (new layout)
    or a robot_raw*.csv (old flat layout). NOT true for a condition/parent dir — those only
    reach robot data via rglob, which _gather_new_layout would falsely accept."""
    try:
        d = Path(d)
        if not d.is_dir():
            return False
        if any(_robot_leaf_id(s) is not None and _has_raw(s)
               for s in d.iterdir() if s.is_dir()):
            return True
        return bool(_has_raw(d))
    except Exception:                                     # noqa: BLE001
        return False


def _discover_selections(path):
    """Return an ordered list of (condition_label, attempt_dir) under a TRIAL PARENT.

    Accepts the parent (whose children are condition folders V / A+V / H+V / H+A+V), a single
    condition dir (children are attempts), or one attempt dir. Conditions are ordered by the
    study order; attempts by folder name."""
    p = Path(os.path.abspath(path))
    if p.is_file():
        p = p.parent
    if _is_attempt_dir(p):                                # a single attempt dir
        return [(p.parent.name if p.parent else p.name, p)]
    if p.is_dir():                                        # a condition dir (children = attempts)
        direct = sorted((a for a in p.iterdir() if _is_attempt_dir(a)), key=lambda a: a.name)
        if direct:
            return [(p.name, a) for a in direct]
    pairs = []                                            # the parent (children = conditions)
    if p.is_dir():
        conds = sorted((d for d in p.iterdir() if d.is_dir()),
                       key=lambda d: _cond_sort_key(d.name))
        for cd in conds:
            for a in sorted((a for a in cd.iterdir() if _is_attempt_dir(a)),
                            key=lambda a: a.name):
                pairs.append((cd.name, a))
    if not pairs and p.is_dir() and list(p.glob("robot_raw*.csv")):
        pairs.append((p.parent.name if p.parent else p.name, p))
    return pairs


def _best_bases(selections, dm_min=15.0, upright_min=0.7):
    """Best-localized base pose per robot across a list of {'condition','attempt','sessions'}
    selections. Symmetric across robots (master AND peer): whichever attempt localized a given
    robot best (most confident head-cam frames) supplies that robot's base. Returns
    {rid: (T_world_base, n_used, 'condition/attempt')}."""
    best = {}
    for sel in selections:
        for rid, s in sel["sessions"].items():
            T, n = load_base_pose(s["session_dir"], dm_min, upright_min)
            if T is None:
                continue
            if rid not in best or n > best[rid][1]:
                best[rid] = (T, n, f"{sel['condition']}/{sel['attempt']}")
    return best


def gather_bases_from(path, dm_min=15.0, upright_min=0.7):
    """Best base pose per robot from ANY path — a whole different SESSION, a trial, a condition,
    or a single attempt. Used to borrow a base when the current session failed to localize a
    robot in EVERY trial (so there's nothing in-session to reuse). The robot must not have moved
    between the two sessions. Returns ({rid: T_world_base}, {rid: 'condition/attempt'})."""
    sels = []
    for cond, adir in _discover_selections(path):
        try:
            sess, _pid, _rel = gather_sessions(str(adir))
        except Exception:                                    # noqa: BLE001
            continue
        sels.append({"condition": cond, "attempt": Path(adir).name, "sessions": sess})
    best = _best_bases(sels, dm_min, upright_min)
    return ({rid: v[0] for rid, v in best.items()},
            {rid: v[2] for rid, v in best.items()})


def gather_trial(path, dm_min=15.0, upright_min=0.7):
    """Discover every (condition, attempt) under a trial parent, gather each attempt's robot
    sessions, and pick a CANONICAL base pose per robot (the best-localized one across the whole
    trial). The robot doesn't move between attempts, so an attempt whose base wasn't localized
    reuses a good one instead of floating above the table.

    Returns (selections, canonical_base, base_source, pid) where selections is an ordered list
    of {'condition','attempt','dir','sessions'}, canonical_base is {rid: T_world_base}, and
    base_source is {rid: 'condition/attempt'} naming where each canonical base came from."""
    pairs = _discover_selections(path)
    if not pairs:
        raise FileNotFoundError(f"no attempts with robot data found under {path}")
    selections, pid_any = [], None
    for cond, adir in pairs:
        try:
            sess, pid, _rel = gather_sessions(str(adir))
        except Exception as exc:                          # noqa: BLE001
            print(f"[twin] skip {cond}/{Path(adir).name}: {exc}")
            continue
        pid_any = pid_any or pid
        selections.append({"condition": cond, "attempt": Path(adir).name,
                           "dir": Path(adir), "sessions": sess})
    if not selections:
        raise FileNotFoundError(f"no readable attempts under {path}")

    best = _best_bases(selections, dm_min, upright_min)   # rid -> (T, n_used, 'cond/att')
    canonical = {rid: v[0] for rid, v in best.items()}
    source = {rid: v[2] for rid, v in best.items()}

    print(f"[twin] trial: {len(selections)} attempt(s) across "
          f"{len(set(s['condition'] for s in selections))} condition(s)")
    for rid in sorted(set(r for sel in selections for r in sel["sessions"])):
        if rid in canonical:
            print(f"[twin] robot{rid}: canonical base from {source[rid]}  "
                  f"pos={np.round(canonical[rid][:3, 3], 3).tolist()}")
        elif all(is_solo(sel["sessions"]) for sel in selections if rid in sel["sessions"]):
            print(f"[twin] robot{rid}: solo take(s) — no base localization recorded; the "
                  "robot is shown at the world origin.")
        else:
            print(f"[twin] robot{rid}: NO base localized in ANY attempt — it will float "
                  "(no head_base_*.csv anywhere).")
    return selections, canonical, source, pid_any


def _menu_layout(conditions, per_cond, cols, tw, th,
                 top=52, head_h=28, m=18, foot=40):
    """Pure geometry for the thumbnail menu (no cv2). Returns
    (rects, headers, (W, H)) where rects maps a selection index -> the tile's BORDER
    rectangle (x0, y0, x1, y1) in canvas pixels, and headers is [(condition, text_y)].
    The thumbnail itself sits 2 px inside each rect. Kept separate from the drawing so the
    click hit-boxes are exactly the drawn tiles and the layout is unit-testable — the whole
    point of the menu is that clicks land where the tiles are."""
    cell_w, cell_h = tw + 16, th + 44
    rects, headers = {}, []
    y = top
    for c in conditions:
        headers.append((c, y + 21))
        y += head_h
        idxs = per_cond[c]
        for j, idx in enumerate(idxs):
            r, cc = divmod(j, cols)
            x0 = m + cc * cell_w + 8
            y0 = y + r * cell_h + 6
            rects[idx] = (x0 - 2, y0 - 2, x0 + tw + 2, y0 + th + 2)
        y += (-(-len(idxs) // cols)) * cell_h            # ceil-div rows for this condition
    W = max(m * 2 + cols * cell_w, 560)
    H = y + foot
    return rects, headers, (W, H)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(selections, canonical_base, base_source, args, external_base=None):
    import dataclasses
    import time

    import mujoco
    import mujoco.viewer
    from inspire_hand_vis_utils import (
        build_mujoco_sensor_overlay_values_from_snapshot,
        create_mujoco_sensor_overlay_runtime,
        prepare_live_scalar_arrow_values,
        render_mujoco_sensor_overlay,
    )

    model_path = args.model or we.DEFAULT_MODEL_PATH
    table_pos = np.asarray(args.table_pos, dtype=np.float64)
    table_half = np.asarray(args.table_size, dtype=np.float64) * 0.5
    table_mesh = _resolve_asset(args.table_stl, "table.stl")
    crate_mesh = _resolve_asset(args.crate_stl, "crate.stl")
    crate_half = np.asarray(args.crate_size, dtype=np.float64) * 0.5
    fs = args.force_scale
    all_rids = sorted({rid for sel in selections for rid in sel["sessions"]})

    # Ordered unique conditions + per-condition attempt numbers (for the menu labels).
    conditions = []
    for sel in selections:
        if sel["condition"] not in conditions:
            conditions.append(sel["condition"])
    attempt_no, _seen = [], {}
    for sel in selections:
        _seen[sel["condition"]] = _seen.get(sel["condition"], 0) + 1
        attempt_no.append(_seen[sel["condition"]])

    # Which robots localized their OWN base in each attempt (the rest float unless a canonical/
    # borrowed base fills in) — surfaced as a marker on the menu tile.
    base_ok = []
    for sel in selections:
        ok = set()
        for rid, s in sel["sessions"].items():
            T, _n = load_base_pose(s["session_dir"], args.dm_min, args.upright_min)
            if T is not None:
                ok.add(rid)
        base_ok.append(ok)

    menu_win = "digital_twin_menu"
    play_win = "digital_twin"

    # Menu geometry. The menu window is WINDOW_AUTOSIZE (shown 1:1 so mouse clicks map exactly
    # to pixels — the whole reason the old scaled tab bar's clicks missed). Because AUTOSIZE
    # windows can't be resized, size the tiles so the WHOLE grid fits a target box (else tiles
    # would spill off-screen, unreachable by click). Grab thumbnails at that computed size.
    COLS_MAX = 5
    MENU_MAXW, MENU_MAXH = 1500, 980
    MENU_M, MENU_TOP, MENU_FOOT, MENU_HEAD = 18, 52, 40, 28
    _per_cond = {c: [i for i, s in enumerate(selections) if s["condition"] == c]
                 for c in conditions}
    MENU_COLS = max(1, min(COLS_MAX, max((len(v) for v in _per_cond.values()), default=1)))
    _rows_total = sum(-(-len(_per_cond[c]) // MENU_COLS) for c in conditions) or 1
    _tw_w = (MENU_MAXW - 2 * MENU_M) // MENU_COLS - 16
    _avail_h = MENU_MAXH - MENU_TOP - MENU_FOOT - len(conditions) * MENU_HEAD
    _th_h = max(40, _avail_h // _rows_total - 44)         # height budget per tile → TH
    TW = int(max(120, min(240, _tw_w, _th_h * 16 // 9)))  # keep ~16:9, clamp to a sane range
    TH = int(max(70, TW * 9 // 16))

    # ── Menu thumbnails: first readable ZED frame per attempt (robot1 preferred) ──
    def _grab_thumb(sessions):
        if cv2 is None:
            return None
        for rid in sorted(sessions):
            s = sessions[rid]
            for key, crop_left in (("left_video", False),
                                   ("stereo_video", True),
                                   ("right_video", False)):
                path = s.get(key)
                if not path or not os.path.exists(path):
                    continue
                cap = cv2.VideoCapture(path)
                fr = None
                for _ in range(8):                       # a few frames in (skip black intro)
                    ok, f = cap.read()
                    if not ok or f is None:
                        break
                    fr = f
                cap.release()
                if fr is not None:
                    if crop_left:
                        fr = fr[:, :fr.shape[1] // 2]
                    try:
                        return cv2.resize(fr, (TW, TH))
                    except Exception:                     # noqa: BLE001
                        return None
        return None

    thumbs = []
    if cv2 is not None:
        print(f"[twin] building {len(selections)} menu thumbnail(s)…")
        for sel in selections:
            thumbs.append(_grab_thumb(sel["sessions"]))

    # Persistent state shared with the menu mouse callback.
    menu = {"click": None, "hits": [], "sel": 0}
    base_override = {}                                    # rid -> T (overrides canonical)
    external_base = external_base or {}                   # rid -> T from --base-from (other session)

    def _menu_mouse(event, x, y, flags, _param):
        if cv2 is None or event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x0, y0, x1, y1, idx) in menu["hits"]:
            if x0 <= x <= x1 and y0 <= y <= y1:
                menu["click"] = idx
                return

    def _mmss(secs):
        m, s = divmod(int(max(0.0, secs)), 60)
        return f"{m}:{s:02d}"

    # ── Menu canvas: thumbnails grouped by condition, current pick highlighted ──
    def _build_menu(selected):
        rects, headers, (W, H) = _menu_layout(
            conditions, _per_cond, MENU_COLS, TW, TH,
            top=MENU_TOP, head_h=MENU_HEAD, m=MENU_M, foot=MENU_FOOT)
        canvas = np.full((H, W, 3), 24, dtype=np.uint8)
        cv2.putText(canvas, "Digital Twin - pick a trial / attempt to play back",
                    (MENU_M, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        for c, hy in headers:
            cv2.putText(canvas, f"{c}", (MENU_M, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (120, 210, 255), 2)
        hits = []
        for idx, (x0, y0, x1, y1) in rects.items():
            tx0, ty0 = x0 + 2, y0 + 2                     # thumbnail sits 2 px inside the rect
            th = thumbs[idx] if idx < len(thumbs) else None
            if th is not None:
                canvas[ty0:ty0 + TH, tx0:tx0 + TW] = th
            else:
                cv2.rectangle(canvas, (tx0, ty0), (tx0 + TW, ty0 + TH), (50, 50, 50), -1)
                cv2.putText(canvas, "no video", (tx0 + TW // 4, ty0 + TH // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
            active = (idx == selected)
            cv2.rectangle(canvas, (x0, y0), (x1, y1),
                          (90, 230, 90) if active else (110, 110, 110),
                          3 if active else 1)
            label = f"A{attempt_no[idx]}  {selections[idx]['attempt']}"
            if len(label) > 26:
                label = label[:25] + "…"
            cv2.putText(canvas, label, (tx0, ty0 + TH + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255) if active else (200, 200, 200), 1)
            missing = [r for r in all_rids if r in selections[idx]["sessions"]
                       and r not in base_ok[idx]
                       and not is_solo(selections[idx]["sessions"])]
            if missing:
                cv2.putText(canvas, f"base borrowed (r{','.join(map(str, missing))})",
                            (tx0, ty0 + TH + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (60, 170, 255), 1)
            hits.append((x0, y0, x1, y1, idx))
        cv2.putText(canvas, "click a tile   |   a/d = move   Enter = play   q = quit",
                    (MENU_M, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)
        menu["hits"] = hits
        return canvas

    def show_menu():
        """Return the selected selection-index, or None to quit."""
        if cv2 is None:
            return 0 if selections else None
        cv2.namedWindow(menu_win, cv2.WINDOW_AUTOSIZE)    # AUTOSIZE → 1:1, clicks map exactly
        try:
            cv2.setWindowTitle(menu_win, "Digital Twin - trial menu")
        except Exception:                                 # noqa: BLE001
            pass
        cv2.setMouseCallback(menu_win, _menu_mouse)
        menu["click"] = None
        sel = min(menu["sel"], len(selections) - 1)
        while True:
            cv2.imshow(menu_win, _build_menu(sel))
            key = cv2.waitKey(20) & 0xFF
            if menu["click"] is not None:
                sel = menu["click"]
                menu["click"] = None
                menu["sel"] = sel
                return sel
            if key in (ord('q'), 27):
                return None
            elif key in (ord('d'), ord('n')):
                sel = (sel + 1) % len(selections)
            elif key in (ord('a'), ord('p')):
                sel = (sel - 1) % len(selections)
            elif key in (13, 10, ord(' ')):
                menu["sel"] = sel
                return sel
            elif ord('1') <= key <= ord('9') and (key - ord('1')) < len(selections):
                menu["sel"] = key - ord('1')
                return menu["sel"]
            if cv2.getWindowProperty(menu_win, cv2.WND_PROP_VISIBLE) < 1:
                return None

    def _load_tactile(tracks, model):
        tac_overlay, tac_by_robot = None, {}
        if not any(getattr(tr, "tactile", None) for tr in tracks.values()):
            return tac_overlay, tac_by_robot
        try:
            tac_overlay = create_mujoco_sensor_overlay_runtime(
                model, sensor_direction_overrides=sv._sensor_direction_overrides())
            tac_overlay["running_max"] = 0.0
            tac_overlay["smoothed_values"] = None
            all_pts = tac_overlay["sensor_points"]
            for rid, tr in tracks.items():
                if not getattr(tr, "tactile", None):
                    continue
                pfx = tr.prefix
                idx, stripped = [], []
                for i, pt in enumerate(all_pts):
                    if pt.mesh_name.startswith(pfx):
                        idx.append(i)
                        stripped.append(dataclasses.replace(
                            pt, mesh_name=pt.mesh_name[len(pfx):]))
                if idx:
                    tac_by_robot[rid] = (np.asarray(idx, dtype=int), stripped)
        except Exception as exc:                          # noqa: BLE001
            print(f"[twin] tactile overlay unavailable: {exc}")
            tac_overlay, tac_by_robot = None, {}
        return tac_overlay, tac_by_robot

    # ── Play ONE attempt (with its own base-fix rebuild loop). Returns 'menu' or 'quit'. ──
    def play_attempt(idx):
        sel = selections[idx]
        sessions = sel["sessions"]
        SEEK_MAX = 1000
        seek = {"req": None, "prog": False, "last": -1, "ok": False}

        def _on_seek(val):
            if not seek["prog"]:
                seek["req"] = int(val)

        if cv2 is not None:
            cv2.namedWindow(play_win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            try:
                cv2.setWindowTitle(play_win, f"Digital Twin - {sel['condition']} / "
                                             f"{sel['attempt']}  (m/ESC=menu  q=quit)")
            except Exception:                             # noqa: BLE001
                pass
            try:
                cv2.resizeWindow(play_win, 1600, 950)
            except Exception:                             # noqa: BLE001
                pass
            try:
                cv2.createTrackbar("seek", play_win, 0, SEEK_MAX, _on_seek)
                seek["ok"] = True
            except Exception as exc:                      # noqa: BLE001
                print(f"[twin] seek bar unavailable ({exc}) — a/d/A/D scrubbing still works.")

        # SOLO take (single-operator recording): one robot, no table/crate unless this take
        # actually holds crate data. Study attempts keep the full table + crate scene.
        solo = is_solo(sessions)
        solo_scene = solo and not any(s.get("crate_csv") for s in sessions.values())
        if solo:
            print("[twin] solo take — " + ("no crate data, so the table + crate are omitted; "
                                           "the robot stands at the world origin."
                                           if solo_scene else
                                           "crate data present, so the table + crate are shown."))

        t_mesh = None if solo_scene else table_mesh
        c_mesh = None if solo_scene else crate_mesh

        while True:                                       # base-fix rebuild loop
            base_T = {rid: (base_override[rid] if rid in base_override
                            else canonical_base.get(rid)) for rid in all_rids}
            model, prefixes, frame_T, crate_qadr = build_twin_model(
                mujoco, model_path, base_T, table_pos, table_half,
                table_mesh=t_mesh, crate_mesh=c_mesh,
                table_scale=args.table_stl_scale, crate_scale=args.crate_stl_scale,
                table_z_down=args.table_z_down,
                table_offset=args.table_offset, table_rot=args.table_rot,
                floor=not args.no_floor, table=not solo_scene)
            data = mujoco.MjData(model)

            def _mesh_geom_ids():
                tg = cg = None
                for gi in range(model.ngeom):
                    if model.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
                        continue
                    bid = int(model.geom_bodyid[gi])
                    if bid == 0 and t_mesh:
                        tg = gi
                    elif c_mesh and model.body(bid).name == "twin_crate":
                        cg = gi
                return tg, cg

            nudge_gid = dict(zip(("table", "crate"), _mesh_geom_ids()))
            nudge = {"table": {"off": np.asarray(args.table_offset, float).copy(),
                               "rpy": np.asarray(args.table_rot, float).copy()},
                     "crate": {"off": np.asarray(args.crate_offset, float).copy(),
                               "rpy": np.asarray(args.crate_rot, float).copy()}}
            nudge_sel = {"t": "crate" if nudge_gid["crate"] is not None else "table"}
            nudge_step = {"pos": 0.001, "rot": 0.5}
            table_flip_R = np.diag([1.0, -1.0, -1.0]) if args.table_z_down else np.eye(3)

            def _apply_nudge(target):
                if target != "table":
                    return
                gid = nudge_gid["table"]
                if gid is None:
                    return
                st = nudge["table"]
                model.geom_pos[gid] = np.asarray(table_pos, float) + st["off"]
                model.geom_quat[gid] = _mat_to_quat_wxyz(
                    _rpy_to_mat(st["rpy"]) @ table_flip_R)

            def _nudge_key(key):
                tgt = nudge_sel["t"]
                st = nudge[tgt]
                ps, rs = nudge_step["pos"], nudge_step["rot"]
                if key == ord('1'):
                    nudge_sel["t"] = "table"
                elif key == ord('2'):
                    nudge_sel["t"] = "crate"
                elif key == ord('h'): st["off"][0] -= ps; _apply_nudge(tgt)
                elif key == ord('l'): st["off"][0] += ps; _apply_nudge(tgt)
                elif key == ord('j'): st["off"][1] -= ps; _apply_nudge(tgt)
                elif key == ord('k'): st["off"][1] += ps; _apply_nudge(tgt)
                elif key == ord('n'): st["off"][2] -= ps; _apply_nudge(tgt)
                elif key == ord('m'): st["off"][2] += ps; _apply_nudge(tgt)
                elif key == ord('['): st["rpy"][2] -= rs; _apply_nudge(tgt)
                elif key == ord(']'): st["rpy"][2] += rs; _apply_nudge(tgt)
                elif key == ord('-'):
                    nudge_step["pos"] = max(0.0001, nudge_step["pos"] / 2)
                    nudge_step["rot"] = max(0.05, nudge_step["rot"] / 2)
                elif key == ord('='):
                    nudge_step["pos"] = min(0.05, nudge_step["pos"] * 2)
                    nudge_step["rot"] = min(10.0, nudge_step["rot"] * 2)
                elif key == ord('0'):
                    st["off"][:] = 0.0; st["rpy"][:] = 0.0; _apply_nudge(tgt)
                elif key == ord('p'):
                    n = nudge
                    print("[twin] nudges → "
                          f"--table-offset {n['table']['off'][0]:.4f} "
                          f"{n['table']['off'][1]:.4f} {n['table']['off'][2]:.4f} "
                          f"--table-rot {n['table']['rpy'][0]:.2f} "
                          f"{n['table']['rpy'][1]:.2f} {n['table']['rpy'][2]:.2f} "
                          f"--crate-offset {n['crate']['off'][0]:.4f} "
                          f"{n['crate']['off'][1]:.4f} {n['crate']['off'][2]:.4f} "
                          f"--crate-rot {n['crate']['rpy'][0]:.2f} "
                          f"{n['crate']['rpy'][1]:.2f} {n['crate']['rpy'][2]:.2f}")
                else:
                    return False
                return True

            nudge_show = {"on": False}
            have_nudge = (nudge_gid["table"] is not None or nudge_gid["crate"] is not None)

            tracks = {rid: RobotTrack(mujoco, rid, sessions[rid], model, prefixes[rid],
                                      frame_T[rid], model_path,
                                      pins=getattr(args, "pin_map", {}).get(rid))
                      for rid in sorted(sessions) if rid in prefixes}
            t0 = min(tr.t_epoch[0] for tr in tracks.values())
            t1 = max(tr.t_epoch[-1] for tr in tracks.values())
            duration = float(t1 - t0)
            crate_src = None
            if not args.no_crate:
                for rid in sorted(sessions):
                    cs = sv.CrateSource(sessions[rid].get("crate_csv"))
                    if cs.available:
                        crate_src = cs
                        if not args.no_crate_smooth:
                            mmz = (_read_stl_min_z(crate_mesh, args.crate_stl_scale)
                                   if crate_mesh else None)
                            smooth_crate_track(crate_src, mesh_min_z=mmz,
                                               z_bias=args.crate_z_bias)
                        break
            video = MultiVideo(sessions) if (not args.no_video and cv2 is not None) else None
            tac_overlay, tac_by_robot = _load_tactile(tracks, model)
            crate_hold = {"raw": None}
            print(f"[twin] ▶ {sel['condition']} / {sel['attempt']}  "
                  f"({len(tracks)} robot(s), {duration:.1f}s"
                  + (f", crate {crate_src.n} frames" if crate_src is not None else "")
                  + (f", base-fix {sorted(base_override)}" if base_override else "") + ")")

            outcome = {"mode": None}                      # 'menu' | 'quit' | 'rebuild'

            with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                              show_right_ui=False) as viewer:
                viewer.cam.distance = 3.2
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -18
                viewer.cam.lookat[:] = [table_pos[0], table_pos[1], 0.4]

                def render_frame(abs_epoch):
                    data.qpos[:] = model.qpos0
                    info = {}
                    for rid, tr in tracks.items():
                        k, in_range = tr.index_at(abs_epoch)
                        tr.pose_into(data, model.qpos0, k)
                        info[rid] = (k, in_range)
                    crate_pose = crate_src.at(abs_epoch) if crate_src is not None else None
                    crate_tracked = crate_pose is not None
                    if crate_qadr is not None:
                        if crate_tracked:
                            p_c, q_c = crate_pose
                            crate_hold["raw"] = _make_T(sv._quat_to_mat(q_c),
                                                        np.asarray(p_c, float))
                        if crate_hold["raw"] is not None:
                            nb = nudge["crate"]
                            T_body = crate_hold["raw"] @ _make_T(
                                _rpy_to_mat(nb["rpy"]), np.asarray(nb["off"], float))
                            data.qpos[crate_qadr:crate_qadr + 7] = [
                                *T_body[:3, 3], *_mat_to_quat_wxyz(T_body[:3, :3])]
                    mujoco.mj_forward(model, data)
                    if tac_overlay is not None and tac_by_robot:
                        raw = np.zeros(len(tac_overlay["sensor_points"]), dtype=np.float64)
                        for rid, (idx2, stripped) in tac_by_robot.items():
                            tac = tracks[rid].tactile
                            j = int(np.clip(np.searchsorted(tac["t"], abs_epoch), 0,
                                            tac["v"].shape[0] - 1))
                            snap = we.reconstruct_touch_snapshot(tac["v"][j], tac["layout"])
                            raw[idx2] = build_mujoco_sensor_overlay_values_from_snapshot(
                                snap, stripped, normalize=False)
                        vals = prepare_live_scalar_arrow_values(raw, tac_overlay,
                                                                **sv._LIVE_ARROW_CFG)
                        render_mujoco_sensor_overlay(viewer, data,
                                                     tac_overlay["sensor_points"], vals)
                    else:
                        viewer.user_scn.ngeom = 0
                    sv._add_triad(viewer.user_scn, table_pos, np.eye(3), 0.15, 0.006)
                    fmags = {}
                    for rid, tr in tracks.items():
                        k, in_range = info[rid]
                        if tr.pelvis_bid is not None:
                            sv._add_triad(viewer.user_scn, data.xpos[tr.pelvis_bid],
                                          data.xmat[tr.pelvis_bid].reshape(3, 3), 0.12, 0.006)
                        if not in_range:
                            continue
                        for side in ("left", "right"):
                            w, p = tr.wrench.at(k, side)
                            if w is None:
                                continue
                            F_world = tr.R_frame @ w[:3]
                            p_world = tr.to_world(p)
                            fmag = float(np.linalg.norm(F_world))
                            fmags[(rid, side)] = fmag
                            sv._add_sphere(viewer.user_scn, p_world, 0.012,
                                           (0.2, 0.4, 1.0, 0.9))
                            if fmag > 1e-3:
                                a = min(1.0, fmag / max(1e-6, args.force_full))
                                sv._add_arrow(viewer.user_scn, p_world,
                                              p_world + F_world * fs, 0.012,
                                              (a, 1.0 - a, 0.1, 0.9))
                    if crate_tracked:
                        p_c, q_c = crate_pose
                        R_c = sv._quat_to_mat(q_c)
                        sv._add_triad(viewer.user_scn, p_c, R_c, 0.10, 0.006)
                        if crate_qadr is None:
                            sv._add_box(viewer.user_scn, p_c, R_c, crate_half,
                                        (0.95, 0.6, 0.1, 0.5))
                    return fmags, crate_tracked, info

                play = True
                speed = float(args.speed)
                tp = 0.0
                last = time.monotonic()

                while viewer.is_running():
                    now = time.monotonic()
                    dt = now - last
                    last = now
                    if seek["req"] is not None and duration > 0:
                        tp = seek["req"] / SEEK_MAX * duration
                        seek["req"] = None
                    elif play:
                        tp += dt * speed
                        if tp >= duration:
                            tp = duration if not args.loop else 0.0
                    tp = min(max(tp, 0.0), duration)
                    abs_epoch = t0 + tp

                    fmags, crate_tracked, info = render_frame(abs_epoch)
                    viewer.sync()

                    if cv2 is not None:
                        frame = (video.frame_at(abs_epoch)
                                 if (video and video.available) else None)
                        if frame is None:
                            frame = np.zeros((200, 1000, 3), dtype=np.uint8)
                        if seek["ok"] and duration > 0:
                            pos = int(tp / duration * SEEK_MAX)
                            if pos != seek["last"]:
                                seek["prog"] = True
                                cv2.setTrackbarPos("seek", play_win, pos)
                                seek["prog"] = False
                                seek["last"] = pos
                        rng = "".join("✓" if info[r][1] else "·" for r in sorted(info))
                        hud = (f"{_mmss(tp)} / {_mmss(duration)}  "
                               f"{'PLAY' if play else 'PAUSE'} x{speed:.2f}  "
                               f"[{sel['condition']} / {sel['attempt']}]  robots[{rng}]")
                        if crate_src is not None:
                            hud += f"  crate:{'ok' if crate_tracked else '—'}"
                        if base_override:
                            hud += f"  base-fix:{sorted(base_override)}"
                        hud += "   m/ESC=menu  b/B base" + ("/x ext" if external_base else "")
                        if have_nudge:
                            hud += "  [o]=nudge" + (" ON" if nudge_show["on"] else "")
                        show_nudge_bar = have_nudge and nudge_show["on"]
                        bar_h = 46 if show_nudge_bar else 26
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), (0, 0, 0), -1)
                        cv2.putText(frame, hud, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (230, 230, 230), 1)
                        if show_nudge_bar:
                            tgt = nudge_sel["t"]
                            o = nudge[tgt]["off"] * 1000.0
                            yaw = nudge[tgt]["rpy"][2]
                            n_hud = (f"nudge[{tgt}]  x{o[0]:+.0f} y{o[1]:+.0f} "
                                     f"z{o[2]:+.0f}mm  yaw{yaw:+.1f}deg  "
                                     f"step {nudge_step['pos']*1000:.1f}mm/"
                                     f"{nudge_step['rot']:.1f}deg   (1/2 pick  hljknm move "
                                     f" [] yaw  -/= step  0 reset  p print)")
                            cv2.putText(frame, n_hud, (8, 39), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.42, (120, 220, 255), 1)
                        cv2.imshow(play_win, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            outcome["mode"] = "quit"
                            break
                        elif key in (ord('m'), 27):       # menu / ESC → back to the menu
                            outcome["mode"] = "menu"
                            break
                        elif key == ord('o'):
                            nudge_show["on"] = not nudge_show["on"]
                        elif nudge_show["on"] and _nudge_key(key):
                            pass
                        elif key == ord('b'):             # borrow THIS attempt's bases
                            picked = {}
                            for rid, s in sessions.items():
                                T, _n = load_base_pose(s["session_dir"], args.dm_min,
                                                       args.upright_min)
                                if T is not None:
                                    picked[rid] = T
                            if picked:
                                base_override.update(picked)
                                print(f"[twin] base-fix: using {sel['condition']}/"
                                      f"{sel['attempt']} base for robot(s) "
                                      f"{sorted(picked)} across ALL attempts")
                                outcome["mode"] = "rebuild"
                                break
                            print("[twin] base-fix: this attempt has no localized base "
                                  "to borrow.")
                        elif key == ord('B'):             # reset to canonical bases
                            if base_override:
                                base_override.clear()
                                print("[twin] base-fix cleared — back to canonical bases")
                                outcome["mode"] = "rebuild"
                                break
                        elif key == ord('x'):             # force the --base-from (other-session) base
                            if external_base:
                                base_override.update(external_base)
                                print("[twin] base-fix: forcing the --base-from base for "
                                      f"robot(s) {sorted(external_base)} across ALL attempts")
                                outcome["mode"] = "rebuild"
                                break
                            print("[twin] no --base-from base loaded — pass "
                                  "--base-from <other session/attempt path> to borrow one.")
                        elif key == ord(' '):
                            play = not play
                        elif key == ord('a'):
                            tp = max(0.0, tp - 5.0)
                        elif key == ord('d'):
                            tp = min(duration, tp + 5.0)
                        elif key == ord('A'):
                            tp = max(0.0, tp - 30.0)
                        elif key == ord('D'):
                            tp = min(duration, tp + 30.0)
                        elif key == ord(','):
                            speed = max(0.1, speed / 1.5)
                        elif key == ord('.'):
                            speed = min(8.0, speed * 1.5)
                        elif key == ord('r'):
                            tp = 0.0
                        if cv2.getWindowProperty(play_win, cv2.WND_PROP_VISIBLE) < 1:
                            outcome["mode"] = "menu"
                            break
                    else:
                        time.sleep(max(0.0, 1.0 / 60.0 - (time.monotonic() - now)))

            if video is not None:
                try:
                    video.release()
                except Exception:                         # noqa: BLE001
                    pass
            if outcome["mode"] == "rebuild":
                continue                                  # base changed → rebuild same attempt
            if cv2 is not None:
                try:
                    cv2.destroyWindow(play_win)
                except Exception:                         # noqa: BLE001
                    pass
            return outcome["mode"] or "menu"

    # ── Top level: menu ↔ playback ──
    if cv2 is None:
        if selections:
            play_attempt(0)
        return
    try:
        while True:
            idx = show_menu()
            if idx is None:
                break
            try:
                cv2.destroyWindow(menu_win)
            except Exception:                             # noqa: BLE001
                pass
            if play_attempt(idx) == "quit":
                break
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Digital-twin reconstruction: table + crate + BOTH G1 robots "
                    "in one MuJoCo scene, placed by head-cam base localization.")
    p.add_argument("session", help="the TRIAL PARENT folder (holding the V / A+V / H+V / "
                                    "H+A+V condition folders) to tab across — or a single "
                                    "condition/attempt dir / robot_raw*.csv (pair auto-found)")
    p.add_argument("--model", default=None, help="override MuJoCo XML path")
    p.add_argument("--no-video", action="store_true", help="skip ZED video playback")
    p.add_argument("--no-crate", action="store_true", help="skip the crate overlay")
    p.add_argument("--no-floor", action="store_true",
                   help="no floor plane at all (default: one floor at the height the "
                        "robots stand on — NOT at z=0, which is the table top)")
    p.add_argument("--no-crate-smooth", action="store_true",
                   help="play the crate track RAW (default: playback-side clean-up — "
                        "zero-phase smoothing, freeze while stationary, and a "
                        "rest-on-table z-bias correction from the crate STL)")
    p.add_argument("--crate-z-bias", type=float, default=None,
                   help="constant z bias (m) to subtract from the crate track. Default: "
                        "auto from the lowest upright stationary segment resting on the "
                        "table (needs the crate STL). 0 disables the correction.")
    p.add_argument("--speed", type=float, default=1.0, help="initial playback speed")
    p.add_argument("--loop", action="store_true", help="loop playback")
    p.add_argument("--force-scale", type=float, default=0.02,
                   help="wrench arrow length (m) per Newton")
    p.add_argument("--force-full", type=float, default=10.0,
                   help="force (N) mapped to a fully-red arrow")
    p.add_argument("--crate-size", type=float, nargs=3, default=(0.3, 0.3, 0.3),
                   metavar=("LX", "LY", "LZ"), help="crate box full extents (m)")
    p.add_argument("--table-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"),
                   help="table-top center in the world/tag frame (default origin)")
    p.add_argument("--table-size", type=float, nargs=3, default=(1.2, 0.8, 0.74),
                   metavar=("LX", "LY", "LZ"),
                   help="table box full extents (m) — used only WITHOUT a table STL")
    p.add_argument("--table-stl", default=None,
                   help="CAD .stl for the table, ITS ORIGIN AT TABLE TAG 0 "
                        "(default: twin_assets/table.stl if present)")
    p.add_argument("--crate-stl", default=None,
                   help="CAD .stl for the crate, ITS ORIGIN AT CRATE TAG 10 "
                        "(default: twin_assets/crate.stl if present)")
    p.add_argument("--table-stl-scale", type=float, default=1.0,
                   help="table STL scale (use 0.001 if the STL is in millimetres)")
    p.add_argument("--crate-stl-scale", type=float, default=1.0,
                   help="crate STL scale (use 0.001 if the STL is in millimetres)")
    # Table STL is authored in the tag-0 frame (Z into the table); the twin world is Z-up,
    # so it's flipped 180° about X by default. Match the --table-z-down used at record time.
    p.add_argument("--table-z-down", dest="table_z_down", action="store_true", default=True,
                   help="table STL authored in the tag-0 (Z-into-table) frame — flip it to "
                        "the Z-up world (default; matches the rig's --table-z-down)")
    p.add_argument("--no-table-z-down", dest="table_z_down", action="store_false",
                   help="table STL already authored Z-up (no flip)")
    # Fine alignment nudges for when the CAD origin doesn't perfectly match the tag.
    p.add_argument("--table-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"),
                   help="translation nudge (m, WORLD frame) for the table STL")
    p.add_argument("--table-rot", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("R", "P", "Y"),
                   help="rotation nudge (deg roll/pitch/yaw, world frame, about the table "
                        "origin) for the table STL")
    p.add_argument("--crate-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"),
                   help="translation nudge (m, in the CRATE/tag-10 frame) for the crate STL")
    p.add_argument("--crate-rot", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("R", "P", "Y"),
                   help="rotation nudge (deg roll/pitch/yaw, in the crate frame) for the "
                        "crate STL")
    p.add_argument("--dm-min", type=float, default=15.0,
                   help="min AprilTag decision margin for a base-pose sample")
    p.add_argument("--upright-min", type=float, default=0.7,
                   help="min |upright| for a base-pose sample")
    p.add_argument("--base-from", dest="base_from", default=None,
                   help="borrow the robot base pose(s) from a DIFFERENT session/trial/attempt "
                        "(any path) when THIS session failed to localize — e.g. every trial had "
                        "a base error. Fills in any robot (master OR peer) the current trial "
                        "couldn't localize; press 'x' in the viewer to FORCE it over an in-trial "
                        "base. The robot must not have moved between the two sessions.")
    p.add_argument("--pin-joint", dest="pin_joint", action="append", default=None,
                   metavar="RID:JOINT=RAD",
                   help="Pin a joint to a FIXED angle (radians) in playback for robot RID, "
                        "overriding its recorded value — repeatable. Use for a broken actuator so "
                        "the twin matches the real robot. E.g. a sheared LEFT thumb "
                        "abduction/adduction on robot 2: --pin-joint 2:left_thumb_1_joint=0.35 "
                        "(left_thumb_1_joint = thumb rotation; the 4 thumb bend joints are "
                        "thumb_2/3/4). Tune the value to the physical stuck angle.")
    args = p.parse_args(argv)

    # Parse --pin-joint 'RID:joint=rad' entries → {rid: {joint_name: value}} for RobotTrack.
    try:
        args.pin_map = _parse_pin_specs(args.pin_joint)
    except ValueError as exc:                        # noqa: BLE001
        p.error(str(exc))
    if args.pin_map:
        print(f"[twin] joint pins (broken-actuator hold): {args.pin_map}")

    if not os.path.exists(args.session):
        p.error(f"not found: {args.session}")

    selections, canonical_base, base_source, pid = gather_trial(
        args.session, dm_min=args.dm_min, upright_min=args.upright_min)
    print(f"[twin] participant {pid or '?'}  →  {len(selections)} attempt(s): "
          + ", ".join(f"{s['condition']}/{s['attempt']}" for s in selections))

    # Cross-session base borrow: source base pose(s) from a good session and fill any robot the
    # current trial couldn't localize (master or peer). 'x' in the viewer forces it everywhere.
    external_base = {}
    if args.base_from:
        if not os.path.exists(args.base_from):
            p.error(f"--base-from not found: {args.base_from}")
        external_base, ext_src = gather_bases_from(
            args.base_from, dm_min=args.dm_min, upright_min=args.upright_min)
        if not external_base:
            print(f"[twin] --base-from {args.base_from}: no localized base found there.")
        for rid, T in external_base.items():
            if rid not in canonical_base:                # current session floated this robot
                canonical_base[rid] = T
                base_source[rid] = f"[other session] {ext_src[rid]}"
                print(f"[twin] robot{rid}: base BORROWED from another session ({ext_src[rid]})  "
                      f"pos={np.round(T[:3, 3], 3).tolist()}")
            else:
                print(f"[twin] robot{rid}: --base-from base available ({ext_src[rid]}); the "
                      "current trial already localized it — press 'x' to force the borrowed one.")

    if cv2 is None and not args.no_video:
        print("[twin] OpenCV (cv2) unavailable — running MuJoCo only.")
        args.no_video = True
    if cv2 is None and len(selections) > 1:
        print("[twin] NOTE: OpenCV unavailable — the clickable tab bar is disabled; only "
              f"the first attempt ({selections[0]['condition']}/{selections[0]['attempt']}) "
              "is shown.")
    run(selections, canonical_base, base_source, args, external_base=external_base)


if __name__ == "__main__":
    main()
