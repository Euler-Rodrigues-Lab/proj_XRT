"""digital_twin.py loads + composes a single-operator SOLO take (one robot, no partner, no
head-cam base, no crate, no video) without opening a window — and a two-robot study attempt
still composes the full table + both robots."""
from __future__ import annotations

import csv
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
dt = pytest.importorskip("projects.inspire_hand_teleop.digital_twin")
we = dt.we


def _write_take(leaf, meta, n=20):
    """A robot_raw CSV in the exact SessionRecorder format + its meta sidecar."""
    leaf.mkdir(parents=True)
    path = leaf / "robot_raw_take_20261006_120000.csv"
    fields = we.session_fieldnames()
    t0 = time.time()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in range(n):
            row = {c: 0.0 for c in fields}
            row["timestamp"] = round(t0 + 0.02 * k, 4)
            row["q_left_0"] = 0.01 * k
            w.writerow(row)
    we.save_capture_meta(str(path), {"model_path": we.DEFAULT_MODEL_PATH,
                                     "record_start_epoch": t0, **meta})
    return path


def _compose(sessions, base_T, table):
    return dt.build_twin_model(mujoco, we.DEFAULT_MODEL_PATH, base_T, np.zeros(3),
                               np.array([0.6, 0.4, 0.37]), table=table)


def _world_boxes(model):
    return sum(1 for g in range(model.ngeom) if model.geom_bodyid[g] == 0
               and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)


def test_solo_take_loads_and_builds_a_one_robot_model(tmp_path):
    root = tmp_path / "solo_20261006_120000"
    _write_take(root / "solo" / "attempt1" / "robot2",
                {"solo": True, "robot_id": 2, "robots_total": 1})
    sels, base, _src, pid = dt.gather_trial(str(root))
    assert pid == "solo" and base == {}
    assert [(s["condition"], s["attempt"]) for s in sels] == [("solo", "attempt1")]
    sessions = sels[0]["sessions"]
    assert list(sessions) == [2] and dt.is_solo(sessions)
    assert sessions[2]["crate_csv"] is None and sessions[2]["stereo_video"] is None
    # Same discovery from the attempt dir / the robot leaf (View Last / session_viewer paths).
    for p in (root / "solo" / "attempt1", root / "solo" / "attempt1" / "robot2"):
        s2, _pid, _rel = dt.gather_sessions(str(p))
        assert list(s2) == [2]

    model, prefixes, frame_T, crate_qadr = _compose(sessions, {2: None}, table=False)
    assert prefixes == {2: "r2_"} and crate_qadr is None
    assert _world_boxes(model) == 0                       # no table slab
    model.body("r2_pelvis")
    with pytest.raises(Exception):
        model.body("r1_pelvis")
    # Robot at the world origin (pelvis over x=y=0).
    data = mujoco.MjData(model)
    track = dt.RobotTrack(mujoco, 2, sessions[2], model, prefixes[2], frame_T[2],
                          we.DEFAULT_MODEL_PATH)
    k, in_range = track.index_at(track.t_epoch[-1])
    assert in_range and k == track.n - 1
    track.pose_into(data, model.qpos0, k)
    mujoco.mj_forward(model, data)
    assert np.allclose(data.xpos[track.pelvis_bid][:2], 0.0, atol=1e-6)
    # No video → MultiVideo reports unavailable instead of crashing (--no-video path too).
    assert not dt.MultiVideo(sessions).available


def test_two_robot_study_attempt_still_composes_table_and_both_robots(tmp_path):
    attempt = tmp_path / "P1_20261006_1200" / "V" / "attempt1"
    for n in (1, 2):
        _write_take(attempt / f"robot{n}", {"robot_id": n, "robots_total": 2})
    sessions, pid, rel = dt.gather_sessions(str(attempt))
    assert sorted(sessions) == [1, 2] and not dt.is_solo(sessions)
    assert pid == "P1" and rel == "V/attempt1"
    model, prefixes, _frame_T, _cq = _compose(sessions, {1: None, 2: None}, table=True)
    assert sorted(prefixes) == [1, 2] and _world_boxes(model) == 1
