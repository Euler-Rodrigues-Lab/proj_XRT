"""Operator modes of the study manager: single operator (default) vs dual-operator study.

Builds the real StudyManagerApp headless (offscreen Qt, --dry_run) with HOME pointed at a
temp dir so the persisted ~/.xrt_study_config.json is never the real one.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

if importlib.util.find_spec("geo_kin") is not None and not os.environ.get("GEO_KIN_LICENSE"):
    pytest.skip("licensed geo_kin wheel installed but GEO_KIN_LICENSE is unset",
                allow_module_level=True)

M = pytest.importorskip("projects.inspire_hand_teleop.XRT_user_study_manager")
from proj_xrt.g1_adapter import STUDY_GEOMETRY_CONFIG  # noqa: E402


def _pump(app, cond, timeout=20.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


@pytest.fixture
def make_manager(tmp_path, monkeypatch):
    """Factory: build a dry-run manager with extra CLI args; records which study services
    the startup / a mode switch asked for (the real ones would open sockets / mics)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(M.StudyManagerApp, "_start_video_streaming_default", lambda self: None)
    monkeypatch.setattr(M.StudyManagerApp, "_init_link", lambda self: calls.append("link"))
    monkeypatch.setattr(M.StudyManagerApp, "_init_transcriber",
                        lambda self: calls.append("transcriber"))
    real_zed_check = M.StudyManagerApp._check_zed_calibration
    monkeypatch.setattr(M.StudyManagerApp, "_check_zed_calibration",
                        lambda self: (calls.append("zedcal"), real_zed_check(self))[1])
    # Confirm the mode-switch dialog without a modal loop.
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    built = []

    def _make(*argv):
        args = M._build_argparser().parse_args(["--dry_run", *argv])
        w = M.StudyManagerApp(args)
        built.append(w)
        assert _pump(app, lambda: w._init_done and w._state != M._S_INIT), "init never finished"
        # Let the startup singleShots (link 600 ms, ZED check 900 ms, transcriber +2.5 s) fire.
        _pump(app, lambda: False, timeout=3.0)
        return w

    yield app, _make, calls, tmp_path
    for w in built:
        w.close()                # stops the control thread + the WebRTC server (port 8080)
        app.processEvents()
        w.deleteLater()
    app.processEvents()


def _shown(w, widget):
    return widget.isVisibleTo(w)


def _study_widgets(w):
    return [w._btn_link_connect, w._link_role_combo, w._pid_edit, w._btn_crate_track,
            w._btn_start_rec, w._btn_force_calib, w._btn_headcam, w._status_crate_box]


def _basic_widgets(w):
    return [w._btn_start_teleop, w._btn_stop_teleop, w._btn_bootup, w._btn_estop,
            w._btn_quit, w._btn_stream_toggle, w._btn_reboot_zed, w._btn_reboot_webrtc,
            w._btn_audio_haptics, w._btn_vibrotactile, w._btn_viewport, w._btn_motor_monitor,
            w._btn_reboot_neck, w._btn_reboot_hands, w._btn_eco, w._btn_live_viewer,
            w._robot_combo, w._btn_operator_mode]


def _assert_single(w):
    assert w._dual_study is False
    assert "Single-Operator" in w.windowTitle()
    assert not any(_shown(w, x) for x in _study_widgets(w))
    assert all(_shown(w, x) for x in _basic_widgets(w))
    # Status panel: only THIS robot's box, no head-cam / ZED-calib study rows.
    assert _shown(w, w._status_robot_boxes[w._robot_id])
    assert not _shown(w, w._status_robot_boxes[3 - w._robot_id])
    assert not _shown(w, w._light_rows[(w._robot_id, "headcam")][0])
    assert w._link is None
    s = w._solver
    assert s.wrist_rate_limit is False
    assert s._session_config.get("geometry_config_path") is None


def _assert_study(w):
    assert w._dual_study is True
    assert "User Study Manager" in w.windowTitle()
    assert all(_shown(w, x) for x in _study_widgets(w))
    assert all(_shown(w, x) for x in _basic_widgets(w))
    assert all(_shown(w, g) for g in w._status_robot_boxes.values())
    s = w._solver
    assert s.wrist_rate_limit is True
    assert s._session_config.get("geometry_config_path") == STUDY_GEOMETRY_CONFIG


def test_single_is_default_and_toggles_to_study_and_back(make_manager):
    app, make, calls, home = make_manager
    w = make("--link-role", "master")
    assert w._cfg["operator_mode"] == "single"
    _assert_single(w)
    # Single mode starts no study service even with a persisted link role.
    assert calls == []
    assert w._btn_operator_mode.isEnabled()
    # Basic teleop works without any session / link / calibration.
    w._on_start_teleop()
    assert w._state == M._S_TELEOP
    assert not w._btn_operator_mode.isEnabled()         # switching is blocked mid-teleop
    w._on_stop_teleop()
    _pump(app, lambda: w._state not in (M._S_TELEOP, M._S_WAITING), timeout=5.0)
    w._update_buttons()
    assert w._btn_operator_mode.isEnabled()

    w._on_toggle_operator_mode()
    _assert_study(w)
    assert {"link", "zedcal", "transcriber"} <= set(calls)
    cfg = json.loads((home / ".xrt_study_config.json").read_text())
    assert cfg["operator_mode"] == "study"

    w._on_toggle_operator_mode()
    _assert_single(w)
    cfg = json.loads((home / ".xrt_study_config.json").read_text())
    assert cfg["operator_mode"] == "single"


def test_study_startup_shows_study_ui_and_starts_services(make_manager):
    _app, make, calls, _home = make_manager
    w = make("--operator-mode", "study", "--link-role", "master")
    _assert_study(w)
    assert {"link", "zedcal", "transcriber"} <= set(calls)


def test_config_saved_before_operator_modes_stays_study_and_kneeling(make_manager):
    _app, make, _calls, home = make_manager
    (home / ".xrt_study_config.json").write_text(json.dumps({"robot_id": 2, "link_role": "peer"}))
    w = make()
    assert w._dual_study is True
    assert w._leg_stance == "kneeling"
    assert w._args.kneeling is True


def test_saved_single_standing_choice_is_kept(make_manager):
    _app, make, _calls, home = make_manager
    (home / ".xrt_study_config.json").write_text(
        json.dumps({"operator_mode": "single", "leg_stance": "standing"}))
    w = make()
    assert w._dual_study is False
    assert w._args.kneeling is False


def test_mode_switch_refused_when_it_would_change_a_booted_stance(make_manager):
    _app, make, _calls, _home = make_manager
    w = make()                                       # single + standing
    w._stance_locked = True                          # as after a real hardware bootup
    assert "leg stance" in w._operator_mode_block_reason()
    w._on_toggle_operator_mode()
    assert w._dual_study is False and w._args.kneeling is False


def test_mode_switch_allowed_when_the_booted_stance_is_kept(make_manager):
    _app, make, _calls, _home = make_manager
    w2 = make("--leg-stance", "kneeling")            # single + kneeling: study kneels too
    w2._stance_locked = True
    assert w2._operator_mode_block_reason() == ""
    w2._on_toggle_operator_mode()
    assert w2._dual_study is True and w2._args.kneeling is True


def test_standing_bootup_requires_the_gantry_box(make_manager):
    _app, make, _calls, _home = make_manager
    w = make()
    box = QtWidgets.QMessageBox(w)
    chk = w._add_gantry_check(box)
    assert chk is not None and not chk.isChecked()
    assert w._gantry_confirmed(chk, "bootup") is False
    chk.setChecked(True)
    assert w._gantry_confirmed(chk, "bootup") is True


def test_kneeling_bootup_has_no_gantry_box(make_manager):
    _app, make, _calls, _home = make_manager
    k = make("--leg-stance", "kneeling")
    assert k._add_gantry_check(QtWidgets.QMessageBox(k)) is None


def test_leg_gains_follow_the_stance():
    args = M._build_argparser().parse_args(["--dry_run"])
    args.kneeling = False
    assert M._leg_gain_scales(args) == (1.0, 1.0, 1.0, 1.0)
    args.kneeling = True
    leg_kp, leg_kd, ankle_kp, _ankle_kd = M._leg_gain_scales(args)
    assert (leg_kp, leg_kd) == (args.leg_kp_scale, args.leg_kd_scale)
    assert ankle_kp == 0.0


# ── Single-operator SOLO recording ────────────────────────────────────────────────────
def test_single_mode_solo_recording_writes_a_playable_take(make_manager):
    app, make, _calls, home = make_manager
    out = home / "rec"
    w = make("--output_dir", str(out))
    _assert_single(w)
    # Solo controls + playback buttons are shown in single mode.
    for b in (w._btn_solo_rec, w._btn_view_recording, w._btn_open_twin):
        assert _shown(w, b)
    assert not w._btn_view_recording.isEnabled()        # nothing recorded yet
    assert w._btn_solo_rec.isEnabled() and "Start Recording" in w._btn_solo_rec.text()

    w._on_toggle_solo_recording()
    assert w._solo_rec_active and "Stop Recording" in w._btn_solo_rec.text()
    # Recording blocks the operator-mode switch (and nothing else from the study starts).
    assert "recording" in w._operator_mode_block_reason()
    assert not w._btn_operator_mode.isEnabled()
    w._on_toggle_operator_mode()
    assert w._dual_study is False and w._solo_rec_active
    assert not w._recording_active and not w._session_open()
    assert not w._btn_view_recording.isEnabled() and not w._robot_combo.isEnabled()
    _pump(app, lambda: False, timeout=1.5)              # ~50 Hz tick → dry-run sim frames
    w._on_toggle_solo_recording()
    assert not w._solo_rec_active and "Start Recording" in w._btn_solo_rec.text()
    assert w._operator_mode_block_reason() == ""

    roots = list(out.glob("solo_*"))
    assert len(roots) == 1
    attempt = roots[0] / "solo" / "attempt1"
    leaf = attempt / "robot1"
    assert w._last_solo_dir == str(attempt)
    assert w._btn_view_recording.isEnabled()
    csvs = list(leaf.glob("robot_raw_solo_robot1_attempt1_*.csv"))
    assert len(csvs) == 1
    meta = json.loads(Path(str(csvs[0])[:-4] + ".meta.json").read_text())
    assert meta["solo"] is True and meta["robot_id"] == 1 and meta["robots_total"] == 1
    assert meta["operator_mode"] == "single" and meta["robot_state_source"] == "sim"
    assert meta["leg_stance"] == "standing" and meta["model_path"]
    assert meta["record_stop_epoch"] >= meta["record_start_epoch"]
    rows = csvs[0].read_text().splitlines()
    assert len(rows) > 10                               # header + sim frames
    header = rows[0].split(",")
    assert header == M.we.session_fieldnames()
    # Camera / crate / head-cam / transcript / study logs are NOT recorded.
    names = [p.name for p in leaf.iterdir()]
    assert not [n for n in names if n.startswith(("zed", "crate", "head_base", "summary",
                                                  "feedback", "latency", "transcript"))]
    # The digital twin's loader accepts the take as a one-robot SOLO selection.
    import projects.inspire_hand_teleop.digital_twin as dt
    sels, base, _src, pid = dt.gather_trial(str(roots[0]))
    assert [(s["condition"], s["attempt"]) for s in sels] == [("solo", "attempt1")]
    assert list(sels[0]["sessions"]) == [1] and dt.is_solo(sels[0]["sessions"])
    assert base == {} and pid == "solo"

    # A second take in the same run → attempt2 under the same root.
    w._on_toggle_solo_recording()
    _pump(app, lambda: False, timeout=0.5)
    w._solo_stop_recording("test")
    assert w._last_solo_dir == str(roots[0] / "solo" / "attempt2")


def test_estop_finalizes_a_solo_recording(make_manager):
    app, make, _calls, home = make_manager
    w = make("--output_dir", str(home / "rec"))
    w._on_toggle_solo_recording()
    _pump(app, lambda: False, timeout=0.5)
    w._on_estop_kill()
    assert not w._solo_rec_active and not w._session_recorder._active
    assert w._last_solo_dir is not None


def test_solo_recording_is_unavailable_in_study_mode(make_manager):
    _app, make, _calls, _home = make_manager
    w = make("--operator-mode", "study")
    assert not _shown(w, w._btn_solo_rec)
    w._solo_start_recording()
    assert not w._solo_rec_active
