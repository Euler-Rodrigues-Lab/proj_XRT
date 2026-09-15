# XRT — Extended-Reality Teleoperation

XRT is the application and study layer for XR teleoperation of a Unitree G1 with
Inspire tactile hands, built on the shared G1 robot and XR device packages.

Robot models, simulation controllers, arm/torso/hand retargeting, collision filtering,
and the base G1 hardware interfaces are owned by the pinned `g1_teleop` submodule.
XRT_devices owns WebRTC/OpenXR input, process isolation, transport telemetry, body-pose
recording, and the optional ZED streaming launcher. This repository owns the study manager,
tactile and haptic policy, camera/localization orchestration, hardware safety extensions, and
digital-twin playback.  The compatibility modules under `projects/g1_*` are adapters,
not a second copy of the retargeting implementation.

## Clone and install

Use Python 3.10–3.12 (the study dependencies currently exclude Python 3.13).
Clone recursively so uv can install the nested editable packages:

```bash
git clone --recurse-submodules https://github.com/Euler-Rodrigues-Lab/proj_XRT.git
cd proj_XRT
uv sync --locked --python 3.12 --extra test
source .venv/bin/activate
python -m pytest -q
```

For an existing checkout, initialize dependencies with
`git submodule update --init --recursive` before syncing. Do not update over
uncommitted changes inside a submodule.

The command above installs the XR transport and public fallback/test dependencies
for simulation. For the full hardware and study environment, use:

```bash
uv sync --locked --python 3.12 --extra hardware --extra study --extra test
```

Add `--extra transcription` if using speech transcription. This project's extra
is named `hardware`, not G1's `hw`. Repeat the extras you need on every sync:
`uv sync` removes packages outside the selected dependencies. `--inexact` preserves
additional installed packages, but explicitly selecting extras is more reproducible.
The Unitree SDK resolves from its public Git source. ZED SDK, GStreamer, MediaMTX
and device/system drivers remain separate prerequisites; see the study preflight
checklist. Installing dependencies does not itself start hardware, but the live
application commands below require `--dry_run` to avoid hardware operation.

If hardware sync fails with `Could not locate cyclonedds`, install the native
Cyclone DDS library first, then export `CYCLONEDDS_HOME` to its installation
prefix and repeat the sync. Unitree pins the Python binding to version 0.10.2;
its [official installation FAQ](https://github.com/unitreerobotics/unitree_sdk2_python#faq)
documents building the compatible `releases/0.10.x` native library with CMake.
This native prerequisite is not installed by uv. For an existing native install:

```bash
export CYCLONEDDS_HOME=/absolute/path/to/cyclonedds/install
uv sync --locked --python 3.12 --extra hardware --extra study --extra test
```

The licensed solver wheel and license are supplied separately. Follow the
[G1 provisioning instructions](external/g1_teleop/README.md#install) to register
them, then run `geo-kin-provision install --product g1-inspire` from this project
with its environment activated. Do not commit wheels or license files.

The licensed Rust `geo_kin` wheel is selected automatically when installed.  Without
it, `g1_teleop` falls back to its public backend.  Simulation-only and data-analysis
tools do not import the Unitree, Inspire, Ostrich-neck, ZED, or RealSense SDKs until
those features are started.

## Run

```bash
# Live XR + MuJoCo, no commands sent to hardware
python -m projects.inspire_hand_teleop.demo_g1_inspire_xr_robot_teleop_full_body_hw --dry_run

# G1 + Inspire hardware
python -m projects.inspire_hand_teleop.demo_g1_inspire_xr_robot_teleop_full_body_hw --net eno1

# Full two-robot study manager
python -m projects.inspire_hand_teleop.XRT_user_study_manager --help

# Offline solver/controller smoke replay supplied by the G1 submodule
python -m g1_teleop.demos.replay_offline --hand inspire --headless --no-loop
```

For the study workflow, cameras, networking, safety systems, and recorded dataset
layout, see [XRT_USER_STUDY_MANAGER.md](projects/inspire_hand_teleop/XRT_USER_STUDY_MANAGER.md)
and [STUDY_PREFLIGHT_CHECKLIST.txt](projects/inspire_hand_teleop/STUDY_PREFLIGHT_CHECKLIST.txt).

The Quest client connects to the WebRTC server on port 8080.  The headset installation
guide is maintained at <https://xr-robot-teleop-website.pages.dev/>.

## Source layout

```text
external/g1_teleop/                 pinned G1 base (Git submodule)
external/g1_teleop/external/XRT_devices/  shared input, feedback, recording, video
projects/g1_upper_body_teleop/      XRT's study-specific DDS safety extensions
projects/g1_full_body_kinematic/    legacy API adapters onto g1_teleop
projects/inspire_hand_teleop/       manager, haptics, sensing, calibration, playback
```

The model files visible under `projects/inspire_hand_teleop/g1_with_inspire_hand/`
and the mesh path under `projects/g1_full_body_kinematic/g1/` are relative symlinks into
the submodule.  This preserves old recording metadata and script paths without
duplicating G1 assets.

The actively tuned XRT head/chest collision geometry remains application-owned at
`projects/g1_full_body_kinematic/config/g1_geometry.yaml`. The compatibility solver
passes it to the Python reference backend; licensed Rust wheels use their equivalent
compiled-in safety geometry, as required by that API. General G1 simulation, XR, and
offline-replay entry points are supplied by `external/g1_teleop/g1_teleop/demos/`
instead of retaining stale duplicate demos here.

Study transport imports `StudyXRDevice` from `xrt_devices.study` or
`WebRTCServerProxy` from `xrt_devices.study_process`. Both require explicit
`start()` and `close()`. The proxy acknowledges server startup and recording
changes; poses use a bounded latest-value mailbox. The Quest creates the
`unity_cmds`, `haptics`, and `motor_stats` channels. Take recording is configured
atomically with `configure_recording(...)`; shutdown drains the CSV writer.
The ZED Bash launcher is resolved from the installed `xrt_devices.video` package;
GStreamer, ZED SDK, NVIDIA encoding, and MediaMTX remain optional system tools.

Local migration changes in the nested device checkout must be committed and
published upstream, then the robot and application gitlinks updated, before a
fresh remote clone contains the new study APIs. No live headset or hardware
validation is implied by the offline/loopback tests.
