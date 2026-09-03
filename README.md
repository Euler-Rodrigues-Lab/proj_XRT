# XRT — Extended-Reality Teleoperation

XRT is the application and study layer for XR teleoperation of a Unitree G1 with
Inspire tactile hands.  It was extracted from
`SEW-Geometric-Teleop@proj_xrt/inspire_hand_control_hw` at commit `32bef869`.

Robot models, simulation controllers, arm/torso/hand retargeting, collision filtering,
and the base G1 hardware interfaces are owned by the pinned `g1_teleop` submodule.
This repository owns the live WebRTC/OpenXR device, the XRT study manager, tactile and
haptic feedback, camera/localization tools, hardware safety extensions, recording, and
digital-twin playback.  The compatibility modules under `projects/g1_*` are adapters,
not a second copy of the retargeting implementation.

## Clone and install

```bash
git clone --recurse-submodules https://github.com/Euler-Rodrigues-Lab/proj_XRT.git
cd proj_XRT
python -m venv .venv
source .venv/bin/activate
python -m pip install -e external/g1_teleop
python -m pip install -e '.[hardware,study]'
```

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
projects/shared_devices/            OpenXR/WebRTC capture + process proxy
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
