# XRT User Study Manager

`XRT_user_study_manager.py` — the mission-control application for a **two-robot,
XR-teleoperated, multi-modal-feedback user study**, in which pairs of Unitree G1
humanoids (with Inspire tactile hands) cooperatively lift and flip a crate on a shared
table. The manager runs on one laptop per robot, drives the full teleoperation stack,
delivers the study's feedback conditions, records every sensor stream into a
digital-twin-ready dataset, and keeps both machines in lock-step over a network link —
so one experimenter click controls both robots.

The recorded data trains a **centralized diffusion policy** for the cooperative
manipulation task; every stream is raw + timestamped so the whole workspace can be
reconstructed offline.

Companion docs: `APRILTAG_STUDY_SETUP.md` (camera / Jetson / NTP setup),
`STUDY_PREFLIGHT_CHECKLIST.txt` (operating checklist), `twin_assets/README.md`
(table/crate CAD alignment).

---

## 1. Study design

| Element | Value |
|---|---|
| Task | Two G1 humanoids on opposite sides of a table cooperatively lift / flip a crate |
| Operators | One human operator per robot, each wearing a Meta Quest 3 |
| Feedback conditions (within-subject trials) | **V** (viewport video only) · **A+V** (audio + video) · **H+V** (vibrotactile haptics + video) · **H+A+V** (all) |
| Attempts per trial condition | 4 ("runs") |
| Session | One participant (pair) = 4 conditions × 4 attempts = 16 recorded takes |
| Condition order | Experimenter drag-ranks the trial order per participant group (counterbalancing) |
| Feedback delivery | The manager auto-switches the Quest app's viewport / audio / vibrotactile channels per condition at each take start, and logs every feedback event to prove delivery |

---

## 2. Hardware inventory

### Per robot station (×2, mirrored)
- **Unitree G1** humanoid (29 DoF), kneeling on a 3-D-printed test rig at the table.
  Low-level control via DDS (CycloneDDS / `unitree_sdk2py`), `LowCmd`/`LowState` at
  **500 Hz** over the robot's wired network.
- **Inspire RH56DFTP** hands (6 motors/hand) with full-surface **tactile skin**:
  per finger a 3×3 tip patch + 12×8 pad patch, plus palm — ~**1300 taxels/frame**
  total. Modbus TCP; commanded at **50 Hz**.
- **Ostrich neck**: custom 2-DoF Dynamixel gimbal (yaw ±90°, pitch ±60°) carrying the
  head camera pair; follows the operator's head pose.
- **ZED Mini** stereo camera on the neck — the operator's eyes.
  HD720@60; streamed to the headset and recorded as a **top-bottom stereo composite**
  (1280×1440: left eye above right).
- **G1 head Intel RealSense D435i** on the robot's onboard **Jetson** — used for
  robot-base self-localization and as an extra crate-tracking viewpoint.
  Served by `g1_headcam_server.py` on the Jetson (JPEG over TCP :5005,
  **multi-client broadcast** — several consumers share one camera).
- **Operator laptop** running this manager (PyQt5).
- **Meta Quest 3** headset (Unity XR app) — sends body pose, receives stereo video,
  haptics, audio and UI commands over WebRTC.

### Master station only
- **2 × Intel RealSense D435** external "crate cameras" on tripods viewing the table
  (USB3, 1280×720 @ 15 fps color).

### Shared workspace
- **Table AprilTag (id 0, 36h11, 65 mm)** on the tabletop — the **world origin** for
  everything.
- **Crate** skinned with an **11-tag 36h11 bundle** (anchor tag id 10 at the
  inside-centre-base); bundle geometry learned once by `apriltag_crate_calibrate.py`
  → `crate_bundle.json` (auto-pushed master → peer over the link).
- Optional CAD: `twin_assets/table.stl` (origin at tag 0) and `crate.stl` (origin at
  tag 10) skin the digital twin.

---

## 3. Network architecture

```
                       ┌────────────── WiFi router A (master's) ─────────────┐
                       │ 192.168.7.x  ── wired/downlink ──  WiFi router B    │
                       │                                     (peer's)        │
   Quest 3 #1 ~~6GHz~~ ┤                                        ├~~6GHz~~ Quest 3 #2
                       │                                        │
              ┌────────┴────────┐                      ┌────────┴────────┐
              │  MASTER laptop  │ ── StudyLink TCP ──▶ │   PEER laptop   │
              │  (robot 1)      │    :5070 (JSON)      │   (robot 2)     │
              └───┬────────┬────┘                      └───┬────────┬────┘
        USB-eth   │        │ USB3                USB-eth   │        │
   192.168.123.x  │        │ 2× RealSense   192.168.123.x  │        │ (no ext. cams)
        ┌─────────┴──┐     │ crate cams          ┌─────────┴──┐
        │  G1 #1     │     │                     │  G1 #2     │
        │  .161 MCU  │     │                     │  .161 MCU  │
        │  .164 Jet. │◀────┘  head-cam TCP :5005 │  .164 Jet. │
        └────────────┘                           └────────────┘
```

- **Inter-laptop link**: two **chained WiFi routers** — the master's router feeds the
  peer's router downstream (plain private LAN, `192.168.7.x`; **no VPN/Tailscale**).
  Master laptop e.g. `192.168.7.226`.
- **StudyLink** (`study_link.py`): a single persistent TCP socket (**:5070**, master
  listens, peer dials, auto-reconnect). Carries tiny JSON messages only — never video:
  - **commands** master↔peer, executed at a **shared epoch** (`_sync_or_local`) so both
    robots start/stop takes together;
  - **status** heartbeats ~2 Hz (full component health → both UIs show both robots);
  - **crate candidates** (peer's head-cam crate observations → master's fusion);
  - **clock pings** (NTP-style min-RTT offset estimate, typically ±3 ms — logged into
    every take's metadata for cross-robot time alignment; both laptops also run NTP).
- **Robot network** (per laptop): USB-ethernet adapter, `192.168.123.x` — laptop
  `.162`, G1 MCU `.161` (DDS 500 Hz), Jetson `.164` (head-cam server :5005).
- **Quest link** (per station): WebRTC — laptop is the server (**:8080** signaling).
  Data channels: `body_pose` (operator skeleton, ~60 Hz), `unity_cmds` (manager →
  headset toggles), `unity_state` (headset → manager UI mirror), `haptics`
  (**unordered / maxRetransmits=0** — reliable-ordered delivery head-of-line-blocks on
  WiFi loss and adds 0.5–2 s), `motor_stats`, `apriltag_pose`.
  Video: ZED → GStreamer (`zedsrc` → NVENC H.264, 24 Mb/s, GOP 30, zero-reorder,
  repeat-SPS) → RTSP → **MediaMTX** (:8554) → WebRTC to the headset.
- **Data delivery** peer → master: `rsync -az --partial` over SSH. The destination is
  **auto-derived** (master's user + recordings dir from its live link status, IP from
  the link socket), verified by an SSH preflight; the peer auto-sends its finished
  session at Stop Session so all data lands on the master.

---

## 4. Software architecture

### Master / peer roles
Both machines run the *same* manager; a Setup-bar role selects **Master** (robot 1,
listens, fuses the crate, hosts the external cameras, receives all data) or **Peer**
(robot 2, dials in, contributes its head cam to crate tracking, ships its data to the
master). All study controls (teleop, session, attempts, calibration) show **"(BOTH)"**
when linked and drive the two robots synchronously; either side's status is mirrored to
the other's UI. Role-specific UI is hidden on the other machine.

### Process model — "no per-frame vision inside the manager process"
CPython's GIL makes any sustained per-frame work inside the manager process starve the
control loops (measured: ~30 full-res AprilTag detections/s in-process → multi-second
teleop lag via input-queue backlog). The architecture therefore isolates every
frame-rate consumer into its **own process**:

| Process | What it does | IPC |
|---|---|---|
| **Manager** (PyQt5) | UI, control threads, recorders, link | — |
| **Crate tracker** (`apriltag_tracker.py --serve`) | Opens 2× RealSense + the robot's head-cam stream, detects the crate bundle, robust-fuses a world crate pose @ 15 Hz, records crate CSVs | socketpair, length-prefixed JSON (numpy-safe) |
| **Head-cam detection worker** (`head_cam_localize.py --serve-cam`) | JPEG-decodes + AprilTag-detects the Jetson stream for base localization (detection capped at 8 Hz) | socketpair, length-prefixed JSON |
| **ZED pipeline** (`zed_to_mediamtx.sh` → gst-launch) | Capture + NVENC encode + RTSP publish | RTSP |
| **MediaMTX** | RTSP/WebRTC relay (an already-running instance is *adopted*, not duplicated) | RTSP/HTTP |
| **ffmpeg** (per take) | `-c copy` remux of the RTSP stream to `zed_stereo_*.mp4` — **zero re-encode**, recording adds no latency | RTSP |
| Digital twin / viewers / dataset builder | Offline playback + analysis | files |

### Thread model (inside the manager)
- **ControlThread** — the hot path: 60 Hz IK tick (SEW-geometric arm IK for both 7-DoF
  arms + waist hold + head→neck), teleop blending, per-step timing telemetry,
  stale-pose failsafe, compliance/eco updates, haptics payload send.
- **Hardware send thread** — 500 Hz `LowCmd` publisher (CRC + DDS write).
- **BoneProcessor** — drop-newest operator-pose processing (a queued backlog is
  impossible by construction; always consumes the freshest skeleton).
- **WebRTC asyncio thread** (uvicorn/aiortc) — headset I/O.
- **NeckWorker** — fire-and-forget Dynamixel serial writes (serial I/O never blocks the
  IK loop).
- **Hand actuator worker** — 50 Hz Modbus finger commanding, decoupled from IK via a
  goal buffer; tactile reader threads cache sensor state.
- **SessionRecorder / wrench / latency workers** — sampling + file writes.
- **GUI thread** — Qt, 5 Hz status refresh with **change-detection on every widget
  update** (`setStyleSheet`/`setText` only when the rendered value changed; unconditional
  restyling of the ~40-tile motor monitor was measured as the #1 GIL consumer).

### Control & data flow (one station)
```
Quest 3 body pose ──WebRTC──▶ BoneProcessor (drop-newest) ──▶ ControlThread IK @60Hz
                                                                 │ q_goal arms/waist
                                                                 ├─▶ 500 Hz LowCmd ─▶ G1
                                                                 ├─▶ NeckWorker ─▶ Dynamixel neck
                                                                 ├─▶ hand goal buffer ─▶ 50 Hz Modbus ─▶ Inspire hands
                                                                 └─▶ haptics payload ─▶ Quest (unreliable channel)
ZED Mini ─▶ gst/NVENC ─▶ MediaMTX ─▶ WebRTC ─▶ Quest display   (+ ffmpeg -c copy per take)
tactile skin ─▶ reader threads ─▶ haptics intensities + tactile.bin recorder
G1 LowState @500Hz ─▶ state cache ─▶ recorder / compliance / eco / watchdog / wrench
```

---

## 5. Coordinate frames & localization

**World frame = table AprilTag 0**, on the tabletop (z = 0 *is the table surface*;
`world_from_table` flips the tag's z-down convention to a z-up world). Everything —
both robots, the crate, and the ZED cameras — is expressed in this one frame, which is
what makes the two independently-recorded robots composable into a single digital twin.

| What | How | Output |
|---|---|---|
| **Robot base** (each robot, live) | Head D435 detects tag 0 → chain `T_world_base = inv(T_optical_table) · inv(T_pelvis_torso(q_waist) · T_torso_d435 · T_d435_optical)` — URDF head-cam extrinsic + live waist FK (MuJoCo). Averaged over ~1.5 s and **locked** (the base is static). | `head_base_*.csv` per take |
| **Crate** (live, master fuses) | Each camera detects bundle tags → `T_world_crate = T_world_cam · T_cam_tag · inv(T_crate_tag)` per tag per camera → robust mean (rotation mean + weighted centroid) @ 15 Hz. Cameras: 2 RealSense + master head cam + **peer head cam** (streamed over the link as world-frame candidates → "4/4 cameras"). | `crate_pose_*.csv` (fused) + `crate_tags_raw_*.csv` (**re-fusable** per-tag/per-camera) |
| **Camera extrinsics** (one-time per session) | Each crate camera averages tag-0 detections ~1.5 s → `T_world_cam` anchored; tag 0 may then be occluded (single bumped camera can be re-anchored live without moving the world frame). | in-memory + meta |
| **ZED cameras** (offline) | The ZED rides the moving neck, so its pose is recovered **per frame** from tag 0 in the recorded video (`zed_localize.py`): `T_world_zed = world_from_table · inv(T_zed_tag0)`. A moving camera is expected; frames without the tag are gap-interpolated by the dataset builder. Recorded intrinsics land in the meta automatically. | `zed_pose_<side>_*.csv` |

---

## 6. Calibration features

- **🚀 Prepare Study** — one click, once per participant group, calibrates *everything
  on both robots*: starts + anchors the crate cameras, verifies both head-cam base
  localizations, pushes the crate bundle to the peer, engages the peer's crate
  contribution, measures NTP + link clock offset, verifies the ZED factory calibration
  conf. All camera anchoring also re-runs automatically at Start Session (a cold
  network camera is awaited by a warm-up gate before anchoring).
- **📐 Calibrate Cameras Now / Anchor / Re-Anchor Bumped Cam** — manual redo paths;
  bumped-camera re-anchor works *during* recording (tag 0 is the fixed origin).
- **🔧 Recalibrate Crate** — re-learn the 11-tag bundle geometry.
- **⚡ Fingertip force calibration** (per-hand window): hardware zero (register 1009) →
  software **baseline + noise-floor** capture per tactile sub-region — the noise floor
  becomes a live deadband that eliminates phantom haptic triggers. One-click **Auto
  Calibrate** does both hands in the correct order; optional known-mass points fit
  grams = slope·(raw − baseline) through the origin. Persisted to JSON, auto-loaded
  every run.
- **Clock sync** — NTP status + link offset (min-RTT, ±few ms) measured at prep and
  after each synced start; stamped into every take's meta for offline alignment.
- **Camera-calibration recording gate** — Start Session/Attempt refuses (with an
  override dialog) unless every camera is anchored to tag 0, so no take is recorded
  outside the shared frame.

---

## 7. Session & recording model

Recording is **per-take**: the session stays open across a participant while file
writers run only during attempts — downtime is never recorded.

```
🎬 Start Session      opens the session (gates: cameras anchored + peer→master SSH ok;
                      subsystems auto-started; folder created; ARMED — nothing recording)
●  Start Attempt      records the current (condition, run). ONE press per slot —
                      greys out for that attempt once pressed.
⏹  Stop Attempt       ends + saves the take (pointer stays; classify/redo first —
                      they act on the take while it records)
⏭  Continue to <cond> Run <n>
                      one button: next attempt, and after run 4 rolls into the next
                      ranked trial (confirmation at the trial boundary)
✓/✗ Mark Good/Bad     labels the take (→ meta; dataset builder can keep only good)
🔄 Redo Attempt/Trial confirmation-guarded discard + re-record (mid-take, or the
                      armed post-stop escape hatch)
■  Stop Session       confirmation-guarded: finalizes everything; the PEER then
                      auto-rsyncs its whole session tree onto the master
```

All of these are **epoch-synchronized to both robots** over the link.

### Folder layout (identical on both machines → merges cleanly)
```
{output_dir}/{participant}_{date}_{time}/          ← shared session id
    {condition}/attempt{K}/robot{N}/               ← one leaf per robot per take
```

### Per-take file inventory (each robot leaf)
| File | Contents |
|---|---|
| `robot_raw_*.csv` + `.meta.json` | Every G1 joint: q, dq, ddq, tau_est, temperature, commanded goal @ ~50 Hz; meta: model path, EE offsets, ZED intrinsics (both eyes), clock-sync numbers, condition/attempt, quality label, tactile layout, wrench baseline, feedback map |
| `tactile_*.bin` | Full ~1300-taxel tactile arrays per frame (binary sidecar) |
| `body_pose_data_*.csv` (+ `csv_start_time.txt`) | Operator XR skeleton (all bones) + processed action targets per frame |
| `zed_stereo_*.mp4` | Operator's-eye stereo video (top-bottom composite, stream-copied — no re-encode) |
| `crate_pose_*.csv` / `crate_tags_raw_*.csv` | Fused crate world pose + re-fusable per-tag observations (master leaf) |
| `head_base_*.csv` | Robot base pose in the table frame over time |
| `feedback_events_*.csv` | Timestamped viewport/audio/vibrotactile on/off events — proof of condition delivery |
| `session_summary_*.csv`, `latency_summary.json` | Take summary + end-to-end latency statistics |

### Condition delivery
At each take start the manager sets the Quest app's feedback modalities to match the
take's condition (V / A+V / H+V / H+A+V), mirrors the UI toggles, and opens the
feedback-event log with that state — the recorded events prove what the participant
actually experienced.

---

## 8. Safety systems

| Layer | Trigger | Response |
|---|---|---|
| **Robot-comms watchdog** | `LowState` age > 0.3 s (e.g. bumped ethernet) | Pause teleop + arm damping (anti-snap); on recovery re-acquires low-level mode, re-syncs, requires manual resume; UI light turns red |
| **Stale-pose failsafe** | No fresh operator pose > 0.5 s (Quest/WebRTC/WiFi stall) mid-teleop | HOLD the commanded pose (stop chasing the frozen target); on recovery **blend** back to the live pose (≤1.5 s) instead of snapping |
| **Compliance / overcurrent softening** | Any arm joint's \|tau_est\| approaching its limit (e.g. crate jam) | Per-joint stiffness blend softens to yield, re-stiffens + returns when the force passes (rate-limited logging) |
| **Eco mode** | Arms idle (no operator motion) / shoulder-pitch temperature climbing | Idle auto-relax + a thermal governor that softens before the over-temperature fault (static-hold heat is gravity-torque/posture-driven) |
| **Abduction clamp** | Always on the final arm goals | Prevents shoulder heat-sink self-collision |
| **Head/neck keep-out** | Always, inside the SEW IK collision filter | A vertical **cylinder** (capsule) around the head covers the head-mounted Ostrich-neck gimbal + ZED — the arms are smoothly deflected (5 cm soft push, 2 cm hard stop) at a **felt** standoff of ~3 in beyond the head front/sides and ~6 in above it, so an operator raising hands to their own head (e.g. removing the headset mid-teleop) cannot drive the robot's hands into its neck mechanism (`g1_geometry.yaml`; sized including the hand-capsule radius + soft margin ≈ 11.5 cm outside the primitive) |
| **Start/shutdown choreography** | 🤖 Bootup Robot (manual, confirmed — the manager never moves the robot at launch) / Stop & Lower | Boot: abduct→raise-back→rise, clear of the table. Shutdown: arms sweep to a drooped A-pose with **straight** elbows (G1 elbow is straight at +76°, not 0°), then forward-tilt + damp — hands never cross the table or chest |
| **E-stop ergonomics** | Operator inactivity | Cursor auto-parks over the E-stop (suppressed while the calibration window is open); window sized so E-stop is always visible |
| **Robot / app separation** | — | E-STOP (double-click, instant damp) and Safe Shutdown (controlled descent) act on the **robot only** — the manager app stays open (recordings, data transfer, playback, peer link all survive; teleop needs a manager restart). The app is closed only by the separate confirmation-guarded **Quit Manager** button, whose graceful teardown finalizes sessions, stops cameras/link, and damps the robot on exit if it wasn't already down |
| **In-app re-boot** | After Safe Shutdown / E-STOP | **♻ Reboot** re-runs the bootup choreography and restarts the control loop with the session, camera anchors, calibration and peer link all preserved — mid-study recovery without a manager restart (pose + control reset only; firmware faults still need a G1 power cycle) |
| **Mid-study save/resume** | Auto at every session state change + 💾/📥 buttons | `study_resume.json` snapshots the open session (id, participant, pointer, recorded-takes ledger, trial order); after a manager relaunch — one machine or both — **📥 Resume Study** restores the ARMED session on both sides in one click (payload-carried state), completed takes stay locked out, and the file is removed on a clean Stop Session |
| **UI guards** | — | One-press-per-attempt / one-press-per-session; confirmation dialogs on Stop Session, Redo, and trial-boundary Continue; redo deletion fenced inside the session folder |

**Known limitation** (documented from a real incident): if the robot's ethernet drops,
G1 firmware can *autonomously* act (e.g. stiff stand-up) — no manager-side watchdog can
command a robot over a dead link. Physical mitigations (locking connectors, strain
relief, rig hardening) are the true first line of defense.

---

## 9. Latency engineering & telemetry

Teleop targets ~0.1 s end-to-end. The design rules that keep it there (each learned
from a measured regression):

1. **No per-frame vision in the manager process** — AprilTag detection and JPEG decode
   live in subprocesses (own GIL). ~30 in-process detections/s reproducibly caused
   multi-second lag.
2. **Drop-newest everywhere on the input path** — the freshest pose always wins;
   backlogs are structurally impossible.
3. **Qt change-detection** — periodic UI refreshes may only touch widgets whose
   rendered value changed (`setStyleSheet` ≈ ms-scale GIL each; ~800/s was the
   measured #1 GIL consumer during a lag incident).
4. **Unreliable-unordered WebRTC channels for streams** — reliable/ordered channels
   head-of-line-block on WiFi loss (0.5–2 s); pose/haptics streams must drop, not
   retransmit.
5. **Stream-copy recording** — ffmpeg `-c copy` from the already-encoded RTSP stream;
   recording adds no encode load.
6. **Serial/Modbus I/O off the control thread** — neck and hands are fed through
   worker threads with goal buffers.

Live telemetry: per-step IK timings, pose receive rate, WebRTC pipeline age
(`pipeline / bone_proc / stale_after` ms), a live latency graph widget, an
actuation-latency estimator (motor-response cross-correlation, on its own thread), and
a per-take `latency_summary.json`.

---

## 10. Playback & analysis tools

- **🧊 Digital twin** (`digital_twin.py`) — reconstructs the *whole workspace* in one
  MuJoCo scene: the table (STL at the world origin), the crate (tracked pose driving a
  free-joint STL), and **both robots** — each posed from its recorded joints and placed
  at its head-cam-localized base — synced on the absolute-epoch timeline with both
  robots' ZED videos (L | R per robot, robot 1 stacked above robot 2), a
  YouTube-style **seek bar** (+ ±5 s / ±30 s keys, speed, play/pause), and a togglable
  STL-nudge overlay (`o`) for CAD alignment.
  The crate track gets a playback-side clean-up: zero-phase smoothing, a
  **stationary freeze** (still segments pinned — measured 20× jitter reduction:
  30.5 → 1.5 mm/s median), and a **rest-on-table** z-bias correction derived from the
  crate STL's base height.
- **Session viewer** (`session_viewer.py`) — single-robot playback: MuJoCo robot +
  ZED video + estimated **end-effector wrench arrows** + tactile overlay.
- **Wrench estimation** (`wrench_estimation.py`, `wrench_playback.py`) — quasi-static
  EE wrench from recorded torques: `w = lstsq(Jᵀ, qfrc_bias − τ)`; live wrench panel in
  the manager is verification-only (known-mass accuracy check supported).
- **Dataset builder** (`build_dataset.py`) — one click from the manager: localizes the
  ZED per frame, time-aligns both robots (clock-sync meta), resamples to 15 Hz, crops
  the localization eye out of the stereo video (side-by-side *and* top-bottom layouts),
  decodes image observations, optionally keeps only good-labeled takes → training-ready
  **HDF5**.

---

## 11. Operating flow (summary)

**Per day**: robots + Jetsons on → managers on both laptops (launch touches no hardware) → **🤖 Bootup Robot** on each (confirmation-guarded; runs the physical bootup + starts the control loop) → link light green → Quests
connected (check "Unity command channel OPEN") → peer shows "Master link OK ✓".

**Per participant**: Pre-Flight Checklist → drag-rank trial order → **Prepare Study**
(all green) → set Participant ID → **Start Teleop (BOTH)** → **Start Session** → per
take: *Start Attempt → task → (Mark/Redo) → Stop Attempt → Continue* × 16 → **Stop
Session** → wait for the peer's "Sent to master ✓".

Teleop and recording are deliberately decoupled: engage teleop once per participant
(every engage/resume runs a blend ramp that would pollute take starts), record takes
inside it.

---

## 12. Module map

| File | Role |
|---|---|
| `XRT_user_study_manager.py` | The manager (UI, control, recording, link, safety) |
| `study_link.py` | Master↔peer TCP link (commands / status / candidates / clock) |
| `apriltag_tracker.py` | Crate tracker (subprocess server + standalone/debug modes) |
| `apriltag_crate_calibrate.py` | Crate bundle calibration → `crate_bundle.json` |
| `head_cam_localize.py` | Robot-base localization + detached detection worker + peer crate contribution |
| `g1_headcam_server.py` | Jetson head-camera TCP server (multi-client) |
| `zed_to_mediamtx.sh` / MediaMTX | ZED capture/encode/relay pipeline |
| `zed_localize.py` | Offline per-frame ZED pose from tag 0 |
| `digital_twin.py` | Two-robot workspace reconstruction + playback |
| `session_viewer.py` | Single-robot playback (video + wrench + tactile) |
| `wrench_estimation.py` / `wrench_playback.py` | EE wrench math / playback |
| `build_dataset.py` | Session → training-ready HDF5 |
| `inspire_fingertip_force_calibration.py` | Tactile calibration model (baselines, noise floors, gram fits) |
| `inspire_hand_haptics_streamer.py` | Live tactile → Quest haptics payloads |
| `xrt_devices.study` / `xrt_devices.study_process` | WebRTC device: pose intake, channels, body-pose CSV |

---

## 13. Diagram / presentation fact sheet

Everything below is stated elsewhere in this document; collected here as the raw
material for a system diagram or slide bullets.

**Nodes**: 2× operator (Quest 3) · 2× operator laptop (manager) · 2× G1 (MCU + Jetson +
head D435i) · 2× Inspire hand pair (tactile) · 2× Ostrich neck (Dynamixel) · 2× ZED
Mini · 2× external RealSense D435 (master only) · 2× chained WiFi routers · table
tag 0 · 11-tag crate.

**Links & rates**: Quest↔laptop WebRTC (pose ~60 Hz in; H.264 24 Mb/s video out;
haptics/audio/commands); laptop↔G1 DDS 500 Hz (USB-eth, 192.168.123.x); laptop↔Jetson
JPEG TCP :5005 (~30 fps, multi-client); laptop↔hands Modbus TCP 50 Hz; laptop↔neck
Dynamixel serial; master↔peer StudyLink TCP :5070 (commands @ shared epoch, status
2 Hz, crate candidates ~10 Hz, clock pings); peer→master rsync/SSH (post-session);
MediaMTX RTSP :8554; WebRTC signaling :8080.

**Compute split (per laptop)**: manager process (GUI 5 Hz change-detected · IK 60 Hz ·
send 500 Hz · BoneProcessor drop-newest · WebRTC asyncio · neck/hand/tactile/recorder
workers) + isolated processes (crate tracker 15 Hz fusion · head-cam detection 8 Hz ·
gst/NVENC · MediaMTX · ffmpeg -c copy).

**Frames**: world = table tag 0 (z = 0 at the tabletop) → robots via head-cam chain
(URDF + waist FK), crate via multi-camera bundle fusion (4 viewpoints incl. the peer's
over the link), ZED per-frame offline.

**Headline numbers**: ~0.1 s teleop latency (telemetry: pipeline/bone-proc/stale ms) ·
16 takes/participant (4 conditions × 4 runs) · ~1300 taxels/frame tactile · 500 Hz
robot state/command · 29 DoF/robot + 12 finger motors · 4/4 crate cameras · ±3 ms
cross-robot clock offset · crate playback jitter 30.5 → 1.5 mm/s (20×) after
smoothing/freeze · stereo recording 1280×1440@60 stream-copied · watchdog 0.3 s /
stale-pose 0.5 s safety thresholds · neck pitch ±60° · G1 elbow straight at +76°.

**Safety story (one line each)**: comms watchdog → damp; stale-pose → hold + blend
back; per-joint overcurrent compliance; thermal governor/eco; abduction clamp;
choreographed boot/shutdown; guarded destructive UI; documented firmware-autonomy
limitation with physical mitigations.
