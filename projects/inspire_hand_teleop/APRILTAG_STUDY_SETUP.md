# AprilTag Study — Full Setup (cameras + manager linking)

End-to-end setup for the dual-G1 AprilTag crate-tracking study: external RealSense
cameras, the G1 head cameras (served from each robot's Jetson), the cross-internet
link between the two user-study managers (plain TCP over the master's reachable IP),
and clock sync (NTP).

Everything fuses into **one coordinate frame with the table AprilTag (id 0) at the
origin**. Tags: `tag36h11`, **65 mm**, id 0 = table/world origin, ids 1–10 = crate
faces (id 10 = inside-centre-base = crate body origin).

## Machines

| Machine | Role | Cameras it uses |
|---|---|---|
| **Robot 1 laptop** | MASTER (crate fusion + digital twin) | 2 external D435 (USB) + robot 1 head cam (over robot 1's Jetson) |
| **Robot 2 laptop** | PEER | robot 2 head cam (over robot 2's Jetson) |
| **Each G1 Jetson** | head-camera server | the G1's built-in head D435i |

Repo path below is assumed to be `~/proj_XRT`; the scripts live in
`~/proj_XRT/projects/inspire_hand_teleop`.

---

## 1. Both laptops — Python deps (one time)

These are additive; they don't disturb the existing teleop stack.

```bash
cd ~/proj_XRT
# activate the project venv first (however you normally do), then:
pip install pupil-apriltags pyrealsense2
```
(`scipy`, `opencv-python`, `mujoco` are already in the env.)

---

## 2. Networking the two managers (the link)

The link is **plain TCP** — no VPN required. **Which machine "listens" is decoupled
from which is Master:** in the Setup bar, the machine with the reachable/open IP
leaves **Link IP blank** (it listens); the other machine enters that listener's IP
(it connects). Control role (Master/Peer) is independent.

> **eduroam caveat:** two devices on eduroam usually **cannot reach each other**
> (client isolation blocks device-to-device; different APs put you on different
> NAT'd subnets). So a direct socket between two eduroam clients typically fails —
> prefer Option A.

### Option A (preferred for co-located robots) — a private cable/switch
The robots sit at the same table, so link them **off eduroam entirely**: an Ethernet
cable between the two machines, or both into a cheap unmanaged switch, on a private
subnet (e.g. `192.168.50.1` / `.2`) — a 3rd interface alongside eduroam (internet)
and each G1's Jetson link. Fast, private, and it never touches the institutional
network (no IT concern).

Make the **laptop (robot 1, master) the LISTENER** and the **IT PC (robot 2, peer)
CONNECT out to it** — locked-down machines usually allow outbound connections but
restrict inbound, so the IT PC only needs an *outbound* connection (no inbound port
to open on their side):

| Machine | Private IP | Setup-bar **Link IP** | Direction |
|---|---|---|---|
| Robot 1 laptop (master) | `192.168.50.1` | **blank** | listens (you control it; `sudo ufw allow 5070/tcp`) |
| Robot 2 IT PC (peer)    | `192.168.50.2` | **`192.168.50.1`** | connects out to the laptop |

- *Only IT ask:* a **static IP on a second NIC** of their PC for this private link
  (no inbound port needed). They offered to help with networking — this is the ask.

### Option B — direct to the IT desktop's static IP (test it first)
If the IT PC's static IP is reachable from the laptop's eduroam connection, point the
link at it. Whichever machine has the reachable IP **listens** (Link IP blank); the
other connects to it. Open the port on the listener if a host firewall is active
(`sudo ufw allow 5070/tcp`). Test from the connecting machine:
```bash
ping <listener-ip>                  # routing
nc -vz <listener-ip> 5070           # port reachable -> "succeeded"/"open"
```
If `ping` fails, eduroam has no route to that IP (common — eduroam is often
internet-egress-only) → use Option A or C.

### Option C — reverse SSH tunnel (if SSH works but the port isn't routable)
From the connecting machine: `ssh -N -L 5070:localhost:5070 <user>@<listener-ip>`,
then set that machine's **Link IP = 127.0.0.1**. Fully logged; no VPN.

### Option D — Tailscale (ONLY if IT approves it)
It can bypass network monitoring, so confirm with IT first. If approved, install with
the single-use auth key from your admin console (Add device → Linux server):
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --authkey tskey-auth-xxxxxxxxxxxx
tailscale ip -4                     # the listener's Link IP = its 100.x.y.z
```

---

## 3. Both laptops — NTP / clock sync (required for same-timestamp recording)

Aligns both managers' clocks to ~ms so a "start at epoch T" command records in
lockstep. Use chrony (tighter than the default `timesyncd`).

```bash
sudo apt update && sudo apt install -y chrony
sudo systemctl enable --now chrony
# verify (do this on BOTH laptops):
timedatectl            # expect: System clock synchronized: yes / NTP service: active
chronyc tracking       # 'System time' offset should be a few ms or less
```

Both reaching the public NTP pool is enough — they don't need to point at each other.

---

## 4. Each G1 Jetson — head-camera server

The G1's head D435i is wired to the Jetson, not the laptop, and the robot does NOT
expose it via the SDK `VideoClient` (returns error 3102). So a small server on the
Jetson streams its frames to the laptop. Do this on **both** robots' Jetsons.

### 4a. Copy the server to the Jetson (from the laptop)
```bash
scp ~/proj_XRT/projects/inspire_hand_teleop/g1_headcam_server.py \
    unitree@<jetson-ip>:~/        # robot 1 Jetson is typically 192.168.123.164
```
Log in as the Jetson's `unitree` user (see Unitree's G1 documentation; change the factory password).

### 4b. Free the camera from the SONIC server + verify (on the Jetson)
The head D435i is held by NVIDIA's `composed_camera_server.service` (the SONIC /
`proj_vla_nav` stack). It must be stopped for our server to open the camera.
```bash
ssh unitree@<jetson-ip>
sudo systemctl stop composed_camera_server.service
sudo fuser -v /dev/video*                 # should now list NO process
rs-enumerate-devices -s                    # confirm the D435I + note its serial
python3 -c "import pyrealsense2, cv2; print('deps ok')"
```

### 4c. Install the server as a boot service (on the Jetson)
So it auto-starts on every reboot — no SSH per session.
```bash
sudo tee /etc/systemd/system/g1-headcam.service > /dev/null <<'EOF'
[Unit]
Description=G1 head-camera AprilTag frame server
After=network-online.target
Wants=network-online.target
Conflicts=composed_camera_server.service

[Service]
Type=simple
User=unitree
WorkingDirectory=/home/unitree
ExecStart=/usr/bin/python3 /home/unitree/g1_headcam_server.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

# stop SONIC from grabbing the camera at boot, enable ours:
sudo systemctl disable --now composed_camera_server.service
sudo systemctl daemon-reload
sudo systemctl enable --now g1-headcam.service
systemctl status g1-headcam.service --no-pager      # expect active (running)
```
- Confirm interpreter deps if status fails: `which python3` must be the one with
  `pyrealsense2`+`cv2`; edit `ExecStart=` if it's not `/usr/bin/python3`.
- Watch logs: `journalctl -u g1-headcam -f`

### 4d. Test the stream from the laptop
```bash
cd ~/proj_XRT/projects/inspire_hand_teleop
python head_cam_localize.py --host <jetson-ip> --table-tag-size 0.065
```
Aim the head at table tag 0 → expect `up ≈ +1.0` (green, "UPRIGHT ok"). Keys:
`o` optical convention, `f` z-flip, `s` save, `q` quit. **Confirmed defaults on this
rig: optical = `ros`, table-z-down = on** (both are the defaults; no flags needed).

### 4e. REVERT to the SONIC server (when you're done with the study)
```bash
ssh unitree@<jetson-ip>
sudo systemctl disable --now g1-headcam.service
sudo systemctl enable --now composed_camera_server.service
sudo fuser -v /dev/video*            # confirm SONIC's python now holds the camera
```
(The `Conflicts=` line means starting one always stops the other, so they never
fight over the camera.)

---

## 5. Robot 1 laptop — external cameras + crate calibration

### 5a. Plug in & identify the 2 external D435s (USB3)
```bash
cd ~/proj_XRT/projects/inspire_hand_teleop
python apriltag_probe.py --list      # lists laptop-USB RealSense serials
python apriltag_probe.py --serial <SN>   # open one to confirm it's an external (q to quit)
```
(The head cam is NOT here — it's on the Jetson.)

### 5b. Calibrate the crate tag bundle (one time per crate build)
Uses BOTH external cameras for faster, multi-angle coverage. Produces
`crate_bundle.json` next to the script (also reachable from the manager's
🔧 Recalibrate Crate button).
```bash
python apriltag_crate_calibrate.py        # both externals, ids 1-10, anchor id 10
# SPACE = capture; rotate the crate slowly so every face overlaps a neighbour;
# watch "not-yet-linked" empty out; s = solve+save; q = quit.
```
Copy `crate_bundle.json` to **robot 2's laptop** too (it needs the same bundle to
contribute crate candidates once linking is enabled):
```bash
scp crate_bundle.json <robot2-host>:~/proj_XRT/projects/inspire_hand_teleop/
```

---

## 6. Running the managers

**No AprilTag/link/head-cam flags needed** — everything is configured in the UI and
**persisted per-machine** to `~/.xrt_study_config.json`. Launch exactly as before:

```bash
cd ~/proj_XRT
python projects/inspire_hand_teleop/XRT_user_study_manager.py --net <your-iface>
```

### Configure once per machine in the Setup bar (top of the window)
| Control | Robot 1 (master) | Robot 2 (peer) |
|---|---|---|
| **Robot** | `1` | `2` |
| **Head cam (Jetson) IP** | robot 1's Jetson IP | robot 2's Jetson IP |
| **Fuse head cam into crate** | ✓ | ✓ |
| **Link** | `Master` | `Peer` |
| **Link IP** | **blank** (laptop listens) | the laptop's link IP, e.g. `192.168.50.1` (IT PC connects out) — see §2 |
| **🔗 Connect Link** | click to (re)start the link after editing | same |

These persist, so on the next launch they're already set (the link auto-connects on
startup if a role is saved). CLI flags (`--link-role`, `--headcam-host`, …) still
exist as optional one-off overrides but are not needed. Universal defaults (not in
the UI): optical = `ros`, table-z-down on, tag size 65 mm, link port 5070.

### Operate
- **Robot 1 (master):** 🎯 Start Crate Tracking → 📍 Start Head-Cam Localize →
  📐 Anchor Cameras (tag 0 visible to all cameras). The **Master Control window**
  opens automatically — use it to drive both robots.
- **Robot 2 (peer):** 🎯 Start Crate Tracking (head-cam only) → 📍 Start Head-Cam
  Localize → 📐 Anchor Cameras. It then obeys the master and streams its crate
  candidates to robot 1.

### Master Control window (robot 1 only)
- **Peer status** line: connected? + robot 2's teleop / recording / base-up.
- **▶/■ Teleop (BOTH)** — start/stop teleop on both at once.
- **●/■ Recording (BOTH)** — start/stop recording on both **at a shared timestamp**
  (`now + 0.8 s`); the master's Participant ID + Trial are pushed to the peer so both
  write the same `session_label`. **This is why both machines must be NTP-synced
  (section 3).**

---

## 7. Per-session operating order

1. Power up both robots; Jetson head-cam servers auto-start (section 4c).
2. Launch both managers (section 6) — the link auto-connects from saved config.
   Confirm the master's **Master Control** window shows **Peer: CONNECTED**.
3. On **each** robot: 🎯 Start Crate Tracking → 📍 Start Head-Cam Localize →
   📐 **Anchor Cameras** (tag 0 visible to every camera).
4. Confirm status: Crate "tracked", Base "localized (up≈+1)"; master shows the
   peer's teleop/recording state.
5. Drive the session from the **Master Control window**: Start Teleop (BOTH), then
   Start Recording (BOTH) — both begin at a shared timestamp.

### Verify the link is healthy (once networking + NTP are up)
- `nc -vz <master-ip> 5070` from robot 2 succeeds; master shows **Peer: CONNECTED**.
- After a synchronized Start Recording, compare `record_start_epoch` in each robot's
  `meta.json` — they should match within a few ms (the NTP payoff).
- Robot 2's crate estimate appears in robot 1's fusion as a `remote` source (the
  crate `Δcam` / tag count reflects the extra input).

---

## 8. Calibration files — per-robot vs shared (don't cross-contaminate)

Each robot keeps its OWN copy of the per-robot calibrations; only the crate bundle
is shared (same physical crate). **All of these are git-ignored**, so they are not
committed and won't sync between robots via git — generate or copy them per the
rules below.

| File (in `projects/inspire_hand_teleop/`) | Scope | Rule |
|---|---|---|
| `fingertip_force_calibration.json` | per-robot (that hand's sensors) | each robot generates its own — **never copy** |
| `kneeling_pose_calibration.json` | per-robot | each robot generates its own — **never copy** |
| `arm_abduction_limit.json` | per-robot | each robot generates its own — **never copy** |
| `robot_base_from_headcam.json` | per-robot (base pose in table frame) | each robot generates its own — **never copy** |
| `robot_base_calibration.json` | per-robot (touch-cal base pose) | each robot generates its own — **never copy** |
| `crate_bundle.json` | per-**crate** (same crate both robots) | calibrate once on robot 1, **scp to robot 2** |
| `~/.xrt_study_config.json` | per-machine (robot id, head-cam IP, link role/host, fuse-head-cam) | set in the UI Setup bar on each machine; lives in `$HOME`, not the repo (auto-migrates the old `~/.xrt_study_robot_id.json`) |

**Not files** (shared CLI defaults, identical for both robots): `--headcam-optical ros`,
`--table-z-down` (on), `--table-tag-size 0.065` — same camera mount on identical
robots and the same shared table, so there's nothing per-robot to leak.

If you ever find a per-robot file *tracked* by git (e.g. it was committed before
being ignored), untrack it and push:
```bash
git rm --cached projects/inspire_hand_teleop/<file>.json
git commit -m "stop tracking per-robot calibration" && git push
```
On a freshly-set-up robot, **delete any per-robot `*.json` that arrived via an old
pull** so that robot regenerates its own; copy **only `crate_bundle.json`** between
robots.

---

## 9. Troubleshooting

- **`Device or resource busy` opening the head cam on the Jetson** → the SONIC
  server still holds it: `sudo systemctl disable --now composed_camera_server.service`
  (section 4b).
- **`Get image sample error. code: 3102`** → expected; the G1 doesn't serve the head
  cam over the SDK VideoClient. Use the Jetson server (section 4).
- **`Crate: not anchored` / `Base: searching`** → tag 0 not visible to that camera
  when you anchored; reposition so id 0 is visible, then Anchor again.
- **`up ≈ -1` in head-cam localize** → wrong world flip; ensure `--table-z-down`
  (default on). `up ≈ +0.5` (tilted) → wrong optical convention; use `ros` (default).
- **Native "more than one new minima found." spam** → harmless; already silenced by
  default (`APRILTAG_VERBOSE=1` to see it).
- **Two managers can't reach each other** → from robot 2: `ping <master-ip>` (routing)
  and `nc -vz <master-ip> 5070` (port). Open `5070/tcp` on the master's firewall;
  if on different subnets, have IT allow inbound TCP 5070 to the desktop. See §2.
