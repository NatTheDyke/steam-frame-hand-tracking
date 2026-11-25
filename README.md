# steam-frame-hand-tracking — optical hand tracking for the Steam Frame

> [!WARNING]
> **Work in progress — and it makes the headset run hot.**
>
> - Hand tracking runs on the Frame's CPU on top of Valve's own streaming and tracking work, which
>   already keeps the CPU at roughly 70–85 °C. With tracking on, the hottest core regularly reaches
>   the high 80s and has hit **93–95 °C**. The headset **rebooted once** under heavy load during
>   development (the cause was not proven).
> - A built-in watchdog stops hand tracking if any CPU sensor stays at **≥ 93 °C for 3 seconds**. It
>   is a safety net, not a guarantee. Track **one hand** where you can (roughly half the work), and
>   watch the temperature on the panel.
> - Tracking is usable but laggy on fast wrist turns, can lose a hand, and may occasionally pick the
>   wrong hand. Heavy load has also caused faint vertical banding in passthrough. See
>   [REPORT.md](REPORT.md).
> - This modifies how SteamVR runs on the headset (a user-level Vulkan layer in the compositor).
>   Use at your own risk.

Use your **bare hands** in PC VR games streamed to the Steam Frame. The Frame's own front
passthrough cameras track your hands; the PC sees them as two Frame controllers (hand pose + finger
skeleton, no buttons). **Set a controller down and that hand takes over; pick it up again and you're
back on the controller.**

- Runs on the Frame itself; camera images never leave the headset, only hand poses (~150 bytes,
  60 times a second) go to the PC.
- Works with Valve's normal Frame streaming. Any app with Frame-controller bindings works.
- Status, measurements and limitations: **[REPORT.md](REPORT.md)**; latest latency work: [LATENCY_REPORT_V2.md](LATENCY_REPORT_V2.md); next research: [RESEARCH_BRIEF_V3.md](RESEARCH_BRIEF_V3.md).

## Hands in games: the tracking cameras (`--source slam`)

XRService pauses the passthrough cameras whenever room view is off, which is all the time inside a
game, so the passthrough source sees nothing there. `--source slam` uses the Frame's two lower
**tracking (SLAM) cameras** instead. They keep running whenever head tracking runs, including in
games, and see more of the space in front of you (wide fisheye: in a synthetic test 400/400 hand
positions in both cameras, against 351/400 for passthrough).

- `capture/slamgrab` reads the frames read-only from buffers XRService already owns (the reader never
  writes to them or pauses XRService). It publishes the scene exposures (30 Hz per camera, 1056x1024)
  to `/dev/shm/mrhand-slam` with their exact capture time, and drops the controller-LED exposures in
  between. Frame age when the tracker reads it: ~10 ms, ~8 ms of it inside XRService. CPU: ~4% of a core.
- It needs one capability, and only around the buffer handoff:
  `sudo setcap cap_sys_ptrace+p capture/build/slamgrab` (again after every rebuild; it does not work
  from a `nosuid` mount such as `/tmp`).
- `headset/start_args` selects it, and `start.sh` starts `slamgrab` instead of `ptgrab`. Use it with
  `tools/profile_slam.conf` as `~/.config/mrhand/profile`, the PC driver from this branch (protocol v4)
  and `pcdriver/framehands/framehands_live.ini` (prediction and world-space smoothing on).
- Status 2026-11-18: hands are tracked in Half-Life: Alyx with passthrough off. Smoother and less laggy
  than it was, but not yet good: position is 30 Hz, and orientation and fingers lag ~100+ ms.
  Off-centre hands are tracked less well.
- Layout facts the reader relies on, from XRService in SteamVR 2.19.2 on the headset: image and metadata
  buffers are allocated in pairs, right camera first. The reader refuses a layout it does not
  recognise rather than guessing.

## FrameNPU inference

Mercury tracking uses a pluggable FrameNPU inference backend with automatic
CPU fallback; `setup.sh --cpu` leaves it out. Available crops are submitted immediately
in groups of one to three; the worker never waits for a third crop.
On the headset it runs through a small root service (`headset/setup.sh` installs it), so
the tracker itself stays a normal user process; `headset/start_args` selects it with
`--merc-inference service` and falls back to CPU when the service is not installed.
Setup, selection and validation: [docs/FRAMENPU.md](docs/FRAMENPU.md).

## The panel

![Hand Tracking panel](overlay/panel_preview.png)

Open the SteamVR dashboard on the Frame and click **Hand Tracking** in the bottom bar.

| | |
|---|---|
| **LEFT / RIGHT** | Turn hand tracking on per hand. The first hand on starts the tracker; the last hand off stops it. |
| green **ON** | That hand is tracked and can take over from its controller. |
| red **OFF** | That hand always stays on its controller, and costs nothing. |
| amber **STOPPED** | Enabled, but the tracker isn't running (usually the thermal watchdog). Tap to restart. |
| status line | Which hands are on and the hottest CPU temperature. |

**One hand runs cooler.** A disabled hand isn't processed at all, so one hand is roughly half the
tracking work. The Frame already runs hot while streaming.

## Everyday use

1. Stream a game from the PC as usual.
2. Open the **Hand Tracking** panel and turn on the hand(s) you want.
3. Put that controller down somewhere and leave it, with your hand **up in front of you** where the
   front cameras can see it. After ~1 second the game's controller becomes your hand.
4. Pick the controller up and move it: you're back on the controller immediately.

How loose the set-down / pick-up rules are is in `pcdriver/framehands/framehands_live.ini`
(re-read every second, no restart):

| Setting | Default | Meaning |
|---|---|---|
| `stillSeconds` | 0.8 | how long the controller must lie still |
| `stillCm`, `stillDeg` | 2 cm, 6° | how still "still" is |
| `handAwayCm` | 10 cm | your tracked palm must be at least this far from the controller (a controller held still in your hand never switches) |
| `pickupCm`, `pickupDeg` | 4 cm, 15° | moving the controller this much returns to the controller |

### Smoothing

The wrist/finger smoothing can be changed live with `~/.config/mrhand/profile` (no restart; delete the
file to return to the default). Do not smooth position in the tracker (head frame): with the PC driver
from this branch, use `posMinCutoff` / `posBeta` in `framehands_live.ini` (world space) and
`tools/profile_slam.conf`. Head-frame smoothing makes a still hand follow head turns.
`pcdriver/tools/pose_recorder` records what SteamVR presents (Windows and Linux); hold a hand still and
turn your head to measure this. The default favours smooth slow movement with less lag than before.
For the previous, smoothest setting: `cp ~/handtrack/src/tools/profile_smooth_v1.conf ~/.config/mrhand/profile`.
Keys and measurements: [LATENCY_REPORT_V2.md](LATENCY_REPORT_V2.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Set the controller down, nothing happens | Is that hand **ON** in the panel? Is your hand in front of you (the cameras see roughly what passthrough shows)? Check `/tmp/live.log`: `hands/run 0.00` means no hand detected. |
| Hand tracking stopped by itself, panel shows STOPPED | Thermal watchdog (≥ 93 °C for 3 s). Let it cool, use one hand, tap to restart. Log: `/tmp/watchdog.log`. |
| Panel says on, PC shows nothing | PC driver not installed / SteamVR not restarted, or wrong `headsetHost` in the ini. The PC SteamVR log (`vrserver.txt`) shows `framehands stats: … packets/s`. |
| Hands wobble / lag on fast turns | Known: see REPORT.md (finger and wrist updates are ~13/s per hand). |

---

## Install

You need: a Steam Frame with SSH enabled, on the same network as the PC, and a sudo password for
`steamos` (set one with `passwd`). The camera reader knows the buffer layout of SteamVR 2.19.2 on the
headset and refuses a layout it does not recognise.

### Headset

```
git clone https://github.com/NatTheDyke/steam-frame-hand-tracking ~/handtrack/src
~/handtrack/src/headset/setup.sh
```

`setup.sh` checks before it changes anything, then builds the Python environment, the pinned libraries
([framenpu](https://github.com/NatTheDyke/framenpu),
[framenpu-cdsp](https://github.com/NatTheDyke/framenpu-cdsp),
[framehands](https://github.com/NatTheDyke/framehands), versions in [headset/PINS](headset/PINS)) and
the camera reader, and fetches the models by hash. The first run tells you how to make a two-second
SteamVR tracking recording, the only place your headset's own camera calibration can be read from.

Then start and stop tracking with `~/handtrack/start.sh` and `~/handtrack/stop.sh`, or add the
dashboard panel with `~/handtrack/src/overlay/install_overlay.sh`.

The models run on the Frame's DSP through a small root service, which takes most of the inference off the
CPU. It is experimental: it uses undocumented DSP access and shares the DSP with Valve's own tracking.
It works on one firmware only (SteamOS 0.5.8, CDSP ABI 9); on any other, setup uses the CPU by itself.
During development a fault left the DSP needing a headset reboot several times; the likely cause is fixed in
framenpu-cdsp 0.1.1. **`setup.sh --cpu`** leaves the DSP alone and runs the models on the CPU, which runs
hotter. `sudo headset/uninstall_npu_service.sh` removes the service. Details:
[docs/FRAMENPU.md](docs/FRAMENPU.md).

### PC (Linux SteamVR)

```
pcdriver/install.sh HEADSET_IP              # needs g++; close SteamVR first
```

Proton games use the native Linux SteamVR, so they see the hands too.

### PC (Windows, VS 2022 Build Tools)

```
pcdriver\build.bat
vrpathreg adddriver <repo>\pcdriver\framehands
```

Set `headsetHost` in `pcdriver\framehands\framehands_live.ini` to the Frame's IP, then restart
SteamVR. The driver DLL can only be replaced while PC SteamVR is closed.

### Updating

On the headset: `~/handtrack/stop.sh`, `git -C ~/handtrack/src pull`, then `setup.sh` again.
With the NPU service installed it then prints the `headset/install.sh` command that updates the
service to the same versions and keeps a rollback. Your `start_args` and `profile` are kept.

### Passthrough cameras (optional)

The default source is the tracking cameras, which also work in games. The passthrough source
(`--source passthrough`) needs the Vulkan layer: `capture/build.sh && capture/install.sh`,
`touch ~/.config/mrhand/enable-capture`, `systemctl --user restart steamvr`.

### Model licenses

The weights are not in this repository and are not MIT: the `mercury-ft7` student is CC BY-NC-SA 4.0,
and Mercury's own models come from Monado's repository, which publishes no license.

### Removing it

- Headset: panel → both hands OFF (or `~/handtrack/stop.sh`); `capture/uninstall.sh` removes the
  layer; `overlay/install_overlay.sh --remove` removes the panel; `headset/rollback.sh` removes the
  `camera/enableCamera` setting an early probe added.
- PC: `vrpathreg removedriver <repo>\pcdriver\framehands` (Linux: `vrpathreg.sh removedriver ...`).

---

## How it works

```
Frame:  passthrough cameras ─▶ vrcompositor ─(our passive Vulkan layer shares the camera buffers)─▶
        ptgrab (GPU downscale, 640² × 2) ─▶ tracker (optical flow 60 Hz + MediaPipe discovery +
        Mercury keypoints) ─▶ UDP hand packets
PC:     framehands SteamVR driver ─▶ two virtual Frame controllers (pose + 31-bone skeleton) ─▶ game
```

- The driver decides per hand when to switch (it sees the real controllers' positions), and reports a
  virtual controller as disconnected while you hold the real one, so apps never pick the wrong one.
- The panel writes the enabled hands to `~/.config/mrhand/hands` (`left`, `right`, `left right` or
  empty); the tracker re-reads it every second and sends the set to the PC in each packet.
- Details: [SYSTEM.md](SYSTEM.md) (architecture), [REPORT.md](REPORT.md) (measurements, latency,
  limitations), [FINDINGS.md](FINDINGS.md) (investigation log),
  [docs/CHANGES_headset.log](docs/CHANGES_headset.log) (headset change log).

| Path | Runs on | What |
|---|---|---|
| `capture/` | Frame | `handcam_layer.cpp`: passive Vulkan layer in vrcompositor that shares the camera buffers (opt-in flag + crash guard). `ptgrab.cpp`: reads them on its own GPU queue, downscales, publishes `/dev/shm/mrhand-pt`. `slamgrab.cpp`: tracking-camera frames from XRService's buffers -> `/dev/shm/mrhand-slam` |
| `handtrack/` | Frame | the tracker: `live.py` main loop, `fasttrack.py` optical flow, `merctrack.py`/`mercury_kp.py` Mercury keypoints, `detect.py`/`mpworker.py` MediaPipe, `palm.py` pose + UDP bridge, `watchdog.py` |
| `headset/` | Frame | `start.sh`, `stop.sh`, `rollback.sh`, `start_args` |
| `overlay/` | Frame | the Hand Tracking dashboard panel |
| `pcdriver/` | PC | SteamVR driver `framehands` (+ Valve's hand-simulation sample, BSD-3, in `third_party/`), live ini, pose recorder |
| `tools/` | both | telemetry report, replay/ablation harness, validation tests, model fetch |
| `probe/` | Frame | early camera-API probes |

Not in the repo: camera images, recordings and calibration copies (`data/`, they show the room),
models (`tools/fetch_models.sh`; the Mercury model's license is unverified, so it is fetched, not
redistributed), virtual environments, build outputs.

Separate project: the chroma-key passthrough (mr-chroma), not part of this repo.

## License

MIT (see [LICENSE](LICENSE)) for this project's code. `pcdriver/third_party/openvr_handsim` is Valve's
OpenVR SDK sample under its own BSD-3 license (see its NOTICE.md). Models are not included; they
are downloaded by `tools/fetch_models.sh` under their own terms.
