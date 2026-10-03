# omeye

Look at a window and it gets focus. omeye watches you through a webcam, works out which Hyprland window you are looking at, and focuses it. It replaces `Super + arrow` or reaching for the mouse. Toggle it on and off with `omeye toggle` (or a key bound to it).

Status: version 0.1. The full pipeline runs, including from a phone camera streamed through Flux, but it has not tracked a real face yet. On lunar-gouda the lid is shut, and the first phone stream only sent black frames.

## How it works

```
webcam frame ─► face + iris landmarks ─► gaze features ─► point on screen ─► window ─► focus
  (OpenCV)      (MediaPipe Face           (iris inside each   (calibrated       (Hyprland   (after a 0.4 s
                 Landmarker, 478 points)   eye + head pose)    regression)       layout)     dwell)
```

1. **Features.** For each eye, where the iris sits between the corners (`u`, across) and between the lids (`v`, down), in eye widths, so head roll cancels out. Both eyes are averaged. Head yaw, pitch and position relative to the camera come from MediaPipe's face transformation matrix.
2. **Calibration** (`omeye calibrate`) shows 15 dots. While you look at each one, omeye records about 30 frames. A ridge regression maps the features (plus `u²`, `v²`, `uv`) to monitor fractions. Leave-one-dot-out cross-validation picks the regularisation and reports the error you can expect.
3. **Smoothing.** A One Euro filter smooths the gaze point a lot while it holds still and very little when it jumps. Blink frames (eyelids below 60% of your usual opening) are skipped.
4. **Picking a window.** Gaze has to be 50 pixels inside a window's edge before that window counts, and the focused window extends 50 pixels past its own edges. That hysteresis stops jitter along a border from flipping focus.
5. **Switching.** The same window must stay under your gaze for 0.4 s; glances away of up to 0.15 s (blinks) don't restart the count. Switching pauses while you type (any input in the last 0.7 s, through the Wayland idle-notify protocol, so no access to `/dev/input` is needed) and for 2 s after the mouse moves. The mouse always wins.

A webcam gives gaze to roughly 2–4° (3–5 cm at arm's length). That's plenty for choosing between tiled windows on a 32-inch screen, and not enough to aim at buttons.

## Setup

1. Put a camera on top of the monitor, centred and facing you. Either:
   - a plug-in webcam: any 1080p one works, and infrared isn't needed; or
   - a phone running Flux (`omarchy-flux`), started as a webcam from the phone. It appears as the virtual camera "Flux Camera".

   With `camera = "auto"`, omeye takes a camera that is sending video, and prefers anything to the laptop's built-in one. `omeye cameras` shows what it sees.
2. Install:

   ```bash
   sudo pacman -S --needed gtk4-layer-shell python-gobject python-cairo wayland-protocols v4l-utils
   ./install.sh
   ```

   `install.sh` creates the Python environment (`uv sync`), downloads the face landmark model to `~/.local/share/omeye/models/`, builds the input-activity helper (`make`), links `omeye` into `~/.local/bin` and installs the `omeye.service` systemd user unit. The unit is not started at login.
3. Calibrate: `omeye calibrate`. Sit as you normally do, press Space, and follow the dots with your eyes (about 30 s). Recalibrate after moving the camera or your chair.
4. Check: `omeye preview` draws a ring where omeye thinks you are looking (green when it is on the focused window) without changing focus. `omeye preview --switch` changes focus too.
5. Use: `omeye on`, `omeye off`, `omeye toggle`.

## Commands

| Command | What it does |
|---|---|
| `omeye on` / `off` / `toggle` | Start or stop the service, with a desktop notification |
| `omeye status` | Service state, calibration age and error, chosen camera |
| `omeye calibrate [--points N] [--monitor NAME]` | Dot calibration (9, 12, 15, 20 or 24 dots) |
| `omeye preview [--switch]` | Show the gaze point; `--switch` also changes focus |
| `omeye run [--preview] [--dry-run] [-v]` | The tracking loop in the foreground (what the service runs) |
| `omeye cameras` | List cameras and mark the one in use |
| `omeye bench [--seconds S]` | Landmark speed and processor load |

`calibrate`, `preview` and `bench` pause the service while they use the camera, then start it again.

The service stays with the camera it was calibrated with. If that camera isn't sending video (the phone stream is off), the service waits for it and picks it up when it comes back. It notices a stream that stops within about 2 s.

## Settings

Optional, in `~/.config/omeye/config.toml`. Every key and its default is in [`omeye/config.py`](omeye/config.py). The ones you are most likely to change:

```toml
camera = "auto"          # or "/dev/video2", or part of the camera's name
delegate = "cpu"         # "gpu" runs the landmark model on the integrated GPU
dwell_ms = 400           # how long to look before focus moves
typing_grace_ms = 700    # no switching until this long after the last key
mouse_grace_ms = 2000    # the mouse wins for this long after it moves
margin_px = 50           # border hysteresis
```

## Processor load

Measured on a Ryzen 7 5800H at 30 frames per second (fps), 1280×720, with a test portrait replayed as the input:

| Where the model runs | Face in view | Nobody there |
|---|---|---|
| central processing unit (CPU) | 9.7 ms per frame, 31% of one core | 8% |
| integrated graphics processing unit (GPU), Radeon Vega | 9.7 ms per frame, 23% of one core | 8% |

After 3 s without a face, omeye checks only every sixth frame. The service runs at `Nice=10`.

## Traps found while building this

- **MediaPipe was killed with SIGKILL at start-up.** `mediapipe` imports `sounddevice` for its audio tasks. Initialising PortAudio goes through the Advanced Linux Sound Architecture (ALSA) into PipeWire, whose realtime module leaves the process with a realtime CPU-time limit of 0, and the kernel kills it once inference starts. omeye has no audio, so `tracker.py` installs an empty `sounddevice` module before importing MediaPipe.
- **The GPU delegate silently ran in software.** MediaPipe opens the first render node, which on this laptop is the NVIDIA card. Mesa can't drive that, so it fell back to `llvmpipe`. With `delegate = "gpu"`, omeye sets `DRI_PRIME` to the first non-NVIDIA Peripheral Component Interconnect (PCI) device (here `pci-0000_07_00_0`, the Radeon).
- **A virtual camera only offers video while something feeds it.** The Flux camera is a v4l2loopback device. With no stream it accepts video in but offers none out, and OpenCV then can't open it. `tracker.py` asks each device what it can do (`VIDIOC_QUERYCAP`) and skips idle ones. When a stream stops, the next read blocks for OpenCV's default 10 s; `OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT=2` shortens that, because OpenCV doesn't support `CAP_PROP_READ_TIMEOUT_MSEC` for V4L2.
- **Hyprland 0.56 has Lua dispatchers.** Focus is `dispatch hl.dsp.focus({ window = "address:0x…" })` on the request socket. `hypr.py` falls back to the old `focuswindow` form on older releases.
- **Overlay.** The calibration and preview overlay is a GTK 4 layer-shell surface. It needs the system Python (PyGObject) and `LD_PRELOAD=/usr/lib/libgtk4-layer-shell.so`, so `overlay_client.py` starts it as a separate process and talks to it in lines of text on standard input and output.

## Layout

```
omeye/            Python package
  cli.py          commands
  daemon.py       tracking loop
  calibrate.py    dot calibration
  tracker.py      camera, MediaPipe, gaze features
  model.py        calibrated regression
  focus.py        window hit test and dwell logic
  filters.py      One Euro filter
  activity.py     typing and mouse activity
  hypr.py         Hyprland socket
  overlay.py      GTK 4 overlay (system Python)
src/omeye-idle.c  input-activity helper (ext-idle-notify-v1)
systemd/          user unit template
tests/            pytest: uv run pytest
```

## Ideas for later

- Learn from clicks: when you click, you are almost always looking at the pointer, so each click is a free calibration sample.
- A bar indicator and toggle as an Omarchy plugin, next to `rb.monitor` and `rb.overview` in `omarchy-rb-plugins`.
- Gaze across several monitors (calibration currently covers the one the camera sits on).
