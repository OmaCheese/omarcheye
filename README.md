# omeye

Look at a window and it gets focus. omeye watches you through a webcam, works out which Hyprland window you are looking at, and focuses it. It replaces `Super + arrow` or reaching for the mouse. Toggle it on and off with `omeye toggle` (or a key bound to it).

Status: version 0.5. It runs on lunar-gouda with a OnePlus 13 streamed through Flux as the camera; the latest calibration (rich features, 15 dots) has a cross-validated error of 6.4% of the screen width (about 4.5 cm). Settings are tuned by replaying the saved calibration frames, not yet by live use.

## How it works

```
webcam frame ─► face + iris landmarks ─► gaze features ─► point on screen ─► window ─► focus
  (OpenCV)      (MediaPipe Face           (iris and lids,     (calibrated       (chance of  (99%: now;
                 Landmarker, 478 points)   eye-direction       regression,       each window 90%: 0.25 s)
                                           scores, head pose)  kept on screen)   in the layout)
```

1. **Features.** Two sets are recorded, and calibration keeps whichever predicts better on your data. **Basic:** where the iris sits between the eye corners (`u`, across) and across the corner line (`v`, down), in eye widths so head roll cancels out, with both eyes averaged. **Rich:** each eye's iris separately, each eye's upper and lower lid position (the upper lid follows the eye up and down, which helps the weak vertical direction), and MediaPipe's eight eye-direction scores (`eyeLookUp/Down/In/Out`, per eye). Both sets include head yaw, pitch and position relative to the camera, from MediaPipe's face transformation matrix.
2. **Calibration** (`omeye calibrate`) shows 15 dots. While you look at each one, omeye records about 30 frames. A ridge regression maps the features (plus `u²`, `v²`, `uv`) to monitor fractions. Leave-one-dot-out cross-validation picks the regularisation and reports the error you can expect. Calibration also fits the geometric model (below), and keeps whichever of the three (basic regression, rich regression, geometric) has the lowest cross-validated error.
3. **Refining** (`omeye refine`, optional) adds pointer samples: you move the mouse slowly and keep your eyes on the pointer. Frames count only while the pointer has rested within 2% of the screen width for 0.4 s, so camera lag and the eyes trailing a moving pointer don't matter. The samples join the dot samples and the model is refitted. The screen is split into 6×4 cells; each cell is held out in turn, so the error reported is for places the fit didn't learn from. The new fit is kept only if it beats the old calibration on the same samples. Running `refine` again adds more.
4. **On screen, and steady.** A predicted point slightly outside the monitor is clamped to its edge. One more than 15% outside means you are looking away, and it doesn't count. A One Euro filter smooths the point the preview shows: a lot while it holds still, very little when it jumps. (A fixation filter replaced it for a while; replayed on the calibration frames it was no steadier and lagged up to 0.5 s behind the eyes, so it went.) Blink frames (eyelids below 60% of your usual opening) are skipped.
5. **Using the layout.** A predicted point is uncertain by about the calibration's error, so omeye treats it as a blob, not a dot. For each window on screen it works out how much of the blob falls inside: that's the chance you are looking at that window. Floating windows on top claim their area first, and what falls in gaps or outside the windows counts as looking away. Looking into the middle of a big window gives it nearly all the chance; a point near a border splits it. A running belief combines the frames, assuming your gaze usually stays put and moves between windows about 1.5 times a second at most. Each frame is softened because consecutive frames share most of their error.
6. **Switching.** How long omeye waits depends on how sure it is. At 99% focus moves at once, which happens within a few frames when you look clearly into a window: about 0.1 s in simulation, 0.2 s replayed on real calibration data for side-by-side or 2×2 windows. Otherwise a window needs 90% for 0.25 s. A flick of the eyes (two frames) doesn't switch, and staring at the border between two windows flips focus about three times a minute. Windows stacked top and bottom stay slower (about 0.5 s on the replay): up-down is the weaker direction for a camera above the screen, so the evidence rarely reaches 99%. Switching pauses while you type (any input in the last 0.7 s, through the Wayland idle-notify protocol, so no access to `/dev/input` is needed) and for 2 s after the mouse moves, but the belief keeps tracking. The mouse always wins. In `omeye preview` the ring sits at the centre of the most likely window, outlined with how likely it is; the small dot is the steady gaze estimate itself. The colour says what the service would do: green, the focused window; amber, sure enough to switch (the label says when typing or the mouse holds it back); white, only the likeliest. With `--verbose` (the service's default) the log names each switch with its probability, and every 30 s says how long a ready switch was held back by typing or the mouse.

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
3. Aim the camera: `omeye camera` shows its view. Your face should sit in the middle with a green outline. Then calibrate: `omeye calibrate`. Sit as you normally do, press Space, and follow the dots with your eyes (about 30 s). Recalibrate after moving the camera or your chair.
4. Optional, and worth it: `omeye refine` (up to 60 s). Move the mouse slowly over the screen, resting it here and there, with your eyes on the pointer. Cells turn green as they fill; Enter finishes early.
5. Check: `omeye preview` draws a ring where omeye thinks you are looking (green when it is on the focused window) without changing focus. `omeye preview --switch` changes focus too.
6. Use: `omeye on`, `omeye off`, `omeye toggle`.

## Commands

| Command | What it does |
|---|---|
| `omeye on` / `off` / `toggle` | Start or stop the service, with a desktop notification |
| `omeye status` | Service state, calibration age and error, chosen camera |
| `omeye calibrate [--points N] [--monitor NAME]` | Dot calibration (9, 12, 15, 20 or 24 dots); starts the samples afresh |
| `omeye test` | 9 dots between the calibration ones: how good the calibration is now, in your current posture, and how each kind of model fitted on your samples does; Enter switches to a clearly better one |
| `omeye refine [--seconds S]` | Follow the mouse pointer with your eyes; adds samples and refits |
| `omeye preview [--switch]` | Show the gaze point; `--switch` also changes focus |
| `omeye run [--preview] [--dry-run] [-v]` | The tracking loop in the foreground (what the service runs) |
| `omeye camera` | Show what the camera sees, with the tracking drawn on; Ctrl+C closes it |
| `omeye cameras` | List cameras and mark the one in use |
| `omeye bench [--seconds S]` | Landmark speed, processor load, eye size in pixels and landmark jitter (look at one spot while it runs) |

The camera view also appears, large, on the start screens of `calibrate` and `refine`, and in the bottom-right corner during `preview`. It draws a dot per face landmark, circles on the irises and the face outline: green when the face is well placed, amber near an edge of the image, red when it's cut off. Below it: the eye's width in pixels, brightness, frames per second and what to do about the camera's aim. Frames go to the overlay in memory and are never written to disk.

`calibrate`, `refine`, `preview`, `camera` and `bench` pause the service while they use the camera, then start it again.

The service stays with the camera it was calibrated with. If that camera isn't sending video (the phone stream is off), the service waits for it and picks it up when it comes back. It notices a stream that stops within about 2 s.

## Settings

Optional, in `~/.config/omeye/config.toml`. Every key and its default is in [`omeye/config.py`](omeye/config.py). The ones you are most likely to change:

```toml
camera = "auto"          # or "/dev/video2", or part of the camera's name
delegate = "cpu"         # "gpu" runs the landmark model on the integrated GPU
dwell_ms = 250           # the likely window must stay confident this long
confidence = 0.9         # how sure omeye must be before focus moves (higher: fewer flips on borders)
quick_confidence = 0.99  # this sure, and focus moves after quick_ms (0) instead; 0.98 is faster, a little less safe
switch_rate = 1.5        # expected gaze moves between windows per second (higher: follows faster, flips more)
typing_grace_ms = 700    # no switching until this long after the last key
mouse_grace_ms = 2000    # the mouse wins for this long after it moves
offscreen = 0.15         # further outside the monitor than this counts as looking away
```

## Geometric model

The regressions learn how features map to the screen within the postures seen during calibration, and extrapolate when you sit differently. The geometric model (`geometry.py`) follows the physical setup instead, in millimetres and in the camera's coordinates:

- **Screen:** a rectangle of the size the monitor reports (700 × 390 mm here).
- **Camera:** `dx` mm right of the screen's centre, `lift` mm above its top edge, tilted by `tilt` and `pan`.
- **Your eye:** where MediaPipe puts your head, times a scale `s`. MediaPipe assumes a 63° lens; a phone's differs, which scales every distance it reports. On this setup it put the face at 47 cm.
- **Your gaze:** a ray along the head's forward axis, turned by the eye's own rotation: `kx·u + ox` across and `ky·v + kl·lid + oy` up.
- **Where you look:** where the ray meets the screen plane.

Calibration fits those ten numbers (Levenberg-Marquardt, weak priors for the directions the dots can't separate, mirrored image tried both ways). Moving your head is then handled by geometry rather than extrapolation. On a simulated desk (calibrate at 65 cm, then move), the regression went from 2.2% to 5.4–5.7% of the screen width when sitting 9 cm closer or moving in several directions, while the geometric model stayed at 1.1–1.2%. That assumes the eye model holds for real eyes, which `omeye test` checks on yours.

## Processor load

Measured on a Ryzen 7 5800H with the OnePlus 13 streaming 1920×1080 at 30 frames per second (fps) through Flux, face in view:

| Where the model runs | Time per frame | Processor use |
|---|---|---|
| central processing unit (CPU) | 9.2 ms | about 36% of one core |
| integrated graphics processing unit (GPU), Radeon Vega (`delegate = "gpu"`) | 13.1 ms | about 40% of one core |

Before OpenCV was pinned to one thread (see Traps), the same runs used 170–190% of a core. After 3 s without a face, omeye checks only every sixth frame. The service runs at `Nice=10`.

At 1080p the face is already larger than the 256-pixel crop MediaPipe's landmark model works on, so 1080p mostly costs conversion time; whether it steadies the landmarks is what `omeye bench` (look at one spot) measures.

## What the first calibration data showed

Analysis of the first 1,342 calibration frames (15 dots plus 24 pointer cells, phone camera at 720p), with each dot or cell held out in turn:

| | Error, share of the screen width |
|---|---|
| Model as built (basic features) | 6.6% |
| Steady offset per spot / frame-to-frame jitter | 5.8% / 3.5% |
| Average of 20 frames instead of one | 6.1% |
| Eyes only / head only | 11.6% / 17.7% |
| Full quadratic, kernel ridge, robust (Huber) fit | 6.5%, 7.0–9.6%, 6.7% |

Up-down was weaker than left-right, and 5% of the frames landed off the screen. No model did better than the plain ridge fit, so the remaining error comes from the measurements, not the fitting. That led to clamping (off-screen points), the rich feature set (better measurements) and, in place of a first per-frame vote, using the layout: deciding between windows with the calibration's own uncertainty.

### Replaying the calibration through each version

The calibration frames were replayed in order, each predicted by a fit that left its dot out, with the frames during each eye movement filled in from the same data. Rich features, 15 dots; "right window" is the share of time the window you looked at had focus, in three layouts:

| Decision | Top/bottom | Left/right | 2×2 | Time to follow the eyes |
|---|---|---|---|---|
| Dwell on a smoothed point (v0.1) | 90.4% | 84.7% | 78.1% | 0.45 s |
| 70% of per-frame votes over 0.4 s (v0.2) | 93.6% | 89.5% | 87.9% | 0.30 s |
| Belief, 80% for 0.4 s (v0.3) | 90.0% | 83.5% | 81.2% | 0.47 s |
| Belief, 90% for 0.25 s (now) | 92.5% | 87.9% | 86.2% | 0.33 s |

Replayed again on 1,288 frames (15 dots, 24 pointer cells) for the quick switch at 99%: top/bottom 0.53 s (unchanged), left/right 0.33 → 0.22 s, 2×2 0.37 → 0.23 s, with no more wrong switches than before.

None of them switched to a wrong window. The shown point moved 2.1% of the width per frame raw, 1.1% with either the One Euro or the fixation filter, but in the first 0.5 s after the eyes moved the fixation filter was off by 12.5% against 7.4% for One Euro.

## Traps found while building this

- **MediaPipe was killed with SIGKILL at start-up.** `mediapipe` imports `sounddevice` for its audio tasks. Initialising PortAudio goes through the Advanced Linux Sound Architecture (ALSA) into PipeWire, whose realtime module leaves the process with a realtime CPU-time limit of 0, and the kernel kills it once inference starts. omeye has no audio, so `tracker.py` installs an empty `sounddevice` module before importing MediaPipe.
- **The GPU delegate silently ran in software.** MediaPipe opens the first render node, which on this laptop is the NVIDIA card. Mesa can't drive that, so it fell back to `llvmpipe`. With `delegate = "gpu"`, omeye sets `DRI_PRIME` to the first non-NVIDIA Peripheral Component Interconnect (PCI) device (here `pci-0000_07_00_0`, the Radeon).
- **A virtual camera only offers video while something feeds it.** The Flux camera is a v4l2loopback device. With no stream it accepts video in but offers none out, and OpenCV then can't open it. `tracker.py` asks each device what it can do (`VIDIOC_QUERYCAP`) and skips idle ones. When a stream stops, the next read blocks for OpenCV's default 10 s; `OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT=2` shortens that, because OpenCV doesn't support `CAP_PROP_READ_TIMEOUT_MSEC` for V4L2.
- **With the GPU delegate, the preview turned the screen black.** `DRI_PRIME` leaked from the tracker into the overlay process, so GTK drew the overlay on the Radeon while the HDMI monitor is driven by the NVIDIA card. The transparent surface came out solid black. `overlay_client.py` removes `DRI_PRIME` from the overlay's environment.
- **OpenCV's thread pool burned more than a core.** Its worker threads busy-wait between frames, so ~2 ms of colour conversion per frame cost 140–170% of a core at 30 fps. `tracker.py` calls `cv2.setNumThreads(1)`.
- **Hyprland 0.56 has Lua dispatchers.** Focus is `dispatch hl.dsp.focus({ window = "address:0x…" })` on the request socket. `hypr.py` falls back to the old `focuswindow` form on older releases.
- **Overlay.** The calibration and preview overlay is a GTK 4 layer-shell surface. It needs the system Python (PyGObject) and `LD_PRELOAD=/usr/lib/libgtk4-layer-shell.so`, so `overlay_client.py` starts it as a separate process and talks to it in lines of text on standard input and output.

## Layout

```
omeye/            Python package
  cli.py          commands
  daemon.py       tracking loop
  calibrate.py    dot calibration
  camview.py      camera view for the overlay
  tracker.py      camera, MediaPipe, gaze features
  model.py        calibrated regressions, choosing a model
  geometry.py     geometric model: screen, camera, eyes in millimetres
  validate.py     omeye test
  refine.py       pointer-following refinement
  samples.py      stored calibration samples
  focus.py        chance of each window, belief, when to switch
  filters.py      One Euro filter
  activity.py     typing and mouse activity
  hypr.py         Hyprland socket
  overlay.py      GTK 4 overlay (system Python)
src/omeye-idle.c  input-activity helper (ext-idle-notify-v1)
systemd/          user unit template
tests/            pytest: uv run pytest
```

## Ideas for later

- Learn from clicks while the service runs: when you click, you are almost always looking at the pointer, so each click is a free calibration sample (`omeye refine` does this on purpose).
- A bar indicator and toggle as an Omarchy plugin, next to `rb.monitor` and `rb.overview` in `omarchy-rb-plugins`.
- Gaze across several monitors (calibration currently covers the one the camera sits on).
