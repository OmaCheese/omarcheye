"""The tracking loop: camera frame -> gaze point -> window -> focus."""

import signal
import sys
import threading
import time

from .activity import CursorWatch, InputActivity
from .config import CALIBRATION_PATH, Config
from .filters import OneEuro2D
from .focus import AWAY, Belief, BeliefParams, on_screen, window_chances
from .hypr import Hypr, HyprError
from .model import GazeModel
from .overlay_client import OverlayProcess
from .camview import CameraFeed
from .tracker import Camera, FaceTracker, framing_advice, pick_camera

STATS_EVERY = 30.0


def log(msg: str) -> None:
    print(f"omeye: {msg}", file=sys.stderr, flush=True)


class LayoutPoller:
    """Keeps a fresh Hyprland layout snapshot, read by the tracking loop."""

    def __init__(self, hypr: Hypr, interval: float = 0.25):
        self.hypr = hypr
        self.layout = hypr.layout()
        self.interval = interval
        self.stopped = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self.stopped.wait(self.interval):
            try:
                self.layout = self.hypr.layout()
            except (OSError, ValueError, HyprError) as e:
                log(f"layout: {e}")


def new_stats() -> dict:
    return {"frames": 0, "faces": 0, "busy": 0.0, "switches": 0, "held_typing": 0, "held_mouse": 0}


def open_camera(cfg: Config, model: GazeModel, stop: threading.Event) -> Camera | None:
    """The calibrated camera, once it sends video (a phone webcam may come and
    go). None if asked to stop first."""
    spec = model.camera if cfg.camera == "auto" and model.camera else cfg.camera
    said = None
    while not stop.is_set():
        try:
            info = pick_camera(spec, quiet=True)
            cam = Camera(info.device, cfg.width, cfg.height, cfg.fps)
            log(f"camera: {info.name} ({info.device})")
            return cam
        except RuntimeError as e:
            if str(e) != said:
                log(f"waiting for the camera: {e}")
                said = str(e)
        stop.wait(2)
    return None


def run(cfg: Config, preview: bool = False, dry_run: bool = False, verbose: bool = False) -> int:
    if not CALIBRATION_PATH.exists():
        log("not calibrated yet; run `omeye calibrate`")
        return 1
    model = GazeModel.load(CALIBRATION_PATH)
    hypr = Hypr()
    poller = LayoutPoller(hypr)
    mon = poller.layout.monitor(model.monitor)
    if mon is None:
        log(f"calibrated monitor {model.monitor} is not connected")
        return 1

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tracker = FaceTracker(delegate=cfg.delegate)
    activity = InputActivity(cfg.typing_grace_ms)
    cursor = CursorWatch()
    smooth = OneEuro2D(cfg.filter_min_cutoff, cfg.filter_beta)  # the point the preview shows
    belief = Belief(BeliefParams.from_config(cfg))
    sigma = max(model.error / 1.2533, 0.02)  # per-axis gaze error in monitor widths (mean radial error / sqrt(pi/2))
    overlay = OverlayProcess(model.monitor, "follow") if preview else None
    feed = CameraFeed(overlay, "corner") if overlay else None
    lost = cfg.lost_ms / 1000
    away = cfg.away_ms / 1000

    log(f"on {model.monitor}, calibration error {model.error * mon.w:.0f} px" + (" [dry run]" if dry_run else ""))
    cam = open_camera(cfg, model, stop)
    last_face = last_frame = time.monotonic()
    frame_no = 0
    stats = new_stats()
    stats_since = time.monotonic()
    try:
        while cam is not None and not stop.is_set():
            frame_no += 1
            if time.monotonic() - last_face > away and frame_no % 6:
                cam.skip()  # nobody there: look only every 6th frame
                continue
            frame = cam.read()
            now = time.monotonic()
            if frame is None:
                if now - last_frame > 1.5:
                    log("camera stopped sending video")
                    cam.close()
                    cam = open_camera(cfg, model, stop)
                    smooth.reset()
                    belief.reset()
                    last_face = last_frame = time.monotonic()
                continue
            last_frame = now
            seen = tracker.process(frame, now)
            sample = None if seen is None or seen.cut_off else seen  # cut-off face: eye landmarks are guesses
            if feed:
                feed.update(frame, seen)
            stats["busy"] += time.monotonic() - now
            stats["frames"] += 1

            layout = poller.layout
            mon = layout.monitor(model.monitor)
            try:
                cursor.update(now, hypr.cursor())
            except (OSError, ValueError):
                pass

            if sample is not None and sample.openness < model.blink:
                continue  # blink: hold everything as it is
            # Each frame's own point, with the calibration's uncertainty, gives
            # the chance of each window on screen; the belief combines frames.
            # The One Euro filter gives the steady point the preview shows.
            chances = gaze = None
            if sample is not None:
                stats["faces"] += 1
                if now - last_face > lost:
                    smooth.reset()
                last_face = now
                point = on_screen(model.predict_sample(sample), cfg.offscreen) if mon else None
                if point:
                    chances = window_chances(layout.windows, *mon.to_global(*point), sigma * mon.w)
                    aspect = mon.h / mon.w
                    fx, fy = smooth(now, point[0], point[1] * aspect)
                    gaze = mon.to_global(fx, fy / aspect)
            if chances is None:  # no face, or looking away
                chances = {**{w.address: 0.0 for w in layout.windows}, AWAY: 1.0}

            chosen = belief.step(now, chances, layout.focused, activity.last(now), cursor.last_move)
            top, prob = belief.top()
            busy = ("typing" if now - activity.last(now) < belief.p.typing_grace
                    else "mouse" if now - cursor.last_move < belief.p.mouse_grace else "")
            ready = top is not AWAY and top != layout.focused and prob >= belief.p.confidence
            if ready and busy:
                stats[f"held_{busy}"] += 1
            if chosen:
                name = next((w.cls for w in layout.windows if w.address == chosen), chosen)
                if verbose or dry_run:
                    log(f"{'would focus' if dry_run else 'focus'} {name} ({100 * prob:.0f}% likely)")
                if not dry_run:
                    try:
                        hypr.focus(chosen)
                        layout.focused = chosen
                        cursor.expect_warp(now)
                        stats["switches"] += 1
                    except (OSError, HyprError) as e:
                        log(f"focus: {e}")

            if overlay and mon:
                # The ring sits on the centre of the most likely window; the
                # small dot is the steady gaze estimate itself.
                if gaze:
                    overlay.send(cmd="point", x=gaze[0] - mon.x, y=gaze[1] - mon.y)
                else:
                    overlay.send(cmd="point")
                # Colours say what the service would do: green = the focused
                # window, amber = sure enough to switch, white = only likeliest.
                win = next((w for w in layout.windows if w.address == top), None)
                if win and prob >= 0.5:
                    state = "focused" if top == layout.focused else "ready" if ready else "likely"
                    label = f"{100 * prob:.0f}% likely" + (f", held: {busy}" if ready and busy else "")
                    overlay.send(cmd="gaze", x=win.x + win.w / 2 - mon.x, y=win.y + win.h / 2 - mon.y, state=state)
                    overlay.send(cmd="rect", x=win.x - mon.x, y=win.y - mon.y, w=win.w, h=win.h,
                                 label=label, state=state)
                else:
                    overlay.send(cmd="gaze")
                    overlay.send(cmd="rect")
                advice = framing_advice(seen.pos, seen.margin) if seen else ""
                overlay.send(cmd="text", text=f"omeye preview{' (dry run)' if dry_run else ''}"
                             f"   face {'yes' if sample else 'no'}   {busy}" + (f"\n{advice}" if advice else ""))

            if verbose and now - stats_since >= STATS_EVERY:
                n = max(stats["frames"], 1)
                fps = n / (now - stats_since)
                log(f"{fps:.1f} fps, {1000 * stats['busy'] / n:.1f} ms/frame, face {100 * stats['faces'] / n:.0f}%, "
                    f"{stats['switches']} switches; a ready switch was held {stats['held_typing'] / fps:.1f} s "
                    f"by typing, {stats['held_mouse'] / fps:.1f} s by the mouse")
                stats = new_stats()
                stats_since = now
        return 0
    finally:
        poller.stopped.set()
        activity.close()
        if overlay:
            overlay.close()
        if cam is not None:
            cam.close()
        tracker.close()
