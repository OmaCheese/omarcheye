"""The tracking loop: camera frame -> gaze point -> window -> focus."""

import signal
import sys
import threading
import time

from .activity import CursorWatch, InputActivity
from .config import CALIBRATION_PATH, Config
from .filters import FixationFilter
from .focus import Vote, VoteParams, hit_test, on_screen
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
    fixation = FixationFilter(cfg.fixation_radius, cfg.fixation_confirm)
    vote = Vote(VoteParams.from_config(cfg))
    overlay = OverlayProcess(model.monitor, "follow") if preview else None
    feed = CameraFeed(overlay, "corner") if overlay else None
    lost = cfg.lost_ms / 1000
    away = cfg.away_ms / 1000

    log(f"on {model.monitor}, calibration error {model.error * mon.w:.0f} px" + (" [dry run]" if dry_run else ""))
    cam = open_camera(cfg, model, stop)
    last_face = last_frame = time.monotonic()
    frame_no = 0
    stats = {"frames": 0, "faces": 0, "busy": 0.0, "switches": 0}
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
                    fixation.reset()
                    vote.reset()
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
            # Each frame's own point votes for a window; the fixation filter
            # gives the steady point the preview shows.
            target = gaze = None
            if sample is not None:
                stats["faces"] += 1
                if now - last_face > lost:
                    fixation.reset()
                last_face = now
                point = on_screen(model.predict_sample(sample), cfg.offscreen) if mon else None
                if point:
                    target = hit_test(layout.windows, *mon.to_global(*point), layout.focused, cfg.margin_px)
                    aspect = mon.h / mon.w
                    fx, fy = fixation(point[0], point[1] * aspect)
                    gaze = mon.to_global(fx, fy / aspect)

            chosen = vote.step(now, target, layout.focused, activity.last(now), cursor.last_move)
            if chosen:
                if dry_run:
                    log(f"would focus {chosen}")
                else:
                    try:
                        hypr.focus(chosen)
                        layout.focused = chosen
                        cursor.expect_warp(now)
                        stats["switches"] += 1
                    except (OSError, HyprError) as e:
                        log(f"focus: {e}")

            if overlay and mon:
                if gaze:
                    on = hit_test(layout.windows, *gaze, layout.focused, cfg.margin_px) == layout.focused
                    overlay.send(cmd="gaze", x=gaze[0] - mon.x, y=gaze[1] - mon.y, on=on)
                else:
                    overlay.send(cmd="gaze")
                leader, share = vote.leader()
                win = next((w for w in layout.windows if w.address == leader), None)
                if win:
                    overlay.send(cmd="rect", x=win.x - mon.x, y=win.y - mon.y, w=win.w, h=win.h,
                                 label=f"{100 * share:.0f}% of votes", on=leader == layout.focused)
                else:
                    overlay.send(cmd="rect")
                busy = ("typing" if now - activity.last(now) < vote.p.typing_grace
                        else "mouse" if now - cursor.last_move < vote.p.mouse_grace else "")
                advice = framing_advice(seen.pos, seen.margin) if seen else ""
                overlay.send(cmd="text", text=f"omeye preview{' (dry run)' if dry_run else ''}"
                             f"   face {'yes' if sample else 'no'}   {busy}" + (f"\n{advice}" if advice else ""))

            if verbose and now - stats_since >= STATS_EVERY:
                n = max(stats["frames"], 1)
                log(f"{n / (now - stats_since):.1f} fps, {1000 * stats['busy'] / n:.1f} ms/frame, "
                    f"face {100 * stats['faces'] / n:.0f}%, {stats['switches']} switches")
                stats = {"frames": 0, "faces": 0, "busy": 0.0, "switches": 0}
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
