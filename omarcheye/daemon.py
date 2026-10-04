"""The tracking loop: camera frame -> gaze point -> window -> focus."""

import math
import signal
from collections import deque
from dataclasses import dataclass, field
import sys
import threading
import time

import numpy as np

from .activity import CursorWatch, InputActivity
from .config import CALIBRATION_PATH, DRIFT_PATH, Config
from .drift import Drift, Record
from .filters import OneEuro2D
from .focus import AWAY, Belief, BeliefParams, Glance, on_screen, runner_up, window_chances
from .hypr import Hypr, HyprError, Layout, Monitor, Window
from .model import GazeModel, load_model
from .overlay_client import OverlayProcess
from .camview import CameraFeed
from .tracker import Camera, FaceTracker, framing_advice, pick_camera

STATS_EVERY = 30.0
CORRECT_S = 3.0  # moving focus yourself this soon after omarcheye did corrects it
# While you type (the pointer still), you are mostly looking at the window you
# type into: every TYPED_GAP s, a second of typing makes a light record.
TYPED_S, TYPED_GAP, TYPED_WEIGHT = 1.0, 4.0, 0.3
SAVE_EVERY = 60.0


def log(msg: str) -> None:
    print(f"omarcheye: {msg}", file=sys.stderr, flush=True)


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


OFF_SCREEN_HINT = "Most gaze estimates fall off the screen: sit as when you calibrated, or run omarcheye calibrate"


@dataclass
class Chain:
    """omarcheye's last focus change and the retries that followed it."""
    t: float  # when omarcheye last moved focus
    raw: tuple[float, float]  # the raw gaze estimate behind it (monitor fractions)
    tried: list[str]  # windows omarcheye focused, the current one last
    known: set[str] = field(default_factory=set)  # windows on screen at the time
    record: Record | None = None  # what the drift learned from this chain


def rect_of(win: Window, mon: Monitor) -> tuple[float, float, float, float]:
    """A window's rectangle in monitor fractions."""
    x0, y0 = mon.to_local(win.x, win.y)
    x1, y1 = mon.to_local(win.x + win.w, win.y + win.h)
    return x0, y0, x1, y1


def find(layout: Layout, address: str | None, mon: Monitor | None = None) -> Window | None:
    """The window with this address; with `mon`, only if it sits on that monitor."""
    w = next((w for w in layout.windows if w.address == address), None)
    if w is None or mon is None:
        return w
    cx, cy = mon.to_local(w.x + w.w / 2, w.y + w.h / 2)
    return w if 0 <= cx <= 1 and 0 <= cy <= 1 else None


def new_stats() -> dict:
    return {"frames": 0, "faces": 0, "busy": 0.0, "switches": 0, "retries": 0, "held_typing": 0, "held_mouse": 0,
            "off_screen": 0}


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
        log("not calibrated yet; run `omarcheye calibrate`")
        return 1
    model = load_model(CALIBRATION_PATH)
    hypr = Hypr()
    poller = LayoutPoller(hypr)
    mon = poller.layout.monitor(model.monitor)
    if mon is None:
        log(f"calibrated monitor {model.monitor} is not connected")
        return 1

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tracker = FaceTracker(delegate=cfg.delegate, eyenet=cfg.eyenet)
    activity = InputActivity(cfg.typing_grace_ms)
    cursor = CursorWatch()
    smooth = OneEuro2D(cfg.filter_min_cutoff, cfg.filter_beta)  # the point the preview shows
    belief = Belief(BeliefParams.from_config(cfg))
    sigma = max(model.error / 1.2533, 0.02)  # per-axis gaze error in monitor widths (mean radial error / sqrt(pi/2))
    aspect = mon.h / mon.w
    drift = Drift(aspect, sigma, model.created, DRIFT_PATH)  # the shift since calibration (drift.py)
    learn = cfg.learn and not dry_run
    glance = Glance(aspect, belief.p.retry, belief.p.glance)
    chain: Chain | None = None
    recent_raw: deque = deque(maxlen=3)  # the last few raw estimates, for a switch's anchor
    typed: list = []  # raw estimates while typing into typed_in
    typed_in: str | None = None
    typed_last = saved = -math.inf
    overlay = OverlayProcess(model.monitor, "follow") if preview else None
    feed = CameraFeed(overlay, "corner") if overlay else None
    lost = cfg.lost_ms / 1000
    away = cfg.away_ms / 1000

    log(f"on {model.monitor}, calibration error {model.error * mon.w:.0f} px" + (" [dry run]" if dry_run else ""))
    if drift.records:
        log(f"learned {drift.describe()}")

    def name(address: str | None) -> str:
        return next((w.cls for w in poller.layout.windows if w.address == address), str(address))

    def move_focus(address: str, now: float, layout: Layout) -> None:
        if dry_run:
            return
        try:
            hypr.focus(address)
            layout.focused = poller.layout.focused = address
            cursor.expect_warp(now)
            stats["switches"] += 1
        except (OSError, HyprError) as e:
            log(f"focus: {e}")

    def learn_from(c: Chain, raw: tuple[float, float], win: Window, m: Monitor) -> None:
        """You were looking at `win` when the raw estimate was `raw`."""
        if c.record is None:
            c.record = drift.add(*raw, rect_of(win, m))
        else:
            c.record.x, c.record.y = raw
            if not drift.retarget(c.record, rect_of(win, m)):
                drift.forget(c.record)
                c.record = None
        if c.record is not None:
            drift.save()
            log(f"learned {drift.describe()}")
    cam = open_camera(cfg, model, stop)
    last_face = last_frame = time.monotonic()
    frame_no = 0
    recent_off: deque = deque(maxlen=90)  # face found but gaze off the screen, last ~3 s
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
                    glance.disarm()
                    chain = None
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
            chances = gaze = raw = None
            if sample is not None:
                stats["faces"] += 1
                if now - last_face > lost:
                    smooth.reset()
                last_face = now
                raw = model.predict_sample(sample)
                if not all(map(math.isfinite, raw)):
                    raw = None
                else:
                    recent_raw.append(raw)
                point = on_screen(drift.apply(*raw), cfg.offscreen) if mon and raw else None
                recent_off.append(point is None)
                stats["off_screen"] += point is None
                if point:
                    chances = window_chances(layout.windows, *mon.to_global(*point), sigma * mon.w)
                    aspect = mon.h / mon.w
                    fx, fy = smooth(now, point[0], point[1] * aspect)
                    gaze = mon.to_global(fx, fy / aspect)
            if chances is None:  # no face, or looking away
                chances = {**{w.address: 0.0 for w in layout.windows}, AWAY: 1.0}

            busy = ("typing" if now - activity.last(now) < belief.p.typing_grace
                    else "mouse" if now - cursor.last_move < belief.p.mouse_grace else "")
            if chain and glance.armed and not glance.away:
                chain.raw = glance.anchor  # where you look, steadier than at the switch
            if busy:
                glance.disarm()  # using the window accepts it
            if chain and now - chain.t > CORRECT_S:
                chain = None
            # You moved focus yourself right after omarcheye did: it picked wrong,
            # and the window you chose is where you were looking. (The first
            # half second is skipped: the layout may not show omarcheye's own move yet.)
            if chain and not dry_run and now - chain.t > 0.5 and layout.focused not in (None, chain.tried[-1]):
                win = find(layout, layout.focused, mon)
                if win and layout.focused in chain.known and find(layout, chain.tried[-1]):
                    log(f"you moved focus from {name(chain.tried[-1])} to {name(win.address)}")
                    if learn:
                        learn_from(chain, chain.raw, win, mon)
                    belief.settle(win.address)
                chain = None
                glance.disarm()

            # Typing: a light record that you look at the window you type into.
            if learn and busy == "typing" and now - cursor.last_move > 1.0 and raw and layout.focused:
                if typed_in != layout.focused:
                    typed, typed_in = [], layout.focused
                typed.append(raw)
                win = find(layout, typed_in, mon)
                if len(typed) >= TYPED_S * cfg.fps and now - typed_last >= TYPED_GAP and win:
                    if drift.add(*np.median(np.array(typed), 0), rect_of(win, mon), TYPED_WEIGHT):
                        typed_last = now
                    typed = []
            elif busy != "typing":
                typed, typed_in = [], None
            if learn and drift.records and now - saved > SAVE_EVERY:
                drift.save()
                saved = now

            if glance.update(now, raw) and chain and mon:
                # A glance away and back: omarcheye picked wrong. Try the likeliest
                # window next to where you were looking that it hasn't tried.
                x, y = mon.to_global(*drift.apply(*glance.spot))
                nxt = runner_up(layout.windows, x, y, sigma * mon.w, chain.tried)
                if nxt is None:
                    log(f"retry: no other window near {name(chain.tried[-1])}")
                    chain = None
                else:
                    log(f"{'would retry' if dry_run else 'retry'}: {name(chain.tried[-1])} -> {name(nxt)}")
                    move_focus(nxt, now, layout)
                    stats["retries"] += 1
                    chain.tried.append(nxt)
                    chain.t = now
                    chain.raw = glance.spot
                    if learn:
                        learn_from(chain, glance.spot, find(layout, nxt), mon)
                    belief.settle(nxt)
                    glance.arm(now, glance.spot)

            chosen = belief.step(now, chances, layout.focused, activity.last(now), cursor.last_move,
                                 hold=glance.holding, avoid=chain.tried[:-1] if chain else ())
            top, prob = belief.top()
            ready = top is not AWAY and top != layout.focused and prob >= belief.p.confidence
            if ready and busy:
                stats[f"held_{busy}"] += 1
            if chosen:
                if verbose or dry_run:
                    log(f"{'would focus' if dry_run else 'focus'} {name(chosen)} ({100 * prob:.0f}% likely)")
                move_focus(chosen, now, layout)
                chain = None
                glance.disarm()
                if recent_raw and raw is not None:
                    anchor = tuple(np.median(np.array(recent_raw), 0))
                    chain = Chain(now, anchor, [chosen], {w.address for w in layout.windows})
                    glance.arm(now, anchor)

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
                if not advice and len(recent_off) > 30 and sum(recent_off) > 0.8 * len(recent_off):
                    advice = OFF_SCREEN_HINT
                overlay.send(cmd="text", text=f"omarcheye preview{' (dry run)' if dry_run else ''}"
                             f"   face {'yes' if sample else 'no'}   {busy}" + (f"\n{advice}" if advice else ""))

            if verbose and now - stats_since >= STATS_EVERY:
                n = max(stats["frames"], 1)
                fps = n / (now - stats_since)
                log(f"{fps:.1f} fps, {1000 * stats['busy'] / n:.1f} ms/frame, face {100 * stats['faces'] / n:.0f}%, "
                    f"{stats['switches']} switches, {stats['retries']} retries; a ready switch was held "
                    f"{stats['held_typing'] / fps:.1f} s by typing, {stats['held_mouse'] / fps:.1f} s by the mouse"
                    + (f"; gaze off the screen in {100 * stats['off_screen'] / max(stats['faces'], 1):.0f}% of face frames"
                       if stats["off_screen"] else ""))
                stats = new_stats()
                stats_since = now
        return 0
    finally:
        if learn:
            drift.save()
        poller.stopped.set()
        activity.close()
        if overlay:
            overlay.close()
        if cam is not None:
            cam.close()
        tracker.close()
