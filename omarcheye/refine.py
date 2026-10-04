"""omarcheye refine: follow the mouse pointer with your eyes. Frames taken while
the pointer rests are added to the calibration samples, and the model is
refitted on everything.

Only frames where the pointer has stayed nearly still for STILL_S count: the
camera stream lags (a phone over Wi-Fi more so) and the eyes trail a moving
pointer, but neither matters while the pointer rests. The screen is split
into cells; each cell is one group for cross-validation, so the reported
error is for places the model did not learn from. The new calibration is
kept only if it beats the old one on these samples.
"""

import time
from collections import deque

import numpy as np

from . import samples
from .samples import DESC, NET
from .calibrate import Cancelled, check_keys, compare, wait_for_start
from .config import CALIBRATION_PATH, Config
from .hypr import Hypr
from .model import MOUSE, data_error, fit_samples, load_model, subset
from .overlay_client import OverlayProcess
from .tracker import FEATURES, RICH, Camera, FaceTracker, pick_camera

COLS, ROWS = 6, 4
PER_CELL = 15  # frames that make a cell covered
STILL_S = 0.4  # the pointer must have rested this long ...
STILL_FRAC = 0.02  # ... within this share of the monitor width
MIN_CELLS = 8

INTRO = (
    "omarch-eye refine\n\n"
    "Move the mouse slowly over the whole screen and keep your eyes on the pointer.\n"
    "Rest it here and there: frames count while the pointer is nearly still.\n"
    "Cells turn green as they fill.\n\n"
    "Space: start      Enter: finish early      Esc: cancel"
)


def pointer_still(history, t: float, aspect: float, window: float = STILL_S, limit: float = STILL_FRAC) -> bool:
    """Has the pointer stayed within `limit` monitor widths for the last `window` s?
    history: (time, x, y) with x, y in monitor fractions, oldest first."""
    if not history or t - history[0][0] < window:
        return False
    recent = np.array([(x, y) for ht, x, y in history if t - ht <= window])
    span = recent.max(0) - recent.min(0)
    return span[0] <= limit and span[1] * aspect <= limit


def cell(nx: float, ny: float) -> int:
    return min(int(ny * ROWS), ROWS - 1) * COLS + min(int(nx * COLS), COLS - 1)


def _collect(ov, cam, tracker, hypr, mon, seconds, aspect):
    feats, rich, pose, net, desc, opens, groups, targets = [], [], [], [], [], [], [], []
    counts = np.zeros(COLS * ROWS, int)
    history: deque = deque(maxlen=90)
    end = time.monotonic() + seconds
    last_frame = time.monotonic()
    frame_no = 0
    while (now := time.monotonic()) < end and counts.min() < PER_CELL:
        if check_keys(ov) in ("Return", "KP_Enter"):
            break
        frame = cam.read()
        if frame is None:
            if time.monotonic() - last_frame > 3:
                raise RuntimeError("the camera stopped sending video")
            continue
        t = last_frame = time.monotonic()
        try:
            nx, ny = mon.to_local(*hypr.cursor())
        except (OSError, ValueError):
            continue
        history.append((t, nx, ny))
        s = tracker.process(frame, t)
        if (s is not None and not s.cut_off and s.rich is not None and 0 <= nx < 1 and 0 <= ny < 1
                and pointer_still(history, t, aspect)):
            c = cell(nx, ny)
            feats.append(s.feat)
            rich.append(s.rich)
            pose.append(s.pose)
            net.append(s.net if s.net is not None else np.full(NET, np.nan))
            desc.append(s.desc.astype(np.float16) if s.desc is not None else np.full(DESC, np.nan, np.float16))
            opens.append(s.openness)
            groups.append(MOUSE + c)
            targets.append((nx, ny))
            counts[c] += 1
        frame_no += 1
        if frame_no % 5 == 0:
            ov.send(cmd="grid", cols=COLS, rows=ROWS, fill=np.minimum(counts / PER_CELL, 1).tolist())
            ov.send(cmd="text", text=f"Covered {(counts >= PER_CELL).sum()} of {COLS * ROWS} cells"
                    f"   {max(0, end - now):.0f} s left\n\nEnter: finish now      Esc: cancel")
    ov.send(cmd="grid")
    return {"feats": np.array(feats).reshape(-1, len(FEATURES)), "rich": np.array(rich).reshape(-1, len(RICH)),
            "pose": np.array(pose).reshape(-1, 16), "net": np.array(net).reshape(-1, NET),
            "desc": np.array(desc, np.float16).reshape(-1, DESC), "opens": np.array(opens), "groups": np.array(groups, int),
            "targets": np.array(targets).reshape(-1, 2)}


def run(cfg: Config, seconds: float = 60) -> int:
    hypr = Hypr()
    layout = hypr.layout()
    previous = load_model(CALIBRATION_PATH) if CALIBRATION_PATH.exists() else None
    name = (previous.monitor if previous else "") or cfg.monitor
    mon = layout.monitor(name) if name else next((m for m in layout.monitors if m.focused), None)
    if mon is None:
        print(f"omarcheye: monitor {name!r} not found")
        return 1
    try:
        info = pick_camera(cfg.camera)
    except RuntimeError as e:
        print(f"omarcheye: {e}")
        return 1
    if previous and previous.camera and previous.camera != info.name:
        print(f"omarcheye: the calibration is for {previous.camera!r}, not {info.name!r}; starting a new one")
        previous = None
    stored = samples.load(info.name, mon.name)
    print(f"omarcheye: refining {mon.name} with {info.name}"
          + (f" ({len(stored['groups'])} stored samples)" if stored else " (no stored samples)"))

    tracker = FaceTracker(delegate=cfg.delegate, eyenet=cfg.eyenet, patches=cfg.patches)
    cam = Camera(info.device, cfg.width, cfg.height, cfg.fps)
    ov = OverlayProcess(mon.name, "calibrate")
    try:
        ready = ov.wait_for("ready", 8)
        aspect = ready["height"] / ready["width"]
        wait_for_start(ov, cam, tracker, info.name, "", INTRO)
        new = _collect(ov, cam, tracker, hypr, mon, seconds, aspect)
        ov.send(cmd="text", text="Fitting…")
        cells = {int(g) for g in np.unique(new["groups"]) if (new["groups"] == g).sum() >= 8}
        if len(cells) < MIN_CELLS:
            raise RuntimeError(f"only {len(cells)} cells got enough frames; rest the pointer in more places")

        merged = samples.merge(stored, new)
        model, used, n_frames, errors = fit_samples(merged, aspect, mon.name, info.name, score=cells,
                                                    screen_mm=hypr.physical_mm(mon.name))
        after = model.error
        before = None
        if previous:
            ok = new["opens"] >= previous.blink
            if ok.any():
                before = data_error(previous, subset(new, ok), aspect)

        width = ready["width"]
        line = f"on the pointer samples: {100 * after:.0f}% of the screen width ({after * width:.0f} px)"
        if before is not None and before <= after:
            title = "Kept the previous calibration"
            text = f"New fit {line}, old one {100 * before:.0f}%."
        else:
            model.save(CALIBRATION_PATH)
            samples.save(info.name, mon.name, merged)
            title = "Calibration refined"
            text = (f"Typical error {line}" + (f", was {100 * before:.0f}%" if before is not None else "")
                    + f"\n{len(used)} dots and cells, {n_frames} frames\n{compare(errors, model.kind)}")
        print(f"omarcheye: {title}. {text}".replace("\n", "; "))
        ov.send(cmd="text", text=f"{title}\n\n{text}\n\nPress any key")
        end = time.monotonic() + 8
        while time.monotonic() < end and check_keys(ov) is None:
            time.sleep(0.05)
        return 0
    except Cancelled:
        print("omarcheye: refine cancelled; calibration unchanged")
        return 1
    except RuntimeError as e:
        print(f"omarcheye: refine failed: {e}")
        ov.send(cmd="text", text=f"Refine failed\n\n{e}")
        time.sleep(3)
        return 1
    finally:
        ov.close()
        cam.close()
        tracker.close()
