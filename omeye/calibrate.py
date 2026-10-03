"""omeye calibrate: show dots, record gaze features per dot, fit, save."""

import random
import time

import numpy as np

from .config import CALIBRATION_PATH, Config
from .hypr import Hypr
from .model import fit, inliers
from .overlay_client import OverlayProcess
from .tracker import Camera, FaceTracker, pick_camera

SETTLE = 0.9  # seconds for the eyes to land on a new dot
COLLECT = 1.1  # seconds of frames kept per dot
MIN_FRAMES = 8
MIN_DOTS = 8

INTRO = (
    "omeye calibration\n\n"
    "Look at each dot until it shrinks away.\n"
    "Move your head as you normally would.\n\n"
    "Space: start      Esc: cancel"
)


class Cancelled(Exception):
    pass


def grid(n: int) -> list[tuple[float, float]]:
    cols, rows = (3, 3) if n <= 9 else (4, 3) if n <= 12 else (5, 3) if n <= 15 else (5, 4) if n <= 20 else (6, 4)
    return [(x, y) for y in np.linspace(0.07, 0.93, rows) for x in np.linspace(0.05, 0.95, cols)]


def _check_keys(ov: OverlayProcess) -> str | None:
    while ev := ov.poll():
        if ev.get("event") == "closed" or ev.get("key") == "Escape":
            raise Cancelled
        if ev.get("event") == "key":
            return ev["key"]
    return None


def _wait_for_start(ov: OverlayProcess, cam: Camera, tracker: FaceTracker) -> None:
    seen: list[bool] = []
    shown = None
    while True:
        frame = cam.read()
        if frame is not None:
            seen = (seen + [tracker.process(frame, time.monotonic()) is not None])[-15:]
        face = sum(seen) >= 10
        status = "Face found" if face else "No face found: is the camera looking at you?"
        if status != shown:
            ov.send(cmd="text", text=f"{INTRO}\n\n{status}")
            shown = status
        if _check_keys(ov) == "space" and face:
            return


def _collect(ov, cam, tracker, dots, width, height):
    feats, opens, groups, targets = [], [], [], []
    for i, (nx, ny) in enumerate(dots):
        ov.send(cmd="dot", x=nx * width, y=ny * height, ms=(SETTLE + COLLECT) * 1000)
        start = time.monotonic()
        while (elapsed := time.monotonic() - start) < SETTLE + COLLECT:
            _check_keys(ov)
            frame = cam.read()
            if frame is None:
                continue
            s = tracker.process(frame, time.monotonic())
            if s is not None and elapsed >= SETTLE:
                feats.append(s.feat)
                opens.append(s.openness)
                groups.append(i)
                targets.append((nx, ny))
    ov.send(cmd="dot")
    return np.array(feats), np.array(opens), np.array(groups), np.array(targets)


def run(cfg: Config, monitor: str = "", points: int = 0) -> int:
    hypr = Hypr()
    layout = hypr.layout()
    name = monitor or cfg.monitor
    mon = layout.monitor(name) if name else next((m for m in layout.monitors if m.focused), None)
    if mon is None:
        print(f"omeye: monitor {name!r} not found")
        return 1
    mm = next((m.get("physicalWidth", 0) for m in hypr.json("monitors") if m["name"] == mon.name), 0)
    device, cam_name = pick_camera(cfg.camera)
    print(f"omeye: calibrating {mon.name} with {cam_name} ({device})")

    tracker = FaceTracker(delegate=cfg.delegate)
    cam = Camera(device, cfg.width, cfg.height, cfg.fps)
    ov = OverlayProcess(mon.name, "calibrate")
    try:
        ready = ov.wait_for("ready", 8)
        width, height = ready["width"], ready["height"]
        _wait_for_start(ov, cam, tracker)
        dots = grid(points or cfg.calibration_points)
        random.shuffle(dots)
        f, opens, groups, targets = _collect(ov, cam, tracker, dots, width, height)
        ov.send(cmd="text", text="Fitting…")

        if len(f) == 0:
            raise RuntimeError("no face frames recorded")
        blink = 0.6 * float(np.median(opens))
        keep = opens >= blink
        keep[keep] = inliers(f[keep], groups[keep])
        counts = np.bincount(groups[keep], minlength=len(dots))
        keep &= counts[groups] >= MIN_FRAMES
        n_dots = int((counts >= MIN_FRAMES).sum())
        if n_dots < MIN_DOTS:
            raise RuntimeError(f"only {n_dots} of {len(dots)} dots had a clear view of your eyes")

        model = fit(f[keep], targets[keep], groups[keep], height / width, mon.name, blink, cam_name)
        model.save(CALIBRATION_PATH)

        err_px = model.error * width
        err_cm = f" ≈ {model.error * mm / 10:.1f} cm" if mm else ""
        summary = f"Typical error {err_px:.0f} px{err_cm} ({n_dots}/{len(dots)} dots, {keep.sum()} frames)"
        print(f"omeye: {summary}; saved {CALIBRATION_PATH}")
        ov.send(cmd="text", text=f"Calibrated\n\n{summary}\n\nPress any key")
        end = time.monotonic() + 6
        while time.monotonic() < end and _check_keys(ov) is None:
            time.sleep(0.05)
        return 0
    except Cancelled:
        print("omeye: calibration cancelled")
        return 1
    except RuntimeError as e:
        print(f"omeye: calibration failed: {e}")
        ov.send(cmd="text", text=f"Calibration failed\n\n{e}")
        time.sleep(3)
        return 1
    finally:
        ov.close()
        cam.close()
        tracker.close()
