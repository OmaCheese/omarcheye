"""omeye calibrate: show dots, record gaze features per dot, fit, save."""

import random
import time

import numpy as np

from . import samples
from .config import CALIBRATION_PATH, Config
from .hypr import Hypr
from .model import fit_samples
from .overlay_client import OverlayProcess
from .camview import CameraFeed
from .tracker import FEATURES, RICH, Camera, FaceTracker, framing_advice, list_cameras, pick_camera

SETTLE = 0.9  # seconds for the eyes to land on a new dot
COLLECT = 1.1  # seconds of frames kept per dot
MIN_DOTS = 8
POOR = 0.15  # cross-validated error above this share of the screen width: warn

INTRO = (
    "omeye calibration\n\n"
    "Look at each dot until it shrinks away.\n"
    "Move your head as you normally would.\n\n"
    "Space: start      Esc: cancel"
)


class Cancelled(Exception):
    pass


def compare(errors: dict[str, float], kind: str) -> str:
    """One line on which feature set won, e.g. "rich features 5.1%, basic 6.6%: using rich"."""
    if len(errors) < 2:
        return ""
    return ", ".join(f"{k} features {100 * e:.1f}%" for k, e in sorted(errors.items(), key=lambda x: x[1])) + f": using {kind}"


def grid(n: int) -> list[tuple[float, float]]:
    cols, rows = (3, 3) if n <= 9 else (4, 3) if n <= 12 else (5, 3) if n <= 15 else (5, 4) if n <= 20 else (6, 4)
    return [(x, y) for y in np.linspace(0.07, 0.93, rows) for x in np.linspace(0.05, 0.95, cols)]


def check_keys(ov: OverlayProcess) -> str | None:
    while ev := ov.poll():
        if ev.get("event") == "closed" or ev.get("key") == "Escape":
            raise Cancelled
        if ev.get("event") == "key":
            return ev["key"]
    return None


def wait_for_start(ov: OverlayProcess, cam: Camera, tracker: FaceTracker, camera: str, note: str,
                    intro: str = INTRO) -> None:
    seen: list = []  # recent samples, None where no face
    shown = None
    feed = CameraFeed(ov, "center")
    last_frame = time.monotonic()
    brightness = 128.0
    while True:
        frame = cam.read()
        if frame is not None:
            last_frame = time.monotonic()
            brightness = float(frame[::16, ::16].mean())
            seen = (seen + [tracker.process(frame, last_frame)])[-15:]
            feed.update(frame, seen[-1])
        found = [s for s in seen if s is not None]
        face = len(found) >= 10
        if time.monotonic() - last_frame > 2:
            status = f"No video from {camera}"
        elif face:
            pos = tuple(np.median([s.pos for s in found], 0))
            advice = framing_advice(pos, float(np.median([s.margin for s in found])))
            status = f"Face found, but: {advice}" if advice else "Face found"
        elif brightness < 10:
            status = "The camera image is black: is the lens covered, or the phone camera off?"
        else:
            status = "No face found: is the camera looking at you?"
        if status != shown:
            ov.send(cmd="text", text=f"{intro}\n\nCamera: {camera}\n{status}" + (f"\n\n{note}" if note else ""))
            shown = status
        if check_keys(ov) == "space" and face:
            feed.hide()
            return


def _collect(ov, cam, tracker, dots, width, height):
    feats, rich, opens, groups, targets, framing = [], [], [], [], [], []
    for i, (nx, ny) in enumerate(dots):
        ov.send(cmd="dot", x=nx * width, y=ny * height, ms=(SETTLE + COLLECT) * 1000)
        start = last_frame = time.monotonic()
        while (elapsed := time.monotonic() - start) < SETTLE + COLLECT:
            check_keys(ov)
            frame = cam.read()
            if frame is None:
                if time.monotonic() - last_frame > 3:
                    raise RuntimeError("the camera stopped sending video")
                continue
            last_frame = time.monotonic()
            s = tracker.process(frame, time.monotonic())
            if s is not None and elapsed >= SETTLE:
                framing.append((*s.pos, s.margin))
                if s.cut_off or s.rich is None:
                    continue
                feats.append(s.feat)
                rich.append(s.rich)
                opens.append(s.openness)
                groups.append(i)
                targets.append((nx, ny))
    ov.send(cmd="dot")
    data = {"feats": np.array(feats).reshape(-1, len(FEATURES)), "rich": np.array(rich).reshape(-1, len(RICH)),
            "opens": np.array(opens), "groups": np.array(groups, int), "targets": np.array(targets).reshape(-1, 2)}
    return data, np.array(framing).reshape(-1, 3)


def run(cfg: Config, monitor: str = "", points: int = 0) -> int:
    hypr = Hypr()
    layout = hypr.layout()
    name = monitor or cfg.monitor
    mon = layout.monitor(name) if name else next((m for m in layout.monitors if m.focused), None)
    if mon is None:
        print(f"omeye: monitor {name!r} not found")
        return 1
    mm = next((m.get("physicalWidth", 0) for m in hypr.json("monitors") if m["name"] == mon.name), 0)
    try:
        info = pick_camera(cfg.camera)
    except RuntimeError as e:
        print(f"omeye: {e}")
        return 1
    idle = [c for c in list_cameras() if not c.live]
    note = idle[0].not_live_reason() if info.builtin and idle else ""
    print(f"omeye: calibrating {mon.name} with {info.name} ({info.device})")

    tracker = FaceTracker(delegate=cfg.delegate)
    cam = Camera(info.device, cfg.width, cfg.height, cfg.fps)
    ov = OverlayProcess(mon.name, "calibrate")
    try:
        ready = ov.wait_for("ready", 8)
        width, height = ready["width"], ready["height"]
        wait_for_start(ov, cam, tracker, info.name, note)
        dots = grid(points or cfg.calibration_points)
        random.shuffle(dots)
        data, framing = _collect(ov, cam, tracker, dots, width, height)
        ov.send(cmd="text", text="Fitting…")
        samples.save(info.name, mon.name, data, framing=framing, size=np.array([width, height]))

        advice = ""
        if len(framing):
            advice = framing_advice(tuple(np.median(framing[:, :2], 0)), float(np.median(framing[:, 2])))
        if len(data["groups"]) == 0:
            raise RuntimeError("no usable face frames" + (f". {advice}" if advice else ""))
        try:
            model, used, n_frames, errors = fit_samples(data["feats"], data["opens"], data["groups"], data["targets"],
                                                        height / width, mon.name, info.name, rich=data["rich"])
        except ValueError:
            used, n_frames = [], 0
        if len(used) < MIN_DOTS:
            raise RuntimeError(f"only {len(used)} of {len(dots)} dots had a clear view of your eyes"
                               + (f". {advice}" if advice else ""))
        model.save(CALIBRATION_PATH)

        err_px = model.error * width
        err_cm = f" ≈ {model.error * mm / 10:.1f} cm" if mm else ""
        summary = (f"Typical error {err_px:.0f} px{err_cm}, {100 * model.error:.0f}% of the screen width "
                   f"({len(used)}/{len(dots)} dots, {n_frames} frames)\n{compare(errors, model.kind)}")
        if model.error > POOR:
            why = advice or "Check that the camera sees your eyes clearly, and look straight at each dot"
            title, summary = "Calibration is poor: eye focus will jump around", f"{summary}\n\n{why}"
        else:
            title = "Calibrated"
        print(f"omeye: {title}. {summary}".replace("\n\n", ". ") + f"; saved {CALIBRATION_PATH}")
        ov.send(cmd="text", text=f"{title}\n\n{summary}\n\nPress any key")
        end = time.monotonic() + 6
        while time.monotonic() < end and check_keys(ov) is None:
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
