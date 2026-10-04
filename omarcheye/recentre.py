"""omarcheye recentre: look at one dot to tell omarcheye how you sit now.

Sitting differently shifts every gaze estimate by about the same amount (see
drift.py). One dot in the middle of the screen measures that shift in two
seconds; it replaces whatever omarcheye had learned from your corrections, which
then carry on from there.
"""

import time

import numpy as np

from .calibrate import Cancelled, _collect, check_keys
from .config import CALIBRATION_PATH, DRIFT_PATH, Config
from .drift import Drift
from .hypr import Hypr
from .model import load_model, subset
from .overlay_client import OverlayProcess
from .tracker import Camera, FaceTracker, pick_camera

WEIGHT = 3.0  # counts as much as three corrections
MIN_FRAMES = 10


def pause(ov: OverlayProcess, seconds: float) -> None:
    """Leave a message up until a key or `seconds`."""
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end and check_keys(ov) is None:
            time.sleep(0.05)
    except Cancelled:
        pass


def run(cfg: Config) -> int:
    if not CALIBRATION_PATH.exists():
        print("omarcheye: not calibrated yet; run `omarcheye calibrate`")
        return 1
    model = load_model(CALIBRATION_PATH)
    mon = Hypr().layout().monitor(model.monitor)
    if mon is None:
        print(f"omarcheye: calibrated monitor {model.monitor} is not connected")
        return 1
    try:
        info = pick_camera(model.camera if cfg.camera == "auto" and model.camera else cfg.camera)
    except RuntimeError as e:
        print(f"omarcheye: {e}")
        return 1

    tracker = FaceTracker(delegate=cfg.delegate, eyenet=cfg.eyenet, patches=cfg.patches)
    cam = Camera(info.device, cfg.width, cfg.height, cfg.fps)
    ov = OverlayProcess(mon.name, "calibrate")
    try:
        ready = ov.wait_for("ready", 8)
        width, height = ready["width"], ready["height"]
        aspect = height / width
        ov.send(cmd="text", text="omarcheye recentre\n\nLook at the dot")
        data, _ = _collect(ov, cam, tracker, [(0.5, 0.5)], width, height)
        check_keys(ov)
        data = subset(data, data["opens"] >= model.blink)
        raw = model.predict_data(data) if len(data["groups"]) else np.empty((0, 2))
        raw = raw[np.isfinite(raw).all(1)]
        if len(raw) < MIN_FRAMES:
            raise RuntimeError("no clear view of your eyes; check `omarcheye camera`")
        x, y = np.median(raw, 0)
        drift = Drift(aspect, max(model.error / 1.2533, 0.02), model.created, DRIFT_PATH)
        drift.records = []
        if drift.add(x, y, (0.5, 0.5, 0.5, 0.5), WEIGHT, margin=0.0) is None:
            raise RuntimeError(f"you look {100 * np.hypot(0.5 - x, (0.5 - y) * aspect):.0f}% of the screen width away "
                               "from where the calibration expects: run `omarcheye calibrate`")
        drift.save()
        text = (f"The estimate was {100 * abs(0.5 - x):.0f}% {'left' if x < 0.5 else 'right'} and "
                f"{100 * abs(0.5 - y) * aspect:.0f}% {'above' if y < 0.5 else 'below'} the dot "
                f"(share of the screen width); omarcheye now shifts it back")
        print(f"omarcheye: recentred. {text}")
        ov.send(cmd="text", text=f"Recentred\n\n{text}")
        pause(ov, 3)
        return 0
    except Cancelled:
        print("omarcheye: recentre cancelled")
        return 1
    except RuntimeError as e:
        print(f"omarcheye: recentre failed: {e}")
        ov.send(cmd="text", text=f"Recentre failed\n\n{e}")
        pause(ov, 4)
        return 1
    finally:
        ov.close()
        cam.close()
        tracker.close()
