"""omarcheye test: look at 9 dots to see how well the calibration does now.

Calibration scores itself on frames from the same sitting. This scores it
later, in whatever posture you are in, which is what using it is like. It
also fits every kind of model (basic and rich regressions, geometric) on the
stored calibration samples and scores each on the same 9 dots; if another
kind does clearly better, Enter switches to it. The test frames are kept in
~/.local/state/omarcheye/tests/ (numbers only) for later analysis.
"""

import time

import numpy as np

from . import samples
from .calibrate import Cancelled, _collect, check_keys, wait_for_start
from .config import CALIBRATION_PATH, STATE_DIR, Config
from .hypr import Hypr
from .model import data_error, fit_all, load_model, subset
from .overlay_client import OverlayProcess
from .tracker import Camera, FaceTracker, pick_camera

DOTS = [(x, y) for y in (0.2, 0.5, 0.8) for x in (0.15, 0.5, 0.85)]  # between the calibration dots
BETTER = 0.9  # another kind must beat the current one by 10% to be offered
NAMES = {"basic": "basic features", "rich": "rich features", "geometric": "geometric model"}

INTRO = (
    "omarcheye test\n\n"
    "Look at each dot until it shrinks away, sitting as you normally do.\n"
    "Nothing is changed unless you choose so at the end.\n\n"
    "Space: start      Esc: cancel"
)


def run(cfg: Config) -> int:
    if not CALIBRATION_PATH.exists():
        print("omarcheye: not calibrated yet; run `omarcheye calibrate`")
        return 1
    current = load_model(CALIBRATION_PATH)
    hypr = Hypr()
    mon = hypr.layout().monitor(current.monitor)
    if mon is None:
        print(f"omarcheye: calibrated monitor {current.monitor} is not connected")
        return 1
    try:
        info = pick_camera(cfg.camera)
    except RuntimeError as e:
        print(f"omarcheye: {e}")
        return 1
    screen_mm = hypr.physical_mm(mon.name)

    tracker = FaceTracker(delegate=cfg.delegate)
    cam = Camera(info.device, cfg.width, cfg.height, cfg.fps)
    ov = OverlayProcess(mon.name, "calibrate")
    try:
        ready = ov.wait_for("ready", 8)
        width, height = ready["width"], ready["height"]
        aspect = height / width
        wait_for_start(ov, cam, tracker, info.name, "", INTRO)
        dots = DOTS.copy()
        np.random.default_rng().shuffle(dots)
        data, _ = _collect(ov, cam, tracker, dots, width, height)
        ov.send(cmd="text", text="Scoring…")

        tests = STATE_DIR / "tests"
        tests.mkdir(parents=True, exist_ok=True)
        np.savez(tests / f"test-{time.strftime('%Y%m%d-%H%M%S')}.npz", camera=np.array(info.name),
                 monitor=np.array(mon.name), **data)
        if len(data["groups"]) == 0:
            raise RuntimeError("no usable face frames")

        def score(model) -> float:
            return data_error(model, subset(data, data["opens"] >= model.blink), aspect)

        now = score(current)
        mm = f" (≈ {now * screen_mm[0] / 10:.1f} cm)" if screen_mm else ""
        lines = [f"Current calibration ({NAMES.get(current.kind, current.kind)}): "
                 f"{100 * now:.1f}% of the screen width{mm}, against {100 * current.error:.1f}% when calibrated"]
        best = None
        stored = samples.load(info.name, mon.name)
        if stored is not None:
            models, *_ = fit_all(stored, aspect, mon.name, info.name, screen_mm=screen_mm)
            scores = sorted(((score(m), m) for m in models), key=lambda x: x[0])
            lines.append("Fitted on your calibration samples, on these dots: "
                         + ", ".join(f"{NAMES.get(m.kind, m.kind)} {100 * e:.1f}%" for e, m in scores))
            if scores[0][0] < BETTER * now:
                best = scores[0]
        print("omarcheye: " + "\nomarcheye: ".join(lines))
        prompt = (f"\n\nEnter: switch to the {NAMES.get(best[1].kind, best[1].kind)}      any other key: keep"
                  if best else "\n\nPress any key")
        ov.send(cmd="text", text="omarcheye test\n\n" + "\n".join(lines) + prompt)
        end = time.monotonic() + 30
        while time.monotonic() < end:
            key = check_keys(ov)
            if key is not None:
                if best and key in ("Return", "KP_Enter"):
                    best[1].save(CALIBRATION_PATH)
                    print(f"omarcheye: switched to the {NAMES.get(best[1].kind, best[1].kind)}")
                break
            time.sleep(0.05)
        return 0
    except Cancelled:
        print("omarcheye: test cancelled")
        return 1
    except RuntimeError as e:
        print(f"omarcheye: test failed: {e}")
        ov.send(cmd="text", text=f"Test failed\n\n{e}")
        time.sleep(3)
        return 1
    finally:
        ov.close()
        cam.close()
        tracker.close()
