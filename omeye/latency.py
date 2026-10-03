"""omeye latency: how long the camera takes to show omeye what happens.

The screen flashes white a few times; the light falls on your face (and the
room), and omeye times how long until a camera frame shows it brighter. That
is the camera's latency as omeye sees it: exposure, the phone's encoding and
the stream for Flux, decoding, and reading the frame. Sit in front of the
camera; a dark room shows the flashes best. The time counts from when the
overlay drew the flash, so it includes the monitor's own delay (a refresh or two).
"""

import random
import time

import numpy as np

from .calibrate import Cancelled, check_keys
from .config import Config
from .hypr import Hypr
from .overlay_client import OverlayProcess
from .tracker import Camera, pick_camera

FLASHES = 8
ON_S = 0.8
OFF_S = (1.2, 1.8)


def edge(times: np.ndarray, light: np.ndarray, t0: float, rising: bool) -> float | None:
    """Seconds from t0 until the first frame halfway between the light before
    t0 and the extreme within ON_S after it; None if there's no clear step."""
    before = light[(times >= t0 - 0.4) & (times < t0)]
    after = (times >= t0) & (times < t0 + ON_S)
    if len(before) < 3 or after.sum() < 3:
        return None
    base = float(np.median(before))
    extreme = float(light[after].max() if rising else light[after].min())
    if abs(extreme - base) < 1.5 or abs(extreme - base) < 6 * float(np.std(before)):
        return None  # no step that stands out from the noise
    half = (base + extreme) / 2
    crossed = (light > half) if rising else (light < half)
    hits = np.flatnonzero(after & crossed)
    return float(times[hits[0]] - t0) if len(hits) else None


def run(cfg: Config) -> int:
    try:
        info = pick_camera(cfg.camera)
    except RuntimeError as e:
        print(f"omeye: {e}")
        return 1
    hypr = Hypr()
    mon = next((m for m in hypr.layout().monitors if m.focused), None)
    cam = Camera(info.device, cfg.width, cfg.height, cfg.fps)
    ov = OverlayProcess(mon.name, "calibrate")
    try:
        ov.wait_for("ready", 8)
        ov.send(cmd="text", text=f"omeye latency\n\nKeep your face in view of {info.name}.\n"
                                 f"The screen flashes white {FLASHES} times.\n\nEsc: cancel")
        times, light, painted = [], [], []
        # (seconds from the start, level): dark, then flashes at irregular times
        schedule, t = [(1.5, 0.0)], 1.5
        for _ in range(FLASHES):
            t += random.uniform(*OFF_S)
            schedule += [(t, 1.0), (t + ON_S, 0.0)]
            t += ON_S
        start = time.monotonic()
        while schedule:
            frame = cam.read()
            now = time.monotonic()
            if frame is not None:
                times.append(now)
                light.append(float(frame[::8, ::8].mean()))
            if now - start >= schedule[0][0]:
                ov.send(cmd="fill", level=schedule.pop(0)[1])
            while ev := ov.poll():
                if ev.get("event") == "painted":
                    painted.append(ev["t"])
                elif ev.get("event") == "closed" or ev.get("key") == "Escape":
                    raise Cancelled
            if now - start > 60:
                raise RuntimeError("the camera sent too few frames")
        end = time.monotonic() + 1.0
        while time.monotonic() < end:  # the last flash's falling edge
            frame = cam.read()
            if frame is not None:
                times.append(time.monotonic())
                light.append(float(frame[::8, ::8].mean()))
            while ev := ov.poll():
                if ev.get("event") == "painted":
                    painted.append(ev["t"])
        check_keys(ov)
        times, light = np.array(times), np.array(light)
        # painted: the dark start, then on, off per flash
        rises = [edge(times, light, t, True) for t in painted[1::2]]
        falls = [edge(times, light, t, False) for t in painted[2::2]]
        found = [x for x in rises + falls if x is not None]
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
        if len(found) < 3:
            text = ("The flashes didn't show clearly in the camera image "
                    f"({len(found)} of {2 * FLASHES} edges): sit closer, or darken the room")
        else:
            ms = 1000 * np.array(found)
            text = (f"Camera latency {np.median(ms):.0f} ms (median of {len(found)} edges, "
                    f"{np.min(ms):.0f}–{np.max(ms):.0f} ms), {fps:.1f} frames per second from {info.name}")
        print(f"omeye: {text}")
        ov.send(cmd="fill", level=0)
        ov.send(cmd="text", text=f"omeye latency\n\n{text}\n\nPress any key")
        end = time.monotonic() + 8
        while time.monotonic() < end and check_keys(ov) is None:
            time.sleep(0.05)
        return 0
    except Cancelled:
        print("omeye: latency test cancelled")
        return 1
    except RuntimeError as e:
        print(f"omeye: latency test failed: {e}")
        return 1
    finally:
        ov.close()
        cam.close()
