"""The raw calibration samples behind the current calibration: gaze features
and where you were looking, numbers only (no images). `omarcheye calibrate`
starts them afresh; `omarcheye refine` adds to them and refits."""

import numpy as np

from .config import SAMPLES_PATH

KEYS = ("feats", "rich", "pose", "opens", "groups", "targets", "net")
NET = 4  # eyenet.FEATURES; samples from before the eye network get NaN there


def load(camera: str, monitor: str) -> dict | None:
    """The stored samples, if they were taken with this camera and monitor."""
    if not SAMPLES_PATH.exists():
        return None
    with np.load(SAMPLES_PATH) as d:
        if "camera" not in d or str(d["camera"]) != camera or str(d["monitor"]) != monitor:
            return None
        if any(k not in d for k in KEYS if k != "net"):
            return None  # from an older omarcheye: start afresh
        data = {k: d[k] for k in KEYS if k in d}
    data.setdefault("net", np.full((len(data["groups"]), NET), np.nan))
    return data


def save(camera: str, monitor: str, data: dict, **extra) -> None:
    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.setdefault("net", np.full((len(data["groups"]), NET), np.nan))
    np.savez(SAMPLES_PATH, camera=np.array(camera), monitor=np.array(monitor),
             **{k: data[k] for k in KEYS}, **extra)


def merge(old: dict | None, new: dict) -> dict:
    if old is None or len(old["groups"]) == 0:
        return new
    for d in (old, new):
        d.setdefault("net", np.full((len(d["groups"]), NET), np.nan))
    return {k: np.concatenate([old[k], new[k]]) for k in KEYS}
