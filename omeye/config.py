"""Settings, file locations and the user config file (~/.config/omeye/config.toml)."""

import os
import sys
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config")) / "omeye"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", HOME / ".local/state")) / "omeye"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", HOME / ".local/share")) / "omeye"

CONFIG_PATH = CONFIG_DIR / "config.toml"
CALIBRATION_PATH = STATE_DIR / "calibration.json"
SAMPLES_PATH = STATE_DIR / "calibration-samples.npz"  # raw numbers from the last calibration
MODEL_PATH = DATA_DIR / "models/face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
IDLE_HELPER = ROOT / "build/omeye-idle"
SERVICE = "omeye.service"


@dataclass
class Config:
    # Camera: "auto" prefers a plug-in webcam over the laptop's built-in one.
    camera: str = "auto"
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Where the landmark model runs: "cpu", or "gpu" = the integrated GPU
    # (OpenGL ES through Mesa; the NVIDIA card is skipped).
    delegate: str = "cpu"

    # Monitor the camera sits on; "" = the monitor focused at calibration.
    monitor: str = ""
    calibration_points: int = 15

    # Focus switching.
    dwell_ms: int = 250  # the likely window must stay confident this long (the belief itself takes a few frames)
    confidence: float = 0.9  # how sure omeye must be that you look at a window before focusing it
    quick_confidence: float = 0.99  # this sure, and focus moves after quick_ms instead
    quick_ms: int = 0
    switch_rate: float = 1.5  # expected gaze moves between windows per second
    typing_grace_ms: int = 700  # no switching until this long after the last key/click
    mouse_grace_ms: int = 2000  # the mouse wins for this long after it moves
    cooldown_ms: int = 300  # minimum time between two switches
    lost_ms: int = 500  # face missing this long resets the gaze filter
    away_ms: int = 3000  # face missing this long: check only every 6th frame
    offscreen: float = 0.15  # gaze further outside the monitor than this (fraction) counts as looking away

    # One Euro filter on the gaze point the preview shows (units: monitor widths).
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.5

    notify: bool = True


def load(path: Path = CONFIG_PATH) -> Config:
    cfg = Config()
    if not path.exists():
        return cfg
    with path.open("rb") as f:
        data = tomllib.load(f)
    known = {f.name: f.type for f in fields(Config)}
    for key, value in data.items():
        if key not in known:
            print(f"omeye: ignoring unknown setting {key!r} in {path}", file=sys.stderr)
            continue
        setattr(cfg, key, type(getattr(cfg, key))(value))
    return cfg
