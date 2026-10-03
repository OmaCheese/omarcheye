"""Camera frames -> face landmarks (MediaPipe Face Landmarker) -> gaze features.

Features per frame: where each iris sits inside its eye (u across, v down,
in eye widths; both eyes averaged) plus head pose (yaw, pitch, and position
relative to the camera). The calibration (model.py) maps them to the screen.
"""

import math
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import MODEL_PATH

# Landmark indices. "A" is the eye on the image's left (the subject's right).
A_LEFT, A_RIGHT, A_UPPER, A_LOWER, A_IRIS = 33, 133, 159, 145, 468
B_LEFT, B_RIGHT, B_UPPER, B_LOWER, B_IRIS = 362, 263, 386, 374, 473

FEATURES = ("u", "v", "yaw", "pitch", "hx", "hy", "hz")


@dataclass
class Sample:
    t: float
    feat: np.ndarray  # ordered as FEATURES
    openness: float  # eyelid gap / eye width, both eyes averaged


def eye_features(p: np.ndarray, left: int, right: int, upper: int, lower: int, iris: int):
    """Iris offset from the eye centre, in eye widths, along and across the
    corner-to-corner line (so head roll cancels out), and eyelid openness."""
    a, b = p[left], p[right]
    width = float(np.linalg.norm(b - a))
    ex = (b - a) / width
    ey = np.array([-ex[1], ex[0]])
    d = p[iris] - (a + b) / 2
    return float(d @ ex) / width, float(d @ ey) / width, float(np.linalg.norm(p[upper] - p[lower])) / width


def head_pose(matrix) -> tuple[float, float, float, float, float]:
    """Yaw and pitch (radians) of the face's forward axis, and its position:
    sideways and vertical offset per unit distance, and distance (cm)."""
    m = np.asarray(matrix, dtype=float)
    forward = m[:3, :3] @ np.array([0.0, 0.0, 1.0])
    yaw = math.atan2(forward[0], forward[2])
    pitch = math.atan2(forward[1], math.hypot(forward[0], forward[2]))
    tx, ty, tz = m[:3, 3]
    dist = max(-tz, 1.0)
    return yaw, pitch, tx / dist, ty / dist, dist


def features(points: np.ndarray, matrix, t: float) -> Sample:
    ua, va, oa = eye_features(points, A_LEFT, A_RIGHT, A_UPPER, A_LOWER, A_IRIS)
    ub, vb, ob = eye_features(points, B_LEFT, B_RIGHT, B_UPPER, B_LOWER, B_IRIS)
    feat = np.array([(ua + ub) / 2, (va + vb) / 2, *head_pose(matrix)])
    return Sample(t, feat, (oa + ob) / 2)


def _import_mediapipe():
    # mediapipe imports sounddevice for its audio tasks, and initialising
    # PortAudio goes through ALSA into PipeWire, whose realtime module leaves
    # this process with a realtime CPU-time limit of 0. The kernel then
    # SIGKILLs it as soon as inference starts. omeye records no audio, so
    # mediapipe gets an empty sounddevice module.
    sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    return mp, BaseOptions, vision


def integrated_gpu() -> str | None:
    """DRI_PRIME name (pci-0000_07_00_0) of the first non-NVIDIA render node."""
    for link in sorted(Path("/dev/dri/by-path").glob("pci-*-render")):
        node = Path("/sys/class/drm") / link.resolve().name
        try:
            if (node / "device/vendor").read_text().strip() != "0x10de":
                return link.name.removesuffix("-render").replace(":", "_").replace(".", "_")
        except OSError:
            continue
    return None


class FaceTracker:
    def __init__(self, model_path: Path = MODEL_PATH, delegate: str = "cpu"):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} is missing; run ./install.sh")
        if delegate == "gpu":
            # MediaPipe opens EGL on the first render node, which is the NVIDIA
            # card here; Mesa can't drive that and silently falls back to
            # software rendering. DRI_PRIME points Mesa at the integrated GPU.
            gpu = integrated_gpu()
            if gpu is None:
                raise RuntimeError("delegate = \"gpu\" but no integrated GPU found")
            os.environ["DRI_PRIME"] = gpu
        elif delegate != "cpu":
            raise ValueError(f"delegate must be \"cpu\" or \"gpu\", not {delegate!r}")
        self.mp, BaseOptions, vision = _import_mediapipe()
        Delegate = BaseOptions.Delegate
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path), delegate=Delegate.GPU if delegate == "gpu" else Delegate.CPU
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_ms = -1

    def process(self, frame_bgr: np.ndarray, t: float) -> Sample | None:
        ms = max(int(t * 1000), self.last_ms + 1)  # VIDEO mode needs rising timestamps
        self.last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return None
        h, w = frame_bgr.shape[:2]
        points = np.array([(q.x * w, q.y * h) for q in result.face_landmarks[0]])
        return features(points, result.facial_transformation_matrixes[0], t)

    def close(self) -> None:
        self.landmarker.close()


BUILT_IN = re.compile(r"integrated|wide.?vision|built.?in|internal|laptop", re.I)


def list_cameras() -> list[tuple[str, str]]:
    """(device, name) for every video capture node."""
    cams = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        try:
            if (node / "index").read_text().strip() != "0":
                continue  # metadata node of the same camera
            cams.append((f"/dev/{node.name}", (node / "name").read_text().strip()))
        except OSError:
            continue
    return cams


def pick_camera(spec: str) -> tuple[str, str]:
    cams = list_cameras()
    if spec.startswith("/dev/"):
        return spec, dict(cams).get(spec, spec)
    if spec != "auto":
        for dev, name in cams:
            if spec.lower() in name.lower():
                return dev, name
        raise RuntimeError(f"no camera named like {spec!r}; `omeye cameras` lists them")
    if not cams:
        raise RuntimeError("no camera found")
    plug_in = [c for c in cams if not BUILT_IN.search(c[1])]
    return (plug_in or cams)[0]


class Camera:
    def __init__(self, device: str, width: int, height: int, fps: int):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, not a queue of old ones

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def skip(self) -> bool:
        """Take the next frame without decoding it."""
        return self.cap.grab()

    def size(self) -> tuple[int, int]:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def close(self) -> None:
        self.cap.release()
