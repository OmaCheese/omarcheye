"""Camera frames -> face landmarks (MediaPipe Face Landmarker) -> gaze features.

Features per frame: where each iris sits inside its eye (u across, v down,
in eye widths; both eyes averaged) plus head pose (yaw, pitch, and position
relative to the camera). The calibration (model.py) maps them to the screen.
"""

import fcntl
import math
import os
import re
import struct
import sys
import types
from dataclasses import dataclass
from pathlib import Path

# A camera that stops sending (a phone webcam whose stream ends) makes
# OpenCV's V4L2 read() wait this many seconds before failing; the default is 10.
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# OpenCV's own thread pool busy-waits between frames: with it, the colour
# conversions cost well over a core of spinning at 30 fps for ~2 ms of work.
cv2.setNumThreads(1)

from .config import MODEL_PATH

# Landmark indices. "A" is the eye on the image's left (the subject's right).
A_LEFT, A_RIGHT, A_UPPER, A_LOWER, A_IRIS = 33, 133, 159, 145, 468
B_LEFT, B_RIGHT, B_UPPER, B_LOWER, B_IRIS = 362, 263, 386, 374, 473

HEAD = ("yaw", "pitch", "hx", "hy", "hz")
FEATURES = ("u", "v", *HEAD)  # "basic": iris offsets of both eyes averaged, head pose
BLEND = ("eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
         "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight")
# "rich": each eye's iris offsets and lid positions, MediaPipe's eye-direction scores, head pose
RICH = ("uA", "vA", "uB", "vB", "upA", "loA", "upB", "loB", *BLEND, *HEAD)


EDGE = 0.01  # a landmark closer than this to the image border (fraction of the image): face cut off


@dataclass
class Sample:
    t: float
    feat: np.ndarray  # ordered as FEATURES
    openness: float  # eyelid gap / eye width, both eyes averaged
    pos: tuple[float, float] = (0.5, 0.5)  # face centre in the camera image, 0..1
    margin: float = 1.0  # nearest landmark to the image border, as a fraction of the image
    points: np.ndarray | None = None  # all landmarks in image pixels, for the camera view
    rich: np.ndarray | None = None  # ordered as RICH
    pose: np.ndarray | None = None  # MediaPipe's face transformation matrix, flattened (16): rotation, position in cm

    @property
    def cut_off(self) -> bool:
        """Part of the face is outside the image, so the eye landmarks are guesses."""
        return self.margin < EDGE


def framing_advice(pos: tuple[float, float], margin: float) -> str:
    """What to do about the camera's aim, or "" when the face sits well."""
    x, y = pos
    if y < 0.3:
        return "Your face is near the top of the camera image: tilt the camera up"
    if y > 0.7:
        return "Your face is near the bottom of the camera image: tilt the camera down"
    if not 0.25 <= x <= 0.75:
        return "Your face is near the side of the camera image: turn the camera towards you"
    if margin < EDGE:
        return "Part of your face is outside the camera image: move further from the camera"
    return ""


def eye_features(p: np.ndarray, left: int, right: int, upper: int, lower: int, iris: int):
    """Iris offset from the eye centre, in eye widths, along and across the
    corner-to-corner line (so head roll cancels out), and eyelid openness."""
    u, v, up, lo = eye_detail(p, left, right, upper, lower, iris)
    return u, v, lo - up


def eye_detail(p: np.ndarray, left: int, right: int, upper: int, lower: int, iris: int):
    """Iris offset (u across, v down) and the upper and lower lid's offset
    across the corner line, all in eye widths. The upper lid follows the
    eye up and down, which helps the weak vertical direction."""
    a, b = p[left], p[right]
    width = float(np.linalg.norm(b - a))
    ex = (b - a) / width
    ey = np.array([-ex[1], ex[0]])
    c = (a + b) / 2
    d = p[iris] - c
    return (float(d @ ex) / width, float(d @ ey) / width,
            float((p[upper] - c) @ ey) / width, float((p[lower] - c) @ ey) / width)


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


def features(points: np.ndarray, matrix, t: float, blend: dict[str, float] | None = None) -> Sample:
    ua, va, upa, loa = eye_detail(points, A_LEFT, A_RIGHT, A_UPPER, A_LOWER, A_IRIS)
    ub, vb, upb, lob = eye_detail(points, B_LEFT, B_RIGHT, B_UPPER, B_LOWER, B_IRIS)
    head = head_pose(matrix)
    feat = np.array([(ua + ub) / 2, (va + vb) / 2, *head])
    rich = None
    if blend is not None:
        rich = np.array([ua, va, ub, vb, upa, loa, upb, lob, *(blend.get(n, 0.0) for n in BLEND), *head])
    return Sample(t, feat, (loa - upa + lob - upb) / 2, rich=rich, pose=np.asarray(matrix, float).ravel())


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
            output_face_blendshapes=True,
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
        blend = {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else None
        sample = features(points, result.facial_transformation_matrixes[0], t, blend)
        lo, hi = points.min(0), points.max(0)
        sample.pos = (float(lo[0] + hi[0]) / 2 / w, float(lo[1] + hi[1]) / 2 / h)
        sample.margin = float(min(lo[0] / w, lo[1] / h, 1 - hi[0] / w, 1 - hi[1] / h))
        sample.points = points
        return sample

    def close(self) -> None:
        self.landmarker.close()


BUILT_IN = re.compile(r"integrated|wide.?vision|built.?in|internal|laptop", re.I)

VIDIOC_QUERYCAP = 0x80685600  # _IOR('V', 0, struct v4l2_capability), 104 bytes
CAP_VIDEO_CAPTURE = 0x00000001
CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
CAP_DEVICE_CAPS = 0x80000000


@dataclass
class CameraInfo:
    device: str
    name: str
    live: bool  # offers video capture now; a virtual camera only does while something feeds it
    virtual: bool  # v4l2loopback, e.g. Flux's phone webcam
    builtin: bool

    def not_live_reason(self) -> str:
        if self.virtual:
            return (f"{self.name} ({self.device}) is a virtual camera and nothing is feeding it; "
                    "for Flux, start the webcam from the phone")
        return f"{self.name} ({self.device}) is not offering video"


def _device_caps(device: str) -> int | None:
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        buf = fcntl.ioctl(fd, VIDIOC_QUERYCAP, bytes(104))
    except OSError:
        return None
    finally:
        os.close(fd)
    caps, device_caps = struct.unpack_from("<II", buf, 84)
    return device_caps if caps & CAP_DEVICE_CAPS else caps


def list_cameras() -> list[CameraInfo]:
    """Every camera: capture nodes, plus virtual cameras even while idle.
    Metadata nodes (a second /dev/video per webcam) are left out."""
    cams = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        device = f"/dev/{node.name}"
        try:
            name = (node / "name").read_text().strip()
            virtual = "/virtual/" in os.path.realpath(node / "device")
        except OSError:
            continue
        caps = _device_caps(device)
        live = caps is not None and bool(caps & (CAP_VIDEO_CAPTURE | CAP_VIDEO_CAPTURE_MPLANE))
        if live or virtual:
            cams.append(CameraInfo(device, name, live, virtual, bool(BUILT_IN.search(name))))
    return cams


def pick_camera(spec: str, quiet: bool = False, cams: list[CameraInfo] | None = None) -> CameraInfo:
    """The camera named by `spec` ("auto", a /dev path or part of a name).

    "auto" takes a camera that is sending video, preferring anything over the
    laptop's built-in one. Raises RuntimeError when the camera isn't there.
    """
    cams = list_cameras() if cams is None else cams
    if spec.startswith("/dev/"):
        cam = next((c for c in cams if c.device == spec), None) or CameraInfo(spec, spec, True, False, False)
    elif spec != "auto":
        cam = next((c for c in cams if spec.lower() in c.name.lower()), None)
        if cam is None:
            raise RuntimeError(f"no camera named like {spec!r}; `omeye cameras` lists them")
    else:
        live = [c for c in cams if c.live]
        idle = [c for c in cams if not c.live]
        if not live:
            raise RuntimeError(idle[0].not_live_reason() if idle else "no camera found")
        cam = next((c for c in live if not c.builtin), live[0])
        if cam.builtin and idle and not quiet:
            print(f"omeye: {idle[0].not_live_reason()}; using {cam.name}", file=sys.stderr)
    if not cam.live:
        raise RuntimeError(cam.not_live_reason())
    return cam


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
