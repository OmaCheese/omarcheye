"""Gaze from full-resolution eye crops: Intel's gaze-estimation-adas-0002.

MediaPipe works on a 256-pixel face crop, so its iris is a few pixels wide.
This network looks at each eye at the camera's own resolution instead: a
60x60 crop per eye, plus the head's yaw, pitch and roll, and returns where
the eyes point. It is a small convolutional network (1.9 million weights)
from Intel's Open Model Zoo (Apache-2.0), run with OpenVINO on the central
processing unit (CPU): about 2 ms per run on one core of a Ryzen 7 5800H.

The preprocessing follows Open Model Zoo's gaze_estimation_demo (C++):

- eye crop: a square 1.8 eye widths wide (corner to corner), centred
  between the corners, cut out of the full frame;
- head roll: the crop is turned about its centre by the roll, so the eye
  lies level, and the network gets roll 0; its answer is turned back by the
  roll afterwards;
- head pose: yaw, pitch and roll in degrees as head-pose-estimation-adas-0001
  gives them (axes X towards the camera, Y right, Z up;
  R = Rz(yaw) Ry(pitch) Rx(roll); yaw > 0 turns the face to the image's
  right, pitch > 0 down, roll > 0 clockwise in the image); here they come
  from MediaPipe's face transformation matrix;
- "left_eye_image" is the eye on the image's left (the subject's right);
- crops are BGR, 0..255, resized with bicubic interpolation.

The output's axes, checked on MPIIFaceGaze (the model card says z points
from the eyes to the camera; for this model it points the other way): x to
the image's right, y up, z along the line of sight from the camera to the
eyes. Turned into the camera's axes with that line of sight, the gaze
direction is 7.8° off the truth on average over 35,000 MPIIFaceGaze images,
with no calibration (7.3° averaged with the mirrored crops, below).

With `flip` (the default), each frame also runs on the mirrored crops (left
and right swapped, yaw negated) and the two answers are averaged: one more
run, and a steadier, slightly more accurate gaze.

Features (FEATURES), in MediaPipe's camera axes (x right, y up, z towards
the viewer, as tracker.head_pose uses):
  gyaw, gpitch  the gaze's direction (radians; > 0: to the image's right, up)
  hitx, hity    where the gaze ray from the eyes meets the camera's plane
                (cm, x right, y up from the camera; the eyes' position from
                MediaPipe's distance). With the camera on the screen, this
                is nearly a point on the screen, so a linear calibration
                maps it well even when the head moves.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR, INSTALL

EYENET_PATH = DATA_DIR / "models/gaze-estimation-adas-0002.xml"
EYENET_URL = ("https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
              "models_bin/1/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002")  # + .xml and .bin

# MediaPipe landmarks. "A" is the eye on the image's left (the subject's right).
A_OUT, A_IN, A_UP, A_LO = 33, 133, 159, 145
B_IN, B_OUT, B_UP, B_LO = 362, 263, 386, 374

FEATURES = ("gyaw", "gpitch", "hitx", "hity")
SIZE = 60  # the network's eye crops are SIZE x SIZE pixels
SCALE = 1.8  # crop side, in eye widths (corner to corner)
MP_VFOV = 63.0  # MediaPipe's face geometry assumes this vertical field of view (degrees)

# MediaPipe camera axes (x right, y up, z towards the viewer) -> the head pose
# network's (X towards the camera, Y right, Z up): OMZ = PERM @ MP.
PERM = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _import_openvino():
    # `import openvino` also imports its model converter, which sends a Google
    # Analytics event on every import unless the user has opted out. omarcheye
    # converts nothing: an empty entry makes that import fail, and openvino
    # skips it.
    sys.modules.setdefault("openvino.tools.ovc", None)
    import openvino

    return openvino


def omz_angles(matrix) -> tuple[float, float, float]:
    """Yaw, pitch, roll (degrees) of MediaPipe's face transformation matrix,
    in head-pose-estimation-adas-0001's convention."""
    r = np.asarray(matrix, float)[:3, :3]
    r = r / np.linalg.norm(r, axis=0)  # in case the matrix carries a scale
    r = PERM @ r @ PERM.T
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    yaw = math.atan2(r[1, 0], r[0, 0])
    roll = math.atan2(-r[2, 1], r[2, 2])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def sight_frame(ray: np.ndarray) -> np.ndarray:
    """Columns: the network's output axes (x right, y up, z away from the
    camera along `ray`) in camera coordinates (x right, y down, z forward)."""
    z = ray / np.linalg.norm(ray)
    up = np.array([0.0, -1.0, 0.0])
    y = up - (up @ z) * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(z, y), y, z], 1)


def eye_crop(frame: np.ndarray, p1, p2, roll: float, scale: float = SCALE) -> np.ndarray | None:
    """The demo's eye crop: a square `scale` eye widths wide around the corners'
    midpoint, turned by `roll` degrees (counter-clockwise) about its centre,
    resized to SIZE x SIZE. None when it leaves the frame or is tiny."""
    p1 = np.rint(p1).astype(int)
    p2 = np.rint(p2).astype(int)
    w = int(scale * float(np.hypot(*(p1 - p2))))
    mx, my = np.rint((p1 + p2) / 2).astype(int)
    x, y = mx - w // 2, my - w // 2
    h, fw = frame.shape[:2]
    if w < 8 or x < 0 or y < 0 or x + w > fw or y + w > h:
        return None
    crop = frame[y:y + w, x:x + w]
    if roll:
        m = cv2.getRotationMatrix2D((float(w // 2), float(w // 2)), float(roll), 1.0)
        crop = cv2.warpAffine(crop, m, (w, w), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)


def openness(points: np.ndarray, out: int, inner: int, up: int, lo: int) -> float:
    """Eyelid gap over eye width."""
    return float(np.linalg.norm(points[lo] - points[up]) / max(np.linalg.norm(points[out] - points[inner]), 1e-6))


class EyeNet:
    """Load the network once; call it with a frame, MediaPipe's points (pixels)
    and its 4x4 face transformation matrix.

    `focal` is the camera's focal length in pixels, for the line of sight to
    the eyes; None takes MediaPipe's own assumption (63° vertical field of
    view), which also sets the distance it reports.
    """

    def __init__(self, model_path: Path = EYENET_PATH, threads: int = 1, focal: float | None = None,
                 flip: bool = True, min_width: float = 12.0, min_open: float = 0.12):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} is missing; run {INSTALL}, or download {EYENET_URL}.xml and .bin next to it")
        core = _import_openvino().Core()
        net = core.compile_model(core.read_model(str(model_path)), "CPU",
                                 {"INFERENCE_NUM_THREADS": int(threads), "PERFORMANCE_HINT": "LATENCY"})
        self.request = net.create_infer_request()
        self.focal = focal
        self.flip = flip
        self.min_width = min_width  # eye narrower than this (pixels): too small to read
        self.min_open = min_open  # lid gap below this share of the eye width: closed or blinking
        self.last: dict | None = None  # details of the last call, for debugging and drawing

    def infer(self, left: np.ndarray, right: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        """The network's gaze (unit vector) for two SIZE x SIZE BGR crops, head roll 0."""
        self.request.infer({
            "left_eye_image": left.transpose(2, 0, 1)[None].astype(np.float32),
            "right_eye_image": right.transpose(2, 0, 1)[None].astype(np.float32),
            "head_pose_angles": np.array([[yaw, pitch, 0.0]], np.float32),
        })
        g = np.array(self.request.get_output_tensor(0).data[0], float)
        return g / np.linalg.norm(g)

    def __call__(self, frame: np.ndarray, points: np.ndarray, matrix) -> np.ndarray | None:
        """FEATURES for one frame, or None when the eyes aren't usable."""
        self.last = None
        p = np.asarray(points, float)
        if not np.isfinite(p[[A_OUT, A_IN, A_UP, A_LO, B_IN, B_OUT, B_UP, B_LO]]).all():
            return None
        if min(np.linalg.norm(p[A_OUT] - p[A_IN]), np.linalg.norm(p[B_OUT] - p[B_IN])) < self.min_width:
            return None
        if min(openness(p, A_OUT, A_IN, A_UP, A_LO), openness(p, B_OUT, B_IN, B_UP, B_LO)) < self.min_open:
            return None
        m = np.asarray(matrix, float)
        yaw, pitch, roll = omz_angles(m)
        left = eye_crop(frame, p[A_OUT], p[A_IN], roll)
        right = eye_crop(frame, p[B_IN], p[B_OUT], roll)
        if left is None or right is None:
            return None
        g = self.infer(left, right, yaw, pitch)
        if self.flip:
            g2 = self.infer(cv2.flip(right, 1), cv2.flip(left, 1), -yaw, pitch)
            g = g + g2 * (-1.0, 1.0, 1.0)
            g /= np.linalg.norm(g)
        cs, sn = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        g = np.array([g[0] * cs + g[1] * sn, -g[0] * sn + g[1] * cs, g[2]])  # undo the levelling
        h, w = frame.shape[:2]
        f = self.focal or (h / 2) / math.tan(math.radians(MP_VFOV / 2))
        mid = (p[A_OUT] + p[A_IN] + p[B_IN] + p[B_OUT]) / 4
        ray = np.array([(mid[0] - w / 2) / f, (mid[1] - h / 2) / f, 1.0])
        c = sight_frame(ray) @ g  # camera coordinates: x right, y down, z forward
        if c[2] > -0.2:  # looking more than ~78° away from the camera's axis
            return None
        eye = ray * max(-m[2, 3], 1.0)  # cm, at MediaPipe's distance
        hit = eye[:2] - eye[2] / c[2] * c[:2]
        self.last = {"left": left, "right": right, "head": (yaw, pitch, roll), "sight": g, "gaze": c,
                     "ray": ray, "eye": eye}
        return np.array([math.atan2(c[0], -c[2]), math.atan2(-c[1], math.hypot(c[0], c[2])), hit[0], -hit[1]])
