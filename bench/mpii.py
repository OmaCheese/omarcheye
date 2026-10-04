"""MPIIFaceGaze: annotations and per-subject calibration (screen, camera, monitor pose).

Dataset layout (~/.cache/omarcheye/bench/MPIIFaceGaze/pXX):
  pXX.txt                 one line per image, see `load_annotations`
  Calibration/Camera.mat  cameraMatrix (3x3), distCoeffs
  Calibration/monitorPose.mat  rvecs, tvecs: the screen plane in camera coordinates (mm)
  Calibration/screenSize.mat   width/height in pixels and mm

The .mat files are MATLAB v5; a minimal reader below (numeric arrays only,
zlib-compressed or not) avoids a scipy dependency.

Camera coordinates: x right, y down, z forward (into the scene), mm.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache/omarcheye/bench"
ROOT = CACHE / "MPIIFaceGaze"
SUBJECTS = tuple(f"p{i:02d}" for i in range(15))

# ---------------------------------------------------------------- MATLAB v5

_NUMERIC = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}


def _element(buf: bytes, pos: int) -> tuple[int, bytes, int]:
    """(type, data, next position) of the data element at `pos`."""
    typ, n = struct.unpack_from("<II", buf, pos)
    if typ >> 16:  # small data element: 4-byte tag, up to 4 bytes of data
        n, typ = typ >> 16, typ & 0xFFFF
        return typ, buf[pos + 4:pos + 4 + n], pos + 8
    data = buf[pos + 8:pos + 8 + n]
    if typ == 15:  # miCOMPRESSED is not padded
        return typ, data, pos + 8 + n
    return typ, data, pos + 8 + n + (-n % 8)


def _matrix(data: bytes) -> tuple[str, np.ndarray | None]:
    _, flags, pos = _element(data, 0)
    cls = flags[0]
    _, dims, pos = _element(data, pos)
    shape = tuple(np.frombuffer(dims, "<i4"))
    _, name, pos = _element(data, pos)
    if cls not in range(6, 16):  # not a numeric array (cell, struct, char, ...)
        return name.decode(), None
    typ, real, pos = _element(data, pos)
    arr = np.frombuffer(real, _NUMERIC[typ]).astype(float).reshape(shape, order="F")
    return name.decode(), arr


def loadmat(path: Path) -> dict[str, np.ndarray]:
    """Numeric variables of a MATLAB v5 .mat file, as float arrays."""
    buf = Path(path).read_bytes()
    out, pos = {}, 128
    while pos + 8 <= len(buf):
        typ, data, pos = _element(buf, pos)
        if typ == 15:
            typ, data, _ = _element(zlib.decompress(data), 0)
        if typ == 14:
            name, arr = _matrix(data)
            if arr is not None:
                out[name] = arr
    return out


# ---------------------------------------------------------------- dataset


@dataclass
class Calibration:
    screen_px: np.ndarray  # (2,) width, height in pixels
    screen_mm: np.ndarray  # (2,) width, height in mm
    camera: np.ndarray  # (3, 3) intrinsic matrix
    dist: np.ndarray  # distortion coefficients
    monitor_R: np.ndarray  # (3, 3) screen -> camera rotation
    monitor_t: np.ndarray  # (3,) screen origin (top-left corner) in camera mm

    def px_to_mm(self, px: np.ndarray) -> np.ndarray:
        """Screen pixels -> mm from the screen's top-left corner (x right, y down)."""
        return np.asarray(px, float) * (self.screen_mm / self.screen_px)

    def screen_to_camera(self, mm: np.ndarray) -> np.ndarray:
        """Points on the screen (mm, x right, y down from top-left) -> camera coords (mm)."""
        mm = np.atleast_2d(np.asarray(mm, float))
        p = np.column_stack([mm, np.zeros(len(mm))])
        return p @ self.monitor_R.T + self.monitor_t


@dataclass
class Annotations:
    files: np.ndarray  # (n,) "dayNN/NNNN.jpg"
    day: np.ndarray  # (n,) int
    target_px: np.ndarray  # (n, 2) gaze target on screen, pixels
    landmarks6: np.ndarray  # (n, 6, 2) eye corners and mouth corners, image pixels
    head_rvec: np.ndarray  # (n, 3)
    head_tvec: np.ndarray  # (n, 3) mm
    face_center: np.ndarray  # (n, 3) mm, camera coords
    gaze_target: np.ndarray  # (n, 3) mm, camera coords
    eye: np.ndarray  # (n,) "left" / "right"

    def __len__(self) -> int:
        return len(self.files)


def _rodrigues(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float).ravel()
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


@lru_cache(maxsize=None)
def load_calibration(subject: str) -> Calibration:
    d = ROOT / subject / "Calibration"
    cam = loadmat(d / "Camera.mat")
    mon = loadmat(d / "monitorPose.mat")
    scr = loadmat(d / "screenSize.mat")
    return Calibration(
        screen_px=np.array([scr["width_pixel"].item(), scr["height_pixel"].item()]),
        screen_mm=np.array([scr["width_mm"].item(), scr["height_mm"].item()]),
        camera=cam["cameraMatrix"],
        dist=cam["distCoeffs"].ravel(),
        monitor_R=_rodrigues(mon["rvects"] if "rvects" in mon else mon["rvecs"]),
        monitor_t=(mon["tvecs"]).ravel(),
    )


@lru_cache(maxsize=None)
def load_annotations(subject: str) -> Annotations:
    rows = [line.split() for line in (ROOT / subject / f"{subject}.txt").read_text().splitlines() if line.strip()]
    files = np.array([r[0] for r in rows])
    num = np.array([[float(x) for x in r[1:27]] for r in rows])
    return Annotations(
        files=files,
        day=np.array([int(f.split("/")[0][3:]) for f in files]),
        target_px=num[:, 0:2],
        landmarks6=num[:, 2:14].reshape(-1, 6, 2),
        head_rvec=num[:, 14:17],
        head_tvec=num[:, 17:20],
        face_center=num[:, 20:23],
        gaze_target=num[:, 23:26],
        eye=np.array([r[27] for r in rows]),
    )


def image_path(subject: str, file: str) -> Path:
    return ROOT / subject / file


if __name__ == "__main__":
    for s in SUBJECTS:
        c, a = load_calibration(s), load_annotations(s)
        days = np.unique(a.day, return_counts=True)
        print(s, len(a), "screen", c.screen_px, c.screen_mm.round(1), "days", dict(zip(*days)))
