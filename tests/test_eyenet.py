import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from omarcheye import eyenet
from omarcheye.eyenet import (A_IN, A_LO, A_OUT, A_UP, B_IN, B_LO, B_OUT, B_UP, EYENET_PATH, SIZE, EyeNet,
                              eye_crop, omz_angles, sight_frame)


def mp_matrix(rot=np.eye(3), dist=50.0):
    m = np.eye(4)
    m[:3, :3] = rot
    m[2, 3] = -dist
    return m


def rot_y(a):  # MediaPipe axes: turns the face's forward axis (+z) towards +x, the image's right
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def rot_x(a):  # turns the forward axis towards -y (down) for a > 0
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_z(a):  # counter-clockwise as the camera sees the face, for a > 0
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def test_head_angles_follow_open_model_zoo():
    assert np.allclose(omz_angles(mp_matrix()), 0)
    yaw, pitch, roll = omz_angles(mp_matrix(rot_y(0.3)))  # face turned to the image's right
    assert yaw == pytest.approx(math.degrees(0.3)) and abs(pitch) < 1e-9 and abs(roll) < 1e-9
    _, pitch, _ = omz_angles(mp_matrix(rot_x(0.2)))  # face looking down
    assert pitch == pytest.approx(math.degrees(0.2))
    _, _, roll = omz_angles(mp_matrix(rot_z(-0.1)))  # head tilted clockwise in the image
    assert roll == pytest.approx(math.degrees(0.1))


def test_sight_frame():
    assert np.allclose(sight_frame(np.array([0.0, 0.0, 1.0])), np.diag([1.0, -1.0, 1.0]))
    ray = np.array([0.3, -0.2, 1.0])
    n = sight_frame(ray)
    assert np.allclose(n.T @ n, np.eye(3))
    assert np.allclose(n[:, 2], ray / np.linalg.norm(ray))
    assert n[0, 0] > 0.9 and n[1, 1] < -0.9  # x still right, y still up


def test_eye_crop_size_and_edges():
    frame = np.zeros((200, 300, 3), np.uint8)
    crop = eye_crop(frame, (100, 100), (140, 100), 0.0)
    assert crop.shape == (SIZE, SIZE, 3)
    assert eye_crop(frame, (2, 100), (42, 100), 0.0) is None  # runs off the left edge
    assert eye_crop(frame, (100, 100), (102, 100), 0.0) is None  # tiny


def test_eye_crop_levels_a_tilted_eye():
    # Head tilted clockwise by 15°: the corner line falls to the right. The
    # roll from the head pose (> 0 clockwise) must turn it level.
    a = math.radians(15)
    frame = np.full((300, 300, 3), 255, np.uint8)
    c, d = np.array([150.0, 150.0]), np.array([math.cos(a), math.sin(a)]) * 40
    cv2.line(frame, tuple(np.rint(c - 2 * d).astype(int)), tuple(np.rint(c + 2 * d).astype(int)), (0, 0, 0), 3)
    crop = eye_crop(frame, c - d, c + d, 15.0)
    rows = np.flatnonzero((crop[..., 0] < 128).sum(1) > SIZE // 2)
    assert len(rows) and abs(rows.mean() - SIZE / 2) < 3 and np.ptp(rows) < 6


needs_model = pytest.mark.skipif(not EYENET_PATH.exists(), reason=f"{EYENET_PATH} not downloaded")


def synthetic_face(iris=(0.0, 0.0), open_=True, w=1280, h=720, ew=60):
    """Two drawn eyes, 120 px apart, and MediaPipe-like points for their corners and lids."""
    img = np.full((h, w, 3), (120, 150, 200), np.uint8)
    pts = np.zeros((478, 2))
    gap = ew // 5 if open_ else 1
    for cx, (outer, inner, up, lo) in ((w / 2 - 60, (A_OUT, A_IN, A_UP, A_LO)), (w / 2 + 60, (B_OUT, B_IN, B_UP, B_LO))):
        centre = (int(cx), h // 2)
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, centre, (ew // 2, gap), 0, 0, 360, 255, -1)
        eye = np.full_like(img, 235)
        ix, iy = int(cx + iris[0] * ew), int(h / 2 + iris[1] * ew)
        cv2.circle(eye, (ix, iy), int(ew * 0.22), (60, 80, 110), -1, cv2.LINE_AA)
        cv2.circle(eye, (ix, iy), int(ew * 0.09), (10, 10, 10), -1, cv2.LINE_AA)
        img[mask > 0] = eye[mask > 0]
        cv2.ellipse(img, centre, (ew // 2, gap), 0, 0, 360, (40, 50, 70), 2, cv2.LINE_AA)
        ends = [(cx - ew / 2, h / 2), (cx + ew / 2, h / 2)]
        if outer == A_OUT:  # image-left eye: outer corner on the left
            pts[outer], pts[inner] = ends
        else:
            pts[inner], pts[outer] = ends
        pts[up], pts[lo] = (cx, h / 2 - gap), (cx, h / 2 + gap)
    return img, pts, mp_matrix()


@needs_model
def test_gaze_follows_the_iris():
    net = EyeNet()
    feats = {s: net(*synthetic_face(s)) for s in [(-0.15, 0), (0.15, 0), (0, -0.08), (0, 0.08)]}
    assert all(f is not None and np.isfinite(f).all() for f in feats.values())
    assert feats[(0.15, 0)][0] > 0.1 > -0.1 > feats[(-0.15, 0)][0]  # yaw > 0: to the image's right
    assert feats[(0, -0.08)][1] > feats[(0, 0.08)][1]  # iris up: pitch up
    assert feats[(0.15, 0)][2] > 0 > feats[(-0.15, 0)][2]  # the hit point moves the same way


@needs_model
def test_mirrored_frame_mirrors_the_gaze():
    net = EyeNet()
    img, pts, m = synthetic_face((0.1, 0.05))
    f = net(img, pts, m)
    mirrored = pts.copy()
    mirrored[:, 0] = img.shape[1] - 1 - pts[:, 0]
    s = np.diag([-1.0, 1.0, 1.0, 1.0])
    g = net(np.ascontiguousarray(img[:, ::-1]), mirrored, s @ m @ s)
    assert g[0] == pytest.approx(-f[0], abs=0.03) and g[1] == pytest.approx(f[1], abs=0.03)


@needs_model
def test_unusable_eyes():
    net = EyeNet()
    assert net(*synthetic_face(open_=False)) is None  # closed
    img, pts, m = synthetic_face()
    pts[A_OUT] = np.nan
    assert net(img, pts, m) is None
    assert net(*synthetic_face(ew=10)) is None  # too small to read


def test_missing_model_says_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match=eyenet.EYENET_URL):
        EyeNet(tmp_path / "nothing.xml")


def test_openvino_sends_no_telemetry():
    # `import openvino` would import its model converter, which reports each import to Google Analytics.
    code = ("import sys; from omarcheye.eyenet import _import_openvino; _import_openvino(); "
            "sys.exit('openvino_telemetry' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent.parent).returncode == 0
