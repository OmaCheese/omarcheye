import math

import cv2
import numpy as np
import pytest

from omarcheye import eyepatch as ep
from omarcheye.tracker import features


def ring_offsets(n: int, width: float, gap: float) -> np.ndarray:
    """Eyelid margin of an almond-shaped eye, in the order of Eye.ring:
    left corner, along the lower lid, right corner, back along the upper lid."""
    half = n // 2
    t = np.linspace(0, math.pi, half + 1)
    lower = np.column_stack([-np.cos(t) * width / 2, np.sin(t) * gap / 2])  # left -> right
    upper = np.column_stack([np.cos(t[1:-1]) * width / 2, -np.sin(t[1:-1]) * gap / 2])  # right -> left
    return np.vstack([lower, upper])


def synthetic(iris=(0.05, 0.02), roll=0.0, width=60.0, size=(640, 400), seed_error=(3.0, -2.0)):
    """A frame with two drawn eyes and MediaPipe-like landmarks.
    iris: true iris offset (u, v) in eye widths; seed_error: px added to
    MediaPipe's iris centre. Returns (frame, points, true iris centres)."""
    c, s = math.cos(roll), math.sin(roll)
    rot = np.array([[c, -s], [s, c]])
    img = np.full((size[1], size[0]), 140, np.uint8)
    points = np.zeros((478, 2))
    truth = []
    gap, r = 0.42 * width, 0.21 * width
    for eye, centre in ((ep.EYE_A, (230.0, 200.0)), (ep.EYE_B, (410.0, 200.0))):
        centre = np.array(centre)

        def place(local):
            return np.asarray(local) @ rot.T + centre

        ring = place(ring_offsets(len(eye.ring), width, gap))
        for k, idx in enumerate(eye.ring):
            points[idx] = ring[k]
        points[eye.left], points[eye.right] = place([-width / 2, 0]), place([width / 2, 0])
        true = place([iris[0] * width, iris[1] * width])
        truth.append(true)
        # draw: sclera inside the lids, dark iris, darker pupil, all clipped by the lids
        eye_mask = np.zeros_like(img)
        cv2.fillPoly(eye_mask, [np.round(ring * 16).astype(np.int32)], 1, cv2.LINE_AA, 4)
        layer = np.full_like(img, 225)
        cv2.circle(layer, tuple(np.round(true * 16).astype(int)), int(r * 16), 70, -1, cv2.LINE_AA, 4)
        cv2.circle(layer, tuple(np.round(true * 16).astype(int)), int(0.4 * r * 16), 25, -1, cv2.LINE_AA, 4)
        img = np.where(eye_mask > 0, layer, img)
        seed = true + np.array(seed_error)
        points[eye.iris] = seed
        for k, ang in enumerate((0, -math.pi / 2, math.pi, math.pi / 2)):
            points[eye.edge[k]] = seed + r * np.array([math.cos(ang), math.sin(ang)]) @ rot.T
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), points, np.array(truth)


def test_transform_puts_the_corner_line_level_and_centred():
    _, pts, _ = synthetic(roll=0.4)
    m = ep.eye_transform(pts, ep.EYE_A, 37.5, (60, 36))
    a = m[:, :2] @ pts[ep.EYE_A.left] + m[:, 2]
    b = m[:, :2] @ pts[ep.EYE_A.right] + m[:, 2]
    assert np.allclose(a, (30 - 37.5 / 2, 18), atol=1e-6)
    assert np.allclose(b, (30 + 37.5 / 2, 18), atol=1e-6)


def test_patches_shape_and_contrast():
    frame, pts, _ = synthetic()
    p = ep.eye_patches(frame, pts)
    assert p.shape == (2, 36, 60) and p.dtype == np.uint8
    assert p.min() < 20 and p.max() > 235  # equalised
    big = ep.eye_patches(frame, pts, (100, 60))
    assert big.shape == (2, 60, 100)


def test_patch_is_the_same_under_head_roll():
    f0, p0, _ = synthetic(roll=0.0)
    f1, p1, _ = synthetic(roll=0.35)
    a = ep.eye_patch(f0, p0, ep.EYE_A, how="none").astype(float)
    b = ep.eye_patch(f1, p1, ep.EYE_A, how="none").astype(float)
    assert np.abs(a - b).mean() < 4.0


def test_patch_scales_with_the_eye():
    f0, p0, _ = synthetic(width=60.0)
    f1, p1, _ = synthetic(width=120.0, size=(1000, 500))
    a = ep.eye_patch(f0, p0, ep.EYE_A, how="none").astype(float)
    b = ep.eye_patch(f1, p1, ep.EYE_A, how="none").astype(float)
    assert np.abs(a - b).mean() < 6.0


def test_downscaling_is_antialiased():
    stripes = np.zeros((400, 400), np.uint8)
    stripes[:, ::2] = 255  # 1-pixel stripes: averaging gives mid-grey, point sampling aliases
    m = np.array([[0.15, 0, 0], [0, 0.15, 0]])
    out = ep.warp(stripes, m, (60, 36))
    assert abs(out.mean() - 127.5) < 10 and out.std() < 10


@pytest.mark.parametrize("method", ["timm", "disk", "limbus", "mean"])
def test_refinement_moves_towards_the_true_iris(method):
    frame, pts, truth = synthetic(iris=(0.08, 0.03), roll=0.2, seed_error=(3.5, -2.5))
    for e, eye in enumerate(ep.EYES):
        p = ep.refine_iris(frame, pts, eye, method)
        assert p is not None
        before = np.hypot(*(pts[eye.iris] - truth[e]))
        after = np.hypot(*(p - truth[e]))
        assert after < 0.5 * before, (method, before, after)


def test_refined_points_give_u_and_v_through_the_tracker():
    frame, pts, truth = synthetic(iris=(0.1, 0.0), roll=0.3, seed_error=(4.0, 0.0))
    refined, n = ep.refine_points(frame, pts, "disk")
    assert n == 2
    changed = np.flatnonzero(np.any(refined != pts, axis=1))
    assert set(changed) <= {ep.EYE_A.iris, ep.EYE_B.iris}
    m = np.eye(4)
    m[2, 3] = -50
    u_seed = features(pts, m, 0.0).feat[0]
    u_ref = features(refined, m, 0.0).feat[0]
    assert abs(u_ref - 0.1) < abs(u_seed - 0.1)
    assert abs(u_ref - 0.1) < 0.02


def test_refinement_fails_gracefully_on_a_blank_frame():
    _, pts, _ = synthetic()
    blank = np.full((400, 640, 3), 128, np.uint8)
    refined, n = ep.refine_points(blank, pts, "limbus")
    assert n == 0 and np.array_equal(refined, pts)
    for method in ("timm", "disk"):
        refined, _ = ep.refine_points(blank, pts, method)  # finds something or nothing, but stays near the seed
        r = 0.21 * 60
        assert np.all(np.hypot(*(refined - pts).T) <= ep.MAX_MOVE * r * 1.5 + 1e-9)


def test_hog_responds_to_where_the_iris_is():
    f0, p0, _ = synthetic(iris=(-0.1, 0.0))
    f1, p1, _ = synthetic(iris=(0.1, 0.0))
    d0, d1 = ep.eye_descriptor(f0, p0), ep.eye_descriptor(f1, p1)
    assert d0.shape == (2 * 1620,) and np.isfinite(d0).all()
    same = ep.eye_descriptor(*synthetic(iris=(-0.1, 0.0))[:2])
    assert np.allclose(d0, same)
    assert np.linalg.norm(d0 - d1) > 0.1 * np.linalg.norm(d0)


def test_hog_of_a_flat_patch_is_zero():
    assert not ep.hog(np.full((36, 60), 90, np.uint8)).any()
