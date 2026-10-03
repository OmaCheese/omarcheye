"""The geometric model against a simulated desk: a 700 x 390 mm screen, a
camera above it, a person calibrating in one posture and then sitting
differently."""

import math

import numpy as np

from omarcheye import geometry
from omarcheye.model import data_error, fit_samples
from omarcheye.tracker import RICH, head_pose

SCREEN = (700.0, 390.0)
TRUE = np.array([2.6, -2.2, 0.0, 0.04, -0.03, math.log(1.4), 0.12, 0.0, 15.0, 25.0])  # kx ky kl ox oy log_s tilt pan dx lift


def rot(yaw, pitch):
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return ry @ rx


def screen_point_cam(a, b, theta=TRUE):
    """Camera coordinates (mm) of the point a, b mm from the screen's top-left corner."""
    *_, tilt, pan, dx, lift = theta
    rc = geometry._rx(tilt) @ geometry._ry(pan)
    return rc @ np.array([a - SCREEN[0] / 2 - dx, -(b + lift), 0.0])


def session(rng, head_mm, n_dots=15, frames=20, noise=0.004, wobble=15.0):
    """Frames of someone at head_mm (camera coordinates, mm) looking at a 5x3
    grid of dots, with the head moving a little and iris noise."""
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, 5), np.linspace(0.07, 0.93, 3))
    dots = np.column_stack([xs.ravel(), ys.ravel()])[:n_dots]
    kx, ky, kl, ox, oy, log_s = TRUE[:6]
    rows = {k: [] for k in ("feats", "rich", "pose", "opens", "groups", "targets")}
    for g, (nx, ny) in enumerate(dots):
        target = screen_point_cam(nx * SCREEN[0], ny * SCREEN[1])
        for _ in range(frames):
            origin = np.asarray(head_mm) + rng.normal(0, wobble, 3)
            head_dir = -origin / np.linalg.norm(origin)  # face roughly towards the camera ...
            yaw = math.atan2(head_dir[0], head_dir[2]) + rng.normal(0, 0.05)  # ... give or take
            pitch = -math.atan2(head_dir[1], head_dir[2]) + rng.normal(0, 0.04)
            R = rot(yaw, pitch)
            gaze = (target - origin) / np.linalg.norm(target - origin)
            gh = R.T @ gaze  # in the head's frame
            beta, alpha = math.asin(gh[1]), math.atan2(gh[0], gh[2])
            u = (alpha - ox) / kx + rng.normal(0, noise)
            v = (beta - oy) / ky + rng.normal(0, noise)
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = R, origin / math.exp(log_s) / 10  # MediaPipe's cm, off by the lens scale
            rich = np.zeros(len(RICH))
            rich[[0, 2]], rich[[1, 3]] = u, v
            rich[-5:] = head_pose(m)
            rows["feats"].append(np.array([u, v, *head_pose(m)]))
            rows["rich"].append(rich)
            rows["pose"].append(m.ravel())
            rows["opens"].append(0.25)
            rows["groups"].append(g)
            rows["targets"].append((nx, ny))
    return {k: np.array(v) for k, v in rows.items()}


def test_predicts_the_true_setup():
    data = session(np.random.default_rng(0), (0.0, -120.0, -650.0), noise=0.0, wobble=20.0)
    x = geometry.inputs(data)
    mm = geometry.predict_mm(TRUE, x, SCREEN, False)
    assert np.nanmax(np.abs(mm - data["targets"] * SCREEN)) < 1e-6


def test_fit_recovers_the_screen_points():
    data = session(np.random.default_rng(1), (0.0, -120.0, -650.0))
    model = geometry.fit(data, data["groups"], SCREEN, "TEST-1", 0.1)
    assert model.error < 0.03  # under 3% of the width, leaving each dot out


def test_geometry_holds_when_you_sit_differently():
    rng = np.random.default_rng(2)
    calib = session(rng, (0.0, -120.0, -650.0))
    later = session(rng, (120.0, -60.0, -560.0))  # 12 cm to the right, 6 cm higher, 9 cm closer
    aspect = SCREEN[1] / SCREEN[0]
    geo = geometry.fit(calib, calib["groups"], SCREEN, "TEST-1", 0.1)
    reg, *_ = fit_samples({k: v for k, v in calib.items() if k != "pose"}, aspect, "TEST-1", "cam")
    geo_later, reg_later = data_error(geo, later, aspect), data_error(reg, later, aspect)
    assert geo_later < 0.05
    assert geo_later < reg_later / 2


def test_fit_samples_picks_geometry_when_poses_are_known(tmp_path):
    data = session(np.random.default_rng(3), (0.0, -120.0, -650.0))
    model, used, frames, errors = fit_samples(data, SCREEN[1] / SCREEN[0], "TEST-1", "cam", screen_mm=SCREEN)
    assert set(errors) == {"basic", "rich", "geometric"}
    model.save(tmp_path / "cal.json")
    from omarcheye.model import load_model

    again = load_model(tmp_path / "cal.json")
    assert again.kind == model.kind
    assert np.allclose(again.predict_data(data)[:5], model.predict_data(data)[:5])
