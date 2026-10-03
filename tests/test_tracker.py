import math

import numpy as np

from omeye.tracker import BUILT_IN, eye_features, head_pose


def eye(center=(100.0, 50.0), width=40.0, iris=(0.0, 0.0), roll=0.0, gap=10.0):
    """Landmarks 0..4: left corner, right corner, upper lid, lower lid, iris."""
    c, s = math.cos(roll), math.sin(roll)
    rot = np.array([[c, -s], [s, c]])
    local = np.array([(-width / 2, 0), (width / 2, 0), (0, -gap / 2), (0, gap / 2), (iris[0] * width, iris[1] * width)])
    return local @ rot.T + np.array(center)


def test_centred_iris():
    u, v, openness = eye_features(eye(), 0, 1, 2, 3, 4)
    assert abs(u) < 1e-9 and abs(v) < 1e-9 and abs(openness - 0.25) < 1e-9


def test_iris_offsets_survive_head_roll():
    for roll in (0.0, 0.3, -0.5):
        u, v, _ = eye_features(eye(iris=(0.1, 0.05), roll=roll), 0, 1, 2, 3, 4)
        assert abs(u - 0.1) < 1e-9 and abs(v - 0.05) < 1e-9


def test_head_pose_straight_ahead():
    m = np.eye(4)
    m[:3, 3] = (0, 0, -60)
    yaw, pitch, hx, hy, dist = head_pose(m)
    assert abs(yaw) < 1e-9 and abs(pitch) < 1e-9 and dist == 60 and hx == 0 and hy == 0


def test_head_pose_turned():
    a = 0.3
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    m[:3, 3] = (6, 0, -60)
    yaw, _, hx, _, _ = head_pose(m)
    assert abs(yaw - a) < 1e-9 and abs(hx - 0.1) < 1e-9


def test_built_in_camera_names():
    assert BUILT_IN.search("HP Wide Vision HD Camera: HP Wi")
    assert BUILT_IN.search("Integrated Camera: Integrated C")
    assert not BUILT_IN.search("Logitech BRIO")


def cams(phone_live: bool):
    from omeye.tracker import CameraInfo

    return [
        CameraInfo("/dev/video0", "HP Wide Vision HD Camera: HP Wi", True, False, True),
        CameraInfo("/dev/video2", "Flux Camera", phone_live, True, False),
    ]


def test_auto_prefers_a_live_phone_camera():
    from omeye.tracker import pick_camera

    assert pick_camera("auto", quiet=True, cams=cams(True)).device == "/dev/video2"


def test_auto_skips_an_idle_virtual_camera():
    from omeye.tracker import pick_camera

    assert pick_camera("auto", quiet=True, cams=cams(False)).device == "/dev/video0"


def test_named_idle_camera_explains_itself():
    import pytest

    from omeye.tracker import pick_camera

    with pytest.raises(RuntimeError, match="nothing is feeding it.*Flux"):
        pick_camera("Flux Camera", cams=cams(False))
    with pytest.raises(RuntimeError, match="no camera named"):
        pick_camera("Logitech", cams=cams(True))


def test_framing_advice():
    from omeye.tracker import framing_advice

    assert "tilt the camera up" in framing_advice((0.57, 0.07), -0.1)
    assert "tilt the camera down" in framing_advice((0.5, 0.8), 0.05)
    assert "side" in framing_advice((0.1, 0.5), 0.05)
    assert "outside" in framing_advice((0.5, 0.5), 0.0)
    assert framing_advice((0.5, 0.45), 0.1) == ""
