import base64

import cv2
import numpy as np

from omarcheye.camview import caption, thumbnail
from omarcheye.tracker import Sample


def face_sample(pos=(0.5, 0.5), margin=0.2):
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(500, 780, 478), rng.uniform(250, 470, 478)])
    return Sample(0.0, np.zeros(7), 0.25, pos, margin, points)


def decode(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


def test_thumbnail_is_a_scaled_jpeg_with_tracking_drawn():
    frame = np.full((720, 1280, 3), 90, np.uint8)
    plain = decode(thumbnail(frame, None, width=480))
    drawn = decode(thumbnail(frame, face_sample(), width=480))
    assert plain.shape == drawn.shape == (270, 480, 3)
    assert np.abs(drawn.astype(int) - plain.astype(int)).sum() > 0


def test_thumbnail_survives_landmarks_outside_the_image():
    s = face_sample(pos=(0.5, 0.05), margin=-0.2)
    s.points[:, 1] -= 400  # face cut off at the top
    assert decode(thumbnail(np.zeros((720, 1280, 3), np.uint8), s)).shape[1] == 480


def test_caption():
    assert "black" in caption(None, 3, 30)
    assert "Well placed" in caption(face_sample(), 110, 30)
    assert "tilt the camera up" in caption(face_sample(pos=(0.5, 0.07)), 110, 30)
