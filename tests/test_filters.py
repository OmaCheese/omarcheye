import numpy as np

from omeye.filters import FixationFilter


def test_holds_still_through_jitter():
    rng = np.random.default_rng(1)
    f = FixationFilter(radius=0.06)
    out = np.array([f(0.3 + rng.normal(0, 0.02), 0.2 + rng.normal(0, 0.02)) for _ in range(60)])
    raw_sd = 0.02
    assert out[15:].std(0).max() < raw_sd / 3  # far steadier than the input
    assert np.allclose(out[-1], [0.3, 0.2], atol=0.02)


def test_jumps_after_confirm_frames():
    f = FixationFilter(radius=0.06, confirm=3)
    for _ in range(10):
        f(0.2, 0.2)
    assert f(0.8, 0.3) == (0.2, 0.2)  # one far point: wait
    assert f(0.8, 0.3) == (0.2, 0.2)
    assert f(0.8, 0.3) == (0.8, 0.3)  # third in a row: jump


def test_ignores_lone_strays():
    f = FixationFilter(radius=0.06, confirm=3)
    for i in range(30):
        x, y = (0.9, 0.5) if i % 4 == 0 else (0.2, 0.2)  # every 4th frame a stray
        out = f(x, y)
    assert out == (0.2, 0.2)


def test_strays_in_different_directions_are_not_a_jump():
    f = FixationFilter(radius=0.06, confirm=3)
    for _ in range(5):
        f(0.5, 0.3)
    f(0.1, 0.3)
    f(0.9, 0.3)
    assert f(0.5, 0.0) == (0.5, 0.3)
