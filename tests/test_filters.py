import numpy as np

from omeye.filters import OneEuro2D


def test_one_euro_calms_jitter_and_follows_jumps():
    rng = np.random.default_rng(1)
    f = OneEuro2D(1.0, 0.5)
    t = np.arange(90) / 30
    out = np.array([f(ti, 0.3 + rng.normal(0, 0.02), 0.2 + rng.normal(0, 0.02)) for ti in t[:60]])
    assert np.abs(np.diff(out[20:], axis=0)).mean() < 0.02 / 3  # far steadier than the input
    jumped = [f(ti, 0.8, 0.2) for ti in t[60:]]
    assert abs(jumped[-1][0] - 0.8) < 0.02  # caught up within a second
