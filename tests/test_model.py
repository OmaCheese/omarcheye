import numpy as np

from omarcheye.model import GazeModel, fit, fit_samples, inliers
from omarcheye.tracker import RICH


def synthetic(n_dots=15, frames=25, noise=0.002, seed=1):
    """Frames whose features follow a known gaze map, with head movement."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, 5), np.linspace(0.07, 0.93, 3))
    dots = np.column_stack([xs.ravel(), ys.ravel()])[:n_dots]
    f, t, g = [], [], []
    for i, (x, y) in enumerate(dots):
        yaw = rng.normal(0, 0.05, frames)
        pitch = rng.normal(0, 0.03, frames)
        hx = rng.normal(0, 0.02, frames)
        hy = rng.normal(0, 0.02, frames)
        hz = rng.normal(65, 2, frames)
        # Eyes make up whatever the head doesn't.
        u = 0.25 * (x - 0.5) - 0.5 * yaw - 0.3 * hx + rng.normal(0, noise, frames)
        v = 0.15 * (y - 0.5) - 0.4 * pitch - 0.2 * hy + rng.normal(0, noise, frames)
        f.append(np.column_stack([u, v, yaw, pitch, hx, hy, hz]))
        t.append(np.tile([x, y], (frames, 1)))
        g.append(np.full(frames, i))
    return np.vstack(f), np.vstack(t), np.concatenate(g)


def test_fit_recovers_map(tmp_path):
    f, t, g = synthetic()
    model = fit(f, t, g, aspect=9 / 16, monitor="TEST-1", blink=0.1)
    assert model.error < 0.02  # under 2% of the screen width
    x, y = model.predict(f[0])
    assert abs(x - t[0, 0]) < 0.05 and abs(y - t[0, 1]) < 0.05

    path = tmp_path / "cal.json"
    model.save(path)
    again = GazeModel.load(path)
    assert again.predict(f[5]) == model.predict(f[5])
    assert again.monitor == "TEST-1"


def test_noise_shows_in_error():
    f, t, g = synthetic(noise=0.02)
    assert fit(f, t, g, 9 / 16, "TEST-1", 0.1).error > 0.03


def test_inliers_drop_saccade_frames():
    f, t, g = synthetic(noise=0.001)
    f[3, 0] += 0.2  # one frame caught mid-saccade
    keep = inliers(f, g)
    assert not keep[3] and keep.mean() > 0.95


def test_rich_features_win_when_they_carry_more():
    """Vertical gaze hidden in the lids: the rich set sees it, the basic one barely."""
    f, t, g = synthetic(noise=0.002)
    rng = np.random.default_rng(5)
    f[:, 1] = rng.normal(0, 0.01, len(f))  # basic v: no vertical signal left
    rich = np.zeros((len(f), len(RICH)))
    rich[:, 0] = rich[:, 2] = f[:, 0]
    rich[:, 4] = 0.2 * (t[:, 1] - 0.5) + rng.normal(0, 0.002, len(f))  # upper lid follows gaze down
    rich[:, -5:] = f[:, 2:]
    data = {"feats": f, "rich": rich, "opens": np.full(len(f), 0.25), "groups": g, "targets": t}
    model, used, frames, errors = fit_samples(data, 9 / 16, "TEST-1", "cam")
    assert model.kind == "rich" and errors["rich"] < errors["basic"]


def test_rich_model_round_trip(tmp_path):
    f, t, g = synthetic()
    rich = np.column_stack([f[:, 0], f[:, 1], f[:, 0], f[:, 1], np.zeros((len(f), 12)), f[:, 2:]])
    model = fit(rich, t, g, 9 / 16, "TEST-1", 0.1, kind="rich")
    model.save(tmp_path / "cal.json")
    again = GazeModel.load(tmp_path / "cal.json")
    assert again.kind == "rich" and again.predict(rich[3]) == model.predict(rich[3])


def test_a_head_held_still_during_calibration_does_not_throw_predictions_off():
    """Calibrated with the distance nearly constant; later 5 cm closer."""
    f, t, g = synthetic()
    rng = np.random.default_rng(7)
    f[:, 6] = 47.5 + rng.normal(0, 0.25, len(f))  # hz barely moves
    f[:, 0] += 0.002 * (f[:, 6] - 47.5)  # and happens to correlate a little with u
    model = fit(f, t, g, 9 / 16, "TEST-1", 0.1)
    later = f[:50].copy()
    later[:, 6] -= 5.0
    p = np.array([model.predict(x) for x in later])
    assert np.all(np.abs(p - t[:50]) < 0.25)  # still on the screen, near the dots


def test_choose_prefers_geometry_within_the_margin():
    from types import SimpleNamespace

    from omarcheye.model import choose

    basic, geo = SimpleNamespace(kind="basic", error=0.077), SimpleNamespace(kind="geometric", error=0.083)
    assert choose([basic, geo]) is geo
    geo.error = 0.12
    assert choose([basic, geo]) is basic


def test_eye_network_model_is_fitted_and_wins_when_it_sees_more():
    """The network's hit point carries the gaze cleanly; the landmarks are noisy.
    Frames the network couldn't read (NaN) are left out of its fit only."""
    from omarcheye.tracker import Sample

    f, t, g = synthetic(noise=0.03, seed=3)
    rng = np.random.default_rng(3)
    rich = np.column_stack([f[:, 0], f[:, 1], f[:, 0], f[:, 1], np.zeros((len(f), 12)), f[:, 2:]])
    net = np.column_stack([np.zeros((len(f), 2)), 30 * (t - 0.5) + rng.normal(0, 0.05, t.shape)])
    net[::7] = np.nan
    data = {"feats": f, "rich": rich, "net": net, "opens": np.full(len(f), 0.25), "groups": g, "targets": t}
    model, _, _, errors = fit_samples(data, 9 / 16, "TEST-1", "cam")
    assert model.kind == "eyenet" and errors["eyenet"] < errors["rich"] / 2
    x, y = model.predict_sample(Sample(0.0, f[1], 0.25, rich=rich[1], net=net[1]))
    assert abs(x - t[1, 0]) < 0.05 and abs(y - t[1, 1]) < 0.05
    assert all(map(np.isnan, model.predict_sample(Sample(0.0, f[1], 0.25, rich=rich[1], net=None))))
