import numpy as np

from omeye.model import GazeModel, fit, inliers


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
