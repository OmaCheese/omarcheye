import types

import numpy as np

from omeye import refine, samples
from omeye.hypr import Monitor
from omeye.model import MOUSE, fit_samples
from omeye.tracker import Sample


def test_pointer_still():
    rest = [(t / 30, 0.5, 0.5) for t in range(20)]
    assert refine.pointer_still(rest, 19 / 30, aspect=9 / 16)
    moving = [(t / 30, 0.5 + t * 0.01, 0.5) for t in range(20)]
    assert not refine.pointer_still(moving, 19 / 30, aspect=9 / 16)
    assert not refine.pointer_still(rest[:5], 4 / 30, aspect=9 / 16)  # not rested long enough yet


def test_cells():
    assert refine.cell(0, 0) == 0
    assert refine.cell(0.999, 0.999) == refine.COLS * refine.ROWS - 1
    assert refine.cell(1.0, 1.0) == refine.COLS * refine.ROWS - 1


class Clock:
    t = 0.0

    def monotonic(self):
        return self.t


def path(t):
    """The pointer rests 1 s on each cell centre in turn, 0.3 s moves between."""
    centres = [((c + 0.5) / refine.COLS, (r + 0.5) / refine.ROWS) for r in range(refine.ROWS) for c in range(refine.COLS)]
    i, k = divmod(t, 1.3)
    a = centres[int(i) % len(centres)]
    b = centres[(int(i) + 1) % len(centres)]
    w = max(0.0, (k - 1.0) / 0.3)
    return a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w


def eyes(x, y, rng):
    """Features of someone looking at (x, y) with a steady head."""
    u = 0.25 * (x - 0.5) + rng.normal(0, 0.004)
    v = 0.15 * (y - 0.5) + rng.normal(0, 0.004)
    return np.array([u, v, 0.0, -0.3, 0.05, 0.5, 60.0]) + rng.normal(0, [0, 0, 0.01, 0.01, 0.003, 0.003, 0.3])


def test_collect_fills_cells_and_fits(monkeypatch):
    clock = Clock()
    rng = np.random.default_rng(3)
    monkeypatch.setattr(refine, "time", types.SimpleNamespace(monotonic=clock.monotonic))
    mon = Monitor("TEST-1", 0, 0, 3072, 1728, True)

    class Cam:
        def read(self):
            clock.t += 1 / 30
            return np.zeros((2, 2, 3), np.uint8)

    class Tracker:
        def process(self, frame, t):
            return Sample(t, eyes(*path(t), rng), 0.25)

    class Hypr:
        def cursor(self):
            return mon.to_global(*path(clock.t))

    class Overlay:
        def poll(self):
            return None

        def send(self, **cmd):
            pass

    data = refine._collect(Overlay(), Cam(), Tracker(), Hypr(), mon, 60, 9 / 16)
    cells = set(np.unique(data["groups"]) - MOUSE)
    assert cells == set(range(refine.COLS * refine.ROWS))
    assert clock.t < 40  # stopped once every cell was covered, not at the time limit

    model, used, frames = fit_samples(data["feats"], data["opens"], data["groups"], data["targets"],
                                      9 / 16, "TEST-1", "cam", score={MOUSE + c for c in cells})
    assert model.error < 0.05
    assert len(used) == len(cells)


def test_samples_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(samples, "SAMPLES_PATH", tmp_path / "s.npz")
    a = {"feats": np.ones((3, 7)), "opens": np.ones(3), "groups": np.arange(3), "targets": np.zeros((3, 2))}
    samples.save("cam", "MON", a)
    back = samples.load("cam", "MON")
    assert back is not None and np.array_equal(back["groups"], a["groups"])
    assert samples.load("other cam", "MON") is None
    merged = samples.merge(back, a)
    assert len(merged["groups"]) == 6 and merged["feats"].shape == (6, 7)
