import numpy as np

from omarcheye.drift import MAX_SHIFT, Drift

ASPECT = 9 / 16
MARGIN = 0.066  # about the calibration's error, monitor widths
COLS = [(c / 4 + 0.003, r / 2 + 0.005, (c + 1) / 4 - 0.003, (r + 1) / 2 - 0.005) for r in range(2) for c in range(4)]


def inside(rect, x, y):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def test_one_correction_moves_the_estimate_well_inside_the_window():
    d = Drift(ASPECT, MARGIN)
    # Looking at the second column; the estimate sat in the first one.
    d.add(0.2, 0.25, COLS[1])
    x, y = d.apply(0.2, 0.25)
    assert inside(COLS[1], x, y) and x > COLS[1][0] + 0.03
    assert abs(y - 0.25) < 1e-6  # nothing said about up-down: no shift that way


def test_corrections_from_many_windows_find_a_steady_shift():
    # The estimate sits 16% of the width left of and 9% (of the width) above
    # where you look, as in the later sitting; you correct omarcheye in a few windows.
    rng = np.random.default_rng(0)
    true = np.array([-0.16, -0.09 / ASPECT])
    d = Drift(ASPECT, MARGIN)
    for _ in range(12):
        rect = COLS[rng.integers(len(COLS))]
        look = np.array([rng.uniform(rect[0], rect[2]), rng.uniform(rect[1], rect[3])])
        raw = look + true + rng.normal(0, 0.02, 2)
        x, y = d.apply(*raw)
        if not inside(rect, x, y):  # omarcheye would pick wrong: you correct it
            d.add(*raw, rect)
    hits = 0
    for _ in range(200):
        rect = COLS[rng.integers(len(COLS))]
        cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        hits += inside(rect, *d.apply(*(np.array([cx, cy]) + true)))
    assert hits > 190


def test_a_correction_needing_a_huge_shift_is_not_learned():
    d = Drift(ASPECT, MARGIN)
    assert d.add(0.05, 0.25, COLS[3]) is None
    assert d.size == 0 and not d.records
    assert d.add(0.2, 0.25, COLS[1]) is not None and d.size <= MAX_SHIFT


def test_retarget_and_forget():
    d = Drift(ASPECT, MARGIN)
    r = d.add(0.15, 0.25, COLS[1])
    assert d.dx > 0
    assert d.retarget(r, COLS[0])  # it was the first window after all
    assert abs(d.dx) < 1e-4
    d.forget(r)
    assert d.size == 0


def test_saved_shift_belongs_to_its_calibration(tmp_path):
    path = tmp_path / "drift.json"
    d = Drift(ASPECT, MARGIN, "cal-1", path)
    d.add(0.2, 0.25, COLS[1])
    d.save()
    again = Drift(ASPECT, MARGIN, "cal-1", path)
    assert abs(again.dx - d.dx) < 1e-6 and len(again.records) == 1
    assert Drift(ASPECT, MARGIN, "cal-2", path).size == 0  # recalibrated: start afresh
