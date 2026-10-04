import numpy as np

from omarcheye.appearance import DESC, AppearanceModel, fit
from omarcheye.model import fit_samples
from omarcheye.tracker import RICH, Sample


def synthetic(n_dots=15, frames=20, seed=0):
    """The eye descriptor carries the gaze (a fixed random projection of the
    target plus noise); the landmark features are mostly noise."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, 5), np.linspace(0.07, 0.93, 3))
    dots = np.column_stack([xs.ravel(), ys.ravel()])[:n_dots]
    proj = rng.normal(0, 1, (2, DESC))
    t = np.repeat(dots, frames, 0)
    g = np.repeat(np.arange(n_dots), frames)
    desc = (t - 0.5) @ proj + rng.normal(0, 0.3, (len(t), DESC))
    rich = rng.normal(0, 0.05, (len(t), len(RICH)))
    rich[:, -1] += 60  # head distance, cm
    net = np.full((len(t), 4), np.nan)
    return {"feats": rich[:, [0, 1, 16, 17, 18, 19, 20]], "rich": rich, "net": net,
            "desc": desc.astype(np.float16), "opens": np.full(len(t), 0.25), "groups": g, "targets": t}


def test_appearance_model_learns_from_the_descriptor_and_wins(tmp_path):
    data = synthetic()
    model, _, _, errors = fit_samples(data, 9 / 16, "TEST-1", "cam")
    assert model.kind == "appearance" and errors["appearance"] < 0.03 < errors["rich"]
    assert model.wh == 0  # no network features in these frames: not used, not needed live
    s = Sample(0.0, data["feats"][3], 0.25, rich=data["rich"][3], desc=data["desc"][3].astype(np.float32))
    x, y = model.predict_sample(s)
    assert abs(x - data["targets"][3, 0]) < 0.05 and abs(y - data["targets"][3, 1]) < 0.05

    path = tmp_path / "cal.json"
    model.save(path)
    again = AppearanceModel.load(path)
    assert np.allclose(again.predict_sample(s), (x, y))
    assert np.isnan(again.predict_sample(Sample(0.0, data["feats"][3], 0.25, rich=data["rich"][3]))[0])


def test_held_out_dots_not_frames_set_the_error():
    # Frames of one dot share a steady offset: leaving single frames out would
    # call this fit perfect; leaving the dot out sees the offset.
    data = synthetic(seed=1)
    rng = np.random.default_rng(1)
    offsets = rng.normal(0, 0.03, (15, 2))
    data["targets"] = data["targets"] + offsets[data["groups"]]
    model = fit(data, data["groups"], 9 / 16, "TEST-1", 0.1)
    assert model.error > 0.02
