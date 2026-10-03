"""Calibrated map from gaze features to a point on the monitor.

Ridge regression on standardised features plus the quadratic eye terms
(u², v², uv). The ridge strength is picked by leave-one-dot-out
cross-validation, and that same error is the accuracy calibration reports.
Two feature sets are fitted ("basic" and "rich", see tracker.py) and the one
with the lower cross-validated error is kept. Outputs are monitor fractions
(0..1 across, 0..1 down), so a calibration stays valid when the monitor's
scale or resolution changes.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracker import FEATURES, RICH, Sample

KINDS = {"basic": FEATURES, "rich": RICH}

# Smallest spread a head feature is scaled by (yaw, pitch in radians; hx, hy
# per unit distance; hz in MediaPipe's cm). Calibration often holds the head
# almost still; scaling by that tiny spread would let a few centimetres of
# leaning later count as a huge change and throw predictions off the screen.
HEAD_FLOOR = np.array([0.035, 0.035, 0.02, 0.02, 2.0])
CLAMP = 2.0  # live features are clamped to the calibration's range, widened by this many spreads
GEOMETRIC_MARGIN = 1.3  # prefer the geometric model unless its error is 30% worse than the best regression's


def floors(kind: str) -> np.ndarray:
    """Minimum spread per raw feature of a kind (0: no minimum)."""
    f = np.zeros(len(KINDS[kind]))
    f[-len(HEAD_FLOOR):] = HEAD_FLOOR
    return f

LAMBDAS = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


def expand(f: np.ndarray, kind: str = "basic") -> np.ndarray:
    f = np.atleast_2d(f)
    if kind == "rich":
        u, v = (f[:, 0] + f[:, 2]) / 2, (f[:, 1] + f[:, 3]) / 2
    else:
        u, v = f[:, 0], f[:, 1]
    return np.column_stack([f, u * u, v * v, u * v])


def vectors(data: dict, kind: str) -> np.ndarray:
    """The feature matrix of a sample set for one kind of model."""
    return data["rich"] if kind == "rich" else data["feats"]


def _ridge(z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, m = z.shape
    return np.linalg.solve(z.T @ z + lam * n * np.eye(m), z.T @ y)


def _solve(z: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    zm, ym = z.mean(0), y.mean(0)
    coef = _ridge(z - zm, y - ym, lam)
    return coef, ym - zm @ coef


def inliers(f: np.ndarray, groups: np.ndarray, k: float = 3.5) -> np.ndarray:
    """Per dot, drop frames whose eye features stray more than k robust
    deviations from that dot's median (saccades, half blinks)."""
    keep = np.ones(len(f), bool)
    for g in np.unique(groups):
        idx = groups == g
        eye = f[idx, :2]
        dev = np.abs(eye - np.median(eye, 0))
        mad = np.maximum(np.median(dev, 0), 1e-3)
        keep[idx] = (dev <= k * 1.4826 * mad).all(1)
    return keep


@dataclass
class GazeModel:
    monitor: str
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray  # (terms, 2)
    intercept: np.ndarray  # (2,)
    lam: float
    blink: float  # openness below this = eyes closed, frame ignored
    error: float  # cross-validated mean error (leave one dot or cell out), in monitor widths
    camera: str = ""
    created: str = ""
    kind: str = "basic"  # which feature set: see KINDS
    lo: np.ndarray | None = None  # live features are clamped to [lo, hi]
    hi: np.ndarray | None = None

    def _z(self, f: np.ndarray) -> np.ndarray:
        f = np.atleast_2d(f)
        if self.lo is not None:
            f = np.clip(f, self.lo, self.hi)
        return (expand(f, self.kind) - self.mean) / self.std

    def predict(self, feat: np.ndarray) -> tuple[float, float]:
        x, y = self._z(feat)[0] @ self.coef + self.intercept
        return float(x), float(y)

    def predict_sample(self, s: Sample) -> tuple[float, float]:
        return self.predict(s.rich if self.kind == "rich" else s.feat)

    def predict_data(self, data: dict) -> np.ndarray:
        """(n, 2) predictions in monitor fractions for a sample set."""
        return self._z(vectors(data, self.kind)) @ self.coef + self.intercept

    def describe(self) -> str:
        return f"{self.kind} features"

    def group_error(self, f: np.ndarray, targets: np.ndarray, groups: np.ndarray, aspect: float) -> float:
        """Mean distance from prediction to target, averaged per group, in monitor widths."""
        d = self._z(f) @ self.coef + self.intercept - targets
        e = np.hypot(d[:, 0], d[:, 1] * aspect)
        return float(np.mean([e[groups == g].mean() for g in np.unique(groups)]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        data["features"] = list(KINDS[self.kind])
        path.write_text(json.dumps(data, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> "GazeModel":
        data = json.loads(path.read_text())
        if data.pop("features", None) != list(KINDS.get(data.get("kind", "basic"), ())):
            raise ValueError(f"{path} was made by another omarcheye version; run `omarcheye calibrate` again")
        for k in ("mean", "std", "coef", "intercept", "lo", "hi"):
            if data.get(k) is not None:
                data[k] = np.array(data[k])
        return cls(**data)


MIN_FRAMES = 8  # a dot or cell with fewer usable frames is left out
MOUSE = 1000  # group ids from here up are pointer cells (omarcheye refine); below are dots


def fit(f: np.ndarray, targets: np.ndarray, groups: np.ndarray, aspect: float,
        monitor: str, blink: float, camera: str = "", score: set[int] | None = None,
        kind: str = "basic") -> GazeModel:
    """f: (n, features) frames; targets: (n, 2) monitor fractions of where
    each frame was looking; groups: dot or cell per frame; aspect: height/width.
    The reported error is the cross-validated error over the `score` groups
    (default: all of them)."""
    x = expand(f, kind)
    mean, std = x.mean(0), x.std(0)
    raw_floor = floors(kind)
    std[: len(raw_floor)] = np.maximum(std[: len(raw_floor)], raw_floor)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std
    pad = CLAMP * np.maximum(f.std(0), raw_floor)
    lo, hi = f.min(0) - pad, f.max(0) + pad

    def cv_errors(lam: float) -> dict[int, float]:
        errs = {}
        for g in np.unique(groups):
            test = groups == g
            coef, b = _solve(z[~test], targets[~test], lam)
            d = z[test] @ coef + b - targets[test]
            errs[int(g)] = float(np.hypot(d[:, 0], d[:, 1] * aspect).mean())
        return errs

    per_lam = {lam: cv_errors(lam) for lam in LAMBDAS}
    lam = min(per_lam, key=lambda k: np.mean(list(per_lam[k].values())))
    scored = [e for g, e in per_lam[lam].items() if score is None or g in score]
    error = float(np.mean(scored or list(per_lam[lam].values())))
    coef, b = _solve(z, targets, lam)
    return GazeModel(monitor, mean, std, coef, b, lam, blink, error, camera,
                     time.strftime("%Y-%m-%dT%H:%M:%S"), kind, lo, hi)


def load_model(path: Path):
    """A saved calibration, whichever kind it is."""
    if json.loads(path.read_text()).get("kind") == "geometric":
        from .geometry import GeoModel

        return GeoModel.load(path)
    return GazeModel.load(path)


def data_error(model, data: dict, aspect: float) -> float:
    """A model's mean error on a sample set (per dot or cell, then averaged),
    in monitor widths; points it can't place count as a full width."""
    d = model.predict_data(data) - data["targets"]
    e = np.hypot(d[:, 0], d[:, 1] * aspect)
    e = np.where(np.isfinite(e), e, 1.0)
    g = data["groups"]
    return float(np.mean([e[g == k].mean() for k in np.unique(g)]))


def subset(data: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim and len(v) == len(mask)}


def prepare(data: dict) -> tuple[np.ndarray, list[int], float]:
    """Which frames to fit on: no blinks, no stray frames within a dot, and
    only dots or cells with enough frames left. Returns (keep, groups used, blink)."""
    feats, opens, groups = data["feats"], data["opens"], data["groups"]
    blink = 0.6 * float(np.median(opens))
    keep = opens >= blink
    dots = keep & (groups < MOUSE)
    keep[dots] = inliers(feats[dots], groups[dots])
    counts = {int(g): int((keep & (groups == g)).sum()) for g in np.unique(groups)}
    keep &= np.array([counts[int(g)] >= MIN_FRAMES for g in groups], bool)
    return keep, sorted(g for g, c in counts.items() if c >= MIN_FRAMES), blink


def fit_all(data: dict, aspect: float, monitor: str, camera: str, score: set[int] | None = None,
            screen_mm: tuple[float, float] | None = None) -> tuple[list, list[int], int]:
    """Every model there is data for: the basic and rich regressions, and the
    geometric model when head poses and the screen's size are known.
    Returns (models, groups used, frames used)."""
    keep, used, blink = prepare(data)
    if not used:
        raise ValueError("no dot or cell had enough clear frames")
    scored = None if score is None else score & set(used)
    d = subset(data, keep)
    models = [fit(d["feats"], d["targets"], d["groups"], aspect, monitor, blink, camera, scored)]
    if "rich" in d and len(d["rich"]):
        models.append(fit(d["rich"], d["targets"], d["groups"], aspect, monitor, blink, camera, scored, "rich"))
        if screen_mm and "pose" in d and len(d["pose"]):
            from . import geometry

            models.append(geometry.fit(d, d["groups"], screen_mm, monitor, blink, camera, scored))
    return models, used, int(keep.sum())


def choose(models: list):
    """The model to use: the lowest cross-validated error, except that the
    geometric model wins unless it is GEOMETRIC_MARGIN worse. Cross-validation
    within one sitting flatters the regressions (they lean on how the head
    happened to sit); in a later sitting the geometric model held up best."""
    best = min(models, key=lambda m: m.error)
    geo = next((m for m in models if m.kind == "geometric"), None)
    if geo is not None and geo.error <= GEOMETRIC_MARGIN * best.error:
        return geo
    return best


def fit_samples(data: dict, aspect: float, monitor: str, camera: str, score: set[int] | None = None,
                screen_mm: tuple[float, float] | None = None):
    """Fit every model there is data for and choose one (see choose()).
    Returns (model, groups used, frames used, {kind: error})."""
    models, used, frames = fit_all(data, aspect, monitor, camera, score, screen_mm)
    return choose(models), used, frames, {m.kind: m.error for m in models}
