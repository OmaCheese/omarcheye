"""Calibrated map from gaze features to a point on the monitor.

Ridge regression on standardised features plus the quadratic eye terms
(u², v², uv). The ridge strength is picked by leave-one-dot-out
cross-validation, and that same error is the accuracy calibration reports.
Outputs are monitor fractions (0..1 across, 0..1 down), so a calibration
stays valid when the monitor's scale or resolution changes.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracker import FEATURES

LAMBDAS = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


def expand(f: np.ndarray) -> np.ndarray:
    f = np.atleast_2d(f)
    u, v = f[:, 0], f[:, 1]
    return np.column_stack([f, u * u, v * v, u * v])


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

    def predict(self, feat: np.ndarray) -> tuple[float, float]:
        z = (expand(feat)[0] - self.mean) / self.std
        x, y = z @ self.coef + self.intercept
        return float(x), float(y)

    def group_error(self, f: np.ndarray, targets: np.ndarray, groups: np.ndarray, aspect: float) -> float:
        """Mean distance from prediction to target, averaged per group, in monitor widths."""
        z = (expand(f) - self.mean) / self.std
        d = z @ self.coef + self.intercept - targets
        e = np.hypot(d[:, 0], d[:, 1] * aspect)
        return float(np.mean([e[groups == g].mean() for g in np.unique(groups)]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        data["features"] = list(FEATURES)
        path.write_text(json.dumps(data, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> "GazeModel":
        data = json.loads(path.read_text())
        if data.pop("features", None) != list(FEATURES):
            raise ValueError(f"{path} was made by another omeye version; run `omeye calibrate` again")
        for k in ("mean", "std", "coef", "intercept"):
            data[k] = np.array(data[k])
        return cls(**data)


MIN_FRAMES = 8  # a dot or cell with fewer usable frames is left out
MOUSE = 1000  # group ids from here up are pointer cells (omeye refine); below are dots


def fit(f: np.ndarray, targets: np.ndarray, groups: np.ndarray, aspect: float,
        monitor: str, blink: float, camera: str = "", score: set[int] | None = None) -> GazeModel:
    """f: (n, features) frames; targets: (n, 2) monitor fractions of where
    each frame was looking; groups: dot or cell per frame; aspect: height/width.
    The reported error is the cross-validated error over the `score` groups
    (default: all of them)."""
    x = expand(f)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std

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
    return GazeModel(monitor, mean, std, coef, b, lam, blink, error, camera, time.strftime("%Y-%m-%dT%H:%M:%S"))


def fit_samples(feats: np.ndarray, opens: np.ndarray, groups: np.ndarray, targets: np.ndarray,
                aspect: float, monitor: str, camera: str, score: set[int] | None = None):
    """Drop blinks and stray frames, then fit. Returns (model, groups used, frames used)."""
    blink = 0.6 * float(np.median(opens))
    keep = opens >= blink
    dots = keep & (groups < MOUSE)
    keep[dots] = inliers(feats[dots], groups[dots])
    counts = {int(g): int((keep & (groups == g)).sum()) for g in np.unique(groups)}
    keep &= np.array([counts[int(g)] >= MIN_FRAMES for g in groups], bool)
    used = sorted(g for g, c in counts.items() if c >= MIN_FRAMES)
    if not used:
        raise ValueError("no dot or cell had enough clear frames")
    model = fit(feats[keep], targets[keep], groups[keep], aspect, monitor, blink, camera,
                None if score is None else score & set(used))
    return model, used, int(keep.sum())
