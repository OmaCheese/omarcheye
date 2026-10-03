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
    error: float  # leave-one-dot-out mean error, in monitor widths
    camera: str = ""
    created: str = ""

    def predict(self, feat: np.ndarray) -> tuple[float, float]:
        z = (expand(feat)[0] - self.mean) / self.std
        x, y = z @ self.coef + self.intercept
        return float(x), float(y)

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


def fit(f: np.ndarray, targets: np.ndarray, groups: np.ndarray, aspect: float,
        monitor: str, blink: float, camera: str = "") -> GazeModel:
    """f: (n, features) frames; targets: (n, 2) monitor fractions of the dot
    each frame was looking at; groups: dot index per frame; aspect: height/width."""
    x = expand(f)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std

    def cv_error(lam: float) -> float:
        errs = []
        for g in np.unique(groups):
            test = groups == g
            coef, b = _solve(z[~test], targets[~test], lam)
            d = z[test] @ coef + b - targets[test]
            errs.append(np.hypot(d[:, 0], d[:, 1] * aspect).mean())
        return float(np.mean(errs))

    errors = {lam: cv_error(lam) for lam in LAMBDAS}
    lam = min(errors, key=errors.get)
    coef, b = _solve(z, targets, lam)
    return GazeModel(monitor, mean, std, coef, b, lam, blink, errors[lam], camera,
                     time.strftime("%Y-%m-%dT%H:%M:%S"))
