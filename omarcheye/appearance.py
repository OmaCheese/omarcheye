"""Per-person appearance model: where you look, learned from how your eyes
look in your own calibration.

The landmark features compress each eye into a few numbers measured on
MediaPipe's 256-pixel face crop. This model reads more of the eye: the
histogram of oriented gradients (HOG) of each eye's patch, cut from the
full-resolution frame (eyepatch.eye_descriptor: 2 x 1620 numbers). With
only a few hundred calibration frames it can't learn eyes in general, but
it can learn yours: a ridge regression from four blocks to the screen:

- the eye descriptor, centred and scaled to unit mean squared length;
- head pose (yaw, pitch, position), spreads floored and live values clamped
  as in model.py, so sitting differently can't throw it off the screen;
- the tracker's eye features (the rich set without head pose);
- where the eye network's gaze meets the camera's plane (eyenet.py).

Each block gets a weight; the weights and the ridge strength are picked by
leaving one dot (or pointer cell) out at a time, because frames of one dot
look alike and leaving single frames out would flatter the fit. That same
error is what calibration compares with the other models. On MPIIFaceGaze
(bench/, calibrate on 30 images of one day): 6.4% of the screen width
within a day and 14.5% on other days, against 10.6% and 17.3% for the rich
features alone.

Calibration samples keep each frame's descriptor (float16 numbers, not an
image) so `omarcheye refine` and `omarcheye test` can refit it.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import CLAMP, HEAD_FLOOR

DESC = 3240  # eyepatch.eye_descriptor's length for 60x36 patches
EYE_COLS = 16  # rich[:16]: iris, lids, eye-direction scores; rich[16:]: head pose
D_WEIGHTS = (0.3, 1.0, 3.0, 10.0)  # descriptor block against the feature blocks
H_WEIGHTS = (0.0, 1.0, 3.0)  # eye network's hit point (0: not used, and not needed live)
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)  # ridge strength per frame
SELECT_FRAMES = 600  # weights and strength are picked on at most this many frames


def _stats(x: np.ndarray, floor: np.ndarray | None = None):
    mu = x.mean(0)
    sd = x.std(0)
    if floor is not None:
        sd = np.maximum(sd, floor)
    sd[sd < 1e-9] = 1.0
    return mu, sd, x.min(0) - CLAMP * sd, x.max(0) + CLAMP * sd


@dataclass
class AppearanceModel:
    monitor: str
    d_mu: np.ndarray  # descriptor mean
    d_scale: float  # root mean squared length of the centred descriptors
    f_mu: np.ndarray  # rich features: mean, spread, clamp range
    f_sd: np.ndarray
    f_lo: np.ndarray
    f_hi: np.ndarray
    h_mu: np.ndarray  # eye network's hit point: mean, spread, clamp range
    h_sd: np.ndarray
    h_lo: np.ndarray
    h_hi: np.ndarray
    wd: float  # block weights
    wh: float
    lam: float
    coef: np.ndarray  # (DESC + 21 + 2, 2)
    intercept: np.ndarray  # (2,)
    blink: float
    error: float  # cross-validated (leave one dot or cell out), monitor widths
    camera: str = ""
    created: str = ""
    kind: str = "appearance"

    def _z(self, desc: np.ndarray, rich: np.ndarray, hit: np.ndarray) -> np.ndarray:
        return _blocks(self, np.atleast_2d(desc), np.atleast_2d(rich), np.atleast_2d(hit), self.wd, self.wh)

    def predict_data(self, data: dict) -> np.ndarray:
        """(n, 2) predictions in monitor fractions; NaN where a needed input is missing."""
        hit = data["net"][:, 2:4] if self.wh else np.zeros((len(data["rich"]), 2))
        return self._z(data["desc"].astype(np.float32), data["rich"], hit) @ self.coef + self.intercept

    def predict_sample(self, s) -> tuple[float, float]:
        if s.desc is None or s.rich is None or (self.wh and s.net is None):
            return math.nan, math.nan
        hit = s.net[2:4] if self.wh else np.zeros(2)
        x, y = (self._z(s.desc, s.rich, hit) @ self.coef + self.intercept)[0]
        return float(x), float(y)

    def describe(self) -> str:
        return (f"appearance model (descriptor weight {self.wd:g}, eye network weight {self.wh:g}, "
                f"ridge {self.lam:.3g})")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        path.write_text(json.dumps(data) + "\n")

    @classmethod
    def load(cls, path: Path) -> "AppearanceModel":
        data = json.loads(path.read_text())
        for k in ("d_mu", "f_mu", "f_sd", "f_lo", "f_hi", "h_mu", "h_sd", "h_lo", "h_hi", "coef", "intercept"):
            data[k] = np.array(data[k])
        return cls(**data)


def _blocks(p, desc, rich, hit, wd, wh) -> np.ndarray:
    """The design matrix: weighted, normalised blocks side by side. `p` has the
    normalisation (an AppearanceModel, or a namespace from fit)."""
    d = (desc - p.d_mu) / p.d_scale
    f = (np.clip(rich, p.f_lo, p.f_hi) - p.f_mu) / p.f_sd / math.sqrt(rich.shape[1])
    h = (np.clip(hit, p.h_lo, p.h_hi) - p.h_mu) / p.h_sd / math.sqrt(2)
    return np.hstack([math.sqrt(wd) * d, f, math.sqrt(wh) * h])


def _group_cv(z: np.ndarray, y: np.ndarray, groups: np.ndarray, aspect: float) -> tuple[float, float]:
    """(error, alpha): leave-one-group-out error of kernel ridge on z for the
    best ALPHAS, from one eigendecomposition (the held-out residuals of a
    group are (I - H_gg)^-1 times its residuals in the full fit)."""
    n = len(z)
    z = z - z.mean(0)  # with the intercept, the hat matrix is 1/n plus the centred one
    k = z @ z.T
    s, q = np.linalg.eigh(k)
    s = np.maximum(s, 0)
    yc = y - y.mean(0)
    qty = q.T @ yc
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    best = (math.inf, ALPHAS[0])
    for a in ALPHAS:
        shrink = s / (s + a * n)
        resid = yc - q @ (shrink[:, None] * qty)
        errs = []
        for i in idx:
            qg = q[i]
            hgg = (qg * shrink) @ qg.T + 1.0 / n  # the intercept's share
            e = np.linalg.solve(np.eye(len(i)) - hgg, resid[i])
            errs.append(np.hypot(e[:, 0], e[:, 1] * aspect).mean())
        err = float(np.mean(errs))
        if err < best[0]:
            best = (err, a)
    return best


def fit(data: dict, groups: np.ndarray, aspect: float, monitor: str, blink: float,
        camera: str = "", seed: int = 0) -> AppearanceModel:
    """Fit on a sample set with "desc", "rich", "net" and "targets" (monitor
    fractions); frames missing the descriptor are left out by the caller."""
    desc = data["desc"].astype(np.float32)
    rich = data["rich"]
    y = data["targets"]
    hit_ok = np.isfinite(data["net"][:, 2:4]).all(1)
    hit = np.where(hit_ok[:, None], data["net"][:, 2:4], 0.0)

    class P:  # normalisation from all frames
        pass

    P.d_mu = desc.mean(0)
    P.d_scale = float(math.sqrt(max(((desc - P.d_mu) ** 2).sum(1).mean(), 1e-12)))
    floor = np.zeros(rich.shape[1])
    floor[-len(HEAD_FLOOR):] = HEAD_FLOOR
    P.f_mu, P.f_sd, P.f_lo, P.f_hi = _stats(rich, floor)
    P.h_mu, P.h_sd, P.h_lo, P.h_hi = _stats(hit[hit_ok]) if hit_ok.any() else (np.zeros(2), np.ones(2),
                                                                               np.full(2, -1.0), np.ones(2))

    # Pick the weights and strength on a subset of frames, keeping every group.
    rng = np.random.default_rng(seed)
    pick = np.arange(len(y))
    if len(pick) > SELECT_FRAMES:
        pick = np.sort(rng.choice(pick, SELECT_FRAMES, replace=False))
    best = (math.inf, None)
    for wh in H_WEIGHTS:
        if wh and hit_ok[pick].mean() < 0.9:
            continue  # the network read too few of these frames to lean on it
        for wd in D_WEIGHTS:
            z = _blocks(P, desc[pick], rich[pick], hit[pick], wd, wh)
            err, a = _group_cv(z, y[pick], groups[pick], aspect)
            if err < best[0]:
                best = (err, (wd, wh, a))
    error, (wd, wh, a) = best

    use = hit_ok if wh else np.ones(len(y), bool)
    z = _blocks(P, desc[use], rich[use], hit[use], wd, wh)
    n = len(z)
    z_mu, ym = z.mean(0), y[use].mean(0)
    zc = z - z_mu
    alpha = np.linalg.solve(zc @ zc.T + a * n * np.eye(n), y[use] - ym)
    coef = zc.T @ alpha
    return AppearanceModel(monitor, P.d_mu, P.d_scale, P.f_mu, P.f_sd, P.f_lo, P.f_hi, P.h_mu, P.h_sd, P.h_lo,
                           P.h_hi, wd, wh, a * n, coef, ym - z_mu @ coef, blink, error, camera,
                           time.strftime("%Y-%m-%dT%H:%M:%S"))
