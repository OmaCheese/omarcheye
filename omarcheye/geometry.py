"""Geometric gaze model: the screen, the camera and your eyes, in millimetres.

The regression in model.py learns how features map to the screen within the
postures seen during calibration, and extrapolates when you sit differently.
This model follows the physics instead, so head movement is handled by
geometry and calibration only has to find a few physical numbers.

Coordinates are the camera's (MediaPipe's convention): x to the right of the
image, y up, z towards the viewer; the face sits at negative z.

- Your eye is at the head position MediaPipe reports, times a scale `s`
  (MediaPipe assumes a 63° lens; a phone camera's differs, which scales all
  its distances).
- Your gaze leaves the eye along the head's forward axis, turned by the eye's
  own rotation: across by kx·u + ox and up by ky·v + kl·lid + oy (u, v: the
  iris offsets; lid: the upper lid, which follows the eye up and down).
- The screen is a W×H mm rectangle. The camera sits `dx` mm right of its
  centre and `lift` mm above its top edge, tilted by `tilt` (about x) and
  `pan` (about y).
- Where the gaze ray meets the screen plane is where you look.

Calibration fits the ten numbers by least squares, with weak priors that
keep directions the dots can't tell apart physically sensible. Whether the
image is mirrored (Flux has a mirror switch) is tried both ways.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARAMS = ("kx", "ky", "kl", "ox", "oy", "log_s", "tilt", "pan", "dx", "lift")
# prior mean and spread per parameter (radians, mm, log scale)
PRIOR_MEAN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.log(1.3), 0.0, 0.0, 0.0, 20.0])
PRIOR_SD = np.array([4.0, 4.0, 3.0, 0.3, 0.3, 0.6, 0.35, 0.17, 100.0, 40.0])
PRIOR_MM = 5.0  # a one-spread move of any parameter costs as much as 5 mm on every frame


def inputs(data: dict) -> dict:
    """What the model reads from a sample set (needs "rich" and "pose")."""
    r, pose = data["rich"], data["pose"].reshape(-1, 4, 4)
    return {
        "u": (r[:, 0] + r[:, 2]) / 2,
        "v": (r[:, 1] + r[:, 3]) / 2,
        "lid": (r[:, 4] + r[:, 6]) / 2,
        "R": pose[:, :3, :3],
        "t": pose[:, :3, 3] * 10.0,  # cm to mm
    }


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def predict_mm(theta: np.ndarray, x: dict, screen: tuple[float, float], mirror: bool) -> np.ndarray:
    """(n, 2) points on the screen in mm from its top-left corner (NaN when
    the gaze ray misses the screen plane)."""
    kx, ky, kl, ox, oy, log_s, tilt, pan, dx, lift = theta
    W, _ = screen
    u, R, t = x["u"], x["R"], x["t"]
    if mirror:  # undo a mirrored image: flip x everywhere
        flip = np.array([-1.0, 1.0, 1.0])
        u, t, R = -u, t * flip, R * flip[:, None] * flip[None, :]
    origin = math.exp(log_s) * t
    alpha = kx * u + ox
    beta = ky * x["v"] + kl * x["lid"] + oy
    g_head = np.stack([np.sin(alpha) * np.cos(beta), np.sin(beta), np.cos(alpha) * np.cos(beta)], axis=1)
    g = np.einsum("nij,nj->ni", R, g_head)
    rc = _rx(tilt) @ _ry(pan)
    normal = rc[:, 2]
    corner = rc @ np.array([-(W / 2 + dx), -lift, 0.0])  # the screen's top-left corner
    towards = g @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = ((corner - origin) @ normal) / towards
    hit = origin + lam[:, None] * g
    local = hit @ rc  # back into the screen's own frame
    out = np.stack([local[:, 0] + W / 2 + dx, -local[:, 1] - lift], axis=1)
    out[(towards <= 1e-3) | (lam <= 0)] = np.nan
    return out


def _residuals(theta, x, target_mm, screen, mirror, prior_w):
    d = predict_mm(theta, x, screen, mirror) - target_mm
    d = np.where(np.isfinite(d), d, 1e4).ravel()
    return np.concatenate([d, prior_w * (theta - PRIOR_MEAN) / PRIOR_SD])


def _least_squares(fun, x0: np.ndarray, iters: int = 80) -> np.ndarray:
    """Levenberg-Marquardt with a numerical Jacobian."""
    x = x0.astype(float).copy()
    r = fun(x)
    cost = r @ r
    damping = 1e-3
    for _ in range(iters):
        jac = np.empty((len(r), len(x)))
        for i in range(len(x)):
            step = 1e-5 * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += step
            jac[:, i] = (fun(xp) - r) / step
        a, g = jac.T @ jac, jac.T @ r
        while True:
            dx = np.linalg.solve(a + damping * np.diag(np.diag(a) + 1e-9), -g)
            r2 = fun(x + dx)
            c2 = r2 @ r2
            if c2 < cost:
                improved = cost - c2
                x, r, cost = x + dx, r2, c2
                damping = max(damping / 3, 1e-9)
                break
            damping *= 4
            if damping > 1e9:
                return x
        if improved < 1e-9 * cost:
            break
    return x


def fit_theta(x: dict, target_mm: np.ndarray, screen, mirror: bool, start: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Best parameters and their data cost (mean squared mm). Without a
    start, the four sign combinations of the eye gains are tried."""
    prior_w = PRIOR_MM * math.sqrt(len(target_mm))
    starts = [start] if start is not None else [
        PRIOR_MEAN + np.array([sx * 2.5, sy * 2.5, 0, 0, 0, 0, 0, 0, 0, 0]) for sx in (1, -1) for sy in (1, -1)
    ]
    best = None
    for s0 in starts:
        theta = _least_squares(lambda th: _residuals(th, x, target_mm, screen, mirror, prior_w), s0)
        d = predict_mm(theta, x, screen, mirror) - target_mm
        cost = float(np.nanmean(np.sum(d * d, 1))) if np.isfinite(d).any() else math.inf
        if best is None or cost < best[1]:
            best = (theta, cost)
    return best


def _subset(x: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in x.items()}


@dataclass
class GeoModel:
    monitor: str
    theta: np.ndarray
    mirror: bool
    screen: tuple[float, float]  # mm
    blink: float
    error: float  # cross-validated mean error, in monitor widths
    camera: str = ""
    created: str = ""
    kind: str = "geometric"

    def predict_data(self, data: dict) -> np.ndarray:
        """(n, 2) predictions in monitor fractions."""
        return predict_mm(self.theta, inputs(data), self.screen, self.mirror) / np.array(self.screen)

    def predict_sample(self, s) -> tuple[float, float]:
        if s.rich is None or s.pose is None:
            return math.nan, math.nan
        x, y = self.predict_data({"rich": s.rich[None], "pose": s.pose[None]})[0]
        return float(x), float(y)

    def describe(self) -> str:
        th = dict(zip(PARAMS, self.theta))
        return (f"camera {th['lift']:.0f} mm above the screen, {th['dx']:+.0f} mm from its centre, "
                f"tilted {math.degrees(th['tilt']):+.0f}°, distances ×{math.exp(th['log_s']):.2f}"
                + (", image mirrored" if self.mirror else ""))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        data["params"] = dict(zip(PARAMS, self.theta.tolist()))
        path.write_text(json.dumps(data, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> "GeoModel":
        data = json.loads(path.read_text())
        data.pop("params", None)
        data["theta"] = np.array(data["theta"])
        data["screen"] = tuple(data["screen"])
        return cls(**data)


def fit(data: dict, groups: np.ndarray, screen: tuple[float, float], monitor: str, blink: float,
        camera: str = "", score: set[int] | None = None) -> GeoModel:
    """Fit on all frames; the reported error leaves each dot or cell out in
    turn (refitting from the full fit's parameters)."""
    x = inputs(data)
    target_mm = data["targets"] * np.array(screen)
    fits = [(mirror, *fit_theta(x, target_mm, screen, mirror)) for mirror in (False, True)]
    mirror, theta, _ = min(fits, key=lambda f: f[2])
    errs = []
    for g in np.unique(groups):
        if score is not None and int(g) not in score:
            continue
        test = groups == g
        th, _ = fit_theta(_subset(x, ~test), target_mm[~test], screen, mirror, start=theta)
        d = predict_mm(th, _subset(x, test), screen, mirror) - target_mm[test]
        e = np.hypot(d[:, 0], d[:, 1]) / screen[0]
        errs.append(float(np.nanmean(np.where(np.isfinite(e), e, 1.0))))
    return GeoModel(monitor, theta, mirror, tuple(screen), blink, float(np.mean(errs)), camera,
                    time.strftime("%Y-%m-%dT%H:%M:%S"))
