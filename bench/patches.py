"""Route A: normalised eye patches + a per-person appearance model.

    uv run python -m bench.patches extract    # eye patches of every image -> patches/pXX.npz
    uv run python -m bench.patches evaluate   # score appearance models with bench.evaluate
    uv run python -m bench.patches write      # the best one's features -> features/<name>/pXX.npz
    uv run python -m bench.patches sweep      # other-sitting error against calibration size K
    uv run python -m bench.patches recentred  # with recentring on R images of each test day
    uv run python -m bench.patches time       # milliseconds per frame of the live parts
    uv run python -m bench.patches show       # a montage of patches (scratch PNG)

Patches (omarcheye.eyepatch.eye_patch): each eye cut out of the original
image along its corner line, scaled so the corner distance is a fixed share
of the patch (1/1.6 of its width), stored raw (grayscale, uint8):

  g60   60x36, similarity transform (roll removed, scaled by corner distance)
  g100  100x60, the same, larger (about the dataset's native resolution)
  z60   60x36, head-pose normalisation (Zhang, Sugano & Bulling 2018): a
        virtual camera turned to look straight at the eye, its x axis along the
        head's, at a fixed distance; MediaPipe's own camera model (63° vertical
        field of view) and head pose, as the live tracker would have them

Contrast normalisation (equalised histogram, CLAHE, none) and smaller sizes
(30x18 from g60) are applied when loading.

The model is fitted per person on the calibration images only, inside
bench.evaluate.evaluate() with a custom fitter (`appearance()`), so the
splits, seeds and scoring are exactly the benchmark's. Features handed to
evaluate() are [patch descriptor of both eyes | rich (21)], and the fitter
splits them again. The fitter is a ridge regression (dual form, an n x n
kernel) on blocks: the patch descriptor (pixels, or their top PCA
components from the calibration images, or HOG), head pose (yaw, pitch, hx,
hy, hz with omarcheye.model's minimum spreads and clamping) and optionally
the tracker's eye features (iris u, v per eye, lids, eye-direction scores).
The patch block's weight against the others and the ridge strength are
picked by leave-one-out on the calibration images.

Result (% of the screen width, same sitting 5-fold / other sitting K=30 /
K=100; baseline rich 10.6 / 17.3 / 14.5): pixels and head pose 7.7 / 19.3 /
15.6 (great within a sitting, worse across); HOG of the patches, head pose
and the tracker's eye features 7.8 / 16.0 / 13.2; the same plus the
disk-refined iris u, v 7.4 / 15.3 / 12.6. Larger patches (100x60), CLAHE,
PCA, an RBF kernel and the head-pose normalisation did not help; the
equalised histogram did for pixels (raw: 24.1 at K=30). From K=20
calibration images on, the HOG model beats the baseline (17.2 vs 18.9); at
K=10 it is worse (24.6 vs 21.2).
"""

from __future__ import annotations

import os

# One BLAS thread per process: the benchmark runs at most 4 processes.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import multiprocessing as mp
import sys
import time

import cv2
import numpy as np

from omarcheye import eyepatch as ep
from omarcheye.model import CLAMP, HEAD_FLOOR

from .evaluate import ALPHAS, PROTOCOLS, evaluate, load, table
from .mpii import CACHE, SUBJECTS, image_path

OUT = CACHE / "patches"
CHUNK = 50
STORE = {"g60": (60, 36), "g100": (100, 60), "z60": (60, 36)}

# Pose normalisation: MediaPipe's camera (63° vertical field of view), the eye
# put at D_N cm, and a focal length that makes MediaPipe's typical eye (3.19 cm
# corner to corner in its metric face model) as wide as in the g60 patches.
MP_FOV = 63.0
D_N = 50.0
EYE_CM = 3.19


def mp_camera(w: int, h: int) -> np.ndarray:
    f = (h / 2) / math.tan(math.radians(MP_FOV / 2))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def pose_warp(points: np.ndarray, matrix: np.ndarray, eye: ep.Eye, cam: np.ndarray,
              size: tuple[int, int], span: float = ep.SPAN) -> np.ndarray:
    """3x3 homography image -> patch for the head-pose normalisation."""
    a, b = points[eye.left], points[eye.right]
    centre = (a + b) / 2
    flip = np.diag([1.0, -1.0, -1.0])  # MediaPipe (x right, y up, z to the viewer) -> OpenCV (y down, z forward)
    rot = flip @ matrix[:3, :3] @ flip
    depth = -float(matrix[2, 3])
    ray = np.linalg.solve(cam, [centre[0], centre[1], 1.0])
    e = ray * depth / ray[2]
    z = e / np.linalg.norm(e)
    y = np.cross(z, rot[:, 0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    rn = np.vstack([x, y, z])
    scale = np.diag([1.0, 1.0, D_N / np.linalg.norm(e)])
    f_n = (size[0] / span) * D_N / EYE_CM
    cam_n = np.array([[f_n, 0, size[0] / 2], [0, f_n, size[1] / 2], [0, 0, 1.0]])
    return cam_n @ scale @ rn @ np.linalg.inv(cam)


def _extract(job: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, dict]:
    cv2.setNumThreads(1)
    subject, rows = job
    d = load(subject)
    out = {k: np.zeros((len(rows), 2, s[1], s[0]), np.uint8) for k, s in STORE.items()}
    for j, i in enumerate(rows):
        img = cv2.imread(str(image_path(subject, str(d["files"][i]))), cv2.IMREAD_GRAYSCALE)
        pts = d["points"][i].astype(float)
        cam = mp_camera(img.shape[1], img.shape[0])
        for e, eye in enumerate(ep.EYES):
            for k in ("g60", "g100"):
                s = STORE[k]
                out[k][j, e] = ep.warp(img, ep.eye_transform(pts, eye, s[0] / ep.SPAN, s), s)
            h = pose_warp(pts, d["matrix"][i].astype(float), eye, cam, STORE["z60"])
            out["z60"][j, e] = cv2.warpPerspective(img, h, STORE["z60"], flags=cv2.INTER_AREA,
                                                   borderMode=cv2.BORDER_REPLICATE)
    return subject, rows, out


def extract(subjects=SUBJECTS, workers: int = 4) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in subjects:
        rows = np.flatnonzero(load(s)["ok"])
        jobs += [(s, rows[k:k + CHUNK]) for k in range(0, len(rows), CHUNK)]
    acc = {s: {} for s in subjects}
    t0 = time.monotonic()
    with mp.get_context("fork").Pool(min(workers, 4)) as pool:
        for n, (s, rows, out) in enumerate(pool.imap_unordered(_extract, jobs), 1):
            nrows = len(load(s)["day"])
            for k, a in out.items():
                acc[s].setdefault(k, np.zeros((nrows, *a.shape[1:]), np.uint8))[rows] = a
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} chunks, {time.monotonic() - t0:.0f} s", file=sys.stderr, flush=True)
    for s in subjects:
        np.savez(OUT / f"{s}.npz", ok=load(s)["ok"], **acc[s])
    print(f"extracted {len(subjects)} subjects in {time.monotonic() - t0:.0f} s", file=sys.stderr)


# ---------------------------------------------------------------- descriptors

hog = ep.hog


def descriptors(subject: str, store: str = "g60", size: tuple[int, int] | None = None,
                how: str = "equalize", kind: str = "pixels") -> np.ndarray:
    """(n, d) float32 descriptor of both eyes per row (NaN where no face):
    the normalised patch's pixels (scaled to 0..1), its HOG ("hog": 6-pixel
    cells, "hog4": 4-pixel cells, ...), or several joined by "+"."""
    with np.load(OUT / f"{subject}.npz") as z:
        ok, raw = z["ok"], z[store]
    n = len(ok)
    first = None
    out = None
    for i in np.flatnonzero(ok):
        parts = []
        for e in range(2):
            p = raw[i, e]
            if size is not None and size != (p.shape[1], p.shape[0]):
                p = cv2.resize(p, size, interpolation=cv2.INTER_AREA)
            p = ep.normalise(p, how)
            for k in kind.split("+"):
                if k.startswith("hog"):
                    parts.append(hog(p, int(k[3:] or 6)))
                else:
                    parts.append(p.ravel().astype(np.float32) / 255.0)
        v = np.concatenate(parts)
        if first is None:
            first = v
            out = np.full((n, len(v)), np.nan, np.float32)
        out[i] = v
    return out


def feature_loader(store="g60", size=None, how="equalize", kind="pixels", extra: str | None = None,
                   hit: bool = False):
    """load_features for evaluate(): [descriptor | rich (21) | extra eye features | hit point].
    `extra`: a features/<name> set whose first four columns (uA, vA, uB, vB)
    are appended, e.g. "iris-disk-rich" for the refined iris centres. `hit`:
    the eye network's hit point (features/eyenet X[:, 2:4]) appended last."""
    def load_features(s: str) -> np.ndarray:
        parts = [descriptors(s, store, size, how, kind), load(s)["rich"]]
        if extra:
            with np.load(CACHE / "features" / extra / f"{s}.npz") as z:
                parts.append(z["X"][:, :4])
        if hit:
            with np.load(CACHE / "features" / "eyenet" / f"{s}.npz") as z:
                parts.append(z["X"][:, 2:4])
        return np.hstack(parts)
    return load_features


# ---------------------------------------------------------------- fitter

WEIGHTS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # patch block's kernel weight against the feature blocks
NO_PATCH = 0.0


def _loo(k: np.ndarray, Y: np.ndarray, alphas) -> tuple[float, float]:
    """Best (LOO mean Euclidean error, lambda) for kernel ridge on a centred kernel k."""
    n = len(k)
    lam_k, q = np.linalg.eigh(k)
    lam_k = np.maximum(lam_k, 0)
    yc = Y - Y.mean(0)
    qty = q.T @ yc
    best = (np.inf, 1.0)
    for a in alphas:
        lam = a * n
        shrink = lam_k / (lam_k + lam)
        fitted = q @ (shrink[:, None] * qty)
        h = (q ** 2) @ shrink + 1.0 / n
        loo = (yc - fitted) / np.maximum(1 - h, 1e-6)[:, None]
        err = float(np.hypot(loo[:, 0], loo[:, 1]).mean())
        if err < best[0]:
            best = (err, lam)
    return best


HIT_WEIGHTS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)  # the hit-point block's weight against the feature blocks


def appearance(npix: int, pca: int | None = None, eye: bool = False, head: bool = True,
               weights=WEIGHTS, alphas=ALPHAS, rbf: bool = False, hit: int = 0, hit_weights=HIT_WEIGHTS):
    """Fitter for evaluate(): features are [descriptor (npix) | rich (21)].

    Blocks, each centred on the calibration images and scaled to unit mean
    squared norm: the descriptor (optionally projected on its top `pca`
    components), head pose (rich[-5:], spreads floored and clamped as in
    omarcheye.model) and, with `eye`, the tracker's eye features (rich[:16]).
    With `hit` > 0, the last `hit` columns (e.g. the eye network's hit point)
    are a fourth block, standardised and clamped like the features, with its
    own weight. Kernel = weight * descriptor kernel + feature kernel (+ hit
    weight * hit kernel), linear, or RBF on the descriptor with `rbf`; the
    weights and the ridge strength by leave-one-out."""

    def fit(X: np.ndarray, Y: np.ndarray, info: dict):
        n = len(X)
        P, R = X[:, :npix], X[:, npix:X.shape[1] - hit]
        H = X[:, X.shape[1] - hit:]
        h_mu, h_sd = H.mean(0), H.std(0)
        h_sd[h_sd < 1e-9] = 1.0
        h_lo, h_hi = H.min(0) - CLAMP * h_sd, H.max(0) + CLAMP * h_sd
        hz = (H - h_mu) / h_sd / math.sqrt(max(hit, 1))
        kh = hz @ hz.T
        mu = P.mean(0)
        pc = P - mu
        proj = None
        if pca:
            _, s, vt = np.linalg.svd(pc, full_matrices=False)
            proj = vt[: min(pca, len(s))].T
            pc = pc @ proj
        p_scale = math.sqrt(max((pc ** 2).sum(1).mean(), 1e-12))

        feats, f_lo, f_hi, f_mu, f_sd = [], [], [], [], []
        cols = []
        if head:
            cols += list(range(16, 21))  # rich: yaw, pitch, hx, hy, hz
        if eye:
            cols += list(range(16)) + list(range(21, R.shape[1]))  # rich's eye features, then any extra
        F = R[:, cols]
        floor = np.zeros(len(cols))
        if head:
            floor[:5] = HEAD_FLOOR
        f_mu = F.mean(0)
        f_sd = np.maximum(F.std(0), floor)
        f_sd[f_sd < 1e-9] = 1.0
        f_lo, f_hi = F.min(0) - CLAMP * f_sd, F.max(0) + CLAMP * f_sd
        fz = (F - f_mu) / f_sd / math.sqrt(max(len(cols), 1))

        def p_block(Pt):
            q = Pt - mu
            if proj is not None:
                q = q @ proj
            return q / p_scale

        pz = pc / p_scale
        if rbf:
            sq = (pz ** 2).sum(1)
            d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * pz @ pz.T, 0)
            gamma = 1.0 / max(np.median(d2[np.triu_indices(n, 1)]), 1e-12)
            kp = np.exp(-gamma * d2)
            one = np.full((n, n), 1.0 / n)
            kp_c = kp - one @ kp - kp @ one + one @ kp @ one
        else:
            kp_c = pz @ pz.T
        kf = fz @ fz.T
        best = (np.inf, None, None, None)
        for w in weights:
            for wh in (hit_weights if hit else (0.0,)):
                if w == 0 and wh == 0 and not cols:
                    continue
                k = w * kp_c + kf + wh * kh
                err, lam = _loo(k, Y, alphas)
                if err < best[0]:
                    best = (err, w, wh, lam)
        _, w, wh, lam = best
        k = w * kp_c + kf + wh * kh
        ym = Y.mean(0)
        alpha = np.linalg.solve(k + lam * np.eye(n), Y - ym)

        def predict(Xt: np.ndarray) -> np.ndarray:
            ht = (np.clip(Xt[:, Xt.shape[1] - hit:], h_lo, h_hi) - h_mu) / h_sd / math.sqrt(max(hit, 1))
            Ft = np.clip(Xt[:, npix:Xt.shape[1] - hit][:, cols], f_lo, f_hi)
            ft = (Ft - f_mu) / f_sd / math.sqrt(max(len(cols), 1))
            pt = p_block(Xt[:, :npix])
            if rbf:
                sqt = (pt ** 2).sum(1)
                kt = np.exp(-gamma * np.maximum(sqt[:, None] + sq[None, :] - 2 * pt @ pz.T, 0))
                kt_c = kt - kt.mean(1, keepdims=True) - kp.mean(0)[None, :] + kp.mean()
            else:
                kt_c = pt @ pz.T
            return (w * kt_c + ft @ fz.T + wh * ht @ hz.T) @ alpha + ym

        predict.choice = (w, wh, lam)
        return predict

    fit.__qualname__ = (f"appearance(pca={pca}, eye={eye}, head={head}" + (", rbf" if rbf else "")
                        + (f", hit={hit}" if hit else "") + ")")
    return fit


# ---------------------------------------------------------------- runs

VARIANTS = {
    # name: (store, size, normalisation, descriptor, fitter options[, extra eye features])
    "patch-g60-eq": ("g60", None, "equalize", "pixels", {}),
    "patch-g60-eq-pca20": ("g60", None, "equalize", "pixels", {"pca": 20}),
    "patch-g30-eq": ("g60", (30, 18), "equalize", "pixels", {}),
    "patch-g100-eq": ("g100", None, "equalize", "pixels", {}),
    "patch-g60-clahe": ("g60", None, "clahe", "pixels", {}),
    "patch-g60-raw": ("g60", None, "none", "pixels", {}),
    "patch-z60-eq": ("z60", None, "equalize", "pixels", {}),
    "patch-g60-hog": ("g60", None, "equalize", "hog", {}),
    "patch-g60-eq-rbf": ("g60", None, "equalize", "pixels", {"rbf": True}),
    "patch-g60-eq+eye": ("g60", None, "equalize", "pixels", {"eye": True}),
    "patch-g30-eq+eye": ("g60", (30, 18), "equalize", "pixels", {"eye": True}),
    "patch-g60-hog+eye": ("g60", None, "equalize", "hog", {"eye": True}),
    "patch-g60-hog4+eye": ("g60", None, "equalize", "hog4", {"eye": True}),
    "patch-g60-hog9+eye": ("g60", None, "equalize", "hog9", {"eye": True}),
    "patch-g100-hog10+eye": ("g100", None, "equalize", "hog10", {"eye": True}),
    "patch-g100-hog+eye": ("g100", None, "equalize", "hog", {"eye": True}),
    "patch-g60-raw-hog+eye": ("g60", None, "none", "hog", {"eye": True}),
    "patch-z60-hog+eye": ("z60", None, "equalize", "hog", {"eye": True}),
    "patch-g60-hog+pixels+eye": ("g60", None, "equalize", "hog+pixels", {"eye": True}),
    "patch-g60-hog+eye+disk": ("g60", None, "equalize", "hog", {"eye": True}, "iris-disk-rich"),
    "patch-g100-hog+eye+disk": ("g100", None, "equalize", "hog", {"eye": True}, "iris-disk-rich"),
    "patch-g60-hog4+eye+disk": ("g60", None, "equalize", "hog4", {"eye": True}, "iris-disk-rich"),
    # with the eye network's hit point (features/eyenet X[:, 2:4]) as a fourth block
    "patch-g60-hog+eye+disk+hit": ("g60", None, "equalize", "hog", {"eye": True, "hit": 2}, "iris-disk-rich", True),
    "patch-g60-hog+eye+hit": ("g60", None, "equalize", "hog", {"eye": True, "hit": 2}, None, True),
}


def npix_of(store, size, kind) -> int:
    w, h = size or STORE[store]
    n = 0
    for k in kind.split("+"):
        n += 2 * (len(hog(np.zeros((h, w), np.uint8), int(k[3:] or 6))) if k.startswith("hog") else w * h)
    return n


def run(names, workers: int = 4, protocols=PROTOCOLS, max_days=None) -> list[dict]:
    out = []
    for name in names:
        store, size, how, kind, opts, *extra = VARIANTS[name]
        t0 = time.monotonic()
        res = evaluate(name, feature_loader(store, size, how, kind, *extra),
                       appearance(npix_of(store, size, kind), **opts),
                       protocols=protocols, workers=workers, max_days=max_days, verbose=False,
                       save=max_days is None)
        print(f"{name}: {time.monotonic() - t0:.0f} s", file=sys.stderr, flush=True)
        print(table([res]), file=sys.stderr, flush=True)
        out.append(res)
    return out


def write_features(name: str = "patch-g60-hog+eye+disk", subjects=SUBJECTS) -> None:
    """A variant's evaluate() features -> features/<name>/pXX.npz (key X, float32,
    NaN where no face), with `npix` (the descriptor's length) for appearance()."""
    store, size, how, kind, opts, *extra = VARIANTS[name]
    loader = feature_loader(store, size, how, kind, *extra)
    out = CACHE / "features" / name
    out.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        np.savez(out / f"{s}.npz", X=loader(s).astype(np.float32), npix=npix_of(store, size, kind))


def integrated_fitter(npix: int = 3240):
    """Adapter: omarcheye.appearance.fit (as integrated) as an evaluate() fitter
    for [descriptor | rich (21) | hit point (2)] features; Y in mm, aspect 1,
    one group per image."""
    from omarcheye import appearance as app

    def split(X):
        net = np.full((len(X), 4), np.nan)
        net[:, 2:4] = X[:, npix + 21:npix + 23]
        return {"desc": X[:, :npix], "rich": X[:, npix:npix + 21], "net": net}

    def fit(X, Y, info):
        data = split(X)
        data["targets"] = Y
        m = app.fit(data, np.arange(len(X)), 1.0, "", 0.0)
        return lambda Xt: m.predict_data(split(Xt))

    fit.__qualname__ = "omarcheye.appearance.fit"
    return fit


def score_integrated(workers: int = 4) -> dict:
    name = "patch-g60-hog+eye+hit (omarcheye.appearance)"
    res = evaluate(name, feature_loader("g60", None, "equalize", "hog", None, True), integrated_fitter(),
                   protocols=("same", "other30", "other100"), workers=workers, verbose=False)
    print(table([res]))
    X = feature_loader("g60", None, "equalize", "hog", None, True)("p00")
    X = X[np.isfinite(X).all(1)]
    Y = load("p00")["target_px"][np.isfinite(feature_loader("g60", None, "equalize", "hog", None, True)("p00")).all(1)]
    for k in (30, 450):
        t = time.perf_counter()
        integrated_fitter()(X[:k], Y[:k], {})
        print(f"fit time, {k} frames: {time.perf_counter() - t:.2f} s")
    return res


def sweep(ks=(10, 20, 30, 50, 100, 150), names=("patch-g60-hog+eye+disk",), workers: int = 4) -> list[dict]:
    """Other-sitting error against the number of calibration images K, for
    appearance models and the baselines (omarcheye.model on rich, and on rich
    with the disk-refined iris), on identical splits. Saved as "<name> [K sweep]"
    so the standard results aren't overwritten."""
    from .evaluate import feat_rich, omarcheye_fitter
    from .iris import feature_loader as iris_features

    protos = tuple(f"other{k}" for k in ks)
    runs = [("rich (omarcheye.model)", feat_rich, omarcheye_fitter("rich")),
            ("iris-disk-rich (omarcheye.model)", iris_features("iris-disk-rich"), omarcheye_fitter("rich"))]
    for name in names:
        store, size, how, kind, opts, *extra = VARIANTS[name]
        runs.append((name, feature_loader(store, size, how, kind, *extra), appearance(npix_of(store, size, kind), **opts)))
    out = []
    for name, feats, fitter in runs:
        res = evaluate(f"{name} [K sweep]", feats, fitter, protocols=protos, workers=workers, verbose=False)
        out.append(res)
        print(name + ": " + ", ".join(f"K={k} {res[f'other{k}']['pct']:.1f}%" for k in ks if f"other{k}" in res),
              file=sys.stderr, flush=True)
    print("| K | " + " | ".join(r["name"].removesuffix(" [K sweep]") for r in out) + " | days (subjects) |")
    print("|---" * (len(out) + 2) + "|")
    for k in ks:
        p = f"other{k}"
        cells = [f"{r[p]['pct']:.1f}%" if p in r else "—" for r in out]
        print(f"| {k} | " + " | ".join(cells) + f" | {out[0][p]['fits'] if p in out[0] else 0} ({out[0][p]['subjects'] if p in out[0] else 0}) |")
    return out


def recentred(names=("patch-g60-hog+eye+disk", "patch-g60-hog+eye"), workers: int = 4,
              protocols=("other30", "other30+r1", "other30+r3", "other100", "other100+r3")) -> list[dict]:
    """Other sitting with and without recentring on R images of each test day,
    for the baseline, the disk-refined iris and appearance models; saved as
    "<name> [recentred]" so the standard results aren't overwritten."""
    from .evaluate import feat_rich, omarcheye_fitter
    from .iris import feature_loader as iris_features

    runs = [("rich (omarcheye.model)", feat_rich, omarcheye_fitter("rich")),
            ("iris-disk-rich (omarcheye.model)", iris_features("iris-disk-rich"), omarcheye_fitter("rich"))]
    for name in names:
        store, size, how, kind, opts, *extra = VARIANTS[name]
        runs.append((name, feature_loader(store, size, how, kind, *extra), appearance(npix_of(store, size, kind), **opts)))
    out = []
    for name, feats, fitter in runs:
        out.append(evaluate(f"{name} [recentred]", feats, fitter, protocols=protocols, workers=workers, verbose=False))
    print("| Features | " + " | ".join(protocols) + " |")
    print("|---" * (len(protocols) + 1) + "|")
    for r in out:
        print(f"| {r['name']} | " + " | ".join(f"{r[p]['pct']:.1f}%" if p in r else "—" for p in protocols) + " |")
    return out


def timing(n: int = 200, calib: int = 450) -> None:
    """Milliseconds per frame of the live parts, on MPIIFaceGaze frames scaled
    by 1.5 (the face about 400 px wide, as from the phone at 1080p): both
    eyes' patches, their HOG, the disk-refined iris centres, and one
    prediction of the appearance model fitted on `calib` frames."""
    d = load("p00")
    rows = np.flatnonzero(d["ok"])[:n]
    frames = [cv2.resize(cv2.imread(str(image_path("p00", str(d["files"][i])))), None, fx=1.5, fy=1.5)
              for i in rows]
    pts = [d["points"][i].astype(float) * 1.5 for i in rows]

    def per_frame(fn):
        t = time.perf_counter()
        for f, p in zip(frames, pts):
            out = fn(f, p)
        return 1000 * (time.perf_counter() - t) / len(frames), out

    t_patch, _ = per_frame(lambda f, p: ep.eye_patches(f, p))
    t_desc, desc = per_frame(lambda f, p: ep.eye_descriptor(f, p))
    t_disk, _ = per_frame(lambda f, p: ep.refine_points(f, p, "disk"))
    t_limbus, _ = per_frame(lambda f, p: ep.refine_points(f, p, "limbus"))
    t_timm, _ = per_frame(lambda f, p: ep.refine_points(f, p, "timm"))
    rng = np.random.default_rng(0)
    X = np.hstack([rng.random((calib, len(desc))), rng.normal(size=(calib, 25))])
    Y = rng.uniform(0, 300, (calib, 2))
    t = time.perf_counter()
    predict = appearance(len(desc), eye=True)(X, Y, {})
    t_fit = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(200):
        predict(X[i % calib: i % calib + 1])
    t_pred = 1000 * (time.perf_counter() - t) / 200
    print(f"per frame at 1080p scale: patches {t_patch:.2f} ms, patches + HOG {t_desc:.2f} ms, "
          f"iris refinement (both eyes) disk {t_disk:.2f} ms, limbus {t_limbus:.2f} ms, timm {t_timm:.2f} ms; "
          f"appearance model on {calib} frames: fit {t_fit:.2f} s, predict {t_pred:.3f} ms")


def show(subject: str = "p00", n: int = 12, path: str = "patches.png") -> None:
    """Montage: rows of images, columns g60 A, g60 B, z60 A, z60 B, equalised."""
    with np.load(OUT / f"{subject}.npz") as z:
        ok = np.flatnonzero(z["ok"])[:: max(1, len(np.flatnonzero(z["ok"])) // n)][:n]
        rows = []
        for i in ok:
            tiles = [ep.normalise(z[k][i, e], "equalize") for k in ("g60", "z60") for e in range(2)]
            rows.append(np.hstack(tiles))
    cv2.imwrite(path, cv2.resize(np.vstack(rows), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("extract", "evaluate", "write", "sweep", "recentred", "time", "show",
                                     "integrated"))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--protocols", nargs="*", default=list(PROTOCOLS))
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="patches.png")
    args = ap.parse_args(argv)
    if args.step == "extract":
        extract(workers=args.workers)
    elif args.step == "evaluate":
        res = run(args.variants, args.workers, args.protocols, args.max_days)
        print(table(res))
    elif args.step == "write":
        for name in args.variants if args.variants != list(VARIANTS) else ["patch-g60-hog+eye+disk"]:
            write_features(name)
    elif args.step == "sweep":
        sweep(names=args.variants if args.variants != list(VARIANTS) else ("patch-g60-hog+eye+disk",),
              workers=args.workers)
    elif args.step == "recentred":
        recentred(workers=args.workers)
    elif args.step == "time":
        timing()
    elif args.step == "integrated":
        score_integrated(args.workers)
    elif args.step == "show":
        show(path=args.out)


if __name__ == "__main__":
    main()
