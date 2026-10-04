"""Evaluation protocol for gaze features on MPIIFaceGaze (see bench/README.md).

    from bench.evaluate import evaluate, load
    res = evaluate("my-features", lambda s: my_features(s))   # (n, d) per subject, NaN rows skipped
    print(res["other30"]["mm"], res["other30"]["deg"])

    uv run python -m bench.evaluate --baseline   # today's features: basic, rich, geometric
    uv run python -m bench.evaluate --table      # every saved result, as one markdown table
    uv run python -m bench.evaluate --table --units mm --protocols other30 other30+r5 --match rich

load_features(subject) returns an (n, d) float array aligned with the rows of
pXX.txt (and of landmarks/pXX.npz). Rows with any NaN are skipped, and by
default so are rows where MediaPipe found no whole face (`ok` False), so that
every feature set is scored on the same images.

fitter(X, Y, info) -> predict: X (k, d) calibration features, Y (k, 2) their
targets in mm on the screen (from the top-left corner, x right, y down), info
a dict with subject, screen_mm, screen_px. It returns a function mapping
(m, d) features to (m, 2) mm. The default is `ridge()`.

Protocols, per subject, then averaged over subjects (each subject counts once):
  same      within each day with >= 50 usable images, 5-fold random CV; the
            subject's error is the mean over all its test images
  same30    calibrate on 30 random images of one day, test on the rest of that
            day; repeated with each day with >= 50 usable images (a diagnostic:
            same calibration size as other30, but no change of sitting)
  other30   THE MAIN NUMBER. Calibrate on 30 random images of one day, test on
            every usable image of all other days; repeated with each day with
            >= 30 usable images as the calibration day
  other100  the same with 100 images (days with >= 100 usable images)
  otherK+rR other sitting + recentre: as otherK, then on each test day R
            images (seeded, the first R of RECENTRE_POOL = 5 set aside per test
            day and never scored) give a constant offset, the per-axis median of
            true - predicted, added to every prediction of that day; the day's
            other images are scored (test days with >= 10 usable images). Like
            `omarcheye recentre` (one dot) plus drift.py's learned shift.
            Default: R = 1, 3, 5 for K = 30, 100.
  otherK+aR the same with a per-day affine map, ridge-regularised towards the
            identity (AFFINE_LAMBDA), instead of the offset; on request only.
  otherK+r0 added with any recentre protocol: otherK on the recentre test images.
For sameK/otherK the subject's error is the mean over calibration days of the
mean error over that day's test images; the K images of a calibration day are
the same for sameK and otherK.
Errors: Euclidean mm on the screen; degrees = the angle at the face centre
(the dataset's face_center) between the rays to the true and the predicted
point, both placed in camera space with the subject's monitor pose; and % of
the screen width. Predictions are clamped to the screen rectangle (as the live
pipeline clamps points just outside it); `offscreen` is the share of
predictions more than 15% of the screen outside it before clamping (the live
pipeline would treat those as looking away), and NaN predictions count as the
screen centre. All randomness is seeded from (seed, subject, day, K).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
import warnings
from collections.abc import Callable
from functools import lru_cache

import numpy as np

from .mpii import CACHE, SUBJECTS, load_calibration

LANDMARKS = CACHE / "landmarks"
RESULTS = CACHE / "results"
PROTOCOLS = ("same", "same30", "other30", "other30+r1", "other30+r3", "other30+r5",
             "other100", "other100+r1", "other100+r3", "other100+r5")
# sameK, otherK, and otherK+rR / otherK+aR: other sitting, recentred on R images of each test day
PROTO_RE = re.compile(r"(same|other)(\d+)(?:\+([ra])(\d+))?")
LABELS = {"same": "Same sitting, 5-fold", "same30": "Same sitting, K=30",
          "other30": "Other sitting, K=30", "other100": "Other sitting, K=100"}
RECENTRE_POOL = 5  # images per test day set aside for recentring (R of them used), never scored
RECENTRE_TEST_MIN = 5  # a test day needs this many images left to score
AFFINE_LAMBDA = 100.0**2  # mm² per image: affine recentre's pull towards the identity (tried 20-400 mm on rich, flat 50-200)
SAME_MIN = 50  # a day needs this many usable images for the "same sitting" protocol
SAME_TEST_MIN = 20  # sameK: a day needs K + this many usable images
FOLDS = 5
OFFSCREEN = 0.15

Predict = Callable[[np.ndarray], np.ndarray]
Fitter = Callable[[np.ndarray, np.ndarray, dict], Predict]


@lru_cache(maxsize=None)
def load(subject: str) -> dict:
    """landmarks/pXX.npz as a dict of arrays (cached; don't modify them)."""
    with np.load(LANDMARKS / f"{subject}.npz") as z:
        return {k: z[k] for k in z.files}


def targets_mm(subject: str) -> np.ndarray:
    d = load(subject)
    return d["target_px"] * (d["screen_mm"] / d["screen_px"])


# ---------------------------------------------------------------- fitters

ALPHAS = np.logspace(-6, 2, 17)  # ridge strength per sample, on standardised features


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, s = x.mean(0), x.std(0)
    s[s < 1e-9] = 1.0
    return m, s


def _quad(z: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(z.shape[1])
    return np.column_stack([z, z[:, i] * z[:, j]])


def ridge(quadratic: bool = False, alphas: np.ndarray = ALPHAS) -> Fitter:
    """Standardised ridge regression, the strength picked by leave-one-out on the
    calibration set (closed form; criterion: mean Euclidean error). With
    `quadratic`, all squares and pairwise products of the standardised features
    are added (d(d+3)/2 terms; keep d small)."""

    def fit(X: np.ndarray, Y: np.ndarray, info: dict) -> Predict:
        m1, s1 = _standardise(X)
        z = (X - m1) / s1
        if quadratic:
            z = _quad(z)
        m2, s2 = _standardise(z)
        z = (z - m2) / s2
        n = len(z)
        ym = Y.mean(0)
        u, sv, vt = np.linalg.svd(z, full_matrices=False)
        uty = u.T @ (Y - ym)
        best = None
        for a in alphas:
            lam = a * n
            shrink = sv**2 / (sv**2 + lam)
            fitted = u @ (shrink[:, None] * uty)
            h = (u**2) @ shrink + 1.0 / n
            loo = (Y - ym - fitted) / np.maximum(1 - h, 1e-6)[:, None]
            err = np.hypot(loo[:, 0], loo[:, 1]).mean()
            if best is None or err < best[0]:
                best = (err, lam)
        lam = best[1]
        coef = vt.T @ ((sv / (sv**2 + lam))[:, None] * uty)

        def predict(Xt: np.ndarray) -> np.ndarray:
            zt = (Xt - m1) / s1
            if quadratic:
                zt = _quad(zt)
            return ((zt - m2) / s2) @ coef + ym

        return predict

    return fit


def omarcheye_fitter(kind: str) -> Fitter:
    """omarcheye.model.fit as calibration runs it ('basic' or 'rich'): quadratic
    eye terms, head spreads floored, leave-one-group-out over its own lambda grid
    (one group per image here), live features clamped to the calibration's range."""
    from omarcheye import model

    def fit(X: np.ndarray, Y: np.ndarray, info: dict) -> Predict:
        mm = np.asarray(info["screen_mm"], float)
        aspect = info["screen_px"][1] / info["screen_px"][0]
        m = model.fit(X, Y / mm, np.arange(len(X)), aspect, "", 0.0, kind=kind)
        return lambda Xt: m.predict_data({"feats": Xt, "rich": Xt}) * mm

    return fit


def geometric_fitter() -> Fitter:
    """omarcheye.geometry.fit (both mirror options, four sign starts). X is
    rich (21) then pose (16). Its leave-one-dot-out error isn't needed here, so
    score=set() skips it; the fitted parameters are the same."""
    from omarcheye import geometry

    def fit(X: np.ndarray, Y: np.ndarray, info: dict) -> Predict:
        mm = np.asarray(info["screen_mm"], float)
        data = {"rich": X[:, :21], "pose": X[:, 21:37], "targets": Y / mm}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # mean of no scored groups
            m = geometry.fit(data, np.arange(len(X)), (float(mm[0]), float(mm[1])), "", 0.0, score=set())
        return lambda Xt: m.predict_data({"rich": Xt[:, :21], "pose": Xt[:, 21:37]}) * mm

    return fit


# ---------------------------------------------------------------- protocol


def _errors(subject: str, pred: np.ndarray, rows: np.ndarray) -> dict:
    """Per-image errors of predictions (mm on the screen) for the given rows."""
    d = load(subject)
    cal = load_calibration(subject)
    mm = d["screen_mm"]
    pred = np.asarray(pred, float).copy()
    bad = ~np.isfinite(pred).all(1)
    pred[bad] = mm / 2
    margin = OFFSCREEN * mm
    off = ((pred < -margin) | (pred > mm + margin)).any(1) & ~bad
    pred = np.clip(pred, 0, mm)
    true = targets_mm(subject)[rows]
    e_mm = np.hypot(*(pred - true).T)
    fc = d["face_center"][rows]
    a = d["gaze_target"][rows] - fc
    b = cal.screen_to_camera(pred) - fc
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    e_deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return {"mm": e_mm, "deg": e_deg, "pct": 100 * e_mm / mm[0], "nan": bad, "off": off}


def _merge(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _subject(subject: str, load_features, fitter: Fitter, cfg: dict) -> dict:
    t0 = time.monotonic()
    d = load(subject)
    n = len(d["day"])
    X = np.asarray(load_features(subject), float)
    if X.ndim != 2 or len(X) != n:
        raise ValueError(f"{subject}: features must be (n={n}, d), got {X.shape}")
    usable = np.isfinite(X).all(1)
    if cfg["mask_ok"]:
        usable &= d["ok"]
    Y = targets_mm(subject)
    info = {"subject": subject, "screen_mm": d["screen_mm"], "screen_px": d["screen_px"]}
    sid = int(subject[1:])
    days = np.unique(d["day"])
    seed = cfg["seed"]
    out = {"usable": int(usable.sum()), "rows": n}

    def run(train: np.ndarray, test: np.ndarray) -> dict:
        predict = fitter(X[train], Y[train], info)
        return _errors(subject, predict(X[test]), test)

    if "same" in cfg["protocols"]:
        parts, fits = [], 0
        eligible = [day for day in days if usable[d["day"] == day].sum() >= SAME_MIN]
        if cfg["max_days"] and len(eligible) > cfg["max_days"]:
            pick = np.random.default_rng([seed, sid, 999]).choice(len(eligible), cfg["max_days"], replace=False)
            eligible = [eligible[i] for i in sorted(pick)]
        for day in eligible:
            rows = np.flatnonzero(d["day"] == day)
            fold = np.empty(len(rows), int)
            fold[np.random.default_rng([seed, sid, int(day), 0]).permutation(len(rows))] = np.arange(len(rows)) % FOLDS
            for f in range(FOLDS):
                train = rows[(fold != f) & usable[rows]]
                test = rows[(fold == f) & usable[rows]]
                if len(test):
                    parts.append(run(train, test))
                    fits += 1
        if parts:
            e = _merge(parts)
            out["same"] = _summary(e, fits=fits, days=len(eligible))

    # sameK and otherK (with any recentre variants): one fit per calibration day
    ks = sorted({(m[1], int(m[2])) for m in map(PROTO_RE.fullmatch, cfg["protocols"]) if m})
    for kind, k in ks:
        other = kind == "other"
        wanted = [p for p in cfg["protocols"] if (m := PROTO_RE.fullmatch(p)) and (m[1], int(m[2])) == (kind, k)]
        variants = sorted({(m[3], int(m[4])) for m in map(PROTO_RE.fullmatch, wanted) if m[3]})
        if variants:  # the same test images without recentring, for reference
            variants = [("r", 0)] + [v for v in variants if v != ("r", 0)]
            wanted = list(dict.fromkeys(wanted + [f"{kind}{k}+r0"]))
            pool_n = max(RECENTRE_POOL, max(r for _, r in variants))
            rc, rest = _recentre_split(d["day"], usable, days, seed, sid, pool_n)
        need = k if other else k + SAME_TEST_MIN
        eligible = [day for day in days if usable[d["day"] == day].sum() >= need]
        if cfg["max_days"] and len(eligible) > cfg["max_days"]:
            pick = np.random.default_rng([seed, sid, k, 999]).choice(len(eligible), cfg["max_days"], replace=False)
            eligible = [eligible[i] for i in sorted(pick)]
        per_day = {p: [] for p in wanted}
        for day in eligible:
            rows = np.flatnonzero(d["day"] == day)
            perm = np.random.default_rng([seed, sid, int(day), k]).permutation(rows)
            train = perm[usable[perm]][:k]
            if other:
                test = np.flatnonzero(usable & (d["day"] != day))
            else:
                test = np.setdiff1d(rows[usable[rows]], train)
            predict = fitter(X[train], Y[train], info)
            pred = np.asarray(predict(X[test]), float)
            if f"{kind}{k}" in per_day:
                per_day[f"{kind}{k}"].append(_errors(subject, pred, test))
            if variants and other:
                pos = np.full(n, -1)
                pos[test] = np.arange(len(test))
                for how, r in variants:
                    parts = []
                    for td in rc:
                        if td == day:
                            continue
                        cal_rows, score_rows = rc[td][:r], rest[td]
                        corrected = _recentre(pred[pos[score_rows]], pred[pos[cal_rows]], Y[cal_rows], how)
                        parts.append(_errors(subject, corrected, score_rows))
                    if parts:
                        per_day[f"{kind}{k}+{how}{r}"].append(_merge(parts))
        for proto, got in per_day.items():
            if got:
                # each calibration day counts once: average the per-day means
                out[proto] = {key: float(np.mean([_summary(p)[key] for p in got]))
                              for key in ("mm", "deg", "pct", "nan", "offscreen")}
                out[proto].update(fits=len(got), days=len(got), n_test=int(np.mean([len(p["mm"]) for p in got])))
    out["seconds"] = time.monotonic() - t0
    return out


def _recentre_split(day: np.ndarray, usable: np.ndarray, days, seed: int, sid: int, pool_n: int):
    """Per test day: `pool_n` seeded random usable images to recentre on (the
    first R of them for R images) and the day's other usable images, which are
    scored. Only days with at least RECENTRE_TEST_MIN images left to score."""
    rc, rest = {}, {}
    for td in days:
        rows = np.flatnonzero((day == td) & usable)
        if len(rows) < pool_n + RECENTRE_TEST_MIN:
            continue
        perm = np.random.default_rng([seed, sid, int(td), 7]).permutation(rows)
        rc[td], rest[td] = perm[:pool_n], np.sort(perm[pool_n:])
    return rc, rest


def _recentre(pred: np.ndarray, cal_pred: np.ndarray, cal_true: np.ndarray, how: str) -> np.ndarray:
    """Predictions corrected from a few images of the same day: "r", a constant
    offset (the median of true - predicted per axis, as `omarcheye recentre`
    and drift.py's learned shift); "a", an affine map A p + b fitted by least
    squares with A pulled towards the identity (ridge, AFFINE_LAMBDA)."""
    ok = np.isfinite(cal_pred).all(1)
    if not ok.any():
        return pred
    p, t = cal_pred[ok], cal_true[ok]
    if how == "r" or len(p) < 2:
        return pred + np.median(t - p, 0)
    pm, tm = p.mean(0), t.mean(0)
    pc, tc = p - pm, t - tm
    lam = AFFINE_LAMBDA * len(p) * np.eye(2)
    a = (tc.T @ pc + lam) @ np.linalg.inv(pc.T @ pc + lam)
    return (pred - pm) @ a.T + tm


def _summary(e: dict, **extra) -> dict:
    return {"mm": float(e["mm"].mean()), "deg": float(e["deg"].mean()), "pct": float(e["pct"].mean()),
            "nan": float(e["nan"].mean()), "offscreen": float(e["off"].mean()), "n_test": len(e["mm"]), **extra}


_JOB = None


def _job(subject: str) -> dict:
    load_features, fitter, cfg = _JOB
    return _subject(subject, load_features, fitter, cfg)


def evaluate(name: str, load_features: Callable[[str], np.ndarray], fitter: Fitter | None = None, *,
             subjects=SUBJECTS, protocols=PROTOCOLS, seed: int = 0, mask_ok: bool = True,
             max_days: int | None = None, workers: int = 1, save: bool = True, update: bool = False,
             verbose: bool = True) -> dict:
    """Score a feature set (and fitter) with the protocols above.

    Returns {"name", "fitter", "coverage", "seconds", "per_subject": {s: ...},
    and per protocol {"mm", "deg", "pct", "offscreen", "nan", "subjects"}}:
    means over the subjects that had a qualifying day. With `max_days`, at
    most that many days per subject and protocol are used (a seeded random
    choice). `workers` > 1 runs subjects in forked processes (load_features
    and fitter needn't be picklable, but avoid it after loading threaded
    libraries such as MediaPipe or torch in this process). With `save`, the
    result goes to ~/.cache/omarcheye/bench/results/<name>.json; with `update`,
    the protocols run now are merged into a result saved earlier under that
    name (its other protocols are kept). Recentre protocols (otherK+rR,
    otherK+aR) also add otherK+r0: the same test images, not recentred."""
    global _JOB
    fitter = fitter or ridge()
    cfg = {"protocols": tuple(protocols), "seed": seed, "mask_ok": mask_ok, "max_days": max_days}
    t0 = time.monotonic()
    subjects = list(subjects)
    if workers > 1:
        _JOB = (load_features, fitter, cfg)
        with mp.get_context("fork").Pool(min(workers, len(subjects))) as pool:
            per = dict(zip(subjects, pool.map(_job, subjects)))
        _JOB = None
    else:
        per = {}
        for s in subjects:
            per[s] = _subject(s, load_features, fitter, cfg)
            if verbose:
                print(f"  {name} {s}: " + ", ".join(f"{p} {per[s][p]['mm']:.1f} mm" for p in protocols if p in per[s])
                      + f" ({per[s]['seconds']:.1f} s)", file=sys.stderr, flush=True)
    res = {"name": name, "fitter": getattr(fitter, "__qualname__", repr(fitter)), "seed": seed,
           "max_days": max_days, "mask_ok": mask_ok,
           "coverage": sum(p["usable"] for p in per.values()) / sum(p["rows"] for p in per.values()),
           "seconds": time.monotonic() - t0, "per_subject": per}
    found = list(protocols) + [p for p in dict.fromkeys(k for v in per.values() for k in v)
                               if PROTO_RE.fullmatch(p) and p not in protocols]
    for proto in found:
        got = [per[s][proto] for s in subjects if proto in per[s]]
        if not got:
            continue
        res[proto] = {k: float(np.mean([g[k] for g in got])) for k in ("mm", "deg", "pct", "offscreen", "nan")}
        res[proto]["subjects"] = len(got)
        res[proto]["fits"] = int(sum(g["fits"] for g in got))
    if verbose:
        print(table([res]), file=sys.stderr, flush=True)
    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"{slug(name)}.json"
        if update and path.exists():
            res = _merged(json.loads(path.read_text()), res)
        path.write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


def _merged(old: dict, new: dict) -> dict:
    """An earlier saved result with the protocols of a new run added or replaced."""
    out = dict(old)
    for k, v in new.items():
        if k == "per_subject":
            out[k] = {s: {**old.get(k, {}).get(s, {}), **v.get(s, {})} for s in {*old.get(k, {}), *v}}
        elif k not in ("seconds",):
            out[k] = v
    out["updated"] = sorted({*old.get("updated", []), *(p for p in new if p == "same" or PROTO_RE.fullmatch(p))})
    return out


def slug(name: str) -> str:
    """File name for a result: results/<slug>.json."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def label(p: str) -> str:
    if p in LABELS:
        return LABELS[p]
    m = PROTO_RE.fullmatch(p)
    if m and m[3]:
        how = "recentre" if m[3] == "r" else "affine recentre"
        return f"{LABELS.get(f'{m[1]}{m[2]}', m[1] + m[2])} + {how} R={m[4]}" if m[4] != "0" else \
            f"{LABELS.get(f'{m[1]}{m[2]}', m[1] + m[2])}, recentre test images"
    return p


UNITS = {"mm": "{:.1f} mm", "deg": "{:.2f}°", "pct": "{:.1f}%"}


def table(results: list[dict], units=("mm", "deg", "pct"), protocols=None) -> str:
    """Markdown table: mean error per protocol (mm on the screen / degrees / %
    of width, or the `units` given), for `protocols` (default: every protocol
    any result has, standard ones first; otherK+r0 only when asked for)."""
    if protocols is None:
        protocols = [p for p in PROTOCOLS if any(p in r for r in results)]
        protocols += sorted({p for r in results for p in r if PROTO_RE.fullmatch(p) and p not in protocols
                             and not p.endswith("+r0")}, key=_order)
        protocols.sort(key=_order)
    head = "| Features | " + " | ".join(label(p) for p in protocols) + " | Usable images |\n"
    head += "|---" * (len(protocols) + 2) + "|\n"
    rows = []
    for r in results:
        cells = []
        for p in protocols:
            if p in r:
                cells.append(" / ".join(UNITS[u].format(r[p][u]) for u in units))
            else:
                cells.append("—")
        rows.append(f"| {r['name']} | " + " | ".join(cells) + f" | {100 * r['coverage']:.1f}% |")
    return head + "\n".join(rows)


def _order(p: str):
    m = PROTO_RE.fullmatch(p)
    if not m:
        return (0, 0, 0, "", 0)
    return (1 if m[1] == "same" else 2, int(m[2]), 0 if not m[3] else 1, m[3] == "a", int(m[4] or 0))


# ---------------------------------------------------------------- baselines


def feat_none(s: str) -> np.ndarray:
    """No features: ridge() then predicts the calibration images' mean target."""
    return np.zeros((len(load(s)["day"]), 1))


def feat_basic(s: str) -> np.ndarray:
    return load(s)["feat"]


def feat_rich(s: str) -> np.ndarray:
    return load(s)["rich"]


def feat_geometric(s: str) -> np.ndarray:
    d = load(s)
    return np.hstack([d["rich"], d["pose"]])


def baselines(workers: int = 4, geo_max_days: int | None = None) -> list[dict]:
    out = []
    for name, feats, fitter, md in (
        ("none (calibration mean)", feat_none, ridge(), None),
        ("basic (omarcheye.model)", feat_basic, omarcheye_fitter("basic"), None),
        ("rich (omarcheye.model)", feat_rich, omarcheye_fitter("rich"), None),
        ("basic, plain ridge", feat_basic, ridge(), None),
        ("rich, plain ridge", feat_rich, ridge(), None),
        ("rich, quadratic ridge", feat_rich, ridge(quadratic=True), None),
        ("geometric (omarcheye.geometry)", feat_geometric, geometric_fitter(), geo_max_days),
    ):
        t0 = time.monotonic()
        r = evaluate(name, feats, fitter, workers=workers, max_days=md, verbose=False)
        print(f"{name}: {time.monotonic() - t0:.0f} s", file=sys.stderr, flush=True)
        out.append(r)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="MPIIFaceGaze gaze benchmark")
    ap.add_argument("--baseline", action="store_true", help="score today's features and print the table")
    ap.add_argument("--table", action="store_true", help="print every saved result as one table")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--geo-max-days", type=int, default=None,
                    help="at most this many days per subject and protocol for the geometric model")
    ap.add_argument("--units", nargs="+", choices=tuple(UNITS), default=list(UNITS),
                    help="--table: which units per cell (default: mm deg pct)")
    ap.add_argument("--protocols", nargs="+", help="--table: these columns, e.g. other30 other30+r5 other30+a5")
    ap.add_argument("--match", nargs="+", help="--table: only results whose name contains one of these")
    args = ap.parse_args(argv)
    if args.baseline:
        print(table(baselines(args.workers, args.geo_max_days)))
    if args.table:
        res = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
        if args.match:
            res = [r for r in res if any(m in r["name"] for m in args.match)]
        print(table(res, args.units, args.protocols))


if __name__ == "__main__":
    main()
