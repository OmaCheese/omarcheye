"""Route B: MediaPipe's iris centre refined on the full-resolution eye -> u, v.

    uv run python -m bench.iris refine     # refined centres of every image -> iris/pXX.npz
    uv run python -m bench.iris features   # tracker features from them -> features/iris-*/pXX.npz
    uv run python -m bench.iris evaluate   # score them with bench.evaluate, next to the baselines
    uv run python -m bench.iris time       # milliseconds per eye for each method
    uv run python -m bench.iris low        # MediaPipe on the image shrunk to 0.6, refinement on the original
    uv run python -m bench.iris low-features && uv run python -m bench.iris low-evaluate
    uv run python -m bench.iris combo      # MediaPipe's rich features plus the refined u, v (plain ridge)

Each refinement method (omarcheye.eyepatch) gets MediaPipe's landmarks and
the original image, moves the two iris centres, and the tracker's own
features() computes u, v (and the rest of "basic" and "rich") from the
moved centres, exactly as the live tracker would. An eye where a method
fails keeps MediaPipe's centre, as live.

iris/pXX.npz: for every row of pXX.txt, `seed` (n, 2, 2): MediaPipe's centres
of eye A and B in image pixels, `r` (n, 2): the iris radius in image pixels,
and one (n, 2, 2) array per variant with the refined centres (NaN: failed or
no face).

Result (other sitting, K=30, % of the screen width; baseline rich 17.3, basic
18.7): disk 16.8 rich / 20.3 basic, limbus 17.3 / 22.0, Timm & Barth 18.1 /
23.0. Every refinement makes u and v alone noisier than MediaPipe's; disk's
helps next to MediaPipe's eye-direction scores. With MediaPipe given the
image at 0.6 (as live, where the camera has ~1.5x the resolution MediaPipe
uses) its own iris did as well as at full size, and the gain from disk was
the same.
"""

from __future__ import annotations

import os

# One BLAS thread per process: the benchmark runs at most 4 processes.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import functools
import multiprocessing as mp
import sys
import time

import cv2
import numpy as np

from omarcheye import eyepatch as ep
from omarcheye.tracker import features

from .evaluate import PROTOCOLS, evaluate, load, omarcheye_fitter, ridge, table
from .mpii import CACHE, SUBJECTS, image_path

OUT = CACHE / "iris"
FEATURES = CACHE / "features"
CHUNK = 50

# Refinement variants: name -> function of an EyeCrop returning a crop point or None.
VARIANTS = {
    "timm": functools.partial(ep.timm, dark=1.0),
    "timm0": functools.partial(ep.timm, dark=0.0),
    "disk": ep.disk,
    "limbus": ep.limbus,
}


def _checked(c: ep.EyeCrop, p) -> np.ndarray | None:
    if p is None or not np.all(np.isfinite(p)) or np.hypot(*(p - c.seed)) > ep.MAX_MOVE * c.r:
        return None
    return p


def _refine(job: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, dict]:
    cv2.setNumThreads(1)
    subject, rows = job
    d = load(subject)
    out = {v: np.full((len(rows), 2, 2), np.nan) for v in VARIANTS}
    seed = np.full((len(rows), 2, 2), np.nan)
    rad = np.full((len(rows), 2), np.nan)
    for j, i in enumerate(rows):
        img = cv2.imread(str(image_path(subject, str(d["files"][i]))))
        pts = d["points"][i].astype(float)
        for e, eye in enumerate(ep.EYES):
            c = ep.eye_crop(img, pts, eye)
            seed[j, e] = pts[eye.iris]
            rad[j, e] = c.r / abs(np.linalg.det(c.m[:, :2])) ** 0.5
            for v, fn in VARIANTS.items():
                p = _checked(c, fn(c))
                if p is not None:
                    out[v][j, e] = ep.to_frame(c, p)
    out["seed"], out["r"] = seed, rad
    return subject, rows, out


def refine(subjects=SUBJECTS, workers: int = 4) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in subjects:
        rows = np.flatnonzero(load(s)["ok"])
        jobs += [(s, rows[k:k + CHUNK]) for k in range(0, len(rows), CHUNK)]
    acc = {s: {} for s in subjects}
    t0 = time.monotonic()
    with mp.get_context("fork").Pool(min(workers, 4)) as pool:
        for n, (s, rows, out) in enumerate(pool.imap_unordered(_refine, jobs), 1):
            nrows = len(load(s)["day"])
            for k, a in out.items():
                acc[s].setdefault(k, np.full((nrows, *a.shape[1:]), np.nan))[rows] = a
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} chunks, {time.monotonic() - t0:.0f} s", file=sys.stderr, flush=True)
    for s in subjects:
        np.savez(OUT / f"{s}.npz", **acc[s])
    print(f"refined {len(subjects)} subjects in {time.monotonic() - t0:.0f} s", file=sys.stderr)


def centres(subject: str, variant: str, blend: float = 1.0) -> np.ndarray:
    """(n, 2, 2) iris centres of a variant ("mediapipe", a VARIANTS name, or
    names joined by "+" for their mean), moved `blend` of the way from
    MediaPipe's; an eye where the variant failed keeps MediaPipe's centre."""
    with np.load(OUT / f"{subject}.npz") as z:
        seed = z["seed"]
        if variant == "mediapipe":
            return seed
        stack = np.stack([z[v] for v in variant.split("+")])
    with np.errstate(invalid="ignore"):
        ref = np.nanmean(stack, 0)
    ref = np.where(np.isfinite(ref), ref, seed)
    return seed + blend * (ref - seed)


def feature_arrays(subject: str, variant: str, blend: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """("basic" (n, 7), "rich" (n, 21)) from tracker.features with the refined centres; NaN rows where no face."""
    d = load(subject)
    names = list(d["blend_names"])
    cen = centres(subject, variant, blend)
    n = len(d["day"])
    basic, rich = np.full((n, 7), np.nan), np.full((n, 21), np.nan)
    for i in np.flatnonzero(d["ok"]):
        pts = d["points"][i].astype(float)
        pts[ep.EYE_A.iris], pts[ep.EYE_B.iris] = cen[i, 0], cen[i, 1]
        s = features(pts, d["matrix"][i].astype(float), 0.0, dict(zip(names, d["blend"][i])))
        basic[i], rich[i] = s.feat, s.rich
    return basic, rich


def write_features(variant: str, blend: float = 1.0, subjects=SUBJECTS) -> str:
    tag = f"iris-{variant}" + ("" if blend == 1.0 else f"-b{blend:g}")
    for s in subjects:
        basic, rich = feature_arrays(s, variant, blend)
        for kind, x in (("basic", basic), ("rich", rich)):
            (FEATURES / f"{tag}-{kind}").mkdir(parents=True, exist_ok=True)
            np.savez(FEATURES / f"{tag}-{kind}" / f"{s}.npz", X=x)
    return tag


def write_combo(variant: str, subjects=SUBJECTS, low: float | None = None) -> str:
    """MediaPipe's rich features (21) plus the refined uA, vA, uB, vB (4):
    both centres side by side, for bench.evaluate's plain ridge."""
    tag = f"iris-{variant}-plus-rich" if low is None else f"iris-low{low:g}-{variant}-plus-rich"
    src = f"iris-{variant}-rich" if low is None else f"iris-low{low:g}-{variant}-rich"
    base = None if low is None else f"iris-low{low:g}-mediapipe-rich"
    for s in subjects:
        rich = load(s)["rich"] if base is None else feature_loader(base)(s)
        ref = feature_loader(src)(s)[:, :4]
        (FEATURES / tag).mkdir(parents=True, exist_ok=True)
        np.savez(FEATURES / tag / f"{s}.npz", X=np.hstack([rich, ref]))
    return tag


def feature_loader(name: str):
    def load_features(s: str) -> np.ndarray:
        with np.load(FEATURES / name / f"{s}.npz") as z:
            return z["X"]
    return load_features


def score(tag: str, workers: int = 4, protocols=PROTOCOLS, plain: bool = False) -> list[dict]:
    """Score a feature tag's basic and rich sets with the fitters the baselines use."""
    out = []
    for kind in ("basic", "rich"):
        name = f"{tag}-{kind}"
        out.append(evaluate(f"{name} (omarcheye.model)", feature_loader(name), omarcheye_fitter(kind),
                            protocols=protocols, workers=workers, verbose=False))
        if plain:
            out.append(evaluate(f"{name}, plain ridge", feature_loader(name), ridge(),
                                protocols=protocols, workers=workers, verbose=False))
    return out


# ---------------------------------------------------------------- low resolution
#
# MPIIFaceGaze's faces are about 260 px wide, so MediaPipe (a 256-pixel face
# crop) already sees them at their native resolution and full-resolution
# refinement has nothing extra to work with. Live, the phone's 1080p frame
# shows the face about 400 px wide, 1.5x more than MediaPipe uses. To
# simulate that, MediaPipe runs on the image shrunk by LOW (its landmarks
# scaled back up), and the refinement works on the original image.

LOW = 0.6


def _low_dir(scale: float):
    return CACHE / f"iris-low{scale:g}"


def _process_low(job):
    from . import landmarks as lm
    from omarcheye.tracker import EDGE

    subject, rows, scale = job
    d = load(subject)
    names = list(d["blend_names"])
    n = len(rows)
    res = {"ok": np.zeros(n, bool), "points": np.full((n, 478, 2), np.nan, np.float32),
           "matrix": np.full((n, 4, 4), np.nan, np.float32), "blend": np.full((n, 52), np.nan, np.float32),
           "seed": np.full((n, 2, 2), np.nan), "r": np.full((n, 2), np.nan)}
    res.update({v: np.full((n, 2, 2), np.nan) for v in VARIANTS})
    for j, i in enumerate(rows):
        img = cv2.imread(str(image_path(subject, str(d["files"][i]))))
        h, w = img.shape[:2]
        small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        out = lm._landmarker.detect(lm._mp.Image(image_format=lm._mp.ImageFormat.SRGB,
                                                 data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        if not out.face_landmarks or not out.facial_transformation_matrixes or not out.face_blendshapes:
            continue
        pts = np.array([(q.x * w, q.y * h) for q in out.face_landmarks[0]])
        lo, hi = pts.min(0), pts.max(0)
        if min(lo[0] / w, lo[1] / h, 1 - hi[0] / w, 1 - hi[1] / h) < EDGE:
            continue
        blend = {c.category_name: c.score for c in out.face_blendshapes[0]}
        res["ok"][j] = True
        res["points"][j] = pts
        res["matrix"][j] = np.asarray(out.facial_transformation_matrixes[0], float)
        res["blend"][j] = [blend.get(k, np.nan) for k in names]
        for e, eye in enumerate(ep.EYES):
            c = ep.eye_crop(img, pts, eye)
            res["seed"][j, e] = pts[eye.iris]
            res["r"][j, e] = c.r / abs(np.linalg.det(c.m[:, :2])) ** 0.5
            for v, fn in VARIANTS.items():
                p = _checked(c, fn(c))
                if p is not None:
                    res[v][j, e] = ep.to_frame(c, p)
    return subject, rows, res


def refine_low(scale: float = LOW, subjects=SUBJECTS, workers: int = 4) -> None:
    """MediaPipe on the shrunk image, refinement on the original -> iris-low<scale>/pXX.npz."""
    from .landmarks import _init_worker

    out_dir = _low_dir(scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in subjects:
        rows = np.flatnonzero(load(s)["ok"])
        jobs += [(s, rows[k:k + CHUNK], scale) for k in range(0, len(rows), CHUNK)]
    acc = {s: {} for s in subjects}
    t0 = time.monotonic()
    with mp.get_context("spawn").Pool(min(workers, 4), initializer=_init_worker) as pool:
        for n, (s, rows, res) in enumerate(pool.imap_unordered(_process_low, jobs), 1):
            nrows = len(load(s)["day"])
            for k, a in res.items():
                fill = False if a.dtype == bool else np.nan
                acc[s].setdefault(k, np.full((nrows, *a.shape[1:]), fill, a.dtype))[rows] = a
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} chunks, {time.monotonic() - t0:.0f} s", file=sys.stderr, flush=True)
    for s in subjects:
        np.savez(out_dir / f"{s}.npz", **acc[s])
    print(f"low-resolution landmarks and refinement: {time.monotonic() - t0:.0f} s", file=sys.stderr)


def write_low_features(variant: str, scale: float = LOW, subjects=SUBJECTS) -> str:
    """Features from the shrunk image's landmarks, with the iris centres
    MediaPipe found there ("mediapipe") or refined on the original."""
    tag = f"iris-low{scale:g}-{variant}"
    for s in subjects:
        d = load(s)
        names = list(d["blend_names"])
        with np.load(_low_dir(scale) / f"{s}.npz") as z:
            low = {k: z[k] for k in z.files}
        seed = low["seed"]
        if variant == "mediapipe":
            cen = seed
        else:
            with np.errstate(invalid="ignore"):
                ref = np.nanmean(np.stack([low[v] for v in variant.split("+")]), 0)
            cen = np.where(np.isfinite(ref), ref, seed)
        n = len(d["day"])
        basic, rich = np.full((n, 7), np.nan), np.full((n, 21), np.nan)
        for i in np.flatnonzero(low["ok"]):
            pts = low["points"][i].astype(float)
            pts[ep.EYE_A.iris], pts[ep.EYE_B.iris] = cen[i, 0], cen[i, 1]
            smp = features(pts, low["matrix"][i].astype(float), 0.0, dict(zip(names, low["blend"][i])))
            basic[i], rich[i] = smp.feat, smp.rich
        for kind, x in (("basic", basic), ("rich", rich)):
            (FEATURES / f"{tag}-{kind}").mkdir(parents=True, exist_ok=True)
            np.savez(FEATURES / f"{tag}-{kind}" / f"{s}.npz", X=x)
    return tag


def timing(n: int = 200) -> None:
    """Milliseconds per eye: the crop, each method; and at 1080p (the image scaled by 1.5)."""
    d = load("p00")
    rows = np.flatnonzero(d["ok"])[:n]
    imgs = [cv2.imread(str(image_path("p00", str(d["files"][i])))) for i in rows]
    pts = [d["points"][i].astype(float) for i in rows]
    for scale in (1.0, 1.5):
        frames = imgs if scale == 1 else [cv2.resize(im, None, fx=scale, fy=scale) for im in imgs]
        crops = []
        t = time.perf_counter()
        for im, p in zip(frames, pts):
            crops += [ep.eye_crop(im, p * scale, eye) for eye in ep.EYES]
        res = {"crop": (time.perf_counter() - t) / len(crops)}
        for v, fn in VARIANTS.items():
            t = time.perf_counter()
            for c in crops:
                fn(c)
            res[v] = (time.perf_counter() - t) / len(crops)
        print(f"scale {scale}: " + ", ".join(f"{k} {1000 * v:.2f} ms" for k, v in res.items()) + " per eye")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("refine", "features", "evaluate", "time", "low", "low-features",
                                     "low-evaluate", "combo"))
    ap.add_argument("--low", action="store_true", help="combo: from the low-resolution run")
    ap.add_argument("--scale", type=float, default=LOW, help="image scale MediaPipe sees (low-*)")
    ap.add_argument("--variants", nargs="*", default=["timm", "timm0", "disk", "limbus", "disk+limbus",
                                                         "timm0+disk+limbus"])
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plain", action="store_true", help="also score with bench.evaluate's plain ridge")
    ap.add_argument("--protocols", nargs="*", default=list(PROTOCOLS))
    args = ap.parse_args(argv)
    if args.step == "refine":
        refine(workers=args.workers)
    elif args.step == "features":
        for v in args.variants:
            print(write_features(v, args.blend), file=sys.stderr, flush=True)
    elif args.step == "evaluate":
        res = []
        for v in args.variants:
            tag = f"iris-{v}" + ("" if args.blend == 1.0 else f"-b{args.blend:g}")
            res += score(tag, args.workers, args.protocols, plain=args.plain)
            print(table(res[-2 - 2 * args.plain:]), file=sys.stderr, flush=True)
        print(table(res))
    elif args.step == "time":
        timing()
    elif args.step == "combo":
        res = []
        for v in args.variants:
            tag = write_combo(v, low=args.scale if args.low else None)
            res.append(evaluate(f"{tag}, plain ridge", feature_loader(tag), ridge(), workers=args.workers,
                                verbose=False))
            print(table(res[-1:]), file=sys.stderr, flush=True)
        print(table(res))
    elif args.step == "low":
        refine_low(args.scale, workers=args.workers)
    elif args.step == "low-features":
        for v in ["mediapipe", *args.variants]:
            print(write_low_features(v, args.scale), file=sys.stderr, flush=True)
    elif args.step == "low-evaluate":
        res = []
        for v in ["mediapipe", *args.variants]:
            res += score(f"iris-low{args.scale:g}-{v}", args.workers, args.protocols, plain=args.plain)
            print(table(res[-2 - 2 * args.plain:]), file=sys.stderr, flush=True)
        print(table(res))


if __name__ == "__main__":
    main()
