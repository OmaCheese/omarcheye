"""MediaPipe Face Landmarker over every MPIIFaceGaze image -> landmarks/pXX.npz.

    uv run python -m bench.landmarks [--workers 4] [--subjects p00 p01 ...] [--force]

Same model file and options as omarcheye's tracker (one face, facial
transformation matrix and blendshapes on, CPU delegate), but IMAGE mode:
the dataset's images are independent stills, not a video.

Each pXX.npz has arrays aligned with the rows of pXX.txt (see README.md).
Rows where `ok` is False (no face, or a landmark within 1% of the image
border, i.e. the tracker's `cut_off`) are NaN in every per-image float
array computed from MediaPipe. `found` and `margin` keep the raw detection.
When all 15 subjects are written, landmarks/DONE is created.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from .mpii import CACHE, SUBJECTS, image_path, load_annotations, load_calibration

OUT = CACHE / "landmarks"
CHUNK = 40

_landmarker = None
_mp = None


def _init_worker() -> None:
    global _landmarker, _mp
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from omarcheye.config import MODEL_PATH
    from omarcheye.tracker import _import_mediapipe

    _mp, BaseOptions, vision = _import_mediapipe()
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH), delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
        output_face_blendshapes=True,
    )
    _landmarker = vision.FaceLandmarker.create_from_options(options)


def _process(job: tuple[str, list[tuple[int, str]]]) -> list[tuple]:
    """Landmarks and the tracker's features for a chunk of one subject's images."""
    import cv2

    from omarcheye.tracker import EDGE, features

    subject, items = job
    out = []
    for i, file in items:
        frame = cv2.imread(str(image_path(subject, file)))
        if frame is None:
            out.append((i, None))
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _landmarker.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            out.append((i, None))
            continue
        # As FaceTracker.process does.
        points = np.array([(q.x * w, q.y * h) for q in result.face_landmarks[0]])
        blend = {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else None
        matrix = np.asarray(result.facial_transformation_matrixes[0], float)
        sample = features(points, matrix, 0.0, blend)
        lo, hi = points.min(0), points.max(0)
        margin = float(min(lo[0] / w, lo[1] / h, 1 - hi[0] / w, 1 - hi[1] / h))
        names = [c.category_name for c in result.face_blendshapes[0]] if result.face_blendshapes else []
        scores = [c.score for c in result.face_blendshapes[0]] if result.face_blendshapes else []
        out.append((i, dict(points=points, matrix=matrix, names=names, blend=scores, feat=sample.feat,
                            rich=sample.rich, pose=sample.pose, openness=sample.openness, margin=margin,
                            ok=margin >= EDGE, size=(w, h))))
    return out


def run_subject(pool, subject: str) -> dict:
    ann = load_annotations(subject)
    cal = load_calibration(subject)
    n = len(ann)
    items = list(enumerate(ann.files))
    jobs = [(subject, items[k:k + CHUNK]) for k in range(0, n, CHUNK)]
    found = np.zeros(n, bool)
    ok = np.zeros(n, bool)
    margin = np.full(n, np.nan)
    image_size = np.zeros((n, 2), np.int32)
    points = np.full((n, 478, 2), np.nan, np.float32)
    matrix = np.full((n, 4, 4), np.nan, np.float32)
    blend = np.full((n, 52), np.nan, np.float32)
    feat = np.full((n, 7), np.nan)
    rich = np.full((n, 21), np.nan)
    pose = np.full((n, 16), np.nan)
    openness = np.full(n, np.nan)
    blend_names = None
    t0 = time.monotonic()
    done = 0
    for res in pool.imap_unordered(_process, jobs):
        for i, r in res:
            if r is None:
                continue
            found[i] = True
            margin[i] = r["margin"]
            image_size[i] = r["size"]
            if blend_names is None and r["names"]:
                blend_names = np.array(r["names"])
            if not r["ok"]:
                continue
            assert blend_names is not None and list(blend_names) == r["names"]
            ok[i] = True
            points[i] = r["points"]
            matrix[i] = r["matrix"]
            blend[i] = r["blend"]
            feat[i] = r["feat"]
            rich[i] = r["rich"]
            pose[i] = r["pose"]
            openness[i] = r["openness"]
        done += len(res)
        if done % (CHUNK * 10) < CHUNK or done == n:
            dt = time.monotonic() - t0
            print(f"  {subject}: {done}/{n} images, {dt:.0f} s, {done / dt:.1f} img/s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{subject}.tmp.npz"
    np.savez(
        tmp,
        files=ann.files, day=ann.day, ok=ok, found=found, margin=margin, image_size=image_size,
        points=points, matrix=matrix, blend=blend,
        blend_names=blend_names if blend_names is not None else np.array([], str),
        feat=feat, rich=rich, pose=pose, openness=openness,
        target_px=ann.target_px, screen_px=cal.screen_px, screen_mm=cal.screen_mm,
        face_center=ann.face_center, gaze_target=ann.gaze_target,
        head_rvec=ann.head_rvec, head_tvec=ann.head_tvec,
    )
    tmp.rename(OUT / f"{subject}.npz")
    return dict(n=n, found=int(found.sum()), ok=int(ok.sum()), seconds=time.monotonic() - t0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subjects", nargs="*", default=list(SUBJECTS))
    ap.add_argument("--force", action="store_true", help="redo subjects already written")
    args = ap.parse_args(argv)
    workers = min(args.workers, 4)
    todo = [s for s in args.subjects if args.force or not (OUT / f"{s}.npz").exists()]
    print(f"landmarks: {len(todo)} subjects to do, {workers} workers -> {OUT}", flush=True)
    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for s in todo:
            st = run_subject(pool, s)
            print(f"{s}: {st['n']} images, face found {st['found']}, ok {st['ok']} "
                  f"({st['n'] - st['ok']} unusable), {st['seconds']:.0f} s; total {time.monotonic() - t0:.0f} s",
                  flush=True)
    if all((OUT / f"{s}.npz").exists() for s in SUBJECTS):
        (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        print(f"all {len(SUBJECTS)} subjects written; created {OUT / 'DONE'}", flush=True)
    print(f"landmarks: {time.monotonic() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
