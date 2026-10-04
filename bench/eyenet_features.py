"""Eye-crop gaze network (omarcheye.eyenet) on MPIIFaceGaze.

    uv run python -m bench.eyenet_features             # features/eyenet/pXX.npz (4 processes, ~3 min)
    uv run python -m bench.eyenet_features --check     # angular error against the truth, no calibration
    uv run python -m bench.eyenet_features --evaluate  # score feature combinations with bench.evaluate

Each pXX.npz is aligned with the rows of pXX.txt (and landmarks/pXX.npz):
  X      (n, 4) omarcheye.eyenet.FEATURES (flip-averaged, MediaPipe's focal length); NaN where it failed
  sight  (n, 3) the network's gaze, roll undone: x right, y up, z along the sight line away from the camera
  mid    (n, 2) the four eye corners' mean, pixels
  head   (n, 3) yaw, pitch, roll fed to the network (degrees, Open Model Zoo's convention)
  size   (2,)   image width, height
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np

from .mpii import CACHE, SUBJECTS, image_path, load_calibration

OUT = CACHE / "features/eyenet"
LANDMARKS = CACHE / "landmarks"


def extract(subject: str) -> str:
    import cv2

    cv2.setNumThreads(1)
    from omarcheye.eyenet import FEATURES, EyeNet

    t0 = time.monotonic()
    net = EyeNet(min_open=0.0)  # MPIIFaceGaze has no blinks worth skipping; keep every image
    with np.load(LANDMARKS / f"{subject}.npz") as z:
        files, found, points, matrix = z["files"], z["found"], z["points"], z["matrix"]
    n = len(files)
    X = np.full((n, len(FEATURES)), np.nan)
    sight, mid, head = np.full((n, 3), np.nan), np.full((n, 2), np.nan), np.full((n, 3), np.nan)
    size = (0, 0)
    for i in np.flatnonzero(found):
        img = cv2.imread(str(image_path(subject, str(files[i]))))
        size = img.shape[1::-1]
        f = net(img, points[i], matrix[i])
        if f is None:
            continue
        X[i] = f
        sight[i] = net.last["sight"]
        head[i] = net.last["head"]
        mid[i] = points[i][[33, 133, 362, 263]].mean(0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{subject}.npz", X=X, sight=sight, mid=mid, head=head, size=np.array(size))
    return f"{subject}: {np.isfinite(X).all(1).sum()}/{n} in {time.monotonic() - t0:.0f} s"


def load_eyenet(subject: str) -> dict:
    with np.load(OUT / f"{subject}.npz") as z:
        return {k: z[k] for k in z.files}


def check() -> dict:
    """Angle between the network's gaze (camera axes through the true focal
    length) and the truth: from the eyes (the corners' mean, at the face
    centre's depth) to the gaze target. Images where MediaPipe found a whole face."""
    from omarcheye.eyenet import sight_frame

    from .evaluate import load

    per, every = {}, []
    for s in SUBJECTS:
        d, e = load(s), load_eyenet(s)
        kinv = np.linalg.inv(load_calibration(s).camera)
        errs = []
        for i in np.flatnonzero(d["ok"] & np.isfinite(e["sight"]).all(1)):
            ray = kinv @ np.array([*e["mid"][i], 1.0])
            truth = d["gaze_target"][i] - ray / ray[2] * d["face_center"][i, 2]
            g = sight_frame(ray) @ e["sight"][i]
            errs.append(np.degrees(np.arccos(np.clip(g @ truth / np.linalg.norm(truth), -1, 1))))
        per[s] = float(np.mean(errs))
        every += errs
    out = {"images": len(every), "mean": float(np.mean(every)), "median": float(np.median(every)),
           "per_subject": per}
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", action="store_true", help="angular error with no calibration")
    ap.add_argument("--evaluate", action="store_true", help="score feature combinations")
    ap.add_argument("subjects", nargs="*", default=list(SUBJECTS))
    args = ap.parse_args(argv)
    if args.check:
        check()
    if args.evaluate:
        from .evaluate import RESULTS, evaluate, load, ridge, slug, table

        X = lambda s: load_eyenet(s)["X"]  # noqa: E731
        head = lambda s: load(s)["feat"][:, 2:]  # noqa: E731  MediaPipe's yaw, pitch, hx, hy, distance
        combos = [
            ("eyenet: gaze angles", lambda s: X(s)[:, :2]),
            ("eyenet: hit point", lambda s: X(s)[:, 2:]),
            ("eyenet: angles + hit", X),
            ("eyenet: angles + head", lambda s: np.hstack([X(s)[:, :2], head(s)])),
            ("eyenet: hit + head", lambda s: np.hstack([X(s)[:, 2:], head(s)])),
            ("eyenet: hit + rich", lambda s: np.hstack([X(s)[:, 2:], load(s)["rich"]])),
            ("eyenet: angles + hit + rich", lambda s: np.hstack([X(s), load(s)["rich"]])),
        ]
        results = [evaluate(name, f, ridge(), workers=args.workers, verbose=False) for name, f in combos]
        for name in ("rich, plain ridge", "rich (omarcheye.model)", "geometric (omarcheye.geometry)"):
            for path in (RESULTS / f"{slug(name)}.json", RESULTS / f"{name}.json"):
                if path.exists():
                    results.append(json.loads(path.read_text()))
                    break
        print(table(results))
    if not (args.check or args.evaluate):
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for line in pool.imap_unordered(extract, args.subjects):
                print(line, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
