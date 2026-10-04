"""Face-based comparison: MobileGaze (yakhyo/gaze-estimation, MIT), MobileOne-S0
trained on Gaze360 only, on MPIIFaceGaze.

    uv run python -m bench.mobilegaze_features             # features/mobilegaze/pXX.npz (4 processes)
    uv run python -m bench.mobilegaze_features --check     # angular error, no calibration
    uv run python -m bench.mobilegaze_features --evaluate  # score with bench.evaluate

Model: https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx
(~/.local/share/omarcheye/models/), run with OpenVINO's ONNX reader. Preprocessing as its
onnx_inference.py: the face box (RetinaFace there; here the box around MediaPipe's 478
points) resized to 448x448, RGB, ImageNet mean and deviation; the output is 90 bins of 4°
per angle, decoded as the softmax mean minus 180°. Its gaze_to_3d: camera coordinates
(x right, y down, z forward) = (-cos p sin y, -sin p, -cos p cos y).

pXX.npz: X (n, 4) = gaze yaw, pitch (radians, > 0 right, up, camera axes) and the
hit point on the camera's plane (cm, as omarcheye.eyenet); g (n, 3) the 3D gaze in
camera coordinates; NaN where it failed.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time

import numpy as np

from omarcheye.config import DATA_DIR

from .mpii import CACHE, SUBJECTS, image_path

MODEL = DATA_DIR / "models/mobileone_s0_gaze.onnx"
OUT = CACHE / "features/mobilegaze"
LANDMARKS = CACHE / "landmarks"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class MobileGaze:
    def __init__(self, threads: int = 1):
        from omarcheye.eyenet import _import_openvino

        core = _import_openvino().Core()
        net = core.compile_model(core.read_model(str(MODEL)), "CPU",
                                 {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"})
        self.request = net.create_infer_request()

    def __call__(self, frame: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        """Gaze in camera coordinates (x right, y down, z forward), unit vector."""
        import cv2

        lo, hi = np.floor(points.min(0)).astype(int), np.ceil(points.max(0)).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, frame.shape[1::-1])
        if (hi - lo).min() < 32:
            return None
        face = cv2.cvtColor(frame[lo[1]:hi[1], lo[0]:hi[0]], cv2.COLOR_BGR2RGB)
        x = (cv2.resize(face, (448, 448)).astype(np.float32) / 255 - MEAN) / STD
        self.request.infer({"input": x.transpose(2, 0, 1)[None]})
        ang = []
        for name in ("yaw", "pitch"):
            logits = np.asarray(self.request.get_tensor(name).data[0], float)
            p = np.exp(logits - logits.max())
            ang.append(math.radians((p / p.sum()) @ np.arange(90) * 4 - 180))
        y, p = ang
        return np.array([-math.cos(p) * math.sin(y), -math.sin(p), -math.cos(p) * math.cos(y)])


def extract(subject: str) -> str:
    import cv2

    cv2.setNumThreads(1)
    from omarcheye.eyenet import MP_VFOV

    t0 = time.monotonic()
    net = MobileGaze()
    with np.load(LANDMARKS / f"{subject}.npz") as z:
        files, found, points, matrix = z["files"], z["found"], z["points"], z["matrix"]
    n = len(files)
    X, G = np.full((n, 4), np.nan), np.full((n, 3), np.nan)
    for i in np.flatnonzero(found):
        img = cv2.imread(str(image_path(subject, str(files[i]))))
        p = points[i].astype(float)
        if not np.isfinite(p).all():
            continue
        c = net(img, p)
        if c is None or c[2] > -0.2:
            continue
        h, w = img.shape[:2]
        f = (h / 2) / math.tan(math.radians(MP_VFOV / 2))
        mid = p[[33, 133, 362, 263]].mean(0)
        eye = np.array([(mid[0] - w / 2) / f, (mid[1] - h / 2) / f, 1.0]) * max(-matrix[i][2, 3], 1.0)
        hit = eye[:2] - eye[2] / c[2] * c[:2]
        G[i] = c
        X[i] = (math.atan2(c[0], -c[2]), math.atan2(-c[1], math.hypot(c[0], c[2])), hit[0], -hit[1])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{subject}.npz", X=X, g=G)
    return f"{subject}: {np.isfinite(X).all(1).sum()}/{n} in {time.monotonic() - t0:.0f} s"


def check() -> dict:
    """Angle between the predicted gaze and the truth (face centre to gaze target)."""
    from .evaluate import load

    per, every = {}, []
    for s in SUBJECTS:
        d = load(s)
        with np.load(OUT / f"{s}.npz") as z:
            g = z["g"]
        ok = d["ok"] & np.isfinite(g).all(1)
        t = d["gaze_target"][ok] - d["face_center"][ok]
        e = np.degrees(np.arccos(np.clip(np.sum(g[ok] * t, 1) / np.linalg.norm(t, axis=1), -1, 1)))
        per[s] = float(e.mean())
        every.append(e)
    every = np.concatenate(every)
    out = {"images": len(every), "mean": float(every.mean()), "median": float(np.median(every)), "per_subject": per}
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("subjects", nargs="*", default=list(SUBJECTS))
    args = ap.parse_args(argv)
    if args.check:
        check()
    if args.evaluate:
        from .evaluate import evaluate, load, ridge, table

        X = lambda s: np.load(OUT / f"{s}.npz")["X"]  # noqa: E731
        combos = [
            ("mobilegaze: gaze angles", lambda s: X(s)[:, :2]),
            ("mobilegaze: hit point", lambda s: X(s)[:, 2:]),
            ("mobilegaze: hit + rich", lambda s: np.hstack([X(s)[:, 2:], load(s)["rich"]])),
        ]
        print(table([evaluate(name, f, ridge(), workers=args.workers, verbose=False) for name, f in combos]))
    if not (args.check or args.evaluate):
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for line in pool.imap_unordered(extract, args.subjects):
                print(line, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
