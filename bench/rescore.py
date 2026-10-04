"""Add the recentre protocols to saved results whose features can be rebuilt.

    uv run python -m bench.rescore [--workers 4] [--only NAME ...]

For every result in ~/.cache/omarcheye/bench/results whose feature set and
fitter this script knows how to rebuild (the baselines, two eyenet
combinations, MobileGaze, the iris-refined and eye-patch sets), the
other-sitting protocols are run again with recentring (otherK+rR, and for the
rich sets otherK+a5) and merged into the saved result; its same-sitting
columns are kept. The re-run otherK must match the saved one, or the features
changed since: such results are reported as stale.
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import re
import sys
import time

import numpy as np

from . import evaluate as E

RECENTRE = ("other30", "other30+r1", "other30+r3", "other30+r5",
            "other100", "other100+r1", "other100+r3", "other100+r5")
AFFINE = ("other30+a5", "other100+a5")
WITH_AFFINE = {"rich (omarcheye.model)", "rich, plain ridge", "eyenet: hit + rich"}


def builders() -> dict:
    """name -> (load_features, fitter) for every result that can be rebuilt."""
    out = {
        "none (calibration mean)": (E.feat_none, E.ridge()),
        "basic (omarcheye.model)": (E.feat_basic, E.omarcheye_fitter("basic")),
        "rich (omarcheye.model)": (E.feat_rich, E.omarcheye_fitter("rich")),
        "basic, plain ridge": (E.feat_basic, E.ridge()),
        "rich, plain ridge": (E.feat_rich, E.ridge()),
        "rich, quadratic ridge": (E.feat_rich, E.ridge(quadratic=True)),
        "geometric (omarcheye.geometry)": (E.feat_geometric, E.geometric_fitter()),
    }
    feats = E.CACHE / "features"

    def npz(sub: str, key: str = "X"):
        return lambda s: np.load(feats / sub / f"{s}.npz")[key]

    eye = npz("eyenet")
    out["eyenet: hit point"] = (lambda s: eye(s)[:, 2:], E.ridge())
    out["eyenet: hit + rich"] = (lambda s: np.hstack([eye(s)[:, 2:], E.load(s)["rich"]]), E.ridge())
    mg = npz("mobilegaze")
    out["mobilegaze: gaze angles"] = (lambda s: mg(s)[:, :2], E.ridge())
    out["mobilegaze: hit point"] = (lambda s: mg(s)[:, 2:], E.ridge())
    out["mobilegaze: hit + rich"] = (lambda s: np.hstack([mg(s)[:, 2:], E.load(s)["rich"]]), E.ridge())

    for p in E.RESULTS.glob("*.json"):
        name = json.loads(p.read_text())["name"]
        m = re.fullmatch(r"(iris-\S+?)( \(omarcheye\.model\)|, plain ridge)", name)
        if m and (feats / m[1]).is_dir():
            kind = "rich" if m[1].endswith("-rich") else "basic"
            fitter = E.omarcheye_fitter(kind) if "model" in m[2] else E.ridge()
            out[name] = (npz(m[1]), fitter)

    from . import patches as P

    for name, (store, size, how, kind, opts, *extra) in P.VARIANTS.items():
        out[name] = (P.feature_loader(store, size, how, kind, *extra), P.appearance(P.npix_of(store, size, kind), **opts))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="result names to redo (default: every rebuildable one)")
    args = ap.parse_args(argv)
    build = builders()
    saved = {}
    for p in sorted(E.RESULTS.glob("*.json")):
        r = json.loads(p.read_text())
        saved[r["name"]] = (p, r)
    todo = [n for n in saved if n in build and (not args.only or n in args.only)]
    skipped = [n for n in saved if n not in build]
    print(f"rescore: {len(todo)} results; not rebuildable: {', '.join(skipped)}", file=sys.stderr, flush=True)
    stale = []
    t0 = time.monotonic()
    for name in todo:
        path, old = saved[name]
        protos = RECENTRE + (AFFINE if name in WITH_AFFINE else ())
        t = time.monotonic()
        loader, fitter = build[name]
        new = E.evaluate(name, loader, fitter, protocols=protos, workers=args.workers, save=False, verbose=False)
        diffs = [abs(old[k]["mm"] - new[k]["mm"]) for k in ("other30", "other100") if k in old and k in new]
        if any(d > 0.05 for d in diffs):
            stale.append(name)
            new["stale_same"] = "features changed since the same-sitting columns were computed"
        path.write_text(json.dumps(E._merged(old, new), indent=1, default=float) + "\n")
        print(f"{name}: {time.monotonic() - t:.0f} s; other30 {new['other30']['mm']:.1f} mm "
              f"(saved {old.get('other30', {}).get('mm', float('nan')):.1f}), "
              f"+r5 {new['other30+r5']['mm']:.1f}" + (" STALE" if name in stale else ""), file=sys.stderr, flush=True)
    print(f"rescore: {time.monotonic() - t0:.0f} s; stale: {stale}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
