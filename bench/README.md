# omarch-eye benchmark: MPIIFaceGaze

One data cache, one evaluation protocol and baseline numbers, so that every
gaze feature set for omarch-eye is scored the same way. The dataset is
MPIIFaceGaze (15 people, 37,667 laptop-webcam images at 1280×720, each taken
while the person looked at a dot on the laptop screen, over days to months).

## Running it

From the repository root:

```bash
uv run python -m bench.mpii                    # check the dataset: images, screen size, days per subject
uv run python -m bench.landmarks               # MediaPipe over every image -> ~/.cache/omarcheye/bench/landmarks/
uv run python -m bench.evaluate --baseline     # today's features, printed as a markdown table
uv run python -m bench.evaluate --table        # every saved result (results/*.json) as one table
uv run python -m bench.evaluate --table --units mm --protocols other30 other30+r5 --match rich   # a slice
uv run python -m bench.rescore                 # add the recentre protocols to every rebuildable saved result
```

The dataset is expected unpacked at `~/.cache/omarcheye/bench/MPIIFaceGaze/p00..p14`.
`bench.landmarks` skips subjects already written (`--force` redoes them) and
creates `landmarks/DONE` once all 15 are there.

| Step | Time on lunar-gouda |
|---|---|
| `bench.landmarks` (4 worker processes, central processing unit (CPU) delegate, about 11 ms per image per worker) | 2 min 17 s |
| `bench.evaluate --baseline`, the six regression rows (4 subjects at a time) | 3 min 20 s, most of it the quadratic ridge (138 s); omarcheye.model 26–33 s, plain ridge 1 s |
| `bench.evaluate --baseline`, geometric model, every repeat (2,119 fits) | 3 min 36 s |
| `bench.evaluate --baseline`, all of it | 6 min 56 s |
| `bench.rescore`, 71 saved results (other-sitting protocols only; recentring adds no fits) | 14 min 39 s, most of it the eye-patch sets |
| `evaluate()` with the default ridge on a 21-feature set | about 1 s |

## Files

- `mpii.py`: annotations (`load_annotations`) and per-subject calibration
  (`load_calibration`: screen size in pixels and mm, camera matrix, monitor
  pose, `px_to_mm`, `screen_to_camera`). It reads the MATLAB files itself (no
  scipy). Note the dataset's monitor-pose rotation is stored as `rvects`.
- `landmarks.py`: writes `~/.cache/omarcheye/bench/landmarks/pXX.npz`. MediaPipe
  Face Landmarker in IMAGE mode with omarch-eye's own model file
  (`~/.local/share/omarcheye/models/face_landmarker.task`) and options (one
  face, transformation matrix and blendshapes on, CPU), imported through
  `omarcheye.tracker._import_mediapipe` (a plain `import mediapipe` gets the
  process SIGKILLed on this machine).
- `evaluate.py`: the protocol (`evaluate()`), fitters (`ridge()`,
  `omarcheye_fitter(kind)`, `geometric_fitter()`), the baselines and `table()`.
  Results are saved to `~/.cache/omarcheye/bench/results/<name>.json`.

### landmarks/pXX.npz

Every per-image array is aligned with the rows of `pXX.txt`.

| Array | Shape | What |
|---|---|---|
| `files`, `day` | (n,) | `dayNN/NNNN.jpg`, NN |
| `ok` | (n,) bool | a face was found and no landmark is within 1% of the image border (the tracker's `cut_off`) |
| `found`, `margin`, `image_size` | (n,), (n,), (n,2) | face found at all; nearest landmark to the border as a fraction of the image (negative: outside); width, height |
| `points` | (n,478,2) float32 | landmarks in pixels of the original image |
| `matrix` | (n,4,4) float32 | MediaPipe's facial transformation matrix (cm) |
| `blend`, `blend_names` | (n,52) float32, (52,) | blendshape scores and their names |
| `feat`, `rich`, `pose`, `openness` | (n,7), (n,21), (n,16), (n,) | `omarcheye.tracker.features()` exactly as `FaceTracker.process` calls it (see `tracker.FEATURES` and `tracker.RICH` for the columns) |
| `target_px` | (n,2) | gaze target on the screen, pixels |
| `screen_px`, `screen_mm` | (2,) | screen width, height |
| `face_center`, `gaze_target` | (n,3) | camera coordinates, mm (x right, y down, z forward) |
| `head_rvec`, `head_tvec` | (n,3) | the dataset's own head pose (6-point model) |

Every MediaPipe-derived float array is NaN where `ok` is False.

## The protocol

`evaluate(name, load_features, fitter=None)` scores one feature set.
`load_features(subject)` returns an (n, d) array aligned with the rows; rows
with a NaN are skipped, and so are rows where `ok` is False (`mask_ok=False`
turns that off), so every feature set is scored on the same images unless its
own features are missing somewhere (the `Usable images` column shows it). The
fitter maps calibration features and targets (mm on the screen) to a predict
function; the default `ridge()` standardises the features and picks the ridge
strength by closed-form leave-one-out on the calibration images only
(`ridge(quadratic=True)` adds all squares and products).

Per subject, then averaged over the 15 subjects (each counts once):

| Protocol | What |
|---|---|
| `same` | Within each day with at least 50 usable images, 5-fold random cross-validation. The subject's error is the mean over all its test images. |
| `same30` | Calibrate on 30 random images of one day, test on the rest of that day (days with at least 50 usable images). The same 30 images as `other30`: the difference between the two is the cost of sitting differently. |
| `other30` | **The main number.** Calibrate on 30 random images of one day, test on every usable image of all the other days. Repeated with each day that has at least 30 usable images as the calibration day; each calibration day counts once. |
| `other100` | The same with 100 calibration images (days with at least 100 usable images). Every subject has at least one. |
| `other30+rR`, `other100+rR` | Other sitting + recentre, R = 1, 3, 5. The same fits as `otherK`; then on each test day R images (seeded, the first R of 5 set aside per test day and never scored, whatever R) give a constant offset, the per-axis median of true − predicted, added to every prediction of that day; the day's other images are scored (test days with at least 10 usable images). It mimics `omarcheye recentre` (one dot) plus the shift `drift.py` learns from a few corrections. |
| `other30+a5`, `other100+a5` | The same with a per-day affine map `A p + b` instead of the offset, least squares on the R images with `A` pulled towards the identity (ridge, (100 mm)² per image; 20–400 mm were tried on rich, flat from 50 to 200). Only when asked for. |
| `otherK+r0` | Added with any recentre protocol: `otherK` on the recentre test images, not recentred. It matches `otherK` within 0.05 mm on every set checked, so the two are comparable. |

Errors are reported three ways: mm on the screen; degrees, the angle at the
dataset's face centre between the rays to the true and the predicted point,
both placed in camera space with the subject's monitor pose (exact, not the
atan approximation; the monitor pose reproduces the dataset's three-dimensional (3D) gaze targets
to 0.00 mm); and % of the screen width (286.5 mm on 14 subjects, 331.7 mm on
p06). Predictions are clamped to the screen rectangle, as the live pipeline
clamps points just outside it. Each result also records `offscreen`, the share
of predictions more than 15% of the screen outside it before clamping (live,
those count as looking away), and `nan`. Splits are seeded from (seed, subject,
day, K) over all rows of a day, so feature sets with the same usable rows get
identical calibration and test sets.

Usage:

```python
from bench.evaluate import evaluate, load, ridge, table

def my_features(subject):                       # (n, d), aligned with pXX.txt, NaN where unusable
    d = load(subject)                           # the landmarks/pXX.npz arrays (cached)
    return np.hstack([my_eye_features(d), d["feat"][:, 2:]])   # e.g. new eye features + head pose

res = evaluate("my-features", my_features)                  # default fitter: ridge()
res = evaluate("my-features-quad", my_features, ridge(quadratic=True), workers=4)
print(res["other30"]["deg"], res["same"]["mm"]); print(table([res]))
```

A custom fitter is `fitter(X, Y, info) -> predict`, with `X` (k, d), `Y` (k, 2)
mm from the screen's top-left corner (x right, y down), `info` = `{subject,
screen_mm, screen_px}`, and `predict(X_test)` returning (m, 2) mm. Other
keywords: `subjects`, `protocols`, `seed`, `max_days` (subsample calibration
days), `workers` (forked processes, one subject each), `save`, `verbose`.

## Baselines: today's features

Mean error, mm on the screen / degrees / % of the screen width.

| Features | Same sitting, 5-fold | Same sitting, K=30 | Other sitting, K=30 | Other sitting, K=100 | Usable images |
|---|---|---|---|---|---|
| none (calibration mean) | 92.4 mm / 10.85° / 31.9% | 92.8 mm / 10.94° / 32.1% | 93.3 mm / 10.97° / 32.2% | 92.4 mm / 10.87° / 31.9% | 92.9% |
| basic (omarcheye.model) | 36.5 mm / 4.14° / 12.6% | 40.1 mm / 4.57° / 13.8% | 54.3 mm / 6.16° / 18.7% | 47.1 mm / 5.33° / 16.2% | 92.9% |
| rich (omarcheye.model) | 30.8 mm / 3.47° / 10.6% | 36.0 mm / 4.09° / 12.4% | 50.2 mm / 5.69° / 17.3% | 42.0 mm / 4.76° / 14.5% | 92.9% |
| basic, plain ridge | 37.1 mm / 4.20° / 12.8% | 39.8 mm / 4.53° / 13.7% | 55.0 mm / 6.23° / 19.0% | 47.2 mm / 5.35° / 16.3% | 92.9% |
| rich, plain ridge | 31.4 mm / 3.54° / 10.8% | 36.2 mm / 4.10° / 12.5% | 52.2 mm / 5.92° / 18.1% | 42.9 mm / 4.86° / 14.8% | 92.9% |
| rich, quadratic ridge | 38.1 mm / 4.33° / 13.1% | 62.6 mm / 7.21° / 21.6% | 91.3 mm / 10.43° / 31.5% | 58.3 mm / 6.59° / 20.1% | 92.9% |
| geometric (omarcheye.geometry) | 39.6 mm / 4.50° / 13.7% | 39.6 mm / 4.53° / 13.7% | 50.7 mm / 5.75° / 17.5% | 47.6 mm / 5.41° / 16.4% | 92.9% |

- **none**: no features, the calibration images' mean target: the floor any
  feature set has to beat by a wide margin.
- **basic / rich (omarcheye.model)**: `omarcheye.model.fit` exactly as
  calibration calls it (quadratic eye terms, head spreads floored at 2° and
  2 cm, lambda by leave-one-group-out, live features clamped to the
  calibration's range), with one group per image. Blink and stray-frame
  filtering (`model.prepare`) is not applied: it works per dot, and these
  images are single frames.
- **plain ridge**: the protocol's default fitter on the same features.
- **geometric (omarcheye.geometry)**: `omarcheye.geometry.fit` with the
  subject's screen size in mm, both mirror options and its four sign starts;
  its own leave-one-dot-out error is skipped (`score=set()`), which doesn't
  change the fitted parameters. Every repeat was run (about 0.4 s per fit of 30–100 images, 1 s for 500), none subsampled.

## Recentring: one dot, a few corrections

Other sitting, mean error in mm (`bench.evaluate --table` has degrees and %
of width too). R images of each test day correct a constant offset (`+rR`) or
an affine map (`+a5`).

| Features | K=30 | +R=1 | +R=3 | +R=5 | affine R=5 | K=100 | +R=1 | +R=3 | +R=5 | affine R=5 |
|---|---|---|---|---|---|---|---|---|---|---|
| none (calibration mean) | 93.3 | 122.3 | 111.6 | 105.8 | — | 92.4 | 122.4 | 111.5 | 105.8 | — |
| basic (omarcheye.model) | 54.3 | 64.3 | 55.5 | 52.6 | — | 47.1 | 56.9 | 48.9 | 46.2 | — |
| rich (omarcheye.model) | 50.2 | 58.6 | 50.9 | 48.2 | 46.1 | 42.0 | 51.2 | 43.7 | 41.3 | 40.1 |
| rich, plain ridge | 52.2 | 60.3 | 52.9 | 50.2 | 47.6 | 42.9 | 51.9 | 44.5 | 42.1 | 40.9 |
| geometric (omarcheye.geometry) | 50.7 | 58.9 | 51.4 | 48.8 | — | 47.6 | 56.7 | 49.2 | 46.8 | — |
| eyenet: hit point | 51.4 | 55.8 | 49.1 | 46.9 | — | 49.3 | 55.0 | 48.3 | 46.1 | — |
| eyenet: hit + rich | 48.1 | 53.3 | 46.9 | 44.4 | 42.1 | 40.9 | 45.8 | 40.0 | 37.8 | 36.8 |
| iris-disk-rich (omarcheye.model) | 48.5 | 57.4 | 49.6 | 46.8 | — | 40.8 | 50.0 | 42.8 | 40.4 | — |
| mobilegaze: hit + rich | 50.6 | 57.7 | 50.5 | 48.0 | — | 41.9 | 50.0 | 42.9 | 40.6 | — |
| patch-g60-hog+eye+disk | 44.3 | 52.7 | 45.6 | 43.1 | — | 36.4 | 44.2 | 37.6 | 35.6 | — |
| patch-g60-hog+eye+disk+hit | 41.5 | 45.3 | 39.5 | 37.5 | — | 34.0 | 38.7 | 33.5 | 31.5 | — |

In degrees, rich (omarcheye.model) goes from 5.69° to 5.48° (R=5) and 5.24°
(affine) with K=30; eyenet hit + rich from 5.44° to 5.04° and 4.79°.

- One image makes things worse for every feature set (8–10 mm for the
  regressions), three about break even, five gain 2–4 mm. A single image's
  own error (about 36 mm within a sitting, rich) is as large as the shift it
  is meant to measure. Live, `omarcheye recentre` averages about 60 frames
  on its dot, which removes the frame-to-frame part of that noise (not the
  error specific to that spot), so R=1 here is pessimistic for it.
- Even the best possible constant offset per day (the median over all the
  day's images, an oracle) only takes rich (omarcheye.model) from 50.2 to
  43.1 mm at K=30. Most of the other-sitting error isn't a shift of the whole
  day; an affine map with 5 images gets further than any constant offset
  from 5 (46.1 vs 48.2 mm).
- Sets that include the eye network's hit point gain the most from
  recentring (hit + rich −3.7 mm with R=5 and −6.0 mm affine; the patch set
  with the hit point −4.0 mm with R=5), so more of their cross-sitting error
  is a per-day shift.

## What stands out

- MediaPipe found a face in 37,595 of 37,667 images (72 misses), but the
  tracker's cut-off rule (any landmark within 1% of the image border) rejects
  2,600 more, so 34,995 (92.9%) are usable. The rejected faces are mostly cut
  by the bottom of the image (the chin; 178 of 206 checked, the other 28 at
  the top), and in all 206 checked the eye landmarks were well inside the
  image. p09 loses 19% of its images this way, p08 14%, p13 14%, p11 14%, p05
  none. The cache follows the tracker (NaN for those rows); a feature set
  that wants them needs `found`, `margin` and its own landmarks.
- Changing sitting costs more than having few calibration images: with the
  same 30 calibration images, the rich features' error grows from 36 mm
  (rest of the same day) to 50 mm (other days). 100 images from the one day
  recover part of it (42 mm).
- The two regressions are close: omarch-eye's floors and clamping help a
  little over a plain ridge across sittings (50.2 vs 52.2 mm for rich). A
  quadratic expansion of all 21 rich features overfits badly with 30 images
  (91 mm, no better than no features): keep `quadratic=True` for small feature
  sets.
- The geometric model doesn't earn its keep here. Across sittings it matches
  the rich regression with 30 images (50.7 vs 50.2 mm) and is worse with 100
  (47.6 vs 42.0 mm); within a sitting it is the worst of the three (39.6 mm)
  and doesn't improve with more images (39.6 mm from 30 images and from 4/5
  of a whole day, 40–600 images), so something other than the amount of data
  limits it, most likely its eye model. It puts 7.7% of `other30` predictions
  more than 15% off the screen, about as many as the regressions (7.8–9.4%).
- Subjects differ a lot: rich, `other30`, from 34 mm (p01) to 71 mm (p12).
- In degrees, the rich regression's 3.5° within a sitting is close to the
  2–4° the main README expects from a webcam; % of the screen width looks
  worse here than live because these laptop screens are only 287 mm wide.
