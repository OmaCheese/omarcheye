"""Eyes at the camera's full resolution: normalised eye patches and a sharper iris centre.

MediaPipe's Face Landmarker finds the face on a 256-pixel crop, so on a
1080p camera its iris centre is placed at a fraction of the resolution the
camera gives. This module goes back to the full-resolution frame, seeded by
MediaPipe's landmarks:

- eye_patch() cuts an eye out along its corner line (so head roll is
  removed), scaled by the corner distance, resampled to a fixed size, in
  grayscale with an equalised histogram. eye_descriptor() is the histogram
  of oriented gradients (HOG) of both eyes' patches: what a per-person
  appearance model reads (bench/patches.py fits one on the calibration).
- refine_iris() moves MediaPipe's iris centre to where the full-resolution
  eye says it is: "disk" (the iris-sized disk darkest against the ring
  around it), "limbus" (a circle through the iris's left and right edges)
  or "timm" (gradient-based eye centre, Timm & Barth 2011).
  refine_points() puts the refined centres back into the landmark array, so
  tracker.features() computes u and v from them unchanged.

On MPIIFaceGaze (bench/iris.py) "disk" is the one worth having: alone its
u and v are noisier than MediaPipe's, but next to MediaPipe's eye-direction
scores (the "rich" features) it lowers the error by about half a point of
the screen width. "timm" made every feature set worse.

Everything works on one eye at a time in a small crop of the frame, so the
cost does not depend on the camera's resolution.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Eye:
    left: int  # corner on the image's left
    right: int  # corner on the image's right
    iris: int  # iris centre
    edge: tuple[int, ...]  # four points on the iris's edge
    ring: tuple[int, ...]  # eyelid margin, in order around the eye


# "A" is the eye on the image's left (the subject's right), as in tracker.py.
EYE_A = Eye(33, 133, 468, (469, 470, 471, 472),
            (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246))
EYE_B = Eye(362, 263, 473, (474, 475, 476, 477),
            (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398))
EYES = (EYE_A, EYE_B)

PATCH = (60, 36)  # patch size (width, height) in pixels
SPAN = 1.6  # the patch is this many corner distances wide (and SPAN * 36 / 60 high)


def eye_transform(points: np.ndarray, eye: Eye, scale: float, size: tuple[int, int],
                  shift: float = 0.0) -> np.ndarray:
    """2x3 affine map from image pixels to patch pixels. The eye's corner line
    becomes horizontal, the corners' midpoint (moved `shift` corner distances
    down the eye) lands in the middle of the patch, and the corner distance
    becomes `scale` patch pixels."""
    a, b = np.asarray(points[eye.left], float), np.asarray(points[eye.right], float)
    width = float(np.hypot(*(b - a)))
    ex = (b - a) / max(width, 1e-6)
    ey = np.array([-ex[1], ex[0]])
    c = (a + b) / 2 + shift * width * ey
    s = scale / max(width, 1e-6)
    rot = s * np.vstack([ex, ey])
    return np.hstack([rot, (np.array(size, float) / 2 - rot @ c)[:, None]])


def warp(image: np.ndarray, m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample `image` through the affine map `m` into a patch of `size`.
    When the patch is smaller than the eye in the image, it is cut at an
    integer multiple of its size first and shrunk by area averaging, so fine
    detail doesn't alias."""
    zoom = float(np.sqrt(abs(np.linalg.det(m[:, :2]))))  # patch pixels per image pixel
    k = max(1, int(np.ceil(1.0 / max(zoom, 1e-6) - 0.25)))
    if k == 1:
        return cv2.warpAffine(image, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    big = cv2.warpAffine(image, k * m, (size[0] * k, size[1] * k), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(big, size, interpolation=cv2.INTER_AREA)


def gray(patch: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch


def normalise(patch: np.ndarray, how: str = "equalize") -> np.ndarray:
    """Contrast-normalised grayscale patch: "equalize" (histogram equalised),
    "clahe" (local equalisation) or "none"."""
    g = gray(patch)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    if how == "equalize":
        return cv2.equalizeHist(g)
    if how == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)).apply(g)
    if how == "none":
        return g
    raise ValueError(f"unknown normalisation {how!r}")


def eye_patch(frame: np.ndarray, points: np.ndarray, eye: Eye, size: tuple[int, int] = PATCH,
              span: float = SPAN, shift: float = 0.0, how: str = "equalize") -> np.ndarray:
    """One eye of a full-resolution frame (BGR or grayscale) as a normalised
    grayscale patch of `size`, `span` corner distances wide. `points` are
    MediaPipe's landmarks in the frame's pixels."""
    m = eye_transform(points, eye, size[0] / span, size, shift)
    return normalise(warp(frame, m, size), how)


def eye_patches(frame: np.ndarray, points: np.ndarray, size: tuple[int, int] = PATCH,
                span: float = SPAN, shift: float = 0.0, how: str = "equalize") -> np.ndarray:
    """Both eyes' patches, (2, height, width) uint8, eye A first."""
    return np.stack([eye_patch(frame, points, e, size, span, shift, how) for e in EYES])


def hog(p: np.ndarray, cell: int = 6, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients of one patch (OpenCV 5 dropped its
    HOGDescriptor): unsigned orientations in `bins` bins, each pixel voting
    into the two nearest, `cell`-pixel cells, 2x2-cell blocks with a
    one-cell stride, each block normalised (L2, clipped at 0.2, again L2)."""
    h, w = p.shape[0] // cell * cell, p.shape[1] // cell * cell
    g = p[:h, :w].astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    pos = (ang % 180.0) / (180.0 / bins)
    b0 = np.floor(pos).astype(int) % bins
    frac = pos - np.floor(pos)
    ch, cw = h // cell, w // cell
    cidx = (np.arange(h)[:, None] // cell) * cw + np.arange(w)[None, :] // cell
    hist = np.bincount((cidx * bins + b0).ravel(), (mag * (1 - frac)).ravel(), ch * cw * bins)
    hist += np.bincount((cidx * bins + (b0 + 1) % bins).ravel(), (mag * frac).ravel(), ch * cw * bins)
    hist = hist.reshape(ch, cw, bins)
    blocks = np.concatenate([hist[:-1, :-1], hist[:-1, 1:], hist[1:, :-1], hist[1:, 1:]], axis=2)
    blocks = blocks / np.sqrt((blocks ** 2).sum(2, keepdims=True) + 1e-6)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / np.sqrt((blocks ** 2).sum(2, keepdims=True) + 1e-6)
    return blocks.ravel().astype(np.float32)


def eye_descriptor(frame: np.ndarray, points: np.ndarray, size: tuple[int, int] = PATCH,
                   cell: int = 6) -> np.ndarray:
    """What a per-person appearance model reads from a frame: the HOG of
    both eyes' equalised patches, joined (2 x 1620 numbers for 60x36)."""
    return np.concatenate([hog(p, cell) for p in eye_patches(frame, points, size)])


# ---------------------------------------------------------------- iris centre

WORK = 64.0  # the corner distance in pixels of the crop the iris is refined on
CROP = (96, 56)  # that crop's size: 1.5 by 0.875 corner distances


@dataclass
class EyeCrop:
    """An eye resampled so the corner distance is WORK pixels, smoothed, with
    MediaPipe's iris centre and radius and the eyelid mask in its pixels."""
    img: np.ndarray  # float32 grayscale
    m: np.ndarray  # image pixels -> crop pixels
    seed: np.ndarray  # MediaPipe's iris centre (x, y)
    r: float  # iris radius from MediaPipe's iris edge points, sanity-limited
    mask: np.ndarray  # bool: inside the eyelids, a little away from them


def eye_crop(frame: np.ndarray, points: np.ndarray, eye: Eye, erode: int = 2) -> EyeCrop:
    m = eye_transform(points, eye, WORK, CROP)
    img = cv2.GaussianBlur(gray(warp(frame, m, CROP)).astype(np.float32), (0, 0), 1.0)
    pts = np.asarray(points, float)

    def to_crop(p):
        return p @ m[:, :2].T + m[:, 2]

    seed = to_crop(pts[eye.iris])
    r = float(np.mean(np.hypot(*(to_crop(pts[list(eye.edge)]) - seed).T)))
    r = float(np.clip(r, 0.15 * WORK, 0.3 * WORK))
    mask = np.zeros(CROP[::-1], np.uint8)
    cv2.fillPoly(mask, [np.round(to_crop(pts[list(eye.ring)]) * 4).astype(np.int32)], 1, cv2.LINE_8, 2)
    if erode:
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)))
    return EyeCrop(img, m, seed, r, mask.astype(bool))


def _peak(score: np.ndarray, ix: int, iy: int) -> tuple[float, float]:
    """Sub-pixel position of a maximum on a grid, by a parabola through its neighbours."""
    def offset(lo, mid, hi):
        den = lo - 2 * mid + hi
        return 0.0 if not np.isfinite(den) or den >= 0 else float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5))

    h, w = score.shape
    dx = offset(score[iy, ix - 1], score[iy, ix], score[iy, ix + 1]) if 0 < ix < w - 1 else 0.0
    dy = offset(score[iy - 1, ix], score[iy, ix], score[iy + 1, ix]) if 0 < iy < h - 1 else 0.0
    return ix + dx, iy + dy


def _candidates(c: EyeCrop, reach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer grid around the seed, `reach` iris radii in every direction."""
    n = int(np.ceil(reach * c.r))
    x0, y0 = np.round(c.seed).astype(int)
    xs = np.arange(x0 - n, x0 + n + 1)
    ys = np.arange(y0 - n, y0 + n + 1)
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)


def timm(c: EyeCrop, reach: float = 0.6, dark: float = 1.0) -> np.ndarray | None:
    """Gradient-based eye centre (Timm & Barth 2011): the point most image
    gradients point away from, over the strong gradients inside the eyelids.
    Only gradients pointing outwards (dark iris to bright sclera) count, only
    within 0.6..1.5 iris radii of the candidate, and a candidate is weighted by
    how dark it is (to the power `dark`; 0: not at all)."""
    gx = cv2.Sobel(c.img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(c.img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    inside = mag[c.mask]
    if inside.size < 20:
        return None
    keep = c.mask & (mag > inside.mean() + 0.3 * inside.std())
    py, px = np.nonzero(keep)
    if px.size < 10:
        return None
    g = np.column_stack([gx[keep], gy[keep]]) / mag[keep][:, None]
    pix = np.column_stack([px, py]).astype(np.float32)
    xs, ys, cand = _candidates(c, reach)
    d = pix[None, :, :] - cand[:, None, :]  # (candidates, pixels, 2)
    dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
    dot = (d[..., 0] * g[None, :, 0] + d[..., 1] * g[None, :, 1]) / dist
    ring = (dist > 0.6 * c.r) & (dist < 1.5 * c.r)
    score = (np.maximum(dot, 0) ** 2 * ring).sum(1)
    if dark:
        blur = cv2.GaussianBlur(c.img, (0, 0), c.r / 3)
        h, w = c.img.shape
        cx, cy = np.clip(cand[:, 0].astype(int), 0, w - 1), np.clip(cand[:, 1].astype(int), 0, h - 1)
        score = score * (255.0 - blur[cy, cx]) ** dark
    score = score.reshape(len(ys), len(xs))
    iy, ix = np.unravel_index(np.argmax(score), score.shape)
    fx, fy = _peak(score, int(ix), int(iy))
    return np.array([xs[0] + fx, ys[0] + fy])


def disk(c: EyeCrop, reach: float = 0.6) -> np.ndarray | None:
    """Template match: the centre of the iris-sized disk that is darkest
    against the ring around it, counting only pixels inside the eyelids."""
    r = c.r
    n = int(np.ceil(1.35 * r))
    xs, ys, _ = _candidates(c, reach)
    h, w = c.mask.shape
    xs, ys = xs[(xs >= 0) & (xs < w)], ys[(ys >= 0) & (ys < h)]
    if not len(xs) or not len(ys):
        return None
    # Masked intensity and mask as two channels, filtered with the disk and
    # the larger disk only where the candidates need them (the crop extended
    # by reflection, as filter2D does at its border).
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    rr = np.hypot(xx, yy)
    k_in = (rr <= r).astype(np.float32)
    k_big = (rr <= 1.35 * r).astype(np.float32)
    m = c.mask.astype(np.float32)
    two = cv2.copyMakeBorder(cv2.merge([c.img * m, m]), n, n, n, n, cv2.BORDER_REFLECT_101)
    roi = two[ys[0]:ys[-1] + 2 * n + 1, xs[0]:xs[-1] + 2 * n + 1]
    inner = cv2.filter2D(roi, -1, k_in, borderType=cv2.BORDER_CONSTANT)[n:-n, n:-n]
    big = cv2.filter2D(roi, -1, k_big, borderType=cv2.BORDER_CONSTANT)[n:-n, n:-n]
    s_in, n_in = inner[..., 0], inner[..., 1]
    s_out, n_out = big[..., 0] - s_in, big[..., 1] - n_in
    with np.errstate(invalid="ignore", divide="ignore"):
        sub = s_out / n_out - s_in / n_in
    sub[(n_in < 0.3 * k_in.sum()) | (n_out < 0.1 * (k_big.sum() - k_in.sum()))] = -np.inf
    if not np.isfinite(sub).any():
        return None
    iy, ix = np.unravel_index(np.argmax(np.where(np.isfinite(sub), sub, -1e9)), sub.shape)
    fx, fy = _peak(np.where(np.isfinite(sub), sub, -1e9), int(ix), int(iy))
    return np.array([xs[0] + fx, ys[0] + fy])


def limbus(c: EyeCrop, rounds: int = 2) -> np.ndarray | None:
    """Fit a circle to the iris's edge: along rays from the centre into the
    left and right sides (top and bottom are under the lids), the strongest
    dark-to-bright step between 0.6 and 1.5 iris radii; then a weighted
    least-squares circle through those points, twice."""
    ang = np.deg2rad(np.r_[np.arange(-55, 56, 5), np.arange(125, 236, 5)])
    rho = np.arange(0.6, 1.5, 0.04) * c.r
    centre = c.seed.astype(np.float64)
    h, w = c.img.shape
    for i in range(rounds):
        x = (centre[0] + np.cos(ang)[:, None] * rho[None, :]).astype(np.float32)
        y = (centre[1] + np.sin(ang)[:, None] * rho[None, :]).astype(np.float32)
        prof = cv2.remap(c.img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        step = np.diff(prof, axis=1)
        j = np.argmax(step, axis=1)
        strength = step[np.arange(len(ang)), j]
        ex = (x[np.arange(len(ang)), j] + x[np.arange(len(ang)), j + 1]) / 2
        ey = (y[np.arange(len(ang)), j] + y[np.arange(len(ang)), j + 1]) / 2
        xi, yi = np.clip(ex.astype(int), 0, w - 1), np.clip(ey.astype(int), 0, h - 1)
        ok = (strength > 2.0) & c.mask[yi, xi]
        if ok.sum() < 6 or ok[np.cos(ang) > 0].sum() < 2 or ok[np.cos(ang) < 0].sum() < 2:
            return None if i == 0 else centre
        fit = _circle(ex[ok], ey[ok], strength[ok], c.r)
        if fit is None or np.hypot(*(fit - c.seed)) > MAX_MOVE * c.r:
            return None if i == 0 else centre
        centre = fit
    return centre


def _circle(x: np.ndarray, y: np.ndarray, wt: np.ndarray, r0: float) -> np.ndarray | None:
    """Weighted algebraic circle fit, with a weak pull of the radius towards r0
    (with edges on two sides only, the radius and the vertical position trade
    off), then one pass dropping points more than 2 px off."""
    wt = wt / wt.sum()
    keep = np.ones(len(x), bool)
    centre = None
    for _ in range(2):
        a = np.column_stack([x, y, np.ones_like(x)])[keep]
        b = -(x ** 2 + y ** 2)[keep]
        sw = np.sqrt(wt[keep])
        # x^2 + y^2 + D x + E y + F = 0, plus prior: D^2/4 + E^2/4 - F = r0^2 (linearised around previous fit)
        try:
            sol, *_ = np.linalg.lstsq(a * sw[:, None], b * sw, rcond=None)
        except np.linalg.LinAlgError:
            return centre
        cx, cy = -sol[0] / 2, -sol[1] / 2
        r = np.sqrt(max(cx ** 2 + cy ** 2 - sol[2], 1e-9))
        if not 0.5 * r0 < r < 1.6 * r0:
            # radius implausible: fit the centre with the radius fixed at r0
            cx, cy = _fixed_radius(x[keep], y[keep], wt[keep], r0, np.array([cx, cy]))
            r = r0
        centre = np.array([cx, cy])
        res = np.abs(np.hypot(x - cx, y - cy) - r)
        new = res < 2.0
        if new.sum() < 6 or (new == keep).all():
            break
        keep = new
    return centre


def _fixed_radius(x, y, wt, r, start):
    c = start.astype(float)
    for _ in range(10):
        dx, dy = c[0] - x, c[1] - y
        dist = np.hypot(dx, dy) + 1e-9
        res = dist - r
        jac = np.column_stack([dx / dist, dy / dist])
        sw = np.sqrt(wt)
        step, *_ = np.linalg.lstsq(jac * sw[:, None], -res * sw, rcond=None)
        c = c + step
        if np.hypot(*step) < 1e-3:
            break
    return c


METHODS = {"timm": timm, "disk": disk, "limbus": limbus}


MAX_MOVE = 0.8  # a refined centre further than this many iris radii from MediaPipe's is a failure


def refine_crop(c: EyeCrop, method: str = "disk") -> np.ndarray | None:
    """Refined iris centre in the crop's pixels, or None. "mean" averages
    the methods that found something."""
    if method == "mean":
        found = [p for p in (refine_crop(c, m) for m in METHODS) if p is not None]
        return np.mean(found, 0) if found else None
    p = METHODS[method](c)
    if p is None or not np.all(np.isfinite(p)) or np.hypot(*(p - c.seed)) > MAX_MOVE * c.r:
        return None
    return p


def to_frame(c: EyeCrop, p: np.ndarray) -> np.ndarray:
    """Crop pixels -> frame pixels."""
    inv = cv2.invertAffineTransform(c.m)
    return inv[:, :2] @ p + inv[:, 2]


def refine_iris(frame: np.ndarray, points: np.ndarray, eye: Eye, method: str = "disk",
                blend: float = 1.0) -> np.ndarray | None:
    """The iris centre of one eye in frame pixels, refined on the
    full-resolution eye; None when the method finds nothing. `blend` < 1
    moves only part of the way from MediaPipe's centre."""
    c = eye_crop(frame, points, eye)
    p = refine_crop(c, method)
    if p is None:
        return None
    return to_frame(c, c.seed + blend * (p - c.seed))


def refine_points(frame: np.ndarray, points: np.ndarray, method: str = "disk",
                  blend: float = 1.0) -> tuple[np.ndarray, int]:
    """A copy of the landmarks with both iris centres refined (each left as
    MediaPipe put it when refinement fails), and how many were refined. Pass
    the copy to tracker.features() for u and v from the refined centres."""
    out = np.array(points, dtype=float, copy=True)
    n = 0
    for eye in EYES:
        p = refine_iris(frame, points, eye, method, blend)
        if p is not None:
            out[eye.iris] = p
            n += 1
    return out, n
