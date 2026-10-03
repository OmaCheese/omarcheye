"""Which window you are most likely looking at, using the screen layout, and
when that becomes a focus change.

A predicted gaze point is uncertain by about the calibration's error, so it
is treated as a Gaussian blob rather than a point. The share of the blob
inside each window is the chance you are looking at that window (the
topmost window owns any overlap; what falls outside every window is "away").
A running belief over the windows combines these per-frame chances, assuming
the gaze mostly stays where it was. Focus moves when another window's
belief stays above `confidence` for `dwell` seconds, or sooner when omeye is
very sure: above `quick` for `quick_dwell` seconds. Looking clearly into a
window gets there within a few frames; borders and weak evidence don't.

When omeye picks the wrong window, a quick glance away and back (Glance)
sends focus on to the runner-up: the likeliest window next to where you are
looking, leaving out the ones already tried.
"""

import math
from dataclasses import dataclass

from .hypr import Window

AWAY = None  # the belief's state for gaps, outside the windows, or no face


def on_screen(point: tuple[float, float], offscreen: float) -> tuple[float, float] | None:
    """A predicted point (monitor fractions) clamped onto the monitor, or None
    when it lies more than `offscreen` outside: then you are looking away."""
    x, y = point
    if not (-offscreen <= x <= 1 + offscreen and -offscreen <= y <= 1 + offscreen):
        return None
    return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)


def _cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def window_chances(windows: list[Window], x: float, y: float, sigma: float) -> dict[str | None, float]:
    """Chance that the gaze is in each window, for a predicted point (x, y)
    with Gaussian error `sigma` per axis (all in pixels). Windows are topmost
    first; AWAY gets the rest."""

    def mass(x0: float, y0: float, x1: float, y1: float) -> float:
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return ((_cdf((x1 - x) / sigma) - _cdf((x0 - x) / sigma))
                * (_cdf((y1 - y) / sigma) - _cdf((y0 - y) / sigma)))

    chances: dict[str | None, float] = {}
    above: list[tuple[float, float, float, float]] = []
    for w in windows:
        r = (w.x, w.y, w.x + w.w, w.y + w.h)
        m = mass(*r)
        for a in above:  # the part a window above covers belongs to that window
            m -= mass(max(r[0], a[0]), max(r[1], a[1]), min(r[2], a[2]), min(r[3], a[3]))
        chances[w.address] = max(m, 0.0)
        above.append(r)
    chances[AWAY] = max(1.0 - sum(chances.values()), 0.0)
    return chances


@dataclass
class BeliefParams:
    dwell: float = 0.25  # seconds the belief must stay confident
    confidence: float = 0.9  # belief a window needs before focus moves
    quick: float = 0.99  # ... unless it is at least this sure:
    quick_dwell: float = 0.0  # then this long is enough
    switch_rate: float = 1.5  # expected gaze moves between windows per second
    temper: float = 0.5  # frames aren't independent (the error is mostly a steady offset): soften each one
    typing_grace: float = 0.7
    mouse_grace: float = 2.0
    cooldown: float = 0.3
    retry: float = 2.0  # seconds after a switch in which a glance away and back retries
    glance: float = 0.7  # a look away longer than this is a move, not a glance

    @classmethod
    def from_config(cls, cfg) -> "BeliefParams":
        return cls(cfg.dwell_ms / 1000, cfg.confidence, cfg.quick_confidence, cfg.quick_ms / 1000, cfg.switch_rate,
                   0.5, cfg.typing_grace_ms / 1000, cfg.mouse_grace_ms / 1000, cfg.cooldown_ms / 1000,
                   cfg.retry_ms / 1000, cfg.glance_ms / 1000)


class Belief:
    """Running probability of which window you are looking at (a forward
    filter over the windows on screen plus AWAY)."""

    FLOOR = 1e-3  # no single frame rules a window out completely

    def __init__(self, p: BeliefParams):
        self.p = p
        self.b: dict[str | None, float] = {}
        self.last_t: float | None = None
        self.candidate: str | None = None
        self.since = 0.0
        self.sure_since: float | None = None  # when the candidate passed `quick`
        self.blocked_until = 0.0

    def reset(self) -> None:
        self.b = {}
        self.last_t = None
        self.candidate = None

    def settle(self, window: str) -> None:
        """Focus moved to `window` for a reason the frames so far didn't show
        (a retry, or you moved it): start from there, so the old evidence
        doesn't pull focus straight back."""
        rest = [s for s in self.b if s != window]
        self.b = {window: 1 - self.FLOOR * len(rest), **{s: self.FLOOR for s in rest}}
        self.candidate = self.sure_since = None

    def observe(self, now: float, chances: dict[str | None, float]) -> None:
        """One frame's evidence: chances from window_chances (or {AWAY: 1})
        over the states on screen now."""
        states = list(chances)
        k = len(states)
        dt = 1 / 30 if self.last_t is None else min(max(now - self.last_t, 0.0), 1.0)
        self.last_t = now
        h = 1 - math.exp(-self.p.switch_rate * dt) if self.b else 1.0  # chance the gaze moved since the last frame
        post = {}
        for s in states:
            prior_s = self.b.get(s, 0.0)
            prior = (1 - h) * prior_s + h * (1 - prior_s) / max(k - 1, 1) if k > 1 else 1.0
            post[s] = prior * max(chances[s], self.FLOOR) ** self.p.temper
        total = sum(post.values())
        self.b = {s: v / total for s, v in post.items()}

    def top(self) -> tuple[str | None, float]:
        """The most likely state and its probability."""
        if not self.b:
            return AWAY, 0.0
        s = max(self.b, key=self.b.get)
        return s, self.b[s]

    def step(self, now: float, chances: dict[str | None, float], focused: str | None,
             last_input: float, last_mouse: float, hold: bool = False, avoid=()) -> str | None:
        """Update with one frame and return a window to focus, or None.
        hold: track, but don't switch (a glance may be under way); avoid:
        windows not to switch to (just rejected with a glance)."""
        self.observe(now, chances)
        p = self.p
        if (hold or now - last_input < p.typing_grace or now - last_mouse < p.mouse_grace
                or now < self.blocked_until):
            self.candidate = self.sure_since = None
            return None
        top, prob = self.top()
        if top is AWAY or top == focused or top in avoid or prob < p.confidence:
            self.candidate = self.sure_since = None
            return None
        if top != self.candidate:
            self.candidate, self.since, self.sure_since = top, now, None
        if prob < p.quick:
            self.sure_since = None
        elif self.sure_since is None:
            self.sure_since = now
        sure = self.sure_since is not None and now - self.sure_since >= p.quick_dwell
        if not sure and now - self.since < p.dwell:
            return None
        self.candidate = self.sure_since = None
        self.blocked_until = now + p.cooldown
        return top


class Glance:
    """After omeye moves focus, notices a quick look away and back: the sign
    that it picked the wrong window.

    Points are raw gaze estimates in monitor fractions (None: no face). While
    you look where you did at the switch, the anchor follows that spot; for
    the first `settle` seconds it just follows, because the estimate is still
    settling from the eye movement that led to the switch. Then a look counts
    as away once two frames in a row are `away` (monitor widths) from the
    anchor, or have no face; it counts as back once two frames are within `back`.
    Away for longer than `longest` is a real move, and omeye switches as usual.
    A look down and back is at the keyboard, not a glance. Blinks never reach
    here. Typing or moving the mouse disarms it: using the window accepts it.
    """

    FRAMES = 2

    def __init__(self, aspect: float, window: float = 2.0, longest: float = 0.7,
                 away: float = 0.12, back: float = 0.08, settle: float = 0.3):
        self.aspect = aspect
        self.window, self.longest, self.settle = window, longest, settle
        self.settled = -math.inf
        self.far, self.near = away, back
        self.until = -math.inf
        self.anchor: tuple[float, float] | None = None
        self.spot: tuple[float, float] | None = None  # where you were looking before the last glance
        self.disarm()

    @property
    def armed(self) -> bool:
        return self.anchor is not None

    @property
    def away(self) -> bool:
        """A look away is under way: hold switching until it's clear what it was."""
        return self.away_since is not None

    def arm(self, now: float, anchor: tuple[float, float]) -> None:
        self.disarm()
        self.until = now + self.window
        self.settled = now + self.settle
        self.anchor = anchor

    def disarm(self) -> None:
        self.anchor = self.away_since = None
        self.count = 0
        self.first = 0.0
        self.sum = [0.0, 0.0]  # where the look away went, summed (monitor widths from the anchor)

    def _offset(self, p: tuple[float, float]) -> tuple[float, float]:
        return p[0] - self.anchor[0], (p[1] - self.anchor[1]) * self.aspect

    def update(self, now: float, point: tuple[float, float] | None) -> bool:
        """One frame; True when a glance away and back has just finished."""
        if self.anchor is None:
            return False
        off = None if point is None or not all(map(math.isfinite, point)) else self._offset(point)
        if self.away_since is None:
            if now > self.until and self.count == 0:
                self.disarm()
                return False
            if now < self.settled:
                if off is not None:
                    ax, ay = self.anchor
                    self.anchor = (ax + 0.3 * (point[0] - ax), ay + 0.3 * (point[1] - ay))
                return False
            if off is None or math.hypot(*off) > self.far:
                if self.count == 0:
                    self.first, self.sum = now, [0.0, 0.0]
                self.count += 1
                if self.count >= self.FRAMES:
                    self.away_since, self.count = self.first, 0
            else:
                self.count = 0
                ax, ay = self.anchor
                self.anchor = (ax + 0.2 * (point[0] - ax), ay + 0.2 * (point[1] - ay))
            if off is not None:
                self.sum = [self.sum[0] + off[0], self.sum[1] + off[1]]
            return False
        if now - self.away_since > self.longest:
            self.disarm()
            return False
        if off is not None and math.hypot(*off) < self.near:
            self.count += 1
            if self.count >= self.FRAMES:
                down = self.sum[1] > abs(self.sum[0])
                spot, until = self.anchor, self.until
                self.disarm()
                if down:  # at the keyboard: carry on as before
                    self.anchor, self.until = spot, until
                    return False
                self.spot = spot
                return True
        else:
            self.count = 0
            if off is not None:
                self.sum = [self.sum[0] + off[0], self.sum[1] + off[1]]
        return False


def runner_up(windows: list[Window], x: float, y: float, sigma: float, exclude) -> str | None:
    """The likeliest window around (x, y) other than those in `exclude`
    (pixels; sigma widened, since the estimate just proved wrong)."""
    chances = window_chances(windows, x, y, 2 * sigma)
    best = max((w.address for w in windows if w.address not in exclude), key=lambda a: chances[a], default=None)
    return best if best is not None and chances[best] > 1e-4 else None
