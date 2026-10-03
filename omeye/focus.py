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

    @classmethod
    def from_config(cls, cfg) -> "BeliefParams":
        return cls(cfg.dwell_ms / 1000, cfg.confidence, cfg.quick_confidence, cfg.quick_ms / 1000, cfg.switch_rate,
                   0.5, cfg.typing_grace_ms / 1000, cfg.mouse_grace_ms / 1000, cfg.cooldown_ms / 1000)


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
             last_input: float, last_mouse: float) -> str | None:
        """Update with one frame and return a window to focus, or None."""
        self.observe(now, chances)
        p = self.p
        if now - last_input < p.typing_grace or now - last_mouse < p.mouse_grace or now < self.blocked_until:
            self.candidate = self.sure_since = None
            return None
        top, prob = self.top()
        if top is AWAY or top == focused or prob < p.confidence:
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
