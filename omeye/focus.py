"""Which window the gaze is on, and when that becomes a focus change."""

from dataclasses import dataclass

from .hypr import Window


def hit_test(windows: list[Window], x: float, y: float, focused: str | None, margin: float) -> str | None:
    """Window under (x, y), topmost first.

    Other windows count only `margin` inside their edges and the focused one
    reaches `margin` beyond its own, so gaze jitter along a border can't flip
    focus back and forth.
    """
    for w in windows:
        m = -margin if w.address == focused else margin
        if w.x + m <= x < w.x + w.w - m and w.y + m <= y < w.y + w.h - m:
            return w.address
    return None


@dataclass
class DwellParams:
    dwell: float = 0.4
    gap: float = 0.15
    typing_grace: float = 0.7
    mouse_grace: float = 2.0
    cooldown: float = 0.3

    @classmethod
    def from_config(cls, cfg) -> "DwellParams":
        return cls(
            cfg.dwell_ms / 1000, cfg.gap_ms / 1000, cfg.typing_grace_ms / 1000,
            cfg.mouse_grace_ms / 1000, cfg.cooldown_ms / 1000,
        )


class Dwell:
    """Turns a stream of gaze targets into focus requests.

    step() returns a window address to focus, or None. A target must hold for
    `dwell` seconds; looking elsewhere for less than `gap` doesn't restart the
    count. Typing (input within `typing_grace`), recent mouse motion and the
    cooldown after a switch all hold switching off.
    """

    def __init__(self, p: DwellParams):
        self.p = p
        self.candidate: str | None = None
        self.since = 0.0
        self.last_on = 0.0  # last time the gaze was on the candidate
        self.blocked_until = 0.0

    def reset(self) -> None:
        self.candidate = None

    def step(self, now: float, target: str | None, focused: str | None,
             last_input: float, last_mouse: float) -> str | None:
        p = self.p
        if now - last_input < p.typing_grace or now - last_mouse < p.mouse_grace or now < self.blocked_until:
            self.candidate = None
            return None
        if self.candidate is not None and target != self.candidate:
            if now - self.last_on < p.gap:
                return None  # brief glance away: keep counting
            self.candidate = None
        if target is None or target == focused:
            self.candidate = None
            return None
        if self.candidate is None:
            self.candidate, self.since = target, now
        self.last_on = now
        if now - self.since >= p.dwell:
            self.candidate = None
            self.blocked_until = now + p.cooldown
            return target
        return None
