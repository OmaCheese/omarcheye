"""Which window the gaze is on, and when that becomes a focus change."""

from collections import Counter, deque
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


def on_screen(point: tuple[float, float], offscreen: float) -> tuple[float, float] | None:
    """A predicted point (monitor fractions) clamped onto the monitor, or None
    when it lies more than `offscreen` outside: then you are looking away."""
    x, y = point
    if not (-offscreen <= x <= 1 + offscreen and -offscreen <= y <= 1 + offscreen):
        return None
    return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)


@dataclass
class VoteParams:
    window: float = 0.4  # seconds of frames that vote
    share: float = 0.7  # a window needs this share of the votes to take focus
    typing_grace: float = 0.7
    mouse_grace: float = 2.0
    cooldown: float = 0.3

    @classmethod
    def from_config(cls, cfg) -> "VoteParams":
        return cls(cfg.dwell_ms / 1000, cfg.vote_share, cfg.typing_grace_ms / 1000,
                   cfg.mouse_grace_ms / 1000, cfg.cooldown_ms / 1000)


class Vote:
    """Focus goes to the window that most frames looked at.

    Every frame votes for the window under that frame's gaze point (None when
    there is no face or the gaze is off the monitor). A window other than the
    focused one takes focus when it holds `share` of the votes over a full
    `window` of seconds. Typing (input within `typing_grace`), recent mouse
    motion and the cooldown after a switch clear the votes.
    """

    def __init__(self, p: VoteParams):
        self.p = p
        self.votes: deque = deque()
        self.since: float | None = None
        self.blocked_until = 0.0

    def reset(self) -> None:
        self.votes.clear()
        self.since = None

    def leader(self) -> tuple[str | None, float]:
        """The window with the most votes and its share of all votes."""
        counts = Counter(t for _, t in self.votes if t is not None)
        if not counts:
            return None, 0.0
        target, n = counts.most_common(1)[0]
        return target, n / len(self.votes)

    def step(self, now: float, target: str | None, focused: str | None,
             last_input: float, last_mouse: float) -> str | None:
        p = self.p
        if now - last_input < p.typing_grace or now - last_mouse < p.mouse_grace or now < self.blocked_until:
            self.reset()
            return None
        if self.since is None:
            self.since = now
        self.votes.append((now, target))
        while self.votes[0][0] < now - p.window:
            self.votes.popleft()
        if now - self.since < p.window:
            return None  # no full window of votes since the last reset
        leader, share = self.leader()
        if leader is None or leader == focused or share < p.share:
            return None
        self.reset()
        self.blocked_until = now + p.cooldown
        return leader
