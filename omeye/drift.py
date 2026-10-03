"""How far the gaze estimate has shifted since calibration, learned while
omeye runs.

Sitting differently moves every estimate by about the same amount. In a later
sitting the calibration was off by 19% of the screen width; one shift, the
same for every dot, explained all of it but 4.7%, which is as good as the
calibration was in its own sitting. Learning that shift is what keeps omeye
from picking the neighbouring window.

omeye records the raw estimate whenever it is clear which window you were
looking at: you sent focus on with a glance (a retry), or moved it yourself
right after omeye moved it; and, as a lighter record, while you type into a
window (you mostly look at it). `omeye recentre` records one dot, heavily.
The shift is the smallest one that puts each
recorded estimate well inside its window (a margin of the calibration's error
from the edges), recent records counting most. Positions are monitor fractions;
distances are in monitor widths, so up-down counts the same as across.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HALF_LIFE = 3600.0  # seconds: an hour-old record counts half
KEEP = 40  # records kept, newest
PRIOR = 0.1  # pull towards no shift, as a weight against the records' (1 each)
MAX_SHIFT = 0.3  # never shift more than this (monitor widths)
ROBUST = 0.08  # a record this far (monitor widths) from agreeing with the rest counts half


@dataclass
class Record:
    t: float  # wall-clock time
    x: float  # raw estimate, monitor fractions
    y: float
    rect: tuple[float, float, float, float]  # the window you were looking at, inset by the margin (x0, y0, x1, y1)
    weight: float = 1.0


def inset(rect: tuple[float, float, float, float], margin_x: float, margin_y: float) -> tuple[float, float, float, float]:
    """A window's rectangle shrunk by a margin, but never below its middle third."""
    x0, y0, x1, y1 = rect
    mx = min(margin_x, (x1 - x0) / 3)
    my = min(margin_y, (y1 - y0) / 3)
    return x0 + mx, y0 + my, x1 - mx, y1 - my


class Drift:
    def __init__(self, aspect: float, margin: float, calibration: str = "", path: Path | None = None):
        """aspect: monitor height / width; margin: how far inside its window a
        record should land, in monitor widths (about the calibration's error)."""
        self.aspect = aspect
        self.margin = margin
        self.calibration = calibration
        self.path = path
        self.records: list[Record] = []
        self.dx = self.dy = 0.0  # monitor fractions
        if path is not None:
            self._load()

    @property
    def size(self) -> float:
        """The shift's length in monitor widths."""
        return math.hypot(self.dx, self.dy * self.aspect)

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return x + self.dx, y + self.dy

    def add(self, x: float, y: float, rect: tuple[float, float, float, float], weight: float = 1.0,
            margin: float | None = None) -> Record | None:
        """Record that the raw estimate (x, y) was a look into `rect` (monitor
        fractions). Returns the record, or None when that would need a shift
        larger than MAX_SHIFT (then it wasn't a correction of this estimate)."""
        m = self.margin if margin is None else margin
        r = Record(time.time(), x, y, inset(rect, m, m / self.aspect), weight)
        if self._needed(r) > MAX_SHIFT:
            return None
        self.records = (self.records + [r])[-KEEP:]
        self._refit()
        return r

    def retarget(self, record: Record, rect: tuple[float, float, float, float]) -> bool:
        """The same look was into another window after all (a further retry)."""
        new = inset(rect, self.margin, self.margin / self.aspect)
        if self._needed(Record(record.t, record.x, record.y, new)) > MAX_SHIFT:
            return False
        record.rect = new
        self._refit()
        return True

    def forget(self, record: Record) -> None:
        if record in self.records:
            self.records.remove(record)
            self._refit()

    def _needed(self, r: Record) -> float:
        """The length of the smallest shift that puts r inside its rectangle, from no shift."""
        x0, y0, x1, y1 = r.rect
        return math.hypot(min(max(r.x, x0), x1) - r.x, (min(max(r.y, y0), y1) - r.y) * self.aspect)

    def _refit(self, now: float | None = None) -> None:
        """Minimise sum(w * distance(estimate + shift, rectangle)^2) + PRIOR * |shift|^2.
        Each step moves the shift to the weighted mean of the shifts that would
        just satisfy each record (a contraction, so it converges). A record
        that disagrees with the shift the others give counts less (Cauchy
        weights), so the odd wrong one (you typed here while reading there)
        doesn't drag the shift along."""
        if not self.records:
            self.dx = self.dy = 0.0
            return
        now = time.time() if now is None else now
        a = np.array([1.0, self.aspect])  # to monitor widths
        p = np.array([(r.x, r.y) for r in self.records]) * a
        lo = np.array([r.rect[:2] for r in self.records]) * a
        hi = np.array([r.rect[2:] for r in self.records]) * a
        w = np.array([r.weight * 0.5 ** (max(now - r.t, 0.0) / HALF_LIFE) for r in self.records])
        o = np.array([self.dx, self.dy]) * a
        for _ in range(300):
            want = np.clip(p + o, lo, hi) - p
            off = np.hypot(*(want - o).T)  # how far each record is from agreeing
            rw = w / (1 + (off / ROBUST) ** 2) if len(w) > 2 else w
            new = (rw[:, None] * want).sum(0) / (rw.sum() + PRIOR)
            done = np.abs(new - o).max() < 1e-7
            o = new
            if done:
                break
        length = math.hypot(*o)
        if length > MAX_SHIFT:
            o *= MAX_SHIFT / length
        self.dx, self.dy = float(o[0]), float(o[1] / self.aspect)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"calibration": self.calibration, "aspect": self.aspect, "shift": [self.dx, self.dy],
                "records": [[r.t, r.x, r.y, *r.rect, r.weight] for r in self.records]}
        self.path.write_text(json.dumps(data) + "\n")

    def _load(self) -> None:
        """Records from earlier runs, if they belong to this calibration."""
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if data.get("calibration") != self.calibration:
            return
        for t, x, y, x0, y0, x1, y1, weight in data.get("records", [])[-KEEP:]:
            self.records.append(Record(t, x, y, (x0, y0, x1, y1), weight))
        self._refit()

    def describe(self) -> str:
        if not self.records:
            return "no shift learned yet"
        return (f"shift {100 * self.dx:+.1f}% across, {100 * self.dy * self.aspect:+.1f}% down (of the width), "
                f"from {len(self.records)} correction{'s' if len(self.records) != 1 else ''}")
