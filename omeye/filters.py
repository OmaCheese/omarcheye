"""Fixation filter: the gaze point holds still while the eyes rest and moves
only when they clearly jump.

Points are in monitor widths (y scaled by height/width), so a distance means
the same across and down. A point within `radius` of the current fixation
joins it, and the output is the median of the fixation's last `window`
points. A point further away starts a possible jump; `confirm` such points in
a row that agree with each other (within `radius`) become the new fixation.
A lone stray point is ignored, and so is jitter within the radius.
"""

import numpy as np


class FixationFilter:
    def __init__(self, radius: float = 0.06, confirm: int = 3, window: int = 15):
        self.radius, self.confirm, self.window = radius, confirm, window
        self.reset()

    def reset(self) -> None:
        self.points: list[np.ndarray] = []
        self.pending: list[np.ndarray] = []

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        p = np.array([x, y])
        if not self.points:
            self.points = [p]
            return x, y
        centre = np.median(self.points, 0)
        if np.linalg.norm(p - centre) <= self.radius:
            self.points = (self.points + [p])[-self.window:]
            self.pending = []
        else:
            if self.pending and np.linalg.norm(p - np.median(self.pending, 0)) > self.radius:
                self.pending = []  # strays in different directions: not a jump
            self.pending.append(p)
            if len(self.pending) >= self.confirm:
                self.points, self.pending = self.pending, []
        centre = np.median(self.points, 0)
        return float(centre[0]), float(centre[1])
