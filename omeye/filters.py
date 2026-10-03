"""One Euro filter (Casiez, Roussel & Vogel, 2012): heavy smoothing when the
gaze holds still, little lag when it jumps."""

import math


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.t = None
        self.x = 0.0
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float) -> float:
        if self.t is None or t <= self.t:
            self.t, self.x, self.dx = t, x, 0.0
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


class OneEuro2D:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5):
        self.fx = OneEuro(min_cutoff, beta)
        self.fy = OneEuro(min_cutoff, beta)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def __call__(self, t: float, x: float, y: float) -> tuple[float, float]:
        return self.fx(t, x), self.fy(t, y)
