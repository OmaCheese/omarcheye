"""When did you last type or move the mouse?"""

import math
import subprocess
import sys
import threading
import time

from .config import IDLE_HELPER


class InputActivity:
    """Keyboard (and any other input) activity, from the omarcheye-idle helper.

    The helper says "active" on input and "idle" once `grace_ms` pass without
    any, so last() is "now" while input keeps coming.
    """

    def __init__(self, grace_ms: int):
        self.grace = grace_ms / 1000
        self.active = False
        self.last_input = -math.inf
        self.proc = None
        if not IDLE_HELPER.exists():
            print(f"omarcheye: {IDLE_HELPER} not built (make); typing won't pause switching", file=sys.stderr)
            return
        self.proc = subprocess.Popen([str(IDLE_HELPER), str(grace_ms)], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            state = line.strip()
            if state == "active":
                self.active = True
            elif state == "idle":
                self.active = False
                self.last_input = time.monotonic() - self.grace

    def last(self, now: float) -> float:
        return now if self.active else self.last_input

    def close(self) -> None:
        if self.proc:
            self.proc.terminate()


class CursorWatch:
    """Notices the mouse moving. Our own focus changes may warp the pointer,
    so jumps right after one are not counted."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.pos = None
        self.last_move = -math.inf
        self.ignore_until = -math.inf

    def update(self, now: float, pos: tuple[float, float]) -> None:
        if self.pos is not None and now >= self.ignore_until:
            if math.dist(pos, self.pos) > self.threshold:
                self.last_move = now
        self.pos = pos

    def expect_warp(self, now: float, seconds: float = 0.15) -> None:
        self.ignore_until = now + seconds
