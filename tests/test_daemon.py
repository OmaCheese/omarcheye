"""The tracking loop end to end, with a scripted camera, model and Hyprland."""

import math
import types
from dataclasses import dataclass

from omarcheye import daemon
from omarcheye.config import Config
from omarcheye.hypr import Layout, Monitor, Window

W, H = 3072, 1728
COLS = [Window(f"0x{i}", i * W / 4 + 10, 47, W / 4 - 20, H - 57, False, i, f"app{i}") for i in range(4)]
SHIFT = (-0.2, 0.0)  # the estimate sits this far left of where you look (monitor fractions)


@dataclass
class Look:
    """A scripted frame: where you look (monitor fractions), or None for no face."""
    x: float
    y: float
    openness: float = 1.0
    cut_off: bool = False


class Clock:
    def __init__(self):
        self.t = 100.0

    def monotonic(self) -> float:
        return self.t


class FakeHypr:
    def __init__(self):
        self.focused = COLS[0].address
        self.focus_calls: list[str] = []

    def layout(self) -> Layout:
        return Layout([Monitor("HDMI-A-1", 0, 0, W, H, True)], list(COLS), self.focused)

    def cursor(self):
        return 0.0, 0.0

    def focus(self, address: str) -> None:
        self.focused = address
        self.focus_calls.append(address)


class SyncPoller:
    """The layout as Hyprland has it now (the real poller refreshes in real time)."""

    def __init__(self, hypr):
        self.hypr = hypr
        self.stopped = types.SimpleNamespace(set=lambda: None)

    @property
    def layout(self) -> Layout:
        return self.hypr.layout()


class FakeModel:
    monitor, camera, created, error, blink = "HDMI-A-1", "Fake", "2026-10-04T00:00:00", 0.06, 0.2

    def predict_sample(self, s: Look):
        return s.x + SHIFT[0], s.y + SHIFT[1]


def run_script(monkeypatch, tmp_path, frames: list[Look | None], hook=None) -> tuple[FakeHypr, list[str]]:
    """Run the loop over the frames; hook(i, hypr) runs before frame i."""
    clock = Clock()
    hypr = FakeHypr()
    script = iter(enumerate(frames))
    logs: list[str] = []

    class Cam:
        def read(self):
            clock.t += 1 / 30
            i, frame = next(script, (None, None))
            if hook and i is not None:
                hook(i, hypr)
            return frame

        def skip(self):
            clock.t += 1 / 30

        def close(self):
            pass

    opened = []

    def open_camera(cfg, model, stop):
        opened.append(1)
        return Cam() if len(opened) == 1 else None

    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    monkeypatch.setattr(daemon, "time", types.SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(daemon, "CALIBRATION_PATH", calibration)
    monkeypatch.setattr(daemon, "DRIFT_PATH", tmp_path / "drift.json")
    monkeypatch.setattr(daemon, "load_model", lambda path: FakeModel())
    monkeypatch.setattr(daemon, "Hypr", lambda: hypr)
    monkeypatch.setattr(daemon, "LayoutPoller", SyncPoller)
    monkeypatch.setattr(daemon, "FaceTracker", lambda delegate, eyenet=True: types.SimpleNamespace(
        process=lambda frame, now: frame, close=lambda: None))
    monkeypatch.setattr(daemon, "InputActivity", lambda grace: types.SimpleNamespace(
        last=lambda now: -math.inf, close=lambda: None))
    monkeypatch.setattr(daemon, "open_camera", open_camera)
    monkeypatch.setattr(daemon, "log", logs.append)
    assert daemon.run(Config(), verbose=True) == 0
    return hypr, logs


def at(win: int, seconds: float, y: float = 0.5) -> list[Look]:
    """Look at the middle of a column for a while."""
    return [Look((win + 0.5) / 4, y)] * round(seconds * 30)


def test_a_glance_retries_and_the_shift_is_learned(monkeypatch, tmp_path):
    frames = (at(1, 0.8)  # look at column 1; the estimate sits in column 0, which has focus: nothing happens
              + at(2, 0.8)  # column 2: the estimate is in column 1, so omarcheye focuses column 1 (wrong)
              + [Look(0.62, -0.4)] * 9  # glance up at the camera ...
              + at(2, 1.5)  # ... and back: omarcheye retries
              + at(0, 1.5)  # column 0 (the estimate is off the screen's left edge, clamped into column 0)
              + at(3, 1.5))  # column 3: with the shift learned, straight to the right one
    hypr, logs = run_script(monkeypatch, tmp_path, frames)
    assert hypr.focus_calls[:2] == ["0x1", "0x2"], logs
    assert any(line.startswith("retry: app1 -> app2") for line in logs), logs
    assert any(line.startswith("learned shift +") for line in logs), logs
    assert hypr.focus_calls[-1] == "0x3" and "0x2" not in hypr.focus_calls[2:], logs
    assert (tmp_path / "drift.json").exists()


def test_moving_focus_yourself_right_after_a_switch_teaches_the_shift(monkeypatch, tmp_path):
    def super_right(i, hypr):
        if i == 45:  # 1.5 s in, omarcheye has focused column 1; you wanted column 2
            hypr.focused = "0x2"

    hypr, logs = run_script(monkeypatch, tmp_path, at(2, 3.0) + at(3, 1.5), super_right)
    assert hypr.focus_calls[0] == "0x1", logs
    assert any("you moved focus from app1 to app2" in line for line in logs), logs
    assert any(line.startswith("learned shift +") for line in logs), logs
    assert hypr.focus_calls[-1] == "0x3", logs
