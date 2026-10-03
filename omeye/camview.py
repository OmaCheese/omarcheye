"""Camera feedback: a picture of what the camera sees, with what the tracker
makes of it drawn on, for the overlay. Frames go to the overlay process in
memory only; nothing is written to disk.

Drawn on: a dot per face landmark, the face outline (green: well placed,
amber: near an edge of the image, red: cut off) and circles on the irises.
"""

import base64
import signal
import threading
import time

import cv2
import numpy as np

from .config import Config
from .hypr import Hypr
from .overlay_client import OverlayProcess
from .tracker import A_IRIS, A_LEFT, A_RIGHT, B_IRIS, B_LEFT, B_RIGHT, Camera, FaceTracker, Sample, framing_advice, pick_camera

WIDTH = 480  # pixels across the picture sent to the overlay
GREEN, AMBER, RED, WHITE = (120, 200, 80), (40, 190, 250), (70, 70, 230), (255, 255, 255)  # BGR


def thumbnail(frame: np.ndarray, sample: Sample | None, width: int = WIDTH) -> str:
    """The frame scaled to `width` with the tracking drawn on, as base64 JPEG."""
    h, w = frame.shape[:2]
    k = width / w
    img = cv2.resize(frame, (width, round(h * k)), interpolation=cv2.INTER_AREA)
    if sample is not None and sample.points is not None:
        p = sample.points * k
        color = RED if sample.cut_off else AMBER if framing_advice(sample.pos, sample.margin) else GREEN
        q = np.round(p[:468]).astype(int)
        inside = (q[:, 0] >= 0) & (q[:, 0] < img.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < img.shape[0])
        img[q[inside, 1], q[inside, 0]] = color
        cv2.polylines(img, [cv2.convexHull(q.astype(np.int32))], True, color, 2, cv2.LINE_AA)
        for c in (A_IRIS, B_IRIS):
            r = np.linalg.norm(p[c + 1:c + 5] - p[c], axis=1).mean()
            cv2.circle(img, tuple(np.round(p[c]).astype(int)), max(2, round(r)), WHITE, 1, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(jpg.tobytes()).decode()


def caption(sample: Sample | None, brightness: float, fps: float) -> str:
    stats = f"brightness {brightness:.0f}   {fps:.0f} fps"
    if sample is None:
        hint = "The image is black: lens covered, or camera off?" if brightness < 10 else "Turn the camera to your face"
        return f"No face   {stats}\n{hint}"
    eye = 0.0
    if sample.points is not None:
        p = sample.points
        eye = (np.linalg.norm(p[A_RIGHT] - p[A_LEFT]) + np.linalg.norm(p[B_RIGHT] - p[B_LEFT])) / 2
    advice = framing_advice(sample.pos, sample.margin)
    return f"Face   eye {eye:.0f} px wide   {stats}" + (f"\n{advice}" if advice else "\nWell placed")


class CameraFeed:
    """Sends the camera picture to an overlay at most `rate` times a second."""

    def __init__(self, overlay: OverlayProcess, place: str, rate: float = 10):
        self.ov, self.place, self.every = overlay, place, 1 / rate
        self.last_sent = 0.0
        self.last_frame = None
        self.fps = 0.0

    def update(self, frame: np.ndarray, sample: Sample | None) -> None:
        now = time.monotonic()
        if self.last_frame is not None and now > self.last_frame:
            self.fps += 0.1 * (1 / (now - self.last_frame) - self.fps)
        self.last_frame = now
        if now - self.last_sent < self.every:
            return
        self.last_sent = now
        brightness = float(frame[::16, ::16].mean())
        self.ov.send(cmd="camera", jpeg=thumbnail(frame, sample), caption=caption(sample, brightness, self.fps),
                     place=self.place)

    def hide(self) -> None:
        self.ov.send(cmd="camera")


def show(cfg: Config) -> int:
    """omeye camera: the camera view in the middle of the screen until Ctrl+C."""
    try:
        info = pick_camera(cfg.camera)
    except RuntimeError as e:
        print(f"omeye: {e}")
        return 1
    layout = Hypr().layout()
    mon = (layout.monitor(cfg.monitor) if cfg.monitor else None) or next(m for m in layout.monitors if m.focused)
    print(f"omeye: showing {info.name} ({info.device}) on {mon.name}; Ctrl+C to close")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    tracker = FaceTracker(delegate=cfg.delegate)
    cam = Camera(info.device, cfg.width, cfg.height, cfg.fps)
    ov = OverlayProcess(mon.name, "follow")
    feed = CameraFeed(ov, "center")
    try:
        ov.wait_for("ready", 8)
        while not stop.is_set():
            frame = cam.read()
            if frame is not None:
                feed.update(frame, tracker.process(frame, time.monotonic()))
        return 0
    finally:
        ov.close()
        cam.close()
        tracker.close()
