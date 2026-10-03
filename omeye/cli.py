"""omeye command line."""

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time

from . import config
from .config import CALIBRATION_PATH, SERVICE


def systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def service_active() -> bool:
    return systemctl("is-active", "--quiet", SERVICE).returncode == 0


def notify(cfg: config.Config, text: str) -> None:
    if cfg.notify and shutil.which("notify-send"):
        subprocess.run(["notify-send", "-a", "omeye", "-i", "camera-web", "-t", "2000", "omeye", text])


@contextlib.contextmanager
def camera_free():
    """Stop the service while a foreground command needs the camera."""
    was = service_active()
    if was:
        systemctl("stop", SERVICE)
    try:
        yield
    finally:
        if was:
            systemctl("start", SERVICE)


def cmd_on(cfg, args) -> int:
    if not CALIBRATION_PATH.exists():
        print("omeye: not calibrated yet; run `omeye calibrate` first", file=sys.stderr)
        notify(cfg, "Not calibrated yet: run omeye calibrate")
        return 1
    r = systemctl("start", SERVICE)
    if r.returncode:
        print(r.stderr.strip() or f"omeye: could not start {SERVICE}; run ./install.sh", file=sys.stderr)
        return 1
    from .tracker import pick_camera

    camera = json.loads(CALIBRATION_PATH.read_text()).get("camera") or cfg.camera
    try:
        pick_camera(camera if cfg.camera == "auto" else cfg.camera, quiet=True)
        notify(cfg, "Eye focus on")
    except RuntimeError as e:
        print(f"omeye: on, waiting for the camera: {e}", file=sys.stderr)
        notify(cfg, f"Eye focus on, waiting for the camera: {e}")
    return 0


def cmd_off(cfg, args) -> int:
    systemctl("stop", SERVICE)
    notify(cfg, "Eye focus off")
    return 0


def cmd_toggle(cfg, args) -> int:
    return cmd_off(cfg, args) if service_active() else cmd_on(cfg, args)


def cmd_status(cfg, args) -> int:
    print(f"service:     {'on' if service_active() else 'off'}")
    if CALIBRATION_PATH.exists():
        c = json.loads(CALIBRATION_PATH.read_text())
        print(f"calibration: {c['created']} on {c['monitor']} with {c['camera']!r}, {c.get('kind', 'basic')} model, "
              f"error {100 * c['error']:.1f}% of screen width")
    else:
        print("calibration: none (run `omeye calibrate`)")
    from .tracker import pick_camera

    try:
        cam = pick_camera(cfg.camera, quiet=True)
        print(f"camera:      {cam.name} ({cam.device})")
    except RuntimeError as e:
        print(f"camera:      {e}")
    return 0


def cmd_run(cfg, args) -> int:
    from .daemon import run

    return run(cfg, preview=args.preview, dry_run=args.dry_run, verbose=args.verbose)


def cmd_preview(cfg, args) -> int:
    from .daemon import run

    with camera_free():
        return run(cfg, preview=True, dry_run=not args.switch, verbose=True)


def cmd_calibrate(cfg, args) -> int:
    from .calibrate import run

    with camera_free():
        return run(cfg, args.monitor, args.points)


def cmd_refine(cfg, args) -> int:
    from .refine import run

    with camera_free():
        return run(cfg, args.seconds)


def cmd_test(cfg, args) -> int:
    from .validate import run

    with camera_free():
        return run(cfg)


def cmd_camera(cfg, args) -> int:
    from .camview import show

    with camera_free():
        return show(cfg)


def cmd_cameras(cfg, args) -> int:
    from .tracker import list_cameras, pick_camera

    try:
        chosen = pick_camera(cfg.camera, quiet=True).device
    except RuntimeError:
        chosen = None
    for c in list_cameras():
        state = "sending video" if c.live else "idle: nothing feeds it" if c.virtual else "no video"
        kind = "virtual" if c.virtual else "built-in" if c.builtin else "plug-in"
        print(f"{'*' if c.device == chosen else ' '} {c.device}  {c.name}  ({kind}, {state})")
    print(f"(* = used with camera = {cfg.camera!r})")
    return 0


def cmd_bench(cfg, args) -> int:
    import numpy as np

    from .tracker import A_LEFT, A_RIGHT, B_LEFT, B_RIGHT, Camera, FaceTracker, pick_camera

    try:
        info = pick_camera(cfg.camera)
    except RuntimeError as e:
        print(f"omeye: {e}", file=sys.stderr)
        return 1
    print("omeye: look at one spot on the screen until it finishes")
    with camera_free():
        tracker = FaceTracker(delegate=cfg.delegate)
        cam = Camera(info.device, cfg.width, cfg.height, cfg.fps)
        print(f"omeye: {args.seconds} s on {info.name} ({info.device}) at {'x'.join(map(str, cam.size()))}, "
              f"landmarks on {cfg.delegate.upper()}")
        times, faces, eyes, new_frames, last_sig = [], 0, [], 0, None
        cpu0, wall0 = time.process_time(), time.monotonic()
        while time.monotonic() - wall0 < args.seconds:
            frame = cam.read()
            if frame is None:
                continue
            sig = frame[::16, ::16].tobytes()
            new_frames += sig != last_sig  # a camera that runs slower repeats frames
            last_sig = sig
            t = time.monotonic()
            s = tracker.process(frame, t)
            times.append(time.monotonic() - t)
            if s is not None:
                faces += 1
                p = s.points
                width = (np.linalg.norm(p[A_RIGHT] - p[A_LEFT]) + np.linalg.norm(p[B_RIGHT] - p[B_LEFT])) / 2
                eyes.append((width, *s.rich[:6]))
        wall, cpu = time.monotonic() - wall0, time.process_time() - cpu0
        cam.close()
        tracker.close()
    ms = 1000 * np.array(times or [0])
    print(f"frames {len(times)} ({len(times) / wall:.1f} fps, {new_frames / wall:.1f} of them new), "
          f"face in {100 * faces / max(len(times), 1):.0f}%")
    print(f"landmarks {np.median(ms):.1f} ms median, {np.percentile(ms, 95):.1f} ms p95")
    print(f"CPU {100 * cpu / wall:.0f}% of one core")
    if len(eyes) > 10:
        # Frame-to-frame jitter: robust spread (median absolute difference) of
        # the change between consecutive frames, over sqrt 2. Slow head drift
        # doesn't count, and neither do the few frames around a blink.
        e = np.array(eyes)
        jitter = 1.4826 * np.median(np.abs(np.diff(e[:, 1:], axis=0)), 0) / np.sqrt(2)
        w = np.median(e[:, 0])
        names = ("iris across", "iris down", "iris across", "iris down", "upper lid", "lower lid")
        print(f"eye {w:.0f} px wide; landmark jitter (eye widths, pixels):")
        for name, eye_name, j in zip(names, ("A", "A", "B", "B", "A", "A"), jitter):
            print(f"  {name:12s} eye {eye_name}  {j:.4f}  {j * w:.2f} px")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="omeye", description="Focus the Hyprland window you look at.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("on", help="start eye focus (systemd user service)")
    sub.add_parser("off", help="stop eye focus")
    sub.add_parser("toggle", help="on if off, off if on")
    sub.add_parser("status", help="service, calibration and camera")
    p = sub.add_parser("calibrate", help="follow dots on screen to calibrate")
    p.add_argument("--monitor", default="", help="monitor the camera sits on (default: focused)")
    p.add_argument("--points", type=int, default=0, help="number of dots (9, 12, 15, 20 or 24)")
    p = sub.add_parser("refine", help="follow the mouse pointer with your eyes to improve the calibration")
    p.add_argument("--seconds", type=float, default=60, help="stop after this long (default 60)")
    sub.add_parser("test", help="look at 9 dots: how good the calibration is now, and which model does best")
    p = sub.add_parser("preview", help="show where omeye thinks you look")
    p.add_argument("--switch", action="store_true", help="also switch focus")
    p = sub.add_parser("run", help="tracking loop in the foreground (what the service runs)")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="never change focus")
    p.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("camera", help="show what the camera sees, with the tracking drawn on (Ctrl+C to close)")
    sub.add_parser("cameras", help="list cameras")
    p = sub.add_parser("bench", help="measure landmark speed and CPU use")
    p.add_argument("--seconds", type=float, default=10)
    args = ap.parse_args(argv)
    cfg = config.load()
    return globals()[f"cmd_{args.cmd}"](cfg, args)
