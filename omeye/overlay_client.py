"""Starts overlay.py under the system Python and talks to it."""

import json
import os
import queue
import subprocess
import threading
import time

from .config import ROOT

SYSTEM_PYTHON = "/usr/bin/python3"
LAYER_SHELL = "/usr/lib/libgtk4-layer-shell.so"


class OverlayProcess:
    def __init__(self, monitor: str, mode: str):
        # gtk4-layer-shell must load before libwayland-client, hence LD_PRELOAD.
        env = dict(os.environ, LD_PRELOAD=LAYER_SHELL, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        self.proc = subprocess.Popen(
            [SYSTEM_PYTHON, "-m", "omeye.overlay", "--monitor", monitor, "--mode", mode],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, cwd=ROOT,
        )
        self.events: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            try:
                self.events.put(json.loads(line))
            except json.JSONDecodeError:
                pass
        self.events.put({"event": "closed"})

    def send(self, **cmd) -> None:
        try:
            self.proc.stdin.write(json.dumps(cmd) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def poll(self) -> dict | None:
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

    def wait_for(self, name: str, timeout: float) -> dict:
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            try:
                ev = self.events.get(timeout=left)
            except queue.Empty:
                break
            if ev.get("event") == name:
                return ev
            if ev.get("event") == "closed":
                break
        raise RuntimeError(f"overlay did not report {name!r}")

    def close(self) -> None:
        self.send(cmd="quit")
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.proc.wait(2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
