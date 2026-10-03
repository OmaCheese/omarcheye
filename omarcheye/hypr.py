"""Hyprland over its request socket: layout snapshots, cursor position, focus."""

import json
import os
import socket
from dataclasses import dataclass


class HyprError(RuntimeError):
    pass


@dataclass
class Monitor:
    name: str
    x: float
    y: float
    w: float  # logical size: pixels / scale, swapped when rotated
    h: float
    focused: bool

    def to_global(self, nx: float, ny: float) -> tuple[float, float]:
        return self.x + nx * self.w, self.y + ny * self.h

    def to_local(self, gx: float, gy: float) -> tuple[float, float]:
        return (gx - self.x) / self.w, (gy - self.y) / self.h


@dataclass
class Window:
    address: str
    x: float
    y: float
    w: float
    h: float
    floating: bool
    history: int  # focusHistoryID: 0 = focused most recently
    cls: str = ""


@dataclass
class Layout:
    monitors: list[Monitor]
    windows: list[Window]  # on visible workspaces, topmost first
    focused: str | None

    def monitor(self, name: str) -> Monitor | None:
        for m in self.monitors:
            if m.name == name:
                return m
        return None


def parse_monitors(data: list[dict]) -> list[Monitor]:
    out = []
    for m in data:
        if m.get("disabled"):
            continue
        w, h = m["width"] / m["scale"], m["height"] / m["scale"]
        if m.get("transform", 0) % 2:
            w, h = h, w
        out.append(Monitor(m["name"], m["x"], m["y"], w, h, bool(m.get("focused"))))
    return out


def visible_workspaces(monitors: list[dict]) -> set[int]:
    ids = set()
    for m in monitors:
        if m.get("disabled"):
            continue
        ids.add(m["activeWorkspace"]["id"])
        special = m.get("specialWorkspace", {}).get("id", 0)
        if special:
            ids.add(special)
    return ids


def parse_layout(monitors: list[dict], clients: list[dict], active: dict | None) -> Layout:
    shown = visible_workspaces(monitors)
    wins = [
        c for c in clients
        if c.get("mapped", True) and not c.get("hidden") and c["workspace"]["id"] in shown
    ]
    # A fullscreen window hides everything else on its workspace.
    fullscreen = {c["workspace"]["id"]: c for c in wins if c.get("fullscreen")}
    wins = [c for c in wins if c["workspace"]["id"] not in fullscreen or fullscreen[c["workspace"]["id"]] is c]
    # Floating windows sit above tiled ones; among each, recently focused is likelier on top.
    wins.sort(key=lambda c: (not c["floating"], c.get("focusHistoryID", 0)))
    windows = [
        Window(c["address"], *c["at"], *c["size"], c["floating"], c.get("focusHistoryID", 0), c.get("class", ""))
        for c in wins
    ]
    focused = (active or {}).get("address") or None
    return Layout(parse_monitors(monitors), windows, focused)


class Hypr:
    def __init__(self):
        sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
        if not sig:
            raise HyprError("HYPRLAND_INSTANCE_SIGNATURE is not set; is Hyprland running?")
        runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        self.path = f"{runtime}/hypr/{sig}/.socket.sock"
        self._lua_dispatch = True

    def request(self, command: str) -> bytes:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(self.path)
            s.sendall(command.encode())
            chunks = []
            while chunk := s.recv(65536):
                chunks.append(chunk)
        return b"".join(chunks)

    def json(self, what: str):
        return json.loads(self.request("j/" + what))

    def layout(self) -> Layout:
        active = self.json("activewindow")
        return parse_layout(self.json("monitors"), self.json("clients"), active if isinstance(active, dict) else None)

    def physical_mm(self, name: str) -> tuple[float, float] | None:
        """The monitor's picture size in millimetres as it reports it (EDID), or None."""
        for m in self.json("monitors"):
            if m["name"] == name and m.get("physicalWidth") and m.get("physicalHeight"):
                w, h = float(m["physicalWidth"]), float(m["physicalHeight"])
                return (h, w) if m.get("transform", 0) % 2 else (w, h)
        return None

    def cursor(self) -> tuple[float, float]:
        p = self.json("cursorpos")
        return p["x"], p["y"]

    def focus(self, address: str) -> None:
        # Hyprland 0.56 takes Lua dispatchers; older releases take the plain form.
        if self._lua_dispatch:
            reply = self.request(f'dispatch hl.dsp.focus({{ window = "address:{address}" }})')
            if reply.strip() == b"ok":
                return
            self._lua_dispatch = False
        reply = self.request(f"dispatch focuswindow address:{address}")
        if reply.strip() != b"ok":
            raise HyprError(f"focus {address}: {reply.decode(errors='replace').strip()}")
