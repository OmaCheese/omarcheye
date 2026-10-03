"""Full-monitor layer-shell overlay drawn with GTK 4.

Runs under the system Python, which has PyGObject; omeye's own environment
starts it through OverlayProcess (overlay_client.py). Commands arrive as
JSON lines on stdin and events leave as JSON lines on stdout.

  calibrate mode: opaque, takes the keyboard (reports key presses)
  follow mode:    transparent and click-through; draws the gaze point

Commands: {"cmd": "text", "text": ...}, {"cmd": "dot", "x", "y", "ms"} (no x
to hide), {"cmd": "gaze", "x", "y", "state"} (no x to hide), {"cmd": "grid",
"cols", "rows", "fill": [0..1 per cell, row by row]} (no cols to hide),
{"cmd": "camera", "jpeg": base64, "caption", "place": "center" | "corner"}
(no jpeg to hide), {"cmd": "rect", "x", "y", "w", "h", "label", "state"} (no x to
hide), {"cmd": "point", "x", "y"} (a small dot; no x to hide), {"cmd": "fill",
"level": 0..1} (the whole monitor grey to white, 0 to clear; reports
{"event": "painted", "t": monotonic time} when drawn), {"cmd": "quit"}.
Coordinates are logical pixels from the monitor's top-left corner.
"""

import argparse
import base64
import json
import signal
import sys
import threading
import time
import warnings

import cairo
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gdk, GdkPixbuf, Gio, GLib, Gtk  # noqa: E402
from gi.repository import Gtk4LayerShell as LayerShell  # noqa: E402

# Still the simplest way to paint a JPEG with cairo; GTK 4 marks it deprecated.
warnings.filterwarnings("ignore", "Gdk.cairo_set_source_pixbuf", DeprecationWarning)

# focused window, sure enough to switch, only the likeliest
STATE_RGBA = {"focused": (0.3, 0.85, 0.5, 0.85), "ready": (0.98, 0.78, 0.25, 0.85), "likely": (0.9, 0.9, 0.95, 0.6)}

CSS = b"""
window.omeye-calibrate { background: #111318; }
window.omeye-follow { background: transparent; }
"""


def emit(**event) -> None:
    print(json.dumps(event), flush=True)


class Overlay(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, monitor: str, mode: str):
        super().__init__(application=app)
        self.mode = mode
        self.text = ""
        self.dot = None  # (x, y, start, seconds)
        self.gaze = None  # (x, y, state)
        self.grid = None  # (cols, rows, fill per cell)
        self.camera = None  # (pixbuf, caption, place)
        self.rect = None  # (x, y, w, h, label, state)
        self.point = None  # (x, y)
        self.fill = 0.0  # 0..1: the whole monitor grey to white (omeye latency)
        self.fill_changed = False
        self.add_css_class(f"omeye-{mode}")

        LayerShell.init_for_window(self)
        LayerShell.set_namespace(self, "omeye")
        LayerShell.set_layer(self, LayerShell.Layer.OVERLAY)
        for edge in (LayerShell.Edge.TOP, LayerShell.Edge.BOTTOM, LayerShell.Edge.LEFT, LayerShell.Edge.RIGHT):
            LayerShell.set_anchor(self, edge, True)
        LayerShell.set_exclusive_zone(self, -1)  # cover the bar too: the whole monitor
        LayerShell.set_keyboard_mode(
            self, LayerShell.KeyboardMode.EXCLUSIVE if mode == "calibrate" else LayerShell.KeyboardMode.NONE
        )
        monitors = self.get_display().get_monitors()
        for i in range(monitors.get_n_items()):
            if monitors.get_item(i).get_connector() == monitor:
                LayerShell.set_monitor(self, monitors.get_item(i))

        self.area = Gtk.DrawingArea()
        self.area.set_draw_func(self.draw)
        self.area.connect("resize", lambda _a, w, h: emit(event="ready", width=w, height=h))
        self.set_child(self.area)

        if mode == "calibrate":
            keys = Gtk.EventControllerKey()
            keys.connect("key-pressed", self.on_key)
            self.add_controller(keys)
        else:
            self.connect("map", lambda _w: self.get_surface().set_input_region(cairo.Region()))
        self.add_tick_callback(self.tick)

    def on_key(self, _ctl, keyval, _code, _state) -> bool:
        emit(event="key", key=Gdk.keyval_name(keyval))
        return True

    def tick(self, _widget, _clock) -> bool:
        if self.dot:
            self.area.queue_draw()
        return GLib.SOURCE_CONTINUE

    def handle(self, msg: dict) -> bool:
        cmd = msg.get("cmd")
        if cmd == "text":
            self.text = msg.get("text", "")
        elif cmd == "dot":
            self.dot = (msg["x"], msg["y"], time.monotonic(), msg.get("ms", 1000) / 1000) if "x" in msg else None
        elif cmd == "camera":
            if "jpeg" in msg:
                loader = GdkPixbuf.PixbufLoader.new_with_type("jpeg")
                loader.write(base64.b64decode(msg["jpeg"]))
                loader.close()
                self.camera = (loader.get_pixbuf(), msg.get("caption", ""), msg.get("place", "corner"))
            else:
                self.camera = None
        elif cmd == "point":
            self.point = (msg["x"], msg["y"]) if "x" in msg else None
        elif cmd == "rect":
            self.rect = (msg["x"], msg["y"], msg["w"], msg["h"], msg.get("label", ""), msg.get("state", "likely")) \
                if "x" in msg else None
        elif cmd == "grid":
            self.grid = (msg["cols"], msg["rows"], msg["fill"]) if "cols" in msg else None
        elif cmd == "gaze":
            self.gaze = (msg["x"], msg["y"], msg.get("state", "likely")) if "x" in msg else None
        elif cmd == "fill":
            self.fill = float(msg.get("level", 0))
            self.fill_changed = True
        elif cmd == "quit":
            self.get_application().quit()
        self.area.queue_draw()
        return GLib.SOURCE_REMOVE

    def draw(self, _area, cr: cairo.Context, width: int, height: int) -> None:
        if self.mode == "follow":
            cr.set_operator(cairo.OPERATOR_SOURCE)
            cr.set_source_rgba(0, 0, 0, 0)
            cr.paint()
            cr.set_operator(cairo.OPERATOR_OVER)
        if self.fill:
            cr.set_source_rgb(self.fill, self.fill, self.fill)
            cr.paint()
        if self.fill_changed:
            self.fill_changed = False
            emit(event="painted", t=time.monotonic())
        if self.grid:
            cols, rows, fill = self.grid
            cw, ch = width / cols, height / rows
            cr.set_line_width(1)
            for i, level in enumerate(fill):
                row, col = divmod(i, cols)
                cr.rectangle(col * cw + 3, row * ch + 3, cw - 6, ch - 6)
                cr.set_source_rgba(0.3, 0.85, 0.5, 0.05 + 0.3 * level)
                cr.fill_preserve()
                cr.set_source_rgba(1, 1, 1, 0.08)
                cr.stroke()
        if self.dot:
            x, y, start, seconds = self.dot
            k = min((time.monotonic() - start) / seconds, 1.0)
            radius = 30 - 22 * (1 - (1 - k) ** 2)
            cr.set_source_rgb(0.98, 0.78, 0.25)
            cr.arc(x, y, radius, 0, 6.2832)
            cr.fill()
            cr.set_source_rgb(0.07, 0.07, 0.09)
            cr.arc(x, y, 3, 0, 6.2832)
            cr.fill()
        if self.rect:
            x, y, w, h, label, state = self.rect
            cr.set_line_width(4)
            cr.set_source_rgba(*STATE_RGBA.get(state, STATE_RGBA["likely"]))
            cr.rectangle(x + 2, y + 2, w - 4, h - 4)
            cr.stroke()
            if label:
                cr.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                cr.set_font_size(16)
                cr.move_to(x + 12, y + 26)
                cr.show_text(label)
        if self.point:
            x, y = self.point
            cr.set_source_rgba(0.07, 0.07, 0.09, 0.6)
            cr.arc(x, y, 8, 0, 6.2832)
            cr.fill()
            cr.set_source_rgba(1, 1, 1, 0.7)
            cr.arc(x, y, 5, 0, 6.2832)
            cr.fill()
        if self.gaze:
            x, y, state = self.gaze
            cr.set_line_width(4)
            cr.set_source_rgba(*STATE_RGBA.get(state, STATE_RGBA["likely"]))
            cr.arc(x, y, 26, 0, 6.2832)
            cr.stroke()
        if self.text:
            self.draw_text(cr, width, height)
        if self.camera:
            self.draw_camera(cr, width, height)

    def draw_camera(self, cr: cairo.Context, width: int, height: int) -> None:
        pixbuf, caption, place = self.camera
        lines = caption.split("\n") if caption else []
        size, pad = 17, 12
        w = min(640, width * 0.4) if place == "center" else min(400, width * 0.25)
        scale = w / pixbuf.get_width()
        h = pixbuf.get_height() * scale
        box_h = h + pad + len(lines) * size * 1.4
        if place == "center":
            x = (width - w) / 2
            y = height * 0.56 if self.mode == "calibrate" else (height - box_h) / 2
        else:
            x, y = width - w - 24, height - box_h - 24
        cr.set_source_rgba(0.07, 0.07, 0.09, 0.88)
        cr.rectangle(x - pad, y - pad, w + 2 * pad, box_h + 2 * pad)
        cr.fill()
        cr.save()
        cr.translate(x, y)
        cr.scale(scale, scale)
        Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
        cr.paint()
        cr.restore()
        cr.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(size)
        cr.set_source_rgb(0.92, 0.92, 0.94)
        ty = y + h + pad + size
        for line in lines:
            cr.move_to(x, ty)
            cr.show_text(line)
            ty += size * 1.4

    def draw_text(self, cr: cairo.Context, width: int, height: int) -> None:
        size = 26 if self.mode == "calibrate" else 18
        cr.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(size)
        lines = self.text.split("\n")
        y = height * 0.38 - len(lines) * size * 0.7 if self.mode == "calibrate" else height - 24 - len(lines) * size * 1.4
        for line in lines:
            ext = cr.text_extents(line)
            x = (width - ext.width) / 2 if self.mode == "calibrate" else 24
            if self.mode == "follow":
                cr.set_source_rgba(0, 0, 0, 0.6)
                cr.rectangle(x - 6, y - size, ext.width + 12, size * 1.35)
                cr.fill()
            cr.set_source_rgb(0.92, 0.92, 0.94)
            cr.move_to(x, y)
            cr.show_text(line)
            y += size * 1.4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", required=True)
    ap.add_argument("--mode", choices=("calibrate", "follow"), default="calibrate")
    args = ap.parse_args()

    # Ctrl+C in the terminal reaches this process too; omeye closes the
    # overlay itself (stdin ends), so ignore it instead of dying with a traceback.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if not LayerShell.is_supported():
        print("omeye overlay: layer shell unsupported (is LD_PRELOAD set?)", file=sys.stderr)
        sys.exit(1)

    provider = Gtk.CssProvider()
    provider.load_from_data(CSS)
    app = Gtk.Application(flags=Gio.ApplicationFlags.NON_UNIQUE)

    def activate(app: Gtk.Application) -> None:
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), provider, 600)
        win = Overlay(app, args.monitor, args.mode)
        win.present()

        def read_stdin() -> None:
            for line in sys.stdin:
                try:
                    GLib.idle_add(win.handle, json.loads(line))
                except json.JSONDecodeError:
                    pass
            GLib.idle_add(app.quit)

        threading.Thread(target=read_stdin, daemon=True).start()

    app.connect("activate", activate)
    app.run([])


if __name__ == "__main__":
    main()
