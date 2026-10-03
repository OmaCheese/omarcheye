from omarcheye.hypr import parse_layout, parse_monitors


def monitor(name, x, ws, scale=1.0, transform=0, special=0, focused=False):
    return {"name": name, "x": x, "y": 0, "width": 3840, "height": 2160, "scale": scale,
            "transform": transform, "disabled": False, "focused": focused,
            "activeWorkspace": {"id": ws, "name": str(ws)}, "specialWorkspace": {"id": special, "name": ""}}


def client(addr, ws, at, size, floating=False, fullscreen=0, history=0):
    return {"address": addr, "mapped": True, "hidden": False, "at": at, "size": size,
            "workspace": {"id": ws, "name": str(ws)}, "floating": floating,
            "fullscreen": fullscreen, "focusHistoryID": history, "class": addr}


def test_monitor_logical_size():
    (m,) = parse_monitors([monitor("HDMI-A-1", 0, 2, scale=1.25)])
    assert (m.w, m.h) == (3072, 1728)
    (r,) = parse_monitors([monitor("DP-1", 0, 2, scale=2, transform=1)])
    assert (r.w, r.h) == (1080, 1920)
    assert m.to_global(0.5, 0.5) == (1536, 864)
    assert m.to_local(1536, 864) == (0.5, 0.5)


def test_layout_keeps_visible_windows_floating_first():
    mons = [monitor("HDMI-A-1", 0, 2, focused=True), monitor("eDP-1", 3840, 5, special=-98)]
    clients = [
        client("0x1", 2, [0, 0], [1900, 2160], history=1),
        client("0x2", 2, [1920, 0], [1900, 2160], history=0),
        client("0x3", 3, [0, 0], [3840, 2160]),  # hidden workspace
        client("0x4", 2, [500, 500], [800, 600], floating=True, history=2),
        client("0x5", -98, [4000, 100], [1000, 800], floating=True, history=3),  # visible scratchpad
    ]
    layout = parse_layout(mons, clients, {"address": "0x2"})
    assert [w.address for w in layout.windows] == ["0x4", "0x5", "0x2", "0x1"]
    assert layout.focused == "0x2"


def test_fullscreen_hides_its_workspace():
    mons = [monitor("HDMI-A-1", 0, 2)]
    clients = [
        client("0x1", 2, [0, 0], [1900, 2160]),
        client("0x2", 2, [0, 0], [3840, 2160], fullscreen=2),
    ]
    layout = parse_layout(mons, clients, {})
    assert [w.address for w in layout.windows] == ["0x2"]
    assert layout.focused is None
