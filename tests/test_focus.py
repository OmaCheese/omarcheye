from omeye.focus import Dwell, DwellParams, hit_test
from omeye.hypr import Window

LEFT = Window("0xa", 0, 0, 1000, 1000, False, 0)
RIGHT = Window("0xb", 1000, 0, 1000, 1000, False, 1)
FLOAT = Window("0xf", 800, 400, 400, 200, True, 2)


def test_hit_test_margin_and_hysteresis():
    wins = [LEFT, RIGHT]
    # Just past the border: still the focused window, not the neighbour.
    assert hit_test(wins, 1030, 500, "0xa", 50) == "0xa"
    # Well inside the neighbour: the neighbour.
    assert hit_test(wins, 1080, 500, "0xa", 50) == "0xb"
    # In the neighbour's margin with the other window focused: nobody.
    assert hit_test(wins, 1030, 500, None, 50) is None


def test_hit_test_floating_on_top():
    wins = [FLOAT, LEFT, RIGHT]
    assert hit_test(wins, 1000, 500, "0xa", 10) == "0xf"


def run(dwell, samples, focused="0xa", last_input=-99.0, last_mouse=-99.0):
    """samples: (time, target) pairs; returns the (time, address) switches."""
    out = []
    for t, target in samples:
        chosen = dwell.step(t, target, focused, last_input, last_mouse)
        if chosen:
            out.append((round(t, 3), chosen))
            focused = chosen
    return out


def ticks(t0, t1, target, dt=1 / 30):
    n = int(round((t1 - t0) / dt))
    return [(t0 + i * dt, target) for i in range(n)]


def test_dwell_switches_after_dwell_time():
    d = Dwell(DwellParams(dwell=0.4))
    switches = run(d, ticks(0, 1, "0xb"))
    assert len(switches) == 1
    assert 0.39 < switches[0][0] < 0.45 and switches[0][1] == "0xb"


def test_short_glance_does_not_switch():
    d = Dwell(DwellParams(dwell=0.4, gap=0.15))
    assert run(d, ticks(0, 0.3, "0xb") + ticks(0.3, 1, "0xa")) == []


def test_blink_gap_keeps_counting():
    d = Dwell(DwellParams(dwell=0.4, gap=0.15))
    samples = ticks(0, 0.2, "0xb") + ticks(0.2, 0.3, None) + ticks(0.3, 0.6, "0xb")
    switches = run(d, samples)
    assert len(switches) == 1 and switches[0][0] < 0.45


def test_typing_and_mouse_hold_switching():
    p = DwellParams(dwell=0.4, typing_grace=0.7, mouse_grace=2.0)
    assert run(Dwell(p), ticks(0, 1, "0xb"), last_input=0.5) == []
    # Typing stopped at 0.5: switching resumes at 1.2, dwell ends at 1.6.
    switches = run(Dwell(p), ticks(0, 2, "0xb"), last_input=0.5)
    assert switches and 1.59 < switches[0][0] < 1.65
    assert run(Dwell(p), ticks(0, 2, "0xb"), last_mouse=0.5) == []


def test_cooldown_between_switches():
    d = Dwell(DwellParams(dwell=0.1, cooldown=0.5, gap=0.0))
    switches = run(d, ticks(0, 0.2, "0xb") + ticks(0.2, 1.2, "0xa"), focused="0xa")
    times = [t for t, _ in switches]
    assert len(times) == 2 and times[1] - times[0] >= 0.5
