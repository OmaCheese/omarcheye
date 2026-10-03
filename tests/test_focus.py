import numpy as np

from omeye.focus import AWAY, Belief, BeliefParams, Glance, on_screen, runner_up, window_chances
from omeye.hypr import Window

W, H = 3072, 1728  # the 4K monitor at scale 1.25, logical pixels
GAP = 10
# 2x2 tiles with gaps, like dwindle with four windows
TILES = [Window(f"0x{i}", c * W / 2 + GAP, r * H / 2 + GAP, W / 2 - 2 * GAP, H / 2 - 2 * GAP, False, i)
         for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])]
SIGMA = 0.051 * W  # calibration error 6.4% of the width, per axis


def test_on_screen_clamps_and_detects_looking_away():
    assert on_screen((1.04, -0.02), 0.15) == (1.0, 0.0)
    assert on_screen((0.5, 0.5), 0.15) == (0.5, 0.5)
    assert on_screen((0.5, 1.3), 0.15) is None


def test_chances_in_the_middle_and_on_a_border():
    mid = window_chances(TILES, W * 0.75, H * 0.25, SIGMA)
    assert mid["0x1"] > 0.85 and abs(sum(mid.values()) - 1) < 1e-9
    border = window_chances(TILES, W * 0.5, H * 0.25, SIGMA)
    assert abs(border["0x0"] - border["0x1"]) < 0.01 and border["0x0"] > 0.4


def test_floating_window_owns_its_area():
    floating = Window("0xf", W * 0.4, H * 0.4, W * 0.2, H * 0.2, True, 0)
    c = window_chances([floating] + TILES, W * 0.5, H * 0.5, 0.02 * W)
    assert c["0xf"] > 0.95
    assert abs(sum(c.values()) - 1) < 1e-6


def test_off_the_windows_is_away():
    c = window_chances(TILES[:1], W * 0.9, H * 0.9, 0.02 * W)
    assert c[AWAY] > 0.99


def look(belief, t0, t1, points, focused="0x0", rng=None, jitter=0.035 * W, last_input=-99.0, dt=1 / 30, avoid=()):
    """Feed frames looking at `points` (cycled) with jitter; return the switches."""
    rng = rng or np.random.default_rng(0)
    out = []
    t = t0
    i = 0
    while t < t1:
        x, y = points[i % len(points)]
        c = window_chances(TILES, x + rng.normal(0, jitter), y + rng.normal(0, jitter), SIGMA)
        chosen = belief.step(t, c, focused, last_input, -99.0, avoid=avoid)
        if chosen:
            out.append((round(t, 2), chosen))
            focused = chosen
        t += dt
        i += 1
    return out


def test_switches_to_the_window_you_look_at():
    b = Belief(BeliefParams())
    look(b, 0, 1, [(W * 0.25, H * 0.25)])  # settle on the focused window
    switches = look(b, 1, 3, [(W * 0.75, H * 0.75)])
    assert len(switches) == 1 and switches[0][1] == "0x3"
    assert switches[0][0] < 1.0 + 1.0  # within a second of looking there


def test_steady_offset_inside_the_window_still_lands():
    # The calibration's error is mostly a steady offset: aim 8% of the width off the window's centre.
    b = Belief(BeliefParams())
    switches = look(b, 0, 2, [(W * 0.75 - 0.08 * W, H * 0.25 + 0.05 * W)])
    assert [s for _, s in switches] == ["0x1"]


def test_gaze_on_a_border_rarely_flips():
    # A minute of staring at the border between two windows, with frame jitter
    # that knows nothing of the border: at most a few flips.
    flips = look(Belief(BeliefParams()), 0, 60, [(W * 0.5, H * 0.25)], focused="0x0")
    assert len(flips) <= 6


def test_a_flick_of_the_eyes_does_not_switch():
    b = Belief(BeliefParams())
    # 2 frames (0.07 s) on the other window, 0.4 s back, repeatedly
    points = [(W * 0.75, H * 0.25)] * 2 + [(W * 0.25, H * 0.25)] * 12
    assert look(b, 0, 4, points) == []


def test_without_the_quick_path_a_glance_does_not_switch():
    b = Belief(BeliefParams(quick=1.01))
    points = [(W * 0.75, H * 0.25)] * 6 + [(W * 0.25, H * 0.25)] * 12  # 0.2 s glances
    assert look(b, 0, 4, points) == []


def test_looking_clearly_into_a_window_switches_quickly():
    b = Belief(BeliefParams())
    look(b, 0, 1, [(W * 0.25, H * 0.25)])
    switches = look(b, 1, 3, [(W * 0.75, H * 0.75)])
    assert switches and switches[0][0] - 1 < 0.2


def test_typing_holds_switching_but_keeps_tracking():
    b = Belief(BeliefParams(typing_grace=0.7))
    assert look(b, 0, 2, [(W * 0.75, H * 0.75)], last_input=1.9) == []
    assert b.top()[0] == "0x3" and b.top()[1] > 0.8


def test_no_face_drifts_to_away_and_windows_can_appear():
    b = Belief(BeliefParams())
    look(b, 0, 1, [(W * 0.25, H * 0.25)])
    for k in range(60):
        b.observe(1 + k / 30, {**{w.address: 0.0 for w in TILES}, AWAY: 1.0})
    assert b.top()[0] is AWAY
    b.observe(3.0, {"0xnew": 0.9, AWAY: 0.1})
    assert set(b.b) == {"0xnew", AWAY}


# A glance away and back after a switch: retry.

ASPECT = H / W


def run_glance(g, frames, t0=0.0, dt=1 / 30):
    """Feed (point or None) frames; return the times a glance finished."""
    out = []
    for i, p in enumerate(frames):
        if g.update(t0 + i * dt, p):
            out.append(round(t0 + i * dt, 2))
    return out


def test_a_glance_up_and_back_is_noticed():
    g = Glance(ASPECT)
    g.arm(0.0, (0.3, 0.4))
    here, up = (0.3, 0.4), (0.32, -0.1)
    done = run_glance(g, [here] * 10 + [up] * 9 + [here] * 3)
    assert len(done) == 1 and abs(g.spot[0] - 0.3) < 0.02 and not g.armed


def test_a_glance_without_a_face_counts_and_one_stray_frame_does_not():
    g = Glance(ASPECT)
    g.arm(0.0, (0.3, 0.4))
    assert run_glance(g, [(0.3, 0.4)] * 5 + [(0.9, 0.4)] + [(0.3, 0.4)] * 10) == []
    assert run_glance(g, [None] * 8 + [(0.3, 0.4)] * 3, t0=1.0) != []


def test_looking_away_for_long_is_a_move_not_a_glance():
    g = Glance(ASPECT, longest=0.7)
    g.arm(0.0, (0.3, 0.4))
    assert run_glance(g, [(0.3, 0.4)] * 12 + [(0.8, 0.4)] * 30 + [(0.3, 0.4)] * 5) == []
    assert not g.armed


def test_a_look_down_at_the_keyboard_is_not_a_glance():
    g = Glance(ASPECT)
    g.arm(0.0, (0.3, 0.4))
    assert run_glance(g, [(0.3, 0.4)] * 12 + [(0.3, 1.2)] * 9 + [(0.3, 0.4)] * 3) == []
    assert g.armed  # still there for a real glance


def test_the_estimate_settling_after_the_switch_is_not_a_glance():
    # Right after a quick switch the estimate is still landing: it wanders off
    # the first anchor and back within the first 0.3 s.
    g = Glance(ASPECT)
    g.arm(0.0, (0.3, 0.4))
    assert run_glance(g, [(0.45, 0.4)] * 5 + [(0.3, 0.4)] * 20) == []
    assert g.armed


def test_no_glance_after_the_retry_window():
    g = Glance(ASPECT, window=2.0)
    g.arm(0.0, (0.3, 0.4))
    assert run_glance(g, [(0.3, 0.4)] * 70 + [(0.3, -0.1)] * 9 + [(0.3, 0.4)] * 3) == []


def test_runner_up_is_the_neighbour_nearest_the_estimate():
    # Estimate in the top-left tile, near its right edge: the top-right tile.
    assert runner_up(TILES, W * 0.45, H * 0.25, SIGMA, ["0x0"]) == "0x1"
    # ... near its bottom edge: the bottom-left tile.
    assert runner_up(TILES, W * 0.25, H * 0.45, SIGMA, ["0x0"]) == "0x2"
    assert runner_up(TILES, W * 0.45, H * 0.25, SIGMA, ["0x0", "0x1"]) in ("0x2", "0x3")
    assert runner_up(TILES[:1], W * 0.25, H * 0.25, SIGMA, ["0x0"]) is None


def test_settle_and_avoid_keep_a_retry_from_being_undone():
    b = Belief(BeliefParams())
    look(b, 0, 1, [(W * 0.25, H * 0.25)])  # the estimate says top-left...
    b.settle("0x1")  # ... but a glance sent focus to top-right
    assert b.top() == ("0x1", b.top()[1]) and b.top()[1] > 0.95
    assert look(b, 1, 2, [(W * 0.25, H * 0.25)], focused="0x1", avoid=("0x0",)) == []
