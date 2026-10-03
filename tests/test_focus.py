from omeye.focus import Vote, VoteParams, hit_test, on_screen
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


def run(vote, samples, focused="0xa", last_input=-99.0, last_mouse=-99.0):
    """samples: (time, target) pairs; returns the (time, address) switches."""
    out = []
    for t, target in samples:
        chosen = vote.step(t, target, focused, last_input, last_mouse)
        if chosen:
            out.append((round(t, 3), chosen))
            focused = chosen
    return out


def ticks(t0, t1, target, dt=1 / 30):
    n = int(round((t1 - t0) / dt))
    return [(t0 + i * dt, target) for i in range(n)]


def mixed(t0, t1, pattern, dt=1 / 30):
    """Targets cycling through `pattern`, one per frame."""
    n = int(round((t1 - t0) / dt))
    return [(t0 + i * dt, pattern[i % len(pattern)]) for i in range(n)]


def test_vote_switches_after_a_full_window():
    switches = run(Vote(VoteParams(window=0.4)), ticks(0, 1, "0xb"))
    assert len(switches) == 1
    assert 0.39 < switches[0][0] < 0.45 and switches[0][1] == "0xb"


def test_vote_survives_stray_frames():
    # 4 of 5 frames on 0xb, the rest jitter onto other windows or nowhere.
    switches = run(Vote(VoteParams(window=0.4, share=0.7)), mixed(0, 1, ["0xb", "0xb", "0xc", "0xb", None, "0xb", "0xb", "0xb", "0xb", "0xb"]))
    assert len(switches) == 1 and switches[0][1] == "0xb"


def test_split_votes_do_not_switch():
    assert run(Vote(VoteParams(window=0.4, share=0.7)), mixed(0, 2, ["0xb", "0xc"])) == []
    assert run(Vote(VoteParams(window=0.4, share=0.7)), mixed(0, 2, ["0xb", None])) == []


def test_short_glance_does_not_switch():
    assert run(Vote(VoteParams(window=0.4)), ticks(0, 0.3, "0xb") + ticks(0.3, 1, "0xa")) == []


def test_typing_and_mouse_hold_switching():
    p = VoteParams(window=0.4, typing_grace=0.7, mouse_grace=2.0)
    assert run(Vote(p), ticks(0, 1, "0xb"), last_input=0.5) == []
    # Typing stopped at 0.5: voting resumes at 1.2, the window is full at 1.6.
    switches = run(Vote(p), ticks(0, 2, "0xb"), last_input=0.5)
    assert switches and 1.59 < switches[0][0] < 1.65
    assert run(Vote(p), ticks(0, 2, "0xb"), last_mouse=0.5) == []


def test_cooldown_between_switches():
    v = Vote(VoteParams(window=0.1, cooldown=0.5))
    switches = run(v, ticks(0, 0.2, "0xb") + ticks(0.2, 1.2, "0xa"), focused="0xa")
    times = [t for t, _ in switches]
    assert len(times) == 2 and times[1] - times[0] >= 0.5


def test_leader():
    v = Vote(VoteParams(window=1.0))
    for t, target in mixed(0, 0.5, ["0xb", "0xb", "0xc", None]):
        v.step(t, target, "0xa", -99, -99)
    assert v.leader()[0] == "0xb" and abs(v.leader()[1] - 0.5) < 0.1


def test_on_screen_clamps_and_detects_looking_away():
    assert on_screen((1.04, -0.02), 0.15) == (1.0, 0.0)
    assert on_screen((0.5, 0.5), 0.15) == (0.5, 0.5)
    assert on_screen((0.5, 1.3), 0.15) is None
