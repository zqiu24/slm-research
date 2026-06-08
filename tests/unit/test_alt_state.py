"""The shared active-side signal: one iteration int, read by layer/optimizer/merge."""

from poet_torch import alt_state


def test_default_iteration_is_zero():
    alt_state.set_iteration(0)
    assert alt_state.get_iteration() == 0


def test_active_side_alternates_every_one():
    for it, expected in [(0, "out"), (1, "in"), (2, "out"), (3, "in")]:
        alt_state.set_iteration(it)
        assert alt_state.active_side(1) == expected


def test_active_side_holds_each_side_for_alternate_every():
    # alternate_every=2 -> out,out,in,in,out,out
    expected = ["out", "out", "in", "in", "out", "out"]
    for it, exp in enumerate(expected):
        alt_state.set_iteration(it)
        assert alt_state.active_side(2) == exp


def test_alternate_every_below_one_is_treated_as_one():
    alt_state.set_iteration(1)
    assert alt_state.active_side(0) == "in"
    assert alt_state.active_side(-5) == "in"
