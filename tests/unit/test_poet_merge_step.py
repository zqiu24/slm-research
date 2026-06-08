"""Tests for the poet_merge_step train_step wrapper helpers (CPU-only: the
module-level helpers must not import megatron)."""


def test_active_side_seeding_helper_sets_alt_state():
    """The merge patch exposes a pure helper that seeds alt_state from an iteration
    (so the layer/optimizer/merge all read the same active side)."""
    from poet_torch import alt_state

    from src.patches.poet_merge_step import _seed_active_side

    _seed_active_side(3)
    assert alt_state.get_iteration() == 3
    assert alt_state.active_side(1) == "in"
    _seed_active_side(4)
    assert alt_state.active_side(1) == "out"
