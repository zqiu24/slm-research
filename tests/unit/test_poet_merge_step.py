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


def test_profile_target_iteration_parses_env(monkeypatch):
    from src.patches.poet_merge_step import _profile_target_iteration

    monkeypatch.delenv("POET_PROFILE_STEP", raising=False)
    assert _profile_target_iteration() is None

    monkeypatch.setenv("POET_PROFILE_STEP", "20")
    assert _profile_target_iteration() == 20

    monkeypatch.setenv("POET_PROFILE_STEP", "0")
    assert _profile_target_iteration() is None  # non-positive -> off

    monkeypatch.setenv("POET_PROFILE_STEP", "notanint")
    assert _profile_target_iteration() is None  # malformed -> off


def test_torch_profile_enabled(monkeypatch):
    from src.patches.poet_merge_step import _torch_profile_enabled

    monkeypatch.delenv("POET_PROFILE_TORCH", raising=False)
    assert _torch_profile_enabled() is False
    monkeypatch.setenv("POET_PROFILE_TORCH", "1")
    assert _torch_profile_enabled() is True
    monkeypatch.setenv("POET_PROFILE_TORCH", "TRUE")
    assert _torch_profile_enabled() is True
    monkeypatch.setenv("POET_PROFILE_TORCH", "0")
    assert _torch_profile_enabled() is False


def test_dominant_phase_picks_largest_leaf():
    from src.patches.poet_merge_step import _dominant_phase

    assert _dominant_phase({}) is None
    # train_step_total is the sum and must be excluded from the leaf comparison.
    timings = {"train_step_total": 100.0, "forward_backward": 70.0, "optimizer": 25.0, "merge": 5.0}
    assert _dominant_phase(timings) == "forward_backward"
    assert _dominant_phase({"optimizer": 9.0, "merge": 40.0}) == "merge"


def test_format_profile_orders_and_labels():
    from src.patches.poet_merge_step import _format_profile

    out = _format_profile(
        {"train_step_total": 100.0, "forward_backward": 70.0, "optimizer": 25.0, "merge": 5.0}
    )
    assert "[POET-PROFILE]" in out
    # fixed order: train_step_total before forward_backward before optimizer before merge
    assert (
        out.index("train_step_total")
        < out.index("forward_backward")
        < out.index("optimizer")
        < out.index("merge")
    )
    assert "dominant component: forward_backward" in out
