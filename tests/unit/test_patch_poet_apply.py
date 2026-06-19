"""Tests for poet_apply_to_model patch registration."""

import importlib
import sys

import pytest

from src.patches import apply_patches, registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_apply_to_model", None)
    sys.modules.pop("src.patches.model_unfuse_linears", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_apply_to_model", None)
    sys.modules.pop("src.patches.model_unfuse_linears", None)


def test_patch_registers():
    importlib.import_module("src.patches.poet_apply_to_model")
    reg = registered_patches()
    assert "poet_apply_to_model" in reg
    targets = reg["poet_apply_to_model"].targets
    assert any("training.training.get_model" in t for t in targets)


def test_apply_returns_hash():
    importlib.import_module("src.patches.poet_apply_to_model")
    import src.patches._registry as reg_mod

    reg_mod._REGISTRY["poet_apply_to_model"].apply_fn = lambda: None
    h = apply_patches(["poet_apply_to_model"])
    assert len(h) == 16 and not h.startswith("noop")


def test_unfuse_patch_registers_on_model_provider():
    """The (separate, optimizer-agnostic) unfuse patch targets model_provider,
    distinct from poet_apply_to_model's get_model target (so no PatchConflict)."""
    importlib.import_module("src.patches.model_unfuse_linears")
    reg = registered_patches()
    assert "model_unfuse_linears" in reg
    targets = reg["model_unfuse_linears"].targets
    assert any("model_provider" in t for t in targets)
    assert not any("get_model" in t for t in targets)


def test_apply_to_chunk_forwards_one_sided_and_pins_alt_state(monkeypatch):
    from types import SimpleNamespace

    from poet_torch import alt_state

    import src.patches.poet_apply_to_model as ap

    captured = {}

    def _fake_replace(m, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr(ap, "replace_linears_with_poet", _fake_replace)
    alt_state.set_fixed_side(None)

    args = SimpleNamespace(
        poet_block_size=256,
        poet_block_count=1,
        poet_single_step_x=True,
        poet_single_step_x_one_sided="in",
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=None,
    )
    try:
        ap._apply_poet_to_chunk(object(), args)
        assert captured["single_step_x_one_sided"] == "in"
        alt_state.set_iteration(0)  # would be "out" under the toggle
        assert alt_state.active_side(1) == "in"
    finally:
        alt_state.set_fixed_side(None)


def test_apply_to_chunk_leaves_alt_state_unpinned_when_unset(monkeypatch):
    from types import SimpleNamespace

    from poet_torch import alt_state

    import src.patches.poet_apply_to_model as ap

    monkeypatch.setattr(ap, "replace_linears_with_poet", lambda m, **kw: 0)
    alt_state.set_fixed_side(None)

    args = SimpleNamespace(
        poet_block_size=256,
        poet_block_count=1,
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=None,
    )
    ap._apply_poet_to_chunk(object(), args)
    alt_state.set_iteration(0)
    assert alt_state.active_side(1) == "out"  # toggle intact
