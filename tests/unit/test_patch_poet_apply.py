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
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_apply_to_model", None)


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


def test_wrapped_get_model_runs_split_before_poet(monkeypatch):
    """The wrapped get_model calls split_fused_linears (with the split flags)
    before replace_linears_with_poet.

    Requires a real megatron.training import (transformer_engine / CUDA), so it
    is skipped on CPU-only environments. The Task 4-7 poet_split tests are the
    authoritative correctness coverage; this test only checks wiring order.
    """
    try:
        import megatron.training  # noqa: F401
    except Exception as exc:  # OSError (TE .so) as well as ImportError
        pytest.skip(f"megatron.training not importable: {exc}")

    import src.optim.poet_layers as pl
    import src.optim.poet_split as ps

    calls = []

    class _Args:
        poet = True
        poet_block_size = 16
        poet_block_count = None
        poet_init_type = "none"
        poet_mup_alpha = 1.0
        poet_cache_mode = "none"
        poet_split_qkv = True
        poet_split_fc1 = True

    class _StubModel:
        def parameters(self):
            return iter(())

    import megatron.training as mtt

    monkeypatch.setattr(mtt, "get_args", lambda: _Args(), raising=False)
    monkeypatch.setattr(
        ps, "split_fused_linears", lambda model, **kw: calls.append(("split", kw)) or 0
    )
    monkeypatch.setattr(
        pl,
        "replace_linears_with_poet",
        lambda model, **kw: calls.append(("replace", kw)) or 0,
    )

    from megatron.training import training as _mt

    import src.patches.poet_apply_to_model as mod

    monkeypatch.setattr(_mt, "get_model", lambda *a, **k: _StubModel(), raising=False)
    mod.apply()
    _mt.get_model("x")

    assert [c[0] for c in calls] == ["split", "replace"]
    assert calls[0][1]["split_qkv"] is True
    assert calls[0][1]["split_fc1"] is True
