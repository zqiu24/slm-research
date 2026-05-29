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
