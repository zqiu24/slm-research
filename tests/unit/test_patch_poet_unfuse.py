"""Tests for poet_unfuse_te_impl patch registration."""

import importlib
import sys

import pytest

from src.patches import apply_patches, registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_unfuse_te_impl", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_unfuse_te_impl", None)


def test_patch_registers():
    importlib.import_module("src.patches.poet_unfuse_te_impl")
    reg = registered_patches()
    assert "poet_unfuse_te_impl" in reg
    targets = reg["poet_unfuse_te_impl"].targets
    assert any("transformer_config_from_args" in t for t in targets)


def test_apply_returns_hash():
    importlib.import_module("src.patches.poet_unfuse_te_impl")
    # Apply will try to import megatron — skip the side effect by monkeypatching
    # the underlying apply() to a no-op for this registration-only test.
    import src.patches._registry as reg_mod

    reg_mod._REGISTRY["poet_unfuse_te_impl"].apply_fn = lambda: None
    h = apply_patches(["poet_unfuse_te_impl"])
    assert len(h) == 16 and not h.startswith("noop")
