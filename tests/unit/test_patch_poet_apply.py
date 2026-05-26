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
