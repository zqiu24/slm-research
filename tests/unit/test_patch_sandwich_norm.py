"""Tests for the sandwich_norm_apply patch registration."""

import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.sandwich_norm_apply", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.sandwich_norm_apply", None)


def test_patch_registers_on_arguments_and_gpt_builder():
    importlib.import_module("src.patches.sandwich_norm_apply")
    reg = registered_patches()
    assert "sandwich_norm_apply" in reg
    targets = reg["sandwich_norm_apply"].targets
    assert any("gpt_builder" in t for t in targets)
    assert any("core_transformer_config_from_args" in t for t in targets)
