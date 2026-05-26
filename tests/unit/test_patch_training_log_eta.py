"""Tests for training_log_eta patch registration."""

import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.training_log_eta", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.training_log_eta", None)


def test_eta_patch_registers():
    importlib.import_module("src.patches.training_log_eta")
    reg = registered_patches()
    assert "training_log_eta" in reg
    assert any("training.training_log" in t for t in reg["training_log_eta"].targets)
