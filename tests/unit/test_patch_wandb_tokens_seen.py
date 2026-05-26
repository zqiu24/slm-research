"""Tests for training_log_wandb_tokens_seen patch registration."""

import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.training_log_wandb_tokens_seen", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.training_log_wandb_tokens_seen", None)


def test_wandb_tokens_seen_patch_registers():
    importlib.import_module("src.patches.training_log_wandb_tokens_seen")
    reg = registered_patches()
    assert "training_log_wandb_tokens_seen" in reg
