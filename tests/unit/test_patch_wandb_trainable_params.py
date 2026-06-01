"""Tests for the wandb_trainable_params patch (Megatron interceptor).

Import-safety + registration and the pure W&B-config payload builder are
CPU-testable here; the all-reduce and config.update wiring are exercised by a
GPU smoke run (see the spec).
"""

import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.wandb_trainable_params", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.wandb_trainable_params", None)


def test_patch_registers_with_empty_targets():
    # targets=() so the runtime setup_model_and_optimizer wrapper never raises a
    # PatchConflict against any static owner; import must be CPU-safe (no torch).
    importlib.import_module("src.patches.wandb_trainable_params")
    reg = registered_patches()
    assert "wandb_trainable_params" in reg
    assert reg["wandb_trainable_params"].targets == ()


def test_config_payload_normal():
    mod = importlib.import_module("src.patches.wandb_trainable_params")
    # 4 trainable oft_R of 104 total -> 3.8462%
    assert mod._config_payload(4, 104) == {
        "trainable_params": 4,
        "total_params": 104,
        "trainable_pct": 3.8462,
    }


def test_config_payload_full_trainable_is_100pct():
    mod = importlib.import_module("src.patches.wandb_trainable_params")
    assert mod._config_payload(15, 15)["trainable_pct"] == 100.0


def test_config_payload_zero_total_no_div_by_zero():
    mod = importlib.import_module("src.patches.wandb_trainable_params")
    assert mod._config_payload(0, 0) == {
        "trainable_params": 0,
        "total_params": 0,
        "trainable_pct": 0.0,
    }
