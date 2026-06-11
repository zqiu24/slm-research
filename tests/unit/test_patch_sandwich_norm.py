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


def test_sandwich_targets_only_gpt_builder():
    # The patch owns ONLY gpt_builders.gpt_builder. The config-stamp is done via
    # a temporary wrapper inside the builder, so it does NOT own the
    # core_transformer_config_from_args target (which poet_unfuse_te_impl owns).
    importlib.import_module("src.patches.sandwich_norm_apply")
    reg = registered_patches()
    assert "sandwich_norm_apply" in reg
    assert reg["sandwich_norm_apply"].targets == ("gpt_builders.gpt_builder",)


def test_sandwich_composes_with_poet_patchset():
    """The full optim/poet patch-set plus sandwich_norm_apply must co-register
    without a PatchConflict (sandwich must NOT own core_transformer_config_from_args,
    which poet_unfuse_te_impl owns)."""
    from omegaconf import OmegaConf

    from launchers.submit import _register_experiment_patches

    cfg = OmegaConf.create(
        {
            "experiment": {
                "patches": [
                    "model_unfuse_linears",
                    "poet_optimizer_setup",
                    "poet_unfuse_te_impl",
                    "poet_apply_to_model",
                    "poet_merge_step",
                    "sandwich_norm_apply",
                ]
            }
        }
    )
    h = _register_experiment_patches(cfg)  # raises PatchConflict before the fix
    reg = registered_patches()
    assert "sandwich_norm_apply" in reg and "poet_unfuse_te_impl" in reg
    assert len(h) == 16 and not h.startswith("noop")
