"""The per-rank launcher applies experiment patches PLUS always-on patches.

``wandb_trainable_params`` must run on every Megatron run regardless of the
experiment's patch list. The resolution is a pure function so it's CPU-testable
without importing megatron (which the actual apply step needs).
"""

from __future__ import annotations

from omegaconf import OmegaConf

from launchers.pretrain_gpt_slm import _resolve_runtime_patch_names


def test_empty_experiment_patches_still_gets_always_on():
    cfg = OmegaConf.create({"experiment": {"patches": []}})
    assert _resolve_runtime_patch_names(cfg) == ["wandb_trainable_params"]


def test_experiment_patches_are_kept_and_always_on_appended():
    cfg = OmegaConf.create({"experiment": {"patches": ["poet_merge_step", "training_log_eta"]}})
    names = _resolve_runtime_patch_names(cfg)
    assert names[:2] == ["poet_merge_step", "training_log_eta"]
    assert "wandb_trainable_params" in names


def test_always_on_not_duplicated_if_experiment_lists_it():
    cfg = OmegaConf.create({"experiment": {"patches": ["wandb_trainable_params"]}})
    assert _resolve_runtime_patch_names(cfg) == ["wandb_trainable_params"]


def test_missing_experiment_block_defaults_to_always_on():
    cfg = OmegaConf.create({})
    assert _resolve_runtime_patch_names(cfg) == ["wandb_trainable_params"]
