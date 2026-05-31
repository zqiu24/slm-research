"""The adam/champion experiments must enable wandb_metric_normalize, and it must
co-register with log_grad_norm_extra (both touch training_log) without conflict."""

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.patches._registry import _reset_for_tests

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    for name in (
        "src.patches.log_grad_norm_extra",
        "src.patches.wandb_metric_normalize",
        "src.patches.training_log_eta",
        "src.patches.model_unfuse_linears",
    ):
        sys.modules.pop(name, None)
    yield
    _reset_for_tests()


@pytest.mark.parametrize("rel", ["optim/adam.yaml", "champion.yaml"])
def test_experiment_lists_wandb_normalize(rel):
    cfg = OmegaConf.load(_REPO / "configs" / "experiments" / rel)
    assert "wandb_metric_normalize" in list(cfg.experiment.patches)


def test_wandb_normalize_composes_with_grad_norm_extra():
    # Both wrap training_log; one declares the target, the other targets=().
    # _register_experiment_patches imports + hashes them (no apply, CPU-safe).
    from launchers.submit import _register_experiment_patches
    from src.patches import registered_patches

    cfg = OmegaConf.create(
        {"experiment": {"patches": ["log_grad_norm_extra", "wandb_metric_normalize"]}}
    )
    h = _register_experiment_patches(cfg)
    reg = registered_patches()
    assert "wandb_metric_normalize" in reg and "log_grad_norm_extra" in reg
    assert len(h) == 16 and not h.startswith("noop")
