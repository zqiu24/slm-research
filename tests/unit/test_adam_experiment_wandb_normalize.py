"""The adam/champion experiments must enable wandb_metric_normalize, and the
logging patches must co-register without a PatchConflict (wandb_metric_normalize
wraps training_log via a runtime monkeypatch declared with targets=())."""

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
        "src.patches.wandb_metric_normalize",
        "src.patches.training_log_eta",
    ):
        sys.modules.pop(name, None)
    yield
    _reset_for_tests()


@pytest.mark.parametrize("rel", ["optim/adam.yaml", "champion.yaml"])
def test_experiment_lists_wandb_normalize(rel):
    cfg = OmegaConf.load(_REPO / "configs" / "experiments" / rel)
    assert "wandb_metric_normalize" in list(cfg.experiment.patches)


def test_logging_patches_register_without_conflict():
    # training_log_eta (wraps print_rank_last) and wandb_metric_normalize
    # (targets=(), wraps training_log at runtime) must co-register cleanly.
    # _register_experiment_patches imports + hashes them (no apply, CPU-safe).
    from launchers.submit import _register_experiment_patches
    from src.patches import registered_patches

    cfg = OmegaConf.create(
        {"experiment": {"patches": ["training_log_eta", "wandb_metric_normalize"]}}
    )
    h = _register_experiment_patches(cfg)
    reg = registered_patches()
    assert "wandb_metric_normalize" in reg and "training_log_eta" in reg
    assert reg["wandb_metric_normalize"].targets == ()
    assert len(h) == 16 and not h.startswith("noop")
