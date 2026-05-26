"""Tests for src/optim get_optimizer dispatcher."""

import pytest
import torch

from src.optim import OptimizerCfg, get_optimizer


def _dummy_params():
    return [torch.nn.Parameter(torch.zeros(4, 4))]


def test_dispatch_adam_returns_torch_adam():
    cfg = OptimizerCfg(kind="adam", lr=1e-4)
    opt = get_optimizer(cfg, _dummy_params(), mcore_cfg=None)
    assert isinstance(opt, torch.optim.Adam | torch.optim.AdamW)


def test_dispatch_adamw_returns_torch_adamw():
    cfg = OptimizerCfg(kind="adamw", lr=1e-4)
    opt = get_optimizer(cfg, _dummy_params(), mcore_cfg=None)
    assert isinstance(opt, torch.optim.AdamW)


def test_dispatch_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown optimizer kind"):
        OptimizerCfg(kind="nonexistent", lr=1e-4)


def test_poet_kind_is_known_but_routes_via_dedicated_builder():
    """``OptimizerCfg(kind='poet', ...)`` validates. POET dispatch goes
    through ``src.optim.poet.get_megatron_poet_optimizer`` (Task 4);
    ``get_optimizer`` raises on direct ``poet`` dispatch.
    """
    cfg = OptimizerCfg(kind="poet", lr=1e-4, poet_merge_period=100)
    assert cfg.kind == "poet"
    with pytest.raises(ValueError, match="POET"):
        get_optimizer(cfg, _dummy_params(), mcore_cfg=None)
