"""Tests for poet_merge_step patch registration."""

import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_merge_step", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_merge_step", None)


def test_patch_registers_and_targets_train_step():
    importlib.import_module("src.patches.poet_merge_step")
    reg = registered_patches()
    assert "poet_merge_step" in reg
    assert any("training.train_step" in t for t in reg["poet_merge_step"].targets)


def test_run_merge_invalidates_cache_on_cached_poet_linear():
    """After merge_then_reinitialize, the layer's R cache must be cleared
    so the next forward recomputes against the new weight + new perms."""
    import torch
    import torch.nn as nn

    from src.optim import poet_cache as pc
    from src.optim.poet_layers import POETMegatronLinear
    from src.patches.poet_merge_step import _run_merge

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer._R_cache_version = 7
    layer._R_out_leaf = torch.zeros(2, 8, 8)
    layer._R_in_leaf = torch.zeros(1, 8, 8)
    pc.register_poet_layer(layer)

    wrapper = POETMegatronLinear(layer)
    model = nn.Module()
    model.fc = wrapper

    class _FakeDist:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_initialized():
            return False

    # Stub the merge math (touches torch.ops.poet, unavailable on CPU).
    layer.merge_then_reinitialize = lambda **kw: None
    _run_merge([model], _FakeDist, iteration=1)

    assert layer._R_cache_version == -1
    assert layer._R_out_leaf is None
    assert layer._R_in_leaf is None


def test_merge_decision_poet0_folds_every_step_reinits_every_400():
    from src.patches.poet_merge_step import _merge_decision

    # merge_period=1 (fold every step), reinit_period=400.
    assert _merge_decision(1, 1, 400) == (True, False)
    assert _merge_decision(399, 1, 400) == (True, False)
    assert _merge_decision(400, 1, 400) == (True, True)
    assert _merge_decision(800, 1, 400) == (True, True)


def test_merge_decision_legacy_folds_and_reinits_together():
    from src.patches.poet_merge_step import _merge_decision

    # merge_period=400, reinit_period=0 -> falls back to merge_period, so fold
    # and reinit always coincide (byte-identical to today's behavior).
    assert _merge_decision(200, 400, 0) == (False, False)
    assert _merge_decision(400, 400, 0) == (True, True)
    assert _merge_decision(800, 400, 0) == (True, True)


def test_merge_decision_disabled_or_iter_zero():
    from src.patches.poet_merge_step import _merge_decision

    assert _merge_decision(0, 1, 400) == (False, False)  # iteration 0 never merges
    assert _merge_decision(10, 0, 400) == (False, False)  # merge_period<=0 disables fold


def _make_reset_fixture():
    """Model with one oft_R param + a fake Megatron optimizer holding a separate
    fp32 master with nonzero Adam moments. Mirrors _iter_model_master_pairs'
    plain-Float16 layout (float16_groups / fp32_from_float16_groups)."""
    import torch
    import torch.nn as nn

    model = nn.Module()
    model.oft_R_in = nn.Parameter(torch.ones(4))  # the bf16 model tensor
    master = nn.Parameter(torch.full((4,), 3.0))  # separate fp32 master, nonzero
    torch_opt = torch.optim.Adam([master], lr=1e-3)
    torch_opt.state[master] = {
        "exp_avg": torch.ones(4),
        "exp_avg_sq": torch.ones(4),
        "step": torch.tensor(5.0),
    }

    class _FakeInner:
        def __init__(self):
            self.float16_groups = [[model.oft_R_in]]
            self.fp32_from_float16_groups = [[master]]
            self.optimizer = torch_opt

    return model, _FakeInner(), torch_opt, master


def test_reset_vanilla_oft_state_keeps_moments_when_reset_moments_false():
    import torch

    from src.patches.poet_merge_step import _reset_vanilla_oft_state

    model, opt, torch_opt, master = _make_reset_fixture()
    _reset_vanilla_oft_state(opt, model, iteration=5, reset_moments=False)

    # Master VALUE always zeroed (prevents spring-back) ...
    assert torch.count_nonzero(master.data) == 0
    # ... but momentum + step preserved (poet0 persists momentum).
    assert torch.count_nonzero(torch_opt.state[master]["exp_avg"]) == 4
    assert torch_opt.state[master]["step"].item() == 5.0


def test_reset_vanilla_oft_state_zeros_moments_when_reset_moments_true():
    import torch

    from src.patches.poet_merge_step import _reset_vanilla_oft_state

    model, opt, torch_opt, master = _make_reset_fixture()
    _reset_vanilla_oft_state(opt, model, iteration=400, reset_moments=True)

    assert torch.count_nonzero(master.data) == 0
    assert torch.count_nonzero(torch_opt.state[master]["exp_avg"]) == 0
    assert torch_opt.state[master]["step"].item() == 0.0


def test_run_merge_forwards_reinit_perm_false_keeps_perm():
    import torch
    import torch.nn as nn
    from poet_torch import POETLinear

    from src.optim.poet_layers import POETMegatronLinear
    from src.patches.poet_merge_step import _run_merge

    torch.manual_seed(0)
    pl = POETLinear(
        in_features=8,
        out_features=8,
        block_count=1,
        dtype=torch.float32,
        parameterization="exp",
    )
    pl.random_init_parameters()
    wrapper = POETMegatronLinear(pl)
    model = nn.Module()
    model.layer = wrapper

    perm_in_before = pl.perm_in.clone()

    class _FakeDist:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_initialized():
            return False

    _run_merge(model, _FakeDist(), iteration=5, reinit_perm=False)

    assert torch.equal(pl.perm_in, perm_in_before)
    assert torch.count_nonzero(pl.oft_R_in) == 0


def test_merge_decision_never_reinit_when_reinit_period_negative():
    from src.patches.poet_merge_step import _merge_decision

    # reinit_period < 0 -> fold EVERY step, NEVER reinit (no Ψ resample, no
    # momentum reset): constant-merge persistent-momentum mode for block_count=1.
    assert _merge_decision(1, 1, -1) == (True, False)
    assert _merge_decision(20, 1, -1) == (True, False)
    assert _merge_decision(400, 1, -1) == (True, False)
    assert _merge_decision(999, 1, -1) == (True, False)
