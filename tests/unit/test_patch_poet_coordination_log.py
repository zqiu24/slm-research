# tests/unit/test_patch_poet_coordination_log.py
"""Patch-glue coverage for poet_coordination_log: registration, the
setup_model_and_optimizer wrap + step-hook, and the fake-wandb log payload.
Mirrors test_patch_poet_grad_conditioning.py (the pattern this patch follows)."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SLM_POET_COORD_DIAG", raising=False)
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_coordination_log", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_coordination_log", None)


def test_patch_registers_with_no_owned_target():
    importlib.import_module("src.patches.poet_coordination_log")
    reg = registered_patches()
    assert "poet_coordination_log" in reg
    # runtime wrapper of setup_model_and_optimizer -> owns no static target.
    assert reg["poet_coordination_log"].targets == ()


def test_install_on_setup_wraps_and_hooks():
    import torch

    from src.patches.poet_coordination_log import _install_coordination_on_setup

    class FakeOpt:
        def __init__(self):
            self.n = 0

        def step(self, *a, **k):
            self.n += 1

    class FakePOET(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.block_size_in = b
            self.block_size_out = b
            self.oft_R_in = torch.nn.Parameter(torch.zeros(1, b * (b - 1) // 2))
            self.oft_R_out = torch.nn.Parameter(torch.zeros(1, b * (b - 1) // 2))

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = FakePOET(4)  # name matches _WANTED, two-sided

    model, opt = FakeModel(), FakeOpt()

    def orig_setup(*a, **k):
        return model, opt, "sched"

    wrapped = _install_coordination_on_setup(orig_setup, interval=1)
    m, o, s = wrapped()
    assert (m, o, s) == (model, opt, "sched")
    # step-hook installed; empty lie-state lookup -> logs nothing, never crashes.
    o.step()
    o.step()
    assert opt.n == 2  # underlying .step still runs through the wrapper


def test_log_coordination_emits_per_layer_and_mean(monkeypatch):
    import sys as _sys
    import types

    import torch

    from src.patches.poet_coordination_log import _build_lie_state_lookup, _log_coordination

    captured = {}
    fake_wandb = types.SimpleNamespace(run=object(), log=lambda d, step=None: captured.update(d))
    monkeypatch.setitem(_sys.modules, "wandb", fake_wandb)

    b = 4
    ne = b * (b - 1) // 2
    oft_in = torch.nn.Parameter(torch.zeros(2, ne))
    oft_in.main_grad = torch.randn(2, ne)
    oft_out = torch.nn.Parameter(torch.zeros(2, ne))
    oft_out.main_grad = torch.randn(2, ne)
    layer = SimpleNamespace(
        oft_R_in=oft_in,
        oft_R_out=oft_out,
        weight=torch.randn(2 * b, 2 * b),
        perm_out_inv=torch.arange(2 * b, dtype=torch.int32),
        perm_in_inv=torch.arange(2 * b, dtype=torch.int32),
        block_size_in=b,
        block_size_out=b,
    )
    # FP32 layout (master == model): no float16 groups -> _iter_model_master_pairs
    # yields (p, p) from optimizer.param_groups, and state holds lie_m.
    torch_opt = SimpleNamespace(
        param_groups=[{"params": [oft_in, oft_out]}],
        state={oft_in: {"lie_m": torch.randn(2, ne)}, oft_out: {"lie_m": torch.randn(2, ne)}},
    )
    optimizer = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=torch_opt)])

    lookup = _build_lie_state_lookup(optimizer)
    label = "decoder.layers.0.mlp.linear_fc1"
    _log_coordination([{"label": label, "layer": layer}], lookup, iteration=0)

    assert f"poet_coord/{label}/mom_cos_out" in captured
    assert f"poet_coord/{label}/cos_D_out_D_in" in captured
    assert f"poet_coord/{label}/gram_cond" in captured
    assert "poet_coord/_mean/mom_cos_out" in captured
    assert all(isinstance(v, float) for v in captured.values())
