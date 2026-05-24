"""Unit tests for src.optim.poet.POETAdam (no Megatron required)."""

import pytest
import torch

from src.optim.poet import POETAdam


def _adam_with_state():
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.AdamW([p], lr=0.1)
    # populate state by stepping once
    p.grad = torch.ones_like(p.data)
    base.step()
    return p, base


def test_lr_scaling_applies_on_init():
    p, base = _adam_with_state()
    wrapped = POETAdam(base, poet_merge_period=0, poet_scale=0.5)
    for g in wrapped.param_groups:
        assert g["lr"] == pytest.approx(0.05)
        assert g["max_lr"] == pytest.approx(0.05)


def test_lr_scaling_is_noop_when_scale_is_one():
    p, base = _adam_with_state()
    before = [g["lr"] for g in base.param_groups]
    POETAdam(base, poet_merge_period=0, poet_scale=1.0)
    after = [g["lr"] for g in base.param_groups]
    assert before == after


def test_momentum_reset_at_merge_period():
    p, base = _adam_with_state()
    wrapped = POETAdam(base, poet_merge_period=2, poet_scale=1.0)
    # state has non-zero exp_avg after the priming step
    assert torch.any(base.state[p]["exp_avg"] != 0)
    # step 1 — no reset
    p.grad = torch.ones_like(p.data)
    wrapped.step()
    assert torch.any(base.state[p]["exp_avg"] != 0)
    # step 2 — reset fires
    p.grad = torch.ones_like(p.data)
    wrapped.step()
    assert torch.all(base.state[p]["exp_avg"] == 0)
    assert torch.all(base.state[p]["exp_avg_sq"] == 0)


def test_proxy_attrs_pass_through():
    p, base = _adam_with_state()
    wrapped = POETAdam(base, poet_merge_period=0, poet_scale=1.0)
    assert wrapped.param_groups is base.param_groups
    assert wrapped.state is base.state


def test_poetadam_init_sets_cache_mode():
    import torch

    from src.optim import poet_cache as pc
    from src.optim.poet import POETAdam

    pc.reset_for_testing()
    p = torch.nn.Parameter(torch.zeros(1))
    base = torch.optim.Adam([p], lr=1e-3)
    POETAdam(base, poet_cache_mode="cached_fwd_bwd")
    assert pc.get_cache_mode() == "cached_fwd_bwd"


def test_poetadam_step_bumps_version_when_cache_active():
    import torch

    from src.optim import poet_cache as pc
    from src.optim.poet import POETAdam

    pc.reset_for_testing()
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.zeros(1)
    base = torch.optim.Adam([p], lr=1e-3)
    opt = POETAdam(base, poet_cache_mode="cached_fwd")
    v0 = pc.get_poet_version()
    opt.step()
    assert pc.get_poet_version() == v0 + 1


def test_poetadam_step_does_not_bump_version_when_cache_none():
    import torch

    from src.optim import poet_cache as pc
    from src.optim.poet import POETAdam

    pc.reset_for_testing()
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.zeros(1)
    base = torch.optim.Adam([p], lr=1e-3)
    opt = POETAdam(base, poet_cache_mode="none")
    v0 = pc.get_poet_version()
    opt.step()
    assert pc.get_poet_version() == v0


def test_poetadam_load_state_dict_bumps_version_and_invalidates():
    """Spec §11: checkpoint load must invalidate caches. Otherwise the
    next forward would reuse R blocks built against an oft_R from the
    pre-load state."""
    import torch

    from src.optim import poet_cache as pc
    from src.optim.poet import POETAdam

    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer._R_cache_version = 99
    pc.register_poet_layer(layer)

    p = torch.nn.Parameter(torch.zeros(1))
    base = torch.optim.Adam([p], lr=1e-3)
    opt = POETAdam(base, poet_cache_mode="cached_fwd")

    v0 = pc.get_poet_version()
    sd = opt.state_dict()
    opt.load_state_dict(sd)
    assert pc.get_poet_version() == v0 + 1
    assert layer._R_cache_version == -1


def test_install_poet_step_hook_runs_flush_before_orig_step():
    """The hook must call _flush_poet_caches_for_step before the original
    optimizer.step()."""
    import torch

    from src.optim import poet_cache as pc
    from src.optim.poet import _install_poet_step_hook

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    order: list[str] = []
    layer = pc.CachedPOETLinear(
        in_features=8,
        out_features=16,
        bsz=8,
        bias=False,
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer._flush_R_grads_to_oft_R = lambda: order.append("flush")
    pc.register_poet_layer(layer)

    class FakeWrappedOpt:
        def step(self, *a, **kw):
            order.append("orig_step")
            return "result"

    fake = FakeWrappedOpt()
    _install_poet_step_hook(fake, cache_mode="cached_fwd_bwd")
    assert fake.step() == "result"
    assert order == ["flush", "orig_step"]


def test_install_poet_step_hook_noop_when_cache_mode_not_a():
    """Hook installation is skipped for cache_mode != 'cached_fwd_bwd'."""
    from src.optim.poet import _install_poet_step_hook

    class FakeWrappedOpt:
        def step(self, *a, **kw):
            return "orig"

    fake = FakeWrappedOpt()
    orig_step = fake.step
    _install_poet_step_hook(fake, cache_mode="none")
    assert fake.step == orig_step  # bound-method equality; `is` fails in Python 3
    _install_poet_step_hook(fake, cache_mode="cached_fwd")
    assert fake.step == orig_step  # bound-method equality; `is` fails in Python 3
