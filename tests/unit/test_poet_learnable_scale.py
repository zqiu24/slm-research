"""CPU tests for POET learnable per-layer scale (ScaledPOETLinear)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from poet_torch import POETLinear, SingleStepPOETLinear

from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet
from src.optim.poet_scaled_layer import (
    ScaledPOETLinear,
    ScaledSingleStepPOETLinear,
)


def _plain_poet_forward_runs_here() -> bool:
    """The plain (non-single-step) cayley ``POETLinear`` forward routes through
    ``torch.compile``/triton; on a CPU-only box with no GPU driver it raises
    ``RuntimeError: 0 active drivers``. The single-step path is pure-eager and
    always runs. Probe once so the plain-path FORWARD cases skip cleanly off-GPU
    while the single-step path (the production ``single_step_native=true`` path)
    still fully exercises the shared gain mixin on CPU."""
    try:
        p = POETLinear(
            4, 4, block_count=1, bias=False, parameterization="cayley", dtype=torch.float32
        )
        with torch.no_grad():
            p.weight.normal_(std=0.02)
            p(torch.zeros(1, 4))
        return True
    except Exception:
        return False


_PLAIN_POET_CPU = _plain_poet_forward_runs_here()
_needs_plain_forward = pytest.mark.skipif(
    not _PLAIN_POET_CPU,
    reason="plain POETLinear cayley forward needs torch.compile/triton (GPU "
    "driver); single-step path covers the shared gain mixin on CPU",
)


def _make_pair(scaled_cls, base_cls):
    """A scaled layer and a base twin sharing the same frozen weight + perms."""
    scaled = scaled_cls(
        8, 16, block_count=1, bias=False, parameterization="cayley", dtype=torch.float32
    )
    base = base_cls(
        8, 16, block_count=1, bias=False, parameterization="cayley", dtype=torch.float32
    )
    # POETLinear leaves the frozen weight as torch.empty (possibly NaN); give it a
    # finite value so torch.equal is meaningful (NaN != NaN would fail spuriously),
    # then mirror it + the permutation buffers into the base twin so the two compute
    # identically. oft_R_{in,out} init to zeros (R=I) in both, so g=1 ⇒ equal output.
    with torch.no_grad():
        scaled.weight.normal_(std=0.02)
        base.weight.copy_(scaled.weight)
        for buf in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(base, buf).copy_(getattr(scaled, buf))
    return scaled, base


def test_gain_initialized_to_one_scalar():
    layer = ScaledPOETLinear(8, 16, block_count=1, bias=False, dtype=torch.float32)
    assert isinstance(layer.gain, nn.Parameter)
    assert layer.gain.requires_grad
    assert layer.gain.shape == torch.Size([])  # 0-dim scalar
    assert float(layer.gain) == 1.0


@pytest.mark.parametrize(
    "scaled_cls,base_cls",
    [
        pytest.param(ScaledPOETLinear, POETLinear, marks=_needs_plain_forward),
        (ScaledSingleStepPOETLinear, SingleStepPOETLinear),
    ],
)
def test_gain_one_is_bit_exact_base(scaled_cls, base_cls):
    scaled, base = _make_pair(scaled_cls, base_cls)
    x = torch.randn(4, 8)
    with torch.no_grad():
        out_scaled = scaled(x)
        out_base = base(x)
    assert torch.equal(out_scaled, out_base)  # exact, not allclose


@pytest.mark.parametrize(
    "scaled_cls,base_cls",
    [
        pytest.param(ScaledPOETLinear, POETLinear, marks=_needs_plain_forward),
        (ScaledSingleStepPOETLinear, SingleStepPOETLinear),
    ],
)
def test_gain_scales_output(scaled_cls, base_cls):
    scaled, base = _make_pair(scaled_cls, base_cls)
    with torch.no_grad():
        scaled.gain.fill_(2.5)
    x = torch.randn(4, 8)
    with torch.no_grad():
        out_scaled = scaled(x)
        out_base = base(x)
    assert torch.allclose(out_scaled, 2.5 * out_base, atol=1e-6)


@pytest.mark.parametrize(
    "scaled_cls",
    [
        pytest.param(ScaledPOETLinear, marks=_needs_plain_forward),
        ScaledSingleStepPOETLinear,
    ],
)
def test_grad_flows_to_gain(scaled_cls):
    layer = scaled_cls(8, 16, block_count=1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.normal_(std=0.02)  # finite frozen base (torch.empty may be NaN)
    x = torch.randn(4, 8)
    layer(x).sum().backward()
    assert layer.gain.grad is not None
    assert layer.gain.grad.shape == torch.Size([])


def test_bias_is_rejected():
    with pytest.raises(ValueError, match="bias=False"):
        ScaledPOETLinear(8, 16, block_count=1, bias=True, dtype=torch.float32)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)


def test_replace_with_learnable_scale_swaps_scaled_class():
    m = _ToyModel()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        learnable_scale=True,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, ScaledPOETLinear)
    assert hasattr(pl, "gain") and float(pl.gain) == 1.0


def test_replace_without_flag_has_no_gain():
    m = _ToyModel()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
    )
    assert not hasattr(m.fc1.poet_linear, "gain")


def test_learnable_scale_rejects_head_aligned():
    m = _ToyModel()
    with pytest.raises(NotImplementedError, match="learnable_scale"):
        replace_linears_with_poet(
            m,
            block_count=1,
            init_type="none",
            extra_linear_types=(nn.Linear,),
            learnable_scale=True,
            head_aligned_attn=True,
            head_dim=4,
        )
