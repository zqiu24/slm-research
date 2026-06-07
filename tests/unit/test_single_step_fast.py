"""CPU equivalence test for the single-step (R=I) fast path.

At oft_R=0 the real POET chain (cayley_batch + chain_layer_x_fast_decoupled,
both pure-torch and CPU-runnable) must produce the SAME forward output and the
SAME oft_R gradients as SingleStepPOETFunction. We compare against poet's actual
cayley_batch (the Neumann series the Triton kernel implements), so this is a
faithful check of the production math, not a toy reimplementation.
"""

import pytest
import torch
from poet_torch import (
    HeadAlignedPOETLinear,
    HeadAlignedSingleStepFunction,
    POETLinear,
    SingleStepPOETFunction,
)
from poet_torch.head_aligned_layer import chain_noperm
from poet_torch.poet_layer import (
    cayley_batch,
    chain_layer_x_fast_decoupled,
    pytorch_skew_symmetric,
)


def _reference_chain(pl, x):
    """Forward through the REAL chain with R built from oft_R via cayley_batch."""
    q_in = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    q_out = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    r_in, r_out = cayley_batch(q_in), cayley_batch(q_out)
    return chain_layer_x_fast_decoupled(
        x,
        r_in,
        pl.weight,
        pl.bias,
        r_out,
        pl.perm_in_inv,
        pl.perm_in,
        pl.perm_out,
        pl.perm_out_inv,
        pl.block_size_in,
        pl.block_size_out,
    )


def _fast(pl, x):
    return SingleStepPOETFunction.apply(
        x,
        pl.oft_R_in,
        pl.oft_R_out,
        pl.weight,
        pl.bias,
        pl.perm_in_inv,
        pl.perm_in,
        pl.perm_out,
        pl.perm_out_inv,
        pl.rows_in,
        pl.cols_in,
        pl.rows_out,
        pl.cols_out,
        pl.block_size_in,
        pl.block_size_out,
    )


@pytest.mark.parametrize(
    "in_f,out_f,bc,bias", [(12, 8, 1, False), (12, 8, 2, False), (16, 16, 4, True)]
)
def test_fast_matches_chain_at_zero(in_f, out_f, bc, bias):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=in_f, out_features=out_f, block_count=bc, bias=bias)
    with torch.no_grad():
        pl.weight.normal_()
        if bias:
            pl.bias.normal_()
    # oft_R is the deployed-invariant value: 0 (R=I). Keep it 0; both paths read it.
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0

    x = torch.randn(5, in_f, requires_grad=True)
    gy = torch.randn(5, out_f)

    # forward equality
    y_ref = _reference_chain(pl, x)
    y_fast = _fast(pl, x)
    assert torch.allclose(y_ref, y_fast, atol=1e-10), (y_ref - y_fast).abs().max()

    # grad equality (oft_R_in/out and x). Two independent backward passes.
    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    x_ref = x.detach().clone().requires_grad_(True)
    (_reference_chain(pl, x_ref) * gy).sum().backward()
    g_in_ref, g_out_ref = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone()
    gx_ref = x_ref.grad.clone()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    x_fast = x.detach().clone().requires_grad_(True)
    (_fast(pl, x_fast) * gy).sum().backward()
    g_in_fast, g_out_fast = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone()
    gx_fast = x_fast.grad.clone()

    assert torch.allclose(g_in_ref, g_in_fast, atol=1e-9), (g_in_ref - g_in_fast).abs().max()
    assert torch.allclose(g_out_ref, g_out_fast, atol=1e-9), (g_out_ref - g_out_fast).abs().max()
    assert torch.allclose(gx_ref, gx_fast, atol=1e-9), (gx_ref - gx_fast).abs().max()


def test_layer_forward_uses_fast_path_when_flagged():
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=2, bias=False)
    with torch.no_grad():
        pl.weight.normal_()
    pl.single_step_fast = True
    x = torch.randn(3, 12, requires_grad=True)
    gy = torch.randn(3, 8)
    # layer forward (fast) must equal the reference chain at oft_R=0
    y = pl(x)
    assert torch.allclose(y, _reference_chain(pl, x), atol=1e-10)
    (y * gy).sum().backward()
    assert pl.oft_R_in.grad is not None and pl.oft_R_out.grad is not None


def _ha_reference(pl, x):
    """Forward through the REAL permutation-free chain (R from cayley_batch)."""
    q_in = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    q_out = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    r_in, r_out = cayley_batch(q_in), cayley_batch(q_out)
    return chain_noperm(x, r_in, pl.weight, pl.bias, r_out, pl.block_size_in, pl.block_size_out)


def _ha_fast(pl, x):
    return HeadAlignedSingleStepFunction.apply(
        x,
        pl.oft_R_in,
        pl.oft_R_out,
        pl.weight,
        pl.bias,
        pl.rows_in,
        pl.cols_in,
        pl.rows_out,
        pl.cols_out,
        pl.block_size_in,
        pl.block_size_out,
        pl.head_side,
    )


@pytest.mark.parametrize("head_side,in_f,out_f", [("out", 12, 8), ("in", 8, 12)])
def test_head_aligned_fast_matches_chain(head_side, in_f, out_f):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = HeadAlignedPOETLinear(
        in_features=in_f,
        out_features=out_f,
        head_side=head_side,
        head_dim=4,
        resid_block_count=1,
        bias=False,
    )
    with torch.no_grad():
        pl.weight.normal_()
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0

    x = torch.randn(5, in_f)
    gy = torch.randn(5, out_f)

    # forward equality
    assert torch.allclose(_ha_reference(pl, x), _ha_fast(pl, x), atol=1e-10)

    # grad equality
    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xr = x.clone().requires_grad_(True)
    (_ha_reference(pl, xr) * gy).sum().backward()
    gin_ref, gout_ref, gx_ref = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xr.grad.clone()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xf = x.clone().requires_grad_(True)
    (_ha_fast(pl, xf) * gy).sum().backward()
    gin_f, gout_f, gx_f = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xf.grad.clone()

    assert torch.allclose(gin_ref, gin_f, atol=1e-9), (gin_ref - gin_f).abs().max()
    assert torch.allclose(gout_ref, gout_f, atol=1e-9), (gout_ref - gout_f).abs().max()
    assert torch.allclose(gx_ref, gx_f, atol=1e-9), (gx_ref - gx_f).abs().max()


@pytest.mark.parametrize("head_side,in_f,out_f", [("out", 12, 8), ("in", 8, 12)])
def test_head_aligned_layer_uses_fast_path_when_flagged(head_side, in_f, out_f):
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)
    pl = HeadAlignedPOETLinear(
        in_features=in_f,
        out_features=out_f,
        head_side=head_side,
        head_dim=4,
        resid_block_count=1,
        bias=False,
    )
    with torch.no_grad():
        pl.weight.normal_()
    pl.single_step_fast = True
    x = torch.randn(3, in_f, requires_grad=True)
    # would call the Triton cayley on the slow path -> CPU error; fast path avoids it.
    y = pl(x)
    assert torch.allclose(y, _ha_reference(pl, x), atol=1e-10)
    (y * torch.randn(3, out_f)).sum().backward()
    assert pl.oft_R_in.grad is not None and pl.oft_R_out.grad is not None
