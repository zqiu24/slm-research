"""CPU equivalence test for the single-step (R=I) fast path.

At oft_R=0 the real POET chain (cayley_batch + chain_layer_x_fast_decoupled,
both pure-torch and CPU-runnable) must produce the SAME forward output and the
SAME oft_R gradients as SingleStepPOETFunction. We compare against poet's actual
cayley_batch (the Neumann series the Triton kernel implements), so this is a
faithful check of the production math, not a toy reimplementation.
"""

import pytest
import torch
from poet_torch import POETLinear, SingleStepPOETFunction
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
