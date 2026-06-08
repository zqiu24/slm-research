"""CPU equivalence tests for the forward-frame (perm-free-forward) POETX path.

POETXSingleStepFunction takes the forward-frame weight Wx = W_perm[perm_out][:,perm_in]
and bias_eff = bias[perm_out] directly, so its forward is a bare GEMM. At oft_R=0 it must
match the real chain (cayley_batch + chain_layer_x_fast_decoupled, pure-torch CPU) to fp64,
and POETXLinear must be a drop-in for the chain forward + reuse the verified merge fold.
"""

import pytest
import torch
from poet_torch import POETLinear, POETXSingleStepFunction
from poet_torch.poet_layer import (
    cayley_batch,
    chain_layer_x_fast_decoupled,
    pytorch_skew_symmetric,
)


def _chain_ref(pl, x):
    qi = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    qo = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    return chain_layer_x_fast_decoupled(
        x,
        cayley_batch(qi),
        pl.weight,
        pl.bias,
        cayley_batch(qo),
        pl.perm_in_inv,
        pl.perm_in,
        pl.perm_out,
        pl.perm_out_inv,
        pl.block_size_in,
        pl.block_size_out,
    )


def _forward_frame(pl):
    """Wx = W_perm[perm_out][:,perm_in]; bias_eff = bias[perm_out] (or None)."""
    Wx = pl.weight.index_select(0, pl.perm_out).index_select(1, pl.perm_in)
    bias_eff = None if pl.bias is None else pl.bias.index_select(0, pl.perm_out)
    return Wx, bias_eff


def _op(pl, Wx, bias_eff, x):
    return POETXSingleStepFunction.apply(
        x,
        pl.oft_R_in,
        pl.oft_R_out,
        Wx,
        bias_eff,
        pl.perm_in_inv,
        pl.perm_out_inv,
        pl.rows_in,
        pl.cols_in,
        pl.rows_out,
        pl.cols_out,
        pl.block_size_in,
        pl.block_size_out,
    )


@pytest.mark.parametrize("bc,bias", [(1, False), (1, True), (2, False), (2, True)])
def test_op_matches_chain_at_zero(bc, bias):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=bc, bias=bias)
    with torch.no_grad():
        pl.weight.normal_()
        if bias:
            pl.bias.normal_()
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0
    Wx, bias_eff = _forward_frame(pl)

    x = torch.randn(5, 12)
    gy = torch.randn(5, 8)

    assert torch.allclose(_chain_ref(pl, x), _op(pl, Wx, bias_eff, x), atol=1e-9), (
        (_chain_ref(pl, x) - _op(pl, Wx, bias_eff, x)).abs().max()
    )

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xr = x.clone().requires_grad_(True)
    (_chain_ref(pl, xr) * gy).sum().backward()
    gi_r, go_r, gx_r = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xr.grad.clone()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xn = x.clone().requires_grad_(True)
    (_op(pl, Wx, bias_eff, xn) * gy).sum().backward()
    gi_n, go_n, gx_n = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xn.grad.clone()

    assert torch.allclose(gi_r, gi_n, atol=1e-9), (gi_r - gi_n).abs().max()
    assert torch.allclose(go_r, go_n, atol=1e-9), (go_r - go_n).abs().max()
    assert torch.allclose(gx_r, gx_n, atol=1e-9), (gx_r - gx_n).abs().max()


def test_op_forward_is_bare_gemm():
    """The forward applies NO permutation: it is exactly x@Wx.t() (+ bias_eff)."""
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=1, bias=True)
    with torch.no_grad():
        pl.weight.normal_()
        pl.bias.normal_()
    Wx, bias_eff = _forward_frame(pl)
    x = torch.randn(3, 12)
    assert (_op(pl, Wx, bias_eff, x) - (x @ Wx.t() + bias_eff)).abs().max().item() == 0.0
