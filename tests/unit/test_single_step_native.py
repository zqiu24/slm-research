"""CPU equivalence test for the native-frame (gather-free) single-step path.

At oft_R=0 the native forward (pure GEMM on the forward-frame weight W_eff) and its
closed-form backward must match the real chain (cayley_batch + chain_layer_x_fast_decoupled,
pure-torch CPU). Verified bit-against the chain to fp64 machine precision.
"""

import pytest
import torch
from poet_torch import NativeSingleStepFunction, POETLinear
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


def _native(pl, x):
    return NativeSingleStepFunction.apply(
        x,
        pl.oft_R_in,
        pl.oft_R_out,
        pl.weight,
        pl.bias,
        pl.perm_in,
        pl.perm_in_inv,
        pl.perm_out,
        pl.perm_out_inv,
        pl.rows_in,
        pl.cols_in,
        pl.rows_out,
        pl.cols_out,
        pl.block_size_in,
        pl.block_size_out,
    )


@pytest.mark.parametrize("bc,bias", [(1, False), (1, True), (2, False), (2, True)])
def test_native_matches_chain_at_zero(bc, bias):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=bc, bias=bias)
    with torch.no_grad():
        pl.weight.normal_()
        if bias:
            pl.bias.normal_()
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0

    x = torch.randn(5, 12)
    gy = torch.randn(5, 8)

    assert torch.allclose(_chain_ref(pl, x), _native(pl, x), atol=1e-9), (
        (_chain_ref(pl, x) - _native(pl, x)).abs().max()
    )

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xr = x.clone().requires_grad_(True)
    (_chain_ref(pl, xr) * gy).sum().backward()
    gi_r, go_r, gx_r = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xr.grad.clone()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xn = x.clone().requires_grad_(True)
    (_native(pl, xn) * gy).sum().backward()
    gi_n, go_n, gx_n = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xn.grad.clone()

    assert torch.allclose(gi_r, gi_n, atol=1e-9), (gi_r - gi_n).abs().max()
    assert torch.allclose(go_r, go_n, atol=1e-9), (go_r - go_n).abs().max()
    assert torch.allclose(gx_r, gx_n, atol=1e-9), (gx_r - gx_n).abs().max()


def test_native_forward_identity_perm_is_bit_identical():
    """With identity perms, the native forward is exactly x@Wᵀ (the bit-identity anchor)."""
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=1, bias=False)
    with torch.no_grad():
        pl.weight.normal_()
        pl.perm_in.copy_(torch.arange(12, dtype=torch.int32))
        pl.perm_in_inv.copy_(torch.arange(12, dtype=torch.int32))
        pl.perm_out.copy_(torch.arange(8, dtype=torch.int32))
        pl.perm_out_inv.copy_(torch.arange(8, dtype=torch.int32))
    x = torch.randn(3, 12)
    assert (_native(pl, x) - x @ pl.weight.t()).abs().max().item() == 0.0
