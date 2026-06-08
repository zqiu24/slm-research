"""AlternatingPOETXLinear: single-side backward (active side matches both-sides
closed form; frozen side returns shape-correct zeros, never None)."""

import pytest
import torch
from poet_torch import POETLinear, POETXSingleStepFunction
from poet_torch.poetx_ops import AlternatingPOETXSingleStepFunction


@pytest.fixture(autouse=True)
def _reset_alt_state():
    # The active-side signal is a module global; reset it around every test so
    # active-side assertions can't leak across tests.
    from poet_torch import alt_state

    alt_state.set_iteration(0)
    yield
    alt_state.set_iteration(0)


def _forward_frame(pl):
    Wx = pl.weight.index_select(0, pl.perm_out).index_select(1, pl.perm_in)
    bias_eff = None if pl.bias is None else pl.bias.index_select(0, pl.perm_out)
    return Wx, bias_eff


def _both(pl, Wx, bias_eff, x):
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


def _alt(pl, Wx, bias_eff, x, active):
    return AlternatingPOETXSingleStepFunction.apply(
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
        active,
    )


def _grads(fn):
    pl = POETLinear(in_features=12, out_features=8, block_count=1, bias=True)
    with torch.no_grad():
        pl.weight.normal_()
        pl.bias.normal_()
    Wx, bias_eff = _forward_frame(pl)
    x = torch.randn(5, 12, requires_grad=True)
    gy = torch.randn(5, 8)
    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    (fn(pl, Wx, bias_eff, x) * gy).sum().backward()
    return pl, x


def test_active_in_matches_both_sides_and_frozen_is_zeros():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    # reference: both-sides grads
    pl_b, _ = _grads(lambda pl, Wx, b, x: _both(pl, Wx, b, x))
    gi_ref = pl_b.oft_R_in.grad.clone()
    # alternating with same seed/weights -> rebuild identical layer
    torch.manual_seed(0)
    pl_a, _ = _grads(lambda pl, Wx, b, x: _alt(pl, Wx, b, x, "in"))
    assert torch.allclose(pl_a.oft_R_in.grad, gi_ref, atol=1e-9)
    # frozen side: shape-correct ZEROS, not None
    assert pl_a.oft_R_out.grad is not None
    assert pl_a.oft_R_out.grad.shape == pl_a.oft_R_out.shape
    assert torch.count_nonzero(pl_a.oft_R_out.grad) == 0


def test_active_out_matches_both_sides_and_frozen_is_zeros():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl_b, _ = _grads(lambda pl, Wx, b, x: _both(pl, Wx, b, x))
    go_ref = pl_b.oft_R_out.grad.clone()
    torch.manual_seed(0)
    pl_a, _ = _grads(lambda pl, Wx, b, x: _alt(pl, Wx, b, x, "out"))
    assert torch.allclose(pl_a.oft_R_out.grad, go_ref, atol=1e-9)
    assert pl_a.oft_R_in.grad is not None
    assert torch.count_nonzero(pl_a.oft_R_in.grad) == 0


def test_grad_x_is_independent_of_active_side():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=1, bias=False)
    with torch.no_grad():
        pl.weight.normal_()
    Wx, bias_eff = _forward_frame(pl)
    gy = torch.randn(5, 8)
    xi = torch.randn(5, 12, requires_grad=True)
    (_alt(pl, Wx, bias_eff, xi, "in") * gy).sum().backward()
    xo = xi.detach().clone().requires_grad_(True)
    (_alt(pl, Wx, bias_eff, xo, "out") * gy).sum().backward()
    assert torch.allclose(xi.grad, xo.grad, atol=1e-9)


def test_layer_forward_is_bare_gemm_and_backward_is_single_side():
    from poet_torch import AlternatingPOETXLinear, alt_state

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    layer = AlternatingPOETXLinear(
        in_features=12, out_features=8, block_count=1, bias=True, alternate_every=1
    )
    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
    x = torch.randn(4, 12)
    # forward = bare GEMM on the stored forward-frame weight (R=I at merge_period=1)
    y = layer(x)
    assert (y - (x @ layer.weight.t() + layer.bias)).abs().max().item() == 0.0

    # iteration 1 -> active "in": only oft_R_in gets a nonzero grad, oft_R_out zeros
    alt_state.set_iteration(1)
    layer.oft_R_in.grad = layer.oft_R_out.grad = None
    gy = torch.randn(4, 8)
    (layer(x) * gy).sum().backward()
    assert torch.count_nonzero(layer.oft_R_in.grad) > 0
    assert torch.count_nonzero(layer.oft_R_out.grad) == 0

    # iteration 2 -> active "out": flips
    alt_state.set_iteration(2)
    layer.oft_R_in.grad = layer.oft_R_out.grad = None
    (layer(x) * gy).sum().backward()
    assert torch.count_nonzero(layer.oft_R_out.grad) > 0
    assert torch.count_nonzero(layer.oft_R_in.grad) == 0


def test_layer_is_poetx_subclass():
    from poet_torch import AlternatingPOETXLinear, POETXLinear

    layer = AlternatingPOETXLinear(in_features=8, out_features=16, block_count=1)
    assert isinstance(layer, POETXLinear)  # merge driver isinstance tuple includes it
    assert layer.alternate_every == 1
