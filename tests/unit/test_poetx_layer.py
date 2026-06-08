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


def _build_R(pl):
    """Pure-torch (CPU) R-build from the current oft_R (cayley)."""
    qi = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    qo = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    return cayley_batch(qo), cayley_batch(qi)  # (R_out, R_in)


def _make_pair(in_f=12, out_f=8, bc=2, bias=True, seed=3):
    """A POETLinear and a POETXLinear sharing identical weights + perms."""
    from poet_torch import POETXLinear

    torch.manual_seed(seed)
    base = POETLinear(in_features=in_f, out_features=out_f, block_count=bc, bias=bias)
    with torch.no_grad():
        base.weight.normal_()
        if bias:
            base.bias.normal_()
    xl = POETXLinear(in_features=in_f, out_features=out_f, block_count=bc, bias=bias)
    with torch.no_grad():
        for b in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(xl, b).copy_(getattr(base, b))
        # POETX stores the FORWARD-FRAME weight: Wx = W_perm[perm_out][:,perm_in].
        xl.weight.copy_(base.weight.index_select(0, base.perm_out).index_select(1, base.perm_in))
        if bias:
            xl.bias.copy_(base.bias.index_select(0, base.perm_out))
    return base, xl


def test_layer_forward_matches_chain():
    torch.set_default_dtype(torch.float64)
    base, xl = _make_pair()
    x = torch.randn(3, 12, requires_grad=True)
    gy = torch.randn(3, 8)
    y = xl(x)  # bare-GEMM forward
    assert torch.allclose(y, _chain_ref(base, x), atol=1e-9), (y - _chain_ref(base, x)).abs().max()
    (y * gy).sum().backward()
    assert xl.oft_R_in.grad is not None and xl.oft_R_out.grad is not None


def test_merge_fold_matches_poetlinear():
    """After folding the SAME stepped R, POETX's stored forward-frame weight equals
    POETLinear's effective weight W_perm[perm_out][:,perm_in] (fp64)."""
    torch.set_default_dtype(torch.float64)
    base, xl = _make_pair()
    with torch.no_grad():  # a real (small) stepped rotation on both
        base.oft_R_in.normal_(std=1e-2)
        base.oft_R_out.normal_(std=1e-2)
        xl.oft_R_in.copy_(base.oft_R_in)
        xl.oft_R_out.copy_(base.oft_R_out)
    R_out, R_in = _build_R(base)
    base._fold_with_R(R_out, R_in, reinit_perm=False)
    xl._fold_with_R(R_out, R_in, reinit_perm=False)
    eff = base.weight.index_select(0, base.perm_out).index_select(1, base.perm_in)
    assert torch.allclose(xl.weight, eff, atol=1e-9), (xl.weight - eff).abs().max()
    assert torch.count_nonzero(xl.oft_R_in) == 0 and torch.count_nonzero(xl.oft_R_out) == 0


def test_merge_reinit_folds_and_resamples_perm():
    """Fold-with-reinit stores the correct (perm-invariant) effective weight AND
    resamples the perms. The effective weight after folding R is built independently
    from the OLD perms; reinit re-permutes storage but the effective weight is the same."""
    from poet_torch.poet_layer import block_diag_lr_matmul_decoupled

    torch.set_default_dtype(torch.float64)
    _, xl = _make_pair()
    with torch.no_grad():
        xl.oft_R_in.normal_(std=1e-2)
        xl.oft_R_out.normal_(std=1e-2)
    R_out, R_in = _build_R(xl)
    perm_in_before = xl.perm_in.clone()
    # Effective (forward-frame) weight the fold must produce, built with the OLD perms:
    W_perm = xl.weight.index_select(0, xl.perm_out_inv).index_select(1, xl.perm_in_inv)
    tmp = block_diag_lr_matmul_decoupled(R_in, W_perm.t(), R_out)
    tmp = tmp.index_select(0, xl.perm_in).index_select(1, xl.perm_out)
    eff_expected = tmp.t()

    xl._fold_with_R(R_out, R_in, reinit_perm=True)

    assert torch.allclose(xl.weight, eff_expected, atol=1e-9), (
        (xl.weight - eff_expected).abs().max()
    )
    assert not torch.equal(xl.perm_in, perm_in_before)  # perms resampled
    assert torch.count_nonzero(xl.oft_R_in) == 0


def test_batched_merge_folds_poetx():
    """POETX folds correctly through the real batched merge primitives
    (_build_R_batched + _fold_with_R), on CPU with the pure-torch cayley_fn."""
    import torch
    from poet_torch import POETXLinear
    from poet_torch.poet_layer import cayley_batch

    from src.patches.poet_merge_step import _build_R_batched

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(5)
    xl = POETXLinear(in_features=12, out_features=8, block_count=2, bias=False)
    with torch.no_grad():
        xl.weight.normal_()
        xl.oft_R_in.normal_(std=1e-2)
        xl.oft_R_out.normal_(std=1e-2)
    w_before = xl.weight.clone()
    built = _build_R_batched([xl], cayley_fn=cayley_batch)  # pure-torch R-build
    R_out, R_in = built[id(xl)]
    xl._fold_with_R(R_out, R_in, reinit_perm=False)
    assert torch.count_nonzero(xl.oft_R_in) == 0 and torch.count_nonzero(xl.oft_R_out) == 0
    assert not torch.allclose(xl.weight, w_before)  # rotation absorbed


def test_run_merge_gate_collects_poetx():
    """The collection filter _run_merge uses must accept a POETX wrapped in
    POETMegatronLinear (it would skip it pre-widen). Built directly (no walk) so
    this task does not depend on the single_step_x walk param added in a later task."""
    from poet_torch import POETLinear, POETXLinear

    from src.optim.poet_layers import POETMegatronLinear

    pl = POETXLinear(in_features=8, out_features=16, block_count=1, bias=False)
    wrapper = POETMegatronLinear(pl)
    # mirror _run_merge's per-module filter (isinstance(mod, POETMegatronLinear) then
    # isinstance(mod.poet_linear, (POETLinear, POETXLinear)) and block_size > 0)
    assert isinstance(wrapper, POETMegatronLinear)
    assert isinstance(wrapper.poet_linear, POETLinear | POETXLinear)
    assert wrapper.poet_linear.block_size > 0


def test_poetx_alternating_flag_defaults_false_and_stores_cadence():
    from poet_torch import POETXLinear

    plain = POETXLinear(in_features=8, out_features=8, block_count=1)
    assert plain.alternating is False
    assert plain.alternate_every == 1

    alt = POETXLinear(
        in_features=8, out_features=8, block_count=1, alternating=True, alternate_every=3
    )
    assert alt.alternating is True
    assert alt.alternate_every == 3


def test_alternating_subclass_sets_flag_and_inherits_fold():
    from poet_torch import AlternatingPOETXLinear, POETXLinear

    layer = AlternatingPOETXLinear(in_features=8, out_features=16, block_count=1, alternate_every=2)
    assert isinstance(layer, POETXLinear)
    assert layer.alternating is True  # routes via the flag in the merge driver
    assert layer.alternate_every == 2
    # _fold_active_side now lives on POETXLinear; the subclass inherits it.
    assert layer._fold_active_side.__qualname__.startswith("POETXLinear.")
