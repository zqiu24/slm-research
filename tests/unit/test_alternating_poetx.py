"""AlternatingPOETXLinear: single-side backward (active side matches both-sides
closed form; frozen side returns shape-correct zeros, never None)."""

import pytest
import torch
from poet_torch import POETLinear, POETXSingleStepFunction
from poet_torch.poetx_ops import AlternatingPOETXSingleStepFunction


@pytest.fixture(autouse=True)
def _reset_alt_state():
    # The active-side signal is a module global; reset it around every test so
    # active-side assertions can't leak across tests. Several tests below set the
    # float64 default dtype — restore float32 after each so we don't leak it into
    # later test files (e.g. the precision-tuned test_poet_lie_orth spectral asserts).
    from poet_torch import alt_state

    alt_state.set_iteration(0)
    yield
    alt_state.set_iteration(0)
    torch.set_default_dtype(torch.float32)


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


def test_active_only_fold_matches_both_sides_when_frozen_is_identity():
    """Folding only the active side == folding both sides when the frozen side's
    oft_R is 0 (identity). fp64 parity."""
    from poet_torch import AlternatingPOETXLinear
    from poet_torch.poet_layer import cayley_batch, pytorch_skew_symmetric

    torch.set_default_dtype(torch.float64)

    def _make():
        # Seed INSIDE so ref and act are bit-identical clones (same perms, weight,
        # and oft_R_in); otherwise the two _make() calls draw divergent RNG state
        # and the both-sides-vs-active-only parity comparison is meaningless.
        torch.manual_seed(5)
        layer = AlternatingPOETXLinear(in_features=12, out_features=8, block_count=1, bias=False)
        with torch.no_grad():
            layer.weight.normal_()
            layer.oft_R_in.normal_(std=1e-2)  # oft_R_out left at 0 (frozen = identity)
        return layer

    def _cayley(layer):
        qi = pytorch_skew_symmetric(
            layer.oft_R_in, layer.block_size_in, layer.rows_in, layer.cols_in
        )
        qo = pytorch_skew_symmetric(
            layer.oft_R_out, layer.block_size_out, layer.rows_out, layer.cols_out
        )
        return cayley_batch(qo), cayley_batch(qi)  # (R_out, R_in)

    # active "in": only oft_R_in nonzero, oft_R_out stays identity
    ref, act = _make(), _make()
    # reference: full both-sides fold
    R_out, R_in = _cayley(ref)
    ref._fold_with_R(R_out, R_in, reinit_perm=False)
    # active-only fold
    act._fold_active_side("in", reinit_perm=False, cayley_fn=cayley_batch)
    assert torch.allclose(act.weight, ref.weight, atol=1e-9), (act.weight - ref.weight).abs().max()
    assert torch.count_nonzero(act.oft_R_in) == 0


def test_merge_layers_routes_alternating_to_active_only_fold(monkeypatch):
    from poet_torch import AlternatingPOETXLinear, alt_state

    import src.patches.poet_merge_step as ms

    alt_state.set_iteration(1)  # active "in" at alternate_every=1
    layer = AlternatingPOETXLinear(in_features=8, out_features=8, block_count=1, bias=False)
    calls = []
    monkeypatch.setattr(
        AlternatingPOETXLinear,
        "_fold_active_side",
        lambda self, side, reinit_perm=False: calls.append(side),
    )
    ms._merge_layers([layer], reinit_perm=False, disable_batch=False)
    assert calls == ["in"]


def test_merge_layers_routes_integrated_poetx_to_active_only_fold(monkeypatch):
    from poet_torch import POETXLinear, alt_state

    import src.patches.poet_merge_step as ms

    alt_state.set_iteration(1)  # active "in" at alternate_every=1
    layer = POETXLinear(in_features=8, out_features=8, block_count=1, bias=False, alternating=True)
    calls = []
    monkeypatch.setattr(
        POETXLinear,
        "_fold_active_side",
        lambda self, side, reinit_perm=False: calls.append(side),
    )
    ms._merge_layers([layer], reinit_perm=False, disable_batch=False)
    assert calls == ["in"]
    alt_state.set_iteration(0)


def test_merge_layers_keeps_plain_poetx_on_both_sides_fold(monkeypatch):
    # alternating=False must NOT route to _fold_active_side; it goes through the
    # batched both-sides fold instead. _build_R_batched + _fold_with_R use the
    # GPU-only Triton cayley, so stub them to keep this routing assertion runnable
    # on a CPU node (the point under test is the partition, not the fold math).
    from poet_torch import POETXLinear, alt_state

    import src.patches.poet_merge_step as ms

    alt_state.set_iteration(1)
    layer = POETXLinear(in_features=8, out_features=8, block_count=1, bias=False)
    with torch.no_grad():
        layer.weight.normal_()
    active_calls = []
    batched = []

    def _stub_build(pls, **kw):
        batched.extend(pls)
        return {id(pl): (None, None) for pl in pls}

    monkeypatch.setattr(
        POETXLinear,
        "_fold_active_side",
        lambda self, side, reinit_perm=False: active_calls.append(side),
    )
    monkeypatch.setattr(ms, "_build_R_batched", _stub_build)
    monkeypatch.setattr(
        POETXLinear, "_fold_with_R", lambda self, R_out, R_in, reinit_perm=False: None
    )
    ms._merge_layers([layer], reinit_perm=False, disable_batch=False)
    assert active_calls == []  # plain POETX did NOT go to the active-only fold
    assert batched == [layer]  # it went through the batched both-sides fold
    alt_state.set_iteration(0)


def test_integrated_alternating_write_side_matches_fold_side():
    """End-to-end consistency: the side the optimizer WRITES each step (driven by
    alt_state) equals the side the merge driver FOLDS (same alt_state)."""
    from poet_torch import POETXLinear, alt_state
    from poet_torch.alt_state import active_side

    import src.patches.poet_merge_step as ms
    from src.optim.poet_lie_orth import LieOrthMomentum

    torch.manual_seed(0)
    layer = POETXLinear(
        in_features=8,
        out_features=8,
        block_count=1,
        bias=False,
        alternating=True,
        alternate_every=1,
    )
    with torch.no_grad():
        layer.weight.normal_()
    opt = LieOrthMomentum(
        [
            dict(params=[layer.oft_R_in], use_skew=True, side="in", lr=0.1),
            dict(params=[layer.oft_R_out], use_skew=True, side="out", lr=0.1),
        ],
        ortho_c=0.05,
        alternating=True,  # integrated both-momenta path (true_single_side stays False)
    )
    # The real fold defaults its cayley to the Triton op (GPU-only); inject the
    # pure-torch cayley_batch so the REAL fold math runs end-to-end on a CPU node.
    from poet_torch.poet_layer import cayley_batch

    folded = []
    _real_fold = POETXLinear._fold_active_side

    def _spy_fold(self, side, reinit_perm=False):
        folded.append(side)
        return _real_fold(self, side, reinit_perm=reinit_perm, cayley_fn=cayley_batch)

    POETXLinear._fold_active_side = _spy_fold
    try:
        for it in range(1, 5):
            alt_state.set_iteration(it)
            layer.oft_R_in.grad = torch.randn_like(layer.oft_R_in)
            layer.oft_R_out.grad = torch.randn_like(layer.oft_R_out)
            opt.step()  # writes the active side only; BOTH momenta advanced
            # Read the side EXACTLY as the optimizer + merge do: active_side takes
            # alternate_every (here 1), reading the iteration set above -- NOT `it`
            # as the cadence (active_side(it) would always be "in" for it>=1).
            active = active_side(layer.alternate_every)
            wrote_in = layer.oft_R_in.abs().sum().item() > 0
            wrote_out = layer.oft_R_out.abs().sum().item() > 0
            assert (wrote_in, wrote_out) == ((active == "in"), (active == "out")), (it, active)
            ms._merge_layers([layer], reinit_perm=False, disable_batch=False)
            assert folded[-1] == active, (it, active, folded[-1])
            # the fold zeroed both sides -> next step starts clean
            assert layer.oft_R_in.abs().sum().item() == 0
            assert layer.oft_R_out.abs().sum().item() == 0
    finally:
        POETXLinear._fold_active_side = _real_fold
        alt_state.set_iteration(0)
