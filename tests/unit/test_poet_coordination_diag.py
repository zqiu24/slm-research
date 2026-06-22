"""Tests for the POET two-sided coordination Tier-0 diagnostics
(src/diag/poet_coordination_diag.py): the pure-math metric functions that
arbitrate the alternating-vs-simultaneous mechanism.

  - momentum_grad_cosine: cos(lie_m, fresh grad) per side -> staleness arbiter.
  - direction_overlap: cos(D_out, D_in), r_joint, gram_cond -> gauge-redundancy
    arbiter (D_out = A_out @ W, D_in = W @ A_in in weight space).

Pure tensors only (no Megatron / CUDA / poet_torch), unit-testable on CPU.
"""

import math

import pytest
import torch

from src.diag.poet_coordination_diag import (
    block_diag_skew,
    cross_term_ratio,
    direction_overlap,
    layer_coordination_metrics,
    momentum_grad_cosine,
    side_directions,
    weight_only_sensitivity,
)


@pytest.fixture(autouse=True)
def _isolate_default_dtype():
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(torch.float32)


# --- momentum_grad_cosine -------------------------------------------------


def test_mgc_identical_vectors_is_one():
    v = torch.randn(1, 28)
    assert momentum_grad_cosine(v, v.clone()).item() == pytest.approx(1.0, abs=1e-6)


def test_mgc_opposite_vectors_is_minus_one():
    v = torch.randn(1, 28)
    assert momentum_grad_cosine(v, -v).item() == pytest.approx(-1.0, abs=1e-6)


def test_mgc_orthogonal_vectors_is_zero():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 2.0, 0.0]])
    assert momentum_grad_cosine(a, b).item() == pytest.approx(0.0, abs=1e-6)


def test_mgc_is_scale_invariant():
    m = torch.randn(3, 28)
    g = torch.randn(3, 28)
    base = momentum_grad_cosine(m, g).item()
    assert momentum_grad_cosine(5.0 * m, 0.1 * g).item() == pytest.approx(base, abs=1e-6)


def test_mgc_is_frobenius_over_all_blocks():
    # A batched (num_blocks, n_elems) tensor reduces over the WHOLE tensor (one scalar),
    # i.e. the Frobenius cosine, not a per-block vector.
    m = torch.randn(4, 28)
    g = torch.randn(4, 28)
    out = momentum_grad_cosine(m, g)
    assert out.dim() == 0
    expected = (m * g).sum() / (m.norm() * g.norm())
    assert out.item() == pytest.approx(expected.item(), abs=1e-6)


def test_mgc_zero_input_is_zero_not_nan():
    m = torch.zeros(1, 28)
    g = torch.randn(1, 28)
    out = momentum_grad_cosine(m, g)
    assert torch.isfinite(out).all()
    assert out.item() == pytest.approx(0.0, abs=1e-12)


# --- direction_overlap ----------------------------------------------------


def test_overlap_identical_directions():
    D = torch.randn(6, 5)
    out = direction_overlap(D, D.clone())
    assert out["cos"].item() == pytest.approx(1.0, abs=1e-6)
    # r_joint = ||2D||^2 / (||D||^2 + ||D||^2) = 4||D||^2 / 2||D||^2 = 2
    assert out["r_joint"].item() == pytest.approx(2.0, abs=1e-6)
    # Gram is rank-1 -> condition number blows up.
    assert out["gram_cond"].item() > 1e6


def test_overlap_anti_parallel_directions_cancel():
    D = torch.randn(6, 5)
    out = direction_overlap(D, -D)
    assert out["cos"].item() == pytest.approx(-1.0, abs=1e-6)
    # ||D - D||^2 = 0 -> full cancellation.
    assert out["r_joint"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["gram_cond"].item() > 1e6


def test_overlap_orthogonal_equal_norm():
    d_out = torch.zeros(2, 2)
    d_out[0, 0] = 3.0
    d_in = torch.zeros(2, 2)
    d_in[1, 1] = 3.0
    out = direction_overlap(d_out, d_in)
    assert out["cos"].item() == pytest.approx(0.0, abs=1e-6)
    # orthogonal, equal energy -> ||D_out + D_in||^2 == ||D_out||^2 + ||D_in||^2
    assert out["r_joint"].item() == pytest.approx(1.0, abs=1e-6)
    # M = diag(9, 9) -> cond 1.
    assert out["gram_cond"].item() == pytest.approx(1.0, abs=1e-5)


def test_overlap_known_numeric_cosine():
    d_out = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    d_in = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    out = direction_overlap(d_out, d_in)
    # <D_out, D_in> = 1, ||D_out|| = 1, ||D_in|| = sqrt(2) -> cos = 1/sqrt(2)
    assert out["cos"].item() == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-6)


def test_overlap_zero_input_is_finite():
    d_out = torch.zeros(3, 3)
    d_in = torch.randn(3, 3)
    out = direction_overlap(d_out, d_in)
    assert torch.isfinite(out["cos"]).all()
    assert out["cos"].item() == pytest.approx(0.0, abs=1e-12)


# --- side_directions (block-diagonal D_out / D_in in the W_perm frame) -----


def _block_diag(blocks):
    # blocks: (n, b, b) -> dense (n*b, n*b) block-diagonal
    return torch.block_diag(*[blocks[i] for i in range(blocks.shape[0])])


def test_side_directions_matches_dense_block_diag_reference():
    # D_out = blockdiag(A_out) @ W ; D_in = W @ blockdiag(A_in), in the W_perm frame
    # where blocks are contiguous. Verify the reshape/bmm/einsum against a dense ref.
    torch.manual_seed(0)
    r_out, b_out = 3, 4  # out_features = 12
    r_in, b_in = 2, 5  # in_features = 10
    a_out = torch.randn(r_out, b_out, b_out)
    a_in = torch.randn(r_in, b_in, b_in)
    w = torch.randn(r_out * b_out, r_in * b_in)

    d_out, d_in = side_directions(a_out, a_in, w)

    d_out_ref = _block_diag(a_out) @ w
    d_in_ref = w @ _block_diag(a_in)
    assert torch.allclose(d_out, d_out_ref, atol=1e-5)
    assert torch.allclose(d_in, d_in_ref, atol=1e-5)


def test_side_directions_square_equal_blocks():
    torch.manual_seed(1)
    r, b = 4, 6  # square: out == in == 24
    a_out = torch.randn(r, b, b)
    a_in = torch.randn(r, b, b)
    w = torch.randn(r * b, r * b)
    d_out, d_in = side_directions(a_out, a_in, w)
    assert torch.allclose(d_out, _block_diag(a_out) @ w, atol=1e-5)
    assert torch.allclose(d_in, w @ _block_diag(a_in), atol=1e-5)


def test_cross_term_ratio_matches_dense_reference():
    # r_cross = ||A_out W A_in||_F / (||A_out W||_F + ||W A_in||_F), the finite
    # bilinear cross term simultaneous carries. Verify against a dense block-diag ref.
    torch.manual_seed(0)
    r_out, b_out = 3, 4
    r_in, b_in = 2, 5
    a_out = torch.randn(r_out, b_out, b_out)
    a_in = torch.randn(r_in, b_in, b_in)
    w = torch.randn(r_out * b_out, r_in * b_in)
    d_out, d_in = side_directions(a_out, a_in, w)

    got = cross_term_ratio(a_out, d_out, d_in)

    cross = _block_diag(a_out) @ w @ _block_diag(a_in)
    ref = cross.norm() / (d_out.norm() + d_in.norm())
    assert got == pytest.approx(ref.item(), rel=1e-5)


def test_cross_term_ratio_scales_linearly_with_angle():
    # cross ~ ||A||^2, denom ~ ||A|| -> ratio ~ ||A||: scaling both generators by s
    # scales r_cross by ~s (so a small operating angle => small cross term).
    torch.manual_seed(1)
    r, b = 2, 4
    a_out = torch.randn(r, b, b)
    a_in = torch.randn(r, b, b)
    w = torch.randn(r * b, r * b)
    d_out, d_in = side_directions(a_out, a_in, w)
    base = cross_term_ratio(a_out, d_out, d_in)
    d_out_s, d_in_s = side_directions(0.1 * a_out, 0.1 * a_in, w)
    scaled = cross_term_ratio(0.1 * a_out, d_out_s, d_in_s)
    assert scaled == pytest.approx(0.1 * base, rel=1e-4)


def test_side_directions_skew_inputs_compose_to_overlap():
    # End-to-end on skew generators: build A from skew vecs, form directions, and
    # confirm direction_overlap consumes them and returns finite geometry.
    from src.diag.skew_conditioning import vec_to_skew

    torch.manual_seed(2)
    r_out, b_out = 2, 4
    r_in, b_in = 3, 4
    a_out = vec_to_skew(torch.randn(r_out, b_out * (b_out - 1) // 2), b_out)
    a_in = vec_to_skew(torch.randn(r_in, b_in * (b_in - 1) // 2), b_in)
    w = torch.randn(r_out * b_out, r_in * b_in)
    d_out, d_in = side_directions(a_out, a_in, w)
    ov = direction_overlap(d_out, d_in)
    assert torch.isfinite(ov["cos"]).all()
    assert -1.0 - 1e-5 <= ov["cos"].item() <= 1.0 + 1e-5


# --- weight-only staleness split (out rotates -> in signal shift) ---------


def _ref_block_diag_skew(mat, b):
    r = mat.shape[0] // b
    blocks = []
    for i in range(r):
        blk = mat[i * b : (i + 1) * b, i * b : (i + 1) * b]
        blocks.append((blk - blk.T) / 2)
    return torch.stack(blocks)


def test_block_diag_skew_matches_reference():
    torch.manual_seed(0)
    r, b = 3, 4
    M = torch.randn(r * b, r * b)
    got = block_diag_skew(M, b)
    assert got.shape == (r, b, b)
    assert torch.allclose(got, _ref_block_diag_skew(M, b), atol=1e-6)


def test_block_diag_skew_is_skew():
    torch.manual_seed(1)
    r, b = 2, 5
    out = block_diag_skew(torch.randn(r * b, r * b), b)
    assert torch.allclose(out, -out.transpose(-1, -2), atol=1e-6)


def test_weight_only_sensitivity_zero_with_no_out_direction():
    # d_out = 0 -> the in-signal does not move -> sensitivity 0.
    torch.manual_seed(0)
    r_out, b_out = 3, 4
    r_in, b_in = 2, 5
    out_f, in_f = r_out * b_out, r_in * b_in
    g = torch.randn(out_f, in_f)
    w = torch.randn(out_f, in_f)
    s = weight_only_sensitivity(g, w, torch.zeros(out_f, in_f), block_size_in=b_in)
    assert s.item() == pytest.approx(0.0, abs=1e-12)


def test_weight_only_sensitivity_is_linear_in_out_direction():
    # sensitivity = ||block_skew(D_out^T G)|| / ||block_skew(W^T G)|| is linear in D_out.
    torch.manual_seed(2)
    r_out, b_out = 3, 4
    r_in, b_in = 2, 5
    out_f, in_f = r_out * b_out, r_in * b_in
    g = torch.randn(out_f, in_f)
    w = torch.randn(out_f, in_f)
    d_out = _block_diag(torch.randn(r_out, b_out, b_out)) @ w
    s1 = weight_only_sensitivity(g, w, d_out, block_size_in=b_in)
    s2 = weight_only_sensitivity(g, w, 3.0 * d_out, block_size_in=b_in)
    assert s1.item() > 0
    assert s2.item() == pytest.approx(3.0 * s1.item(), rel=1e-5)


def test_weight_only_sensitivity_matches_explicit_ratio():
    torch.manual_seed(3)
    r_out, b_out = 2, 4
    r_in, b_in = 3, 4
    out_f, in_f = r_out * b_out, r_in * b_in
    g = torch.randn(out_f, in_f)
    w = torch.randn(out_f, in_f)
    d_out = _block_diag(torch.randn(r_out, b_out, b_out)) @ w
    got = weight_only_sensitivity(g, w, d_out, block_size_in=b_in)
    k = block_diag_skew(w.T @ g, b_in)
    dk = block_diag_skew(d_out.T @ g, b_in)
    assert got.item() == pytest.approx((dk.norm() / k.norm()).item(), rel=1e-5)


def test_wsplit_validation_matches_poet_backward_up_to_sign():
    # The WSPLIT validate_cos compares block_skew(W_perm^T G_perm) (G = captured
    # ambient grad) against oft_R_in.grad. Replicate POETXSingleStepFunction.backward
    # (poetx_ops.py:55-62) and confirm they align up to sign (|cos|~1) -> the frame is
    # correct, so a *run's* validate_cos~0 is a DP local-vs-global mismatch, not a frame bug.
    from src.diag.skew_conditioning import vec_to_skew

    torch.manual_seed(0)
    r_out, b_out = 3, 4
    r_in, b_in = 2, 5
    out_f, in_f = r_out * b_out, r_in * b_in
    n_tok = 7
    x = torch.randn(n_tok, in_f)
    grad_y = torch.randn(n_tok, out_f)
    wx = torch.randn(out_f, in_f)
    perm_in_inv = torch.argsort(torch.randperm(in_f))
    perm_out_inv = torch.argsort(torch.randperm(out_f))
    tri = torch.triu_indices(b_in, b_in, 1)

    def _conj(m, p):
        return m.index_select(0, p).index_select(1, p)

    def _bdsv(full, b, rows, cols, factor=2.0):
        nb = full.shape[0] // b
        diag = full.reshape(nb, b, nb, b)[torch.arange(nb), :, torch.arange(nb), :]
        return factor * (diag - diag.transpose(-1, -2))[:, rows.long(), cols.long()]

    # exact POET backward in-side gradient
    g_their = x.reshape(-1, in_f).t() @ grad_y.reshape(-1, out_f)  # (in, out)
    k_true = vec_to_skew(_bdsv(_conj(g_their @ wx, perm_in_inv), b_in, tri[0], tri[1]), b_in)

    # WSPLIT validation formula (captured G = grad_y^T x)
    g_eff = grad_y.reshape(-1, out_f).t() @ x.reshape(-1, in_f)  # (out, in)
    g_perm = g_eff.index_select(0, perm_out_inv).index_select(1, perm_in_inv)
    w_perm = wx.index_select(0, perm_out_inv).index_select(1, perm_in_inv)
    k_mine = block_diag_skew(w_perm.t() @ g_perm, b_in)

    cos = (k_mine.flatten() @ k_true.flatten()) / (k_mine.norm() * k_true.norm())
    assert abs(cos.item()) > 0.999, cos.item()


# --- layer_coordination_metrics (per-layer assembler) ---------------------


def _ortho5(skew):
    from src.optim.poet_skew_muon import orthogonalize_skew_direction

    return orthogonalize_skew_direction(skew, method="muon", ns_steps=5)


def test_layer_metrics_assembles_finite_float_dict():
    torch.manual_seed(0)
    r_out, b_out = 2, 4
    r_in, b_in = 3, 4
    ne_out = b_out * (b_out - 1) // 2
    ne_in = b_in * (b_in - 1) // 2
    lie_m_out = torch.randn(r_out, ne_out)
    grad_out = torch.randn(r_out, ne_out)
    lie_m_in = torch.randn(r_in, ne_in)
    grad_in = torch.randn(r_in, ne_in)
    w = torch.randn(r_out * b_out, r_in * b_in)

    m = layer_coordination_metrics(
        lie_m_out,
        grad_out,
        lie_m_in,
        grad_in,
        w,
        block_size_out=b_out,
        block_size_in=b_in,
        orthogonalize_fn=_ortho5,
    )
    keys = {
        "mom_cos_out",
        "mom_cos_in",
        "cos_D_out_D_in",
        "r_joint",
        "gram_cond",
        "norm_D_out",
        "norm_D_in",
    }
    assert keys <= set(m)
    for k in keys:
        assert isinstance(m[k], float)
        assert math.isfinite(m[k])


def test_layer_metrics_mom_cos_matches_direct():
    torch.manual_seed(1)
    b = 4
    ne = b * (b - 1) // 2
    lie_m_out = torch.randn(2, ne)
    grad_out = torch.randn(2, ne)
    lie_m_in = torch.randn(2, ne)
    grad_in = torch.randn(2, ne)
    w = torch.randn(2 * b, 2 * b)
    m = layer_coordination_metrics(
        lie_m_out,
        grad_out,
        lie_m_in,
        grad_in,
        w,
        block_size_out=b,
        block_size_in=b,
        orthogonalize_fn=_ortho5,
    )
    assert m["mom_cos_out"] == pytest.approx(
        momentum_grad_cosine(lie_m_out, grad_out).item(), abs=1e-5
    )
    assert m["mom_cos_in"] == pytest.approx(
        momentum_grad_cosine(lie_m_in, grad_in).item(), abs=1e-5
    )


def test_layer_metrics_aligned_momentum_gives_high_cos():
    # When the fresh grad equals the momentum, the staleness arbiter reads ~1.
    torch.manual_seed(2)
    b = 4
    ne = b * (b - 1) // 2
    lie_m_out = torch.randn(2, ne)
    lie_m_in = torch.randn(2, ne)
    w = torch.randn(2 * b, 2 * b)
    m = layer_coordination_metrics(
        lie_m_out,
        lie_m_out.clone(),
        lie_m_in,
        lie_m_in.clone(),
        w,
        block_size_out=b,
        block_size_in=b,
        orthogonalize_fn=_ortho5,
    )
    assert m["mom_cos_out"] == pytest.approx(1.0, abs=1e-5)
    assert m["mom_cos_in"] == pytest.approx(1.0, abs=1e-5)


def test_layer_metrics_includes_raw_overlap_and_cross_term():
    # Tier-1 additions: cos_D_out_D_in_raw (overlap of the RAW -m directions, to tell
    # intrinsic decorrelation from orthogonalizer-induced) and r_cross (finite bilinear
    # cross-term magnitude). Both floats and finite; raw cos in [-1, 1].
    torch.manual_seed(3)
    r_out, b_out = 2, 4
    r_in, b_in = 3, 4
    ne_out = b_out * (b_out - 1) // 2
    ne_in = b_in * (b_in - 1) // 2
    m = layer_coordination_metrics(
        torch.randn(r_out, ne_out),
        torch.randn(r_out, ne_out),
        torch.randn(r_in, ne_in),
        torch.randn(r_in, ne_in),
        torch.randn(r_out * b_out, r_in * b_in),
        block_size_out=b_out,
        block_size_in=b_in,
        orthogonalize_fn=_ortho5,
    )
    for k in ("cos_D_out_D_in_raw", "r_cross"):
        assert k in m and isinstance(m[k], float) and math.isfinite(m[k])
    assert -1.0 - 1e-5 <= m["cos_D_out_D_in_raw"] <= 1.0 + 1e-5
    assert m["r_cross"] >= 0.0


def test_r_cross_scales_with_realized_angle():
    # r_cross is reported PHYSICAL: realized_angle (= eff∠) times the unit-generator
    # cross-term ratio, so it is directly "fraction of movement" at the operating angle.
    # cos/r_joint/gram are angle-invariant.
    torch.manual_seed(4)
    r_out, b_out = 2, 4
    r_in, b_in = 3, 4
    ne_out = b_out * (b_out - 1) // 2
    ne_in = b_in * (b_in - 1) // 2
    args = (
        torch.randn(r_out, ne_out),
        torch.randn(r_out, ne_out),
        torch.randn(r_in, ne_in),
        torch.randn(r_in, ne_in),
        torch.randn(r_out * b_out, r_in * b_in),
    )
    kw = dict(block_size_out=b_out, block_size_in=b_in, orthogonalize_fn=_ortho5)
    m_unit = layer_coordination_metrics(*args, realized_angle=1.0, **kw)
    m_scaled = layer_coordination_metrics(*args, realized_angle=0.02, **kw)
    assert m_scaled["r_cross"] == pytest.approx(0.02 * m_unit["r_cross"], rel=1e-5)
    assert m_scaled["cos_D_out_D_in"] == pytest.approx(m_unit["cos_D_out_D_in"], abs=1e-6)
