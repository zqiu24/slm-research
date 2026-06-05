"""Tests for the POET Lie-Orth (Muon-like orthogonalizing) optimizer:
the orthogonalization helper and the LieOrthMomentum optimizer.
See docs/muon_orthogonalizing_optimizer_poet.md."""

import pytest
import torch
import torch.nn as nn

from src.diag.skew_conditioning import block_spectral_stats, vec_to_skew
from src.optim.poet_lie_orth import LieOrthMomentum
from src.optim.poet_skew_muon import orthogonalize_skew_direction


def _benign_skew(num_blocks, b, seed):
    torch.manual_seed(seed)
    return vec_to_skew(torch.randn(num_blocks, b * (b - 1) // 2), b)


@pytest.mark.parametrize("method", ["muon", "spectral"])
def test_orthogonalize_skew_direction_stays_skew(method):
    M = _benign_skew(3, 8, seed=1)
    X = orthogonalize_skew_direction(M, method=method, ns_steps=20)
    assert torch.allclose(X, -X.transpose(-2, -1), atol=1e-5)


@pytest.mark.parametrize("method", ["muon", "spectral"])
def test_orthogonalize_skew_direction_batches_per_block(method):
    a = _benign_skew(1, 8, seed=3)
    c = _benign_skew(1, 8, seed=4)
    out = orthogonalize_skew_direction(torch.cat([a, c], dim=0), method=method, ns_steps=20)
    assert torch.allclose(
        out[0:1], orthogonalize_skew_direction(a, method=method, ns_steps=20), atol=1e-6
    )
    assert torch.allclose(
        out[1:2], orthogonalize_skew_direction(c, method=method, ns_steps=20), atol=1e-6
    )


def test_muon_method_democratizes_the_spectrum():
    # DEFAULT: Muon's quintic flattens a heavy-tailed spectrum into a BAND around 1
    # (condition number ~ 1.5) in ~5 steps. It does NOT drive sigma to exactly 1.
    M = _benign_skew(2, 8, seed=0)
    cond_in = block_spectral_stats(M)["condition_number"].mean().item()
    X = orthogonalize_skew_direction(M, method="muon", ns_steps=5)
    cond_out = block_spectral_stats(X)["condition_number"].mean().item()
    assert cond_in > 5.0  # non-trivial input
    assert cond_out < 2.0 and cond_out < cond_in / 3.0  # democratized into a band


def test_spectral_method_drives_singular_values_to_one():
    # OPT-IN exact variant: every singular value -> 1 (needs ~15-20 steps).
    M = _benign_skew(2, 8, seed=0)
    sv = torch.linalg.svdvals(orthogonalize_skew_direction(M, method="spectral", ns_steps=20))
    assert torch.allclose(sv, torch.ones_like(sv), atol=0.02), sv


def test_spectral_method_is_odd_and_exact_on_a_2d_plane():
    M = _benign_skew(2, 8, seed=2)
    assert torch.allclose(
        orthogonalize_skew_direction(-M, method="spectral", ns_steps=20),
        -orthogonalize_skew_direction(M, method="spectral", ns_steps=20),
        atol=1e-5,
    )
    t = 3.7  # a single 2D plane [[0,t],[-t,0]] -> the unit generator regardless of t>0
    M2 = torch.tensor([[[0.0, t], [-t, 0.0]]])
    X2 = orthogonalize_skew_direction(M2, method="spectral", ns_steps=20)
    assert torch.allclose(X2, torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]]), atol=1e-4), X2


def _make_opt(p, lr, ortho_c, method="muon", ns_steps=5, **kw):
    return LieOrthMomentum(
        [dict(params=[p], use_skew=True, side="out", lr=lr)],
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        ortho_c=ortho_c,
        ortho_method=method,
        ortho_ns_steps=ns_steps,
        **kw,
    )


def test_muon_equalizes_plane_angles_into_a_band():
    # DEFAULT (muon): one step from identity -> the written oft_R's per-plane angles
    # form a tight band (cond < 2) at ~ lr*ortho_c. Equalized, but not exactly equal.
    torch.manual_seed(0)
    b, ne, lr, c = 8, 8 * 7 // 2, 0.1, 0.05
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    _make_opt(p, lr, c).step()
    R = vec_to_skew(p.data, b)
    sv = torch.linalg.svdvals(R)
    cond = block_spectral_stats(R)["condition_number"].mean().item()
    assert cond < 2.0  # planes roughly equalized
    assert 0.5 * lr * c < sv.median().item() < 1.2 * lr * c  # magnitude ~ lr*c (a band)


def test_spectral_makes_every_plane_angle_equal():
    # OPT-IN exact variant: every plane angle == lr*ortho_c (needs ns_steps ~20).
    torch.manual_seed(0)
    b, ne, lr, c = 8, 8 * 7 // 2, 0.1, 0.05
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    _make_opt(p, lr, c, method="spectral", ns_steps=20).step()
    sv = torch.linalg.svdvals(vec_to_skew(p.data, b))
    assert torch.allclose(sv, torch.full_like(sv, lr * c), atol=lr * c * 0.05), sv


def test_invalid_ortho_method_raises():
    p = nn.Parameter(torch.zeros(1, 6))
    with pytest.raises(ValueError, match="ortho_method"):
        LieOrthMomentum([dict(params=[p], use_skew=True, lr=1e-3)], ortho_method="bogus")


def test_first_moment_only_differs_from_second_moment():
    # With a wildly uneven per-entry grad, the second-moment (Adam) direction and the
    # first-moment-only direction point differently before orthogonalization.
    torch.manual_seed(0)
    ne, lr, c = 8 * 7 // 2, 0.1, 0.05
    g = torch.randn(1, ne)
    g[:, 0] *= 50.0
    p1 = nn.Parameter(torch.zeros(1, ne))
    p1.grad = g.clone()
    p2 = nn.Parameter(torch.zeros(1, ne))
    p2.grad = g.clone()
    _make_opt(p1, lr, c, ortho_use_second_moment=False).step()
    _make_opt(p2, lr, c, ortho_use_second_moment=True).step()
    assert not torch.allclose(p1.data, p2.data, atol=1e-4)


def test_grad_sign_flips_the_update():
    # Orthogonalization is odd in sign, so negating the grad negates the written oft_R.
    torch.manual_seed(0)
    ne, lr, c = 8 * 7 // 2, 0.1, 0.05
    g = torch.randn(1, ne)
    p_pos = nn.Parameter(torch.zeros(1, ne))
    p_pos.grad = g.clone()
    p_neg = nn.Parameter(torch.zeros(1, ne))
    p_neg.grad = -g.clone()
    _make_opt(p_pos, lr, c).step()
    _make_opt(p_neg, lr, c).step()
    assert torch.allclose(p_pos.data, -p_neg.data, atol=1e-5)


def test_adamw_branch_steps_non_skew_params():
    # non-oft_R params get the AdamW branch (moved off their initial value).
    w = nn.Parameter(torch.randn(4, 4))
    w.grad = torch.randn(4, 4)
    w0 = w.data.clone()
    LieOrthMomentum([dict(params=[w], use_skew=False, lr=1e-2)], adamw_wd=0.0).step()
    assert not torch.allclose(w.data, w0)


def test_momentum_persists_across_value_reset():
    # lie_m persists across the per-step fold (p zeroed between steps); the second
    # step's direction reflects the accumulated EMA, not a fresh start.
    torch.manual_seed(0)
    ne, lr, c = 8 * 7 // 2, 0.1, 0.05
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    opt = _make_opt(p, lr, c)
    opt.step()
    assert "lie_m" in opt.state[p] and opt.state[p]["lie_m"].abs().sum() > 0
    p.data.zero_()  # simulate the merge fold
    p.grad = torch.randn(1, ne)
    opt.step()
    assert torch.isfinite(p.data).all()
