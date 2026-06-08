"""Tests for the POET Lie-Orth (Muon-like orthogonalizing) optimizer:
the orthogonalization helper and the LieOrthMomentum optimizer.
See docs/muon_orthogonalizing_optimizer_poet.md."""

import pytest
import torch
import torch.nn as nn

from src.diag.skew_conditioning import block_spectral_stats, vec_to_skew
from src.optim.poet_lie_orth import LieOrthMomentum
from src.optim.poet_skew_muon import orthogonalize_skew_direction


@pytest.fixture(autouse=True)
def _isolate_default_dtype():
    # These tests assume the canonical float32 default. Other test files
    # (e.g. test_poetx_layer.py) set float64 globally and never restore it; when
    # they run earlier in the same process the leaked default breaks the
    # precision-tuned spectral asserts. Force float32 per test and don't leak.
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(torch.float32)


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


def test_first_moment_only_skips_lie_v_buffer():
    # Default (first-moment-only): the direction is just -m, so the second-moment
    # buffer lie_v should never be allocated or maintained.
    torch.manual_seed(0)
    ne = 8 * 7 // 2
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    opt = _make_opt(p, 0.1, 0.05)  # ortho_use_second_moment defaults to False
    opt.step()
    assert "lie_m" in opt.state[p]
    assert "lie_v" not in opt.state[p]


def test_second_moment_allocates_lie_v_buffer():
    # With the second moment on, lie_v is allocated and updated.
    torch.manual_seed(0)
    ne = 8 * 7 // 2
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    opt = _make_opt(p, 0.1, 0.05, ortho_use_second_moment=True)
    opt.step()
    assert "lie_v" in opt.state[p]
    assert opt.state[p]["lie_v"].abs().sum() > 0


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


def test_batched_step_matches_solo_steps_same_block_size():
    # Two skew params of the SAME block size in one group must get the same update
    # batched together as they would stepped alone.
    torch.manual_seed(0)
    b = 8
    ne = b * (b - 1) // 2
    g_a = torch.randn(2, ne)
    g_b = torch.randn(3, ne)

    p_a = nn.Parameter(torch.zeros(2, ne))
    p_a.grad = g_a.clone()
    p_b = nn.Parameter(torch.zeros(3, ne))
    p_b.grad = g_b.clone()
    LieOrthMomentum(
        [dict(params=[p_a, p_b], use_skew=True, side="out", lr=0.1)],
        ortho_c=0.05,
        ortho_method="muon",
        ortho_ns_steps=5,
    ).step()

    p_a2 = nn.Parameter(torch.zeros(2, ne))
    p_a2.grad = g_a.clone()
    _make_opt(p_a2, 0.1, 0.05).step()
    p_b2 = nn.Parameter(torch.zeros(3, ne))
    p_b2.grad = g_b.clone()
    _make_opt(p_b2, 0.1, 0.05).step()

    assert torch.allclose(p_a.data, p_a2.data, atol=1e-6)
    assert torch.allclose(p_b.data, p_b2.data, atol=1e-6)


def test_batched_step_handles_mixed_block_sizes():
    # Different block sizes in one group -> separate buckets; each matches its solo run.
    torch.manual_seed(1)
    b1, b2 = 8, 6
    ne1, ne2 = b1 * (b1 - 1) // 2, b2 * (b2 - 1) // 2
    g1 = torch.randn(1, ne1)
    g2 = torch.randn(1, ne2)

    p1 = nn.Parameter(torch.zeros(1, ne1))
    p1.grad = g1.clone()
    p2 = nn.Parameter(torch.zeros(1, ne2))
    p2.grad = g2.clone()
    LieOrthMomentum(
        [dict(params=[p1, p2], use_skew=True, side="out", lr=0.1)],
        ortho_c=0.05,
        ortho_method="muon",
        ortho_ns_steps=5,
    ).step()

    p1b = nn.Parameter(torch.zeros(1, ne1))
    p1b.grad = g1.clone()
    _make_opt(p1b, 0.1, 0.05).step()
    p2b = nn.Parameter(torch.zeros(1, ne2))
    p2b.grad = g2.clone()
    _make_opt(p2b, 0.1, 0.05).step()

    assert torch.allclose(p1.data, p1b.data, atol=1e-6)
    assert torch.allclose(p2.data, p2b.data, atol=1e-6)


def test_batched_step_alternating_writes_only_active_side():
    # Alternating now reads the SHARED alt_state iteration (not the internal
    # _alt_step counter). The iterations are chosen so they do NOT coincide with the
    # internal counter (which would start at 0): iteration 1 -> active 'in',
    # iteration 2 -> active 'out'. Momentum accrues on BOTH sides; only the active
    # side's oft_R is written. This sequence FAILS against the old _alt_step source
    # (step 1 would pick 'out') and PASSES against the alt_state source.
    from poet_torch import alt_state

    torch.manual_seed(2)
    b = 8
    ne = b * (b - 1) // 2
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_in.grad = torch.randn(1, ne)
    p_out = nn.Parameter(torch.zeros(1, ne))
    p_out.grad = torch.randn(1, ne)
    opt = LieOrthMomentum(
        [
            dict(params=[p_in], use_skew=True, side="in", lr=0.1),
            dict(params=[p_out], use_skew=True, side="out", lr=0.1),
        ],
        ortho_c=0.05,
        ortho_method="muon",
        ortho_ns_steps=5,
        alternating=True,
    )
    alt_state.set_iteration(1)  # active 'in' (old _alt_step=0 would pick 'out')
    opt.step()
    assert p_in.data.abs().sum() > 0 and torch.allclose(p_out.data, torch.zeros_like(p_out))
    p_in.grad = torch.randn(1, ne)
    p_out.grad = torch.randn(1, ne)
    p_in.data.zero_()  # simulate the per-step fold of the just-written side
    alt_state.set_iteration(2)  # active 'out' (old _alt_step=1 would pick 'in')
    opt.step()
    assert p_out.data.abs().sum() > 0 and torch.allclose(p_in.data, torch.zeros_like(p_in))
    alt_state.set_iteration(0)  # restore the module global for later tests


def test_replicated_buffer_owns_all_params():
    # At (dp_rank=0, dp_world=1) every skew param is owned, so its buffer slice is
    # written (non-zero) — the replicated path covers everything.
    torch.manual_seed(0)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2)]
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = LieOrthMomentum([dict(params=ps, use_skew=True, side="out", lr=0.1)], ortho_c=0.05)
    opt._lie_m_update(active=None)
    buf, slices = opt._skew_update_buffer(dp_rank=0, dp_world=1, active=None)
    assert len(slices) == 3
    for off, n, _, _ in slices:
        assert buf[off : off + n].abs().sum() > 0  # written, not left as zeros


@pytest.mark.parametrize("dp_world", [2, 3, 4])
def test_sharded_buffers_sum_to_replicated(dp_world):
    # Build several skew params of DIFFERENT shapes (heterogeneous n_blocks), one opt.
    torch.manual_seed(0)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2, 5)]
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = LieOrthMomentum(
        [dict(params=ps, use_skew=True, side="out", lr=0.1)],
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        ortho_c=0.05,
        ortho_method="muon",
        ortho_ns_steps=5,
    )
    opt._lie_m_update(active=None)  # momentum once (shared across the simulated ranks)
    replicated, _ = opt._skew_update_buffer(dp_rank=0, dp_world=1, active=None)
    summed = torch.zeros_like(replicated)
    for r in range(dp_world):
        buf_r, _ = opt._skew_update_buffer(dp_rank=r, dp_world=dp_world, active=None)
        summed += buf_r
    # Each param is owned by exactly one rank; zeros elsewhere ⇒ sum == replicated, exactly.
    assert torch.equal(summed, replicated), (summed - replicated).abs().max()


def test_sharded_owns_each_param_exactly_once():
    # Every skew param must be written by exactly one rank (no double-count, no drop).
    torch.manual_seed(0)
    ne = 8 * 7 // 2
    ps = [nn.Parameter(torch.zeros(nb, ne)) for nb in (1, 3, 2, 5, 4)]
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = LieOrthMomentum(
        [dict(params=ps, use_skew=True, side="out", lr=0.1)],
        ortho_c=0.05,
    )
    opt._lie_m_update(active=None)
    dp_world = 3
    nonzero_owners = [0] * len(ps)
    for r in range(dp_world):
        buf, slices = opt._skew_update_buffer(dp_rank=r, dp_world=dp_world, active=None)
        for i, (off, n, _, _) in enumerate(slices):
            if buf[off : off + n].abs().sum() > 0:
                nonzero_owners[i] += 1
    assert all(c == 1 for c in nonzero_owners), nonzero_owners


def test_true_single_side_freezes_inactive_momentum(monkeypatch):
    # true_single_side: inactive side's momentum must NOT advance/decay, even with
    # a (zeros) grad present; active side updates as usual. Active comes from alt_state.
    from poet_torch import alt_state

    torch.manual_seed(7)
    b = 8
    ne = b * (b - 1) // 2
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_out = nn.Parameter(torch.zeros(1, ne))
    p_in.grad = torch.randn(1, ne)
    p_out.grad = torch.zeros(1, ne)  # frozen side gets zeros from the layer backward
    opt = LieOrthMomentum(
        [
            dict(params=[p_in], use_skew=True, side="in", lr=0.1),
            dict(params=[p_out], use_skew=True, side="out", lr=0.1),
        ],
        ortho_c=0.05,
        true_single_side=True,
    )
    alt_state.set_iteration(1)  # active "in"
    opt.step()
    assert p_in.data.abs().sum() > 0  # active side written
    assert torch.allclose(p_out.data, torch.zeros_like(p_out))  # inactive not written
    assert "lie_m" in opt.state[p_in]
    # inactive side's momentum buffer must be absent OR all-zero (never advanced)
    assert "lie_m" not in opt.state[p_out] or opt.state[p_out]["lie_m"].abs().sum() == 0


def test_true_single_side_active_flips_with_iteration():
    from poet_torch import alt_state

    torch.manual_seed(8)
    ne = 8 * 7 // 2
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_out = nn.Parameter(torch.zeros(1, ne))
    p_in.grad = torch.zeros(1, ne)
    p_out.grad = torch.randn(1, ne)
    opt = LieOrthMomentum(
        [
            dict(params=[p_in], use_skew=True, side="in", lr=0.1),
            dict(params=[p_out], use_skew=True, side="out", lr=0.1),
        ],
        ortho_c=0.05,
        true_single_side=True,
    )
    alt_state.set_iteration(2)  # active "out"
    opt.step()
    assert p_out.data.abs().sum() > 0
    assert torch.allclose(p_in.data, torch.zeros_like(p_in))
