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


# --- cross-side decorrelation (decorrelate_sides) -------------------------
# Projects each layer's in/out generator off the other's weight-space direction so
# cos(D_out, D_in) -> 0, holding per-side Muon conditioning ~fixed. Isolates the
# inter-side gauge-redundancy channel (ANALYSIS §17.6). The one-sided modes zero the
# overlap exactly; 'symmetric' splits the perturbation and reduces it.
def _decorr_applied_cos(decorrelate, mode, seed=1):
    from src.diag.poet_coordination_diag import side_directions

    torch.manual_seed(seed)
    out_f, in_f, bo, bi = 12, 12, 12, 12  # block_count=1 (champion): one full block/side
    r_out, r_in = out_f // bo, in_f // bi
    oout = nn.Parameter(torch.zeros(r_out, bo * (bo - 1) // 2))
    oin = nn.Parameter(torch.zeros(r_in, bi * (bi - 1) // 2))
    oout.grad = torch.randn_like(oout)
    oin.grad = torch.randn_like(oin)
    W = torch.randn(out_f, in_f)
    lr = 0.05
    opt = LieOrthMomentum(
        [
            dict(params=[oin], use_skew=True, side="in", lr=lr),
            dict(params=[oout], use_skew=True, side="out", lr=lr),
        ],
        ortho_c=0.1,
        decorrelate_sides=decorrelate,
        decorrelate_mode=mode,
        layer_pairs=[(oout, oin, W, bo, bi)] if decorrelate else None,
    )
    opt.step()
    # oft_R started at 0, so .data = lr * generator -> recover the applied generators.
    A_out = vec_to_skew(oout.data / lr, bo)
    A_in = vec_to_skew(oin.data / lr, bi)
    d_out, d_in = side_directions(A_out, A_in, W.float())
    a, b = d_out.flatten(), d_in.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


@pytest.mark.parametrize("mode", ["in_off_out", "out_off_in"])
def test_decorrelate_zeroes_inter_side_overlap(mode):
    base = _decorr_applied_cos(False, mode)
    assert abs(base) > 0.02, f"baseline overlap should be non-trivial, got {base}"
    c = _decorr_applied_cos(True, mode)
    assert abs(c) < 1e-4, f"{mode} should drive cos(D_out,D_in)->0, got {c}"
    assert abs(c) < 0.05 * abs(base), f"{mode} should crush overlap: base={base} c={c}"


def test_decorrelate_symmetric_reduces_overlap():
    base = abs(_decorr_applied_cos(False, "symmetric"))
    sym = abs(_decorr_applied_cos(True, "symmetric"))
    assert sym < 0.5 * base, f"symmetric should reduce overlap: base={base} sym={sym}"


def test_decorrelate_off_is_unchanged():
    # decorrelate_sides=False must be bit-identical to the plain optimizer.
    assert _decorr_applied_cos(False, "in_off_out") == _decorr_applied_cos(False, "out_off_in")


def test_decorrelate_rejects_bad_mode():
    p = nn.Parameter(torch.zeros(1, 28))
    with pytest.raises(ValueError, match="decorrelate_mode"):
        LieOrthMomentum(
            [dict(params=[p], use_skew=True, side="in", lr=0.1)],
            decorrelate_sides=True,
            decorrelate_mode="bogus",
        )


# --- alternating-mode decorrelation (cross-step over-spend control) ----------
# Under the alternating champion only one side is written per step, so the inactive
# side's update buffer is zero and the simultaneous decorrelation is a literal no-op.
# The alternating path instead sources the inactive side's direction from its
# MAINTAINED momentum (lie_m) and projects the ACTIVE written side off it ("don't keep
# pushing along the direction the other side just moved"). Knobs: partial lambda,
# movement-preserving renorm, and a |cos| module-selective gate.
def _alt_decorr_dirs(decorrelate, mode="in_off_out", active_iter=1, seed=3, **extra):
    """One alternating step (active_iter=1 -> 'in' is written). Returns the applied
    active-in weight-space direction D_in and the inactive-out momentum direction
    D_out_mom, both in the W frame, plus the applied generator norm ||D_in||."""
    from poet_torch import alt_state

    from src.diag.poet_coordination_diag import side_directions

    torch.manual_seed(seed)
    f = b = 12
    ne = b * (b - 1) // 2
    oin = nn.Parameter(torch.zeros(1, ne))
    oin.grad = torch.randn(1, ne)
    oout = nn.Parameter(torch.zeros(1, ne))
    oout.grad = torch.randn(1, ne)
    W = torch.randn(f, f)
    lr = 0.05
    kw = dict(
        decorrelate_sides=decorrelate,
        decorrelate_mode=mode,
        layer_pairs=[(oout, oin, W, b, b)] if decorrelate else None,
    )
    kw.update(extra)
    opt = LieOrthMomentum(
        [
            dict(params=[oin], use_skew=True, side="in", lr=lr),
            dict(params=[oout], use_skew=True, side="out", lr=lr),
        ],
        ortho_c=0.1,
        alternating=True,
        **kw,
    )
    alt_state.set_iteration(active_iter)  # 1 -> active 'in'
    opt.step()
    alt_state.set_iteration(0)
    A_in = vec_to_skew(oin.data / lr, b)  # applied generator (oin started at 0)
    m_out = opt.state[oout]["lie_m"]  # inactive side's maintained momentum
    A_out_mom = orthogonalize_skew_direction(vec_to_skew(-m_out, b), method="muon", ns_steps=5)
    d_out_mom, d_in = side_directions(A_out_mom, A_in, W.float())
    return d_in, d_out_mom


def _cos(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def test_alternating_decorrelate_is_not_a_noop():
    # current code: alt+decorr skips (inactive buffer is zero) -> applied write identical
    # to decorrelate-off. The alternating path must instead modify the active write.
    base, _ = _alt_decorr_dirs(decorrelate=False)
    dec, _ = _alt_decorr_dirs(decorrelate=True, mode="in_off_out")
    assert _cos(base, dec) < 0.999, "alternating decorrelation must change the active write"


def test_alternating_decorrelate_removes_inactive_momentum_overlap():
    # in_off_out on an in-write step projects the active in-direction off the inactive
    # out-side momentum direction -> cos(D_in, D_out_mom) -> 0.
    d_in_base, d_out_mom = _alt_decorr_dirs(decorrelate=False)
    base = abs(_cos(d_in_base, d_out_mom))
    assert base > 0.02, f"baseline inactive-momentum overlap should be non-trivial, got {base}"
    d_in, d_out_mom2 = _alt_decorr_dirs(decorrelate=True, mode="in_off_out")
    assert (
        abs(_cos(d_in, d_out_mom2)) < 1e-3
    ), "decorrelation must zero the inactive-momentum overlap"


@pytest.mark.parametrize("lam", [0.25, 0.5, 1.0])
def test_alternating_decorrelate_lambda_scales_overlap(lam):
    # Partial projection leaves a (1-lambda) fraction of the original parallel component:
    # <D_in', D_out_mom> = (1-lambda) <D_in, D_out_mom>  (exact, renorm off).
    d_in0, d_mom = _alt_decorr_dirs(decorrelate=False)
    ip0 = (d_in0.flatten() @ d_mom.flatten()).item()
    assert abs(ip0) > 1e-3, f"baseline parallel component should be non-trivial, got {ip0}"
    d_in, d_mom2 = _alt_decorr_dirs(decorrelate=True, mode="in_off_out", decorrelate_lambda=lam)
    ip = (d_in.flatten() @ d_mom2.flatten()).item()
    assert ip == pytest.approx((1.0 - lam) * ip0, rel=2e-3, abs=1e-4)


def test_alternating_decorrelate_without_renorm_shrinks_movement():
    d_in0, _ = _alt_decorr_dirs(decorrelate=False)
    d_in, _ = _alt_decorr_dirs(decorrelate=True, mode="in_off_out", decorrelate_lambda=1.0)
    assert d_in.norm().item() < 0.999 * d_in0.norm().item(), "projection should reduce ||D_in||"


@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_alternating_decorrelate_renorm_preserves_movement(lam):
    # Movement normalization restores the active side's realized ||D_in|| to its
    # pre-projection value (only the direction changes).
    d_in0, _ = _alt_decorr_dirs(decorrelate=False)
    n0 = d_in0.norm().item()
    d_in, _ = _alt_decorr_dirs(
        decorrelate=True, mode="in_off_out", decorrelate_lambda=lam, decorrelate_renorm=True
    )
    assert d_in.norm().item() == pytest.approx(n0, rel=1e-4)


def test_alternating_decorrelate_threshold_gates():
    # |cos| below threshold -> layer untouched; above threshold -> decorrelated.
    d_in0, d_mom = _alt_decorr_dirs(decorrelate=False)
    cos = abs(_cos(d_in0, d_mom))
    assert cos > 0.02, f"need a non-trivial overlap to gate on, got {cos}"
    skipped, _ = _alt_decorr_dirs(
        decorrelate=True, mode="in_off_out", decorrelate_cos_threshold=cos + 0.1
    )
    assert _cos(d_in0, skipped) > 0.9999, "below-threshold layer must be left untouched"
    fired, _ = _alt_decorr_dirs(
        decorrelate=True, mode="in_off_out", decorrelate_cos_threshold=max(cos - 0.1, 0.0)
    )
    assert _cos(d_in0, fired) < 0.999, "above-threshold layer must be decorrelated"


def test_alternating_decorrelate_mode_targets_active_side():
    # On an in-write step (active_iter=1 -> 'in'): in_off_out & symmetric modify the
    # active in-write; out_off_in targets the inactive out side -> no-op.
    base, _ = _alt_decorr_dirs(decorrelate=False, active_iter=1)
    in_mode, _ = _alt_decorr_dirs(decorrelate=True, mode="in_off_out", active_iter=1)
    out_mode, _ = _alt_decorr_dirs(decorrelate=True, mode="out_off_in", active_iter=1)
    sym, _ = _alt_decorr_dirs(decorrelate=True, mode="symmetric", active_iter=1)
    assert _cos(base, in_mode) < 0.999, "in_off_out must modify the active in-write"
    assert _cos(base, out_mode) > 0.9999, "out_off_in must not touch the active in-write"
    assert _cos(base, sym) < 0.999, "symmetric must modify the active in-write"


# --- per-block dimension-dependent angle scaling (angle_dim_exp) -----------
# gen = ortho_c * (bsz/ref)^p * X. X (the orthogonalized direction) is independent of
# the scalar, so a block of size b gets its applied update scaled EXACTLY by (b/ref)^p
# vs p=0. ref defaults to hidden (passed by poet.py); here we pass it explicitly.
def _applied_norm_for_exp(p, b=8, ref=4, lr=0.05, seed=0):
    import torch

    torch.manual_seed(seed)
    ne = b * (b - 1) // 2
    oin = nn.Parameter(torch.zeros(1, ne))
    oin.grad = torch.randn(1, ne)
    opt = LieOrthMomentum(
        [dict(params=[oin], use_skew=True, side="in", lr=lr)],
        ortho_c=0.1,
        angle_dim_exp=p,
        angle_dim_ref=ref,
    )
    opt.step()
    return oin.data.norm().item()


@pytest.mark.parametrize("p", [-1.0, -0.5, 0.5, 1.0])
def test_angle_dim_exp_scales_block_by_dim_ratio(p):
    import math

    d0 = _applied_norm_for_exp(0.0)
    dp = _applied_norm_for_exp(p)
    assert math.isclose(dp / d0, (8 / 4) ** p, rel_tol=1e-4), (p, dp / d0, (8 / 4) ** p)


def test_angle_dim_exp_zero_is_noop():
    # p=0 (and a missing ref) must reproduce the unscaled champion update exactly.
    import torch

    torch.manual_seed(1)
    ne = 8 * 7 // 2
    a = nn.Parameter(torch.zeros(1, ne))
    a.grad = torch.randn(1, ne)
    b = nn.Parameter(a.detach().clone())
    b.grad = a.grad.clone()
    LieOrthMomentum([dict(params=[a], use_skew=True, side="in", lr=0.05)], ortho_c=0.1).step()
    LieOrthMomentum(
        [dict(params=[b], use_skew=True, side="in", lr=0.05)],
        ortho_c=0.1,
        angle_dim_exp=0.0,
        angle_dim_ref=512,
    ).step()
    assert torch.allclose(a.data, b.data, atol=1e-7)
