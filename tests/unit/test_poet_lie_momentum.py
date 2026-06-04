"""Tests for the Lie-algebra momentum optimizer (POET q_optimizer=lie_algebra)."""

import torch
import torch.nn as nn

from src.diag.skew_conditioning import skew_to_vec, vec_to_skew


def _skew_space_reference(g_vec, b, b1, b2, eps, lr, v_mode):
    """Paper-faithful skew-space Pion first/second-moment step from ZERO state,
    returns the expected oft_R update (vec-space) = lr * skew_to_vec(A)."""
    G = vec_to_skew(g_vec, b)  # (n_blocks, b, b) skew
    M = (1 - b1) * G
    if v_mode == "scalar":
        v = (1 - b2) * (G * G).sum(dim=(-2, -1), keepdim=True)  # ‖G‖_F^2 full matrix
    else:
        v = (1 - b2) * (G * G)
    A = -M / (v.sqrt() + eps)
    return lr * skew_to_vec(A, b)


def _make_opt(p, lr, v_mode):
    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    return LieAlgebraMomentum(
        [dict(params=[p], use_skew=True, lr=lr)],
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        v_mode=v_mode,
    )


def test_first_step_matches_pion_scalar_v():
    torch.manual_seed(0)
    b, ne, lr = 4, 6, 1e-3
    p = nn.Parameter(torch.zeros(1, ne))  # born at identity
    p.grad = torch.randn(1, ne)
    expected = _skew_space_reference(p.grad.clone(), b, 0.9, 0.95, 1e-8, lr, "scalar")
    _make_opt(p, lr, "scalar").step()
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()


def test_first_step_matches_pion_elementwise_v():
    torch.manual_seed(0)
    b, ne, lr = 4, 6, 1e-3
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    expected = _skew_space_reference(p.grad.clone(), b, 0.9, 0.95, 1e-8, lr, "elementwise")
    _make_opt(p, lr, "elementwise").step()
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()


def test_momentum_persists_across_value_reset():
    # Two steps with p zeroed between (simulating the fold) -> lie_m/lie_v are
    # EMAs that ACCUMULATE across the zeroing (NOT reset).
    torch.manual_seed(1)
    ne, lr, b1, b2, eps = 6, 1e-3, 0.9, 0.95, 1e-8
    p = nn.Parameter(torch.zeros(1, ne))
    opt = _make_opt(p, lr, "scalar")
    g1 = torch.randn(1, ne)
    g2 = torch.randn(1, ne)
    p.grad = g1.clone()
    opt.step()
    p.data.zero_()
    p.grad = g2.clone()
    opt.step()
    # hand-compute the 2nd step from the persisted state
    m = b1 * ((1 - b1) * g1) + (1 - b1) * g2
    v = b2 * ((1 - b2) * 2 * (g1 * g1).sum()) + (1 - b2) * 2 * (g2 * g2).sum()
    expected = lr * (-m / (v.sqrt() + eps))
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()
    st = opt.state[p]
    assert torch.allclose(st["lie_m"], m, atol=1e-7)


def test_v_shapes():
    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    ne = 6
    for v_mode, vshape in (("scalar", (2, 1)), ("elementwise", (2, ne))):
        p = nn.Parameter(torch.zeros(2, ne))
        p.grad = torch.randn(2, ne)
        opt = LieAlgebraMomentum([dict(params=[p], use_skew=True, lr=1e-3)], v_mode=v_mode)
        opt.step()
        assert tuple(opt.state[p]["lie_v"].shape) == vshape


def test_adamw_branch_steps_without_error():
    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    p = nn.Parameter(torch.randn(3, 5))
    g = torch.randn(3, 5)
    p.grad = g.clone()
    before = p.data.clone()
    opt = LieAlgebraMomentum([dict(params=[p], use_skew=False, lr=1e-2)])
    opt.step()
    assert not torch.allclose(p.data, before)  # standard AdamW moved it


def test_build_lie_param_groups_side_tagged_and_scaled():
    import torch.nn as nn

    from src.optim.poet_lie_momentum import _build_lie_param_groups

    skew_in = [nn.Parameter(torch.zeros(1, 6))]
    skew_out = [nn.Parameter(torch.zeros(1, 6))]
    adamw = [nn.Parameter(torch.zeros(4))]
    groups = _build_lie_param_groups(skew_in, skew_out, adamw, lr=1e-3, min_lr=1e-5, scale=0.5)

    g_in = next(g for g in groups if g.get("side") == "in")
    g_out = next(g for g in groups if g.get("side") == "out")
    g_adam = next(g for g in groups if not g["use_skew"])
    assert (
        g_in["use_skew"]
        and g_in["lr"] == 5e-4
        and g_in["max_lr"] == 5e-4
        and g_in["min_lr"] == 5e-6
    )
    assert g_out["use_skew"] and g_out["side"] == "out" and g_out["lr"] == 5e-4
    assert g_adam["lr"] == 1e-3 and g_adam["max_lr"] == 1e-3 and g_adam["min_lr"] == 1e-5
    assert g_adam["side"] is None


def test_build_lie_param_groups_drops_empty_sides():
    import torch.nn as nn

    from src.optim.poet_lie_momentum import _build_lie_param_groups

    assert _build_lie_param_groups([], [], [], 1e-3, 1e-5, 0.5) == []
    # only out-side present -> single side-tagged group
    groups = _build_lie_param_groups([], [nn.Parameter(torch.zeros(1, 6))], [], 1e-3, 1e-5, 0.5)
    assert len(groups) == 1 and groups[0]["side"] == "out"


def test_split_poet_lie_params_buckets_by_name():
    from src.optim.poet_lie_momentum import _split_poet_lie_params

    m = nn.Module()
    m.q_oft_R_in = nn.Parameter(torch.zeros(1, 6))
    m.q_oft_R_out = nn.Parameter(torch.zeros(1, 6))
    m.embed = nn.Parameter(torch.zeros(4))
    skew_in, skew_out, adamw = _split_poet_lie_params([m])
    assert len(skew_in) == 1 and skew_in[0] is m.q_oft_R_in
    assert len(skew_out) == 1 and skew_out[0] is m.q_oft_R_out
    assert len(adamw) == 1 and adamw[0] is m.embed


def _make_alt_opt(p_in, p_out, lr, alternate_every=1, alternating=True):
    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    return LieAlgebraMomentum(
        [
            dict(params=[p_in], use_skew=True, side="in", lr=lr),
            dict(params=[p_out], use_skew=True, side="out", lr=lr),
        ],
        v_mode="scalar",
        alternating=alternating,
        alternate_every=alternate_every,
    )


def _alt_run(opt, p_in, p_out, ne, expected_out_active):
    """Drive `opt` one step per entry, zeroing oft_R between steps (the fold), and
    assert exactly the active side is written each step."""
    for expect_out in expected_out_active:
        p_in.data.zero_()
        p_out.data.zero_()
        p_in.grad = torch.randn(1, ne)
        p_out.grad = torch.randn(1, ne)
        opt.step()
        if expect_out:
            assert torch.count_nonzero(p_out.data) > 0 and torch.count_nonzero(p_in.data) == 0
        else:
            assert torch.count_nonzero(p_in.data) > 0 and torch.count_nonzero(p_out.data) == 0


def test_alternating_writes_one_side_and_flips():
    torch.manual_seed(0)
    ne, lr = 6, 1e-3
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_out = nn.Parameter(torch.zeros(1, ne))
    opt = _make_alt_opt(p_in, p_out, lr)
    _alt_run(opt, p_in, p_out, ne, [True, False, True, False])  # out, in, out, in


def test_alternating_accumulates_momentum_on_inactive_side():
    torch.manual_seed(1)
    ne, lr = 6, 1e-3
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_out = nn.Parameter(torch.zeros(1, ne))
    opt = _make_alt_opt(p_in, p_out, lr)
    p_in.grad = torch.randn(1, ne)
    p_out.grad = torch.randn(1, ne)
    opt.step()  # step 0: out active
    assert torch.count_nonzero(p_in.data) == 0  # in NOT written
    assert torch.count_nonzero(opt.state[p_in]["lie_m"]) > 0  # but in momentum accumulated


def test_alternate_every_2_holds_each_side_two_steps():
    torch.manual_seed(2)
    ne, lr = 6, 1e-3
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_out = nn.Parameter(torch.zeros(1, ne))
    opt = _make_alt_opt(p_in, p_out, lr, alternate_every=2)
    _alt_run(opt, p_in, p_out, ne, [True, True, False, False, True])  # out,out,in,in,out


def test_alternating_false_writes_both_sides():
    torch.manual_seed(3)
    ne, lr = 6, 1e-3
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_out = nn.Parameter(torch.zeros(1, ne))
    opt = _make_alt_opt(p_in, p_out, lr, alternating=False)
    p_in.grad = torch.randn(1, ne)
    p_out.grad = torch.randn(1, ne)
    opt.step()
    assert torch.count_nonzero(p_in.data) > 0 and torch.count_nonzero(p_out.data) > 0


def _rms_opt(p, lr, rms, rms_c=0.2, v_mode="elementwise"):
    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    return LieAlgebraMomentum(
        [dict(params=[p], use_skew=True, side="out", lr=lr)],
        v_mode=v_mode,
        rms=rms,
        rms_c=rms_c,
    )


def test_rms_scaling_matches_reference():
    from src.diag.skew_conditioning import block_size_from_nelems

    torch.manual_seed(0)
    ne, lr, rms_c, b1, b2, eps = 6, 1e-3, 0.2, 0.9, 0.95, 1e-8
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    g = p.grad.clone()
    # Stage 1 (elementwise Adam, first step from 0) then Stage 2 (W-free RMS):
    m = (1 - b1) * g
    v = (1 - b2) * (g * g)
    A = -m / (v.sqrt() + eps)
    b = block_size_from_nelems(A.shape[1])
    dim_const = (A.shape[0] * b) ** 0.5
    alpha = rms_c * dim_const / (torch.linalg.norm(A) + eps)
    expected = lr * alpha * A
    _rms_opt(p, lr, rms=True, rms_c=rms_c).step()
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()


def test_rms_makes_oft_r_norm_grad_independent():
    # The whole point: ‖oft_R‖_F = lr·rms_c·√(n_blocks·block_size), regardless of
    # the gradient magnitude (scale consistency).
    from src.diag.skew_conditioning import block_size_from_nelems

    ne, lr, rms_c = 6, 1e-3, 0.2
    b = block_size_from_nelems(ne)
    target = lr * rms_c * (1 * b) ** 0.5
    for scale in (1e-3, 1.0, 1e3):  # wildly different gradient magnitudes
        p = nn.Parameter(torch.zeros(1, ne))
        p.grad = scale * torch.randn(1, ne)
        _rms_opt(p, lr, rms=True, rms_c=rms_c).step()
        assert abs(float(torch.linalg.norm(p.data)) - target) < 1e-6, scale


def test_rms_off_is_unscaled():
    # rms=False reproduces the current oft_R = lr*A (no Stage 2).
    torch.manual_seed(2)
    ne, lr, b1, b2, eps = 6, 1e-3, 0.9, 0.95, 1e-8
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    g = p.grad.clone()
    A = -((1 - b1) * g) / (((1 - b2) * (g * g)).sqrt() + eps)
    expected = lr * A
    _rms_opt(p, lr, rms=False).step()
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()


def test_rms_norm_scales_with_sqrt_d_across_sizes():
    # Two different widths, same rms_c: ‖oft_R‖/√(n_blocks·block_size) is equal.
    from src.diag.skew_conditioning import block_size_from_nelems

    lr, rms_c = 1e-3, 0.2
    ratios = []
    for d in (4, 8):
        ne = d * (d - 1) // 2
        p = nn.Parameter(torch.zeros(1, ne))
        p.grad = torch.randn(1, ne)
        _rms_opt(p, lr, rms=True, rms_c=rms_c).step()
        b = block_size_from_nelems(ne)
        ratios.append(float(torch.linalg.norm(p.data)) / (1 * b) ** 0.5)
    assert abs(ratios[0] - ratios[1]) < 1e-7, ratios


def test_rms_is_per_block_consistent():
    """With RMS on, each block's applied update has Frobenius norm
    rms_c*sqrt(block_size) regardless of that block's update magnitude.

    The generators A=-m/(sqrt(v)+eps) are seeded with genuinely DIFFERENT
    per-block magnitudes (b1=b2=1 so step() leaves the seeded state untouched).
    The OLD global-alpha formula scales every block by one shared factor, so the
    resulting per-block norms stay unequal -> this assertion fails on it. Only
    the per-block formula renormalizes each block to the same target.
    """
    import torch

    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    torch.manual_seed(0)
    bsz = 8
    n_elems = bsz * (bsz - 1) // 2  # 28
    n_blocks = 4
    p = torch.nn.Parameter(torch.zeros(n_blocks, n_elems, dtype=torch.float64))
    p.grad = torch.zeros(n_blocks, n_elems, dtype=torch.float64)  # non-None; b1=b2=1 -> ignored

    rms_c = 0.2
    opt = LieAlgebraMomentum(
        [{"params": [p], "use_skew": True, "side": "out", "lr": 1.0}],
        b1=1.0,
        b2=1.0,
        eps=1e-12,
        v_mode="elementwise",
        rms=True,
        rms_c=rms_c,
    )
    # Seed per-block-DIFFERENT momentum (block scales 10 / 0.1 / 1 / 5) with v=1,
    # so A=-m has genuinely different per-block Frobenius norms. b1=b2=1 means
    # step() does not overwrite these from the (zero) grad.
    m = torch.randn(n_blocks, n_elems, dtype=torch.float64)
    m[0] *= 10.0
    m[1] *= 0.1
    m[2] *= 1.0
    m[3] *= 5.0
    opt.state[p]["lie_m"] = m.clone()
    opt.state[p]["lie_v"] = torch.ones(n_blocks, n_elems, dtype=torch.float64)
    opt.step()

    target = rms_c * (bsz**0.5)  # per-block Frobenius of the (lr=1) update
    per_block = torch.linalg.norm(p.detach(), dim=1)
    assert torch.allclose(
        per_block, torch.full((n_blocks,), target, dtype=torch.float64), atol=1e-6
    ), per_block


def test_rms_block_count_1_unchanged():
    """At n_blocks==1 the per-block RMS equals the old global formula."""
    import torch

    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    torch.manual_seed(1)
    bsz = 8
    n_elems = bsz * (bsz - 1) // 2
    p = torch.nn.Parameter(torch.zeros(1, n_elems, dtype=torch.float64))
    p.grad = torch.randn(1, n_elems, dtype=torch.float64)
    opt = LieAlgebraMomentum(
        [{"params": [p], "use_skew": True, "side": "out", "lr": 1.0}],
        b1=0.0,
        b2=0.0,
        eps=1e-12,
        v_mode="elementwise",
        rms=True,
        rms_c=0.2,
    )
    opt.step()
    assert torch.isclose(
        torch.linalg.norm(p.detach()),
        torch.tensor(0.2 * (bsz**0.5), dtype=torch.float64),
        atol=1e-6,
    )
