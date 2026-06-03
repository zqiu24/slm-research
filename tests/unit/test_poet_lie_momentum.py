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


def test_build_lie_param_groups_scales_skew_lr():
    import torch.nn as nn

    from src.optim.poet_lie_momentum import _build_lie_param_groups

    skew = [nn.Parameter(torch.zeros(1, 6))]
    adamw = [nn.Parameter(torch.zeros(4))]
    groups = _build_lie_param_groups(skew, adamw, lr=1e-3, min_lr=1e-5, scale=0.5)

    g_skew = next(g for g in groups if g["use_skew"])
    g_adam = next(g for g in groups if not g["use_skew"])
    assert g_skew["lr"] == 5e-4 and g_skew["max_lr"] == 5e-4 and g_skew["min_lr"] == 5e-6
    assert g_adam["lr"] == 1e-3 and g_adam["max_lr"] == 1e-3 and g_adam["min_lr"] == 1e-5


def test_build_lie_param_groups_drops_empty_sides():
    from src.optim.poet_lie_momentum import _build_lie_param_groups

    assert _build_lie_param_groups([], [], 1e-3, 1e-5, 0.5) == []
