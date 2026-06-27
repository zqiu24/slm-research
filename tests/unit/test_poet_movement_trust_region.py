"""Realized-movement trust region (M1) on LieOrthUpdateRMSMomentum.

Mirrors the decorrelate-test harness: one alternating step with active 'in'
(active_iter=1), then re-derive the applied in-side weight-space move
D_in = W @ blockdiag(A_in) and its ratio r = ||D_in||_F / ||W||_F.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.diag.poet_coordination_diag import side_directions
from src.diag.skew_conditioning import vec_to_skew
from src.optim.poet_lie_orth_update_rms import LieOrthUpdateRMSMomentum


@pytest.fixture(autouse=True)
def _isolate_state():
    from poet_torch import alt_state

    torch.set_default_dtype(torch.float32)
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)
    yield
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)


def _move_step(mode="off", rho=0.0, lam=1.0, seed=3, active_iter=1, **extra):
    """One alternating update-RMS step (active_iter=1 -> 'in' written). Returns
    (ratio, applied_in_vec, optimizer)."""
    from poet_torch import alt_state

    torch.manual_seed(seed)
    b = 12
    ne = b * (b - 1) // 2
    oin = nn.Parameter(torch.zeros(1, ne))
    oin.grad = torch.randn(1, ne)
    oout = nn.Parameter(torch.zeros(1, ne))
    oout.grad = torch.randn(1, ne)
    W = nn.Parameter(torch.randn(b, b), requires_grad=False)
    opt = LieOrthUpdateRMSMomentum(
        [
            dict(params=[oin], use_skew=True, side="in", weight=W, block_size=b, lr=0.05),
            dict(params=[oout], use_skew=True, side="out", weight=W, block_size=b, lr=0.05),
        ],
        update_rms=0.3,
        max_angle=1.0,  # no clamp, so the trust-region identity is exact
        ortho_method="muon",
        ortho_ns_steps=5,
        move_control_mode=mode,
        move_budget_rho=rho,
        move_lambda=lam,
        **extra,
    )
    alt_state.set_iteration(active_iter)
    opt.step()
    alt_state.set_iteration(0)
    A_in = vec_to_skew(oin.data, b)
    _, d_in = side_directions(torch.zeros(1, b, b), A_in, W.float())
    ratio = (d_in.norm() / W.float().norm()).item()
    return ratio, oin.data.clone(), opt


def test_off_is_bit_identical():
    _, v1, _ = _move_step(mode="off")
    _, v2, _ = _move_step(mode="off")
    assert torch.equal(v1, v2)


def test_clip_noop_when_under_budget():
    r0, v0, _ = _move_step(mode="off")
    r1, v1, _ = _move_step(mode="clip", rho=10.0 * r0)
    assert torch.equal(v0, v1)
    assert r1 == pytest.approx(r0, rel=1e-5)


def test_clip_to_budget_when_over():
    r0, _, _ = _move_step(mode="off")
    rho = 0.5 * r0
    r1, _, _ = _move_step(mode="clip", rho=rho, lam=1.0)
    assert r1 == pytest.approx(rho, rel=2e-3)


def test_clip_lambda_partial():
    r0, _, _ = _move_step(mode="off")
    rho = 0.5 * r0  # rho/r0 = 0.5, f = 1 - 0.5*(1-0.5) = 0.75
    r1, _, _ = _move_step(mode="clip", rho=rho, lam=0.5)
    assert r1 == pytest.approx(0.75 * r0, rel=2e-3)


@pytest.mark.parametrize("mult", [0.5, 2.0])
def test_normalize_hits_budget(mult):
    r0, _, _ = _move_step(mode="off")
    rho = mult * r0
    r1, _, _ = _move_step(mode="normalize", rho=rho)
    assert r1 == pytest.approx(rho, rel=2e-3)


def test_measure_does_not_change_write():
    _, v0, _ = _move_step(mode="off")
    _, v1, opt = _move_step(mode="measure")
    assert torch.equal(v0, v1)
    assert "poet_move/ratio_mean" in opt.last_update_rms_stats


def test_rejects_bad_mode():
    p = nn.Parameter(torch.zeros(1, 1))
    W = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    with pytest.raises(ValueError, match="move_control_mode"):
        LieOrthUpdateRMSMomentum(
            [dict(params=[p], use_skew=True, side="in", weight=W, block_size=2, lr=0.01)],
            move_control_mode="bogus",
        )


def test_requires_rho_when_active():
    p = nn.Parameter(torch.zeros(1, 1))
    W = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    with pytest.raises(ValueError, match="move_budget_rho"):
        LieOrthUpdateRMSMomentum(
            [dict(params=[p], use_skew=True, side="in", weight=W, block_size=2, lr=0.01)],
            move_control_mode="clip",
            move_budget_rho=0.0,
        )
