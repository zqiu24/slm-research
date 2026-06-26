"""Cross-side decorrelation on the update-RMS POET optimizer (alternating path).

Mirrors the alternating-decorrelate tests in test_poet_lie_orth.py, but for
LieOrthUpdateRMSMomentum. Key difference from the lie_ortho version: the buffer here
holds the ANGLE-SCALED generator (theta baked in; scatter uses alpha=1.0), so the
applied generator is `oin.data` directly (no division by lr).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.diag.skew_conditioning import vec_to_skew
from src.optim.poet_lie_orth_update_rms import LieOrthUpdateRMSMomentum
from src.optim.poet_skew_muon import orthogonalize_skew_direction


@pytest.fixture(autouse=True)
def _isolate_state():
    from poet_torch import alt_state

    torch.set_default_dtype(torch.float32)
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)
    yield
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)


def _alt_decorr_dirs(decorrelate, mode="in_off_out", active_iter=1, seed=3, **extra):
    """One alternating update-RMS step (active_iter=1 -> 'in' is written). Returns the
    applied active-in weight-space direction D_in and the inactive-out momentum
    direction D_out_mom, both in the W frame."""
    from poet_torch import alt_state

    from src.diag.poet_coordination_diag import side_directions

    torch.manual_seed(seed)
    b = 12
    ne = b * (b - 1) // 2
    oin = nn.Parameter(torch.zeros(1, ne))
    oin.grad = torch.randn(1, ne)
    oout = nn.Parameter(torch.zeros(1, ne))
    oout.grad = torch.randn(1, ne)
    W = nn.Parameter(torch.randn(b, b), requires_grad=False)
    lr = 0.05
    kw = dict(
        decorrelate_sides=decorrelate,
        decorrelate_mode=mode,
        layer_pairs=[(oout, oin, W, b, b)] if decorrelate else None,
    )
    kw.update(extra)
    opt = LieOrthUpdateRMSMomentum(
        [
            dict(params=[oin], use_skew=True, side="in", weight=W, block_size=b, lr=lr),
            dict(params=[oout], use_skew=True, side="out", weight=W, block_size=b, lr=lr),
        ],
        update_rms=0.3,
        max_angle=1.0,  # avoid the clamp so the projection identity is exact
        ortho_method="muon",
        ortho_ns_steps=5,
        **kw,
    )
    alt_state.set_iteration(active_iter)  # 1 -> active 'in'
    opt.step()
    alt_state.set_iteration(0)
    A_in = vec_to_skew(oin.data, b)  # applied generator (theta baked in; oin started at 0)
    m_out = opt.state[oout]["lie_m"]
    A_out_mom = orthogonalize_skew_direction(vec_to_skew(-m_out, b), method="muon", ns_steps=5)
    d_out_mom, d_in = side_directions(A_out_mom, A_in, W.float())
    return d_in, d_out_mom


def _cos(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def test_off_is_deterministic_and_on_changes_write():
    # decorrelate_sides defaults False -> step() skips the projection entirely. Two off
    # runs are bit-identical; enabling the feature must change the active write.
    off1, _ = _alt_decorr_dirs(decorrelate=False)
    off2, _ = _alt_decorr_dirs(decorrelate=False)
    assert torch.equal(off1, off2)
    on, _ = _alt_decorr_dirs(decorrelate=True, mode="in_off_out")
    assert _cos(off1, on) < 0.999


def test_alternating_decorrelate_removes_inactive_momentum_overlap():
    d_in_base, d_out_mom = _alt_decorr_dirs(decorrelate=False)
    base = abs(_cos(d_in_base, d_out_mom))
    assert base > 0.02, f"baseline inactive-momentum overlap should be non-trivial, got {base}"
    d_in, d_out_mom2 = _alt_decorr_dirs(decorrelate=True, mode="in_off_out")
    assert abs(_cos(d_in, d_out_mom2)) < 1e-3


@pytest.mark.parametrize("lam", [0.25, 0.5, 1.0])
def test_alternating_decorrelate_lambda_scales_overlap(lam):
    # Partial projection leaves a (1-lambda) fraction of the parallel component:
    # <D_in', D_out_mom> = (1-lambda) <D_in, D_out_mom>  (exact, renorm off).
    d_in0, d_mom = _alt_decorr_dirs(decorrelate=False)
    ip0 = (d_in0.flatten() @ d_mom.flatten()).item()
    assert abs(ip0) > 1e-3, f"baseline parallel component should be non-trivial, got {ip0}"
    d_in, d_mom2 = _alt_decorr_dirs(decorrelate=True, mode="in_off_out", decorrelate_lambda=lam)
    ip = (d_in.flatten() @ d_mom2.flatten()).item()
    assert ip == pytest.approx((1.0 - lam) * ip0, rel=2e-3, abs=1e-4)


@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_renorm_preserves_realized_norm(lam):
    # With renorm, the active side's realized ||D|| (theta-inclusive) is restored to its
    # pre-projection value -- only the direction changes. This is the §3.4 subtlety.
    d_in0, _ = _alt_decorr_dirs(decorrelate=False)
    n0 = d_in0.norm().item()
    d_in, _ = _alt_decorr_dirs(
        decorrelate=True, mode="in_off_out", decorrelate_lambda=lam, decorrelate_renorm=True
    )
    assert d_in.norm().item() == pytest.approx(n0, rel=1e-4)


def test_without_renorm_shrinks_movement():
    d_in0, _ = _alt_decorr_dirs(decorrelate=False)
    d_in, _ = _alt_decorr_dirs(decorrelate=True, mode="in_off_out", decorrelate_lambda=1.0)
    assert d_in.norm().item() < 0.999 * d_in0.norm().item()


def test_threshold_gates_module():
    d_in0, d_mom = _alt_decorr_dirs(decorrelate=False)
    cos = abs(_cos(d_in0, d_mom))
    assert cos > 0.02, f"need a non-trivial overlap to gate on, got {cos}"
    skipped, _ = _alt_decorr_dirs(
        decorrelate=True, mode="in_off_out", decorrelate_cos_threshold=cos + 0.1
    )
    assert _cos(d_in0, skipped) > 0.9999, "below-threshold layer must be left untouched"
    fired, _ = _alt_decorr_dirs(
        decorrelate=True, mode="in_off_out", decorrelate_cos_threshold=cos * 0.5
    )
    assert _cos(d_in0, fired) < 0.999, "above-threshold layer must be decorrelated"


def test_rejects_bad_mode():
    p = nn.Parameter(torch.zeros(1, 1))
    W = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    with pytest.raises(ValueError, match="decorrelate_mode"):
        LieOrthUpdateRMSMomentum(
            [dict(params=[p], use_skew=True, side="in", weight=W, block_size=2, lr=0.01)],
            decorrelate_sides=True,
            decorrelate_mode="bogus",
        )
