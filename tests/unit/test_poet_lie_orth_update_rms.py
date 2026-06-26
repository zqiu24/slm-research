"""Tests for the POET Lie-Orth update-RMS optimizer."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.optim.poet import _build_lie_update_rms_param_groups
from src.optim.poet_lie_orth_update_rms import (
    LieOrthUpdateRMSMomentum,
    compute_update_rms_angle,
)


@pytest.fixture(autouse=True)
def _isolate_state():
    from poet_torch import alt_state

    torch.set_default_dtype(torch.float32)
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)
    yield
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)
    torch.set_default_dtype(torch.float32)


def test_compute_update_rms_angle_matches_formula():
    theta = compute_update_rms_angle(lr=0.005, update_rms=0.2, denom=0.064, max_angle=0.024)
    assert theta == pytest.approx(0.015625)


def test_compute_update_rms_angle_clamps():
    theta = compute_update_rms_angle(lr=0.006, update_rms=0.3, denom=0.04, max_angle=0.024)
    assert theta == pytest.approx(0.024)


class _TinyPoetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4), requires_grad=False)
        self.oft_R_in = nn.Parameter(torch.zeros(1, 6))
        self.oft_R_out = nn.Parameter(torch.zeros(1, 6))
        self.dense = nn.Parameter(torch.ones(2))
        self.block_size_in = 4
        self.block_size_out = 4


def test_build_lie_update_rms_param_groups_are_per_layer_and_unscaled():
    model = _TinyPoetModule()
    groups = _build_lie_update_rms_param_groups([model], lr=0.005, min_lr=1.0e-5)
    skew_groups = [group for group in groups if group["use_skew"]]
    assert len(skew_groups) == 2
    assert {group["side"] for group in skew_groups} == {"in", "out"}
    for group in skew_groups:
        assert len(group["params"]) == 1
        assert group["weight"] is model.weight
        assert group["block_size"] == 4
        assert group["lr"] == pytest.approx(0.005)
        assert group["max_lr"] == pytest.approx(0.005)
    adamw_groups = [group for group in groups if not group["use_skew"]]
    assert len(adamw_groups) == 1
    assert adamw_groups[0]["params"] == [model.dense]


def _make_groups(p_in, p_out, weight, lr=0.01):
    return [
        dict(
            params=[p_in],
            use_skew=True,
            side="in",
            weight=weight,
            block_size=2,
            lr=lr,
        ),
        dict(
            params=[p_out],
            use_skew=True,
            side="out",
            weight=weight,
            block_size=2,
            lr=lr,
        ),
    ]


def test_active_side_write_keeps_inactive_momentum_fresh():
    from poet_torch import alt_state

    weight = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    p_in = nn.Parameter(torch.zeros(1, 1))
    p_out = nn.Parameter(torch.zeros(1, 1))
    p_in.grad = torch.tensor([[1.0]])
    p_out.grad = torch.tensor([[2.0]])
    opt = LieOrthUpdateRMSMomentum(
        _make_groups(p_in, p_out, weight),
        b1=0.0,
        update_rms=0.2,
        max_angle=1.0,
        ortho_method="spectral",
        ortho_ns_steps=20,
    )

    alt_state.set_iteration(0)  # active "out"
    opt.step()
    assert torch.allclose(p_in.data, torch.zeros_like(p_in))
    assert p_out.data.abs().sum() > 0
    assert torch.allclose(opt.state[p_in]["lie_m"], torch.tensor([[1.0]]))

    p_in.grad = torch.tensor([[3.0]])
    p_out.grad = torch.tensor([[4.0]])
    out_before = p_out.detach().clone()
    alt_state.set_iteration(1)  # active "in"
    opt.step()
    assert p_in.data.abs().sum() > 0
    assert torch.allclose(p_out.data, out_before)
    assert torch.allclose(opt.state[p_out]["lie_m"], torch.tensor([[4.0]]))


def test_angle_uses_group_lr_directly():
    from poet_torch import alt_state

    weight = nn.Parameter(torch.full((2, 2), 0.064), requires_grad=False)
    p = nn.Parameter(torch.zeros(1, 1))
    p.grad = torch.tensor([[1.0]])
    opt = LieOrthUpdateRMSMomentum(
        [
            dict(
                params=[p],
                use_skew=True,
                side="out",
                weight=weight,
                block_size=2,
                lr=0.005,
            )
        ],
        b1=0.0,
        update_rms=0.2,
        max_angle=0.024,
        ortho_method="spectral",
        ortho_ns_steps=20,
    )
    alt_state.set_iteration(0)
    opt.step()
    assert opt.last_update_rms_angles[id(p)] == pytest.approx(0.015625)


def test_clamp_path_records_max_angle_and_applies_it():
    from poet_torch import alt_state

    weight = nn.Parameter(torch.full((2, 2), 0.04), requires_grad=False)
    p = nn.Parameter(torch.zeros(1, 1))
    p.grad = torch.tensor([[1.0]])
    opt = LieOrthUpdateRMSMomentum(
        [
            dict(
                params=[p],
                use_skew=True,
                side="out",
                weight=weight,
                block_size=2,
                lr=0.006,
            )
        ],
        b1=0.0,
        update_rms=0.3,
        max_angle=0.024,
        ortho_method="spectral",
        ortho_ns_steps=20,
    )
    alt_state.set_iteration(0)
    opt.step()
    assert opt.last_update_rms_angles[id(p)] == pytest.approx(0.024)
    assert p.data.abs().item() == pytest.approx(0.024, rel=0.05)
    assert opt.last_update_rms_stats["poet_update_rms/clamp_fraction"] == pytest.approx(1.0)


def test_side_gamma_default_is_symmetric_noop():
    """gamma=0 must leave the angle bit-for-bit at lr*rho/RMS (champion path)."""
    from poet_torch import alt_state

    # Rectangular weight (fan_out=8 > fan_in=2) so a nonzero gamma WOULD bite.
    weight = nn.Parameter(torch.full((8, 2), 0.064), requires_grad=False)
    p = nn.Parameter(torch.zeros(1, 1))
    p.grad = torch.tensor([[1.0]])
    opt = LieOrthUpdateRMSMomentum(
        [dict(params=[p], use_skew=True, side="out", weight=weight, block_size=2, lr=0.005)],
        b1=0.0,
        update_rms=0.2,
        max_angle=0.5,
        ortho_method="spectral",
        ortho_ns_steps=20,
    )
    alt_state.set_iteration(0)
    opt.step()
    assert opt.last_update_rms_angles[id(p)] == pytest.approx(0.015625)


def test_side_gamma_redistributes_out_vs_in_and_preserves_geomean():
    """gamma>0: out side (fan_out>fan_in) rotates more, in side less, product==symmetric^2."""
    from poet_torch import alt_state

    # fc1-like: fan_out=8, fan_in=2 -> sqrt(d_out*d_in)=4. gamma=0.5:
    #   out factor = (8/4)^0.5 = sqrt(2); in factor = (2/4)^0.5 = 1/sqrt(2); product == 1.
    weight = nn.Parameter(torch.full((8, 2), 0.064), requires_grad=False)
    base = 0.005 * 0.2 / 0.064  # symmetric angle = 0.015625

    def angle_for(side, gamma):
        p = nn.Parameter(torch.zeros(1, 1))
        p.grad = torch.tensor([[1.0]])
        opt = LieOrthUpdateRMSMomentum(
            [dict(params=[p], use_skew=True, side=side, weight=weight, block_size=2, lr=0.005)],
            b1=0.0,
            update_rms=0.2,
            max_angle=0.5,
            side_gamma=gamma,
            ortho_method="spectral",
            ortho_ns_steps=20,
        )
        alt_state.set_iteration(0 if side == "out" else 1)
        opt.step()
        return float(opt.last_update_rms_angles[id(p)])

    a_out = angle_for("out", 0.5)
    a_in = angle_for("in", 0.5)
    assert a_out == pytest.approx(base * math.sqrt(2.0))
    assert a_in == pytest.approx(base / math.sqrt(2.0))
    # geometric mean of the two side angles equals the symmetric angle (pure redistribution).
    assert math.sqrt(a_out * a_in) == pytest.approx(base)
    assert a_out > base > a_in


def test_direction_rms_mode_rejected_before_training():
    p = nn.Parameter(torch.zeros(1, 1))
    weight = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    with pytest.raises(ValueError, match="rms_mode='weight' only"):
        LieOrthUpdateRMSMomentum(
            [
                dict(
                    params=[p],
                    use_skew=True,
                    side="out",
                    weight=weight,
                    block_size=2,
                    lr=0.01,
                )
            ],
            rms_mode="direction",
        )


def test_alternating_required():
    p = nn.Parameter(torch.zeros(1, 1))
    weight = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    with pytest.raises(ValueError, match="requires alternating=True"):
        LieOrthUpdateRMSMomentum(
            [
                dict(
                    params=[p],
                    use_skew=True,
                    side="out",
                    weight=weight,
                    block_size=2,
                    lr=0.01,
                )
            ],
            alternating=False,
        )
