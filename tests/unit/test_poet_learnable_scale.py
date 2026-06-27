"""CPU tests for POET learnable per-layer scale (ScaledPOETLinear)."""

from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn as nn
from poet_torch import POETLinear, SingleStepPOETLinear

from launchers.pretrain_gpt_slm import add_slm_args
from src.optim.poet import _build_lie_update_rms_param_groups
from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet
from src.optim.poet_scaled_layer import (
    ScaledPOETLinear,
    ScaledSingleStepPOETLinear,
)


def _plain_poet_forward_runs_here() -> bool:
    """The plain (non-single-step) cayley ``POETLinear`` forward routes through
    ``torch.compile``/triton; on a CPU-only box with no GPU driver it raises
    ``RuntimeError: 0 active drivers``. The single-step path is pure-eager and
    always runs. Probe once so the plain-path FORWARD cases skip cleanly off-GPU
    while the single-step path (the production ``single_step_native=true`` path)
    still fully exercises the shared gain mixin on CPU."""
    try:
        p = POETLinear(
            4, 4, block_count=1, bias=False, parameterization="cayley", dtype=torch.float32
        )
        with torch.no_grad():
            p.weight.normal_(std=0.02)
            p(torch.zeros(1, 4))
        return True
    except Exception:
        return False


_PLAIN_POET_CPU = _plain_poet_forward_runs_here()
_needs_plain_forward = pytest.mark.skipif(
    not _PLAIN_POET_CPU,
    reason="plain POETLinear cayley forward needs torch.compile/triton (GPU "
    "driver); single-step path covers the shared gain mixin on CPU",
)


def _make_pair(scaled_cls, base_cls):
    """A scaled layer and a base twin sharing the same frozen weight + perms."""
    scaled = scaled_cls(
        8, 16, block_count=1, bias=False, parameterization="cayley", dtype=torch.float32
    )
    base = base_cls(
        8, 16, block_count=1, bias=False, parameterization="cayley", dtype=torch.float32
    )
    # POETLinear leaves the frozen weight as torch.empty (possibly NaN); give it a
    # finite value so torch.equal is meaningful (NaN != NaN would fail spuriously),
    # then mirror it + the permutation buffers into the base twin so the two compute
    # identically. oft_R_{in,out} init to zeros (R=I) in both, so g=1 ⇒ equal output.
    with torch.no_grad():
        scaled.weight.normal_(std=0.02)
        base.weight.copy_(scaled.weight)
        for buf in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(base, buf).copy_(getattr(scaled, buf))
    return scaled, base


def test_gain_initialized_to_one_scalar():
    layer = ScaledPOETLinear(8, 16, block_count=1, bias=False, dtype=torch.float32)
    assert isinstance(layer.gain, nn.Parameter)
    assert layer.gain.requires_grad
    assert layer.gain.shape == torch.Size([])  # 0-dim scalar
    assert float(layer.gain) == 1.0


@pytest.mark.parametrize(
    "scaled_cls,base_cls",
    [
        pytest.param(ScaledPOETLinear, POETLinear, marks=_needs_plain_forward),
        (ScaledSingleStepPOETLinear, SingleStepPOETLinear),
    ],
)
def test_gain_one_is_bit_exact_base(scaled_cls, base_cls):
    scaled, base = _make_pair(scaled_cls, base_cls)
    x = torch.randn(4, 8)
    with torch.no_grad():
        out_scaled = scaled(x)
        out_base = base(x)
    assert torch.equal(out_scaled, out_base)  # exact, not allclose


@pytest.mark.parametrize(
    "scaled_cls,base_cls",
    [
        pytest.param(ScaledPOETLinear, POETLinear, marks=_needs_plain_forward),
        (ScaledSingleStepPOETLinear, SingleStepPOETLinear),
    ],
)
def test_gain_scales_output(scaled_cls, base_cls):
    scaled, base = _make_pair(scaled_cls, base_cls)
    with torch.no_grad():
        scaled.gain.fill_(2.5)
    x = torch.randn(4, 8)
    with torch.no_grad():
        out_scaled = scaled(x)
        out_base = base(x)
    assert torch.allclose(out_scaled, 2.5 * out_base, atol=1e-6)


@pytest.mark.parametrize(
    "scaled_cls",
    [
        pytest.param(ScaledPOETLinear, marks=_needs_plain_forward),
        ScaledSingleStepPOETLinear,
    ],
)
def test_grad_flows_to_gain(scaled_cls):
    layer = scaled_cls(8, 16, block_count=1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.normal_(std=0.02)  # finite frozen base (torch.empty may be NaN)
    x = torch.randn(4, 8)
    layer(x).sum().backward()
    assert layer.gain.grad is not None
    assert layer.gain.grad.shape == torch.Size([])


def test_bias_is_rejected():
    with pytest.raises(ValueError, match="bias=False"):
        ScaledPOETLinear(8, 16, block_count=1, bias=True, dtype=torch.float32)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)


def test_replace_with_learnable_scale_swaps_scaled_class():
    m = _ToyModel()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        learnable_scale=True,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, ScaledPOETLinear)
    assert hasattr(pl, "gain") and float(pl.gain) == 1.0


def test_replace_without_flag_has_no_gain():
    m = _ToyModel()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
    )
    assert not hasattr(m.fc1.poet_linear, "gain")


def test_learnable_scale_rejects_head_aligned():
    m = _ToyModel()
    with pytest.raises(NotImplementedError, match="learnable_scale"):
        replace_linears_with_poet(
            m,
            block_count=1,
            init_type="none",
            extra_linear_types=(nn.Linear,),
            learnable_scale=True,
            head_aligned_attn=True,
            head_dim=4,
        )


class _TinyScaledPoet(nn.Module):
    """Toy stand-in: a skew module that owns a frozen weight + a gain scalar."""

    def __init__(self, gain_value=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((4, 4), 0.5), requires_grad=False)
        self.oft_R_in = nn.Parameter(torch.zeros(1, 6))
        self.oft_R_out = nn.Parameter(torch.zeros(1, 6))
        self.gain = nn.Parameter(torch.tensor(float(gain_value)))
        self.block_size_in = 4
        self.block_size_out = 4


def test_skew_groups_carry_gain():
    model = _TinyScaledPoet()
    groups = _build_lie_update_rms_param_groups([model], lr=0.005, min_lr=1e-5)
    skew = [g for g in groups if g["use_skew"]]
    assert len(skew) == 2
    for g in skew:
        assert g["gain"] is model.gain


def test_gain_lands_in_wd_zero_group():
    model = _TinyScaledPoet()
    groups = _build_lie_update_rms_param_groups([model], lr=0.005, min_lr=1e-5)
    gain_groups = [
        g for g in groups if not g["use_skew"] and any(p is model.gain for p in g["params"])
    ]
    assert len(gain_groups) == 1
    assert gain_groups[0]["weight_decay"] == 0.0


def test_plain_poet_module_has_gain_none():
    # A skew module WITHOUT a gain (plain POETLinear analogue) → group gain is None.
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, 4), requires_grad=False)
            self.oft_R_in = nn.Parameter(torch.zeros(1, 6))
            self.oft_R_out = nn.Parameter(torch.zeros(1, 6))
            self.block_size_in = 4
            self.block_size_out = 4

    groups = _build_lie_update_rms_param_groups([_Tiny()], lr=0.005, min_lr=1e-5)
    for g in groups:
        if g["use_skew"]:
            assert g["gain"] is None


def test_denom_scales_with_gain():
    # Coupling: with gain=2, the angle denom doubles → theta halves (unclamped).
    from src.optim.poet_lie_orth_update_rms import compute_update_rms_angle

    w_rms = 0.064
    theta_g1 = compute_update_rms_angle(lr=0.005, update_rms=0.2, denom=w_rms * 1.0, max_angle=10.0)
    theta_g2 = compute_update_rms_angle(lr=0.005, update_rms=0.2, denom=w_rms * 2.0, max_angle=10.0)
    assert float(theta_g2) == pytest.approx(float(theta_g1) / 2.0, rel=1e-6)


def test_cli_arg_registered_store_true():
    # add_slm_args registers --slm-config-path as required=True, so it must be passed.
    parser = add_slm_args(argparse.ArgumentParser())
    ns = parser.parse_args(["--slm-config-path", "x", "--poet-learnable-scale"])
    assert ns.poet_learnable_scale is True
    ns2 = parser.parse_args(["--slm-config-path", "x"])
    assert ns2.poet_learnable_scale is False


def test_learnable_scale_flag_emitted_only_when_set():
    # Real injection path via _optimizer_args, mirroring test_megatron_args_grouped_poetx.
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    def _poet_cfg(learnable_scale):
        return OmegaConf.create(
            {
                "optim": {
                    "type": "poet",
                    "lr": 5e-3,
                    "weight_decay": 0.1,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "poet": {
                        "block_count": 1,
                        "cache_mode": "none",
                        "init_type": "mup_normalized",
                        "mup_alpha": 4.0,
                        "merge_period": 1,
                        "scale": 1.0,
                        "single_step_fast": True,
                        "single_step_native": True,
                        "lie_alternating": True,
                        "learnable_scale": learnable_scale,
                    },
                }
            }
        )

    assert "--poet-learnable-scale" in _optimizer_args(_poet_cfg(True))
    assert "--poet-learnable-scale" not in _optimizer_args(_poet_cfg(False))


def test_gain_round_trips_through_state_dict():
    layer = ScaledPOETLinear(8, 16, block_count=1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.gain.fill_(1.37)
    sd = layer.state_dict()
    assert "gain" in sd
    fresh = ScaledPOETLinear(8, 16, block_count=1, bias=False, dtype=torch.float32)
    fresh.load_state_dict(sd)
    assert float(fresh.gain) == pytest.approx(1.37)


def test_gain_group_lr_scaled_by_gain_lr_mult():
    model = _TinyScaledPoet()
    groups = _build_lie_update_rms_param_groups([model], lr=0.005, min_lr=1e-5, gain_lr_mult=2.0)
    gain_groups = [
        g for g in groups if not g["use_skew"] and any(p is model.gain for p in g["params"])
    ]
    assert len(gain_groups) == 1
    g = gain_groups[0]
    assert g["lr"] == pytest.approx(0.005 * 2.0)
    assert g["max_lr"] == pytest.approx(0.005 * 2.0)
    assert g["min_lr"] == pytest.approx(1e-5 * 2.0)


def test_gain_lr_mult_default_is_unscaled():
    model = _TinyScaledPoet()
    groups = _build_lie_update_rms_param_groups([model], lr=0.005, min_lr=1e-5)
    gain_groups = [
        g for g in groups if not g["use_skew"] and any(p is model.gain for p in g["params"])
    ]
    assert gain_groups[0]["max_lr"] == pytest.approx(0.005)
    assert gain_groups[0]["min_lr"] == pytest.approx(1e-5)


def test_cli_gain_lr_mult_parsed_as_float():
    parser = add_slm_args(argparse.ArgumentParser())
    ns = parser.parse_args(["--slm-config-path", "x", "--poet-gain-lr-mult", "2.5"])
    assert ns.poet_gain_lr_mult == pytest.approx(2.5)
    ns2 = parser.parse_args(["--slm-config-path", "x"])
    assert ns2.poet_gain_lr_mult == pytest.approx(1.0)


def test_gain_lr_mult_emitted_only_when_set():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    def _poet_cfg(mult):
        return OmegaConf.create(
            {
                "optim": {
                    "type": "poet",
                    "lr": 5e-3,
                    "weight_decay": 0.1,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "poet": {
                        "block_count": 1,
                        "cache_mode": "none",
                        "init_type": "mup_normalized",
                        "mup_alpha": 4.0,
                        "merge_period": 1,
                        "scale": 1.0,
                        "single_step_fast": True,
                        "single_step_native": True,
                        "lie_alternating": True,
                        "learnable_scale": True,
                        "gain_lr_mult": mult,
                    },
                }
            }
        )

    args2 = _optimizer_args(_poet_cfg(2.0))
    assert "--poet-gain-lr-mult" in args2
    assert "2.0" in args2
    assert "--poet-gain-lr-mult" not in _optimizer_args(_poet_cfg(1.0))
