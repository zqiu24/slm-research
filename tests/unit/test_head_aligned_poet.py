"""HeadAlignedPOETLinear: CPU constructor/merge/geometry; GPU forward parity."""

from __future__ import annotations

import pytest
import torch


def test_constructor_out_head_side_shapes():
    from poet_torch import HeadAlignedPOETLinear

    layer = HeadAlignedPOETLinear(
        in_features=512,
        out_features=512,
        head_side="out",
        head_dim=64,
        resid_block_count=1,
        parameterization="exp",
        dtype=torch.float64,
    )
    assert layer.block_size_out == 64 and layer.block_size_in == 512
    assert layer.head_count == 8
    assert layer.oft_R_out.shape == (8, 64 * 63 // 2)
    assert layer.oft_R_in.shape == (1, 512 * 511 // 2)
    assert layer.oft_R_in.requires_grad and layer.oft_R_out.requires_grad
    assert layer.weight.requires_grad is False
    # Head side (out) has identity Psi; residual (in) is a random permutation (resid_permute defaults True).
    assert torch.equal(layer.perm_out, torch.arange(512, dtype=torch.int32))


def test_constructor_in_head_side_for_output_proj():
    from poet_torch import HeadAlignedPOETLinear

    layer = HeadAlignedPOETLinear(
        in_features=512,
        out_features=512,
        head_side="in",
        head_dim=64,
        resid_block_count=4,
        parameterization="exp",
        dtype=torch.float64,
    )
    assert layer.block_size_in == 64 and layer.block_size_out == 128  # 512/4
    assert layer.head_count == 8
    assert torch.equal(layer.perm_in, torch.arange(512, dtype=torch.int32))  # head side identity


def test_constructor_validation():
    from poet_torch import HeadAlignedPOETLinear

    with pytest.raises(ValueError, match="head_side"):
        HeadAlignedPOETLinear(
            in_features=512, out_features=512, head_side="bogus", head_dim=64, resid_block_count=1
        )
    with pytest.raises(ValueError, match="exactly one of resid"):
        HeadAlignedPOETLinear(in_features=512, out_features=512, head_side="out", head_dim=64)
    with pytest.raises(ValueError, match="head_dim 48 doesn't divide"):
        HeadAlignedPOETLinear(
            in_features=512, out_features=512, head_side="out", head_dim=48, resid_block_count=1
        )


def test_merge_matches_stock_poetlinear_when_state_identical():
    """HeadAligned merge math == stock POETLinear(block_count=head_count) merge
    when both have identical state and reinit_perm=False (exp param, CPU)."""
    from poet_torch import HeadAlignedPOETLinear, POETLinear

    torch.manual_seed(0)
    a = HeadAlignedPOETLinear(
        in_features=512,
        out_features=512,
        head_side="out",
        head_dim=64,
        resid_block_count=8,
        parameterization="exp",
        dtype=torch.float64,
    )  # bs_out=64, bs_in=64 == stock block_count=8
    b = POETLinear(
        in_features=512,
        out_features=512,
        block_count=8,
        parameterization="exp",
        dtype=torch.float64,
    )
    with torch.no_grad():
        b.weight.copy_(torch.randn_like(b.weight))
        a.weight.copy_(b.weight)
        for name in ("oft_R_in", "oft_R_out"):
            new = torch.randn_like(getattr(a, name)) * 1e-2
            getattr(a, name).copy_(new)
            getattr(b, name).copy_(new)
        for buf in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(b, buf).copy_(getattr(a, buf))
    a.merge_then_reinitialize(reinit_perm=False)
    b.merge_then_reinitialize(reinit_perm=False)
    assert torch.allclose(a.weight, b.weight, atol=1e-10)
    assert torch.count_nonzero(a.oft_R_in) == 0 and torch.count_nonzero(a.oft_R_out) == 0


def test_merge_resamples_only_residual_side():
    """reinit_perm=True resamples the residual perm; the head perm stays identity."""
    from poet_torch import HeadAlignedPOETLinear

    torch.manual_seed(1)
    layer = HeadAlignedPOETLinear(
        in_features=512,
        out_features=512,
        head_side="out",
        head_dim=64,
        resid_block_count=8,
        parameterization="exp",
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        layer.oft_R_in.normal_(std=1e-2)
        layer.oft_R_out.normal_(std=1e-2)
    perm_in_before = layer.perm_in.clone()
    layer.merge_then_reinitialize(reinit_perm=True)
    # Head side (out) Psi stays identity; residual side (in) Psi changes.
    assert torch.equal(layer.perm_out, torch.arange(512, dtype=torch.int32))
    assert not torch.equal(layer.perm_in, perm_in_before)


def test_merge_resid_permute_false_never_resamples():
    from poet_torch import HeadAlignedPOETLinear

    torch.manual_seed(2)
    layer = HeadAlignedPOETLinear(
        in_features=512,
        out_features=512,
        head_side="out",
        head_dim=64,
        resid_block_count=8,
        resid_permute=False,
        parameterization="exp",
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        layer.oft_R_in.normal_(std=1e-2)
    pin, pout = layer.perm_in.clone(), layer.perm_out.clone()
    layer.merge_then_reinitialize(reinit_perm=True)
    assert torch.equal(layer.perm_in, pin) and torch.equal(layer.perm_out, pout)
