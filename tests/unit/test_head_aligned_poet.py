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
