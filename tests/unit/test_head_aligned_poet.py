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


def test_merge_preserves_singular_values():
    """Folding orthogonal (exp) rotations + permutation preserves W's spectrum."""
    from poet_torch import HeadAlignedPOETLinear

    torch.manual_seed(3)
    layer = HeadAlignedPOETLinear(
        in_features=512,
        out_features=512,
        head_side="out",
        head_dim=64,
        resid_block_count=1,
        parameterization="exp",
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        sv_before = torch.linalg.svdvals(layer.weight.double())
        layer.oft_R_in.normal_(std=1e-2)
        layer.oft_R_out.normal_(std=1e-2)
    layer.merge_then_reinitialize(reinit_perm=False)
    sv_after = torch.linalg.svdvals(layer.weight.double())
    assert torch.allclose(
        sv_before, sv_after, atol=1e-6
    )  # exact-orthogonal exp; loose for the 512-block


def test_no_cross_head_mixing():
    """Perturbing head j's out-side block changes only head j's rows of the
    folded weight (residual side held at identity)."""
    from poet_torch import HeadAlignedPOETLinear

    torch.manual_seed(4)
    w0 = torch.randn(512, 512, dtype=torch.float64)

    def merged_weight(perturb_block=None):
        layer = HeadAlignedPOETLinear(
            in_features=512,
            out_features=512,
            head_side="out",
            head_dim=64,
            resid_block_count=1,
            resid_permute=False,
            parameterization="exp",
            dtype=torch.float64,
        )
        with torch.no_grad():
            layer.weight.copy_(w0)
            if perturb_block is not None:
                layer.oft_R_out[perturb_block].normal_(std=1e-1)
            # oft_R_in stays 0 (residual identity).
        layer.merge_then_reinitialize(reinit_perm=False)
        return layer.weight.detach().clone()

    base = merged_weight(None)
    pert = merged_weight(perturb_block=2)
    diff = (pert - base).abs()
    rows = slice(2 * 64, 3 * 64)  # head 2's rows (out side, identity perm)
    assert diff[rows].max() > 1e-6  # head 2 changed
    mask = torch.ones(512, dtype=torch.bool)
    mask[rows] = False
    assert diff[mask].max() < 1e-12  # all other heads untouched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernel")
def test_forward_matches_reference_and_both_sides_get_grad():
    """The kernel forward (cayley) matches the pure-PyTorch decoupled reference,
    and a backward populates BOTH oft_R grads."""
    from poet_torch import HeadAlignedPOETLinear

    from tests.unit.test_poet_decoupled import poet_reference_forward

    torch.manual_seed(0)
    layer = HeadAlignedPOETLinear(
        in_features=512,
        out_features=512,
        head_side="out",
        head_dim=64,
        resid_block_count=1,
        device="cuda",
        dtype=torch.float32,
    )
    with torch.no_grad():
        # Row-normalize the base weight (the deployed init_type="normalized"
        # regime) so the 512-wide dense contraction stays O(1); a randn base
        # makes outputs ~O(50) where the fused-kernel-vs-pytorch Cayley
        # truncation difference shows up as ~1e-3 absolute float32 noise that
        # scales linearly with output magnitude (not a logic error).
        w = torch.randn_like(layer.weight)
        layer.weight.copy_(w / w.norm(dim=1, keepdim=True))
        layer.oft_R_in.normal_(std=1e-2)
        layer.oft_R_out.normal_(std=1e-2)

    x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
    y = layer(x)
    y_ref = poet_reference_forward(
        x.cpu(),
        layer.weight.detach().cpu(),
        layer.oft_R_in.detach().cpu(),
        layer.oft_R_out.detach().cpu(),
        layer.perm_in.cpu(),
        layer.perm_in_inv.cpu(),
        layer.perm_out.cpu(),
        layer.perm_out_inv.cpu(),
        layer.block_size_in,
        layer.block_size_out,
    )
    # atol=1e-3 / rtol=1e-3 matches the grad-parity precedent in
    # test_poet_decoupled.py (float32 fused-kernel accumulation).
    assert torch.allclose(y.detach().cpu(), y_ref, atol=1e-3, rtol=1e-3)

    y.sum().backward()
    assert layer.oft_R_in.grad is not None and layer.oft_R_in.grad.abs().sum() > 0
    assert layer.oft_R_out.grad is not None and layer.oft_R_out.grad.abs().sum() > 0
