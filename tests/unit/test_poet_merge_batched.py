"""CPU tests for the batched / replicated POET merge.

The real Cayley is a Triton GPU op, so these tests build R with the pure-torch
reference cayley_batch (the Neumann series the Triton kernel implements) and inject
it via cayley_fn. Block-diagonal fold ops are pure torch and run on CPU.
"""

import torch
from poet_torch import HeadAlignedPOETLinear, POETLinear
from poet_torch.poet_layer import cayley_batch, pytorch_skew_symmetric


def _identity_R(n_blocks, b):
    return torch.eye(b).unsqueeze(0).repeat(n_blocks, 1, 1)


def test_fold_with_R_identity_is_noop_poetlinear():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    pl = POETLinear(in_features=12, out_features=8, block_count=2, bias=False)
    with torch.no_grad():
        pl.weight.normal_()
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
    W0 = pl.weight.detach().clone()
    R_in = _identity_R(pl.r_in, pl.block_size_in)
    R_out = _identity_R(pl.r_out, pl.block_size_out)
    pl._fold_with_R(R_out, R_in, reinit_perm=False)
    assert torch.allclose(pl.weight, W0, atol=1e-12), (pl.weight - W0).abs().max()
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0


def test_fold_with_R_identity_is_noop_headaligned():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    pl = HeadAlignedPOETLinear(
        in_features=12,
        out_features=8,
        head_side="out",
        head_dim=4,
        resid_block_count=1,
        bias=False,
    )
    with torch.no_grad():
        pl.weight.normal_()
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
    W0 = pl.weight.detach().clone()
    R_in = _identity_R(pl.r_in, pl.block_size_in)
    R_out = _identity_R(pl.r_out, pl.block_size_out)
    pl._fold_with_R(R_out, R_in, reinit_perm=False)
    assert torch.allclose(pl.weight, W0, atol=1e-12)
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0


def _mixed_layers():
    """A mix that exercises grouping: 3 head-aligned (small head blocks, same
    head_dim=4) + 2 standard (different block sizes)."""
    layers = []
    for _ in range(3):
        pl = HeadAlignedPOETLinear(
            in_features=12,
            out_features=8,
            head_side="out",
            head_dim=4,
            resid_block_count=1,
            bias=False,
        )
        layers.append(pl)
    layers.append(POETLinear(in_features=12, out_features=8, block_count=2, bias=False))
    layers.append(POETLinear(in_features=12, out_features=8, block_count=1, bias=False))
    for pl in layers:
        with torch.no_grad():
            pl.weight.normal_()
            pl.oft_R_in.normal_(std=0.1)
            pl.oft_R_out.normal_(std=0.1)
    return layers


def _per_layer_R(pl, cayley_fn):
    qi = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    qo = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    return cayley_fn(qo), cayley_fn(qi)  # (R_out, R_in)


def test_batched_build_R_matches_per_layer():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    from src.patches.poet_merge_step import _build_R_batched

    layers = _mixed_layers()
    built = _build_R_batched(layers, cayley_fn=cayley_batch, max_batch_block=256)
    for pl in layers:
        R_out_ref, R_in_ref = _per_layer_R(pl, cayley_batch)
        R_out_b, R_in_b = built[id(pl)]
        assert torch.allclose(R_out_b, R_out_ref, atol=1e-12), (R_out_b - R_out_ref).abs().max()
        assert torch.allclose(R_in_b, R_in_ref, atol=1e-12), (R_in_b - R_in_ref).abs().max()
