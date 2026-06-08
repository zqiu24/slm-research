"""HeadAlignedPOETXLinear: a POETXLinear with head_dim blocks + identity perm on the
head side, and a real random perm + multiple blocks on the residual side."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _isolate_default_dtype():
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(torch.float32)


def test_qkv_head_side_out_structure():
    from poet_torch import HeadAlignedPOETXLinear, POETXLinear

    # q/k/v: head_side="out". out=heads*head_dim, in=hidden.
    heads, head_dim, hidden = 4, 8, 16
    layer = HeadAlignedPOETXLinear(
        in_features=hidden,
        out_features=heads * head_dim,
        head_side="out",
        head_dim=head_dim,
        head_resid_block_count=2,
        bias=False,
    )
    assert isinstance(layer, POETXLinear)  # merge driver / isinstance routing
    # head side = out -> block_size_out == head_dim, r_out == heads
    assert layer.block_size_out == head_dim
    assert layer.r_out == heads
    # residual side = in -> split into head_resid_block_count blocks
    assert layer.block_size_in == hidden // 2
    assert layer.r_in == 2
    # head-side perm is identity; residual-side perm is a real permutation
    assert torch.equal(layer.perm_out, torch.arange(heads * head_dim, dtype=torch.int32))
    assert not torch.equal(layer.perm_in, torch.arange(hidden, dtype=torch.int32))
    # perm inverses are consistent
    assert torch.equal(
        layer.perm_in[layer.perm_in_inv.long()], torch.arange(hidden, dtype=torch.int32)
    )


def test_o_head_side_in_structure():
    from poet_torch import HeadAlignedPOETXLinear

    # o: head_side="in". in=heads*head_dim, out=hidden.
    heads, head_dim, hidden = 4, 8, 16
    layer = HeadAlignedPOETXLinear(
        in_features=heads * head_dim,
        out_features=hidden,
        head_side="in",
        head_dim=head_dim,
        head_resid_block_count=2,
        bias=False,
    )
    assert layer.block_size_in == head_dim
    assert layer.r_in == heads
    assert layer.block_size_out == hidden // 2
    assert layer.r_out == 2
    # head side = in -> identity perm there; residual side = out -> real perm
    assert torch.equal(layer.perm_in, torch.arange(heads * head_dim, dtype=torch.int32))
    assert not torch.equal(layer.perm_out, torch.arange(hidden, dtype=torch.int32))


def test_invalid_head_side_raises():
    from poet_torch import HeadAlignedPOETXLinear

    with pytest.raises(ValueError, match="head_side"):
        HeadAlignedPOETXLinear(
            in_features=16,
            out_features=32,
            head_side="bogus",
            head_dim=8,
            head_resid_block_count=2,
        )


def _chain_ref(layer, x):
    """Reference output via the VERIFIED chain_layer_x_fast_decoupled (the same
    reference test_poetx_layer.py uses). That reference wants the W_perm-frame
    weight; the layer stores the FORWARD-frame weight post-bake, so un-permute it
    back. This exercises the asymmetric perms/blocks against the trusted chain."""
    from poet_torch.poet_layer import (
        cayley_batch,
        chain_layer_x_fast_decoupled,
        pytorch_skew_symmetric,
    )

    w_perm = layer.weight.index_select(0, layer.perm_out_inv.long()).index_select(
        1, layer.perm_in_inv.long()
    )
    qi = pytorch_skew_symmetric(layer.oft_R_in, layer.block_size_in, layer.rows_in, layer.cols_in)
    qo = pytorch_skew_symmetric(
        layer.oft_R_out, layer.block_size_out, layer.rows_out, layer.cols_out
    )
    return chain_layer_x_fast_decoupled(
        x,
        cayley_batch(qi),
        w_perm,
        None,
        cayley_batch(qo),
        layer.perm_in_inv,
        layer.perm_in,
        layer.perm_out,
        layer.perm_out_inv,
        layer.block_size_in,
        layer.block_size_out,
    )


def test_forward_matches_poet_chain_at_zero_head_side_out():
    # POETX forward is a bare GEMM that IGNORES oft_R (the rotation applies only at
    # the merge). So forward parity is at oft_R=0 (R=I); this verifies the asymmetric
    # perms (identity head + random resid) + bake give the right effective weight.
    from poet_torch import HeadAlignedPOETXLinear

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    layer = HeadAlignedPOETXLinear(
        in_features=16,
        out_features=32,
        head_side="out",
        head_dim=8,
        head_resid_block_count=2,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.normal_()
        layer.bake_perms_into_weight()  # walk does this after copying the real weight
    x = torch.randn(5, 16)
    assert torch.allclose(layer(x), _chain_ref(layer, x), atol=1e-9), (
        (layer(x) - _chain_ref(layer, x)).abs().max()
    )


def test_forward_matches_poet_chain_at_zero_head_side_in():
    from poet_torch import HeadAlignedPOETXLinear

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    layer = HeadAlignedPOETXLinear(
        in_features=32,
        out_features=16,
        head_side="in",
        head_dim=8,
        head_resid_block_count=2,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.normal_()
        layer.bake_perms_into_weight()
    x = torch.randn(5, 32)
    assert torch.allclose(layer(x), _chain_ref(layer, x), atol=1e-9), (
        (layer(x) - _chain_ref(layer, x)).abs().max()
    )
