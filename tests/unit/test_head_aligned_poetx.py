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
