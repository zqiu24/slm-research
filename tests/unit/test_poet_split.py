"""Tests for POET fused-layer splitting (geometry + surgery)."""

import torch

from src.optim import poet_split as ps


def test_segment_out_dims_mqa_and_gqa():
    # MQA: 16 heads, 1 group, head_dim 384.
    assert ps.qkv_segment_out_dims(16, 1, 384) == (16 * 384, 1 * 384)
    # GQA: 8 heads, 2 groups, head_dim 64.
    assert ps.qkv_segment_out_dims(8, 2, 64) == (8 * 64, 2 * 64)


def test_deinterleave_row_indices_gqa_layout():
    # 4 heads, 2 groups (2 q-heads/group), head_dim 1 → trivial indices.
    # Fused per-group layout: [q,q,k,v] → group0 rows 0,1,2,3 ; group1 rows 4,5,6,7
    q, k, v = ps.qkv_deinterleave_row_indices(4, 2, 1)
    assert q.tolist() == [0, 1, 4, 5]
    assert k.tolist() == [2, 6]
    assert v.tolist() == [3, 7]


def test_interleave_index_roundtrips_weight():
    # Random fused weight; de-interleave then re-interleave must reproduce it.
    nah, ng, hd, in_f = 8, 2, 16, 32
    q_out, kv_out = ps.qkv_segment_out_dims(nah, ng, hd)
    total = q_out + 2 * kv_out
    w = torch.randn(total, in_f)
    qr, kr, vr = ps.qkv_deinterleave_row_indices(nah, ng, hd)
    cat = torch.cat([w[qr], w[kr], w[vr]], dim=0)  # de-interleaved
    idx = ps.qkv_interleave_index(nah, ng, hd)
    assert torch.equal(cat.index_select(0, idx), w)


def test_validate_divisible_raises_with_segment_name():
    import pytest

    with pytest.raises(ValueError, match="linear_k"):
        ps.validate_divisible(
            "decoder.layers.0.self_attention",
            "linear_k",
            in_f=1280,
            out_f=384,
            block_size=256,
            block_count=None,
        )


def test_validate_divisible_ok():
    # 6144 and 1280 both divisible by 256 → no raise.
    ps.validate_divisible(
        "attn", "linear_q", in_f=1280, out_f=6144, block_size=256, block_count=None
    )
