"""Tests for POET fused-layer splitting (geometry + surgery)."""

import torch
import torch.nn as nn

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


class _FakeConfig:
    def __init__(self):
        self.gated_linear_unit = True
        self.activation_func = torch.nn.functional.silu
        self.activation_func_clamp_value = None
        self.glu_linear_offset = 0.0


class _FakeMLP(nn.Module):
    """Stand-in mimicking Megatron MLP's attributes used by the split path."""

    def __init__(self, hidden=8, ffn=16):
        super().__init__()
        self.config = _FakeConfig()
        self.linear_fc1 = nn.Linear(hidden, 2 * ffn, bias=False)  # [gate; up]
        self.linear_fc2 = nn.Linear(ffn, hidden, bias=False)


def test_split_fc1_creates_separate_modules_and_matches_fused():
    torch.manual_seed(0)
    m = _FakeMLP(hidden=8, ffn=16)
    x = torch.randn(3, 8)

    # Reference: fused forward (silu(gate) * up) -> fc2.
    fused = m.linear_fc1(x)
    gate_ref, up_ref = torch.chunk(fused, 2, dim=-1)
    ref = m.linear_fc2(torch.nn.functional.silu(gate_ref) * up_ref)

    n = ps.split_fused_linears(
        m,
        split_qkv=False,
        split_fc1=True,
        block_size=8,
        block_count=None,
        linear_types=(nn.Linear,),
    )
    assert n == 1
    assert hasattr(m, "linear_fc1_gate") and hasattr(m, "linear_fc1_up")
    assert not hasattr(m, "linear_fc1")
    assert isinstance(m.linear_fc1_gate, nn.Linear)

    out, _out_bias = m.forward(x)
    assert torch.allclose(out, ref, atol=1e-6)


def test_split_fc1_hard_errors_on_indivisible_segment():
    import pytest

    m = _FakeMLP(hidden=8, ffn=20)  # ffn 20 not divisible by 8
    with pytest.raises(ValueError, match="linear_fc1_gate"):
        ps.split_fused_linears(
            m,
            split_qkv=False,
            split_fc1=True,
            block_size=8,
            block_count=None,
            linear_types=(nn.Linear,),
        )
