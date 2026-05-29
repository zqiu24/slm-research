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


class _FakeAttnConfig:
    attention_output_gate = False


class _FakeAttention(nn.Module):
    """Stand-in mimicking Megatron SelfAttention attributes used by the split."""

    def __init__(self, hidden=32, num_heads=8, num_groups=2, head_dim=16):
        super().__init__()
        self.config = _FakeAttnConfig()
        self.world_size = 1
        self.hidden_size_per_attention_head = head_dim
        self.num_query_groups_per_partition = num_groups
        self.num_attention_heads_per_partition = num_heads
        self.q_layernorm = None
        self.k_layernorm = None
        q_out = num_heads * head_dim
        kv_out = num_groups * head_dim
        self.linear_qkv = nn.Linear(hidden, q_out + 2 * kv_out, bias=False)

    # Faithful reference of Megatron's TP=1 / no-gate get_query_key_value_tensors.
    def reference_qkv(self, hidden_states):
        mixed, _ = self.linear_qkv(hidden_states), None
        hd = self.hidden_size_per_attention_head
        ng = self.num_query_groups_per_partition
        nqhpg = self.num_attention_heads_per_partition // ng
        mixed = mixed.view(*mixed.size()[:-1], ng, (nqhpg + 2) * hd)
        query, key, value = torch.split(mixed, [nqhpg * hd, hd, hd], dim=-1)
        query = query.reshape(query.size(0), -1, hd)
        return query, key, value


def test_split_qkv_creates_modules_and_matches_reference():
    torch.manual_seed(0)
    a = _FakeAttention(hidden=32, num_heads=8, num_groups=2, head_dim=16)
    x = torch.randn(5, 32)  # [sq*b flattened-ok, hidden]; here treat as [N, hidden]
    q_ref, k_ref, v_ref = a.reference_qkv(x)

    n = ps.split_fused_linears(
        a,
        split_qkv=True,
        split_fc1=False,
        block_size=16,
        block_count=None,
        linear_types=(nn.Linear,),
    )
    assert n == 1
    assert hasattr(a, "linear_q") and hasattr(a, "linear_k") and hasattr(a, "linear_v")
    assert not hasattr(a, "linear_qkv")

    q, k, v = a.get_query_key_value_tensors(x)
    assert torch.allclose(q, q_ref, atol=1e-6)
    assert torch.allclose(k, k_ref, atol=1e-6)
    assert torch.allclose(v, v_ref, atol=1e-6)


def test_split_qkv_mqa_contiguous():
    torch.manual_seed(1)
    a = _FakeAttention(hidden=32, num_heads=4, num_groups=1, head_dim=16)
    x = torch.randn(5, 32)
    q_ref, k_ref, _v_ref = a.reference_qkv(x)
    ps.split_fused_linears(
        a,
        split_qkv=True,
        split_fc1=False,
        block_size=16,
        block_count=None,
        linear_types=(nn.Linear,),
    )
    q, k, _v = a.get_query_key_value_tensors(x)
    assert torch.allclose(q, q_ref, atol=1e-6)
    assert torch.allclose(k, k_ref, atol=1e-6)


def test_split_qkv_hard_errors_on_indivisible_segment():
    import pytest

    # hidden=64 and q_out=128 both divide 64; kv_out=32 does NOT, so the first
    # failing segment is linear_k. (With hidden==kv_out, no block size can fail
    # K while passing Q, hence hidden=64 here.)
    a = _FakeAttention(hidden=64, num_heads=8, num_groups=2, head_dim=16)
    with pytest.raises(ValueError, match="linear_k"):
        ps.split_fused_linears(
            a,
            split_qkv=True,
            split_fc1=False,
            block_size=64,
            block_count=None,
            linear_types=(nn.Linear,),
        )


def test_split_qkv_inert_without_linear_qkv():
    m = nn.Module()  # MLA-like: no linear_qkv
    n = ps.split_fused_linears(
        m,
        split_qkv=True,
        split_fc1=False,
        block_size=16,
        block_count=None,
        linear_types=(nn.Linear,),
    )
    assert n == 0
