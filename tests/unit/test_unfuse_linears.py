"""Tests for architectural unfusing of fused linears (geometry + surgery).

These are optimizer-agnostic; divisibility/hard-error under POET is tested
separately in test_poet_layers.py.
"""

import torch
import torch.nn as nn

from src.model import unfuse_linears as uf


def test_segment_out_dims_mqa_and_gqa():
    # MQA: 16 heads, 1 group, head_dim 384.
    assert uf.qkv_segment_out_dims(16, 1, 384) == (16 * 384, 1 * 384)
    # GQA: 8 heads, 2 groups, head_dim 64.
    assert uf.qkv_segment_out_dims(8, 2, 64) == (8 * 64, 2 * 64)


def test_deinterleave_row_indices_gqa_layout():
    # 4 heads, 2 groups (2 q-heads/group), head_dim 1 → trivial indices.
    # Fused per-group layout: [q,q,k,v] → group0 rows 0,1,2,3 ; group1 rows 4,5,6,7
    q, k, v = uf.qkv_deinterleave_row_indices(4, 2, 1)
    assert q.tolist() == [0, 1, 4, 5]
    assert k.tolist() == [2, 6]
    assert v.tolist() == [3, 7]


def test_interleave_index_roundtrips_weight():
    # Random fused weight; de-interleave then re-interleave must reproduce it.
    nah, ng, hd, in_f = 8, 2, 16, 32
    q_out, kv_out = uf.qkv_segment_out_dims(nah, ng, hd)
    total = q_out + 2 * kv_out
    w = torch.randn(total, in_f)
    qr, kr, vr = uf.qkv_deinterleave_row_indices(nah, ng, hd)
    cat = torch.cat([w[qr], w[kr], w[vr]], dim=0)  # de-interleaved
    idx = uf.qkv_interleave_index(nah, ng, hd)
    assert torch.equal(cat.index_select(0, idx), w)


class _FakeConfig:
    def __init__(self):
        self.gated_linear_unit = True
        self.activation_func = torch.nn.functional.silu
        self.activation_func_clamp_value = None
        self.glu_linear_offset = 0.0


class _FakeMLP(nn.Module):
    """Stand-in mimicking Megatron MLP's attributes used by the unfuse path."""

    def __init__(self, hidden=8, ffn=16):
        super().__init__()
        self.config = _FakeConfig()
        self.linear_fc1 = nn.Linear(hidden, 2 * ffn, bias=False)  # [gate; up]
        self.linear_fc2 = nn.Linear(ffn, hidden, bias=False)


def test_unfuse_fc1_creates_separate_modules_and_matches_fused():
    torch.manual_seed(0)
    m = _FakeMLP(hidden=8, ffn=16)
    x = torch.randn(3, 8)

    # Reference: fused forward (silu(gate) * up) -> fc2.
    fused = m.linear_fc1(x)
    gate_ref, up_ref = torch.chunk(fused, 2, dim=-1)
    ref = m.linear_fc2(torch.nn.functional.silu(gate_ref) * up_ref)

    n = uf.unfuse_fused_linears(m, unfuse_qkv=False, unfuse_fc1=True, linear_types=(nn.Linear,))
    assert n == 1
    assert hasattr(m, "linear_fc1_gate") and hasattr(m, "linear_fc1_up")
    assert not hasattr(m, "linear_fc1")
    assert isinstance(m.linear_fc1_gate, nn.Linear)

    out, _out_bias = m.forward(x)
    assert torch.allclose(out, ref, atol=1e-6)


def test_unfuse_fc1_requires_gated_mlp():
    import pytest

    m = _FakeMLP(hidden=8, ffn=16)
    m.config.gated_linear_unit = False
    with pytest.raises(ValueError, match="gated"):
        uf.unfuse_fused_linears(m, unfuse_qkv=False, unfuse_fc1=True, linear_types=(nn.Linear,))


class _FakeAttnConfig:
    attention_output_gate = False


class _FakeAttention(nn.Module):
    """Stand-in mimicking Megatron SelfAttention attributes used by the unfuse."""

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


def test_unfuse_qkv_creates_modules_and_matches_reference():
    torch.manual_seed(0)
    a = _FakeAttention(hidden=32, num_heads=8, num_groups=2, head_dim=16)
    x = torch.randn(5, 32)  # treat as [N, hidden]
    q_ref, k_ref, v_ref = a.reference_qkv(x)

    n = uf.unfuse_fused_linears(a, unfuse_qkv=True, unfuse_fc1=False, linear_types=(nn.Linear,))
    assert n == 1
    assert hasattr(a, "linear_q") and hasattr(a, "linear_k") and hasattr(a, "linear_v")
    assert not hasattr(a, "linear_qkv")

    q, k, v = a.get_query_key_value_tensors(x)
    assert torch.allclose(q, q_ref, atol=1e-6)
    assert torch.allclose(k, k_ref, atol=1e-6)
    assert torch.allclose(v, v_ref, atol=1e-6)


def test_unfuse_qkv_mqa_contiguous():
    torch.manual_seed(1)
    a = _FakeAttention(hidden=32, num_heads=4, num_groups=1, head_dim=16)
    x = torch.randn(5, 32)
    q_ref, k_ref, _v_ref = a.reference_qkv(x)
    uf.unfuse_fused_linears(a, unfuse_qkv=True, unfuse_fc1=False, linear_types=(nn.Linear,))
    q, k, _v = a.get_query_key_value_tensors(x)
    assert torch.allclose(q, q_ref, atol=1e-6)
    assert torch.allclose(k, k_ref, atol=1e-6)


def test_unfuse_qkv_inert_without_linear_qkv():
    m = nn.Module()  # MLA-like: no linear_qkv
    n = uf.unfuse_fused_linears(m, unfuse_qkv=True, unfuse_fc1=False, linear_types=(nn.Linear,))
    assert n == 0


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = _FakeAttention(hidden=32, num_heads=8, num_groups=2, head_dim=16)
        self.mlp = _FakeMLP(hidden=32, ffn=16)


def test_both_unfuse_compose_and_register_separate_submodules():
    block = _FakeBlock()
    n = uf.unfuse_fused_linears(block, unfuse_qkv=True, unfuse_fc1=True, linear_types=(nn.Linear,))
    assert n == 2
    names = dict(block.named_modules())
    assert "self_attention.linear_q" in names
    assert "self_attention.linear_k" in names
    assert "self_attention.linear_v" in names
    assert "mlp.linear_fc1_gate" in names
    assert "mlp.linear_fc1_up" in names
    assert "self_attention.linear_qkv" not in names
    assert "mlp.linear_fc1" not in names
    # The interleave index is a non-persistent buffer (not a trainable param).
    assert "_unfuse_qkv_interleave_index" not in dict(block.named_parameters())
