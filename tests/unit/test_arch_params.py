"""Unit tests for src/utils/arch_params.py (architecture param accounting).

CPU-only, no torch, no Megatron. All expected values are hand-computed from
the formulas documented in arch_params.py; the runtime ground-truth check is
the wandb_trainable_params log compared during GPU smoke (±2%).
"""

from __future__ import annotations

import pytest

from src.utils.arch_params import (
    active_non_embedding_params,
    attention_params,
    gdn_params,
    mamba_params,
    mla_params,
    moe_layer_active_params,
    moe_layer_params,
    non_embedding_params,
)


def test_attention_gqa():
    # q: 8*2*4=64, kv: 2*8*1*4=64, o: 2*4*8=64
    assert attention_params(hidden=8, num_heads=2, num_groups=1, head_dim=4) == 192


def test_attention_qk_norm_adds_two_head_dims():
    assert attention_params(hidden=8, num_heads=2, num_groups=1, head_dim=4, qk_norm=True) == 200


def test_mla():
    # q: 8*4 + 4 + 4*2*(4+2) = 84; kv: 8*(4+2) + 4 + 4*2*(4+4) = 116; o: 2*4*8 = 64
    assert (
        mla_params(
            hidden=8,
            num_heads=2,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_head_dim=4,
            qk_pos_emb_head_dim=2,
            v_head_dim=4,
        )
        == 264
    )


def test_gdn():
    # qk_dim=8, v_dim=16; in: 8*(16+32+8)=448; conv: 4*32=128; out: 128; small: 2*4+4=12
    assert (
        gdn_params(
            hidden=8,
            num_key_heads=2,
            key_head_dim=4,
            num_value_heads=4,
            value_head_dim=4,
            conv_kernel_dim=4,
        )
        == 716
    )


def test_mamba():
    # d_inner=16, nheads=4, conv_dim=32; in: 8*(32+16+4)=416; conv: 4*32+32=160;
    # out: 128; small: 3*4+16=28
    assert mamba_params(hidden=8, state_dim=4, head_dim=4, num_groups=2) == 732


def test_moe_layer():
    # router: 8*4+4=36; experts: 4*3*8*16=1536; shared: 3*8*16=384
    assert (
        moe_layer_params(
            hidden=8,
            num_experts=4,
            moe_ffn=16,
            shared_ffn=16,
            expert_bias=True,
        )
        == 1956
    )


def test_moe_layer_active():
    # router: 36; topk experts: 2*3*8*16=768; shared: 384
    assert (
        moe_layer_active_params(
            hidden=8,
            num_experts=4,
            topk=2,
            moe_ffn=16,
            shared_ffn=16,
            expert_bias=True,
        )
        == 1188
    )


def _dense_model() -> dict:
    return {
        "num_layers": 2,
        "hidden_size": 8,
        "ffn_hidden_size": 16,
        "num_attention_heads": 2,
        "num_query_groups": 1,
        "head_dim": 4,
        "activation": "SwiGLU",
        "qk_norm": False,
    }


def test_dispatch_dense_gpt():
    # per layer: attn 192 + swiglu 3*8*16=384 + 2 norms 16 = 592; x2 + final 8
    assert non_embedding_params(_dense_model()) == 1192


def test_dispatch_dense_with_mtp():
    # MTP block: one decoder layer 592 + eh_proj 2*8*8=128 + enorm 8 + hnorm 8 + final 8
    model = _dense_model() | {"mtp_num_layers": 1}
    assert non_embedding_params(model) == 1192 + 744


def test_dispatch_gdn_moe():
    model = {
        "num_layers": 2,
        "hidden_size": 8,
        "ffn_hidden_size": 16,
        "num_attention_heads": 2,
        "num_query_groups": 1,
        "head_dim": 4,
        "activation": "SwiGLU",
        "qk_norm": True,
        "linear_attention_freq": "[1, 0]",
        "gdn": {
            "enabled": True,
            "num_key_heads": 2,
            "key_head_dim": 4,
            "num_value_heads": 4,
            "value_head_dim": 4,
            "conv_kernel_dim": 4,
        },
        "moe": {
            "enabled": True,
            "layer_freq": "[1, 1]",
            "num_experts": 4,
            "ffn_hidden_size": 16,
            "shared_expert_intermediate_size": 16,
            "router_enable_expert_bias": True,
            "router_topk": 2,
        },
    }
    # layer0: gdn 716 + moe 1956 + norms 16 = 2688
    # layer1: attn(qk_norm) 200 + moe 1956 + norms 16 = 2172; final 8
    assert non_embedding_params(model) == 4868
    # active: moe -> 1188; layer0 1920 + layer1 1404 + final 8
    assert active_non_embedding_params(model) == 3332


def test_int_layer_freq_rejected():
    # Megatron's int form means different things per field; configs must use
    # explicit list-expression strings.
    model = _dense_model() | {
        "gdn": {
            "enabled": True,
            "num_key_heads": 2,
            "key_head_dim": 4,
            "num_value_heads": 4,
            "value_head_dim": 4,
            "conv_kernel_dim": 4,
        },
        "linear_attention_freq": 2,
    }
    with pytest.raises(ValueError):
        non_embedding_params(model)


def test_dispatch_hybrid_mamba():
    model = {
        "hidden_size": 8,
        "ffn_hidden_size": 16,
        "num_attention_heads": 2,
        "num_query_groups": 1,
        "head_dim": 4,
        "activation": "squared_relu",
        "qk_norm": False,
        "hybrid_layer_pattern": "M*-",
        "mamba": {"state_dim": 4, "head_dim": 4, "num_groups": 2},
    }
    # M 732 + attn 192 + relu2 mlp 2*8*16=256 + 3 per-layer norms 24 + final 8
    assert non_embedding_params(model) == 1212
    # no MoE -> active == total
    assert active_non_embedding_params(model) == 1212


def test_geglu_mlp_matches_swiglu():
    # GeGLU is gated (gate+up+down) like SwiGLU -> identical 3*h*ffn accounting.
    assert non_embedding_params(_dense_model() | {"activation": "GeGLU"}) == 1192


def test_sandwich_norm_adds_two_norms_per_layer():
    # Sandwich norm adds post-attn + post-mlp norm weights = +2*hidden per layer.
    # _dense_model() has 2 layers, hidden 8 -> +2*8*2 = +32.
    assert non_embedding_params(_dense_model() | {"use_sandwich_norm": True}) == 1192 + 32


def test_unknown_activation_rejected():
    with pytest.raises(ValueError):
        non_embedding_params(_dense_model() | {"activation": "gelu"})
