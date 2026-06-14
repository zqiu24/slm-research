"""Arg-emission tests for the new family mechanisms (GDN, hybrid mamba,
non-SwiGLU activation, non-rope positional encoding)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.utils.megatron_args import _model_args


def _cfg(**overrides):
    model = {
        "transformer_impl": None,
        "num_layers": 2,
        "hidden_size": 8,
        "ffn_hidden_size": 16,
        "num_attention_heads": 2,
        "num_query_groups": 2,
        "head_dim": 4,
        "seq_length": 16,
        "positional_encoding": "rope",
        "rotary_base": 10000,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "normalization": "RMSNorm",
        "norm_epsilon": 1.0e-6,
        "init_method_std": 0.02,
        "tie_embeddings": True,
        "activation": "SwiGLU",
    }
    model.update(overrides)
    return OmegaConf.create({"base": {"model": model}})


def _value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_swiglu_default_unchanged():
    args = _model_args(_cfg())
    assert "--swiglu" in args
    assert "--squared-relu" not in args
    assert _value(args, "--rotary-base") == "10000"


def test_squared_relu_replaces_swiglu():
    args = _model_args(_cfg(activation="squared_relu"))
    assert "--squared-relu" in args
    assert "--swiglu" not in args


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        _model_args(_cfg(activation="gelu"))


def test_positional_none_omits_rotary_args():
    args = _model_args(_cfg(positional_encoding="none"))
    assert _value(args, "--position-embedding-type") == "none"
    assert "--rotary-base" not in args
    assert "--rotary-percent" not in args


def test_gdn_emission():
    cfg = _cfg(
        qk_norm=True,
        linear_attention_freq="([1]*1+[0]*1)",
        gdn={
            "enabled": True,
            "num_key_heads": 2,
            "key_head_dim": 4,
            "num_value_heads": 4,
            "value_head_dim": 4,
            "conv_kernel_dim": 4,
        },
    )
    args = _model_args(cfg)
    assert _value(args, "--experimental-attention-variant") == "gated_delta_net"
    assert _value(args, "--linear-attention-freq") == "([1]*1+[0]*1)"
    assert _value(args, "--linear-num-key-heads") == "2"
    assert _value(args, "--linear-key-head-dim") == "4"
    assert _value(args, "--linear-num-value-heads") == "4"
    assert _value(args, "--linear-value-head-dim") == "4"
    assert _value(args, "--linear-conv-kernel-dim") == "4"
    assert "--enable-experimental" in args


def test_hybrid_pattern_and_mamba_dims():
    cfg = _cfg(
        positional_encoding="none",
        num_layers=3,
        hybrid_layer_pattern="M*-",
        mamba={"state_dim": 4, "head_dim": 4, "num_groups": 2},
    )
    args = _model_args(cfg)
    assert _value(args, "--hybrid-layer-pattern") == "M*-"
    assert _value(args, "--mamba-state-dim") == "4"
    assert _value(args, "--mamba-head-dim") == "4"
    assert _value(args, "--mamba-num-groups") == "2"


def test_hybrid_pattern_length_mismatch_raises():
    cfg = _cfg(positional_encoding="none", num_layers=2, hybrid_layer_pattern="M*-")
    with pytest.raises(ValueError):
        _model_args(cfg)


def test_hybrid_rejects_mtp():
    cfg = _cfg(
        positional_encoding="none",
        num_layers=3,
        hybrid_layer_pattern="M*-",
        mtp_num_layers=1,
    )
    with pytest.raises(ValueError):
        _model_args(cfg)


def test_geglu_emits_quick_geglu():
    args = _model_args(_cfg(activation="GeGLU"))
    assert "--quick-geglu" in args
    assert "--swiglu" not in args
    assert "--squared-relu" not in args


def test_layernorm_zero_centered_emits_1p():
    args = _model_args(_cfg(layernorm_zero_centered=True))
    assert "--apply-layernorm-1p" in args


def test_layernorm_zero_centered_default_omits_1p():
    assert "--apply-layernorm-1p" not in _model_args(_cfg())


def test_sliding_window_emission():
    args = _model_args(_cfg(sliding_window={"enabled": True, "window": 1024, "skip_freq": 6}))
    assert _value(args, "--window-size") == "1024,0"
    assert _value(args, "--window-attn-skip-freq") == "6"


def test_sliding_window_disabled_omits_flags():
    args = _model_args(_cfg())
    assert "--window-size" not in args
    assert "--window-attn-skip-freq" not in args
