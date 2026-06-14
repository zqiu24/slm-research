"""Pin guard: the bake-off families rely on these Megatron CLI args existing
in third_party/Megatron-LM (core_v0.17.0). Re-run after any submodule bump
(SPEC.md §4.1 step 2).

Requires the cluster env (TransformerEngine's .so dlopens CUDA libs):
  source load_cuda13_2_nccl_env.sh
  PYTHONPATH=third_party/Megatron-LM <venv>/python -m pytest tests/integration/test_megatron_pin_features.py -v
"""

from __future__ import annotations

import sys

import pytest

REQUIRED_FIELDS = [
    # GatedDeltaNet (qwen3_next family)
    "experimental_attention_variant",
    "linear_attention_freq",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    # Hybrid mamba (nemotron_h family)
    "hybrid_layer_pattern",
    "mamba_state_dim",
    "mamba_head_dim",
    "mamba_num_groups",
    # Activation (nemotron_h family)
    "squared_relu",
    # Gemma 3 family (sliding-window interleave, GeGLU, zero-centered RMSNorm)
    "window_size",
    "window_attn_skip_freq",
    "quick_geglu",
    "layernorm_zero_centered_gamma",
]


def test_pin_exposes_family_flags():
    pytest.importorskip("transformer_engine")
    from megatron.training.arguments import parse_args

    argv, sys.argv = sys.argv, ["pin_guard"]
    try:
        args = parse_args(ignore_unknown_args=True)
    finally:
        sys.argv = argv
    missing = [f for f in REQUIRED_FIELDS if not hasattr(args, f)]
    assert not missing, f"pin lacks expected args: {missing}"
