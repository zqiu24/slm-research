"""pgpt layer-spec structure + v1 guardrails (TP=1, no MoE/MLA)."""

import types

import pytest

from src.specs.pgpt_layer_spec import build_pgpt_layer_spec


def _cfg(**over):
    base = dict(
        tensor_model_parallel_size=1,
        num_moe_experts=None,
        multi_latent_attention=False,
        num_attention_heads=4,
        hidden_size=64,
        ngpt_base_scale=1.0 / 8.0,
        ngpt_sqk_init=1.0,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_spec_wires_pgpt_layer_and_identity_norms():
    from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp

    from src.model.pgpt.layer import PGPTTransformerLayer
    from src.model.pgpt.mlp import PGPTMLP

    spec = build_pgpt_layer_spec(_cfg())
    assert spec.module is PGPTTransformerLayer
    sub = spec.submodules
    assert sub.input_layernorm is IdentityOp
    assert sub.pre_mlp_layernorm is IdentityOp
    assert sub.self_attn_bda is IdentityFuncOp
    assert sub.mlp_bda is IdentityFuncOp
    assert sub.mlp.module is PGPTMLP


def test_spec_rejects_tp_gt_1():
    with pytest.raises(AssertionError):
        build_pgpt_layer_spec(_cfg(tensor_model_parallel_size=2))


def test_spec_rejects_moe():
    with pytest.raises(AssertionError):
        build_pgpt_layer_spec(_cfg(num_moe_experts=8))
