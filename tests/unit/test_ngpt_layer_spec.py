"""CPU tests for the nGPT Megatron spec builder."""

import pytest


def test_build_ngpt_layer_spec_returns_module_spec():
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.spec_utils import ModuleSpec

    from src.model.ngpt.layer import NGPTTransformerLayer
    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec

    class _Cfg:
        hidden_size = 64
        num_attention_heads = 4
        ffn_hidden_size = 256
        num_query_groups = 4
        ngpt_base_scale = 1.0 / 8.0
        ngpt_sqk_init = 1.0
        ngpt_suv_init = 1.0

    spec = build_ngpt_layer_spec(_Cfg())
    assert isinstance(spec, ModuleSpec)
    assert spec.module is NGPTTransformerLayer
    sub = spec.submodules
    assert sub.input_layernorm is IdentityOp
    assert sub.pre_mlp_layernorm is IdentityOp
    # self_attn_bda / mlp_bda must be no-op-equivalent (IdentityFuncOp).
    from megatron.core.transformer.identity_op import IdentityFuncOp

    assert sub.self_attn_bda is IdentityFuncOp
    assert sub.mlp_bda is IdentityFuncOp

    # Regression guard: mlp.module MUST be a class so Megatron's build_module
    # instantiates it. A builder *function* is a types.FunctionType, which
    # build_module returns uninstantiated — leaving the layer's mlp a bare
    # function with no parameters (no mlp.linear_fc1/linear_fc2 weights).
    import types

    from src.model.ngpt.mlp import NGPTMLP

    assert sub.mlp.module is NGPTMLP
    assert not isinstance(sub.mlp.module, types.FunctionType)


def test_build_ngpt_layer_spec_asserts_tp1():
    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec

    class _CfgTp2:
        hidden_size = 64
        num_attention_heads = 4
        ffn_hidden_size = 256
        num_query_groups = 4
        ngpt_base_scale = 1.0 / 8.0
        ngpt_sqk_init = 1.0
        ngpt_suv_init = 1.0
        tensor_model_parallel_size = 2

    with pytest.raises(AssertionError, match="TP"):
        build_ngpt_layer_spec(_CfgTp2())


def test_build_ngpt_layer_spec_asserts_no_moe():
    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec

    class _CfgMoE:
        hidden_size = 64
        num_attention_heads = 4
        ffn_hidden_size = 256
        num_query_groups = 4
        ngpt_base_scale = 1.0 / 8.0
        ngpt_sqk_init = 1.0
        ngpt_suv_init = 1.0
        num_moe_experts = 4

    with pytest.raises(AssertionError, match="MoE"):
        build_ngpt_layer_spec(_CfgMoE())
