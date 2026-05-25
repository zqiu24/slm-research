"""Build the Megatron ModuleSpec for an nGPT transformer layer.

v1 constraints (see plan): TP=1, PP=1, dense (no MoE, no MLA). These are
checked at spec-build time so a misconfigured experiment fails fast at
submit instead of partway into a job.

The softmax-scale override (nGPT uses sqrt(head_dim), not 1/sqrt
(head_dim)) is *not* handled here. It is stamped onto
`TransformerConfig.softmax_scale` by the `ngpt_apply_spec` patch's wrap
of `core_transformer_config_from_args`; from there Megatron's
`SelfAttention.__init__` forwards it into `DotProductAttention`. Keeping
the override in the patch means the unit-tested spec builder stays
config-agnostic.
"""

from __future__ import annotations

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from src.model.ngpt.attention import QKHyperNorm
from src.model.ngpt.layer import NGPTTransformerLayer
from src.model.ngpt.mlp import NGPTMLP


def _qk_hyper_norm_builder(num_heads: int, head_dim: int, sqk_init: float, base_scale: float):
    def _build(hidden_size, eps=None, **_kwargs):
        # Megatron passes hidden_size==head_dim when constructing q/k_layernorm
        # (it does it from the per-head slice). We don't use the eps param.
        return QKHyperNorm(
            num_heads_per_tp=num_heads,
            head_dim=head_dim,
            sqk_init_value=sqk_init,
            base_scale=base_scale,
        )

    return _build


def build_ngpt_layer_spec(config) -> ModuleSpec:
    tp = getattr(config, "tensor_model_parallel_size", 1)
    assert tp == 1, (
        f"nGPT v1 requires TP=1, got tensor_model_parallel_size={tp}. "
        "TP>1 is a v2 follow-up (sqk/suv sharding)."
    )
    assert getattr(config, "num_moe_experts", None) in (None, 0), "nGPT v1 does not support MoE."
    assert not getattr(config, "multi_latent_attention", False), "nGPT v1 does not support MLA."

    num_heads = int(config.num_attention_heads)
    head_dim = int(config.hidden_size) // num_heads
    base_scale = float(getattr(config, "ngpt_base_scale", 1.0 / (config.hidden_size**0.5)))
    sqk_init = float(getattr(config, "ngpt_sqk_init", 1.0))

    submodules = TransformerLayerSubmodules(
        input_layernorm=IdentityOp,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=_qk_hyper_norm_builder(num_heads, head_dim, sqk_init, base_scale),
                k_layernorm=_qk_hyper_norm_builder(num_heads, head_dim, sqk_init, base_scale),
            ),
        ),
        self_attn_bda=IdentityFuncOp,
        pre_mlp_layernorm=IdentityOp,
        mlp=ModuleSpec(module=NGPTMLP),
        mlp_bda=IdentityFuncOp,
    )
    return ModuleSpec(module=NGPTTransformerLayer, submodules=submodules)
