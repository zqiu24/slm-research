"""nGPT transformer block.

This module provides the Megatron-integrated layer and re-exports the
pure-PyTorch parity oracle for back-compat:

* `NGPTBlock` — the pure-PyTorch parity-oracle block now lives in
  `src/model/ngpt/block.py`. It is re-exported here so existing
  `from src.model.ngpt.layer import NGPTBlock` callers keep working.

* `NGPTTransformerLayer` — subclass of Megatron's `TransformerLayer`
  that overrides `forward` to apply nGPT's residual blend. It expects
  the surrounding spec to wire `input_layernorm` and `pre_mlp_layernorm`
  to `IdentityOp` (no pre-norm in nGPT) and `self_attn_bda` /
  `mlp_bda` to `IdentityFuncOp` (we do the residual ourselves; these
  slots are built but never invoked because we override `forward`).
  Learned scaling parameters `attn_alpha` and `mlp_alpha` are built in
  `__init__` — building them in `forward` would mean they don't exist
  when Megatron's optimizer is constructed (which walks
  `model.named_parameters()` before the first forward).
"""

from __future__ import annotations

import torch
from megatron.core.transformer.transformer_layer import TransformerLayer

from src.model.ngpt.attention import QKHyperNorm  # noqa: F401  (used via spec)
from src.model.ngpt.block import (  # _residual_blend reused by forward
    NGPTBlock,  # noqa: F401  -- re-exported for back-compat
    _residual_blend,
)
from src.model.ngpt.mlp import NGPTMLPBody  # noqa: F401  (used via spec)
from src.model.ngpt.scaling_params import LearnedScaling

# ---------------------------------------------------------------------------
# Megatron-integrated layer (T=1, dense). Used when the spec wires this in.
# ---------------------------------------------------------------------------


class NGPTTransformerLayer(TransformerLayer):
    """nGPT layer for Megatron. Overrides `forward` to apply hypersphere blend.

    The companion spec builder (`src/specs/ngpt_layer_spec.py`) wires:

      input_layernorm    = IdentityOp
      pre_mlp_layernorm  = IdentityOp
      self_attn_bda      = IdentityFuncOp     # built but never called
      mlp_bda            = IdentityFuncOp     # built but never called
      self_attention.q_layernorm/k_layernorm = QKHyperNorm
      mlp.module         = NGPTMLPBody

    `attn_alpha` and `mlp_alpha` are constructed in `__init__` so they
    are present in `model.named_parameters()` *before* Megatron's
    optimizer is built. Building them lazily in `forward` would leave
    them out of the optimizer entirely — they'd never receive gradients
    and the run would silently train without them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `forward` is a pure function of its inputs (no dropout — nGPT forces
        # attention/hidden dropout to 0 — and no in-place state), so Megatron's
        # 'full' activation recompute (which wraps the whole layer in
        # tensor_parallel.checkpoint and re-runs it in backward) is safe and
        # frees the per-layer hypersphere activations. 'selective' (core-
        # attention-only recompute) is not yet validated against the override.
        rg = getattr(self.config, "recompute_granularity", None)
        assert rg in (None, "full"), (
            f"nGPT supports recompute_granularity in (None, 'full'), got {rg!r}; "
            "'selective' is not yet validated against the nGPT forward override."
        )

        hidden = int(self.config.hidden_size)
        # These fields are stamped onto the config by `ngpt_apply_spec`'s
        # wrap of `core_transformer_config_from_args`. Falling back to
        # defaults makes layer-only unit-testing easier.
        base_scale = float(getattr(self.config, "ngpt_base_scale", 1.0 / (hidden**0.5)))
        alpha_init = float(getattr(self.config, "ngpt_alpha_init", 0.05))
        self.attn_alpha = LearnedScaling((hidden,), init_value=alpha_init, init_scaling=base_scale)
        self.mlp_alpha = LearnedScaling((hidden,), init_value=alpha_init, init_scaling=base_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        padding_mask: torch.Tensor | None = None,
        inference_params=None,
        **kwargs,
    ):
        # ---- Attention branch ----
        # input_layernorm is wired to IdentityOp, so `hidden_states` is already
        # the attention input. We mirror Megatron's _forward_attention call into
        # self.self_attention (kwargs incl. rotary_pos_cos_sin vary by Megatron
        # version; **kwargs keeps the override forward-compatible) but replace
        # the bias-dropout-add residual with the nGPT hypersphere blend.
        attn_out_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        # attn_out_with_bias is (output, bias). We use only output; bias is None
        # under --disable-bias-linear which nGPT requires.
        attn_out = (
            attn_out_with_bias[0] if isinstance(attn_out_with_bias, tuple) else attn_out_with_bias
        )
        hidden_states = _residual_blend(hidden_states, attn_out, self.attn_alpha)

        # ---- MLP branch ----
        mlp_out_with_bias = self.mlp(hidden_states)
        mlp_out = (
            mlp_out_with_bias[0] if isinstance(mlp_out_with_bias, tuple) else mlp_out_with_bias
        )
        hidden_states = _residual_blend(hidden_states, mlp_out, self.mlp_alpha)

        # Megatron's layer returns (hidden_states, context). nGPT has no context.
        return hidden_states, context
