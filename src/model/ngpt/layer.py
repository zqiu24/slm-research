"""nGPT transformer block.

This module ships two things:

* `NGPTBlock` — a pure-PyTorch transformer block that mirrors the
  reference's `Block` (with the same attention + MLP + residual-blend
  semantics, including the reference's *internal* bf16 cast for
  attention so parity tests aren't fighting a precision delta). It
  exists so the parity test can run CPU-side without a Megatron model.

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
import torch.nn as nn
from megatron.core.transformer.transformer_layer import TransformerLayer

from src.model.ngpt.attention import QKHyperNorm  # noqa: F401  (used via spec)
from src.model.ngpt.mlp import NGPTMLPBody  # noqa: F401  (used via spec)
from src.model.ngpt.normalize import justnorm
from src.model.ngpt.scaling_params import LearnedScaling


def _residual_blend(h: torch.Tensor, h_branch: torch.Tensor, alpha: LearnedScaling) -> torch.Tensor:
    """Hypersphere residual: h <- justnorm(justnorm(h) + |alpha| * (justnorm(h_branch) - justnorm(h)))."""
    lr = torch.abs(alpha.scaled_value()).to(h.dtype)
    a = justnorm(h)
    b = justnorm(h_branch)
    return justnorm(a + lr * (b - a))


def _apply_rope(sinusoidal_pos: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
    """Re-implementation of the reference's apply_rotary_position_embeddings."""
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, (*q.shape[:-1], q.shape[-1] // 2, 2)) * torch.stack(
        (cos, sin), dim=-1
    )
    k_rot = torch.reshape(k_rot, (*k.shape[:-1], k.shape[-1] // 2, 2)) * torch.stack(
        (cos, sin), dim=-1
    )
    return q_rot.reshape(q.shape), k_rot.reshape(k.shape)


def _sinusoidal_embeddings(n_positions: int, dim: int) -> torch.Tensor:
    import math

    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    emb = torch.zeros((n_positions, dim))
    emb[:, 0::2] = torch.sin(position * div_term)
    emb[:, 1::2] = torch.cos(position * div_term)
    return emb


class NGPTBlock(nn.Module):
    """CPU-runnable parity-oracle block. Mirrors reference Block(use_nGPT=1).

    Attention is computed in bf16 inside this method to match the
    reference's hardcoded `q.to(bfloat16)` / `k.to(bfloat16)` /
    `v.to(bfloat16)` casts inside `Block.forward` (model.py:136).
    Without that, the parity test would have to swallow a sustained
    precision gap that has nothing to do with whether nGPT is wired
    correctly.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        base_scale: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Same Linear shapes as reference.
        self.query = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.att_c_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.c_fc = nn.Linear(hidden_size, 2 * ffn_hidden_size, bias=False, dtype=dtype)
        self.mlp_c_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False, dtype=dtype)

        # Scaling params (init matches reference defaults).
        self.sqk = LearnedScaling((hidden_size,), init_value=1.0, init_scaling=base_scale)
        self.suv = LearnedScaling((2 * ffn_hidden_size,), init_value=1.0, init_scaling=1.0)
        self.attn_alpha = LearnedScaling((hidden_size,), init_value=0.05, init_scaling=base_scale)
        self.mlp_alpha = LearnedScaling((hidden_size,), init_value=0.05, init_scaling=base_scale)

        self._ffn_hidden_size = ffn_hidden_size
        self._n_embd_sqrt = float(hidden_size) ** 0.5

    def _attn(self, h: torch.Tensor) -> torch.Tensor:
        b, t, c = h.size()
        q = self.query(h).view(b, t, self.num_heads, self.head_dim)
        k = self.key(h).view(b, t, self.num_heads, self.head_dim)
        v = self.value(h).view(b, t, self.num_heads, self.head_dim)

        sinusoidal_pos = _sinusoidal_embeddings(t, self.head_dim).to(q.device)
        q, k = _apply_rope(sinusoidal_pos, q.transpose(1, 2), k.transpose(1, 2))
        q, k = q.transpose(2, 1), k.transpose(2, 1)

        sqk = self.sqk.scaled_value().view(1, 1, self.num_heads, self.head_dim).to(q.dtype)
        q = sqk * justnorm(q)
        k = sqk * justnorm(k)

        softmax_scale = self._n_embd_sqrt / (self.num_heads**0.5)  # = sqrt(head_dim)
        # Reference (model.py:136) explicitly casts q/k/v to bf16 inside
        # attention, regardless of outer dtype. Mirror that so parity holds.
        q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out_dtype = q.dtype
        q_bf, k_bf, v_bf = (t_.to(torch.bfloat16) for t_ in (q_, k_, v_))
        attn_bf = torch.nn.functional.scaled_dot_product_attention(
            q_bf,
            k_bf,
            v_bf,
            dropout_p=0.0,
            is_causal=True,
            scale=softmax_scale,
        )
        attn = attn_bf.to(out_dtype).transpose(1, 2).contiguous().view(b, t, c)
        return self.att_c_proj(attn)

    def _mlp(self, h: torch.Tensor) -> torch.Tensor:
        uv = self.c_fc(h)
        suv = (self.suv.scaled_value() * self._n_embd_sqrt).to(uv.dtype)
        uv = suv * uv
        u, v = uv.chunk(2, dim=-1)
        return self.mlp_c_proj(u * torch.nn.functional.silu(v))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_att = self._attn(h)
        h = _residual_blend(h, h_att, self.attn_alpha)
        h_mlp = self._mlp(h)
        h = _residual_blend(h, h_mlp, self.mlp_alpha)
        return h


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
        # v1 scope assertion — `forward` skips Megatron's recompute path.
        assert getattr(self.config, "recompute_granularity", None) is None, (
            "nGPT v1 does not support --recompute-granularity; override "
            "expects a single-pass forward."
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
