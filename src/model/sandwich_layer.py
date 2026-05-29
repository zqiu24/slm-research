"""Sandwich-norm transformer layer for first-party Megatron 0.17.

Subclasses Megatron's ``TransformerLayer`` and, when ``config.use_sandwich_norm``
is set, adds a post-norm to the attention and MLP outputs *before* the residual
add (Huawei DeepSeek-3Bv2 "sandwich" norm). The post-norm is injected via a
forward-hook on ``self.self_attention`` / ``self.mlp`` so the (long, version-
coupled) ``_forward_attention`` / ``_forward_mlp`` methods are not copied.

No-op when ``use_sandwich_norm`` is false, so this class is safe as the default
GPT layer module.
"""

from __future__ import annotations

from megatron.core.transformer.transformer_layer import TransformerLayer

from src.model.sandwich_norm_ops import apply_post_norm_scale, make_post_norm_hook


def _sandwich_norm_cls():
    """TENorm if Transformer Engine is available, else WrappedTorchNorm.

    Mirrors the Huawei ``_get_sandwich_norm_impl``: TENorm imports even without
    TE (its __new__ raises at instantiation), so guard on HAVE_TE.
    """
    try:
        from megatron.core.extensions.transformer_engine import HAVE_TE, TENorm

        if HAVE_TE:
            return TENorm
    except ImportError:
        pass
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    return WrappedTorchNorm


class SandwichTransformerLayer(TransformerLayer):
    """TransformerLayer + optional post-attention / post-MLP sandwich norm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        if not getattr(cfg, "use_sandwich_norm", False):
            return
        norm_cls = _sandwich_norm_cls()
        self.post_self_attn_layernorm = norm_cls(
            config=cfg, hidden_size=cfg.hidden_size, eps=cfg.layernorm_epsilon
        )
        self.post_mlp_layernorm = norm_cls(
            config=cfg, hidden_size=cfg.hidden_size, eps=cfg.layernorm_epsilon
        )
        apply_post_norm_scale(
            self.post_self_attn_layernorm, getattr(cfg, "attn_post_norm_scale", 1.0)
        )
        apply_post_norm_scale(self.post_mlp_layernorm, getattr(cfg, "ffn_post_norm_scale", 1.0))
        # Post-norm the sub-layer output before the bias-dropout-residual add.
        self.self_attention.register_forward_hook(
            make_post_norm_hook(self.post_self_attn_layernorm)
        )
        self.mlp.register_forward_hook(make_post_norm_hook(self.post_mlp_layernorm))
