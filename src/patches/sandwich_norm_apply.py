"""Patch: stamp sandwich-norm config + swap in SandwichTransformerLayer.

Mirrors src/patches/ngpt_apply_spec.py. The CLI args (--use-sandwich-norm,
--attn-post-norm-scale, --ffn-post-norm-scale) are registered in add_slm_args;
this patch wraps gpt_builder so that, when sandwich-norm is on, it (1) stamps
the flags onto the TransformerConfig and (2) swaps the transformer-layer class
to SandwichTransformerLayer across every spec path (dense, MoE decoder block,
MTP). Both are installed *temporarily* around the gpt_builder call and removed
in a finally block, so the only owned target is gpt_builders.gpt_builder.

Why wrap gpt_builders.core_transformer_config_from_args (not the
megatron.training.arguments symbol): gpt_builders.py binds the name at import
time (`from megatron.training.arguments import core_transformer_config_from_args`)
and calls the bare local name, so the build-time config comes from THAT binding.
Wrapping it here also avoids owning the arguments-module target, which
poet_unfuse_te_impl owns — so this patch composes with optim/poet.

Megatron imports happen inside apply() so importing this module is CPU-safe.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = ("gpt_builders.gpt_builder",)
logger = logging.getLogger(__name__)


@register_patch(name="sandwich_norm_apply", targets=_TARGET)
def apply() -> None:
    import gpt_builders as _gb
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from src.model.sandwich_layer import SandwichTransformerLayer

    def _sandwichify(spec):
        """Set spec.module = SandwichTransformerLayer wherever a base
        TransformerLayer spec appears (single ModuleSpec, a list of them, or a
        TransformerBlockSubmodules with .layer_specs)."""
        if spec is None:
            return spec
        if isinstance(spec, list | tuple):
            for s in spec:
                _sandwichify(s)
        elif hasattr(spec, "layer_specs"):
            _sandwichify(spec.layer_specs)
        elif getattr(spec, "module", None) is TransformerLayer:
            spec.module = SandwichTransformerLayer
        return spec

    _orig_builder = _gb.gpt_builder
    # Names of the spec-producing functions gpt_builder calls (dense / MoE / MTP).
    _spec_fns = (
        "get_gpt_decoder_block_spec",
        "_get_transformer_layer_spec",
        "get_gpt_decoder_layer_specs",
    )

    def _wrapped_builder(args, *a, **kw):
        if not getattr(args, "use_sandwich_norm", False):
            return _orig_builder(args, *a, **kw)

        # (1) Temporarily stamp sandwich-norm onto the config gpt_builder builds.
        _orig_cfg = _gb.core_transformer_config_from_args

        def _wrapped_cfg(cfg_args, *ca, **ckw):
            config = _orig_cfg(cfg_args, *ca, **ckw)
            config.use_sandwich_norm = True
            config.attn_post_norm_scale = float(getattr(cfg_args, "attn_post_norm_scale", 1.0))
            config.ffn_post_norm_scale = float(getattr(cfg_args, "ffn_post_norm_scale", 1.0))
            return config

        _gb.core_transformer_config_from_args = _wrapped_cfg

        # (2) Temporarily swap the layer class across all spec paths.
        originals = {}
        for name in _spec_fns:
            fn = getattr(_gb, name, None)
            if fn is None:
                continue
            originals[name] = fn

            def _make(orig):
                def wrapped(*aa, **kk):
                    return _sandwichify(orig(*aa, **kk))

                return wrapped

            setattr(_gb, name, _make(fn))
        try:
            model = _orig_builder(args, *a, **kw)
        finally:
            _gb.core_transformer_config_from_args = _orig_cfg
            for name, fn in originals.items():
                setattr(_gb, name, fn)
        logger.info("[sandwich] swapped layer class + stamped config (dense/MoE/MTP)")
        return model

    _gb.gpt_builder = _wrapped_builder
