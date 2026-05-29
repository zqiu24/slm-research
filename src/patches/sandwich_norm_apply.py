"""Patch: stamp sandwich-norm config + swap in SandwichTransformerLayer.

Mirrors src/patches/ngpt_apply_spec.py. The CLI args (--use-sandwich-norm,
--attn-post-norm-scale, --ffn-post-norm-scale) are registered in add_slm_args;
this patch only (1) stamps them onto the TransformerConfig and (2) swaps the
transformer-layer class to SandwichTransformerLayer across every spec path used
by gpt_builder — the dense spec, the MoE decoder block spec (.layer_specs), and
the MTP layer spec. All gated on args.use_sandwich_norm (no-op otherwise).
Megatron imports happen inside apply() so importing this module is CPU-safe.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = (
    "gpt_builders.gpt_builder",
    "megatron.training.arguments.core_transformer_config_from_args",
)
logger = logging.getLogger(__name__)


@register_patch(name="sandwich_norm_apply", targets=_TARGET)
def apply() -> None:
    # ---- stamp config from args ----
    from megatron.training import arguments as _ma

    _orig_cfg = _ma.core_transformer_config_from_args

    def _wrapped_cfg(args, *a, **kw):
        config = _orig_cfg(args, *a, **kw)
        if getattr(args, "use_sandwich_norm", False):
            config.use_sandwich_norm = True
            config.attn_post_norm_scale = float(getattr(args, "attn_post_norm_scale", 1.0))
            config.ffn_post_norm_scale = float(getattr(args, "ffn_post_norm_scale", 1.0))
        return config

    _ma.core_transformer_config_from_args = _wrapped_cfg

    # ---- swap the layer class across all spec paths ----
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
            for name, fn in originals.items():
                setattr(_gb, name, fn)
        logger.info("[sandwich] swapped layer class on all spec paths (dense/MoE/MTP)")
        return model

    _gb.gpt_builder = _wrapped_builder
