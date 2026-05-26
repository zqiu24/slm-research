"""Patch: force ``config.transformer_impl='local'`` when POET is enabled.

Targets ``megatron.training.arguments.core_transformer_config_from_args``.
Without this, Megatron's GPT spec materialises fused
``TELayerNormColumnParallelLinear`` modules which POET cannot replace
(layer-norm payload would be silently dropped).

Upstream SHA (pinned via third_party/Megatron-LM): see docs/megatron_pin.md.
"""

from __future__ import annotations

from src.patches._registry import register_patch

_TARGET = ("megatron.training.arguments.core_transformer_config_from_args",)


@register_patch(name="poet_unfuse_te_impl", targets=_TARGET)
def apply() -> None:
    """Wrap ``core_transformer_config_from_args`` to flip the impl when POET is on."""
    from megatron.training import arguments as _ma

    _orig = _ma.core_transformer_config_from_args

    def _wrapped(args, *a, **kw):
        config = _orig(args, *a, **kw)
        if not getattr(args, "poet", False):
            return config
        if getattr(config, "transformer_impl", None) == "inference_optimized":
            raise ValueError(
                "POET is not supported with --transformer-impl "
                "inference_optimized. Use 'local' (or omit; "
                "transformer_engine will be unfused to local automatically)."
            )
        if getattr(config, "transformer_impl", None) == "transformer_engine":
            config.transformer_impl = "local"
        return config

    _ma.core_transformer_config_from_args = _wrapped
