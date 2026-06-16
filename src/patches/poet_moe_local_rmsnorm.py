"""Patch: fix the MoE decoder block's *final* layer norm under local impl + RMSNorm.

``get_gpt_decoder_block_spec`` (the MoE path, taken when ``--num-experts`` is
set) hardcodes the block's final ``layer_norm`` to ``gpt_layer_specs.LNImpl``,
which is ``FusedLayerNorm`` whenever apex is importable. ``FusedLayerNorm``
asserts ``config.normalization == "LayerNorm"`` and so cannot build an RMSNorm
final norm. The per-layer norms avoid this because the local spec builds them
through ``LocalSpecProvider().layer_norm(rms_norm=True)`` -> ``WrappedTorchNorm``
(``torch.nn.RMSNorm``); only the block-level final norm is left on ``LNImpl``.

This bites POET specifically: POET forces ``transformer_impl=local`` (it cannot
replace fused ``TELayerNormColumnParallelLinear``), and the DeepSeek MoE +
RMSNorm model then hits the final-norm assertion at build. Dense POET runs
(llama3 60m) never hit it — dense uses a single-layer spec, not the MoE block
spec. MoE+RMSNorm runs on the TE path are fine (TENorm handles RMSNorm).

Fix: wrap ``gpt_builders.get_gpt_decoder_block_spec`` (the name as bound in
gpt_builders -- wrapping the originating module would miss the import-time
binding) and, for a local (non-TE) RMSNorm build, swap a ``FusedLayerNorm``
final norm for ``WrappedTorchNorm`` -- exactly the norm the per-layer specs
already use in this same config. No-op for LayerNorm, TE builds, or when the
final norm is already RMSNorm-capable. Owns only this one symbol, so it
composes with sandwich_norm_apply (owns gpt_builder) and the poet patches.

Megatron imports happen inside apply() so importing this module is CPU-safe.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = ("gpt_builders.get_gpt_decoder_block_spec",)
logger = logging.getLogger(__name__)


@register_patch(name="poet_moe_local_rmsnorm", targets=_TARGET)
def apply() -> None:
    import gpt_builders as _gb
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    _orig = _gb.get_gpt_decoder_block_spec

    def _wrapped(config, use_transformer_engine, *args, **kwargs):
        spec = _orig(config, use_transformer_engine, *args, **kwargs)
        normalization = kwargs.get("normalization") or getattr(config, "normalization", None)
        if (
            not use_transformer_engine
            and normalization == "RMSNorm"
            and getattr(spec, "layer_norm", None) is FusedLayerNorm
        ):
            spec.layer_norm = WrappedTorchNorm
            logger.info(
                "[poet_moe_local_rmsnorm] block final norm FusedLayerNorm -> "
                "WrappedTorchNorm (local impl + RMSNorm)"
            )
        return spec

    _gb.get_gpt_decoder_block_spec = _wrapped
