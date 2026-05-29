"""Patch: replace ParallelLinear modules with POETMegatronLinear after model build.

Targets ``megatron.training.training.get_model``. Mirrors the fork-2
``model_provider.py`` customisation that called ``apply_poet_to_model``
immediately after ``model_builder(...)`` returned.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.get_model",)
logger = logging.getLogger(__name__)


@register_patch(name="poet_apply_to_model", targets=_TARGET)
def apply() -> None:
    from megatron.training import get_args
    from megatron.training import training as _mt

    from src.optim.poet_layers import replace_linears_with_poet

    _orig = _mt.get_model

    def _wrapped(*a, **kw):
        model = _orig(*a, **kw)
        args = get_args()
        if not getattr(args, "poet", False):
            return model
        block = getattr(args, "poet_block_size", 256)
        block_count = getattr(args, "poet_block_count", None)
        init = getattr(args, "poet_init_type", "normalized")
        mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        cache_mode = getattr(args, "poet_cache_mode", "none")
        chunks = model if isinstance(model, list) else [model]

        split_qkv = getattr(args, "poet_split_qkv", False)
        split_fc1 = getattr(args, "poet_split_fc1", False)
        if split_qkv or split_fc1:
            from src.optim.poet_split import split_fused_linears

            for m in chunks:
                split_fused_linears(
                    m,
                    split_qkv=split_qkv,
                    split_fc1=split_fc1,
                    block_size=block,
                    block_count=block_count,
                )

        total = 0
        for m in chunks:
            total += replace_linears_with_poet(
                m,
                block_size=block,
                block_count=block_count,
                init_type=init,
                mup_alpha=mup_alpha,
                cache_mode=cache_mode,
            )
        trainable = sum(p.numel() for m in chunks for p in m.parameters() if p.requires_grad)
        frozen = sum(p.numel() for m in chunks for p in m.parameters() if not p.requires_grad)
        ratio = trainable / max(trainable + frozen, 1) * 100
        logger.info(
            "[POET] replaced %d linears | trainable=%d frozen=%d (%.2f%%)",
            total,
            trainable,
            frozen,
            ratio,
        )
        return model

    _mt.get_model = _wrapped
