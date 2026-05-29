"""Patch: replace ParallelLinear modules with POETMegatronLinear after model build.

Targets ``megatron.training.training.get_model``. Mirrors the fork-2
``model_provider.py`` customisation that called ``apply_poet_to_model``
immediately after ``model_builder(...)`` returned.

Unfusing fused linears (qkv / fc1) is handled separately and earlier by the
``model_unfuse_linears`` patch (at ``model_provider`` time); by the time this
runs, the model already has whatever (fused or unfused) linears it will have,
and POET simply wraps each eligible one.
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
        # Per-parameter dump (name | shape | requires_grad) so the
        # block_count -> param-count mapping is inspectable from a
        # single-GPU smoke run. Rank-0 only to avoid 8x spam on real runs.
        import torch

        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_dist else 0
        if rank == 0:
            print("[POET] ===== parameter dump (name | shape | requires_grad) =====", flush=True)
            for m in chunks:
                for pname, p in m.named_parameters():
                    print(
                        f"[POET] {pname:<78} {tuple(p.shape)!s:<22} "
                        f"requires_grad={p.requires_grad} numel={p.numel()}",
                        flush=True,
                    )
            print("[POET] ===== end parameter dump =====", flush=True)

        trainable = sum(p.numel() for m in chunks for p in m.parameters() if p.requires_grad)
        frozen = sum(p.numel() for m in chunks for p in m.parameters() if not p.requires_grad)
        ratio = trainable / max(trainable + frozen, 1) * 100
        if rank == 0:
            print(
                f"[POET] replaced {total} linears | trainable={trainable} "
                f"frozen={frozen} ({ratio:.2f}%)",
                flush=True,
            )
        logger.info(
            "[POET] replaced %d linears | trainable=%d frozen=%d (%.2f%%)",
            total,
            trainable,
            frozen,
            ratio,
        )
        return model

    _mt.get_model = _wrapped
