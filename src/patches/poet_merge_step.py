"""Patch: periodic POET merge-and-reinitialize in the training loop.

Targets ``megatron.training.training.train_step``. After each step,
if ``args.poet`` is set and ``iteration % args.poet_merge_period == 0``,
calls ``POETLinear.merge_then_reinitialize()`` on every POET layer and
broadcasts the updated state across ranks.

The fork-2 equivalent called ``poet_check_and_merge(model, iter, gap)``
from inside the training loop body. We instead wrap ``train_step``, which
receives ``(forward_step_func, data_iterator, model, optimizer,
opt_param_scheduler, config, forward_backward_func, iteration=None)`` —
``model`` is the 3rd positional arg, ``iteration`` the 8th kwarg/positional.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.train_step",)
logger = logging.getLogger(__name__)


@register_patch(name="poet_merge_step", targets=_TARGET)
def apply() -> None:
    import torch.distributed as dist
    from megatron.training import get_args
    from megatron.training import training as _mt

    _orig_train_step = _mt.train_step

    def _wrapped(*args, **kwargs):
        ret = _orig_train_step(*args, **kwargs)
        opts = get_args()
        if not getattr(opts, "poet", False):
            return ret
        gap = getattr(opts, "poet_merge_period", 0)
        if gap <= 0:
            return ret
        iteration = kwargs.get("iteration")
        if iteration is None and len(args) >= 8:
            iteration = args[7]
        if iteration is None:
            iteration = getattr(opts, "iteration", 0)
        if iteration <= 0 or iteration % gap != 0:
            return ret
        model = args[2] if len(args) >= 3 else kwargs.get("model")
        if model is None:
            logger.warning("[POET] merge step skipped: model not found in train_step args")
            return ret
        _run_merge(model, dist, iteration)
        return ret

    _mt.train_step = _wrapped


def _run_merge(model, dist, iteration: int) -> None:
    import torch
    from poet_torch import POETLinear

    from src.optim.poet_layers import POETMegatronLinear

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    chunks = model if isinstance(model, list) else [model]
    for m in chunks:
        for _, mod in m.named_modules():
            if not isinstance(mod, POETMegatronLinear):
                continue
            pl = mod.poet_linear
            if not isinstance(pl, POETLinear) or pl.block_size <= 0:
                continue
            with torch.no_grad():
                if rank == 0:
                    pl.merge_then_reinitialize()
                if is_dist:
                    for buf in (
                        pl.oft_R_in.data,
                        pl.oft_R_out.data,
                        pl.weight.data,
                        pl.perm_in,
                        pl.perm_in_inv,
                        pl.perm_out,
                        pl.perm_out_inv,
                    ):
                        dist.broadcast(buf, src=0)
            # Cache invalidation: weight and oft_R both changed under merge,
            # so any cached R blocks are stale. Guard with hasattr because
            # upstream POETLinear (cache_mode=none) doesn't have this method.
            if hasattr(pl, "_invalidate_R_cache"):
                pl._invalidate_R_cache()
    logger.info("[POET] merged at iteration %d", iteration)
