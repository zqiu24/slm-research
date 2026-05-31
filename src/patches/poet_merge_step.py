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
import os

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
        # Vanilla A/B path: POETAdam isn't in the loop to reset momentum, so do
        # it here (same cadence as the weight merge).
        if os.environ.get("POET_VANILLA_OPT") == "1":
            optimizer = args[3] if len(args) >= 4 else kwargs.get("optimizer")
            if optimizer is not None:
                _reset_vanilla_momentum(optimizer, model, iteration)
        return ret

    _mt.train_step = _wrapped


def _reset_vanilla_momentum(optimizer, model, iteration: int) -> None:
    """POETAdam-faithful momentum reset for the ``POET_VANILLA_OPT`` path.

    POETAdam reset only its oft_R branch (exp_avg / exp_avg_sq / step), leaving
    the embedding/norm Adam state untouched. We reproduce that exactly: zero the
    Adam state for the oft_R params ONLY.

    The stock bf16 optimizer keys its Adam state by the fp32 *master* params, not
    the model's bf16 oft_R tensors, so we map model->master via the optimizer's
    parallel ``float16_groups`` / ``fp32_from_float16_groups`` lists (state is
    keyed by the master). Falls back to ``fp32_from_fp32_groups`` and, for an
    FP32Optimizer, to the model params directly.
    """
    import torch

    chunks = model if isinstance(model, list) else [model]
    oft_ids = {
        id(p)
        for m in chunks
        for name, p in m.named_parameters()
        if "oft_R" in name and p.requires_grad
    }

    def _zero(master_param, torch_opt) -> int:
        st = torch_opt.state.get(master_param)
        if not st:
            return 0
        if "exp_avg" in st:
            st["exp_avg"].zero_()
        if "exp_avg_sq" in st:
            st["exp_avg_sq"].zero_()
        if "step" in st:
            if torch.is_tensor(st["step"]):
                st["step"].zero_()
            else:
                st["step"] = 0
        return 1

    inner = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    n = 0
    for opt in inner:
        torch_opt = getattr(opt, "optimizer", None)
        if torch_opt is None:
            continue
        f16 = getattr(opt, "float16_groups", None)
        fp32_master = getattr(opt, "fp32_from_float16_groups", None)
        if f16 is not None and fp32_master is not None:
            # bf16/fp16 optimizer: Adam state lives on the fp32 master copies.
            for f16_grp, master_grp in zip(f16, fp32_master, strict=False):
                for model_p, master_p in zip(f16_grp, master_grp, strict=False):
                    if id(model_p) in oft_ids:
                        n += _zero(master_p, torch_opt)
            for grp in getattr(opt, "fp32_from_fp32_groups", None) or []:
                for p in grp:
                    if id(p) in oft_ids:
                        n += _zero(p, torch_opt)
        else:
            # FP32Optimizer: Adam state is keyed by the model params directly.
            for group in torch_opt.param_groups:
                for p in group["params"]:
                    if id(p) in oft_ids:
                        n += _zero(p, torch_opt)
    logger.info("[POET] vanilla momentum reset (oft_R only) at iter %d (%d params)", iteration, n)


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
