"""Patch: log trainable / total param counts into the W&B run *config* (Megatron).

Writes three static fields to the W&B Overview -> Config table (NOT charts):

* ``trainable_params`` — Σ ``p.numel()`` over params with ``requires_grad=True``
  (for POET this is ``oft_R``; for plain Adam it equals ``total_params``).
* ``total_params``      — Σ ``p.numel()`` over all params (incl. frozen base weights).
* ``trainable_pct``     — ``100 * trainable / total``.

Why a patch + ``config.update`` (not a chart, not a config arg): W&B's run
config is a snapshot taken at ``wandb.init()`` time — before the model exists —
so the counts can't be known then. Megatron itself uses exactly this pattern,
calling ``get_wandb_writer().config.update({...})`` right after
``setup_model_and_optimizer`` (training.py, for ``slurm_job_name``). We hook the
same point.

Hook: wrap ``setup_model_and_optimizer``. Counting happens *after* it returns,
so ``requires_grad`` reflects the final trainable set (POET freezes base weights
at model-build and the custom-POETAdam path restores requires_grad before
returning). Composes with ``poet_optimizer_setup``, which wraps the optimizer
*builders*, not ``setup_model_and_optimizer``.

Parallelism: counts are reduced (SUM) over the model-parallel group (TPxPP);
DP/CP ranks are replicas and excluded. No-op in the current DP-only setup
(model-parallel size 1). Expert parallelism (EP>1) is NOT separately aggregated;
we emit a warning so the number is never silently wrong.

Registered with ``targets=()`` (runtime wrapper, no static target ownership).
Module import is CPU-safe (torch / megatron / wandb imported only inside the
functions that need them).
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)


def _config_payload(trainable: int, total: int) -> dict:
    """Build the W&B-config dict; ``trainable_pct`` is 0.0 when ``total`` is 0."""
    pct = round(100.0 * trainable / total, 4) if total else 0.0
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": pct,
    }


def _reduce_model_parallel_sum(trainable: int, total: int) -> tuple[int, int]:
    """SUM ``(trainable, total)`` across the model-parallel group (TPxPP).

    DP/CP ranks hold replicas and are excluded, so the result is the global
    count for one replica. No-op when distributed is uninitialized or the
    model-parallel group is a single rank (the current DP-only setup).
    """
    import torch

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return trainable, total
    from megatron.core import parallel_state as mpu

    group = mpu.get_model_parallel_group()
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    counts = torch.tensor([trainable, total], dtype=torch.long, device=device)
    torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=group)
    return int(counts[0].item()), int(counts[1].item())


def _maybe_warn_expert_parallel() -> None:
    """Warn once if EP>1: expert params aren't aggregated across the EP group."""
    try:
        from megatron.training import get_args

        if int(getattr(get_args(), "expert_model_parallel_size", 1) or 1) > 1:
            logger.warning(
                "wandb_trainable_params: expert_model_parallel_size > 1; expert "
                "params are not aggregated across the expert-parallel group, so the "
                "logged counts undercount experts by that factor."
            )
    except Exception:  # logging must never crash training
        pass


@register_patch(name="wandb_trainable_params", targets=())
def apply() -> None:
    from megatron.training import training as _mt

    orig = _mt.setup_model_and_optimizer
    if getattr(orig, "_slm_trainable_params", False):
        return

    def _wrapped(*args, **kwargs):
        model, optimizer, opt_param_scheduler = orig(*args, **kwargs)

        # Count + reduce. This block runs identically on EVERY rank and must stay
        # OUTSIDE the logging-rank gate below: the all-reduce is a collective, so
        # a rank that skipped it would hang the ranks that didn't.
        trainable, total = 0, 0
        try:
            from src.utils.param_count import count_local_params

            chunks = model if isinstance(model, list | tuple) else [model]
            trainable, total = count_local_params(chunks)
            _maybe_warn_expert_parallel()
            trainable, total = _reduce_model_parallel_sum(trainable, total)
        except Exception:  # logging must never crash training; fall back to zeros
            logger.warning("wandb_trainable_params: counting failed", exc_info=True)
            trainable, total = 0, 0

        # Write to the run config on the W&B-logging rank only (writer is None
        # elsewhere). Mirrors Megatron's own post-setup config.update.
        try:
            writer = _mt.get_wandb_writer()
            if writer is not None:
                writer.config.update(_config_payload(trainable, total), allow_val_change=True)
        except Exception:  # logging must never crash training
            logger.warning("wandb_trainable_params: config.update failed", exc_info=True)

        return model, optimizer, opt_param_scheduler

    _wrapped._slm_trainable_params = True
    _mt.setup_model_and_optimizer = _wrapped
