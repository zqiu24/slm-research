"""Patch: log POET Lie-Orth update-RMS optimizer diagnostics to W&B.

The optimizer computes cheap per-step tensors in
``LieOrthUpdateRMSMomentum.last_update_rms_stats``. This patch is intentionally
separate from the optimizer hot path: it wraps Megatron's
``setup_model_and_optimizer``, then logs those cached tensors after
``optimizer.step`` at the normal training log interval.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)


def _iter_inner_optimizers(optimizer) -> Iterable[object]:
    inner = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    for opt in inner:
        torch_opt = getattr(opt, "optimizer", None)
        yield torch_opt if torch_opt is not None else opt


def _update_rms_optimizers(optimizer) -> list[object]:
    return [
        opt for opt in _iter_inner_optimizers(optimizer) if hasattr(opt, "last_update_rms_stats")
    ]


def _to_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _stats_payload(update_rms_opts: list[object]) -> dict[str, float]:
    by_key: dict[str, list[float]] = {}
    for opt in update_rms_opts:
        stats = getattr(opt, "last_update_rms_stats", {}) or {}
        for key, value in stats.items():
            by_key.setdefault(str(key), []).append(_to_float(value))
    return {key: sum(vals) / len(vals) for key, vals in by_key.items() if vals}


def _log_update_rms(update_rms_opts: list[object], iteration: int) -> None:
    try:
        import wandb
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return
    payload = _stats_payload(update_rms_opts)
    if payload:
        wandb.log(payload, step=int(iteration))


def _install_step_hook(optimizer, update_rms_opts: list[object], interval: int) -> None:
    orig_step = optimizer.step
    state = {"n": 0}

    def _wrapped_step(*args, **kwargs):
        ret = orig_step(*args, **kwargs)
        step = state["n"]
        if interval > 0 and step % interval == 0:
            try:
                _log_update_rms(update_rms_opts, step)
            except Exception:
                logger.exception("[POET/update-rms] logging failed at step %d", step)
        state["n"] = step + 1
        return ret

    optimizer.step = _wrapped_step


def _resolve_interval(default: int = 100) -> int:
    env = os.environ.get("SLM_POET_UPDATE_RMS_LOG_INTERVAL")
    if env is not None:
        return max(1, int(env))
    try:
        from megatron.training import get_args

        opts = get_args()
        return max(
            1,
            int(
                getattr(opts, "tensorboard_log_interval", None)
                or getattr(opts, "log_interval", None)
                or default
            ),
        )
    except Exception:
        return default


def _install_on_setup(orig_setup, interval: int | None = None):
    def _wrapped_setup(*args, **kwargs):
        model, optimizer, opt_param_scheduler = orig_setup(*args, **kwargs)
        update_rms_opts = _update_rms_optimizers(optimizer)
        if update_rms_opts:
            log_interval = _resolve_interval() if interval is None else max(1, int(interval))
            _install_step_hook(optimizer, update_rms_opts, log_interval)
            logger.warning(
                "[POET/update-rms] W&B diagnostics enabled for %d optimizer(s), interval=%d",
                len(update_rms_opts),
                log_interval,
            )
        return model, optimizer, opt_param_scheduler

    return _wrapped_setup


@register_patch(name="poet_update_rms_log", targets=())
def apply() -> None:
    from megatron.training import training as _mt

    _mt.setup_model_and_optimizer = _install_on_setup(_mt.setup_model_and_optimizer)
