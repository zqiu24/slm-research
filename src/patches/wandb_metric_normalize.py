"""Patch: normalize Megatron's W&B metric KEYS to the canonical schema, and add
the canonical metrics Megatron doesn't emit to W&B in our config.

Two orthogonal changes, both guarded so logging never crashes training:

1. **Rename keys.** Megatron's W&B "writer" IS the ``wandb`` module itself
   (``megatron.training.global_vars._GLOBAL_WANDB_WRITER = wandb``), so every
   ``wandb_writer.log({...}, it)`` is ``wandb.log(...)``. We wrap ``wandb.log`` to
   run ``src.utils.wandb_metrics.normalize(d, "megatron")`` on the dict first
   (``lm loss`` -> ``train/loss``, ``learning-rate`` -> ``train/lr``, ...).
   Unmapped keys (grad-norm-clipped, params-norm, ...) pass through.

2. **Add computed metrics.** Our runs don't set --log-timers-to-tensorboard, so
   Megatron logs neither iteration-time nor throughput to W&B. We wrap
   ``training_log`` to additively emit ``train/tokens_seen`` (from
   ``consumed_train_samples * seq_length``) and ``perf/step_time_s`` (a
   perf-counter window), on the W&B-logging rank only, gated on the SAME interval
   as the native metrics (``tensorboard_log_interval`` or ``log_interval``).
   Throughput is intentionally NOT emitted: Megatron's would be a global-aggregate
   tokens/sec while torchtitan's ``throughput(tps)`` is normalized by
   ``non_data_parallel_size`` — not comparable, so each backend keeps its native
   throughput as a passthrough extra.

Registered with ``targets=()`` so it composes with ``log_grad_norm_extra``
(which owns ``training.training_log``). Apply order is sorted-by-name:
``log_grad_norm_extra`` (l) wraps first, ``wandb_metric_normalize`` (w) wraps the
result. Module import is CPU-safe (megatron / wandb imported only inside
``apply()``).
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)


def _extra_metrics(consumed_samples, seq_length, iteration, now, last):
    """Canonical metrics Megatron does not natively log to W&B in our config.

    Returns ``(metrics, new_state)``. ``last`` is the previous emit's state dict
    (or None on the first emit). The first emit carries only the cumulative
    ``train/tokens_seen``; later emits add the per-window ``perf/step_time_s``.
    Throughput is intentionally omitted (see module docstring).
    """
    tokens = int(consumed_samples) * int(seq_length)
    metrics = {"train/tokens_seen": tokens}
    if last is not None:
        dt = float(now) - float(last["time"])
        steps = max(1, int(iteration) - int(last["iter"]))
        if dt > 0:
            metrics["perf/step_time_s"] = dt / steps
    new_state = {"time": float(now), "tokens": tokens, "iter": int(iteration)}
    return metrics, new_state


def _wrap_wandb_log(orig_log):
    """Return a ``wandb.log`` wrapper that canonicalizes Megatron metric keys."""
    from src.utils.wandb_metrics import normalize

    def _log(data, *args, **kwargs):
        try:
            if isinstance(data, dict):
                data = normalize(data, "megatron")
        except Exception:  # logging must never crash training
            pass
        return orig_log(data, *args, **kwargs)

    _log._slm_wandb_normalize = True
    return _log


@register_patch(name="wandb_metric_normalize", targets=())
def apply() -> None:
    import time

    import wandb
    from megatron.training import get_args
    from megatron.training import training as _mt

    # (1) Rename keys at the wandb.log boundary (idempotent guard).
    if not getattr(wandb.log, "_slm_wandb_normalize", False):
        wandb.log = _wrap_wandb_log(wandb.log)

    # (2) Add computed canonical metrics from training_log (idempotent guard).
    _orig = _mt.training_log
    if getattr(_orig, "_slm_wandb_extra", False):
        return
    state = {"last": None}

    def _wrapped(*args, **kwargs):
        ret = _orig(*args, **kwargs)
        try:
            # get_wandb_writer() is None on non-logging ranks -> nothing to do.
            if _mt.get_wandb_writer() is not None:
                opts = get_args()
                iteration = kwargs.get("iteration")
                if iteration is None and len(args) > 3:
                    iteration = args[3]  # training_log(loss_dict, total, lr, iteration, ...)
                # Match the native-metric cadence: training_log gates its wandb
                # block on tensorboard_log_interval (training.py:2052). Mirror the
                # tokens_seen patch's gate so computed metrics aren't 10x sparser.
                interval = int(
                    getattr(opts, "tensorboard_log_interval", None)
                    or getattr(opts, "log_interval", 0)
                    or 0
                )
                if iteration is not None and interval and int(iteration) % interval == 0:
                    metrics, state["last"] = _extra_metrics(
                        getattr(opts, "consumed_train_samples", 0),
                        getattr(opts, "seq_length", 0),
                        int(iteration),
                        time.perf_counter(),
                        state["last"],
                    )
                    # wandb.log is the wrapped (renaming) one; canonical keys pass
                    # through normalize() idempotently.
                    wandb.log(metrics, int(iteration))
        except Exception:  # logging must never crash training
            pass
        return ret

    _wrapped._slm_wandb_extra = True
    _mt.training_log = _wrapped
