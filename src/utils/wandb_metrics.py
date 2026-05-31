"""Canonical W&B metric schema + per-backend key normalization.

Single source of truth for the metric KEY names both training backends
("megatron", "torchtitan") emit to Weights & Biases, so one dashboard overlays
comparable curves. Pure module (stdlib only) — importable in any CPU/unit-test
env; no torch / wandb / megatron / torchtitan dependency.

Both backends are normalized at their wandb-log boundary and call
`normalize(metrics, backend)`:
  * megatron   -> src/patches/wandb_metric_normalize.py (wraps wandb.log)
  * torchtitan -> src/titan_ext/metrics.py (wraps WandBLogger.log)

See docs/superpowers/specs/2026-05-31-unified-wandb-logging-design.md.
"""

from __future__ import annotations

import math

# The cross-backend comparison set: keys both backends converge on. (perf/* and
# train/tokens_seen are *computed* in the megatron interceptor, not renamed.)
CORE_CANONICAL = frozenset(
    {
        "train/loss",
        "train/lr",
        "train/grad_norm",
        "train/tokens_seen",
        "perf/step_time_s",
        "val/loss",
    }
)

# Megatron native key -> canonical. Deliberately EXCLUDES:
#   * "throughput"      — Megatron's is TFLOP/s/GPU; throughput is not normalized
#                         across backends (torchtitan's tps uses a different
#                         normalization), so it stays native on both sides.
#   * "iteration-time"  — gated off in our runs (no --log-timers-to-tensorboard);
#                         step time is COMPUTED in the interceptor instead.
# Unlisted keys pass through unchanged.
MEGATRON_TO_CANONICAL = {
    "lm loss": "train/loss",
    "learning-rate": "train/lr",
    "grad-norm": "train/grad_norm",
    "tokens seen": "train/tokens_seen",  # robustness if the legacy patch is enabled
}

# Torchtitan native key -> canonical. Deliberately EXCLUDES "throughput(tps)":
# it is tokens/sec normalized by non_data_parallel_size (a per-model-parallel-
# group rate), not comparable to a global-aggregate tokens/sec, so it stays
# native (like Megatron's TFLOP/s "throughput"). Other unlisted keys (mfu(%),
# tflops, memory/*, time_metrics/data_loading*, validation_metrics/*) pass through.
TITAN_TO_CANONICAL = {
    "loss_metrics/global_avg_loss": "train/loss",
    "loss_metrics/global_max_loss": "train/loss_max",
    "lr": "train/lr",
    "grad_norm": "train/grad_norm",
    "n_tokens_seen": "train/tokens_seen",
    "time_metrics/end_to_end(s)": "perf/step_time_s",
    "validation_metrics/loss": "val/loss",
}


def _canonical_key(key: str, backend: str) -> str:
    if backend == "megatron":
        if key in MEGATRON_TO_CANONICAL:
            return MEGATRON_TO_CANONICAL[key]
        # Megatron logs validation loss as "lm loss validation" (key + suffix).
        if key.startswith("lm loss") and "validation" in key:
            return "val/loss"
        return key
    if backend == "torchtitan":
        return TITAN_TO_CANONICAL.get(key, key)
    return key  # unknown backend: pass through unchanged


def normalize(metrics: dict | None, backend: str) -> dict:
    """Return a new dict with core overlapping keys renamed to the canonical
    schema; unknown / backend-specific keys pass through unchanged.

    Pure and idempotent on already-canonical input. `backend` is "megatron" or
    "torchtitan"; any other value is a pass-through no-op.
    """
    return {_canonical_key(k, backend): v for k, v in (metrics or {}).items()}


def with_derived(metrics: dict | None) -> dict:
    """Add derived canonical metrics to an already-normalized dict.

    Currently derives ``val/ppl = exp(min(20, val/loss))`` (perplexity) whenever
    ``val/loss`` is present — mirroring Megatron's own clamp
    (``training.py`` ``math.exp(min(20, loss))``). Megatron logs validation PPL
    only to TensorBoard and torchtitan doesn't emit it at all, so deriving it here
    gives both backends a perplexity curve from the canonical ``val/loss``.

    Pure; idempotent (skips if ``val/ppl`` already set); no-op when ``val/loss``
    is absent. Call AFTER :func:`normalize`.
    """
    metrics = dict(metrics or {})
    if "val/loss" in metrics and "val/ppl" not in metrics:
        metrics["val/ppl"] = math.exp(min(20.0, float(metrics["val/loss"])))
    return metrics
