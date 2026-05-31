"""Patch torchtitan's per-step console log to match the Megatron path.

Two changes to the line ``MetricsProcessor.log`` emits each ``metrics.log_freq``
steps (``step: N  loss: ...  grad_norm: ...  tps: ...  tflops: ...  mfu: ...``):

1. **Emit it only on rank 0.** torchtitan's ``init_logger`` installs a stdout
   handler on *every* rank with no gating, so the line is duplicated once per
   rank. Megatron prints the per-iteration line via ``print_rank_last`` (one
   rank only); this brings torchtitan in line.
2. **Append an ETA segment** (``ETA: 1h30m``) computed from the configured total
   step count and the just-elapsed wall time — the same ``H h MM m`` format the
   Megatron ``training_log_eta`` patch injects (``src/patches/training_log_eta.py``).

Implementation: a runtime monkeypatch applied when ``src.titan_ext`` is imported
via torchtitan's ``experimental.custom_import`` hook — the vendored submodule is
never edited. ``MetricsProcessor.log`` references the module-global ``logger``
(``from torchtitan.tools.logging import logger``); we wrap the method to compute
the ETA, temporarily rebind that module global to a proxy that gates ``.info`` to
rank 0 and appends the ETA, then restore it. All upstream metric math, wandb /
tensorboard logging (``self.logger.log``), and per-step state resets run unchanged.

Import-safe on CPU (no torchtitan import at module load); the patch no-ops if
torchtitan is absent or already patched.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_WRAP_FLAG = "_slm_eta_rank0_wrapped"


def _rank() -> int:
    """Current global rank: prefer the initialized process group, else the
    ``RANK`` env torchrun sets, else 0."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return int(os.environ.get("RANK", "0") or 0)


def format_eta(remaining_steps: int, sec_per_step: float) -> str:
    """``H h MM m`` remaining-time string, e.g. ``1h30m`` (matches Megatron)."""
    eta = max(0.0, float(remaining_steps) * float(sec_per_step))
    h = int(eta // 3600)
    mm = int((eta % 3600) // 60)
    return f"{h}h{mm:02d}m"


class _RankZeroEtaLogger:
    """Proxy around torchtitan's module logger for the duration of one ``log``
    call: suppresses ``.info`` off rank 0 and appends ``  ETA: <eta>`` to the
    (single) per-step line. Everything else forwards to the real logger."""

    def __init__(self, real, eta_str: str, is_rank0: bool):
        self._real = real
        self._eta_str = eta_str
        self._is_rank0 = is_rank0

    def info(self, msg, *args, **kwargs):
        if not self._is_rank0:
            return
        if self._eta_str and isinstance(msg, str):
            msg = f"{msg}  ETA: {self._eta_str}"
        return self._real.info(msg, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def apply_titan_metrics_patch() -> bool:
    """Monkeypatch ``MetricsProcessor.log`` for rank-0-only output + ETA.

    Returns True if the patch is in place (or was already), False if torchtitan
    is not importable (e.g. a CPU unit-test environment).
    """
    try:
        import time

        import torchtitan.components.metrics as _m
    except Exception:
        return False

    if getattr(_m.MetricsProcessor.log, _WRAP_FLAG, False):
        return True

    _orig_log = _m.MetricsProcessor.log

    def _wrapped(self, step, *args, **kwargs):
        is_rank0 = _rank() == 0
        # Per-step wall time over the last log_freq steps (mirrors upstream's
        # time_end_to_end). Guard everything: logging must never crash training.
        eta_str = ""
        try:
            log_freq = max(1, int(self.job_config.metrics.log_freq))
            total_steps = int(self.job_config.training.steps)
            time_delta = time.perf_counter() - self.time_last_log
            sec_per_step = time_delta / log_freq
            remaining = max(0, total_steps - int(step))
            eta_str = format_eta(remaining, sec_per_step)
        except Exception:
            eta_str = ""

        real_logger = _m.logger
        _m.logger = _RankZeroEtaLogger(real_logger, eta_str, is_rank0)
        try:
            return _orig_log(self, step, *args, **kwargs)
        finally:
            _m.logger = real_logger

    setattr(_wrapped, _WRAP_FLAG, True)
    _m.MetricsProcessor.log = _wrapped
    logger.info("[titan_ext] patched MetricsProcessor.log (rank-0-only + ETA)")
    return True
