# src/patches/overfit_single_batch.py
"""Patch (Probe 0A): single-batch overfit by replaying the first get_batch.

Env-gated: only active when SLM_OVERFIT_SINGLE_BATCH=1. Inert otherwise, so it
is safe in _ALWAYS_ON_PATCHES and never perturbs a normal run. Targets
``pretrain_gpt.get_batch`` (free; train_step is owned by poet_merge_step).
"""

from __future__ import annotations

import logging
import os

from src.patches._registry import register_patch

_TARGET = ("pretrain_gpt.get_batch",)
logger = logging.getLogger(__name__)


@register_patch(name="overfit_single_batch", targets=_TARGET)
def apply() -> None:
    if os.environ.get("SLM_OVERFIT_SINGLE_BATCH") != "1":
        return  # inert unless explicitly enabled

    import pretrain_gpt as mg

    from src.diag.single_batch import BatchReplay

    _orig_get_batch = mg.get_batch
    _replay = BatchReplay()

    def _wrapped(*args, **kwargs):
        return _replay(lambda: _orig_get_batch(*args, **kwargs))

    mg.get_batch = _wrapped
    logger.warning(
        "[OVERFIT] single-batch overfit ENABLED — replaying the first get_batch every step"
    )
