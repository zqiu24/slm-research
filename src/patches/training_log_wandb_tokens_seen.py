"""Patch: log ``tokens seen`` (consumed_train_samples * seq_length) to wandb.

Ported from fork 1's working-tree edit to ``megatron/training/training.py``.
Targets ``megatron.training.training.training_log``.

This patch *composes onto* whatever already wraps ``training_log`` —
notably ``training_log_eta`` (the ETA injection patch). To avoid a
``PatchConflict`` on overlapping ``targets``, we register with
``targets=()`` and document the implicit dependency in this docstring.

Apply order is sorted-by-name, so ``training_log_eta`` (e) applies first
and ``training_log_wandb_tokens_seen`` (w) wraps ``training_log``. The two
are orthogonal: ``training_log_eta`` wraps ``print_rank_last`` (the stdout
emitter called *inside* ``training_log``), so both run independently.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)


@register_patch(name="training_log_wandb_tokens_seen", targets=())
def apply() -> None:
    from megatron.training import get_args
    from megatron.training import training as _mt

    _orig = _mt.training_log

    def _wrapped(*args, **kwargs):
        ret = _orig(*args, **kwargs)
        try:
            opts = get_args()
            iteration = kwargs.get("iteration")
            if iteration is None and len(args) >= 5:
                iteration = args[4]
            if iteration is None:
                return ret
            log_interval = getattr(opts, "tensorboard_log_interval", None) or getattr(
                opts, "log_interval", 0
            )
            if not log_interval or iteration % log_interval != 0:
                return ret
            import wandb

            if wandb.run is None:
                return ret
            tokens = int(getattr(opts, "consumed_train_samples", 0)) * int(
                getattr(opts, "seq_length", 0)
            )
            wandb.log({"tokens seen": tokens}, step=int(iteration))
        except Exception:
            # Logging extras must never crash training.
            pass
        return ret

    _mt.training_log = _wrapped
