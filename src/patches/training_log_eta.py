"""Patch: rewrite the per-iteration training log line.

Two changes to the stdout line Megatron emits each ``log_interval``:

1. Inject ``ETA: 1h30m`` right after ``iteration X/Y |``.
2. Strip fields that are noise on the console because they either never
   change or are already in wandb/tensorboard: ``learning rate``,
   ``global batch size``, ``loss scale``, ``grad norm``. (learning rate
   and grad norm are logged to wandb in ``training_log`` itself, so
   dropping them here loses nothing.)

The per-iteration line is built with f-strings and emitted via
``megatron.training.training.print_rank_last`` (a plain ``print``), not
through the logging module -- so we wrap ``print_rank_last`` and rewrite
its argument. (An earlier version registered a ``logging.Filter``, which
never fired because the line is printed, not logged.)
"""

from __future__ import annotations

import contextlib
import re

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.print_rank_last",)

# ``/\s*`` tolerates Megatron's right-justified ``iteration {:8d}/{:8d}``
# padding, e.g. ``iteration       50/   45776 |``.
_ITER_RE = re.compile(r"iteration\s+(\d+)/\s*(\d+)\s*\|")
_ELAPSED_RE = re.compile(r"elapsed time per iteration \(ms\): ([\d.]+)")

# Each matches one `` <label>: <value> |`` field for removal.
_STRIP_RES = (
    re.compile(r"\s*learning rate:\s*[\d.eE+\-]+\s*\|"),
    re.compile(r"\s*global batch size:\s*\d+\s*\|"),
    re.compile(r"\s*loss scale:\s*[\d.eE+\-]+\s*\|"),
    re.compile(r"\s*grad norm:\s*[\d.eE+\-]+\s*\|"),
)


def _rewrite(msg: str) -> str:
    """Inject ETA and strip noise fields from a per-iteration log line."""
    if "iteration" not in msg or "elapsed time per iteration" not in msg:
        return msg
    m = _ITER_RE.search(msg)
    e = _ELAPSED_RE.search(msg)
    if m and e:
        curr, total = int(m.group(1)), int(m.group(2))
        sec = float(e.group(1)) / 1000.0
        eta = max(0.0, (total - curr) * sec)
        h, mm = int(eta // 3600), int((eta % 3600) // 60)
        msg = f"{msg[:m.end()]} ETA: {h}h{mm:02d}m |{msg[m.end():]}"
    for pat in _STRIP_RES:
        msg = pat.sub("", msg)
    return msg


@register_patch(name="training_log_eta", targets=_TARGET)
def apply() -> None:
    """Wrap ``print_rank_last`` to rewrite the per-iteration log line."""
    from megatron.training import training as _mt

    _orig = _mt.print_rank_last
    if getattr(_orig, "_slm_eta_wrapped", False):
        return

    def _wrapped(message):
        if isinstance(message, str):
            # Rewriting must never break logging.
            with contextlib.suppress(Exception):
                message = _rewrite(message)
        return _orig(message)

    _wrapped._slm_eta_wrapped = True
    _mt.print_rank_last = _wrapped
