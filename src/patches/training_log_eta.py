"""Patch: prepend ``ETA: 1h30m`` to the per-iteration training log.

Ported from fork 2's training.py customisation (commit bb43fa063).
Targets ``megatron.training.training.training_log``.

The upstream ``training_log`` builds the log string with f-strings, so a
post-hoc replacement on the formatted record is the cleanest path: we
register a logging.Filter that rewrites the per-step log record after
``_orig`` produces it.
"""

from __future__ import annotations

import logging
import re

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.training_log",)
log = logging.getLogger(__name__)

_ITER_RE = re.compile(r"iteration\s+(\d+)/(\d+)")
_ELAPSED_RE = re.compile(r"elapsed time per iteration \(ms\): ([\d.]+)")


class _ETAFilter(logging.Filter):
    """Injects ``ETA: HhMMm`` after ``iteration X/Y |`` in the formatted record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "iteration" not in msg or "/" not in msg or "elapsed time per iteration" not in msg:
            return True
        try:
            m = _ITER_RE.search(msg)
            e = _ELAPSED_RE.search(msg)
            if not m or not e:
                return True
            curr, total = int(m.group(1)), int(m.group(2))
            sec = float(e.group(1)) / 1000.0
            eta = max(0.0, (total - curr) * sec)
            h, m_ = int(eta // 3600), int((eta % 3600) // 60)
            new_msg = msg.replace(
                f"iteration {m.group(1)}/{m.group(2)} |",
                f"iteration {m.group(1)}/{m.group(2)} | ETA: {h}h{m_:02d}m |",
                1,
            )
            record.msg = new_msg
            record.args = ()
        except Exception:
            # The filter must never break logging.
            pass
        return True


@register_patch(name="training_log_eta", targets=_TARGET)
def apply() -> None:
    """Install the ETA filter on Megatron's training logger."""
    from megatron.training import training as _mt  # noqa: F401  (ensures target exists)

    mt_logger = logging.getLogger("megatron.training.training")
    if not any(isinstance(f, _ETAFilter) for f in mt_logger.filters):
        mt_logger.addFilter(_ETAFilter())
