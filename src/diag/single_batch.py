# src/diag/single_batch.py
"""Single-batch-overfit helper (Probe 0A): cache the first batch, replay forever."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class BatchReplay:
    """Caches the first value produced and returns it on every later call.

    The wrapped producer (e.g. Megatron's ``get_batch``) is invoked exactly
    once; thereafter the cached value is returned without advancing it, so the
    model sees one fixed minibatch every step.
    """

    def __init__(self) -> None:
        self._cached: Any = None
        self._have: bool = False
        self.calls: int = 0
        self.producer_calls: int = 0

    def __call__(self, producer: Callable[[], Any]) -> Any:
        self.calls += 1
        if not self._have:
            self._cached = producer()
            self.producer_calls += 1
            self._have = True
        return self._cached
