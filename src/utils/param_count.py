"""Pure parameter-counting helpers (no torch.distributed, no Megatron).

``count_local_params`` returns the (trainable, total) parameter counts for a
list of model chunks on the *local* rank. Global aggregation across
model-parallel ranks is the caller's job (see
``src/patches/wandb_trainable_params.py``); kept separate so the arithmetic is
CPU-unit-testable in isolation.
"""

from __future__ import annotations

from collections.abc import Iterable


def count_local_params(model_chunks: Iterable) -> tuple[int, int]:
    """Return ``(trainable, total)`` local param counts over ``model_chunks``.

    ``trainable`` counts params with ``requires_grad=True`` (e.g. POET's
    ``oft_R``); ``total`` counts every ``nn.Parameter`` (including frozen base
    weights, which remain Parameters with ``requires_grad=False``).
    """
    trainable = 0
    total = 0
    for chunk in model_chunks:
        for p in chunk.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
    return trainable, total
