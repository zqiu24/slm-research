"""Pure parameter-counting helpers (no torch.distributed, no Megatron).

``count_local_params`` returns the (trainable, total, poet) parameter counts for
a list of model chunks on the *local* rank. Global aggregation across
model-parallel ranks is the caller's job (see
``src/patches/wandb_trainable_params.py``); kept separate so the arithmetic is
CPU-unit-testable in isolation.
"""

from __future__ import annotations

from collections.abc import Iterable

# Parameter-name substring identifying POET's trainable orthogonal generators.
# Catches the decoupled ``oft_R_in`` / ``oft_R_out`` and the legacy single
# ``oft_R`` (same detector as the POET parameter dump in poet_apply_to_model).
_POET_PARAM_MARKER = "oft_R"


def count_local_params(model_chunks: Iterable) -> tuple[int, int, int]:
    """Return ``(trainable, total, poet)`` local param counts over ``model_chunks``.

    ``trainable`` counts params with ``requires_grad=True`` (e.g. POET's
    ``oft_R``); ``total`` counts every ``nn.Parameter`` (including frozen base
    weights, which remain Parameters with ``requires_grad=False``); ``poet``
    counts params whose name contains ``oft_R`` — by name, independent of
    ``requires_grad`` — and is ``0`` for any model without POET layers (plain
    adam / muon / ngpt), which is the desired "normally 0" behaviour.
    """
    trainable = 0
    total = 0
    poet = 0
    for chunk in model_chunks:
        for name, p in chunk.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
            if _POET_PARAM_MARKER in name:
                poet += n
    return trainable, total, poet
