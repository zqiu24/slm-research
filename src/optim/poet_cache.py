"""POET Cayley cache — module-level state, registry, and CachedPOETLinear.

See docs/superpowers/specs/2026-05-23-poet-cayley-cache-design.md for the
design. v1 supports float POETLinear only; QPOETLinear /
POETLinearNeurips / POETCayleyLinear are deferred to v2.

Module-level state (single source of truth):
- _POET_CACHE_MODE — set once at startup by POETAdam.__init__.
- _POET_VERSION    — monotonic counter; bumped by POETAdam.step() and
                     by POETAdam.load_state_dict() on resume.
- _POET_LAYER_REGISTRY — weakref list of every live CachedPOETLinear,
                     populated by replace_linears_with_poet.

All access goes through the helpers in this module so tests can drive
the state without touching globals directly.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator
from typing import Literal

logger = logging.getLogger(__name__)

CacheMode = Literal["none", "cached_fwd", "cached_fwd_bwd"]
_VALID_MODES: tuple[CacheMode, ...] = ("none", "cached_fwd", "cached_fwd_bwd")

_POET_CACHE_MODE: CacheMode = "none"
_POET_VERSION: int = 0
_POET_LAYER_REGISTRY: list[weakref.ReferenceType] = []


def get_cache_mode() -> CacheMode:
    return _POET_CACHE_MODE


def set_cache_mode(mode: str) -> None:
    global _POET_CACHE_MODE
    if mode not in _VALID_MODES:
        raise ValueError(f"poet_cache_mode must be one of {_VALID_MODES}, got {mode!r}")
    _POET_CACHE_MODE = mode  # type: ignore[assignment]
    logger.info("[POET cache] mode set to %s", mode)


def get_poet_version() -> int:
    return _POET_VERSION


def bump_poet_version() -> None:
    global _POET_VERSION
    _POET_VERSION += 1


def register_poet_layer(layer) -> None:
    _POET_LAYER_REGISTRY.append(weakref.ref(layer))


def iter_live_layers() -> Iterator:
    """Yield every still-live layer, pruning dead weakrefs lazily."""
    alive: list[weakref.ReferenceType] = []
    for ref in _POET_LAYER_REGISTRY:
        layer = ref()
        if layer is not None:
            alive.append(ref)
            yield layer
    _POET_LAYER_REGISTRY[:] = alive


def reset_for_testing() -> None:
    """Reset module state. Tests only — never call from prod code."""
    global _POET_CACHE_MODE, _POET_VERSION
    _POET_CACHE_MODE = "none"
    _POET_VERSION = 0
    _POET_LAYER_REGISTRY.clear()


import torch  # noqa: E402
from poet_torch import POETLinear  # noqa: E402
from poet_torch.poet_layer import pytorch_skew_symmetric  # noqa: E402
from torch import Tensor  # noqa: E402


def _compute_cayley(
    oft_R: Tensor,  # noqa: N803
    block_size: int,
    rows: Tensor,
    cols: Tensor,
    r_in: int,
    r_out: int,
) -> tuple[Tensor, Tensor]:
    """Build (R_out, R_in) block-orthogonal matrices from oft_R.

    Mirrors get_weight_poet in third_party/poet_torch/poet_layer.py.
    `torch.ops.poet.cayley` is a GPU Triton kernel; this function is
    GPU-only at runtime.

    Spec §5 lists this as a "compiled region"; v1 ships it as plain
    Python — see Task 3 design note for the rationale.
    """
    Q_skew = pytorch_skew_symmetric(oft_R, block_size, rows, cols)  # noqa: N806
    R_cat = torch.ops.poet.cayley(Q_skew)[0]  # noqa: N806
    R_out, R_in = R_cat.split([r_out, r_in], dim=0)  # noqa: N806
    return R_out, R_in


class CachedPOETLinear(POETLinear):
    """POETLinear subclass that supports Cayley-Neumann caching.

    Cache slots are mutable Python attributes that live OUTSIDE any
    torch.compile region. The mode-specific forward dispatch reads
    `_POET_CACHE_MODE` once per call.

    `_R_cache_version` is compared against `_POET_VERSION`: a mismatch
    means the cached R blocks were built under a stale oft_R and must
    be recomputed before reuse.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._R_cache_version: int = -1
        self._R_out_leaf: Tensor | None = None  # detached leaf, requires_grad=True
        self._R_in_leaf: Tensor | None = None
        self._R_out_full: Tensor | None = None  # mode A only: tensor in cayley graph
        self._R_in_full: Tensor | None = None  # mode A only

    def _invalidate_R_cache(self) -> None:  # noqa: N802
        """Drop all cached R blocks. Next forward will recompute."""
        self._R_cache_version = -1
        self._R_out_leaf = None
        self._R_in_leaf = None
        self._R_out_full = None
        self._R_in_full = None

    def forward(self, x: Tensor) -> Tensor:
        mode = get_cache_mode()
        if mode == "none":
            return super().forward(x)
        # Mode-specific paths added in Tasks 4 and 5.
        raise NotImplementedError(f"cache mode {mode!r} not yet implemented")


def invalidate_all_poet_caches() -> None:
    """Force every live POET layer to recompute on next forward.

    Use from checkpoint-load paths and as a manual debug knob. The
    optimizer step uses `bump_poet_version()` instead — lazy invalidation
    via version mismatch is cheaper than walking the registry.
    """
    for layer in iter_live_layers():
        layer._invalidate_R_cache()
