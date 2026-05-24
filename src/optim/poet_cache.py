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
