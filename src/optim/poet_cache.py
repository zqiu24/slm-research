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

CacheMode = Literal["none", "cached_fwd_bwd"]
_VALID_MODES: tuple[CacheMode, ...] = ("none", "cached_fwd_bwd")

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
from poet_torch.poet_layer import (  # noqa: E402
    chain_layer_x_checkpoint_mem_o2,
    pytorch_skew_symmetric,
)
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

    def _get_R_blocks_mode_a(self) -> tuple[Tensor, Tensor]:  # noqa: N802
        """Mode A: cache (R_out_full, R_in_full) WITH the cayley autograd
        graph, plus detached leaves that micro-batch backwards write into.

        The leaves' `.grad` accumulates naturally across micro-batches.
        At end-of-cycle, `_flush_R_grads_to_oft_R` runs one manual VJP
        through `_R_*_full` to push the summed gradient back to `oft_R`.

        We pre-allocate the leaves' `.grad` as fp32 even when the leaves
        themselves are bf16, so autograd's in-place accumulation casts
        each bf16 contribution up to fp32 on the way in. This mirrors
        Megatron's `param.main_grad` pattern: bf16 per-element gradients
        accumulated K times in a bf16 buffer lose precision rapidly,
        but accumulated in an fp32 buffer they stay at the fp32 floor.
        """
        if self._R_cache_version != get_poet_version():
            with torch.enable_grad():
                R_out_full, R_in_full = _compute_cayley(  # noqa: N806
                    self.oft_R,
                    self.block_size,
                    self.rows,
                    self.cols,
                    self.r_in,
                    self.r_out,
                )
            self._R_out_full = R_out_full
            self._R_in_full = R_in_full
            self._R_out_leaf = R_out_full.detach().requires_grad_(True)
            self._R_in_leaf = R_in_full.detach().requires_grad_(True)
            self._R_cache_version = get_poet_version()
        return self._R_out_leaf, self._R_in_leaf

    def _flush_R_grads_to_oft_R(self) -> None:  # noqa: N802
        """Push accumulated R-leaf gradients back to oft_R (mode A only).

        Writes to `oft_R.main_grad` when Megatron's grad buffer exists
        (production training); falls back to `oft_R.grad` otherwise (unit
        tests). The outer optimizer's `prepare_grads` reads `main_grad`,
        not `.grad`, so writing there is what makes the update actually
        reach base_adam — see plan "Optimizer integration timing".

        No-op when no forward happened this cycle.
        """
        if self._R_out_full is None or self._R_in_full is None:
            return
        gR_out = self._R_out_leaf.grad  # noqa: N806
        gR_in = self._R_in_leaf.grad  # noqa: N806
        if gR_out is None and gR_in is None:
            return
        if gR_out is None:
            gR_out = torch.zeros_like(self._R_out_full)  # noqa: N806
        if gR_in is None:
            gR_in = torch.zeros_like(self._R_in_full)  # noqa: N806
        (g,) = torch.autograd.grad(
            outputs=[self._R_out_full, self._R_in_full],
            inputs=self.oft_R,
            grad_outputs=[gR_out, gR_in],
        )
        if hasattr(self.oft_R, "main_grad") and self.oft_R.main_grad is not None:
            # Megatron path: zero-initialized FP32 buffer; copy our VJP in.
            self.oft_R.main_grad.copy_(g.to(self.oft_R.main_grad.dtype))
        else:
            # Test / non-Megatron path.
            if self.oft_R.grad is None:
                self.oft_R.grad = g
            else:
                self.oft_R.grad.copy_(g)
        self._invalidate_R_cache()

    def forward(self, x: Tensor) -> Tensor:
        mode = get_cache_mode()
        if mode == "none":
            return super().forward(x)
        if mode != "cached_fwd_bwd":
            raise ValueError(f"unknown poet_cache_mode: {mode!r}")
        R_out, R_in = self._get_R_blocks_mode_a()  # noqa: N806
        return _cached_chain_layer_core(
            x,
            R_in,
            self.weight,
            self.bias,
            R_out,
            self.perm_in_inv,
            self.perm_in,
            self.perm_out,
            self.perm_out_inv,
            self.block_size,
        )


@torch.compile(fullgraph=True)
def _cached_chain_layer_core(
    x: Tensor,
    R_in: Tensor,  # noqa: N803
    weight: Tensor,
    bias: Tensor,
    R_out: Tensor,  # noqa: N803
    perm_in_inv: Tensor,
    perm_in: Tensor,
    perm_out: Tensor,
    perm_out_inv: Tensor,
    block_size: int,
) -> Tensor:
    """Mode A hot path mirror of upstream `forward_core`.

    Upstream's `forward_core` is wrapped in `@torch.compile(fullgraph=True)`
    and fuses Cayley + chain_layer in one compiled region. Mode A skips
    the Cayley step (it's cached and supplied via R_in/R_out), so we wrap
    only the chain_layer call here with the same decorator. Without this,
    every microbatch's linear call runs uncompiled and the per-call
    overhead drowns the (K-1) Cayley savings for large shapes.
    """
    return chain_layer_x_checkpoint_mem_o2(
        x,
        R_in,
        weight,
        bias,
        R_out,
        perm_in_inv,
        perm_in,
        perm_out,
        perm_out_inv,
        block_size,
    )


def invalidate_all_poet_caches() -> None:
    """Force every live POET layer to recompute on next forward.

    Use from checkpoint-load paths and as a manual debug knob. The
    optimizer step uses `bump_poet_version()` instead — lazy invalidation
    via version mismatch is cheaper than walking the registry.
    """
    for layer in iter_live_layers():
        layer._invalidate_R_cache()
