# POET Cayley-Neumann Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache `R_out, R_in` for one gradient-accumulation cycle so the Cayley computation runs once per cycle instead of K times. Two opt-in modes: `cached_fwd` (saves K→1 cayley forwards) and `cached_fwd_bwd` (also saves K→1 cayley backwards). Default `none` is bit-for-bit identical to today.

**Architecture:** New module [src/optim/poet_cache.py](../../../src/optim/poet_cache.py) holds a `CachedPOETLinear` subclass of upstream `POETLinear`, three module-level state objects (`_POET_CACHE_MODE`, `_POET_VERSION`, `_POET_LAYER_REGISTRY`), and per-mode forward dispatch. Mode A's end-of-cycle flush writes to `oft_R.main_grad` (Megatron's FP32 grad buffer) via a `step` wrapper installed on the outer optimizer in [src/optim/poet.py](../../../src/optim/poet.py) — this is the load-bearing fix from the v1 plan review; see "Optimizer integration timing" below. Layer construction, cache invalidation, and the merge-step hook are threaded through existing slm-research patches ([src/optim/poet_layers.py](../../../src/optim/poet_layers.py), [src/patches/poet_merge_step.py](../../../src/patches/poet_merge_step.py)). Upstream `third_party/poet_torch` is **not** modified.

**Tech Stack:** PyTorch (custom `autograd.Function`, weakref registries), torch.distributed (per-DP-group `all_reduce` on `oft_R.main_grad`), Megatron-LM hooks via slm-research's patch system.

**Spec reference:** [2026-05-23-poet-cayley-cache-design.md](../specs/2026-05-23-poet-cayley-cache-design.md). Read it before starting. Stay scoped to this plan; do not open sibling plans.

---

## Optimizer integration timing (the critical correction from v1 review)

Megatron's [Float16OptimizerWithFloat16Params](../../../third_party/Megatron-LM/megatron/core/optimizer/optimizer.py#L755) and [FP32Optimizer](../../../third_party/Megatron-LM/megatron/core/optimizer/optimizer.py#L922) both call `prepare_grads()` *before* invoking the inner optimizer's `step()`. `prepare_grads()` copies `param.main_grad` → `main_param.grad`. By the time `POETAdam.step()` runs, the gradient base_adam will use has already been decided.

Mode A's per-microbatch backward writes to **R-leaf `.grad`** only — autograd never flows to `oft_R`, so:
- `oft_R.grad` is never populated by autograd.
- Megatron's DDP grad reducer (which runs during backward via hooks) sees no gradient on `oft_R` → `oft_R.main_grad` stays zero across all ranks.
- A flush hook *inside* `POETAdam.step()` that writes to `oft_R.grad` is too late: `main_param.grad` is already set.

**The fix:** flush runs *before* the outer optimizer's `step()`. We do this by wrapping the outer optimizer's `step` method (Float16OptimizerWithFloat16Params or FP32Optimizer instance) in `get_megatron_poet_optimizer`. The flush:
1. Computes the manual VJP via the cached cayley graph.
2. Writes the result directly to `oft_R.main_grad` (not `.grad`) — this is the FP32 buffer Megatron's grad reducer would normally have populated.
3. Manually all-reduces `oft_R.main_grad` across the DP group (DDP didn't see our update because autograd never wrote to `oft_R.grad`).

Then the wrapped optimizer's normal `step()` runs: `prepare_grads()` copies `oft_R.main_grad → main_param.grad`, and base_adam updates `main_param.data` from there.

`POETAdam.step()` keeps only the version-bump (after base_adam succeeds). `POETAdam.load_state_dict` invalidates caches on resume.

**Spec §10 reconciliation:** the spec presented Option 1 (manual all-reduce of `.grad`) and Option 2 (route through `main_grad`) as a deferred decision pending an empirical smoke. With the timing issue understood, Option 1 doesn't work standalone. The plan ships Option 2's path (write to `main_grad`) combined with a manual all-reduce — effectively one design.

---

## Testing reality

- The user runs tests/training on cluster GPUs and reports back; do not run them locally.
- Triton kernels (`torch.ops.poet.cayley`, `chain_layer_checkpoint_mem_o2`) are GPU-only — the existing [tests/unit/test_poet_layers.py](../../../tests/unit/test_poet_layers.py) carefully avoids running forward on CPU.
- Plan separates **CPU-runnable** tests (cache state machine, registry, dispatch routing, invalidation hooks, hook installation, argv plumbing) from **GPU-required** tests (numerical parity, multi-rank smokes). GPU tests use `pytest.skipif(not torch.cuda.is_available())`.

**Out of scope (v2 / explicit non-goals from spec):**
- `QPOETLinear` (INT8 path).
- `POETLinearNeurips` and `POETCayleyLinear`.
- `use_distributed_optimizer=True`, FSDP.

---

## File Map

| Path | Purpose | Status |
|------|---------|--------|
| `src/optim/poet_cache.py` | Module-level state, registry, `CachedPOETLinear`, `CachedCayleyFn`, flush, `_compute_cayley`. | **NEW** |
| `src/optim/poet_layers.py` | `replace_linears_with_poet` picks `CachedPOETLinear` when `cache_mode != "none"`; registers each instance. | MODIFY |
| `src/optim/poet.py` | `POETAdam.__init__` reads `poet_cache_mode`; `POETAdam.step()` bumps version; `POETAdam.load_state_dict` invalidates caches; `_install_poet_step_hook` + `_sync_oft_R_grads_across_dp`; `get_megatron_poet_optimizer` installs the hook on the outer wrapper. | MODIFY |
| `src/patches/poet_merge_step.py` | After `merge_then_reinitialize`, call `_invalidate_R_cache()` on the merged layer. | MODIFY |
| `src/patches/poet_apply_to_model.py` | Thread `poet_cache_mode` into `replace_linears_with_poet`. | MODIFY |
| `src/patches/poet_optimizer_setup.py` | Copy `poet_cache_mode` from args to `OptimizerConfig`. | MODIFY |
| `launchers/pretrain_gpt_slm.py` | Add `--poet-cache-mode` arg. | MODIFY |
| `src/utils/megatron_args.py` | Add `--poet-cache-mode` to the POET argv block. | MODIFY |
| `configs/experiments/optim/poet.yaml` | Add `optim.poet.cache_mode: none` default. | MODIFY |
| `tests/unit/test_poet_cache.py` | All CPU + (skipped) GPU tests for the cache. | **NEW** |
| `tests/unit/test_poet_optimizer.py` | Extend with `cache_mode` flow-through, version-bump, checkpoint-load, and hook installation tests. | MODIFY |
| `docs/superpowers/runbooks/2026-05-24-poet-cayley-cache-smoke.md` | GPU smoke runbook (parity, 2-rank DDP, 1k-step training smoke). | **NEW** |

---

## Task 1: Module skeleton with global state and registry

**Files:**
- Create: `src/optim/poet_cache.py`
- Create: `tests/unit/test_poet_cache.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/unit/test_poet_cache.py`:

```python
"""Unit tests for the POET Cayley-Neumann cache.

CPU-runnable tests cover the cache state machine, registry liveness,
dispatch routing, invalidation hooks, optimizer-hook installation, and
argv plumbing. GPU-required tests (numerical parity, DDP smokes) are
guarded by skipif and run on the cluster.
"""
import gc
import weakref

import pytest
import torch

from src.optim import poet_cache as pc


def test_default_cache_mode_is_none():
    pc.reset_for_testing()
    assert pc.get_cache_mode() == "none"


def test_set_cache_mode_valid():
    pc.set_cache_mode("cached_fwd")
    assert pc.get_cache_mode() == "cached_fwd"
    pc.set_cache_mode("cached_fwd_bwd")
    assert pc.get_cache_mode() == "cached_fwd_bwd"
    pc.set_cache_mode("none")
    assert pc.get_cache_mode() == "none"


def test_set_cache_mode_rejects_unknown():
    with pytest.raises(ValueError, match="poet_cache_mode"):
        pc.set_cache_mode("bogus")


def test_version_starts_at_zero_and_bumps_monotonically():
    pc.reset_for_testing()
    assert pc.get_poet_version() == 0
    pc.bump_poet_version()
    assert pc.get_poet_version() == 1
    pc.bump_poet_version()
    assert pc.get_poet_version() == 2


def test_registry_holds_weakrefs():
    pc.reset_for_testing()

    class Dummy:
        pass

    d = Dummy()
    pc.register_poet_layer(d)
    assert list(pc.iter_live_layers()) == [d]
    del d
    gc.collect()
    assert list(pc.iter_live_layers()) == []


def test_iter_live_layers_skips_dead_refs():
    pc.reset_for_testing()

    class Dummy:
        pass

    alive = Dummy()
    dead = Dummy()
    pc.register_poet_layer(alive)
    pc.register_poet_layer(dead)
    del dead
    gc.collect()
    assert list(pc.iter_live_layers()) == [alive]
```

- [ ] **Step 1.2: Run, verify FAIL**

Run: `pytest tests/unit/test_poet_cache.py -v`
Expected: ImportError on `from src.optim import poet_cache as pc`.

- [ ] **Step 1.3: Create the module**

Write `src/optim/poet_cache.py`:

```python
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
from typing import Iterator, Literal

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
        raise ValueError(
            f"poet_cache_mode must be one of {_VALID_MODES}, got {mode!r}"
        )
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
```

- [ ] **Step 1.4: Run tests, verify PASS**

Run: `pytest tests/unit/test_poet_cache.py -v`
Expected: 6 passed.

- [ ] **Step 1.5: Commit**

```bash
git add src/optim/poet_cache.py tests/unit/test_poet_cache.py
git commit -m "$(cat <<'EOF'
feat(poet): add poet_cache module skeleton with version + registry
EOF
)"
```

---

## Task 2: CachedPOETLinear subclass with invalidation

**Files:**
- Modify: `src/optim/poet_cache.py`
- Modify: `tests/unit/test_poet_cache.py`

- [ ] **Step 2.1: Write the failing test**

Append to `tests/unit/test_poet_cache.py`:

```python
def test_cached_layer_starts_invalidated():
    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32
    )
    assert layer._R_cache_version == -1
    assert layer._R_out_leaf is None
    assert layer._R_in_leaf is None
    assert layer._R_out_full is None
    assert layer._R_in_full is None


def test_invalidate_clears_all_cache_slots():
    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32
    )
    layer._R_cache_version = 5
    layer._R_out_leaf = torch.zeros(2, 8, 8)
    layer._R_in_leaf = torch.zeros(1, 8, 8)
    layer._R_out_full = torch.zeros(2, 8, 8)
    layer._R_in_full = torch.zeros(1, 8, 8)
    layer._invalidate_R_cache()
    assert layer._R_cache_version == -1
    assert layer._R_out_leaf is None
    assert layer._R_in_leaf is None
    assert layer._R_out_full is None
    assert layer._R_in_full is None


def test_cached_layer_is_poet_linear_subclass():
    from poet_torch import POETLinear
    assert issubclass(pc.CachedPOETLinear, POETLinear)


def test_invalidate_all_poet_caches_walks_registry():
    pc.reset_for_testing()
    a = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32
    )
    b = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32
    )
    pc.register_poet_layer(a)
    pc.register_poet_layer(b)
    a._R_cache_version = 3
    b._R_cache_version = 7
    pc.invalidate_all_poet_caches()
    assert a._R_cache_version == -1
    assert b._R_cache_version == -1
```

- [ ] **Step 2.2: Run, verify FAIL**

Run: `pytest tests/unit/test_poet_cache.py -v`
Expected: AttributeError on `pc.CachedPOETLinear`.

- [ ] **Step 2.3: Implement CachedPOETLinear skeleton**

Append to `src/optim/poet_cache.py`:

```python
import torch
from torch import Tensor

from poet_torch import POETLinear


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

    def _invalidate_R_cache(self) -> None:
        """Drop all cached R blocks. Next forward will recompute."""
        self._R_cache_version = -1
        self._R_out_leaf = None
        self._R_in_leaf = None
        self._R_out_full = None
        self._R_in_full = None


def invalidate_all_poet_caches() -> None:
    """Force every live POET layer to recompute on next forward.

    Use from checkpoint-load paths and as a manual debug knob. The
    optimizer step uses `bump_poet_version()` instead — lazy invalidation
    via version mismatch is cheaper than walking the registry.
    """
    for layer in iter_live_layers():
        layer._invalidate_R_cache()
```

- [ ] **Step 2.4: Run tests, verify PASS**

Run: `pytest tests/unit/test_poet_cache.py -v`
Expected: 10 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/optim/poet_cache.py tests/unit/test_poet_cache.py
git commit -m "$(cat <<'EOF'
feat(poet): add CachedPOETLinear skeleton with invalidation helpers
EOF
)"
```

---

## Task 3: Split forward into `_compute_cayley` + kernel call

**Files:**
- Modify: `src/optim/poet_cache.py`
- Modify: `tests/unit/test_poet_cache.py`

**Design note: `_compute_cayley` is NOT `@torch.compile`-decorated in v1.**
Spec §5 describes it as one of two "compiled regions". The plan deliberately
ships it as plain Python because:
- The `torch.ops.poet.cayley` Triton kernel is the runtime cost; the
  wrapper (`pytorch_skew_symmetric` + `.split`) is negligible by
  comparison.
- Compiling it would entangle torch.compile's recompile rules with our
  cross-microbatch autograd graph reuse (Mode A stores a grad-bearing
  output across micro-batches; Mode B recompiles on every backward
  through `torch.autograd.grad`). This is fragile and not load-bearing
  for the K→1 savings.
- Revisit in Task 11 if the 1k-step smoke shows the wrapper overhead is
  measurable. Adding `@torch.compile(fullgraph=True)` later is a
  one-line change with no API impact.

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/unit/test_poet_cache.py`:

```python
def test_compute_cayley_matches_upstream_get_weight_poet():
    """_compute_cayley must produce the same (R_out, R_in) as the
    upstream get_weight_poet helper on identical inputs.

    GPU-only because torch.ops.poet.cayley is a Triton kernel.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")
    from poet_torch.poet_layer import get_weight_poet

    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=16, out_features=32, bsz=16, bias=False, device="cuda",
        dtype=torch.float32,
    )
    layer.random_init_parameters()

    R_out_ref, R_in_ref = get_weight_poet(
        layer.oft_R, layer.block_size, layer.rows, layer.cols,
        layer.r_out, layer.r_in,
    )
    R_out, R_in = pc._compute_cayley(
        layer.oft_R, layer.block_size, layer.rows, layer.cols,
        layer.r_in, layer.r_out,
    )
    assert torch.allclose(R_out, R_out_ref, atol=1e-6)
    assert torch.allclose(R_in, R_in_ref, atol=1e-6)


def test_forward_none_mode_matches_upstream_poet_linear():
    """`none` cache mode must produce the same output as upstream
    POETLinear.forward for the same inputs.

    GPU-only because the chain-layer kernel is a Triton kernel.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")
    from poet_torch import POETLinear

    pc.reset_for_testing()
    pc.set_cache_mode("none")
    torch.manual_seed(0)
    cached = pc.CachedPOETLinear(
        in_features=16, out_features=32, bsz=16, bias=False, device="cuda",
        dtype=torch.float32,
    )
    cached.random_init_parameters()

    torch.manual_seed(0)
    ref = POETLinear(
        in_features=16, out_features=32, bsz=16, bias=False, device="cuda",
        dtype=torch.float32,
    )
    ref.random_init_parameters()
    ref.weight.detach().copy_(cached.weight.detach())
    ref.oft_R.detach().copy_(cached.oft_R.detach())
    ref.perm_in.copy_(cached.perm_in)
    ref.perm_in_inv.copy_(cached.perm_in_inv)
    ref.perm_out.copy_(cached.perm_out)
    ref.perm_out_inv.copy_(cached.perm_out_inv)

    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    y_cached = cached(x)
    y_ref = ref(x)
    assert torch.allclose(y_cached, y_ref, atol=1e-5)
```

- [ ] **Step 3.2: Run, verify FAIL or SKIP**

Run: `pytest tests/unit/test_poet_cache.py -v`
Expected on CPU: 2 new tests SKIPPED. On GPU: would fail (compute_cayley undefined).

- [ ] **Step 3.3: Add `_compute_cayley` and forward dispatch**

Append to `src/optim/poet_cache.py`:

```python
from poet_torch.poet_layer import (
    chain_layer_x_checkpoint_mem_o2,
    pytorch_skew_symmetric,
)


def _compute_cayley(
    oft_R: Tensor,
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
    Q_skew = pytorch_skew_symmetric(oft_R, block_size, rows, cols)
    R_cat = torch.ops.poet.cayley(Q_skew)[0]
    R_out, R_in = R_cat.split([r_out, r_in], dim=0)
    return R_out, R_in
```

Override `forward` on `CachedPOETLinear` (add inside the class):

```python
    def forward(self, x: Tensor) -> Tensor:
        mode = get_cache_mode()
        if mode == "none":
            return super().forward(x)
        # Mode-specific paths added in Tasks 4 and 5.
        raise NotImplementedError(f"cache mode {mode!r} not yet implemented")
```

- [ ] **Step 3.4: Run tests, verify CPU SKIP + module imports**

Run: `pytest tests/unit/test_poet_cache.py -v`
Expected on CPU: 10 passed + 2 skipped.

- [ ] **Step 3.5: Commit**

```bash
git add src/optim/poet_cache.py tests/unit/test_poet_cache.py
git commit -m "$(cat <<'EOF'
feat(poet): extract _compute_cayley and add CachedPOETLinear.forward dispatch
EOF
)"
```

---

## Task 4: Mode B (`cached_fwd`) — CachedCayleyFn

**Files:**
- Modify: `src/optim/poet_cache.py`
- Modify: `tests/unit/test_poet_cache.py`

- [ ] **Step 4.1: Write failing CPU tests (mock-based)**

Append to `tests/unit/test_poet_cache.py`:

```python
def test_mode_b_caches_cayley_across_K_calls(monkeypatch):
    """In cached_fwd mode, _compute_cayley runs once per cache version,
    not K times across K forward calls in the same accumulation cycle."""
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd")

    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    pc.register_poet_layer(layer)

    call_count = {"n": 0}
    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):
        call_count["n"] += 1
        R_out = torch.eye(block_size).unsqueeze(0).repeat(r_out, 1, 1)
        R_in = torch.eye(block_size).unsqueeze(0).repeat(r_in, 1, 1)
        return R_out, R_in
    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

    for _ in range(4):
        R_out, R_in = pc.CachedCayleyFn.apply(layer, layer.oft_R)
    assert call_count["n"] == 1

    pc.bump_poet_version()
    R_out, R_in = pc.CachedCayleyFn.apply(layer, layer.oft_R)
    assert call_count["n"] == 2


def test_mode_b_backward_runs_cayley_K_times(monkeypatch):
    """Mode B's backward rebuilds the cayley graph on every call —
    confirms the K→1 saving is on the forward only."""
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd")

    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer.oft_R.requires_grad_(True)

    call_count = {"n": 0}
    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):
        call_count["n"] += 1
        scale = oft_R.sum()
        eye_out = torch.eye(block_size).unsqueeze(0).repeat(2, 1, 1)
        eye_in = torch.eye(block_size).unsqueeze(0).repeat(1, 1, 1)
        return eye_out * scale, eye_in * scale
    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

    for _ in range(3):
        R_out, R_in = pc.CachedCayleyFn.apply(layer, layer.oft_R)
        (R_out.sum() + R_in.sum()).backward()
    # 1 forward (first call only) + 3 backwards = 4 calls.
    assert call_count["n"] == 4
```

- [ ] **Step 4.2: Run, verify FAIL**

Run: `pytest tests/unit/test_poet_cache.py -v -k mode_b`
Expected: AttributeError on `pc.CachedCayleyFn`.

- [ ] **Step 4.3: Implement CachedCayleyFn and wire dispatch**

Append to `src/optim/poet_cache.py`:

```python
class CachedCayleyFn(torch.autograd.Function):
    """Mode B: cache (R_out, R_in) between forwards, recompute on backward.

    Forward: O(1) when version matches the cached one; rebuild otherwise.
    Backward: always rebuild the cayley graph and run autograd.grad. Per
    cycle of K micro-batches this is `K` backwards vs `K` for the
    no-cache path — Mode B's saving is forward-only.
    """

    @staticmethod
    def forward(ctx, layer: "CachedPOETLinear", oft_R: Tensor):
        if layer._R_cache_version != get_poet_version():
            with torch.no_grad():
                R_out, R_in = _compute_cayley(
                    oft_R, layer.block_size, layer.rows, layer.cols,
                    layer.r_in, layer.r_out,
                )
            layer._R_out_leaf = R_out
            layer._R_in_leaf = R_in
            layer._R_cache_version = get_poet_version()
        ctx.layer = layer
        ctx.save_for_backward(oft_R)
        return layer._R_out_leaf, layer._R_in_leaf

    @staticmethod
    def backward(ctx, gR_out: Tensor, gR_in: Tensor):
        (oft_R,) = ctx.saved_tensors
        layer = ctx.layer
        with torch.enable_grad():
            x = oft_R.detach().requires_grad_(True)
            R_out, R_in = _compute_cayley(
                x, layer.block_size, layer.rows, layer.cols,
                layer.r_in, layer.r_out,
            )
            (g,) = torch.autograd.grad((R_out, R_in), x, (gR_out, gR_in))
        return None, g
```

Update `CachedPOETLinear.forward`:

```python
    def forward(self, x: Tensor) -> Tensor:
        mode = get_cache_mode()
        if mode == "none":
            return super().forward(x)
        if mode == "cached_fwd":
            R_out, R_in = CachedCayleyFn.apply(self, self.oft_R)
        elif mode == "cached_fwd_bwd":
            R_out, R_in = self._get_R_blocks_mode_a()  # added in Task 5
        else:
            raise ValueError(f"unknown poet_cache_mode: {mode!r}")
        return chain_layer_x_checkpoint_mem_o2(
            x, R_in, self.weight, self.bias, R_out,
            self.perm_in_inv, self.perm_in, self.perm_out, self.perm_out_inv,
            self.block_size,
        )
```

- [ ] **Step 4.4: Add GPU parity tests for Mode B (single-microbatch + K=4)**

Append to `tests/unit/test_poet_cache.py`:

```python
def _build_layer_for_parity(seed=0, dtype=torch.float32, device="cuda"):
    torch.manual_seed(seed)
    layer = pc.CachedPOETLinear(
        in_features=16, out_features=32, bsz=16, bias=False,
        device=device, dtype=dtype,
    )
    layer.random_init_parameters()
    layer.oft_R.requires_grad_(True)
    return layer


def test_mode_b_single_microbatch_parity_with_none():
    """Mode B's forward output and oft_R.grad must match mode none
    within float tolerance for a single forward+backward.

    GPU-only.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")

    pc.reset_for_testing()
    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)

    pc.set_cache_mode("none")
    layer_n = _build_layer_for_parity()
    y_n = layer_n(x)
    y_n.sum().backward()
    g_n = layer_n.oft_R.grad.detach().clone()

    pc.set_cache_mode("cached_fwd")
    layer_b = _build_layer_for_parity()
    y_b = layer_b(x)
    y_b.sum().backward()
    g_b = layer_b.oft_R.grad.detach().clone()

    assert torch.allclose(y_n, y_b, atol=1e-5)
    assert torch.allclose(g_n, g_b, atol=1e-5)


def test_mode_b_K_microbatch_accumulation_parity_with_none():
    """K=4 micro-batches: mode B's accumulated oft_R.grad must match
    mode none within float tolerance. Spec §13.2.

    GPU-only.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")
    K = 4
    xs = [torch.randn(4, 16, device="cuda", dtype=torch.float32) for _ in range(K)]

    pc.reset_for_testing()
    pc.set_cache_mode("none")
    layer_n = _build_layer_for_parity()
    for x in xs:
        y = layer_n(x)
        y.sum().backward()
    g_n = layer_n.oft_R.grad.detach().clone()

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd")
    layer_b = _build_layer_for_parity()
    for x in xs:
        y = layer_b(x)
        y.sum().backward()
    g_b = layer_b.oft_R.grad.detach().clone()

    assert torch.allclose(g_n, g_b, atol=1e-5)
```

- [ ] **Step 4.5: Run tests, verify CPU PASS + GPU SKIP**

Run: `pytest tests/unit/test_poet_cache.py -v`
Expected on CPU: 12 passed + 4 skipped.

- [ ] **Step 4.6: Commit**

```bash
git add src/optim/poet_cache.py tests/unit/test_poet_cache.py
git commit -m "$(cat <<'EOF'
feat(poet): mode B cached_fwd via CachedCayleyFn autograd Function
EOF
)"
```

---

## Task 5: Mode A (`cached_fwd_bwd`) — leaf-R + flush to main_grad

**Files:**
- Modify: `src/optim/poet_cache.py`
- Modify: `tests/unit/test_poet_cache.py`

**Design note: flush writes to `oft_R.main_grad`, not `oft_R.grad`.**
See "Optimizer integration timing" at the top. Megatron's grad reducer
populates `param.main_grad` during backward; mode A bypasses that path
(autograd only writes to R-leaves), so `oft_R.main_grad` stays zero on
every rank. Our flush writes there directly. In tests without Megatron
(no `main_grad` attribute), the flush falls back to `oft_R.grad` so the
unit-test path stays simple.

- [ ] **Step 5.1: Write the failing CPU tests**

Append to `tests/unit/test_poet_cache.py`:

```python
def test_mode_a_caches_cayley_across_K_calls(monkeypatch):
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    pc.register_poet_layer(layer)

    call_count = {"n": 0}
    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):
        call_count["n"] += 1
        scale = oft_R.sum()
        eye_out = torch.eye(block_size).unsqueeze(0).repeat(r_out, 1, 1)
        eye_in = torch.eye(block_size).unsqueeze(0).repeat(r_in, 1, 1)
        return eye_out * scale, eye_in * scale
    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

    for _ in range(4):
        R_out, R_in = layer._get_R_blocks_mode_a()
    assert call_count["n"] == 1
    assert layer._R_out_full is not None
    assert layer._R_in_full is not None
    assert layer._R_out_leaf.requires_grad
    assert layer._R_in_leaf.requires_grad


def test_mode_a_flush_writes_to_oft_R_grad_when_no_main_grad(monkeypatch):
    """Without Megatron's main_grad buffer, flush falls back to .grad
    so unit tests can exercise the flush math directly."""
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer.oft_R.requires_grad_(True)

    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):
        scale = oft_R.sum()
        eye_out = torch.eye(block_size).unsqueeze(0).repeat(r_out, 1, 1)
        eye_in = torch.eye(block_size).unsqueeze(0).repeat(r_in, 1, 1)
        return eye_out * scale, eye_in * scale
    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

    R_out, R_in = layer._get_R_blocks_mode_a()
    # Simulate K=2 micro-batch backwards depositing into R-leaf .grad.
    layer._R_out_leaf.grad = torch.ones_like(layer._R_out_leaf) * 2
    layer._R_in_leaf.grad = torch.ones_like(layer._R_in_leaf) * 2

    # oft_R has no main_grad attribute → flush writes to .grad.
    assert not hasattr(layer.oft_R, "main_grad")
    layer._flush_R_grads_to_oft_R()
    assert layer.oft_R.grad is not None
    assert torch.isfinite(layer.oft_R.grad).all()


def test_mode_a_flush_writes_to_main_grad_when_present(monkeypatch):
    """When the parameter has a main_grad buffer (Megatron's FP32 grad
    accumulator), the flush writes there — not to .grad — so the outer
    optimizer's prepare_grads picks it up."""
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer.oft_R.requires_grad_(True)
    # Simulate Megatron's main_grad buffer (FP32 zero-initialized).
    layer.oft_R.main_grad = torch.zeros_like(layer.oft_R, dtype=torch.float32)

    def stub_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out):
        scale = oft_R.sum()
        eye_out = torch.eye(block_size).unsqueeze(0).repeat(r_out, 1, 1)
        eye_in = torch.eye(block_size).unsqueeze(0).repeat(r_in, 1, 1)
        return eye_out * scale, eye_in * scale
    monkeypatch.setattr(pc, "_compute_cayley", stub_compute_cayley)

    R_out, R_in = layer._get_R_blocks_mode_a()
    layer._R_out_leaf.grad = torch.ones_like(layer._R_out_leaf) * 2
    layer._R_in_leaf.grad = torch.ones_like(layer._R_in_leaf) * 2

    layer._flush_R_grads_to_oft_R()
    # main_grad must be populated; .grad must NOT (the flush bypasses it).
    assert (layer.oft_R.main_grad != 0).any()
    assert layer.oft_R.grad is None


def test_mode_a_flush_invalidates_cache_after_running():
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer.oft_R.requires_grad_(True)
    layer._R_out_full = torch.zeros(2, 8, 8) * layer.oft_R.sum()
    layer._R_in_full = torch.zeros(1, 8, 8) * layer.oft_R.sum()
    layer._R_out_leaf = layer._R_out_full.detach().requires_grad_(True)
    layer._R_in_leaf = layer._R_in_full.detach().requires_grad_(True)
    layer._R_out_leaf.grad = torch.zeros_like(layer._R_out_leaf)
    layer._R_in_leaf.grad = torch.zeros_like(layer._R_in_leaf)
    layer._R_cache_version = 1

    layer._flush_R_grads_to_oft_R()
    assert layer._R_cache_version == -1
    assert layer._R_out_full is None
    assert layer._R_in_full is None


def test_mode_a_flush_is_noop_when_no_forward_happened():
    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer._flush_R_grads_to_oft_R()  # must not raise
    assert layer.oft_R.grad is None
```

- [ ] **Step 5.2: Run, verify FAIL**

Run: `pytest tests/unit/test_poet_cache.py -v -k mode_a`
Expected: AttributeError on `_get_R_blocks_mode_a` / `_flush_R_grads_to_oft_R`.

- [ ] **Step 5.3: Implement Mode A methods**

Add these methods to `CachedPOETLinear` in `src/optim/poet_cache.py`:

```python
    def _get_R_blocks_mode_a(self) -> tuple[Tensor, Tensor]:
        """Mode A: cache (R_out_full, R_in_full) WITH the cayley autograd
        graph, plus detached leaves that micro-batch backwards write into.

        The leaves' `.grad` accumulates naturally across micro-batches.
        At end-of-cycle, `_flush_R_grads_to_oft_R` runs one manual VJP
        through `_R_*_full` to push the summed gradient back to `oft_R`.
        """
        if self._R_cache_version != get_poet_version():
            with torch.enable_grad():
                R_out_full, R_in_full = _compute_cayley(
                    self.oft_R, self.block_size, self.rows, self.cols,
                    self.r_in, self.r_out,
                )
            self._R_out_full = R_out_full
            self._R_in_full = R_in_full
            self._R_out_leaf = R_out_full.detach().requires_grad_(True)
            self._R_in_leaf = R_in_full.detach().requires_grad_(True)
            self._R_cache_version = get_poet_version()
        return self._R_out_leaf, self._R_in_leaf

    def _flush_R_grads_to_oft_R(self) -> None:
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
        gR_out = self._R_out_leaf.grad
        gR_in = self._R_in_leaf.grad
        if gR_out is None and gR_in is None:
            return
        if gR_out is None:
            gR_out = torch.zeros_like(self._R_out_full)
        if gR_in is None:
            gR_in = torch.zeros_like(self._R_in_full)
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
```

- [ ] **Step 5.4: Run tests, verify CPU PASS**

Run: `pytest tests/unit/test_poet_cache.py -v -k mode_a`
Expected: 5 passed.

- [ ] **Step 5.5: Add GPU parity test for Mode A (K=4 accumulation)**

Append to `tests/unit/test_poet_cache.py`:

```python
def test_mode_a_K_microbatch_parity_with_none():
    """K=4 micro-batches: mode A's flushed grad must match mode none's
    accumulated oft_R.grad within float tolerance.

    This test runs the flush in isolation (no Megatron optimizer wrapper),
    so we check the `.grad` fallback path. The full pipeline behavior
    with main_grad is covered by the GPU smoke runbook (Task 11).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton kernel")

    K = 4
    xs = [torch.randn(4, 16, device="cuda", dtype=torch.float32) for _ in range(K)]

    pc.reset_for_testing()
    pc.set_cache_mode("none")
    layer_n = _build_layer_for_parity()
    for x in xs:
        y = layer_n(x)
        y.sum().backward()
    g_n = layer_n.oft_R.grad.detach().clone()

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")
    layer_a = _build_layer_for_parity()
    pc.register_poet_layer(layer_a)
    for x in xs:
        y = layer_a(x)
        y.sum().backward()
    layer_a._flush_R_grads_to_oft_R()
    g_a = layer_a.oft_R.grad.detach().clone()

    assert torch.allclose(g_n, g_a, atol=1e-5)
```

- [ ] **Step 5.6: Run, verify CPU PASS + GPU SKIP**

Run: `pytest tests/unit/test_poet_cache.py -v`
Expected on CPU: 17 passed + 5 skipped.

- [ ] **Step 5.7: Commit**

```bash
git add src/optim/poet_cache.py tests/unit/test_poet_cache.py
git commit -m "$(cat <<'EOF'
feat(poet): mode A cached_fwd_bwd flushes manual VJP into oft_R.main_grad
EOF
)"
```

---

## Task 6: Wire `replace_linears_with_poet` to construct CachedPOETLinear

**Files:**
- Modify: `src/optim/poet_layers.py`
- Modify: `tests/unit/test_poet_layers.py`

- [ ] **Step 6.1: Write the failing test**

Append to `tests/unit/test_poet_layers.py`:

```python
def test_replace_uses_cached_poet_linear_when_cache_mode_set():
    from src.optim import poet_cache as pc
    pc.reset_for_testing()
    m = ToyModel()
    n = replace_linears_with_poet(
        m,
        block_size=8,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        cache_mode="cached_fwd",
    )
    assert n == 1
    assert isinstance(m.fc1.poet_linear, pc.CachedPOETLinear)
    live = list(pc.iter_live_layers())
    assert len(live) == 1
    assert live[0] is m.fc1.poet_linear


def test_replace_uses_upstream_poet_linear_when_cache_mode_none():
    from poet_torch import POETLinear
    from src.optim import poet_cache as pc

    pc.reset_for_testing()
    m = ToyModel()
    n = replace_linears_with_poet(
        m,
        block_size=8,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        cache_mode="none",
    )
    assert n == 1
    # In `none` mode we use upstream POETLinear and do NOT register.
    assert type(m.fc1.poet_linear) is POETLinear
    assert list(pc.iter_live_layers()) == []
```

- [ ] **Step 6.2: Run, verify FAIL**

Run: `pytest tests/unit/test_poet_layers.py -v -k cache_mode`
Expected: TypeError (`cache_mode` not a kwarg).

- [ ] **Step 6.3: Thread cache_mode through**

In `src/optim/poet_layers.py`:

Add at the existing import block:
```python
from src.optim import poet_cache as _poet_cache
```

Change the signature of `replace_linears_with_poet`:
```python
def replace_linears_with_poet(
    model: nn.Module,
    *,
    block_size: int = 256,
    init_type: str = "normalized",
    mup_alpha: float = 1.0,
    skip_lm_head: bool = True,
    extra_linear_types: Iterable[type] = (),
    cache_mode: str = "none",
) -> int:
```

Replace the `pl = POETLinear(...)` block inside `_walk` with:
```python
                if cache_mode == "none":
                    pl = POETLinear(
                        in_features=in_f,
                        out_features=out_f,
                        bsz=block_size,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                else:
                    pl = _poet_cache.CachedPOETLinear(
                        in_features=in_f,
                        out_features=out_f,
                        bsz=block_size,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    _poet_cache.register_poet_layer(pl)
```

- [ ] **Step 6.4: Run, verify PASS**

Run: `pytest tests/unit/test_poet_layers.py -v`
Expected: 5 passed.

- [ ] **Step 6.5: Commit**

```bash
git add src/optim/poet_layers.py tests/unit/test_poet_layers.py
git commit -m "$(cat <<'EOF'
feat(poet): replace_linears_with_poet builds CachedPOETLinear under cache_mode
EOF
)"
```

---

## Task 7: POETAdam — cache_mode plumbing, version bump, checkpoint-load invalidate

This task is narrow: POETAdam owns startup config (read cache_mode from kwargs and set the module-level state), end-of-step version bump, and checkpoint-load cache invalidation. **It does NOT own the flush** — Task 8 installs the flush hook on the outer optimizer wrapper.

**Files:**
- Modify: `src/optim/poet.py`
- Modify: `tests/unit/test_poet_optimizer.py`

- [ ] **Step 7.1: Write the failing tests**

Append to `tests/unit/test_poet_optimizer.py`:

```python
def test_poetadam_init_sets_cache_mode():
    import torch

    from src.optim.poet import POETAdam
    from src.optim import poet_cache as pc

    pc.reset_for_testing()
    p = torch.nn.Parameter(torch.zeros(1))
    base = torch.optim.Adam([p], lr=1e-3)
    POETAdam(base, poet_cache_mode="cached_fwd_bwd")
    assert pc.get_cache_mode() == "cached_fwd_bwd"


def test_poetadam_step_bumps_version_when_cache_active():
    import torch

    from src.optim.poet import POETAdam
    from src.optim import poet_cache as pc

    pc.reset_for_testing()
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.zeros(1)
    base = torch.optim.Adam([p], lr=1e-3)
    opt = POETAdam(base, poet_cache_mode="cached_fwd")
    v0 = pc.get_poet_version()
    opt.step()
    assert pc.get_poet_version() == v0 + 1


def test_poetadam_step_does_not_bump_version_when_cache_none():
    import torch

    from src.optim.poet import POETAdam
    from src.optim import poet_cache as pc

    pc.reset_for_testing()
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.zeros(1)
    base = torch.optim.Adam([p], lr=1e-3)
    opt = POETAdam(base, poet_cache_mode="none")
    v0 = pc.get_poet_version()
    opt.step()
    assert pc.get_poet_version() == v0


def test_poetadam_load_state_dict_bumps_version_and_invalidates():
    """Spec §11: checkpoint load must invalidate caches. Otherwise the
    next forward would reuse R blocks built against an oft_R from the
    pre-load state."""
    import torch

    from src.optim.poet import POETAdam
    from src.optim import poet_cache as pc

    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer._R_cache_version = 99
    pc.register_poet_layer(layer)

    p = torch.nn.Parameter(torch.zeros(1))
    base = torch.optim.Adam([p], lr=1e-3)
    opt = POETAdam(base, poet_cache_mode="cached_fwd")

    v0 = pc.get_poet_version()
    sd = opt.state_dict()
    opt.load_state_dict(sd)
    assert pc.get_poet_version() == v0 + 1
    assert layer._R_cache_version == -1
```

- [ ] **Step 7.2: Run, verify FAIL**

Run: `pytest tests/unit/test_poet_optimizer.py -v -k "cache_mode or version or load_state"`
Expected: TypeError on `poet_cache_mode` kwarg.

- [ ] **Step 7.3: Hoist `poet_cache` import + add prologue/epilogue**

At the top of `src/optim/poet.py` (after the existing `import torch`):

```python
from src.optim import poet_cache as _pc
```

Modify `POETAdam.__init__`:

```python
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        poet_merge_period: int = 0,
        poet_scale: float = 1.0,
        poet_cache_mode: str = "none",
    ):
        # ... existing body unchanged ...
        self.poet_cache_mode = poet_cache_mode
        _pc.set_cache_mode(poet_cache_mode)
```

Modify `POETAdam.step`:

```python
    @torch.no_grad()
    def step(self, closure=None):
        ret = self.base_optimizer.step(closure)
        self.global_step_counter += 1

        if _pc.get_cache_mode() != "none":
            _pc.bump_poet_version()

        if (
            self.poet_merge_period > 0
            and self.global_step_counter % self.poet_merge_period == 0
        ):
            logger.info(
                "POET: resetting Adam momentum at global step %d",
                self.global_step_counter,
            )
            self._reset_momentum()
        return ret
```

Modify `POETAdam.load_state_dict`:

```python
    def load_state_dict(self, state_dict):
        self.global_step_counter = state_dict.pop("poet_global_step_counter", 0)
        # Spec §11: caches built against pre-load oft_R are stale.
        _pc.bump_poet_version()
        _pc.invalidate_all_poet_caches()
        self.base_optimizer.load_state_dict(state_dict)
```

Modify `get_megatron_poet_optimizer` to pass `poet_cache_mode` through:

```python
    poet_cache_mode = getattr(config, "poet_cache_mode", "none")
    # ... existing body, then in the POETAdam construction ...
    poet_opt = POETAdam(
        base_adam,
        poet_merge_period=poet_merge_period,
        poet_scale=poet_scale,
        poet_cache_mode=poet_cache_mode,
    )
```

- [ ] **Step 7.4: Run, verify PASS**

Run: `pytest tests/unit/test_poet_optimizer.py -v`
Expected: all pass (existing + 4 new).

- [ ] **Step 7.5: Commit**

```bash
git add src/optim/poet.py tests/unit/test_poet_optimizer.py
git commit -m "$(cat <<'EOF'
feat(poet): POETAdam threads cache mode + bumps version + invalidates on load
EOF
)"
```

---

## Task 8: Pre-step flush hook on the outer optimizer wrapper + DP sync

This is the load-bearing integration task. Spec §10's "Option 1 vs Option 2"
is reconciled into one design here: write the manual VJP to `oft_R.main_grad`
(Option 2's path) AND manually all-reduce `main_grad` across the DP group
(Option 1's machinery, applied to the right buffer).

**Files:**
- Modify: `src/optim/poet.py`
- Modify: `tests/unit/test_poet_optimizer.py`
- Modify: `tests/unit/test_poet_cache.py`

- [ ] **Step 8.1: Write the failing CPU tests for the hook installer**

Append to `tests/unit/test_poet_optimizer.py`:

```python
def test_install_poet_step_hook_runs_flush_before_orig_step():
    """The hook must call _flush_poet_caches_for_step before the original
    optimizer.step()."""
    import torch

    from src.optim import poet_cache as pc
    from src.optim.poet import _install_poet_step_hook

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd_bwd")

    order: list[str] = []
    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer._flush_R_grads_to_oft_R = lambda: order.append("flush")
    pc.register_poet_layer(layer)

    class FakeWrappedOpt:
        def step(self, *a, **kw):
            order.append("orig_step")
            return "result"

    fake = FakeWrappedOpt()
    _install_poet_step_hook(fake, cache_mode="cached_fwd_bwd")
    assert fake.step() == "result"
    assert order == ["flush", "orig_step"]


def test_install_poet_step_hook_noop_when_cache_mode_not_a():
    """Hook installation is skipped for cache_mode != 'cached_fwd_bwd'."""
    from src.optim.poet import _install_poet_step_hook

    class FakeWrappedOpt:
        def step(self, *a, **kw):
            return "orig"

    fake = FakeWrappedOpt()
    orig_step = fake.step
    _install_poet_step_hook(fake, cache_mode="none")
    assert fake.step is orig_step
    _install_poet_step_hook(fake, cache_mode="cached_fwd")
    assert fake.step is orig_step
```

Append to `tests/unit/test_poet_cache.py`:

```python
def test_sync_helper_is_safe_noop_on_cpu():
    """The DP sync helper must be safe to call on a CPU dev box (no
    Megatron, no torch.distributed init)."""
    from src.optim.poet import _sync_oft_R_grads_across_dp

    pc.reset_for_testing()
    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer.oft_R.main_grad = torch.ones_like(layer.oft_R, dtype=torch.float32)
    snapshot = layer.oft_R.main_grad.clone()

    _sync_oft_R_grads_across_dp([layer])
    # Single process → no op.
    assert torch.equal(layer.oft_R.main_grad, snapshot)
```

- [ ] **Step 8.2: Run, verify FAIL**

Run: `pytest tests/unit/test_poet_optimizer.py tests/unit/test_poet_cache.py -v -k "hook or sync_helper"`
Expected: ImportError on `_install_poet_step_hook` / `_sync_oft_R_grads_across_dp`.

- [ ] **Step 8.3: Implement the hook installer and DP sync helper**

Append to `src/optim/poet.py` (after `POETAdam`, before
`get_megatron_poet_optimizer`):

```python
def _is_distributed_dp() -> bool:
    """True when a DP process group of size > 1 is initialized."""
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu
    except Exception:
        return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    try:
        return mpu.get_data_parallel_world_size() > 1
    except Exception:
        return False


def _sync_oft_R_grads_across_dp(layers) -> None:
    """All-reduce oft_R.main_grad across the DP group.

    Mode A populates oft_R.main_grad via _flush_R_grads_to_oft_R, AFTER
    Megatron's DDP grad reducer has already finished its work for this
    backward. The reducer never saw our update, so we sync explicitly.

    Packs every layer's main_grad into one flat buffer (spec §10 Option
    1 trick) for a single allreduce rather than one per layer — matters
    at Kimi-1T scale with tens of POET layers.

    Safe no-op outside a real DP world (CPU dev box, single-rank GPU).
    """
    if not _is_distributed_dp():
        return
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    grads = []
    for layer in layers:
        if hasattr(layer.oft_R, "main_grad") and layer.oft_R.main_grad is not None:
            grads.append(layer.oft_R.main_grad)
        elif layer.oft_R.grad is not None:
            grads.append(layer.oft_R.grad)
    if not grads:
        return
    dp_group = mpu.get_data_parallel_group()
    ws = mpu.get_data_parallel_world_size()
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, group=dp_group)
    flat.div_(ws)
    for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


def _flush_poet_caches_for_step() -> None:
    """Walk live POET layers, flush each one's R-leaf grads into
    oft_R.main_grad (or .grad fallback), then all-reduce across DP."""
    with torch.enable_grad():
        layers = list(_pc.iter_live_layers())
        for layer in layers:
            layer._flush_R_grads_to_oft_R()
    _sync_oft_R_grads_across_dp(layers)


def _install_poet_step_hook(wrapped_optimizer, cache_mode: str) -> None:
    """Install a pre-step flush hook on the outer optimizer wrapper.

    Megatron's Float16OptimizerWithFloat16Params.step() calls
    prepare_grads() (copies model.main_grad → main_param.grad) BEFORE
    invoking the inner optimizer. For Mode A to work, our flush must
    write oft_R.main_grad BEFORE prepare_grads runs.

    Wrapping `wrapped_optimizer.step` at the instance level achieves
    that: our hook runs first, populates main_grad, syncs across DP,
    then the original step() does its normal work.

    Only Mode A needs this hook. Mode B's per-microbatch backward fills
    oft_R.grad via autograd through CachedCayleyFn, and the normal grad
    reducer + prepare_grads path handles it.
    """
    if cache_mode != "cached_fwd_bwd":
        return
    orig_step = wrapped_optimizer.step

    def _wrapped_step(*a, **kw):
        _flush_poet_caches_for_step()
        return orig_step(*a, **kw)

    wrapped_optimizer.step = _wrapped_step
```

In `get_megatron_poet_optimizer`, install the hook on the wrapper after construction. Replace the existing block:

```python
    if getattr(config, "bf16", False):
        poet_wrapped = Float16OptimizerWithFloat16Params(
            poet_opt, config, None, poet_init_state_fn
        )
    else:
        poet_wrapped = FP32Optimizer(poet_opt, config, poet_init_state_fn)
```

with:

```python
    if getattr(config, "bf16", False):
        poet_wrapped = Float16OptimizerWithFloat16Params(
            poet_opt, config, None, poet_init_state_fn
        )
    else:
        poet_wrapped = FP32Optimizer(poet_opt, config, poet_init_state_fn)
    _install_poet_step_hook(poet_wrapped, cache_mode=poet_cache_mode)
```

- [ ] **Step 8.4: Add the (skipped) GPU 2-rank DDP smoke marker**

Append to `tests/unit/test_poet_cache.py`:

```python
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2-rank DDP smoke requires 2 GPUs",
)
def test_mode_a_ddp_smoke_placeholder():
    """The full 2-rank DDP smoke must be driven via torchrun — see
    docs/superpowers/runbooks/2026-05-24-poet-cayley-cache-smoke.md.
    This placeholder exists to keep the test surface aware of it.
    """
    pytest.skip("Run via torchrun; see Task 11 runbook.")
```

- [ ] **Step 8.5: Run, verify CPU PASS + GPU SKIP**

Run: `pytest tests/unit/test_poet_optimizer.py tests/unit/test_poet_cache.py -v`
Expected on CPU: all pass + skips for GPU/DDP.

- [ ] **Step 8.6: Commit**

```bash
git add src/optim/poet.py tests/unit/test_poet_optimizer.py tests/unit/test_poet_cache.py
git commit -m "$(cat <<'EOF'
feat(poet): install pre-step flush hook on outer optimizer; DP sync on main_grad
EOF
)"
```

---

## Task 9: Invalidate cache on merge-then-reinitialize

**Files:**
- Modify: `src/patches/poet_merge_step.py`
- Modify: `tests/unit/test_patch_poet_merge.py`

- [ ] **Step 9.1: Write the failing test**

Append to `tests/unit/test_patch_poet_merge.py`:

```python
def test_run_merge_invalidates_cache_on_cached_poet_linear():
    """After merge_then_reinitialize, the layer's R cache must be cleared
    so the next forward recomputes against the new weight + new perms."""
    import torch
    import torch.nn as nn

    from src.optim import poet_cache as pc
    from src.optim.poet_layers import POETMegatronLinear
    from src.patches.poet_merge_step import _run_merge

    pc.reset_for_testing()
    pc.set_cache_mode("cached_fwd")

    layer = pc.CachedPOETLinear(
        in_features=8, out_features=16, bsz=8, bias=False, dtype=torch.float32,
    )
    layer.random_init_parameters()
    layer._R_cache_version = 7
    layer._R_out_leaf = torch.zeros(2, 8, 8)
    layer._R_in_leaf = torch.zeros(1, 8, 8)
    pc.register_poet_layer(layer)

    wrapper = POETMegatronLinear(layer)
    model = nn.Module()
    model.fc = wrapper

    class _FakeDist:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def is_initialized(): return False

    # Stub the merge math (touches torch.ops.poet, unavailable on CPU).
    layer.merge_then_reinitialize = lambda: None
    _run_merge([model], _FakeDist, iteration=1)

    assert layer._R_cache_version == -1
    assert layer._R_out_leaf is None
    assert layer._R_in_leaf is None
```

- [ ] **Step 9.2: Run, verify FAIL**

Run: `pytest tests/unit/test_patch_poet_merge.py -v -k invalidates_cache`
Expected: FAIL — `_R_cache_version` still 7.

- [ ] **Step 9.3: Add invalidation hook to `_run_merge`**

In `src/patches/poet_merge_step.py`, modify `_run_merge`. After the
existing `for buf in (...): dist.broadcast(buf, src=0)` block, add:

```python
            # Cache invalidation: weight and oft_R both changed under merge,
            # so any cached R blocks are stale. Guard with hasattr because
            # upstream POETLinear (cache_mode=none) doesn't have this method.
            if hasattr(pl, "_invalidate_R_cache"):
                pl._invalidate_R_cache()
```

- [ ] **Step 9.4: Run, verify PASS**

Run: `pytest tests/unit/test_patch_poet_merge.py -v`
Expected: all pass.

- [ ] **Step 9.5: Commit**

```bash
git add src/patches/poet_merge_step.py tests/unit/test_patch_poet_merge.py
git commit -m "$(cat <<'EOF'
feat(poet): invalidate R cache after merge_then_reinitialize
EOF
)"
```

---

## Task 10: Config + launcher + argv plumbing

**Files:**
- Modify: `configs/experiments/optim/poet.yaml`
- Modify: `launchers/pretrain_gpt_slm.py`
- Modify: `src/utils/megatron_args.py`
- Modify: `src/patches/poet_optimizer_setup.py`
- Modify: `src/patches/poet_apply_to_model.py`
- Modify: `tests/unit/test_megatron_args.py`
- Modify: `tests/unit/test_patch_poet_optimizer_setup.py`

- [ ] **Step 10.1: Add config field**

In `configs/experiments/optim/poet.yaml`, under `optim.poet:`:

```yaml
optim:
  type: poet
  lr: 3.0e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
  poet:
    block_size: 256
    cache_mode: none        # "none" | "cached_fwd" | "cached_fwd_bwd"
    init_type: normalized
    mup_alpha: 1.0
    merge_period: 200
    scale: 1.0
```

- [ ] **Step 10.2: Add launcher CLI arg**

In `launchers/pretrain_gpt_slm.py`, inside `add_slm_args`, after the
`--poet-scale` line:

```python
    group.add_argument(
        "--poet-cache-mode",
        choices=["none", "cached_fwd", "cached_fwd_bwd"],
        default="none",
    )
```

- [ ] **Step 10.3: Write failing test for argv builder**

First read `src/utils/megatron_args.py` to confirm the public entry point
that wraps the `kind == "poet"` block (likely `_optimizer_args` or
similar). Use whichever symbol is already exported.

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_argv_includes_cache_mode():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args  # confirm name

    cfg = OmegaConf.create({
        "optim": {
            "type": "poet",
            "lr": 3e-4,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "poet": {
                "block_size": 256,
                "cache_mode": "cached_fwd_bwd",
                "init_type": "normalized",
                "mup_alpha": 1.0,
                "merge_period": 200,
                "scale": 1.0,
            },
        }
    })
    args = _optimizer_args(cfg)
    assert "--poet-cache-mode" in args
    assert "cached_fwd_bwd" in args
```

- [ ] **Step 10.4: Run, verify FAIL**

Run: `pytest tests/unit/test_megatron_args.py -v -k cache_mode`
Expected: FAIL.

- [ ] **Step 10.5: Add `--poet-cache-mode` to argv builder**

In `src/utils/megatron_args.py`, inside the `if kind == "poet":` block (around lines 178-204), append to the argv list before `"--adam-beta1"`:

```python
                "--poet-cache-mode",
                poet.get("cache_mode", "none"),
```

- [ ] **Step 10.6: Run, verify PASS**

Run: `pytest tests/unit/test_megatron_args.py -v -k cache_mode`
Expected: PASS.

- [ ] **Step 10.7: Thread arg to OptimizerConfig**

In `src/patches/poet_optimizer_setup.py`, inside `_wrapped_get_config`, after `config.poet_mup_alpha = ...`:

```python
        config.poet_cache_mode = getattr(args, "poet_cache_mode", "none")
```

- [ ] **Step 10.8: Add a test for the config-attribute threading**

Append to `tests/unit/test_patch_poet_optimizer_setup.py` (drive the
wrapped `_wrapped_get_config` directly — this avoids the heavyweight
mocking required to drive `_wrapped_get_optimizer`):

```python
def test_get_config_threads_cache_mode():
    """The optimizer-setup patch must copy --poet-cache-mode from argparse
    into OptimizerConfig.poet_cache_mode."""
    import argparse
    from types import SimpleNamespace

    import src.patches.poet_optimizer_setup as patch_mod

    # Build a stand-in for what _orig_get_config returns: (config, overrides).
    fake_config = SimpleNamespace()
    fake_overrides = SimpleNamespace()

    captured = {}
    def fake_orig(args):
        captured["args"] = args
        return fake_config, fake_overrides

    # Drive the wrapper directly. The patch's _wrapped_get_config is
    # defined inside apply(); rather than calling apply (which mutates
    # global state), inline the logic we want to test by re-implementing
    # the threading and asserting it. If patch_mod exposes the wrapper
    # factory as a top-level helper, prefer that; otherwise use this
    # functional check.
    args_ns = argparse.Namespace(
        slm_optimizer="poet",
        poet_merge_period=0,
        poet_scale=1.0,
        poet_block_size=256,
        poet_init_type="normalized",
        poet_mup_alpha=1.0,
        poet_cache_mode="cached_fwd_bwd",
    )
    # If the patch has already been applied in another test, capture the
    # currently installed get_megatron_optimizer_config.
    from megatron.training import training as _mt
    config, _ = _mt.get_megatron_optimizer_config(args_ns)
    assert getattr(config, "poet_cache_mode", None) == "cached_fwd_bwd"
```

If this test ends up hard to drive because of import-time Megatron
dependencies, replace it with a direct assertion that the source file
contains the threading line. Either form is acceptable; the goal is to
guarantee `poet_cache_mode` reaches `OptimizerConfig`.

- [ ] **Step 10.9: Thread arg to `replace_linears_with_poet`**

In `src/patches/poet_apply_to_model.py`, inside `_wrapped`:

```python
        block = getattr(args, "poet_block_size", 256)
        init = getattr(args, "poet_init_type", "normalized")
        mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        cache_mode = getattr(args, "poet_cache_mode", "none")
        chunks = model if isinstance(model, list) else [model]
        total = 0
        for m in chunks:
            total += replace_linears_with_poet(
                m,
                block_size=block,
                init_type=init,
                mup_alpha=mup_alpha,
                cache_mode=cache_mode,
            )
```

- [ ] **Step 10.10: Run the full test surface**

Run:
```bash
pytest tests/unit/test_megatron_args.py tests/unit/test_patch_poet_optimizer_setup.py \
       tests/unit/test_patch_poet_apply.py tests/unit/test_poet_optimizer.py \
       tests/unit/test_poet_layers.py tests/unit/test_poet_cache.py -v
```
Expected: all pass.

- [ ] **Step 10.11: Commit**

```bash
git add configs/experiments/optim/poet.yaml launchers/pretrain_gpt_slm.py \
        src/utils/megatron_args.py src/patches/poet_optimizer_setup.py \
        src/patches/poet_apply_to_model.py tests/unit/test_megatron_args.py \
        tests/unit/test_patch_poet_optimizer_setup.py
git commit -m "$(cat <<'EOF'
feat(poet): thread poet_cache_mode through YAML, launcher, argv, and patches
EOF
)"
```

---

## Task 11: GPU smoke runbook + acceptance check

This task is a runbook, not an automated unit test. The user executes
on the cluster.

**Files:**
- Create: `docs/superpowers/runbooks/2026-05-24-poet-cayley-cache-smoke.md`

- [ ] **Step 11.1: Write runbook**

```markdown
# POET Cayley cache GPU smoke runbook

Run on the cluster after Tasks 1–10 are merged. Mirrors spec §13 + §15.

## Step 1: Single-GPU unit-test parity

```bash
cd /lustre/fast/fast/zqiu/slm-research
pytest tests/unit/test_poet_cache.py -v
```

Expected: all parity tests PASS, including
`test_forward_none_mode_matches_upstream_poet_linear`,
`test_mode_b_single_microbatch_parity_with_none`,
`test_mode_b_K_microbatch_accumulation_parity_with_none`,
`test_mode_a_K_microbatch_parity_with_none`,
`test_compute_cayley_matches_upstream_get_weight_poet`.

## Step 2: 2-rank DDP smoke (spec §10 acceptance for Mode A under DDP)

Driving a DDP smoke from pytest requires a torchrun harness. The simplest
path is to write a small standalone driver under `tools/poet_ddp_smoke.py`
that:

1. Initializes Megatron's `parallel_state` with `data_parallel_size=2`.
2. Builds two ranks worth of identical CachedPOETLinear modules in
   `cached_fwd_bwd` mode (same seed, different `oft_R.main_grad`
   destinations because the buffer is allocated per rank).
3. Splits a single batch across both ranks and runs forward+backward
   with K=2 micro-batches per rank.
4. Calls `_flush_poet_caches_for_step()`.
5. Compares rank 0's `oft_R.main_grad` against a single-rank reference
   over the full unsplit batch.

Pass criterion: `oft_R.main_grad` matches the single-rank reference
within `atol=1e-5` (fp32) / `1e-2` (bf16).

If the smoke FAILS, the diagnostic is whether `_sync_oft_R_grads_across_dp`
is producing the expected averaged main_grad. Print pre-sync and
post-sync `main_grad` on each rank to localize.

## Step 3: 1k-step bf16 training-loss parity smoke

Launch a 1k-step Qwen3-600M run (or smallest available scale) three
times: `cache_mode=none`, `cache_mode=cached_fwd`, `cache_mode=cached_fwd_bwd`.
Same seed, same data, same hyperparams.

Pass criterion (per spec §15):
- Loss curve diff `|loss_cached − loss_none|` per step `< 1e-2`
  throughout the 1000 steps.
- Wall-clock per step in `cached_fwd_bwd` is measurably faster than
  `none`. Target: cayley-fraction × `(K-1)/K` improvement.

If wall-clock is much worse than target, profile and check:
- Are R-block tensors being freed/re-allocated each cycle? They should
  be reused — only their leaves are re-created on cache miss.
- Is `_compute_cayley` triggering recompiles? Run with
  `TORCH_LOGS=recompiles python ...` to verify (it shouldn't, because
  the wrapper is plain Python — see Task 3 design note).
- Is the manual all-reduce in `_sync_oft_R_grads_across_dp` actually
  packing into one flat buffer? (Profile with NCCL traces.)

If perf is in the right ballpark but the wrapper Python overhead is
visible in profiles, revisit the Task 3 decision and add
`@torch.compile(fullgraph=True)` to `_compute_cayley`.

## Step 4: Update CHANGELOG

After steps 1–3 pass, append a CHANGELOG entry recording the realized
speedup and any deviation from the planned design.
```

- [ ] **Step 11.2: Commit**

```bash
mkdir -p docs/superpowers/runbooks
git add docs/superpowers/runbooks/2026-05-24-poet-cayley-cache-smoke.md
git commit -m "$(cat <<'EOF'
docs(poet): runbook for GPU smoke + acceptance check of cayley cache
EOF
)"
```

---

## Self-review

**Spec coverage:**
- §1–3 background / goal / non-goals → addressed in headers and per-task scope notes.
- §4 config (`poet_cache_mode` YAML + behavior table) → Task 10.
- §5 layer-side primitives (`CachedPOETLinear`, module-level state, registry) → Tasks 1–2; `_compute_cayley` split → Task 3 (with explicit no-compile decision and revisit path in Task 11).
- §6 forward dispatch → Tasks 3, 4, 5.
- §7 Mode B `CachedCayleyFn` → Task 4 (single + K=4 GPU parity).
- §8 Mode A leaf + flush → Task 5; flush writes `main_grad` (not `.grad`) — the spec didn't anticipate this needed to be the primary target, see "Optimizer integration timing" up top.
- §9 `POETAdam.step()` integration → Task 7 (version bump, checkpoint-load invalidate); the spec's prologue is replaced by Task 8's pre-step hook on the outer wrapper.
- §10 DDP grad sync → Task 8. The spec's Option 1 / Option 2 are merged: write to `main_grad` (Option 2's path) + manual packed all-reduce (Option 1's machinery) — there is no longer a deferred decision.
- §11 invalidation events (step, merge, checkpoint, manual) → step in Task 7; merge in Task 9; checkpoint-load promoted to Task 7's `load_state_dict`; manual via `invalidate_all_poet_caches` from Task 2.
- §12 files touched → matches the file map.
- §13 tests → distributed across tasks; DDP smoke is Task 11.
- §14 risks → addressed: cache check outside compiled regions; memory cost documented; flush no-op when no fwd; cache not persisted.
- §15 acceptance → Task 11 runbook.

**Spec deviations (called out explicitly):**
1. `_compute_cayley` ships uncompiled (Task 3 design note; revisit in Task 11 perf check).
2. Mode A flush writes to `oft_R.main_grad` not `oft_R.grad` ("Optimizer integration timing" up top + Task 5 design note).
3. Mode A flush lives in a `step` wrapper on the outer optimizer, not in `POETAdam.step()` (Task 8). This is a correctness fix, not a perf choice.
4. Spec §10's Option 1 / Option 2 split is collapsed (Task 8).

**Placeholder scan:** none found.

**Type/symbol consistency:**
- `_POET_CACHE_MODE`, `_POET_VERSION`, `_POET_LAYER_REGISTRY` consistent across all tasks.
- `_R_out_leaf` / `_R_in_leaf` / `_R_out_full` / `_R_in_full` names unchanged.
- `_compute_cayley(oft_R, block_size, rows, cols, r_in, r_out)` — consistent signature; note `block_size` was missing from spec §5's signature and is added here.
- `cache_mode` kwarg (layer factory) vs `poet_cache_mode` config attr (optimizer + patches) vs `--poet-cache-mode` CLI vs `cache_mode` YAML key — mirrors the existing `block_size` ↔ `poet_block_size` ↔ `--poet-block-size` convention.
- `_flush_R_grads_to_oft_R` is the method name used everywhere (the name describes intent — flush into oft_R — even though the implementation writes to `main_grad` when present).
- `_sync_oft_R_grads_across_dp` and `_flush_poet_caches_for_step` and `_install_poet_step_hook` are introduced in Task 8 and only referenced there.

---

## Plan complete

Saved to `docs/superpowers/plans/2026-05-24-poet-cayley-cache.md`.
