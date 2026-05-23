# POET Cayley-Neumann cache for gradient-accumulation cycles

**Status:** design (approved 2026-05-23)
**Owner:** zqiu
**Scope:** float `POETLinear` only; `QPOETLinear` and `POETCayleyLinear` deferred to v2.

## 1. Background

POET parameterises each linear layer as
`y = P_out · R_out · P_out^T · W · P_in · R_in · P_in^T · x`,
where `R_out, R_in` are block-diagonal orthogonal matrices built from a
trainable skew-symmetric parameter `oft_R` via a Cayley–Neumann series.

In the current implementation
([`third_party/poet_torch/poet_layer.py:213-241`](../../../third_party/poet_torch/poet_layer.py#L213-L241)),
every forward through a `POETLinear` runs:

1. `R_out, R_in = get_weight_poet(oft_R, …)` — builds skew matrix, runs
   `torch.ops.poet.cayley`.
2. `y = chain_layer_x_checkpoint_mem_o2(x, R_in, W, b, R_out, …)`.

Both steps are welded into one `@torch.compile(fullgraph=True)` graph.

Training uses gradient accumulation: `K` micro-batches' forwards and
backwards are run, then one `optimizer.step()` updates `oft_R`. Inside an
accumulation cycle `oft_R` is constant, so `R_out, R_in = f(oft_R)` is
also constant — yet today we recompute it `K` times, and the cayley
backward also runs `K` times.

## 2. Goal

Cache `R_out, R_in` for the lifetime of one gradient-accumulation cycle
so the cayley computation runs once per cycle instead of `K` times.

By linearity of the cayley VJP,
`Σ_i ∂L_i/∂oft_R = J_f^T · Σ_i ∂L_i/∂(R_out, R_in)`,
so the cayley backward can also run once per cycle on the accumulated
upstream gradient.

## 3. Non-goals (v1)

- `QPOETLinear` (INT8/INT4) — same architecture, but interaction with
  `_dequantize_to` / `_requantize_from_float` and `merge_then_reinitialize`
  needs its own spec.
- `POETLinearNeurips`, `POETCayleyLinear` — different parameterisations.
- `use_distributed_optimizer=True` — already disallowed by the existing
  POET integration ([`src/optim/poet.py:199-200`](../../../src/optim/poet.py#L199-L200)).
- FSDP — not used in this project.

## 4. Configuration

One additive field in the experiment YAML:

```yaml
optimizer:
  poet_cache_mode: none | cached_fwd | cached_fwd_bwd   # default: none
```

| Mode | Behavior | Savings per cycle of K micro-batches |
|------|----------|--------------------------------------|
| `none` | Bit-for-bit current behavior. | 0 (baseline) |
| `cached_fwd` (Approach B) | Cache `R_out, R_in` between forwards; recompute cayley graph + VJP on every backward. | `K → 1` cayley forwards |
| `cached_fwd_bwd` (Approach A) | Cache + accumulate gradients on detached leaves; one manual VJP at end of cycle. | `K → 1` cayley forwards **and** `K → 1` cayley backwards |

`none` is the default. The two cached modes must produce numerically
equivalent results to `none` within float tolerance.

## 5. Layer-side primitives

New module `src/optim/poet_cache.py` introduces a subclass of the
vendored `POETLinear` — we do **not** modify upstream `poet_torch`.

```python
class CachedPOETLinear(POETLinear):
    _R_cache_version: int = -1
    _R_out_leaf: Tensor | None = None   # detached, requires_grad=True
    _R_in_leaf:  Tensor | None = None
    _R_out_full: Tensor | None = None   # mode A only: live tensor in cayley graph
    _R_in_full:  Tensor | None = None   # mode A only
```

Module-level globals in `poet_cache.py`:

- `_POET_CACHE_MODE: Literal["none", "cached_fwd", "cached_fwd_bwd"]`
  — set once at startup from config.
- `_POET_VERSION: int` — monotonic counter. Bumped by `POETAdam.step()`
  and on checkpoint load.
- `_POET_LAYER_REGISTRY: list[weakref[CachedPOETLinear]]` — populated by
  `replace_linears_with_poet`; used by `POETAdam` to flush and
  invalidate without traversing the model.

The existing `forward_core` (`@torch.compile(fullgraph=True)`) is split
into two compiled regions:

- `_compute_cayley(oft_R, rows, cols, r_in, r_out) -> (R_out, R_in)`
- `chain_layer_x_checkpoint_mem_o2(...)` — unchanged kernel call.

The Python cache-check that decides recompute-vs-reuse lives between
them and is **outside** both compiled regions, so torch.compile does
not see the mutable cache state.

## 6. Forward dispatch

```python
def CachedPOETLinear.forward(self, x):
    mode = _POET_CACHE_MODE
    if mode == "none":
        return forward_core(x, self.oft_R, ...)       # existing path
    if mode == "cached_fwd":
        R_out, R_in = CachedCayleyFn.apply(self, self.oft_R)
    else:  # cached_fwd_bwd
        R_out, R_in = self._get_R_blocks_mode_a()
    return chain_layer_x_checkpoint_mem_o2(
        x, R_in, self.weight, self.bias, R_out,
        self.perm_in_inv, self.perm_in, self.perm_out, self.perm_out_inv,
        self.block_size,
    )
```

## 7. Mode B (`cached_fwd`) — `CachedCayleyFn`

```python
class CachedCayleyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer, oft_R):
        if layer._R_cache_version != _POET_VERSION:
            with torch.no_grad():
                R_out, R_in = _compute_cayley(
                    oft_R, layer.rows, layer.cols, layer.r_in, layer.r_out
                )
            layer._R_out_leaf = R_out
            layer._R_in_leaf  = R_in
            layer._R_cache_version = _POET_VERSION
        ctx.layer = layer
        ctx.save_for_backward(oft_R)
        return layer._R_out_leaf, layer._R_in_leaf

    @staticmethod
    def backward(ctx, gR_out, gR_in):
        (oft_R,) = ctx.saved_tensors
        layer = ctx.layer
        with torch.enable_grad():
            x = oft_R.detach().requires_grad_(True)
            R_out, R_in = _compute_cayley(
                x, layer.rows, layer.cols, layer.r_in, layer.r_out
            )
            g, = torch.autograd.grad((R_out, R_in), x, (gR_out, gR_in))
        return None, g
```

- No trainer changes. `oft_R.grad` is populated on every micro-batch
  backward exactly as today, so Megatron DDP grad-bucket handling is
  unchanged.
- Saves `K → 1` cayley forwards per cycle.

## 8. Mode A (`cached_fwd_bwd`) — leaf-R + end-of-accum manual VJP

`_get_R_blocks_mode_a()`:

```python
def _get_R_blocks_mode_a(self):
    if self._R_cache_version != _POET_VERSION:
        with torch.enable_grad():
            R_out_full, R_in_full = _compute_cayley(
                self.oft_R, self.rows, self.cols, self.r_in, self.r_out
            )
        self._R_out_full, self._R_in_full = R_out_full, R_in_full
        self._R_out_leaf = R_out_full.detach().requires_grad_(True)
        self._R_in_leaf  = R_in_full.detach().requires_grad_(True)
        self._R_cache_version = _POET_VERSION
    return self._R_out_leaf, self._R_in_leaf
```

Per micro-batch, the chain-layer kernel's VJP writes into
`R_out_leaf.grad` and `R_in_leaf.grad`. Because both are leaf tensors
with `requires_grad=True`, `.grad` accumulates naturally across
micro-batches.

`_flush_R_grads_to_oft_R()` — called from `POETAdam.step()` prologue,
**before** `base_optimizer.step()`:

```python
def _flush_R_grads_to_oft_R(self):
    if self._R_out_full is None:
        return  # no forward happened this cycle
    gR_out = self._R_out_leaf.grad
    gR_in  = self._R_in_leaf.grad
    torch.autograd.backward(
        tensors=[self._R_out_full, self._R_in_full],
        grad_tensors=[gR_out, gR_in],
    )
    # oft_R.grad now holds Σ_i ∂L_i/∂oft_R locally
    self._invalidate_R_cache()
```

After flush, `oft_R.grad` is correct on each rank in isolation but has
**not** been all-reduced across the data-parallel group — see §10.

Saves `K → 1` cayley forwards **and** `K → 1` cayley backwards.

## 9. `POETAdam.step()` integration

Add a prologue to `POETAdam.step()` at
[`src/optim/poet.py:92-105`](../../../src/optim/poet.py#L92-L105):

```python
@torch.no_grad()
def step(self, closure=None):
    if _POET_CACHE_MODE == "cached_fwd_bwd":
        for ref in list(_POET_LAYER_REGISTRY):
            layer = ref()
            if layer is None: continue
            with torch.enable_grad():
                layer._flush_R_grads_to_oft_R()
        if _is_distributed() and _dp_world_size() > 1:
            _sync_oft_R_grads_across_dp()           # see §10
    ret = self.base_optimizer.step(closure)
    self.global_step_counter += 1
    _bump_poet_version()                            # invalidates all caches
    # existing merge-period logic ...
    return ret
```

`_bump_poet_version()` is the single source of cache invalidation for
optimizer-driven updates. Forward paths in both cached modes check
`self._R_cache_version != _POET_VERSION` and rebuild lazily.

## 10. DDP grad sync for mode A — **deferred decision, both paths specified**

Mode A writes `oft_R.grad` **after** `loss.backward()` returns, which
means Megatron's gradient buckets have already finished their
all-reduce by the time we populate it. Two paths are documented; the
implementation plan picks one after a small empirical check.

### Option 1 — Explicit all-reduce in flush

```python
def _sync_oft_R_grads_across_dp():
    dp_group = mpu.get_data_parallel_group()
    ws = mpu.get_data_parallel_world_size()
    for ref in _POET_LAYER_REGISTRY:
        layer = ref()
        if layer is None or layer.oft_R.grad is None: continue
        torch.distributed.all_reduce(layer.oft_R.grad, group=dp_group)
        layer.oft_R.grad.div_(ws)
```

- Pros: bypasses Megatron's bucket entirely; minimal Megatron internals
  touched; easy to reason about.
- Cons: one allreduce per POET layer (no bucketing). For Kimi-1T this
  could be tens of small allreduces per step. Could mitigate by packing
  all `oft_R.grad` tensors into one flat buffer for a single allreduce.

### Option 2 — Route through `param.main_grad`

- Megatron uses FP32 `param.main_grad` on each parameter, populated by
  the grad reducer. After the manual VJP, write into `main_grad`
  (instead of `param.grad`) and either:
  (a) mark the bucket dirty and re-fire the bucket's all-reduce, or
  (b) rely on the optimizer reading `main_grad` directly and add a
      separate sync for the post-VJP delta.
- Pros: uses existing bucketing infrastructure.
- Cons: deeper Megatron coupling; behaviour depends on `DistributedDataParallel`
  variant in use; fragile across Megatron pin bumps.

### Decision rule (to be exercised at implementation time)

Run a 2-rank DDP smoke test with mode A and Option 1. Compare
`oft_R.grad` after one step against a single-rank reference with the
same batch. If they match within `atol=1e-5` (fp32) or `1e-2` (bf16),
ship Option 1. Otherwise fall back to Option 2 with an explicit
test for grad-bucket re-firing.

## 11. Cache invalidation events

| Event | Action |
|-------|--------|
| `POETAdam.step()` | `_bump_poet_version()` → lazy invalidate on next forward |
| `merge_then_reinitialize()` | `_invalidate_R_cache()` on that layer (both `weight` and `oft_R` change) — added inside the `poet_merge_step` patch |
| Checkpoint load | `_bump_poet_version()` (and explicit `_invalidate_R_cache()` on each layer for safety) |
| Manual debug | `invalidate_all_poet_caches()` callable exposed for safety |

`_invalidate_R_cache()` clears all four cache slots (`_R_out_leaf`,
`_R_in_leaf`, `_R_out_full`, `_R_in_full`) and resets `_R_cache_version
= -1`.

## 12. Files touched

| File | Change |
|------|--------|
| `src/optim/poet_cache.py` | **new** — `CachedPOETLinear`, `CachedCayleyFn`, `_compute_cayley`, module-level state, flush + sync helpers |
| `src/optim/poet_layers.py` | `replace_linears_with_poet` constructs `CachedPOETLinear` when `cache_mode != "none"`; registers each instance in `_POET_LAYER_REGISTRY` |
| `src/optim/poet.py` | `POETAdam.__init__` reads `poet_cache_mode` and sets `_POET_CACHE_MODE`; `step()` prologue runs flush + DP sync + version bump |
| `src/patches/poet_merge_step.py` | call `_invalidate_R_cache()` per merged layer |
| `src/patches/poet_apply_to_model.py` | thread `poet_cache_mode` through to `replace_linears_with_poet` |
| `configs/experiments/optim/poet.yaml` | add `poet_cache_mode: none` default |
| `tests/unit/test_poet_cache.py` | **new** — see §13 |

## 13. Tests

CPU-runnable using `extra_linear_types=(nn.Linear,)`, matching the
pattern of `tests/unit/test_poet_layers.py`.

1. **Single-microbatch parity.** `none`, `cached_fwd`, `cached_fwd_bwd`
   produce identical `y` and `oft_R.grad` for the same input within
   `atol=1e-5` (fp32) / `1e-2` (bf16).
2. **Multi-microbatch accumulation parity.** With K=4 micro-batches and
   identical inputs/seeds across modes, total `oft_R.grad` matches
   `none` mode.
3. **Cache invalidation on step.** Version bump after `POETAdam.step()`
   forces recompute on the next forward; assert `_R_cache_version`
   advances.
4. **Cache invalidation on merge.** Calling
   `merge_then_reinitialize()` (via the merge-step patch) clears all
   four cache slots.
5. **Registry liveness.** Weakrefs are collected when a layer is
   deleted; flush + sync skip dead refs without erroring.
6. **DDP smoke (GPU-only, deferred).** 2-rank, K=2: per-rank manual
   VJP + DP sync (Option 1) matches single-rank reference within
   tolerance. Used to make the §10 decision.

## 14. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| DDP grad sync wrong for mode A (§10) | Explicit 2-rank smoke test before declaring mode A ready; fall back to Option 2 if Option 1 fails. |
| `torch.compile` recompiles on every cache miss | Cache check is plain Python outside compiled region; `_compute_cayley` and `chain_layer_x_checkpoint_mem_o2` each compile once and are called with stable shapes. |
| Cache state leaks across runs / interferes with checkpointing | `_invalidate_R_cache()` runs on checkpoint load; cache tensors are not part of any state dict. |
| Memory cost of cache | Per layer: `(r_in + r_out) × bsz × bsz` floats. Example (`bsz=256`, `d_model=4096`): ~2 MB/layer → ~400 MB for a 32-layer × 6-linear model. Tolerable; documented. |
| Mode A breaks if a forward happens without a backward before the next step | `_flush_R_grads_to_oft_R()` is a no-op when `_R_out_leaf.grad is None`; assert that this only happens in pure-eval contexts and document. |

## 15. Acceptance

v1 is done when:

- All six tests in §13 pass (test 6 deferred until a GPU run is
  available).
- A bf16 training smoke (1k steps, Qwen3-600M scale) shows mode
  `cached_fwd_bwd` produces a loss curve indistinguishable from
  mode `none` within fp16 noise (`abs_diff < 1e-2` per step at the
  loss level).
- Wall-clock per step on the same smoke is measurably faster in mode
  `cached_fwd_bwd` than `none` (target: at least the cayley fraction
  × `(K-1)/K` improvement; revisit if real speedup falls short).
