# POET decoupled block-count implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Goal

Add a `block_count` parameterization to POET layers. Today, POET uses a single
`block_size` for both `R_in` and `R_out`, which forces `block_size` to divide
**both** `in_features` and `out_features`. For non-square layers (e.g.
`4096 × 11008`), the only useful common divisor is often 256 — limiting
Mode A's speedup ceiling on these layers.

With `block_count = n`, both sides have `n` blocks but potentially **different
block sizes**:

```
block_size_in  = in_features  / n
block_size_out = out_features / n
```

This gives more flexibility on FFN-style layers and lets us pick a Cayley
block size that's natural for each side independently.

## Hard constraints

1. **Math equivalence.** When `block_size_in == block_size_out`, the result must
   be bit-equivalent (within kernel ULP) to the current POET implementation.
2. **Mode A (Cayley cache) must keep working.** The cache's K→1 amortization
   is a 1.20× speedup on small attention layers; we don't sacrifice it.
3. **No regression for existing recipes.** Specifying `bsz=256` (current behavior)
   continues to work and produces the same numerical output.

## What changes

The math `y = P_out · R_out · P_out_inv · W · P_in · R_in · P_in_inv · x` is
unchanged. Three things split into two independent halves:

1. **`oft_R`** splits into `oft_R_in` (shape `(r_in, n_elems_in)`) and `oft_R_out`
   (shape `(r_out, n_elems_out)`), where `n_elems_* = block_size_* · (block_size_* − 1) / 2`.
2. **Cayley computation** runs **twice** (once for `R_in`, once for `R_out`) —
   they can't be concatenated into a single kernel call because tile shapes differ.
3. **chain_layer kernel** accepts `(block_size_in, block_size_out)` instead of one
   `block_size`. The internal block-diag muls operate on different block sizes for
   the two sides.

## Architectural decisions

**Decision 1: Single layer class, optional decoupled.** Don't fork
`POETLinear` into `POETLinear` + `POETLinearDecoupled`. Instead, extend
`POETLinear` so that internally **everything is always stored as two
potentially-different block sizes**. The legacy `bsz=int` API constructs a
layer where `block_size_in == block_size_out`. The new `block_count=int` API
constructs a layer where they may differ. Single code path, two constructor entry points.

**Decision 2: New Triton op alongside existing.** Don't break the existing
`torch.ops.poet.chain_layer_checkpoint_mem_o2` signature. Add a new op
`chain_layer_checkpoint_mem_o2_decoupled` that takes two block sizes. The
new layer dispatches to the new op; the existing kernel is untouched and
still used by any code that calls `POETLinear(bsz=256)` and routes through
the legacy path.

Actually — since "Decision 1" says single code path, we always need the new
op regardless. We'll route through the new op for both equal- and unequal-
block-size cases. The old op can be retired once the new one is shipped and
tested. For correctness validation we compare new-op-with-equal-blocks vs
old-op.

**Decision 3: Mode A cache extends naturally.** `CachedPOETLinear` already
holds `_R_in_*`, `_R_out_*` separately; the cache mechanism doesn't care
about block sizes. Two changes:
- `_compute_cayley` becomes two calls (one per side) on the decoupled layer.
- The flush runs two manual VJPs (one for `oft_R_in`, one for `oft_R_out`),
  writing into two separate `main_grad` buffers.

## File map

| Path | Purpose | Status |
|------|---------|--------|
| `third_party/poet_torch/poet_layer.py` | `POETLinear.__init__` accepts `block_count`. Storage uses `block_size_in`, `block_size_out`, `oft_R_in`, `oft_R_out`. `get_weight_poet` runs Cayley twice. `forward` dispatches to new op. `merge_then_reinitialize` handles both sides. | MODIFY |
| `third_party/poet_torch/poet_ops.py` | Add new `chain_layer_checkpoint_mem_o2_decoupled` op (Triton kernel) with two block-size args. Keep the existing op. | MODIFY |
| `third_party/poet_torch/poet_layer.py` | `chain_layer_x_checkpoint_mem_o2_decoupled` Python wrapper. | MODIFY |
| `src/optim/poet_cache.py` | `_compute_cayley` returns from two calls. `_get_R_blocks_mode_a` builds independent in/out caches. `_flush_R_grads_to_oft_R` runs two VJPs. | MODIFY |
| `src/optim/poet_layers.py` | `replace_linears_with_poet` accepts `block_count` (mutually exclusive with `block_size`). | MODIFY |
| `src/patches/poet_apply_to_model.py` | Thread `block_count` from `args.poet_block_count`. | MODIFY |
| `src/utils/megatron_args.py` | Emit `--poet-block-count` when `optim.poet.block_count` is set. | MODIFY |
| `launchers/pretrain_gpt_slm.py` | Add `--poet-block-count` argparse arg. | MODIFY |
| `configs/experiments/optim/poet.yaml` | Document `block_count` as alternative to `block_size`. | MODIFY |
| `tests/unit/test_poet_decoupled.py` | All tests for decoupled mode (CPU + GPU). | **NEW** |
| `tests/unit/test_poet_layers.py` | Add tests for `block_count` plumbing. | MODIFY |
| `tests/unit/test_poet_cache.py` | Add parity test: cache works with decoupled block sizes. | MODIFY |
| `tools/poet_cache_bench.py` | Add `--block-count` flag and sweep it. | MODIFY |

## Testing reality

- Triton kernels are GPU-only. CPU tests use Python-only reference paths.
- The user runs all tests on the cluster and reports back.
- New tests are split into CPU-runnable (state, plumbing, dispatch, mathematical
  equivalence on a tiny pure-PyTorch reference) and GPU-required (full kernel parity).

---

## Task 1: Pure-PyTorch reference implementation of decoupled POET

**Files:**
- Create: `tests/unit/test_poet_decoupled.py`

Build a pure-PyTorch reference that implements the math
`y = P_out · R_out · P_out_inv · W · P_in · R_in · P_in_inv · x` with potentially
different block sizes for `R_in` and `R_out`. This is the ground-truth oracle
for all subsequent kernel work.

- [ ] **Step 1.1: Implement pure-PyTorch reference**

```python
def poet_reference_forward(x, W, oft_R_in, oft_R_out, perm_in, perm_in_inv,
                           perm_out, perm_out_inv, block_size_in, block_size_out):
    # 1. Build R_in (block-diag from oft_R_in via Cayley)
    R_in_blocks  = _cayley_pytorch(oft_R_in, block_size_in)    # (r_in,  bs_in,  bs_in)
    R_out_blocks = _cayley_pytorch(oft_R_out, block_size_out)  # (r_out, bs_out, bs_out)
    # 2. Apply chain in pure pytorch (slow, but reference)
    x_p = x.index_select(-1, perm_in_inv)
    x_pR = _apply_block_diag(x_p, R_in_blocks, block_size_in)
    x_pRp = x_pR.index_select(-1, perm_in)
    y_pre = x_pRp @ W.T
    y_pre_p = y_pre.index_select(-1, perm_out_inv)
    y_pre_pR = _apply_block_diag(y_pre_p, R_out_blocks, block_size_out)
    y = y_pre_pR.index_select(-1, perm_out)
    return y
```

- [ ] **Step 1.2: Write tests proving equivalence with existing POETLinear**

For each `(in_features, out_features)` pair where `in == out` and `block_size`
divides both, test that the reference matches upstream `POETLinear.forward`
within tolerance. Use small dims for CPU speed.

```python
def test_reference_matches_poet_linear_when_block_sizes_equal():
    # in=out=32, block_size=8 → reference with bs_in=bs_out=8 should match upstream.
    ...
```

- [ ] **Step 1.3: Run, verify PASS**

CPU. ~5 tests. All pass.

- [ ] **Step 1.4: Commit**

```
feat(poet): pure-pytorch reference for decoupled-block POET forward
```

## Task 2: Decoupled Cayley (two calls instead of fused)

**Files:**
- Modify: `third_party/poet_torch/poet_layer.py`
- Modify: `tests/unit/test_poet_decoupled.py`

- [ ] **Step 2.1: Add `get_weight_poet_decoupled`**

```python
def get_weight_poet_decoupled(oft_R_in, oft_R_out,
                              block_size_in, block_size_out,
                              rows_in, cols_in, rows_out, cols_out):
    Q_in  = pytorch_skew_symmetric(oft_R_in,  block_size_in,  rows_in,  cols_in)
    Q_out = pytorch_skew_symmetric(oft_R_out, block_size_out, rows_out, cols_out)
    R_in,  _ = torch.ops.poet.cayley(Q_in)
    R_out, _ = torch.ops.poet.cayley(Q_out)
    return R_out, R_in
```

Two separate Cayley kernel calls. Each consumes a `(r, bs, bs)` tensor with
its own `bs`. Existing Cayley op signature is fine — it handles any `B`.

- [ ] **Step 2.2: Add parity test (GPU)**

Verify `get_weight_poet_decoupled` with equal block sizes produces same
result as `get_weight_poet`.

- [ ] **Step 2.3: Run, verify PASS on GPU**

- [ ] **Step 2.4: Commit**

```
feat(poet): decoupled Cayley computation (two kernel calls)
```

## Task 3: New `chain_layer` Triton op accepting two block sizes

This is the deepest piece of work. The existing kernel hardcodes a single
`block_size`. We need a variant that accepts `block_size_in` and `block_size_out`
and uses each on its respective side.

**Files:**
- Modify: `third_party/poet_torch/poet_ops.py`
- Modify: `third_party/poet_torch/poet_layer.py`

- [ ] **Step 3.1: Read the existing kernel carefully**

`chain_layer_checkpoint_mem_o2` in `poet_ops.py` is implemented as a series
of view/reshape + matmul ops. Notably the relevant block-diag muls are:

```python
xb = x.view(N, rin, block_size)              # reshape input into blocks
xR_r = block_diag_matmul(xb, Rin)            # block-wise R_in mul
xR = xR_r.transpose(...).reshape(N, in)      # collapse blocks
yb_flat = xR @ W.t()                         # big GEMM
yb = yb_flat.view(N, rout, block_size)       # reshape into output blocks
yR = block_diag_matmul(yb, Rout)             # block-wise R_out mul
y = ...                                      # collapse blocks
```

The `block_size` only appears as a reshape dimension. So conceptually, splitting
into `block_size_in` and `block_size_out` is mechanical — just use the right
one for each reshape.

- [ ] **Step 3.2: Implement `chain_layer_checkpoint_mem_o2_decoupled` op**

New op definition with explicit `block_size_in: int, block_size_out: int`
parameters. Mirror the existing implementation but with separated block sizes
on the two sides. Register fake meta + backward (parallel to existing).

- [ ] **Step 3.3: Python wrapper**

`chain_layer_x_checkpoint_mem_o2_decoupled(x, R_in, weight, bias, R_out,
perm_in_inv, perm_in, perm_out, perm_out_inv, block_size_in, block_size_out)`
in `poet_layer.py`.

- [ ] **Step 3.4: GPU parity test (equal block sizes)**

Compare new op output to existing op output when `block_size_in == block_size_out`.
Must be bit-identical within Triton kernel ULP.

- [ ] **Step 3.5: GPU parity test (unequal block sizes)**

Compare new op output to the pure-PyTorch reference from Task 1, with
unequal block sizes.

- [ ] **Step 3.6: Run tests, verify both PASS**

- [ ] **Step 3.7: Backward parity test**

Verify gradients flow correctly. Specifically that `∂L/∂R_in` and `∂L/∂R_out`
match the reference's autograd.

- [ ] **Step 3.8: Commit**

```
feat(poet): decoupled chain_layer kernel (two block sizes)
```

## Task 4: Refactor `POETLinear.__init__` to use decoupled block sizes internally

**Files:**
- Modify: `third_party/poet_torch/poet_layer.py`
- Modify: `tests/unit/test_poet_decoupled.py`

The goal: a single layer class where every storage tensor uses
`(block_size_in, block_size_out)` even if they happen to be equal.

- [ ] **Step 4.1: New constructor signature**

```python
def __init__(self, in_features, out_features, bsz=None,
             block_count=None, bias=False, device=None, dtype=None,
             mem_efficient_mode=False):
    if (bsz is None) == (block_count is None):
        raise ValueError("exactly one of bsz or block_count must be set")
    if bsz is not None:
        block_size_in = block_size_out = bsz
    else:
        if in_features % block_count != 0 or out_features % block_count != 0:
            raise ValueError(f"block_count {block_count} doesn't divide "
                             f"in={in_features} or out={out_features}")
        block_size_in  = in_features  // block_count
        block_size_out = out_features // block_count
    self.block_size_in  = block_size_in
    self.block_size_out = block_size_out
    self.r_in  = in_features  // block_size_in
    self.r_out = out_features // block_size_out
    # Two oft_R parameters, two rows/cols, two perms (unchanged on perm).
    ...
```

- [ ] **Step 4.2: Two `oft_R` parameters**

Replace `self.oft_R` (single tensor) with:
- `self.oft_R_in`: shape `(r_in, n_elems_in)` where `n_elems_in = bs_in*(bs_in-1)/2`
- `self.oft_R_out`: shape `(r_out, n_elems_out)` where `n_elems_out = bs_out*(bs_out-1)/2`

Update `random_init_parameters` to init both. Update `merge_then_reinitialize`
to fold both. Update `perform_permutation` (unchanged structurally).

- [ ] **Step 4.3: Two `rows`/`cols` buffers**

`self.rows_in`, `self.cols_in`, `self.rows_out`, `self.cols_out`.

- [ ] **Step 4.4: Update `forward`**

```python
def forward(self, x):
    R_out, R_in = get_weight_poet_decoupled(
        self.oft_R_in, self.oft_R_out,
        self.block_size_in, self.block_size_out,
        self.rows_in, self.cols_in, self.rows_out, self.cols_out,
    )
    return chain_layer_x_checkpoint_mem_o2_decoupled(
        x, R_in, self.weight, self.bias, R_out,
        self.perm_in_inv, self.perm_in, self.perm_out, self.perm_out_inv,
        self.block_size_in, self.block_size_out,
    )
```

Always use the decoupled ops, regardless of whether the two block sizes are equal.

- [ ] **Step 4.5: Backward-compat test**

Construct `POETLinear(in_features=32, out_features=32, bsz=8)` and
`POETLinear(in_features=32, out_features=32, block_count=4)` — they should
produce identical results given identical `oft_R_in`, `oft_R_out`, `W`, `perms`.

- [ ] **Step 4.6: Decoupled test**

Construct `POETLinear(in_features=32, out_features=64, block_count=4)` —
`block_size_in=8`, `block_size_out=16` — and verify against the pure-PyTorch
reference.

- [ ] **Step 4.7: Run tests, verify PASS**

- [ ] **Step 4.8: Commit**

```
feat(poet): POETLinear stores decoupled block sizes internally; legacy bsz routes through
```

## Task 5: Update `merge_then_reinitialize` for decoupled mode

**Files:**
- Modify: `third_party/poet_torch/poet_layer.py`

- [ ] **Step 5.1: Adapt merge math**

`merge_then_reinitialize` computes the effective weight and folds it back
into `W`. Adapt for two separate `oft_R_in`, `oft_R_out` and two block sizes.

- [ ] **Step 5.2: Adapt `update_permutation`**

The perm-updates are already per-side (in_features, out_features) and so
don't need to change. The `perform_permutation` index_select is unchanged.

- [ ] **Step 5.3: Equivalence test (equal block sizes)**

Run 200 cycles of training on a small toy model with `bsz=8` and with
`block_count=4` (both yield block_size_in==block_size_out==8). After each
merge, the model state should match.

- [ ] **Step 5.4: Commit**

```
feat(poet): merge_then_reinitialize handles decoupled block sizes
```

## Task 6: Extend `CachedPOETLinear` (Mode A) to decoupled mode

**Files:**
- Modify: `src/optim/poet_cache.py`
- Modify: `tests/unit/test_poet_cache.py`

The cache should "just work" with decoupled layers because it already stores
`R_in_*` and `R_out_*` independently. Two changes needed:

- [ ] **Step 6.1: Update `_compute_cayley` to two calls**

The current `_compute_cayley` runs one Cayley call on concatenated R. Split
into separate calls:

```python
def _compute_cayley_decoupled(oft_R_in, oft_R_out, bs_in, bs_out,
                              rows_in, cols_in, rows_out, cols_out):
    Q_in  = pytorch_skew_symmetric(oft_R_in,  bs_in,  rows_in,  cols_in)
    Q_out = pytorch_skew_symmetric(oft_R_out, bs_out, rows_out, cols_out)
    R_in,  _ = torch.ops.poet.cayley(Q_in)
    R_out, _ = torch.ops.poet.cayley(Q_out)
    return R_out, R_in
```

- [ ] **Step 6.2: Update `_get_R_blocks_mode_a`**

```python
def _get_R_blocks_mode_a(self):
    if self._R_cache_version != get_poet_version():
        with torch.enable_grad():
            R_out_full, R_in_full = _compute_cayley_decoupled(
                self.oft_R_in, self.oft_R_out,
                self.block_size_in, self.block_size_out,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            )
        # ... rest unchanged: store full, detach leaves, version bump
```

- [ ] **Step 6.3: Update `_flush_R_grads_to_oft_R` to run TWO VJPs**

```python
def _flush_R_grads_to_oft_R(self):
    if self._R_out_full is None or self._R_in_full is None: return
    gR_out, gR_in = self._R_out_leaf.grad, self._R_in_leaf.grad
    if gR_out is None and gR_in is None: return
    if gR_out is None: gR_out = torch.zeros_like(self._R_out_full)
    if gR_in  is None: gR_in  = torch.zeros_like(self._R_in_full)
    # Two separate VJPs since the autograd graphs are independent.
    (g_in,)  = torch.autograd.grad(self._R_in_full,  self.oft_R_in,  gR_in)
    (g_out,) = torch.autograd.grad(self._R_out_full, self.oft_R_out, gR_out)
    # Write to two separate main_grad buffers.
    for param, g in [(self.oft_R_in, g_in), (self.oft_R_out, g_out)]:
        if hasattr(param, "main_grad") and param.main_grad is not None:
            param.main_grad.copy_(g.to(param.main_grad.dtype))
        else:
            if param.grad is None:
                param.grad = g
            else:
                param.grad.copy_(g)
    self._invalidate_R_cache()
```

- [ ] **Step 6.4: CPU tests (mocked)**

Verify that the cache state machine still works with decoupled layers. Verify
flush invalidates both caches.

- [ ] **Step 6.5: GPU parity tests**

For a decoupled layer (in≠out, block_count=4, so block_size_in≠block_size_out),
verify Mode A's K-microbatch accumulated gradient matches `none`-mode's within
expected bf16 floor.

- [ ] **Step 6.6: Commit**

```
feat(poet): Mode A cache supports decoupled block sizes (two oft_R, two VJPs)
```

## Task 7: DP sync helper handles two oft_R buffers

**Files:**
- Modify: `src/optim/poet.py`

`_sync_oft_R_grads_across_dp` currently iterates over each layer's `oft_R`
and all_reduces `main_grad`. With two `oft_R` params per layer, it must
iterate over both.

- [ ] **Step 7.1: Update sync to iterate over both params**

```python
def _sync_oft_R_grads_across_dp(layers):
    grads = []
    for layer in layers:
        for name in ("oft_R_in", "oft_R_out", "oft_R"):
            p = getattr(layer, name, None)
            if p is not None and hasattr(p, "main_grad") and p.main_grad is not None:
                grads.append(p.main_grad)
    if not grads or not torch.distributed.is_initialized(): return
    # Existing flat-buffer all-reduce, but over the doubled list.
    ...
```

(Iterate over both `oft_R_in` and `oft_R_out` on decoupled layers; fall back
to `oft_R` for legacy layers if anything still uses it.)

- [ ] **Step 7.2: Update test for CPU no-op behavior**

- [ ] **Step 7.3: Commit**

```
feat(poet): DP sync iterates over both oft_R_in and oft_R_out
```

## Task 8: Config + CLI plumbing

**Files:**
- Modify: `src/utils/megatron_args.py`
- Modify: `src/patches/poet_apply_to_model.py`
- Modify: `src/optim/poet_layers.py`
- Modify: `launchers/pretrain_gpt_slm.py`
- Modify: `configs/experiments/optim/poet.yaml`

- [ ] **Step 8.1: Add `--poet-block-count` CLI arg**

In `launchers/pretrain_gpt_slm.py`, add a new argparse arg that's mutually
exclusive with `--poet-block-size`.

- [ ] **Step 8.2: Plumb through Hydra config**

In `configs/experiments/optim/poet.yaml`, document `block_count` as an
alternative to `block_size`:

```yaml
optim:
  poet:
    # Provide EITHER block_size OR block_count, not both.
    block_size: 256        # same block size on both sides (default).
    # block_count: 8       # potentially different block_size_in / block_size_out.
```

- [ ] **Step 8.3: `megatron_args.py` emits the right flag**

Inspect `optim.poet`: if `block_count` is set, emit `--poet-block-count N`;
else emit `--poet-block-size N`.

- [ ] **Step 8.4: `replace_linears_with_poet` accepts `block_count`**

Add `block_count` as a kwarg. Pass it through to `POETLinear(block_count=...)`
instead of `bsz=...` when set.

- [ ] **Step 8.5: Tests**

`test_megatron_args.py`: verify the right argv is emitted for both configs.
`test_poet_layers.py`: verify `replace_linears_with_poet` builds the right
layers in both modes.

- [ ] **Step 8.6: Commit**

```
feat(poet): plumb block_count through Hydra config and CLI
```

## Task 9: Benchmark sweep includes block_count

**Files:**
- Modify: `tools/poet_cache_bench.py`

- [ ] **Step 9.1: Add `--block-count` flag**

Construct layers with `block_count` instead of `block_size`. Compute the
implied `block_size_in`/`block_size_out` and skip rows where they're not
integer multiples.

- [ ] **Step 9.2: Add to default sweep**

Default sweep can include `block_count ∈ {4, 8, 16, 32}` as a knob, with
the existing shapes. Some combos will be invalid; skip.

- [ ] **Step 9.3: Document expected Cayley fraction shifts**

Block_count=8 on a 4096×11008 layer gives `bs_in=512, bs_out=1376`. The
out-side Cayley fraction should be ~2× the equal-block-size case. Confirm
empirically.

- [ ] **Step 9.4: Commit**

```
feat(poet-bench): sweep across block_count configurations
```

## Task 10: Integration smoke — full training run with decoupled blocks

**Files:**
- Modify: `scripts/train_poet.sh` (optional — only if you want to flip
  the default to `block_count`)

- [ ] **Step 10.1: Validation run, equal blocks via block_count**

Run `train_poet.sh optim.poet.block_count=6` (i.e. block_size_in=block_size_out=256
for a 1536-dim model). Compare ~100-step loss curve to a `block_size=256` baseline.
Should match within bf16 noise.

- [ ] **Step 10.2: Validation run, true decoupled**

Run with a different `block_count` that yields unequal block sizes on FFN layers.
Verify training loss is reasonable (not NaN, monotonically decreasing).

- [ ] **Step 10.3: Speed comparison**

Same recipe, run with `block_size=256` and with `block_count` chosen to give
`block_size_out` higher on FFN-up layers. Measure step time and report speedup
or regression.

- [ ] **Step 10.4: Commit + CHANGELOG**

```
feat(poet): decoupled block_count parameterization, end-to-end validated
```

Update `CHANGELOG.md` with the realized speedup numbers.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| The Triton kernel modification is more invasive than expected | Task 3 has multiple substeps. If kernel proves intractable, fall back to a pure-PyTorch chain_layer for decoupled paths (slow but works). Profile reveals whether this matters. |
| Bit-equivalence with old POETLinear when block sizes are equal | Tests in Task 4.5 are gating. If we can't match, the refactor isn't shippable as a drop-in. |
| Mode A flush math (two VJPs instead of one) introduces precision drift | Mode A already accepts bf16-floor drift. Two VJPs are still better than 2K VJPs (baseline). |
| Decoupled DP sync incompatible with existing checkpoint format | `oft_R_in` and `oft_R_out` are new parameter names; existing checkpoints don't have them. Old checkpoints need a migration script (load `oft_R`, split into `oft_R_in`/`oft_R_out` based on r_in/r_out). |
| `update_permutation` and merge: the perm-update logic for unequal block counts | Perm tensors are per-side and don't depend on the block size of the other side. Already correct. |

## Out of scope for v1

- Quantized (Q8/4-bit) layer variants. `POETQuantizedLinear` and `QPOETLinear`
  use the same fused Cayley path; extending them to decoupled is deferred.
- `POETLinearNeurips` and `POETCayleyLinear` — alternate kernel variants in
  `poet_cayley_layer.py`. Not currently used by the slm-research training
  pipeline.
- Per-layer `block_count` (different values per layer). v1 uses one global
  value; per-layer overrides via a config map are a v2 feature.

## Migration path for existing checkpoints

When loading a checkpoint trained with the legacy `bsz=N` POETLinear into a
new `block_count`-aware layer:
- The new layer has `oft_R_in` and `oft_R_out` parameters; the checkpoint has
  a single `oft_R`. Split it: `oft_R_in = oft_R[:r_in]`, `oft_R_out = oft_R[r_in:]`.
- Provide a small migration utility in `src/optim/poet_decoupled_migration.py`
  that runs at checkpoint load time when the old layout is detected.

## Estimated effort

- Task 1–2: 1 day (pure PyTorch reference, validate Cayley split)
- Task 3: 2–3 days (new Triton op + tests)
- Task 4–5: 1 day (refactor `POETLinear`, merge logic)
- Task 6–7: 1 day (Mode A + DP sync)
- Task 8–9: 0.5 day (plumbing + bench)
- Task 10: 0.5 day (validation)

**Total: ~6–7 days of focused work.**
