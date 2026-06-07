# POET Merge: Batched Cayley + Replicated (no-broadcast) Fold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the per-step POET merge cost two ways — (1) **batch the launch-bound Cayley calls** across layers that share a block size (collapsing ~72 tiny head-side Cayley launches/step into a handful), and (2) **fold on every DP rank instead of rank-0-then-broadcast** (zero merge communication), valid because the fold is a deterministic function of DP-identical `(oft_R, W)`.

**Architecture:** The merge currently runs `merge_then_reinitialize()` per layer on rank 0, then broadcasts every weight/perm to all ranks ([poet_merge_step.py:_run_merge](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L235)). We (a) split each layer's merge into "build R via Cayley" (`_merge_R`, exists) and "fold given R" (new `_fold_with_R`); (b) add a batched orchestrator that groups cayley layers by block size and issues **one** Cayley per (group, side) for small blocks; (c) run that orchestrator on **all ranks** and drop the broadcast when permutations are not being resampled (`reinit_perm=False`). Both are **bit-identical** to today's merge (Cayley is per-block independent; replicated deterministic fold keeps DP replicas in sync exactly as DDP keeps weights in sync without broadcasting them).

**Tech Stack:** PyTorch (custom autograd already in place), the vendored `poet_torch` package, Megatron `train_step` patch, `torch.distributed`, pytest (CPU). The real Cayley is a Triton GPU op (`torch.ops.poet.cayley`); CPU tests use the pure-torch reference `cayley_batch` (the Neumann series the Triton kernel implements) via a `cayley_fn` injection seam.

**Relation to the existing distributed-merge plan (`660ab37`):** that plan enumerated distribution modes A/B/C. This plan commits to the **replicate** mode for the deployed replicated-DDP regime (`use_distributed_optimizer=False`, `no_shard`) and supersedes the "shard-then-all-gather" mode for that regime — a full-weight all-gather every step scales worse than redundant (parallel, free-bubble) folding, and degenerates naturally into per-shard folding if weights are ever sharded. Batched Cayley is orthogonal and composes with any distribution mode.

**Correctness invariants this relies on (do not break):**
- With `merge_period=1, reinit_period<0` (deployed), `reinit_perm=False` every step → no `randperm` → the merge is a deterministic function of `(oft_R, W)`.
- `oft_R` is DP-identical after the step (grads all-reduced by DDP; `lie_ortho_distributed` all-reduces to an identical result), and `W` was identical coming in → replicated fold yields bit-identical `W` on every rank with no communication. (Embeddings/norms already rely on exactly this; they are never broadcast.)
- Cayley acts **independently per `[b,b]` block**, so concatenating blocks across layers and running one kernel is bit-identical to per-layer calls.

---

## File Structure

- **Modify** `third_party/poet_torch/poet_layer.py` — split `POETLinear.merge_then_reinitialize` into `_merge_R` (exists) + new `_fold_with_R(R_out, R_in, reinit_perm)`; `merge_then_reinitialize` becomes a 2-line wrapper.
- **Modify** `third_party/poet_torch/head_aligned_layer.py` — same split for `HeadAlignedPOETLinear` (perm-free fold).
- **Modify** `src/patches/poet_merge_step.py` — new `_build_R_batched(layers, cayley_fn, max_batch_block)`; rewrite `_run_merge` to (a) use batched build + `_fold_with_R`, (b) replicate on all ranks when `reinit_perm=False`, with env escape hatches.
- **Create** `tests/unit/test_poet_merge_batched.py` — CPU tests: `_fold_with_R` identity no-op; batched R == per-layer R; batched merge == per-layer merge (bit-identical W).

No config/CLI changes: batched merge is bit-identical and always on for cayley layers; escape hatches are env vars (`POET_DISABLE_BATCHED_MERGE`, `POET_FORCE_MERGE_BROADCAST`) for debugging only.

---

## Task 1: Split the per-layer merge into `_merge_R` + `_fold_with_R`

**Files:**
- Modify: `third_party/poet_torch/poet_layer.py` (`POETLinear.merge_then_reinitialize`, lines 686-723)
- Modify: `third_party/poet_torch/head_aligned_layer.py` (`HeadAlignedPOETLinear.merge_then_reinitialize`, lines 252-264)
- Create: `tests/unit/test_poet_merge_batched.py`

- [ ] **Step 1: Write the failing test** (`_fold_with_R` with identity R is a no-op on W and zeros oft_R)

Create `tests/unit/test_poet_merge_batched.py`:

```python
"""CPU tests for the batched / replicated POET merge.

The real Cayley is a Triton GPU op, so these tests build R with the pure-torch
reference cayley_batch (the Neumann series the Triton kernel implements) and inject
it via cayley_fn. Block-diagonal fold ops are pure torch and run on CPU.
"""
import torch
from poet_torch import HeadAlignedPOETLinear, POETLinear
from poet_torch.poet_layer import (
    block_diag_lr_matmul_decoupled,
    cayley_batch,
    pytorch_skew_symmetric,
)


def _identity_R(n_blocks, b):
    return torch.eye(b).unsqueeze(0).repeat(n_blocks, 1, 1)


def test_fold_with_R_identity_is_noop_poetlinear():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    pl = POETLinear(in_features=12, out_features=8, block_count=2, bias=False)
    with torch.no_grad():
        pl.weight.normal_()
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
    W0 = pl.weight.detach().clone()
    R_in = _identity_R(pl.r_in, pl.block_size_in)
    R_out = _identity_R(pl.r_out, pl.block_size_out)
    pl._fold_with_R(R_out, R_in, reinit_perm=False)
    assert torch.allclose(pl.weight, W0, atol=1e-12), (pl.weight - W0).abs().max()
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0


def test_fold_with_R_identity_is_noop_headaligned():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    pl = HeadAlignedPOETLinear(
        in_features=12, out_features=8, head_side="out", head_dim=4,
        resid_block_count=1, bias=False,
    )
    with torch.no_grad():
        pl.weight.normal_()
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
    W0 = pl.weight.detach().clone()
    R_in = _identity_R(pl.r_in, pl.block_size_in)
    R_out = _identity_R(pl.r_out, pl.block_size_out)
    pl._fold_with_R(R_out, R_in, reinit_perm=False)
    assert torch.allclose(pl.weight, W0, atol=1e-12)
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_batched.py -x -q`
Expected: FAIL — `AttributeError: 'POETLinear' object has no attribute '_fold_with_R'`.

- [ ] **Step 3: Extract `_fold_with_R` in `POETLinear`.** In `third_party/poet_torch/poet_layer.py`, replace the body of `merge_then_reinitialize` (lines 686-723) with a wrapper + the extracted method:

```python
    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        R_out, R_in = self._merge_R()
        self._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)

    @torch.no_grad()
    def _fold_with_R(self, R_out, R_in, reinit_perm: bool = True) -> None:
        """Fold given (already-built) rotations into the frozen weight and zero
        oft_R. Split out of merge_then_reinitialize so a batched orchestrator can
        build R for many layers in one Cayley call, then fold each here."""
        W = self.weight.detach().clone()
        tmp = W.t()
        tmp = block_diag_lr_matmul_decoupled(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        if reinit_perm:
            device = self.weight.device
            perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
            perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
            perm_in_inv = torch.argsort(perm_in).to(torch.int32)
            perm_out_inv = torch.argsort(perm_out).to(torch.int32)
            expected = expected.index_select(0, perm_out_inv).index_select(1, perm_in_inv)
            self.weight.detach().copy_(expected)
            self.perm_in.copy_(perm_in)
            self.perm_in_inv.copy_(perm_in_inv)
            self.perm_out.copy_(perm_out)
            self.perm_out_inv.copy_(perm_out_inv)
        else:
            expected = expected.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
            self.weight.detach().copy_(expected)

        self.oft_R_in.zero_()
        self.oft_R_out.zero_()
```

- [ ] **Step 4: Extract `_fold_with_R` in `HeadAlignedPOETLinear`.** In `third_party/poet_torch/head_aligned_layer.py`, replace `merge_then_reinitialize` (lines 252-264) with:

```python
    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        R_out, R_in = self._merge_R()
        self._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)

    @torch.no_grad()
    def _fold_with_R(self, R_out, R_in, reinit_perm: bool = True) -> None:
        # Permutation-free fold (identity Ψ): weight <- (R_in @ Wᵀ @ R_out)ᵀ, then
        # reset generators. reinit_perm is accepted for API parity (no-op here).
        W = self.weight.detach().clone()
        tmp = block_diag_lr_matmul_decoupled(R_in, W.t(), R_out)
        self.weight.detach().copy_(tmp.t())
        self.oft_R_in.zero_()
        self.oft_R_out.zero_()
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_batched.py -x -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/poet_layer.py third_party/poet_torch/head_aligned_layer.py tests/unit/test_poet_merge_batched.py
git commit -m "refactor(poet): split merge into _merge_R + _fold_with_R"
```

---

## Task 2: Batched R-build orchestrator

**Files:**
- Modify: `src/patches/poet_merge_step.py` (add `_build_R_batched`)
- Modify: `tests/unit/test_poet_merge_batched.py`

- [ ] **Step 1: Write the failing test** (batched R == per-layer Cayley, mixed layers/block sizes). Append to `tests/unit/test_poet_merge_batched.py`:

```python
def _mixed_layers():
    """A mix that exercises grouping: 3 head-aligned (small head blocks, same
    head_dim=4) + 2 standard (different block sizes)."""
    layers = []
    for _ in range(3):
        pl = HeadAlignedPOETLinear(in_features=12, out_features=8, head_side="out",
                                   head_dim=4, resid_block_count=1, bias=False)
        layers.append(pl)
    layers.append(POETLinear(in_features=12, out_features=8, block_count=2, bias=False))
    layers.append(POETLinear(in_features=12, out_features=8, block_count=1, bias=False))
    for pl in layers:
        with torch.no_grad():
            pl.weight.normal_()
            pl.oft_R_in.normal_(std=0.1)
            pl.oft_R_out.normal_(std=0.1)
    return layers


def _per_layer_R(pl, cayley_fn):
    qi = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    qo = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    return cayley_fn(qo), cayley_fn(qi)  # (R_out, R_in)


def test_batched_build_R_matches_per_layer():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    from src.patches.poet_merge_step import _build_R_batched

    layers = _mixed_layers()
    built = _build_R_batched(layers, cayley_fn=cayley_batch, max_batch_block=256)
    for pl in layers:
        R_out_ref, R_in_ref = _per_layer_R(pl, cayley_batch)
        R_out_b, R_in_b = built[id(pl)]
        assert torch.allclose(R_out_b, R_out_ref, atol=1e-12), (R_out_b - R_out_ref).abs().max()
        assert torch.allclose(R_in_b, R_in_ref, atol=1e-12), (R_in_b - R_in_ref).abs().max()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_batched.py::test_batched_build_R_matches_per_layer -x -q`
Expected: FAIL — `ImportError: cannot import name '_build_R_batched'`.

- [ ] **Step 3: Implement `_build_R_batched`.** In `src/patches/poet_merge_step.py`, add near the top (after the imports / before `_run_merge`):

```python
def _build_R_batched(layers, cayley_fn=None, max_batch_block: int = 256):
    """Build (R_out, R_in) for every layer, batching the Cayley across layers that
    share a block size on a side (small blocks only). Returns {id(layer): (R_out, R_in)}.

    Cayley acts independently per [b,b] block, so concatenating blocks across layers
    and running one kernel is bit-identical to per-layer calls. Sides with block_size
    > max_batch_block (e.g. block_count=1 dense) are built per-layer to bound the
    transient memory of stacking big blocks.

    cayley_fn(Q[*, b, b]) -> R[*, b, b]; defaults to the Triton op. Tests inject the
    pure-torch cayley_batch.
    """
    import torch
    from poet_torch.poet_layer import pytorch_skew_symmetric

    if cayley_fn is None:
        def cayley_fn(Q):
            return torch.ops.poet.cayley(Q)[0]

    result = {id(pl): [None, None] for pl in layers}  # [R_out, R_in]
    # side_idx 0 -> out, 1 -> in
    for side_idx, side in enumerate(("out", "in")):
        groups = {}  # block_size -> list of (pl, oft, rows, cols)
        for pl in layers:
            if side == "out":
                b, oft, rows, cols = pl.block_size_out, pl.oft_R_out, pl.rows_out, pl.cols_out
            else:
                b, oft, rows, cols = pl.block_size_in, pl.oft_R_in, pl.rows_in, pl.cols_in
            groups.setdefault(int(b), []).append((pl, oft, rows, cols))
        for b, items in groups.items():
            if b <= max_batch_block and len(items) > 1:
                rows, cols = items[0][2], items[0][3]
                skews = [pytorch_skew_symmetric(oft, b, rows, cols) for (_, oft, _, _) in items]
                sizes = [s.shape[0] for s in skews]
                R = cayley_fn(torch.cat(skews, dim=0))  # ONE Cayley for the whole group
                off = 0
                for (pl, _, _, _), n in zip(items, sizes):
                    result[id(pl)][side_idx] = R[off : off + n]
                    off += n
            else:
                for (pl, oft, rows, cols) in items:
                    R = cayley_fn(pytorch_skew_symmetric(oft, b, rows, cols))
                    result[id(pl)][side_idx] = R
    return {k: (v[0], v[1]) for k, v in result.items()}
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_batched.py::test_batched_build_R_matches_per_layer -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_merge_step.py tests/unit/test_poet_merge_batched.py
git commit -m "feat(poet): batched Cayley R-build for the merge"
```

---

## Task 3: Wire batched build + `_fold_with_R` into `_run_merge` (single-process path)

**Files:**
- Modify: `src/patches/poet_merge_step.py` (`_run_merge`, lines 235-271)
- Modify: `tests/unit/test_poet_merge_batched.py`

- [ ] **Step 1: Write the failing test** (end-to-end batched merge == per-layer merge, bit-identical W). Append:

```python
def test_batched_merge_equals_per_layer_merge():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    from src.patches.poet_merge_step import _build_R_batched

    layers = _mixed_layers()
    # snapshot inputs and compute the PER-LAYER reference (cayley_batch + _fold_with_R)
    import copy
    ref_layers = copy.deepcopy(layers)
    for pl in ref_layers:
        R_out, R_in = _per_layer_R(pl, cayley_batch)
        pl._fold_with_R(R_out, R_in, reinit_perm=False)

    # BATCHED path on the originals
    built = _build_R_batched(layers, cayley_fn=cayley_batch, max_batch_block=256)
    for pl in layers:
        R_out, R_in = built[id(pl)]
        pl._fold_with_R(R_out, R_in, reinit_perm=False)

    for pl, ref in zip(layers, ref_layers):
        assert torch.allclose(pl.weight, ref.weight, atol=1e-12), (pl.weight - ref.weight).abs().max()
        assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0
```

- [ ] **Step 2: Run to verify it passes already** (this test only uses Task 1/2 APIs)

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_batched.py::test_batched_merge_equals_per_layer_merge -x -q`
Expected: PASS. (It validates the orchestration contract that `_run_merge` will use; if it fails, fix the batching before wiring it into `_run_merge`.)

- [ ] **Step 3: Rewrite `_run_merge` to use the batched path.** Replace the body of `_run_merge` (lines 235-271) in `src/patches/poet_merge_step.py` with:

```python
def _run_merge(model, dist, iteration: int, reinit_perm: bool = True) -> None:
    import os

    import torch
    from poet_torch import POETLinear

    from src.optim.poet_layers import POETMegatronLinear

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    # Collect the POET layers to merge (same filter as before).
    pls = []
    chunks = model if isinstance(model, list) else [model]
    for m in chunks:
        for _, mod in m.named_modules():
            if not isinstance(mod, POETMegatronLinear):
                continue
            pl = mod.poet_linear
            if not isinstance(pl, POETLinear) or pl.block_size <= 0:
                continue
            pls.append(pl)

    # Escape hatches (debugging only): force the legacy rank-0 + broadcast path,
    # and/or disable Cayley batching.
    force_broadcast = os.environ.get("POET_FORCE_MERGE_BROADCAST") == "1"
    disable_batch = os.environ.get("POET_DISABLE_BATCHED_MERGE") == "1"

    # REPLICATE: when permutations are NOT being resampled, the fold is a
    # deterministic function of DP-identical (oft_R, W), so every rank folds its
    # own replica to a bit-identical result with NO communication (same reason DDP
    # never broadcasts weights). reinit_perm=True (randperm) is rank-divergent, so
    # fall back to rank-0 + broadcast for that (rare/disabled) case.
    replicate = (not reinit_perm) and (not force_broadcast)

    if replicate:
        with torch.no_grad():
            _merge_layers(pls, reinit_perm=False, disable_batch=disable_batch)
        for pl in pls:
            if hasattr(pl, "_invalidate_R_cache"):
                pl._invalidate_R_cache()
        return

    # Legacy path: rank-0 folds, then broadcast (covers reinit_perm=True and the
    # forced escape hatch).
    with torch.no_grad():
        if rank == 0:
            _merge_layers(pls, reinit_perm=reinit_perm, disable_batch=disable_batch)
        if is_dist:
            for pl in pls:
                for buf in (
                    pl.oft_R_in.data, pl.oft_R_out.data, pl.weight.data,
                    pl.perm_in, pl.perm_in_inv, pl.perm_out, pl.perm_out_inv,
                ):
                    dist.broadcast(buf, src=0)
    for pl in pls:
        if hasattr(pl, "_invalidate_R_cache"):
            pl._invalidate_R_cache()


def _merge_layers(pls, reinit_perm: bool, disable_batch: bool) -> None:
    """Fold every layer. With batching on, split layers into cayley (batched
    R-build) and non-cayley (per-layer merge_then_reinitialize, e.g. exp)."""
    if disable_batch:
        for pl in pls:
            pl.merge_then_reinitialize(reinit_perm=reinit_perm)
        return
    cayley_pls = [pl for pl in pls if getattr(pl, "parameterization", "cayley") == "cayley"]
    other_pls = [pl for pl in pls if getattr(pl, "parameterization", "cayley") != "cayley"]
    for pl in other_pls:
        pl.merge_then_reinitialize(reinit_perm=reinit_perm)
    if cayley_pls:
        built = _build_R_batched(cayley_pls)  # default cayley_fn = Triton op
        for pl in cayley_pls:
            R_out, R_in = built[id(pl)]
            pl._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)
```

- [ ] **Step 4: Run the full merge test file + py_compile**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_batched.py -q`
Expected: PASS (all).
Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/poet_merge_step.py`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_merge_step.py tests/unit/test_poet_merge_batched.py
git commit -m "feat(poet): replicated + batched merge in _run_merge"
```

---

## Task 4: CPU regression + GPU correctness/perf handoff

**Files:** (verification only; no GPU runs launched here)

- [ ] **Step 1: Run the POET CPU suite (no regressions)**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_batched.py tests/unit/test_single_step_fast.py tests/unit/test_poet_layers.py -q`
Expected: PASS except the known pre-existing `test_sharded_state_dict_is_deduped_replicated_and_complete` (megatron.core importorskip fails on the CPU env — unrelated).

- [ ] **Step 2: ruff**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m ruff check src/patches/poet_merge_step.py third_party/poet_torch/poet_layer.py third_party/poet_torch/head_aligned_layer.py tests/unit/test_poet_merge_batched.py`
Expected: `All checks passed!`

- [ ] **Step 3: Hand the GPU checks to the user (do NOT launch).** Two things to verify on GPU:

  **(a) Single-GPU bit-equivalence of the merge** — confirms batched Cayley + the refactor match the old merge. Run the same loss A/B as before; loss must overlap the pre-change run within bf16 noise:
  ```bash
  codexlog merge_batch_fast bash scripts/train_poet_lie_orth.sh llama3 \
    optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
    optim.poet.single_step_fast=true
  ```
  Compare loss/ms vs the prior `ss_full_fast` run; expect lower or equal ms/iter and overlapping loss.

  **(b) Multi-GPU cross-rank weight consistency** — the load-bearing check for the replicate change. On a ≥2-GPU run, after a merge step, every rank's frozen `W` must be bit-identical. Add a temporary debug assert (env-gated) to the merge patch for one run, OR run with `POET_FORCE_MERGE_BROADCAST=1` once and confirm the loss matches the replicate run (they must be identical since both are deterministic). Suggested debug assert to drop in `_run_merge` after `_merge_layers` (rank-0 prints max cross-rank drift):
  ```python
  if os.environ.get("POET_CHECK_MERGE_SYNC") == "1" and is_dist:
      for pl in pls[:1]:  # one layer is enough; all share the same determinism
          w = pl.weight.data.clone()
          ref = w.clone(); dist.broadcast(ref, src=0)
          drift = (w - ref).abs().max()
          if rank == 0:
              print(f"[POET] merge cross-rank drift (rank-vs-0): {drift.item():.2e}", flush=True)
  ```
  Acceptance: drift `== 0.0` on every step (proves replicas stay in sync with no broadcast). If non-zero, a kernel is non-deterministic on this hardware → set `POET_FORCE_MERGE_BROADCAST=1` and file it.

- [ ] **Step 4: Final commit (any doc/CHANGELOG updates)**

```bash
git add -A && git commit -m "docs(poet): batched/replicated merge verification notes"
```

---

## Self-Review

- **Spec coverage:** batched Cayley = Task 2 (`_build_R_batched`, grouped by block size, small-blocks-only) + Task 3 (wired in). Replicate-no-broadcast = Task 3 (`replicate` branch, gated on `reinit_perm=False`). Refactor enabling both = Task 1 (`_fold_with_R`). Escape hatches (`POET_DISABLE_BATCHED_MERGE`, `POET_FORCE_MERGE_BROADCAST`) = Task 3. Cross-rank correctness gate = Task 4 Step 3(b).
- **Placeholder scan:** none — every code step is complete. CPU tests inject `cayley_fn=cayley_batch` (real Cayley is Triton/GPU); the GPU steps are explicitly the user's to run, with exact commands and acceptance criteria.
- **Type consistency:** `_fold_with_R(R_out, R_in, reinit_perm)` signature identical in both layer classes, the orchestrator, and tests. `_build_R_batched(layers, cayley_fn, max_batch_block) -> {id(pl): (R_out, R_in)}`; callers always unpack `(R_out, R_in)` in that order (matches `_per_layer_R` and `get_weight_poet_decoupled`'s `(R_out, R_in)` convention). `cayley_fn(Q) -> R` (default does the `[0]` unwrap of the Triton op; tests pass `cayley_batch` which already returns `R`).
- **Determinism caveat is explicit:** replicate is gated on `reinit_perm=False`; `reinit_perm=True` keeps rank-0 + broadcast. If reinit is ever re-enabled with replicate, the follow-up is to seed the perm RNG identically across ranks (sync a seed, not weights) — noted but out of scope here.
- **Composes with distribution:** `_merge_layers` is distribution-agnostic; a future sharded-weight mode would partition `pls` across ranks and call the same `_merge_layers` per shard.
