# lie_ortho Optimizer Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `lie_ortho` POET optimizer (`q_optimizer=lie_ortho`) ~as fast as its `lie_rms` sibling (target: ~1.18 s/step → ~0.3 s/step on 60m/8×H100) **without changing its numerical result**, so the champion run (`5sbgancm`, val/loss 3.5669) is preserved.

**Architecture:** The slowdown is *not* the algorithm. The Newton–Schulz orthogonalization is only ~50 ms/step of real compute, but `LieOrthMomentum.step` costs ~960 ms because, **per skew param, per step**, it (a) rebuilds million-element `torch.triu_indices` on CPU and copies them to GPU, (b) materializes dense `(1, b, b)` skew matrices and scatters into them, (c) does all this in a ~250-iteration Python loop. The fast `lie_rms` avoids dense matrices entirely (it norms the packed vector). Fix = three numerically-exact changes: cache `triu_indices` on-device, and batch all same-block-size params into one `vec_to_skew → NS → skew_to_vec` call instead of a per-param loop. An optional bf16 NS path is included separately (it changes numerics, so it is opt-in and off by default).

**Tech Stack:** PyTorch, pytest. CPU-only unit tests (the optimizer math runs on CPU); one GPU validation step run by the user.

**Test runner:** all pytest/py_compile commands use the slm_env venv:
`PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`

---

## Background — measured root cause (already verified)

Step times (60m/40tpp, 8×H100, W&B `perf/step_time_s`):

| config | s/step | note |
|---|---|---|
| adam | 0.127 | baseline |
| `poet_lie_rms` (lie_algebra) | 0.272 | RMS-scales the packed vector — no dense matrices |
| `poet_lie_orth` (lie_ortho) | **1.180** | orthogonalizes per-block — the +0.91 s |

Decomposition runs proved it is **not** head-alignment (lie_algebra 0.272→0.286 head on; lie_ortho 1.180→1.195 head off). Compute-budget check: NS arithmetic is ~50 ms/step (fp32) vs ~910 ms measured → ~18× overhead, i.e. plumbing, not math. Hotspots, all in [poet_lie_orth.py:124-132](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L124-L132) → [skew_conditioning.py](/lustre/fast/fast/zqiu/slm-research/src/diag/skew_conditioning.py) `vec_to_skew`/`skew_to_vec`:
1. `torch.triu_indices(b,b,1)` rebuilt every call (CPU tensor → implicit H2D copy of ~1.2M int64 for `b=1536`).
2. dense `(1,b,b)` alloc + advanced-index scatter, per param.
3. ~250-param Python loop (kernel-launch latency).
`block_count=1` makes the FFN blocks `b=1536` (worst case), but changing `block_count` alters the model, so it is out of scope here.

---

## File Structure

- `tools/lie_orth_profile.py` — **new**. Standalone GPU microbenchmark that times a `LieOrthMomentum.step()` on representative 60m shapes and isolates (full step) vs (skew↔vec conversions) vs (NS matmuls). Confirms the hotspot before/after each change. One responsibility: profiling.
- `src/diag/skew_conditioning.py` — **modify** `vec_to_skew` / `skew_to_vec` to use an on-device cached `triu_indices` helper `_triu_idx(b, device)`. Pure speedup, numerically identical.
- `src/optim/poet_lie_orth.py` — **modify** `LieOrthMomentum.step` to batch all same-block-size skew params into one `vec_to_skew → orthogonalize → skew_to_vec` per block size, instead of one call per param. Numerically identical (batched NS == per-param NS, already covered by `test_orthogonalize_skew_direction_batches_per_block`).
- `src/optim/poet_skew_muon.py` — **modify** (Task 4, optional) `orthogonalize_skew_blocks` / `orthogonalize_skew_direction` to accept an optional `compute_dtype` (bf16) for the matmul loop.
- `src/optim/poet.py`, `src/utils/megatron_args.py`, `launchers/pretrain_gpt_slm.py` — **modify** (Task 4, optional) to thread a `poet_lie_ortho_compute_dtype` flag (default `fp32`, preserves results).
- `tests/unit/test_diag_skew_conditioning.py` — **modify**: add cache-correctness tests.
- `tests/unit/test_poet_lie_orth.py` — **modify**: add batched-step equivalence tests.

---

## Task 1: Profiler to confirm the hotspot (GPU, user-run)

**Files:**
- Create: `tools/lie_orth_profile.py`

- [ ] **Step 1: Write the profiler**

```python
# tools/lie_orth_profile.py
"""Microbenchmark for LieOrthMomentum.step on representative 60m shapes.

Isolates: (a) full optimizer step, (b) skew<->vec conversions only, (c) NS only.
Run on a GPU node (uses the training env). CPU run works too but is not
representative of the H100 timing.

    PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python
    $PY tools/lie_orth_profile.py --device cuda --steps 30
"""
from __future__ import annotations

import argparse
import time

import torch

from src.diag.skew_conditioning import skew_to_vec, vec_to_skew
from src.optim.poet_lie_orth import LieOrthMomentum
from src.optim.poet_skew_muon import orthogonalize_skew_direction

# 60m, block_count=1: per layer ~3 FFN R_out blocks of 1536 and ~9 blocks of 512.
LAYERS = 18
BLOCK_SIZES = ([1536] * 3 + [512] * 9) * LAYERS


def _make_params(device):
    ps = []
    for b in BLOCK_SIZES:
        ne = b * (b - 1) // 2
        p = torch.nn.Parameter(torch.zeros(1, ne, device=device))
        ps.append(p)
    return ps


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    dev = args.device

    ps = _make_params(dev)
    grads = [torch.randn_like(p) for p in ps]
    opt = LieOrthMomentum(
        [dict(params=ps, use_skew=True, side="out", lr=3e-3)],
        ortho_c=8.0,
        ortho_method="muon",
        ortho_ns_steps=5,
    )

    # (a) full step
    for p, g in zip(ps, grads):
        p.grad = g
    opt.step()  # warmup (allocs buffers)
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for p, g in zip(ps, grads):
            p.grad = g
        opt.step()
    _sync(dev)
    full = (time.perf_counter() - t0) / args.steps

    # (b) conversions only: vec_to_skew -> skew_to_vec round-trip
    dirs = [torch.randn(1, b * (b - 1) // 2, device=dev) for b in BLOCK_SIZES]
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for d, b in zip(dirs, BLOCK_SIZES):
            skew_to_vec(vec_to_skew(d, b), b)
    _sync(dev)
    conv = (time.perf_counter() - t0) / args.steps

    # (c) NS only (on pre-built dense skew)
    skews = [vec_to_skew(d, b) for d, b in zip(dirs, BLOCK_SIZES)]
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for s in skews:
            orthogonalize_skew_direction(s, method="muon", ns_steps=5)
    _sync(dev)
    ns = (time.perf_counter() - t0) / args.steps

    print(f"device={dev} steps={args.steps}")
    print(f"  full step          : {full*1000:8.1f} ms")
    print(f"  conversions only   : {conv*1000:8.1f} ms")
    print(f"  NS only            : {ns*1000:8.1f} ms")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile-check it**

Run: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python; $PY -m py_compile tools/lie_orth_profile.py`
Expected: no output (exit 0).

- [ ] **Step 3: (User, GPU) run the baseline profile**

Run: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python; $PY tools/lie_orth_profile.py --device cuda --steps 30`
Expected: `conversions only` ≫ `NS only` (the plan's premise — most of the ~900 ms is conversions, NS is tens of ms). Record the numbers; re-run after Tasks 2–3 to measure the win.

- [ ] **Step 4: Commit**

```bash
git add tools/lie_orth_profile.py
git commit -m "perf(poet): add lie_ortho step microbenchmark to isolate the hotspot"
```

---

## Task 2: Cache triu_indices on-device (numerically exact)

**Files:**
- Modify: `src/diag/skew_conditioning.py`
- Test: `tests/unit/test_diag_skew_conditioning.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_diag_skew_conditioning.py`:

```python
def test_triu_idx_is_cached():
    from src.diag.skew_conditioning import _TRIU_CACHE, _triu_idx

    _TRIU_CACHE.clear()
    a = _triu_idx(8, torch.device("cpu"))
    b = _triu_idx(8, torch.device("cpu"))
    assert a[0] is b[0] and a[1] is b[1]  # second call reuses, does not rebuild
    assert (8, "cpu") in _TRIU_CACHE


def test_vec_to_skew_correct_after_caching():
    import torch

    from src.diag.skew_conditioning import skew_to_vec, vec_to_skew

    torch.manual_seed(0)
    b = 8
    vec = torch.randn(3, b * (b - 1) // 2)
    q = vec_to_skew(vec, b)
    assert torch.allclose(q, -q.transpose(-2, -1))           # skew-symmetric
    rows, cols = torch.triu_indices(b, b, 1)                  # reference (uncached)
    assert torch.allclose(q[:, rows, cols], vec)             # entries placed correctly
    assert torch.allclose(skew_to_vec(q, b), vec)            # round-trips
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python; $PY -m pytest tests/unit/test_diag_skew_conditioning.py::test_triu_idx_is_cached -v`
Expected: FAIL with `ImportError` / `cannot import name '_triu_idx'`.

- [ ] **Step 3: Add the on-device cache and use it**

In `src/diag/skew_conditioning.py`, after the imports add:

```python
# Cache of strictly-upper-triangular indices, keyed by (block_size, device-string),
# built directly on the target device. vec_to_skew/skew_to_vec are called per skew
# param per optimizer step; rebuilding triu_indices (and copying it H2D) every call
# is the dominant cost of the lie_ortho optimizer (see
# docs/superpowers/plans/2026-06-06-lie-orth-optimizer-speedup.md).
_TRIU_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _triu_idx(b: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(b), str(device))
    cached = _TRIU_CACHE.get(key)
    if cached is None:
        idx = torch.triu_indices(b, b, 1, device=device)
        cached = (idx[0].contiguous(), idx[1].contiguous())
        _TRIU_CACHE[key] = cached
    return cached
```

Then change `vec_to_skew` to replace `rows, cols = torch.triu_indices(b, b, 1)` with:

```python
    rows, cols = _triu_idx(b, vec.device)
```

and change `skew_to_vec` to replace `rows, cols = torch.triu_indices(b, b, 1)` with:

```python
    rows, cols = _triu_idx(b, skew.device)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python; $PY -m pytest tests/unit/test_diag_skew_conditioning.py tests/unit/test_poet_lie_orth.py -v`
Expected: PASS (new cache tests pass; all existing skew/lie_orth tests still pass — the change is numerically identical).

- [ ] **Step 5: Commit**

```bash
git add src/diag/skew_conditioning.py tests/unit/test_diag_skew_conditioning.py
git commit -m "perf(poet): cache on-device triu_indices in vec_to_skew/skew_to_vec"
```

---

## Task 3: Batch same-block-size skew params in LieOrthMomentum.step (numerically exact)

**Files:**
- Modify: `src/optim/poet_lie_orth.py` (the `use_skew` branch of `step`, [poet_lie_orth.py:86-132](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L86-L132))
- Test: `tests/unit/test_poet_lie_orth.py`

- [ ] **Step 1: Write the failing equivalence tests**

Add to `tests/unit/test_poet_lie_orth.py`:

```python
def test_batched_step_matches_solo_steps_same_block_size():
    # Two skew params of the SAME block size in one group must get the same update
    # batched-together as they would stepped alone (batched NS == per-param NS).
    torch.manual_seed(0)
    b = 8
    ne = b * (b - 1) // 2
    gA = torch.randn(2, ne)
    gB = torch.randn(3, ne)

    pA = nn.Parameter(torch.zeros(2, ne))
    pA.grad = gA.clone()
    pB = nn.Parameter(torch.zeros(3, ne))
    pB.grad = gB.clone()
    LieOrthMomentum(
        [dict(params=[pA, pB], use_skew=True, side="out", lr=0.1)],
        ortho_c=0.05,
        ortho_method="muon",
        ortho_ns_steps=5,
    ).step()

    pA2 = nn.Parameter(torch.zeros(2, ne))
    pA2.grad = gA.clone()
    _make_opt(pA2, 0.1, 0.05).step()
    pB2 = nn.Parameter(torch.zeros(3, ne))
    pB2.grad = gB.clone()
    _make_opt(pB2, 0.1, 0.05).step()

    assert torch.allclose(pA.data, pA2.data, atol=1e-6)
    assert torch.allclose(pB.data, pB2.data, atol=1e-6)


def test_batched_step_handles_mixed_block_sizes():
    # Different block sizes in one group -> separate buckets; each matches its solo run.
    torch.manual_seed(1)
    b1, b2 = 8, 6
    ne1, ne2 = b1 * (b1 - 1) // 2, b2 * (b2 - 1) // 2
    g1 = torch.randn(1, ne1)
    g2 = torch.randn(1, ne2)

    p1 = nn.Parameter(torch.zeros(1, ne1))
    p1.grad = g1.clone()
    p2 = nn.Parameter(torch.zeros(1, ne2))
    p2.grad = g2.clone()
    LieOrthMomentum(
        [dict(params=[p1, p2], use_skew=True, side="out", lr=0.1)],
        ortho_c=0.05,
        ortho_method="muon",
        ortho_ns_steps=5,
    ).step()

    p1b = nn.Parameter(torch.zeros(1, ne1))
    p1b.grad = g1.clone()
    _make_opt(p1b, 0.1, 0.05).step()
    p2b = nn.Parameter(torch.zeros(1, ne2))
    p2b.grad = g2.clone()
    _make_opt(p2b, 0.1, 0.05).step()

    assert torch.allclose(p1.data, p1b.data, atol=1e-6)
    assert torch.allclose(p2.data, p2b.data, atol=1e-6)


def test_batched_step_alternating_writes_only_active_side():
    # Alternating: step 0 writes 'out' only, step 1 writes 'in' only; momentum accrues both.
    torch.manual_seed(2)
    b = 8
    ne = b * (b - 1) // 2
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_in.grad = torch.randn(1, ne)
    p_out = nn.Parameter(torch.zeros(1, ne))
    p_out.grad = torch.randn(1, ne)
    opt = LieOrthMomentum(
        [
            dict(params=[p_in], use_skew=True, side="in", lr=0.1),
            dict(params=[p_out], use_skew=True, side="out", lr=0.1),
        ],
        ortho_c=0.05,
        ortho_method="muon",
        ortho_ns_steps=5,
        alternating=True,
    )
    opt.step()  # _alt_step 0 -> active "out"
    assert p_out.data.abs().sum() > 0 and torch.allclose(p_in.data, torch.zeros_like(p_in))
    p_in.grad = torch.randn(1, ne)
    p_out.grad = torch.randn(1, ne)
    p_out.data.zero_()  # simulate the per-step fold
    opt.step()  # _alt_step 1 -> active "in"
    assert p_in.data.abs().sum() > 0 and torch.allclose(p_out.data, torch.zeros_like(p_out))
```

- [ ] **Step 2: Run them to verify the new ones fail / behavior is correct**

Run: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python; $PY -m pytest tests/unit/test_poet_lie_orth.py -v`
Expected: the three new tests PASS already on the *current* per-param code (it is equivalent by construction). This is intentional: they are the **equivalence guardrail** that must keep passing after the refactor in Step 3. (If any fails now, stop — the assumption is wrong.)

- [ ] **Step 3: Replace the per-param loop with a batched implementation**

In `src/optim/poet_lie_orth.py`, replace the entire `if group["use_skew"]:` block (the body from `side = group["side"]` through the `p.add_(gen.to(p.dtype), alpha=lr)` line) with:

```python
            if group["use_skew"]:
                side = group["side"]
                b1, b2, eps, v_mode = group["b1"], group["b2"], group["eps"], group["v_mode"]
                # Pass 1 (per param): update momentum (and 2nd moment) for ALL params,
                # collect the to-be-written direction bucketed by block size so the
                # expensive vec_to_skew -> orthogonalize -> skew_to_vec runs ONCE per
                # block size instead of once per param.
                buckets: dict[int, list] = {}
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    g = g.float()
                    st = self.state[p]
                    if "lie_m" not in st:
                        st["lie_m"] = torch.zeros_like(g)
                        if self.ortho_use_second_moment:
                            if v_mode == "scalar":
                                st["lie_v"] = torch.zeros(
                                    g.shape[0], 1, dtype=g.dtype, device=g.device
                                )
                            else:
                                st["lie_v"] = torch.zeros_like(g)
                    m = st["lie_m"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    if self.ortho_use_second_moment:
                        v = st["lie_v"]
                        if v_mode == "scalar":
                            v.mul_(b2).add_(2.0 * (g * g).sum(dim=-1, keepdim=True), alpha=1 - b2)
                        else:
                            v.mul_(b2).add_(g * g, alpha=1 - b2)
                    if self.alternating and side != active:
                        continue
                    A_dir = -m / (v.sqrt() + eps) if self.ortho_use_second_moment else -m
                    bsz = block_size_from_nelems(A_dir.shape[1])
                    buckets.setdefault(bsz, []).append((p, A_dir))
                # Pass 2 (per block size): one batched orthogonalization, then scatter
                # the result back to each param (scaled by its group lr).
                for bsz, items in buckets.items():
                    A_cat = torch.cat([a for _, a in items], dim=0)  # (sum_nb, n_elems)
                    X = orthogonalize_skew_direction(
                        vec_to_skew(A_cat, bsz),
                        method=self.ortho_method,
                        ns_steps=self.ortho_ns_steps,
                    )
                    gen_cat = skew_to_vec(self.ortho_c * X, bsz)  # (sum_nb, n_elems)
                    off = 0
                    for p, a in items:
                        nb = a.shape[0]
                        p.add_(gen_cat[off : off + nb].to(p.dtype), alpha=lr)
                        off += nb
```

(The `lr = group["lr"]` line above the `if group["use_skew"]:` stays; the `else:` AdamW branch is unchanged.)

- [ ] **Step 4: Run the full lie_orth + skew suites to verify equivalence held**

Run: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python; $PY -m pytest tests/unit/test_poet_lie_orth.py tests/unit/test_diag_skew_conditioning.py tests/unit/test_poet_lie_momentum.py -v`
Expected: PASS (all original behavior tests + the three new batching guardrails).

- [ ] **Step 5: (User, GPU) re-run the profiler to confirm the win**

Run: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python; $PY tools/lie_orth_profile.py --device cuda --steps 30`
Expected: `full step` drops sharply vs the Task 1 baseline (conversions now batched into a few calls; indices cached).

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_lie_orth.py tests/unit/test_poet_lie_orth.py
git commit -m "perf(poet): batch lie_ortho skew params by block size in step()"
```

---

## Task 4 (OPTIONAL): bf16 Newton–Schulz (opt-in; changes numerics)

> Only do this if Task 1/3 profiling shows NS matmuls are still a meaningful slice. It is a small win (~tens of ms) and it **changes the numerical result**, so it defaults OFF and the champion stays fp32.

**Files:**
- Modify: `src/optim/poet_skew_muon.py`, `src/optim/poet_lie_orth.py`, `src/optim/poet.py`, `src/utils/megatron_args.py`, `launchers/pretrain_gpt_slm.py`
- Test: `tests/unit/test_poet_skew_muon.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_poet_skew_muon.py`:

```python
def test_bf16_orthogonalize_close_to_fp32_and_skew():
    import torch

    from src.diag.skew_conditioning import vec_to_skew
    from src.optim.poet_skew_muon import orthogonalize_skew_direction

    torch.manual_seed(0)
    b = 8
    M = vec_to_skew(torch.randn(2, b * (b - 1) // 2), b)
    fp32 = orthogonalize_skew_direction(M, method="muon", ns_steps=5)
    bf16 = orthogonalize_skew_direction(M, method="muon", ns_steps=5, compute_dtype=torch.bfloat16)
    assert torch.allclose(bf16, -bf16.transpose(-2, -1), atol=1e-2)  # stays skew
    assert torch.allclose(bf16, fp32, atol=3e-2)                     # close to fp32
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python; $PY -m pytest tests/unit/test_poet_skew_muon.py::test_bf16_orthogonalize_close_to_fp32_and_skew -v`
Expected: FAIL with `unexpected keyword argument 'compute_dtype'`.

- [ ] **Step 3: Add the compute_dtype param**

In `src/optim/poet_skew_muon.py`, change `orthogonalize_skew_blocks` signature and body:

```python
def orthogonalize_skew_blocks(
    Q: torch.Tensor, ns_steps: int, compute_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Batched quintic Newton-Schulz over a (num_blocks, b, b) batch.

    compute_dtype: if given (e.g. torch.bfloat16), run the NS matmuls in that dtype
    and cast back to Q's dtype (faster on tensor cores; the caller re-skews). Default
    None = Q's dtype (fp32 path, exact-reproducible).
    """
    work_dtype = compute_dtype or Q.dtype
    norm = torch.linalg.matrix_norm(Q, ord="fro", dim=(-2, -1), keepdim=True)
    X = (Q / (norm + 1e-7)).to(work_dtype)
    for _ in range(ns_steps):
        A = X @ X.transpose(-2, -1)
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    return X.to(Q.dtype)
```

Then thread it through `orthogonalize_skew_direction` — change its signature to add `compute_dtype: torch.dtype | None = None` and pass it to the `muon` branch:

```python
    if method == "muon":
        X = orthogonalize_skew_blocks(A, ns_steps, compute_dtype=compute_dtype)
        return 0.5 * (X - X.transpose(-2, -1))
```

- [ ] **Step 4: Wire an opt-in flag through the optimizer + config**

In `src/optim/poet_lie_orth.py`: add `ortho_compute_dtype: torch.dtype | None = None` to `LieOrthMomentum.__init__`, store `self.ortho_compute_dtype = ortho_compute_dtype`, and pass `compute_dtype=self.ortho_compute_dtype` into the `orthogonalize_skew_direction(...)` call in the batched Pass 2.

In `launchers/pretrain_gpt_slm.py` (near the other `--poet-lie-ortho-*` flags, [pretrain_gpt_slm.py:98-101](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L98-L101)):

```python
    group.add_argument(
        "--poet-lie-ortho-compute-dtype", choices=["fp32", "bf16"], default="fp32"
    )
```

In `src/optim/poet.py` where `LieOrthMomentum(...)` is constructed ([poet.py:606](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L606)), add:

```python
            ortho_compute_dtype=(
                torch.bfloat16
                if getattr(config, "poet_lie_ortho_compute_dtype", "fp32") == "bf16"
                else None
            ),
```

In `src/utils/megatron_args.py` near the other `lie_ortho` reads ([megatron_args.py:315-341](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L315-L341)), emit the flag from config:

```python
        poet_args += ["--poet-lie-ortho-compute-dtype", str(poet.get("lie_ortho_compute_dtype", "fp32"))]
```

- [ ] **Step 5: Run tests + compile-check**

Run: `PY=/lustre/fast/fast/zqiu/slm_env/.venv/bin/python; $PY -m pytest tests/unit/test_poet_skew_muon.py tests/unit/test_poet_lie_orth.py -v && $PY -m py_compile src/optim/poet.py src/utils/megatron_args.py launchers/pretrain_gpt_slm.py`
Expected: PASS, no compile errors.

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_skew_muon.py src/optim/poet_lie_orth.py src/optim/poet.py src/utils/megatron_args.py launchers/pretrain_gpt_slm.py tests/unit/test_poet_skew_muon.py
git commit -m "perf(poet): optional bf16 Newton-Schulz for lie_ortho (opt-in, default fp32)"
```

---

## Task 5: GPU validation (user-run) + record in tracker

**Files:**
- Modify: `POET_dev.md` (record the new step time)

- [ ] **Step 1: (User, GPU) re-run the champion and confirm parity + speed**

Run:
```bash
codexlog poet_lie_orth_fast bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 \
  optim.poet.lie_ortho_c=8
```
Expected: `perf/step_time_s` drops from ~1.18 s toward ~0.3 s; the early-step `train/loss` curve tracks the original `5sbgancm` (the default fp32 path is numerically exact, so loss should match closely). If you also enable `optim.poet.lie_ortho_compute_dtype=bf16`, expect a small extra speedup and a slightly different (not identical) curve.

- [ ] **Step 2: Record the measured step time in the tracker**

In `POET_dev.md` §2.1 / §2.5, add the post-optimization `lie_ortho` step time next to the old 1.18 s so the tracker reflects the fix. (Exact numbers come from Step 1.)

- [ ] **Step 3: Commit**

```bash
git add POET_dev.md
git commit -m "docs(poet): record lie_ortho post-speedup step time"
```

---

## Self-Review

**Spec coverage:**
- Hotspot 1 (triu_indices rebuilt + H2D) → Task 2 (on-device cache). ✓
- Hotspot 3 (per-param Python loop / dense materialization) → Task 3 (batch by block size). ✓
- Hotspot driver fp32 NS → Task 4 (optional bf16, opt-in). ✓
- "Don't change the result" → Tasks 2 & 3 are numerically exact (cache = same indices; batched NS == per-param NS, guarded by `test_batched_step_matches_solo_steps_same_block_size` + the existing `test_orthogonalize_skew_direction_batches_per_block`); bf16 is opt-in and off by default. ✓
- Confirm-before-fix → Task 1 profiler runs first; re-run after Task 3. ✓
- `block_count` (model change) explicitly out of scope. ✓

**Placeholder scan:** none — every code step shows complete content; every run step has an exact command + expected output. GPU-only steps are marked "(User, GPU)" per the repo's GPU policy.

**Type/name consistency:** `_triu_idx(b, device)` / `_TRIU_CACHE` defined in Task 2 and used by `vec_to_skew`/`skew_to_vec`; `buckets`/`A_cat`/`gen_cat` local to Task 3; `compute_dtype` added to `orthogonalize_skew_blocks` (Task 4 Step 3) and consumed by `orthogonalize_skew_direction` (same step) and `LieOrthMomentum` (Step 4); flag name `--poet-lie-ortho-compute-dtype` ↔ config key `lie_ortho_compute_dtype` ↔ `config.poet_lie_ortho_compute_dtype` consistent across launcher/megatron_args/poet.py. Batched `orthogonalize_skew_direction(vec_to_skew(A_cat, bsz), ...)` matches its `(num_blocks, b, b)` contract.

**Risk note:** Task 3 is the structural change. Its equivalence is guaranteed by (a) cat/split preserving param order and (b) batched NS == stacked single NS (already an existing test). The three new guardrail tests are written to pass on the *current* code first (Step 2), then must keep passing after the refactor (Step 4) — that is the equivalence proof.
