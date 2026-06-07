# Distributed POET merge — investigation + design

**Date:** 2026-06-07
**Status:** Investigate-only. This document is the deliverable. **No code changes** are
proposed yet — the implementation designs below are specified so they are mechanical to
build *if and when* the measurement (§4) justifies it.
**Target regime:** large model / Kimi-1T direction, high data-parallel (DP) world size.

---

## 1. Problem & current behavior

POET keeps trainable rotation parameters `oft_R_in`/`oft_R_out` per linear and periodically
**merges** them into the (frozen) base `weight`. With `merge_period: 1` (the default in the
best run, `scripts/train_poet_lie_orth.sh`, and in `configs/experiments/optim/poet_lie_head.yaml`)
the merge runs **every step**.

The merge is driven by `_run_merge` in
[`src/patches/poet_merge_step.py`](../../../src/patches/poet_merge_step.py) (lines ~235–271).
Its core, per POET layer:

```python
with torch.no_grad():
    if rank == 0:
        pl.merge_then_reinitialize(reinit_perm=reinit_perm)   # ONLY rank 0 computes
    if is_dist:
        for buf in (oft_R_in, oft_R_out, weight, perm_in, perm_in_inv, perm_out, perm_out_inv):
            dist.broadcast(buf, src=0)                          # everyone copies rank 0's answer
```

`merge_then_reinitialize` (in `third_party/poet_torch/poet_layer.py`) builds the rotation from
the skew params (Cayley / matrix-exp) and applies a block-diagonal left/right matmul to the
weight — **O(d³) per layer** at `block_count=1` (dense full-matrix rotation).

What physically happens each step:

1. **Rank 0** loops over every POET layer doing the O(d³) merge math. Cost = `T_merge`.
2. **Ranks 1…N-1** hit `dist.broadcast` and **block** — they do *zero* merge compute. They wait
   for rank 0, then receive 7 tensors per layer over the network (dominated by the large
   `weight` matrices). Cost = `C`.

So the critical path of **every step** is `T_merge + C`, and all non-zero ranks idle through the
compute.

**Latent correctness note (out of scope to fix here, but recorded):** the broadcast uses `src=0`
over the **default world group**. Under tensor parallelism (TP>1) each TP rank holds a *different*
shard of `weight`, so broadcasting rank 0's shard to all ranks would be **wrong**. Current
experiments run TP=1, where world ≡ the relevant group, so it works today by accident. Both
designs below scope their collective to the DP group, which removes this hazard as a side effect.

---

## 2. The invariant that unlocks everything

The merge runs **after** `train_step` returns. Under the `DistributedOptimizer`
(`distributed_optimizer: true`, e.g. `cluster=h100_de`), the updated bf16 params are
**all-gathered** at the end of the step. Therefore, immediately before the merge:

> Every DP rank holds **bit-identical** `weight` and `oft_R`.

And `merge_then_reinitialize` is a **deterministic pure function** of `(weight, oft_R, perms)`.
Same inputs + same math + same GPU architecture ⇒ same output, on every rank.

**Consequence:** the broadcast in §1 is a *choice*, not a necessity. The authors chose
"rank 0 computes and ships the answer." But since every rank can reproduce the answer from inputs
it already has, you can instead have **every rank compute the merge itself** and skip the network
entirely. This is Approach B (§5).

This invariant is load-bearing for Approach B and is therefore the first thing the investigation
(§4, Phase 0) **verifies empirically** rather than assumes.

---

## 3. Cost model & three regimes

Let `T_merge` = full per-rank merge compute (sum over all POET layers), `C` = broadcast cost
(dominated by weight bytes), `dp` = DP world size.

| Regime | Description | Compute on critical path | Comm on critical path | Total |
|---|---|---|---|---|
| **A — current** | rank 0 computes, broadcast to all | `T_merge` | `C` | `T_merge + C` |
| **B — comm-free** | every rank merges all layers, drop broadcast | `T_merge` | `0` | `T_merge` |
| **C — sharded** | round-robin owners, broadcast results over DP group | `T_merge / dp` | `C` | `T_merge/dp + C` |

Two consequences:

- **B strictly beats A.** Same compute latency (the other ranks were idling at the broadcast
  barrier anyway), minus the entire broadcast. It is a *deletion*, not an addition.
- **B vs C is a genuine trade**, decided by measurement:
  - **C wins** (compute-bound) iff `T_merge/dp + C < T_merge` ⟺ `C < T_merge·(1 − 1/dp)`.
  - **B wins** (comm-bound) otherwise.

**For the Kimi-1T / high-DP target, comm is expected to dominate:** broadcasting full (huge)
weight matrices every step is exactly the cost that explodes as weights and `dp` grow, while C
*keeps* that comm and only shaves compute. So **B is the recommended primary** and **C the
compute-bound fallback**. A packed-buffer `all_reduce` variant (mirroring the lie_ortho optimizer
path exactly) is deliberately **not** pursued: at Kimi scale it requires a full-model-sized extra
fp32 buffer — worst of both worlds.

> **Why not "best of both" (shard compute *and* skip comm)?** Impossible: parallel compute means
> each rank only produces a subset of results, so it *must* receive the rest → comm is required.
> Comm-free *requires* redundant full compute. The trade is real; §4 measures which side wins.

---

## 4. Investigation protocol (what you run)

Goal: produce the numbers that pick B, C, or "do nothing," and de-risk B's invariant. All GPU
runs are yours to launch (per project policy); this section hands you the exact instrumentation
and commands. The instrumentation is **throwaway measurement scaffolding** (env-gated), distinct
from any feature code.

### Phase 0 — verify the §2 invariant (de-risk B)

Before trusting redundant compute, prove the inputs really are bit-identical across DP. Right
before the merge, on each rank compute a checksum of every POET `weight` and `oft_R`
(e.g. `float64` sum of `t.double()` plus `t.abs().sum()`), pack into one tensor, and
`all_reduce(MAX)` and `all_reduce(MIN)` over the DP group; assert `MAX == MIN` elementwise.
Gate behind `POET_MERGE_VERIFY=1`. Run ~20 steps.

- **Pass:** inputs identical → B is safe. Proceed.
- **Fail:** inputs already diverge under A → B unsafe as-is; investigate *why* (the divergence is
  also latent risk for A's own consistency assumptions). Record findings.

### Phase 1 — split merge-compute vs merge-comm vs step

Add CUDA-event timers (`torch.cuda.Event(enable_timing=True)`, with `torch.cuda.synchronize()` at
read-out) around:

- (a) the `merge_then_reinitialize` loop (compute) → `T_merge_ms`,
- (b) the `dist.broadcast` loop (comm) → `C_ms`,
- (c) the whole `train_step` (already wrapped) → `step_ms`.

Gate behind `POET_MERGE_PROFILE=1`. Accumulate over ~100 steps and log mean/median/p95 of each,
plus the derived quantities:

- `merge_fraction = (T_merge_ms + C_ms) / step_ms`
- `projected_B_savings = C_ms / step_ms`
- `projected_C_savings = (T_merge_ms·(1 − 1/dp)) / step_ms`

### Command

Run the best recipe with profiling on (adjust scale/DP to the regime you care about):

```bash
POET_MERGE_VERIFY=1 POET_MERGE_PROFILE=1 \
codexlog lieorth_merge_profile bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true
```

Read `T_merge_ms`, `C_ms`, `step_ms` from `/lustre/home/zqiu/log/lieorth_merge_profile.log`.

### Decision gate

| Observation | Decision |
|---|---|
| `merge_fraction < ~1–2%` | **Do nothing** — not worth the complexity. |
| comm-bound: `C_ms ≳ T_merge_ms·(1 − 1/dp)` | **Ship B** (comm-free). |
| compute-bound: `C_ms < T_merge_ms·(1 − 1/dp)` | **Ship C** (sharded). |
| Phase 0 fails | Resolve input divergence before B; C still available (it forces agreement). |

---

## 5. Design — Approach B (comm-free redundant merge)

**Idea:** every rank computes the merge for every layer; delete the broadcast.

**Change to `_run_merge`** (conceptual; behind the flag in §7):

```python
with torch.no_grad():
    pl.merge_then_reinitialize(reinit_perm=reinit_perm)   # EVERY rank, identically
    # no broadcast — the result is already identical on every rank
# oft_R is zeroed by merge_then_reinitialize locally on each rank (consistent).
```

**Why correct (steady state, `reinit_period=-1`):** inputs are DP-identical (§2, verified in
Phase 0), the merge is deterministic, and `reinit_perm=False` means **no `randperm`** is drawn —
so the only mutated tensors (`weight`, zeroed `oft_R`) end up identical on every rank with zero
communication.

**Unchanged and already correct:**
- `_reset_vanilla_oft_state` (the fp32-master zero + Adam-moment reset) operates on each rank's
  **local** master shard and is independent of who computed the merge — keep as-is on all ranks.
- `_invalidate_R_cache` is local — keep on all ranks.

**Drift anchor (optional safety net):** because B *assumes* rather than *forces* agreement, add an
optional periodic re-broadcast from rank 0 every `K` steps (`K` configurable; `K=0` disables).
This bounds any hypothetical FP drift to `K` steps at negligible amortized comm. Default: enabled
with a large `K` (e.g. 1000) while B is new; can be turned off once Phase 0 + bit-exactness
testing build confidence.

**Risks & mitigations:**
- *Silent divergence* if inputs drift or a kernel is nondeterministic → Phase 0 verification +
  drift anchor + the bit-exactness test in §8.
- *Reinit `randperm` divergence* (each rank draws a different permutation) → does **not** occur at
  `reinit_period=-1`; for the general case see §9 (shared-seed permutations), deferred.

---

## 6. Design — Approach C (sharded compute + sync)

**Idea (your original framing):** split the merge compute round-robin across DP ranks, then sync
results. This is the right choice **only if measurement shows the merge is compute-bound**.

**Two-phase structure** (do **not** interleave compute and broadcast per layer — a per-layer
broadcast inside the compute loop re-serializes everyone at each layer and kills the parallelism):

```python
# Phase 1 — parallel compute, NO comm. Each rank merges only its owned layers.
for i, pl in enumerate(layers):
    if i % dp_world == dp_rank:
        pl.merge_then_reinitialize(reinit_perm=reinit_perm)   # owner folds -> weight, zeros oft_R
    else:
        pl.oft_R_in.zero_(); pl.oft_R_out.zero_()             # match owner's zeroing locally

# Phase 2 — sync. Broadcast each layer's changed weight from its owner over the DP group.
for i, pl in enumerate(layers):
    src = torch.distributed.get_global_rank(dp_group, i % dp_world)
    dist.broadcast(pl.weight.data, src=src, group=dp_group)
    if reinit_perm:                                            # perms only change on reinit
        for buf in (pl.perm_in, pl.perm_in_inv, pl.perm_out, pl.perm_out_inv):
            dist.broadcast(buf, src=src, group=dp_group)
    # oft_R already zeroed everywhere in Phase 1 -> no broadcast needed
```

**Key points:**
- **Scope the collective to `mpu.get_data_parallel_group()`** (not the world group). This is both
  required for the sharding to be correct and removes the TP>1 hazard from §1. Translate the
  DP-local owner index to a global rank via `torch.distributed.get_global_rank(dp_group, …)`.
- **Comm is *reduced* vs A** in fold-only steps: only `weight` is broadcast (not all 7 buffers),
  because `oft_R` is zeroed locally and perms are unchanged when `reinit_perm=False`.
- **Ownership:** round-robin `i % dp_world` for v1. POET linears have non-uniform shapes
  (qkv/o/gate/up/down), so at high `dp` round-robin can imbalance compute; a **cost-aware
  assignment** (sort layers by `weight.numel()` desc, greedily assign each to the least-loaded
  rank) balances it. Noted as a refinement; v1 = round-robin.

---

## 7. Shared implementation concerns

**Single flag, mirroring the existing `lie_ortho_distributed` plumbing** (yaml →
`src/utils/megatron_args.py` → `src/patches/poet_optimizer_setup.py` → config → patch; the
lie_ortho path is the template, see `poet.py` builder ~599–627):

- `optim.poet.merge_distributed: off | comm_free | sharded`
  - `off` (default) → **today's behavior A**, fully backward-compatible.
  - `comm_free` → Approach B.
  - `sharded` → Approach C.
- `optim.poet.merge_reanchor_period: K` (B only; `0` disables the drift anchor).

`_run_merge` reads the resolved mode and dispatches. The DP group / rank / world come from `mpu`
(`get_data_parallel_group`, `get_data_parallel_rank`, `get_data_parallel_world_size`), exactly as
the lie_ortho optimizer already does.

---

## 8. Validation plan (when built)

- **Bit-exactness:** for a fixed seeded state, B and C must produce a `weight` **bit-identical** to
  A's single-rank reference, and identical across all ranks. Assert via cross-rank checksum
  (reuse the Phase-0 machinery). This is achievable precisely because the merge is deterministic.
- **Numerical parity:** short training run; overlay loss curves for A / B / C — must coincide.
- **CPU-testable (run locally, no GPU):** flag plumbing unit tests mirroring the existing
  `tests/unit/test_megatron_args.py` and `tests/unit/test_patch_poet_optimizer_setup.py` for
  `lie_ortho_distributed` (yaml→argv emission, argparse acceptance, config copy, defaults).
- **Throughput:** re-run the Phase-1 profile with the chosen mode on; confirm `step_ms` drops by
  ≈ the projected savings.

---

## 9. Risks & open questions

- **B determinism / drift** — primary risk; mitigated by Phase-0 verification, the drift anchor,
  and the bit-exactness test.
- **Kimi INT4 base weight** — *open*: what is `pl.weight` on the quantized path, and does the
  block-diagonal fold even apply to a quantized base, or is there a dequant/requant around it?
  This must be checked separately before claiming either design works at Kimi scale. Both B and C
  assume `merge_then_reinitialize` is correct for the target weight representation.
- **Reinit `randperm` under B** — for `reinit_period >= 0`, B needs **shared-seed permutations**
  (derive the permutation from a per-iteration seed set identically on all ranks, e.g.
  `torch.manual_seed(base_seed + iteration)` scoped around the `randperm`) so all ranks draw the
  same "random" permutation with zero comm. Deferred (steady-state target uses `reinit_period=-1`).
- **TP>1 merge-math correctness** — pre-existing question (does the block-diagonal rotation align
  with a column/row-parallel weight shard?). Out of scope here; flagged. Today's A path only works
  for TP=1; C's DP-group scoping is necessary-but-not-sufficient for TP>1.

---

## 10. Scope / non-goals

- **Investigate-only:** this doc + the measurement scaffolding are the deliverable. No feature code
  is written until the §4 decision gate justifies it.
- `merge_period > 1` cadence is unchanged.
- Full reinit/TP generality (shared-seed perms, TP>1 merge math) is deferred — steady-state
  (`reinit_period=-1`, TP=1) is designed first; the rest is documented as clearly-scoped extensions.

---

## Appendix — key code references

- Merge driver: [`src/patches/poet_merge_step.py`](../../../src/patches/poet_merge_step.py)
  (`_run_merge` ~235–271; rank-0 compute + broadcast ~252–265; master-state reset
  `_reset_vanilla_oft_state` ~153–232; `_merge_decision` ~48–71).
- Merge math: `third_party/poet_torch/poet_layer.py` (`merge_then_reinitialize` ~687–723).
- Distributed q-update template: `src/optim/poet_lie_orth.py`
  (`_skew_update_buffer` round-robin ~123–166; `step` `all_reduce` over DP group ~186–193).
- Flag plumbing template (`lie_ortho_distributed`): `src/utils/megatron_args.py` ~343–344;
  `src/patches/poet_optimizer_setup.py` ~60; `src/optim/poet.py` builder ~599–627.
- Best run: `scripts/train_poet_lie_orth.sh`; config `configs/experiments/optim/poet_lie_head.yaml`
  (`merge_period: 1`, `reinit_period: -1`, `block_count: 1`).
