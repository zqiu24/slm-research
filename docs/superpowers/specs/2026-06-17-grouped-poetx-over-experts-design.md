# Grouped-POETX over Experts — Design

**Date:** 2026-06-17
**Status:** Design approved; implementation **gated** on a profiler confirmation (see §9).
**Topic:** Batch POET's per-expert rotation backward across the routed MoE experts to remove the dominant POET throughput tax, while keeping POET on every expert.

---

## 1. Goal

Make POET-on-experts materially faster on DeepSeek-3Bv2 **without dropping POET from any routed expert**. The lever is the POETX rotation *backward*: collapse the per-expert orthogonal-gradient computation into one **block-sparse, expert-batched** operation.

Concretely: replace the `2 × num_experts` independent per-expert POETX backward calls per MoE layer with a single grouped module whose backward computes the rotation gradient (a) only on the block diagonal it actually uses (a `block_count×` FLOP cut) and (b) batched over the expert axis (an `E×` launch cut).

Non-goal: speeding up the expert GEMM. The profiler showed the GEMM is *not* the bottleneck (Adam runs the identical SequentialMLP at full speed). The forward stays a per-expert ragged GEMM; only the rotation backward changes.

---

## 2. Motivation & evidence

### 2.1 Profiler finding (recorded, commit 3e52dba)

On the 8-GPU `full` run (64 experts, `grouped_gemm=false` → `SequentialMLP`, DP=8):

- `forward_backward` ≈ **99.5%**, `optimizer` (lie_ortho) ≈ **0.5%**, `merge` ≈ **2.6%**.
- Adam runs the **same unfused SequentialMLP model** (`optim/adam` also sets `unfuse_qkv/unfuse_fc1=true`) at **7.5 TFLOP/s (~24s/iter)**; POET at **4.2 TFLOP/s (~43s/iter)** → POET ~1.79× slower.
- The "distributed lie_ortho is the bottleneck" hypothesis is **refuted**. The entire gap is inside forward/backward and is POET-specific.

The profiled config is **already** on the fast forward-frame path — [poet_lie_orth_alt.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_orth_alt.yaml#L50): `single_step_x: true`, `single_step_fast: true`, `merge_period: 1`. The forward is already a bare GEMM. So the cost is the **backward**, not forward materialization.

### 2.2 Mechanism (read from the code)

[`POETXSingleStepFunction.backward`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_ops.py#L48), per wrapped expert linear, per micro-batch:

| op | shape | cost | POET-specific |
|---|---|---|---|
| `grad_x = grad_y @ Wx` | `[T,out]·[out,in]` | `O(T·d²)` | no (Adam-equivalent input grad) |
| `G = xᵀ @ grad_y` | `[in,T]·[T,out]→[in,out]` | `O(T·d²)` | no (Adam-equivalent weight grad) |
| **`M_in = G @ Wx`** | `[in,out]·[out,in]→[in,in]` | **`O(in²·out)`** | **yes — the tax** |
| **`M_out = Wx @ G`** | `[out,in]·[in,out]→[out,out]` | **`O(out²·in)`** | **yes — the tax** |

The two `M` GEMMs are the entire POET tax: ~`2·d³` per expert linear, **independent of token count `T` and independent of `block_count`**. After top-k routing across 64 experts, tokens-per-expert `T < d`, so the `d³` tax dwarfs the `T·d²` useful work — explaining the 1.79× slowdown.

Crucially, [`_blockdiag_skew_vec`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/single_step.py#L32) reshapes `M` to `[nb, b, nb, b]` and keeps **only the diagonal blocks** — every off-block-diagonal entry of the full `[d,d]` `M` is computed and thrown away. So the current code computes ~`block_count×` more than it uses.

### 2.3 Two levers, composable

1. **Block-sparse `M` (FLOP cut).** Compute only the block-diagonal blocks: `M_in_block_k = G[rows_k,:] @ Wx[:,cols_k] → [b,b]`. Mathematically identical (off-diagonal is discarded anyway); `block_count×` fewer FLOPs.
2. **Batch over experts (launch cut).** Block-sparse turns each `M` into `n_blocks` small `[b,b]` GEMMs → launch-bound. Stacking the experts' frozen weights lets all `E·n_blocks` block-GEMMs fold into one `bmm`.

This design takes **both**, in one grouped module.

---

## 3. Settled decisions

1. **Install via the POET walk** (post-build, pre-DDP) — no Megatron spec/backend change. The walk detects `SequentialMLP` and swaps in a grouped expert path.
2. **POETX champion path only** — forward-frame `single_step_x` + `lie_alternating` both-momenta (`POETXLinear(alternating=True)`), `merge_period=1`, `oft_R≡0` regime. Natural-frame `POETLinear` is out of scope.
3. **Both reductions** — block-sparse `M` *and* expert-batched `bmm`, in the grouped module.

---

## 4. Architecture & install (no Megatron spec change)

`SequentialMLP` holds experts as a `ModuleList` of `MLP`s, each `linear_fc1 → activation → linear_fc2` (with `unfuse_fc1` splitting the gated fc1). Its [forward](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/transformer/moe/experts.py#L783) splits the permuted tokens by `tokens_per_expert` and loops experts.

The walk ([replace_linears_with_poet](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L179)) gains a `SequentialMLP` branch that, pre-DDP:

1. Reads each expert's per-role linear weights, stacks them across `E` into forward-frame grouped buffers (`Wx[E,out,in]`), and builds one `GroupedPOETXLinear` per **role** (fc1, fc2, plus the extra split under `unfuse_fc1`).
2. **Replaces `SequentialMLP.forward`** with a grouped version that runs each role once over all experts.

**Why replace the forward, not just swap linears:** batching the backward requires all experts' rotation gradients to be computed together. You cannot keep 64 independent per-expert backward calls and still batch them — they fire separately. So the experts for a given role must flow through one autograd Function. The per-expert *GEMM* is preserved — it becomes a ragged loop *inside* that Function, so forward compute is unchanged.

This mirrors the surgery the walk already performs (in-place module replacement, pre-DDP); it adds a method swap on `SequentialMLP.forward`.

---

## 5. Components & interfaces

### 5.1 `GroupedPOETXLinear` (`third_party/poet_torch/grouped_poetx_layer.py`)

One responsibility: own all `E` experts' rotation state for a single linear role and expose a batched forward.

- Parameters / buffers:
  - `weight: Parameter[E, out, in]`, `requires_grad=False` — the only **stacked** state (forward-frame `Wx`).
  - `oft_R_in: ParameterList` of `E` tensors `[r_in, n_in]`; `oft_R_out: ParameterList` of `E` tensors `[r_out, n_out]`, `requires_grad=True` — names **must** contain `oft_R`. **Kept as E separate 2-D params** (not a 3-D param) so the optimizer + merge see them exactly as today (see §8). The forward stacks them transiently with `torch.stack`; autograd splits the stacked grad back to the E leaves.
  - per-expert `perm_in/out`, `perm_*_inv`, block index buffers; `block_size_in/out`, `r_in/r_out`; `alternating: bool`, `alternate_every: int`.
- Methods:
  - `forward(concat_tokens, tokens_per_expert) -> concat_out` — calls `GroupedPOETXFunction.apply(...)`.
  - `effective_weight() -> Tensor[E,out,in]` — for parity tests (at `oft_R≡0`, equals `weight`).
  - `merge_then_reinitialize(reinit_perm: bool)` — batched fold (§7).
  - `_fold_active_side_grouped(active, reinit_perm)` — batched active-only fold.
- Reuses POETX's perm/fold helpers ([poetx_layer.py](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_layer.py)) batched over the leading `E` axis.

### 5.2 `GroupedPOETXFunction` (`third_party/poet_torch/grouped_poetx_ops.py`)

The core — one `torch.autograd.Function` spanning all experts for one role.

- **forward(ctx, concat_x, oft_R_in, oft_R_out, Wx[E,out,in], …, tokens_per_expert):** split `concat_x` by `tokens_per_expert`; per expert `y_e = x_e @ Wx_e.t()`; concat → `y`. Save `concat_x`, `Wx`, perms, block buffers, split sizes. (`oft_R_*` are inputs only, so autograd routes the closed-form grads to them — same trick as POETX.)
- **backward(ctx, grad_y):** split `grad_y`; per expert `G_e = x_eᵀ @ grad_y_e` → stack `G[E,in,out]` (ragged, Adam-equivalent, not the tax). Then **block-sparse + batched**: for each block `k`, gather `G[:, rows_k, :]` and `Wx[:, :, cols_k]` and run `M_in_blocks = bmm(...)` over `[E·n_blocks, b, …]`; likewise `M_out`. Project to skew vectors → `grad_oft_R_in/out[E, r, n_elems]`. Return `grad_x` (concat) + the two stacked `grad_oft_R` + `None`s.
- Both `M_in` and `M_out` every step: the champion is `POETXLinear(alternating=True)` → `POETXSingleStepFunction` (both grads). `alternating` only changes the merge, not the backward.

### 5.3 Grouped `SequentialMLP.forward`

`h = grouped_fc1(x, tpe)` → elementwise activation on the concat tensor (gate/up split handled per the existing swiglu path) → `out = grouped_fc2(h, tpe)`. The token split/concat is the existing ragged structure. Target the **bf16, non-fp8** path; the `fp8`/`fp4` padding branches and `moe_apply_probs_on_input` are preserved or explicitly asserted out of scope.

---

## 6. Data flow

- **Forward:** ragged per-expert bare GEMM, concatenated — unchanged cost.
- **Backward:** ragged per-expert `G` (Adam-equivalent), then the **batched block-sparse `M`** replacing `2·E` full `[d,d]` GEMMs with two batched-block `bmm`s — `block_count×` fewer FLOPs, `E×` fewer launches. This is the entire win.

---

## 7. Merge (batched, active-only, over experts)

Extend the merge driver ([poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py)) to collect `GroupedPOETXLinear` and fold batched over the expert axis:

- forward-frame → W_perm round-trip per expert (batched index-selects);
- active-side block-sparse Cayley only (the frozen side's `oft_R≡0` → identity, skipped — mirrors [`_fold_active_side`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_layer.py#L156));
- re-permute → forward frame.

Bit-identical per expert to today's per-layer fold. Merge is the cheap ~2.6% — this task is correctness-only, not perf. Grouped layers fold in their **own** loop (they aren't Cayley-batchable with the 2-D per-layer path, whose batch axis is layers, not experts).

---

## 8. Optimizer / DDP integration — resolved

The earlier "#1 risk" (does `lie_ortho` handle a stacked `E` axis?) is **resolved by construction**: don't stack `oft_R`.

- Reading [`LieOrthMomentum._skew_update_buffer`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L150): the optimizer assumes each skew param is **2-D `(n_blocks, n_elems)`** — `block_size_from_nelems(A_dir.shape[1])` reads `n_elems` from the last axis, `vec_to_skew(A_cat, bsz)` expects 2-D, and it **already concatenates** all skew params sharing a block size into one Newton-Schulz call (`buckets`/`A_cat = torch.cat(..., dim=0)`). A stacked 3-D `oft_R` would break all three.
- Therefore `GroupedPOETXLinear` keeps **E separate 2-D `oft_R` params**, identical in shape/name/count to the per-expert `POETXLinear`s they replace. The optimizer and merge see **no change** — and the experts' skew updates are *already* batched inside the optimizer's NS.
- Only the frozen **weight** is stacked (a `[E,out,in]` buffer, `requires_grad=False`, no optimizer concern). The forward stacks `oft_R` transiently via `torch.stack`; the stacked grad from `GroupedPOETXFunction` flows back through the `stack` op to each leaf param's `.grad` (2-D), exactly what the optimizer expects.
- Names retain the `oft_R` substring, so the optimizer per-group LR glob ([poet_optimizer_setup.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py)) and the merge-reset filter `_reset_vanilla_oft_state` ([poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L322)) catch them unchanged. The walk runs **pre-DDP**, so `oft_R` lands in the DDP grad buffer.
- A guard test (plan Task 7) pins the optimizer's 2-D assumption so a future stacked-param refactor can't silently regress it.

---

## 9. Gating

**The build is conditional on the per-op profile** confirming `M_in`/`M_out` dominate forward/backward. The prior `torch.profiler` drill-down was killed aggregating a 32-microbatch × 64-expert trace; re-run with grad-accum forced to 1:

```bash
env POET_PROFILE_STEP=20 POET_PROFILE_TORCH=1 \
  bash scripts/train_deepseek_poet.sh full \
  training.global_batch_size=8 training.micro_batch_size=1 training.log_interval=1
```

If the top ops are the expert `M`/GEMM rows → proceed. If something else dominates → revisit before building.

---

## 10. Parity gate (the correctness bar)

At every layer, `GroupedPOETXLinear` over `E` experts must be **bit-comparable to `E` independent `POETXLinear`s** on the same weights / `oft_R`, for:

- **forward** output,
- **`grad_oft_R_in/out`** (backward — the load-bearing equivalence),
- **post-merge weight**.

fp32 exact; bf16 ≤ 1e-5. Pure-torch, CPU-only — no GPU needed. No silent skips: non-divisible expert dims raise at construction and at the wrap site.

---

## 11. File structure

- Create `third_party/poet_torch/grouped_poetx_ops.py` — `GroupedPOETXFunction` (block-sparse batched backward).
- Create `third_party/poet_torch/grouped_poetx_layer.py` — `GroupedPOETXLinear` + batched perm/fold helpers.
- Create `third_party/poet_torch/tests_poet/test_grouped_poetx.py` — CPU parity (forward / backward / merge) vs `E` independent `POETXLinear`s.
- Modify `src/optim/poet_layers.py` — `SequentialMLP` detection + grouped install + `SequentialMLP.forward` swap.
- Create `tests/optim/test_grouped_expert_wrap.py` — walk wraps a fake SequentialMLP.
- Modify `src/patches/poet_merge_step.py` — collect + batched fold for grouped modules.
- Create `tests/patches/test_grouped_poetx_merge.py` — batched fold correctness.
- Modify `src/utils/megatron_args.py` — a flag (e.g. `optim.poet.group_experts`) gating the grouped path.
- Modify `configs/experiments/optim/poet_lie_orth_alt.yaml` (or a sibling) + `docs/experiments/...md` — enable + document.

---

## 12. Implementation sequencing (de-risk order)

1. **Pin the optimizer's 2-D `oft_R` assumption** (§8) with a guard test, so the "keep E separate params" decision is enforced.
2. `GroupedPOETXFunction` with **block-sparse `M`**, single-expert path — parity vs `POETXSingleStepFunction` (FLOP cut, no batching yet).
3. **Batch over experts** — `bmm` over `[E·n_blocks, …]`; parity vs `E` independent functions.
4. `GroupedPOETXLinear` module + batched merge — parity (forward/backward/merge).
5. Walk wiring + `SequentialMLP.forward` swap — fake-SequentialMLP CPU test; existing 2-D POET path unchanged.
6. Config flag + GPU smoke + throughput/loss A/B vs the current per-expert POET run (record in `docs/experiments`).

---

## 13. Risks & open questions

- **`lie_ortho` stacked-axis** (§8) — **resolved** by keeping `oft_R` as E separate 2-D params; pinned by a guard test (Task 7).
- **`unfuse_fc1` role enumeration** — the grouped install must stack the correct set of per-expert linear roles (fc1/fc2, or gate/up/fc2 under unfuse). Enumerate from the wrapped linears, don't hard-code.
- **`SequentialMLP.forward` fidelity** — probs application, quantization padding, `num_local_experts==1` branch. Target bf16/non-fp8; assert the rest out of scope rather than silently mishandle.
- **Win is conditional** (§9) — block-sparse is a guaranteed FLOP cut; the batched-`bmm` launch cut only helps if the block-GEMMs are launch/occupancy-bound. The profile decides.
- **Expert/data parallelism** — POET pins TP=1; this design assumes each rank holds all `E` local experts (DP replicates). Re-check if EP is ever introduced.
