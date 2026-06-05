# POET: Parameter-Efficient Orthogonal Training

> **Last updated: 2026-06-05.** Part 1 below is the conceptual reference (math,
> kernel, cache). For the living status of every implemented modification, which
> designs actually help, and the best-run leaderboard, jump to
> **[Part 2 — Modifications & results tracker](#part-2--modifications--results-tracker)**.

## The core idea

POET replaces a standard linear layer

```
y = W · x
```

with a parameterization that keeps `W` **frozen** and instead trains two small
**block-orthogonal** matrices that pre- and post-multiply `W`:

```
y = R_out · W · R_in · x
```

The trainable parameters are the small ones; the big base matrix `W` only
changes at periodic "merge" events. Two random permutations (`P_in`, `P_out`)
get interleaved into the chain to break correlations between blocks:

```
y = P_out · R_out · P_out_inv · W · P_in · R_in · P_in_inv · x
```

This is the math the layer computes every forward pass.

## What `R_in` and `R_out` look like

Both are **block-diagonal orthogonal** matrices built from a small trainable
parameter `oft_R`:

- `oft_R`: a tensor of skew-symmetric block parameters, shape
  `(r_in + r_out, block_size · (block_size − 1) / 2)`, where
  `r_in = in_features / block_size` and `r_out = out_features / block_size`.

For each block, an orthogonal matrix is built via the **Cayley transform**:

```
Q     = skew_symmetric(oft_R_block)    # antisymmetric: Qᵀ = −Q
R_blk = (I − Q) · (I + Q)⁻¹             # orthogonal by construction
```

Assembled along the block diagonal, the `r_in` blocks form `R_in`
(`in × in`), and the `r_out` blocks form `R_out` (`out × out`).
Only the small skew vectors in `oft_R` are trainable.

## Time-scale taxonomy

POET has three nested time scales:

| Quantity        | Changes at            | Period (approx) |
|-----------------|-----------------------|-----------------|
| `x` (input)     | every microbatch      | 1 step          |
| `oft_R`         | every `optimizer.step` | 1 cycle (K microbatches) |
| `R_in`, `R_out` | derived from `oft_R`  | every cycle     |
| `W` (base)      | merge events          | every `merge_period` cycles |
| `perm_in`, `perm_out` | merge events    | every `merge_period` cycles |

This structure is what makes caching effective — `R` is constant across the
K microbatches of a gradient-accumulation cycle.

## The merge step

Every `merge_period` cycles (default 200), `merge_then_reinitialize` runs:

1. Fold the current `R_in`, `R_out` into `W`:
   `W := P_out · R_out · P_out_inv · W · P_in · R_in · P_in_inv` (the
   "effective" weight at this point).
2. Reset `oft_R` to zero, so `R_in = R_out = I`.
3. Randomize `perm_in`, `perm_out`.
4. Pre-permute `W` with the new perms (`perform_permutation`) so subsequent
   forwards produce the same effective math.

After merge, training continues with new random perms, fresh `oft_R = 0`,
and the same trainable surface area.

## Per-microbatch forward (current default kernel)

A single fused Triton kernel, `chain_layer_x_checkpoint_mem_o2`, performs
the full chain:

1. Load `x` via `perm_in_inv` (index translation, no buffer)
2. Block-diag multiply `x @ R_in` (one block at a time)
3. Re-index via `perm_in`
4. Dense matmul `@ W` (the bulk of the FLOPs)
5. Re-index via `perm_out_inv`
6. Block-diag multiply `@ R_out`
7. Store via `perm_out`

The "checkpoint_mem_o2" suffix means the kernel does internal **gradient
checkpointing** — it doesn't store the intermediates from steps 2–6 for
backward; instead it recomputes them on demand. This trades extra compute
for activation memory savings.

## Backward

The backward pass propagates gradients of the loss through the kernel, all
the way back to `oft_R` via the Cayley graph. The flow is:

```
∂L/∂y  →  chain_layer backward  →  ∂L/∂R_in, ∂L/∂R_out
                                              ↓
                                    Cayley backward
                                              ↓
                                         ∂L/∂oft_R
                                              ↓
                                  Megatron's main_grad buffer
                                  (fp32 accumulator across K microbatches)
```

Across K microbatches in a cycle, the gradient accumulates in fp32 via
Megatron's `main_grad` mechanism. After K microbatches, `optimizer.step()`
updates `oft_R`, and the cycle starts over.

## Cayley cache (Mode A)

Because `oft_R` is **constant within a cycle**, the Cayley-derived
`R_in` and `R_out` are **identical for every one of the K microbatches**.
Computing them K times is redundant.

Mode A (in `src/optim/poet_cache.py`) caches them:

- **Cache miss** (first microbatch of a cycle): build `R_full` with the
  Cayley autograd graph alive, detach into `R_leaf` tensors used by the
  layer's forward.
- **Cache hit** (microbatches 2..K): just return the cached `R_leaf`
  tensors. Skip the Cayley work.
- **Flush** (end of cycle, before `optimizer.step()`): run one manual VJP
  through the cached Cayley graph (`R_full → oft_R`), pushing the
  K-accumulated `R_leaf.grad` back to `oft_R.main_grad`. Then invalidate
  the cache so the next cycle rebuilds with the updated `oft_R`.

Speedup ceiling: `1 / (1 − cayley_fraction × (K−1)/K)`. For small attention
projections (1536²) at K=64 in bf16 this is ~1.20×. For big FFN layers
(7168²+) it's ~1.03×.

## Hyperparameters worth knowing

| Parameter | Typical value | What it does |
|---|---|---|
| `block_size` | 256 or 512 | Size of each Cayley block. Bigger = more orthogonal freedom per block, more compute, bigger speedup ceiling for Mode A. Must divide both `in_features` and `out_features`. |
| `merge_period` | 200 steps | How often `R` is folded into `W` and `oft_R` reset. |
| `init_type` | `normalized` | Whether to normalize `W` rows at init (per-row spectral norm). |
| `mup_alpha` | 1.0 | μP-style spectral scaling. |
| `cache_mode` | `cached_fwd_bwd` (Mode A) or `none` | Whether to use the Cayley cache. |

## Where to read more

- Math + structure: `third_party/poet_torch/poet_layer.py`
- Triton kernels: `third_party/poet_torch/poet_ops.py`
- Cayley cache implementation: `src/optim/poet_cache.py`
- Cache design doc: `docs/superpowers/specs/2026-05-23-poet-cayley-cache-design.md`
- Cache implementation plan: `docs/superpowers/plans/2026-05-24-poet-cayley-cache.md`
- Standalone benchmark: `tools/poet_cache_bench.py`

---

# Part 2 — Modifications & results tracker

> **Living section.** Update it whenever a modification lands, a verdict changes,
> or a run beats the leaderboard. The goal is a single place that answers: *what
> have we built, what actually helps, and what is the best run so far?*
>
> **Results cohort** (unless a row says otherwise): llama3 **60m**, seq 256,
> **40 tokens/param** (≈2.4B tokens, **9,155 steps**, global batch 1024), seed 42,
> `cluster=h100_de` (8×GPU). Metric is the W&B **`val/loss`** (lower is better;
> `val/ppl` in parentheses), with **`train/loss`** shown for continuity. Note the
> metrics were renamed by the `wandb_metric_normalize` patch — the live keys are
> `val/loss` / `train/loss` / `val/ppl`, **not** the old `lm loss`. Loss is only
> comparable *within* one (scale, token-budget) cohort.

## 2.1 Implemented modifications — status & verdict

| Modification | Config key / CLI flag | What it changes (mechanism) | Status | Verdict @ 60m/40tpp |
|---|---|---|---|---|
| Frozen-W block-orthogonal core | always on | `y = R_out·W·R_in·x`; train only the small skew `oft_R`, fold into `W` at merges | ✅ | baseline (POET) |
| Single-step merge + decoupled reinit | `merge_period=1`, `reinit_period` (`poet0`) | fold `R→W` every step; `reinit_period` separately controls Ψ-resample / Adam-momentum reset | ✅ | helps vs `merge_period=400` (≈3.65 vs ≈3.70+) |
| Lie-algebra momentum on Q | `optim.poet.q_optimizer=lie_algebra` / `--poet-q-optimizer` (`poet_lie`) | Adam-like 1st/2nd moment kept in the skew algebra so(n), persists across merges ([poet_lie_momentum.py:161](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L161)) | ✅ | **helps** (3.647 vs vanilla 3.70) — the strongest POET base |
| Stage-2 W-free RMS scaling | `optim.poet.lie_rms=true`, `lie_rms_c` / `--poet-lie-rms[-c]` (`poet_lie_rms`) | per-block `α = c·√blk / (‖A‖_F+eps)` → dimension-consistent rotation angle, no `W` access ([poet_lie_momentum.py:161-171](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L161-L171)) | ✅ | **helps with tuned lr** — best POET overall (3.626 @ lr 3e-3, c=4) |
| Head-aligned attention rotation | `optim.poet.head_aligned_attn=true` / `--poet-head-aligned-attn` (`poet_lie_head`, `poet_h_*`) | swap q/k/v/o to `HeadAlignedPOETLinear`: per-head block-diagonal rotation (block=head_dim, fixed identity Ψ), needs unfused qkv ([head_aligned_layer.py:28](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/head_aligned_layer.py#L28), [poet_layers.py:245-257](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L245-L257)) | ✅ | **neutral→hurts** at 60m (3.654 vs non-head 3.634 at matched lr/c) |
| Residual-side perm off | `optim.poet.head_resid_perm=false` / `--poet-no-head-resid-perm` (`poet_h_noperm_*`) | freeze the residual (non-head) side's Ψ in head-aligned mode | ✅ | neutral (3.6536 vs 3.6541) |
| Alternating single-sided update | `optim.poet.lie_alternating=true`, `lie_alternate_every` (`poet_lie_alt`) | write only one rotation side per step ([poet_lie_momentum.py:126-130](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L126-L130)) | ✅ | **hurts** (3.709 vs poet_lie 3.647) |
| `exp` parameterization | `optim.poet.parameterization=exp` | exact matrix-exponential orthogonal map (vs truncated Cayley); incompatible with caching | ✅ | **hurts** vs cayley (3.70–3.82) |
| Muon-on-Q (SkewMuon) | `optim.poet.q_optimizer=muon`, `muon_theta/ns_steps/momentum` | per-block Newton-Schulz orthogonalize + constant-angle θ rescale; built for the no-reset regime ([poet_skew_muon.py:120](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_skew_muon.py#L120)) | ✅ | hurts so far (≈3.79); needs `merge_period=0` tuning, not yet done |
| Cayley cache (Mode A) | `optim.poet.cache_mode=cached_fwd_bwd` | cache `R` within a grad-accum cycle, flush one VJP at cycle end ([poet_cache.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_cache.py)) | ✅ | perf-only; measured dead-end for small K (no quality effect) |
| Normalized / μP base init | `optim.poet.init_type`, `mup_alpha` | row-normalize frozen `W` (+ optional μP spectral scale) ([poet_layers.py:44-62](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L44-L62)) | ✅ | `normalized` is default; not separately ablated (sweeps fixed `mup_alpha=1.0`) |
| Single-sided rotation (freeze output) | `optim.poet.train_output_rotation=false` / `--poet-freeze-output-rotation` | train only `R_in`, freeze `R_out=I` | ✅ | not ablated at scale |

Q-optimizer dispatch lives at [poet.py:644-652](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L644-L652) (`lie_algebra` / `muon` / default `adam`); CLI→flag routing at [megatron_args.py:249-339](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L249-L339).

## 2.2 Experiment configs (the variants)

All under [configs/experiments/optim/](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/). Common to all POET configs: `block_count=1`, `scale=0.5`, `init_type=normalized`, `parameterization=cayley`, `train_output_rotation=true`.

| Config | q_opt | merge / reinit | lie_rms (c) | head-aligned | alternating | Purpose |
|---|---|---|---|---|---|---|
| [poet](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml) | adam | 400 / 0 | — | no | no | Baseline POET (periodic merge) |
| [poet0](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet0.yaml) | adam | 1 / 400 | — | no | no | Single-step merge, Ψ/momentum held for 400 steps |
| [poet_lie](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie.yaml) | lie_algebra | 1 / −1 | — | no | no | Pion **Stage 1**: Lie-algebra momentum, never resample Ψ |
| [poet_lie_alt](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_alt.yaml) | lie_algebra | 1 / −1 | — | no | yes (every 1) | Stage 1 + §6 alternating single-sided update |
| [poet_lie_head](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_head.yaml) | lie_algebra | 1 / −1 | — | **yes** | no | Stage 1 + per-head attention rotation |
| [poet_lie_rms](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_rms.yaml) | lie_algebra | 1 / −1 | true (0.2) | no | no | Pion **Stage 2**: W-free RMS angle scaling |

The `poet_h_*` / `poet_dense_*` runs in §2.4 are CLI sweeps over `poet_lie_rms` (± `head_aligned_attn`, varying `lie_rms_c`), not separate config files.

## 2.3 Results — which designs are useful

Best completed run per setting, ranked by `val/loss` (60m / 40 tokens-per-param):

| # | Setting | val/loss | (ppl) | train | lr | lie_rms_c | head | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | **muon_kimi** | **3.5352** | 34.30 | 3.4261 | 1e-3 | — | — | best overall |
| 2 | adam (baseline) | 3.5570 | 35.06 | 3.4575 | 1e-3 | — | — | best dense baseline |
| 3 | muon_hybrid | 3.5698 | 35.51 | 3.4705 | — | — | — | |
| 4 | **poet_lie_rms** | **3.6257** | 37.55 | 3.5220 | **3e-3** | 4 | no | **best POET** (run `tx67fwih`) |
| 5 | poet_dense_rms (c8) | 3.6344 | 37.88 | 3.5367 | 1e-3 | 8 | no | |
| 6 | poet_lie_rms (c8) | 3.6404 | 38.11 | 3.5367 | 1e-3 | 8 | no | |
| 7 | poet_lie | 3.6474 | 38.37 | 3.5437 | 1e-3 | — | no | Stage 1 |
| 8 | poet_lie_rms (c4) | 3.6496 | 38.46 | 3.5478 | 1e-3 | 4 | no | same as #4 but lr 1e-3 |
| 9 | poet0 | 3.6518 | 38.55 | 3.5484 | 1e-3 | — | no | |
| 10 | **poet_h_noperm_rms_c8** | 3.6536 | 38.61 | 3.5578 | 1e-3 | 8 | **yes** | best head-aligned |
| 11 | poet_h_rms_c8 | 3.6541 | 38.63 | 3.5588 | 1e-3 | 8 | yes | |
| 12 | poet (vanilla, cayley) | ≈3.70 | ≈40.6 | ≈3.60 | 1e-3 | — | no | weakest POET family |
| — | poet `exp` / Muon-on-Q | 3.70–3.82 | 41–46 | — | 1e-3 | — | no | regressions |

**Conclusions (what's useful):**
- **POET still trails the strong baselines** at this scale: best POET (3.626) is **+0.07** val/loss vs adam (3.557) and **+0.09** vs muon_kimi (3.535). Closing this gap is the open problem.
- **The useful POET stack** is *single-step merge + Lie-algebra momentum + Stage-2 RMS scaling*, with **lr pushed to 3e-3** and **`lie_rms_c` ≈ 4–8**. Each layer of that stack helps: vanilla `poet` (≈3.70) → `poet_lie` (3.647) → `poet_lie_rms` (3.626).
- **`lie_rms_c` has a sweet spot:** 4–8 are best; **c=12 degrades and c=16 is much worse** (train ≈3.78). Larger c over-rotates.
- **Higher lr helps the RMS variant:** at c=4, lr 3e-3 (3.626) clearly beats lr 1e-3 (3.650).
- **Head-aligned attention does NOT help at 60m.** Matched at lr 1e-3 / c=8: head-aligned (3.654) is ~0.015–0.02 *worse* than non-head-aligned (3.634–3.640). Turning the residual-side permutation off (`noperm`) is a wash.
- **Alternating, `exp` parameterization, and Muon-on-Q are current regressions** for the reset-based recipe. Muon-on-Q was designed for the `merge_period=0` no-reset regime and hasn't been retuned for it yet.

## 2.4 Head-aligned + RMS sweep (the `poet_h_*` / `poet_dense_*` runs)

All at lr 1e-3, `lie_rms_c` swept (the `aNNN` token in run names is the paired nominal-angle annotation, **not** `mup_alpha`, which stayed 1.0). Sorted best-first by `val/loss`:

| Run | head-aligned | lie_rms_c | resid_perm | val/loss | train |
|---|---|---|---|---|---|
| poet_dense_rms_c8_a004 | no | 8 | — | 3.6344 | 3.5367 |
| poet_dense_rms_c12_a006 | no | 12 | — | 3.6576 | 3.5617 |
| poet_h_noperm_rms_c8 | yes | 8 | off | 3.6536 | 3.5578 |
| poet_h_rms_c8_a004 | yes | 8 | on | 3.6541 | 3.5588 |
| poet_h_rms_c12_a006 | yes | 12 | on | 3.6743 | 3.5810 |
| poet_h_rms_c4_a002 | yes | 4 | on | 3.6781 | 3.5815 |
| poet_h_norms | yes | — (no rms) | on | 3.6927 | 3.5949 |
| poet_h_exp_rms_c8 | yes | 8 (exp) | on | — | crashed @ step 4256 |

Takeaway: **non-head-aligned (`dense`) beats head-aligned at every matched c**, RMS-on beats RMS-off, c=8 > c=12 > c=4 for the head-aligned family, and the `exp` head-aligned run crashed.

## 2.5 Best runs leaderboard (settings + result)

> Keep this current: when a run beats its family's entry, replace it (cite the run dir + W&B id).

**🏆 Overall best (60m/40tpp):** [`muon_kimi-…-20260602T134241Z`](/lustre/fast/fast/zqiu/slm-research/runs/muon_kimi-llama3-60m-s42-20260602T134241Z) — **val/loss 3.5352, ppl 34.30**, train 3.4261, 9155 steps, lr 1e-3.

**🥇 Best POET:** [`poet_lie_rms-…-20260604T140255Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_rms-llama3-60m-s42-20260604T140255Z) (W&B `zeju-qiu/slm-zeju-dev/tx67fwih`) — **val/loss 3.6257, ppl 37.55**, train 3.5220, 9155 steps. Settings: `experiment=optim/poet_lie_rms`, **lr=0.003**, **lie_rms_c=4** (all other knobs = config default). Reproduced by its twin [`…-20260604T124303Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_rms-llama3-60m-s42-20260604T124303Z) (identical val/loss). Command:
```bash
codexlog poet_lie_rms_best bash scripts/train_poet_lie_rms.sh llama3 \
  optim.lr=0.003 \
  optim.poet.lie_rms_c=4
```

**Per-family best:**

| Family | Run dir | val/loss | (ppl) | key settings |
|---|---|---|---|---|
| muon_kimi | [muon_kimi-…20260602T134241Z](/lustre/fast/fast/zqiu/slm-research/runs/muon_kimi-llama3-60m-s42-20260602T134241Z) | 3.5352 | 34.30 | lr 1e-3 |
| adam | [adam-…20260601T221123Z](/lustre/fast/fast/zqiu/slm-research/runs/adam-llama3-60m-s42-20260601T221123Z) | 3.5570 | 35.06 | lr 1e-3 |
| muon_hybrid | [muon-…20260602T001936Z](/lustre/fast/fast/zqiu/slm-research/runs/muon-llama3-60m-s42-20260602T001936Z) | 3.5698 | 35.51 | |
| poet_lie_rms | [poet_lie_rms-…20260604T140255Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_rms-llama3-60m-s42-20260604T140255Z) | 3.6257 | 37.55 | lr 3e-3, c=4 |
| poet_lie | [poet_lie-…20260603T183821Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie-llama3-60m-s42-20260603T183821Z) | 3.6474 | 38.37 | lr 1e-3 |
| poet0 | [poet0-…20260603T165332Z](/lustre/fast/fast/zqiu/slm-research/runs/poet0-llama3-60m-s42-20260603T165332Z) | 3.6518 | 38.55 | lr 1e-3 |
| head-aligned | [poet_h_noperm_rms_c8-…20260605T112512Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_h_noperm_rms_c8-llama3-60m-s42-20260605T112512Z) | 3.6536 | 38.61 | lr 1e-3, c=8, noperm |
| poet (vanilla) | `runs/poet-llama3-60m-s42-*` | ≈3.70 | ≈40.6 | lr 1e-3, merge_period 400 |

## 2.6 How to update this tracker

- **Cohort matters:** only compare runs at the same scale + tokens/param. Everything above is 60m / 40tpp. A 300m or 20x table would be a separate block.
- **Pull results** from each run's W&B summary: `runs/<dir>/**/wandb-summary.json`, keys `val/loss` / `train/loss` / `val/ppl` / `_step`. Treat `_step < 9000` as crashed/short for this cohort (full run = 9155 steps).
- **Settings** come from `runs/<dir>/resolved_config.yaml` (`optim.lr`, `optim.poet.*`, `training.tokens_per_param`).
- **Data-quality caveats from this snapshot:** the vanilla `poet` family had a high crash rate (many `_step ≪ 9155`); duplicate rows in the raw scan are same-setting reruns (e.g. the two `poet_lie_rms` lr-3e-3/c-4 dirs); `poet_h_exp_rms_c8` crashed at step 4256.
