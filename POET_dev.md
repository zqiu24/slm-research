# POET: Parameter-Efficient Orthogonal Training

> **Last updated: 2026-06-19.** Part 1 below is the conceptual reference (math,
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
| Stage-2 W-free RMS scaling | `optim.poet.lie_rms=true`, `lie_rms_c` / `--poet-lie-rms[-c]` (`poet_lie_rms`) | per-block `α = c·√blk / (‖A‖_F+eps)` → dimension-consistent rotation angle, no `W` access ([poet_lie_momentum.py:161-171](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L161-L171)) | ✅ | **helps with tuned lr** (3.626 @ lr 3e-3, c=4) — best of the *RMS* family; superseded by `lie_ortho` ↓ |
| **Muon-like orthogonalizing Q-opt** | `optim.poet.q_optimizer=lie_ortho`, `lie_ortho_c`/`_method`/`_ns_steps` (`poet_lie_orth`) | standalone `LieOrthMomentum`: same Lie 1st-moment momentum, but **orthogonalize** the skew direction (all planes → ~same angle) instead of RMS-scaling; `muon` band (~5 NS steps) or `spectral` exact `A(−A²)^{-1/2}` (~20) ([poet_lie_orth.py:27](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L27)) | ✅ | **BEST POET / best PEFT** (**3.5231** @ lr **4e-3** / scale 0.5 / c8 → **eff∠ 0.016**, head-OFF + `lie_alternating`, `ghsu7t8y`) — **beats the old under-tuned muon_kimi 3.5321 by −0.009** but now sits behind **re-tuned muon_kimi** (3.4514, +0.072), **nGPT** (3.4583, +0.065), and **re-tuned dense adam** (**3.4935** @ lr 3e-3, `ebndt1qj`, +0.030); the 2026-06-09 grid found a hotter optimum (angle 0.016 > old 0.012, dense lr 4e-3 > 3e-3). Prior champ 3.5332 @ eff∠0.012 (`1ynrrimu`) |
| DP-sharded `lie_ortho` orthogonalization | `optim.poet.lie_ortho_distributed=true` / `--poet-lie-ortho-distributed` | round-robin the Newton-Schulz skew blocks across data-parallel ranks, then re-sync with one zero-padded `all_reduce(SUM)` of update deltas | ✅ | perf-only win: completed distributed run matches replicated quality (3.566695 vs 3.566730) and is faster (0.2785 vs 0.3764 s/step); now `true` in all POET experiment configs |
| Head-aligned attention rotation | `optim.poet.head_aligned_attn=true` / `--poet-head-aligned-attn` (`poet_lie_head`, `poet_h_*`) | swap q/k/v/o to `HeadAlignedPOETLinear`: per-head block-diagonal rotation (block=head_dim, fixed identity Ψ), needs unfused qkv ([head_aligned_layer.py:28](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/head_aligned_layer.py#L28), [poet_layers.py:245-257](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L245-L257)) | ✅ | **neutral→hurts** at 60m (3.654 vs non-head 3.634 at matched lr/c) |
| Residual-side perm off | `optim.poet.head_resid_perm=false` / `--poet-no-head-resid-perm` (`poet_h_noperm_*`) | freeze the residual (non-head) side's Ψ in head-aligned mode | ✅ | neutral (3.6536 vs 3.6541) |
| Alternating single-sided update | `optim.poet.lie_alternating=true`, `lie_alternate_every` (`poet_lie_alt`) | write only ONE rotation side per step (out even / in odd) but keep BOTH Lie momenta advancing every step ([poet_lie_orth.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py)) | ✅ | **optimizer-dependent: HELPS `lie_ortho` (champion 3.5332 → tuned to 3.5231 by the lr×scale×c grid, ~4% faster/step), but HURT `lie_algebra`** (3.709 vs 3.647). Needs fresh both-side momentum — the d³-optimized true-single-side (frozen momentum) regresses to 4.22 (`au92x0pj`). Matched-movement diagnostic (§2.5 sweep H, `c5pzfkzb`) **rules out step-size**: at eff∠=θ/√2 both-sides = 3.5416, still −0.0185 behind alternating → the win is **Gauss–Seidel coupling**, not a smaller step |
| Nesterov look-ahead (`lie_ortho`) | `optim.poet.lie_ortho_nesterov=true` / `--poet-lie-ortho-nesterov` | orthogonalize the Muon look-ahead direction `(1−b1)·g + b1·m` (= modern Muon's `grad.lerp(m, β)`) instead of the bare first moment `m`; skew/rotation branch only, AdamW untouched ([poet_lie_orth.py:160-176](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L160-L176)) | ⏳ implemented, **untested** | sweep ready ([sweep_lie_orth_nesterov_lr.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_nesterov_lr.sh)), no run yet @ 60m/40tpp |
| `exp` parameterization | `optim.poet.parameterization=exp` | exact matrix-exponential orthogonal map (vs truncated Cayley); incompatible with caching | ✅ | **hurts** vs cayley (3.70–3.82) |
| Muon-on-Q (SkewMuon) | `optim.poet.q_optimizer=muon`, `muon_theta/ns_steps/momentum` | per-block Newton-Schulz orthogonalize + constant-angle θ rescale; built for the no-reset regime ([poet_skew_muon.py:120](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_skew_muon.py#L120)) | ✅ | hurts so far (≈3.79); needs `merge_period=0` tuning, not yet done |
| Cayley cache (Mode A) | `optim.poet.cache_mode=cached_fwd_bwd` | cache `R` within a grad-accum cycle, flush one VJP at cycle end ([poet_cache.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_cache.py)) | ✅ | perf-only; measured dead-end for small K (no quality effect) |
| Normalized / μP base init | `optim.poet.init_type`, `mup_alpha` | row-normalize frozen `W` (+ optional μP spectral scale) ([poet_layers.py:44-62](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L44-L62)) | ✅ | `normalized` is default; not separately ablated (sweeps fixed `mup_alpha=1.0`) |
| Single-sided rotation (freeze output) | `optim.poet.train_output_rotation=false` / `--poet-freeze-output-rotation` | train only `R_in`, freeze `R_out=I` | ✅ | not ablated at scale |

Q-optimizer dispatch (`lie_algebra` / `lie_ortho` / `muon` / default `adam`) lives in [poet.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py); `lie_algebra` and `lie_ortho` share the same builder, which branches to construct `LieOrthMomentum` at [poet.py:596](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L596). CLI→flag routing in [megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py).

**The `lie_ortho` optimizer (new — current best POET).** A standalone `LieOrthMomentum` ([poet_lie_orth.py:27](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L27)), selected by `optim.poet.q_optimizer=lie_ortho`. It keeps the same Lie-algebra **first-moment** momentum on `oft_R` as `lie_algebra` (persists across folds), but replaces the direction→generator transform: instead of RMS-scaling, it **orthogonalizes** the per-block skew direction so **every rotation plane turns by ~the same angle** — Muon's "trust the subspace, not the per-direction magnitude" bet, applied to *rotational* updates. Two methods (`optim.poet.lie_ortho_method`):
- **`muon`** (default, ~5 NS steps): Muon's quintic Newton–Schulz on the direction, then a `½(X−Xᵀ)` cleanup. NS *preserves skew* on a skew input (verified to ~1e-15) and lands the singular values in a **band** around 1 — cheap, approximately-equal angles.
- **`spectral`** (~15–20 NS steps): the exact Löwdin form `A·(−A²)^{-1/2}` — drives every σ to *exactly* 1, ≈4× the cost.

Realized per-plane angle = `lr · scale · ortho_c` (under `muon` the band makes `ortho_c` *nominal*, ≈0.75–1.0× that). First-moment-only by default (a second moment is partly undone by orthogonalization). Design doc: [docs/muon_orthogonalizing_optimizer_poet.md](/lustre/fast/fast/zqiu/slm-research/docs/muon_orthogonalizing_optimizer_poet.md); plan: [docs/superpowers/plans/2026-06-05-poet-lie-orth-optimizer.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/plans/2026-06-05-poet-lie-orth-optimizer.md). **Status (2026-06-09):** champion is now **`ghsu7t8y`** (`cos_lr4_s50_c8`) at **val/loss 3.5231** — the head-OFF + `lie_alternating` recipe with the angle/dense-lr tuned UP by the 2026-06-09 lr×scale×c grid (**lr 4e-3 / scale 0.5 / c8 → eff∠ 0.016**). This **beats the old under-tuned muon_kimi (3.5321) by −0.009** (best PEFT method), but the leaderboard above it has since moved: **re-tuned muon_kimi** (`vtw9k55h`, **3.4514** @ lr 4e-3/wd 0.1) and **nGPT** (`5zycv3p5`, **3.4583** @ lr 1e-2) now co-lead overall (~−0.065–0.072 ahead of POET), and re-tuned **dense adam at lr 3e-3** (`ebndt1qj`, **3.4935**) is −0.030 ahead — so POET is the best PEFT/non-dense method at 60m/40tpp, not the outright best (see §2.3). It supersedes the prior POET champion `1ynrrimu` (3.5332 @ lr3e-3/eff∠0.012), which the grid reproduced exactly (`li3sflwl`, `wj68pgey`). It beats the **head-aligned-OFF both-sides** run `dwynpk9y` (c=8, lr 3e-3, muon, **`head_aligned_attn=false`**, `lie_ortho_distributed=true`) at **3.5528** by **−0.020**. Both wins compound: turning head-alignment OFF beat the head-on twin `7lncmww7` (3.5667) by **−0.014** and overtook the old adam baseline (3.557, lr 1e-3 — but the re-tuned adam at lr 3e-3 = 3.4935 now leads overall); alternating then added **−0.017–0.020** more. The earlier head-on champion `7lncmww7` matched its replicated twin `l5w0n7gq` to ~3e-5 while cutting W&B `perf/step_time_s` from **0.3764 → 0.2785**; the nohead champion runs at a comparable **0.2822 s/step**. Relative to the pre-speedup tracker baseline (1.180 s/step), the current path is ~4× faster end-to-end, and the targeted optimizer hot path was confirmed ~3× faster on GPU. Full sweep verdicts are in §2.5; headline: **ortho ≫ RMS** (matched-angle RMS sibling diverged), **muon-band ≈ exact `spectral`**, **1st-moment > 2nd**, **the angle sweet spot is eff∠ ~0.016** (head-OFF+alt tolerates higher angles than the old 0.012 ceiling; only eff∠ 0.024 diverged), **dense lr wants 4e-3 ≳ 3e-3** (decoupling-down falsified), **min_lr_ratio 0.01 is the floor sweet spot** (0.1 and 0.001 both slightly worse), and **head-alignment OFF > ON even with ortho**. The DP-sharded path round-robins the Newton-Schulz work across data-parallel ranks and re-syncs with one zero-padded `all_reduce(SUM)` of update deltas; it is now enabled in all POET experiment YAMLs (effective for `q_optimizer=lie_ortho`, no-op at `dp_world=1`). Sweeps: [_lr](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_lr.sh), [_scale](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_scale.sh), [_variants](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_variants.sh), [_c](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_c.sh) (the c-sweep traces the same effective-angle axis as `_scale`), [_nesterov_lr](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_nesterov_lr.sh) (champion recipe + Muon Nesterov look-ahead, lr {1,2,3,4,6}e-3 — **untested**).

## 2.2 Experiment configs (the variants)

All under [configs/experiments/optim/](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/). Common to all POET configs: `block_count=1`, `scale=0.5`, `init_type=normalized`, `parameterization=cayley`, `train_output_rotation=true`, `lie_ortho_distributed=true` (effective only when `q_optimizer=lie_ortho`).

| Config | q_opt | merge / reinit | lie_rms (c) | head-aligned | alternating | Purpose |
|---|---|---|---|---|---|---|
| [poet](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml) | adam | 400 / 0 | — | no | no | Baseline POET (periodic merge) |
| [poet0](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet0.yaml) | adam | 1 / 400 | — | no | no | Single-step merge, Ψ/momentum held for 400 steps |
| [poet_lie](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie.yaml) | lie_algebra | 1 / −1 | — | no | no | Pion **Stage 1**: Lie-algebra momentum, never resample Ψ |
| [poet_lie_alt](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_alt.yaml) | lie_algebra | 1 / −1 | — | no | yes (every 1) | Stage 1 + §6 alternating single-sided update |
| [poet_lie_head](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_head.yaml) | lie_algebra | 1 / −1 | — | **yes** | no | Stage 1 + per-head attention rotation |
| [poet_lie_rms](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_rms.yaml) | lie_algebra | 1 / −1 | true (0.2) | no | no | Pion **Stage 2**: W-free RMS angle scaling |
| [poet_lie_orth](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_orth.yaml) | **lie_ortho** | 1 / −1 | — (ortho c=4, muon) | yes (YAML default) | no | **Muon-like orthogonalizing** optimizer (equal-angle planes) — **current best POET**, but the best run overrides `head_aligned_attn=false` (head-off wins, see §2.5) |

The `poet_h_*` / `poet_dense_*` runs in §2.4 are CLI sweeps over `poet_lie_rms` (± `head_aligned_attn`, varying `lie_rms_c`), not separate config files.

## 2.3 Results — which designs are useful

Best completed run per setting, ranked by `val/loss` (60m / 40 tokens-per-param):

| # | Setting | val/loss | (ppl) | train | lr | ortho c (eff∠) | head | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | **muon_kimi (lr 4e-3, wd 0.1)** | **3.4514** | 31.54 | 3.3482 | **4e-3** | — | — | **🏆 NEW BEST OVERALL** (`vtw9k55h`) — re-tuned muon dense baseline; **≈tie with nGPT** (−0.007, ~seed-noise), beats tuned dense adam by **−0.042**, best POET by **−0.072** |
| 2 | nGPT (architecture, lr 1e-2) | 3.4583 | 31.76 | 3.3573 | **1e-2** | — | — | normalized-GPT *architecture* (`ngpt_lr100`/`5zycv3p5`) — co-best (−0.007 behind muon_kimi); beats tuned adam by −0.035, best POET by −0.065 |
| 3 | adam (dense, lr 3e-3) | 3.4935 | 32.90 | 3.3935 | **3e-3** | — | — | re-tuned dense baseline (`ebndt1qj`); −0.042 behind muon_kimi, beats best POET by **−0.030** |
| 4 | poet_lie_orth (+alt, no-head, lr4e-3/c8) | 3.5231 | 33.89 | 3.4233 | **4e-3** | 8 (**0.016**) | **no** | **best POET / best PEFT** (`ghsu7t8y`, cosine grid); behind dense adam by +0.030 |
| 5 | poet_lie_orth (+alt, no-head, lr3e-3/c12) | 3.5274 | 34.04 | 3.4277 | 3e-3 | 12 (0.018) | no | `owcyd976` — angle 0.018 also stable+strong (old doc wrongly called 0.018 divergent) |
| 6 | poet_lie_orth (+alt, no-head, lr4e-3/s0.25/c12) | 3.5288 | 34.08 | 3.4278 | 4e-3 | 12 (0.012) | no | `q60mrt7u` — at angle 0.012, dense-lr 4e-3 beats 3e-3 (hotter dense helps) |
| 7 | poet_lie_orth (+alt, no-head, lr3e-3/c8) | 3.5332 | 34.23 | 3.4334 | 3e-3 | 8 (0.012) | no | prior best POET (`1ynrrimu`); reproduced by `li3sflwl`, `wj68pgey` |
| 8 | muon_kimi (lr 1e-3 — old baseline) | 3.5321 | 34.20 | 3.4219 | 1e-3 | — | — | under-tuned; lr 4e-3 + wd 0.1 (#1) is **−0.081** better (`of4bakqd`; `ijq33tle` rerun 3.5251) |
| 9 | poet_lie_orth (c8, no-head, both-sides) | 3.5528 | 34.91 | 3.4557 | 3e-3 | 8 (0.012) | no | both-sides head-off (`dwynpk9y`; fresh rerun `f4f49v4f` = 3.5504) |
| 10 | adam (dense, lr 1e-3 — old baseline) | 3.5570 | 35.06 | 3.4575 | 1e-3 | — | — | under-tuned; lr 3e-3 (#3) is −0.064 better (`ylrd45af`) |
| 11 | poet_lie_orth (c8, head) | 3.5667 | 35.40 | 3.4693 | 3e-3 | 8 (0.012) | yes | head-aligned twin (`7lncmww7`, distributed=true) |
| 12 | muon_hybrid | 3.5698 | 35.51 | 3.4705 | — | — | — | |
| 13 | poet_lie_orth (c4) | 3.5715 | 35.57 | 3.4701 | 3e-3 | 4 (0.006) | yes | run `z1gpz9y7` |
| 14 | poet_lie_rms | 3.6193 | 37.31 | 3.5220 | 3e-3 | 4 (rms) | no | best RMS-family (`98293d1u`; head-aligned twin `l2pzawa4` 3.6335 — worse) |
| 15 | poet_dense_rms (c8) | 3.6344 | 37.88 | 3.5367 | 1e-3 | 8 (rms) | no | |
| 16 | poet_lie_rms (c8) | 3.6404 | 38.11 | 3.5367 | 1e-3 | 8 (rms) | no | |
| 17 | poet_lie | 3.6474 | 38.37 | 3.5437 | 1e-3 | — | no | Stage 1 |
| 18 | poet_lie_rms (c4) | 3.6496 | 38.46 | 3.5478 | 1e-3 | 4 (rms) | no | same as #14 but lr 1e-3 |
| 19 | poet0 | 3.6518 | 38.55 | 3.5484 | 1e-3 | — | no | |
| 20 | **poet_h_noperm_rms_c8** | 3.6536 | 38.61 | 3.5578 | 1e-3 | 8 (rms) | **yes** | best head-aligned (RMS family) |
| 21 | poet_h_rms_c8 | 3.6541 | 38.63 | 3.5588 | 1e-3 | 8 (rms) | yes | |
| 22 | poet (vanilla, cayley) | ≈3.70 | ≈40.6 | ≈3.60 | 1e-3 | — | no | weakest POET family |
| — | poet `exp` / Muon-on-Q / true-single-side (`au92x0pj` 4.22) / WSD df0.2 (`lodwi7cw` 3.5699) | 3.57–4.22 | — | — | — | — | no | regressions / dead-ends |

**Conclusions (what's useful):**
- **The top of the board is now a near-tie between re-tuned muon_kimi and nGPT, both clearly ahead of POET and even tuned adam.** On the identical 60m/40tpp cohort + cosine min_lr 0.1 schedule: **muon_kimi lr 4e-3 / wd 0.1 → val 3.4514** (`vtw9k55h`, ppl 31.54) and **nGPT lr 1e-2 → 3.4583** (`5zycv3p5`) are within −0.007 (~seed noise), both beating **tuned dense adam (3.4935)** by ~0.035–0.042 and the best POET (3.5231) by ~0.065–0.072. So among same-arch (llama3) optimizers, **muon_kimi — not adam — is now strongest**, and POET remains the best PEFT method but not the best overall.
- **muon_kimi was badly under-tuned in the old tracker — lr and weight-decay both matter.** The old entry (lr 1e-3 → 3.5321, `of4bakqd`) left a **−0.081** improvement on the table: pushing lr 1e-3 → 4e-3 and turning on **wd 0.1** gives 3.4514. Both knobs help monotonically here — at fixed lr, **wd 0.1 beats wd 0** by ≈−0.025 (4e-3: 3.4514 vs 3.4774; 3e-3: 3.4604 vs 3.4850), and higher lr helps (1e-3 3.5251 → 3e-3 3.4604 → 4e-3 3.4514; 6e-3 not yet run). ⚠️ single seed; (no lr 1e-4 run exists on disk — the win is at lr 4e-3).
- **nGPT** ([sweep_ngpt_lr.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_ngpt_lr.sh), adam-matched recipe) peaks at **lr 1e-2 → 3.4583** (lr90 next at 3.4645, optimum bracketed). ⚠️ this is a **cross-architecture** comparison — nGPT changes the model (hypersphere-normalized weights/activations + learned scaling), not just the optimizer; single seed; the *reference*-recipe nGPT sweep (wd 0, no warmup, [sweep_ngpt_lr_reference.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_ngpt_lr_reference.sh)) is still running and so far trails this recipe (lr5–50 all ≈0.01–0.04 worse).
- **Tuned dense adam (lr 3e-3 → 3.4935, `ebndt1qj`)** is now 3rd overall — a −0.064 jump over the old lr-1e-3 baseline (3.5570), still beating the best POET by −0.030, but **−0.042 behind muon_kimi**. ⚠️ adam ran with cosine **min_lr_ratio 0.1** and has not been swept on min_lr / 2–3 seeds yet.
- **POET is still the best PEFT method and the strongest non-dense optimizer — it just lost the overall crown.** Best POET (poet_lie_orth + `lie_alternating`, head-OFF, distributed, **lr 4e-3 / scale 0.5 / c8 → eff∠ 0.016**) hits **val 3.5231** (`ghsu7t8y`), still **beating the old under-tuned muon_kimi (3.5321) by −0.009** and the prior champion (3.5332 @ lr3e-3/eff∠0.012) by −0.010, but now behind re-tuned muon_kimi (3.4514, +0.072), nGPT (3.4583, +0.065), and dense adam (3.4935, +0.030). The POET-internal stack that got here: head-OFF (−0.014) → `lie_alternating` (−0.017) → **tune the angle up to ~0.016 + dense lr up to 4e-3 (−0.010)**. ⚠️ single seed; worth a 2–3 seed confirm.
- **The optimum rotation angle is ~0.016, not 0.012 — the old champion was under-rotating.** The grid's top three (3.5231 @ eff∠0.016, 3.5274 @ 0.018, 3.5288 @ 0.012/dense-4e-3) all beat the 0.012 champion. And **dense lr wants to be HIGHER**: at fixed eff∠0.012, dense 4e-3 (3.5288) > 3e-3 (3.5332), and the decoupling sweep is monotone — lowering dense lr to 1e-3 *hurt* (3.5332→3.5563). The earlier decoupling hypothesis (3e-3 too hot) is **falsified**: POET wants it hot on both axes.
- **Alternating the single-sided write HELPS `lie_ortho` (reversing the `lie_algebra` verdict).** Writing one rotation side per step while keeping **both** momenta fresh is the new champion (3.5332, ahead at every checkpoint, ~4% faster/step). This *flips* the earlier finding that alternating hurt the `lie_algebra` family (3.709 vs 3.647) — the difference is the optimizer: orthogonalized updates take a full-magnitude step along the momentum *direction*, so giving each side a 2-step-accumulated (smoother) momentum + Gauss–Seidel one-factor-at-a-time decoupling is a net win; under RMS/`lie_algebra` it was not. Crucial caveat: this only works while **both momenta stay fresh** — the d³-optimized true-single-side variant (which *freezes* the inactive momentum to skip its gradient) **regresses to 4.22** (`au92x0pj`). Fresh both-side momentum is load-bearing.
- **Orthogonalizing the rotation direction (`lie_ortho`) beats RMS-scaling it (`lie_rms`)** — now confirmed by the variants sweep: at the matched champion angle the RMS sibling `lierms_c8` **diverged** (val 6.38) while ortho held 3.567. Also from the sweeps: **muon-band ≈ exact `spectral`** (3.5669 vs 3.5703 — the cheap ~5-step quintic is enough, no need to pay for exact σ=1), **1st-moment beats 2nd** (3.5669 vs 3.5702), and for the angle **c=8 ≳ c=4** (3.5669 vs 3.5715). The `head`-on/off arm (`lieorth_c8_nohead`) is now resolved — **head-off wins** (3.5528 vs head-on 3.5667).
- **The useful POET stack** is *single-step merge + Lie-algebra momentum + an angle-equalizing transform + alternating write*: vanilla `poet` (≈3.70) → `poet_lie` (3.647) → `poet_lie_rms` (3.619) → `poet_lie_orth` (3.5528, head-off) → **+ `lie_alternating` (3.5332)**, all with **lr 3e-3**.
- **The rotation-angle ceiling is RECIPE-DEPENDENT and moved up — best is now eff∠ ~0.016.** The old "ceiling at 0.012, 0.018 diverges (val 4.55)" was the **both-sides head-ON** recipe. For the current **head-OFF + alternating** recipe, eff∠ **0.016 (lr4/s0.5/c8 = 3.5231) and 0.018 (lr3/s0.5/c12 = 3.5274) are stable AND best**; only **eff∠ 0.024 (lr4/s0.5/c12) diverged**. Alternating writes one side/step, so each side rotates less per step → a higher *nominal* angle is tolerable. New sweet spot ≈ 0.016; ceiling sits between 0.018 and 0.024. (c and scale remain interchangeable under `muon`: it's the product eff∠=lr·scale·c that matters.)
- **The `c` knob has a sweet spot:** for RMS, c≈4 best at lr 3e-3 (c=8 *diverged* at this lr); for ortho c=8 ≳ c=4 and the band is wider. Larger c over-rotates.
- **Head-aligned attention does NOT help — now confirmed for the ortho family too.** The `lieorth_c8_nohead` arm finished: head-OFF `dwynpk9y` **3.5528** beats the head-ON twin `7lncmww7` **3.5667** by **−0.014** at the otherwise-identical champion recipe. This matches the RMS evidence (lr 3e-3/c=4: head `l2pzawa4` 3.6335 vs no-head `98293d1u` 3.6193; and lr 1e-3/c=8: head 3.654 vs dense 3.634). So head-alignment hurts across **both** families — the best POET is now head-OFF.
- **`exp` parameterization and (reset-regime) Muon-on-Q are current regressions.** Muon-on-Q (SkewMuon) was built for the `merge_period=0` no-reset regime and hasn't been retuned for it. (Alternating is NO LONGER a regression — it is the new champion under `lie_ortho`; see the alternating bullet above.)

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

## 2.5 lie_ortho sweep results (as of 2026-06-21)

Champion-quality setting (**updated 2026-06-09**): **lr 4e-3, scale 0.5, c=8** (eff∠ **0.016**), muon, **head-aligned OFF**, **+ `lie_alternating=true`**, cosine min_lr 0.01. Current best completed run is `cos_lr4_s50_c8` (`ghsu7t8y`) = **val/loss 3.5231** — **beats the old under-tuned muon_kimi 3.5321** (but behind the re-tuned muon_kimi 3.4514, `vtw9k55h`) — found by the lr×scale×c grid (arm **F** below). The prior champion `1ynrrimu` (lr3e-3/eff∠0.012) = 3.5332 is arm **E**. The head-off **both-sides** run `dwynpk9y` = **3.5528** (the `lieorth_c8_nohead` arm — head-off beats head-on by −0.014; fresh both-sides rerun `f4f49v4f` = 3.5504). Among the *head-aligned* arms, the original anchor `5sbgancm` = **3.5669**, reproduced 4× (`lieorth_c8_muon`, `lieorth_lr0.003`, `lieorth_scale0.5`, + the original anchor), with the distributed rerun `7lncmww7` = **3.5667**.

**Speed/distributed status:** cached skew indices + same-block batching removed the big Python/scatter overhead, and DP-sharded orthogonalization is now wired end-to-end. Completed same-config pair:

| Run | distributed | val/loss | train | s/step | runtime |
|---|---:|---:|---:|---:|---:|
| `7lncmww7` | true | **3.566695** | 3.4693 | **0.2785** | 3070s |
| `l5w0n7gq` | false | 3.566730 | 3.4680 | 0.3764 | 3949s |

Quality is unchanged at the 60m/40tpp level; the distributed path gives a clear perf win on top of the optimizer speedup. Completed sweep arms, by sweep:

**A — global LR** ([sweep_lie_orth_lr.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_lr.sh)): 1e-3 → 3.6259 · 2e-3 → 3.5683 · **3e-3 → 3.5669** · (4e-3, 6e-3 not yet complete). Peak at 3e-3 so far.

**B — POET scale / rotation-lr** ([sweep_lie_orth_scale.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_scale.sh)): 0.25 → 3.5715 · **0.5 → 3.5669** · **0.75 → 4.5527 (DIVERGED)** · (1.0, 1.5 incomplete). Stability ceiling at scale 0.5 (eff∠ 0.012).

**C — variants at the champion angle** ([sweep_lie_orth_variants.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_variants.sh)):

| arm | change vs control | val/loss |
|---|---|---|
| `lieorth_c8_muon` (control) | — | 3.5669 |
| `lieorth_c8_spectral` | exact σ=1 (ns=20) | 3.5703 |
| `lieorth_c8_2mom` | second moment on | 3.5702 |
| `lierms_c8` | RMS instead of ortho | **6.3797 (DIVERGED)** |
| `lieorth_c8_nohead` | head-aligned off | **3.5528 (BEST — `dwynpk9y`)** |

→ muon-band ≈ exact spectral; 1st-moment ≥ 2nd; **ortho far more stable than RMS** at this angle; **head-aligned OFF wins** (3.5528 vs head-on 3.5667 — head-alignment hurts the ortho family too).

**D — native c** ([sweep_lie_orth_c.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_c.sh)): not yet run. Degenerate with sweep B (eff∠ = lr·scale·c, and `orthogonalize(−m)` is magnitude-free), so c≥12 (eff∠ ≥ 0.018) will likely diverge like scale=0.75; the informative band is c≤8–10.

**E — alternating single-sided write** (champion recipe + `lie_alternating=true`, `lie_alternate_every=1`):

| arm | change vs both-sides control | val/loss | s/step |
|---|---|---|---:|
| both-sides head-off (`f4f49v4f`) | — | 3.5504 | 0.2150 |
| **alternating, both momenta (`1ynrrimu`)** | write 1 side/step, both momenta fresh | **3.5332** (then-best; superseded by grid arm F → 3.5231) | **0.2068** |
| true-single-side (`au92x0pj`) | write 1 side/step, **freeze** inactive momentum (d³-skip layer `single_step_x_alternating`) | 4.2201 (REGRESS) | 0.1923 |

→ Alternating the *write* while keeping **both momenta fresh** is the champion (−0.017 vs both-sides, ahead at every checkpoint, ~4% faster). Freezing the inactive momentum to also skip its gradient (the d³-optimized layer) **breaks** it — the fresh both-side momentum is the load-bearing ingredient. This run was a CLI override on `poet_lie_orth` (no dedicated config yet); a dedicated `poet_lie_orth_alt` config + a POETX-structured alternating layer are planned.

**F — lr × scale × c grid** ([sweep_lie_orth_grid_cosine.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_grid_cosine.sh), cosine min_lr 0.01, champion base): 16 cells = lr {1,2,3,4}e-3 × scale {0.25,0.5} × c {8,12}. **This grid produced the best POET run.** Top cells (val/loss; eff∠ = lr·scale·c):

| run | lr | scale | c | eff∠ | val/loss |
|---|---|---|---|---|---|
| **`cos_lr4_s50_c8` (`ghsu7t8y`)** | 4e-3 | 0.5 | 8 | **0.016** | **3.5231 🏆** |
| `cos_lr3_s50_c12` (`owcyd976`) | 3e-3 | 0.5 | 12 | 0.018 | 3.5274 |
| `cos_lr4_s25_c12` (`q60mrt7u`) | 4e-3 | 0.25 | 12 | 0.012 | 3.5288 |
| `cos_lr3_s50_c8` (`li3sflwl`, = champ) | 3e-3 | 0.5 | 8 | 0.012 | 3.5332 |
| `cos_lr2_s50_c12` | 2e-3 | 0.5 | 12 | 0.012 | 3.5385 |
| `cos_lr4_s50_c12` | 4e-3 | 0.5 | 12 | 0.024 | **DIVERGED** |

→ **eff∠ 0.016 is the new sweet spot** (0.018 close behind); the curve trains *hotter* (higher loss mid-run) but anneals to a lower endpoint, ahead from step ~8.5k on. **0.024 diverged** → ceiling for this recipe is between 0.018 and 0.024 (vs the old 0.012 ceiling for the both-sides head-on recipe). At fixed eff∠ 0.012, dense lr **4e-3 (3.5288) > 3e-3 (3.5332)** — hotter dense helps. Monotone below: lower lr/angle steadily worse (down to 3.7875 @ eff∠ 0.002).

**G — dense-LR decoupling** ([sweep_lie_orth_decouple.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_decouple.sh), cosine): hold c=8 and the rotation-group lr (eff∠) fixed, push AdamW dense lr 3e-3→1e-3 via scale, × min_lr_ratio {0.01, 0.001}. **No winner — three clean monotone trends, all "POET wants it hot":**
- At fixed eff∠ 0.012 (m01): dense 3e-3 (3.5332) < 2e-3 (3.5385) < 1.5e-3 (3.5432) < 1e-3 (3.5563). **Lowering dense lr HURTS** → decoupling-down hypothesis falsified; the high dense lr is *good*.
- eff∠ 0.012 > 0.008 at every dense lr (e.g. dense 3e-3: 3.5332 vs 3.5565).
- **min_lr_ratio 0.01 > 0.001** for all 8 twins (deeper floor slightly hurts). Combined with `9mvs5hsg` (cosine min_lr 0.1 = 3.5413), **0.01 is the floor sweet spot** (0.1 and 0.001 both worse).

**H — alternating vs both-sides at MATCHED per-step movement (step-size vs coupling)** (2026-06-13): isolate whether alternating's win over both-sides is the Gauss–Seidel / fresh-momentum coupling or merely a smaller effective step. Both-sides applies two rotations/step (‖ΔW‖≈√2·θ), alternating one (≈θ), so match per-step `W`-movement by running both-sides at eff∠ = θ/√2 — shift `scale` (not `lr`) so the dense AdamW optimization stays identical. Same lr 4e-3, c8, head-off, distributed:

| arm | eff∠ | per-step ‖ΔW‖ | val/loss | s/step |
|---|---|---|---|---:|
| **A — alternating (`ghsu7t8y`)** | 0.016 | θ | **3.5231** | 0.207 |
| **B — both-sides, matched (`c5pzfkzb`)** | 0.0113 (scale 0.35355) | θ (matched) | **3.5416** | 0.212 |

→ **Step-size RULED OUT — the coupling is real.** At matched per-step movement *and* matched dense lr, alternating still wins by **−0.0185** (~3.7× the ~0.005 seed-noise band) and is ~2% faster (folds one side/step). Removing the √2 step-size confound did **not** shrink the gap (the raw same-nominal-angle gap was −0.0172) — had step-size been the driver, matching movement would have let both-sides catch up; it did not. Mechanism: both-sides = **Jacobi** (both factors stepped from gradients at the *same* W, neglecting the bilinear cross-term `Q_out·W·Q_in`); alternating = **Gauss–Seidel** (fold one side so the other's gradient sees it) + fresh both-side momentum carrying the coupling between writes. The magnitude-free orthogonalizer takes a full-length step regardless of ‖m‖ (no step-shrink safety net), so direction-coherence pays in full — which is also why alternating helps `lie_ortho` but hurt `lie_algebra`/RMS (sweep E). ⚠️ single seed; 3.5416 sits just inside the "confirm" band and the √2 is a first-order ‖ΔW‖ estimate, so an **optimum-to-optimum closer is pending** — a both-sides arm at eff∠ ~0.0141 (scale 0.44194, lr 4e-3) to verify both-sides isn't merely under-rotated here. Justifies pursuing coupling-quality improvements (symmetric Gauss–Seidel, per-side angle/cadence asymmetry).

**I — pure one-sided POET (in_only / out_only)** (2026-06-19, [sweep_poet_lie_orth_in_only.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_poet_lie_orth_in_only.sh) / […_out_only.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_poet_lie_orth_out_only.sh)): train exactly ONE FIXED rotation side for the whole run (`optim.poet.single_step_x_one_sided=in|out` — `InOnlyPOETXLinear` trains only `oft_R_in`, `OutOnlyPOETXLinear` only `oft_R_out`; the frozen side stays at its 0-init identity so its forward, backward, momentum, and merge fold are all short-circuited). Each side swept lr×c×scale = {3,4,5,6}e-3,1e-2,2e-2 × c{4,8} × scale{0.5,1.0} = 24 cells (champion lie_ortho base: muon, head-off, distributed, merge_period 1, cayley, cosine min_lr 0.01).

| arm | run (W&B) | lr | c | scale | eff∠ | val/loss |
|---|---|---|---|---|---|---|
| **out_only champion** | `outonly_lr0.006_c4_s0.5` (`vgj9ywrd`) | 6e-3 | 4 | 0.5 | 0.012 | **3.6289** |
| out_only runner-up | `outonly_lr0.005_c4_s0.5` (`z4xen8f7`) | 5e-3 | 4 | 0.5 | 0.010 | 3.6347 |
| **in_only champion** | `inonly_lr0.006_c4_s0.5` (`xef9sj7f`) | 6e-3 | 4 | 0.5 | 0.012 | **3.6794** |
| out_only worst (diverged) | `outonly_lr0.004_c8_s1.0` (`ppfjzs2l`) | 4e-3 | 8 | 1.0 | 0.032 | 7.39 💥 |

→ **out_only beats in_only in 23/24 cells** by a steady **−0.045 to −0.05** (best 3.6289 vs 3.6794) — the **output-side rotation `oft_R_out` carries more capacity than the input side**. But **both one-sided modes lose clearly to both-sides**: out_only is **+0.106 vs the alternating champion (3.5231)**, **+0.078 vs the both-sides head-off (3.5504)**; in_only +0.156 / +0.129. So freezing one rotation side costs ~0.08–0.16 val/loss — the second side is real, non-redundant capacity, and one-sided is not a free lunch. **Angle sweet spot ≈ 0.010–0.012** (same as both-sides): optimum at `c4/scale0.5/lr 5–6e-3`, degrading above eff∠ 0.016 and bad ≥0.04; the one-sided optimum sits at a *higher dense lr* (6e-3) than the both-sides champ (3–4e-3) for the same eff∠ — with one side frozen the model leans harder on the AdamW dense params. **Implementation sanity check passed:** within each lr, `(c4,scale1.0)` and `(c8,scale0.5)` give **bit-identical** val/loss (same `scale·c` product → same rotation, same dense lr), confirming `eff∠ = lr·scale·c` and the one-sided optimizer/merge wiring. **Crucially this isolates the §2.5-E regression:** the d³-skip layer here uses the SAME `true_single_side` optimizer + active-only fold as the regressing `au92x0pj` (3.5504→4.2201), but with the side **FIXED** instead of alternating, the trained side's momentum advances every step → healthy 3.63–3.68, **not** 4.22. So the 4.22 blow-up was momentum-*staleness* from alternating, **not** the d³ layer — the d³ short-circuit itself is correct. ⚠️ single seed each (in-vs-out gap ~0.05 ≫ seed noise; one-sided-vs-both gap ~0.1 robust); `outonly_lr0.006_c4_s0.5` worth a 2–3 seed confirm before treating 3.6289 as final.

**J — alternate_every (write-cadence / momentum averaging-window)** (2026-06-21, [sweep_lie_orth_alternate_every.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_alternate_every.sh)): hold the champion recipe (lr 4e-3 / scale 0.5 / c8 → eff∠ 0.016, muon, head-off, alternating) and sweep `lie_alternate_every ∈ {1,2,4,8}` — write each side for k consecutive steps then rest it k steps (both momenta keep advancing). eff∠ is unchanged (0.016 throughout); only the per-side write cadence changes. `SLM_POET_COORD_DIAG=1` on to read whether more averaging shifts the momentum SNR. This axis had been PINNED at 1 and never swept.

| arm | k | val/loss | mom_cos_in | mom_cos_out | gram_cond | cos(D_out,D_in) |
|---|---|---|---|---|---|---|
| **alt1 (champ)** | 1 | **3.5231** | +0.016 | +0.009 | 1.229 | −0.002 |
| alt4 | 4 | 3.5241 | +0.003 | +0.002 | 1.232 | −0.001 |
| alt2 | 2 | 3.5262 | +0.008 | +0.015 | 1.228 | +0.001 |
| alt8 | 8 | 3.5285 | −0.012 | −0.009 | 1.262 | +0.000 |

⚠️ **Frame-bug correction (2026-06-22, commit `aac95a2`).** The `gram_cond` and `cos(D_out,D_in)` columns above were computed with a buggy `w_perm_frame` that re-permuted the *already*-block-frame `POETLinear.weight`, scrambling every `side_directions`-derived metric to ~noise (the forward permutes the activations, not the weight). The frame-fixed re-run of `alt1` (W&B `g9i51g5l`, identical recipe) gives **cos(D_out,D_in) ≈ +0.44** (range 0.41–0.47, all run), **gram_cond ≈ 2.67**, **r_joint ≈ 1.44**, **cos_D_out_D_in_raw ≈ 0.54** (overlap intrinsic to the momenta, not NS-induced), **r_cross ≈ 0.004** (peak 0.0074 → 0.0001). The k 2/4/8 rows were not re-run, but their cos/gram columns are equally invalid. `mom_cos_*` and `val/loss` are **unaffected** (they don't pass through the frame) and reproduce exactly — `alt1` val 3.518 = the champion within seed noise, confirming the fix is diagnostic-only. See ANALYSIS §17.6.

→ **k=1 is already optimal; no interior optimum.** Whole-sweep spread is only **0.0055 nats** (weak axis, ≈1 seed-band), but the envelope is monotone-worse with k (the k2<k4 inversion is within noise; **k1-best / k8-worst** is the clean signal), and `alt1` reproduces the champion 3.5231 *exactly*. The coord diag shows **longer averaging does NOT raise the momentum SNR — it lowers it**: `mom_cos` trends down with k and goes **negative at k=8** (the worst run), i.e. longer rest makes the per-step rotation gradient staler / anti-correlated, not sharper. So the alternating win is **per-step fresh re-evaluation of the rotation gradient, not a wider EMA averaging window** — consistent with the frozen-EMA blow-up (`au92x0pj` → 4.22, §2.5-E): the momentum must keep advancing AND be re-applied every step. **Gauge-redundancy is SUPPORTED, not falsified** — the frame-fixed `cos(D_out,D_in) ≈ 0.44` (not ≈0; see correction note above and ANALYSIS §17.6), so the two sides DO share a reinforcing spatial-overlap direction (`gram_cond` ≈ 2.67). Its *physical* per-step cross-term is still small, though (`r_cross` ≤0.7%). This sweep nonetheless isolates the **temporal** axis via `mom_cos` (frame-independent), and the spatial overlap is ~constant across k, so it does not drive the k-dependence here — the win on this axis is **per-step fresh re-evaluation**, unchanged by the fix. (Whether the now-confirmed spatial overlap is *causally* part of the alternating advantage is answered by the decorrelation A/B in **J.2** below.) **Verdict: keep `lie_alternate_every` pinned at 1.**

**J.2 — is the spatial overlap causal? simultaneous ±cross-side-decorrelation** (2026-06-22, [scripts/train_sim_decorrelate.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_sim_decorrelate.sh)): J.1 confirmed the two sides overlap (`cos≈0.44`), but is that overlap *causally* part of the alternating win or benign? Test it head-on: run the champion recipe **simultaneously** (`lie_alternating=false`, both sides written each step) with vs without **cross-side decorrelation** (`optim.poet.lie_ortho_decorrelate=true`, mode `in_off_out`) — which projects each layer's in/out generator off the other's weight-space direction so the *applied-update* `cos(D_out,D_in)→0` while leaving per-side Muon whitening intact ([poet_lie_orth.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py) `_decorrelate_buf`). Matched seed 1234, lr4e-3 / c8 / head-off.

| arm | config | val/loss |
|---|---|---|
| A0 alternating champion (`g9i51g5l`) | `alt=true` | **3.5181** |
| A1 simultaneous baseline (`1tpkj44a`) | `alt=false`, decorr off | 3.5768 |
| A2 simultaneous + decorrelate (`c9l15mmy`) | `alt=false`, decorr **on** | **3.5624** |

→ **Both channels are causal; temporal dominates ~3:1.** Decorrelation recovers **A1−A2 = +0.0144 ≈ 25%** of the alternating advantage (A1−A0 = 0.0587) — so the spatial gauge-redundancy is genuinely **harmful** in a simultaneous step (over-spending the shared direction), the first *direct causal* proof it is real, not benign. But it leaves **A2−A0 = 0.0443 ≈ 75%** unrecovered — the **temporal Gauss–Seidel** channel (fold one side so the other's gradient sees the moved `W`; a first-order spatial projection applied simultaneously can't reproduce fresh re-evaluation), consistent with the small physical cross-term (`r_cross` ≤0.7%, J.1) and the `alternate_every`/frozen-EMA evidence above. **Verdict: alternating win ≈ 75% temporal (fresh re-eval) + 25% spatial (gauge-redundancy)** — not purely temporal (the old buggy-frame claim) nor purely spatial. ⚠️ single seed each, `in_off_out` only — a 2–3-seed + `symmetric`/`out_off_in` confirm would harden the 25/75 split. (Impl note: decorrelation silently no-op'd through three layers — frame `aac95a2`, args→config `ebbc73c`, bf16 master-param remap `b6a9f3f` — before this run produced a non-baseline result; a 0-pairs-matched guard now logs loudly if it regresses.)

**Schedule (settled):** **cosine beats WSD.** Matched-recipe WSD df0.2 (`lodwi7cw`) = 3.5699, **+0.037 vs cosine** — holding the angle at the ceiling through the stable phase keeps loss high and the 20% tail can't recover; WSD→cosine as df→1 so it can't win here. min_lr 0.01 cosine (champ) beats min_lr 0.1 cosine (`9mvs5hsg` 3.5413) by +0.008.

## 2.6 Best runs leaderboard (settings + result)

> Keep this current: when a run beats its family's entry, replace it (cite the run dir + W&B id).

**🏆 Overall best (60m/40tpp) — re-tuned muon_kimi:** [`muon_kimi-…-20260609T141524Z`](/lustre/fast/fast/zqiu/slm-research/runs/muon_kimi-llama3-60m-s42-20260609T141524Z) (W&B `vtw9k55h`) — **val/loss 3.4514, ppl 31.54**, train 3.3482, 9155 steps. The Kimi/Muon optimizer (per-block Newton-Schulz orthogonalization + RMS scaling, `muon_momentum=0.95`, Nesterov, `ns_steps=5`) at **lr 4e-3 with weight_decay 0.1**, cosine **min_lr_ratio 0.1**, warmup 0.01. A **−0.081** jump over the old tracker entry (3.5321 @ lr 1e-3) — both higher lr and turning wd on help. **Edges nGPT (3.4583) by −0.007 (≈seed-noise, effectively co-best)** and beats tuned dense adam (3.4935) by **−0.042**, best POET (3.5231) by −0.072, on the identical cohort + schedule. ⚠️ single seed (no lr 1e-4 run exists — the win is at lr 4e-3); 6e-3 not yet tried. Command:
```bash
codexlog muon_kimi scripts/train_muon_dev.sh optim.lr=0.004 optim.weight_decay=0.1 experiment.name=muon_kimi
```

**🥈 Co-best overall — nGPT architecture:** [`ngpt_lr100-…-20260617T150127Z`](/lustre/fast/fast/zqiu/slm-research/runs/ngpt_lr100-llama3-60m-s42-20260617T150127Z) (W&B `5zycv3p5`) — **val/loss 3.4583, ppl 31.76**, train 3.3573, 9155 steps. The **normalized-GPT architecture** (hypersphere-normalized weights/activations + learned `alpha/sqk/suv/sz` scaling, `ngpt_adamw`) at **lr 1e-2**, optimizer recipe matched to the dense-adam baseline (warmup 0.01, wd 0.1, betas 0.9/0.95, cosine **min_lr_ratio 0.1**). Sweep minimum (lr90 = 3.4645 next), so the optimum is bracketed. Within −0.007 of muon_kimi; **beats tuned dense adam (3.4935) by −0.035**, best POET (3.5231) by −0.065. ⚠️ **cross-architecture comparison** — nGPT changes the model, not just the optimizer; single seed; the reference-recipe nGPT sweep (wd 0 / no warmup) is still running and so far trails this recipe. Command:
```bash
bash scripts/sweep_ngpt_lr.sh   # or, single run:
codexlog ngpt_lr100 scripts/train_ngpt_dev.sh optim.lr=0.01 optim.weight_decay=0.1 optim.ngpt.no_warmup=false experiment.name=ngpt_lr100
```

**🥉 3rd overall — re-tuned dense adam:** [`adam_lr30-…-20260609T112229Z`](/lustre/fast/fast/zqiu/slm-research/runs/adam_lr30-llama3-60m-s42-20260609T112229Z) (W&B `ebndt1qj`) — **val/loss 3.4935, ppl 32.90**, train 3.3935, 9155 steps. Plain **adamw at lr 3e-3** (betas 0.9/0.95, wd 0.1, eps 1e-8), cosine **min_lr_ratio 0.1**, warmup 0.01. A **−0.064** jump over the old adam baseline (3.5570 @ lr 1e-3, `ylrd45af`); **beats the best POET (3.5231) by −0.030** — but now **+0.042 behind muon_kimi**. ⚠️ single seed, and adam has not been swept on min_lr (POET's best used 0.01); worth a 2–3 seed confirm. Command:
```bash
codexlog adam_lr30 bash scripts/train_adam.sh llama3 optim.lr=0.003
```

**Best POET (best PEFT method, 4th overall):** [`cos_lr4_s50_c8-…-20260609T080009Z`](/lustre/fast/fast/zqiu/slm-research/runs/cos_lr4_s50_c8-llama3-60m-s42-20260609T080009Z) (W&B `ghsu7t8y`) — **val/loss 3.5231, ppl 33.89**, train 3.4233, 9155 steps. The head-off `lie_ortho` + alternating champion with the angle/dense-lr tuned UP by the 2026-06-09 grid: **lr 4e-3, scale 0.5, c8 → eff∠ 0.016** (vs the prior 3e-3/0.012). Beats the old under-tuned `muon_kimi` (3.5321) by **−0.009** and the prior POET champion `1ynrrimu` (3.5332) by **−0.010** (but re-tuned muon_kimi 3.4514 / nGPT 3.4583 / adam 3.4935 all now lead it). All other knobs = the head-off champion recipe (method=muon, ns_steps=5, 1st-moment, cayley, merge_period=1, reinit_period=−1, block_count=1, distributed, cosine min_lr 0.01). ⚠️ single seed — worth a 2–3 seed confirm. Command:
```bash
codexlog poet_best_grid bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.004 \
  optim.poet.scale=0.5 \
  optim.poet.lie_ortho_c=8 \
  optim.poet.lie_ortho_distributed=true \
  optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true \
  optim.poet.lie_alternate_every=1
```
*Previous best POET (lr3e-3/eff∠0.012):* [`poet_lie_orth-…-20260608T133306Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_orth-llama3-60m-s42-20260608T133306Z) (W&B `1ynrrimu`) — **val/loss 3.5332**, lr 3e-3, c=8, scale 0.5, head-off, alternating, distributed (reproduced by `li3sflwl`, `wj68pgey`).
*Previous best POET (both-sides head-off):* [`poet_lie_orth-…-20260607T172750Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_orth-llama3-60m-s42-20260607T172750Z) (W&B `dwynpk9y`) — **val/loss 3.5528** @ lr 3e-3, c=8, head-off, distributed (a fresh both-sides rerun `f4f49v4f` reproduced this at **3.5504**). Note: the *d³-optimized* true-single-side layer (`single_step_x_alternating`, run `au92x0pj`) **regresses badly** (3.5504 → **4.2201**) — it freezes the inactive side's momentum to skip the frozen gradient, and *fresh both-side momentum is exactly what makes alternating win* (see §2.1 alternating row).
*Previous best POET (head-aligned twin):* [`poet_lie_orth-…-20260607T111231Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_orth-llama3-60m-s42-20260607T111231Z) (W&B `7lncmww7`) — val/loss 3.5667 @ lr 3e-3, c=8, head-aligned + distributed (replicated twin `l5w0n7gq` 3.5667, original anchor `5sbgancm` 3.5669).
*Previous best POET (RMS family):* [`poet_lie_rms-…-20260604T140255Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_rms-llama3-60m-s42-20260604T140255Z) (W&B `tx67fwih`) — val/loss 3.6257 @ lr 3e-3, c=4 (twin [`…-20260604T124303Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_rms-llama3-60m-s42-20260604T124303Z), identical).

**Per-family best:**

| Family | Run dir | val/loss | (ppl) | key settings |
|---|---|---|---|---|
| **muon_kimi (tuned)** | [muon_kimi-…20260609T141524Z](/lustre/fast/fast/zqiu/slm-research/runs/muon_kimi-llama3-60m-s42-20260609T141524Z) (`vtw9k55h`) | **3.4514** | 31.54 | **lr 4e-3, wd 0.1**, momentum 0.95, nesterov, ns_steps 5, cosine min_lr 0.1 — **🏆 overall best** (old lr-1e-3 baseline `of4bakqd` = 3.5321; −0.081) |
| **nGPT (architecture)** | [ngpt_lr100-…20260617T150127Z](/lustre/fast/fast/zqiu/slm-research/runs/ngpt_lr100-llama3-60m-s42-20260617T150127Z) (`5zycv3p5`) | 3.4583 | 31.76 | **lr 1e-2**, adam-matched recipe (warmup 0.01, wd 0.1, cosine min_lr 0.1) — **🥈 co-best** (cross-arch; −0.007 behind muon_kimi; lr sweep min) |
| adam (dense, tuned) | [adam_lr30-…20260609T112229Z](/lustre/fast/fast/zqiu/slm-research/runs/adam_lr30-llama3-60m-s42-20260609T112229Z) (`ebndt1qj`) | 3.4935 | 32.90 | **lr 3e-3**, cosine min_lr 0.1, wd 0.1 — 🥉 3rd overall (old lr-1e-3 baseline `ylrd45af` = 3.5570) |
| poet_lie_orth (+alt, tuned) | [cos_lr4_s50_c8-…20260609T080009Z](/lustre/fast/fast/zqiu/slm-research/runs/cos_lr4_s50_c8-llama3-60m-s42-20260609T080009Z) (`ghsu7t8y`) | 3.5231 | 33.89 | **lr 4e-3, scale 0.5, c=8 (eff∠ 0.016)**, muon, head-off, distributed, `lie_alternating=true` — **best POET / best PEFT (4th overall)** |
| muon_hybrid | [muon-…20260602T001936Z](/lustre/fast/fast/zqiu/slm-research/runs/muon-llama3-60m-s42-20260602T001936Z) | 3.5698 | 35.51 | |
| poet_lie_orth (+alt, prior champ) | [poet_lie_orth-…20260608T133306Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_orth-llama3-60m-s42-20260608T133306Z) (`1ynrrimu`) | 3.5332 | 34.23 | lr 3e-3, c=8 (eff∠ 0.012), head-off, distributed, alternating |
| poet_lie_orth **out_only** (one-sided) | [outonly_lr0.006_c4_s0.5-…20260619T172803Z](/lustre/fast/fast/zqiu/slm-research/runs/outonly_lr0.006_c4_s0.5-llama3-60m-s42-20260619T172803Z) (`vgj9ywrd`) | 3.6289 | 37.67 | **fixed OUT side only** (`oft_R_out`), lr 6e-3, c=4, scale 0.5 (eff∠ 0.012) — best one-sided; +0.106 vs alt champ (§2.5-I) |
| poet_lie_rms (RMS family) | [poet_lie_rms-…20260605T142434Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_rms-llama3-60m-s42-20260605T142434Z) (`98293d1u`) | 3.6193 | 37.31 | lr 3e-3, c=4 |
| poet_lie_orth **in_only** (one-sided) | [inonly_lr0.006_c4_s0.5-…20260619T172106Z](/lustre/fast/fast/zqiu/slm-research/runs/inonly_lr0.006_c4_s0.5-llama3-60m-s42-20260619T172106Z) (`xef9sj7f`) | 3.6794 | 39.62 | **fixed IN side only** (`oft_R_in`), lr 6e-3, c=4, scale 0.5 (eff∠ 0.012); out_only beats it by −0.05 (§2.5-I) |
| poet_lie | [poet_lie-…20260603T183821Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie-llama3-60m-s42-20260603T183821Z) | 3.6474 | 38.37 | lr 1e-3 |
| poet0 | [poet0-…20260603T165332Z](/lustre/fast/fast/zqiu/slm-research/runs/poet0-llama3-60m-s42-20260603T165332Z) | 3.6518 | 38.55 | lr 1e-3 |
| head-aligned | [poet_h_noperm_rms_c8-…20260605T112512Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_h_noperm_rms_c8-llama3-60m-s42-20260605T112512Z) | 3.6536 | 38.61 | lr 1e-3, c=8, noperm |
| poet (vanilla) | `runs/poet-llama3-60m-s42-*` | ≈3.70 | ≈40.6 | lr 1e-3, merge_period 400 |

## 2.7 Weight-norm monitoring — POET vs Adam vs Muon (no weight decay)

**Question.** POET trains with **no weight decay**, yet applies *both* a left
($R_\text{out}$) and right ($R_\text{in}$) rotation around a frozen base. Does the
effective weight $W_\text{eff}=R_\text{out}\,W_0\,R_\text{in}$ grow in norm over
training, and how does that compare to additive optimizers (Adam / Muon) with and
without decoupled weight decay?

**How measured.** The `weight_norm_monitor` patch
([src/patches/weight_norm_monitor.py](/lustre/fast/fast/zqiu/slm-research/src/patches/weight_norm_monitor.py))
logs, per selected layer (`first,mid,last`) and matrix type, the **mean** of the
per-row and per-column **RMS** norms of the *post-step* weight
([compute_matrix_norm_stats](/lustre/fast/fast/zqiu/slm-research/src/patches/weight_norm_monitor.py#L125-L150)):
`row_rms = ‖W[i,:]‖₂ / √in`, `col_rms = ‖W[:,j]‖₂ / √out`. The `/√dim` divides out
matrix width so different-shaped matrices are comparable (≈ per-element weight std).
For POET the weight is read **post-merge** (`merge_period=1` ⇒ base `== W_eff` every
step). `row_rms ≈ col_rms` in every run (growth is symmetric between input/output
sides), so only `row_rms` is quoted. Enable per-run with
`training.log_weight_norms=true training.log_weight_norms_interval=100`.

**Runs.** 60m / llama3 / `ablation_40x` (9155 steps) / lr 1e-3 / seed 42. All use
`--unfuse-qkv`/`--unfuse-fc1`, so types are `q,k,v,proj,fc1_gate,fc1_up,fc2`. POET =
`poet_lie_orth` (single-step, `merge_period=1`, cayley, `q_optimizer=lie_ortho`).

| run | optimizer | wd | base init | dir |
|---|---|---|---|---|
| POET (raw init) | poet | 0.0 | `init_type=none` (raw Megatron 0.02) | [poet_lie_orth-…125743Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_orth-llama3-60m-s42-20260618T125743Z) |
| POET (norm init) | poet | 0.0 | `init_type=normalized` (unit row-norm) | [poet_lie_orth-…122249Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_orth-llama3-60m-s42-20260618T122249Z) |
| Adam | adamw | 0.0 | raw 0.02 | [adam-…133519Z](/lustre/fast/fast/zqiu/slm-research/runs/adam-llama3-60m-s42-20260618T133519Z) |
| Adam | adamw | 0.1 | raw 0.02 | [adam-…122414Z](/lustre/fast/fast/zqiu/slm-research/runs/adam-llama3-60m-s42-20260618T122414Z) |
| Muon | muon_kimi | 0.0 | raw 0.02 | [muon_kimi-…134627Z](/lustre/fast/fast/zqiu/slm-research/runs/muon_kimi-llama3-60m-s42-20260618T134627Z) |
| Muon | muon_kimi | 0.1 | raw 0.02 | [muon_kimi-…131357Z](/lustre/fast/fast/zqiu/slm-research/runs/muon_kimi-llama3-60m-s42-20260618T131357Z) |

**Result — matched (wd=0, raw init): the fair comparison.** `row_rms/mean`,
averaged over first/mid/last layers and all matrix types, at step 100 → 9100:

| run | start | @1k | @3k | @6k | final | growth |
|---|---|---|---|---|---|---|
| **POET** (raw init) | 0.0146 | 0.0151 | 0.0155 | 0.0157 | **0.0158** | **1.08×** (flat) |
| **Adam** (wd 0) | 0.0160 | 0.0280 | 0.0420 | 0.0505 | **0.0517** | **3.23×** |
| **Muon** (wd 0) | 0.0166 | 0.0335 | 0.0477 | 0.0551 | **0.0562** | **3.39×** |

**Effect of adding weight decay (Adam & Muon).** `row_rms/mean`, step 100 → 9100,
wd=0 vs wd=0.1 head-to-head:

| optimizer | wd | start | @1k | @3k | @6k | final | growth | Δfinal vs wd=0 |
|---|---|---|---|---|---|---|---|---|
| Adam | 0.0 | 0.0160 | 0.0280 | 0.0420 | 0.0505 | 0.0517 | 3.23× | — |
| Adam | 0.1 | 0.0159 | 0.0264 | 0.0368 | 0.0409 | 0.0403 | 2.54× | **−22%** |
| Muon | 0.0 | 0.0166 | 0.0335 | 0.0477 | 0.0551 | 0.0562 | 3.39× | — |
| Muon | 0.1 | 0.0165 | 0.0322 | 0.0430 | 0.0464 | 0.0457 | 2.77× | **−19%** |

- **Decay trims the final per-element RMS ~20%** for both (Adam −22%, Muon −19%) —
  but neither comes near POET's flat line; even decayed they still grow **~2.5–2.8×**
  from init. So decoupled weight decay does real, measurable work, yet doesn't
  replicate POET's intrinsic norm preservation.
- **It also changes the *shape*, not just the level.** With wd=0 the norm rises
  monotonically and is **still climbing at the end** (Adam 0.0505 @6k → 0.0517;
  Muon 0.0551 → 0.0562). With wd=0.1 it reaches a **quasi-equilibrium mid-run and
  slightly recedes** as the LR cosine-decays (Adam peaks ~0.041 @6k → 0.0403; Muon
  ~0.046 → 0.0457). Decay converts unbounded growth into a bounded plateau — the
  classic decoupled-decay ↔ effective-norm-target behavior.

**POET's normalized init (wd 0), for reference** — flat regardless of starting point:

| run | start | final | growth |
|---|---|---|---|
| POET (norm init) | 0.0416 | 0.0440 | 1.06× (flat) |
| POET (raw init) | 0.0146 | 0.0158 | 1.08× (flat) |

**Per-type final `row_rms/mean` (wd=0 trio).** POET stays near its (type-dependent)
init; Adam/Muon inflate every type to a common ~0.045–0.060 band:

| type | POET (raw) | Adam | Muon |
|---|---|---|---|
| q | 0.0216 | 0.0549 | 0.0513 |
| k | 0.0216 | 0.0540 | 0.0519 |
| v | 0.0212 | 0.0449 | 0.0587 |
| proj | 0.0078 | 0.0457 | 0.0523 |
| fc1_gate | 0.0220 | 0.0549 | 0.0602 |
| fc1_up | 0.0079 | 0.0550 | 0.0597 |
| fc2 | 0.0080 | 0.0526 | 0.0592 |

**Interpretation.**

- **POET does not grow the effective-weight norm, even with no weight decay**
  (~1.06–1.08× over 9000 steps, independent of init scheme), vs **~3.2–3.4× for
  Adam/Muon**. POET behaves as if strongly weight-decayed without any decay term.
- **Why:** $W_\text{eff}=R_\text{out}W_0R_\text{in}$ with $W_0$ frozen and $R$
  orthogonal (Cayley / Muon-orthogonalized). Orthogonal rotations preserve the
  Frobenius norm, so the merged weight's per-element RMS is pinned near $W_0$'s
  init. The norm constraint is **baked into the parameterization**, which is why
  POET needs no weight decay.
- **Decoupled weight decay does real work for the additive optimizers** but doesn't
  match POET: Adam 0.052 (wd0) → 0.040 (wd0.1); Muon 0.056 → 0.046. Even *with*
  decay they grow ~2.5–2.8×.
- **Muon grows slightly more than Adam** at matched wd=0 (3.39× vs 3.23×).
- **`row_rms ≈ col_rms` everywhere** ⇒ for POET, neither the left ($R_\text{out}$,
  rows) nor the right ($R_\text{in}$, cols) rotation lopsidedly inflates the
  effective weight — both stay norm-preserving, as the orthogonal-rotation theory
  predicts. No left/right asymmetry observed.

**Repro (8-GPU node).** Same three overrides on each optimizer's dev launcher; POET
uses `optim.poet.init_type=none` to match Adam/Muon's raw init, and `_nowd` runs add
`optim.weight_decay=0.0`:

```bash
bash scripts/train_poet_dev.sh experiment=optim/poet_lie_orth optim.poet.init_type=none \
  training.log_weight_norms=true training.log_weight_norms_interval=100 training.weight_norm_layers=first,mid,last
bash scripts/train_adam_dev.sh optim.weight_decay=0.0 \
  training.log_weight_norms=true training.log_weight_norms_interval=100 training.weight_norm_layers=first,mid,last
bash scripts/train_muon_dev.sh optim.weight_decay=0.0 \
  training.log_weight_norms=true training.log_weight_norms_interval=100 training.weight_norm_layers=first,mid,last
```

W&B keys: `weightnorm/L{i}/{type}/{row,col,row_rms,col_rms}/mean` + per-layer
`weightnorm/L{i}/{row,col}_rms_hist` (mean-only scalars as of 2026-06-18).

## 2.8 How to update this tracker

- **Cohort matters:** only compare runs at the same scale + tokens/param. Everything above is 60m / 40tpp. A 300m or 20x table would be a separate block.
- **Pull results** from each run's W&B summary: `runs/<dir>/**/wandb-summary.json`, keys `val/loss` / `train/loss` / `val/ppl` / `_step`. Treat `_step < 9000` as crashed/short for this cohort (full run = 9155 steps).
- **Settings** come from `runs/<dir>/resolved_config.yaml` (`optim.lr`, `optim.poet.*`, `training.tokens_per_param`).
- **Data-quality caveats from this snapshot:** the vanilla `poet` family had a high crash rate (many `_step ≪ 9155`); duplicate rows in the raw scan are same-setting reruns (e.g. the two `poet_lie_rms` lr-3e-3/c-4 dirs); `poet_h_exp_rms_c8` crashed at step 4256.
