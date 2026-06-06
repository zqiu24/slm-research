# POET: Parameter-Efficient Orthogonal Training

> **Last updated: 2026-06-06.** Part 1 below is the conceptual reference (math,
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
| **Muon-like orthogonalizing Q-opt** | `optim.poet.q_optimizer=lie_ortho`, `lie_ortho_c`/`_method`/`_ns_steps` (`poet_lie_orth`) | standalone `LieOrthMomentum`: same Lie 1st-moment momentum, but **orthogonalize** the skew direction (all planes → ~same angle) instead of RMS-scaling; `muon` band (~5 NS steps) or `spectral` exact `A(−A²)^{-1/2}` (~20) ([poet_lie_orth.py:27](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L27)) | ✅ | **best POET so far** (3.567 @ lr 3e-3, c=8) — closes the gap to adam (3.557); sweeps in progress |
| Head-aligned attention rotation | `optim.poet.head_aligned_attn=true` / `--poet-head-aligned-attn` (`poet_lie_head`, `poet_h_*`) | swap q/k/v/o to `HeadAlignedPOETLinear`: per-head block-diagonal rotation (block=head_dim, fixed identity Ψ), needs unfused qkv ([head_aligned_layer.py:28](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/head_aligned_layer.py#L28), [poet_layers.py:245-257](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L245-L257)) | ✅ | **neutral→hurts** at 60m (3.654 vs non-head 3.634 at matched lr/c) |
| Residual-side perm off | `optim.poet.head_resid_perm=false` / `--poet-no-head-resid-perm` (`poet_h_noperm_*`) | freeze the residual (non-head) side's Ψ in head-aligned mode | ✅ | neutral (3.6536 vs 3.6541) |
| Alternating single-sided update | `optim.poet.lie_alternating=true`, `lie_alternate_every` (`poet_lie_alt`) | write only one rotation side per step ([poet_lie_momentum.py:126-130](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L126-L130)) | ✅ | **hurts** (3.709 vs poet_lie 3.647) |
| `exp` parameterization | `optim.poet.parameterization=exp` | exact matrix-exponential orthogonal map (vs truncated Cayley); incompatible with caching | ✅ | **hurts** vs cayley (3.70–3.82) |
| Muon-on-Q (SkewMuon) | `optim.poet.q_optimizer=muon`, `muon_theta/ns_steps/momentum` | per-block Newton-Schulz orthogonalize + constant-angle θ rescale; built for the no-reset regime ([poet_skew_muon.py:120](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_skew_muon.py#L120)) | ✅ | hurts so far (≈3.79); needs `merge_period=0` tuning, not yet done |
| Cayley cache (Mode A) | `optim.poet.cache_mode=cached_fwd_bwd` | cache `R` within a grad-accum cycle, flush one VJP at cycle end ([poet_cache.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_cache.py)) | ✅ | perf-only; measured dead-end for small K (no quality effect) |
| Normalized / μP base init | `optim.poet.init_type`, `mup_alpha` | row-normalize frozen `W` (+ optional μP spectral scale) ([poet_layers.py:44-62](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L44-L62)) | ✅ | `normalized` is default; not separately ablated (sweeps fixed `mup_alpha=1.0`) |
| Single-sided rotation (freeze output) | `optim.poet.train_output_rotation=false` / `--poet-freeze-output-rotation` | train only `R_in`, freeze `R_out=I` | ✅ | not ablated at scale |

Q-optimizer dispatch (`lie_algebra` / `lie_ortho` / `muon` / default `adam`) lives in [poet.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py); `lie_algebra` and `lie_ortho` share the same builder, which branches to construct `LieOrthMomentum` at [poet.py:596](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L596). CLI→flag routing in [megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py).

**The `lie_ortho` optimizer (new — current best POET).** A standalone `LieOrthMomentum` ([poet_lie_orth.py:27](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L27)), selected by `optim.poet.q_optimizer=lie_ortho`. It keeps the same Lie-algebra **first-moment** momentum on `oft_R` as `lie_algebra` (persists across folds), but replaces the direction→generator transform: instead of RMS-scaling, it **orthogonalizes** the per-block skew direction so **every rotation plane turns by ~the same angle** — Muon's "trust the subspace, not the per-direction magnitude" bet, applied to *rotational* updates. Two methods (`optim.poet.lie_ortho_method`):
- **`muon`** (default, ~5 NS steps): Muon's quintic Newton–Schulz on the direction, then a `½(X−Xᵀ)` cleanup. NS *preserves skew* on a skew input (verified to ~1e-15) and lands the singular values in a **band** around 1 — cheap, approximately-equal angles.
- **`spectral`** (~15–20 NS steps): the exact Löwdin form `A·(−A²)^{-1/2}` — drives every σ to *exactly* 1, ≈4× the cost.

Realized per-plane angle = `lr · scale · ortho_c` (under `muon` the band makes `ortho_c` *nominal*, ≈0.75–1.0× that). First-moment-only by default (a second moment is partly undone by orthogonalization). Design doc: [docs/muon_orthogonalizing_optimizer_poet.md](/lustre/fast/fast/zqiu/slm-research/docs/muon_orthogonalizing_optimizer_poet.md); plan: [docs/superpowers/plans/2026-06-05-poet-lie-orth-optimizer.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/plans/2026-06-05-poet-lie-orth-optimizer.md). **Status (2026-06-06):** champion is `5sbgancm` (c=8, lr 3e-3, muon, head-aligned) at **val/loss 3.5669**, **reproduced 4× exactly** across the lr/scale/variant sweeps — nothing beats it. Full sweep verdicts are in §2.5; headline: **ortho ≫ RMS** (matched-angle RMS sibling diverged), **muon-band ≈ exact `spectral`**, **1st-moment > 2nd**, and **`scale=0.5` is the stability ceiling** (`scale=0.75` diverged). Sweeps: [_lr](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_lr.sh), [_scale](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_scale.sh), [_variants](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_variants.sh), [_c](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_c.sh) (the c-sweep traces the same effective-angle axis as `_scale`). Still open: the `head`-on/off arm (`lieorth_c8_nohead`).

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
| [poet_lie_orth](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_orth.yaml) | **lie_ortho** | 1 / −1 | — (ortho c=4, muon) | **yes** | no | **Muon-like orthogonalizing** optimizer (equal-angle planes) — **current best POET** |

The `poet_h_*` / `poet_dense_*` runs in §2.4 are CLI sweeps over `poet_lie_rms` (± `head_aligned_attn`, varying `lie_rms_c`), not separate config files.

## 2.3 Results — which designs are useful

Best completed run per setting, ranked by `val/loss` (60m / 40 tokens-per-param):

| # | Setting | val/loss | (ppl) | train | lr | rms/ortho c | head | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | **muon_kimi** | **3.5321** | 34.20 | 3.4219 | 1e-3 | — | — | best overall (run `of4bakqd`) |
| 2 | adam (baseline) | 3.5570 | 35.06 | 3.4575 | 1e-3 | — | — | best dense baseline |
| 3 | **poet_lie_orth (c8)** | **3.5669** | 35.41 | 3.4691 | **3e-3** | 8 (ortho) | yes | **best POET** (`5sbgancm`, **reproduced 4×** across sweeps) |
| 4 | muon_hybrid | 3.5698 | 35.51 | 3.4705 | — | — | — | |
| 5 | poet_lie_orth (c4) | 3.5715 | 35.57 | 3.4701 | 3e-3 | 4 (ortho) | yes | run `z1gpz9y7` |
| 6 | poet_lie_rms | 3.6193 | 37.31 | 3.5220 | 3e-3 | 4 (rms) | no | best RMS-family (`98293d1u`; head-aligned twin `l2pzawa4` 3.6335 — worse) |
| 7 | poet_dense_rms (c8) | 3.6344 | 37.88 | 3.5367 | 1e-3 | 8 (rms) | no | |
| 8 | poet_lie_rms (c8) | 3.6404 | 38.11 | 3.5367 | 1e-3 | 8 (rms) | no | |
| 9 | poet_lie | 3.6474 | 38.37 | 3.5437 | 1e-3 | — | no | Stage 1 |
| 10 | poet_lie_rms (c4) | 3.6496 | 38.46 | 3.5478 | 1e-3 | 4 (rms) | no | same as #6 but lr 1e-3 |
| 11 | poet0 | 3.6518 | 38.55 | 3.5484 | 1e-3 | — | no | |
| 12 | **poet_h_noperm_rms_c8** | 3.6536 | 38.61 | 3.5578 | 1e-3 | 8 (rms) | **yes** | best head-aligned (RMS family) |
| 13 | poet_h_rms_c8 | 3.6541 | 38.63 | 3.5588 | 1e-3 | 8 (rms) | yes | |
| 14 | poet (vanilla, cayley) | ≈3.70 | ≈40.6 | ≈3.60 | 1e-3 | — | no | weakest POET family |
| — | poet `exp` / Muon-on-Q | 3.70–3.82 | 41–46 | — | 1e-3 | — | no | regressions |

**Conclusions (what's useful):**
- **`lie_ortho` is the breakthrough — POET now nearly matches the baselines.** Best POET (poet_lie_orth, **3.5669** @ c=8) is only **+0.010** val/loss vs adam (3.557), **beats muon_hybrid** (3.570), and sits **#3 overall** behind only muon_kimi (3.532) and adam. That's a **−0.057** jump over the previous best POET (poet_lie_rms 3.626) — the long-standing POET gap is now ~closed. The champion is **reproduced 4× at exactly 3.5669**, so it's not noise.
- **Orthogonalizing the rotation direction (`lie_ortho`) beats RMS-scaling it (`lie_rms`)** — now confirmed by the variants sweep: at the matched champion angle the RMS sibling `lierms_c8` **diverged** (val 6.38) while ortho held 3.567. Also from the sweeps: **muon-band ≈ exact `spectral`** (3.5669 vs 3.5703 — the cheap ~5-step quintic is enough, no need to pay for exact σ=1), **1st-moment beats 2nd** (3.5669 vs 3.5702), and for the angle **c=8 ≳ c=4** (3.5669 vs 3.5715). The `head`-on/off arm (`lieorth_c8_nohead`) is still running.
- **The useful POET stack** is *single-step merge + Lie-algebra momentum + an angle-equalizing transform*: vanilla `poet` (≈3.70) → `poet_lie` (3.647) → `poet_lie_rms` (3.619) → **`poet_lie_orth` (3.567)**, all with **lr 3e-3**.
- **The rotation angle has a hard ceiling.** `scale=0.5` (eff∠ 0.012) is best and stable; `scale=0.75` (eff∠ 0.018) **diverged** (val 4.55). Since c and scale are interchangeable under `muon` (`scale0.25/c8` ≡ `scale0.5/c4` → both 3.5715), the c-sweep's c≥12 arms (eff∠ ≥ 0.018) are expected to diverge too — keep the angle near eff∠ 0.012.
- **The `c` knob has a sweet spot:** for RMS, c≈4 best at lr 3e-3 (c=8 *diverged* at this lr); for ortho c=8 ≳ c=4 and the band is wider. Larger c over-rotates.
- **Head-aligned attention does NOT help.** Confirmed again at the best RMS recipe (lr 3e-3/c=4): head-aligned `l2pzawa4` 3.6335 vs no-head `98293d1u` 3.6193 (and earlier at lr 1e-3/c=8: head 3.654 vs dense 3.634). The best `lie_ortho` runs **are** head-aligned, but whether head-alignment helps *with* ortho is unresolved until `lieorth_c8_nohead` finishes.
- **Alternating, `exp` parameterization, and (reset-regime) Muon-on-Q are current regressions.** Muon-on-Q (SkewMuon) was built for the `merge_period=0` no-reset regime and hasn't been retuned for it.

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

## 2.5 lie_ortho sweep results (as of 2026-06-06)

Champion `5sbgancm` (lr 3e-3, scale 0.5, c=8, muon, head-aligned) = **val/loss 3.5669**, reproduced 4× (`lieorth_c8_muon`, `lieorth_lr0.003`, `lieorth_scale0.5`, + the original anchor). Completed arms, by sweep:

**A — global LR** ([sweep_lie_orth_lr.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_lr.sh)): 1e-3 → 3.6259 · 2e-3 → 3.5683 · **3e-3 → 3.5669** · (4e-3, 6e-3 not yet complete). Peak at 3e-3 so far.

**B — POET scale / rotation-lr** ([sweep_lie_orth_scale.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_scale.sh)): 0.25 → 3.5715 · **0.5 → 3.5669** · **0.75 → 4.5527 (DIVERGED)** · (1.0, 1.5 incomplete). Stability ceiling at scale 0.5 (eff∠ 0.012).

**C — variants at the champion angle** ([sweep_lie_orth_variants.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_variants.sh)):

| arm | change vs control | val/loss |
|---|---|---|
| `lieorth_c8_muon` (control) | — | 3.5669 |
| `lieorth_c8_spectral` | exact σ=1 (ns=20) | 3.5703 |
| `lieorth_c8_2mom` | second moment on | 3.5702 |
| `lierms_c8` | RMS instead of ortho | **6.3797 (DIVERGED)** |
| `lieorth_c8_nohead` | head-aligned off | *running* |

→ muon-band ≈ exact spectral; 1st-moment ≥ 2nd; **ortho far more stable than RMS** at this angle; head on/off unresolved.

**D — native c** ([sweep_lie_orth_c.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_lie_orth_c.sh)): not yet run. Degenerate with sweep B (eff∠ = lr·scale·c, and `orthogonalize(−m)` is magnitude-free), so c≥12 (eff∠ ≥ 0.018) will likely diverge like scale=0.75; the informative band is c≤8–10.

## 2.6 Best runs leaderboard (settings + result)

> Keep this current: when a run beats its family's entry, replace it (cite the run dir + W&B id).

**🏆 Overall best (60m/40tpp):** [`muon_kimi-…-20260605T142324Z`](/lustre/fast/fast/zqiu/slm-research/runs/muon_kimi-llama3-60m-s42-20260605T142324Z) (W&B `of4bakqd`) — **val/loss 3.5321, ppl 34.20**, train 3.4219, 9155 steps, lr 1e-3. (Earlier `…20260602T134241Z` was 3.5352; this is a same-config rerun.)

**🥇 Best POET:** [`poet_lie_orth-…-20260605T190018Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_orth-llama3-60m-s42-20260605T190018Z) (W&B `zeju-qiu/slm-zeju-dev/5sbgancm`) — **val/loss 3.5669, ppl 35.41**, train 3.4691, 9155 steps. The Muon-band orthogonalizing optimizer: `experiment=optim/poet_lie_orth`, **lr=0.003**, **lie_ortho_c=8** (method=muon, head-aligned; all other knobs = config default). Beats the previous best POET by **−0.059** and nearly matches the adam baseline (3.557). Command:
```bash
codexlog poet_lie_orth_best bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 \
  optim.poet.lie_ortho_c=8
```
*Previous best POET (RMS family):* [`poet_lie_rms-…-20260604T140255Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_rms-llama3-60m-s42-20260604T140255Z) (W&B `tx67fwih`) — val/loss 3.6257 @ lr 3e-3, c=4 (twin [`…-20260604T124303Z`](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_rms-llama3-60m-s42-20260604T124303Z), identical).

**Per-family best:**

| Family | Run dir | val/loss | (ppl) | key settings |
|---|---|---|---|---|
| muon_kimi | [muon_kimi-…20260605T142324Z](/lustre/fast/fast/zqiu/slm-research/runs/muon_kimi-llama3-60m-s42-20260605T142324Z) (`of4bakqd`) | 3.5321 | 34.20 | lr 1e-3 |
| adam | [adam-…20260605T142335Z](/lustre/fast/fast/zqiu/slm-research/runs/adam-llama3-60m-s42-20260605T142335Z) (`ylrd45af`) | 3.5570 | 35.06 | lr 1e-3 |
| muon_hybrid | [muon-…20260602T001936Z](/lustre/fast/fast/zqiu/slm-research/runs/muon-llama3-60m-s42-20260602T001936Z) | 3.5698 | 35.51 | |
| **poet_lie_orth** | [poet_lie_orth-…20260605T190018Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_orth-llama3-60m-s42-20260605T190018Z) (`5sbgancm`) | **3.5669** | 35.41 | lr 3e-3, c=8, muon, head-aligned |
| poet_lie_rms (RMS family) | [poet_lie_rms-…20260605T142434Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie_rms-llama3-60m-s42-20260605T142434Z) (`98293d1u`) | 3.6193 | 37.31 | lr 3e-3, c=4 |
| poet_lie | [poet_lie-…20260603T183821Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_lie-llama3-60m-s42-20260603T183821Z) | 3.6474 | 38.37 | lr 1e-3 |
| poet0 | [poet0-…20260603T165332Z](/lustre/fast/fast/zqiu/slm-research/runs/poet0-llama3-60m-s42-20260603T165332Z) | 3.6518 | 38.55 | lr 1e-3 |
| head-aligned | [poet_h_noperm_rms_c8-…20260605T112512Z](/lustre/fast/fast/zqiu/slm-research/runs/poet_h_noperm_rms_c8-llama3-60m-s42-20260605T112512Z) | 3.6536 | 38.61 | lr 1e-3, c=8, noperm |
| poet (vanilla) | `runs/poet-llama3-60m-s42-*` | ≈3.70 | ≈40.6 | lr 1e-3, merge_period 400 |

## 2.7 How to update this tracker

- **Cohort matters:** only compare runs at the same scale + tokens/param. Everything above is 60m / 40tpp. A 300m or 20x table would be a separate block.
- **Pull results** from each run's W&B summary: `runs/<dir>/**/wandb-summary.json`, keys `val/loss` / `train/loss` / `val/ppl` / `_step`. Treat `_step < 9000` as crashed/short for this cohort (full run = 9155 steps).
- **Settings** come from `runs/<dir>/resolved_config.yaml` (`optim.lr`, `optim.poet.*`, `training.tokens_per_param`).
- **Data-quality caveats from this snapshot:** the vanilla `poet` family had a high crash rate (many `_step ≪ 9155`); duplicate rows in the raw scan are same-setting reruns (e.g. the two `poet_lie_rms` lr-3e-3/c-4 dirs); `poet_h_exp_rms_c8` crashed at step 4256.
