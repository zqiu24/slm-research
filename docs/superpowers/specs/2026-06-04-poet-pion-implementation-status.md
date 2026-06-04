# POET × Pion — Implementation Status & Results (increments 0–3)

**Date:** 2026-06-04
**Status:** Implemented & GPU-validated on the 60m LLaMA-3 dev scale
**Scope:** This document records **only what has been built and measured** so far
in the POET-X × Pion line. Forward-looking ideas (μP spectral / Newton–Schulz on
the generators, second-order exp, per-head blocks, Pion-faithful `‖A·W‖`) are
*not* part of this status and appear only in the closing "Not yet implemented"
section as a scope boundary.
**Related:**
[docs/poetx_pion_pipeline.md](/lustre/fast/fast/zqiu/slm-research/docs/poetx_pion_pipeline.md),
[docs/rms_normalization_poet_interval1.md](/lustre/fast/fast/zqiu/slm-research/docs/rms_normalization_poet_interval1.md),
[poet0 design](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-03-poet0-single-step-design.md),
[lie-momentum design](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-03-poet-lie-momentum-design.md),
[lie-alternating design](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-03-poet-lie-alternating-design.md),
Pion paper Algorithm 1.

---

## 1. The regime everything runs in

All four increments share one POET configuration — the **single-step, full-matrix,
no-resample** regime — so that POET reduces (to O(η²)) to *direct-on-W Pion*:

| Knob | Value | Meaning |
|---|---|---|
| `merge_period` | `1` | Fold `R(oft_R)` into the frozen base `W` **every** step; `oft_R` is reborn at 0 (identity) each step. |
| `block_count` | `1` | One block = the **full** `d×d` skew matrix (no block-diagonal structure). |
| `reinit_period` | `-1` | **Never** resample the permutation Ψ or reset momentum — one coherent coordinate frame for the whole run. |
| `scale` | `0.5` | `oft_R` LR multiplier (`group_lr = base_lr · scale`). |
| `parameterization` | `cayley` (k=3) | Exp map used by the merge (unchanged from stock POET). |

Why this matters: at `merge_period=1` the rotation is born at identity, so the
Cayley/exp Jacobian → I, and **autograd's ambient `oft_R.grad` equals the
skew-projected tangent gradient `G_skew` to O(angle²)** — Pion's `Γ ≈ 2P`. No new
gradient plumbing is needed. At `block_count=1` the single block *is* the full
tangent gradient, so it is the correctness oracle. This is implemented in
[poet_merge_step.py `_merge_decision`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L48):
`reinit_period < 0` → fold every step, never reinit; the fp32 **master value** of
`oft_R` is zeroed every fold (anti spring-back) while momentum is left untouched.

---

## 2. What has been implemented

### Increment 0 — poet0: single-step POET

Decoupled the **fold** cadence (`merge_period`) from the **resample-Ψ + reset-momentum**
cadence (`reinit_period`), so the rotation can be folded into `W` every step while
momentum stays coherent. `reinit_period`: `>0` resample every N (multiple of
`merge_period`), `==0` legacy (resample on every fold), `<0` never resample.
Stock Megatron-Adam on `oft_R`, k=3 Cayley, two-sided. Imports **none** of the
Pion geometry — it is the substrate the other three increments sit on.
Config: [poet0.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet0.yaml)
(`merge_period=1, reinit_period=400, block_count=1, scale=0.5, q_optimizer=adam`).

### Increment 1 — Lie-algebra momentum (`q_optimizer=lie_algebra`)

New optimizer [`LieAlgebraMomentum`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L79):
skew branch on `oft_R`, AdamW branch on everything else (shaped like SkewMuon).
Pion's **first + second moment** momentum on the identity-point skew gradient,
**persisting across the per-step fold**:

```python
m ← β1·m + (1−β1)·g                      # 1st moment (Lie algebra)
v ← β2·v + (1−β2)·g²                      # 2nd moment (elementwise)  [or scalar: 2·Σg² per block]
A = −m / (√v + ε)                         # normalized skew direction
p ← lr · A                               # p born at 0 → p = lr·A; merge exponentiates & folds
```

Defaults `b1=0.9, b2=0.95, eps=1e-8`, no bias correction (matches Pion Algorithm 1).
State buffers are named `lie_m`/`lie_v` (**not** `exp_avg`) so the merge patch's
`_zero_moments` can never clobber them. `lie_v_mode ∈ {elementwise, scalar}`;
**default `elementwise`** (paper Algorithm 1). The `in`/`out` sides are split into
separate param groups (`side="in"/"out"`) — the enabler for alternating.
Config: [poet_lie.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie.yaml).

### Increment 2 — Alternating single-sided update (Pion §6 / Eq. 8)

`lie_alternating=true`: write **one** rotation side per step (out on even, in on
odd) while accumulating momentum on **both** sides every step; the inactive side
stays at identity that step (no-op fold). `lie_alternate_every` holds each side
for N steps before flipping. Implemented as the `active`-side gate in
[`step()`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L159).
Config: [poet_lie_alt.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_alt.yaml).

### Increment 3 — RMS scaling, W-free (Pion §2.4.1 / Stage 2)

`lie_rms=true`: after forming the Adam direction `A`, rescale the **generator** so
the per-plane rotation angle is dimension-consistent across matrices, **without**
touching `W`:

```python
α = rms_c · √(n_blocks · block_size) / (‖A‖ + ε)     # → ‖α·A‖ = rms_c·√d, gradient-independent
A ← α · A
```

This is the **Frobenius / W-free** variant: it normalizes the generator's own
Frobenius norm (`‖A‖`, not `‖A·W‖`), justified because POET freezes the spectrum.
The single new hyperparameter is `rms_c` (default `0.2`). The dimension constant
is `√(n_blocks·block_size) = √d`, which is **blocking-invariant** (same per-plane
angle whether `block_count=1` or per-head). Implemented at
[poet_lie_momentum.py:162-169](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L162).
Config: [poet_lie_rms.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_rms.yaml).

---

## 3. Files & touch points

| File | What it carries |
|---|---|
| [src/optim/poet_lie_momentum.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py) | `LieAlgebraMomentum` (skew + AdamW branches), `_split_poet_lie_params`, `_build_lie_param_groups`. All four increments live here. |
| [src/optim/poet.py:528](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L528) | `get_megatron_poet_lie_momentum_optimizer` builder; dispatch `if poet_q_optimizer=="lie_algebra"` at [L644](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L644) (before the muon branch). |
| [src/patches/poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py) | `_merge_decision` (fold/reinit cadence), `_reset_vanilla_oft_state` (master-value zero, conditional moment reset). |
| [launchers/pretrain_gpt_slm.py:70-91](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L70) | CLI flags: `--poet-q-optimizer {adam,muon,lie_algebra}`, `--poet-lie-b1/-b2/-eps`, `--poet-lie-v-mode`, `--poet-lie-alternating`, `--poet-lie-alternate-every`, `--poet-lie-rms`, `--poet-lie-rms-c`, `--poet-merge-period`, `--poet-reinit-period`. |
| [src/utils/megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py) | `kind=="poet"` branch emits all the above from the resolved config. |
| [src/patches/poet_optimizer_setup.py:46-53](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py#L46) | Threads `poet_lie_*` onto the OptimizerConfig. |
| Configs | [poet0](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet0.yaml), [poet_lie](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie.yaml), [poet_lie_alt](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_alt.yaml), [poet_lie_rms](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_rms.yaml). |
| Scripts | [train_poet0.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_poet0.sh), [train_poet_lie.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_poet_lie.sh), [train_poet_lie_alt.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_poet_lie_alt.sh), [train_poet_lie_rms.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_poet_lie_rms.sh) (60m default, `weight_decay=0.1`, `scheduler=cosine_poet`). |

**Constraints inherited from the builder** (dev-only, acceptable for the 60m
ablation): no distributed/sharded optimizer, no TP/PP > 1, bf16 (not fp16).

---

## 4. Results (60m LLaMA-3 dev, `slm-zeju-dev`)

All runs: 60m LLaMA-3, seq 256, GBS 1024, `ablation_40x` token budget,
`block_count=1, merge_period=1, reinit_period=-1`, `weight_decay=0.1`,
`min_lr_ratio` from `cosine_poet`. Numbers are final val loss.

| Optimizer on the rotation | Config | Val loss | Note |
|---|---|---|---|
| AdamW baseline | `experiment=optim/adam` | **3.42** | reference floor |
| Muon (hybrid: Muon on hidden 2D, AdamW on rest) | `experiment=optim/muon_hybrid` | **3.44** | the target to match |
| Lie momentum, **scalar** v, no RMS | `poet_lie` `lie_v_mode=scalar` | **5.37** | plateaus — diagnosed below |
| Lie momentum, **elementwise** v, no RMS | `poet_lie` (default) | **~3.50–3.51** | off-baseline toward Muon; the working Stage-1 result |
| Lie momentum, elementwise v, **RMS** (best) | `poet_lie_rms`, eff∠ ≈ 0.006 (run `bl9241ve`) | **3.48** | **beats no-RMS 3.51, approaches Muon 3.44** |
| Lie momentum, elementwise v, **RMS** (too hot) | `poet_lie_rms`, scale=2, c=4, eff∠ ≈ 0.008 (run `tk4864zk`) | worse | overshoot, not a bug |

### The governing variable for RMS runs

RMS makes the step magnitude **gradient-independent**, so the loss is a clean
function of a single **effective per-plane rotation angle**:

```
eff∠ = lr · scale · rms_c     (= the RMS angle the generator rotates each plane)
```

Empirically on this scale:

| eff∠ | behavior |
|---|---|
| 0.002 – 0.006 | **sweet spot** — best run (`bl9241ve`, eff∠ 0.006) = **3.48** |
| ~0.008 (`tk4864zk`) | too hot — overshoots, worse than the sweet spot |
| ≥ 0.012 | unstable — loss spikes (max Δ ≈ 2.96) |

---

## 5. What we learned (diagnoses behind the numbers)

1. **Elementwise v is the right default, not scalar.** Scalar-v normalizes by the
   *block* Frobenius norm (`2·Σg²` per block), which √d-suppresses the per-entry
   step → 5.37 plateau. Elementwise (paper Algorithm 1) reaches 3.50. This is why
   `lie_v_mode` **defaults to elementwise** in the optimizer, configs, and scripts.
2. **At interval-1 the ambient `oft_R.grad` *is* the Lie gradient.** The
   identity-point shortcut held: no exact-tangent-gradient plumbing was needed to
   move the loss off the Adam baseline.
3. **Lie momentum persists across the per-step fold.** `lie_m`/`lie_v` naming +
   `reset_moments=False` at `reinit_period=-1` keep momentum coherent across folds;
   no periodic spikes (unlike the Adam path's momentum-reset spikes).
4. **RMS at the *right* angle helps; the failure mode is calibration, not bug.**
   The "RMS made it worse" runs were simply too hot (eff∠ > 0.006). At the sweet
   spot, RMS (3.48) **beats** the no-RMS Lie run (3.51) and closes most of the gap
   to Muon (3.44) / Adam (3.42). Pion's `rms_c=0.2` is a *per-entry* calibration;
   our `√d` constant is *per-plane*, so the matching `c` is ≈ √d larger — this is
   the folded-in degeneracy `eff∠ = lr·scale·c`.
5. **Muon and POET wrap the same layers.** Both unfuse qkv/fc1 and target the
   hidden 2D linears (q/k/v/o/gate/up/down), both skip lm_head + embeddings →
   AdamW. The head-to-head differs only in update rule + LRs, so the comparison is
   apples-to-apples.

---

## 6. Verification status

- **CPU tests (run, passing):** arg-translation + flag-default tests in
  [tests/unit/test_pretrain_gpt_slm.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_pretrain_gpt_slm.py)
  (lie / alternating / rms flags), `test_megatron_args.py`, optimizer-math unit
  tests (vec-space ≡ skew-space; scalar-v ×2 Frobenius factor; `‖α·A‖ = rms_c·√d`
  gradient-independence), `py_compile` / `ruff` on edited files, script dry-runs.
- **GPU runs (user-launched):** the 60m dev ablations above on `slm-zeju-dev`.

---

## 7. Not yet implemented (scope boundary — not part of this status)

These are the deferred Pion levers, listed only to delimit "what we have done":

- **μP spectral-norm control / Newton–Schulz on the generators** (Pion §2.6,
  Scheme I/II) — the current RMS controls the *Frobenius* (average) angle, not the
  *spectral* (max) angle μP requires for LR transfer across widths. The discussed
  next step.
- **Second-order exp `E2` / low-order Cayley** (Pion §2.4.4) — still using the
  merge's Cayley k=3.
- **Per-head blocks** (`block_count = n_heads`, Pion App. D.1) — currently
  `block_count=1` (full matrix).
- **Pion-faithful `‖A·W‖` RMS** — we implemented the W-free `‖A‖` variant only.
- **Exact two-sided tangent gradient**, **sharded merge** (multi-GPU), **transported
  ambient momentum** (Pion found Lie better), **bilateral normalization** (Pion did
  not adopt).
