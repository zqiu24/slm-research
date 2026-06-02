# POET × Muon: Spectral Optimization of the Orthogonal Tangent Space

**Goal for the coding agent:** implement and evaluate the hypothesis that POET's performance gap to Muon comes not from POET's frozen *weight* spectrum, but from the fact that POET optimizes its skew-symmetric tangent parameter `Q` with AdamW — an optimizer blind to matrix geometry. We test whether applying Muon-style update orthogonalization to `∂f/∂Q` closes (or exceeds) the gap.

This is a research spec, not a fixed recipe. Execute it as **gated stages**: do not proceed to a later stage until the gate of the previous stage passes. **Start at Stage 0 — it is two cheap probes that decide whether this whole line of work is even on the right axis, and either one can kill it before any optimizer code is written.** Respect the gates.

---

## 0. Background context the agent needs

POET reparameterizes each weight as `W = R W0 P` with `W0` a fixed random matrix and `R, P` trainable orthogonal matrices. Neither `R` nor `P` is optimized directly. Each is built from a block-diagonal set of small `b×b` orthogonal blocks, and each block is parameterized by a skew-symmetric `Q ∈ R^{b×b}` (`Q = -Qᵀ`) via the Cayley–Neumann parameterization (CNP), with `k=3` terms in POET-X:

```
G ≈ I + 2Q + 2Q² + 2Q³ + Q⁴        (forward, CNP k=3)
```

The actual trainable tensor is the upper-triangular part of `Q` (`b(b-1)/2` params per block). The optimizer steps on `Q`; CNP then maps `Q → G` (orthogonal block); blocks + permutations assemble `R, P`.

**Two distinct spectra — do not conflate them:**
- *Weight spectrum* `σ(W)`: frozen by design (POET's headline property). Already well-behaved. Leave it alone.
- *Update spectrum*: the singular values of the **step taken on `Q` each iteration**. POET imposes no control here; AdamW produces it. **This is the object we suspect is degenerate, and the object Muon is designed to fix.**

**Core hypothesis (H):** `∂f/∂Q` is ill-conditioned (heavy-tailed singular values) under normal POET training. AdamW under-serves the small-singular-value directions. Because `Q` lives on a curved space (the Lie algebra `so(b)`, tangent to the rotation group at identity), poor step conditioning is amplified when CNP maps the step back onto the manifold. Muon-style orthogonalization of `∂f/∂Q` equalizes the step across directions and should improve final perplexity.

**Reference code & environment (from the papers):**
- POET / POET-X is built on top of the GaLore codebase: `https://github.com/jiaweizzhao/GaLore` (Apache 2.0).
- The `∂f/∂Q` expression for CNP (k=3) backward, from POET-X §3.4 — implement/verify against this:
  ```
  ∇1 = ∂f/∂G
  ∇2 = ∇1 Qᵀ + Qᵀ ∇1
  ∂f/∂Q = 2∇1 + 2∇2 + 2∇3 + ∇4
        = 2(∇1 + ∇2) + (2Qᵀ + (Q²)ᵀ)∇2 + (2∇1 + ∇2)(Q²)ᵀ
  ```
  (`∇3, ∇4` defined in the paper; the fused form above is what the Triton kernel computes.)
- POET-X kernels are Triton; the merge-then-reinitialize ("reset") cadence is `Tm = 400` steps by default.

---

## Stage 0 — DIAGNOSTIC BATTERY: is POET training even optimization-limited, and does Muon-on-Q have traction?

**Run this entire stage before writing any optimizer code.** It contains two cheap, decisive probes that partition the hypothesis space. Each has a pre-committed *null* (what "no slack here" looks like) and *positive* (what "exploitable slack" looks like). Do not interpret a plot that has no stated null — if you can't say in advance what "fine" looks like, the probe is worthless.

These two probes answer two orthogonal questions:
- **0A** — *Can* POET represent a good solution at all (representation-limited vs. optimization-limited)?
- **0B** — *If* it's optimization-limited, is the specific fix (Muon-on-Q) viable (is `∂f/∂Q` ill-conditioned)?

Run **0A first** — if it says "representation-limited," the entire Muon-on-Q line (Stages 1–4) is the wrong tree and you should report that and stop.

---

### Probe 0A — Single-batch overfit: optimization-limited vs. representation-limited

**Question it settles:** is the POET↔Muon gap because POET *cannot reach* good solutions (frozen spectrum / block structure imposes a representational floor), or because it *can reach but optimizes there poorly*? This is the single most important fork in the investigation, and right now it's unknown which side we're on. Everything downstream (optimizer vs. parameterization) depends on the answer.

**Setup:**
1. Small model (Llama-60M is fine), single fixed minibatch, **all regularization off** (no weight decay, no dropout). Goal is pure memorization capacity, not generalization.
2. Train to drive **training loss on that one batch toward zero**, as far as it will go, for a fixed large step budget.
3. Run three arms, identical batch/seed/budget:
   - **AdamW-direct** (standard training, no POET) — the representational ceiling reference.
   - **POET** (vanilla, AdamW on Q, normal `Tm`).
   - **POET, reset disabled** (no merge-then-reinitialize for this run) — isolates whether the reset, not the parameterization, is what caps memorization.
4. Plot training-loss-vs-step for all three on the same axes. Report the floor each reaches.

**Gate criterion:**
- **Null (optimization-limited — the good case for this project):** POET reaches a floor *comparable to* AdamW-direct (within small margin). Representation is fine; the gap is optimization/generalization → **the Muon-on-Q line is well-motivated. Proceed to 0B.**
- **Positive (representation-limited):** POET cannot drive the batch loss as low as AdamW-direct, by a clear margin, *even with reset disabled*. There is a representational floor baked into the parameterization → **no optimizer fixes this.** Stop the Muon-on-Q line; redirect effort to the parameterization (learnable-Σ, Appendix A; or block/mixing structure, the expressiveness probes). Report which.
- **Diagnostic sub-case:** if POET-with-reset is floored but POET-without-reset matches AdamW, the *reset* is the representational bottleneck, not the parameterization — this directly motivates the momentum-transport work (Stage 3) and means `Tm` tuning alone may recover most of the gap.

**Why it's near-free:** one small model, one batch, three short runs. An afternoon. It can save the entire Stage 1–4 effort from being spent on the wrong axis.

**Deliverable:** `stage0a_overfit.md` — the three-arm loss-floor plot and a one-line verdict: OPTIMIZATION-LIMITED (→ 0B) / REPRESENTATION-LIMITED (→ Appendix A) / RESET-LIMITED (→ Stage 3).

---

### Probe 0B — Conditioning of `∂f/∂Q` (only if 0A says OPTIMIZATION-LIMITED)

**Question it settles:** given that POET *can* reach good solutions but optimizes there poorly, is the poor optimization specifically due to ill-conditioned steps on `Q` that Muon could fix? If `∂f/∂Q` is already well-conditioned, AdamW is not under-serving any directions and Muon-on-Q buys nothing.

This is the gate previously written as "Stage 1"; its implementation is detailed immediately below (now **Stage 1**). Treat Stage 1 as the body of Probe 0B.

**Gate criterion (summary; full detail in Stage 1):**
- **Null:** flat spectrum, stable rank ≈ b, condition number ~O(1) and stable over training → Muon-on-Q has nothing to bite on. Stop; pivot to learnable-Σ (Appendix A).
- **Positive:** heavy-tailed singular values, stable rank ≪ b, condition number growing over training → AdamW is starving the tail directions; Muon-on-Q is well-motivated. Proceed to Stage 2. These plots become Figure 1.

---

### Stage 0 decision summary

| 0A result | 0B result | Action |
|-----------|-----------|--------|
| Optimization-limited | `∂f/∂Q` ill-conditioned | **Proceed to Stage 2** (Muon-on-Q) — the main line |
| Optimization-limited | `∂f/∂Q` well-conditioned | Pivot to **Appendix A** (learnable-Σ); optimizer is not the issue |
| Reset-limited | (run anyway) | Prioritize **Stage 3** (momentum transport / `Tm` tuning) |
| Representation-limited | — | Stop Muon line; redirect to **parameterization** (Appendix A / mixing structure) |

**Do not implement any optimizer changes until this table resolves to a row.** Stop here for human review after producing `stage0a_overfit.md` and (if reached) the Stage 1 conditioning report.

---

## Stage 1 — GATE: measure the conditioning of `∂f/∂Q` (this is the body of Probe 0B)

**This stage decides whether the rest of the project is worth doing. If the gradient is already well-conditioned, the hypothesis is dead — stop and report.**

### Task
1. Take a **vanilla POET (or POET-X) run on Llama-130M**, C4, standard hyperparameters (AdamW on `Q`, `lr(POET)=5e-4·γ`, `γ=0.5`, `Tm=400`, block size `b=256`). Use the existing codebase unchanged.
2. Add a non-invasive hook that, every N steps (e.g. every 2000), captures `∂f/∂Q` for a fixed sample of blocks — pick ~8 blocks spanning different layers (early/mid/late) and different projections (q_proj, v_proj, mlp.down_proj, mlp.up_proj), for both `R` and `P` factors.
3. For each captured per-block gradient (reconstruct the full skew-symmetric `b×b` from the upper-triangular trainable vector first), compute its singular values via `torch.linalg.svdvals`. Skew-symmetric matrices have paired singular values — that's expected, account for it.
4. Log to W&B (`wandb.nk-slm.com` is available): per-block singular-value histograms over training, plus summary scalars: condition number `σ_max/σ_min`, stable rank `‖·‖_F² / ‖·‖_2²`, and the ratio `σ_max / σ_median`.

### Gate criterion
- **Heavy-tailed / high condition number / low stable rank** (e.g. stable rank ≪ b, condition number growing over training) → hypothesis is plausible. **Proceed to Stage 2.** Save these plots; they are Figure 1 of the paper.
- **Flat spectrum / stable rank ≈ b / condition number ~O(1) and stable** → AdamW is not under-serving any directions. **Stop the Muon-on-Q line.** Muon-on-Q will buy little. Pivot to the spectrum branch (Appendix A) — note its gate is the *alignment* measurement (A0), not spectrum velocity, since POET Figure 1 already shows the loss moves the spectrum substantially under direct training and Figure 6 shows POET already has the highest entropy. The open question is alignment, not motion.

### Deliverable
A short markdown report (`stage1_conditioning.md`) with the plots and a one-line verdict: PROCEED or STOP, with the numbers that justify it.

---

## Stage 2 — Minimal Muon-on-Q experiment (only if Stage 1 says PROCEED)

### The key confound to control: the merge-then-reinitialize reset
Muon relies on **momentum**. POET resets `Q → 0` and resamples the permutation every `Tm=400` steps, which destroys any momentum buffer on `Q`. A naive Muon drop-in will be crippled by this reset. So the minimal experiment **must remove this confound** before drawing conclusions.

### Implementation
1. Implement a **Muon-style update on `Q`**. Per block, per step:
   - Take the per-block gradient `g = ∂f/∂Q` (full skew-symmetric `b×b`).
   - Apply momentum: `m ← μ·m + g` (Nesterov optional, μ≈0.95).
   - Orthogonalize the momentum via **Newton–Schulz iteration** (the standard Muon ~5-iteration quintic; reuse the Gram Newton–Schulz / symmetric-GEMM kernels already in the nk-slm stack if available — they target exactly this). Call the result `O ≈ orthogonalize(m)`.
   - **Symmetrize to stay in the tangent space:** the orthogonalized step must remain skew-symmetric. Project: `O_skew = (O - Oᵀ)/2`. (Newton–Schulz on a skew-symmetric input should approximately preserve skew-symmetry, but project explicitly to be safe — verify this empirically, see checks below.)
   - Step: `Q ← Q - step`, where `step` is computed by the **rotation-angle scaling rule defined in the subsection below** — NOT a naive Muon RMS-match. This is load-bearing; read the scaling subsection before implementing this line.
2. **Only update the 2D `Q` blocks with Muon.** Everything that is not a `Q` block (embeddings, norms, biases, any scalar/1D params, the base optimizer for non-POET params) stays on AdamW — same hybrid discipline as standard Muon usage.

### Update scaling — control the realized rotation angle, not the Q-step magnitude

**Why a naive RMS-match is wrong here.** In vanilla Muon the orthogonalized update *is* the weight update, so matching its magnitude to Adam's is the whole story. In POET the orthogonalized step lands on `Q`, not on `W`, and reaches `W` only after two non-magnitude-preserving hops:

```
step on Q  →  Q  →  G = CNP(Q) ≈ I + 2Q + 2Q² + 2Q³ + Q⁴  →  W ← R(G) · W · P
```

So a unit of `‖step on Q‖` does not map linearly to a unit of `‖ΔW‖`:
1. **CNP linearization:** for small Q, `G ≈ I + 2Q`, i.e. the induced rotation angle ≈ `2·‖step on Q‖`. The factor of 2 is exactly what the existing `η_POET = γ·η_AdamW`, `γ=0.5` was partly compensating — but that γ was tuned against AdamW's step-magnitude *distribution*. Newton–Schulz destroys gradient magnitude and replaces it with a flat `σ≈1` profile, so the magnitude distribution is completely different and γ=0.5 is now calibrated against the wrong reference.
2. **Multiplicative composition:** `‖ΔW‖` from a fixed-angle rotation depends on the energy of the block being rotated, so the same Q-step produces different `‖ΔW‖` on different blocks.

**What to actually hold constant.** The smooth, appropriate quantity is the **realized rotation angle per step**, calibrated against what healthy vanilla-POET+AdamW produces in Phase II (measure this once; it is the reference, exactly as Adam is Muon's reference). Newton–Schulz already gives a magnitude-free direction (`O_skew`, all singular values ≈ 1); we re-inject a *controlled* magnitude so the **realized** angle hits a target:

```
O_skew = skew(NS(momentum))                 # direction only; magnitude discarded
step   = lr · θ_target · O_skew / ‖O_skew‖_*  # re-inject controlled magnitude
```

Crucially, **close the loop on the realized angle, not on `‖step‖`.** Fold the CNP factor-of-2 (and any normalization `‖O_skew‖_*`) into the constant so that the *measured* `‖G − I‖` (equivalently the vector-probing cosine drift `vᵀGv`) hits `θ_target`. Do not trust the `G ≈ I + 2Q` linearization to set the scale analytically — near the end of a merge cycle `Q` is at its largest and the higher-order `2Q²+2Q³+Q⁴` terms leak, so the analytic angle deviates from `2·‖step‖`. Measuring `‖G−I‖` directly sidesteps this.

**Free instrument:** POET's existing **vector-probing** hook already measures the realized per-block rotation angle (`vᵀGv` / `vᵀRv`). Use it to (a) verify the scaling produces a smooth controlled angle and (b) detect stalled or over-rotating blocks — the scaling design and the optimization-health diagnostic are the *same* measurement. Log `‖G − I‖` per block per step.

**Reset-boundary behavior:** right after each merge-then-reinitialize, `Q = 0` so `G = I` and the angle ramps from zero. A constant-angle target handles this gracefully (always asking for the same incremental angle); any scheme keyed to `‖Q‖` degenerates at the boundary when `‖Q‖ ≈ 0`. This is a strong reason to prefer the constant-angle formulation.

**Ablation — what to hold constant (run both, compare):**
- **Constant rotation angle per step** (geometric; the honest Muon analogue — democratize rotation across blocks regardless of weight energy). Recommended first cut: simpler, scale-free, reset-safe, and directly measurable via vector probing.
- **Constant relative weight change `‖ΔW‖/‖W‖`** (energy-matched; accounts for the multiplicative hop, so high/low-energy blocks rotate by different angles to produce equal relative effect on `W`). Requires per-block weight-norm bookkeeping. Use as the refinement if constant-angle shows blocks drifting unevenly.

`θ_target` is now the single tunable, and it has a physical meaning (per-step rotation angle), so it transfers across scales far better than a raw lr. Grid it on 130M against validation PPL; report the value and the realized-angle curve.

### Experiment matrix (Llama-130M, C4, identical everything else)
Run these and compare final + curve of validation perplexity:

| Run | Optimizer on Q | Tm (reset cadence) | Purpose |
|-----|----------------|--------------------|---------|
| A (baseline) | AdamW | 400 | reproduce vanilla POET number |
| B | Muon-on-Q | 400 | naive drop-in (expected: hobbled by reset) |
| C | Muon-on-Q | 1600 | loosen reset so momentum can accumulate |
| D | Muon-on-Q | no reset for the run | upper bound on Muon-on-Q benefit, ignoring CNP-norm drift |

Note for D: with no reset, CNP's Neumann series can drift out of its convergence regime (operator norm of `Q` exceeding 1). Monitor the orthogonality approximation error `‖RRᵀ − I‖_F / ‖I‖_F` per block. If it blows up, D is invalid past that point — report the step at which it diverges and treat C as the cleaner comparison.

### Gate criterion
- If **C or D** shows validation perplexity moving meaningfully toward (or past) the Muon baseline number, while A reproduces vanilla POET → **the mechanism is real.** Proceed to Stage 3.
- If even D shows no improvement over A → the conditioning seen in Stage 1 was not actionable via Muon. Report and stop.

### Correctness checks (must pass before trusting any perplexity number)
- **Skew-symmetry preserved:** assert `‖Q + Qᵀ‖_F < tol` after each update.
- **Orthogonality of resulting G:** track `‖GGᵀ − I‖_F`; should stay comparable to AdamW baseline, not worse.
- **Gradient correctness:** unit-test the `∂f/∂Q` reconstruction against `torch.autograd.gradcheck` on a small `b` (e.g. b=8) before running at scale.
- **Realized-angle calibration (not RMS parity):** log `‖G − I‖` (and the vector-probing cosine) per block per step. Confirm the scaling rule from the "Update scaling" subsection produces a smooth, controlled rotation angle matching the vanilla-POET Phase-II reference. If the realized angle is erratic or block-dependent when it shouldn't be, the lr/`θ_target` is not transferable and the comparison is unfair.

---

## Stage 3 — Momentum transport through the reset (the real fix)

If Stage 2 confirms the mechanism but shows the reset is the bottleneck (B ≪ C/D), the production fix is to **transport the Muon momentum buffer across the merge-then-reinitialize boundary** rather than dropping it or just lengthening `Tm`.

### Why transport, and what to transport
At a reset, `Q → 0` and the permutation `Ψ` is resampled to `Ψ'`. The momentum buffer `m` lives in the old block's coordinate frame. Carrying it forward requires relabeling it under the permutation change.

- **First moment only.** Transport the (orthogonalized) first-moment / momentum buffer. **Do NOT transport a second moment** — second-moment transport across a frame change is known to fail (it's not a covariant object under the relabeling and introduces bias). Muon has no second moment, which is convenient; if any Adam-style state exists on `Q`, reset it.
- **Permutation relabel:** if `m` is indexed by the old permutation `Ψ` and the new frame uses `Ψ'`, apply the index remap `Ψ' ∘ Ψ⁻¹` to `m` (use the existing CUDA permutation operator / index-mapping from POET-X §3.2 — it's the same machinery, no new kernel).
- Since `Q` itself resets to 0 (identity rotation) but the *weight* `W` absorbs the merged rotation, the momentum represents "the direction we were rotating in" — transporting it preserves optimization inertia across the otherwise-amnesiac boundary.

### Experiment
Repeat the C/D-winning config at `Tm=400` (the memory-friendly cadence) but **with momentum transport**. Compare against:
- A (AdamW baseline, Tm=400)
- C (Muon, Tm=1600, no transport)
- E (Muon, Tm=400, **with transport**) ← the proposed production setting

**Success = E matches or beats C while keeping the cheap `Tm=400` reset cadence.** That means you get Muon's benefit without paying the stability/memory cost of long reset intervals.

---

## Stage 4 — Scale up and the headline claim

Only after Stages 1–3 pass on 130M:
1. Confirm on **Llama-350M and 1.3B** (C4, Chinchilla-ish token budgets per the POET papers).
2. **The claim to test is not "we matched Muon."** It's: *POET-with-Muon-on-Q **beats** standalone Muon*, because the orthogonal-equivalence weight parameterization (well-conditioned weights) and the spectral tangent optimizer (well-conditioned updates) are **complementary, not competing** — they act on different objects. Run standalone Muon at the same scale/budget as the direct comparison.
3. Headline table: {AdamW, Muon, POET(AdamW-Q), POET(Muon-Q)+transport} × {130M, 350M, 1.3B}, validation PPL + peak memory. The selling point is the bottom-right cell winning on PPL while retaining POET's memory profile.

---

## Hard constraints / gotchas (read before coding)

- **Do not touch the weight spectrum *in the Muon-on-Q runs*.** The learnable-Σ work is a separate, parallel branch (Appendix A) with its own gate. Keep the Muon-on-Q study clean: it isolates the *optimizer-on-Q* axis only. The two branches recombine only in the deliberate 2×2 (Appendix A3).
- **Hybrid optimizer discipline:** Muon updates only 2D `Q` blocks. All other parameters stay AdamW. Getting this wrong silently corrupts the comparison.
- **Reproduce A first.** If you can't reproduce the vanilla POET 130M perplexity from the paper within noise, fix that before running any Muon variant. A miscalibrated baseline makes every downstream number meaningless.
- **Newton–Schulz on small `b`:** the standard Muon NS coefficients are tuned for the gradient-orthogonalization regime; verify they behave on skew-symmetric `b×b` inputs (paired imaginary eigenvalues → paired singular values). If NS is unstable here, fall back to an explicit polar factor via SVD for the gate experiment (slow but correct), then optimize the kernel later.
- **Fair lr / calibration:** do NOT RMS-match the raw step. Use the rotation-angle scaling rule (Stage 2 "Update scaling" subsection): the tunable is `θ_target`, calibrated so the *realized* per-block angle (`‖G−I‖`, measured via vector probing) matches healthy vanilla-POET Phase-II. Grid `θ_target` on 130M; report the value and the realized-angle curve.
- **Seeds:** ≥2 seeds per run at 130M; the gap to Muon is small, so single-seed differences may be noise.

---

## Appendix A — the spectrum branch: resolving the entropy↔performance paradox

This is a **parallel line of investigation to the Muon-on-Q work**, not merely a fallback. It is pursued if Stage 0 says representation-limited, OR if 0B says `∂f/∂Q` is well-conditioned, OR simply in parallel because the central question here is independent and arguably deeper. Treat it as its own potential paper.

### The paradox (this is the spine of the section)

POET's SVD-entropy (Figure 6 of the POET paper) is **higher than both AdamW and Muon** across essentially all projection types — POET has the *most* uniform/diverse weight spectrum of the three. The Kimi/Muon explanation (POET ref [46]) attributes Muon's superiority over AdamW precisely to *higher spectral diversity*. So by that theory, POET — which dominates on spectral diversity — should have the best performance. **Yet POET loses to Muon on perplexity.**

This is a contradiction in the accepted theory, not a tuning gap. Resolving it is the contribution. There are two candidate resolutions, and the section's job is to determine which is true:

- **(a) Entropy is a correlate, not the cause.** Muon's high update-entropy is a *side-effect* of well-conditioned dynamic steps that are independently doing something useful; the entropy itself isn't the mechanism. POET achieves high entropy the cheap, static way (frozen balanced spectrum by construction) and thereby copied the side-effect while missing the mechanism. If demonstrated cleanly, this **bounds/falsifies the Kimi explanation** — diversity is necessary-ish but not sufficient.
- **(b) Right entropy, wrong axis.** SVD entropy is blind to *orientation*: it sees the distribution of singular *values* but not *which directions* carry them. POET's high entropy sits on a **frozen, randomly-oriented** singular basis (the spectrum is frozen at init; R,P rotate vectors but the *values* stay bound to their original random directions). Muon's comparable entropy sits on a **task-aligned** basis. Identical entropy, completely different objects. Orientation is what the loss cares about.

**(b) is the more likely truth and the more actionable one.** It implies POET's structural limitation: POET can learn directions (R, P) OR keep magnitudes (frozen Σ₀), but cannot *bind a chosen magnitude to a chosen direction*. Free joint optimization does exactly this binding — a large singular value attached to a specific learned direction — and POET structurally cannot, because the magnitude is frozen and the direction is learned separately.

### Why this does NOT contradict POET's generalization story (important)

Naively "letting Σ move" sounds like chasing AdamW's heavy tail — the exact pathology POET's intro argues against. The resolution is that there are **three distinct spectra**, and the goal is none of the obvious ones:
- **AdamW:** heavy-tailed, large σ_max on *uncontrolled* directions → bad generalization (POET's stated enemy).
- **POET:** bounded/flat, but on *random* directions → bounded-but-misaligned.
- **Target (what Muon may approximate):** bounded/balanced AND *aligned to task directions* → **bounded-and-aligned**.

POET has bounded-but-misaligned; AdamW has aligned-but-unbounded. The sweet spot is *bounded-and-aligned*, which **neither** can express. The whole point of the spectrum branch is a parameterization that can express "bounded AND aligned" — large singular values permitted, but only on learned directions, and only within a controlled band. That keeps POET's conditioning/generalization guarantee (bounded spectral norm → the Appendix C / Eq. 7 margin bound survives) while removing the misalignment.

### Conceptual lineage (the narrative for the paper)

energy-preserving (orthogonal training) ⊂ spectrum-*preserving* (POET) ⊂ spectrum-*constrained* (this work). Each step relaxes one constraint while keeping the conditioning guarantee. POET is the degenerate special case where the allowed spectral band collapses to the init point. This is a generalization of POET, not a contradiction of it — exactly how POET generalized orthogonal training.

**What survives the relaxation:** bounded spectral norm (→ margin/generalization bound, Eq. 7) survives, because the band is bounded. **What dies:** exact spectrum-preservation (Theorem 1's *equality*) and the exact hyperspherical-energy-invariance argument (needed exact Gaussian-distribution invariance). Energy was already relaxed once by POET ("in expectation"); relaxing to "bounded energy" is in the same spirit. State this honestly in any writeup.

---

### Probe A0 — GATE: the alignment measurement (distinguishes (a) from (b); do before building anything)

Entropy collapses too much information. The decisive measurement is **where the singular-value mass sits relative to the learned/data directions**, for all three methods.

1. Train small models (Llama-60M/130M, C4) with **AdamW, Muon, and POET** — same budget. (Muon's trained spectrum is the panel *missing* from POET Figure 1; produce it and overlay all three SVD-over-training curves as a first sub-result.)
2. For each method over training, compute **not just SVD entropy** but the **alignment between top singular directions and actual data/gradient energy**: take the top-k left singular vectors of each W, measure the fraction of activation/gradient energy they capture (e.g. project a batch of activations onto the top-k singular subspace; report captured energy ratio). Do this per projection type.
3. Overlay: entropy(method) vs. alignment(method), over training.

**Gate criterion:**
- **(b) confirmed — the productive case:** POET shows *high entropy but low direction-data alignment*; Muon shows *comparable entropy but high alignment*. → The gap is magnitude-direction misalignment. **Proceed to A1** (co-adaptive band-constrained Σ). This overlay is the paper's central figure.
- **(a) confirmed:** entropy tracks performance poorly for *all* methods / Muon's edge persists with no alignment difference. → Entropy isn't the operative axis; the gap is dynamic/optimizer-side. Redirect to the Muon-on-Q line; the spectrum branch is a (publishable) negative result bounding the Kimi explanation.
- **Muon's spectrum looks like AdamW's tail:** then heavy tail isn't actually harmful and POET's generalization framing needs revisiting — a different, harder paper. Flag for human decision.

**Deliverable:** `appendixA0_alignment.md` — the three-way entropy-vs-alignment overlay, the Muon spectrum panel for Fig 1, and verdict (a)/(b)/revisit.

---

### A1 — Co-adaptive, band-constrained Σ (only if A0 confirms (b))

The fix is **not** free Σ (breaks guarantees, chases AdamW's tail). It is: let Σ move *so magnitude can concentrate on the directions R, P are emphasizing*, while constraining the *set* of magnitudes to stay balanced/bounded.

Reparameterize each weight: `W = R · Σ · P` (Σ diagonal, replacing the frozen Σ₀). Enforce the band **by construction**, not by penalty — the spiritual match to how CNP enforces orthogonality by construction rather than penalizing it:

- **Main method (by-construction band):** `σᵢ = (1−√λ) + (2√λ)·sigmoid(sᵢ)`, with `sᵢ` the unconstrained trainable. σ is *always* in `[1−√λ, 1+√λ]` (the band POET's own Appendix C proves well-conditioned) at every step exactly — no regularization weight to tune, boundedness guarantee holds always. Cost: `min(m,n)` extra params per matrix — negligible vs. the Q blocks.
- **Initialize `sᵢ` so Σ starts at Σ₀** (i.e. the POET init spectrum), so the relaxed method is statistically equivalent to vanilla POET at step 0 and can only depart as the loss pulls it.

### A2 — The δ-leash ablation (the figure that quantifies the cost of freezing)

To *measure* how much freezing was costing, parameterize a continuous knob between full-POET and freer-Σ:

`σᵢ = σᵢ⁰ · exp(εᵢ)`, with `‖ε‖_∞ ≤ δ` (enforced by `εᵢ = δ·tanh(eᵢ)`).

- δ → 0 recovers vanilla POET *exactly* (continuous interpolation).
- Sweep δ ∈ {0, small, …} and plot validation PPL vs δ. The shape of that curve is direct evidence of how over-constrained the freezing was. A monotone improvement up to some δ* then degradation (toward AdamW's tail regime) would be the ideal story: "freezing was over-constrained; here is the optimal leash length."

### A3 — Optimizer choice for Σ and interaction with the Q-optimizer

- Σ is a diagonal/1D parameter set → it stays on **AdamW** (do not Muon-ify 1D params; same hybrid discipline as everywhere in this spec).
- If the Muon-on-Q line is *also* live, the clean experiment is the 2×2: {frozen Σ₀, band-constrained Σ} × {AdamW-on-Q, Muon-on-Q}. This directly tests whether magnitude-direction co-adaptation (Σ) and well-conditioned vector steps (Muon-on-Q) are **complementary** — the strongest possible result, since it would show POET's two structural limitations (frozen magnitude, ill-conditioned direction steps) are independent and both fixable.

### Correctness / honesty checks
- Confirm Σ stays in the band at every step (assert, don't trust).
- Confirm step-0 equivalence to vanilla POET (Σ = Σ₀, same init): PPL curves should overlay vanilla POET initially.
- Track the **alignment metric from A0** *during* A1 training: the claim is that band-constrained Σ *increases* direction-data alignment relative to vanilla POET at equal entropy. If alignment does **not** improve, the mechanism story is wrong even if PPL improves — investigate before claiming the mechanism.
- Re-verify the surviving generalization bound empirically: spectral norm stays bounded (it must, by construction), so report `σ_max` trajectories to confirm no drift toward AdamW's tail.

### Suggested order for the spectrum branch
A0 (alignment gate) → **stop for human review** → A1 (by-construction band Σ) + A2 (δ-sweep) on 130M → A3 (2×2 with the Q-optimizer) → scale up. A0 is cheap and decisive; do not build A1 until A0 confirms (b).

---

## Suggested first action for the agent

Start with **Stage 0, Probe 0A (single-batch overfit)** — it is the cheapest probe and it decides the entire axis (optimizer vs. parameterization) before any optimizer code exists. Produce `stage0a_overfit.md` with the three-arm loss-floor plot and a verdict. **Only if 0A returns OPTIMIZATION-LIMITED**, proceed to Probe 0B / Stage 1: wire the `∂f/∂Q` conditioning hook into a vanilla POET-130M run, produce the conditioning report with singular-value plots and the PROCEED/STOP verdict.

**Stop after Stage 0 for human review** before implementing any optimizer changes. The Stage 0 decision table must resolve to a row first; everything downstream depends on which row.


## PyTorch Implementation: The Vectorized Optimizer

To save memory, POET stores only the upper-triangular elements of $Q$ as a flattened vector $v$. However, the Newton-Schulz algorithm requires dense matrix multiplications.To solve this, the optimizer performs a fast memory-view cycle:Inflate: Vector $\to$ Dense Skew-Symmetric Matrix.Orthogonalize: Run Newton-Schulz on the dense matrix.Deflate: Dense Matrix $\to$ Vector.

import torch

@torch.no_grad()
def muon_poet_vector_step(v_param: torch.Tensor, grad_v: torch.Tensor, lr: float, b: int, ns_steps: int = 5):
    """
    Applies the Muon optimization principle to a vectorized skew-symmetric generator Q.

    Args:
        v_param: The flattened upper-triangular parameters. Shape: (k, b*(b-1)//2)
        grad_v: The gradients corresponding to v_param. Shape: (k, b*(b-1)//2)
        lr: The learning rate (target aggregate rotation angle).
        b: The block dimension size.
        ns_steps: Number of Newton-Schulz iterations (typically 5 or 6).
    """
    k = v_param.shape[0]
    device = v_param.device
    dtype = v_param.dtype

    # Pre-compute indices for the upper triangle (offset=1 excludes the diagonal)
    i, j = torch.triu_indices(b, b, offset=1, device=device)

    # ==========================================
    # 1. INFLATE: Vector -> Matrix
    # ==========================================
    # Create empty block matrices: shape (k, b, b)
    G = torch.zeros((k, b, b), device=device, dtype=dtype)

    # Fill the upper triangle with the gradient vector
    G[:, i, j] = grad_v

    # Enforce strict skew-symmetry: G_skew = G - G^T
    G_skew = G - G.mT

    # ==========================================
    # 2. DEMOCRATIZE: Newton-Schulz Iteration
    # ==========================================
    # Pre-scale to get into the convergence radius of Newton-Schulz.
    # Using dim=(1,2) to compute norms independently per-block in the batch.
    G_norm = torch.linalg.matrix_norm(G_skew, ord='fro', dim=(1, 2), keepdim=True)

    # Add epsilon to prevent division by zero for inactive blocks
    X = G_skew / (G_norm + 1e-8)

    # Run the Newton-Schulz orthogonalization loop
    for _ in range(ns_steps):
        A = X @ X.mT
        X = 1.5 * X - 0.5 * (A @ X)

    # X is now our orthogonalized gradient matrix (\hat{G}_skew).
    # All canonical angles (singular values) are approximately 1.

    # ==========================================
    # 3. SCALE & DEFLATE: Matrix -> Vector
    # ==========================================
    # Scale to the target aggregate angle (learning rate)
    X_norm = torch.linalg.matrix_norm(X, ord='fro', dim=(1, 2), keepdim=True)
    update_matrix = lr * (X / (X_norm + 1e-8))

    # Extract the upper triangle back into a flat vector
    update_v = update_matrix[:, i, j]

    # ==========================================
    # 4. APPLY THE STEP
    # ==========================================
    v_param.sub_(update_v)
