# POET × Muon-on-Q — Stage 0 execution plan (gated)

**Status:** discussion output, not yet executed. **No optimizer code is written in this
plan.** It covers only the Stage 0 diagnostic battery from
[`docs/poet_muon_q_spec.md`](../../poet_muon_q_spec.md), adapted to this repo, and ends at a
**hard stop for human review**. Downstream stages (Muon-on-`oft_R`, momentum transport,
Appendix A learnable-Σ) are deferred until the Stage 0 decision table resolves to a row.

## Goal

Decide *cheaply* whether our confirmed POET-vs-baseline gap at 60m is worth attacking with
Muon-style orthogonalization of `∂f/∂Q`, **before** building any Newton–Schulz / hybrid-optimizer
code. Stage 0 can kill or redirect the whole line in an afternoon.

## Repo ↔ spec mapping (this repo *is* the spec's POET-X)

| Spec term | This repo |
|---|---|
| trainable skew `Q` (upper-tri) | `oft_R` — Cayley-parameterized skew vector, `triu_indices(b,b,1)` ([src/optim/poet_layers.py](../../../src/optim/poet_layers.py)) |
| "AdamW on `Q`" | **stock Megatron Adam** on `oft_R` (default `optim.poet.use_poet_adam=false`) |
| CNP `Q→G` (k=3) | `torch.ops.poet.cayley` (Triton) / `pytorch_skew_symmetric` |
| `W = R W0 P` | `W_eff = R_out @ W_0 @ R_in` with block permutations |
| reset cadence `Tm=400` | `optim.poet.merge_period=400` — merge-then-reinit, **resets Adam momentum**, re-randomizes permutations ([src/patches/poet_merge_step.py](../../../src/patches/poet_merge_step.py)) |
| `γ=0.5` LR multiplier | `optim.poet.scale=0.5` |
| `∂f/∂Q` gradcheck | already exists: [tools/poet_gradcheck.py](../../../tools/poet_gradcheck.py) |
| Muon optimizer | already exists: [src/optim/muon.py](../../../src/optim/muon.py), `experiment=optim/muon_hybrid` |

Block size here is **b = 512** (`optim.poet.block_size=512`, the 60m config is built so every
weight dim divides 512), not the spec's 256. Account for it; otherwise identical.

## Common run settings (apply to every arm, for apples-to-apples)

60m llama3, `seq_length=256`, `training_regime=ablation_40x`, `global_batch_size=1024`,
`micro_batch_size=128`, untied embeddings, `weight_decay=0`, dropout 0 (family default),
`wandb.project=slm-zeju-dev`. Same seed across arms; **≥2 seeds** for the final go/no-go since
the gap is small. Training launches are wrapped with `codexlog NAME …`.

The dev scripts already inject most of this. The **Muon baseline is the exception** — its script
defaults to 300m / gbs=512, so override explicitly.

---

## Probe −1 — `merge_period` sweep (FREE: zero new code)

Because POET already wipes Adam momentum every `merge_period` steps, sweeping it on the *existing*
AdamW-on-`oft_R` setup is simultaneously the spec's "reset-disabled" arm and its Stage-2 "loosen
the reset" idea — with no optimizer code. Run **first**; it localizes the gap.

```bash
# baselines (align Muon to the dev harness settings)
codexlog s0_adam   bash scripts/train_adam_dev.sh
codexlog s0_muon   bash scripts/train_muon.sh base/scale=60m base.model.seq_length=256 \
                        training_regime=ablation_40x training.global_batch_size=1024 \
                        training.micro_batch_size=128 base.model.tie_embeddings=false \
                        wandb.project=slm-zeju-dev
# POET arms
codexlog s0_poet400  bash scripts/train_poet_dev.sh                                  # Tm=400 (baseline)
codexlog s0_poet1600 bash scripts/train_poet_dev.sh optim.poet.merge_period=1600     # loosened
codexlog s0_poet0    bash scripts/train_poet_dev.sh optim.poet.merge_period=0        # disabled
```

**Monitor** `‖RRᵀ − I‖_F` per block on the `merge_period=0` arm: with no merges `oft_R` grows and
CNP's Neumann series can leave its convergence regime. If it diverges, that arm is invalid past the
divergence step — report the step and treat `1600` as the clean "loosened" comparison.

**Gate (reading the sweep):**
- Gap **mostly closes** when the reset is lengthened/disabled → **RESET-LIMITED.** The fix is
  Stage 3 (momentum transport across the reset) or simply tuning `merge_period`; Muon-on-Q is
  secondary. Likely the cheapest real win.
- Gap **barely moves** → not the reset. Proceed; the gap is in AdamW-on-`Q` conditioning (→ 0B) or
  representation (→ 0A says which).

---

## Probe 0A — single-batch overfit (capacity: representation- vs optimization-limited)

**One piece of new code** (the only build in this plan): an "overfit a single fixed minibatch"
mode — cache the first micro-batch and feed it every step, constant LR, large fixed step budget.
Regularization is already off (`weight_decay=0`, dropout 0). Realized as an env-gated patch
(`SLM_OVERFIT_SINGLE_BATCH=1`) plus `scheduler=constant`; see the implementation plan for wiring.

Three arms, identical batch/seed/budget, 60m, DP=1:
```bash
SLM_OVERFIT_SINGLE_BATCH=1 codexlog s0a_adam    bash scripts/train_adam_dev.sh scheduler=constant
SLM_OVERFIT_SINGLE_BATCH=1 codexlog s0a_poet400 bash scripts/train_poet_dev.sh scheduler=constant
SLM_OVERFIT_SINGLE_BATCH=1 codexlog s0a_poet0   bash scripts/train_poet_dev.sh scheduler=constant optim.poet.merge_period=0
```
Plot train-loss-vs-step floors on shared axes.

**Gate (decision table fork):**
| Observation | Verdict | Next |
|---|---|---|
| POET (either reset setting) reaches ≈ AdamW-direct floor | **OPTIMIZATION-LIMITED** | → Probe 0B |
| POET can't reach AdamW floor *even with reset off* | **REPRESENTATION-LIMITED** | → Appendix A (learnable-Σ); no optimizer fixes this |
| POET-Tm400 floored but POET-Tm0 matches AdamW | **RESET-LIMITED** | → Stage 3 (momentum transport) |

---

## Probe 0B = Stage 1 — conditioning of `∂f/∂Q` (only if 0A = OPTIMIZATION-LIMITED)

**New code:** a non-invasive hook on a vanilla POET run (Tm=400). Every ~2k steps, for ~8 blocks
spanning early/mid/late layers × {q_proj, v_proj, mlp.down, mlp.up}, both `R_in` and `R_out`:
capture `oft_R.grad`, reconstruct the skew `b×b` (reuse the reconstruction in
[tools/poet_gradcheck.py](../../../tools/poet_gradcheck.py)), `torch.linalg.svdvals`. Skew matrices
have **paired** singular values — account for it. Log to W&B: per-block singular-value histograms
over training + scalars: condition number `σ_max/σ_min`, stable rank `‖·‖_F²/‖·‖_2²`,
`σ_max/σ_median`.

```bash
SLM_POET_GRAD_CONDITIONING=1 codexlog s0b_cond  bash scripts/train_poet_dev.sh
```

**Gate:**
- Heavy-tailed / condition number growing / stable rank ≪ b → **PROCEED** (Muon-on-Q motivated).
  These plots are Figure 1.
- Flat / stable rank ≈ b / condition number O(1) → **STOP** the Muon-on-Q line; pivot to Appendix A.

---

## Decision table → next action (HARD STOP for human review here)

| 0A | 0B | Action |
|---|---|---|
| Optimization-limited | ill-conditioned | Stage 2 (Muon-on-`oft_R`) — main line |
| Optimization-limited | well-conditioned | Appendix A (learnable-Σ) |
| Reset-limited | (run anyway) | Stage 3 (momentum transport / `merge_period` tuning) |
| Representation-limited | — | Stop Muon line; Appendix A / mixing structure |

**Do not write any Muon / Newton–Schulz / hybrid-optimizer code until this resolves to a row.**

## Deliverables

- `stage0a_overfit.md` — `merge_period` sweep curves + three-arm overfit floors + verdict
  (OPTIMIZATION / REPRESENTATION / RESET-limited).
- `stage1_conditioning.md` (if reached) — singular-value plots + PROCEED/STOP with the numbers.

## Deferred (not in this plan)

Vector-probing realized-angle hook (`‖G−I‖`), Muon-on-`oft_R` hybrid optimizer, rotation-angle
scaling rule, momentum transport across the reset, Stage 4 scale-up, and the entire Appendix A
learnable-Σ branch. Each is gated behind the decision table above.

## Caveats

- **b=512, not 256** — larger blocks; the conditioning story may differ from the paper's numbers.
- **60m representativeness** — chosen for cost; a 60m verdict may not transfer to the scales we
  ultimately care about. Re-confirm the decisive verdict at 300m before committing to Stage 2.
- **Muon baseline alignment** — its script defaults to 300m/gbs=512; the override above is
  load-bearing for a fair comparison.
- **`merge_period=0` drift** — monitor `‖RRᵀ−I‖_F`; treat `1600` as the clean loosened point.
