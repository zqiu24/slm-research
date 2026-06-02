# POET Muon-on-Q (Stage 2) — Design

**Date:** 2026-06-02
**Status:** Approved design, ready for implementation plan
**Parent spec:** [`docs/poet_muon_q_spec.md`](../../poet_muon_q_spec.md) (Stage 2 §119–185; reference `muon_poet_vector_step` §319–387)
**Gate cleared:** Probe 0A → optimization-limited; Probe 0B → `∂f/∂Q` decisively heavy-tailed at every step (attention `q.R_out` stable rank ~5–12 / 512, condition 10⁴–10⁶; MLP milder ~110/512). Human-review PROCEED granted.

## Goal

Step POET's skew generators `oft_R` with a **Muon-style spectral update** — orthogonalize the per-block skew gradient (Newton–Schulz) and rescale to a **constant rotation angle** — instead of AdamW, to fix the heavy-tailed `∂f/∂Q` conditioning Probe 0B confirmed. Everything non-`oft_R` stays AdamW (hybrid discipline). Selected by a flag on the POET path; default is byte-identical to today.

## Scope

**In:** Stage 2 only — the Muon-on-Q optimizer + config + correctness/diagnostics, runnable as a 2×2 comparison. **Out:** Stage 3 momentum-transport-across-reset, Stage 4 scale-up, the `‖ΔW‖/‖W‖`-matched scaling refinement, and distributed-optimizer (DP-sharded) state for the new optimizer.

**No merge-reset.** Per the approved simplification, Stage-2 runs disable the periodic merge (`merge_period=0`). There is therefore no reset event, the SkewMuon momentum **accumulates over the whole run** (which is what Muon needs — the reset is exactly what destroys Muon's momentum), the permutation is fixed at init, and **no reset/momentum-zeroing logic is built**. The POET merge code is untouched; it simply never fires.

## Experiment (the user's GPU runs; the build delivers the knobs)

A 2×2, all `merge_period=0`, isolating the two axes independently:

| | cayley | exp |
|---|---|---|
| **AdamW-on-Q** (`q_optimizer=adam`) | A-cay | A-exp |
| **SkewMuon-on-Q** (`q_optimizer=muon`) | M-cay | M-exp |

- Down each column: optimizer effect (AdamW vs Muon). Across each row: parameterization effect (cayley vs exp).
- These are pure config combinations (`q_optimizer` × `optim.poet.parameterization` × `merge_period=0`). The plan delivers the optimizer + flags; the runs are the user's.
- **Cayley drifts in the no-reset regime** (`‖Q‖₂` can exceed the Cayley–Neumann convergence radius → `‖RRᵀ−I‖` blows up; the parent spec flags this for the no-reset "D" run). **`exp` is exactly orthogonal for any `Q`** (built for this case). The `‖RRᵀ−I‖` monitor (Unit 4) validates the cayley arms; an arm is invalid past the step where it diverges.

## Architecture

### Unit 1 — `SkewMuon` optimizer (the one genuinely new piece)

A `torch.optim.Optimizer` that applies the Muon principle on the **block skew matrices**, parameterization-agnostic (it only touches `oft_R` and its grad). New file `src/optim/poet_skew_muon.py`.

Per `oft_R` param (shape `(n_blocks, n_elems)`), per step:
1. Derive block size `b` from `n_elems` (`n_elems = b(b−1)/2` → `b = (1 + √(1+8·n_elems))/2`).
2. Momentum: `m ← μ·m + g` (μ≈`muon_momentum`, default 0.95; Nesterov optional).
3. **Inflate** each block's vector → `b×b` skew matrix (reuse `src/diag/skew_conditioning.vec_to_skew`; same `triu_indices(b,b,1)` layout POET stores).
4. **Newton–Schulz orthogonalize** the skew batch (reuse `src/optim/_kimi_muon.zeropower_via_newtonschulz5`, `ns_steps` default 5), then re-skew-symmetrize `X̂ ← (X̂ − X̂ᵀ)/2`.
5. **Constant-angle scale** (per block): `step_skew = θ_target · X̂ / ‖X̂‖_F`. (`θ_target` = `muon_theta`, the single tunable; under `exp` this is the exact aggregate rotation angle, under cayley ≈ `2·arctan`-scaled — still a valid knob.)
6. **Deflate** `step_skew` → upper-triangular vector; `oft_R ← oft_R − step_vec`.

Standard Muon on the raw `(n_blocks, n_elems)` tensor would orthogonalize the wrong axes (mixing blocks and skew entries); the inflate→NS→deflate is exactly why this is custom.

### Unit 2 — hybrid wiring in the POET optimizer path

In `src/optim/poet.py` (`get_megatron_poet_optimizer`), when `q_optimizer=muon`: split trainable params into `oft_R*` → `SkewMuon` and everything else (embeddings, norms, lm_head) → AdamW, mirroring the split in [`src/optim/muon_kimi.py`](../../../src/optim/muon_kimi.py#L50-L61). `oft_R` keeps its **pre-DDP grad-buffer placement** (so `main_grad` is populated and DP-reduced before `step()`, exactly as Probe 0B relied on); SkewMuon steps the DP-reduced grad, so DP replicas stay in sync without a distributed (sharded) optimizer. When `q_optimizer=adam` (default), the path is unchanged. No `poet_merge_step` interaction (no reset).

### Unit 3 — config plumbing

`optim.poet.q_optimizer: "adam" | "muon"` (default `"adam"`), plus `optim.poet.muon_theta`, `optim.poet.muon_ns_steps` (5), `optim.poet.muon_momentum` (0.95) — threaded through the same chain `parameterization` already uses: [`configs/experiments/optim/poet.yaml`](../../../configs/experiments/optim/poet.yaml) → [`src/utils/megatron_args.py`](../../../src/utils/megatron_args.py) (`--poet-q-optimizer` etc.) → [`launchers/pretrain_gpt_slm.py`](../../../launchers/pretrain_gpt_slm.py) → the builder.

### Unit 4 — correctness + diagnostics

- **CPU unit tests** (the SkewMuon math, no GPU): skew-symmetry preserved after a step (`‖Q+Qᵀ‖_F < tol`); NS **flattens the spectrum** (stable rank ↑ toward `b`, condition ↓ — verified with the existing `block_spectral_stats`); constant-angle scaling makes the realized per-block step norm hit `θ_target`; `b`-from-`n_elems` inversion; param-group routing (`oft_R`→SkewMuon, rest→Adam).
- **Realized-angle logging** `‖G−I‖_F` per probed block (the parent spec's calibration check, §184) — confirms the rotation angle is smooth/controlled and `θ_target`-transferable.
- **Orthogonality-drift monitor** `‖RRᵀ−I‖_F` per block — validity gate for the cayley no-reset arms; reuses the parameterization's R-builder (`get_weight_poet_decoupled` / `_exp`). Both can ride the existing env-gated conditioning-probe infrastructure ([`src/patches/poet_grad_conditioning.py`](../../../src/patches/poet_grad_conditioning.py)) or a sibling probe.

## Reuse (don't rebuild)

- `vec_to_skew`, `block_spectral_stats` — [`src/diag/skew_conditioning.py`](../../../src/diag/skew_conditioning.py).
- `zeropower_via_newtonschulz5` — [`src/optim/_kimi_muon.py:18`](../../../src/optim/_kimi_muon.py#L18).
- Hybrid param-split pattern — [`src/optim/muon_kimi.py`](../../../src/optim/muon_kimi.py).
- POET layer `oft_R` storage / skew layout — [`third_party/poet_torch/poet_layer.py`](../../../third_party/poet_torch/poet_layer.py) (`pytorch_skew_symmetric`, `triu_indices`).

## Risks

- **Megatron mixed-precision integration of a custom step.** Threading SkewMuon through the fp32-master / grad-buffer / clipping stack. Mitigation: follow the proven `muon_kimi` (non-sharded) pattern; the fp32 master `oft_R` is what SkewMuon steps; DP replicas stay in sync because every rank steps the same DP-reduced grad.
- **NS on small `b` / skew inputs.** Standard Muon NS coefficients are tuned for the gradient-orthogonalization regime; verify they behave on skew (paired singular values). Parent-spec gotcha §223: if unstable, fall back to an explicit polar factor (SVD) for the gate, optimize later. CPU test covers this.
- **`θ_target` calibration.** It is the single tunable; grid it on the 60m dev runs and read the realized-angle curve. Not a correctness risk; a tuning one.
