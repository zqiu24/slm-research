# Alternating-on-POETX (integrated, both-momenta) — Design Spec

**Date:** 2026-06-08
**Status:** approved (brainstorm), pending implementation plan

## Goal

Bring the new POET champion — **alternate the rotation *write* every step while keeping *both* Lie momenta fresh** (`lie_ortho` + `lie_alternating`, val/loss **3.5332**, a virtual tie with the overall-best `muon_kimi`) — onto the **POETX forward-frame layer** (`POETXLinear`, the canonical single-step-merge layer), and add the one POETX-native speedup that is compatible with both-momenta: an **active-only merge fold** (skip the frozen side's Cayley, which is identity in alternating mode).

## Why (context)

- The champion run (`1ynrrimu`) used `single_step_native` + `lie_alternating`. The winning ingredient is purely an **optimizer** behavior: write one side per step, but advance **both** momenta every step (so each written side gets a smoother, 2-step-accumulated, Gauss–Seidel-decoupled direction). See POET_dev.md §2.5 arm E and the §2.1 Alternating row.
- A prior attempt at a *true single-side* POETX layer (`single_step_x_alternating`, run `au92x0pj`) **regressed to 4.22** because it froze the inactive side's momentum to skip the frozen gradient (the d³ backward saving). A CPU isolation (3 optimizer modes on a faithful mini-POET) confirmed the momentum-freeze — not the layer math — is the killer: with **fresh both-side momentum** the optimizer modes are within ~1.6%.
- **Fresh both-side momentum is load-bearing.** Therefore the backward must stay both-sides (no d³ backward saving). The only safe POETX speedup is at the **merge**: in alternating mode the inactive side's `oft_R` is exactly 0 → its rotation is identity → its Cayley build can be skipped.

## Key facts already verified

- `POETXSingleStepFunction.backward` returns **both** `grad_oft_R_in` and `grad_oft_R_out` → both momenta stay fed. (`third_party/poet_torch/poetx_ops.py`)
- The optimizer's `alternating` mode is layer-agnostic — it gates the *write* by the param group's `side`. (`src/optim/poet_lie_orth.py`)
- `megatron_args` does **not** block `single_step_x` + `lie_alternating`; only `single_step_x_alternating` is mutually exclusive with `lie_alternating`.
- `_fold_active_side` (active-only fold, frozen side = identity) is already verified **bit-identical** to the both-sides fold for the `"in"` side at fp64 (Task 7). The `"out"` side is currently **untested**.

## Design

### Phase 1 — Parity (config only, zero new code)

A champion-recipe config with `single_step_x=true` + `lie_alternating=true` (+ `lie_alternate_every=1`), head-off, lr 3e-3, c=8, distributed, `single_step_x_alternating=false`.

- The walk already builds `POETXLinear` for `single_step_x`; the optimizer's `alternating` mode (from `lie_alternating`) writes one side; the standard `POETXLinear.merge_then_reinitialize` folds **both** sides (the frozen side's `oft_R=0` → `R=I` → no-op).
- **Acceptance:** a GPU run reproduces ≈**3.5332** at the POETX forward speed. This de-risks the whole approach before any layer surgery.

### Phase 2 — POETX-native active-only merge (integrated into `POETXLinear`)

1. **`POETXLinear` gains `alternating: bool = False`, `alternate_every: int = 1`.** `_fold_active_side(active, …)` moves up from the (research) `AlternatingPOETXLinear` subclass into `POETXLinear`. When `alternating` is set, the layer's merge folds **only the active side**.
2. **Single active-side source of truth.** Both the optimizer's *write* and the merge's *fold* must target the same side, or the written rotation is dropped. Both read `poet_torch.alt_state` (seeded once per training step by the `poet_merge_step` train_step wrapper — Tasks 1 & 6). The optimizer's `alternating` mode switches its active-side source **from the internal `_alt_step` counter to `alt_state`** (`LieOrthMomentum._active_side`). This is **quality-neutral** for the both-sides-merge case (the champion folds both sides, so the write phase never mattered) and **required** for active-only correctness.
3. **Walk dispatch.** `replace_linears_with_poet` builds `POETXLinear(alternating=True, alternate_every=…)` when `single_step_x` **and** `lie_alternating` are set (and `single_step_x_alternating` is off). `poet_apply_to_model` threads `lie_alternating` through.
4. **Merge-driver dispatch.** `_merge_layers` routes a layer to the active-only fold by the **`alternating` flag** (`getattr(pl, "alternating", False)`), not by isinstance — folding `active_side(pl.alternate_every)`.

### Active-side data flow (per step)

```
poet_merge_step._wrapped:  alt_state.set_iteration(iteration)   # BEFORE forward
  -> POETXLinear.forward   : both-sides bare GEMM (no active-side dependence)
  -> backward              : both grads (both momenta fed)
  -> LieOrthMomentum.step(): writes active_side(alt_state) only
  -> merge (active-only)   : folds active_side(alt_state) only   # SAME iteration -> SAME side
```

### Research path (kept, gated off)

The true-single-side machinery stays in, documented as a regression needing momentum-refresh research:
`AlternatingPOETXSingleStepFunction` (single-side backward), `AlternatingPOETXLinear` subclass, `LieOrthMomentum(true_single_side=True)`, the `--poet-single-step-x-alternating` flag/config. It remains mutually exclusive with `lie_alternating`. A future experiment may try to recover the d³ backward saving without the quality hit (lagged / periodic-refresh inactive momentum).

## Components & files

| File | Change |
|---|---|
| `third_party/poet_torch/poetx_layer.py` | `POETXLinear`: add `alternating`/`alternate_every`; host `_fold_active_side`; alternating-aware merge |
| `src/optim/poet_lie_orth.py` | `_active_side`: `alternating` mode reads `alt_state` (not `_alt_step`) |
| `src/optim/poet_layers.py` | walk builds `POETXLinear(alternating=True)` for `single_step_x` + `lie_alternating` |
| `src/patches/poet_apply_to_model.py` | thread `lie_alternating` into the walk |
| `src/patches/poet_merge_step.py` | `_merge_layers`: dispatch active-only fold on the `alternating` flag |
| `configs/experiments/optim/poet_lie_orth_alt.yaml` (new) | champion + `single_step_x` + `lie_alternating` |
| `docs/experiments/poet_lie_orth_alt.md` (new), `scripts/train_poet_lie_orth_alt.sh` (new) | experiment doc + launcher |
| tests | both-side fold parity; optimizer alt reads alt_state; write==fold consistency; walk-selection |

## Testing & correctness gates

- **Fold parity (CPU):** `_fold_active_side` == both-sides fold for **both** `"in"` and `"out"` (closes the Task 7 `"out"` gap), fp64.
- **Optimizer active-side (CPU):** `alternating` mode writes the side given by `alt_state` parity (update the existing alternating test to seed `alt_state`).
- **Write==fold consistency (CPU):** drive an integrated `POETXLinear(alternating=True)` for several steps; assert the side the optimizer wrote each step equals the side the merge folded.
- **Walk-selection (CPU):** `single_step_x` + `lie_alternating` → `POETXLinear(alternating=True)`; `single_step_x_alternating` → `AlternatingPOETXLinear` (research path unchanged).
- **Full CPU suite green** (keep the dtype-isolation fixtures).
- **GPU (user):** Phase 1 reproduces 3.5332; Phase 2 active-only merge keeps 3.5332 (bit-identical fold) and shows a merge-time `perf/step_time_s` drop.

## Out of scope

- The d³ **backward** saving (skip the frozen gradient) — incompatible with fresh both-side momentum; lives in the gated research path.
- A genuine one-sided fold matmul that also skips the frozen identity-fold (Task 7 v2) — optional follow-up behind its own parity test.

## Open questions

None — goal, integrate-vs-dedicated (integrate), and keep-research-flag are settled.
