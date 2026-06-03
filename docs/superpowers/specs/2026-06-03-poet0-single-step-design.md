# poet0 — Single-Step POET (fold every step, periodic permute + momentum reset)

**Date:** 2026-06-03
**Status:** Design approved, ready for implementation plan
**Related:** [docs/poetx_pion_pipeline.md](/lustre/fast/fast/zqiu/slm-research/docs/poetx_pion_pipeline.md) §1

---

## 1. Goal

Add a new POET training variant, **poet0**, that runs the existing POET
optimizer in the *single-step* regime of
[poetx_pion_pipeline.md](/lustre/fast/fast/zqiu/slm-research/docs/poetx_pion_pipeline.md)
§1: the block rotation `Q` is **folded into the live weight `W` every step**
(born at identity, one small step, exponentiated, merged, reset), instead of
accumulating against a frozen `W` over a 400-step interval.

The central refinement of poet0 over a naïve "merge interval = 1" is that the
two things the legacy merge bundles together are **split onto two cadences**:

- **Fold (`merge_period = 1`)** — every step: fold `R(Q)` into `W`, zero `Q`,
  **keep** the block permutation Ψ, **keep** the optimizer momentum. `Q` is
  reborn at identity each step so the per-step angle stays small.
- **Reinit (`reinit_period = 400`, the *original* merge period)** — every 400
  steps: **resample Ψ** *and* **reset the optimizer momentum**. Resampling Ψ is
  necessary so the block-diagonal rotation eventually covers all neuron pairs
  (a fixed Ψ would only ever train one fixed subset). But a new Ψ is a new
  coordinate system, in which the carried momentum is meaningless — so momentum
  must be reset exactly when (and only when) Ψ changes.

Net: Ψ and the momentum coordinate frame stay **fixed and coherent for
~400-step stretches**; at each stretch boundary both flip together. Between
boundaries the optimizer behaves like one continuous run (momentum persists,
`Q` reborn each step).

poet0 is the **baseline** of the pipeline. It imports *none* of the
Pion-specific machinery (tangent-space gradient §2, scalar-`v` Lie momentum §3,
RMS-α §4, low-order Cayley §5, alternating single-sided §6, sharded merge §8).
Those are later ablations layered on top. poet0 keeps the current optimizer
exactly — ambient-space Megatron-Adam on `oft_R`, k=3 truncated Cayley,
two-sided, `scale=0.5`.

The value of poet0: it isolates whether folding every step with a
born-at-identity `Q` (small per-step angle) is stable and trains well *before*
any Pion geometry changes, while still getting full neuron-pair coverage via
periodic Ψ resampling.

## 2. Background: how POET merge/reset works today

The default POET path (`optim.poet.use_poet_adam=false`) trains `oft_R` as a
normal Megatron-Adam parameter and periodically merges via a training-loop
patch:

- **Config → CLI → args.** `optim.poet.*`
  ([poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml))
  is translated to `--poet-*` flags by
  [megatron_args.py:250](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L250),
  registered as argparse args in
  [pretrain_gpt_slm.py:31](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L31),
  read off `get_args()` at runtime.
- **Merge trigger.**
  [poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py)
  wraps `train_step`; after each step, if `iteration % poet_merge_period == 0`
  it calls `_run_merge` then (on the default Adam path)
  `_reset_vanilla_oft_state`.
- **`_run_merge`** ([line 216](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L216))
  calls `POETLinear.merge_then_reinitialize()` per layer and broadcasts the
  updated tensors across ranks.
- **`merge_then_reinitialize`** ([poet_layer.py:682](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L682))
  does three things in one call: (i) folds `R(oft_R)` into the frozen base
  weight using the **current** Ψ, (ii) generates a **new** Ψ and re-permutes the
  folded weight into the new layout (lines 693–709), (iii) zeros the bf16 model
  `oft_R`.
- **`_reset_vanilla_oft_state`** ([line 128](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L128))
  does **two** distinct jobs for the `oft_R` masters:
  - **(a) zero the fp32 master *value*** so it matches the merge's zeroed bf16
    model tensor. Load-bearing: in mixed precision the optimizer steps the fp32
    master, not the bf16 model tensor — if only the model tensor is zeroed, the
    next `optimizer.step()` copies the still-nonzero master back into the model
    and `oft_R` springs back to its pre-merge value, re-applying the just-merged
    rotation a second time → a recurring loss spike at every merge.
  - **(b) zero the Adam *moments*** (`exp_avg`, `exp_avg_sq`, per-param and
    per-group `step`).

Both optimizer layouts (plain `Float16OptimizerWithFloat16Params` and the
sharded `DistributedOptimizer`) are handled by `_iter_model_master_pairs`.

**The key observation:** the legacy path fuses *fold*, *Ψ-resample*, and
*momentum-reset* at a single cadence (`merge_period`). poet0 needs fold at
cadence 1 and Ψ-resample + momentum-reset at cadence 400. So all three must be
made independently gateable.

## 3. Design decisions (resolved)

| Decision | Choice | Rationale |
|---|---|---|
| Fold cadence | `merge_period = 1` | Doc §1 single-step: `Q` born at identity each step, small angle, folded into `W`, reset. |
| Ψ-resample + momentum-reset cadence | `reinit_period = 400` (new key; the legacy merge period) | Periodic Ψ resampling is needed for full neuron-pair coverage; momentum must reset with Ψ since a new permutation is a new coordinate frame. |
| Momentum between reinit boundaries | **Persist** | Within a fixed-Ψ stretch the moment buffers live in one coordinate frame and carry cleanly; Adam runs as one continuous optimization instead of restarting cold every step. |
| `oft_R` master value at fold | **Zeroed every step** | Correctness §2(a): without zeroing the fp32 master, `oft_R` springs back and re-applies the rotation → loss spike. Not optional; independent of the momentum decision. |
| Cayley/exp order | Unchanged (k=3 Cayley) | Low-order Cayley (§5) is a later ablation. |
| Q optimizer | Unchanged (Megatron-Adam, `scale=0.5`) | Lie-algebra / Muon geometry is later. |
| Sides | Unchanged (two-sided) | Alternating single-sided (§6) is later. |
| Backward compatibility | `reinit_period` unset ⇒ defaults to `merge_period` | Existing poet (`merge_period=400`, no `reinit_period`) keeps folding+resampling+resetting all at 400 — byte-identical to today. |

### Per-step decomposition

At each step the merge patch computes two booleans:

```
folding = (merge_period > 0 and iteration % merge_period == 0)
reinit  = (folding and reinit_period > 0 and iteration % reinit_period == 0)
```

and dispatches:

| | fold `R→W`, zero `Q` (model+master value) | resample Ψ | reset Adam moments |
|---|---|---|---|
| `folding` & not `reinit` | yes | no | no (momentum persists) |
| `folding` & `reinit` | yes | yes | yes |

For poet0 (`merge_period=1`, `reinit_period=400`): every step folds; every 400th
step also resamples Ψ and resets momentum. For legacy poet
(`merge_period=400`, `reinit_period`→400): `folding` and `reinit` coincide every
400 steps → identical to today.

**Constraint:** `reinit_period` should be an integer multiple of `merge_period`
so reinit boundaries always land on a folding step (else a scheduled reinit is
silently skipped). Validate at setup and error loudly if violated.

## 4. Scope of changes

Six edits.

### 4.1 New int arg — `launchers/pretrain_gpt_slm.py`

Register next to the other `--poet-*` flags
([line 45](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L45)):

```python
# Cadence (in optimizer steps) at which the block permutation is resampled AND
# Adam momentum is reset. 0 = fall back to --poet-merge-period (legacy: fold,
# resample, and reset all happen together). poet0 sets merge_period=1 (fold every
# step) and reinit_period=400, so Ψ + momentum stay fixed/coherent for 400-step
# stretches while Q is folded each step.
group.add_argument("--poet-reinit-period", type=int, default=0)
```

Default `0` → `args.poet_reinit_period == 0` for every existing recipe, which the
patch reads as "fall back to `merge_period`".

### 4.2 Emit the arg — `src/utils/megatron_args.py`

In the `kind == "poet"` branch
([line 250](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L250)),
alongside `--poet-merge-period`:

```python
"--poet-reinit-period",
poet.get("reinit_period", 0),
```

`reinit_period` defaults to `0` via `.get(..., 0)`, so existing poet configs
(which don't set it) emit `--poet-reinit-period 0` → fall-back-to-merge_period →
unchanged behavior.

### 4.3 Fold-only mode — `third_party/poet_torch/poet_layer.py`

Add a `reinit_perm: bool = True` parameter to `POETLinear.merge_then_reinitialize`
([line 682](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L682)):

- The fold (lines 684–691) is unchanged — it always uses the **current** Ψ.
- When `reinit_perm=True` (legacy): keep the new-permutation block exactly as
  today (generate new Ψ at lines 695–698, re-permute the folded weight with the
  **new** inverse perms at line 701, update the perm buffers at lines 706–709).
- When `reinit_perm=False` (poet0 non-boundary step): **skip** the new-Ψ
  generation and the buffer update; at line 701 re-permute the folded weight back
  into the **current** Ψ's layout using `self.perm_out_inv` / `self.perm_in_inv`
  (not freshly generated ones). Ψ buffers are left untouched.
- `oft_R_in.zero_()` / `oft_R_out.zero_()` (lines 711–712) run in both modes.

This is purely a fold (rotation absorbed into `W`, `oft_R` reset) with the
permutation held constant; the next forward with the unchanged Ψ is
mathematically consistent.

(Confirm during implementation whether any quantized sibling —
`merge_then_reinitialize` at lines 781 / 949 — is on an active path for the
configs poet0 targets; the default non-quantized dev path uses the float
`POETLinear` at line 509. If a sibling is reachable, thread the same parameter.)

### 4.4 Split the cadences — `src/patches/poet_merge_step.py`

- `_run_merge` ([line 216](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L216))
  gains a `reinit_perm: bool` parameter, passed through to
  `pl.merge_then_reinitialize(reinit_perm=reinit_perm)`. The cross-rank broadcast
  of `oft_R`, `weight`, and the perm buffers stays unconditional (weight + oft_R
  change every fold; broadcasting unchanged perms on non-boundary steps is a
  harmless no-op that keeps ranks in sync).
- `_reset_vanilla_oft_state` ([line 128](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L128))
  gains a `reset_moments: bool` parameter. The master-value zero (line 187) runs
  **unconditionally**; the `_zero_moments` call (line 189) and the per-group
  `step` reset loop (lines 195–206) run **only when `reset_moments`**. The log
  line reports whether moments were kept or reset.
- `_wrapped` ([line 56](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L56))
  computes the two booleans from §3 and dispatches:

  ```python
  merge_gap = getattr(opts, "poet_merge_period", 0)
  if merge_gap <= 0:  # fold disabled
      return ret
  # ... resolve iteration ...
  if iteration <= 0 or iteration % merge_gap != 0:
      return ret
  reinit_gap = getattr(opts, "poet_reinit_period", 0) or merge_gap
  do_reinit = (reinit_gap > 0 and iteration % reinit_gap == 0)
  _run_merge(model, dist, iteration, reinit_perm=do_reinit)
  if not getattr(opts, "poet_use_poet_adam", False):
      optimizer = args[3] if len(args) >= 4 else kwargs.get("optimizer")
      if optimizer is not None:
          _reset_vanilla_oft_state(optimizer, model, iteration, reset_moments=do_reinit)
  ```

  The `reinit_gap % merge_gap == 0` constraint (§3) is validated once at
  optimizer setup, not per step.

### 4.5 New experiment config — `configs/experiments/optim/poet0.yaml`

Clone of [poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml)
with:

- `experiment.name: poet0`, updated `description` documenting the single-step /
  periodic-reinit regime.
- Same `patches` list (`poet_merge_step` still applied — it now reads the new
  `reinit_period`).
- `optim.poet.merge_period: 1`.
- `optim.poet.reinit_period: 400` (new key; the only structural addition).
- Everything else identical (`block_count: 1`, `parameterization: cayley`,
  `q_optimizer: adam`, `scale: 0.5`, `train_output_rotation: true`,
  `use_poet_adam: false`, unfusing on).

A matching **`docs/experiments/poet0.md`** must be added in the same change — the
repo pre-commit hook *"Every experiment YAML has a matching
docs/experiments/<name>.md"* fails the commit otherwise.

### 4.6 New training script — `scripts/train_poet0.sh`

Clone of [train_poet_dev.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_poet_dev.sh)
with the single change `experiment=optim/poet` → `experiment=optim/poet0`. Same
defaults preserved: 60m dev scale, `seq_length=256`,
`training_regime=ablation_40x`, `scheduler=cosine_poet`, `cluster=h100_de`,
untied embeddings, `weight_decay=0.0`, GBS 1024 / MBS 128,
`wandb.project=slm-zeju-dev`, and `"$@"` passthrough so any CLI override wins.

## 5. Data flow (per step, poet0)

```
NON-BOUNDARY step (iteration % 400 != 0):
  forward/backward  → grad on oft_R (born at 0 this step), Ψ fixed
  optimizer.step()  → Adam updates oft_R using PERSISTED moments
  _run_merge(reinit_perm=False)
                    → fold R(oft_R) into W with CURRENT Ψ; keep Ψ;
                      zero bf16 model oft_R; broadcast
  _reset_vanilla_oft_state(reset_moments=False)
                    → zero fp32 master oft_R VALUE (no spring-back)
                      KEEP exp_avg / exp_avg_sq / step (momentum persists)

BOUNDARY step (iteration % 400 == 0):
  ... same fold ...
  _run_merge(reinit_perm=True)
                    → fold with current Ψ, then RESAMPLE Ψ, broadcast new Ψ
  _reset_vanilla_oft_state(reset_moments=True)
                    → zero master VALUE + zero moments (fresh coord frame)
```

## 6. Testing & verification

CPU-testable (no GPU/Megatron runtime needed):

1. **Arg-translation unit test** — drive the `megatron_args` `kind == "poet"`
   branch with a poet0-shaped config and assert the emitted argv contains
   `--poet-merge-period 1` **and** `--poet-reinit-period 400`; and that a stock
   poet config emits `--poet-merge-period 400` and `--poet-reinit-period 0`
   (fall-back). Use the established CPU test venv
   (`/lustre/fast/fast/zqiu/slm_env/.venv/bin/python` or the launchers test env).
2. **Cadence-dispatch unit test** — a small pure-python test over the two
   booleans (`folding`, `reinit`) for `(merge_period, reinit_period)` =
   `(1, 400)` and `(400, 0)`: assert fold-every-step + reinit-every-400 for
   poet0, and fold==reinit-every-400 for legacy. (Factor the boolean logic into a
   tiny helper so it is importable without Megatron.)
3. **Config resolution** — `--dry-run` the poet0 launcher invocation to confirm
   `experiment=optim/poet0` resolves and the new key flows through without schema
   errors.
4. **`py_compile` / `ruff`** on the edited Python files.

Not run here (the user's to launch): the GPU smoke run on the 60m dev scale to
confirm no per-step loss spike, correct Ψ-resample + momentum-reset at the 400
boundary, and that loss tracks/beats the `merge_period=400` baseline.

## 7. Out of scope (future ablations, per pipeline doc)

- §2 tangent-space gradient (`G_out`, `G_in`)
- §3 scalar-`v` Lie-algebra momentum
- §4 RMS-α step-size normalization
- §5 low-order Cayley / exp (k=1,2)
- §6 alternating single-sided update
- §8 block-sharded merge under DDP

Each is a separate spec → plan, layered on the poet0 baseline.
