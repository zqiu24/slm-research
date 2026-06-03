# poet0 — Single-Step POET (merge every step, persistent momentum)

**Date:** 2026-06-03
**Status:** Design approved, ready for implementation plan
**Related:** [docs/poetx_pion_pipeline.md](/lustre/fast/fast/zqiu/slm-research/docs/poetx_pion_pipeline.md) §1

---

## 1. Goal

Add a new POET training variant, **poet0**, that runs the existing POET
optimizer in the *single-step* regime described in
[poetx_pion_pipeline.md](/lustre/fast/fast/zqiu/slm-research/docs/poetx_pion_pipeline.md)
§1: the merge interval is **1** instead of 400, so every step the block
rotation `Q` is born at identity, takes one small step, is exponentiated, folded
into the live weight `W`, and reset.

poet0 is deliberately the **baseline** of that pipeline. It imports *none* of
the Pion-specific machinery (tangent-space gradient §2, scalar-`v` Lie momentum
§3, RMS-α §4, low-order Cayley §5, alternating single-sided §6, sharded merge
§8). Those are later ablations layered on top of this baseline. poet0 keeps the
current optimizer exactly — ambient-space Megatron-Adam on `oft_R`, k=3
truncated Cayley, two-sided, `scale=0.5` — and changes only **two** things
relative to `experiment=optim/poet`:

1. `merge_period: 400 → 1` (merge every step).
2. Optimizer momentum on `oft_R` **persists** across the per-step merge instead
   of being reset.

The value of poet0: it isolates whether merge-every-step with a born-at-identity
`Q` (small per-step angle) is stable and trains well *before* any of the Pion
geometry changes are introduced, giving a clean control for the rest of the
pipeline.

## 2. Background: how POET merge/reset works today

The default POET path (`optim.poet.use_poet_adam=false`) trains `oft_R` as a
normal Megatron-Adam parameter and periodically merges via a training-loop
patch:

- **Config → CLI → args.** `optim.poet.*`
  ([poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml))
  is translated to `--poet-*` flags by
  [megatron_args.py:250](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L250),
  which are registered as argparse args in
  [pretrain_gpt_slm.py:31](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L31)
  and read off `get_args()` at runtime.
- **Merge trigger.**
  [poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py)
  wraps `train_step`; after each step, if `iteration % poet_merge_period == 0`
  it calls `_run_merge` then (on the default Adam path)
  `_reset_vanilla_oft_state`.
- **`_run_merge`** ([line 216](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L216))
  calls `POETLinear.merge_then_reinitialize()` per layer: folds `R(oft_R)` into
  the frozen base weight, zeros the **bf16 model** `oft_R`, and **re-randomizes
  the block permutations** (Ψ).
- **`_reset_vanilla_oft_state`** ([line 128](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L128))
  does **two** distinct jobs for the `oft_R` masters:
  - **(a) zero the fp32 master *value*** so it matches the merge's zeroed bf16
    model tensor. Load-bearing: in mixed precision the optimizer steps the fp32
    master, not the bf16 model tensor — if only the model tensor is zeroed, the
    next `optimizer.step()` copies the still-nonzero master back into the model
    and `oft_R` springs back to its pre-merge value, re-applying the
    just-merged rotation a second time → a recurring loss spike at every merge.
  - **(b) zero the Adam *moments*** (`exp_avg`, `exp_avg_sq`, per-param and
    per-group `step`) so the post-merge restart gets fresh momentum + bias
    correction.

Both layouts (plain `Float16OptimizerWithFloat16Params` and the sharded
`DistributedOptimizer`) are handled by `_iter_model_master_pairs`.

## 3. Design decisions (resolved)

| Decision | Choice | Rationale |
|---|---|---|
| Merge interval | `merge_period = 1` | Doc §1 single-step: `Q` born at identity each step, small angle, folded into `W`, reset. |
| Momentum at merge | **Persist** (do *not* reset moments) | Doc §3: because every step starts at identity, the moment buffers live in the same tangent space and carry cleanly; resetting every step would leave Adam memoryless (degenerate). |
| `oft_R` master value at merge | **Still zeroed** | Correctness §2(a): without zeroing the fp32 master, `oft_R` springs back and the rotation is re-applied → loss spike. Not optional. |
| Block permutation Ψ | **Resample every step** (current merge behavior) | Doc §1 per-step block-stochastic Ψ. Accepted tradeoff: persisted moments then refer to shifting neuron pairs → geometrically *approximate*; revisit if it hurts. No code change (the existing merge already resamples). |
| Cayley/exp order | Unchanged (k=3 Cayley default) | Low-order Cayley (§5) is a later ablation, not poet0. |
| Q optimizer | Unchanged (Megatron-Adam, `scale=0.5`) | Lie-algebra / Muon geometry is later. |
| Sides | Unchanged (two-sided, `train_output_rotation=true`) | Alternating single-sided is §6, later. |

### Key correctness consequence

"Keep the momentum" splits the existing reset into its two jobs:

- **Always** run job (a): zero the fp32 master value.
- **Skip** job (b): leave `exp_avg` / `exp_avg_sq` / `step` untouched.

Net per-step effect: `oft_R` is reborn at 0 (value zeroed on both model and
master, no spring-back), while the Adam moments on it persist — Adam behaves as
one continuous optimization rather than restarting cold every step.

## 4. Scope of changes

Five edits. The flag follows the existing **off-switch precedent**
(`--poet-freeze-output-rotation`, emitted only when
`train_output_rotation: false`): default-absent preserves today's behavior, and
only poet0 emits it.

### 4.1 New off-switch arg — `launchers/pretrain_gpt_slm.py`

Register, next to the other `--poet-*` flags
([line 45](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L45)):

```python
# Skip the post-merge Adam-momentum reset (keep exp_avg/exp_avg_sq/step across
# merges). The master VALUE is still zeroed by the merge patch regardless. Used
# by the single-step poet0 regime (merge_period=1) so momentum persists instead
# of restarting cold every step.
group.add_argument("--poet-no-momentum-reset", action="store_true")
```

`store_true` → `args.poet_no_momentum_reset` defaults `False` (current reset
behavior) for every existing recipe.

### 4.2 Emit the flag — `src/utils/megatron_args.py`

In the `kind == "poet"` branch
([line 250](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L250)),
after the `--poet-freeze-output-rotation` block, append the flag when the config
opts out of the reset:

```python
# store_true: keep Adam momentum across the per-step merge (single-step poet0).
if not poet.get("reset_momentum_on_merge", True):
    poet_args.append("--poet-no-momentum-reset")
```

`reset_momentum_on_merge` defaults `True` via `.get(..., True)`, so existing
poet configs (which don't set it) are unaffected.

### 4.3 Split the reset — `src/patches/poet_merge_step.py`

In `_wrapped` ([line 80](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L80))
pass the flag through, and in `_reset_vanilla_oft_state`
([line 128](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L128))
gate **only the moment-zeroing**, never the value-zeroing:

- Add a `reset_moments: bool` parameter to `_reset_vanilla_oft_state`.
- The master-value zero (`master_p.detach().zero_()`,
  [line 187](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L187))
  runs unconditionally.
- Wrap the `_zero_moments(...)` call
  ([line 189](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L189))
  and the per-group `step` reset loop
  ([lines 195–206](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L195))
  in `if reset_moments:`.
- Caller passes
  `reset_moments=not getattr(opts, "poet_no_momentum_reset", False)`.
- The log line should report whether moments were kept vs reset, for run
  forensics.

`_run_merge` is **unchanged** — it still runs every step, so Ψ resamples every
step (the accepted §1 behavior) and the model+master `oft_R` value is zeroed.

### 4.4 New experiment config — `configs/experiments/optim/poet0.yaml`

Clone of [poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml)
with:

- `experiment.name: poet0`, updated `description` documenting the single-step /
  persistent-momentum regime.
- Same `patches` list as poet (`poet_merge_step` still applied — it now reads
  the new flag).
- `optim.poet.merge_period: 1`.
- `optim.poet.reset_momentum_on_merge: false` (new key; the only addition).
- Everything else identical (`block_count: 1`, `parameterization: cayley`,
  `q_optimizer: adam`, `scale: 0.5`, `train_output_rotation: true`,
  `use_poet_adam: false`, unfusing on).

### 4.5 New training script — `scripts/train_poet0.sh`

Clone of [train_poet_dev.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_poet_dev.sh)
with the single change `experiment=optim/poet` → `experiment=optim/poet0`. Same
defaults preserved: 60m dev scale, `seq_length=256`, `training_regime=ablation_40x`,
`scheduler=cosine_poet`, `cluster=h100_de`, untied embeddings, `weight_decay=0.0`,
GBS 1024 / MBS 128, `wandb.project=slm-zeju-dev`, and `"$@"` passthrough so any
CLI override still wins.

## 5. Data flow (per step, poet0)

```
forward/backward  → grad on oft_R (born at 0 this step)
optimizer.step()  → Adam updates oft_R using PERSISTED exp_avg/exp_avg_sq
                    (master oft_R now nonzero; model copies master)
_run_merge        → fold R(oft_R) into W; zero bf16 model oft_R;
                    resample Ψ (perm_in/out re-randomized)
_reset_vanilla_oft_state(reset_moments=False):
                    zero fp32 master oft_R VALUE (no spring-back)
                    KEEP exp_avg / exp_avg_sq / step  (momentum persists)
→ next step: oft_R born at 0 again, momentum intact
```

## 6. Testing & verification

CPU-testable (no GPU/Megatron runtime needed):

1. **Arg-translation unit test** — drive `megatron_args` (the `kind == "poet"`
   branch) with a poet0-shaped config and assert the emitted argv contains
   `--poet-merge-period 1` **and** `--poet-no-momentum-reset`; and that a stock
   poet config emits neither (no `--poet-no-momentum-reset`, `--poet-merge-period
   400`). Use the established CPU test venv
   (`/lustre/fast/fast/zqiu/slm_env/.venv/bin/python` or the launchers test env).
2. **Config resolution** — `--dry-run` the poet0 script / launcher invocation to
   confirm `experiment=optim/poet0` resolves and the new key flows through
   without schema errors.
3. **`py_compile` / `ruff`** on the three edited Python files.

Not run here (the user's to launch): the GPU smoke run on the 60m dev scale to
confirm no per-step loss spike and that loss tracks/beats the `merge_period=400`
baseline.

## 7. Out of scope (future ablations, per pipeline doc)

- §2 tangent-space gradient (`G_out`, `G_in`)
- §3 scalar-`v` Lie-algebra momentum
- §4 RMS-α step-size normalization
- §5 low-order Cayley / exp (k=1,2)
- §6 alternating single-sided update
- §8 block-sharded merge under DDP

Each is a separate spec → plan, layered on the poet0 baseline.
