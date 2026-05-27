# Decoupled LR-scheduler config axis — design

**Date:** 2026-05-27
**Status:** approved (design), pending implementation plan
**Scope:** single-GPU / config-layer only; no Megatron training-loop surgery

---

## Problem

LR-scheduler configuration today lives in two places, and they disagree:

1. **A phantom `scheduler:` block** inside every `configs/training_regime/*.yaml`
   (`type: wsd`, `peak_lr`, `stable_fraction`, `decay_fraction`, `decay_shape`,
   `min_lr_ratio`, `warmup_tokens`). **None of these fields are read by any Python.**
   `grep` for `peak_lr` / `stable_fraction` / `decay_fraction` in `src/` and
   `launchers/` returns nothing.

2. **The real, consumed surface**: `training.lr_decay_style` (default `"cosine"`)
   plus a warmup hardcoded in `src/utils/megatron_args.py` as
   `warmup_samples = max(1, (total_tokens // seq_length) // 500)` (a fixed 0.2%),
   plus `--min-lr` from `training.min_lr`.

Consequences:

- Every regime declares `type: wsd` but **actually runs cosine** (the default
  `lr_decay_style`). WSD has never executed.
- Warmup is not configurable without editing Python.
- A user tuning `scheduler.peak_lr` sees no effect — a silent footgun.

The repo's Megatron build (`third_party/Megatron-LM`) **natively supports** the
full set of decay styles we want, so this is a *wiring + config* problem, not a
"port a scheduler" problem:

- `lr_decay_style ∈ {constant, linear, cosine, inverse-square-root, WSD}`
  (`megatron/core/optimizer_param_scheduler.py`, `get_lr`).
- `lr_wsd_decay_style ∈ {exponential, linear, cosine, minus_sqrt}` and
  `lr_wsd_decay_samples` for the WSD anneal tail.
- `--lr-warmup-fraction` / `--lr-warmup-samples`, `--min-lr`, `--lr-decay-samples`.
- `--finetune` and `--override-opt-param-scheduler` for resume.

## Goals

1. **Make every declared scheduler genuinely run** (cosine + WSD + the other
   Megatron-native styles). No more phantom config.
2. **Clean per-run switching** of schedulers, with per-field tweaks, via a
   first-class composable config axis.
3. **Decay-only WSD resume** (compute reuse for decay-stage ablations).
4. **Remove the step-decay legacy** (Megatron has no native `step`; we don't
   want to maintain the patch).

## Non-goals

- No multi-GPU-specific work; behavior is identical across ranks (LR schedule
  is global).
- No Megatron training-loop patches. Everything is flags + config + one pure
  resolver function.
- POET dead-code cleanup (old `chain_layer` op, `forward_core`,
  `merge_then_reinitialize_working`) is **out of scope** — separate effort.

---

## Decisions (settled during brainstorming)

| # | Decision |
|---|---|
| D1 | **Approach C**: a single pure resolver function `scheduler_args()` in a new `src/utils/scheduler.py`. `_training_args` calls it in one line. Mirrors `src/utils/ladder_math.py` (pure, no Megatron import, laptop-testable). |
| D2 | **Scheduler is its own Hydra config group** `configs/scheduler/`, one file per scheduler type, decoupled from `training_regime` (which keeps only token budget / batch / checkpointing). |
| D3 | **Types**: `constant`, `linear`, `cosine`, `inverse-square-root`, `wsd` (+ `wsd_decay_only`). All Megatron-native. **No `step`.** |
| D4 | **Peak LR stays `optim.lr`** (`--lr`); `scheduler.*` owns shape only. |
| D5 | **`min_lr_ratio`** (× `optim.lr`), not absolute `min_lr`. Resolver multiplies and emits absolute `--min-lr`. |
| D6 | **Warmup**: support both `warmup_fraction` (default, → `--lr-warmup-fraction`) and `warmup_tokens` (→ `--lr-warmup-samples`, ÷ seq_length). Mutually exclusive, validated. **Default `warmup_fraction: 0.01` (1%)** in the scheduler files. |
| D7 | **Type casing** accepted lowercase in config (`wsd`, `inverse_square_root`), normalized to Megatron's (`WSD`, `inverse-square-root`). |
| D8 | **Type-foreign fields are rejected** (e.g. `wsd_decay_style` on `cosine` → `ValueError`), killing the phantom-config bug class. |
| D9 | **Decay-only = two-run, flag-only (Option 4a)**: a `constant` "stable" run saves a final checkpoint; a decay-only run resumes from it with `--finetune --override-opt-param-scheduler` and `warmup_fraction: 0`. No force-save hook. |
| D10 | **Remove step-decay legacy** entirely (recoverable from git). |

---

## Architecture

### Config schema (`scheduler:` block, `# @package _global_`)

```yaml
scheduler:
  type: cosine              # constant | linear | cosine | inverse-square-root | wsd

  # Warmup — exactly one (fraction is the default):
  warmup_fraction: 0.01     # → --lr-warmup-fraction
  # warmup_tokens: 60_000_000  # → --lr-warmup-samples (÷ seq_length)

  # LR floor (all types):
  min_lr_ratio: 0.1         # min_lr = min_lr_ratio × optim.lr → --min-lr

  # wsd-only:
  wsd_decay_fraction: 0.2   # tail anneal length, fraction of total → --lr-wsd-decay-samples
  wsd_decay_style: cosine   # exponential | linear | cosine | minus_sqrt → --lr-wsd-decay-style
```

### Config group `configs/scheduler/` (new)

| File | Contents |
|---|---|
| `cosine.yaml` | `type: cosine`, `warmup_fraction: 0.01`, `min_lr_ratio: 0.1` |
| `wsd.yaml` | cosine fields + `wsd_decay_fraction: 0.2`, `wsd_decay_style: cosine` |
| `constant.yaml` | `type: constant`, `warmup_fraction: 0.01` |
| `linear.yaml` | `type: linear`, `warmup_fraction: 0.01`, `min_lr_ratio: 0.1` |
| `inverse_square_root.yaml` | `type: inverse-square-root`, `warmup_fraction: 0.01`, `min_lr_ratio: 0.1` |
| `wsd_decay_only.yaml` | `type: wsd`, `warmup_fraction: 0`, `wsd_decay_fraction: 1.0`, `wsd_decay_style: cosine` (composed with the decay-only regime) |

Default added to `configs/launch/config.yaml` defaults list:

```yaml
defaults:
  - base/family: qwen3
  - base/scale: 300m
  - scheduler: cosine        # NEW
  - experiment: champion
  - training_regime: ablation_20x
  - ...
```

Selection / tweaking:

```bash
scripts/train_adam.sh llama3 scheduler=wsd
scripts/train_adam.sh llama3 scheduler=cosine scheduler.warmup_fraction=0.02
```

### Resolver (`src/utils/scheduler.py`, new)

```python
def scheduler_args(
    sched: Mapping,        # cfg.scheduler
    *,
    peak_lr: float,        # cfg.optim.lr
    total_tokens: int,     # already computed in _training_args
    seq_length: int,
) -> list[str]:
    """Validate the scheduler block and return Megatron LR/decay CLI flags."""
```

Three phases:

1. **Validate** — known `type`; required fields per type (`wsd` needs
   `wsd_decay_fraction`); exactly one of `warmup_fraction` / `warmup_tokens`;
   `0 <= warmup_fraction < 1`; `0 < wsd_decay_fraction <= 1`; type-foreign
   fields rejected. Clear `ValueError` messages.
2. **Resolve** — normalize `type` casing; `min_lr = min_lr_ratio × peak_lr`;
   warmup fraction→flag or `warmup_tokens // seq_length`;
   `wsd_decay_samples = round(wsd_decay_fraction × total_tokens) // seq_length`.
3. **Emit** — `--lr-decay-style`, `--min-lr`, the warmup flag, and for WSD
   `--lr-wsd-decay-style` + `--lr-wsd-decay-samples`.

### Integration (`src/utils/megatron_args.py::_training_args`)

Remove (lines ~150–171): hardcoded `warmup_samples`, inline `--lr-decay-style`
read, the `if lr_decay_style == "step":` branch, and the `training.min_lr`
`--min-lr` line.

Replace with:

```python
_add(args, "--lr", peak_lr)                          # optim.lr — unchanged
_add(args, "--train-samples", total_tokens // seq_length)
_add(args, "--lr-decay-samples", total_tokens // seq_length)
args.extend(scheduler_args(cfg.scheduler, peak_lr=peak_lr,
                           total_tokens=total_tokens, seq_length=seq_length))
```

### Decay-only resume (Option 4a)

Workflow (two launches, both pure flags):

1. **Stable run** — `scheduler=constant`, regime token budget = stable-phase
   length. Holds peak LR; saves a normal final checkpoint = the stable checkpoint.
2. **Decay-only run** — `scheduler=wsd_decay_only training_regime=final_wsd_decay_only`.
   When `training.resume_from_stable_stage` is set, the arg builder points
   `--load` at the stable checkpoint dir and adds `--finetune`
   `--override-opt-param-scheduler`; `warmup_fraction: 0` plus
   `wsd_decay_fraction: 1.0` make the whole run a pure anneal.

This is a small branch in the checkpoint-args section of `megatron_args.py`
(currently `--save`/`--load` both point at `{archive}/checkpoints`,
~line 335). The exact "where does the stable checkpoint dir come from" wiring is
a launcher ergonomics detail to be specified in the plan; the mechanism is flags
only.

### Removals (step-decay legacy)

| Path | Action |
|---|---|
| `src/patches/lr_decay_style_step.py` | delete |
| `configs/experiments/optim/adamw_step_decay.yaml` | delete |
| `tests/unit/test_megatron_args_step_decay.py` | delete |
| `launchers/pretrain_gpt_slm.py` (`--lr-decay-step-ratio/coeff`) | remove args |
| `src/utils/megatron_args.py` (inline `step` branch) | remove (subsumed by resolver) |

Phantom `scheduler:` blocks are removed from all 5 `training_regime/*.yaml`
(regimes keep only token budget / batch / checkpointing; the decay-only regime
keeps `resume_from_stable_stage` / `decay_tokens`).

---

## Testing

- **`tests/unit/test_scheduler.py`** (new, CPU, no Megatron): drives
  `scheduler_args()` directly. One happy path per type asserting the exact
  emitted flag list; validation-failure cases (wsd missing `wsd_decay_fraction`;
  both warmup forms set; type-foreign field; out-of-range fractions).
- **`tests/unit/test_megatron_args.py`** (extend): each `scheduler/*.yaml`
  composes into the right argv end-to-end.
- **No GPU required** for any test — pure config→flags.

All existing tests must still pass after the step-decay test file is deleted.

---

## Risks

| Risk | Mitigation |
|---|---|
| Default warmup change (0.2% → 1%) alters de-facto schedule vs past runs | Intentional (D6); documented; reproducibility-sensitive runs pin `warmup_fraction`. |
| Decay-only "stable checkpoint dir" wiring under-specified | Mechanism (flags) is settled; ergonomics deferred to the implementation plan. |
| Removing `adamw_step_decay` drops a real (referenced) capability | Recoverable from git; user opted in explicitly. |
| `--lr-warmup-fraction` vs `--lr-warmup-samples` both emitted | Validator enforces exactly one; Megatron also asserts mutual exclusivity. |

## Out of scope

- POET dead-code cleanup.
- Per-parameter-group schedules (decoupled embedding LR via `--decoupled-lr`).
- A force-save-at-anneal-boundary single-run decay-only (Option 4b).
