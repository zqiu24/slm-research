# Log trainable / total params to the W&B run config (Megatron)

**Date:** 2026-06-01
**Status:** Design — pending review
**Scope:** Megatron backend only (POET, adam, muon, ngpt). Torchtitan is out of scope.

## Goal

Surface each run's parameter counts in the **W&B Overview → Config** table (a
static per-run value), *not* as a time-series chart. The motivating case is
POET, where the point is that trainable params (`oft_R`) are a tiny fraction of
the total — that ratio belongs next to the other run config, not on a curve.

Three fields:

| field | meaning |
|---|---|
| `trainable_params` | Σ `p.numel()` over params with `requires_grad=True` |
| `total_params`     | Σ `p.numel()` over all params |
| `trainable_pct`    | `100 × trainable_params / total_params` |

## Requirements

1. Values appear in the W&B run **config** (overview table), not as logged
   metrics/charts.
2. Counts are **global** for the run: deduplicated across data- and
   context-parallel replicas, summed across tensor-/pipeline-/expert-parallel
   shards. In the current DP-only configuration (TP=PP=EP=1) global == local, so
   this is a no-op today but stays correct if TP/PP/EP > 1 later.
3. Counted **after** the optimizer is set up, so `requires_grad` reflects the
   final trainable set (POET freezes base weights and unfreezes `oft_R` during
   optimizer construction).
4. Applies to **every** Megatron run automatically — no per-experiment opt-in.
5. Logging must **never crash training** (wrapped in try/except; failures fall
   back to writing `0` so the fields still appear).

## Non-goals

- Torchtitan support (its W&B config is a `JobConfig.to_dict()` snapshot; a
  separate hook would be needed — deferred).
- Per-layer / per-component breakdowns. Only the three run-level numbers.
- Any new metric chart.

## Design

### New patch: `src/patches/wandb_trainable_params.py`

Modeled on [`wandb_metric_normalize.py`](../../../src/patches/wandb_metric_normalize.py).
CPU-safe at import time (megatron / wandb / torch imported only inside
`apply()`), registered with `register_patch(name="wandb_trainable_params",
targets=())` (runtime wrapper, no static target ownership).

**Hook point.** Wrap `megatron.training.training.setup_model_and_optimizer`.
Call the original, then count from the returned `model` (a list of model
chunks). This runs after POET's freeze/unfreeze, so `requires_grad` is final.
Compose-safe: wrap the current symbol at `apply()` time and call through,
returning the original's `(model, optimizer, scheduler)` tuple unchanged.

**Timing.** `wandb.init()` runs inside `initialize_megatron()` at the top of
`pretrain()` — before `setup_model_and_optimizer` — so `wandb.config` already
exists when the wrapper fires.

### Counting (pure, CPU-testable)

A standalone helper:

```
count_local_params(model_chunks) -> (trainable: int, total: int)
    trainable = sum(p.numel() for mc in model_chunks for p in mc.parameters() if p.requires_grad)
    total     = sum(p.numel() for mc in model_chunks for p in mc.parameters())
```

No torch-dist, no Megatron — unit-tested on CPU with toy `nn.Module`s
(including a module with some params frozen, to assert trainable < total).

### Global aggregation (collective, all ranks)

All ranks call the wrapper and participate in the reductions (so no rank
deadlocks waiting on a collective the logging rank skips):

- All-reduce SUM `local_trainable` and `local_total` over the **model-parallel
  group** (`parallel_state.get_model_parallel_group()` → TP×PP).
- If expert parallelism is enabled (`expert_model_parallel_size > 1`), expert
  parameters are additionally summed across the expert-model-parallel group.
  (The plan pins the exact group calls + a test; not exercised by current runs.)
- Guarded: if `torch.distributed` is uninitialized (single-process), skip the
  reduction and use local counts.

### Writing to W&B config (logging rank only)

After the (collective) reduction, only the W&B-logging rank writes:

```
if megatron.training.training.get_wandb_writer() is not None:
    wandb.config.update(
        {"trainable_params": t, "total_params": n,
         "trainable_pct": round(100 * t / n, 4) if n else 0.0},
        allow_val_change=True,
    )
```

`get_wandb_writer()` is `None` on non-logging ranks, so only one rank writes.
The whole block is wrapped in try/except; on any failure it attempts to write
the three keys as `0` so they still appear in the overview.

### Enablement (always-on)

Applied unconditionally in
[`launchers/pretrain_gpt_slm.py`](../../../launchers/pretrain_gpt_slm.py)
alongside `_apply_runtime_patches(cfg)` (which runs before `pretrain()`), e.g.
`apply_patches(["wandb_trainable_params"])` after the experiment patches —
deduped if the experiment list already contains it. Not added to any
`experiment.patches` YAML; coverage is universal and automatic.

## Testing

- **CPU unit test** (`tests/unit/`): `count_local_params` on toy modules — all
  trainable, some frozen (POET-like), all frozen — asserting the trainable/total
  split. No GPU, no dist.
- **GPU smoke (user-run):** a short POET run; confirm `trainable_params`,
  `total_params`, `trainable_pct` show in the W&B overview config and that
  `trainable_pct` is small (oft_R ≪ total). Confirm a non-W&B rank does not error.

## Edge cases

- **POET base weights as frozen `Parameter`s** (not buffers): they count toward
  `total_params` but not `trainable_params` — the desired denominator. The plan
  verifies the adapter freezes via `requires_grad=False` rather than converting
  to buffers (which would drop them from `total`).
- **`total_params == 0`** (degenerate): `trainable_pct` → `0.0`, no divide-by-zero.
- **W&B disabled / offline:** `get_wandb_writer()` is `None`/offline-safe; the
  `update` is a no-op or skipped. No crash.
- **Resume:** `allow_val_change=True` lets the value be (re)written on restart.
