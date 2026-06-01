# Log trainable / total params to the W&B run config (Megatron)

**Date:** 2026-06-01
**Status:** Design — pending review
**Scope:** Megatron backend only (POET, adam, muon, ngpt). Torchtitan is out of scope.

## Goal

Surface each run's parameter counts in the **W&B Overview → Config** table (a
static per-run value), *not* as a time-series chart. The motivating case is
POET, where the point is that trainable params (`oft_R`) are a tiny fraction of
the total — that ratio belongs next to the other run config, not on a curve.

Four fields:

| field | meaning |
|---|---|
| `trainable_params` | Σ `p.numel()` over params with `requires_grad=True` |
| `total_params`     | Σ `p.numel()` over all params |
| `trainable_pct`    | `100 × trainable_params / total_params` |
| `poet_params`      | Σ `p.numel()` over params whose name contains `oft_R` (POET's trainable orthogonal generators). `0` for non-POET runs (adam / muon / ngpt) — they have no such params, so the field is always present and reads `0` unless POET is applied. |

`poet_params` is the motivating extra: for POET it isolates the orthogonal-delta
count (`oft_R`) from the rest of the trainable set (norms / embeddings / LM head,
which are *also* trained — see the trainable-vs-POET distinction), so the
overview shows at a glance both "how little is trainable" (`trainable_pct`) and
"how much of that is POET's delta" (`poet_params`). For any non-POET optimizer it
is identically `0`.

## Requirements

1. Values appear in the W&B run **config** (overview table), not as logged
   metrics/charts.
2. Counts are **global** for the run: deduplicated across data- and
   context-parallel replicas, summed across tensor-/pipeline-/expert-parallel
   shards. In the current DP-only configuration (TP=PP=EP=1) global == local, so
   this is a no-op today but stays correct if TP/PP/EP > 1 later.
3. Counted **after** the optimizer is set up, so `requires_grad` reflects the
   final trainable set (POET freezes base weights and unfreezes `oft_R` during
   optimizer construction). `poet_params` is counted by parameter **name**
   (`oft_R`), independent of `requires_grad`, and is `0` for any run without
   POET layers.
4. Applies to **every** Megatron run automatically — no per-experiment opt-in.
5. Logging must **never crash training** (wrapped in try/except; failures fall
   back to writing `0` so the fields still appear).

## Non-goals

- Torchtitan support (its W&B config is a `JobConfig.to_dict()` snapshot; a
  separate hook would be needed — deferred).
- Per-layer / per-component breakdowns. Only the three run-level numbers.
- Any new metric chart.

## Why this works (precedent)

Megatron already does exactly this: after `setup_model_and_optimizer`, on the
W&B-logging rank, it calls
`wandb_writer.config.update({'slurm_job_name': ...})`
([training.py:1166-1169](../../../third_party/Megatron-LM/megatron/training/training.py#L1166-L1169),
where `wandb_writer = get_wandb_writer()` is the `wandb` module). This proves
the mechanism (`config.update` lands in the Overview config), the timing (the
writer is live post-setup), and the rank gating (`get_wandb_writer()` is `None`
off the logging rank). The wandb writer is set inside `initialize_megatron`
(`_set_wandb_writer` → `wandb.init` →
[`_GLOBAL_WANDB_WRITER = wandb`](../../../third_party/Megatron-LM/megatron/training/global_vars.py#L253-L254)),
which runs at [training.py:891](../../../third_party/Megatron-LM/megatron/training/training.py#L891),
before `setup_model_and_optimizer` at
[training.py:1026](../../../third_party/Megatron-LM/megatron/training/training.py#L1026).

## Design

### New patch: `src/patches/wandb_trainable_params.py`

Modeled on [`wandb_metric_normalize.py`](../../../src/patches/wandb_metric_normalize.py).
CPU-safe at import time (megatron / wandb / torch imported only inside
`apply()`), registered with `register_patch(name="wandb_trainable_params",
targets=())` (runtime wrapper, no static target ownership).

**Hook point.** Wrap `megatron.training.training.setup_model_and_optimizer`.
Call the original, then count from the returned `model` (a list of model
chunks), so `requires_grad` reflects the final trainable set.

**Compose-safe with POET.** The POET patch
[`poet_optimizer_setup`](../../../src/patches/poet_optimizer_setup.py) wraps the
*optimizer builders* (`get_megatron_optimizer_config` / `get_megatron_optimizer`
→ `get_megatron_poet_optimizer`), **not** `setup_model_and_optimizer` — so our
wrapper composes cleanly. The trainable set is fixed earlier, at model-build
time, by [`poet_apply_to_model`](../../../src/patches/poet_apply_to_model.py)
(frozen base weight + `requires_grad=True` `oft_R`); the custom-POETAdam path's
transient `requires_grad` toggling is fully restored before
`setup_model_and_optimizer` returns. Either way, counting post-return is
correct. Wrap the current symbol at `apply()` time, call through, and return the
original's `(model, optimizer, scheduler)` tuple unchanged.

### Counting (pure, CPU-testable)

A standalone helper in a shared util (e.g. `src/utils/param_count.py`):

```
count_local_params(model_chunks) -> (trainable: int, total: int, poet: int)
    trainable = sum(p.numel() for mc in model_chunks for p in mc.parameters() if p.requires_grad)
    total     = sum(p.numel() for mc in model_chunks for p in mc.parameters())
    poet      = sum(p.numel() for mc in model_chunks
                    for name, p in mc.named_parameters() if "oft_R" in name)
```

`trainable`/`total` are the same arithmetic already inlined in
[`poet_apply_to_model.py:162-163`](../../../src/patches/poet_apply_to_model.py#L162-L163)
(a debug print), which also confirms POET's frozen base weights remain
`nn.Parameter`s with `requires_grad=False` — so they count toward `total_params`
but not `trainable_params`, exactly the denominator we want. Extracting the
helper gives one tested source of truth; rewiring that existing debug print to
call it is optional (nice-to-have, not required).

`poet` is counted **by name** (`"oft_R" in name`), not by `requires_grad`,
matching the `oft_R` detector already used in the POET parameter dump
([`poet_apply_to_model.py:152`](../../../src/patches/poet_apply_to_model.py#L152)).
The substring catches all three layouts: the decoupled `oft_R_in` / `oft_R_out`
and the legacy single `oft_R`. For any non-POET model no parameter name contains
`oft_R`, so the sum is `0` — that is exactly the "normally 0" behaviour wanted
for adam / muon / ngpt, with no optimizer-type branching needed.

No torch-dist, no Megatron — unit-tested on CPU with toy `nn.Module`s
(all-trainable, some-frozen POET-like, all-frozen) asserting the split.

### Global aggregation (collective, all ranks)

The reduction runs **uniformly on every rank** — the patch is applied per-rank
in the launcher, so every rank installs the wrapper and reaches the same
collective. The collective MUST stay outside the logging-rank gate, or ranks
that skip it would hang the ones that don't. Only the final `config.update` is
rank-gated.

- All-reduce SUM `local_trainable`, `local_total`, **and `local_poet`** over the
  **model-parallel group**
  ([`parallel_state.get_model_parallel_group()`](../../../third_party/Megatron-LM/megatron/core/parallel_state.py#L1377)
  → TP×PP). `poet_params` rides along as a third element of the same reduced
  vector — identical group, identical reduction — so it gets the same
  dedup/sum treatment for free. DP/CP ranks are replicas, so they're excluded —
  no double-counting. No-op today (all sizes = 1).
- If expert parallelism is enabled (`expert_model_parallel_size > 1`), expert
  parameters are additionally summed across the expert-model-parallel group.
  This applies to `poet_params` too: POET-ized expert-FFN linears hold their
  `oft_R` on the expert-parallel shards, so the poet count is split across the
  EP group exactly like `total`/`trainable` and must be reduced over the same
  group. (The plan pins the exact group calls + a test; not exercised by current
  runs.)
- Guarded: if `torch.distributed` is uninitialized (single-process), skip the
  reduction and use local counts.

### Writing to W&B config (logging rank only)

After the (collective) reduction, only the W&B-logging rank writes:

```
writer = megatron.training.training.get_wandb_writer()
if writer is not None:
    writer.config.update(
        {"trainable_params": t, "total_params": n,
         "trainable_pct": round(100 * t / n, 4) if n else 0.0,
         "poet_params": pp},
        allow_val_change=True,
    )
```

Use `get_wandb_writer().config.update(...)` (Megatron's own idiom at
[training.py:1169](../../../third_party/Megatron-LM/megatron/training/training.py#L1169)),
not the bare `wandb` module. `get_wandb_writer()` is `None` on non-logging
ranks, so only one rank writes. The write is wrapped in try/except; on any
failure it attempts to write the four keys as `0` so they still appear in the
overview. (The collective reduction stays *outside* this try/except and this
rank gate — see above.)

### Enablement (always-on)

Applied unconditionally in
[`launchers/pretrain_gpt_slm.py`](../../../launchers/pretrain_gpt_slm.py)
alongside `_apply_runtime_patches(cfg)` (which runs before `pretrain()`):
`import src.patches.wandb_trainable_params` (the registry raises `UnknownPatch`
for un-imported modules) then `apply_patches(["wandb_trainable_params"])` after
the experiment patches. `apply_patches` is idempotent — it skips entries whose
`applied` flag is set ([`_registry.py:101`](../../../src/patches/_registry.py#L101)) —
so this is a no-op if some experiment list ever adds it too. Not added to any
`experiment.patches` YAML; coverage is universal and automatic.

## Testing

- **CPU unit test** (`tests/unit/`): `count_local_params` on toy modules — all
  trainable, some frozen (POET-like), all frozen — asserting the trainable/total
  split. For `poet_params`: a POET-like toy registering a parameter whose name
  contains `oft_R` (e.g. via a submodule with an `oft_R` attribute) asserts
  `poet == that param's numel`; a toy with no such name asserts `poet == 0`
  (covers the adam/muon "normally 0" case). Also assert `poet` counts the
  decoupled `oft_R_in` / `oft_R_out` names. No GPU, no dist.
- **GPU smoke (user-run):** a short POET run; confirm `trainable_params`,
  `total_params`, `trainable_pct`, **`poet_params`** show in the W&B overview
  config, that `trainable_pct` is small (oft_R ≪ total), and that `poet_params`
  is nonzero and ≤ `trainable_params`. Confirm a non-W&B rank does not error. As
  a control, a short non-POET run (adam) should show `poet_params == 0`.

## Edge cases

- **POET base weights as frozen `Parameter`s** (not buffers): they count toward
  `total_params` but not `trainable_params` — the desired denominator. Verified:
  [`poet_apply_to_model.py:162-163`](../../../src/patches/poet_apply_to_model.py#L162-L163)
  reaches the frozen base weights via `.parameters()` (so they are `Parameter`s
  with `requires_grad=False`, not buffers — buffers would drop from `total`).
- **`total_params == 0`** (degenerate): `trainable_pct` → `0.0`, no divide-by-zero.
- **Non-POET runs** (adam / muon / ngpt): no parameter name contains `oft_R`, so
  `poet_params == 0`. The field is still written, so every Megatron run's
  overview carries a `poet_params` value (`0` or the real count) — uniform schema
  across optimizers, no per-optimizer branching.
- **W&B disabled / offline:** `get_wandb_writer()` is `None`/offline-safe; the
  `update` is a no-op or skipped. No crash.
- **Resume:** `allow_val_change=True` lets the value be (re)written on restart.
