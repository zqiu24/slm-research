# Unified W&B logging across Megatron and torchtitan backends — design

**Date:** 2026-05-31
**Status:** approved (design), implemented.
**Update 2026-06-01:** the `log_grad_norm_extra` patch (the `grad-norm-clipped` /
`grad-norm-clip-coeff` POET-debug scalars) has since been **removed** from all
experiments — only the raw grad-norm (`train/grad_norm`) is kept. Mentions of it
below (passthrough lists, `targets=()` composition) are historical context.
**Scope:** normalize the **W&B metric keys** emitted by the two training
backends (`backend=megatron` via `scripts/train_adam.sh`, `backend=torchtitan`
via `scripts/train_adam_titan.sh`) onto a single, neutral, namespaced schema, so
the same dashboard shows comparable curves regardless of backend. A single
first-party module owns the mapping; two thin interceptors call it. **No edits to
the vendored submodules** (`third_party/Megatron-LM`, `third_party/torchtitan`).

---

## Problem

The two backends already share a W&B **run name** (`src/utils/wandb_naming.py`:
`[megatron] …` / `[torchtitan] …`, same canonical base) and one project, so runs
land side-by-side on one dashboard. But the **per-metric keys** are completely
divergent, so the curves do not overlay:

| Concept | Megatron emits | Torchtitan emits |
|---|---|---|
| training loss | `lm loss` | `loss_metrics/global_avg_loss` |
| learning rate | `learning-rate` | `lr` |
| grad norm (raw) | `grad-norm` | `grad_norm` |
| tokens seen | `tokens seen` (patch, **off** in adam) | `n_tokens_seen` |
| step time | `iteration-time` (seconds) | `time_metrics/end_to_end(s)` |
| throughput | `throughput` = **TFLOP/s/GPU** | `throughput(tps)` = **tokens/s** |
| max loss / mfu / tflops / mem | mostly absent in W&B | `loss_metrics/global_max_loss`, `mfu(%)`, `tflops`, `memory/*` |
| val loss | `lm loss validation` | `validation_metrics/loss` |

Naming differs by separator (space vs `-` vs `_`), namespace (flat vs
`loss_metrics/`-style), and inline units (`(tps)`, `(%)`). Worse, `throughput`
names the **same word for two different physical quantities** — a naive
name-merge would plot TFLOP/s against tokens/s on one axis.

### Where logging is wired today (the hook points already exist)

- **Megatron** logs from `training_log()` in
  `third_party/Megatron-LM/megatron/training/training.py`, which calls
  `get_wandb_writer().log({...}, iteration)` per metric. slm-research already
  monkey-patches this surface via the SHA-keyed patch registry
  (`src/patches/_registry.py`): `log_grad_norm_extra` wraps `training_log`,
  `training_log_eta` wraps `print_rank_last`, and `training_log_wandb_tokens_seen`
  adds `tokens seen` (registered but **not** in the `optim/adam` patch list).
- **Torchtitan** logs from `MetricsProcessor.log()` in
  `third_party/torchtitan/torchtitan/components/metrics.py`, which assembles the
  metrics dict and hands it to `WandBLogger.log({...}, step)` →
  `wandb.log(...)`. slm-research already wraps `MetricsProcessor.log` from
  `src/titan_ext/metrics.py` (rank-0-only console + ETA), loaded via torchtitan's
  `experimental.custom_import` hook.

Both backends step by `iteration`, so the x-axis is already consistent.

### Verified facts (load-bearing for the mapping)

1. Megatron's W&B `iteration-time` scalar is in **seconds**
   (`elapsed_time_per_iteration`); only the *stdout* line multiplies by 1000 for
   ms. So it matches torchtitan's `time_metrics/end_to_end(s)` with **no unit
   conversion**. (`training.py:2195,2208,2210,2219`)
2. Megatron's W&B `throughput` =
   `num_floating_point_operations / (elapsed_time_per_iteration · 1e12 ·
   world_size)` = **TFLOP/s/GPU**, gated on `log_throughput +
   log_timers_to_tensorboard`. It is **not** tokens/sec.
   (`training.py:2197,2221-2227`)
3. `iteration-time` / `throughput` are gated on `log_timers_to_tensorboard`;
   they may be absent. The interceptor must not depend on them being present.
4. The `optim/adam` experiment (used by both target scripts) enables patches
   `[model_unfuse_linears, training_log_eta, log_grad_norm_extra]` — so today the
   Megatron adam run logs **no** `tokens seen`, and `grad-norm-clipped` /
   `grad-norm-clip-coeff` (extra grad metrics) are present.

---

## Goals / non-goals

**Goals**
- One neutral, namespaced canonical schema; both backends remapped onto it.
- The mapping lives in exactly **one** first-party module, unit-testable on CPU.
- Core overlapping metrics are **replaced/remapped** (one curve per concept, no
  native duplicate). Backend-specific extras pass through untouched.
- No vendored-submodule edits; reuse the existing patch/`titan_ext` hook points.
- Logging never crashes training (every interceptor degrades to the original
  `.log` on error).

**Non-goals**
- TensorBoard key normalization (W&B only; the same `normalize()` is reusable
  later as a trivial follow-up).
- Cross-filling every missing metric (e.g. computing MFU/TFLOPs for Megatron, or
  `params-norm` for torchtitan). The one small computed value we add is described
  in §3.
- Changing run names, project, entity, or the x-axis (already unified).
- The aspirational 3-backend `UNIFIED_LOGGER_SPEC.md` (W&B+TB+Aim fan-out) — out
  of scope; this is a focused key-normalization layer.

---

## 1. Canonical schema (neutral, namespaced)

| Canonical key | Meaning | Unit |
|---|---|---|
| `train/loss` | training loss (mean) | nats |
| `train/loss_max` | max micro-batch loss (titan-sourced) | nats |
| `train/lr` | learning rate | — |
| `train/grad_norm` | raw (pre-clip) gradient norm | — |
| `train/tokens_seen` | cumulative tokens consumed | tokens |
| `perf/step_time_s` | wall-time per iteration | seconds |
| `val/loss` | validation loss | nats |

`CORE_CANONICAL` (the set both backends must converge on) =
`{train/loss, train/lr, train/grad_norm, train/tokens_seen, perf/step_time_s,
val/loss}`. `train/loss_max` is canonical but not required on both.

**Throughput is intentionally excluded from the canonical set** (see the §2 note):
the two backends compute tokens/sec with incompatible normalization, so each keeps
its native throughput key as a passthrough.

Everything outside the canonical set is **passthrough** (logged under its native
key, unchanged):
- Megatron: `num-zeros`, `params-norm`, `loss-scale`, `world-size`,
  `batch-size`, `samples vs steps`, `grad-norm-clipped`,
  `grad-norm-clip-coeff`, `throughput` (TFLOP/s/GPU), `iteration-time`,
  MoE/MTP losses if present.
- Torchtitan: `throughput(tps)`, `mfu(%)`, `tflops`, `memory/*`,
  `time_metrics/data_loading(s)`, `time_metrics/data_loading(%)`.

---

## 2. Mapping table

| Canonical | Megatron source key | Torchtitan source key |
|---|---|---|
| `train/loss` | `lm loss` | `loss_metrics/global_avg_loss` |
| `train/loss_max` | — | `loss_metrics/global_max_loss` |
| `train/lr` | `learning-rate` | `lr` |
| `train/grad_norm` | `grad-norm` | `grad_norm` |
| `train/tokens_seen` | computed (see §3) | `n_tokens_seen` |
| `perf/step_time_s` | computed (see §3) | `time_metrics/end_to_end(s)` |
| `val/loss` | `lm loss validation` | `validation_metrics/loss` |

**Throughput is deliberately not in the table** (and not mapped on either side).
Megatron's `throughput` is **TFLOP/s/GPU**. Torchtitan's `throughput(tps)` is
tokens/sec **normalized by `non_data_parallel_size`** (a per-model-parallel-group
rate, `metrics.py:422-423`), while a Megatron-computed tokens/sec would be the
**global aggregate** — these differ by roughly the data-parallel degree, so a
shared key would overlay two curves a large constant factor apart. Each backend's
native throughput passes through unchanged; `perf/step_time_s` (plain
wall-seconds per iteration, parallelism-independent) is the comparable perf
metric. Megatron's `iteration-time`/`throughput` aren't even logged to W&B in our
runs (no `--log-timers-to-tensorboard`), which is why `perf/step_time_s` is
computed rather than renamed.

The `lm loss` source key is the loss-dict key Megatron uses for GPT pretraining
(`pretrain_gpt.py`); the validation key is `f"{key} validation{suffix}"`. The
mapper keys off the leading `lm loss` token so the validation variant routes to
`val/loss`.

---

## 3. Computed Megatron metrics (the only "fill")

Megatron does not natively emit cumulative tokens or per-step wall time to W&B in
the adam path (no `--log-timers-to-tensorboard`). Rather than enable and re-map
the legacy `tokens seen` patch, the Megatron interceptor computes both inside one
`training_log` wrap (it already has `get_args()` in scope), gated on the **same**
interval as the native metrics (`tensorboard_log_interval or log_interval`) so
they log at matching cadence:

- `train/tokens_seen = args.consumed_train_samples × args.seq_length`
- `perf/step_time_s = Δwall_time / Δiteration` over the window between W&B-log
  points (a `time.perf_counter()` delta tracked in patch-local state).

Both are guarded; any failure skips the extra metrics and never touches the
original call. Throughput is intentionally **not** computed here (see §2 — a
global-aggregate tokens/sec is not comparable to torchtitan's
`non_data_parallel_size`-normalized `throughput(tps)`).

---

## 4. Components

### 4.1 `src/utils/wandb_metrics.py` (new — pure, no torch/wandb import)
Single source of truth for the schema and mapping.
- Constants: `MEGATRON_TO_CANONICAL`, `TITAN_TO_CANONICAL` (dicts),
  `CORE_CANONICAL` (frozenset). (Units live in the WANDB_SETUP.md doc table, not
  in code.)
- `normalize(metrics: dict[str, float], backend: str) -> dict[str, float]`:
  for each input key, if it (or its `lm loss`-prefix form) is in the backend's
  map, emit under the canonical key and **drop** the native key; otherwise pass
  the key through unchanged. Returns a new dict; pure and idempotent on already
  -canonical input.
- No imports beyond stdlib so it loads in any CPU/unit-test env.

### 4.2 Megatron interceptor — `src/patches/wandb_metric_normalize.py` (new patch)
- Registered via `@register_patch(name="wandb_metric_normalize", targets=())`
  (empty targets so it composes with `log_grad_norm_extra`, which owns
  `training.training_log` — same pattern the existing tokens-seen patch uses).
- `apply()` wraps **`wandb.log`**: Megatron's W&B writer *is* the `wandb` module
  (`global_vars.py:254` sets `_GLOBAL_WANDB_WRITER = wandb`), so every
  `wandb_writer.log(dict, step)` is literally `wandb.log(...)`. The wrapper runs
  `normalize(d, "megatron")` before delegating. This catches *all* callers — core
  `training_log` metrics, validation, and the `grad-norm-clipped` extras (which
  pass through). Guarded by an idempotent flag on the wrapped `wandb.log`.
- `apply()` also wraps `training_log` to additively emit the two computed metrics
  from §3, gated to `iteration % (tensorboard_log_interval or log_interval) == 0`
  (the cadence the native W&B block uses), on the W&B-logging rank only.
- Added to the `patches:` list of `configs/experiments/optim/adam.yaml` (and
  `champion.yaml`, kept in sync per that file's contract). Other experiments can
  opt in the same way.

### 4.3 Torchtitan interceptor — extend `src/titan_ext/metrics.py`
- New `apply_titan_wandb_normalize()` wraps
  `torchtitan.components.metrics.WandBLogger.log(metrics, step)` to run
  `normalize(metrics, "torchtitan")` before the upstream body (which then applies
  any `tag/` prefix and calls `wandb.log`). Idempotent guard flag, mirroring the
  existing `_WRAP_FLAG` pattern. Import-safe on CPU (no-op if torchtitan absent).
- Called from the same place `apply_titan_metrics_patch()` is invoked when
  `src.titan_ext` is imported via `experimental.custom_import`.

---

## 5. Data flow

```
Megatron:
  training_log
    ├─ wandb.log({...}, it) ──► [wrapped] normalize(d,"megatron") ──► wandb.log(canonical)
    └─ [interceptor adds] train/tokens_seen, perf/step_time_s  (native cadence)

Torchtitan:
  MetricsProcessor.log → WandBLogger.log({...}, step) ──► normalize(d,"torchtitan") ──► wandb.log(canonical)
```

Both step by `iteration` → x-axis already consistent. `train/tokens_seen` present
on both → a token-based x-axis works for cross-backend overlay.

---

## 6. Error handling

- Each interceptor wraps its transform + delegate in `try/except`; on any
  exception it logs once to stderr and calls the **original** `.log` with the
  untransformed dict. Training correctness is never affected (matches the
  invariant in every existing `src/patches/*` and `titan_ext` wrapper).
- The mapper itself never raises on unknown keys (passthrough) or empty dicts.

---

## 7. Testing

Pure CPU unit tests (no GPU — operator runs the GPU smoke separately):
- `tests/unit/test_wandb_metrics.py`:
  - golden mapping: a representative Megatron dict and a representative
    torchtitan dict each normalize to the expected canonical keys;
  - unknown/native keys pass through untouched;
  - `normalize` is idempotent on already-canonical input;
  - **guard:** neither backend's throughput is normalized — Megatron
    `{"throughput": x}` and torchtitan `{"throughput(tps)": x}` both pass through
    unchanged;
  - both backends' core outputs are exactly `CORE_CANONICAL` (minus computed
    extras, which are added by the interceptor not the mapper).
- A patch-registry test asserting `wandb_metric_normalize` registers and composes
  with `log_grad_norm_extra` without `PatchConflict`.
- Interceptor behavior (writer-wrap remaps keys; titan WandBLogger-wrap remaps
  keys) tested with a fake writer/logger object — no real wandb, no GPU.

Operator-run validation (documented, not run here): a short 2-backend smoke;
confirm `train/loss`, `train/lr`, `train/grad_norm`, `train/tokens_seen`,
`perf/step_time_s` appear under identical keys for both `[megatron]` and
`[torchtitan]` runs, and the curves overlay.

---

## 8. Deliverables

1. `src/utils/wandb_metrics.py` — schema + `normalize()` (single source of truth).
2. `src/patches/wandb_metric_normalize.py` — Megatron interceptor patch.
3. `src/titan_ext/metrics.py` — `apply_titan_wandb_normalize()` added + wired.
4. `configs/experiments/optim/adam.yaml` (+ `champion.yaml`) — patch list updated.
5. Tests in §7.
6. `CHANGELOG.md` entry under Unreleased.
7. Short note in `docs/` (or `WANDB_SETUP.md`) documenting the canonical schema
   table so dashboard authors know the keys.

### Out of scope (explicit)
- TensorBoard normalization, MFU/TFLOPs cross-fill, Aim/multi-backend fan-out,
  run-name/x-axis changes (already unified).
