# Delta-W Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a flag-gated monitor that logs the actual materialized weight displacement `delta_W = W_after - W_before` for selected transformer linear weights in Adam, Muon, and POET runs. This replaces the current ambiguous comparison between dense weight-gradient spectra and POET's packed `delta_Q`; the comparable object is the realized movement in weight space.

**Architecture:** Create a new patch named `weight_delta_monitor` that wraps `megatron.training.training.train_step` as an outer wrapper. On a logging step it snapshots selected 2-D weights before calling the inner `train_step`, snapshots them again after the inner call returns, and logs metrics on `W_after - W_before`. For Adam and Muon this is the optimizer-applied dense parameter update. For POET with `merge_period=1`, the after-snapshot is post-merge, so `module.weight` is the materialized effective weight after the induced rotation has been folded into `W`. The monitor is inert unless `--log-delta-w` is set.

**Tech Stack:** Python, PyTorch, Megatron-LM (vendored), OmegaConf configs, W&B, pytest. CPU test interpreter: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`.

**Key design facts:**
- `grad_conditioning` logs `grad_cond/*` on raw 2-D `dL/dW`, and `grad_update/*` on a Newton-Schulz transform of that gradient, before the optimizer consumes it ([grad_conditioning.py](/lustre/fast/fast/zqiu/slm-research/src/patches/grad_conditioning.py#L73)). Those are signal/update-direction diagnostics, not realized parameter displacements.
- `block_spectral_stats` already supports ordinary 2-D matrices by adding a batch dimension ([skew_conditioning.py](/lustre/fast/fast/zqiu/slm-research/src/diag/skew_conditioning.py#L27)). Reuse it for `delta_W` spectra, with an explicit zero-delta guard so stable/effective rank do not become `NaN`.
- Patch registry applies patches in sorted order; later wrappers are outer wrappers ([\_registry.py](/lustre/fast/fast/zqiu/slm-research/src/patches/_registry.py#L88)). `weight_delta_monitor` sorts after `poet_merge_step`, so its after-snapshot happens after POET fold/reset.
- `poet_merge_step` folds POET state into base weights after the inner `train_step` returns ([poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L220)). Wrapping `optimizer.step` directly would miss this post-optimizer materialization.
- The existing weight-norm monitor already contains target-layer classification, layer selection, post-merge wrapper mechanics, and W&B rank gating that can be reused or mirrored ([weight_norm_monitor.py](/lustre/fast/fast/zqiu/slm-research/src/patches/weight_norm_monitor.py#L224)).
- Scope v1 to tensor-parallel size 1, matching current POET dev runs. For TP>1, log local shards with an explicit warning or skip until a gather-aware version is designed.

---

## File Structure

- **Create** `src/patches/weight_delta_monitor.py` - pure helpers, snapshot collection, `delta_W` metric computation, W&B logging, and train-step wrapper.
- **Create** `tests/unit/test_patch_weight_delta_monitor.py` - CPU tests for layer selection reuse, zero/rank metric behavior, wrapper before/after semantics, POET cadence, and registry registration.
- **Modify** `launchers/pretrain_gpt_slm.py` - add CLI flags and add `weight_delta_monitor` to `_ALWAYS_ON_PATCHES`.
- **Modify** `src/utils/megatron_args.py` - emit delta-W monitor flags from the YAML `training` config.
- **Modify** `tests/unit/test_megatron_args.py` - assert the YAML-to-CLI plumbing.
- **Optionally modify** `configs/base/training/*.yaml` or experiment configs only if a default diagnostic profile is desired. Do not enable by default in normal training configs.

---

## Task 1: CLI and Config Plumbing

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py`
- Modify: `src/utils/megatron_args.py`
- Modify: `tests/unit/test_megatron_args.py`

- [ ] Add argparse flags:
  - `--log-delta-w`
  - `--log-delta-w-interval`, default `250` or `500`
  - `--delta-w-layers`, default `first,mid,last`
  - `--delta-w-max-targets`, default `0` meaning no cap after layer selection
- [ ] Add `_delta_w_args(training)` next to `_weight_norm_args(training)` in `src/utils/megatron_args.py`.
- [ ] Emit the flags only when `training.log_delta_w: true` is set. Keep this monitor off by default because it copies real weights for before/after snapshots.
- [ ] Add tests that a minimal training config emits the expected flags and that the default config emits nothing.

**Acceptance:**
- `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "delta_w or weight_norm" -v`

---

## Task 2: Pure Delta-W Metric Helpers

**Files:**
- Create: `src/patches/weight_delta_monitor.py`
- Create: `tests/unit/test_patch_weight_delta_monitor.py`

- [ ] Implement `compute_delta_w_stats(before, after, eps=1e-12) -> dict[str, float]`.
- [ ] Compute and log:
  - `fro_abs = ||delta_W||_F`
  - `fro_rel = ||delta_W||_F / max(||W_before||_F, eps)`
  - `w_fro_before`, `w_fro_after`, `w_fro_ratio`
  - `cos_to_w = <delta_W, W_before> / (||delta_W||_F * ||W_before||_F)`
  - `row_rms_delta_mean = mean(||delta_W[i, :]||_2 / sqrt(in))`
  - `col_rms_delta_mean = mean(||delta_W[:, j]||_2 / sqrt(out))`
  - `stable_rank`, `effective_rank`, `condition_number`, `sigma_max_over_median`
  - `stable_rank_frac` and `effective_rank_frac`, divided by `min(out, in)`
- [ ] Add a zero-delta branch that returns finite zeros for delta-only metrics and leaves `w_fro_ratio=1` when `before == after`.
- [ ] Test known cases:
  - identical matrices produce `fro_abs=0`, `fro_rel=0`, finite rank metrics.
  - rank-1 delta has stable/effective rank approximately 1.
  - identity delta has stable/effective rank equal to matrix size.
  - positive and negative radial deltas give positive/negative `cos_to_w`.

**Acceptance:**
- `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_delta_monitor.py -k "stats" -v`

---

## Task 3: Target Collection and Snapshot Lifecycle

**Files:**
- Create: `src/patches/weight_delta_monitor.py`
- Modify or import from: `src/patches/weight_norm_monitor.py`

- [ ] Reuse or mirror `parse_layer_selection`, `classify_linear`, and `collect_target_weights` from `weight_norm_monitor` so the same matrices are tracked across `weightnorm/*` and `deltaw/*`.
- [ ] Implement `snapshot_target_weights(model, selected_layers, max_targets=0)` returning stable target records with:
  - layer index
  - matrix type
  - module name
  - CPU `float32` clone of `module.weight`
  - original shape
- [ ] Compute snapshots under `torch.no_grad()` and immediately move clones to CPU to avoid persistent GPU memory pressure.
- [ ] Match before/after records by module name, layer, type, and shape. If a target is missing or changes shape, skip that target and emit a one-time warning.
- [ ] Keep W&B keys compact:
  - `deltaw/L{layer}/{type}/{metric}`
  - `deltaw/_mean/{metric}` for scalar means across logged targets
- [ ] Do not log `delta_Q` in this patch. It is a parameter-space sidecar for POET and is not comparable to Adam/Muon weight displacement.

**Acceptance:**
- Unit test with a fake model where one linear weight is mutated by the wrapped function and the logged `deltaw/*/fro_abs` equals the known mutation norm.

---

## Task 4: Train-Step Wrapper and POET Cadence

**Files:**
- Create: `src/patches/weight_delta_monitor.py`
- Modify: `launchers/pretrain_gpt_slm.py`

- [ ] Register the patch as `@register_patch(name="weight_delta_monitor", targets=())`.
- [ ] Add `weight_delta_monitor` to `_ALWAYS_ON_PATCHES` so the flag is available for Adam, Muon, and POET without changing each experiment's patch list.
- [ ] Build the wrapper around `train_step`:
  - read `iteration` from `kwargs`, positional Megatron args, or `opts.iteration`
  - if not enabled or not on cadence, call through without snapshots
  - on cadence, snapshot before weights
  - call the inner `train_step`
  - snapshot after weights
  - log `delta_W` metrics
- [ ] Preserve the post-step logging order: snapshot after the inner wrapper returns. This is what captures POET's folded effective weight.
- [ ] POET behavior:
  - `merge_period=1`: log on normal cadence.
  - `merge_period>1`: log only when `iteration % merge_period == 0` and warn once if the requested interval is not a multiple of `merge_period`.
  - `merge_period<=0`: warn once and skip, because the base weight is frozen and `W_eff` is not materialized.
- [ ] Dense Adam/Muon behavior:
  - log every `log_delta_w_interval` steps.
  - no special handling for optimizer internals; the monitor only observes `W_after - W_before`.

**Acceptance:**
- Unit test confirms wrapper call order: `before` snapshot sees the original weight, inner function mutates it, and `after` snapshot sees the mutation.
- Unit test confirms `weight_delta_monitor` sorts after `poet_merge_step`.

---

## Task 5: Run Validation

**Files:**
- Modify tests only unless bugs are found.

- [ ] Run CPU checks:
  - `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_delta_monitor.py tests/unit/test_megatron_args.py -k "delta_w or weight_norm" -v`
  - `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/weight_delta_monitor.py`
- [ ] Do not launch GPU training from the implementation task. Hand off exact run commands because this monitor is meant to be validated on real Adam/Muon/POET runs.
- [ ] Suggested POET validation command shape:

```bash
SLM_POET_GRAD_CONDITIONING=1 \
SLM_GRAD_CONDITIONING=1 \
SLM_POET_COORD_DIAG=1 \
SLM_POET_WSPLIT=1 \
bash scripts/train_poet_lie_orth_alt.sh llama3 \
  training.log_delta_w=true \
  training.log_delta_w_interval=250 \
  training.delta_w_layers=first,mid,last
```

- [ ] Suggested Adam validation: run `bash scripts/train_adam_dev.sh llama3 training.log_delta_w=true training.log_delta_w_interval=250 training.delta_w_layers=first,mid,last` with the same data seed and token budget.
- [ ] Suggested Muon validation: run `bash scripts/train_muon_dev.sh llama3 training.log_delta_w=true training.log_delta_w_interval=250 training.delta_w_layers=first,mid,last` with the same data seed and token budget.

**Analysis checklist for the first real runs:**
- Compare `deltaw/*/stable_rank` and `deltaw/*/effective_rank` across Adam, Muon, POET. These are the actual update spectra.
- Compare `deltaw/*/fro_rel` across optimizers. A much larger POET value would explain fast weight-norm growth even if the Q-space update looks small.
- Compare `deltaw/*/cos_to_w` and `w_fro_ratio`. Sustained positive radial components mean the step itself increases weight norms; near-zero radial components shift suspicion to merge/materialization drift or other code paths.
- Treat `grad_cond/*`, `poet_cond/*`, and `poet_update/*` as upstream signal diagnostics only. The optimizer-comparable monitor is `deltaw/*`.

---

## Done Definition

- `deltaw/*` appears in W&B for Adam, Muon, and POET when `training.log_delta_w=true`.
- Logged metrics are finite on zero updates and small/dev runs.
- POET logs are taken after fold/reset on merge-boundary steps.
- CPU unit tests pass.
- No default training config pays snapshot overhead unless explicitly enabled.
