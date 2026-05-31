# Changelog

## Unreleased

### Fixed — POET merge loss spike

- **POET merge-and-reinitialize no longer spikes the loss every
  `poet_merge_period` steps** (`src/patches/poet_merge_step.py`). On Megatron's
  mixed-precision optimizers the merge folds the rotation into the frozen weight
  and zeros the **bf16 model** `oft_R`, but the optimizer steps the **fp32
  master** copy — which was left nonzero, so the next `optimizer.step()` copied
  the stale master back and re-applied the just-merged rotation a second time (a
  huge recurring spike the plain-PyTorch GaLore reference never has).
  `_reset_vanilla_oft_state` now also zeros the master *value* (not just the Adam
  moments), and discovers masters on **both** layouts via
  `_iter_model_master_pairs`: `float16_groups`/`fp32_from_float16_groups`
  (single-GPU `Float16OptimizerWithFloat16Params`) and
  `model_float16_groups`/`shard_fp32_from_float16_groups` (multi-GPU
  `DistributedOptimizer`, where the prior id-based reset silently matched 0
  params). Verified on 8-GPU `h100_de` and single-GPU `dev`: loss descends
  smoothly through every merge.

### Changed — POET run name carries oft_R scale

- **POET run names now include the oft_R LR multiplier** (`src/utils/wandb_naming.py`):
  `wandb_base_name` appends a `-scale<v>` segment (`optim.poet.scale`, formatted
  with `:g`) after the block parameterization, e.g.
  `poet-llama3-300m-lr0.001-bc4-scale0.05`, so POET scale sweeps are
  distinguishable on the W&B dashboard. Only the canonical W&B run name changes;
  the on-disk run-dir name (`exp-family-scale-s<seed>-<ts>`) is unchanged.

### Added — unified W&B logging

- **Unified W&B metric keys across backends** (`src/utils/wandb_metrics.py`):
  both backends now normalize their core training-health metrics onto one
  canonical schema (`train/loss`, `train/lr`, `train/grad_norm`,
  `train/tokens_seen`, `perf/step_time_s`, `val/loss`) so Megatron and torchtitan
  runs overlay on one dashboard. Applied via a registered Megatron patch
  (`wandb_metric_normalize`, wraps `wandb.log` **lazily inside the `training_log`
  wrapper** — `wandb.init()` rebinds the module-level `wandb.log`, so wrapping at
  patch-apply time was silently clobbered and left Megatron's native keys
  unrenamed; it also computes tokens_seen / step_time, which our runs don't log to
  W&B) and a torchtitan `WandBLogger.log` wrapper in `src/titan_ext/metrics.py`.
  Throughput is deliberately NOT normalized
  (Megatron's `throughput` is TFLOP/s/GPU; torchtitan's `throughput(tps)` is
  normalized by `non_data_parallel_size` — not comparable), so each backend keeps
  its native throughput as a passthrough. Backend-specific extras pass through
  unchanged; no vendored-submodule edits. Design + plan in
  `docs/superpowers/{specs,plans}/2026-05-31-unified-wandb-logging*.md`.
- **Eval/validation in W&B for both backends.** Added a derived `val/ppl`
  (`exp(min(20, val/loss))`, via `wandb_metrics.with_derived`) so eval perplexity
  shows alongside `val/loss` — Megatron logs val PPL only to TensorBoard and
  torchtitan didn't emit it. Enabled **torchtitan validation**: `torchtitan_args`
  emits a `[validation]` block mirroring the Megatron eval cadence
  (`eval_interval`→`freq`, `eval_iters`→`steps`), and `src/titan_ext` monkeypatches
  torchtitan's validation dataloader (`build_text_validation_dataloader`) to a
  **val split of the same Megatron-indexed `GPTDataset`** (same `data.split`), so
  torchtitan's `val/loss` is computed on the same held-out documents as Megatron's
  eval. The vendored `Validator` hardcodes a C4 raw-text loader with no TrainSpec
  hook, hence the monkeypatch (consistent with the existing `titan_ext` pattern).
  Also patched `BaseValidator.should_validate` to **drop torchtitan's step-1
  eval** (`step == 1 or step % freq == 0` → `step % freq == 0`): evaluating an
  untrained model at step 1 produced a huge first point that distorted the curve
  (Megatron only evals at `eval_interval` boundaries). First eval is now at `freq`.
- **`wandb_metric_normalize` enabled in every Megatron experiment**, not just
  adam/champion: added to `poet`, `arch/ngpt`, `muon_hybrid`, and the experiment
  `_template` (alongside the always-on `training_log_eta`). Previously
  `train_poet`/`train_ngpt`/`train_muon` logged native Megatron keys (`lm loss`,
  `learning-rate`, `grad-norm`) instead of the canonical `train/*` schema. The
  torchtitan side is already unconditional (applied on `titan_ext` import).
  Verified all five experiments' patch lists co-register without a `PatchConflict`
  (e.g. POET's `poet_merge_step` wraps `train_step`, `wandb_metric_normalize`
  wraps `training_log`).
- **Removed the `log_grad_norm_extra` patch** (and its `grad-norm-clipped` /
  `grad-norm-clip-coeff` W&B/TensorBoard scalars) from all experiments
  (adam, champion, poet, ngpt, muon_hybrid, template). It was a POET grad-norm
  debugging aid; the POET issue is resolved, so only the raw grad-norm
  (`train/grad_norm`, from Megatron's native `grad-norm`) is kept.

### Added — torchtitan training backend (M1–M2)

- Vendored **torchtitan v0.2.2** as `third_party/torchtitan` (pinned
  `73a0e6979`); pin + bump procedure in `docs/torchtitan_pin.md`; runtime deps in
  the `torchtitan` extra (`pyproject.toml`) and `install_slm_env.sh`.
- New first-class **`backend ∈ {megatron, torchtitan}`** config field (default
  `megatron`). torchtitan runs get a `-torchtitan-` run-name segment and record
  `torchtitan_sha` in the resolved config + launch metadata; Megatron run names
  stay byte-identical.
- `scripts/train_adam.sh --backend torchtitan` dispatches to the new
  `launchers/train_torchtitan.py`, which resolves the same six-axis config, emits
  `<run_dir>/torchtitan.toml`, and builds a `torchrun -m torchtitan.train`
  command. The non-AdamW wrappers (`train_muon/ngpt/poet`) reject
  `--backend torchtitan` with a clear message (AdamW-only in M1).
- `src/utils/torchtitan_args.py`: pure `cfg → (toml, overrides)` mapper (bf16
  AdamW + FSDP2 + WSD scheduler), with **warn-and-skip** for Megatron-only knobs
  (patches, sandwich-norm) so the same experiment yaml runs on either backend.
- `src/titan_ext/` runtime extension (loaded via torchtitan's
  `experimental.custom_import`, **no edits to the vendored submodule**): registers
  an `slm_<family>` TrainSpec cloned from torchtitan's native llama3/qwen3/
  deepseek_v3, adds an `slm_<scale>` flavor for the dense families, and swaps in a
  **Megatron-`GPTDataset`-backed dataloader** so torchtitan reads the same
  `.bin/.idx` corpus. Byte-level data parity vs the Megatron path is the M2 gate
  (`tests/integration/test_titan_megatron_data_parity.py`).
- W&B project/run-name routed via env so both backends share one dashboard.
- **M3 functional training-health gate** (`tests/numerics/test_titan_training_health.py`,
  `@pytest.mark.gpu`) + operator runbook
  (`docs/superpowers/runbooks/2026-05-30-torchtitan-training-health.md`) —
  per-family curves are pending an operator GPU run.
- API surface recorded in `docs/torchtitan_api_notes.md`; design + plan in
  `docs/superpowers/{specs,plans}/2026-05-30-torchtitan-backend*.md`.
- Emit `[comm].init_timeout_seconds` (default **3600s**, override via
  `cluster.comm_init_timeout_seconds`) so rank 0's cold dataset-index build
  doesn't trip torchtitan's 300s default and crash the other ranks at the
  startup barrier. The torchtitan path always cold-builds (different cache hash
  than Megatron), so the 5-min default was insufficient on real corpora.
- `collate_fn=_collate_megatron_to_titan` reshapes each Megatron `GPTDataset`
  sample-dict into torchtitan's required `(input_dict, labels)` 2-tuple
  (`{"input": tokens}`, pre-shifted `labels`), dropping the unused
  attention_mask/loss_mask/position_ids. Without it the first training step
  crashed in `batch_generator` with `too many values to unpack (expected 2)`.
- **Unified W&B run naming** (`src/utils/wandb_naming.py`): both backends now
  derive the same canonical name (`wandb_base_name`: `<exp>-<family>-<scale>-lr<lr>`
  + optimizer segments) and tag it with a `[megatron]` / `[torchtitan]` prefix via
  `wandb_run_name`. torchtitan's `WANDB_NAME` switched from the timestamped run-dir
  name to this shared name, so the two backends' W&B names differ **only** by the
  prefix and land side-by-side on one dashboard. (Run-*dir* names are unchanged.)
- **Megatron 300m aligned to the torchtitan model build** so a Megatron run
  reproduces the torchtitan training curve. torchtitan builds llama3 from its
  native flavor (ignores `ffn_hidden_size`, derives the FFN from `dim`; untied
  embeddings; depth-scaled init), so `configs/base/scale/300m.yaml` now sets
  `ffn_hidden_size: 2816` (matches torchtitan's `dim`-derived width),
  `tie_embeddings: false`, and a new `titan_init: true` flag. The init scheme has
  no Megatron config knob, so `src/model/titan_init.py` re-initializes the built
  model to torchtitan's exact per-weight recipe (std=1.0 embeddings, fixed-0.02
  fan-in / gate, per-layer depth-scaled output·up·down, `dim**-0.5` LM head) —
  wired in `launchers/pretrain_gpt_slm.py` as a post-build, pre-DDP
  `model_provider` wrapper (after unfuse), gated on the config, TP=PP=1, seeded
  for data-parallel-replica determinism. No-op on the torchtitan backend.
  Covered by `tests/unit/test_titan_init.py`.
- **torchtitan per-step console line now matches the Megatron path**
  (`src/titan_ext/metrics.py`, applied via the `custom_import` hook — no submodule
  edit): the `step: … loss: … grad_norm: … tps: … tflops: … mfu: …` line is
  emitted **only on rank 0** (torchtitan's `init_logger` otherwise duplicates it
  per rank) and gains an **`ETA: 1h30m`** segment derived from
  `training.steps` + just-elapsed wall time — same `HhMMm` format as the Megatron
  `training_log_eta` patch. Runtime-wraps `MetricsProcessor.log`, rebinding the
  metrics module logger to a rank-0/ETA proxy for the call; all upstream metric
  math and wandb/tb logging are unchanged. Helpers covered by
  `tests/unit/test_titan_metrics_patch.py`.
- **Fixed torchtitan data starvation** (`src/titan_ext/dataloader.py`): the
  `ParallelAwareDataloader` was built with no `num_workers`, so it defaulted to
  `0` (synchronous, main-process) and the GPU idled ~98% per step waiting for the
  next batch from the shared Megatron `GPTDataset` (mfu ~1.7%; ~5× slower wall
  clock than the Megatron path at 300m/seq256). Now passes `num_workers`
  (= `data.num_workers`, same value Megatron's `--num-workers` uses), `pin_memory`,
  `prefetch_factor`, and `persistent_workers`, so data loading overlaps compute.
  Worker prefetch does not change sample selection or order (same sampler), so the
  training curve is unaffected. Kwargs builder covered by
  `tests/unit/test_titan_dataloader_collate.py`.
