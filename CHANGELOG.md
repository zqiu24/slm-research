# Changelog

## Unreleased

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
