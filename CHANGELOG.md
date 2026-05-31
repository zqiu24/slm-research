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
