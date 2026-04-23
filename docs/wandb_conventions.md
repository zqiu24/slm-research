# W&B conventions

Authoritative source: SPEC.md §8. Summary:

## Projects (one entity: `neckariumai-research`)

| Project | Purpose |
|---|---|
| `pretrain-ablations-300m` / `-600m` / `-1_2b` / `-2_4b` / `-7b` | Per-scale ablation runs |
| `pretrain-champion` | Current baseline, rerun at promotion |
| `pretrain-final-1_2b`, `pretrain-final-2_4b` | Final overtrained deliverables |
| `sandbox-<username>` | Personal debug/scratch — **required** for in-progress work |

## Per-run fields

- `group` = `config_hash` (16 hex chars). Seeds of the same config share a group.
- `job_type` ∈ `{ablation, promotion_gate, extrapolation, final, champion_baseline, sandbox}`
- **Tags:** `person:<user>`, `family:<ablation-family>`, `base_family:<llama3|qwen3|...>`,
  `scale:<300m|600m|1_2b|2_4b|7b>`, `cluster:<h800_cn|h100_de|a100_de|b200_de|hpc_de>`,
  `precision:<bf16|fp8|fp4>`, `status:<candidate|promoted|deprecated>`,
  `regime:<ablation_20x|...>`, `month:<YYYY-MM>`
- **Config fields:** `git_sha`, `megatron_sha`, `patch_set_hash`, `dataset_hash`,
  `config_hash`, **`config_diff_from_champion`** (primary comparison column),
  `experiment_summary`, `required_capabilities`, `achieved_parallelism`,
  `non_embedding_params`, `total_params`, `launch_timestamp_utc`

## Metric prefixes

- `train/*` — per-step training metrics
- `eval/<task>@<tokens>` — token-milestone evals, e.g. `eval/hellaswag@20B`
- `perf/*` — throughput, MFU
- `system/*` — GPU memory, utilization (optional)

## Offline mode

All jobs run with `WANDB_MODE=offline`. `tools/sync_wandb.py` pushes to the
self-hosted server; HPC compute nodes have no internet, so this is
mandatory and the same recipe applies everywhere for consistency.
