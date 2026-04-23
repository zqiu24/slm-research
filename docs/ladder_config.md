# Frozen ladder configuration

These values are **globally frozen** across the entire scale ladder and
every cluster. Changing any of them invalidates the ladder's scaling slope
and must be flagged as a separate experimental track (SPEC.md §1.3).

| Knob | Frozen value | Rationale |
|---|---|---|
| `training.global_batch_size_tokens` | `4_194_304` (4M) | Matches MiniCPM; reasonable across 300M–7B |
| `training.seq_length` | `4096` | Longer context is a separate research track |
| Tokenizer | single dataset tokenizer | Confound-removal; research is about the method |
| Dataset | hashed manifest at job start | Reproducibility anchor |

## Parameter-counting convention

The ladder is defined in **non-embedding parameters** (`base.non_embedding_params`).
Token budgets use this number: `training.total_tokens = tokens_per_param * non_embedding_params`.
Total params (including embeddings + LM head) are derived at launch and
logged, but do not drive ladder math.

## How the launcher enforces this

`launchers/submit.py` refuses to submit when `training_regime` overrides
`global_batch_size_tokens` or `seq_length` unless the caller passes
`--override-ladder-config` (which also forces `wandb.project` to be a
sandbox).

## Scale ladder

| Scale | `non_embedding_params` | Role | Tokens (ablation) |
|---|---:|---|---:|
| 300M  | 300_000_000   | Optional smoke / slope anchor | 6B @ 20x  |
| 600M  | 600_000_000   | Daily dev iteration           | 24B @ 40x |
| 1.2B  | 1_200_000_000 | Scale check + final deliverable | 24B @ 20x |
| 2.4B  | 2_400_000_000 | Promotion gate + final deliverable | 48B @ 20x |
| 7B    | 7_000_000_000 | HPC extrapolation anchor (champion only) | 140B @ 20x |
