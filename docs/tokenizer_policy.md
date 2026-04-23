# Tokenizer policy

The pretraining dataset is tokenized exactly once with a single tokenizer.
All experiments — regardless of declared base family (Llama-3, Qwen-3, …)
— consume that frozen tokenization.

## Why

Tokenizer effects and vocab-size effects confound training-method effects.
Since the research question is about training algorithms and architecture,
not tokenization, we remove the variable.

## How it's enforced

1. `tools/freeze_dataset.py` pre-tokenizes the corpus and emits a manifest
   JSON with per-shard SHA256 hashes and a top-level `dataset_hash`.
2. At job start the launcher verifies every shard against the manifest
   (SPEC.md §6.1). Mismatches abort.
3. Family YAMLs (`configs/base/family/*.yaml`) declare a
   `tokenizer.nominal_name` and `tokenizer.nominal_vocab_size` — these are
   **descriptive only**. The runtime vocab size comes from the dataset
   manifest.

## Cross-family comparison implications

- `base.family` is included in the `config_hash`; different families →
  different hashes → curves not expected to match.
- The scaling-slope plot must be filtered to a single family; mixing
  Llama-3 and Qwen-3 points in one slope conflates method with family
  effects.
- If multiple families are active, the team maintains parallel champions
  (`champion_qwen3.yaml`, `champion_llama3.yaml`). At seed stage we
  recommend committing to one family.

## Bumping to a new tokenizer / dataset version

This is a major event — `dataset_hash` changes, so `config_hash` changes
for every experiment, so every prior run is effectively a different
ladder. Treat it as a ladder-reset, not a routine change.
