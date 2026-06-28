# Data Preprocessing

Nemotron-CC-v2 preprocessing has two stages:

1. Convert parquet shards to JSONL with `tools/preprocess_nemotron_parquet_to_jsonl.sh`.
2. Tokenize JSONL into Megatron indexed dataset files with `tools/preprocess_nemotron_tokenize.sh`.

The default data config points at:

`/lustre/fast/fast/groups/ei-slm/Nemotron-CC-v2/nemotron_cc_v2_high_quality_text_document_llama31_8b`

Scratch variants are registered under `configs/data/`:

- `nemotron_cc_v2_scratch_llama31`
- `nemotron_cc_v2_scratch_qwen3`
- `nemotron_cc_v2_scratch_qwen35`

Use a registered dataset from training scripts with:

```bash
scripts/train_adam.sh llama3 data=nemotron_cc_v2_scratch_qwen3 --dry-run
```

The tokenizer policy remains: compare optimizer and architecture runs only
within one data config unless the experiment is explicitly about tokenizers.
