# DeepSeek training data-prep — design

**Date:** 2026-05-28
**Status:** approved-pending-review
**Scope:** Add a DeepSeek-flavored data-prep pipeline to `slm-research/tools/` that produces
Megatron mmap `.bin/.idx` tokenized with the **DeepSeek-V3 tokenizer**, byte-compatible with the
DeepSeek trainer in `Megatron-poet`.

## Problem

The DeepSeek training scripts live in `/lustre/fast/fast/zqiu/tmp/Megatron-poet`
(e.g. [`training_scripts/train_DeepSeek_3b.sh`](../../../../tmp/Megatron-poet/training_scripts/train_DeepSeek_3b.sh)).
They consume Megatron mmap `.bin/.idx` via `--data-path` (optionally through a `.list` blend file),
tokenized with `--tokenizer-type HuggingFaceTokenizer`.

The existing `slm-research/tools/` pipeline (`preprocess_nemotron_*`) already does
parquet → jsonl → tokenize, but tokenizes with **Llama-3.1-8B**. DeepSeek needs the same flow with
the tokenizer swapped to DeepSeek-V3, and — to eliminate format skew — tokenized with the **same
Megatron repo that trains** (Megatron-poet).

## Tokenizer — what the source actually requires

Traced through `Megatron-poet/megatron/training/tokenizer/tokenizer.py`:

- On the `HuggingFaceTokenizer` path, `args.vocab_size` is **never passed** to the tokenizer
  ([tokenizer.py:49-50](../../../../tmp/Megatron-poet/megatron/training/tokenizer/tokenizer.py#L49-L50));
  it is consumed only by `NullTokenizer` / `TikTokenizer` / `NullMultimodalTokenizer`. So the
  `--vocab-size 129280` in the shell script is a **no-op** for real training (it only bites in the
  `MOCK`/`NullTokenizer` branch).
- The model's vocab is derived from `len(tokenizer)`
  ([tokenizer.py:148-150](../../../../tmp/Megatron-poet/megatron/training/tokenizer/tokenizer.py#L148-L150)),
  then padded to a multiple of `make_vocab_size_divisible_by × TP`
  ([tokenizer.py:109-121](../../../../tmp/Megatron-poet/megatron/training/tokenizer/tokenizer.py#L109-L121)).
- The YAML hardcodes [`--make-vocab-size-divisible-by: 3232`](../../../../tmp/Megatron-poet/training_scripts/model_args/DeepSeek-3B.yaml#L34).
  `3232 × 40 = 129280` = the official DeepSeek-V3 `config.json` vocab. The divisor is
  reverse-engineered for the DeepSeek-V3 tokenizer specifically.

**Verified empirically** (downloaded tokenizer, ran the real code path):
`len(DeepSeek-V3 tokenizer) = 128815` → padded with divisor `3232` (TP=1 and TP=2) = **129280**.
`LlamaTokenizerFast`, **EOS id = 1** (`<｜end▁of▁sentence｜>`), BOS id = 0.

**The only hard requirement the code imposes:** preprocess the `.bin/.idx` with the *exact same* HF
tokenizer used for training. Nothing hardcodes "DeepSeek"; the configs are tuned for DeepSeek-V3.

### Tokenizer asset (already done)

Downloaded **tokenizer files only** (no weights) via `snapshot_download(..., allow_patterns=
["tokenizer.json","tokenizer_config.json"])` into:

`/lustre/fast/fast/zqiu/hf_models/DeepSeek-V3-tokenizer`  (≈7.85 MB total)

## Design

### New files (in `slm-research/tools/`)

1. **`preprocess_deepseek_tokenize.sh`** — core tokenize step. Invokes **Megatron-poet's**
   `tools/preprocess_data.py` (same `indexed_dataset` writer + `build_tokenizer` the trainer reads
   with → byte-compatible output). Args (all env-overridable):
   - `--tokenizer-type HuggingFaceTokenizer`
   - `--tokenizer-model` ← default `/lustre/fast/fast/zqiu/hf_models/DeepSeek-V3-tokenizer`
   - `--append-eod` (writes EOS id 1 as the doc separator)
   - `--workers` (default 8), `--partitions` (default **1** = one process writes the single
     `_text_document.{bin,idx}` directly, no merge / no temp partition files; raise only for
     very large corpora, and then `workers % partitions == 0`)
   - **no `--vocab-size`** (proven dead for HF tokenizer)
   - env: `MEGATRON_POET_ROOT` (default `/lustre/fast/fast/zqiu/tmp/Megatron-poet`),
     `INPUT_FILE`, `OUTPUT_PREFIX`, `TOKENIZER_MODEL`, `WORKERS`, `PARTITIONS`.

2. **`preprocess_deepseek_pipeline.sh`** — orchestrator. Flags: `--input-dir`,
   `--output-prefix` (or `--output-dir`), `--tokenizer-model`, `--workers`, `--partitions`,
   `--skip-stage {1|2|3}`. Stages:
   - **Stage 1** parquet → jsonl shards via existing `preprocess_parquet_to_jsonl.py`
     (auto-skipped if `--input-dir` already holds `.jsonl`).
   - **Stage 2** cat shards → one merged jsonl.
   - **Stage 3** tokenize merged jsonl → `<output-prefix>_text_document.bin/.idx` via file #1
     (default **partitions=1, workers=8** — single file written directly, matching the
     `zqiu24/Megatron-LM` fork's recipe; `--partitions>1` is available for very large corpora).

3. **`preprocess_deepseek_nemotron.sh`** — hardcoded one-command driver for the full
   Nemotron-CC-v2 corpus. Input is already one merged jsonl
   (`/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_full.jsonl`, 2.7 TB), so it skips
   stages 1-2 and calls file #1 directly (workers=8, partitions=1). Output follows the sibling
   naming convention → `nemotron_cc_v2_high_quality_deepseek_v3_tokenizer_text_document.{bin,idx}`
   (alongside the existing `_llama31_tokenizer` / `_qwen3_tokenizer` / `_qwen35_tokenizer` sets).

### Reused unchanged
- `preprocess_parquet_to_jsonl.py` (tokenizer-agnostic; sharded jsonl writer).

### Run environment
The DeepSeek conda env (`megatron-lm-014`, per `train_DeepSeek_3b.sh`), so `import megatron`
resolves for the poet preprocessor.

### Output / consumption
`<output-prefix>.bin/.idx` → trainer `--data-path <output-prefix>` (single dataset), or referenced
from a `.list` blend file.

## Data flow

```
parquet dir ──stage1──> jsonl shards ──stage2(cat)──> merged.jsonl
   ──stage3 (poet preprocess_data.py, HF DeepSeek-V3 tok, --append-eod, --partitions N)──>
      <prefix>.bin / <prefix>.idx  ──>  trainer --data-path
```

## Error handling
- `set -euo pipefail` in both scripts (matches existing tools).
- Stage 3 fails fast if `--tokenizer-model` dir or `MEGATRON_POET_ROOT/tools/preprocess_data.py`
  is missing.
- Stage 1 auto-detects parquet vs pre-existing jsonl input.

## Testing / verification
- Tokenizer load + padding math: **already verified** (len 128815 → 129280).
- Smoke: run the pipeline on a tiny parquet/jsonl slice, then read back the `.idx` header and
  decode a few sequences with the DeepSeek-V3 tokenizer to confirm round-trip + EOD separators.
- (Compute is run by the user; harness has no training env.)

## Out of scope (YAGNI)
- Auto-generating `.list` blend weights (single-dataset `--data-path` works directly).
- A separate per-shard + `merge_datasets.py` mode. Noted as the path for very large (2 TB-scale)
  corpora to avoid one giant merged jsonl; **not** built now.
</content>
</invoke>
