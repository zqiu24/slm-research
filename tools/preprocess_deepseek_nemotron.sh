#!/usr/bin/env bash
# Hardcoded DeepSeek-V3 tokenization of the full Nemotron-CC-v2 high-quality corpus.
#
# The input is already ONE merged jsonl, so this skips parquet->jsonl->cat and runs
# tokenization directly: workers=8, partitions=1 -> the single output file is written
# directly (no merge, no temp partition files).
#
# Output naming follows the siblings already in the dir
# (..._llama31_tokenizer / _qwen3_tokenizer / _qwen35_tokenizer), so the result is:
#   nemotron_cc_v2_high_quality_deepseek_v3_tokenizer_text_document.{bin,idx}
#
# Run under the DeepSeek trainer's conda env (e.g. `megatron-lm-014`) so `import megatron`
# resolves for Megatron-poet's preprocess_data.py.
set -euo pipefail

INPUT_FILE="/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_full.jsonl"
OUTPUT_PREFIX="/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_deepseek_v3_tokenizer"
TOKENIZER_MODEL="/lustre/fast/fast/zqiu/hf_models/DeepSeek-V3-tokenizer"
WORKERS="8"
PARTITIONS="1"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_FILE="$INPUT_FILE" \
OUTPUT_PREFIX="$OUTPUT_PREFIX" \
TOKENIZER_MODEL="$TOKENIZER_MODEL" \
WORKERS="$WORKERS" \
PARTITIONS="$PARTITIONS" \
  bash "$HERE/preprocess_deepseek_tokenize.sh"
