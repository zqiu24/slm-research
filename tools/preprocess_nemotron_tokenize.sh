#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${INPUT_FILE:-/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_full.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_llama31_tokenizer}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B}"
WORKERS="${WORKERS:-8}"

python third_party/Megatron-LM/tools/preprocess_data.py \
  --input "${INPUT_FILE}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --tokenizer-type "${TOKENIZER_TYPE}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --workers "${WORKERS}" \
  --append-eod
