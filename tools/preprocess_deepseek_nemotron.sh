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
# Self-contained: sets up the env (CUDA-13.2 loader + slm_env venv) then tokenizes.
# The CUDA-13.2 loader is REQUIRED here: slm_env's transformer_engine references a
# libcublasLt.so.13 symbol that only the system cuBLAS exports, and Megatron-poet's
# megatron package imports transformer_engine eagerly at import time (even though
# tokenization itself is CPU-only). The loader's LD_PRELOAD fixes that.
# Set SKIP_ENV_SETUP=1 to skip this and use whatever env is already active.
set -euo pipefail

CUDA_ENV="${CUDA_ENV:-/lustre/fast/fast/zqiu/slm-research/load_cuda13_2_nccl_env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/fast/zqiu/slm_env/.venv/bin/activate}"
if [[ "${SKIP_ENV_SETUP:-0}" != "1" ]]; then
  # shellcheck disable=SC1090
  source "$CUDA_ENV"
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

INPUT_FILE="/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_full.jsonl"
OUTPUT_PREFIX="/lustre/scratch/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_deepseek_v3_tokenizer"
TOKENIZER_MODEL="/lustre/fast/fast/zqiu/hf_models/DeepSeek-V3-tokenizer"
WORKERS="${WORKERS:-16}"
PARTITIONS="${PARTITIONS:-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_FILE="$INPUT_FILE" \
OUTPUT_PREFIX="$OUTPUT_PREFIX" \
TOKENIZER_MODEL="$TOKENIZER_MODEL" \
WORKERS="$WORKERS" \
PARTITIONS="$PARTITIONS" \
  bash "$HERE/preprocess_deepseek_tokenize.sh"
