#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/High-Quality}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/High-Quality_jsonl}"
IDX="${1:-}"

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${IDX}" ]]; then
  python -m tools.preprocess_parquet_to_jsonl --input "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" --idx "${IDX}"
else
  python -m tools.preprocess_parquet_to_jsonl --input "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}"
fi
