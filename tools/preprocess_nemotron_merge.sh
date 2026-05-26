#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/High-Quality_processed}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/Nemotron-CC-v2-High-Quality_merged}"

python third_party/Megatron-LM/tools/merge_datasets.py \
  --input "${INPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}"
