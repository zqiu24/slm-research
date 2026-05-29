#!/usr/bin/env bash
set -euo pipefail
# Thin entry for the vendored Huawei POET DeepSeek-3B stack (Megatron-core
# 0.14, isolated under poet_torch_huawei/). This deliberately does NOT use
# slm-research's hydra launcher (launchers.train_megatron) — the vendored
# scripts carry their own env activation + torchrun invocation.
#
#   scripts/train_poet_huawei.sh dev    # single-GPU mock smoke (block-size 128)
#   scripts/train_poet_huawei.sh full   # reference 8-GPU EP=8 DeepSeek-3B run
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUAWEI_SCRIPTS="$SLM_REPO/poet_torch_huawei/training_scripts"

MODE="${1:-dev}"
shift || true
case "${MODE}" in
  dev)
    exec bash "$HUAWEI_SCRIPTS/train_DeepSeek_dev_mock_1gpu.sh" "$@"
    ;;
  full)
    exec bash "$HUAWEI_SCRIPTS/train_DeepSeek_3bv3_sandwich_mqa_poet.sh" "$@"
    ;;
  *)
    echo "Usage: scripts/train_poet_huawei.sh [dev|full] [extra args]" >&2
    echo "  dev  - single-GPU mock-data smoke (block-size 128)" >&2
    echo "  full - reference 8-GPU EP=8 DeepSeek-3B POET run" >&2
    exit 2
    ;;
esac
