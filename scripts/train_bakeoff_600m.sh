#!/usr/bin/env bash
set -euo pipefail

# Architecture-family bake-off at the 600M non-embedding budget
# (docs/experiments/arch_bakeoff_600m.md). One run per family; everything
# except base/family + base/scale is identical across runs.
#
# Usage:
#   bash scripts/train_bakeoff_600m.sh <family> [overrides...]
#   family ∈ {qwen3, deepseek_v3, qwen3_next, nemotron_h}
# Examples:
#   bash scripts/train_bakeoff_600m.sh deepseek_v3 cluster=h100_de
#   bash scripts/train_bakeoff_600m.sh nemotron_h cluster=h100_de training.micro_batch_size=8
#
# micro_batch_size: ablation_40x leaves it null, which megatron_args derives to
# min(64, gbs)=64 -> OOMs at the first forward on 80GB H100 (seq 4096, tp=1).
# Default to a value that fits every family; override via MICRO_BATCH_SIZE=... or
# a trailing training.micro_batch_size=N (the latter wins, last override = winner).
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

FAMILY="${1:?usage: train_bakeoff_600m.sh <family> [overrides...]}"
shift
case "$FAMILY" in
  qwen3)       SCALE="600m" ;;            # dense control (existing dev rung)
  deepseek_v3) SCALE="600m_deepseek_v3" ;;
  qwen3_next)  SCALE="600m_qwen3_next" ;;
  nemotron_h)  SCALE="600m_nemotron_h" ;;
  *) echo "unknown family: $FAMILY (qwen3|deepseek_v3|qwen3_next|nemotron_h)" >&2; exit 1 ;;
esac

python -m launchers.train_megatron \
  "base/family=$FAMILY" \
  "base/scale=$SCALE" \
  "experiment=optim/adam" \
  "training_regime=ablation_40x" \
  "scheduler=wsd" \
  "seed=42" \
  "training.micro_batch_size=$MICRO_BATCH_SIZE" \
  "$@"
