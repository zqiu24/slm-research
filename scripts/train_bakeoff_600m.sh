#!/usr/bin/env bash
set -euo pipefail

# Architecture-family bake-off at the 600M non-embedding budget
# (docs/experiments/arch_bakeoff_600m.md). One run per family; everything
# except base/family + base/scale is identical across runs.
#
# Usage:
#   bash scripts/train_bakeoff_600m.sh <family> [overrides...]
#   family ∈ {qwen3, deepseek_v3, deepseek_v3_dense, qwen3_next, nemotron_h,
#             llama3, minicpm, gemma3}
# Examples:
#   bash scripts/train_bakeoff_600m.sh deepseek_v3 cluster=h100_de
#   bash scripts/train_bakeoff_600m.sh nemotron_h cluster=h100_de training.micro_batch_size=8
#
# training_regime: fixed 12B-token budget (total_tokens, NOT tokens_per_param)
# so all four families train on the EXACT same token count and share one
# GPTDataset cache despite their slightly different non_embedding_params.
# Override via REGIME=fixed_50b etc. for a different fixed budget.
REGIME="${REGIME:-fixed_12b}"

# seq_length: default to 4096 — the protocol length from the bake-off design doc
# (docs/experiments/arch_bakeoff_600m.md) and the base pretraining length used by
# DeepSeek-V3 and Qwen3 (Nemotron-H uses 8K); long-context signal (Mamba/GDN/MLA)
# is faithfully exercised here. Override via SEQ_LENGTH=... or a trailing
# base.model.seq_length=N for cheaper/faster iteration.
SEQ_LENGTH="${SEQ_LENGTH:-4096}"

# micro_batch_size: ablation_40x leaves it null, which megatron_args derives to
# min(64, gbs)=64 -> OOMs at the first forward on 80GB H100 (seq 4096, tp=1).
# Default to 4, a conservative value that fits every family at seq 4096; raise
# via MICRO_BATCH_SIZE=... or a trailing training.micro_batch_size=N (the latter
# wins, last override = winner) if a family has headroom.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"

# SIZE: non-embedding budget rung; selects base/scale=<SIZE>_<family>. Defaults
# to 600m (the original bake-off budget). Set SIZE=1b to run the 1B dense rung
# (configs/base/scale/1b_{qwen3,llama3,minicpm,gemma3,deepseek_v3_dense}.yaml).
SIZE="${SIZE:-600m}"
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

FAMILY="${1:?usage: train_bakeoff_600m.sh <family> [overrides...]}"
shift
case "$FAMILY" in
  qwen3)             ;;      # dense GQA, budget-matched
  deepseek_v3)       ;;
  deepseek_v3_dense) ;;      # MLA + MTP, MoE off
  qwen3_next)        ;;
  nemotron_h)        ;;
  llama3)            ;;      # dense GQA (Llama-style)
  minicpm)           ;;      # dense GQA (MiniCPM, depth-scaled)
  gemma3)            ;;
  *) echo "unknown family: $FAMILY (qwen3|deepseek_v3|deepseek_v3_dense|qwen3_next|nemotron_h|llama3|minicpm|gemma3)" >&2; exit 1 ;;
esac
SCALE="${SIZE}_${FAMILY}"
if [[ ! -f "$SLM_REPO/configs/base/scale/$SCALE.yaml" ]]; then
  echo "no scale config for SIZE=$SIZE family=$FAMILY (expected configs/base/scale/$SCALE.yaml)" >&2
  exit 1
fi

python -m launchers.train_megatron \
  "base/family=$FAMILY" \
  "base/scale=$SCALE" \
  "experiment=optim/adam" \
  "training_regime=$REGIME" \
  "scheduler=wsd" \
  "seed=42" \
  "base.model.seq_length=$SEQ_LENGTH" \
  "training.micro_batch_size=$MICRO_BATCH_SIZE" \
  "$@"
