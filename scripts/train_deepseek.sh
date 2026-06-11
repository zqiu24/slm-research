#!/usr/bin/env bash
set -euo pipefail

# First-party DeepSeek-3Bv2 (MQA + sandwich-norm) training, de-vendored from
# poet_torch_huawei. Defaults to plain AdamW. sandwich_norm_apply is wired into
# optim/adam, optim/muon_hybrid, AND optim/poet, so it composes with POET:
#   bash scripts/train_deepseek.sh experiment=optim/poet ...
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

python -m launchers.train_megatron \
  "base/family=deepseek_v3_mqa" \
  "base/scale=deepseek_3bv2" \
  "cluster=h100_de" \
  "experiment=optim/adam" \
  "scheduler=wsd" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=4" \
  "optim.lr=8.6e-4" \
  "optim.min_lr=7e-6" \
  "$@"
