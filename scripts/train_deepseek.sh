#!/usr/bin/env bash
set -euo pipefail

# First-party DeepSeek-3Bv2 (MQA + sandwich-norm) training, de-vendored from
# poet_torch_huawei. Defaults to plain AdamW, under which the sandwich_norm_apply
# patch is active (it is wired into optim/adam and optim/muon_hybrid). POET is
# deferred: optim/poet does not yet carry sandwich_norm_apply (its
# poet_unfuse_te_impl patch already owns the core_transformer_config_from_args
# target), so do NOT expect sandwich-norm under experiment=optim/poet yet.
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
