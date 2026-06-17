#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-3Bv2 (MQA + sandwich-norm) Muon baseline -- the optimizer-only
# counterpart to train_deepseek_poet.sh. SAME model / scheduler (wsd) / data /
# token budget / batch / parallelism; the ONLY change vs the POET run is the
# optimizer: experiment=optim/muon_hybrid instead of optim/poet_lie_orth_alt.
#
#   bash scripts/train_deepseek_muon.sh dev    [hydra overrides...]   # 1-GPU smoke
#   bash scripts/train_deepseek_muon.sh full   [hydra overrides...]   # 8-GPU cluster
#
# muon_hybrid = Muon (Newton-Schulz orthogonalized) on 2D linear weights, AdamW
# on embeddings / norms / biases / LM head, with separate per-group LR schedules.
#
# Parity with train_deepseek_poet.sh (kept identical so the comparison is
# one-variable):
#   * tp=1 / sequence_parallel=false. The shared deepseek_3bv2 recipe runs with
#     unfused qkv/fc1 (optim/muon_hybrid sets base.model.unfuse_qkv/unfuse_fc1=
#     true, same as POET); the unfuse path requires tp=1 -- so this pins tp=1
#     exactly like the POET full mode (pure data parallel, dp=8 on 8 GPUs).
#   * base.model.transformer_impl=local mirrors the local forward path the POET
#     optimizer auto-forces. SP off is required by the local-impl
#     WrappedTorchNorm (and is a no-op at tp=1). muon_tp_mode=blockwise is moot
#     at tp=1.
#   * training_regime=fixed_10b, seq_length=256, gbs=1024, mbs=4 -- all matched.
#
# LR: uses the muon_hybrid config defaults -- Muon side lr 2e-3, Adam side lr
# 1e-3 (configs/experiments/optim/muon_hybrid.yaml). These are Muon's own tuned
# LRs, NOT POET's 4e-3. The scheduler's peak LR / decay floor track the Adam-side
# lr (min_lr_ratio=0.1 => ~1e-4). Override with optim.muon.lr=... / optim.adam.lr=...
#
# Trailing key=value args pass through to launchers.train_megatron (last wins).
# Set SLM_DRYRUN_PRINT=1 to print the resolved command without launching.

MODE="${1:-dev}"
shift || true

SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Skip the CUDA env loader when only dry-printing (it derefs $HOME under set -u).
if [[ "${SLM_DRYRUN_PRINT:-0}" != "1" ]]; then
  source "$SLM_REPO/load_cuda13_2_nccl_env.sh"
fi

COMMON=(
  "base/family=deepseek_v3_mqa"
  "base/scale=deepseek_3bv2"
  "experiment=optim/muon_hybrid"
  "scheduler=wsd"
  # Match the POET run's local forward path (POET auto-forces local; Muon does
  # not, so set it explicitly).
  "base.model.transformer_impl=local"
)

case "${MODE}" in
  dev)
    # Single-GPU smoke: cluster=dev (local torchrun, 1 GPU), tiny token budget,
    # no checkpoints, offline wandb, expandable allocator to limit fragmentation.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    RUN=(python -m launchers.train_megatron
      "${COMMON[@]}"
      "cluster=dev"
      "training.global_batch_size=8"
      "training.micro_batch_size=1"
      "training.total_tokens=600000"
      "training.log_interval=1"
      "training.save_enabled=false"
      "wandb.project=slm-zeju-dev"
      "$@")
    ;;
  full)
    # 8-GPU cluster run. Force tp=1 => pure data parallel across the node's GPUs
    # (dp=8 on 8). SP off is required by the local-impl WrappedTorchNorm (and is
    # a no-op at tp=1). Budget pinned to fixed_10b (10B tokens), seq 256, to
    # mirror train_deepseek_poet.sh exactly. Override training_regime=... or
    # training.total_tokens=... / base.model.seq_length=... to change.
    RUN=(python -m launchers.train_megatron
      "${COMMON[@]}"
      "cluster=h100_de"
      "parallelism.tp_size_rules=[{model_params_lt: 1.0e15, tp: 1, pp: 1}]"
      "parallelism.sequence_parallel=false"
      "training_regime=fixed_10b"
      "base.model.seq_length=256"
      "training.global_batch_size=1024"
      "training.micro_batch_size=4"
      "training.save_enabled=true"
      "$@")
    ;;
  *)
    echo "Usage: scripts/train_deepseek_muon.sh [dev|full] [hydra overrides...]" >&2
    echo "  dev  - single-GPU smoke (cluster=dev, no checkpoints)" >&2
    echo "  full - 8-GPU cluster run (cluster=h100_de, checkpoints on)" >&2
    exit 2
    ;;
esac

if [[ "${SLM_DRYRUN_PRINT:-0}" == "1" ]]; then
  printf '%s ' "${RUN[@]}"; echo
else
  "${RUN[@]}"
fi
