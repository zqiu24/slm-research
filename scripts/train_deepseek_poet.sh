#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-3Bv2 (MQA + sandwich-norm) trained with the slm-research POET champion
# recipe (experiment=optim/poet_lie_orth_alt: head-OFF lie_ortho + alternating
# POETX, scale 0.5, lie_ortho_c 8, lr 4e-3 -> eff angle ~0.016). Same model as
# train_deepseek.sh, but POET instead of AdamW, with dev/full modes mirroring
# train_poet_huawei.sh.
#
#   bash scripts/train_deepseek_poet.sh dev    [hydra overrides...]   # 1-GPU smoke
#   bash scripts/train_deepseek_poet.sh full   [hydra overrides...]   # 8-GPU cluster
#
# POET needs no --transformer-impl flag: megatron_args forces local for the poet
# optimizer (the GPT spec reads args.transformer_impl, which the runtime config
# flip never reached). sandwich_norm_apply + poet_moe_local_rmsnorm are already in
# the poet_lie_orth_alt patch list, so MQA + sandwich + MoE compose out of the box.
#
# block_count (BLOCK_COUNT env, default 8): POET's oft_R is ~dim^2 / block_count
# per linear. The 60m champion used block_count=1 (full-dim rotations), but at this
# 3B / 64-expert MoE scale bc=1 inflates oft_R to ~3.8B params (~178GB peak) and
# does NOT fit on one GPU -- nor on an 80GB H100 under data parallel, because POET's
# weight unfuse forces TP=1, so each rank holds the whole model (DP replicates, it
# does not shard). bc=8 shrinks oft_R ~8x (~0.74B, ~31GB peak) and fits. The exact
# bc=1 champion needs model sharding (pipeline parallel), which is untested with
# POET. Override the block count with:
#   BLOCK_COUNT=4 bash scripts/train_deepseek_poet.sh dev
#
# Trailing key=value args pass through to launchers.train_megatron (last wins).
# Set SLM_DRYRUN_PRINT=1 to print the resolved command without launching.

MODE="${1:-dev}"
shift || true

BLOCK_COUNT="${BLOCK_COUNT:-8}"

SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Skip the CUDA env loader when only dry-printing (it derefs $HOME under set -u).
if [[ "${SLM_DRYRUN_PRINT:-0}" != "1" ]]; then
  source "$SLM_REPO/load_cuda13_2_nccl_env.sh"
fi

COMMON=(
  "base/family=deepseek_v3_mqa"
  "base/scale=deepseek_3bv2"
  "experiment=optim/poet_lie_orth_alt"
  "scheduler=wsd"
  "optim.lr=4e-3"
  "optim.poet.block_count=${BLOCK_COUNT}"
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
    # 8-GPU cluster run (data parallel; POET forces TP=1). Checkpoints on.
    # Budget pinned to fixed_10b (10B tokens) for testing rather than inheriting
    # the repo default ablation_20x (which is 20x*3B = 60B). Override the regime
    # (e.g. training_regime=ablation_20x) or training.total_tokens=... to change.
    # mbs=1 is the safe default; raise it if the per-GPU memory headroom allows.
    RUN=(python -m launchers.train_megatron
      "${COMMON[@]}"
      "cluster=h100_de"
      "training_regime=fixed_10b"
      "training.global_batch_size=1024"
      "training.micro_batch_size=1"
      "training.save_enabled=true"
      "optim.min_lr=7e-6"
      "$@")
    ;;
  *)
    echo "Usage: scripts/train_deepseek_poet.sh [dev|full] [hydra overrides...]" >&2
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
