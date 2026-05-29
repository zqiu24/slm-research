#!/bin/bash
# Single-GPU, mock-data SMOKE run of DeepSeek-3B + POET (block-size 128, local impl).
# Isolated huawei vendor stack (Megatron-core 0.14). Proves POET wraps and
# trains end-to-end before scaling to the real 8-GPU run. No external data.
set -o pipefail

# --- Environment: the `poet` conda env (torch 2.8 / triton 3.4, no TE) -----
source /home/zqiu/anaconda3/etc/profile.d/conda.sh && conda activate poet

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Megatron's _compile_dependencies() unconditionally calls legacy fused_kernels.load(),
# which probes `$CUDA_HOME/bin/nvcc -V` for the CUDA version. The `poet` env ships only
# torch's bundled CUDA runtime (no nvcc) so CUDA_HOME is unset -> TypeError(None + str).
# load() compiles nothing (its helper is unused), so a valid nvcc for the probe is enough.
# cuda/12.9 matches torch 2.8.0+cu129. Override via env if your cluster path differs.
export CUDA_HOME="${CUDA_HOME:-/is/software/nvidia/cuda-12.9}"

# Offline W&B for a smoke (no key needed).
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-huawei_poet_dev}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-deepseek3b_poet_dev_smoke}"

# --- Paths -----------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
CODE_BASE_DIR=$(cd "$SCRIPT_DIR/../" && pwd)   # .../slm-research/poet_torch_huawei
export PYTHONPATH=$CODE_BASE_DIR
cd "$CODE_BASE_DIR"

OUT_DIR="${OUT_DIR:-$CODE_BASE_DIR/runs_dev/poet_dev_smoke}"
mkdir -p "$OUT_DIR/data_cache" "$OUT_DIR/checkpoints" "$OUT_DIR/tb"

# --- Model config (shrunk dev YAML) ----------------------------------------
YAML_CONFIG="$SCRIPT_DIR/model_args/DeepSeek-3Bv2-sandwich-mqa-poet-dev.yaml"
source "$SCRIPT_DIR/parse_yaml.sh" "$YAML_CONFIG"
read -ra MODEL_ARGS <<< "$MODEL_ARGS_FROM_CONFIG"

# --- Single-GPU launch -----------------------------------------------------
DISTRIBUTED_ARGS=( --nproc_per_node 1 --nnodes 1 --node_rank 0
                   --master_addr localhost --master_port "${MASTER_PORT:-6000}" )

TRAINING_SPECIFIC_ARGS=(
  --micro-batch-size 1
  --global-batch-size 8
  --lr 8.6e-4 --min-lr 7e-6
  --lr-warmup-iters 2 --lr-decay-style WSD --lr-wsd-decay-style cosine --lr-wsd-decay-iters 10
  --train-iters 30
  --seq-length 512
  --calculate-per-token-loss
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
)

# block-size 128 explicitly (reference script default is 256).
POET_CLI_ARGS=( --poet-block-size 128 --poet-merge-interval 20 )

DATA_ARGS_LIST=(
  --mock-data
  --tokenizer-type NullTokenizer
  --vocab-size 151936
  --data-cache-path "$OUT_DIR/data_cache"
  --split 99,1,0
  --no-create-attention-mask-in-dataloader
  --no-mmap-bin-files
  --num-workers 1
)

# Checkpoint saving is OFF by default for the smoke (set SAVE_CKPT=1 to enable).
# Without --save, Megatron writes no checkpoints — including the implicit
# end-of-run save (every save_checkpoint() call is guarded by `if args.save`).
# This also sidesteps the post-merge heterogeneous-optimizer-step save assert.
CHECKPOINT_LOGGING_ARGS=(
  --tensorboard-dir "$OUT_DIR/tb"
  --eval-iters 0 --eval-interval 1000 --save-interval 1000
  --ckpt-format torch_dist
  --log-interval 1
)
if [[ "${SAVE_CKPT:-0}" != "0" ]]; then
  CHECKPOINT_LOGGING_ARGS+=( --save "$OUT_DIR/checkpoints" --load "$OUT_DIR/checkpoints" )
fi

echo "[dev] launching DeepSeek-3B POET smoke: 1 GPU, mock data, block-size 128, 30 steps"
# Launch via the active env's python (poet, 3.12) — a bare `torchrun` resolves to
# ~/.local/bin/torchrun (shebang /usr/bin/python3 = system 3.10), which escapes the
# conda env and runs workers under the wrong interpreter. `python -m torch.distributed.run`
# pins both the agent and workers to the poet env python validated in Task 4.
python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" pretrain_gpt_poet.py \
  "${MODEL_ARGS[@]}" "${TRAINING_SPECIFIC_ARGS[@]}" \
  "${POET_CLI_ARGS[@]}" "${DATA_ARGS_LIST[@]}" "${CHECKPOINT_LOGGING_ARGS[@]}"
