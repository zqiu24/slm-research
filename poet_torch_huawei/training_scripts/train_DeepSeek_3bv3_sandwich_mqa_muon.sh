#!/bin/bash
source /home/miniconda3/bin/activate megatron-lm-014
# DeepSeek 3B Training Script with YAML Configuration + Muon optimizer.
# Derived from train_DeepSeek_3bv3_sandwich_mqa.sh; the only delta is the
# pretrain entry point (pretrain_gpt_muon.py) and the Muon CLI flags
# appended to the torchrun invocation.
# Optimized for H200 GPUs (141GB VRAM)

export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr

export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

export NVTE_WITH_USER_CUDA=1 # must turn on when multiple-nodes
export CUDNN_PATH=/usr # must turn on when multiple-nodes
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda-12.8/include # must turn on when multiple-nodes

export NCCL_NVLS_ENABLE=0
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=12000
export NCCL_NET_GDR_LEVEL=2  # Enable GPUDirect RDMA if RDMA is available # optim0129
export NCCL_MIN_NCHANNELS=4  # Increase NCCL channels # optim0129

# ==============================================================================
# Distributed Training Setup
# ==============================================================================
GPUS_PER_NODE=8
NUM_NODES=${HOST_NUM:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

# ==============================================================================
# Parallelism Configuration (Script-level only, not in YAML)
# ==============================================================================
TP=${2:-1}  # Tensor parallel size
PP=${3:-1}  # Pipeline parallel size
CP=${4:-1}  # Context parallel size
EP=${5:-8}  # Expert parallel size
MBS=4              # Micro batch size
GBS=1024           # Global batch size
SEQ_LEN=4096       # Sequence length
EVAL_STEP=200
EVAL_ITERS=32
SAVE_STEP=2000
MONITOR_STEP=1000

# Training-specific parameters
TRAIN_ITERS=48000 # 200B
WSD_DECAY_ITERS=12000 # 50B
LR_WARMUP_ITERS=2000
LR=8.6e-4
MIN_LR=7e-6

# ==============================================================================
# Muon-specific hyperparameters
# Following Moonshot's Muon recipe. `muon_matched_adamw_rms` ~ 0.2-0.4 scales
# the Muon spectral-norm step to match the effective RMS of AdamW at the same
# LR; `muon_ns_steps` is the fixed (non-converging) Newton-Schulz count.
# `muon_ns_num_head_groups` only matters for MLA up-projections (ignored here
# because this config is MQA, but kept so the flag is visible).
# ==============================================================================
MUON_MATCHED_ADAMW_RMS=0.2
MUON_MOMENTUM=0.95
MUON_NS_STEPS=5
MUON_NS_NUM_HEAD_GROUPS=-1  # -1 disables head-group reshape (MQA config has no MLA)

# ==============================================================================
# Paths
# ==============================================================================

## main dir
# MAIN_DIR=/public/shihan/experiments # 本地服务器
MAIN_DIR=/experiments # 云服务器
SAVE_DIR=$MAIN_DIR/baseline_3bv3_mqa_muon

if [ -z "$SAVE_DIR" ]; then
    echo "Error: SAVE_DIR is not set." >&2
    exit 1
fi

## data and tokenizer
DATA_NAME=dolma3-Shuffle-Stack-part0
VALID_DATA_NAME=dolma3_Shuffle-Stack-part1
DATA=/public/Datasets/04_mixing_lists/DeepSeek-V3+dolma3_mix-6T+Shuffle+Stack.part0.list
VALID_DATA=/public/Datasets/04_mixing_lists/DeepSeek-V3+dolma3_mix-6T+Shuffle+Stack.part1.list
TOKENIZER=/public/Datasets/00_tokenizers/DeepSeek-V3

RUNTIME=$(date +%m%d%H%M)
DATE=$(date +%Y%m%d)
export WANDB_PROJECT="H200_baseline_3Bv3_mqa_muon"
export WANDB_EXP_NAME="DPSK3bv2_bf16_muon_seqlen${SEQ_LEN}_node${NUM_NODES}_data${DATA_NAME}_mbs${MBS}_gbs${GBS}_TP${TP}_PP${PP}_CP${CP}_EP${EP}_eval${EVAL_STEP}_save${SAVE_STEP}_${RUNTIME}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_SYNC_TENSORBOARD=false

EXP_NAME=$WANDB_EXP_NAME

# ==============================================================================
# Command-line Arguments
# ==============================================================================
CHECKPOINT_PATH=${1:-"$SAVE_DIR/checkpoints/$EXP_NAME"}
TENSORBOARD_LOGS_PATH=${2:-"$SAVE_DIR/tensorboard_logs/$EXP_NAME"}
TOKENIZER_ARG=${3:-"$TOKENIZER"}
DATA_ARG=${4:-"$DATA"}
VALID_DATA_ARG=$VALID_DATA

# Create directories
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

PRETRAIN_SCRIPT_PATH="pretrain_gpt_muon.py"
DATA_CACHE_PATH="$MAIN_DIR/data_cache/benchmark_cache_deepseek_3bv2_${DATA_NAME}"
mkdir -p "$DATA_CACHE_PATH"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ==============================================================================
# Parse YAML Configuration
# ==============================================================================
YAML_CONFIG="$SCRIPT_DIR/model_args/DeepSeek-3Bv2-sandwich-mqa.yaml"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "Error: YAML config not found at $YAML_CONFIG"
    exit 1
fi

echo "==============================================="
echo "Loading configuration from: $YAML_CONFIG"
echo "==============================================="

PARSE_YAML_SCRIPT="${SCRIPT_DIR}/parse_yaml.sh"

if [ ! -f "$PARSE_YAML_SCRIPT" ]; then
    echo "Error: parse_yaml.sh not found at $PARSE_YAML_SCRIPT"
    exit 1
fi

source "$PARSE_YAML_SCRIPT" "$YAML_CONFIG"

read -ra MODEL_ARGS <<< "$MODEL_ARGS_FROM_CONFIG"

echo "Configuration loaded successfully"
echo "TP=$TP, PP=$PP, CP=$CP, EP=$EP"
echo "MBS=$MBS, GBS=$GBS, SEQ_LEN=$SEQ_LEN"
echo "Optimizer=muon (momentum=$MUON_MOMENTUM, ns_steps=$MUON_NS_STEPS, matched_adamw_rms=$MUON_MATCHED_ADAMW_RMS)"
echo "==============================================="

# ==============================================================================
# Distributed Arguments
# ==============================================================================
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ==============================================================================
# Training-specific Arguments (not in YAML)
# ==============================================================================
TRAINING_SPECIFIC_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --lr $LR
    --min-lr $MIN_LR
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --lr-decay-style WSD \
    --lr-wsd-decay-style cosine
    --lr-wsd-decay-iters $WSD_DECAY_ITERS
    --train-iters $TRAIN_ITERS
    --seq-length $SEQ_LEN
    --calculate-per-token-loss
    --empty-unused-memory-level 1
    --recompute-granularity selective
    --grad-reduce-in-bf16
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --expert-model-parallel-size $EP
    --moe-per-layer-logging
    --reset-attention-mask
    --reset-position-ids
    --eod-mask-loss
)

# ==============================================================================
# Muon Optimizer Arguments
# ==============================================================================
MUON_ARGS=(
    --optimizer muon
    --muon-matched-adamw-rms $MUON_MATCHED_ADAMW_RMS
    --muon-momentum $MUON_MOMENTUM
    --muon-ns-steps $MUON_NS_STEPS
    --muon-ns-num-head-groups $MUON_NS_NUM_HEAD_GROUPS
)

# ==============================================================================
# Data Arguments
# ==============================================================================
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type" "NullTokenizer"
        "--vocab-size" "151936"
        "--data-cache-path" "${DATA_CACHE_PATH}"
        "--tiktoken-pattern" "v2"
        "--split" "99,1,0"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers" "1"
    )
else
    if [[ "$DATA_ARG" == *.list ]]; then
        if [ -f "$DATA_ARG" ]; then
            DATA_PATH=$(grep -v '^#' "$DATA_ARG" | grep -v '^$' | xargs)
        else
            echo "Error: Data list file $DATA_ARG not found."
            exit 1
        fi
    else
        DATA_PATH="$DATA_ARG"
    fi

    if [[ "$VALID_DATA_ARG" == *.list ]]; then
        if [ -f "$VALID_DATA_ARG" ]; then
            VALID_DATA_PATH=$(grep -v '^#' "$VALID_DATA_ARG" | grep -v '^$' | xargs)
        else
            echo "Error: Data list file $VALID_DATA_ARG not found."
            exit 1
        fi
    else
        VALID_DATA_PATH="$VALID_DATA_ARG"
    fi

    DATA_ARGS_LIST+=(
        "--train-data-path" "$DATA_PATH"
	    "--valid-data-path" "$VALID_DATA_PATH"
        "--tokenizer-type" "HuggingFaceTokenizer"
        "--tokenizer-model" "$TOKENIZER_ARG"
        "--data-cache-path" "${DATA_CACHE_PATH}"
        "--vocab-size" "129280"
	    "--num-workers" "8"
    )
fi

# ==============================================================================
# WandB Configuration
# ==============================================================================
WANDB_ARGS=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    WANDB_ARGS=(
        "--wandb-project" "${WANDB_PROJECT}"
        "--wandb-exp-name" "${WANDB_EXP_NAME:-deepseek_3b_muon}"
        "--wandb-save-dir" "${TENSORBOARD_LOGS_PATH}"
    )
fi

# ==============================================================================
# Checkpoint and Logging Arguments
# ==============================================================================
CHECKPOINT_LOGGING_ARGS=(
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --eval-iters $EVAL_ITERS
    --eval-interval $EVAL_STEP
    --save-interval $SAVE_STEP
    --ckpt-format torch_dist
    --use-pytorch-profiler
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --enable-transformer-stats-monitor
    --enable-attention-stats-monitor
    --stability-log-per-layer
    --stability-monitor-freq $MONITOR_STEP
    --stability-monitor-sample-tokens 1024
    --enable-monitor
    --monitor-interval $MONITOR_STEP
    "${WANDB_ARGS[@]}"
)

# ==============================================================================
# Validation
# ==============================================================================
CODE_BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)
export PYTHONPATH=$CODE_BASE_DIR
cd $CODE_BASE_DIR

if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt_muon.py not found at $PRETRAIN_SCRIPT_PATH"
    exit 1
fi

# ==============================================================================
# Run Training
# ==============================================================================
echo "==============================================="
echo "Starting DeepSeek 3B Training (Muon)"
echo "==============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "TensorBoard: $TENSORBOARD_LOGS_PATH"
echo "Data: $DATA_ARG"
echo "Tokenizer: $TOKENIZER_ARG"
echo "GPUs: $GPUS_PER_NODE"
echo "Parallelism: TP=$TP, PP=$PP, EP=$EP"
echo "Batch: MBS=$MBS, GBS=$GBS"
echo "==============================================="

torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_SPECIFIC_ARGS[@]} \
    ${MUON_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${CHECKPOINT_LOGGING_ARGS[@]}

set +x
