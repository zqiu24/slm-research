#!/bin/bash
source /home/miniconda3/bin/activate megatron-lm-014
# DeepSeek 3B Training Script with YAML Configuration
# This script uses DeepSeek-3B.yaml for model arguments
# Optimized for H200 GPUs (141GB VRAM)

export CUDA_DEVICE_MAX_CONNECTIONS=1
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
TP=${2:-1}  # Tensor parallel size (increased for 30B model)
PP=${3:-1}  # Pipeline parallel size (increased for 30B model)
CP=${4:-1}  # Context parallel size
EP=${5:-8}  # Expert parallel size (8 GPUs / (TP=2 * PP=2) = 2, but we can use EP=8 with TP=1, PP=1)
            # Adjusted: With 8 GPUs, TP=2, PP=2, we get DP=2, so EP should divide evenly
            # Better config: TP=1, PP=2, EP=4 -> 8 GPUs total
MBS=4              # Micro batch size (not in YAML, script-specific)
GBS=1024 # for 4nodes 1T else GBS 1node 200B set to 320            # Global batch size (not in YAML, script-specific)
SEQ_LEN=4096       # Sequence length
EVAL_STEP=200 ## to 200 ??
EVAL_ITERS=32
SAVE_STEP=2000 ## real: 1000
MONITOR_STEP=1000 ## real: 100

# Training-specific parameters (not in YAML)
TRAIN_ITERS=48000 # 200B
WSD_DECAY_ITERS=12000 # 50B
LR_WARMUP_ITERS=2000
LR=8.6e-4
MIN_LR=7e-6

# ==============================================================================
# Paths
# ==============================================================================

## main dir
MAIN_DIR=/public/shihan/experiments ## ??
SAVE_DIR=$MAIN_DIR/baseline_3bv3_mqa_cx_tiny

if [ -z "$SAVE_DIR" ]; then
    echo "Error: SAVE_DIR is not set." >&2
    exit 1
fi

## data and tokenizer
DATA_NAME=dolma3-Shuffle-Stack-part0 ## ??
VALID_DATA_NAME=dolma3_Shuffle-Stack-part1
DATA=/public/Datasets/04_mixing_lists/DeepSeek-V3+dolma3_mix-6T+Shuffle+Stack.part0.list ## ??
VALID_DATA=/public/Datasets/04_mixing_lists/DeepSeek-V3+dolma3_mix-6T+Shuffle+Stack.part1.list
TOKENIZER=/public/Datasets/00_tokenizers/DeepSeek-V3

RUNTIME=$(date +%m%d%H%M)
DATE=$(date +%Y%m%d)
export WANDB_PROJECT="H200_baseline_3Bv3_mqa" ## ??
export WANDB_EXP_NAME="DPSK3bv2_bf16_seqlen${SEQ_LEN}_node${NUM_NODES}_data${DATA_NAME}_mbs${MBS}_gbs${GBS}_TP${TP}_PP${PP}_CP${CP}_EP${EP}_eval${EVAL_STEP}_save${SAVE_STEP}_${RUNTIME}"
export WANDB_API_KEY="4604e54e9c69942344bf98f695b966bc710a6a90" ## ??
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
# WANDB_API_KEY=${5:-$WANDB_API_KEY}

# Create directories
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"
DATA_CACHE_PATH="$MAIN_DIR/data_cache/benchmark_cache_deepseek_3bv2_${DATA_NAME}"
mkdir -p "$DATA_CACHE_PATH"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ==============================================================================
# Parse YAML Configuration
# ==============================================================================
YAML_CONFIG="$SCRIPT_DIR/model_args/DeepSeek-3Bv2-sandwich-mqa-cx.yaml"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "Error: YAML config not found at $YAML_CONFIG"
    exit 1
fi

echo "==============================================="
echo "Loading configuration from: $YAML_CONFIG"
echo "==============================================="

# 使用 parse_yaml.sh 解析 YAML 配置
# Use parse_yaml.sh to parse YAML configuration
PARSE_YAML_SCRIPT="${SCRIPT_DIR}/parse_yaml.sh"

if [ ! -f "$PARSE_YAML_SCRIPT" ]; then
    echo "Error: parse_yaml.sh not found at $PARSE_YAML_SCRIPT"
    exit 1
fi

# 执行 parse_yaml.sh 获取模型参数
# Execute parse_yaml.sh to get model parameters
source "$PARSE_YAML_SCRIPT" "$YAML_CONFIG"

# Convert string to array for use in torchrun command
read -ra MODEL_ARGS <<< "$MODEL_ARGS_FROM_CONFIG"

echo "Configuration loaded successfully"
echo "TP=$TP, PP=$PP, CP=$CP, EP=$EP"
echo "MBS=$MBS, GBS=$GBS, SEQ_LEN=$SEQ_LEN"
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
# These parameters are specific to this training run and not in the YAML config
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
    # Handle .list file for data source
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

    # Handle .list file for data source
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
        "--wandb-exp-name" "${WANDB_EXP_NAME:-deepseek_3b}"
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
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    exit 1
fi

# ==============================================================================
# Run Training
# ==============================================================================
echo "==============================================="
echo "Starting DeepSeek 3B Training"
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
    ${DATA_ARGS_LIST[@]} \
    ${CHECKPOINT_LOGGING_ARGS[@]}

set +x
