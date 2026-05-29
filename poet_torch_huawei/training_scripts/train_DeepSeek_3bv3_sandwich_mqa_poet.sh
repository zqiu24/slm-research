#!/bin/bash
# Fail fast AND propagate pipeline failures so `tee` doesn't swallow
# torchrun's non-zero exit code.
set -o pipefail
source /home/miniconda3/bin/activate megatron-lm-014
# DeepSeek 3B training with POET-X reparameterization.
# Baseline: train_DeepSeek_3bv3_sandwich_mqa.sh
# POET paper: https://arxiv.org/abs/2506.08001
# POET-X paper: https://arxiv.org/abs/2603.05500
# POET code: https://github.com/Sphere-AI-Lab/poet

export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr

export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

export NVTE_WITH_USER_CUDA=1
export CUDNN_PATH=/usr
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda-12.8/include

export NCCL_NVLS_ENABLE=0
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=12000
export NCCL_NET_GDR_LEVEL=2
export NCCL_MIN_NCHANNELS=4

# Force line-buffered stdout so iteration logs actually reach the .log file /
# tmux scrollback. With block buffering + tee, iteration prints can sit in
# libc buffers for minutes while dataset-loader chatter pushes them out of
# the scrollback window, making a healthy run look hung.
export PYTHONUNBUFFERED=1

# ============================================================================
# Optional: enable PyTorch profiler. OFF by default.
#
# Empirically on this stack torch.profiler.export_chrome_trace spends >>10
# minutes (often effectively forever) serializing the trace for this MoE +
# POET model at the end of the profile window. Symptoms seen in practice:
#   * training progresses through iter 1..profile-step-end normally,
#   * then main thread sits in
#     torch/profiler/profiler.py:step -> _trace_ready -> handler_fn ->
#     export_chrome_trace (torch/autograd/profiler.py:490),
#   * GPU util stays non-zero (CUDA events still draining), no new iters.
# Contributors: (a) MoE + many layers produce a huge event set,
# (b) torch.compile'd subgraphs elsewhere in the stack (e.g. sepllm flex
# attention, poet_torch forward_core if ever imported) generate Inductor
# kernel events the profiler exporter handles very poorly.
# Use Nsight Systems (nsys profile torchrun ...) if you need a real trace.
# ============================================================================
POET_ENABLE_PROFILER=${POET_ENABLE_PROFILER:-0}

# ==============================================================================
# Distributed Training Setup
# ==============================================================================
GPUS_PER_NODE=8
NUM_NODES=${HOST_NUM:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

# ==============================================================================
# Parallelism Configuration
# POET note: we force TP=1 so that each parallel linear's local out_dim stays
# divisible by --poet-block-size (256). With TP=2, e.g. the QKV output
# (6912) would split to 3456, which is not a multiple of 256.
# CP=1 is also required (POET does not interact with context parallelism).
# ==============================================================================
TP=${2:-1}
PP=${3:-1}
CP=${4:-1}
EP=${5:-8}

if [ "$TP" != "1" ] || [ "$CP" != "1" ]; then
    echo "[POET] WARNING: TP=$TP CP=$CP -- POET requires local dims divisible"
    echo "       by --poet-block-size. Double-check your config or drop block_size"
    echo "       to 128 if you must increase TP."
fi

MBS=4
GBS=1024
SEQ_LEN=4096
EVAL_STEP=200 #200
EVAL_ITERS=32
SAVE_STEP=2000
MONITOR_STEP=1000

# Training schedule.
TRAIN_ITERS=48000
WSD_DECAY_ITERS=12000
LR_WARMUP_ITERS=2000
LR=8.6e-4
MIN_LR=7e-6

# ==============================================================================
# POET hyperparameters (also set in the YAML, but can be overridden here).
# ==============================================================================
POET_BLOCK_SIZE=${POET_BLOCK_SIZE:-256}
POET_MERGE_INTERVAL=${POET_MERGE_INTERVAL:-200}
POET_MEM_EFFICIENT=${POET_MEM_EFFICIENT:-0}   # 1 -> enable POET-X memory mode
POET_QUANTIZE=${POET_QUANTIZE:-0}             # 1 -> enable POET-XQ (INT8)

# ==============================================================================
# Paths
# ==============================================================================
MAIN_DIR=/public/shihan/experiments # 本地服务器
# MAIN_DIR=/experiments # 云服务器
SAVE_DIR=$MAIN_DIR/poet_3bv3_mqa

if [ -z "$SAVE_DIR" ]; then
    echo "Error: SAVE_DIR is not set." >&2
    exit 1
fi

DATA_NAME=dolma3-Shuffle-Stack-part0
VALID_DATA_NAME=dolma3_Shuffle-Stack-part1
DATA=/public/Datasets/04_mixing_lists/DeepSeek-V3+dolma3_mix-6T+Shuffle+Stack.part0.list
VALID_DATA=/public/Datasets/04_mixing_lists/DeepSeek-V3+dolma3_mix-6T+Shuffle+Stack.part1.list
TOKENIZER=/public/Datasets/00_tokenizers/DeepSeek-V3

RUNTIME=$(date +%m%d%H%M)
DATE=$(date +%Y%m%d)
export WANDB_PROJECT="H200_poet_3Bv3_mqa"
export WANDB_EXP_NAME="DPSK3bv2_POET_bf16_seq${SEQ_LEN}_node${NUM_NODES}_mbs${MBS}_gbs${GBS}_bs${POET_BLOCK_SIZE}_mi${POET_MERGE_INTERVAL}_TP${TP}_PP${PP}_CP${CP}_EP${EP}_${RUNTIME}"
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

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# POET wrapper script instead of pretrain_gpt.py.
PRETRAIN_SCRIPT_PATH="pretrain_gpt_poet.py"
DATA_CACHE_PATH="$MAIN_DIR/data_cache/benchmark_cache_deepseek_3bv2_${DATA_NAME}"
mkdir -p "$DATA_CACHE_PATH"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ==============================================================================
# Parse YAML Configuration
# ==============================================================================
YAML_CONFIG="$SCRIPT_DIR/model_args/DeepSeek-3Bv2-sandwich-mqa-poet.yaml"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "Error: YAML config not found at $YAML_CONFIG"
    exit 1
fi

echo "==============================================="
echo "Loading POET configuration from: $YAML_CONFIG"
echo "==============================================="

PARSE_YAML_SCRIPT="${SCRIPT_DIR}/parse_yaml.sh"

if [ ! -f "$PARSE_YAML_SCRIPT" ]; then
    echo "Error: parse_yaml.sh not found at $PARSE_YAML_SCRIPT"
    exit 1
fi

source "$PARSE_YAML_SCRIPT" "$YAML_CONFIG"
read -ra MODEL_ARGS <<< "$MODEL_ARGS_FROM_CONFIG"

echo "POET config loaded. TP=$TP, PP=$PP, CP=$CP, EP=$EP"
echo "MBS=$MBS, GBS=$GBS, SEQ_LEN=$SEQ_LEN"
echo "POET: block_size=$POET_BLOCK_SIZE merge_interval=$POET_MERGE_INTERVAL "
echo "      mem_efficient=$POET_MEM_EFFICIENT quantize=$POET_QUANTIZE"
echo "==============================================="

# ==============================================================================
# Distributed Arguments
# ==============================================================================
# Persist every rank's stdout+stderr to its own file under TORCHRUN_LOG_DIR,
# and tee rank-0 to the parent console. --redirects 3 / --tee 3 : ALL.
TORCHRUN_LOG_DIR="$SAVE_DIR/logs/$EXP_NAME/torchrun"
mkdir -p "$TORCHRUN_LOG_DIR"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --log-dir "$TORCHRUN_LOG_DIR"
    --redirects 3
    --tee 3
)

# ==============================================================================
# Training-specific Arguments
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

# Allow shell-level overrides of the POET flags declared in the YAML.
POET_CLI_ARGS=(
    --poet-block-size $POET_BLOCK_SIZE
    --poet-merge-interval $POET_MERGE_INTERVAL
)
if [ "$POET_MEM_EFFICIENT" = "1" ]; then
    POET_CLI_ARGS+=(--poet-mem-efficient)
fi
if [ "$POET_QUANTIZE" = "1" ]; then
    POET_CLI_ARGS+=(--poet-quantize)
fi

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
        "--wandb-exp-name" "${WANDB_EXP_NAME:-deepseek_3b_poet}"
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
if [ "$POET_ENABLE_PROFILER" = "1" ]; then
    # See the long comment at the top of this file. Expect the run to stall
    # at the end of the profile window while the chrome trace is serialized.
    CHECKPOINT_LOGGING_ARGS+=(
        --use-pytorch-profiler
        --profile
        --profile-step-start 4
        --profile-step-end 6
    )
fi

# ==============================================================================
# Validation
# ==============================================================================
CODE_BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)
export PYTHONPATH=$CODE_BASE_DIR
cd $CODE_BASE_DIR

if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: $PRETRAIN_SCRIPT_PATH not found in $CODE_BASE_DIR"
    exit 1
fi

# ==============================================================================
# Run Training
# ==============================================================================
echo "==============================================="
echo "Starting DeepSeek 3B + POET-X Training"
echo "==============================================="
echo "Checkpoint:    $CHECKPOINT_PATH"
echo "TensorBoard:   $TENSORBOARD_LOGS_PATH"
echo "Data:          $DATA_ARG"
echo "Tokenizer:     $TOKENIZER_ARG"
echo "GPUs:          $GPUS_PER_NODE"
echo "Parallelism:   TP=$TP, PP=$PP, CP=$CP, EP=$EP"
echo "Batch:         MBS=$MBS, GBS=$GBS"
echo "POET:          block=$POET_BLOCK_SIZE, merge_interval=$POET_MERGE_INTERVAL"
echo "==============================================="

RUN_LOG_FILE="$SAVE_DIR/logs/$EXP_NAME/run.$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$RUN_LOG_FILE")"
echo "==============================================="
echo "Run log:       $RUN_LOG_FILE"
echo "Per-rank logs: $TORCHRUN_LOG_DIR/<rank>/{stdout,stderr}.log"
echo "==============================================="

torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_SPECIFIC_ARGS[@]} \
    ${POET_CLI_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${CHECKPOINT_LOGGING_ARGS[@]} 2>&1 | tee "$RUN_LOG_FILE"

TORCHRUN_EXIT=${PIPESTATUS[0]}
echo "torchrun exited with code $TORCHRUN_EXIT"
echo "Full log:      $RUN_LOG_FILE"
echo "Per-rank logs: $TORCHRUN_LOG_DIR"

set +x
exit $TORCHRUN_EXIT
