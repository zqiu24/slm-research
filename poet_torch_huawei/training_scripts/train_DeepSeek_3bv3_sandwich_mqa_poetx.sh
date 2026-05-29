#!/bin/bash
# Fail fast AND propagate pipeline failures so `tee` doesn't swallow
# torchrun's non-zero exit code.
set -o pipefail
source /home/miniconda3/bin/activate megatron-lm-014
# DeepSeek 3B training with POET-X_fast (activation-space) reparameterization.
#
# Algorithmically identical to train_DeepSeek_3bv3_sandwich_mqa_poet.sh; the
# difference is purely system-level speed:
#   * --poet-variant poetx routes forwards through
#     ``poet_torch.core.ops.forward_core``, which is @torch.compile(fullgraph)
#     and calls the Triton Cayley-Neumann kernel.
#   * POET_MEM_EFFICIENT=1 additionally switches to the
#     ``chain_layer_x_checkpoint_mem_o2`` Triton kernel (recomputes the
#     x-rotation chain in backward -> ~half activation memory).
#
# Both of these live in poet_torch/core/{ops,triton_ops}.py vendored from
# https://github.com/Sphere-AI-Lab/poet. They compose with Megatron's
# distributed optimizer + ZeRO path unchanged.
#
# NOTE: the YAML forces --transformer-impl local because the activation-space
# path cannot live inside TE's fused LayerNorm+Linear (RMSNorm's gamma does
# not commute with R_in). If you want TE, use the weight-space poet.sh.

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

# torch.compile knobs: POET-X fast relies on torch.compile fullgraph for the
# forward chain. We have to lift *three* separate Dynamo caps because:
#   1. Different POET linear shapes (qkv/proj/fc1/fc2/router/...) each take
#      a cache slot during warmup.
#   2. Every merge-then-reinit (every --poet-merge-interval steps) bumps
#      ``_version`` on weight + perm_* buffers, invalidating every old
#      compiled entry's guards. New entries are created instead of replacing
#      old ones, so usage grows linearly with merge events until a cap hits.
#   3. ``fullgraph=True`` sets ``one_graph=True`` inside Dynamo, which guards
#      against recompilation with the separate ``recompile_limit`` (default 8)
#      and raises ``FailOnRecompileLimitHit`` (instead of graceful fallback)
#      when exceeded -- this is exactly what bit the first 2-node run at
#      step 201 (first merge was at step 200).
#
# pretrain_gpt_poet.py also sets these caps inside Python *before* importing
# poet_torch, and the adapter calls ``torch._dynamo.reset()`` after each
# merge. The env vars below are belt-and-suspenders for any subprocess /
# launcher that forks before the Python-side config runs.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}
export TORCHDYNAMO_CACHE_SIZE_LIMIT=${TORCHDYNAMO_CACHE_SIZE_LIMIT:-512}
export TORCHDYNAMO_RECOMPILE_LIMIT=${TORCHDYNAMO_RECOMPILE_LIMIT:-512}
export TORCHDYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT=${TORCHDYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT:-1024}
export TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT=${TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT:-1024}
# Force line-buffered stdout so iteration logs actually reach the .log file
# (default block buffering + tee can hide hangs for many minutes).
export PYTHONUNBUFFERED=1

# ============================================================================
# Optional: enable PyTorch profiler. Off by default because
# torch.profiler + torch.compile(fullgraph=True) deadlocks during
# stop_trace when Inductor-generated kernels are being recorded -- the
# main thread gets stuck in profiler.__exit__ while background workers
# keep draining CUDA events, so GPU util stays > 0 but no iterations
# advance. Symptom: log ends at "after 1 iterations" and never
# progresses. Use Nsight Systems (nsys profile torchrun ...) instead if
# you need a trace of the fast path.
# ============================================================================
POET_X_ENABLE_PROFILER=${POET_X_ENABLE_PROFILER:-0}

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
# POET-X note: TP=1 is required. RowParallelLinear's R_out at TP>1 would need
# all ranks to agree on the same permutation (we generate per-rank permutations
# today); the adapter hard-asserts TP=1 on the fast path.
# ==============================================================================
TP=${2:-1}
PP=${3:-1}
CP=${4:-1}
EP=${5:-8}

if [ "$TP" != "1" ] || [ "$CP" != "1" ]; then
    echo "[POET-X] ERROR: TP=$TP CP=$CP -- POET-X fast path requires TP=1, CP=1."
    echo "         Use the weight-space poet.sh if you need TP>1."
fi

MBS=4
GBS=1024
SEQ_LEN=4096
EVAL_STEP=200
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
# POET-X hyperparameters (also set in the YAML, but can be overridden here).
# ==============================================================================
POET_BLOCK_SIZE=${POET_BLOCK_SIZE:-128}
POET_MERGE_INTERVAL=${POET_MERGE_INTERVAL:-200}
# POET_MEM_EFFICIENT=1 -> use chain_layer_x_checkpoint_mem_o2 Triton kernel
# (recomputes x-rotation chain in bwd for ~half activation mem).
POET_MEM_EFFICIENT=${POET_MEM_EFFICIENT:-0}
POET_QUANTIZE=${POET_QUANTIZE:-0}             # not wired in Megatron adapter yet
# POET_NORMALIZE_WEIGHTS=1 -> row-L2-normalize each wrapped W_0 at install time
# (upstream POET default). Off by default here because:
#   * Megatron uses --init-method-std=0.006 -> row-norm ~= 0.006 * sqrt(in_dim)
#     ~= 0.21 for hidden=1280; row-normalizing pushes that to 1.0 (~4.6x scale
#     blow-up of every wrapped linear).
#   * Combined with --use-sandwich-norm + small post-norm-scale (0.03) and the
#     LR/WSD schedule tuned for the AdamW baseline, this 4.6x scale change makes
#     the network's activation/gradient dynamics no longer comparable to the
#     baseline -- empirically POET-X then trains *worse* than AdamW even though
#     the paper claims parity.
#   * --qk-layernorm absorbs the scale on the Q/K side, but V / attn-output /
#     fc1 / fc2 / shared_experts still see the full blowup.
# Set POET_NORMALIZE_WEIGHTS=1 if you specifically want to reproduce the upstream
# install-time normalization (e.g. for the LLaMA-style init the POET paper uses).
POET_NORMALIZE_WEIGHTS=${POET_NORMALIZE_WEIGHTS:-1}

if [ "$POET_QUANTIZE" = "1" ]; then
    echo "[POET-X] ERROR: POET_QUANTIZE=1 is not supported in this Megatron integration yet."
    echo "         The adapter still uses float base weights and does not route to QPOET/forward_core_q8."
    echo "         Leave POET_QUANTIZE=0 for train_DeepSeek_3bv3_sandwich_mqa_poetx.sh."
    exit 1
fi

# ==============================================================================
# Paths
# ==============================================================================
# MAIN_DIR=/public/shihan/experiments # 本地服务器
MAIN_DIR=/experiments # 云服务器
SAVE_DIR=$MAIN_DIR/poetx_3bv3_mqa_fixed

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
export WANDB_PROJECT="H200_poetx_3Bv3_mqa"
export WANDB_EXP_NAME="DPSK3bv2_POETX_bf16_seq${SEQ_LEN}_node${NUM_NODES}_mbs${MBS}_gbs${GBS}_bs${POET_BLOCK_SIZE}_mi${POET_MERGE_INTERVAL}_me${POET_MEM_EFFICIENT}_norm${POET_NORMALIZE_WEIGHTS}_TP${TP}_PP${PP}_CP${CP}_EP${EP}_${RUNTIME}"
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
YAML_CONFIG="$SCRIPT_DIR/model_args/DeepSeek-3Bv2-sandwich-mqa-poetx.yaml"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "Error: YAML config not found at $YAML_CONFIG"
    exit 1
fi

echo "==============================================="
echo "Loading POET-X configuration from: $YAML_CONFIG"
echo "==============================================="

PARSE_YAML_SCRIPT="${SCRIPT_DIR}/parse_yaml.sh"

if [ ! -f "$PARSE_YAML_SCRIPT" ]; then
    echo "Error: parse_yaml.sh not found at $PARSE_YAML_SCRIPT"
    exit 1
fi

source "$PARSE_YAML_SCRIPT" "$YAML_CONFIG"
read -ra MODEL_ARGS <<< "$MODEL_ARGS_FROM_CONFIG"

echo "POET-X config loaded. TP=$TP, PP=$PP, CP=$CP, EP=$EP"
echo "MBS=$MBS, GBS=$GBS, SEQ_LEN=$SEQ_LEN"
echo "POET-X: block_size=$POET_BLOCK_SIZE merge_interval=$POET_MERGE_INTERVAL "
echo "        mem_efficient=$POET_MEM_EFFICIENT normalize_weights=$POET_NORMALIZE_WEIGHTS"
echo "==============================================="

# ==============================================================================
# Distributed Arguments
# ==============================================================================
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
    --poet-variant poetx
)
if [ "$POET_MEM_EFFICIENT" = "1" ]; then
    POET_CLI_ARGS+=(--poet-mem-efficient)
fi
if [ "$POET_NORMALIZE_WEIGHTS" != "1" ]; then
    # Disable upstream's row-L2-normalize on each W_0 at install time. See the
    # POET_NORMALIZE_WEIGHTS comment block above for why this is the default.
    POET_CLI_ARGS+=(--poet-no-normalize-weights)
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
        "--wandb-exp-name" "${WANDB_EXP_NAME:-deepseek_3b_poetx}"
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
if [ "$POET_X_ENABLE_PROFILER" = "1" ]; then
    # WARNING: torch.profiler + torch.compile deadlocks during stop_trace.
    # Only turn this on if you've also disabled torch.compile in
    # poet_torch/core/ops.py (wrap forward_core with @torch.compiler.disable),
    # otherwise training will hang at step 6 / 7.
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
echo "Starting DeepSeek 3B + POET-X_fast Training"
echo "==============================================="
echo "Checkpoint:    $CHECKPOINT_PATH"
echo "TensorBoard:   $TENSORBOARD_LOGS_PATH"
echo "Data:          $DATA_ARG"
echo "Tokenizer:     $TOKENIZER_ARG"
echo "GPUs:          $GPUS_PER_NODE"
echo "Parallelism:   TP=$TP, PP=$PP, CP=$CP, EP=$EP"
echo "Batch:         MBS=$MBS, GBS=$GBS"
echo "POET-X:        block=$POET_BLOCK_SIZE merge_interval=$POET_MERGE_INTERVAL mem_efficient=$POET_MEM_EFFICIENT normalize_weights=$POET_NORMALIZE_WEIGHTS"
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
