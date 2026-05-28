#!/usr/bin/env bash
# Tokenize a JSONL corpus into Megatron mmap (.bin/.idx) with the DeepSeek-V3
# tokenizer.
#
# It calls Megatron-poet's OWN tools/preprocess_data.py on purpose: that script
# writes via the same megatron.core.datasets.indexed_dataset and builds the
# tokenizer via the same build_tokenizer the DeepSeek trainer reads with, so the
# output is byte-compatible with training.
#
# Activate an env that can `import megatron` (incl. transformer_engine) FIRST, e.g.:
#   source /lustre/fast/fast/zqiu/slm-research/load_cuda13_2_nccl_env.sh   # CUDA-13.2 + LD_PRELOAD
#   source /fast/zqiu/slm_env/.venv/bin/activate
# The CUDA-13.2 loader is required: poet's megatron imports transformer_engine eagerly,
# and slm_env's TE needs the system libcublasLt.so.13 the loader LD_PRELOADs.
#
# Tokenizer note: on the HuggingFaceTokenizer path Megatron derives the vocab
# from len(tokenizer) and ignores --vocab-size, so it is intentionally NOT
# passed here. The DeepSeek-V3 tokenizer has len 128815, which pads to 129280
# (the model config vocab) under --make-vocab-size-divisible-by 3232.
#
# All inputs are env-overridable; see defaults below.
set -euo pipefail

INPUT_FILE="${INPUT_FILE:?set INPUT_FILE=/path/to/merged.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:?set OUTPUT_PREFIX=/path/to/out_prefix (no extension)}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fast/fast/zqiu/hf_models/DeepSeek-V3-tokenizer}"
WORKERS="${WORKERS:-8}"
# Default partitions=1: one process tokenizes the whole jsonl and writes the single
# {OUTPUT_PREFIX}_text_document.{bin,idx} directly (no merge, no temp partition files).
# Raise it only for very large corpora; WORKERS must then be a multiple of PARTITIONS.
PARTITIONS="${PARTITIONS:-1}"
JSON_KEYS="${JSON_KEYS:-text}"
MEGATRON_POET_ROOT="${MEGATRON_POET_ROOT:-/lustre/fast/fast/zqiu/tmp/Megatron-poet}"

PREP="${MEGATRON_POET_ROOT}/tools/preprocess_data.py"

# --- validation ---------------------------------------------------------------
[[ -f "$INPUT_FILE" ]]      || { echo "INPUT_FILE not found: $INPUT_FILE" >&2; exit 1; }
[[ -d "$TOKENIZER_MODEL" ]] || { echo "TOKENIZER_MODEL dir not found: $TOKENIZER_MODEL" >&2; exit 1; }
[[ -f "$PREP" ]]           || { echo "preprocess_data.py not found: $PREP (set MEGATRON_POET_ROOT)" >&2; exit 1; }
if (( WORKERS % PARTITIONS != 0 )); then
  echo "WORKERS ($WORKERS) must be a multiple of PARTITIONS ($PARTITIONS)" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

# NOTE: preprocess_data.py appends "_<json-key>_document" to --output-prefix, so the
# real files are ${OUTPUT_PREFIX}_${JSON_KEYS}_document.{bin,idx} (default: _text_document).
echo "[deepseek-tokenize] ${INPUT_FILE} -> ${OUTPUT_PREFIX}_${JSON_KEYS}_document.{bin,idx}"
echo "  tokenizer=${TOKENIZER_MODEL} (${TOKENIZER_TYPE})  workers=${WORKERS}  partitions=${PARTITIONS}"

python "$PREP" \
  --input "$INPUT_FILE" \
  --output-prefix "$OUTPUT_PREFIX" \
  --json-keys "$JSON_KEYS" \
  --tokenizer-type "$TOKENIZER_TYPE" \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --workers "$WORKERS" \
  --partitions "$PARTITIONS" \
  --append-eod
