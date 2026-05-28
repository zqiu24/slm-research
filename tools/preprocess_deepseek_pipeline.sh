#!/usr/bin/env bash
# End-to-end DeepSeek-V3 data-prep driver. Chains:
#   1. parquet -> jsonl shards   (tools/preprocess_parquet_to_jsonl.py)
#   2. cat shards -> one merged jsonl
#   3. tokenize jsonl -> Megatron mmap (.bin/.idx) with the DeepSeek-V3 tokenizer
#      via tools/preprocess_deepseek_tokenize.sh (merged + --partitions strategy)
#
# Stage 1 is auto-skipped when --input-dir already holds .jsonl (and no .parquet).
# Any stage can be force-skipped with --skip-stage {1|2|3} (repeatable).
#
# Run under the same conda env as the DeepSeek trainer (e.g. `megatron-lm-014`).
set -euo pipefail

INPUT_DIR=""
JSONL_DIR=""
JSONL_MERGED=""
OUTPUT_PREFIX=""
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fast/fast/zqiu/hf_models/DeepSeek-V3-tokenizer}"
TEXT_COLUMN="text"
WORKERS="32"
PARTITIONS="1"
MEGATRON_POET_ROOT="${MEGATRON_POET_ROOT:-/lustre/fast/fast/zqiu/tmp/Megatron-poet}"
SKIPS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  --input-dir DIR          parquet shards (stage 1 in) OR jsonl shards (stage 1 auto-skipped)
  --jsonl-dir DIR          per-shard jsonl out / cat input   (default: <out-dir>/jsonl)
  --jsonl-merged FILE      merged jsonl out / tokenize in     (default: <out-dir>/merged.jsonl)
  --output-prefix PREFIX   final .bin/.idx prefix (no extension)   [required]
  --tokenizer-model PATH   HF tokenizer dir (default: DeepSeek-V3-tokenizer)
  --text-column NAME       parquet text column (default: text)
  --workers N              tokenize workers (default: 32; must be a multiple of partitions)
  --partitions N           tokenize partitions (default: 1)
  --megatron-poet-root DIR Megatron-poet checkout (default: $MEGATRON_POET_ROOT)
  --skip-stage {1|2|3}     skip that stage (repeatable)
  -h | --help              this text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)          INPUT_DIR="$2"; shift 2 ;;
    --jsonl-dir)          JSONL_DIR="$2"; shift 2 ;;
    --jsonl-merged)       JSONL_MERGED="$2"; shift 2 ;;
    --output-prefix)      OUTPUT_PREFIX="$2"; shift 2 ;;
    --tokenizer-model)    TOKENIZER_MODEL="$2"; shift 2 ;;
    --text-column)        TEXT_COLUMN="$2"; shift 2 ;;
    --workers)            WORKERS="$2"; shift 2 ;;
    --partitions)         PARTITIONS="$2"; shift 2 ;;
    --megatron-poet-root) MEGATRON_POET_ROOT="$2"; shift 2 ;;
    --skip-stage)         SKIPS+=("$2"); shift 2 ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

want_stage() {
  local s="$1"
  for x in "${SKIPS[@]:-}"; do [[ "$x" == "$s" ]] && return 1; done
  return 0
}

[[ -n "$OUTPUT_PREFIX" ]] || { echo "--output-prefix is required" >&2; usage >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="$(dirname "$OUTPUT_PREFIX")"
JSONL_DIR="${JSONL_DIR:-${OUT_DIR}/jsonl}"
JSONL_MERGED="${JSONL_MERGED:-${OUT_DIR}/merged.jsonl}"

# Auto-skip stage 1 when input is already jsonl (no parquet present).
if want_stage 1 && [[ -n "$INPUT_DIR" ]]; then
  shopt -s nullglob
  parquet_files=("$INPUT_DIR"/**/*.parquet "$INPUT_DIR"/*.parquet)
  jsonl_files=("$INPUT_DIR"/*.jsonl)
  shopt -u nullglob
  if [[ ${#parquet_files[@]} -eq 0 && ${#jsonl_files[@]} -gt 0 ]]; then
    echo "[stage 1] input-dir holds jsonl, no parquet -> auto-skipping stage 1; using it as --jsonl-dir"
    JSONL_DIR="$INPUT_DIR"
    SKIPS+=("1")
  fi
fi

# --- Stage 1: parquet -> jsonl shards ----------------------------------------
if want_stage 1; then
  [[ -n "$INPUT_DIR" ]] || { echo "stage 1 needs --input-dir" >&2; exit 2; }
  mkdir -p "$JSONL_DIR"
  echo "[stage 1] parquet -> jsonl ($INPUT_DIR -> $JSONL_DIR)"
  python -m tools.preprocess_parquet_to_jsonl \
    --input "$INPUT_DIR" \
    --output-dir "$JSONL_DIR" \
    --prefix deepseek \
    --text-column "$TEXT_COLUMN"
fi

# --- Stage 2: cat shards -> merged jsonl -------------------------------------
if want_stage 2; then
  [[ -d "$JSONL_DIR" ]] || { echo "stage 2 needs $JSONL_DIR to exist" >&2; exit 2; }
  echo "[stage 2] cat $JSONL_DIR/*.jsonl -> $JSONL_MERGED"
  mkdir -p "$(dirname "$JSONL_MERGED")"
  : > "$JSONL_MERGED"
  shopt -s nullglob
  shards=("$JSONL_DIR"/*.jsonl)
  shopt -u nullglob
  [[ ${#shards[@]} -gt 0 ]] || { echo "no jsonl shards in $JSONL_DIR" >&2; exit 2; }
  for f in "${shards[@]}"; do cat "$f" >> "$JSONL_MERGED"; done
fi

# --- Stage 3: tokenize merged jsonl -> .bin/.idx -----------------------------
if want_stage 3; then
  [[ -f "$JSONL_MERGED" ]] || { echo "stage 3 needs $JSONL_MERGED" >&2; exit 2; }
  echo "[stage 3] tokenize -> ${OUTPUT_PREFIX}.{bin,idx}"
  INPUT_FILE="$JSONL_MERGED" \
  OUTPUT_PREFIX="$OUTPUT_PREFIX" \
  TOKENIZER_MODEL="$TOKENIZER_MODEL" \
  WORKERS="$WORKERS" \
  PARTITIONS="$PARTITIONS" \
  MEGATRON_POET_ROOT="$MEGATRON_POET_ROOT" \
    bash tools/preprocess_deepseek_tokenize.sh
fi

echo "[done] ${OUTPUT_PREFIX}.bin / ${OUTPUT_PREFIX}.idx"
