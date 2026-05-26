#!/usr/bin/env bash
# End-to-end Nemotron preprocessing driver. Chains:
#   1. parquet -> jsonl (per-shard or whole-dir)
#   2. cat per-shard jsonl files into one
#   3. tokenize jsonl -> Megatron mmap (.bin/.idx)
#
# Stages may be skipped with --skip-stage {1|2|3} (repeatable).
#
# Each stage delegates to the wrappers added by the canonical plan
# (Task 7 of 2026-05-16-megatron-runner-data-port.md). This file does
# not duplicate their logic; it only orders them and gates them on flags.
set -euo pipefail

INPUT_DIR=""
JSONL_DIR=""
JSONL_MERGED=""
OUTPUT_PREFIX=""
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL=""
WORKERS="8"
IDX=""
SKIPS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  --input-dir DIR              parquet shards (stage 1 input)
  --jsonl-dir DIR              per-shard jsonl out / cat input
  --jsonl-merged FILE          concatenated jsonl out / tokenize in
  --output-prefix PREFIX       .bin/.idx output prefix (no extension)
  --tokenizer-type NAME        (default: HuggingFaceTokenizer)
  --tokenizer-model PATH       HF model dir / SentencePiece .model
  --workers N                  (default: 8)
  --idx N                      stage-1 parquet shard index (optional)
  --skip-stage {1|2|3}         skip that stage (repeatable)
  -h | --help                  this text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)        INPUT_DIR="$2"; shift 2 ;;
    --jsonl-dir)        JSONL_DIR="$2"; shift 2 ;;
    --jsonl-merged)     JSONL_MERGED="$2"; shift 2 ;;
    --output-prefix)    OUTPUT_PREFIX="$2"; shift 2 ;;
    --tokenizer-type)   TOKENIZER_TYPE="$2"; shift 2 ;;
    --tokenizer-model)  TOKENIZER_MODEL="$2"; shift 2 ;;
    --workers)          WORKERS="$2"; shift 2 ;;
    --idx)              IDX="$2"; shift 2 ;;
    --skip-stage)       SKIPS+=("$2"); shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

want_stage() {
  local s="$1"
  for x in "${SKIPS[@]:-}"; do [[ "$x" == "$s" ]] && return 1; done
  return 0
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${REPO_ROOT}"

if want_stage 1; then
  [[ -n "$INPUT_DIR" && -n "$JSONL_DIR" ]] || \
    { echo "stage 1 needs --input-dir and --jsonl-dir" >&2; exit 2; }
  mkdir -p "$JSONL_DIR"
  echo "[stage 1] parquet -> jsonl"
  INPUT_DIR="$INPUT_DIR" OUTPUT_DIR="$JSONL_DIR" \
    bash tools/preprocess_nemotron_parquet_to_jsonl.sh ${IDX:-}
fi

if want_stage 2; then
  [[ -n "$JSONL_DIR" && -n "$JSONL_MERGED" ]] || \
    { echo "stage 2 needs --jsonl-dir and --jsonl-merged" >&2; exit 2; }
  echo "[stage 2] cat $JSONL_DIR/*.jsonl -> $JSONL_MERGED"
  : > "$JSONL_MERGED"
  for f in "$JSONL_DIR"/*.jsonl; do cat "$f" >> "$JSONL_MERGED"; done
fi

if want_stage 3; then
  [[ -n "$JSONL_MERGED" && -n "$OUTPUT_PREFIX" && -n "$TOKENIZER_MODEL" ]] || \
    { echo "stage 3 needs --jsonl-merged, --output-prefix, --tokenizer-model" >&2; exit 2; }
  mkdir -p "$(dirname "$OUTPUT_PREFIX")"
  echo "[stage 3] tokenize -> ${OUTPUT_PREFIX}.{bin,idx}"
  INPUT_FILE="$JSONL_MERGED" \
  OUTPUT_PREFIX="$OUTPUT_PREFIX" \
  TOKENIZER_TYPE="$TOKENIZER_TYPE" \
  TOKENIZER_MODEL="$TOKENIZER_MODEL" \
  WORKERS="$WORKERS" \
    bash tools/preprocess_nemotron_tokenize.sh
fi

echo "[done]"
