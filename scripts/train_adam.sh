#!/usr/bin/env bash
set -euo pipefail

# Auto-source the cluster env loader so the user doesn't have to remember.
# Provides: cuda/13.2 on PATH (nvcc), LD_PRELOAD=libcublasLt.so.13 (TE
# symbol fix), and the older system cudnn-9.10.2 unloaded (torch wants
# the venv-bundled 9.19.0). All three are load-bearing for training to
# pass `import transformer_engine` and the first forward step.
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" || "${ARCH}" == "deepseek_v3_3b" ]]; then
  shift || true
else
  ARCH="llama3"
fi

case "${ARCH}" in
  llama3)
    FAMILY="llama3"
    DEFAULT_SCALE="300m"            # smallest dense scale; override with base/scale=...
    ;;
  deepseek_v3)
    FAMILY="deepseek_v3"
    DEFAULT_SCALE="deepseek_v3_proxy_small"
    ;;
  deepseek_v3_3b)
    # DeepSeek-V3-style 3B-total / ~520M-activated. Ported from
    # Megatron-poet/training_scripts/model_args/DeepSeek-3B.yaml.
    FAMILY="deepseek_v3"
    DEFAULT_SCALE="deepseek_v3_3b"
    ;;
  *)
    echo "Unknown architecture: ${ARCH}. Use llama3, deepseek_v3, or deepseek_v3_3b." >&2
    exit 2
    ;;
esac

# Extract --backend {megatron,torchtitan} (default megatron) from the passthrough
# args, route to the matching launcher, and inject backend=<value> so it lands in
# the resolved config (and the run name / torchtitan_sha).
BACKEND="megatron"
NEWARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --backend=*) BACKEND="${1#*=}"; shift ;;
    *) NEWARGS+=("$1"); shift ;;
  esac
done
set -- "${NEWARGS[@]}"

case "${BACKEND}" in
  megatron)   LAUNCHER="launchers.train_megatron"; BACKEND_OVERRIDE=() ;;
  torchtitan) LAUNCHER="launchers.train_torchtitan"; BACKEND_OVERRIDE=("backend=torchtitan") ;;
  *) echo "Unknown backend: ${BACKEND}. Use megatron or torchtitan." >&2; exit 2 ;;
esac

# Only inject the scale default if the user did not pass base/scale=...
USER_SET_SCALE="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
  esac
done

SCALE_ARGS=()
if [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]]; then
  SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
fi

RUN=(python -m "${LAUNCHER}" \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${BACKEND_OVERRIDE[@]}" \
  "cluster=h100_de" \
  "experiment=optim/adam" \
  "training.global_batch_size=512" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "$@")
if [[ "${SLM_DRYRUN_PRINT:-0}" == "1" ]]; then
  printf '%s ' "${RUN[@]}"; echo
else
  "${RUN[@]}"
fi
