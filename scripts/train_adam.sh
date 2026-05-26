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
    DEFAULT_SCALE=""                # inherit launch config default (1_2b)
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

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "experiment=champion" \
  "base.model.seq_length=256" \
  "training.seq_length=256" \
  "base.model.transformer_impl=local" \
  "$@"
