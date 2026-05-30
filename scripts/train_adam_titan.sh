#!/usr/bin/env bash
set -euo pipefail

# Adam(W) training on the torchtitan backend.
#
# Torchtitan-default sibling of scripts/train_adam.sh: equivalent to
#   scripts/train_adam.sh <arch> --backend torchtitan <overrides...>
# but without having to remember the flag. Resolves the same 6-axis slm config
# and launches torchtitan via launchers.train_torchtitan, which emits
# <run_dir>/torchtitan.toml, builds `torchrun -m torchtitan.train`, and injects
# the slm wiring (slm_<family> TrainSpec, slm_<scale> flavor, Megatron-indexed
# dataloader, WSD scheduler) through torchtitan's experimental.custom_import hook
# — the vendored submodule is never edited.
#
# torchtitan is AdamW-only in milestone 1, so Adam is the supported optimizer.

# Auto-source the cluster env loader so the user doesn't have to remember.
# Provides: cuda/13.2 on PATH (nvcc), LD_PRELOAD=libcublasLt.so.13 (TE
# symbol fix), and the older system cudnn-9.10.2 unloaded (torch wants
# the venv-bundled 9.19.0). All three are load-bearing for training to
# pass `import transformer_engine` and the first forward step.
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Skip the CUDA env loader when only printing the resolved command — dry-print
# needs no GPU, and the loader derefs $HOME under `set -u` (fails in a clean env).
if [[ "${SLM_DRYRUN_PRINT:-0}" != "1" ]]; then
  source "$SLM_REPO/load_cuda13_2_nccl_env.sh"
fi

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

# This wrapper is hard-wired to torchtitan. Consume any --backend flag so it
# never leaks into the launcher's hydra overrides, and reject a non-torchtitan
# value (use scripts/train_adam.sh for the megatron backend).
NEWARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      if [[ "${2:-}" != "torchtitan" ]]; then
        echo "train_adam_titan.sh is torchtitan-only; use scripts/train_adam.sh for --backend ${2:-}." >&2
        exit 2
      fi
      shift 2 ;;
    --backend=*)
      if [[ "${1#*=}" != "torchtitan" ]]; then
        echo "train_adam_titan.sh is torchtitan-only; use scripts/train_adam.sh for --backend ${1#*=}." >&2
        exit 2
      fi
      shift ;;
    *) NEWARGS+=("$1"); shift ;;
  esac
done
set -- "${NEWARGS[@]}"

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

RUN=(python -m launchers.train_torchtitan \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "backend=torchtitan" \
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
