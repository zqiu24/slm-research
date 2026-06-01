#!/usr/bin/env bash
set -euo pipefail

# torchtitan is AdamW-only in milestone 1; reject --backend torchtitan here so the
# same flag fails fast on this non-AdamW wrapper (see scripts/train_adam.sh).
case " $* " in
  *" --backend torchtitan "*|*" --backend=torchtitan "*)
    echo "This optimizer is not yet supported on torchtitan (milestone 1 is AdamW only)." >&2
    exit 2 ;;
esac

# Auto-source the cluster env loader so the user doesn't have to remember.
# Provides: cuda/13.2 on PATH (nvcc), LD_PRELOAD=libcublasLt.so.13 (TE
# symbol fix), and the older system cudnn-9.10.2 unloaded (torch wants
# the venv-bundled 9.19.0). All three are load-bearing for training to
# pass `import transformer_engine` and the first forward step.
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" ]]; then
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
  *)
    echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2
    exit 2
    ;;
esac

# Inject debug defaults unless overridden on the command line:
#   scale=300m (smallest dense scale), seq_length=256 (short ctx for fast debug).
USER_SET_SCALE="no"
USER_SET_SEQ="no"
USER_SET_SCHED="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
    base.model.seq_length=*) USER_SET_SEQ="yes" ;;
    scheduler=*) USER_SET_SCHED="yes" ;;
  esac
done

SCALE_ARGS=()
if [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]]; then
  SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
fi

SEQ_ARGS=()
if [[ "${USER_SET_SEQ}" == "no" ]]; then
  SEQ_ARGS=("base.model.seq_length=256")
fi

# POET defaults to a 1% min-LR floor (scheduler/cosine_poet.yaml,
# min_lr_ratio=0.01) instead of the global cosine default's 10%, matching the
# POET reference recipe. Override with scheduler=... on the command line.
SCHED_ARGS=()
if [[ "${USER_SET_SCHED}" == "no" ]]; then
  SCHED_ARGS=("scheduler=cosine_poet")
fi

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${SEQ_ARGS[@]}" \
  "${SCHED_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=optim/poet" \
  "training.global_batch_size=512" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "$@"
