#!/usr/bin/env bash
set -euo pipefail

# poet_lie_orth_in_only: pure one-sided POETX (InOnlyPOETXLinear, trains ONLY oft_R_in;
# oft_R_out stays identity) on the champion lie_ortho recipe. Same harness as
# train_poet_lie_orth_alt_x.sh, experiment swapped (side FIXED, not alternating).

case " $* " in
  *" --backend torchtitan "*|*" --backend=torchtitan "*)
    echo "This optimizer is not yet supported on torchtitan (milestone 1 is AdamW only)." >&2
    exit 2 ;;
esac

SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" ]]; then
  shift || true
else
  ARCH="llama3"
fi

case "${ARCH}" in
  llama3) FAMILY="llama3"; DEFAULT_SCALE="60m" ;;
  deepseek_v3) FAMILY="deepseek_v3"; DEFAULT_SCALE="deepseek_v3_proxy_small" ;;
  *) echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2; exit 2 ;;
esac

USER_SET_SCALE="no"; USER_SET_SEQ="no"; USER_SET_SCHED="no"; USER_SET_REGIME="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
    base.model.seq_length=*) USER_SET_SEQ="yes" ;;
    scheduler=*) USER_SET_SCHED="yes" ;;
    training_regime=*) USER_SET_REGIME="yes" ;;
  esac
done

SCALE_ARGS=(); [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]] && SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
REGIME_ARGS=(); [[ "${USER_SET_REGIME}" == "no" ]] && REGIME_ARGS=("training_regime=ablation_40x")
SEQ_ARGS=(); [[ "${USER_SET_SEQ}" == "no" ]] && SEQ_ARGS=("base.model.seq_length=256")
SCHED_ARGS=(); [[ "${USER_SET_SCHED}" == "no" ]] && SCHED_ARGS=("scheduler=cosine_poet")

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${REGIME_ARGS[@]}" \
  "${SEQ_ARGS[@]}" \
  "${SCHED_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=optim/poet_lie_orth_in_only" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=128" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "base.model.tie_embeddings=false" \
  "optim.weight_decay=0.1" \
  "wandb.project=slm-zeju-dev" \
  "$@"
