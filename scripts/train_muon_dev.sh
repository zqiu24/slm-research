#!/usr/bin/env bash
set -euo pipefail

# Dev variant of train_muon.sh: same harness, but defaults to the tiny 60m
# scale (configs/base/scale/60m.yaml, hidden=512, ~61M non-embedding params)
# for fast local iteration instead of 300m. Untied embeddings are forced on
# (overridable). Everything else is identical to train_muon.sh, and any "$@"
# override still wins.

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
    DEFAULT_SCALE="60m"            # tiny dev scale; override with base/scale=...
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

# Inject dev defaults unless overridden on the command line:
#   scale=60m (tiny dev scale) and the 40x-tokens-per-param dev regime
#   (training_regime=ablation_40x — 40 * non_embedding_params tokens, 2x the
#   base 20x default).
USER_SET_SCALE="no"
USER_SET_REGIME="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
    training_regime=*) USER_SET_REGIME="yes" ;;
  esac
done

SCALE_ARGS=()
if [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]]; then
  SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
fi

REGIME_ARGS=()
if [[ "${USER_SET_REGIME}" == "no" ]]; then
  REGIME_ARGS=("training_regime=ablation_40x")
fi

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${REGIME_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=optim/muon_hybrid" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=128" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "base.model.tie_embeddings=false" \
  "wandb.project=slm-zeju-dev" \
  "$@"
