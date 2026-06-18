#!/usr/bin/env bash
set -euo pipefail

# Dev launcher for Nemotron-H (base/family=nemotron_h — the Mamba2/attention/MLP
# hybrid, arXiv:2504.03624). Mirrors train_ngpt_dev.sh / train_adam_dev.sh in
# structure (auto-source env, inject scale+regime defaults, dry-run print, dev
# wandb, any "$@" override wins) but with NEMOTRON-SPECIFIC settings — the llama3
# dev defaults do not transfer to the hybrid Mamba path:
#
#   * SCALE is the first positional arg (600m | 1b) — the whole point of this
#     script. There is no tiny <600M nemotron scale, so 600m_nemotron_h is the
#     dev default; pass `1b` for 1b_nemotron_h. A trailing base/scale=... still
#     wins for anything else (configs/base/scale/<scale>.yaml).
#   * NO transformer_impl=local. The mamba stack spec is built from TE layers
#     (TELayerNormColumnParallelLinear/TERowParallelLinear); leaving
#     transformer_impl unset uses Megatron's transformer_engine default, which is
#     what the bake-off (scripts/train_bakeoff_600m.sh) runs. Forcing `local`
#     breaks the mamba builder.
#   * NO tie_embeddings=false. The nemotron scale configs set tie_embeddings=true
#     (faithful to the reference); we respect that instead of forcing untied.
#   * micro_batch_size=4, NOT 128. 128 OOMs the 600M hybrid at seq 4096 on 80GB;
#     4 is the conservative value train_bakeoff_600m.sh fits every family with.
#     Raise via a trailing training.micro_batch_size=N if a scale has headroom.
#   * experiment=optim/adam (nemotron is not an nGPT/optimizer experiment).
#
# Usage:
#   bash scripts/train_nemotron_dev.sh [600m|1b] [hydra overrides...]
# Examples:
#   bash scripts/train_nemotron_dev.sh                       # 600m_nemotron_h
#   bash scripts/train_nemotron_dev.sh 1b                    # 1b_nemotron_h
#   bash scripts/train_nemotron_dev.sh 600m base.model.seq_length=8192
#   bash scripts/train_nemotron_dev.sh 1b training.micro_batch_size=2
#   SLM_DRYRUN_PRINT=1 bash scripts/train_nemotron_dev.sh 1b  # print cmd, no GPU
#
# NOTE: unlike the tiny-60m llama3 dev scripts, the smallest nemotron scale is
# 600M, so the default ablation_40x regime (~24B tokens) is a REAL run, not a
# fast smoke. For a quick "does it build + step" check, cap iterations, e.g.:
#   bash scripts/train_nemotron_dev.sh 600m training.train_iters=10

# Mamba/hybrid cannot run on the AdamW-only torchtitan backend; fail fast so the
# flag errors here rather than deep in the launcher (mirrors train_ngpt_dev.sh).
case " $* " in
  *" --backend torchtitan "*|*" --backend=torchtitan "*)
    echo "Nemotron-H (hybrid Mamba) is not supported on torchtitan (AdamW-only)." >&2
    exit 2 ;;
esac

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

FAMILY="nemotron_h"

# First positional arg selects the scale rung. Bare 600m/1b map to the
# *_nemotron_h scale configs; a leading flag/override or empty leaves the
# default and is treated as a passthrough arg.
DEFAULT_SCALE="600m_nemotron_h"
case "${1:-}" in
  600m|600m_nemotron_h) DEFAULT_SCALE="600m_nemotron_h"; shift ;;
  1b|1b_nemotron_h)     DEFAULT_SCALE="1b_nemotron_h";   shift ;;
  ""|-*|*=*)            : ;;  # no scale token; keep default, pass arg through
  *) echo "Unknown scale: ${1}. Use 600m or 1b (or a trailing base/scale=...)." >&2
     exit 2 ;;
esac

# Inject dev defaults unless overridden on the command line:
#   base/scale=<DEFAULT_SCALE> and the 40x-tokens-per-param dev regime
#   (training_regime=ablation_40x — 40 * non_embedding_params tokens).
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

RUN=(python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${REGIME_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=optim/adam" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=4" \
  "base.model.seq_length=4096" \
  "training.save_enabled=true" \
  "wandb.project=slm-zeju-dev" \
  "$@")
if [[ "${SLM_DRYRUN_PRINT:-0}" == "1" ]]; then
  printf '%s ' "${RUN[@]}"; echo
else
  "${RUN[@]}"
fi
