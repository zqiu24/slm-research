#!/usr/bin/env bash
set -euo pipefail

# Dev launcher for Nemotron-H (base/family=nemotron_h — the Mamba2/attention/MLP
# hybrid, arXiv:2504.03624). Mirrors train_ngpt_dev.sh / train_adam_dev.sh in
# structure AND defaults (tiny 60m scale, 40x regime, gbs 1024, dev wandb,
# experiment=optim/adam, any "$@" override wins) so nemotron dev runs are
# directly comparable to the llama3 ones. NEMOTRON-SPECIFIC deviations — the
# llama3 dev defaults that do NOT transfer to the hybrid Mamba path:
#
#   * SCALE is the first positional arg (60m | 600m | 1b) — the whole point of
#     this script. Defaults to 60m (60m_nemotron_h, the tiny hybrid dev rung,
#     ~60M non-embedding, seq 256). A trailing base/scale=... still wins for
#     anything else (configs/base/scale/<scale>.yaml).
#   * micro_batch_size is scale-dependent: 60m -> 128 (matches the llama3 dev
#     launchers at seq 256), 600m/1b -> 4 (mbs 128 OOMs the larger hybrids at
#     seq 4096 on 80GB; 4 is the value train_bakeoff_600m.sh fits every family
#     with). A trailing training.micro_batch_size=N overrides.
#   * seq_length is left to the scale config (60m -> 256, 600m/1b -> 4096), so
#     each rung trains at its own protocol length. Override base.model.seq_length=N.
#   * NO transformer_impl=local. The mamba stack spec is built from TE layers
#     (TELayerNormColumnParallelLinear/TERowParallelLinear); leaving
#     transformer_impl unset uses Megatron's transformer_engine default, which is
#     what the bake-off (scripts/train_bakeoff_600m.sh) runs. Forcing `local`
#     breaks the mamba builder.
#   * NO tie_embeddings=false. The nemotron scale configs set tie_embeddings=true
#     (faithful to the reference); we respect that instead of forcing untied.
#
# Usage:
#   bash scripts/train_nemotron_dev.sh [60m|600m|1b] [hydra overrides...]
# Examples:
#   bash scripts/train_nemotron_dev.sh                       # 60m_nemotron_h
#   bash scripts/train_nemotron_dev.sh 600m                  # 600m_nemotron_h
#   bash scripts/train_nemotron_dev.sh 1b                    # 1b_nemotron_h
#   bash scripts/train_nemotron_dev.sh 600m base.model.seq_length=8192
#   bash scripts/train_nemotron_dev.sh 1b training.micro_batch_size=2
#   SLM_DRYRUN_PRINT=1 bash scripts/train_nemotron_dev.sh 1b  # print cmd, no GPU

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

# First positional arg selects the scale rung. Bare 60m/600m/1b map to the
# *_nemotron_h scale configs; a leading flag/override or empty leaves the
# default (60m) and is treated as a passthrough arg.
SCALE_SEL="60m"
case "${1:-}" in
  60m|60m_nemotron_h)   SCALE_SEL="60m";   shift ;;
  600m|600m_nemotron_h) SCALE_SEL="600m";  shift ;;
  1b|1b_nemotron_h)     SCALE_SEL="1b";    shift ;;
  ""|-*|*=*)            : ;;  # no scale token; keep default, pass arg through
  *) echo "Unknown scale: ${1}. Use 60m, 600m, or 1b (or a trailing base/scale=...)." >&2
     exit 2 ;;
esac

# Map the rung to its scale config + a memory-safe micro-batch. 60m mirrors the
# llama3 dev launchers (seq 256 from the config, mbs 128); the larger hybrid
# rungs run at seq 4096 where mbs 128 OOMs, so they default to 4.
case "${SCALE_SEL}" in
  60m)  DEFAULT_SCALE="60m_nemotron_h";  DEFAULT_MBS=128 ;;
  600m) DEFAULT_SCALE="600m_nemotron_h"; DEFAULT_MBS=4 ;;
  1b)   DEFAULT_SCALE="1b_nemotron_h";   DEFAULT_MBS=4 ;;
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
  "training.micro_batch_size=${DEFAULT_MBS}" \
  "training.save_enabled=true" \
  "wandb.project=slm-zeju-dev" \
  "$@")
if [[ "${SLM_DRYRUN_PRINT:-0}" == "1" ]]; then
  printf '%s ' "${RUN[@]}"; echo
else
  "${RUN[@]}"
fi
