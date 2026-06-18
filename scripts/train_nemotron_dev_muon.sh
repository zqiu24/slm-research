#!/usr/bin/env bash
set -euo pipefail

# Dev launcher for Nemotron-H trained with the vendored Kimi/Moonlight Muon
# optimizer (experiment=optim/muon_kimi) instead of Adam. Mirrors
# train_nemotron_dev.sh line-for-line EXCEPT for the experiment, so the
# Adam-vs-Muon ablation is directly comparable on the same hybrid backbone and
# training regime (tiny 60m scale, 40x regime, gbs 1024, dev wandb, any "$@"
# override wins). See train_nemotron_dev.sh / configs/base/family/nemotron_h.yaml
# for the hybrid Mamba2/attention/MLP architecture (arXiv:2504.03624).
#
# EXPERIMENTAL — Muon on the hybrid Mamba path is unvalidated. muon_kimi applies
# Muon to 2D non-embedding weights (internal AdamW on the rest) after the
# `model_unfuse_linears` patch + --unfuse-qkv/--unfuse-fc1 split the fused
# attention/MLP linears. Unlike POET it does NOT require transformer_impl=local
# (no impl gating), so we keep the mamba-mandatory TE impl. But whether the
# unfuse + Muon coverage behaves on the TE-built attention layers AND on the
# Mamba mixer in/out projections needs a GPU smoke before the loss is trusted
# (cf. the nGPT+Muon smoke in CHANGELOG). muon_kimi is single-process: it raises
# on tensor/pipeline parallelism, so keep tp=pp=1 (the dev scales already do).
#
# NEMOTRON-SPECIFIC deviations from the llama3 muon dev launcher
# (train_muon_dev.sh), all inherited from train_nemotron_dev.sh:
#   * SCALE is the first positional arg (60m | 600m | 1b); defaults to 60m
#     (60m_nemotron_h, ~60M non-embedding, seq 256). A trailing base/scale=...
#     still wins (configs/base/scale/<scale>.yaml).
#   * micro_batch_size is scale-dependent: 60m -> 128 (matches the llama3 dev
#     launchers at seq 256), 600m/1b -> 4 (mbs 128 OOMs the larger hybrids at
#     seq 4096 on 80GB). A trailing training.micro_batch_size=N overrides.
#   * seq_length is left to the scale config (60m -> 256, 600m/1b -> 4096).
#   * NO transformer_impl=local (the mamba stack spec is built from TE layers;
#     leaving it unset uses Megatron's transformer_engine default). NO
#     tie_embeddings=false (nemotron scale configs tie embeddings).
#
# Usage:
#   bash scripts/train_nemotron_dev_muon.sh [60m|600m|1b] [hydra overrides...]
# Examples:
#   bash scripts/train_nemotron_dev_muon.sh                  # 60m_nemotron_h
#   bash scripts/train_nemotron_dev_muon.sh 600m             # 600m_nemotron_h
#   bash scripts/train_nemotron_dev_muon.sh 60m optim.lr=2e-3
#   SLM_DRYRUN_PRINT=1 bash scripts/train_nemotron_dev_muon.sh  # print cmd, no GPU

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
  "experiment=optim/muon_kimi" \
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
