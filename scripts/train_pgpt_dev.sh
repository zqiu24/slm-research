#!/usr/bin/env bash
set -euo pipefail

# Dev launcher for pgpt (experiment=arch/pgpt): the nGPT hypersphere
# architecture with the explicit per-step weight projection REMOVED, trained
# with POET (frozen base weight + trained block-orthogonal delta). Distinct from
# train_ngpt_dev_poet.sh, which keeps vanilla nGPT (and its per-step renorm) and
# only swaps the optimizer. Mirrors train_ngpt_dev.sh EXCEPT for the experiment
# (and the POET cosine_poet scheduler default, as in train_poet_dev.sh), so the
# comparison stays on the same llama3 backbone and training regime: tiny 60m
# scale, 40x-tokens-per-param regime, gbs 1024 / mbs 128, local transformer impl,
# untied embeddings, dev wandb project. Any "$@" override still wins.
#
# EXPERIMENTAL -- GPU-smoke before trusting the loss. Unlike ngpt_poet, pgpt has
# NO train_step renorm, so poet_merge_step does not collide and is included
# (inert at merge_period=0; flip optim.poet.merge_period>0 to enable merges). The
# only surviving renorm is embedding+lm_head, installed by pgpt_optimizer_setup's
# optimizer.step hook. Confirm "[pgpt] optimizer setup ..." and "[nGPT] applied
# spec" (pgpt reuses that log) AND POET orbit logs all appear, and that ortho_err
# stays bounded. See configs/experiments/arch/pgpt.yaml for the full rationale.

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
# Skip the CUDA env loader when only printing the resolved command — dry-print
# needs no GPU, and the loader derefs $HOME under `set -u` (fails in a clean env).
if [[ "${SLM_DRYRUN_PRINT:-0}" != "1" ]]; then
  source "$SLM_REPO/load_cuda13_2_nccl_env.sh"
fi

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
#   scale=60m (tiny dev scale), the 40x-tokens-per-param dev regime
#   (training_regime=ablation_40x — 40 * non_embedding_params tokens, 2x the
#   base 20x default), and POET's 1% min-LR cosine floor (scheduler=cosine_poet,
#   matching the POET reference recipe / train_poet_dev.sh).
USER_SET_SCALE="no"
USER_SET_REGIME="no"
USER_SET_SCHED="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
    training_regime=*) USER_SET_REGIME="yes" ;;
    scheduler=*) USER_SET_SCHED="yes" ;;
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

# POET defaults to a 1% min-LR floor (scheduler/cosine_poet.yaml,
# min_lr_ratio=0.01) instead of the global cosine default's 10%, matching the
# POET reference recipe. Override with scheduler=... (e.g. scheduler=cosine to
# match the nGPT baseline exactly) on the command line.
SCHED_ARGS=()
if [[ "${USER_SET_SCHED}" == "no" ]]; then
  SCHED_ARGS=("scheduler=cosine_poet")
fi

# NB: POET never uses the Megatron distributed optimizer — its optimizer builder
# (src/optim/poet.py get_megatron_poet_optimizer) rejects it unconditionally, so
# the launcher declines to emit --use-distributed-optimizer for any POET run
# (_distributed_optimizer_supported returns False; regression-guarded by
# tests/unit/test_megatron_args.py). No per-run override is needed here.
RUN=(python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${REGIME_ARGS[@]}" \
  "${SCHED_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=arch/pgpt" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=128" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "base.model.tie_embeddings=false" \
  "wandb.project=slm-zeju-dev" \
  "$@")
if [[ "${SLM_DRYRUN_PRINT:-0}" == "1" ]]; then
  printf '%s ' "${RUN[@]}"; echo
else
  "${RUN[@]}"
fi
