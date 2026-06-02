#!/usr/bin/env bash
set -euo pipefail

# Probe 0B convenience wrapper: run a normal POET dev run with the ∂f/∂Q
# conditioning probe turned ON, then delegate to train_poet_dev.sh.
#
# Why a wrapper (not flags baked into train_poet_dev.sh): the conditioning probe
# is opt-in diagnostics — it logs per-block singular-value stats to W&B every
# SLM_POET_GRAD_CONDITIONING_INTERVAL steps and would pollute every normal POET
# dev run if it were always on. Exporting the env vars HERE (rather than as a
# `VAR=1 codexlog ...` prefix) is also what makes them propagate reliably: they
# land in this script's process, so the python launcher and its torchrun workers
# inherit them via os.environ.copy() (launchers/train_megatron.py).
#
# Usage (one arm per invocation; each grabs all 8 GPUs):
#   codexlog s0b_cond     bash scripts/train_poet_dev_cond.sh
#   codexlog s0b_cond_exp bash scripts/train_poet_dev_cond.sh optim.poet.parameterization=exp
# Override the capture cadence with SLM_POET_GRAD_CONDITIONING_INTERVAL=250 ... etc.

# Recompute chain: the default cayley path uses the fast (non-recompute) chain,
# which OOMs at 60m/mbs128 (the per-token rotation activations leave no room for
# the fp32 CE-logits buffer). POET_MEM_EFFICIENT=1 forces the recompute chain so
# the conditioning run fits. (exp already defaults to recompute; harmless no-op.)
export POET_MEM_EFFICIENT=1

# Turn on the conditioning probe (src/patches/poet_grad_conditioning.py).
export SLM_POET_GRAD_CONDITIONING=1
export SLM_POET_GRAD_CONDITIONING_INTERVAL="${SLM_POET_GRAD_CONDITIONING_INTERVAL:-500}"

SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$SLM_REPO/scripts/train_poet_dev.sh" "$@"
