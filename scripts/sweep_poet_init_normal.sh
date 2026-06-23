#!/usr/bin/env bash
# POET init sweep — NORMAL / raw spectrum (init_type=none), operating-NORM axis.
#
# init_type=none keeps Megatron's native init: the Marchenko-Pastur Gaussian spectrum AND
# the residual 1/sqrt(2L) downscale on proj/fc2 (so it is type-structured, large condition
# number). Native per-element RMS is ~0.016 (§2.7) — ~3x BELOW normalized's 0.044 and well
# below the Adam/Muon equilibrium (~0.045-0.056). Because POET can't grow the frozen norm,
# raw init may simply be under-scaled; init_scale lifts it across the same band as the other
# scripts. The s275 cell (~0.044) is a matched-NORM A/B vs normalized@1.0 → isolates whether
# the raw (residual-structured, ill-conditioned) SHAPE helps or hurts at equal norm.
#
# Baseline to beat: nestON_lr4 = val/loss 3.5160 (normalized @ scale 1.0).
# All cells ride the champion recipe (lr 4e-3 / eff∠ 0.016, head-off, alt, Nesterov b1.95).
#   bash scripts/sweep_poet_init_normal.sh        # default GPU 0-3 (dp=4, global_batch=1024), sequential
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # default first half-node (GPU 0-3)
export MASTER_PORT="${MASTER_PORT:-6000}"                         # torchrun rendezvous (override + CUDA_VISIBLE_DEVICES=4,5,6,7 to pair a 2nd lane)
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe; only optim.poet.init_scale varies (init_type pinned none = raw).
HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 optim.poet.lie_ortho_c=8 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=none \
cluster.gpus_per_node=4"

run () {  # <name> <init_scale>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.init_scale="$2" experiment.name="$1"
}

run init_none_s100 1.0    # row_rms ~0.016  (raw native baseline)
run init_none_s150 1.5    # row_rms ~0.024
run init_none_s275 2.75   # row_rms ~0.044  (matched-NORM A/B vs normalized@1.0)
run init_none_s400 4.0    # row_rms ~0.064  (~Muon equilibrium)
run init_none_s550 5.5    # row_rms ~0.088  (matched to normalized@2.0)

echo "=== POET init/normal(raw) sweep complete (baseline 3.5160) ==="
