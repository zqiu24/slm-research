#!/usr/bin/env bash
# POET Lie-Orth update-RMS - in/out side-asymmetry (gamma) sweep, gamma=-0.25.
#
# On top of the §2.11 update-RMS champion (rho=0.30, lr=5e-3, max_angle=0.024),
# this varies ONLY the per-side angle exponent:
#   theta_side = lr * rho / RMS(W) * (d_side / sqrt(d_out*d_in)) ** gamma
# Geometric-mean ref => PURE redistribution (factor_out*factor_in == 1): each
# layer's average angle is unchanged; only the R_out vs R_in split moves.
#   gamma=0  -> symmetric champion (already run: norm 3.4765, mup 3.4758)
#   gamma>0  -> larger-dim side (e.g. fc1 d_out=4*d_in) rotates MORE
#   gamma<0  -> larger-dim side rotates LESS
#
# 2 runs: the two best inits at rho=0.30 - normalized (scale 2) and mup (alpha 4).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-6000}"
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
export SLM_POET_UPDATE_RMS_LOG_INTERVAL="${SLM_POET_UPDATE_RMS_LOG_INTERVAL:-10}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

EXTRA_ARGS=("$@")

GAMMA=-0.25
LABEL=gM25

HELD="base/scale=60m training_regime=ablation_40x \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.lie_ortho_max_angle=0.024 \
optim.poet.lie_ortho_rms_mode=weight optim.poet.scale=1.0 \
optim.poet.lie_ortho_update_rms=0.30 \
optim.poet.lie_ortho_update_rms_side_gamma=${GAMMA} \
optim.lr=0.005 cluster.gpus_per_node=8"

run () {
  local name="$1"; shift
  codexlog "$name" scripts/train_poet_lie_orth_update_rms.sh llama3 $HELD \
    "$@" experiment.name="$name" "${EXTRA_ARGS[@]}"
}

run "urms_${LABEL}_norm_r030_lr5" optim.poet.init_type=normalized optim.poet.init_scale=2.0
run "urms_${LABEL}_mup_r030_lr5"  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4

echo "=== POET update-RMS side-gamma sweep complete: gamma=${GAMMA}, norm+mup at rho0.30, 2 runs ==="
