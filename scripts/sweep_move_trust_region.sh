#!/usr/bin/env bash
# M1 realized-movement trust region — Phase 0 (measure) + Phase 1 (2x2 vs decorr).
# Base recipe = the §2.15c decorrelation champion (mup a4, side_gamma +0.25, rho0.30,
# lr5, decorrelate lambda0.25 renorm=off), cloned verbatim from
# scripts/sweep_update_rms_decorrelate_gp25.sh. The M1 rho_move grid (RA/RB) is set from
# the Phase-0 p50/p90 of poet_move/ratio_* (read off wandb after the measure arm finishes).
#
# Usage: bash scripts/sweep_move_trust_region.sh <arm>
#   arm in: measure | clip_off_rA | clip_off_rB | clip_decorr_rA | clip_decorr_rB
#
# TRIPWIRE: every clip/measure startup must print the optimizer's move-control banner with
# the arm's mode/rho; the decorr arms must also print the CROSS-SIDE DECORRELATION ON line
# (mode=symmetric, lambda=0.25, renorm=false). A mismatch => an override was dropped — kill+fix.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

ARM="${1:?usage: sweep_move_trust_region.sh <arm>}"

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
# Log the cos(D_out,D_in) trajectory so the overlap is visibly tracked during the run.
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

# --- champion base (COPIED verbatim from sweep_update_rms_decorrelate_gp25.sh, mup arm) ---
# Decorrelation flags are split out below (DECORR_ON/DECORR_OFF) so each arm can toggle them.
BASE_OVERRIDES="base/scale=60m training_regime=ablation_40x \
  optim.poet.scale=1.0 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.25 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true \
  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 \
  optim.lr=0.005 cluster.gpus_per_node=8"

# §2.15c champion decorrelation (symmetric, partial-lambda 0.25, renorm off) vs OFF.
DECORR_ON="optim.poet.lie_ortho_decorrelate=true \
  optim.poet.lie_ortho_decorrelate_mode=symmetric \
  optim.poet.lie_ortho_decorrelate_cos_threshold=0.0 \
  optim.poet.lie_ortho_decorrelate_lambda=0.25 \
  optim.poet.lie_ortho_decorrelate_renorm=false"
DECORR_OFF="optim.poet.lie_ortho_decorrelate=false"

# rho_move grid RA/RB — SET THESE from Phase 0 (the `measure` arm's wandb
# poet_move/ratio_p50 and ratio_p90) BEFORE running any clip arm.
RA="__SET_FROM_PHASE0_P50__"
RB="__SET_FROM_PHASE0_P90__"

case "$ARM" in
  measure)        MOVE="optim.poet.lie_move_control_mode=measure" ;             DECORR="$DECORR_ON"  ;;
  clip_off_rA)    MOVE="optim.poet.lie_move_control_mode=clip optim.poet.lie_move_budget_rho=${RA}" ; DECORR="$DECORR_OFF" ;;
  clip_off_rB)    MOVE="optim.poet.lie_move_control_mode=clip optim.poet.lie_move_budget_rho=${RB}" ; DECORR="$DECORR_OFF" ;;
  clip_decorr_rA) MOVE="optim.poet.lie_move_control_mode=clip optim.poet.lie_move_budget_rho=${RA}" ; DECORR="$DECORR_ON"  ;;
  clip_decorr_rB) MOVE="optim.poet.lie_move_control_mode=clip optim.poet.lie_move_budget_rho=${RB}" ; DECORR="$DECORR_ON"  ;;
  *) echo "unknown arm: $ARM" >&2 ; exit 1 ;;
esac

NAME="mtr_${ARM}"
echo ">>> ${NAME} move='${MOVE}' decorr='${DECORR}' starting"
scripts/train_poet_lie_orth_update_rms.sh llama3 ${BASE_OVERRIDES} ${DECORR} ${MOVE} \
  experiment.name="${NAME}" \
  2>&1 | tee "${CODEX_LOG_DIR}/${NAME}.log"
echo "<<< ${NAME} done (status ${PIPESTATUS[0]}) — log: ${CODEX_LOG_DIR}/${NAME}.log"
echo "=== M1 trust-region arm ${ARM} complete; anchors: 3.4745 (no-decorr) / 3.4686 (record) ==="
