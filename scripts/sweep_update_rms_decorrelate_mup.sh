#!/usr/bin/env bash
# Stage 1: cross-side decorrelation ("the split, with a scale") on the SYMMETRIC update-RMS
# champion — MUP init (init_type=mup_normalized, mup_alpha=4, ρ0.30, side_γ=0, lr5).
# Recipe mirrors scripts/sweep_poet_lie_orth_update_rms_mup.sh (the run that produced the
# baseline), with ρ fixed at its optimum and λ swept instead.
#
# Baseline to beat: val 3.4758 (POET_dev.md §2.11/§2.6). λ swept {0.25,0.50,0.75}; mode
# symmetric (fires every step), renorm on (direction-only change), all layers
# (cos_threshold=0). λ=1.0 EXCLUDED (catastrophic in §J.3; NB `decorrelate_lambda` defaults
# to 1.0, so every arm sets it explicitly). 3 runs, SEQUENTIAL — split across GPUs to parallelize.
#
# TRIPWIRE: each run's startup log must print
#   [POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=symmetric, lambda=<L>, renorm=True,
#          cos_threshold=0.0, alternating=True)
# If lambda/mode/renorm/alternating do NOT match the arm, an override was dropped — kill + fix.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
# Log the cos(D_out,D_in) trajectory so the overlap is visibly driven down during the run.
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

HELD="base/scale=60m training_regime=ablation_40x \
  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.scale=1.0 \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.0 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true \
  optim.lr=0.005 cluster.gpus_per_node=8"

for LAM in 0.25 0.50 0.75; do
  TAG="${LAM/./p}"                       # 0.25 -> 0p25
  NAME="urms_decorr_mup_l${TAG}"
  echo ">>> ${NAME} (mup, rho0.30, lambda=${LAM}) starting — baseline 3.4758"
  scripts/train_poet_lie_orth_update_rms.sh llama3 ${HELD} \
    optim.poet.lie_ortho_decorrelate=true \
    optim.poet.lie_ortho_decorrelate_mode=symmetric \
    optim.poet.lie_ortho_decorrelate_renorm=true \
    optim.poet.lie_ortho_decorrelate_lambda="${LAM}" \
    optim.poet.lie_ortho_decorrelate_cos_threshold=0.0 \
    experiment.name="${NAME}" \
    2>&1 | tee "${CODEX_LOG_DIR}/${NAME}.log"
  echo "<<< ${NAME} done (status ${PIPESTATUS[0]}) — log: ${CODEX_LOG_DIR}/${NAME}.log"
done
echo "=== update-RMS decorrelation (MUP α4, ρ0.30) sweep complete: λ {0.25,0.50,0.75} vs baseline 3.4758 ==="
