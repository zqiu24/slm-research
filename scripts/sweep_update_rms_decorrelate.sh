#!/usr/bin/env bash
# Stage 1: cross-side decorrelation ("the split, with a scale") stacked on the SYMMETRIC
# update-RMS champion (POET_dev.md §2.11/§2.12). Baseline to beat: the symmetric
# mup_normalized a4 / rho0.30 / side_gamma=0 / lr5 run = val 3.4758 (§2.11). If the §J.3
# lambda=0.5 win (-0.0070) carries over -> ~3.4688, a new POET record.
#
# Held recipe = the symmetric baseline; only the decorrelation knobs move. The inactive
# side's direction is sourced from its maintained momentum (lie_m) and the active write is
# projected off it (cross-step over-spend control). mode=symmetric (fires every step),
# renorm=true (direction-only change), all layers (cos_threshold=0). lambda is swept; 1.0
# is deliberately EXCLUDED (catastrophic in §J.3 via the renorm pathology).
#
# Runs the three lambda arms SEQUENTIALLY; background or split across GPUs to parallelize.
#
# TRIPWIRE: each run's startup log must print
#   [POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=symmetric, lambda=<L>, renorm=True,
#          cos_threshold=0.0, alternating=True)
# If lambda/mode/renorm/alternating do NOT match the arm, an override was silently dropped
# — kill it and fix before trusting the result.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
# Log the cos(D_out,D_in) trajectory so the overlap is visibly driven down during the run.
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

HELD="scheduler=cosine_poet training_regime=ablation_40x \
  optim.lr=0.005 \
  optim.poet.lie_ortho_update_rms=0.30 \
  optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.0 \
  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.scale=1.0 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true optim.poet.lie_ortho_rms_mode=weight"

for LAM in 0.25 0.50 0.75; do
  TAG="${LAM/./p}"                       # 0.25 -> 0p25
  NAME="urms_decorr_sym_renorm_l${TAG}"
  echo ">>> ${NAME} (lambda=${LAM}) starting"
  codexlog "${NAME}" scripts/train_poet_lie_orth_update_rms.sh llama3 ${HELD} \
    optim.poet.lie_ortho_decorrelate=true \
    optim.poet.lie_ortho_decorrelate_mode=symmetric \
    optim.poet.lie_ortho_decorrelate_renorm=true \
    optim.poet.lie_ortho_decorrelate_lambda="${LAM}" \
    optim.poet.lie_ortho_decorrelate_cos_threshold=0.0 \
    experiment.name="${NAME}"
  echo "<<< ${NAME} done (status $?)"
done
echo "=== update-RMS decorrelation Stage-1 sweep complete: lambda {0.25,0.50,0.75} vs baseline 3.4758 ==="
