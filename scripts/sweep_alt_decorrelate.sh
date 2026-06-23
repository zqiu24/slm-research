#!/usr/bin/env bash
# Stage 1 of the alternating-champion decorrelation sweep (ANALYSIS plan, item 1:
# "partial, movement-normalized decorrelation on the actual champion").
#
# Unlike the §17.6 simultaneous A/B (scripts/train_sim_decorrelate.sh, alt=false), this
# runs the ALTERNATING champion (alt=true) WITH cross-side decorrelation ON. Under
# alternating only one side writes per step, so the inactive side's direction is sourced
# from its MAINTAINED momentum (lie_m) and the active write is projected off it — i.e.
# "don't keep pushing along the direction the other side just moved" (cross-step
# over-spend control). Knobs: symmetric mode (fires every step), movement-preserving
# renorm (direction-only change, realized ||dW|| held fixed), partial lambda swept,
# all layers (cos_threshold=0).
#
# Stage 1 finds the best lambda (and whether ANY decorrelation helps the champion).
# Compare each arm against the alternating champion baseline g9i51g5l (val 3.5181).
# Stage 2 (at the best lambda) then sweeps mode {in_off_out, out_off_in} and the
# module-selective gate (cos_threshold=0.3) — held until Stage 1 picks lambda.
#
# All overrides are baked in (the §17.6 A/B lost overrides to truncation three times).
# Just run:   bash scripts/sweep_alt_decorrelate.sh
# Runs the three lambda arms SEQUENTIALLY; background or split across GPUs to parallelize.
#
# TRIPWIRE: each run's startup log prints
#   [POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=symmetric, lambda=<L>, renorm=True,
#          cos_threshold=0.0, alternating=True)
# If lambda/renorm/alternating do NOT match the arm, an override was silently dropped —
# kill it and fix before trusting the result.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
# Log the cos(D_out,D_in) trajectory so the overlap is visibly driven down during the run.
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

for LAM in 0.25 0.50 1.00; do
  TAG="${LAM/./p}"                       # 0.25 -> 0p25
  NAME="alt_decorr_sym_renorm_l${TAG}"
  echo ">>> ${NAME} (lambda=${LAM}) starting"
  scripts/train_poet_lie_orth.sh \
    base/scale=60m \
    training_regime=ablation_40x \
    optim.lr=4e-3 \
    optim.poet.scale=0.5 \
    optim.poet.lie_ortho_c=8 \
    optim.poet.lie_ortho_method=muon \
    optim.poet.head_aligned_attn=false \
    optim.poet.lie_alternating=true \
    optim.poet.lie_ortho_decorrelate=true \
    optim.poet.lie_ortho_decorrelate_mode=symmetric \
    optim.poet.lie_ortho_decorrelate_renorm=true \
    optim.poet.lie_ortho_decorrelate_lambda="${LAM}" \
    optim.poet.lie_ortho_decorrelate_cos_threshold=0.0 \
    experiment.name="${NAME}" \
    2>&1 | tee "${CODEX_LOG_DIR}/${NAME}.log"
  echo "<<< ${NAME} done (status ${PIPESTATUS[0]}) — log: ${CODEX_LOG_DIR}/${NAME}.log"
done
