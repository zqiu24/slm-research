#!/usr/bin/env bash
# A2 of the §17.6 simultaneous ±decorrelation A/B (ANALYSIS / POET_dev arm J).
#
# ALL overrides are baked in here so nothing can be lost to copy-paste/line-continuation
# truncation (three prior attempts ran with the overrides silently dropped). Just run:
#
#     bash scripts/train_sim_decorrelate.sh
#
# It tees to $CODEX_LOG_DIR/sim_decorrelate2.log itself (codexlog is an interactive
# shell function and does NOT expand in a non-interactive script). Compare against the
# already-valid simultaneous baseline `sim_baseline` (val ~3.577) and the alternating
# champion g9i51g5l (3.518).
#
# Recipe = champion (lr 4e-3 / scale 0.5 / c8 / muon / head-OFF) but SIMULTANEOUS
# (lie_alternating=false) WITH cross-side decorrelation ON (cos(D_out,D_in)->0).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

NAME=sim_decorrelate2

scripts/train_poet_lie_orth.sh \
  base/scale=60m \
  training_regime=ablation_40x \
  optim.lr=4e-3 \
  optim.poet.scale=0.5 \
  optim.poet.lie_ortho_c=8 \
  optim.poet.lie_ortho_method=muon \
  optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=false \
  optim.poet.lie_ortho_decorrelate=true \
  optim.poet.lie_ortho_decorrelate_mode=in_off_out \
  experiment.name="${NAME}" \
  2>&1 | tee "${CODEX_LOG_DIR}/${NAME}.log"

echo "<<< ${NAME} done (status ${PIPESTATUS[0]}) — log: ${CODEX_LOG_DIR}/${NAME}.log"
