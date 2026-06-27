#!/usr/bin/env bash
# §2.16(a) FOLLOW-UP — bracket the normalized optimum. In the finer-λ scan, normalized@side_γ=0
# renorm=off improved MONOTONICALLY to the grid edge (λ0.30 = 3.4682, ties the basin floor and
# beats its no-decorr base 3.4765 by −0.0083), so its true minimum may sit ABOVE 0.30. Extend the
# scan to λ {0.35, 0.40, 0.45, 0.50} to bracket it. normalized only, side_γ=0, renorm=FALSE,
# mode=symmetric, cos_threshold=0, ρ0.30/lr5/max∠0.024. 4 runs, sequential, ascending λ — once the
# val stops dropping you can kill the remainder (the minimum is bracketed by then).
#   normalized baselines: λ0.30 = 3.4682 (edge), λ0.25 = 3.4703, base (no decorr) = 3.4765.
#
# TRIPWIRE: each startup must print
#   [POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=symmetric, lambda=<L>, renorm=False, ...)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

COMMON="base/scale=60m training_regime=ablation_40x \
  optim.poet.scale=1.0 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.init_type=normalized optim.poet.init_scale=2.0 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.0 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true \
  optim.poet.lie_ortho_decorrelate=true \
  optim.poet.lie_ortho_decorrelate_mode=symmetric \
  optim.poet.lie_ortho_decorrelate_renorm=false \
  optim.poet.lie_ortho_decorrelate_cos_threshold=0.0 \
  optim.lr=0.005 cluster.gpus_per_node=8"

# λ values from args (default = the 4-point edge extension, ascending); allows `… 0.55` etc.
LAMS=("$@"); [ "${#LAMS[@]}" -eq 0 ] && LAMS=(0.35 0.40 0.45 0.50)

for LAM in "${LAMS[@]}"; do
  TAG="${LAM/./p}"                       # 0.35 -> 0p35
  NAME="urms_decorrfine_norm_l${TAG}_rnf"
  echo ">>> ${NAME} (normalized, side_γ0, λ=${LAM}, renorm=off) starting"
  scripts/train_poet_lie_orth_update_rms.sh llama3 ${COMMON} \
    optim.poet.lie_ortho_decorrelate_lambda="${LAM}" \
    experiment.name="${NAME}" \
    2>&1 | tee "${CODEX_LOG_DIR}/${NAME}.log"
  echo "<<< ${NAME} done (status ${PIPESTATUS[0]}) — log: ${CODEX_LOG_DIR}/${NAME}.log"
done
echo "=== normalized finer-λ EXTENSION complete: side_γ0 renorm=off λ{${LAMS[*]}}; bracket vs λ0.30=3.4682 ==="
