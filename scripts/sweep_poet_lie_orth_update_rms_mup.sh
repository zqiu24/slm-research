#!/usr/bin/env bash
# POET Lie-Orth update-RMS sweep - init_type=mup_normalized, fixed mup_alpha=4.
#
# With q_optimizer=lie_ortho_update_rms, the old fixed-angle c is replaced by:
#   theta = min(lr * rho / RMS(W), max_angle)
# so the first useful grid is lr x rho, with max_angle held fixed and monitored
# through poet_update_rms/clamp_fraction.
#
# Grid:
#   lr  = 5e-3
#   rho = {0.20,0.25,0.30,0.35,0.40}
# = 5 sequential 8-GPU runs.
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

HELD="base/scale=60m training_regime=ablation_40x \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=mup_normalized \
optim.poet.mup_alpha=4 optim.poet.lie_ortho_max_angle=0.024 \
optim.poet.lie_ortho_rms_mode=weight optim.poet.scale=1.0 \
optim.lr=0.005 cluster.gpus_per_node=8"

run () {
  codexlog "$1" scripts/train_poet_lie_orth_update_rms.sh llama3 $HELD \
    optim.poet.lie_ortho_update_rms="$2" experiment.name="$1" \
    "${EXTRA_ARGS[@]}"
}

RHOS=("r020:0.20" "r025:0.25" "r030:0.30" "r035:0.35" "r040:0.40")
for rho in "${RHOS[@]}"; do
  run "urms_mup_${rho%%:*}_lr5" "${rho##*:}"
done

echo "=== POET update-RMS sweep complete: init mup_normalized, mup_alpha 4, lr5, 5 runs ==="
