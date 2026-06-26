#!/usr/bin/env bash
# POET Lie-Orth update-RMS sweep - init_type=normalized, fixed init_scale=2.0.
#
# With q_optimizer=lie_ortho_update_rms, the old fixed-angle c is replaced by:
#   theta = min(lr * rho / RMS(W), max_angle)
# so the first useful grid is lr x rho, with max_angle held fixed and monitored
# through poet_update_rms/clamp_fraction.
#
# Grid:
#   lr  = {4,5,6}e-3
#   rho = {0.20,0.25,0.30,0.35,0.40}
# = 15 sequential 8-GPU runs.
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
optim.poet.lie_ortho_distributed=true optim.poet.init_type=normalized \
optim.poet.init_scale=2.0 optim.poet.lie_ortho_max_angle=0.024 \
optim.poet.lie_ortho_rms_mode=weight optim.poet.scale=1.0 \
cluster.gpus_per_node=8"

run () {
  codexlog "$1" scripts/train_poet_lie_orth_update_rms.sh llama3 $HELD \
    optim.lr="$2" optim.poet.lie_ortho_update_rms="$3" experiment.name="$1" \
    "${EXTRA_ARGS[@]}"
}

LRS=("lr4:0.004" "lr5:0.005" "lr6:0.006")
RHOS=("r020:0.20" "r025:0.25" "r030:0.30" "r035:0.35" "r040:0.40")
for rho in "${RHOS[@]}"; do
  for lr in "${LRS[@]}"; do
    run "urms_norm_${rho%%:*}_${lr%%:*}" "${lr##*:}" "${rho##*:}"
  done
done

echo "=== POET update-RMS sweep complete: init normalized, init_scale 2.0, 15 runs ==="
