#!/usr/bin/env bash
# Small user-run sweep for POET Lie-Orth update-RMS:
#   rho in {0.16, 0.20, 0.25, 0.30}
#   lr  in {4e-3, 5e-3, 6e-3}
#   max_angle fixed at 0.024
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-6000}"
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

HELD="base/scale=60m training_regime=ablation_40x \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=mup_normalized \
optim.poet.mup_alpha=4 optim.poet.lie_ortho_max_angle=0.024 \
optim.poet.lie_ortho_rms_mode=weight optim.poet.scale=1.0 cluster.gpus_per_node=8"

run () {
  codexlog "$1" scripts/train_poet_lie_orth_update_rms.sh llama3 $HELD \
    optim.lr="$2" optim.poet.lie_ortho_update_rms="$3" experiment.name="$1"
}

LRS=("lr4:0.004" "lr5:0.005" "lr6:0.006")
RHOS=("r016:0.16" "r020:0.20" "r025:0.25" "r030:0.30")
for rho in "${RHOS[@]}"; do
  for lr in "${LRS[@]}"; do
    run "urms_${rho%%:*}_${lr%%:*}" "${lr##*:}" "${rho##*:}"
  done
done

echo "=== POET Lie-Orth update-RMS sweep complete: 12 runs ==="
