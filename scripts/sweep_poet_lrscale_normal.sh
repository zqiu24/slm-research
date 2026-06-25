#!/usr/bin/env bash
# POET lr x poet-scale sweep — init_type=normalized, FIXED best init_scale=2.0 (§2.9, row_rms ~0.088).
#
# The §2.9 init sweep fixed the frozen-base NORM optimum (normalized @ init_scale 2.0 / c6 = 3.4809;
# 4-GPU twin 3.4787, the grid-min cell). This sweep holds that norm + angle (c6) and instead perturbs
# the TWO optimizer levers around the champion (lr 4e-3, poet.scale 0.5):
#   LR    = optim.lr          {3,4,5,6}e-3   (dense adam lr AND a factor of the rotation angle)
#   SCALE = optim.poet.scale  {0.4,0.5,0.6}  (rotation magnitude ONLY)
# = 12 runs (4 lr x 3 scale). eff∠ = lr*scale*c6 spans 0.0072..0.0216 around the 0.012 optimum.
# Iso-angle diagonals (lr4/ps50 == lr5/ps40 == eff∠ 0.012) disentangle dense-lr from rotation
# magnitude at matched angle: is the lever the optimizer step or the rotation size?
#
# Baseline to beat: hi_norm_s2_c6 = val/loss 3.4809 (the grid center lr4/ps50 reproduces it).
#   bash scripts/sweep_poet_lrscale_normal.sh        # default GPU 0-3 (dp=4, global_batch=1024), sequential
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # default first half-node (GPU 0-3)
export MASTER_PORT="${MASTER_PORT:-6000}"                         # torchrun rendezvous (override + CUDA_VISIBLE_DEVICES=4,5,6,7 to pair a 2nd lane)
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe; init NORM (init_scale=2.0) and ANGLE (c6) PINNED. lr and poet.scale vary per cell.
HELD="base/scale=60m training_regime=ablation_40x \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true \
optim.poet.init_type=normalized optim.poet.init_scale=2.0 optim.poet.lie_ortho_c=6 \
cluster.gpus_per_node=4"

run () {  # <name> <lr> <poet_scale>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.lr="$2" optim.poet.scale="$3" experiment.name="$1"
}

# token:value pairs (token avoids float-to-name arithmetic).
LRS=("lr3:0.003" "lr4:0.004" "lr5:0.005" "lr6:0.006")
SCALES=("ps40:0.4" "ps50:0.5" "ps60:0.6")
for lr in "${LRS[@]}"; do
  for sc in "${SCALES[@]}"; do
    run "lrsc_norm_${lr%%:*}_${sc%%:*}" "${lr##*:}" "${sc##*:}"
  done
done

echo "=== POET lr x scale sweep (init normalized, scale 2.0/c6) complete: 12 runs (baseline 3.4809) ==="
