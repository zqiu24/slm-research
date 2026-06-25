#!/usr/bin/env bash
# POET lr x poet-scale sweep — init_type=mup_normalized, FIXED best mup_alpha=4 (§2.9, row_rms ~0.064).
#
# The §2.9 init sweep fixed the frozen-base NORM optimum (mup @ mup_alpha 4 / c6 = 3.4816; 8-GPU
# twin 3.4803). This sweep holds that norm + angle (c6) and instead perturbs the TWO optimizer
# levers around the champion (lr 4e-3, poet.scale 0.5):
#   LR    = optim.lr          {2,3,4,5,6}e-3   (dense adam lr AND a factor of the rotation angle)
#   SCALE = optim.poet.scale  {0.2,0.5,1.0}    (rotation magnitude ONLY)
# = 15 runs (5 lr x 3 scale). eff∠ = lr*scale*c6 spans 0.0024..0.036 around the 0.012 optimum.
# Iso-angle cells (lr4/ps50 == lr2/ps100 == eff∠ 0.012) disentangle dense-lr from rotation
# magnitude at matched angle: is the lever the optimizer step or the rotation size?
# (mup norm axis is mup_alpha; init_scale stays at its 1.0 default. Nesterov + lie_b1=0.95 come
#  from the poet_lie_orth experiment config; re-stated below for record.)
#
# Baseline to beat: init_mup_a400_c6 = val/loss 3.4816 (the grid center lr4/ps50 reproduces it).
#   bash scripts/sweep_poet_lrscale_mup.sh        # full 8-GPU node (dp=8, global_batch=1024), sequential
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"   # full 8-GPU node (dp=8)
export MASTER_PORT="${MASTER_PORT:-6000}"                                # torchrun rendezvous
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe; init NORM (mup_alpha=4) and ANGLE (c6) PINNED. lr and poet.scale vary per cell.
HELD="base/scale=60m training_regime=ablation_40x \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true \
optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.lie_ortho_c=6 \
cluster.gpus_per_node=8"

run () {  # <name> <lr> <poet_scale>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.lr="$2" optim.poet.scale="$3" experiment.name="$1"
}

# token:value pairs (token avoids float-to-name arithmetic).
LRS=("lr2:0.002" "lr3:0.003" "lr4:0.004" "lr5:0.005" "lr6:0.006")
SCALES=("ps20:0.2" "ps50:0.5" "ps100:1.0")
for lr in "${LRS[@]}"; do
  for sc in "${SCALES[@]}"; do
    run "lrsc_mup_${lr%%:*}_${sc%%:*}" "${lr##*:}" "${sc##*:}"
  done
done

echo "=== POET lr x scale sweep (init mup_normalized, alpha 4/c6) complete: 15 runs (baseline 3.4816) ==="
