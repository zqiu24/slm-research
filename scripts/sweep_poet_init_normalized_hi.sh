#!/usr/bin/env bash
# POET init EXTENSION — NORMALIZED: HIGHER scale x COOLER angle. 8-GPU.
#
# The champion's normalized@scale1.0 (row_rms 0.044) is likely UNDER-SCALED: every shape in
# the 4-GPU grid kept improving with operating norm and none bottomed out, and c6 beat c8/c10.
# This pushes normalized above its default norm with cooler angles, on a full 8-GPU node
# (dp=8, global_batch 1024):
#   NORM  = optim.poet.init_scale {1,2,3,4,5}  -> row_rms ~0.044..0.220 (integer steps)
#   ANGLE = optim.poet.lie_ortho_c {2,4,6} -> eff∠ {0.004,0.008,0.012}
# = 15 runs + 1 sanity (8-GPU repro of the current 4-GPU best none_s400_c6 = 3.4818).
# Baseline to beat: 3.5160.
#   bash scripts/sweep_poet_init_normalized_hi.sh        # one 8-GPU machine
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"   # full 8-GPU node
export MASTER_PORT="${MASTER_PORT:-6000}"
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=normalized \
cluster.gpus_per_node=8"

run () {  # <name> <init_scale> <lie_ortho_c>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.init_scale="$2" optim.poet.lie_ortho_c="$3" experiment.name="$1"
}

SCALES=("s1:1" "s2:2" "s3:3" "s4:4" "s5:5")   # row_rms ~0.044..0.220 (integer +1 steps)
CVALS=("c2:2" "c4:4" "c6:6")                   # eff∠ 0.004/0.008/0.012
for sc in "${SCALES[@]}"; do
  for cc in "${CVALS[@]}"; do
    run "hi_norm_${sc%%:*}_${cc%%:*}" "${sc##*:}" "${cc##*:}"
  done
done

# Sanity: 8-GPU repro of the current 4-GPU best (none_s400_c6 = 3.4818) — explicit init_type=none.
codexlog sanity_none_s4_c6_8g scripts/train_poet_lie_orth.sh \
  base/scale=60m training_regime=ablation_40x optim.lr=0.004 optim.poet.scale=0.5 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true cluster.gpus_per_node=8 \
  optim.poet.init_type=none optim.poet.init_scale=4 optim.poet.lie_ortho_c=6 \
  experiment.name=sanity_none_s4_c6_8g

echo "=== POET init/normalized HI sweep complete: 15 runs + sanity (baseline 3.5160) ==="
