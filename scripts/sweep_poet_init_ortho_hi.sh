#!/usr/bin/env bash
# POET init EXTENSION — ORTHOGONAL (init_type=orthogonal, kappa=1): BIGGER scale x COOLER angle. 4-GPU.
#
# The 4-GPU grid only reached scale 2.0 and only angles c{6,8,10}; orthogonal was STILL FALLING
# with scale at the top edge (s2.0/c6 = 3.5240) and c6 was the best of the three angles. So this
# pushes BOTH unexplored directions:
#   NORM  = optim.poet.init_scale {3,4,5,6}  -> row_rms ~0.13..0.26 (kappa=1; ~0.044*scale)
#   ANGLE = optim.poet.lie_ortho_c {4,6}     -> eff∠ {0.008,0.012} (cooler; c4 never tried for ortho)
# = 8 runs (4 norm x 2 angle). Baseline to beat: best ortho so far 3.5240 (s2.0/c6); overall
# init best is none s3.5-4.0/c6 ~= 3.480.
#   bash scripts/sweep_poet_init_ortho_hi.sh        # default GPU 0-3 (dp=4, global_batch=1024), sequential
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

# Champion recipe; init_scale (NORM) and lie_ortho_c (ANGLE) vary per cell. init_type pinned orthogonal.
HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=orthogonal \
cluster.gpus_per_node=4"

run () {  # <name> <init_scale> <lie_ortho_c>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.init_scale="$2" optim.poet.lie_ortho_c="$3" experiment.name="$1"
}

# token:value pairs (token avoids float-to-name arithmetic). sNNN = scale*100 so names merge
# into the existing init_ortho_* grid (s300=3.0 .. s600=6.0).
SCALES=("s300:3" "s400:4" "s500:5" "s600:6")   # row_rms ~0.13..0.26
CVALS=("c4:4" "c6:6")                             # eff∠ 0.008/0.012
for sc in "${SCALES[@]}"; do
  for cc in "${CVALS[@]}"; do
    run "init_ortho_${sc%%:*}_${cc%%:*}" "${sc##*:}" "${cc##*:}"
  done
done

echo "=== POET init/orthogonal HI sweep complete: 8 runs (bigger scale x cooler angle; baseline ortho 3.5240) ==="
