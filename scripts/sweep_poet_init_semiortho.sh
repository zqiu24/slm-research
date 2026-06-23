#!/usr/bin/env bash
# POET init sweep — SEMI-ORTHOGONAL spectrum (init_type=orthogonal). 2D grid: NORM x ANGLE.
#
# The most POET-specific init: a fresh semi-orthogonal base has ALL singular values equal
# (condition number = 1) — no near-zero directions the frozen-spectrum rotation can never
# revive. Anchored at init_scale=1.0 to the SAME per-element RMS as normalized (1/sqrt(in)),
# so (s100,c8) vs normalized(s100,c8) is a PURE-CONDITIONING A/B at matched norm. Two axes:
#   NORM  = optim.poet.init_scale {0.5,0.7,1.0,1.4,2.0} -> row_rms 0.022..0.088 (kappa=1)
#   ANGLE = optim.poet.lie_ortho_c {6,8,10} -> eff∠ {0.012,0.016,0.020} (lr4e-3*scale0.5*c)
# = 15 runs (5 norm x 3 angle). A well-conditioned base may tolerate a hotter angle — add
# c=12 (0.024) to CVALS to test whether the divergence ceiling moves up.
#
# Baseline to beat: nestON_lr4 = val/loss 3.5160 (normalized @ scale 1.0, c8).
#   bash scripts/sweep_poet_init_semiortho.sh        # default GPU 0-3 (dp=4, global_batch=1024), sequential
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

# token:value pairs (token avoids float-to-name arithmetic).
SCALES=("s050:0.5" "s070:0.7" "s100:1.0" "s140:1.4" "s200:2.0")   # row_rms ~0.022..0.088
CVALS=("c6:6" "c8:8" "c10:10")                                     # eff∠ 0.012/0.016/0.020
for sc in "${SCALES[@]}"; do
  for cc in "${CVALS[@]}"; do
    run "init_ortho_${sc%%:*}_${cc%%:*}" "${sc##*:}" "${cc##*:}"
  done
done

echo "=== POET init/semi-orthogonal 2D sweep complete: 15 runs (baseline 3.5160) ==="
