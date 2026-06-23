#!/usr/bin/env bash
# POET init sweep — NORMAL / raw spectrum (init_type=none). 2D grid: NORM x ANGLE.
#
# init_type=none keeps Megatron's native init: the Marchenko-Pastur Gaussian spectrum AND
# the residual 1/sqrt(2L) downscale on proj/fc2 (type-structured, large condition number).
# Native per-element RMS is ~0.016 (§2.7) — ~3x BELOW normalized's 0.044. Because POET can't
# grow the frozen norm, raw init may simply be under-scaled. Two axes:
#   NORM  = optim.poet.init_scale {1.0,1.5,2.75,4.0,5.5} -> row_rms 0.016..0.088
#           (s275 ~0.044 is a matched-NORM A/B vs normalized@1.0)
#   ANGLE = optim.poet.lie_ortho_c {6,8,10} -> eff∠ {0.012,0.016,0.020} (lr4e-3*scale0.5*c)
# = 15 runs (5 norm x 3 angle). Add c=12 (0.024) to CVALS to probe the divergence ceiling.
#
# Baseline to beat: nestON_lr4 = val/loss 3.5160 (normalized @ scale 1.0, c8).
#   bash scripts/sweep_poet_init_normal.sh        # default GPU 0-3 (dp=4, global_batch=1024), sequential
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

# Champion recipe; init_scale (NORM) and lie_ortho_c (ANGLE) vary per cell. init_type pinned none.
HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=none \
cluster.gpus_per_node=4"

run () {  # <name> <init_scale> <lie_ortho_c>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.init_scale="$2" optim.poet.lie_ortho_c="$3" experiment.name="$1"
}

# token:value pairs (token avoids float-to-name arithmetic).
SCALES=("s100:1.0" "s150:1.5" "s275:2.75" "s400:4.0" "s550:5.5")   # row_rms ~0.016..0.088
CVALS=("c6:6" "c8:8" "c10:10")                                     # eff∠ 0.012/0.016/0.020
for sc in "${SCALES[@]}"; do
  for cc in "${CVALS[@]}"; do
    run "init_none_${sc%%:*}_${cc%%:*}" "${sc##*:}" "${cc##*:}"
  done
done

echo "=== POET init/normal(raw) 2D sweep complete: 15 runs (baseline 3.5160) ==="
