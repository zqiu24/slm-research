#!/usr/bin/env bash
# POET init sweep — NORMALIZED spectrum (unit per-row L2 norm). 2D grid: NORM x ANGLE.
#
# init_type=normalized is the current default & champion anchor. POET freezes W and only
# rotates it, so the init norm IS the operating norm (POET_dev.md §2.7: POET RMS flat
# ~1.07x vs Adam/Muon's ~3.2-3.4x growth to a ~0.045-0.056 equilibrium). normalized lands
# at row_rms = init_scale/sqrt(in) ~ 0.044 at scale 1.0 — close to the Adam equilibrium it
# can't grow into. Two axes (the optimal rotation angle may shift as the base norm moves):
#   NORM  = optim.poet.init_scale {0.5,0.7,1.0,1.4,2.0}  -> row_rms 0.022..0.088
#   ANGLE = optim.poet.lie_ortho_c {6,8,10} -> eff∠ {0.012,0.016,0.020} (lr4e-3*scale0.5*c)
# = 15 runs (5 norm x 3 angle). (s100,c8) = the champion anchor (~3.516). Add c=12 (0.024)
# to CVALS to probe whether a hotter/colder base shifts the divergence ceiling.
#
# Baseline to beat: nestON_lr4 = val/loss 3.5160. All cells: head-off, alt, Nesterov b1.95.
#   bash scripts/sweep_poet_init_normalized.sh        # default GPU 0-3 (dp=4, global_batch=1024), sequential
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

# Champion recipe; init_scale (NORM) and lie_ortho_c (ANGLE) vary per cell. lr/scale fixed
# so the dense AdamW optimization is identical and only the rotation magnitude moves.
HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=normalized \
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
    run "init_norm_${sc%%:*}_${cc%%:*}" "${sc##*:}" "${cc##*:}"
  done
done

echo "=== POET init/normalized 2D sweep complete: 15 runs (baseline 3.5160) ==="
