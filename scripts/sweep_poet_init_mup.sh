#!/usr/bin/env bash
# POET init sweep — muP spectrum (init_type=mup_normalized). 2D grid: NORM x ANGLE.
#
# mup_normalized row-normalizes the frozen W then SPECTRAL-scales it so the top singular
# value = mup_alpha * sqrt(d_out/d_in) (the muP-style spectral target). Its operating-norm
# knob is mup_alpha (a spectral-norm parameterization of the "what frozen norm?" axis the
# other scripts sweep via row_rms). mup_normalized has never been A/B'd. Two axes:
#   NORM  = optim.poet.mup_alpha {0.25,0.5,1.0,2.0,4.0}  (init_scale stays 1.0)
#   ANGLE = optim.poet.lie_ortho_c {6,8,10} -> eff∠ {0.012,0.016,0.020} (lr4e-3*scale0.5*c)
# = 15 runs (5 norm x 3 angle). Add c=12 (0.024) to CVALS to probe the divergence ceiling.
#
# Baseline to beat: nestON_lr4 = val/loss 3.5160 (init_type=normalized @ scale 1.0, c8).
#   bash scripts/sweep_poet_init_mup.sh        # default GPU 0-3 (dp=4, global_batch=1024), sequential
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

# Champion recipe; mup_alpha (NORM) and lie_ortho_c (ANGLE) vary per cell. init_type pinned mup_normalized.
HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=mup_normalized \
cluster.gpus_per_node=4"

run () {  # <name> <mup_alpha> <lie_ortho_c>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.mup_alpha="$2" optim.poet.lie_ortho_c="$3" experiment.name="$1"
}

# token:value pairs (token avoids float-to-name arithmetic).
ALPHAS=("a025:0.25" "a050:0.5" "a100:1.0" "a200:2.0" "a400:4.0")   # spectral-norm target scale
CVALS=("c6:6" "c8:8" "c10:10")                                     # eff∠ 0.012/0.016/0.020
for ac in "${ALPHAS[@]}"; do
  for cc in "${CVALS[@]}"; do
    run "init_mup_${ac%%:*}_${cc%%:*}" "${ac##*:}" "${cc##*:}"
  done
done

echo "=== POET init/mup 2D sweep complete: 15 runs (baseline 3.5160) ==="
