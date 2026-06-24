#!/usr/bin/env bash
# POET init EXTENSION — NORMAL/raw (init_type=none): HIGHER scale x COOLER angle. 8-GPU.
#
# The 4-GPU grid found raw init best (none_s400_c6 = 3.4818, eff∠0.012/scale4.0) and STILL
# FALLING with scale at the top edge, with c6 ≥ c8 ≥ c10 (cooler better). This extends both
# frontiers on a full 8-GPU node (dp=8, global_batch 1024 — same as the 8-GPU champions):
#   NORM  = optim.poet.init_scale {4,5,6,7,8}  -> row_rms ~0.064..0.128 (integer steps)
#   ANGLE = optim.poet.lie_ortho_c {2,4,6} -> eff∠ {0.004,0.008,0.012} (cooler than the c6 floor)
# = 15 runs. The hi_none_s4_c6 cell reproduces the 4-GPU best (3.4818) on 8 GPU -> the
# built-in 4<->8 GPU sanity check (≈no difference expected). Baseline to beat: 3.5160.
#   bash scripts/sweep_poet_init_normal_hi.sh        # one 8-GPU machine
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
optim.poet.lie_ortho_distributed=true optim.poet.init_type=none \
cluster.gpus_per_node=8"

run () {  # <name> <init_scale> <lie_ortho_c>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.init_scale="$2" optim.poet.lie_ortho_c="$3" experiment.name="$1"
}

SCALES=("s4:4" "s5:5" "s6:6" "s7:7" "s8:8")   # row_rms ~0.064..0.128 (integer +1 steps)
CVALS=("c2:2" "c4:4" "c6:6")                   # eff∠ 0.004/0.008/0.012
for sc in "${SCALES[@]}"; do
  for cc in "${CVALS[@]}"; do
    run "hi_none_${sc%%:*}_${cc%%:*}" "${sc##*:}" "${cc##*:}"
  done
done

echo "=== POET init/normal HI sweep complete: 15 runs (hi_none_s4_c6 = 8-GPU parity vs 3.4818) ==="
