#!/usr/bin/env bash
# POET init EXTENSION — muP (init_type=mup_normalized): HIGHER spectral scale x COOLER angle. 8-GPU.
#
# The 4-GPU mup grid kept improving with mup_alpha (a025 3.671 -> a100 3.548 -> a200 3.517)
# and was still falling at the top edge (a400 pending), with c6 ≥ c8 ≥ c10. This pushes the
# spectral-norm target higher with cooler angles, on a full 8-GPU node (dp=8, global_batch 1024):
#   NORM  = optim.poet.mup_alpha {2,3,4,5,6}  (init_scale stays 1.0; integer +1 steps)
#   ANGLE = optim.poet.lie_ortho_c {2,4,6} -> eff∠ {0.004,0.008,0.012}
# = 15 runs + 1 sanity (8-GPU repro of the current 4-GPU best none_s400_c6 = 3.4818).
# Baseline to beat: 3.5160.
#   bash scripts/sweep_poet_init_mup_hi.sh        # one 8-GPU machine
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
optim.poet.lie_ortho_distributed=true optim.poet.init_type=mup_normalized \
cluster.gpus_per_node=8"

run () {  # <name> <mup_alpha> <lie_ortho_c>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.mup_alpha="$2" optim.poet.lie_ortho_c="$3" experiment.name="$1"
}

ALPHAS=("a2:2" "a3:3" "a4:4" "a5:5" "a6:6")   # spectral-norm target (integer +1 steps)
CVALS=("c2:2" "c4:4" "c6:6")                   # eff∠ 0.004/0.008/0.012
for ac in "${ALPHAS[@]}"; do
  for cc in "${CVALS[@]}"; do
    run "hi_mup_${ac%%:*}_${cc%%:*}" "${ac##*:}" "${cc##*:}"
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

echo "=== POET init/mup HI sweep complete: 15 runs + sanity (baseline 3.5160) ==="
