#!/usr/bin/env bash
# Nesterov A/B — arm A: lie_ortho_nesterov=ON, lie_b1=0.95 (Muon's canonical recipe).
#
# Tests whether POET with MUON'S ACTUAL Nesterov recipe (look-ahead + b1=0.95) beats the
# b1=0.9 champion (3.5231). The earlier sweep used the default b1=0.9 (Muon's Nesterov off
# its design momentum) and lost; the only promising legacy run (3.5152) used b1=0.95.
# Pair with sweep_nesterov_b1_95_OFF.sh (same recipe, nesterov OFF) to DECONFOUND
# "nesterov" from "higher momentum b1=0.95".
#
# gpus=4 → pinned to the FIRST half-node (GPU 0-3, port 6000) so it runs CONCURRENTLY
# with the OFF script (GPU 4-7, port 6010). Same global_batch=1024 as the 8-GPU champion
# (dp=4 just does 2 grad-accum microbatches) → results directly comparable.
#   bash scripts/sweep_nesterov_b1_95_on.sh        # run alongside the _off script
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES=0,1,2,3          # first half-node
export MASTER_PORT=6000                       # distinct torchrun rendezvous
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe + b1=0.95 + Nesterov ON; only lr (=> eff∠ = lr·0.5·8) varies. gpus=4.
HELD="base/scale=60m training_regime=ablation_40x optim.poet.scale=0.5 \
optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_b1=0.95 optim.poet.lie_ortho_nesterov=true cluster.gpus_per_node=4"

run () {  # <name> <lr>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD optim.lr="$2" experiment.name="$1"
}

run nestON_lr2 2e-3   # eff∠ 0.008
run nestON_lr3 3e-3   # eff∠ 0.012
run nestON_lr4 4e-3   # eff∠ 0.016  (champion angle)
run nestON_lr5 5e-3   # eff∠ 0.020
run nestON_lr6 6e-3   # eff∠ 0.024

echo "=== nesterov-ON (b1=0.95) sweep complete ==="
