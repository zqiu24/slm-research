#!/usr/bin/env bash
# Nesterov A/B — arm B: lie_ortho_nesterov=OFF, lie_b1=0.95 (control).
#
# Same recipe as sweep_nesterov_b1_95_ON.sh but Nesterov OFF. This is the DECONFOUND arm:
# the legacy 3.5152 win changed BOTH nesterov AND b1 (0.9→0.95), so comparing ON vs OFF at
# fixed b1=0.95 isolates whether the gain is the look-ahead or just higher momentum. The
# b1=0.9 / no-nesterov champion (3.5231) is the third reference point (already have it).
#
# gpus=4 → pinned to the SECOND half-node (GPU 4-7, port 6010) so it runs CONCURRENTLY
# with the ON script (GPU 0-3, port 6000). Same global_batch=1024 → comparable.
#   bash scripts/sweep_nesterov_b1_95_off.sh       # run alongside the _on script
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES=4,5,6,7          # second half-node
export MASTER_PORT=6010                       # distinct torchrun rendezvous
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe + b1=0.95 + Nesterov OFF; only lr varies. gpus=4.
HELD="base/scale=60m training_regime=ablation_40x optim.poet.scale=0.5 \
optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_b1=0.95 optim.poet.lie_ortho_nesterov=false cluster.gpus_per_node=4"

run () {  # <name> <lr>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD optim.lr="$2" experiment.name="$1"
}

run nestOFF_lr2 2e-3   # eff∠ 0.008
run nestOFF_lr3 3e-3   # eff∠ 0.012
run nestOFF_lr4 4e-3   # eff∠ 0.016  (champion angle)
run nestOFF_lr5 5e-3   # eff∠ 0.020
run nestOFF_lr6 6e-3   # eff∠ 0.024

echo "=== nesterov-OFF (b1=0.95) sweep complete ==="
