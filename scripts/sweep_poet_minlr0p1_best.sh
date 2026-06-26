#!/usr/bin/env bash
# POET best-config × min_lr_ratio=0.1 — the three §2.10 champions at a 10% LR floor.
#
# The POET champions used the cosine_poet 1% floor (scheduler.min_lr_ratio=0.01); the dense
# adam / muon_kimi / nGPT baselines all used min_lr_ratio=0.1. §2.5 found 0.01 slightly beat
# 0.1 for the default-init champion — this re-checks that at the NEW init-scaled optima, and
# gives a floor-matched comparison to the baselines. ONLY min_lr_ratio changes vs each best
# config (same init norm + best lr at c6 / poet.scale 0.5, champion recipe).
#
# 3 runs = the three §2.10 best configs (champion init scale, NO nonpoet scaling), floor 0.1:
#   minlr0p1_none_s4 : init none  scale 4, lr 4e-3   (baseline @0.01 = 3.4804)
#   minlr0p1_norm_s2 : init norm  scale 2, lr 5e-3   (baseline @0.01 = 3.4770)
#   minlr0p1_mup_a4  : init mup   alpha 4, lr 5e-3   (baseline @0.01 = 3.4766)
#   bash scripts/sweep_poet_minlr0p1_best.sh        # full 8-GPU node (dp=8, global_batch=1024), sequential
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"   # full 8-GPU node (dp=8)
export MASTER_PORT="${MASTER_PORT:-6000}"                                # torchrun rendezvous
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe; poet.scale + angle (c6) PINNED, scheduler.min_lr_ratio=0.1 (vs cosine_poet's
# 0.01 default). init_type / init norm / lr vary per run (below). Same cosine_poet scheduler,
# only its floor is raised.
HELD="base/scale=60m training_regime=ablation_40x \
scheduler.min_lr_ratio=0.1 \
optim.poet.scale=0.5 optim.poet.lie_ortho_c=6 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true \
cluster.gpus_per_node=8"

NAMES=(minlr0p1_none_s4 minlr0p1_norm_s2 minlr0p1_mup_a4)
OVERRIDES=(
  "optim.lr=0.004 optim.poet.init_type=none optim.poet.init_scale=4.0"
  "optim.lr=0.005 optim.poet.init_type=normalized optim.poet.init_scale=2.0"
  "optim.lr=0.005 optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4"
)

for i in 0 1 2; do
  N="${NAMES[$i]}"; O="${OVERRIDES[$i]}"
  codexlog "$N" scripts/train_poet_lie_orth.sh $HELD $O "experiment.name=$N"
done

echo "=== POET best × min_lr_ratio=0.1 complete: 3 runs (baselines @0.01: none 3.4804 / norm 3.4770 / mup 3.4766) ==="
