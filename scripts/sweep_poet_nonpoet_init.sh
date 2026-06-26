#!/usr/bin/env bash
# POET control — scale the NON-POET layers' init too (sanity check, expected ~no-op).
#
# §2.9/§2.10 found POET wants its FROZEN weights scaled UP (none x4 / normalized x2 / mup a4),
# because POET can't grow a frozen weight so the init norm IS the operating norm. The embeddings
# + LM head are AdamW-trained (they CAN grow to their own equilibrium), so their init norm should
# wash out. This control sets optim.poet.nonpoet_init_scale to the SAME factor as each shape's
# POET scale and asks whether matching them helps. Expectation: ≈null — a null is the clean
# control proving the init lever is *specifically* about the frozen POET weights.
#
# 3 runs = the three §2.10 best configs (init norm + best lr at c6 / poet.scale 0.5), each with
# nonpoet_init_scale = the POET norm multiplier (none 4, normalized 2, mup 4):
#   nonpoet_none_s4  : init none  scale 4, lr 4e-3   (baseline 3.4804)
#   nonpoet_norm_s2  : init norm  scale 2, lr 5e-3   (baseline 3.4770)
#   nonpoet_mup_a4   : init mup   alpha 4, lr 5e-3   (baseline 3.4766)
#   bash scripts/sweep_poet_nonpoet_init.sh        # full 8-GPU node (dp=8, global_batch=1024), sequential
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

# Champion recipe; poet.scale + angle (c6) PINNED. init_type / init norm / lr / nonpoet_init_scale
# vary per run (below). nonpoet_init_scale also scales the embedding + (untied) LM-head init.
HELD="base/scale=60m training_regime=ablation_40x \
optim.poet.scale=0.5 optim.poet.lie_ortho_c=6 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true \
cluster.gpus_per_node=8"

NAMES=(nonpoet_none_s4 nonpoet_norm_s2 nonpoet_mup_a4)
OVERRIDES=(
  "optim.lr=0.004 optim.poet.init_type=none optim.poet.init_scale=4.0 optim.poet.nonpoet_init_scale=4.0"
  "optim.lr=0.005 optim.poet.init_type=normalized optim.poet.init_scale=2.0 optim.poet.nonpoet_init_scale=2.0"
  "optim.lr=0.005 optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.nonpoet_init_scale=4.0"
)

for i in 0 1 2; do
  N="${NAMES[$i]}"; O="${OVERRIDES[$i]}"
  codexlog "$N" scripts/train_poet_lie_orth.sh $HELD $O "experiment.name=$N"
done

echo "=== POET nonpoet-init control complete: 3 runs (baselines none 3.4804 / norm 3.4770 / mup 3.4766) ==="
