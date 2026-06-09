#!/usr/bin/env bash
# adam (AdamW) LEARNING-RATE sweep — everything else at the adam baseline defaults.
# Run on one node (3 sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_adam_lr.sh
#
# Baseline being tuned: adam (AdamW dense baseline) — val/loss 3.5570 @ lr 1e-3
# (`ylrd45af`). The POET grid found its baseline LR was too cold; this probes hotter
# LRs for adam. ONLY optim.lr changes — betas [0.9,0.95], eps, weight_decay (0.1), and
# the stock cosine schedule (min_lr 0.1) are all left at the adam baseline defaults.
#
# Launcher = scripts/train_adam_dev.sh, which reproduces the cohort exactly:
# experiment=optim/adam, llama3-60m, ablation_40x (40 tpp), seq 256, gbs 1024, mbs 128,
# transformer_impl=local, tie_embeddings=false.
#
#   name        lr      note
#   adam_lr20   2e-3    2x baseline
#   adam_lr30   3e-3
#   adam_lr40   4e-3    hottest probe
#   (baseline lr 1e-3 = 3.5570 not re-run here)

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.002 0.003 0.004); LTAGS=(20 30 40)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  name="adam_lr${lt}"
  echo "### ${name}: lr=${lr} (all else = adam baseline defaults)"
  codexlog "$name" scripts/train_adam_dev.sh \
    optim.lr="$lr" experiment.name="$name"
done

echo "=== adam LR sweep complete (3 runs) ==="
