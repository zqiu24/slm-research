#!/usr/bin/env bash
# Pion LEARNING-RATE sweep — everything else at the optim/pion defaults.
# Run on one node (sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_pion_lr.sh
#
# Baseline being tuned: pion (vendored src/optim/_pion.py; Pion on 2-D attn/MLP
# weights, chained AdamW on embeddings/norms/biases/head). ONLY optim.lr changes;
# pion_scaling (rms), pion_rms (0.2), pion_update_side (alternate),
# pion_momentum (transported_ambient_ambient), pion_degree (2), betas, and the
# stock cosine schedule (min_lr 0.1) are all left at the optim/pion defaults.
#
# Launcher = scripts/train_pion_dev.sh, which reproduces the dev cohort exactly:
# experiment=optim/pion, llama3-60m, ablation_40x (40 tpp), seq 256, gbs 1024,
# mbs 128, transformer_impl=local, tie_embeddings=false. Pion uses ONE base
# optim.lr for BOTH the Pion side (scaled internally by pion_rms*sqrt(m*n)) and
# the chained-AdamW side.
#
#   lr           note
#   5e-4         cooler probe
#   1e-3         reference default (opt_llama_60M_pion.sh)
#   2e-3, 3e-3   hotter probes
#   4e-3..1e-2   hot-range extension (7 steps) probing for a higher optimum
# Each run uses experiment.name=pion (distinct run dirs by timestamp).

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01)

for lr in "${LRS[@]}"; do
  name="pion_lr${lr}"
  echo "### ${name}: lr=${lr} (all else = optim/pion defaults)"
  codexlog "$name" scripts/train_pion_dev.sh \
    optim.lr="$lr" experiment.name="pion"
done

echo "=== pion LR sweep complete (${#LRS[@]} runs) ==="
