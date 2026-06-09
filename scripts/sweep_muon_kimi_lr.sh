#!/usr/bin/env bash
# muon_kimi LEARNING-RATE sweep — everything else at the muon_kimi defaults.
# Run on one node (2 sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_muon_kimi_lr.sh
#
# Baseline being tuned: muon_kimi (vendored Kimi/Moonlight Muon on 2D attn/MLP
# weights, internal AdamW on embeddings/norms/head) — best NON-POET run at 60m/40tpp
# (val/loss 3.5321 @ lr 1e-3, `of4bakqd`). The POET grid found its baseline LR was too
# cold; this checks the same for muon_kimi. ONLY optim.lr changes — momentum (0.95),
# nesterov, ns_steps (5), weight_decay (0.1), and the stock cosine schedule (min_lr 0.1)
# are all left at the muon_kimi defaults.
#
# Launcher = scripts/train_muon_dev.sh, which reproduces the cohort exactly:
# experiment=optim/muon_kimi, llama3-60m, ablation_40x (40 tpp), seq 256, gbs 1024,
# mbs 128, transformer_impl=local, tie_embeddings=false. NOTE: muon_kimi uses ONE base
# optim.lr for BOTH the Muon side (internally scaled by 0.2*sqrt(max d_out,d_in)) and
# the internal AdamW side.
#
#   lr      note
#   3e-3    hotter probe (POET wanted ~3-4x the 1e-3 baseline)
#   4e-3    hottest probe
# All runs use experiment.name=muon_kimi (distinct run dirs by timestamp); the
# 1e-3 baseline = of4bakqd (val 3.5321) is not re-run here.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.003 0.004)

for lr in "${LRS[@]}"; do
  name="muon_kimi"
  echo "### ${name}: lr=${lr} (all else = muon_kimi defaults)"
  codexlog "$name" scripts/train_muon_dev.sh \
    optim.lr="$lr" experiment.name="$name"
done

echo "=== muon_kimi LR sweep complete (2 runs) ==="
