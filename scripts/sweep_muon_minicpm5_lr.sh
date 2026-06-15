#!/usr/bin/env bash
# muon (muon_hybrid) LEARNING-RATE sweep on minicpm5_1b — everything else pinned
# to the reference recipe. Runs on the 4 local B200s (3 sequential runs, each
# grabs all 4 GPUs and blocks):
#   bash scripts/sweep_muon_minicpm5_lr.sh
#
# Recipe mirrors runs/adam-minicpm-600m_minicpm-s42-* / adam-nemotron_h-600m_*:
#   experiment=optim/muon_hybrid (Muon on linear weights, Adam on emb/norm/head),
#   base/family=minicpm5, base/scale=minicpm5_1b (679M non-emb), scheduler=wsd
#   (warmup 0.01 / min_lr 0.1 / decay 0.2 cosine), tokens_per_param=17.66 (~12.0B
#   tokens, matching the 600M baselines' absolute budget), data=nemotron_cc_v2_
#   llama31_8b (real, Llama-3.1 tokenizer), gbs 1024, mbs 4, cluster=b200_de with
#   gpus_per_node=4 (tp=1, dp=4), fp8, seed 42. Flash attention is the
#   megatron_args default (--attention-backend flash); no override needed.
#
# ONLY optim.muon.lr changes. optim.adam.lr stays 1e-3 (muon_hybrid default), as
# does momentum 0.95 / quintic NS / weight settings. Baseline = muon.lr 2e-3.
#
#   name              muon.lr   note
#   mcpm5_muon_lr10   1e-3      0.5x baseline
#   mcpm5_muon_lr20   2e-3      baseline (== bs1024 in the batch-size sweep)
#   mcpm5_muon_lr40   4e-3      2x baseline (hottest probe)

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.001 0.002 0.004); LTAGS=(10 20 40)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  name="mcpm5_muon_lr${lt}"
  echo "### ${name}: muon.lr=${lr} (gbs 1024, all else = reference recipe)"
  codexlog "$name" scripts/train_muon.sh minicpm5 \
    cluster=b200_de cluster.gpus_per_node=4 \
    scheduler=wsd \
    training.tokens_per_param=17.66 \
    training.global_batch_size=1024 \
    training.micro_batch_size=4 \
    optim.muon.lr="$lr" \
    wandb.project=slm-arch-dense \
    seed=42 \
    experiment.name="$name"
done

echo "=== muon minicpm5 LR sweep complete (3 runs) ==="
