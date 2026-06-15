#!/usr/bin/env bash
# muon (muon_hybrid) GLOBAL-BATCH-SIZE sweep on minicpm5_1b — everything else
# pinned to the reference recipe. Runs on the 4 local B200s (2 sequential runs,
# each grabs all 4 GPUs and blocks):
#   bash scripts/sweep_muon_minicpm5_bs.sh
#
# Recipe mirrors runs/adam-minicpm-600m_minicpm-s42-* / adam-nemotron_h-600m_*:
#   experiment=optim/muon_hybrid, base/family=minicpm5, base/scale=minicpm5_1b,
#   scheduler=wsd, tokens_per_param=17.66 (~12.0B tokens, FIXED across this sweep
#   so only the step count changes), data=nemotron_cc_v2_llama31_8b, mbs 4,
#   cluster=b200_de with gpus_per_node=4 (tp=1, dp=4), fp8, seed 42. Flash
#   attention is on by default.
#
# ONLY training.global_batch_size changes. muon.lr stays 2e-3 (baseline);
# LR is NOT re-scaled with batch size here (hold-one-fixed coordinate sweep).
# Each gbs is divisible by dp*mbs = 4*4 = 16. The gbs=1024 baseline is omitted
# here because it is identical to mcpm5_muon_lr20 in the LR sweep (run once).
#
#   name                gbs    note
#   mcpm5_muon_bs512    512    0.5x baseline
#   mcpm5_muon_bs2048   2048   2x baseline
#   (gbs 1024 baseline = mcpm5_muon_lr20, run by the LR sweep)

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

GBS=(512 2048); BTAGS=(512 2048)

for i in "${!GBS[@]}"; do
  gbs="${GBS[$i]}"; bt="${BTAGS[$i]}"
  name="mcpm5_muon_bs${bt}"
  echo "### ${name}: gbs=${gbs} (muon.lr 2e-3, all else = reference recipe)"
  codexlog "$name" scripts/train_muon.sh minicpm5 \
    cluster=b200_de cluster.gpus_per_node=4 \
    scheduler=wsd \
    training.tokens_per_param=17.66 \
    training.global_batch_size="$gbs" \
    training.micro_batch_size=4 \
    optim.muon.lr=0.002 \
    wandb.project=slm-arch-dense \
    seed=42 \
    experiment.name="$name"
done

echo "=== muon minicpm5 batch-size sweep complete (2 runs) ==="
