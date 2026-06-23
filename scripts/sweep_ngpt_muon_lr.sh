#!/usr/bin/env bash
# nGPT+Muon LEARNING-RATE sweep — nGPT hypersphere architecture trained with the
# vendored Kimi/Moonlight Muon optimizer (experiment=arch/ngpt_muon), tuned over
# lr only. Run on one node (sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_ngpt_muon_lr.sh
#
# PURPOSE: fill the missing leg of the optimizer-vs-architecture matrix. We have
#   * dense llama3 + adam   → 3.4935 @ lr 3e-3   (sweep_adam_lr.sh)
#   * dense llama3 + muon   → 3.4514 @ lr 4e-3   (sweep_muon_kimi_lr.sh, BEST overall)
#   * nGPT       + adam     → 3.4583 @ lr 1e-2   (sweep_ngpt_lr.sh, co-best)
#   * nGPT       + muon     → ??? (this sweep)
# The nGPT+Muon config (configs/experiments/arch/ngpt_muon.yaml) ships only the
# muon_kimi baseline lr=1e-3, which is COLD: dense muon_kimi gained -0.081 going
# 1e-3 -> 4e-3. So this combo needs its own tune before it can be put on the board.
#
# Launcher = scripts/train_ngpt_dev_muon.sh, whose resolved command is
# byte-identical to scripts/train_muon_dev.sh / train_ngpt_dev.sh EXCEPT
# experiment=arch/ngpt_muon — so model/data/tokens/batch/seq/schedule are shared:
# llama3-60m, ablation_40x (40 tpp), seq 256, gbs 1024, mbs 128,
# transformer_impl=local, tie_embeddings=false, cosine schedule (min_lr 0.1).
#
# RECIPE — matched to the *tuned dense muon_kimi* baseline so the ONLY difference
# vs dense muon is the nGPT architecture (mirrors how sweep_ngpt_lr.sh matched the
# adam baseline). Two knobs are overridden away from the ngpt_muon.yaml defaults:
#   optim.weight_decay=0.1        (muon_kimi/leaderboard default; ngpt_muon.yaml
#                                  ships wd=0.0 for the "nGPT regime". wd 0.1 beat
#                                  wd 0 by ~-0.025 for dense muon and for nGPT-adam.
#                                  CAVEAT: with ngpt_optimizer_setup dropped there
#                                  is no zero-WD bucketing, so wd 0.1 also hits the
#                                  nGPT scaling vectors (alpha/sqk/suv/sz). For the
#                                  reference nGPT regime instead, set wd=0.0 below.)
#   optim.ngpt.no_warmup=false    (use the 1% warmup the muon/adam baselines use,
#                                  instead of nGPT's reference no-warmup; this flag
#                                  is a scheduler knob and still applies on the muon
#                                  path — megatron_args.py:285.)
# Everything else stays at the muon_kimi defaults (momentum 0.95, nesterov,
# ns_steps 5, betas [0.9,0.95], eps 1e-8). For a pure nGPT-reference A/B, flip the
# two knobs above (wd=0.0, no_warmup=true) — that is the muon analog of
# sweep_ngpt_lr_reference.sh.
#
# GRID — brackets dense muon's optimum (~4e-3) and extends up past nGPT-adam's
# hotter optimum (1e-2) to 2e-2, since the hypersphere arch tends to want a
# hotter lr (nGPT-adam pushed its optimum to 1e-2 vs dense adam's 3e-3):
#   name              lr        note
#   ngpt_muon_lr20    0.002
#   ngpt_muon_lr30    0.003
#   ngpt_muon_lr40    0.004     dense muon_kimi optimum
#   ngpt_muon_lr50    0.005
#   ngpt_muon_lr60    0.006
#   ngpt_muon_lr80    0.008
#   ngpt_muon_lr100   0.01      nGPT-adam optimum
#   ngpt_muon_lr200   0.02      hot extension
#
# Idempotent: a run is SKIPPED only if its ${LOGDIR}/<name>.log shows it COMPLETED
# ("after training is done"); missing OR crashed/partial runs are (re-)launched
# (rm a log to force re-run).

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.002 0.003 0.004 0.005 0.006 0.008 0.01 0.02)
LTAGS=(20 30 40 50 60 80 100 200)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  name="ngpt_muon_lr${lt}"
  if [[ -f "${LOGDIR}/${name}.log" ]] && grep -q "after training is done" "${LOGDIR}/${name}.log"; then
    echo "### ${name}: SKIP (already completed; rm ${LOGDIR}/${name}.log to re-run)"
    continue
  fi
  echo "### ${name}: lr=${lr}, wd=0.1, warmup matched to muon_kimi baseline"
  codexlog "$name" scripts/train_ngpt_dev_muon.sh \
    optim.lr="$lr" optim.weight_decay=0.1 optim.ngpt.no_warmup=false experiment.name="$name"
done

echo "=== nGPT+Muon LR sweep complete (${#LRS[@]} runs) ==="
