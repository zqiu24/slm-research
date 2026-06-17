#!/usr/bin/env bash
# nGPT LEARNING-RATE sweep — REFERENCE optimizer recipe (no warmup, no weight
# decay), the canonical NVIDIA/ngpt setting. Run sequentially on one node:
#   bash scripts/sweep_ngpt_lr_reference.sh
#
# PURPOSE: now that the patch-binding fix (post-2026-06-17) makes arch/ngpt
# actually train as nGPT, test whether the CANONICAL nGPT recipe from the
# reference (NVIDIA/ngpt train.py:112-114 -> warmup_iters=0, weight_decay=0.0)
# trains stably with the fixed code, and how its best lr compares to the
# adam-matched warmup+wd=0.1 variant in scripts/sweep_ngpt_lr.sh. Same lr grid
# as that sweep so the two are a clean A/B.
#
# Differences vs sweep_ngpt_lr.sh — ONLY the two reference knobs flip:
#   optim.ngpt.no_warmup=true   -> emits "--lr-warmup-samples 0" (no warmup;
#                                  src/utils/scheduler.py:48)
#   optim.weight_decay=0.0      -> reference nGPT wd (train.py:113)
# Everything else is IDENTICAL (train_ngpt_dev.sh: llama3-60m, ablation_40x,
# seq 256, gbs 1024, mbs 128, transformer_impl=local, tie_embeddings=false,
# cosine schedule) so the ONLY difference vs the warmup+wd sweep is warmup + wd.
#
# NOTE: the cosine schedule still decays to 10% of peak (min_lr_ratio=0.1 from
# arch/ngpt), not to 0 as in the reference — kept at 0.1 to match
# sweep_ngpt_lr.sh so the comparison isolates warmup + wd only.
#
# Idempotent: a run is SKIPPED only if its ${LOGDIR}/<name>.log shows it
# COMPLETED ("[after training is done]"); missing OR crashed/partial runs are
# (re-)launched (rm a log to force re-run).
#
#   name             lr        note
#   ngpt_ref_lr5     0.0005
#   ngpt_ref_lr10    0.001     adam baseline lr
#   ngpt_ref_lr20    0.002
#   ngpt_ref_lr30    0.003
#   ngpt_ref_lr40    0.004
#   ngpt_ref_lr50    0.005
#   ngpt_ref_lr60    0.006
#   ngpt_ref_lr70    0.007
#   ngpt_ref_lr80    0.008
#   ngpt_ref_lr90    0.009
#   ngpt_ref_lr100   0.01

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
LTAGS=(5 10 20 30 40 50 60 70 80 90 100)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  name="ngpt_ref_lr${lt}"
  if [[ -f "${LOGDIR}/${name}.log" ]] && grep -q "after training is done" "${LOGDIR}/${name}.log"; then
    echo "### ${name}: SKIP (already completed; rm ${LOGDIR}/${name}.log to re-run)"
    continue
  fi
  echo "### ${name}: lr=${lr}, wd=0.0, no warmup (reference recipe)"
  codexlog "$name" scripts/train_ngpt_dev.sh \
    optim.lr="$lr" optim.weight_decay=0.0 optim.ngpt.no_warmup=true experiment.name="$name"
done

echo "=== nGPT REFERENCE-recipe LR sweep complete (${#LRS[@]} runs) ==="
