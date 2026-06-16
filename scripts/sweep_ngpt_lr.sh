#!/usr/bin/env bash
# nGPT LEARNING-RATE sweep — everything else at the nGPT reference defaults.
# Run on one node (sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_ngpt_lr.sh
#
# PURPOSE: tune nGPT's lr and compare its best against the adam lr sweep
# (scripts/sweep_adam_lr.sh) on the SAME cohort. Launcher = train_ngpt_dev.sh,
# whose resolved command is byte-identical to train_adam_dev.sh except
# experiment=arch/ngpt — so model/data/tokens/batch/seq/schedule are shared:
# llama3-60m, ablation_40x (40 tpp), seq 256, gbs 1024, mbs 128,
# transformer_impl=local, tie_embeddings=false, cosine schedule (min_lr 0.1).
#
# ONLY optim.lr changes. Held at the nGPT reference recipe (NOT tuned):
#   betas [0.9,0.95], eps, weight_decay=0, no-warmup, and the hypersphere init
#   scales alpha_init=0.05 / sqk_init=1 / suv_init=1 / sz_init=1 / base_scale=1/sqrt(d).
#   (weight_decay=0 and no-warmup are required by the per-step normalization, the
#   same way adam's wd=0.1 + warmup are adam's recipe — each method at its own.)
# nGPT also runs full activation recompute (its default) so it fits at mbs=128;
# recompute is memory-only and does NOT change the loss, so the val/loss is
# directly comparable to the adam runs.
#
# nGPT's reference lr is 15e-4; it likes hotter lr than adam (adam baseline ~1e-3).
# This brackets the reference and probes up to ~8x. Extend LRS if the optimum
# sits at an edge. NOTE: the pre-fix ngpt_lr10..50 runs (2026-06-11) are INVALID
# — they predate the patch-binding fix and trained as plain llama3, not nGPT.
#
#   name         lr       note
#   ngpt_lr15    0.0015   reference / current default (low anchor)
#   ngpt_lr30    0.003    2x reference
#   ngpt_lr60    0.006    4x reference
#   ngpt_lr120   0.012    8x reference (hottest probe)

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.0015 0.003 0.006 0.012); LTAGS=(15 30 60 120)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  name="ngpt_lr${lt}"
  echo "### ${name}: lr=${lr} (all else = nGPT reference defaults)"
  codexlog "$name" scripts/train_ngpt_dev.sh \
    optim.lr="$lr" experiment.name="$name"
done

echo "=== nGPT LR sweep complete (${#LRS[@]} runs) ==="
