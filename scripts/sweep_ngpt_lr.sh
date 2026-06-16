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
# The optimizer hyperparams + schedule are MATCHED to the adam baseline so the
# ONLY difference is the nGPT architecture + ngpt_adamw normalization:
#   optim.weight_decay=0.1        (adam's wd; nGPT's per-step row/col normalize
#                                  washes out the magnitude shrink, scaling
#                                  params stay no-decay — so this stays close to
#                                  nGPT's reference wd=0 while removing the confound)
#   optim.ngpt.no_warmup=false    (use adam's 1% warmup instead of nGPT's none)
# With these, ngpt vs adam now match on warmup (0.01), betas [0.9,0.95], eps,
# weight_decay (0.1), cosine decay + min_lr (0.1), seq/gbs/mbs/tokens. The nGPT
# init scales alpha_init=0.05 / sqk_init=1 / suv_init=1 / sz_init=1 /
# base_scale=1/sqrt(d) are reference-fixed (not tuned — analogous to betas/eps).
# nGPT also runs full activation recompute (its default) so it fits at mbs=128;
# recompute is memory-only and does NOT change the loss, so the val/loss is
# directly comparable to the adam runs.
#
# This grid matches the adam sweep's lr range (so both methods are tuned over
# the same lrs). NOTE: the pre-fix ngpt_lr10..50 runs (2026-06-11) are INVALID
# — they predate the patch-binding fix and trained as plain llama3, not nGPT;
# these runs (post-fix) overwrite those logs.
#
#   name         lr        note
#   ngpt_lr5     0.0005
#   ngpt_lr10    0.001     adam baseline lr
#   ngpt_lr20    0.002
#   ngpt_lr30    0.003
#   ngpt_lr40    0.004
#   ngpt_lr50    0.005

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.0005 0.001 0.002 0.003 0.004 0.005); LTAGS=(5 10 20 30 40 50)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  name="ngpt_lr${lt}"
  echo "### ${name}: lr=${lr}, wd=0.1, warmup matched to adam"
  codexlog "$name" scripts/train_ngpt_dev.sh \
    optim.lr="$lr" optim.weight_decay=0.1 optim.ngpt.no_warmup=false experiment.name="$name"
done

echo "=== nGPT LR sweep complete (${#LRS[@]} runs) ==="
