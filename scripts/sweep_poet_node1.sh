#!/usr/bin/env bash
# POET head-aligned sweep — NODE 1 of 2.  Run on one node:  bash scripts/sweep_poet_node1.sh
# 5 sequential runs, all 60m + ablation_40x (the script defaults; set here explicitly).
# Each run uses the whole node (torchrun) and blocks until done.
#
# FOCUS: the new head-aligned attention rotation (experiment=optim/poet_lie_head)
# and its response to the dominant hyperparameter — the RMS per-plane rotation
# angle.  eff∠ = lr · scale · rms_c ; here lr=1e-3, scale=0.5 ⇒ eff∠ = 5e-4 · rms_c.
# Known sweet spot 0.002–0.006; dense best so far = 3.48 at eff∠ 0.006.
#
#   codexlog NAME          variant                       eff∠    question
#   poet_h_norms           head-aligned, no RMS           —      head effect vs dense poet_lie ≈ 3.50
#   poet_h_rms_c4_a002     head-aligned, RMS c=4          0.002  low end of the sweet spot
#   poet_h_rms_c8_a004     head-aligned, RMS c=8          0.004  mid sweet spot (the reference angle)
#   poet_h_rms_c12_a006    head-aligned, RMS c=12         0.006  dense was best (3.48) here — does head beat it?
#   poet_h_rms_c16_a008    head-aligned, RMS c=16         0.008  overshoot check (dense was too hot here)
#
# Pair with NODE 2's dense controls (c8/c12) for the head-vs-dense A/B at matched eff∠.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

# All runs share: 60m scale, 40x token budget, no checkpointing (sweep = read the
# val-loss curve in wandb; drop save to keep disk light — remove to keep ckpts).
COMMON="base/scale=60m training_regime=ablation_40x training.save_enabled=false"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log, and do
# NOT abort the remaining runs if one fails.
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

codexlog poet_h_norms        scripts/train_poet_lie_head.sh $COMMON experiment.name=poet_h_norms
codexlog poet_h_rms_c4_a002  scripts/train_poet_lie_head.sh $COMMON experiment.name=poet_h_rms_c4_a002  optim.poet.lie_rms=true optim.poet.lie_rms_c=4
codexlog poet_h_rms_c8_a004  scripts/train_poet_lie_head.sh $COMMON experiment.name=poet_h_rms_c8_a004  optim.poet.lie_rms=true optim.poet.lie_rms_c=8
codexlog poet_h_rms_c12_a006 scripts/train_poet_lie_head.sh $COMMON experiment.name=poet_h_rms_c12_a006 optim.poet.lie_rms=true optim.poet.lie_rms_c=12
codexlog poet_h_rms_c16_a008 scripts/train_poet_lie_head.sh $COMMON experiment.name=poet_h_rms_c16_a008 optim.poet.lie_rms=true optim.poet.lie_rms_c=16

echo "=== NODE 1 sweep complete ==="
