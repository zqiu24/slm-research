#!/usr/bin/env bash
# POET head-aligned sweep — NODE 2 of 2.  Run on the other node:  bash scripts/sweep_poet_node2.sh
# 5 sequential runs, all 60m + ablation_40x.  Each run uses the whole node and blocks.
#
# FOCUS: (a) DENSE controls at the same eff∠ as NODE 1's head runs, so head-vs-dense
# is a clean A/B; (b) the orthogonal levers on head-aligned, all held at the
# reference angle c=8 (eff∠ 0.004) so they are comparable to NODE 1's poet_h_rms_c8.
# eff∠ = lr · scale · rms_c ; lr=1e-3, scale=0.5 ⇒ eff∠ = 5e-4 · rms_c.
#
#   codexlog NAME            variant                                  eff∠    question
#   poet_dense_rms_c8_a004   DENSE (poet_lie), RMS c=8                0.004  control for head poet_h_rms_c8
#   poet_dense_rms_c12_a006  DENSE (poet_lie), RMS c=12               0.006  control for head poet_h_rms_c12 (reproduce 3.48)
#   poet_h_noperm_rms_c8     head-aligned, RMS c=8, residual-perm OFF 0.004  does dropping residual Ψ hurt?
#   poet_h_exp_rms_c8        head-aligned, RMS c=8, parameterization=exp 0.004 low-order (matrix_exp) map vs Cayley-k3
#   poet_h_alt_rms_c8        head-aligned, RMS c=8, alternating ON     0.004  alternating single-sided + head-aligned
#
# Compare each head run to NODE 1's poet_h_rms_c8_a004 (same angle, lever off).

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

COMMON="base/scale=60m training_regime=ablation_40x training.save_enabled=false"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee to $LOGDIR/<name>.log; do not abort on a failed run.
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

codexlog poet_dense_rms_c8_a004   scripts/train_poet_lie.sh      $COMMON experiment.name=poet_dense_rms_c8_a004  optim.poet.lie_rms=true optim.poet.lie_rms_c=8
codexlog poet_dense_rms_c12_a006  scripts/train_poet_lie.sh      $COMMON experiment.name=poet_dense_rms_c12_a006 optim.poet.lie_rms=true optim.poet.lie_rms_c=12
codexlog poet_h_noperm_rms_c8     scripts/train_poet_lie_head.sh $COMMON experiment.name=poet_h_noperm_rms_c8 optim.poet.lie_rms=true optim.poet.lie_rms_c=8 optim.poet.head_resid_perm=false
codexlog poet_h_exp_rms_c8        scripts/train_poet_lie_head.sh $COMMON experiment.name=poet_h_exp_rms_c8    optim.poet.lie_rms=true optim.poet.lie_rms_c=8 optim.poet.parameterization=exp
codexlog poet_h_alt_rms_c8        scripts/train_poet_lie_head.sh $COMMON experiment.name=poet_h_alt_rms_c8    optim.poet.lie_rms=true optim.poet.lie_rms_c=8 optim.poet.lie_alternating=true

echo "=== NODE 2 sweep complete ==="
