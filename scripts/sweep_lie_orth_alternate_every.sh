#!/usr/bin/env bash
# POET lie-orth sweep — ALTERNATE_EVERY (write-cadence / momentum averaging-window).
#   bash scripts/sweep_lie_orth_alternate_every.sh
# 4 sequential runs, each uses the whole node (torchrun via the launcher) and blocks.
#
# WHY THIS SWEEP (not blind): the Tier-0 coordination diagnostics on the champion
# (W&B qjapxj18, ANALYSIS.md §17.6) FALSIFIED gauge-redundancy — cos(D_out,D_in)≈0
# and gram_cond≈1.25 for the whole run, so the two sides are orthogonal and the
# alternating win is NOT spatial overlap/cancellation. mom_cos is near-white
# (~−0.15 → 0 as LR decays), i.e. the per-step rotation gradient is low-SNR and the
# persistent EMA does genuine noise-AVERAGING (freezing it → 4.22, au92x0pj). This
# sweep tests that surviving mechanism directly: alternate_every=k writes each side
# for k consecutive steps then rests it k steps (both momenta keep advancing), so
# larger k = longer averaging/rest between write bursts. alternate_every has been
# PINNED at 1 and never swept. SLM_POET_COORD_DIAG=1 is on, so each run logs the
# mom_cos / norm curves to read whether more averaging shifts the SNR.
#
#   codexlog NAME       alternate_every   question
#   lieorth_alt1        1                 champion baseline (reproduces qjapxj18 recipe)
#   lieorth_alt2        2                 mild burst/rest — slightly longer averaging window
#   lieorth_alt4        4                 longer rest between bursts
#   lieorth_alt8        8                 long rest — over-averaging / staleness onset?
#
# alternate_every does NOT change the realized per-plane angle (eff∠ = lr·scale·c =
# 0.016 throughout); it only changes the per-side write cadence. A non-monotone
# curve (interior optimum) ⇒ there is a best averaging window; monotone-worse with k
# ⇒ k=1 already optimal and the win is per-step fresh re-evaluation, not longer EMA.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

# Tier-0 coordination diagnostics ON for every run (exported so torchrun workers,
# where the patch runs, inherit it). Logs poet_coord/* to W&B every 250 steps.
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

# All runs share: 60m scale, 40x token budget (same cohort as the lr/scale sweeps).
COMMON="base/scale=60m training_regime=ablation_40x"
# Held at the champion qjapxj18 non-swept dims (head-OFF + alternating, lr4e-3 /
# scale0.5 / c8 → eff∠ 0.016, val/loss 3.5231).
HELD="optim.lr=4e-3 optim.poet.scale=0.5 optim.poet.lie_ortho_c=8 \
optim.poet.lie_ortho_method=muon optim.poet.lie_alternating=true \
optim.poet.head_aligned_attn=false"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log, and do
# NOT abort the remaining runs if one fails.
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

codexlog lieorth_alt1 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.lie_alternate_every=1 experiment.name=lieorth_alt1
codexlog lieorth_alt2 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.lie_alternate_every=2 experiment.name=lieorth_alt2
codexlog lieorth_alt4 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.lie_alternate_every=4 experiment.name=lieorth_alt4
codexlog lieorth_alt8 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.lie_alternate_every=8 experiment.name=lieorth_alt8

echo "=== lie-orth alternate_every sweep complete ==="
