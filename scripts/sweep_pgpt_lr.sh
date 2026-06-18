#!/usr/bin/env bash
# pgpt LEARNING-RATE sweep — nGPT-anchored recipe (10 runs).
#
# Run on one node (sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_pgpt_lr.sh
#
# WHAT pgpt IS: the nGPT hypersphere architecture with the explicit per-step
# weight projection REMOVED, trained with POET (frozen base + block-orthogonal
# delta oft_R). See configs/experiments/arch/pgpt.yaml.
#
# PURPOSE: find pgpt's lr optimum on the SAME knob + cohort the nGPT lr sweep
# used (scripts/sweep_ngpt_lr.sh), so pgpt-vs-nGPT is directly comparable at each
# lr. Launcher = train_pgpt_dev.sh (llama3-60m, ablation_40x = 40 tpp, seq 256,
# gbs 1024, mbs 128, transformer_impl=local, tie_embeddings=false).
#
# HELD at the nGPT CHAMPION recipe (ngpt_lr100 / W&B 5zycv3p5, val 3.4583) so the
# ONLY axis is lr. These OVERRIDE pgpt's own defaults (wd 0 / no_warmup /
# cosine_poet) to match nGPT exactly:
#   optim.weight_decay=0.1        nGPT champion wd (scaling params stay zero-WD
#                                 via pgpt_optimizer_setup)
#   optim.ngpt.no_warmup=false    turn ON the 1% warmup (matches nGPT/adam)
#   scheduler=cosine              min_lr_ratio 0.1 — nGPT champion floor
#                                 (overrides pgpt's cosine_poet 0.01 floor)
# The POET rotation path stays at pgpt's defaults (q_optimizer=adam, Cayley,
# head-aligned, merge_period=0); optim.lr ALSO scales the rotation group's LR via
# the fixed poet.scale=0.5, so this sweep moves dense AND rotation magnitude
# together (the same coupling the POET grid used). The POET-recipe axis is swept
# separately by scripts/sweep_pgpt_orth_angle.sh.
#
#   name          lr      rotation-LR (= lr*0.5)
#   pgpt_lr10     0.001   0.0005
#   pgpt_lr20     0.002   0.001
#   pgpt_lr30     0.003   0.0015
#   pgpt_lr40     0.004   0.002
#   pgpt_lr50     0.005   0.0025
#   pgpt_lr60     0.006   0.003
#   pgpt_lr70     0.007   0.0035
#   pgpt_lr80     0.008   0.004
#   pgpt_lr90     0.009   0.0045
#   pgpt_lr100    0.01    0.005     (nGPT's own optimum lr)
#
# Idempotent: a run is SKIPPED only if its ${LOGDIR}/<name>.log shows it
# COMPLETED ("after training is done"); missing/crashed/partial runs are
# (re-)launched (rm a log to force re-run).

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in
# a non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log and
# do NOT abort the remaining runs if one fails (e.g. a divergent hot-lr cell).
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# nGPT champion recipe, held across all 10 cells (NOT swept):
HELD="optim.weight_decay=0.1 optim.ngpt.no_warmup=false scheduler=cosine"

LRS=(0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01)
LTAGS=(10 20 30 40 50 60 70 80 90 100)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  name="pgpt_lr${lt}"
  if [[ -f "${LOGDIR}/${name}.log" ]] && grep -q "after training is done" "${LOGDIR}/${name}.log"; then
    echo "### ${name}: SKIP (already completed; rm ${LOGDIR}/${name}.log to re-run)"
    continue
  fi
  echo "### ${name}: lr=${lr}  (nGPT recipe: wd 0.1, warmup on, cosine min_lr 0.1)"
  codexlog "$name" scripts/train_pgpt_dev.sh $HELD \
    optim.lr="$lr" experiment.name="$name"
done

echo "=== pgpt LR sweep complete (${#LRS[@]} runs) ==="
