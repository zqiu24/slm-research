#!/usr/bin/env bash
# pgpt POET ROTATION-ANGLE sweep — POET-champion-anchored (lie_ortho, 10 runs).
#
# Run on one node (sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_pgpt_orth_angle.sh
#
# WHAT pgpt IS: the nGPT hypersphere architecture with the per-step weight
# projection REMOVED, trained with POET. See configs/experiments/arch/pgpt.yaml.
#
# PURPOSE: port the BEST-POET rotation recipe (the lie_ortho champion
# cos_lr4_s50_c8 / W&B ghsu7t8y, val 3.5231) onto pgpt and find pgpt's
# rotation-angle optimum. pgpt's OWN default POET path is q_optimizer=adam
# (Cayley); this sweep OVERRIDES it to the standalone Muon-orthogonalizing
# Lie-momentum optimizer the POET champion used.
#
# HELD at the lie_ortho CHAMPION stack (everything NOT swept) — these OVERRIDE
# pgpt's POET defaults:
#   optim.poet.q_optimizer=lie_ortho        Muon-orthogonalized Lie-momentum
#   optim.poet.lie_ortho_method=muon        quintic Newton-Schulz band (~5 steps)
#   optim.poet.scale=0.5                     rotation-group LR multiplier
#   optim.poet.merge_period=1                single-step merge (champion setting)
#   optim.poet.head_aligned_attn=false       HEAD-OFF (champion was head-off)
#   optim.poet.lie_alternating=true          alternate the written side, both
#   optim.poet.lie_alternate_every=1           momenta fresh (the champion's win)
#   optim.poet.lie_ortho_distributed=true    shard NS across DP (identical result)
#   optim.weight_decay=0.1                    POET champion wd
#   scheduler=cosine_poet                     min_lr_ratio 0.01 (POET floor;
#                                             pgpt's default, passed explicitly)
#
# SWEPT (10): the realized per-plane rotation angle
#   eff∠ = optim.lr * scale * lie_ortho_c   (muon band gives ~0.75-1.0x).
# scale=0.5 and lie_ortho_c=8 are FIXED at the champion pair; optim.lr walks eff∠
# across the champion's 0.016 sweet spot and into the divergence zone, to locate
# pgpt's OWN ceiling (plain-llama3 POET diverged at eff∠ >= ~0.024; pgpt's
# hypersphere arch may differ). NOTE: optim.lr is ALSO the dense AdamW LR for
# embeds/norms/scaling — this single axis couples dense-LR and rotation-angle,
# exactly as the POET grid did. The hot-dense-LR (nGPT) direction is swept
# separately by scripts/sweep_pgpt_lr.sh.
#
#   name           lr      eff∠ (=lr*0.5*8)   note
#   pgpt_orth_a06  0.0015  0.006
#   pgpt_orth_a08  0.002   0.008
#   pgpt_orth_a10  0.0025  0.010
#   pgpt_orth_a12  0.003   0.012     prior-POET-champ angle (1ynrrimu, 3.5332)
#   pgpt_orth_a14  0.0035  0.014
#   pgpt_orth_a16  0.004   0.016     *** POET CHAMPION angle (ghsu7t8y, 3.5231) ***
#   pgpt_orth_a18  0.0045  0.018     POET stability-ceiling edge
#   pgpt_orth_a20  0.005   0.020
#   pgpt_orth_a24  0.006   0.024     plain-POET DIVERGED here (boundary probe)
#   pgpt_orth_a28  0.007   0.028     boundary probe
#
# GUARD: arch/pgpt + q_optimizer=lie_ortho has not been run end-to-end before
# (pgpt's design default is q_optimizer=adam). Before the 10 real runs, a ~30-step
# composition smoke at the champion cell (lr=0.004) confirms the lie_ortho
# optimizer builds, steps, and merges on the pgpt patch stack. If it does not
# complete, the sweep ABORTS before burning the full grid.
#
# Idempotent: a run (and the guard) is SKIPPED only if its ${LOGDIR}/<name>.log
# shows it COMPLETED ("after training is done"); rm a log to force re-run.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in
# a non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log and
# do NOT abort the remaining runs if one fails (e.g. a divergent boundary cell).
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# lie_ortho champion stack, held across every cell (NOT swept):
HELD="optim.weight_decay=0.1 scheduler=cosine_poet \
optim.poet.q_optimizer=lie_ortho optim.poet.lie_ortho_method=muon \
optim.poet.scale=0.5 optim.poet.lie_ortho_c=8 optim.poet.merge_period=1 \
optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_distributed=true"

# ---- composition guard: ~30-step smoke before committing the 10 full runs -----
# training.total_tokens=8e6 -> --train-samples 31250 / gbs 1024 ~= 30 iters.
GUARD_LOG="${LOGDIR}/pgpt_orth_guard.log"
if [[ -f "$GUARD_LOG" ]] && grep -q "after training is done" "$GUARD_LOG"; then
  echo "### guard: SKIP (pgpt+lie_ortho smoke already passed; rm ${GUARD_LOG} to re-run)"
else
  echo "### guard: ~30-step pgpt+lie_ortho composition smoke (champion cell, lr=0.004, eff∠=0.016)"
  codexlog pgpt_orth_guard scripts/train_pgpt_dev.sh $HELD \
    training.total_tokens=8000000 optim.lr=0.004 experiment.name=pgpt_orth_guard
  if ! grep -q "after training is done" "$GUARD_LOG"; then
    echo "!!! guard FAILED — pgpt+lie_ortho did not complete the smoke; ABORTING sweep." >&2
    echo "    inspect ${GUARD_LOG}" >&2
    exit 1
  fi
  echo "### guard PASSED — proceeding to the 10-run angle sweep."
fi

LRS=(0.0015 0.002 0.0025 0.003 0.0035 0.004 0.0045 0.005 0.006 0.007)
ATAGS=(06 08 10 12 14 16 18 20 24 28)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; at="${ATAGS[$i]}"
  ang=$(awk -v l="$lr" 'BEGIN{printf "%.3f", l*0.5*8}')
  name="pgpt_orth_a${at}"
  if [[ -f "${LOGDIR}/${name}.log" ]] && grep -q "after training is done" "${LOGDIR}/${name}.log"; then
    echo "### ${name}: SKIP (already completed; rm ${LOGDIR}/${name}.log to re-run)"
    continue
  fi
  echo "### ${name}: lr=${lr} scale=0.5 c=8  eff∠=${ang}"
  codexlog "$name" scripts/train_pgpt_dev.sh $HELD \
    optim.lr="$lr" experiment.name="$name"
done

echo "=== pgpt POET rotation-angle sweep complete (${#LRS[@]} runs) ==="
