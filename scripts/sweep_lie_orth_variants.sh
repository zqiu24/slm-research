#!/usr/bin/env bash
# POET lie-orth sweep C — VARIANTS / ABLATIONS at the champion angle.
# Run on one node:   bash scripts/sweep_lie_orth_variants.sh
# 5 sequential runs, each uses the whole node (torchrun via the launcher) and blocks.
#
# Sweeps A (sweep_lie_orth_lr.sh) and B (sweep_lie_orth_scale.sh) tune HOW BIG the
# rotation should be (both move eff∠ = lr·scale·c).  This node instead holds the proven
# champion angle and asks WHICH VARIANT wins — the open questions from
# docs/muon_orthogonalizing_optimizer_poet.md that the angle sweeps don't touch.
#
# ALL runs are matched at eff∠ ≈ 0.012 (lr=0.003, scale=0.5, c=8, head-aligned), the
# config of the best run so far (wandb 5sbgancm, val/loss 3.567).  Each run changes ONE
# thing vs the control:
#
#   codexlog NAME           change vs control                 question
#   lieorth_c8_muon         (none — the control = 5sbgancm)   in-node baseline for the ablations
#   lieorth_c8_spectral     method=spectral, ns_steps=20      band vs EXACT σ=1 — is the cheap
#                                                             Muon band as good as exact equal angles?
#   lierms_c8               lie_rms instead of lie_ortho       ORTHO vs RMS (§7 core) — are the
#                                                             gradient's relative plane angles signal or noise?
#   lieorth_c8_2mom         lie_ortho_use_second_moment=true   1st- vs 2nd-moment — does Adam-style v help
#                                                             or get undone by orthogonalization?
#   lieorth_c8_nohead       head_aligned_attn=false            head-aligned vs plain block rotation
#
# Caveat: spectral c=8 realizes exactly ∠0.012, while muon c=8 lands in a band whose
# median is ~0.75-1.0× that — so if spectral wins, re-check at spectral c≈6 to rule out
# the slightly-larger angle.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

# All runs share: 60m scale, 40x token budget. Save stays ENABLED (the train-script
# default) — Megatron's wandb writer derives its dir from --save, so dropping save
# (training.save_enabled=false) leaves args.save=None and crashes _set_wandb_writer.
# save_interval defaults to 1e9, so no checkpoints are actually written during these
# short runs; disk stays light regardless.
COMMON="base/scale=60m training_regime=ablation_40x"
# The champion angle, held across every run (eff∠ = lr·scale·c = 0.012).
ANGLE="optim.lr=0.003 optim.poet.scale=0.5"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log, and do
# NOT abort the remaining runs if one fails.
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# 1. Control — muon-band lie-orth, c=8 (reproduces 5sbgancm).
codexlog lieorth_c8_muon \
  scripts/train_poet_lie_orth.sh $COMMON $ANGLE optim.poet.lie_ortho_c=8 \
  optim.poet.lie_ortho_method=muon experiment.name=lieorth_c8_muon

# 2. Exact σ=1 (Löwdin) — band vs exact.
codexlog lieorth_c8_spectral \
  scripts/train_poet_lie_orth.sh $COMMON $ANGLE optim.poet.lie_ortho_c=8 \
  optim.poet.lie_ortho_method=spectral optim.poet.lie_ortho_ns_steps=20 \
  experiment.name=lieorth_c8_spectral

# 3. RMS sibling at the matched angle — ortho vs RMS (§7).
codexlog lierms_c8 \
  scripts/train_poet_lie_rms.sh $COMMON $ANGLE optim.poet.lie_rms_c=8 \
  experiment.name=lierms_c8

# 4. Second moment on — 1st- vs 2nd-moment ablation.
codexlog lieorth_c8_2mom \
  scripts/train_poet_lie_orth.sh $COMMON $ANGLE optim.poet.lie_ortho_c=8 \
  optim.poet.lie_ortho_method=muon optim.poet.lie_ortho_use_second_moment=true \
  experiment.name=lieorth_c8_2mom

# 5. Head-aligned off — per-head vs plain block rotation.
codexlog lieorth_c8_nohead \
  scripts/train_poet_lie_orth.sh $COMMON $ANGLE optim.poet.lie_ortho_c=8 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  experiment.name=lieorth_c8_nohead

echo "=== lie-orth variants sweep complete ==="
