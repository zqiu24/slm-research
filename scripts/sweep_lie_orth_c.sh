#!/usr/bin/env bash
# POET lie-orth sweep D — ORTHO c (the per-plane rotation angle scale).
# Run on one node:   bash scripts/sweep_lie_orth_c.sh
# 5 sequential runs, each uses the whole node (torchrun via the launcher) and blocks.
#
# ANCHOR: experiment=optim/poet_lie_orth (standalone Muon-band orthogonalizing optimizer,
# q_optimizer=lie_ortho, method=muon, head-aligned), llama3-60m, lr=0.003, poet.scale=0.5.
# The two excellent runs differ only in c:
#     wandb 5sbgancm  c=8  eff∠=0.012  val/loss 3.567   (best — anchored here)
#     wandb z1gpz9y7  c=4  eff∠=0.006  val/loss 3.572
#
# This sweep varies optim.poet.lie_ortho_c at fixed lr=0.003, scale=0.5.
# Realized per-plane angle eff∠ = lr·scale·c = 0.0015·c (nominal; muon band ~0.75-1.0x).
#
# ⚠️ DEGENERACY: orthogonalize(−m) is scale-invariant, so the update is
# `oft_R += (lr·scale·c)·X` — c and poet.scale enter ONLY as the product scale·c, both
# decayed identically by the schedule. This sweep therefore traces the SAME effective-
# angle axis as sweep_lie_orth_scale.sh. RUN ONE OR THE OTHER, not both. c is the more
# interpretable knob (and matches the RMS §2.4 c-sweep for a cross-family comparison),
# so this is the recommended angle sweep.
#
#   codexlog NAME      lie_ortho_c   eff∠     question
#   lieorth_c2         2             0.003    too-small angle (undershoot)
#   lieorth_c4         4             0.006    = z1gpz9y7
#   lieorth_c8         8             0.012    ANCHOR (= 5sbgancm, val 3.567)
#   lieorth_c12        12            0.018    RMS degraded here — does ortho tolerate it?
#   lieorth_c16        16            0.024    RMS was much worse here — does equalization help?

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

# All runs share: 60m scale, 40x token budget. Save stays ENABLED (the train-script
# default) — Megatron's wandb writer derives its dir from --save, so dropping save
# (training.save_enabled=false) leaves args.save=None and crashes _set_wandb_writer.
# save_interval defaults to 1e9, so no checkpoints are actually written during these
# short runs; disk stays light regardless.
COMMON="base/scale=60m training_regime=ablation_40x"
# Held at the best anchor's non-swept dimensions.
HELD="optim.lr=0.003 optim.poet.scale=0.5 optim.poet.lie_ortho_method=muon"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log, and do
# NOT abort the remaining runs if one fails.
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

codexlog lieorth_c2  scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.lie_ortho_c=2  experiment.name=lieorth_c2
codexlog lieorth_c4  scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.lie_ortho_c=4  experiment.name=lieorth_c4
codexlog lieorth_c8  scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.lie_ortho_c=8  experiment.name=lieorth_c8
codexlog lieorth_c12 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.lie_ortho_c=12 experiment.name=lieorth_c12
codexlog lieorth_c16 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.lie_ortho_c=16 experiment.name=lieorth_c16

echo "=== lie-orth c sweep complete ==="
