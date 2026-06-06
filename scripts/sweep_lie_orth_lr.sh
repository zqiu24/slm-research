#!/usr/bin/env bash
# POET lie-orth sweep A — GLOBAL LEARNING RATE.  Run on one node:
#   bash scripts/sweep_lie_orth_lr.sh
# 5 sequential runs, each uses the whole node (torchrun via the launcher) and blocks.
#
# ANCHOR: the two excellent runs were experiment=optim/poet_lie_orth (the standalone
# Muon-band orthogonalizing optimizer, q_optimizer=lie_ortho, method=muon, head-aligned),
# llama3-60m, lr=0.003, poet.scale=0.5, differing only in lie_ortho_c:
#     wandb 5sbgancm  c=8  eff∠=0.012  val/loss 3.567   (best — anchored here)
#     wandb z1gpz9y7  c=4  eff∠=0.006  val/loss 3.572
#
# This sweep varies the GLOBAL learning rate (optim.lr) and holds scale=0.5, c=8,
# method=muon.  Note global lr moves BOTH the AdamW non-rotation params (embeddings,
# norms, ...) AND the rotation angle; the sibling sweep (sweep_lie_orth_scale.sh)
# isolates the rotation lr via poet.scale.  Realized per-plane angle eff∠ = lr·scale·c
# = lr·0.5·8 = 4·lr (nominal; muon band gives ~0.75-1.0x that).
#
#   codexlog NAME       optim.lr   eff∠     question
#   lieorth_lr0.001     0.001      0.004    low lr — undertrained / too-small angle?
#   lieorth_lr0.002     0.002      0.008    just below the anchor
#   lieorth_lr0.003     0.003      0.012    ANCHOR (reproduces 5sbgancm, val 3.567)
#   lieorth_lr0.004     0.004      0.016    just above the anchor
#   lieorth_lr0.006     0.006      0.024    high lr / overshoot check

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

# All runs share: 60m scale, 40x token budget, no checkpointing (sweep = read the
# val-loss curve in wandb; drop save to keep disk light — remove to keep ckpts).
COMMON="base/scale=60m training_regime=ablation_40x training.save_enabled=false"
# Held at the best anchor's non-swept dimensions.
HELD="optim.poet.scale=0.5 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_method=muon"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log, and do
# NOT abort the remaining runs if one fails.
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

codexlog lieorth_lr0.001 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.lr=0.001 experiment.name=lieorth_lr0.001
codexlog lieorth_lr0.002 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.lr=0.002 experiment.name=lieorth_lr0.002
codexlog lieorth_lr0.003 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.lr=0.003 experiment.name=lieorth_lr0.003
codexlog lieorth_lr0.004 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.lr=0.004 experiment.name=lieorth_lr0.004
codexlog lieorth_lr0.006 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.lr=0.006 experiment.name=lieorth_lr0.006

echo "=== lie-orth LR sweep complete ==="
