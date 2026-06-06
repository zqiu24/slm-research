#!/usr/bin/env bash
# POET lie-orth sweep B — POET SCALE (the lr multiplier for the POET rotation params).
# Run on one node:   bash scripts/sweep_lie_orth_scale.sh
# 5 sequential runs, each uses the whole node (torchrun via the launcher) and blocks.
#
# ANCHOR: the two excellent runs were experiment=optim/poet_lie_orth (the standalone
# Muon-band orthogonalizing optimizer, q_optimizer=lie_ortho, method=muon, head-aligned),
# llama3-60m, lr=0.003, poet.scale=0.5, differing only in lie_ortho_c:
#     wandb 5sbgancm  c=8  eff∠=0.012  val/loss 3.567   (best — anchored here)
#     wandb z1gpz9y7  c=4  eff∠=0.006  val/loss 3.572
#
# This sweep varies optim.poet.scale — the per-group lr multiplier applied ONLY to the
# POET rotation params (oft_R).  Unlike the global-lr sweep (sweep_lie_orth_lr.sh),
# this leaves the AdamW non-rotation lr fixed and moves ONLY the rotation angle, so it
# isolates "how fast should the rotation turn?" from the rest of the network.
# Realized per-plane angle eff∠ = lr·scale·c = 0.003·scale·8 = 0.024·scale (nominal;
# muon band gives ~0.75-1.0x that).
#
#   codexlog NAME          poet.scale   eff∠     question
#   lieorth_scale0.25      0.25         0.006    half the anchor angle (= the c=4 sweet spot)
#   lieorth_scale0.5       0.50         0.012    ANCHOR (reproduces 5sbgancm, val 3.567)
#   lieorth_scale0.75      0.75         0.018    rotate faster — does the AdamW lr like it?
#   lieorth_scale1.0       1.00         0.024    rotation lr == global lr
#   lieorth_scale1.5       1.50         0.036    overshoot check (likely too hot)

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
HELD="optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_method=muon"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log, and do
# NOT abort the remaining runs if one fails.
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

codexlog lieorth_scale0.25 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.scale=0.25 experiment.name=lieorth_scale0.25
codexlog lieorth_scale0.5  scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.scale=0.5  experiment.name=lieorth_scale0.5
codexlog lieorth_scale0.75 scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.scale=0.75 experiment.name=lieorth_scale0.75
codexlog lieorth_scale1.0  scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.scale=1.0  experiment.name=lieorth_scale1.0
codexlog lieorth_scale1.5  scripts/train_poet_lie_orth.sh $COMMON $HELD optim.poet.scale=1.5  experiment.name=lieorth_scale1.5

echo "=== lie-orth POET-scale sweep complete ==="
