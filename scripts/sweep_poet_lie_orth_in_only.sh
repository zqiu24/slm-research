#!/usr/bin/env bash
# POET one-sided (IN-ONLY) GRID sweep — COSINE scheduler (cosine_poet, 1% floor).
# Run on one node (24 sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_poet_lie_orth_in_only.sh
#
# Sibling: scripts/sweep_poet_lie_orth_out_only.sh — SAME 24-cell grid, OUT side.
# Both sweep the pure one-sided POET layer (InOnlyPOETXLinear trains ONLY oft_R_in;
# OutOnlyPOETXLinear trains ONLY oft_R_out) so the frozen side never moves W.
#
# BASE CONFIG (held at the champion lie_ortho recipe — everything NOT swept, from
# experiment=optim/poet_lie_orth_in_only):
#   single_step_x=TRUE, single_step_x_one_sided=in (FIXED side, NOT alternating),
#   q_optimizer=lie_ortho, method=muon, lie_ortho_distributed=TRUE,
#   head_aligned_attn=FALSE, merge_period=1, reinit_period=-1, block_count=1, cayley,
#   normalized init, llama3-60m, 40 tokens/param (ablation_40x), seq 256, global batch 1024.
#
# SWEPT (6 x 2 x 2 = 24):
#   optim.lr               in {1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3}  (ALSO the AdamW dense LR)
#   optim.poet.lie_ortho_c in {4, 8}                                 (per-plane angle multiplier)
#   optim.poet.scale       in {0.5, 1.0}                             (scales ONLY the rotation LR)
#
# Realized rotation angle of the TRAINED side  eff∠ = optim.lr * scale * lie_ortho_c
# (muon band ~0.75-1.0x). Known both-sides ceiling: eff∠ ~0.012 best+stable, >= ~0.016
# diverges — the high-angle cells here (up to 0.006*1.0*8 = 0.048) are boundary probes
# and may diverge; codexlog does NOT abort the rest of the grid when one run fails. One
# point of this sweep is to learn whether the one-sided angle ceiling differs.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

PREFIX="inonly"
TRAIN="scripts/train_poet_lie_orth_in_only.sh"
SCHED="scheduler=cosine_poet"

# 60m scale, 40x token budget; save stays ENABLED (train-script default — Megatron's
# wandb writer derives its dir from --save; save_interval defaults to 1e9 so no
# checkpoints are actually written during these short runs).
COMMON="base/scale=60m training_regime=ablation_40x"
# Held across all 24 cells (NOT swept). method=muon + distributed=true mirror the
# experiment YAML; left explicit so a YAML default change cannot silently move the grid.
HELD="optim.poet.lie_ortho_method=muon optim.poet.lie_ortho_distributed=true"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log and do NOT
# abort the remaining runs if one fails (e.g. a divergent high-angle cell).
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.001 0.002 0.003 0.004 0.005 0.006)
CS=(4 8)
SCALES=(0.5 1.0)

for lr in "${LRS[@]}"; do
  for c in "${CS[@]}"; do
    for s in "${SCALES[@]}"; do
      ang=$(awk -v l="$lr" -v s="$s" -v c="$c" 'BEGIN{printf "%.4f", l*s*c}')
      name="${PREFIX}_lr${lr}_c${c}_s${s}"
      echo "### ${name}: lr=${lr} c=${c} scale=${s}  eff∠=${ang}"
      codexlog "$name" "$TRAIN" $COMMON $SCHED $HELD \
        optim.lr="$lr" optim.poet.lie_ortho_c="$c" optim.poet.scale="$s" \
        experiment.name="$name"
    done
  done
done

echo "=== POET in-only grid sweep complete (24 runs) ==="
