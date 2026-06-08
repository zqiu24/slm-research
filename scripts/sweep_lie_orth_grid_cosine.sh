#!/usr/bin/env bash
# POET lie-orth GRID sweep — COSINE scheduler (cosine_poet, 1% floor).
# Run on one node (16 sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_lie_orth_grid_cosine.sh
#
# Sibling: scripts/sweep_lie_orth_grid_wsd.sh — SAME 16-cell grid, WSD scheduler.
# The two scripts are a clean paired cosine-vs-WSD comparison; only the scheduler
# (and the run-name prefix) differ.
#
# BASE CONFIG (held at the current best-POET recipe — everything NOT swept):
#   experiment=optim/poet_lie_orth, q_optimizer=lie_ortho, method=muon,
#   head_aligned_attn=FALSE, lie_alternating=TRUE (alt_every=1, both momenta fresh),
#   lie_ortho_distributed=TRUE, merge_period=1, reinit_period=-1, block_count=1,
#   cayley, normalized init, llama3-60m, 40 tokens/param (ablation_40x), seq 256,
#   global batch 1024.  This is the `1ynrrimu` champion stack minus the swept knobs.
#
# SWEPT (4 x 2 x 2 = 16):
#   optim.lr           in {1e-3, 2e-3, 3e-3, 4e-3}   (ALSO the AdamW dense LR: embeds/norms/head)
#   optim.poet.scale   in {0.25, 0.5}                (scales ONLY the rotation group's LR)
#   optim.poet.lie_ortho_c in {8, 12}                (nominal per-plane angle multiplier)
#
# Realized rotation angle  eff∠ = optim.lr * scale * lie_ortho_c  (muon band ~0.75-1.0x).
# Known stability ceiling: eff∠ ~ 0.012 is best+stable; eff∠ >= ~0.016 diverges.
# Same eff∠ at different optim.lr = the dense-LR DECOUPLING probe (the rotation angle
# is held fixed while the dense AdamW LR changes).
#
#   name             lr    scale  c   eff∠     note
#   cos_lr1_s25_c8   1e-3  0.25   8   0.002    floor of the grid
#   cos_lr1_s25_c12  1e-3  0.25   12  0.003
#   cos_lr1_s50_c8   1e-3  0.50   8   0.004
#   cos_lr1_s50_c12  1e-3  0.50   12  0.006
#   cos_lr2_s25_c8   2e-3  0.25   8   0.004
#   cos_lr2_s25_c12  2e-3  0.25   12  0.006
#   cos_lr2_s50_c8   2e-3  0.50   8   0.008
#   cos_lr2_s50_c12  2e-3  0.50   12  0.012    ceiling angle, dense-LR 2e-3  (decouple)
#   cos_lr3_s25_c8   3e-3  0.25   8   0.006
#   cos_lr3_s25_c12  3e-3  0.25   12  0.009
#   cos_lr3_s50_c8   3e-3  0.50   8   0.012    *** CHAMPION recipe (1ynrrimu, val 3.5332) ***
#   cos_lr3_s50_c12  3e-3  0.50   12  0.018    (!) expected DIVERGE (boundary probe)
#   cos_lr4_s25_c8   4e-3  0.25   8   0.008
#   cos_lr4_s25_c12  4e-3  0.25   12  0.012    ceiling angle, dense-LR 4e-3  (decouple)
#   cos_lr4_s50_c8   4e-3  0.50   8   0.016    (!) expected DIVERGE (boundary probe)
#   cos_lr4_s50_c12  4e-3  0.50   12  0.024    (!) expected DIVERGE (boundary probe)
#
# To save compute you may cull the 3 "(!) expected DIVERGE" cells; they are kept to
# map the WSD-vs-cosine divergence boundary (WSD holds peak longer -> may diverge sooner).

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

SCHED="scheduler=cosine_poet"
PREFIX="cos"

# 60m scale, 40x token budget; save stays ENABLED (train-script default — Megatron's
# wandb writer derives its dir from --save; save_interval defaults to 1e9 so no
# checkpoints are actually written during these short runs).
COMMON="base/scale=60m training_regime=ablation_40x"
# Best-POET base, held across all 16 cells (NOT swept):
HELD="optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_method=muon optim.poet.lie_ortho_distributed=true"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log and do NOT
# abort the remaining runs if one fails (e.g. a divergent boundary cell).
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.001 0.002 0.003 0.004); LTAGS=(1 2 3 4)
SCALES=(0.25 0.5);             STAGS=(25 50)
CS=(8 12)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  for j in "${!SCALES[@]}"; do
    s="${SCALES[$j]}"; st="${STAGS[$j]}"
    for c in "${CS[@]}"; do
      ang=$(awk -v l="$lr" -v s="$s" -v c="$c" 'BEGIN{printf "%.4f", l*s*c}')
      name="${PREFIX}_lr${lt}_s${st}_c${c}"
      echo "### ${name}: lr=${lr} scale=${s} c=${c}  eff∠=${ang}"
      codexlog "$name" scripts/train_poet_lie_orth.sh $COMMON $SCHED $HELD \
        optim.lr="$lr" optim.poet.scale="$s" optim.poet.lie_ortho_c="$c" \
        experiment.name="$name"
    done
  done
done

echo "=== lie-orth COSINE grid sweep complete (16 runs) ==="
