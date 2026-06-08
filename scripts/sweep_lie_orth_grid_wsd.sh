#!/usr/bin/env bash
# POET lie-orth GRID sweep — WSD scheduler (wsd_poet: 1% floor, 20% cosine tail).
# Run on one node (16 sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_lie_orth_grid_wsd.sh
#
# Sibling: scripts/sweep_lie_orth_grid_cosine.sh — SAME 16-cell grid, cosine_poet.
# The two scripts are a clean paired cosine-vs-WSD comparison; only the scheduler
# (and the run-name prefix) differ.
#
# Scheduler starting point taken from the first poet_lie_orth WSD run
# (runs/poet_lie_orth-llama3-60m-s42-20260608T222621Z, lr3e-3/scale0.5/c8): WSD with
# min_lr_ratio=0.01 (POET 1% floor, matches cosine_poet), wsd_decay_fraction=0.2,
# wsd_decay_style=cosine.  Wired as configs/scheduler/wsd_poet.yaml.
#
# WHY WSD for POET: under cosine the rotation angle sits at its safe max only for an
# instant after warmup and decays for the rest of the run, so POET under-rotates
# through the whole middle (its loss curve is convex / late-diving vs adam/muon's
# straight power law). WSD holds the ceiling angle through the stable phase and
# captures the basin in a short deep anneal — a direct attack on that shape.
#
# BASE CONFIG (held at the current best-POET recipe — everything NOT swept):
#   experiment=optim/poet_lie_orth, q_optimizer=lie_ortho, method=muon,
#   head_aligned_attn=FALSE, lie_alternating=TRUE (alt_every=1, both momenta fresh),
#   lie_ortho_distributed=TRUE, merge_period=1, reinit_period=-1, block_count=1,
#   cayley, normalized init, llama3-60m, 40 tokens/param (ablation_40x), seq 256,
#   global batch 1024.
#
# SWEPT (4 x 2 x 2 = 16):
#   optim.lr           in {1e-3, 2e-3, 3e-3, 4e-3}   (ALSO the AdamW dense LR)
#   optim.poet.scale   in {0.25, 0.5}                (scales ONLY the rotation group's LR)
#   optim.poet.lie_ortho_c in {8, 12}                (nominal per-plane angle multiplier)
#
# Realized rotation angle  eff∠ = optim.lr * scale * lie_ortho_c  (muon band ~0.75-1.0x).
# NOTE: WSD holds peak LR through ~80% of the run (vs cosine decaying immediately), so a
# given eff∠ is sustained MUCH longer — the divergence boundary may sit at a LOWER angle
# than under cosine. The lr3e-3/scale0.5/c8 (eff∠ 0.012) starting-point run survived the
# stable phase to step 5.5k without diverging.
#
#   name             lr    scale  c   eff∠     note
#   wsd_lr1_s25_c8   1e-3  0.25   8   0.002    floor of the grid
#   wsd_lr1_s25_c12  1e-3  0.25   12  0.003
#   wsd_lr1_s50_c8   1e-3  0.50   8   0.004
#   wsd_lr1_s50_c12  1e-3  0.50   12  0.006
#   wsd_lr2_s25_c8   2e-3  0.25   8   0.004
#   wsd_lr2_s25_c12  2e-3  0.25   12  0.006
#   wsd_lr2_s50_c8   2e-3  0.50   8   0.008
#   wsd_lr2_s50_c12  2e-3  0.50   12  0.012    ceiling angle, dense-LR 2e-3  (decouple)
#   wsd_lr3_s25_c8   3e-3  0.25   8   0.006
#   wsd_lr3_s25_c12  3e-3  0.25   12  0.009
#   wsd_lr3_s50_c8   3e-3  0.50   8   0.012    *** WSD starting-point recipe (222621Z) ***
#   wsd_lr3_s50_c12  3e-3  0.50   12  0.018    (!) expected DIVERGE (boundary probe)
#   wsd_lr4_s25_c8   4e-3  0.25   8   0.008
#   wsd_lr4_s25_c12  4e-3  0.25   12  0.012    ceiling angle, dense-LR 4e-3  (decouple)
#   wsd_lr4_s50_c8   4e-3  0.50   8   0.016    (!) expected DIVERGE (boundary probe)
#   wsd_lr4_s50_c12  4e-3  0.50   12  0.024    (!) expected DIVERGE (boundary probe)
#
# To save compute you may cull the 3 "(!) expected DIVERGE" cells.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

SCHED="scheduler=wsd_poet"
PREFIX="wsd"

COMMON="base/scale=60m training_regime=ablation_40x"
HELD="optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_method=muon optim.poet.lie_ortho_distributed=true"

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

echo "=== lie-orth WSD grid sweep complete (16 runs) ==="
