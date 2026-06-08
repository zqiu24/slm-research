#!/usr/bin/env bash
# POET lie-orth DENSE-LR DECOUPLING sweep — cosine scheduler.
# Run on one node (16 sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_lie_orth_decouple.sh
#
# Sibling: scripts/sweep_lie_orth_grid_cosine.sh (the lr×scale×c grid).
#
# WHY (signal from completed runs, 2026-06-09):
#   - WSD (df 0.2) at the champion recipe LOST: lodwi7cw val 3.5699 vs cosine champ
#     3.5332 (+0.037). Holding the angle at the ceiling through the stable phase keeps
#     loss high (3.82 @ 6k vs cosine 3.58) and the 20% tail can't recover. WSD->cosine
#     as df->1, so WSD can't beat cosine here -> the WSD grid was dropped.
#   - Deeper floor helps a little: cosine min_lr 0.01 (champ 3.5332) beat min_lr 0.1
#     (9mvs5hsg 3.5413) by +0.008. So we also probe an EVEN deeper floor (0.001).
#
# WHAT THIS ISOLATES: in poet_lie_momentum._build_lie_param_groups the rotation (oft_R)
# group gets lr = optim.lr * scale, while the AdamW DENSE group (embeddings / norms /
# LM head) gets lr = optim.lr (NOT scaled). The champion couples them: optim.lr=3e-3
# sets BOTH the dense LR (3x the adam-optimal 1e-3) AND, via *scale*0.5*c8, the rotation
# angle 0.012. This sweep DECOUPLES them: hold c=8 and the rotation-group LR fixed
# (rot_lr = optim.lr*scale -> fixed angle), and push the DENSE LR down to 1e-3 by
# RAISING scale to compensate. Question: is the 3e-3 dense LR too hot?
#
# BASE CONFIG (held at the best-POET recipe — everything NOT swept): head_aligned_attn
# =FALSE, lie_alternating=TRUE (alt_every=1), method=muon, distributed, merge_period=1,
# reinit_period=-1, block_count=1, cayley, lie_ortho_c=8, llama3-60m, 40 tpp, seq 256,
# global batch 1024, cosine (cosine_poet).
#
# SWEPT (4 x 2 x 2 = 16). rot_lr = optim.lr * scale; angle = rot_lr * c(=8) (muon band):
#   optim.lr (DENSE)         in {1e-3, 1.5e-3, 2e-3, 3e-3}
#   angle (via rot_lr)       in {0.008 (rot_lr 1.0e-3), 0.012 (rot_lr 1.5e-3)} -> scale = rot_lr/lr
#   scheduler.min_lr_ratio   in {0.01, 0.001}
#
#   name               lr      scale   c  angle  min_lr   note
#   dec_d10_a008_m01   1e-3    1.000   8  0.008  0.01
#   dec_d10_a012_m01   1e-3    1.500   8  0.012  0.01     champ angle, dense 1e-3 (cold dense)
#   dec_d15_a008_m01   1.5e-3  0.667   8  0.008  0.01
#   dec_d15_a012_m01   1.5e-3  1.000   8  0.012  0.01
#   dec_d20_a008_m01   2e-3    0.500   8  0.008  0.01
#   dec_d20_a012_m01   2e-3    0.750   8  0.012  0.01
#   dec_d30_a008_m01   3e-3    0.333   8  0.008  0.01
#   dec_d30_a012_m01   3e-3    0.500   8  0.012  0.01     *** = CHAMPION (1ynrrimu, 3.5332) ***
#   dec_*_*_m001       (same 8 cells, scheduler.min_lr_ratio=0.001 — deeper floor probe)
#
# Same angle (e.g. a012) across the four optim.lr rows = the decoupling line: rotation
# behavior fixed, dense AdamW LR varied. All angles are in the stable band (<=0.012); no
# divergence expected.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

PREFIX="dec"
SCHED="scheduler=cosine_poet"   # min_lr_ratio overridden per-run below

COMMON="base/scale=60m training_regime=ablation_40x"
# Best-POET base, held across all 16 cells (c FIXED at 8 — scale carries the sweep):
HELD="optim.poet.lie_ortho_c=8 optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_method=muon optim.poet.lie_ortho_distributed=true"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

DENSE=(0.001 0.0015 0.002 0.003); DTAG=(10 15 20 30)
ROT=(0.001 0.0015);               ATAG=(008 012)   # rot_lr; angle = rot_lr * 8
MINLR=(0.01 0.001);               MTAG=(01 001)

for i in "${!DENSE[@]}"; do
  lr="${DENSE[$i]}"; dt="${DTAG[$i]}"
  for j in "${!ROT[@]}"; do
    rot="${ROT[$j]}"; at="${ATAG[$j]}"
    scale=$(awk -v r="$rot" -v d="$lr" 'BEGIN{printf "%.4f", r/d}')
    ang=$(awk -v r="$rot" 'BEGIN{printf "%.4f", r*8}')
    for k in "${!MINLR[@]}"; do
      ml="${MINLR[$k]}"; mt="${MTAG[$k]}"
      name="${PREFIX}_d${dt}_a${at}_m${mt}"
      echo "### ${name}: dense_lr=${lr} scale=${scale} c=8 angle=${ang} min_lr_ratio=${ml}"
      codexlog "$name" scripts/train_poet_lie_orth.sh $COMMON $SCHED $HELD \
        optim.lr="$lr" optim.poet.scale="$scale" scheduler.min_lr_ratio="$ml" \
        experiment.name="$name"
    done
  done
done

echo "=== lie-orth DENSE-LR DECOUPLING sweep complete (16 runs) ==="
