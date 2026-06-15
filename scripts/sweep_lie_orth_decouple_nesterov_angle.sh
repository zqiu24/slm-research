#!/usr/bin/env bash
# POET lie-orth DECOUPLED ANGLE sweep, Nesterov A/B — cosine scheduler.
# Run on one node (20 sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_lie_orth_decouple_nesterov_angle.sh
#
# Siblings: scripts/sweep_lie_orth_decouple.sh (dense-lr decoupling, angle capped at
# 0.012, Nesterov OFF) and scripts/sweep_lie_orth_nesterov_lr.sh (the COUPLED global-lr
# Nesterov sweep this supersedes).
#
# WHY (motivation from the two completed sweeps):
#   - The coupled Nesterov lr sweep (nest_lr*, 2026-06-13) was still IMPROVING at its
#     top cell lr 6e-3 (val 3.5271) with no divergence -- but that cell raises the DENSE
#     AdamW lr (6e-3) AND the rotation angle (0.024) TOGETHER, so "still improving" is
#     confounded: we can't tell which knob wants to go up.
#   - The decouple sweep (dec_*, 2026-06-09, Nesterov OFF) isolated them and found that
#     at FIXED angle the dense lr wants to go UP (dense 1e-3->3e-3: 3.5563->3.5332 at
#     angle 0.012), and angle 0.012 beat 0.008 at every dense lr. BOTH knobs were still
#     climbing and NEITHER turned over -- but dec_* capped the angle at 0.012.
#   => Pin the dense lr at its optimum and push the ANGLE up past the champion's 0.016
#      until it turns over. Run the champion direction (first-moment, Nesterov OFF) AND
#      Nesterov ON at every angle, so there is finally a matched OFF reference above
#      angle 0.012 to settle whether Nesterov ever wins once the comparison is fair.
#
# DECOUPLING (see src/optim/poet_lie_momentum.py::_build_lie_param_groups): the DENSE
# AdamW group (embeddings / norms / LM head) gets lr = optim.lr; the rotation (oft_R)
# group gets lr = optim.lr*scale, giving realized angle  eff∠ = optim.lr * scale * c.
# Here c is FIXED at 8 and *scale* carries the angle, so optim.lr alone pins the dense
# lr while scale alone moves the rotation. scale = angle / (optim.lr * 8).
#
# DENSE-LR PIN: two rows. dense 3e-3 = the best FULL-AdamW lr at this scale (adam_lr30,
# val 3.4935; 2e-3->3.5065, 4e-3->3.5098). dense 4e-3 = the current best-POET champion
# ghsu7t8y's dense lr (val 3.5231, angle 0.016) -- POET's dense is only embeds/norms/head
# and the decouple data hints it wants the higher lr. Sweeping both settles the pin too.
#
# BASE CONFIG (held at the champion stack -- everything NOT swept): lie_ortho_c=8,
# method=muon, ns_steps=5, head_aligned_attn=FALSE, lie_alternating=TRUE (alt_every=1),
# lie_ortho_distributed=TRUE, merge_period=1, reinit_period=-1, block_count=1, cayley,
# normalized init, llama3-60m, 40 tokens/param (ablation_40x), seq 256, global batch
# 1024, seed 42, cosine_poet (min_lr_ratio 0.01 default -- the dec_* winner over 0.001).
#
# SWEPT (2 dense x 5 angle x 2 nesterov = 20). angle via scale = angle/(lr*8):
#   dense optim.lr  in {3e-3, 4e-3}
#   eff∠            in {0.012, 0.016, 0.020, 0.024, 0.030}
#   nesterov        in {false, true}
#
#   anchors that cross-check the harness against recorded runs:
#     decn_d30_a012_n0  should ~= dec_d30_a012_m01 (3.5332)  [decouple-champ reproduction]
#     decn_d40_a016_n0  should ~= ghsu7t8y         (3.5231)  [best-POET champion reproduction]
#
# BASELINES for the read-out: best POET = 3.5231 (ghsu7t8y); full-AdamW ceiling = 3.4935
# (adam_lr30); Muon-kimi = 3.4514. The angle column should reveal where eff∠ turns over
# at a FIXED, sane dense lr -- the thing the coupled sweep could not show.
#
# RUNTIME: ~37 min/run (from nest_lr* timestamps) x 20 = ~12 h sequential. Trim the
# ANGLE / DENSE / NEST arrays below for a faster pass; a failed cell does not abort the rest.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

PREFIX="decn"
SCHED="scheduler=cosine_poet"                 # min_lr_ratio 0.01 by default (POET floor)
COMMON="base/scale=60m training_regime=ablation_40x"
# Champion stack, held across all 20 cells (c FIXED at 8 -- scale carries the angle;
# optim.lr carries the dense lr; lie_ortho_nesterov is toggled per-cell):
HELD="optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_method=muon \
optim.poet.lie_ortho_ns_steps=5 optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_distributed=true"

# Inline equivalent of the interactive codexlog alias: tee stdout+stderr to
# $LOGDIR/<name>.log and do NOT abort the remaining runs if one cell fails/diverges.
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

DENSE=(0.003 0.004);                         DTAG=(30 40)
ANGLE=(0.012 0.016 0.020 0.024 0.030);       ATAG=(012 016 020 024 030)
NEST=(false true);                           NTAG=(0 1)

for i in "${!DENSE[@]}"; do
  lr="${DENSE[$i]}"; dt="${DTAG[$i]}"
  for n in "${!NEST[@]}"; do
    nest="${NEST[$n]}"; nt="${NTAG[$n]}"
    for j in "${!ANGLE[@]}"; do
      ang="${ANGLE[$j]}"; at="${ATAG[$j]}"
      scale=$(awk -v a="$ang" -v d="$lr" 'BEGIN{printf "%.4f", a/(d*8)}')
      name="${PREFIX}_d${dt}_a${at}_n${nt}"
      echo "### ${name}: dense_lr=${lr} scale=${scale} c=8 angle=${ang} nesterov=${nest}"
      codexlog "$name" scripts/train_poet_lie_orth.sh $COMMON $SCHED $HELD \
        optim.lr="$lr" optim.poet.scale="$scale" \
        optim.poet.lie_ortho_nesterov="$nest" \
        experiment.name="$name"
    done
  done
done

echo "=== lie-orth DECOUPLED ANGLE x Nesterov A/B sweep complete (20 runs) ==="
