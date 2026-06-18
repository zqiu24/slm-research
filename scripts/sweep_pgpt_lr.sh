#!/usr/bin/env bash
# pgpt DENSE-LR sweep — best-POET rotation, angle DECOUPLED (10 runs).
#
# Run on one node (sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_pgpt_lr.sh
#
# WHAT pgpt IS: the nGPT hypersphere architecture with the explicit per-step
# weight projection REMOVED, trained with POET. See configs/experiments/arch/pgpt.yaml.
#
# PURPOSE: does pgpt want a HOT dense Adam LR like nGPT (whose optimum was 1e-2)
# WHEN the rotation already uses the best-POET recipe? This holds the lie_ortho
# CHAMPION rotation (cos_lr4_s50_c8 / W&B ghsu7t8y, val 3.5231) FIXED and sweeps
# only the dense AdamW LR (embeddings / norms / nGPT scaling params). Sibling
# scripts/sweep_pgpt_orth_angle.sh sweeps the rotation ANGLE instead.
#
# DECOUPLING (the point): optim.lr is BOTH the dense AdamW LR and (via poet.scale)
# the rotation-group LR, and for lie_ortho the per-plane angle is
#   eff∠ = optim.lr * scale * lie_ortho_c.
# A naive lr sweep at fixed scale would drag eff∠ from 0.004 to 0.04 and the hot
# cells would diverge on the ANGLE, not the dense LR. So we pin eff∠ at the
# champion 0.016 by setting per cell
#   scale = 0.016 / (lr * c) = 0.002 / lr      (c = 8)
# => rotation-group LR = lr*scale = 0.002 CONSTANT, eff∠ = 0.016 CONSTANT across
# all cells; ONLY the dense LR varies. (Method = POET_dev.md SS2.6 G.)
#
# HELD — lie_ortho CHAMPION rotation + POET-champion schedule (NOT swept):
#   optim.poet.q_optimizer=lie_ortho         Muon-orthogonalized Lie-momentum
#   optim.poet.lie_ortho_method=muon         quintic Newton-Schulz band (~5 steps)
#   optim.poet.lie_ortho_c=8                  nominal per-plane angle multiplier
#   optim.poet.merge_period=1                 single-step merge (champion setting)
#   optim.poet.head_aligned_attn=false        HEAD-OFF (champion was head-off)
#   optim.poet.lie_alternating=true           alternate written side, both momenta
#   optim.poet.lie_alternate_every=1            fresh (the champion's win)
#   optim.poet.lie_ortho_distributed=true     shard NS across DP (identical result)
#   optim.weight_decay=0.1                     POET champion wd
#   optim.ngpt.no_warmup=false                 1% warmup ON
#   scheduler=cosine_poet                      min_lr_ratio 0.01 — POET champion floor
# Same schedule as sweep_pgpt_orth_angle.sh (cosine_poet, 0.01 floor), so A and B
# differ ONLY in the swept axis (A = dense lr, B = rotation angle). The rotation
# group_lr (hence eff∠) anneals on the 0.01 floor, matching ghsu7t8y; the lr=4e-3
# cell is the champion ROTATION RECIPE applied to the pgpt arch. (A still keeps
# warmup ON via no_warmup=false; B leaves pgpt's no_warmup=true default = warmup OFF.)
#
#   name          dense lr   scale (=0.002/lr)   eff∠ (=lr*scale*8)
#   pgpt_lr10     0.001      2.0                 0.016
#   pgpt_lr20     0.002      1.0                 0.016
#   pgpt_lr30     0.003      0.667               0.016
#   pgpt_lr40     0.004      0.5                 0.016   (= champion rotation recipe)
#   pgpt_lr50     0.005      0.4                 0.016
#   pgpt_lr60     0.006      0.333               0.016
#   pgpt_lr70     0.007      0.286               0.016
#   pgpt_lr80     0.008      0.25                0.016
#   pgpt_lr90     0.009      0.222               0.016
#   pgpt_lr100    0.01       0.2                 0.016   (nGPT's own optimum dense lr)
#
# Idempotent: a run is SKIPPED only if its ${LOGDIR}/<name>.log shows it
# COMPLETED ("after training is done"); missing/crashed/partial runs are
# (re-)launched (rm a log to force re-run).

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in
# a non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log and
# do NOT abort the remaining runs if one fails (e.g. a divergent hot-lr cell).
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# lie_ortho champion rotation + POET-champion schedule, held across all 10 cells:
HELD="optim.weight_decay=0.1 optim.ngpt.no_warmup=false scheduler=cosine_poet \
optim.poet.q_optimizer=lie_ortho optim.poet.lie_ortho_method=muon \
optim.poet.lie_ortho_c=8 optim.poet.merge_period=1 \
optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_distributed=true"

LRS=(0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01)
LTAGS=(10 20 30 40 50 60 70 80 90 100)

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"; lt="${LTAGS[$i]}"
  # Hold eff∠ = lr*scale*8 = 0.016  =>  scale = 0.002/lr (rotation group_lr = 0.002).
  scale=$(awk -v l="$lr" 'BEGIN{printf "%.6g", 0.002/l}')
  name="pgpt_lr${lt}"
  if [[ -f "${LOGDIR}/${name}.log" ]] && grep -q "after training is done" "${LOGDIR}/${name}.log"; then
    echo "### ${name}: SKIP (already completed; rm ${LOGDIR}/${name}.log to re-run)"
    continue
  fi
  echo "### ${name}: dense_lr=${lr} scale=${scale}  (eff∠=0.016 held; lie_ortho champion rotation)"
  codexlog "$name" scripts/train_pgpt_dev.sh $HELD \
    optim.lr="$lr" optim.poet.scale="$scale" experiment.name="$name"
done

echo "=== pgpt DENSE-LR sweep complete (${#LRS[@]} runs, eff∠ held at 0.016) ==="
