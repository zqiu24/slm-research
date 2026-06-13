#!/usr/bin/env bash
# POET lie-orth sweep — GLOBAL LEARNING RATE *with Nesterov look-ahead* (NEW knob).
# Run on one node:
#   bash scripts/sweep_lie_orth_nesterov_lr.sh
# 5 sequential runs, each uses the whole node (torchrun via the launcher) and blocks.
#
# WHY: the current best POET run (`ghsu7t8y`, val/loss 3.5231 — see POET_dev.md §2.6)
# uses the plain first-moment EMA direction (orthogonalize -m). It does NOT use a
# Nesterov look-ahead. This sweep turns Nesterov ON and re-finds the best lr for it.
#
# WHAT NESTEROV DOES HERE (optim.poet.lie_ortho_nesterov=true): the lie_m EMA is
# unchanged (m = b1*m + (1-b1)*g); the optimizer instead ORTHOGONALIZES the Muon-style
# look-ahead direction  (1-b1)*g + b1*m  (= grad.lerp(m, b1)) instead of the bare m.
# This is exactly modern Muon's `update = grad.lerp(momentum, beta)`. Skew/rotation
# branch only — the AdamW dense branch (embeds/norms/head) is untouched.
# See src/optim/poet_lie_orth.py and docs/muon_orthogonalizing_optimizer_poet.md.
#
# BASE CONFIG (held at the current best-POET recipe — everything NOT swept; identical
# to scripts/sweep_lie_orth_grid_cosine.sh's HELD, i.e. the `ghsu7t8y` champion stack):
#   experiment=optim/poet_lie_orth, q_optimizer=lie_ortho, method=muon, ns_steps=5,
#   head_aligned_attn=FALSE, lie_alternating=TRUE (alt_every=1, both momenta fresh),
#   lie_ortho_distributed=TRUE, scale=0.5, lie_ortho_c=8, first-moment-only,
#   merge_period=1, reinit_period=-1, block_count=1, cayley, normalized init,
#   llama3-60m, 40 tokens/param (ablation_40x), seq 256, global batch 1024, seed 42,
#   cosine scheduler (cosine_poet, min_lr_ratio 0.01).
#   ===> + optim.poet.lie_ortho_nesterov=true on ALL five cells.
#
# SWEPT: optim.lr in {1e-3, 2e-3, 3e-3, 4e-3, 6e-3} (mirrors scripts/sweep_lie_orth_lr.sh
# so the Nesterov-ON curve is directly comparable to the non-Nesterov lr sweep).
# Global lr moves BOTH the AdamW dense params AND the rotation angle. Realized per-plane
# angle  eff∠ = lr * scale * lie_ortho_c = lr * 0.5 * 8 = 4*lr (nominal; muon band gives
# ~0.75-1.0x that). NOTE: Nesterov makes each step more aggressive than the bare-m
# direction, so the stability ceiling may move DOWN — the eff∠ 0.016 that was the
# non-Nesterov sweet spot (lr 4e-3) could be too hot here, and the optimum may sit at
# lr 3e-3 (eff∠ 0.012) or lower. That shift is exactly what this sweep measures.
#
#   codexlog NAME           optim.lr   eff∠     question
#   nest_lr0.001            0.001      0.004    low lr — undertrained / too-small angle?
#   nest_lr0.002            0.002      0.008    below the non-Nesterov anchor
#   nest_lr0.003            0.003      0.012    prior champ angle (was 3.5332 w/o Nesterov)
#   nest_lr0.004            0.004      0.016    champ angle (ghsu7t8y 3.5231 w/o Nesterov)
#   nest_lr0.006            0.006      0.024    high lr / overshoot — expected DIVERGE (boundary)
#
# BASELINE for the A/B: the existing non-Nesterov champion `ghsu7t8y` (val 3.5231,
# lr 4e-3) — already in POET_dev.md §2.6. No nesterov=off cell is run here; compare
# each nest_* run to that recorded baseline at the matching lr.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

# 60m scale, 40x token budget; save stays ENABLED (train-script default — Megatron's
# wandb writer derives its dir from --save; save_interval defaults to 1e9 so no
# checkpoints are actually written during these short runs).
COMMON="base/scale=60m training_regime=ablation_40x"
SCHED="scheduler=cosine_poet"
# Best-POET base, held across all cells (NOT swept) + Nesterov ON.
HELD="optim.poet.scale=0.5 optim.poet.lie_ortho_c=8 \
optim.poet.lie_ortho_method=muon optim.poet.lie_ortho_ns_steps=5 \
optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_distributed=true \
optim.poet.lie_ortho_nesterov=true"

# Inline equivalent of the interactive `codexlog` alias (aliases do NOT expand in a
# non-interactive script): tee a run's stdout+stderr to $LOGDIR/<name>.log and do NOT
# abort the remaining runs if one fails (e.g. a divergent boundary cell).
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.001 0.002 0.003 0.004 0.006)
for lr in "${LRS[@]}"; do
  ang=$(awk -v l="$lr" 'BEGIN{printf "%.4f", l*0.5*8}')
  name="nest_lr${lr}"
  echo "### ${name}: lr=${lr} scale=0.5 c=8 nesterov=true  eff∠=${ang}"
  codexlog "$name" scripts/train_poet_lie_orth.sh $COMMON $SCHED $HELD \
    optim.lr="$lr" experiment.name="$name"
done

echo "=== lie-orth Nesterov LR sweep complete (5 runs) ==="
