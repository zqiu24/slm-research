#!/usr/bin/env bash
# POET init sweep — the frozen base W's SHAPE (init_type) × NORM (init_scale).
#
# Motivation (POET_dev.md §2.1 "not separately ablated", §2.7 norm-monitor):
# POET freezes W and only rotates it, so (a) W's singular-value spectrum at init is
# PERMANENT (orthogonal rotation preserves singular values) and (b) the init norm IS
# the operating norm — §2.7 shows POET's per-element RMS is flat (~1.07x over 9k steps)
# while Adam/Muon grow ~3.2-3.4x to an equilibrium (~0.045-0.056 row_rms). The default
# `normalized` init lands at row_rms ~0.044 (= 1/sqrt(in)), suspiciously close to where
# Adam equilibrates — but this has never been A/B'd for val/loss.
#
# Two independent axes (a scalar multiply scales every sigma equally, so init_scale
# moves the NORM without touching the SHAPE / condition number):
#   init_type  = spectrum SHAPE : none | normalized | orthogonal(kappa=1)
#   init_scale = operating NORM : final scalar multiply on W (default 1.0 = champion)
#
# Baseline to beat: nestON_lr4 = val/loss 3.5160 (current champion, = normalized @ scale 1.0).
# All cells ride the champion recipe (lr 4e-3 / scale 0.5 / c8 -> eff∠ 0.016, head-off,
# alternating, Nesterov b1.95, distributed, cosine min_lr 0.01) and vary ONLY init.
#
#   bash scripts/sweep_poet_init.sh           # full node (8 GPU), sequential
# To run two lanes concurrently, split the run-list across two copies with
# CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=6000 vs 4,5,6,7 / 6010 and cluster.gpus_per_node=4.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe; only optim.poet.init_type / init_scale vary per cell.
HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 optim.poet.lie_ortho_c=8 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true"

run () {  # <name> <init_type> <init_scale>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.init_type="$2" optim.poet.init_scale="$3" experiment.name="$1"
}

# --- Axis A: operating NORM (shape = normalized, the default). ----------------
# normalized row_rms = init_scale / sqrt(in); the Adam/Muon equilibrium ~0.045-0.056
# sits around scale ~1.0-1.3, so the hypothesis is the optimum is at or slightly above 1.0.
run init_n_s050 normalized 0.5    # row_rms ~0.022 (under-scaled)
run init_n_s070 normalized 0.7    # row_rms ~0.031
run init_n_s100 normalized 1.0    # row_rms ~0.044  == CURRENT CHAMPION ANCHOR (~3.516)
run init_n_s140 normalized 1.4    # row_rms ~0.062  (brackets the Muon equilibrium ~0.056)
run init_n_s200 normalized 2.0    # row_rms ~0.088 (over-scaled)

# --- Axis B: spectrum SHAPE at matched norm. ----------------------------------
# orthogonal @ 1.0 is anchored to the SAME per-element RMS as normalized @ 1.0, so
# normalized vs orthogonal here is a pure conditioning (kappa) A/B. `none` @ 2.75 lifts
# raw init's native ~0.016 up to normalized's ~0.044 -> raw spectrum (MP + residual
# 1/sqrt(2L) downscale) at matched norm. `none` @ 1.0 is the raw native-norm baseline.
run init_o_s100 orthogonal 1.0    # kappa=1, matched norm vs init_n_s100
run init_o_s140 orthogonal 1.4    # kappa=1 at the higher norm (shape x norm interaction)
run init_r_s100 none       1.0    # raw native (row_rms ~0.016, residual-structured)
run init_r_s275 none       2.75   # raw spectrum lifted to ~0.044 (matched-norm raw A/B)

echo "=== POET init sweep complete ==="
echo "Read: runs/<name>-llama3-60m-s42-*/**/wandb-summary.json  (val/loss); baseline 3.5160"
