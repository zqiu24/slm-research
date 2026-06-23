#!/usr/bin/env bash
# POET init sweep — SEMI-ORTHOGONAL spectrum (init_type=orthogonal), operating-NORM axis.
#
# The most POET-specific init: a fresh semi-orthogonal base has ALL singular values equal
# (condition number = 1) — no near-zero directions that the frozen-spectrum rotation can
# never revive. It is anchored at init_scale=1.0 to the SAME per-element RMS as normalized
# (1/sqrt(in)), so init_ortho_s100 vs normalized@1.0 is a PURE-CONDITIONING A/B at matched
# norm; init_scale then sweeps the operating norm (shape stays kappa=1, scalar-invariant).
# Tests: does a perfectly-conditioned frozen base beat normalized beyond just its scale?
#
# Baseline to beat: nestON_lr4 = val/loss 3.5160 (normalized @ scale 1.0).
# All cells ride the champion recipe (lr 4e-3 / eff∠ 0.016, head-off, alt, Nesterov b1.95).
#   bash scripts/sweep_poet_init_semiortho.sh        # default GPU 0-3 (dp=4, global_batch=1024), sequential
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # default first half-node (GPU 0-3)
export MASTER_PORT="${MASTER_PORT:-6000}"                         # torchrun rendezvous (override + CUDA_VISIBLE_DEVICES=4,5,6,7 to pair a 2nd lane)
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe; only optim.poet.init_scale varies (init_type pinned orthogonal).
HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 optim.poet.lie_ortho_c=8 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=orthogonal \
cluster.gpus_per_node=4"

run () {  # <name> <init_scale>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.init_scale="$2" experiment.name="$1"
}

run init_ortho_s050 0.5    # row_rms ~0.022 (kappa=1)
run init_ortho_s070 0.7    # row_rms ~0.031
run init_ortho_s100 1.0    # row_rms ~0.044  (matched-NORM A/B vs normalized@1.0)
run init_ortho_s140 1.4    # row_rms ~0.062  (brackets the Muon equilibrium ~0.056)
run init_ortho_s200 2.0    # row_rms ~0.088

echo "=== POET init/semi-orthogonal sweep complete (baseline 3.5160) ==="
