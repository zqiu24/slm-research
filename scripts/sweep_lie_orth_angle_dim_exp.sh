#!/usr/bin/env bash
# POET_dev arm K — per-block dimension-dependent angle exponent sweep.
#
#   θ_block = lr·scale·ortho_c · (block_size / hidden)^p      (p = angle_dim_exp)
#
# p=0 = the flat champion (every block same per-plane angle). p>0 rotates larger-dim
# blocks (fc-out 1536, fc2-in 1536) MORE than ‖W‖-proportional; p<0 rotates them LESS
# (and the small kv blocks more). b_ref = hidden is auto (scale-stable ratio). Tests the
# in/out dimension asymmetry on top of the ALTERNATING champion (the real best recipe).
#
# All overrides baked in — no paste truncation. Run:
#     bash scripts/sweep_lie_orth_angle_dim_exp.sh
# 5 sequential runs (~56 min each). p=0 reproduces the champion (~3.518) as a sanity
# check; comment it out if you trust g9i51g5l/01q4rx72. Start with em05/ep05 if you only
# want the sign of the effect first.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

# Inline codexlog (the interactive shell function does not expand in a script).
codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe (head-OFF + alternating); ONLY angle_dim_exp varies.
HELD="base/scale=60m training_regime=ablation_40x optim.lr=4e-3 optim.poet.scale=0.5 \
optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1"

run () {  # <name> <exp>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.lie_ortho_angle_dim_exp="$2" experiment.name="$1"
}

run angle_em1  -1.0
run angle_em05 -0.5
run angle_e0    0.0
run angle_ep05  0.5
run angle_ep1   1.0

echo "=== angle_dim_exp sweep complete ==="
