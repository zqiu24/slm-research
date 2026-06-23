#!/usr/bin/env bash
# POET init sweep — muP spectrum (init_type=mup_normalized), operating-NORM axis via mup_alpha.
#
# mup_normalized row-normalizes the frozen W then SPECTRAL-scales it so the top singular
# value = mup_alpha * sqrt(d_out/d_in) (the muP-style spectral target). So its operating-norm
# knob is mup_alpha (NOT init_scale, which stays 1.0) — a spectral-norm parameterization of
# the same "what frozen norm does POET want?" axis the other three scripts sweep via row_rms.
# mup_normalized has never been A/B'd; mup_alpha=1.0 is its untouched default.
#
# Baseline to beat: nestON_lr4 = val/loss 3.5160 (init_type=normalized @ scale 1.0).
# All cells ride the champion recipe (lr 4e-3 / eff∠ 0.016, head-off, alt, Nesterov b1.95).
#   bash scripts/sweep_poet_init_mup.sh        # full node (8 GPU), sequential
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

# Champion recipe; only optim.poet.mup_alpha varies (init_type pinned mup_normalized).
HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 optim.poet.lie_ortho_c=8 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=mup_normalized"

run () {  # <name> <mup_alpha>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.mup_alpha="$2" experiment.name="$1"
}

run init_mup_a025 0.25   # spectral target 0.25*sqrt(d_out/d_in) (under-scaled)
run init_mup_a050 0.5
run init_mup_a100 1.0    # mup_normalized default (never swept)
run init_mup_a200 2.0
run init_mup_a400 4.0    # over-scaled

echo "=== POET init/mup sweep complete (baseline 3.5160) ==="
