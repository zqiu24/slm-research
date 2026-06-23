#!/usr/bin/env bash
# POET init sweep — NORMALIZED spectrum (unit per-row L2 norm), operating-NORM axis.
#
# init_type=normalized is the current default & champion anchor. POET freezes W and only
# rotates it, so the init norm IS the operating norm (POET_dev.md §2.7: POET RMS flat
# ~1.07x vs Adam/Muon's ~3.2-3.4x growth to a ~0.045-0.056 equilibrium). normalized lands
# at row_rms = init_scale/sqrt(in) ~ 0.044 at scale 1.0 — close to the Adam equilibrium it
# can't grow into. This sweeps init_scale to ask: does POET want a HIGHER frozen norm?
#
# Baseline to beat: nestON_lr4 = val/loss 3.5160 (= this script's init_norm_s100 cell).
# All cells ride the champion recipe (lr 4e-3 / eff∠ 0.016, head-off, alt, Nesterov b1.95).
#   bash scripts/sweep_poet_init_normalized.sh        # full node (8 GPU), sequential
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

# Champion recipe; only optim.poet.init_scale varies (init_type pinned normalized).
HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 optim.poet.lie_ortho_c=8 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true optim.poet.init_type=normalized"

run () {  # <name> <init_scale>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.init_scale="$2" experiment.name="$1"
}

run init_norm_s050 0.5    # row_rms ~0.022 (under-scaled)
run init_norm_s070 0.7    # row_rms ~0.031
run init_norm_s100 1.0    # row_rms ~0.044  == CHAMPION ANCHOR (~3.516)
run init_norm_s140 1.4    # row_rms ~0.062  (brackets the Muon equilibrium ~0.056)
run init_norm_s200 2.0    # row_rms ~0.088 (over-scaled)

echo "=== POET init/normalized sweep complete (baseline 3.5160) ==="
