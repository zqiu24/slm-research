#!/usr/bin/env bash
# Arm K (round 2, FIXED) — per-block angle exponent, POSITIVE side.  Run on 8-GPU NODE 2.
#
#   θ_block = lr·scale·ortho_c · (block_size / hidden)^p
#
# p>0 → larger-dim blocks (fc 1536) rotate MORE than ‖W‖-proportional, small kv (64)
# rotate LESS. p=0 = champion (3.518, already have it — skip). Run the NEGATIVE half
# (sweep_angle_dim_exp_neg.sh) on NODE 1 concurrently.
#
# Same fix/guard as the _neg script (f5f05cc): b_ref(hidden) now propagates, so these arms
# genuinely scale (and crash loudly if b_ref is ever missing). Uses all 8 GPUs of the node;
# NO CUDA_VISIBLE_DEVICES pinning. Run:
#     bash scripts/sweep_angle_dim_exp_pos.sh        # on a SECOND 8-GPU node
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export MASTER_PORT=6010
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {  # inline (interactive shell function does not expand in a script)
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

# Champion recipe; only angle_dim_exp varies. gpus=8 (default).
HELD="base/scale=60m training_regime=ablation_40x optim.lr=4e-3 optim.poet.scale=0.5 \
optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1"

run () {  # <name> <exp>
  codexlog "$1" scripts/train_poet_lie_orth.sh $HELD \
    optim.poet.lie_ortho_angle_dim_exp="$2" experiment.name="$1"
}

run angle2_ep025 0.25
run angle2_ep05  0.5
run angle2_ep10  1.0
run angle2_ep15  1.5

echo "=== angle_dim_exp POSITIVE sweep complete ==="
