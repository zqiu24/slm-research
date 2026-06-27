#!/usr/bin/env bash
# A/B for the POET learnable per-layer scale (g). Each run adds the trainable gain
# g (init 1.0) on top of an otherwise-fixed init; g=1 ≡ the no-gain baseline, so any
# delta is purely "operating norm became learnable".
#   arm 1 (champion-init): mup α4 + g   — primary A/B vs the 3.4686 champion
#   arm 2 (neutral-init):  normalized/scale1 + g — "can g replace init tuning?"
# 60m/40tpp, seed 42, 8-GPU. Baseline (no g) = urms side_γ+0.25 champion 3.4745;
# the §2.15c decorrelation record 3.4686 is the no-gain SOTA to beat.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

COMMON="llama3 scheduler=cosine_poet training_regime=ablation_40x \
  optim.lr=0.005 optim.poet.learnable_scale=true \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.25 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.head_aligned_attn=false optim.poet.lie_alternating=true \
  optim.poet.lie_alternate_every=1 optim.poet.lie_ortho_distributed=true"

run() {  # $1 = name ; $2.. = extra overrides
  local name="$1"; shift
  echo ">>> ${name} starting"
  scripts/train_poet_lie_orth_update_rms.sh ${COMMON} "$@" \
    experiment.name="${name}" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< ${name} done (status ${PIPESTATUS[0]}) — ${CODEX_LOG_DIR}/${name}.log"
}

run lscale_mup    optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.init_scale=1.0
run lscale_norm   optim.poet.init_type=normalized     optim.poet.init_scale=1.0
echo "=== learnable-scale A/B done: compare lscale_mup vs no-gain champion 3.4745/record 3.4686 ==="
