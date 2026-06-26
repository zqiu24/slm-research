#!/usr/bin/env bash
set -euo pipefail

# One-shot POET Lie-Orth update-RMS probe. Logs update-RMS diagnostics every
# 10 optimizer steps so the early theta / clamp / implied-rho scale is visible.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export SLM_POET_UPDATE_RMS_LOG_INTERVAL="${SLM_POET_UPDATE_RMS_LOG_INTERVAL:-10}"
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

codexlog poet_urms_r020_lr5 scripts/train_poet_lie_orth_update_rms.sh llama3 \
  scheduler=cosine_poet training_regime=ablation_40x \
  optim.lr=0.005 optim.poet.lie_ortho_update_rms=0.2 \
  optim.poet.lie_ortho_max_angle=0.024 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4.0 \
  optim.poet.head_aligned_attn=false optim.poet.lie_alternating=true \
  optim.poet.lie_alternate_every=1 optim.poet.lie_ortho_distributed=true \
  "$@"
