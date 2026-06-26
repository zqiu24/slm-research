#!/usr/bin/env bash
# update-RMS angle-CEILING sweep — the §2.11 "Next" lever (POET_dev.md:859). The clamp in
# θ = min(lr·ρ/RMS(W), max_angle) currently only bites high-ρ / `none` (§2.11(2)); lowering
# max_angle makes the ceiling actively shape the early peak-LR rotation for the OPTIMUM
# configs. Holds ρ0.30 / lr5 / side_γ=0; sweeps max_angle {0.012, 0.016, 0.024, 0.032}
# (0.024 = baseline anchor) × {mup α4, normalized s2}. 8 runs, sequential.
# Baselines (max∠0.024): mup 3.4758, normalized 3.4765 (§2.11/§2.12).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

COMMON="base/scale=60m training_regime=ablation_40x \
  optim.poet.scale=1.0 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_update_rms_side_gamma=0.0 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true \
  optim.lr=0.005 cluster.gpus_per_node=8"

init_flags() {  # $1 = init key
  case "$1" in
    mup)  echo "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4" ;;
    norm) echo "optim.poet.init_type=normalized optim.poet.init_scale=2.0" ;;
  esac
}

for INIT in mup norm; do
  for MA in 0.012 0.016 0.024 0.032; do
    TAG="${MA/./p}"                      # 0.016 -> 0p016
    NAME="urms_maxangle_${INIT}_a${TAG}"
    echo ">>> ${NAME} (${INIT}, ρ0.30, max_angle=${MA}) starting"
    codexlog "${NAME}" scripts/train_poet_lie_orth_update_rms.sh llama3 ${COMMON} \
      $(init_flags "$INIT") \
      optim.poet.lie_ortho_max_angle="${MA}" \
      experiment.name="${NAME}"
    echo "<<< ${NAME} done (status $?)"
  done
done
echo "=== update-RMS max_angle sweep complete: {mup,norm} × {0.012,0.016,0.024,0.032}; anchor max∠0.024 = mup 3.4758 / norm 3.4765 ==="
