#!/usr/bin/env bash
# update-RMS dense-LR sweep UNDER THE ADAPTIVE ANGLE — revisits the §2.10 lr lever, which was
# only mapped for the fixed-angle path (POET_dev.md:859). The §2.11 ρ-sweep held lr=5e-3, so
# the lr×ρ interaction under the self-scaling angle θ=min(lr·ρ/RMS(W), max∠) is open. Holds
# ρ0.30 / max∠0.024 / side_γ=0; sweeps lr {4e-3, 5e-3, 6e-3} (5e-3 = baseline anchor) ×
# {mup α4, normalized s2}. 6 runs, sequential.
# Baselines (lr5): mup 3.4758, normalized 3.4765 (§2.11) — the lr5 cells should reproduce them.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

COMMON="base/scale=60m training_regime=ablation_40x \
  optim.poet.scale=1.0 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_update_rms_side_gamma=0.0 \
  optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true \
  cluster.gpus_per_node=8"

init_flags() {  # $1 = init key
  case "$1" in
    mup)  echo "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4" ;;
    norm) echo "optim.poet.init_type=normalized optim.poet.init_scale=2.0" ;;
  esac
}

for INIT in mup norm; do
  for LR in 0.004 0.005 0.006; do
    TAG="${LR/./p}"                      # 0.004 -> 0p004
    NAME="urms_lr_${INIT}_${TAG}"
    echo ">>> ${NAME} (${INIT}, ρ0.30, lr=${LR}) starting"
    codexlog "${NAME}" scripts/train_poet_lie_orth_update_rms.sh llama3 ${COMMON} \
      $(init_flags "$INIT") \
      optim.lr="${LR}" \
      experiment.name="${NAME}"
    echo "<<< ${NAME} done (status $?)"
  done
done
echo "=== update-RMS lr sweep complete: {mup,norm} × {4e-3,5e-3,6e-3}; anchor lr5 = mup 3.4758 / norm 3.4765 ==="
