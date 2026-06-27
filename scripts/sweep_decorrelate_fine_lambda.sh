#!/usr/bin/env bash
# Finer-λ RECORD REFINEMENT (§2.15c follow-up). λ0.25 was the smallest non-trivial value
# tested and beat BOTH λ0 and λ0.5, so the true decorrelation optimum may sit below 0.25.
# Scan λ {0.10, 0.15, 0.20, 0.25, 0.30} with renorm=FALSE (§2.15c: renorm=off wins
# everywhere). Each init at its OWN best side_γ:
#   mup        → side_γ=+0.25  (record holder: λ0.25 = 3.4686; base λ0 = 3.4745)
#   normalized → side_γ=0      (its optimum + strongest symmetric decorr responder:
#                               λ0.25 renorm=TRUE = 3.4705; renorm=off untested; base 3.4765)
# mode=symmetric, cos_threshold=0, ρ0.30, lr5, max∠0.024. 10 runs, sequential. The mup λ0.25
# cell anchors to 3.4686 (wiring check). λ=1.0 excluded (catastrophic, §J.3).
#
# TRIPWIRE: each startup must print
#   [POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=symmetric, lambda=<L>, renorm=False, ...)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

COMMON="base/scale=60m training_regime=ablation_40x \
  optim.poet.scale=1.0 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true \
  optim.poet.lie_ortho_decorrelate=true \
  optim.poet.lie_ortho_decorrelate_mode=symmetric \
  optim.poet.lie_ortho_decorrelate_renorm=false \
  optim.poet.lie_ortho_decorrelate_cos_threshold=0.0 \
  optim.lr=0.005 cluster.gpus_per_node=8"

init_flags() {  # $1 = init key — emits init shape + that init's best side_γ
  case "$1" in
    mup)  echo "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.lie_ortho_update_rms_side_gamma=0.25" ;;
    norm) echo "optim.poet.init_type=normalized optim.poet.init_scale=2.0 optim.poet.lie_ortho_update_rms_side_gamma=0.0" ;;
  esac
}

# λ values from args (for the per-node split, e.g. `… 0.10`); default = the full 5-point grid.
LAMS=("$@"); [ "${#LAMS[@]}" -eq 0 ] && LAMS=(0.10 0.15 0.20 0.25 0.30)

for INIT in mup norm; do
  for LAM in "${LAMS[@]}"; do
    TAG="${LAM/./p}"                     # 0.15 -> 0p15
    NAME="urms_decorrfine_${INIT}_l${TAG}_rnf"
    echo ">>> ${NAME} (${INIT}, $( [ "$INIT" = mup ] && echo 'side_γ+0.25' || echo 'side_γ0' ), λ=${LAM}, renorm=off) starting"
    scripts/train_poet_lie_orth_update_rms.sh llama3 ${COMMON} \
      $(init_flags "$INIT") \
      optim.poet.lie_ortho_decorrelate_lambda="${LAM}" \
      experiment.name="${NAME}" \
      2>&1 | tee "${CODEX_LOG_DIR}/${NAME}.log"
    echo "<<< ${NAME} done (status ${PIPESTATUS[0]}) — log: ${CODEX_LOG_DIR}/${NAME}.log"
  done
done
echo "=== finer-λ decorrelation complete: {mup@+0.25, norm@0} × λ{0.10,0.15,0.20,0.25,0.30} renorm=off; mup λ0.25 anchors 3.4686 ==="
