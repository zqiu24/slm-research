#!/usr/bin/env bash
# Decorrelation × side_γ=+0.25 — THE RECORD ATTEMPT. Stacks the §J.3 partial-λ cross-side
# decorrelation on the ASYMMETRIC champion (side_γ=+0.25) instead of the symmetric baseline
# (the §2.14 sweeps use side_γ=0 for clean attribution; this complements them). Sweeps
# λ {0.25,0.50,0.75} × renorm {true,false} × {mup α4, normalized s2}. mode=symmetric,
# cos_threshold=0, ρ0.30, lr5, max∠0.024. 12 runs, sequential. λ=1.0 EXCLUDED (catastrophic,
# §J.3; NB decorrelate_lambda defaults to 1.0 — every arm sets it).
#
# Baselines (side_γ=+0.25, no decorr, §2.12): mup 3.4745 (CHAMPION — the record target),
# normalized 3.4780. NOTE: normalized's OWN optimum is side_γ=0 (3.4765); its +0.25 arm here
# is an asymmetry×decorrelation INTERACTION probe, not normalized's best base.
#
# TRIPWIRE: each startup must print
#   [POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=symmetric, lambda=<L>, renorm=<R>,
#          cos_threshold=0.0, alternating=True)
# If lambda/renorm/mode mismatch the arm, an override was dropped — kill + fix.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
# Log the cos(D_out,D_in) trajectory so the overlap is visibly driven down during the run.
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

COMMON="base/scale=60m training_regime=ablation_40x \
  optim.poet.scale=1.0 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.25 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true \
  optim.poet.lie_ortho_decorrelate=true \
  optim.poet.lie_ortho_decorrelate_mode=symmetric \
  optim.poet.lie_ortho_decorrelate_cos_threshold=0.0 \
  optim.lr=0.005 cluster.gpus_per_node=8"

init_flags() {  # $1 = init key
  case "$1" in
    mup)  echo "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4" ;;
    norm) echo "optim.poet.init_type=normalized optim.poet.init_scale=2.0" ;;
  esac
}

for INIT in mup norm; do
  for LAM in 0.25 0.50 0.75; do
    for RENORM in true false; do
      LTAG="${LAM/./p}"                  # 0.50 -> 0p50
      RTAG="${RENORM:0:1}"               # true->t, false->f
      NAME="urms_decorr_gp25_${INIT}_l${LTAG}_rn${RTAG}"
      echo ">>> ${NAME} (${INIT}, side_γ+0.25, λ=${LAM}, renorm=${RENORM}) starting"
      codexlog "${NAME}" scripts/train_poet_lie_orth_update_rms.sh llama3 ${COMMON} \
        $(init_flags "$INIT") \
        optim.poet.lie_ortho_decorrelate_lambda="${LAM}" \
        optim.poet.lie_ortho_decorrelate_renorm="${RENORM}" \
        experiment.name="${NAME}"
      echo "<<< ${NAME} done (status $?)"
    done
  done
done
echo "=== update-RMS decorrelation × side_γ+0.25 complete: {mup,norm} × λ{0.25,0.50,0.75} × renorm{t,f}; targets mup 3.4745 / norm 3.4780 ==="
