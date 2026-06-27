#!/usr/bin/env bash
# SEED-CONFIRM the new best-POET record. The record `urms_decorr_gp25_mup_l0p25_rnf` = 3.4686
# (§2.15c) beats its no-decorrelation base (side_γ+0.25 update-RMS champion = 3.4745, §2.12)
# by only −0.0059 — BELOW the ~0.01–0.02 60m seed-noise floor. So re-run BOTH at seeds 43 & 44
# (seed 42 already in hand) and compare the 3-seed clouds: if the record cloud sits below the
# base cloud, the decorrelation gain is real, not seed luck. 4 runs, sequential.
#   base   = side_γ+0.25 champion, NO decorrelation   (s42 = 3.4745)
#   record = + decorrelation λ0.25 / renorm=false      (s42 = 3.4686)
#
# TRIPWIRE (record runs only): startup prints
#   [POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=symmetric, lambda=0.25, renorm=False, ...)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

# The side_γ+0.25 update-RMS champion recipe (= the `base`); `record` adds decorrelation.
COMMON="base/scale=60m training_regime=ablation_40x \
  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.scale=1.0 \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.25 optim.poet.lie_ortho_rms_mode=weight \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true \
  optim.lr=0.005 cluster.gpus_per_node=8"

cfg_flags() {  # $1 = base|record
  case "$1" in
    base)   echo "" ;;  # no decorrelation (default poet_lie_ortho_decorrelate=false)
    record) echo "optim.poet.lie_ortho_decorrelate=true \
                  optim.poet.lie_ortho_decorrelate_mode=symmetric \
                  optim.poet.lie_ortho_decorrelate_lambda=0.25 \
                  optim.poet.lie_ortho_decorrelate_renorm=false \
                  optim.poet.lie_ortho_decorrelate_cos_threshold=0.0" ;;
  esac
}

for SEED in 43 44; do
  for CFG in base record; do
    NAME="seedconf_${CFG}_s${SEED}"
    echo ">>> ${NAME} (${CFG}, seed ${SEED}) starting"
    scripts/train_poet_lie_orth_update_rms.sh llama3 ${COMMON} \
      seed="${SEED}" \
      $(cfg_flags "$CFG") \
      experiment.name="${NAME}" \
      2>&1 | tee "${CODEX_LOG_DIR}/${NAME}.log"
    echo "<<< ${NAME} done (status ${PIPESTATUS[0]}) — log: ${CODEX_LOG_DIR}/${NAME}.log"
  done
done
echo "=== seed-confirm complete: base(3.4745@s42) vs record(3.4686@s42) at seeds 43,44 — compare 3-seed clouds ==="
