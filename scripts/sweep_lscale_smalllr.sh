#!/usr/bin/env bash
# POET learnable-scale SMALL-LR probe (§2.18) — follow-up to the NEGATIVE §2.17 sweep.
#
# §2.17 found the learnable gain g regresses at every gain LR tried, monotone toward the
# LR->0 (no-gain) limit, diverging by gain LR 2.5e-3. This probe drops a decade below that
# floor AND adds the same-code no-gain control §2.17 lacked. 8 arms, one per invocation:
#   - 2 controls (learnable_scale=false) — pin each init's §2.12 champion exactly:
#       lss_mup_s1_ctrl  (mup α4, init_scale=1, side_γ0.25) -> anchor 3.4745
#       lss_norm_s2_ctrl (normalized, init_scale=2, side_γ0) -> anchor 3.4765
#   - 6 gentle-gain probes: each init × gain_lr_mult {0.1,0.03,0.01} (gain LR 5e-4/1.5e-4/5e-5)
# Each init uses its OWN champion operating point — mup: init_scale=1 / side_γ+0.25;
# normalized: init_scale=2 / side_γ0 (normalized's best norm is s2, NOT s1 — §2.12/§2.10).
# g still inits at 1.0 (bit-exact baseline at step 0).
# 60m/40tpp, seed 42, 8-GPU per arm. Logs: /lustre/home/zqiu/log/lss_<arm>.log.
#
# Usage (one arm per node):
#   scripts/sweep_lscale_smalllr.sh list
#   scripts/sweep_lscale_smalllr.sh lss_mup_s1_m0p03
#   scripts/sweep_lscale_smalllr.sh all
# NOTE: learnable_scale / side_gamma are set PER ARM (not in COMMON) to avoid duplicate-key
# overrides — the controls simply omit learnable_scale (YAML default false).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

COMMON="llama3 scheduler=cosine_poet training_regime=ablation_40x \
  optim.lr=0.005 \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.head_aligned_attn=false optim.poet.lie_alternating=true \
  optim.poet.lie_alternate_every=1 optim.poet.lie_ortho_distributed=true"

MUP="optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.init_scale=1.0 optim.poet.lie_ortho_update_rms_side_gamma=0.25"
NORM="optim.poet.init_type=normalized optim.poet.init_scale=2.0 optim.poet.lie_ortho_update_rms_side_gamma=0.0"

build_arms() {  # "<name>\t<per-arm overrides>"
  printf '%s\t%s\n' lss_mup_s1_ctrl   "${MUP}"
  printf '%s\t%s\n' lss_mup_s1_m0p1   "${MUP} optim.poet.learnable_scale=true optim.poet.gain_lr_mult=0.1"
  printf '%s\t%s\n' lss_mup_s1_m0p03  "${MUP} optim.poet.learnable_scale=true optim.poet.gain_lr_mult=0.03"
  printf '%s\t%s\n' lss_mup_s1_m0p01  "${MUP} optim.poet.learnable_scale=true optim.poet.gain_lr_mult=0.01"
  printf '%s\t%s\n' lss_norm_s2_ctrl  "${NORM}"
  printf '%s\t%s\n' lss_norm_s2_m0p1  "${NORM} optim.poet.learnable_scale=true optim.poet.gain_lr_mult=0.1"
  printf '%s\t%s\n' lss_norm_s2_m0p03 "${NORM} optim.poet.learnable_scale=true optim.poet.gain_lr_mult=0.03"
  printf '%s\t%s\n' lss_norm_s2_m0p01 "${NORM} optim.poet.learnable_scale=true optim.poet.gain_lr_mult=0.01"
}

run() {  # $1 = name ; $2.. = extra overrides
  local name="$1"; shift
  echo ">>> ${name} starting"
  scripts/train_poet_lie_orth_update_rms.sh ${COMMON} "$@" \
    experiment.name="${name}" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< ${name} done (status ${PIPESTATUS[0]}) — ${CODEX_LOG_DIR}/${name}.log"
}

cmd="${1:-list}"
case "${cmd}" in
  list)
    build_arms | cut -f1
    ;;
  all)
    while IFS=$'\t' read -r name ov; do
      run "${name}" ${ov}
    done < <(build_arms)
    echo "=== small-LR probe done: compare each probe to its ctrl (mup 3.4745 / norm 3.4765) ==="
    ;;
  *)
    ov="$(build_arms | awk -F'\t' -v n="${cmd}" '$1==n {print $2}')"
    if [[ -z "${ov}" ]]; then
      echo "unknown arm: ${cmd}" >&2
      echo "run 'scripts/sweep_lscale_smalllr.sh list' for the 8 valid arm names." >&2
      exit 2
    fi
    run "${cmd}" ${ov}
    ;;
esac
