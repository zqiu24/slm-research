#!/usr/bin/env bash
# POET learnable per-layer scale (g) — 16-arm A/B, one experiment per node.
#
# Every arm sets optim.poet.learnable_scale=true (trainable gain g, init 1.0) on
# the update-RMS champion config, and varies three axes:
#   init_type   : mup_normalized α4  |  normalized
#   init_scale  : 1.0 (champion / neutral)  |  4.0 (4× over-normed norm-recovery test)
#   gain_lr_mult: gain LR = optim.lr × mult; range CENTERED on the expected target g
#                   init_scale=1.0 → g≈1   → mult {0.5, 1.0, 2.0, 4.0}
#                   init_scale=4.0 → g≈0.25 → mult {0.1, 0.25, 0.5, 1.0}
# 60m/40tpp, seed 42, 8-GPU per arm. g=1 ≡ no-gain baseline (bit-exact at init), so
# any delta is purely "operating norm became learnable". Compare each arm's val/loss
# to the no-gain side_γ+0.25 champion 3.4745 and the §2.15c record 3.4686. Watch the
# W&B `gain` histograms leave 1.0 — the s4 arms should drive g → ~0.25.
#
# Usage (one arm per node):
#   scripts/sweep_poet_learnable_scale.sh list          # print the 16 arm names
#   scripts/sweep_poet_learnable_scale.sh ls_mup_s1_m2  # run ONE arm (8-GPU)
#   scripts/sweep_poet_learnable_scale.sh all           # run all 16 sequentially
# Wave plan for 4 nodes (4 arms each): ls_mup_s1_* , ls_mup_s4_* , ls_norm_s1_* , ls_norm_s4_*
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

# Emit "<arm_name>\t<per-arm overrides>" for all 16 arms, in wave order.
build_arms() {
  local init init_ov scale mults mult sdig mdig name
  for init in mup norm; do
    if [[ "${init}" == "mup" ]]; then
      init_ov="optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4"
    else
      init_ov="optim.poet.init_type=normalized"
    fi
    for scale in 1.0 4.0; do
      if [[ "${scale}" == "1.0" ]]; then mults="0.5 1.0 2.0 4.0"; else mults="0.1 0.25 0.5 1.0"; fi
      sdig="${scale%.*}"  # 1.0 -> 1 ; 4.0 -> 4
      for mult in ${mults}; do
        mdig="m${mult//./p}"  # 0.5 -> m0p5 ; 1.0 -> m1p0
        name="ls_${init}_s${sdig}_${mdig}"
        printf '%s\t%s optim.poet.init_scale=%s optim.poet.gain_lr_mult=%s\n' \
          "${name}" "${init_ov}" "${scale}" "${mult}"
      done
    done
  done
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
    echo "=== learnable-scale 16-arm A/B done: compare vs no-gain champion 3.4745 / record 3.4686 ==="
    ;;
  *)
    ov="$(build_arms | awk -F'\t' -v n="${cmd}" '$1==n {print $2}')"
    if [[ -z "${ov}" ]]; then
      echo "unknown arm: ${cmd}" >&2
      echo "run 'scripts/sweep_poet_learnable_scale.sh list' for the 16 valid arm names." >&2
      exit 2
    fi
    run "${cmd}" ${ov}
    ;;
esac
