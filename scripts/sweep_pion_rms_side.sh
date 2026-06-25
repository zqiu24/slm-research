#!/usr/bin/env bash
# Pion pion_rms × pion_update_side sweep — a 3×4 = 12-run factorial at the fixed
# LR-sweep optimum (optim.lr=1e-3); everything else at the optim/pion defaults.
# Run on one node (sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_pion_rms_side.sh
#
# Two axes (the highest-leverage untuned knobs after LR — see POET_dev.md §2.6):
#   pion_rms          0.1  0.2  0.4      (update-magnitude scale: half / default / double)
#   pion_update_side  alternate both in out   (which Lie generator(s) drive the rotation)
#
# Fixed at the optim/pion reference defaults otherwise: pion_scaling=rms,
# pion_momentum=transported_ambient_ambient, pion_degree=2, betas (0.9, 0.95),
# weight_decay 0.1, stock cosine schedule (min_lr 0.1), lr 1e-3.
#
# Self-anchoring: the (rms=0.2, side=alternate) cell reproduces the current Pion
# baseline (val/loss 3.7688 @ iter 9155, run pion-...20260625T152410Z / W&B
# gkh8zu5k) — use it to confirm the grid is on-cohort before reading the others.
#
# Launcher = scripts/train_pion_dev.sh, which reproduces the dev cohort exactly:
# experiment=optim/pion, llama3-60m, ablation_40x (40 tpp), seq 256, gbs 1024,
# mbs 128, transformer_impl=local, tie_embeddings=false. Pion uses ONE base
# optim.lr for BOTH the Pion side (scaled internally by pion_rms*sqrt(m*n)) and
# the chained-AdamW side. Each run uses experiment.name=pion (distinct run dirs
# by timestamp); the codexlog name encodes both axes so the 12 logs are distinct.
#
# NOTE: 12 runs × ~38 min ≈ 7.6 h sequential on one node.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

RMS_VALUES=(0.1 0.2 0.4)
SIDES=(alternate both in out)

for rms in "${RMS_VALUES[@]}"; do
  for side in "${SIDES[@]}"; do
    name="pion_rms${rms}_side${side}"
    echo "### ${name}: pion_rms=${rms} pion_update_side=${side} (lr=1e-3, all else = optim/pion defaults)"
    codexlog "$name" scripts/train_pion_dev.sh \
      optim.lr=0.001 \
      optim.pion_rms="$rms" \
      optim.pion_update_side="$side" \
      experiment.name="pion"
  done
done

echo "=== pion rms×side sweep complete ($((${#RMS_VALUES[@]} * ${#SIDES[@]})) runs) ==="
