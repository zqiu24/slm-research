#!/usr/bin/env bash
# Node 4 of the 16-arm learnable-scale A/B: normalized init, init_scale=4.0 (norm-recovery).
# Sweeps gain_lr_mult {0.1,0.25,0.5,1.0} (gain LR 0.0005/0.00125/0.0025/0.005) sequentially —
# runs ONE arm at a time and waits for each to finish before the next. Expect g → ~0.25.
# Grid + read-outs: scripts/sweep_poet_learnable_scale.sh. Logs: /lustre/home/zqiu/log/<arm>.log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${HERE}/sweep_poet_learnable_scale.sh"
ARMS=(ls_norm_s4_m0p1 ls_norm_s4_m0p25 ls_norm_s4_m0p5 ls_norm_s4_m1p0)
echo ">>> node4 starting ${#ARMS[@]} arms: ${ARMS[*]}"
for arm in "${ARMS[@]}"; do
  "${SWEEP}" "${arm}"
done
echo "<<< node4 done: ${ARMS[*]}"
