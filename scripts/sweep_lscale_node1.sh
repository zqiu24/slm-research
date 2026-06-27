#!/usr/bin/env bash
# Node 1 of the 16-arm learnable-scale A/B: mup_normalized α4, init_scale=1.0.
# Sweeps gain_lr_mult {0.5,1,2,4} (gain LR 0.0025/0.005/0.010/0.020) sequentially —
# runs ONE arm at a time and waits for each to finish before the next.
# Grid + read-outs: scripts/sweep_poet_learnable_scale.sh. Logs: /lustre/home/zqiu/log/<arm>.log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${HERE}/sweep_poet_learnable_scale.sh"
ARMS=(ls_mup_s1_m0p5 ls_mup_s1_m1p0 ls_mup_s1_m2p0 ls_mup_s1_m4p0)
echo ">>> node1 starting ${#ARMS[@]} arms: ${ARMS[*]}"
for arm in "${ARMS[@]}"; do
  "${SWEEP}" "${arm}"
done
echo "<<< node1 done: ${ARMS[*]}"
