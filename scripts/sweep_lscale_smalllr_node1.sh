#!/usr/bin/env bash
# Node 1 of the small-LR learnable-scale probe (§2.18): mup s1 control + largest gentle gain.
#   lss_mup_s1_ctrl  (no gain — anchors §2.12 champion 3.4745)
#   lss_mup_s1_m0p1  (gain LR 5e-4)
# Runs sequentially, one at a time. Grid: scripts/sweep_lscale_smalllr.sh. Logs: /lustre/home/zqiu/log/lss_*.log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${HERE}/sweep_lscale_smalllr.sh"
ARMS=(lss_mup_s1_ctrl lss_mup_s1_m0p1)
echo ">>> smalllr node1 starting ${#ARMS[@]} arms: ${ARMS[*]}"
for arm in "${ARMS[@]}"; do
  "${SWEEP}" "${arm}"
done
echo "<<< smalllr node1 done: ${ARMS[*]}"
