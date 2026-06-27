#!/usr/bin/env bash
# Node 3 of the small-LR learnable-scale probe (§2.18): normalized s2 control + largest gentle gain.
#   lss_norm_s2_ctrl (no gain — anchors §2.12 champion 3.4765; normalized's best norm is s2)
#   lss_norm_s2_m0p1 (gain LR 5e-4)
# Runs sequentially, one at a time. Grid: scripts/sweep_lscale_smalllr.sh. Logs: /lustre/home/zqiu/log/lss_*.log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${HERE}/sweep_lscale_smalllr.sh"
ARMS=(lss_norm_s2_ctrl lss_norm_s2_m0p1)
echo ">>> smalllr node3 starting ${#ARMS[@]} arms: ${ARMS[*]}"
for arm in "${ARMS[@]}"; do
  "${SWEEP}" "${arm}"
done
echo "<<< smalllr node3 done: ${ARMS[*]}"
