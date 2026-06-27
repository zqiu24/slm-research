#!/usr/bin/env bash
# Node 2 of the small-LR learnable-scale probe (§2.18): mup s1, two smallest gentle gains.
#   lss_mup_s1_m0p03 (gain LR 1.5e-4)
#   lss_mup_s1_m0p01 (gain LR 5e-5)
# Runs sequentially, one at a time. Grid: scripts/sweep_lscale_smalllr.sh. Logs: /lustre/home/zqiu/log/lss_*.log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${HERE}/sweep_lscale_smalllr.sh"
ARMS=(lss_mup_s1_m0p03 lss_mup_s1_m0p01)
echo ">>> smalllr node2 starting ${#ARMS[@]} arms: ${ARMS[*]}"
for arm in "${ARMS[@]}"; do
  "${SWEEP}" "${arm}"
done
echo "<<< smalllr node2 done: ${ARMS[*]}"
