#!/usr/bin/env bash
# Small-LR learnable-scale probe (§2.18) — normalized s2, ONE arm per node.
#   norm4: lss_norm_s2_m0p01  (gain LR 5e-5)
# Grid + read-outs: scripts/sweep_lscale_smalllr.sh. Log: /lustre/home/zqiu/log/lss_norm_s2_m0p01.log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/sweep_lscale_smalllr.sh" lss_norm_s2_m0p01
