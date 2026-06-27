#!/usr/bin/env bash
# Small-LR learnable-scale probe (§2.18) — normalized s2, ONE arm per node.
#   norm1: lss_norm_s2_ctrl  (no gain — anchors §2.12 normalized champion 3.4765)
# Grid + read-outs: scripts/sweep_lscale_smalllr.sh. Log: /lustre/home/zqiu/log/lss_norm_s2_ctrl.log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/sweep_lscale_smalllr.sh" lss_norm_s2_ctrl
