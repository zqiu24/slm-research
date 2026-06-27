#!/usr/bin/env bash
# Finer-λ split — NODE 3 of 5: mup@side_γ+0.25 + normalized@side_γ=0 at λ=0.20, renorm=off.
# 2 runs (one per init), sequential. Run on node 3:  bash scripts/sweep_finelam_node3.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
echo "===== FINE-λ NODE 3 (λ=0.20): mup@+0.25 + norm@0, renorm=off — 2 runs ====="
bash scripts/sweep_decorrelate_fine_lambda.sh 0.20
echo "===== FINE-λ NODE 3 DONE (status $?) ====="
