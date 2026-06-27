#!/usr/bin/env bash
# Finer-λ split — NODE 4 of 5: mup@side_γ+0.25 + normalized@side_γ=0 at λ=0.25, renorm=off.  (λ0.25 = the record anchor — mup should reproduce 3.4686)
# 2 runs (one per init), sequential. Run on node 4:  bash scripts/sweep_finelam_node4.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
echo "===== FINE-λ NODE 4 (λ=0.25): mup@+0.25 + norm@0, renorm=off — 2 runs ====="
bash scripts/sweep_decorrelate_fine_lambda.sh 0.25
echo "===== FINE-λ NODE 4 DONE (status $?) ====="
