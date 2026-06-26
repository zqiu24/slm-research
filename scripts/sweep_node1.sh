#!/usr/bin/env bash
# NODE 1 (8-GPU): §2.15(a) max_angle sweep  +  §2.14 mup decorrelation.  ~11 runs, sequential.
#   1) sweep_update_rms_maxangle.sh         — 8 runs: {mup,norm} × max∠{0.012,0.016,0.024,0.032}
#   2) sweep_update_rms_decorrelate_mup.sh  — 3 runs: mup λ{0.25,0.50,0.75}, baseline 3.4758
# Every run is 8-GPU, so the two sub-sweeps (and their internal grids) run SEQUENTIALLY.
# No errexit: sub-sweep 2 runs even if a sub-sweep-1 arm fails. Run:  bash scripts/sweep_node1.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

echo "===== NODE 1 START: max_angle sweep, then mup decorrelation ====="
bash scripts/sweep_update_rms_maxangle.sh
echo "----- NODE 1: max_angle sweep returned (status $?); starting mup decorrelation -----"
bash scripts/sweep_update_rms_decorrelate_mup.sh
echo "===== NODE 1 DONE (status $?) ====="
