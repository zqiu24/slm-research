#!/usr/bin/env bash
# NODE 2 (8-GPU): §2.15(b) lr sweep  +  §2.14 normalized decorrelation.  ~9 runs, sequential.
#   1) sweep_update_rms_lr.sh                      — 6 runs: {mup,norm} × lr{4e-3,5e-3,6e-3}
#   2) sweep_update_rms_decorrelate_normalized.sh  — 3 runs: normalized λ{0.25,0.50,0.75}, baseline 3.4765
# Every run is 8-GPU, so the two sub-sweeps (and their internal grids) run SEQUENTIALLY.
# No errexit: sub-sweep 2 runs even if a sub-sweep-1 arm fails. Run:  bash scripts/sweep_node2.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

echo "===== NODE 2 START: lr sweep, then normalized decorrelation ====="
bash scripts/sweep_update_rms_lr.sh
echo "----- NODE 2: lr sweep returned (status $?); starting normalized decorrelation -----"
bash scripts/sweep_update_rms_decorrelate_normalized.sh
echo "===== NODE 2 DONE (status $?) ====="
