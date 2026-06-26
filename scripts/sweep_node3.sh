#!/usr/bin/env bash
# NODE 3 (8-GPU): §2.15(c) decorrelation × side_γ+0.25 (RECORD ATTEMPT)  +  §2.14 none decorrelation.
#   ~15 runs, sequential (the heaviest node).
#   1) sweep_update_rms_decorrelate_gp25.sh  — 12 runs: {mup,norm} × λ{0.25,0.50,0.75} × renorm{t,f};
#                                              targets mup 3.4745 (champion) / norm 3.4780
#   2) sweep_update_rms_decorrelate_none.sh   — 3 runs: none λ{0.25,0.50,0.75}, baseline 3.4782
# Every run is 8-GPU, so the two sub-sweeps (and their internal grids) run SEQUENTIALLY.
# No errexit: sub-sweep 2 runs even if a sub-sweep-1 arm fails. Run:  bash scripts/sweep_node3.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

echo "===== NODE 3 START: decorrelation×side_γ+0.25, then none decorrelation ====="
bash scripts/sweep_update_rms_decorrelate_gp25.sh
echo "----- NODE 3: gp25 sweep returned (status $?); starting none decorrelation -----"
bash scripts/sweep_update_rms_decorrelate_none.sh
echo "===== NODE 3 DONE (status $?) ====="
