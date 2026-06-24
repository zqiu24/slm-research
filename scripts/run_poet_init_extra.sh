#!/usr/bin/env bash
# POET init EXTRA runs — indexable launcher for spare 4-GPU machines (HTCondor array).
# Usage:  scripts/run_poet_init_extra.sh <index 0-9>     (SLM_DRYRUN=1 to print, not run)
# Submitted via scripts/submit_poet_init_extra.sub (queue idx 0-9).
#
# 4-GPU (dp=4, global_batch 1024) — same cohort as the original 4-GPU init grid + the 3.5160
# champion, so directly comparable. All ride the champion recipe at the OPTIMUM angle c6
# (eff∠ 0.012); per-index cell knobs below.
# Index plan — finer SCALE at c6, ONLY the cells not already done/running/pending in the grids:
#   none @ c6 is covered at {1,1.5,2.75,4,5,5.5,6,7,8}; the descent 1.5->4.0 has only 2.75 -> add 2.0/2.5/3.0/3.5
#   mup  @ c6 is covered at integer 1..6 (hi_mup) -> only the 1->2 gap is open -> add 1.5
#   (dropped as redundant: none 4.5; mup 2.5/3.5/4.5/5.5 — all sit inside already-covered regions)
#   0-3  none (raw)  init_scale {2.0, 2.5, 3.0, 3.5}
#   4    mup         mup_alpha  {1.5}
# Names use the init_*/scale×100 convention so they auto-merge into the analysis tables.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
IDX="${1:?usage: run_poet_init_extra.sh <index 0-9>}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # Condor sets this to the 4 allocated GPUs
export MASTER_PORT="${MASTER_PORT:-$((6000 + IDX))}"            # distinct rendezvous per index
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

HELD="base/scale=60m training_regime=ablation_40x \
optim.lr=0.004 optim.poet.scale=0.5 \
optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
optim.poet.lie_ortho_distributed=true cluster.gpus_per_node=4"

NAMES=(
  init_none_s200_c6 init_none_s250_c6 init_none_s300_c6 init_none_s350_c6
  init_mup_a150_c6
)
OVERRIDES=(
  "optim.poet.init_type=none optim.poet.init_scale=2.0 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=none optim.poet.init_scale=2.5 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=none optim.poet.init_scale=3.0 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=none optim.poet.init_scale=3.5 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=1.5 optim.poet.lie_ortho_c=6"
)

if [ "$IDX" -lt 0 ] || [ "$IDX" -ge "${#NAMES[@]}" ]; then
  echo "bad index '$IDX' (valid 0-$(( ${#NAMES[@]} - 1 )))" >&2; exit 2
fi
N="${NAMES[$IDX]}"; O="${OVERRIDES[$IDX]}"

CMD=(scripts/train_poet_lie_orth.sh $HELD $O "experiment.name=$N")
if [ "${SLM_DRYRUN:-0}" = "1" ]; then
  echo "[$IDX] $N :: ${CMD[*]}"; exit 0
fi
echo ">>> START [$IDX] ${N}  $(date '+%F %T')"
"${CMD[@]}" 2>&1 | tee "${CODEX_LOG_DIR}/${N}.log"
echo "<<< END   [$IDX] ${N}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
