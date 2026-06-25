#!/usr/bin/env bash
# POET init EXTRA runs — indexable launcher for spare 4-GPU machines (HTCondor array).
# Usage:  scripts/run_poet_init_extra.sh <index 0-9>     (SLM_DRYRUN=1 to print, not run)
# Submitted via scripts/submit_poet_init_extra.sub (queue idx 0-9).
#
# 4-GPU (dp=4, global_batch 1024) — same cohort as the original 4-GPU init grid + the 3.5160
# champion, so directly comparable. All ride the champion recipe at the OPTIMUM angle c6
# (eff∠ 0.012); per-index cell knobs below.
# Index plan — non-redundant cells only (skip anything done/running/pending in the grids):
#   0-3  none (raw) @ c6  init_scale {2.0,2.5,3.0,3.5}  — fills the sparse 1.5->4.0 descent (only 2.75 there)
#   4-9  mup HIGHER scale × angle: mup_alpha {7,8} × c{4,6,8}  — hi_mup already sweeps a{2..6}×c{2,4,6},
#        so go ABOVE 6 (mup @ c6 still falling at 4.0=3.4816); angles bracket the c6 optimum (c4/c6/c8).
#        All standard angle columns so they auto-merge into the tables.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
IDX="${1:?usage: run_poet_init_extra.sh <index 0-9>}"

# Env bootstrap: condor lands each array job on a different machine and
# getenv=True only inherits the SUBMIT-host env, so set up the venv + CUDA
# here. Without the venv, `python` falls back to system 3.10 and the launcher
# dies on `from datetime import UTC` (py3.11+) before training starts.
SLM_VENV="${SLM_VENV:-/lustre/fast/fast/zqiu/slm_env/.venv}"
if [ "${VIRTUAL_ENV:-}" != "$SLM_VENV" ] && [ -f "$SLM_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$SLM_VENV/bin/activate"
fi
source /lustre/fast/fast/zqiu/slm-research/load_cuda13_2_nccl_env.sh

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
  init_mup_a700_c4 init_mup_a700_c6 init_mup_a700_c8
  init_mup_a800_c4 init_mup_a800_c6 init_mup_a800_c8
)
OVERRIDES=(
  "optim.poet.init_type=none optim.poet.init_scale=2.0 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=none optim.poet.init_scale=2.5 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=none optim.poet.init_scale=3.0 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=none optim.poet.init_scale=3.5 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=7.0 optim.poet.lie_ortho_c=4"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=7.0 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=7.0 optim.poet.lie_ortho_c=8"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=8.0 optim.poet.lie_ortho_c=4"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=8.0 optim.poet.lie_ortho_c=6"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=8.0 optim.poet.lie_ortho_c=8"
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
