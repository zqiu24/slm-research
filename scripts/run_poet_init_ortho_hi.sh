#!/usr/bin/env bash
# POET init ORTHOGONAL-HI runs — indexable launcher for spare 4-GPU machines (HTCondor array).
# Usage:  scripts/run_poet_init_ortho_hi.sh <index 0-7>     (SLM_DRYRUN=1 to print, not run)
# Submitted via scripts/submit_poet_init_ortho_hi.sub (queue idx 0-7).
#
# 4-GPU (dp=4, global_batch 1024) — same cohort as the original 4-GPU init grid + the 3.5160
# champion, so directly comparable. init_type=orthogonal (kappa=1). BIGGER scale x COOLER angle:
# the 4-GPU grid only reached scale 2.0 / c{6,8,10} and ortho was still falling (s2.0/c6=3.5240),
# so push scale {3,4,5,6} x c{4,6} (eff∠ 0.008/0.012; c4 never tried for ortho).
#   idx 0-1  scale 3.0  x c{4,6}
#   idx 2-3  scale 4.0  x c{4,6}
#   idx 4-5  scale 5.0  x c{4,6}
#   idx 6-7  scale 6.0  x c{4,6}
# Standard angle columns + sNNN scale naming so they auto-merge into the §2.9 ortho grid.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
IDX="${1:?usage: run_poet_init_ortho_hi.sh <index 0-7>}"

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
optim.poet.lie_ortho_distributed=true optim.poet.init_type=orthogonal \
cluster.gpus_per_node=4"

NAMES=(
  init_ortho_s300_c4 init_ortho_s300_c6
  init_ortho_s400_c4 init_ortho_s400_c6
  init_ortho_s500_c4 init_ortho_s500_c6
  init_ortho_s600_c4 init_ortho_s600_c6
)
OVERRIDES=(
  "optim.poet.init_scale=3.0 optim.poet.lie_ortho_c=4"
  "optim.poet.init_scale=3.0 optim.poet.lie_ortho_c=6"
  "optim.poet.init_scale=4.0 optim.poet.lie_ortho_c=4"
  "optim.poet.init_scale=4.0 optim.poet.lie_ortho_c=6"
  "optim.poet.init_scale=5.0 optim.poet.lie_ortho_c=4"
  "optim.poet.init_scale=5.0 optim.poet.lie_ortho_c=6"
  "optim.poet.init_scale=6.0 optim.poet.lie_ortho_c=4"
  "optim.poet.init_scale=6.0 optim.poet.lie_ortho_c=6"
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
