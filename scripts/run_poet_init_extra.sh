#!/usr/bin/env bash
# POET init EXTRA runs — indexable launcher for spare 4-GPU machines (HTCondor array).
# Usage:  scripts/run_poet_init_extra.sh <index 0-9>     (SLM_DRYRUN=1 to print, not run)
# Submitted via scripts/submit_poet_init_extra.sub (queue idx 0-9).
#
# 4-GPU (dp=4, global_batch 1024) — same cohort as the original 4-GPU init grid + the 3.5160
# champion, so directly comparable. All ride the champion recipe; per-index cell knobs below.
# Index plan (targets the two co-leaders none_s400_c6=3.4818 / mup_a400_c6=3.4816):
#   0-3  angle refinement c5/c7 (eff∠ 0.010/0.014) around each leader — pin the angle optimum
#   4-9  3-seed confirmation (seeds 1234/2024/777) of both leaders — kill the single-seed caveat
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
  ext_none_s4_c5 ext_none_s4_c7 ext_mup_a4_c5 ext_mup_a4_c7
  ext_none_s4_c6_s1234 ext_none_s4_c6_s2024 ext_none_s4_c6_s777
  ext_mup_a4_c6_s1234 ext_mup_a4_c6_s2024 ext_mup_a4_c6_s777
)
OVERRIDES=(
  "optim.poet.init_type=none optim.poet.init_scale=4 optim.poet.lie_ortho_c=5"
  "optim.poet.init_type=none optim.poet.init_scale=4 optim.poet.lie_ortho_c=7"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.lie_ortho_c=5"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.lie_ortho_c=7"
  "optim.poet.init_type=none optim.poet.init_scale=4 optim.poet.lie_ortho_c=6 seed=1234"
  "optim.poet.init_type=none optim.poet.init_scale=4 optim.poet.lie_ortho_c=6 seed=2024"
  "optim.poet.init_type=none optim.poet.init_scale=4 optim.poet.lie_ortho_c=6 seed=777"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.lie_ortho_c=6 seed=1234"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.lie_ortho_c=6 seed=2024"
  "optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.lie_ortho_c=6 seed=777"
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
