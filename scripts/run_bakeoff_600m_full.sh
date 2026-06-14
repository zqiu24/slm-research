#!/usr/bin/env bash
set -uo pipefail

# Run the full architecture-family bake-off at the 600M budget — all four
# families, 24B tokens each, ONE AFTER ANOTHER on a single node.
# (docs/experiments/arch_bakeoff_600m.md, Task 11 Step 3.)
#
# train_megatron launches torchrun in the FOREGROUND (blocking), so each
# family fully finishes before the next starts — this script just chains them.
# Per-family stdout+stderr is tee'd to $CODEX_LOG_DIR/<name>.log (codexlog
# style); the real training exit code is captured via PIPESTATUS, not tee's.
#
# Usage:
#   bash scripts/run_bakeoff_600m_full.sh [extra hydra overrides...]
#
# The extra overrides are forwarded IDENTICALLY to every family (fairness:
# if a smoke needed a fallback, e.g. base.model.moe.grouped_gemm=false, pass
# it here so all four get it). Examples:
#   bash scripts/run_bakeoff_600m_full.sh
#   bash scripts/run_bakeoff_600m_full.sh base.model.attention_backend=auto
#
# Knobs (env vars):
#   CLUSTER        target cluster config        (default: h100_de)
#   FAMILIES       space-separated subset/order  (default: qwen3 deepseek_v3 qwen3_next nemotron_h)
#   WANDB_PROJECT  shared W&B project for all four (default: slm-<user>-arch)
#   STOP_ON_FAIL   abort the chain on first failed run (default: 0 = continue)

SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SLM_REPO"

CLUSTER="${CLUSTER:-h100_de}"
FAMILIES="${FAMILIES:-qwen3 deepseek_v3 qwen3_next nemotron_h}"
WANDB_PROJECT="${WANDB_PROJECT:-slm-${USER:-unknown}-arch}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$LOG_DIR"

# family -> codexlog log name (matches the plan's Task 11 Step 3 names)
declare -A LOG_NAME=(
  [qwen3]=bakeoff-600m-qwen3
  [deepseek_v3]=bakeoff-600m-deepseek
  [deepseek_v3_dense]=bakeoff-600m-deepseek-dense
  [qwen3_next]=bakeoff-600m-qwen3next
  [nemotron_h]=bakeoff-600m-nemotron
)

echo "=================================================================="
echo " Full 600M arch bake-off  |  cluster=$CLUSTER"
echo " families: $FAMILIES"
echo " wandb project: $WANDB_PROJECT"
echo " extra overrides (applied to ALL): ${*:-<none>}"
echo " logs: $LOG_DIR/<name>.log"
echo "=================================================================="

declare -A STATUS
overall_rc=0

for family in $FAMILIES; do
  name="${LOG_NAME[$family]:-bakeoff-600m-$family}"
  log="$LOG_DIR/${name}.log"
  echo
  echo ">>> [$(date '+%F %T')] START $family  ->  $log"

  bash scripts/train_bakeoff_600m.sh "$family" "cluster=$CLUSTER" "wandb.project=$WANDB_PROJECT" "$@" 2>&1 | tee "$log"
  rc="${PIPESTATUS[0]}"
  ln -sf "$log" "$LOG_DIR/terminal.log"

  if [[ "$rc" -eq 0 ]]; then
    STATUS[$family]="ok"
    echo ">>> [$(date '+%F %T')] DONE  $family (exit 0)"
  else
    STATUS[$family]="FAILED(rc=$rc)"
    overall_rc=1
    echo ">>> [$(date '+%F %T')] FAIL  $family (exit $rc)" >&2
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      echo ">>> STOP_ON_FAIL=1 -> aborting remaining runs" >&2
      break
    fi
  fi
done

echo
echo "===================== bake-off summary ==========================="
for family in $FAMILIES; do
  printf '  %-14s %s\n' "$family" "${STATUS[$family]:-skipped}"
done
echo "================================================================="
exit "$overall_rc"
