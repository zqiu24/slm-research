# Huawei POET DeepSeek-3B Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On a new `huawei` branch in slm-research, vendor the reference Megatron-poet (core 0.14.1) POET + DeepSeek-3B stack into an isolated `poet_torch_huawei/` subtree and produce a single-GPU, mock-data smoke run of the block-size-128 POET DeepSeek-3B script that trains for ~30 steps and fires a POET merge.

**Architecture:** Isolated vendor. We do **not** integrate into slm-research's existing Megatron 0.17 / `src/patches/poet_*` / `third_party/poet_torch`. Instead we copy the reference repo's *own* runtime (its Megatron 0.14 + `poet_torch` math package + `poet_adapter` glue + `pretrain_gpt_poet.py` + training scripts) verbatim into a single top-level folder `poet_torch_huawei/`, run it under its own env with `PYTHONPATH=poet_torch_huawei/`, and add one shrunk single-GPU/mock dev launcher. Because we run the reference's own Megatron, **none** of the 0.14→0.17 incompatibilities (optimizer param-group API, `train_step` signature, missing sandwich-norm) apply.

**Tech Stack:** Megatron-LM core 0.14.1 (vendored), `poet_torch` (Triton + `torch.compile` POET ops), Transformer-Engine (TE weight-swap POET path), torchrun, bash launch scripts, MoE (SequentialMLP + alltoall dispatcher), MLA-off MQA attention with `poet_split_qkv`.

**Execution note — runs on the `poet` node.** The user is granting access to a node with `conda activate poet` (torch 2.8 / triton 3.4, single GPU), so the import smoke (Task 4) and the training smoke (Task 8) can be executed directly there — no need to hand every command back. If that node is unavailable at execution time, those two tasks fall back to "the user runs and reports." All file copy/create/edit steps run wherever the repo lives.

**Source of truth (reference repo):** `/lustre/fast/fast/zqiu/tmp/Megatron-poet`
**Target repo:** `/lustre/fast/fast/zqiu/slm-research`

---

## File Structure

Everything lands under one isolated folder so it cannot clash with existing slm-research code. The inner `poet_torch` package keeps its name (no rename needed) — isolation comes from the folder + `PYTHONPATH`, and `sys.path` entries win over the editable `poet_torch` finder installed in `slm_env`.

```
slm-research/
├── poet_torch_huawei/                      # NEW — the whole vendored reference stack
│   ├── megatron/                           #   vendored Megatron-core 0.14.1 (incl. poet_adapter + split-qkv + sandwich-norm)
│   ├── poet_torch/                         #   POET math package (Triton ops, POETLinear, POETAdamW) — unchanged
│   ├── pretrain_gpt.py                      #   base entry (dataset/forward/loss)
│   ├── pretrain_gpt_poet.py                 #   POET entry: wraps model_provider + merge hook + optimizer hook
│   ├── training_scripts/
│   │   ├── parse_yaml.sh                     #   YAML→bash MODEL_ARGS (unchanged)
│   │   ├── train_DeepSeek_3bv3_sandwich_mqa_poet.sh   # original 8-GPU script (secrets scrubbed)
│   │   ├── model_args/DeepSeek-3Bv2-sandwich-mqa-poet.yaml       # original config (unchanged)
│   │   ├── model_args/DeepSeek-3Bv2-sandwich-mqa-poet-dev.yaml   # NEW — shrunk dev config
│   │   └── train_DeepSeek_dev_mock_1gpu.sh   # NEW — single-GPU mock smoke launcher
│   └── (tools/ carried along for transitive imports; examples/docs/tests excluded — Task 2)
├── scripts/train_poet_huawei.sh             # NEW — thin entry mirroring train_adam.sh/train_poet.sh
├── docs/superpowers/plans/2026-05-28-huawei-poet-deepseek-3b-port.md   # this plan
└── .gitignore                               # MODIFY — ignore dev run outputs under poet_torch_huawei/
```

**Responsibilities:**
- `poet_torch_huawei/megatron/` — the exact Megatron the reference POET path was validated against; do not edit.
- `poet_torch_huawei/poet_torch/` — pure-torch POET math; no Megatron coupling.
- `pretrain_gpt_poet.py` — the only entry point; resolves `poet_torch` via its own `sys.path` insert.
- `training_scripts/*.sh` — launch glue; the new dev script is the deliverable.

---

## Task 1: Create the `huawei` branch and ignore rules

**Files:**
- Create: `slm-research/.gitignore` entries (modify if it exists)

- [ ] **Step 1: Create the branch explicitly from `main` (current HEAD may be detached)**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research
git fetch --all --quiet || true
# Base off main, NOT the current HEAD (the session shows HEAD is detached).
git switch -c huawei main 2>/dev/null || git switch -c huawei origin/main
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
```
Expected: prints `huawei`, and the last commit matches the tip of `main`.

- [ ] **Step 2: Add ignore rules for dev run outputs (keep large artifacts out of git)**

Append to `/lustre/fast/fast/zqiu/slm-research/.gitignore`:
```gitignore
# huawei vendored POET stack — runtime outputs (not source)
poet_torch_huawei/runs_dev/
poet_torch_huawei/**/__pycache__/
poet_torch_huawei/**/*.pyc
poet_torch_huawei/**/wandb/
poet_torch_huawei/**/*.log
poet_torch_huawei/training_scripts/**/data_cache/
```

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add .gitignore
git commit -m "chore(huawei): branch + ignore rules for vendored POET stack"
```

---

## Task 2: Vendor the reference stack into `poet_torch_huawei/`

**Files:**
- Create: `slm-research/poet_torch_huawei/` (full copy of the reference repo, minus VCS/caches/artifacts)

- [ ] **Step 1: Copy the reference repo into the isolated folder (exclude VCS, caches, logs, checkpoints)**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research
rsync -a \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='wandb/' \
  --exclude='*.log' \
  --exclude='*.pt' \
  --exclude='*.ckpt' \
  --exclude='checkpoints/' \
  --exclude='experiments/' \
  --exclude='examples/' \
  --exclude='images/' \
  --exclude='docs/' \
  --exclude='tasks/' \
  --exclude='tests/' \
  --exclude='docker/' \
  /lustre/fast/fast/zqiu/tmp/Megatron-poet/ \
  /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/
```

The excluded dirs (`examples/`, `images/`, `docs/`, `tasks/`, `tests/`, `docker/`) are not imported by the training path — they only bloat the vendor commit. Step 4 verifies nothing in the kept tree imports them.

- [ ] **Step 2: Verify the load-bearing files are present**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
ls pretrain_gpt.py pretrain_gpt_poet.py \
   megatron/core/poet_adapter/adapter.py \
   poet_torch/__init__.py poet_torch/core/ops.py poet_torch/core/triton_ops.py \
   training_scripts/parse_yaml.sh \
   training_scripts/model_args/DeepSeek-3Bv2-sandwich-mqa-poet.yaml \
   training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poet.sh
```
Expected: all paths listed, no "No such file".

- [ ] **Step 3: Confirm the vendored Megatron is 0.14 (not the target 0.17)**

Run:
```bash
grep -E "MINOR|PATCH" /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/package_info.py
```
Expected: `MINOR = 14`.

- [ ] **Step 4: Confirm nothing in the kept tree imports the excluded dirs**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
grep -rnE "(from|import) (examples|tasks|tests)([. ]|$)" \
  megatron pretrain_gpt.py pretrain_gpt_poet.py poet_torch training_scripts 2>/dev/null \
  || echo "NO CROSS-IMPORTS"
```
Expected: `NO CROSS-IMPORTS`. If a match appears, re-run the rsync without excluding that specific dir.

- [ ] **Step 5: Commit the vendor drop (large commit; source only)**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei
git commit -m "feat(huawei): vendor reference Megatron 0.14 POET + DeepSeek-3B stack"
```

---

## Task 3: Scrub the hardcoded secret and cluster hardcodes from the copied original script

**Files:**
- Modify: `slm-research/poet_torch_huawei/training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poet.sh:127`

- [ ] **Step 1: Replace the hardcoded W&B API key with an env passthrough**

Edit the file. Replace this exact line:
```bash
export WANDB_API_KEY="4604e54e9c69942344bf98f695b966bc710a6a90"
```
with:
```bash
export WANDB_API_KEY="${WANDB_API_KEY:-}"
```

- [ ] **Step 2: Verify no secret remains anywhere in the vendored tree**

Run:
```bash
grep -rn "4604e54e9c69942344bf98f695b966bc710a6a90" /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/ || echo "CLEAN"
```
Expected: prints `CLEAN`.

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poet.sh
git commit -m "fix(huawei): scrub hardcoded W&B key from vendored training script"
```

---

## Task 4: Validate the runtime env (`poet` conda env)

**Decided:** the runtime env is the existing **`poet` conda env** (`/home/zqiu/anaconda3/envs/poet`) — torch 2.8.0+cu129, triton 3.4.0, flash_attn 2.8.3, which match the reference's pins. It has **no Transformer Engine**, so the run uses `--transformer-impl local` (Task 5 Step 7). We do NOT use slm_env (torch 2.11 / TE 2.14 — too new for Megatron-core 0.14). This task is an import smoke before the run; it can be executed directly on the `poet` node once access is available.

- [ ] **Step 1: Activate `poet` and import-smoke the vendored stack (no TE expected)**

Run on the `poet` node:
```bash
source /home/zqiu/anaconda3/etc/profile.d/conda.sh && conda activate poet
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei
PYTHONPATH=. python -c "import torch; print('torch', torch.__version__); \
import triton; print('triton', triton.__version__); \
import poet_torch; print('poet_torch OK'); \
import megatron; from megatron.core import package_info as p; print('megatron-core', p.__version__); \
from megatron.core import poet_adapter; print('poet_adapter OK')"
# parse_yaml.sh (sourced by the launcher) shells out to python3 + PyYAML:
python3 -c "import yaml; print('pyyaml OK')"
```
**Expected:** `torch 2.8.0+cu129`, `triton 3.4.0`, `poet_torch OK`, `megatron-core 0.14.1`, `poet_adapter OK`, `pyyaml OK`. (TE is intentionally absent — local impl doesn't need it.)

- [ ] **Step 2: If `pyyaml` or `einops` is missing, install into `poet`**

`parse_yaml.sh` needs PyYAML; Megatron-core needs `einops`/`packaging`/`numpy<2`. If Step 1 reports a missing module:
```bash
conda activate poet && python -m pip install pyyaml "einops~=0.8" "numpy<2.0.0" packaging
```
Re-run Step 1. If the *megatron* import itself fails (not a trivial missing dep), capture the full traceback and STOP — a deeper 0.14/torch-2.8 incompatibility is a decision point, not a guess.

---

## Task 5: Create the single-GPU mock-data dev config

We keep every *dimension* identical to the real config (so block-size 128 divisibility holds — including the split-qkv projections: q=6144, k=v=384, all divisible by 128) and only shrink *counts* (layers, experts) and merge interval.

**Files:**
- Create: `slm-research/poet_torch_huawei/training_scripts/model_args/DeepSeek-3Bv2-sandwich-mqa-poet-dev.yaml`

- [ ] **Step 1: Copy the real config to the dev filename**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/training_scripts/model_args
cp DeepSeek-3Bv2-sandwich-mqa-poet.yaml DeepSeek-3Bv2-sandwich-mqa-poet-dev.yaml
```

- [ ] **Step 2: Shrink layer count — edit `--num-layers`**

In `DeepSeek-3Bv2-sandwich-mqa-poet-dev.yaml`, replace:
```yaml
  --num-layers: 12
```
with:
```yaml
  --num-layers: 4
```

- [ ] **Step 3: Match MoE layer-freq to the new layer count — edit `--moe-layer-freq`**

Replace:
```yaml
  --moe-layer-freq: ([0]*1+[1]*11)
```
with:
```yaml
  --moe-layer-freq: ([0]*1+[1]*3)
```
(1 dense + 3 MoE = 4 layers; Megatron asserts `len(layer_freq) == num_layers`.)

- [ ] **Step 4: Shrink expert count — edit `--num-experts`**

Replace:
```yaml
  --num-experts: 64
```
with:
```yaml
  --num-experts: 8
```

- [ ] **Step 5: Make router topk valid/cheap for 8 experts — edit `--moe-router-topk`**

Replace:
```yaml
  --moe-router-topk: 6
```
with:
```yaml
  --moe-router-topk: 2
```

- [ ] **Step 6: Shorten the merge interval so a merge fires inside the smoke — edit `--poet-merge-interval`**

Replace:
```yaml
  --poet-merge-interval: 200
```
with:
```yaml
  --poet-merge-interval: 20
```

- [ ] **Step 7: Switch off TE — edit `--transformer-impl` (the `poet` env has no Transformer Engine)**

Replace:
```yaml
  --transformer-impl: transformer_engine
```
with:
```yaml
  --transformer-impl: local
```
POET then wraps Megatron's native `ColumnParallelLinear`/`RowParallelLinear` (the reference `pretrain_gpt_poet.py` already rebinds `gpt_layer_specs.LNImpl = WrappedTorchNorm` so RMSNorm works under local). This is the POET path the `poet` env is set up for.

- [ ] **Step 8: Verify block-size-128 divisibility still holds and counts changed**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/training_scripts/model_args
grep -E "num-layers|moe-layer-freq|num-experts|moe-router-topk|poet-merge-interval|poet-block-size|transformer-impl|hidden-size|ffn-hidden-size|moe-ffn-hidden-size|kv-channels|num-attention-heads|num-query-groups" DeepSeek-3Bv2-sandwich-mqa-poet-dev.yaml
```
Expected: `--num-layers: 4`, `--moe-layer-freq: ([0]*1+[1]*3)`, `--num-experts: 8`, `--moe-router-topk: 2`, `--poet-merge-interval: 20`, `--poet-block-size: 128`, `--transformer-impl: local`, and unchanged dims (hidden 1280, ffn 7168, moe-ffn 896, kv-channels 384, heads 16, query-groups 1). Divisibility by 128: 1280/128=10, 7168/128=56, 896/128=7, qkv split q=16*384=6144→48, k=v=384→3 — all integers.

- [ ] **Step 9: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/training_scripts/model_args/DeepSeek-3Bv2-sandwich-mqa-poet-dev.yaml
git commit -m "feat(huawei): single-GPU mock dev config for DeepSeek-3B POET (block-size 128)"
```

---

## Task 6: Create the single-GPU mock-data dev launcher

A self-contained, short script (not derived from the 380-line cluster script) that runs 1 GPU, EP=1, mock data, offline W&B, block-size 128, 30 steps.

**Files:**
- Create: `slm-research/poet_torch_huawei/training_scripts/train_DeepSeek_dev_mock_1gpu.sh`

- [ ] **Step 1: Write the dev launcher**

Create the file with exactly this content (env = `poet` conda env, per Task 4):
```bash
#!/bin/bash
# Single-GPU, mock-data SMOKE run of DeepSeek-3B + POET (block-size 128, local impl).
# Isolated huawei vendor stack (Megatron-core 0.14). Proves POET wraps and
# trains end-to-end before scaling to the real 8-GPU run. No external data.
set -o pipefail

# --- Environment: the `poet` conda env (torch 2.8 / triton 3.4, no TE) -----
source /home/zqiu/anaconda3/etc/profile.d/conda.sh && conda activate poet

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Offline W&B for a smoke (no key needed).
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-huawei_poet_dev}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-deepseek3b_poet_dev_smoke}"

# --- Paths -----------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
CODE_BASE_DIR=$(cd "$SCRIPT_DIR/../" && pwd)   # .../slm-research/poet_torch_huawei
export PYTHONPATH=$CODE_BASE_DIR
cd "$CODE_BASE_DIR"

OUT_DIR="${OUT_DIR:-$CODE_BASE_DIR/runs_dev/poet_dev_smoke}"
mkdir -p "$OUT_DIR/data_cache" "$OUT_DIR/checkpoints" "$OUT_DIR/tb"

# --- Model config (shrunk dev YAML) ----------------------------------------
YAML_CONFIG="$SCRIPT_DIR/model_args/DeepSeek-3Bv2-sandwich-mqa-poet-dev.yaml"
source "$SCRIPT_DIR/parse_yaml.sh" "$YAML_CONFIG"
read -ra MODEL_ARGS <<< "$MODEL_ARGS_FROM_CONFIG"

# --- Single-GPU launch -----------------------------------------------------
DISTRIBUTED_ARGS=( --nproc_per_node 1 --nnodes 1 --node_rank 0
                   --master_addr localhost --master_port "${MASTER_PORT:-6000}" )

TRAINING_SPECIFIC_ARGS=(
  --micro-batch-size 1
  --global-batch-size 8
  --lr 8.6e-4 --min-lr 7e-6
  --lr-warmup-iters 2 --lr-decay-style WSD --lr-wsd-decay-style cosine --lr-wsd-decay-iters 10
  --train-iters 30
  --seq-length 512
  --calculate-per-token-loss
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
)

# block-size 128 explicitly (reference script default is 256).
POET_CLI_ARGS=( --poet-block-size 128 --poet-merge-interval 20 )

DATA_ARGS_LIST=(
  --mock-data
  --tokenizer-type NullTokenizer
  --vocab-size 151936
  --data-cache-path "$OUT_DIR/data_cache"
  --split 99,1,0
  --no-create-attention-mask-in-dataloader
  --no-mmap-bin-files
  --num-workers 1
)

CHECKPOINT_LOGGING_ARGS=(
  --save "$OUT_DIR/checkpoints" --load "$OUT_DIR/checkpoints"
  --tensorboard-dir "$OUT_DIR/tb"
  --eval-iters 0 --eval-interval 1000 --save-interval 1000
  --ckpt-format torch_dist
  --log-interval 1
)

echo "[dev] launching DeepSeek-3B POET smoke: 1 GPU, mock data, block-size 128, 30 steps"
torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt_poet.py \
  "${MODEL_ARGS[@]}" "${TRAINING_SPECIFIC_ARGS[@]}" \
  "${POET_CLI_ARGS[@]}" "${DATA_ARGS_LIST[@]}" "${CHECKPOINT_LOGGING_ARGS[@]}"
```

- [ ] **Step 2: Make it executable**

Run:
```bash
chmod +x /lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/training_scripts/train_DeepSeek_dev_mock_1gpu.sh
```

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/training_scripts/train_DeepSeek_dev_mock_1gpu.sh
git commit -m "feat(huawei): single-GPU mock-data POET smoke launcher (block-size 128)"
```

---

## Task 7: Add a thin `scripts/train_poet_huawei.sh` entry point

Mirrors the existing `scripts/train_adam.sh` / `scripts/train_poet.sh` for discoverability. It does **not** use slm-research's hydra launcher — it dispatches to the vendored scripts, which carry their own env activation + torchrun.

**Files:**
- Create: `slm-research/scripts/train_poet_huawei.sh`

- [ ] **Step 1: Write the wrapper**

Create the file with exactly this content:
```bash
#!/usr/bin/env bash
set -euo pipefail
# Thin entry for the vendored Huawei POET DeepSeek-3B stack (Megatron-core
# 0.14, isolated under poet_torch_huawei/). This deliberately does NOT use
# slm-research's hydra launcher (launchers.train_megatron) — the vendored
# scripts carry their own env activation + torchrun invocation.
#
#   scripts/train_poet_huawei.sh dev    # single-GPU mock smoke (block-size 128)
#   scripts/train_poet_huawei.sh full   # reference 8-GPU EP=8 DeepSeek-3B run
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUAWEI_SCRIPTS="$SLM_REPO/poet_torch_huawei/training_scripts"

MODE="${1:-dev}"
shift || true
case "${MODE}" in
  dev)
    exec bash "$HUAWEI_SCRIPTS/train_DeepSeek_dev_mock_1gpu.sh" "$@"
    ;;
  full)
    exec bash "$HUAWEI_SCRIPTS/train_DeepSeek_3bv3_sandwich_mqa_poet.sh" "$@"
    ;;
  *)
    echo "Usage: scripts/train_poet_huawei.sh [dev|full] [extra args]" >&2
    echo "  dev  - single-GPU mock-data smoke (block-size 128)" >&2
    echo "  full - reference 8-GPU EP=8 DeepSeek-3B POET run" >&2
    exit 2
    ;;
esac
```

- [ ] **Step 2: Make it executable**

Run:
```bash
chmod +x /lustre/fast/fast/zqiu/slm-research/scripts/train_poet_huawei.sh
```

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add scripts/train_poet_huawei.sh
git commit -m "feat(huawei): thin train_poet_huawei.sh entry (dev|full) for the vendored stack"
```

---

## Task 8: Smoke run (acceptance gate, on the `poet` node)

- [ ] **Step 1: Run the dev smoke on the `poet` node (tee'd to a log)**

Run (via the Task-7 wrapper; the wrapper activates `poet` and dispatches to the dev launcher):
```bash
codexlog poet_huawei_dev bash /lustre/fast/fast/zqiu/slm-research/scripts/train_poet_huawei.sh dev
```

- [ ] **Step 2: Verify the acceptance criteria in the log**

**Expected in `/lustre/home/zqiu/log/poet_huawei_dev.log`:**
1. A POET install summary line: `[POET] variant=poet | wrapped N parallel-linear layers | block_size=128 | merge_interval=20 …` with `N > 0`.
2. A params line: `[POET] params: total=… trainable=… oft_R=… (oft_R = …% of trainable)`.
3. Per-iteration loss logs progressing to iteration 30 (`--log-interval 1`).
4. At least one `[POET] merge-then-reinitialize at step 20` line.
5. Process exits 0 (no traceback).

- [ ] **Step 3: If it fails, triage in this order (record which fix applied)**

- `ModuleNotFoundError` (`yaml`/`einops`/triton) at startup → env problem; revisit Task 4 (install into `poet`). TE is expected to be absent — that is fine under `--transformer-impl local`.
- `make`/`gcc` error while "compiling dataset index builder" → the vendored `megatron/core/datasets` Helpers need a C++ toolchain in the run env (this compile happens even with `--mock-data`); ensure `gcc`/`make` are present (or load the cluster's build module), then re-run.
- `cudaErrorUnsupportedPtxVersion` / masked-softmax-fusion or persistent-layernorm error (local-impl fused kernels too new for the driver) → add `--no-masked-softmax-fusion: true` and `--no-persist-layer-norm: true` to the dev YAML and re-run (this is the same local-impl fix slm-research uses for llama3).
- Sandwich-norm / arg-unknown error → confirm the vendored `megatron/` is 0.14 (Task 2 Step 3); a 0.17 tree would reject `--use-sandwich-norm`.
- Divisibility assert in POET install → re-check Task 5 Step 8 (a changed dim broke 128-divisibility).
- `--reset-attention-mask`/`--reset-position-ids` incompatible with mock data → those two flags were already dropped from the dev script; only re-add if a real run needs them.
- OOM or very slow torch.compile across many expert shapes → lower `--num-experts` to 4 and `--num-layers` to 2 (keep `--moe-layer-freq: ([0]*1+[1]*1)`), re-run.

- [ ] **Step 4: Commit any triage fix applied to the dev config/script**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/training_scripts/
git commit -m "fix(huawei): dev smoke triage — <one line: what changed and why>"
```

---

## Task 9: Document the scale-up path to the real block-size-128 run

**Files:**
- Modify: `slm-research/poet_torch_huawei/training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poet.sh` (header comment only)

- [ ] **Step 1: Add a header note describing how to run the full 8-GPU block-128 config on this cluster**

Insert after the existing top comment block (around line 11) a comment documenting: the env, the `local`-impl requirement, block-size 128, data, and the 8-GPU requirement. Exact text:
```bash
# --- Running on THIS cluster (huawei vendor) -------------------------------
# 1. Env: `source /home/zqiu/anaconda3/etc/profile.d/conda.sh && conda activate poet`
#    (torch 2.8 / triton 3.4; NO Transformer Engine).
# 2. Because `poet` has no TE, edit this script's YAML
#    (model_args/DeepSeek-3Bv2-sandwich-mqa-poet.yaml) to use
#    `--transformer-impl: local` instead of transformer_engine — otherwise the
#    TE weight-swap path will fail to import TE. (A TE-2.7 env would be needed
#    for the exact TE path; not available here.)
# 3. Force block-size 128 (script default is 256):  export POET_BLOCK_SIZE=128
# 4. Provide cluster-local data + tokenizer, OR reuse mock data:
#      bash train_DeepSeek_3bv3_sandwich_mqa_poet.sh "" "" MOCK MOCK
#    (positional $3=tokenizer, $4=data; "MOCK" triggers --mock-data/NullTokenizer)
# 5. EP=8 needs 8 GPUs on one node. The single-GPU smoke is
#    train_DeepSeek_dev_mock_1gpu.sh.
# ---------------------------------------------------------------------------
```

- [ ] **Step 2: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git add poet_torch_huawei/training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poet.sh
git commit -m "docs(huawei): how to run the full 8-GPU block-128 DeepSeek-3B POET config"
```

---

## Self-Review

**Spec coverage:**
- "Port complete POET + 3B DeepSeek code to slm-research" → Task 2 (full vendor copy).
- "poet_torch_huawei folder so it doesn't mess up current code" → Task 2 folder layout + Task 1 ignore rules; isolation via folder + PYTHONPATH + own env (no edits to `third_party/poet_torch`, `src/patches/poet_*`, or vendored 0.17 Megatron).
- "new branch huawei" → Task 1.
- "start from my poet one (clean)" → source is `/lustre/fast/fast/zqiu/tmp/Megatron-poet` (Task 2).
- "run the script that trains POET with block-size 128" → Task 5/6 (dev YAML + launcher force block-size 128), Task 8 (smoke gate), Task 9 (full-run path).
- "add train_poet_huawei.sh" (user request) → Task 7 (thin `dev|full` wrapper next to `train_adam.sh`/`train_poet.sh`).
- Decisions locked with user: vendor 0.14 isolated / mock-data first / single-GPU dev first / dataset deferred (user will re-tokenize) / env = `poet` conda env / `--transformer-impl local` (no TE) — all reflected.

**Placeholder scan:** Triage bullets (Task 8 Step 3) and the scale-up note (Task 9) reference concrete flags/values, not "handle errors". The commit message in Task 8 Step 4 contains a fill-in `<one line>` for the *human-authored reason* — acceptable as it documents an action taken, not unspecified code.

**Type/name consistency:** Folder `poet_torch_huawei/`, inner package `poet_torch` (unchanged), dev YAML `DeepSeek-3Bv2-sandwich-mqa-poet-dev.yaml`, dev script `train_DeepSeek_dev_mock_1gpu.sh`, log tag `poet_huawei_dev` — used consistently across tasks.

**Runtime env — resolved:** the `poet` conda env (torch 2.8 / triton 3.4 / flash_attn 2.8.3) matches the reference pins and is the env to use; no env needs building. It has no TE, so the run uses `--transformer-impl local`. Residual risks, both covered by triage: (a) Megatron-0.14 source importing cleanly under torch 2.8 (Task 4 gate); (b) local-impl fused kernels (`masked_softmax`/`persist_layer_norm`) being too new for the node's driver (Task 8 triage adds the `--no-*-fusion` flags). Neither is a blind guess.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-28-huawei-poet-deepseek-3b-port.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. (Run/verify gates in Tasks 4 and 8 execute on the `poet` node you're granting access to.)

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
