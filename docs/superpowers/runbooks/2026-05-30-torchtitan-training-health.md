# Runbook — torchtitan M3 functional training-health gate

**Date:** 2026-05-30
**Owner action required:** this gate runs on a GPU node and is **not** executed in
the dev harness. Run the per-family short trainings below, then point the health
test at each run's per-step loss log.

The acceptance gate is **functional, not Megatron-parity**: for each wired family
(llama3 → qwen3 → deepseek_v3) at a small scale, bf16 + AdamW + WSD, a short run
should **train and the loss should decrease healthily** (no NaN/Inf, sane curve
shape). An informal side-by-side vs the Megatron run is welcome as a sanity
signal but gates nothing.

> Before trusting a curve, glance at the **param count** torchtitan logs at
> startup — it should be in the right ballpark for the scale. A wildly-off count
> means the `slm_<scale>` flavor dims are wrong (see `src/titan_ext/model_flavor.py`).

## Prerequisites

```bash
# On a GPU node, from the slm-research repo root:
source load_cuda13_2_nccl_env.sh          # cuda/13.2 + cuBLAS LD_PRELOAD (TE symbol fix)
git submodule update --init --recursive   # ensure third_party/torchtitan is checked out
# Install the torchtitan extra into the env if not already present:
#   uv pip install -e ".[torchtitan]"      (torchdata, tomli-w, tiktoken, blobfile, torchao, tyro)
```

## Short runs (one per family)

These reuse the slm wrappers — the only backend-specific change is
`--backend torchtitan`. `codexlog <name> <cmd>` is the repo's run-logger; drop it
if you launch directly.

**GPU count.** These are tiny `300m` / `seq_len=256` smoke runs; the gate only
needs a healthy, decreasing loss curve, not throughput. Parallelism resolves to
TP=1 / PP=1 with FSDP2 (`data_parallel_shard_degree=-1`) sharding over whatever
ranks `torchrun` launches — so **1 GPU is sufficient**. `cluster=h100_de` sets
`gpus_per_node: 4` by default; override `cluster.gpus_per_node=N` to use fewer.

### 1-GPU variant (simplest — recommended for the gate)

```bash
# llama3 (dense) on torchtitan, single GPU
codexlog titan_health_llama3 \
  scripts/train_adam.sh llama3 --backend torchtitan base/scale=300m \
    cluster.gpus_per_node=1 training_regime=ablation_20x seed=7 training.train_iters=200

# qwen3 (dense) on torchtitan, single GPU
codexlog titan_health_qwen3 \
  scripts/train_adam.sh llama3 --backend torchtitan base/family=qwen3 base/scale=300m \
    cluster.gpus_per_node=1 training_regime=ablation_20x seed=7 training.train_iters=200

# deepseek_v3 (MoE + MLA) on torchtitan, single GPU — wired last; runs a
# torchtitan-native flavor in M1 (base.model.titan_flavor, default debugmodel). A
# proper slm-sized deepseek flavor + [parallelism].expert_parallel_degree mapping
# is a follow-on. (Drop cluster.gpus_per_node=1 if you want expert parallel later.)
codexlog titan_health_dsv3 \
  scripts/train_adam.sh deepseek_v3 --backend torchtitan \
    cluster.gpus_per_node=1 training_regime=ablation_20x seed=7 training.train_iters=200
```

### 4-GPU variant (full `h100_de` node, FSDP2 over 4 ranks)

```bash
# llama3 (dense) on torchtitan
codexlog titan_health_llama3 \
  scripts/train_adam.sh llama3 --backend torchtitan base/scale=300m \
    training_regime=ablation_20x seed=7 training.train_iters=200

# qwen3 (dense) on torchtitan
codexlog titan_health_qwen3 \
  scripts/train_adam.sh llama3 --backend torchtitan base/family=qwen3 base/scale=300m \
    training_regime=ablation_20x seed=7 training.train_iters=200

# deepseek_v3 (MoE + MLA) on torchtitan — wired last; runs a torchtitan-native
# flavor in M1 (base.model.titan_flavor, default debugmodel). A proper slm-sized
# deepseek flavor + [parallelism].expert_parallel_degree mapping is a follow-on.
codexlog titan_health_dsv3 \
  scripts/train_adam.sh deepseek_v3 --backend torchtitan \
    training_regime=ablation_20x seed=7 training.train_iters=200
```

## Extracting the per-step loss jsonl

torchtitan logs metrics to its `[metrics]` sink (W&B when `enable_wandb`, else
the TensorBoard folder under the run's `dump_folder`). Produce a jsonl with one
`{"loss": <float>}` per step from whichever sink you have, e.g.:

- **W&B:** export the run history and emit `{"loss": row["loss"]}` per logged step.
- **TensorBoard:** read the `loss`/`loss_metrics/global_avg_loss` scalar from the
  event file under the run's `tb` folder and emit one line per step.

Save it as e.g. `runs/<run_name>/tt_loss.jsonl`.

## Run the health gate

```bash
TT_LOSS_LOG=runs/<run_name>/tt_loss.jsonl \
  pytest -m gpu tests/numerics/test_titan_training_health.py -v
```

Expected: PASS — ≥40 steps, all finite, final-window mean ≥0.1 below the
early-window mean. (Optional sanity: run the same config on `--backend megatron`
and eyeball the two curves side by side — informational only.)

## Report back

Post, per family: the startup param count, the loss curve (or early/late means),
and whether the gate passed. Tighten the thresholds in
`tests/numerics/test_titan_training_health.py` in a follow-up once real curves
exist.
