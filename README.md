# slm-research

Research infrastructure for small language model (SLM) pretraining with novel
training algorithms and architectural variants.

See [SPEC.md](SPEC.md) for the authoritative design document. Everything in
this repo exists to preserve the invariant:

> Two runs with the same `config_hash`, `dataset_hash`, and `git_sha`
> reproduce the same curve up to seed variance.

## Installation

Our code sits on top of the pinned `third_party/Megatron-LM` submodule; the
whole framework is effectively "this repo + Megatron Core, both editable in
one venv." Getting that venv right is the main install step.

We use [`uv`](https://github.com/astral-sh/uv) for env / dependency
management. On a fresh CUDA host:

```bash
# 0. Install uv once (standalone binary, doesn't touch system Python):
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone with submodules (Megatron Core is a pinned submodule — see
#    docs/megatron_pin.md for the current SHA and bump procedure).
git clone --recurse-submodules https://github.com/zqiu24/slm-research.git
cd slm-research
# If you forgot --recurse-submodules:
#   git submodule update --init --recursive

# 2. Create a project-local venv. Pin to a Python the cluster has.
uv venv --python 3.10
source .venv/bin/activate

# 3. Install this package in editable mode, plus dev tooling.
uv pip install -e ".[dev]"

# 4. Install the Megatron Core submodule — editable, no build isolation
#    (mcore vendors CUDA-bound build steps that need the live torch).
#    Only run this on a node with a CUDA toolchain + torch already available.
uv pip install --no-build-isolation -e "./third_party/Megatron-LM[mlm]"

# 5. Install the GPU stack (torch + TransformerEngine) pinned per-cluster;
#    see configs/clusters/<cluster>.yaml for the exact versions. Example
#    for a cluster that pins torch 2.4.0 + TE 1.12.0:
uv pip install "torch==2.4.0"
uv pip install --no-build-isolation "transformer-engine[pytorch]==1.12.0"

# 6. Install git hooks.
pre-commit install

# 7. Sanity checks that need no GPU:
pytest -m "not gpu"
python -m launchers.submit \
    base/family=qwen3 base/scale=600m \
    experiment=champion training_regime=ablation_20x \
    cluster=h800_cn seed=0 \
    wandb.project=sandbox-${USER} allow_dirty=true \
    --dry-run
```

Step 4 is the one that most often fails on non-GPU machines — mcore's
editable install triggers CUDA-bound hooks. On a laptop / login node
without `nvcc` you can skip steps 4 and 5 and still run the unit tests,
launcher dry-run, and config tooling; everything under `tests/unit/` is
CPU-only.

## Updating the Megatron pin

The submodule SHA is load-bearing: changing it invalidates the champion
baseline until a rerun confirms parity. See
[docs/megatron_pin.md](docs/megatron_pin.md) for the current pin and the
full bump procedure. Short version:

```bash
cd third_party/Megatron-LM
git fetch --tags
git checkout <new-sha>
cd ../..
pytest tests/numerics/                 # patch-neutrality must still hold
git add third_party/Megatron-LM
# update docs/megatron_pin.md and commit together
```

## Repository layout

See [SPEC.md §3](SPEC.md) for the canonical layout. Top-level:

- `src/` — architecture, optimizer, precision, patches, eval, utils
- `configs/` — decomposable YAMLs: `base/{family,scale}`, `experiments`,
  `training_regime`, `clusters`, `launch`
- `launchers/` — Hydra entry points + SLURM templates
- `tools/` — reporting, validation, config introspection
- `runs/` — append-only archive of resolved configs (one dir per `config_hash`)
- `docs/experiments/` — lab notebook: one markdown file per experiment
- `third_party/Megatron-LM/` — pinned submodule; never edited

## Implementation progress

See [docs/megatron_pin.md](docs/megatron_pin.md) for the currently pinned
Megatron-LM SHA and bump history, and
[docs/experiments/champion_history.md](docs/experiments/champion_history.md)
for the champion promotion log.

This repo is under active scaffolding; follow the order in SPEC.md §12.
