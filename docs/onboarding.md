# Onboarding

Welcome. This repo is the research infrastructure for SLM pretraining
experiments. The authoritative design is [SPEC.md](../SPEC.md); this doc is
the 15-minute "what do I do on day one" version.

## Mental model

Every run is built by composing six independently-swappable axes:

```
base/family   +   base/scale   +   experiment   +   training_regime   +   cluster   +   seed
  (Llama-3,      (600M, 1.2B,     (optim, attn,    (20x, 40x, 200x)     (h800_cn,      (int)
   Qwen-3, ...)   2.4B, 7B)        precision,...)                        hpc_de, ...)
```

A **`config_hash`** (blake2s, 16 hex chars) is computed over the resolved
config, excluding seed / wandb / parallelism / checkpointing cadence. The
invariant: same `config_hash` ⇒ same curve up to seed variance.

Seeds sharing a `config_hash` share a W&B group — aggregation across seeds
is automatic.

## Your first day

1. Clone with submodules: `git clone --recurse-submodules …`
2. Set up env: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
3. Install hooks: `pre-commit install`
4. Run unit tests: `pytest -m "not gpu"` — everything in `tests/unit/` runs on a laptop.
5. Read [SPEC.md](../SPEC.md). Skim all of §1-§5; §9 is the launcher pipeline you'll live in.
6. Dry-run a launch:
   ```bash
   python -m launchers.submit \
       base/family=qwen3 base/scale=600m \
       experiment=champion training_regime=ablation_20x \
       cluster=h800_cn seed=0 wandb.project=sandbox-${USER} \
       --dry-run
   ```
   This resolves the config, runs every guardrail, writes
   `runs/<config_hash>/resolved_config.yaml`, and exits without submitting.

## Your first experiment

1. Copy `configs/experiments/_template.yaml` to
   `configs/experiments/<family>/<my_idea>.yaml` and fill in the placeholders.
2. Create `docs/experiments/<my_idea>.md` from the template (CI will refuse
   the PR otherwise).
3. Dry-run at 600M across two seeds; eyeball the resolved config.
4. Real-run on `cluster=h800_cn` with `wandb.project=sandbox-<you>`.
5. When results are credible, rerun under `wandb.project=pretrain-ablations-600m`
   with `job_type=ablation`.
6. If the 600M signal is positive, sweep the ladder (600m → 1.2b → 2.4b).
7. Update the doc as you learn.

## Where things live

- Architecture variants: `src/model/`
- Optimizers: `src/optim/`
- Precision recipes: `src/precision/`
- Megatron escape-hatch patches: `src/patches/`
- Evals: `src/eval/`
- Utilities (hashing, git, ladder math): `src/utils/`
- Configs: `configs/{base,experiments,training_regime,clusters,launch}/`
- Launchers: `launchers/submit.py`, `launchers/sweep.py`
- Tooling (reports, audits): `tools/`
- Experiment notes (the lab notebook): `docs/experiments/`

## What NOT to do

- Don't edit `third_party/Megatron-LM/`. Use `src/patches/` and register
  with `@register_patch`.
- Don't override `training.global_batch_size_tokens` or `training.seq_length`
  without the explicit `--override-ladder-config` flag — those are frozen.
- Don't submit dirty working trees to a shared W&B project. Sandbox projects
  are fine for in-progress work.
- Don't push an experiment YAML without its companion `docs/experiments/<name>.md`.
