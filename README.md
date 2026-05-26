# slm-research

Research infrastructure for small language model (SLM) pretraining with novel
training algorithms and architectural variants.

See [SPEC.md](SPEC.md) for the authoritative design document. Everything in
this repo exists to preserve the invariant:

> Two runs with the same `config_hash`, `dataset_hash`, and `git_sha`
> reproduce the same curve up to seed variance.

**Setup:** see [INSTALL.md](INSTALL.md).

## Quick start: data → training

The Nemotron-CC-v2 High-Quality corpus has already been curated and
tokenized with the Llama-3.1-8B tokenizer in the upstream fork; the
Megatron `text_document` prefix lives at

```
/lustre/fast/fast/zqiu/Megatron-LM/Nemotron-CC-v2/nemotron_cc_v2_high_quality_text_document_llama31_8b
```

(2.4 TB `.bin` + 16 GB `.idx`). The preprocessing pipeline that produced
it — parquet → jsonl → tokenized `.bin/.idx` — lives in the fork at
`/lustre/fast/fast/zqiu/Megatron-LM/tools/preprocess_data_parquet_to_jsonl.{py,sh}`
followed by `tools/preprocess_data.sh`. Porting those into
`tools/preprocess_*` inside slm-research is a separate plan; until that
lands, point `data.path` at the existing prefix above.

1. **Install the env** once per cluster (full recipe in [INSTALL.md](INSTALL.md)):

   ```bash
   bash slm-research/install_slm_env.sh <your_name>
   source ../<your_name>/.venv/bin/activate
   ```

2. **Launch a training run** with one of the per-optimizer startup
   wrappers in `scripts/`. Each wrapper fixes the `experiment=` axis and
   forwards extra Hydra overrides through `"$@"`:

   ```bash
   # Llama-3 family
   scripts/train_adam.sh llama3 --dry-run
   scripts/train_muon.sh llama3 --dry-run
   scripts/train_poet.sh llama3 --dry-run

   # DeepSeek-V3-style family
   scripts/train_adam.sh deepseek_v3 --dry-run
   scripts/train_muon.sh deepseek_v3 --dry-run
   scripts/train_poet.sh deepseek_v3 --dry-run

   # Override any axis or scalar config inline
   scripts/train_adam.sh llama3 base/scale=600m data=nemotron_cc_v2_scratch_qwen3 seed=7
   ```

   The default `data=` axis points at the Llama-3.1-tokenized prefix
   above. Switch to scratch tokenizers with `data=nemotron_cc_v2_scratch_llama31`,
   `data=nemotron_cc_v2_scratch_qwen3`, or `data=nemotron_cc_v2_scratch_qwen35`.

   Add `--dry-run` to resolve and archive the config without submitting
   to SLURM. Override any axis or scalar config inline, e.g.
   `scripts/train_adam.sh llama3 base/scale=1_2b training_regime=final_200x seed=7`.

3. **Inspect the run.** The launcher writes the resolved config and
   launch metadata (including `patch_set_hash`, `git_sha`, `megatron_sha`,
   `dataset_hash`) to `runs/<config_hash>/` before any GPU work; W&B logs
   land in `wandb.project` (defaults to `sandbox-${USER}`).

## Repository layout

See [SPEC.md §3](SPEC.md) for the canonical tree. The structural ideas
below are the ones you must internalize before adding code — they are
what enforces the reproducibility invariant.

### A run is a 5-axis composition

Every experiment is `{base_family, base_scale, experiment, training_regime,
cluster, seed}` (SPEC.md §2 principle 2). Each axis lives under its own
sub-directory of `configs/` and composes at launch via Hydra. **A new
ablation is a YAML diff under `configs/experiments/<area>/<name>.yaml`,
not a new launcher script.** The launcher resolves the composition,
hashes it (`config_hash`), and writes the resolved YAML to
`runs/<config_hash>/` before any GPU work happens.

```
configs/
  base/family/{llama3,qwen3,minicpm}.yaml   # arch lineage (norm, rope, tokenizer)
  base/scale/{300m,600m,1_2b,2_4b,7b}.yaml  # dimensional only (layers, hidden, ffn, heads)
  experiments/<area>/<name>.yaml             # the *logical* ablation diff
  training_regime/{ablation_20x,final_200x,...}.yaml
  clusters/{h800_cn,h100_de,...}.yaml        # parallelism, precision, kernels
  launch/{monthly_sweep,weekly_gate,...}.yaml
```

Ablation YAMLs are hardware-agnostic. Cluster-specific realization
(parallelism, FP8/FP4 recipe, attention backend) is mixed in *only* via
`configs/clusters/<cluster>.yaml`. Don't put cluster knobs in an
experiment file or you've broken axis 1.

### Where code goes

- `src/model/{attention,mlp,norm,embedding,positional}/` — architecture
  variants. Each subtree has a `baseline.py` (the reference) plus one
  file per variant. Variants are wired into Megatron Core via
  `src/specs/gpt_layer_specs.py::build_spec(experiment_cfg) -> ModuleSpec`.
- `src/optim/` — `__init__.py` exposes `get_optimizer(cfg, params, mcore_cfg)`,
  one file per optimizer (`muon.py`, `newton_schulz.py`, ...), schedulers
  under `schedulers/`.
- `src/precision/` — TransformerEngine recipe overrides + `capability.py`
  with the `CAPABILITY_TAGS` constants used by submit-time guarding.
- `src/patches/` — disciplined monkey-patch layer for things `ModuleSpec`
  cannot reach. One file per patch with a required docstring; registered
  via `_registry.py` (decorator + conflict detection); applied by
  `apply_patches(names) -> patch_set_hash`. The hash goes into run metadata.
  **This is the only place upstream Megatron behavior is mutated.**
- `src/data/` — deterministic indexed loader + SHA256 manifest verification.
- `src/eval/` — token-milestone evaluation harness + per-task modules.
- `src/utils/` — `config_hash.py`, `git_meta.py`, `checkpoint_convert.py`,
  `ladder_math.py`, `wandb_helpers.py`. Anything used cross-subsystem.

### Megatron Core is read-only

`third_party/Megatron-LM/` is a pinned submodule. **Never edit it.** All
extensions integrate either through `ModuleSpec` (preferred) or through
`src/patches/` (when the spec system cannot reach the call site). The
pin SHA is load-bearing: the reproducibility invariant references
`megatron_sha` directly. Bump procedure in
[docs/megatron_pin.md](docs/megatron_pin.md).

### Run identity & archive

A run is identified by `(git_sha, megatron_sha, config_hash, dataset_hash,
patch_set_hash, seed)`. If any of those isn't pinned and logged, the run
does not go to the shared W&B project — that's the contract.
`runs/<config_hash>/` is the **append-only** local archive: the resolved
config, launch metadata, and an auto-generated README. Don't delete or
edit those directories; they're how `tools/reproduce.py` replays a run.

### Where reasoning lives

YAMLs hold *what*; W&B holds *metrics*; the *why* (hypothesis, what
worked, what didn't, follow-ups) lives in
`docs/experiments/<name>.md`, one file per experiment. Promotion
decisions reference these files. Without them, institutional memory
evaporates when people leave — this is enforced by review, not by code.

### Top-level summary

- `src/` — architecture, optimizer, precision, specs, patches, data, eval, utils
- `configs/` — the 5-axis YAML tree described above
- `launchers/` — Hydra entry points + per-cluster SLURM templates
- `tools/` — reporting, validation, replay, config introspection
- `runs/` — append-only archive of resolved configs (one dir per `config_hash`)
- `docs/experiments/` — lab notebook: one markdown file per experiment
- `third_party/Megatron-LM/` — pinned submodule; never edited

## Implementation progress

See [docs/megatron_pin.md](docs/megatron_pin.md) for the currently pinned
Megatron-LM SHA and bump history, and
[docs/experiments/champion_history.md](docs/experiments/champion_history.md)
for the champion promotion log.

This repo is under active scaffolding; follow the order in SPEC.md §12.
