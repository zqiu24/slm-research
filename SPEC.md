# Research Infrastructure Spec: SLM Pretraining with Novel Training Algorithms

> **Purpose of this document.** This is an implementation spec for a research infrastructure that enables a small team to run ablations on novel training algorithms and architectural variants for small language models (SLMs), across heterogeneous clusters, with reproducible cross-site comparability. The intended reader is an engineer (human or AI coding assistant) implementing the system from scratch.
>
> **This spec is the source of truth for design decisions.** It is not a tutorial. When in doubt about "why," the answer is usually "to preserve cross-site / cross-person / cross-month comparability of ablation results." That invariant takes precedence over convenience.

---

## 1. Research context

### 1.1 Overall goal

Develop novel **training algorithms** (optimizers, precision schemes) and **architectural variants** (attention, MLP, norm, MoE) for small language models positioned for edge deployment. The research target is a seed-stage investor demonstration: we show that our method produces a competitive 1.2B and 2.4B model at substantially less training compute than existing baselines (MiniCPM, Qwen2.5, Llama-3.2, SmolLM2), with a scaling slope that extrapolates favorably to larger sizes.

**The pretraining dataset is fixed.** Innovation lives in the training algorithm and architecture axes only.

### 1.2 Scale ladder

Five scales, geometrically spaced:

| Scale | Role | Tokens (ablation) | Tokens (final) |
|---|---|---|---|
| **600M** | Daily dev iteration | 24B @ 40× | — |
| **1.2B** | Scale check, monthly gate, final deliverable | 24B @ 20× | 360–480B @ 300–400× |
| **2.4B** | Monthly promotion gate, final deliverable | 48B @ 20× | 480B–1T @ 200–425× |
| **7B** | HPC extrapolation anchor (champion only) | 140B @ 20× | — |
| **300M** (optional) | Cheap smoke tests, slope anchor | 6B @ 20× | — |

Final deliverables are **only** at 1.2B and 2.4B.

### 1.3 Frozen ladder configuration

The following are **globally frozen across the entire ladder** and cannot be per-scale-tuned without explicit override:

- **Global batch size (in tokens): 4,194,304** (4M tokens). Chosen to match MiniCPM and because it's a reasonable balance for the ladder range.
- **Sequence length: 4096.** If attention research requires longer, it happens on a separate track.
- **Tokenizer: single frozen tokenizer used across all runs, regardless of declared base family.** The pretraining dataset is tokenized exactly once. Family-level tokenizer declarations are descriptive (documenting what the reference family would use in principle) but do not override the frozen dataset tokenizer. This is a deliberate confound-removal choice: research is about the training method, not tokenizer effects.
- **Dataset: single frozen pretraining corpus.** Hash-verified at job start.

Altering any of these invalidates the ladder's scaling slope and must be flagged as a separate experimental track.

### 1.4 Parameter counting convention

The ladder is defined in **non-embedding parameters**. `training.tokens_per_param` uses non-embedding params; the scale labels (600M / 1.2B / 2.4B / 7B) refer to non-embedding params.

Total params (including input embedding and LM head) are derived at launch and logged to W&B for reference, but do not drive ladder math. This ensures matched comparison across base-architecture families whose tokenizers differ in vocab size (if you ever vary family — see Section 5.1.1).

When reporting model sizes externally (papers, pitch decks, model cards), be explicit about which convention is used. Recommended: use total params for external-facing communication (matches how most published models are counted: Qwen, Llama, etc.) and non-embedding params for internal research (matches MiniCPM convention and ladder math).

### 1.5 Compute environments

| Cluster | Hardware | Role | Access pattern |
|---|---|---|---|
| **h800_cn** | 6× nodes of 8× H800 (48 GPUs) | Primary iteration cluster | Dedicated |
| **h100_de / a100_de / b200_de** | German shared cluster, variable allocation | Elastic ablation overflow | Bid system, dynamic node count |
| **hpc_de** | German public HPC | 7B anchor + 2.4B final | Monthly budget allocation |

**Precision capabilities vary.** A100 has no FP8; B200 has FP4. Capability tagging (Section 5.4) prevents mismatched experiment/cluster pairings.

---

## 2. Design principles

These are load-bearing. Violating them breaks the invariant that "two runs with the same resolved config, `dataset_hash`, and `git_sha` reproduce the same curve up to seed variance."

1. **One canonical definition of an experiment, hardware-agnostic.** An ablation is a small diff of *logical* parameters. Cluster-specific realization (parallelism, precision recipe, kernel backend) lives in separate cluster configs and composes in at launch.
2. **Configs decompose into `{base_family, base_scale, experiment, training_regime, cluster, seed}`.** Each is independently swappable. Base family (e.g., Llama-3, Qwen-3) and base scale (600M, 1.2B, ...) are separate axes so the research direction and the reference-architecture lineage are explicit.
3. **Tokens-seen is the unit of comparison.** Not wall-clock, not steps, not samples. All primary ablation metrics are evaluated at fixed token budgets.
4. **Megatron-LM upstream stays untouched.** Architecture and optimizer extensions live in a parallel `src/` tree and integrate through Megatron Core's `ModuleSpec` system and an optimizer factory. For things the spec system cannot reach, a disciplined `patches/` layer applies monkey-patches, hashed into the run metadata.
5. **Every run is reproducible from its archived `resolved_config.yaml` plus `(git_sha, megatron_sha, patch_set_hash, dataset_hash, seed)`.** If it isn't, it doesn't get logged to the shared W&B project. (Config hashing was removed; see §5.3.)
6. **Capability tagging is enforced at submit time.** An FP4-requiring experiment cannot be launched on a cluster lacking FP4 capability. Prevents silent "wrong numbers" accumulation.
7. **Seed replication is structural, not optional.** Seeds of the same config share a W&B group, automatically, via the launcher. Manual "I forgot" is not possible.
8. **Every experiment has human-written notes.** W&B captures metrics; YAMLs capture configs; but the *reasoning* — hypothesis, what worked, why something was promoted or rejected — lives in `docs/experiments/<name>.md`. Without this layer, institutional memory evaporates when people leave.

---

## 3. Repository layout

```
research/
├── third_party/
│   └── Megatron-LM/                    # git submodule, pinned SHA
│
├── src/
│   ├── __init__.py
│   │
│   ├── model/
│   │   ├── attention/
│   │   │   ├── baseline.py             # standard MHA/GQA reference
│   │   │   ├── gqa.py
│   │   │   ├── latent.py               # variants go here
│   │   │   └── hybrid.py
│   │   ├── mlp/
│   │   │   ├── baseline.py             # SwiGLU reference
│   │   │   ├── swiglu_variants.py
│   │   │   └── moe/
│   │   ├── norm/
│   │   ├── embedding/
│   │   └── positional.py
│   │
│   ├── optim/
│   │   ├── __init__.py                 # factory: get_optimizer(cfg, params, mcore_cfg)
│   │   ├── muon.py
│   │   ├── newton_schulz.py
│   │   ├── orthogonal_adam.py
│   │   └── schedulers/
│   │       ├── wsd.py                  # Warmup-Stable-Decay (MiniCPM-style)
│   │       └── cosine.py
│   │
│   ├── precision/
│   │   ├── __init__.py
│   │   ├── fp8_recipes.py              # TransformerEngine recipe overrides
│   │   ├── fp4_hooks.py
│   │   └── capability.py               # CAPABILITY_TAGS constants
│   │
│   ├── specs/
│   │   ├── __init__.py
│   │   └── gpt_layer_specs.py          # build_spec(experiment_cfg) -> ModuleSpec
│   │
│   ├── patches/
│   │   ├── __init__.py                 # apply_patches(names: list[str]) -> patch_set_hash
│   │   ├── _registry.py                # decorator-based registry, conflict detection
│   │   ├── parallel_layers.py          # one file per patch, docstring required
│   │   └── fp4_loss_scaling.py
│   │
│   ├── data/
│   │   ├── loader.py                   # deterministic indexed dataset
│   │   └── manifest.py                 # SHA256 manifest verification
│   │
│   ├── eval/
│   │   ├── harness.py                  # token-milestone evaluation
│   │   └── tasks/
│   │       ├── hellaswag.py
│   │       ├── piqa.py
│   │       ├── arc_e.py
│   │       ├── code_probe.py           # small HumanEval subset
│   │       └── math_probe.py           # small GSM8K subset
│   │
│   └── utils/
│       ├── config_hash.py              # deterministic config hashing
│       ├── wandb_helpers.py            # group / tag conventions
│       ├── checkpoint_convert.py       # Megatron ↔ safetensors
│       ├── git_meta.py                 # git_sha, megatron_sha, dirty-detection
│       └── ladder_math.py              # tokens_per_param → token count, etc.
│
├── configs/
│   ├── base/
│   │   ├── family/
│   │   │   ├── llama3.yaml              # family-level defaults (norm, rope, tokenizer)
│   │   │   ├── qwen3.yaml
│   │   │   └── minicpm.yaml
│   │   └── scale/
│   │       ├── 300m.yaml                # dimensional only (layers, hidden, ffn, heads)
│   │       ├── 600m.yaml
│   │       ├── 1_2b.yaml
│   │       ├── 2_4b.yaml
│   │       └── 7b.yaml
│   │
│   ├── experiments/
│   │   ├── _template.yaml              # required fields reference
│   │   ├── champion.yaml               # current baseline (updated at promotion)
│   │   ├── optim/
│   │   │   ├── muon_hybrid.yaml
│   │   │   └── newton_schulz_v1.yaml
│   │   ├── attention/
│   │   ├── precision/
│   │   └── moe/
│   │
│   ├── clusters/
│   │   ├── h800_cn.yaml
│   │   ├── h100_de.yaml
│   │   ├── a100_de.yaml
│   │   ├── b200_de.yaml
│   │   └── hpc_de.yaml
│   │
│   ├── training_regime/
│   │   ├── ablation_20x.yaml
│   │   ├── ablation_40x.yaml           # dev mode (for 600M)
│   │   ├── final_200x.yaml
│   │   ├── final_400x.yaml
│   │   └── final_wsd_decay_only.yaml   # re-uses stable-stage checkpoint
│   │
│   └── launch/
│       ├── monthly_sweep.yaml          # ladder × variants × seeds spec
│       ├── weekly_gate.yaml
│       └── promotion_gate.yaml
│
├── launchers/
│   ├── submit.py                       # top-level Hydra entry point
│   ├── sweep.py                        # orchestrates ladder sweeps
│   ├── slurm/
│   │   ├── h800_cn.sbatch.j2
│   │   ├── hpc_de.sbatch.j2
│   │   └── bid_cluster.sbatch.j2       # with preemption handling
│   └── env/
│       ├── h800_cn.env                 # TE version, CUDA version
│       └── hpc_de.env
│
├── tools/
│   ├── reproduce.py                    # replay a run by W&B id
│   ├── promote.py                      # ablation → main gate
│   ├── validate_ladder.py              # end-of-month audit
│   ├── sync_wandb.py                   # offline mode sync helper
│   ├── freeze_dataset.py               # pre-tokenize + emit manifest
│   ├── ladder_plot.py                  # slope plot for one experiment across scales
│   ├── monthly_table.py                # aggregate monthly runs to CSV / DataFrame
│   ├── gen_monthly_report.py           # builds W&B report from ablation projects
│   ├── config_diff.py                  # human-readable diff between two configs
│   └── archive_resolved_configs.py     # writes runs/<hash>/resolved_config.yaml
│
├── runs/                               # append-only archive of resolved configs
│   └── <config_hash>/
│       ├── resolved_config.yaml        # written at launch
│       ├── launch_metadata.json
│       └── README.md                   # auto-generated summary
│
├── tests/
│   ├── unit/
│   │   ├── test_config_hash.py         # hash determinism
│   │   ├── test_capability_check.py
│   │   ├── test_wsd_scheduler.py
│   │   └── test_ladder_math.py
│   ├── integration/
│   │   └── test_smoke_runs.py          # 10-step runs at each scale
│   └── numerics/
│       └── test_patch_neutrality.py    # patches don't change baseline numerics
│
├── docs/
│   ├── onboarding.md
│   ├── ladder_config.md                # frozen batch/seq/tokenizer
│   ├── promotion_protocol.md
│   ├── wandb_conventions.md
│   ├── patches_cookbook.md
│   ├── tokenizer_policy.md             # why one tokenizer, how it's enforced
│   ├── megatron_pin.md                 # current pinned SHA and bump history
│   └── experiments/                    # lab notebook: one file per experiment
│       ├── _template.md
│       ├── champion_history.md         # versioned log of champion promotions
│       ├── muon_hybrid.md
│       ├── latent_attention_v1.md
│       └── fp4_native.md
│
├── pyproject.toml                      # poetry / uv managed
├── .pre-commit-config.yaml
└── .github/workflows/
    └── ci.yaml                          # smoke runs + unit tests per PR
```

---

## 4. Third-party pinning

### 4.1 Megatron-LM

Included as a git submodule under `third_party/Megatron-LM`, pinned to a specific commit SHA. **Never edited.** When upstream changes require adoption, the procedure is:

1. Bump the submodule SHA on a dedicated branch.
2. Re-apply all patches; verify each patch still targets a valid function.
3. Run the full `test_patch_neutrality.py` suite.
4. Run the champion config at 1.2B and 2.4B; verify loss curves match prior champion within seed-variance bounds.
5. Merge the bump to `main` only after the champion reruns pass.

The current pinned SHA and the rationale for the last bump live in `docs/megatron_pin.md` (implementors: please create this).

### 4.2 TransformerEngine

Pinned **per cluster config**, because cluster environments differ. Example in `configs/clusters/h800_cn.yaml`:

```yaml
cluster:
  transformer_engine_version: "1.12.0"
```

At job start, the launcher verifies the installed TE matches the pinned version; mismatch aborts the job. TE bumps have silently changed FP8 numerics in the past; this is not optional.

### 4.3 CUDA / PyTorch

Same rule as TE: pinned per cluster, verified at job start.

---

## 5. Core subsystems

### 5.1 Config composition (Hydra + OmegaConf)

A resolved config is built from six composable axes:

```bash
python -m launchers.submit \
    base/family=qwen3 \
    base/scale=1_2b \
    experiment=optim/muon_hybrid \
    training_regime=ablation_20x \
    cluster=h800_cn \
    seed=42
```

The base architecture splits into two independent axes: **family** (the reference lineage — Llama-3, Qwen-3, MiniCPM) and **scale** (the dimensional choices to hit a target non-embedding param count). Family composes first; scale overrides on top.

**`base/family/<family>.yaml`** — family-specific architectural defaults that do not depend on scale: norm type and epsilon, activation, rotary base, QK-norm, attention/hidden dropout, initialization method, and nominal tokenizer. These are the inherited design choices from the reference architecture.

Example `base/family/qwen3.yaml`:
```yaml
# @package _global_
base:
  family: qwen3
  family_version: "3.0"
  reference: "Qwen Team 2025"
  model:
    normalization: "RMSNorm"
    norm_epsilon: 1.0e-6
    activation: "SwiGLU"
    positional_encoding: "rope"
    rotary_base: 1000000
    rotary_scaling: null
    qk_norm: true
    attention_dropout: 0.0
    hidden_dropout: 0.0
    init_method_std: 0.02
    depth_scaled_init: false
  tokenizer:
    nominal_name: "qwen3"                # descriptive only; actual tokenizer
    nominal_vocab_size: 151936           # is frozen per dataset manifest (see 1.3)
```

Example `base/family/llama3.yaml`:
```yaml
# @package _global_
base:
  family: llama3
  family_version: "3.1"
  reference: "Dubey et al. 2024 (arXiv:2407.21783)"
  model:
    normalization: "RMSNorm"
    norm_epsilon: 1.0e-5
    activation: "SwiGLU"
    positional_encoding: "rope"
    rotary_base: 500000
    rotary_scaling: null
    qk_norm: false
    attention_dropout: 0.0
    hidden_dropout: 0.0
    init_method_std: 0.02
    depth_scaled_init: false
  tokenizer:
    nominal_name: "llama3"
    nominal_vocab_size: 128256
```

**`base/scale/<scale>.yaml`** — scale-specific dimensional choices only: `num_layers`, `hidden_size`, `ffn_hidden_size`, `num_attention_heads`, `num_query_groups`, `head_dim`, `seq_length`, and the scale-conditional architectural choice of embedding tying. Tuned to hit the target **non-embedding** parameter count.

Example `base/scale/1_2b.yaml` (following MiniCPM-1.2B's deep-and-thin convention):
```yaml
# @package _global_
base:
  scale: "1_2b"
  non_embedding_params: 1_200_000_000    # primary target; ladder math uses this

  model:
    # Deep-and-thin: follows MiniCPM-1.2B (Hu et al. 2024)
    num_layers: 52
    hidden_size: 1536
    ffn_hidden_size: 3840                # ~2.5× hidden (SwiGLU convention)
    num_attention_heads: 24
    num_query_groups: 8                  # GQA 3:1
    head_dim: 64
    seq_length: 4096

    # Scale-conditional (tie at small scales, untie at 2.4B+)
    tie_embeddings: true
```

Example `base/scale/2_4b.yaml`:
```yaml
# @package _global_
base:
  scale: "2_4b"
  non_embedding_params: 2_400_000_000

  model:
    num_layers: 40
    hidden_size: 2304
    ffn_hidden_size: 5760
    num_attention_heads: 36
    num_query_groups: 36                 # MHA at this scale (MiniCPM choice)
    head_dim: 64
    seq_length: 4096

    tie_embeddings: false                # untie at 2.4B+
```

#### 5.1.1 Parameter counting and family comparison

Since the tokenizer is frozen globally (Section 1.3), the actual vocab size used at training time is `dataset_manifest.vocab_size` regardless of `base.family.tokenizer.nominal_vocab_size`. At resolution time, the launcher computes:

```python
vocab = dataset_manifest.vocab_size
hidden = cfg.base.model.hidden_size
emb_params = vocab * hidden
lm_head_params = 0 if cfg.base.model.tie_embeddings else emb_params
cfg._derived.embedding_params = emb_params
cfg._derived.lm_head_params = lm_head_params
cfg._derived.total_params = cfg.base.non_embedding_params + emb_params + lm_head_params
```

`total_params` is logged to W&B for reference but does NOT drive ladder math. `training.tokens_per_param * base.non_embedding_params` determines the token count.

**Embedding and LM head are shape-fixed but not weight-fixed.** Fixing the tokenizer fixes `vocab_size`, which together with per-scale `hidden_size` fixes the dimensions of both layers across all experiments at that scale. But the layers still train from scratch — they have gradients, they're updated by the optimizer, they learn their values during training. "Fixed" means shape-fixed, not frozen.

**Cross-family scaling analysis.** The scaling-slope plot must be filtered to a single family. A 600M→2.4B slope mixing Llama-3 and Qwen-3 points conflates the method effect with family-dependent base-architecture effects. The "champion" config is implicitly per-family; if multiple families are actively researched, the team maintains parallel champions (`champion_qwen3.yaml`, `champion_llama3.yaml`). At seed stage we recommend committing to a single family to concentrate effort.

#### 5.1.2 Experiment, training_regime, cluster

**`experiments/<family>/<variant>.yaml`** — *only* the method's overrides. Must declare `family`, `description`, `required_capabilities`, and `patches`. The `family` field here refers to the *ablation family* (optim / attention / precision / etc.), not the base architecture family.

Example `experiments/optim/muon_hybrid.yaml`:
```yaml
# @package _global_
experiment:
  name: muon_hybrid
  family: optim
  description: |
    Muon on linear weights, Adam on embeddings, norms, biases, and LM head.
    Separate LR schedules per group. Hypothesis: orthogonal Newton-Schulz
    update is beneficial for dense weight matrices but unstable for 1D params.
  references:
    - "Jordan 2024 Muon blog post"
    - "Moonlight paper arXiv:2502.16982"
  patches: []
  required_capabilities: []

optim:
  type: muon_hybrid
  muon:
    apply_to: linear_weights
    ns_steps: 5
    ns_coeffs: [3.4445, -4.7750, 2.0315]
    lr: 2.0e-3
  adam:
    apply_to: [embedding, norm, bias, lm_head]
    lr: 1.0e-3
    betas: [0.9, 0.95]
    eps: 1.0e-8
```

The `description` field is **required**. CI refuses PRs introducing experiment YAMLs with missing or empty descriptions.

**`training_regime/<regime>.yaml`** — token budget rule, schedule, batch, checkpointing.

Example `training_regime/ablation_20x.yaml`:
```yaml
# @package _global_
training:
  tokens_per_param: 20
  # Total tokens computed at launch: tokens_per_param * base.non_embedding_params
  global_batch_size_tokens: 4_194_304   # 4M, FROZEN across ladder
  seq_length: 4096                       # FROZEN across ladder
  micro_batch_size: null                 # derived from cluster

scheduler:
  type: wsd
  warmup_tokens: 2_000_000_000
  stable_fraction: 0.8
  decay_fraction: 0.2
  peak_lr: 0.01
  min_lr_ratio: 0.1
  decay_shape: "linear"

checkpointing:
  save_every_tokens: 2_000_000_000
  keep_last: 3
  save_stable_stage_final: true          # for WSD checkpoint reuse
```

**`clusters/<cluster>.yaml`** — capabilities, parallelism rules, precision realization, env pins.

Example `clusters/h800_cn.yaml`:
```yaml
# @package _global_
cluster:
  name: h800_cn
  site: china
  nodes: 6
  gpus_per_node: 8
  interconnect: "nvlink_within_pcie_across"
  capabilities: [bf16, fp16, fp8]
  transformer_engine_version: "1.12.0"
  cuda_version: "12.4"
  pytorch_version: "2.4.0"
  wandb_offline: true                    # required; sync via tools/sync_wandb.py
  slurm_partition: "h800"
  slurm_account: "research"

parallelism:
  tp_size_rules:
    - {model_params_lt: 3.0e9, tp: 1, pp: 1}
    - {model_params_lt: 1.0e10, tp: 4, pp: 1}
    - {model_params_lt: 3.0e10, tp: 8, pp: 1}
  sequence_parallel: true
  distributed_optimizer: true

precision:
  default: fp8
  fp8_recipe: "delayed_scaling"
  bf16_fallback: true
```

**`seed`** — a single integer. Part of the run name (Section 5.3); excluded from the experiment definition so it can vary across replicas.

### 5.2 Resolution pipeline

On `submit.py` invocation:

1. **Hydra composes** the six axes (family, scale, experiment, training_regime, cluster, seed) into one resolved `DictConfig`. Family composes first, then scale overrides on top; both are under `base.*`.
2. **Parallelism derivation**: `parallelism.tp`, `parallelism.pp`, `parallelism.dp` computed from `base.non_embedding_params` and `cluster.*` rules.
3. **Embedding math**: `_derived.embedding_params`, `_derived.lm_head_params`, `_derived.total_params` computed from `dataset_manifest.vocab_size`, `base.model.hidden_size`, and `base.model.tie_embeddings`.
4. **Token count derivation**: `training.total_tokens = training.tokens_per_param * base.non_embedding_params`.
5. **Capability check**: `set(experiment.required_capabilities) ⊆ set(cluster.capabilities)`. On mismatch, abort with a clear error listing the missing capabilities.
6. **TE/CUDA version check**: verify installed versions match `cluster.*_version` pins. Mismatch aborts.
7. **Dataset manifest check**: verify dataset SHA256 manifest matches `data.expected_manifest_hash`. Mismatch aborts.
8. **Patch application**: `apply_patches(experiment.patches)` returns `patch_set_hash`.
9. **Config diff from champion**: compute `config_diff_from_champion` — a compact human-readable string describing what differs from `configs/experiments/champion.yaml` at the same base/family and scale. (See 5.3.1.)
10. **Run identity**: compute `run_name` / `run_dir` — a readable, timestamped name (see 5.3).
11. **Metadata injection**: `git_sha`, `megatron_sha`, `patch_set_hash`, `dataset_hash`, `run_name`, `config_diff_from_champion`, `launch_timestamp_utc` are written into `cfg._derived`.
12. **Resolved-config archive**: write `runs/<run_name>/resolved_config.yaml` and `launch_metadata.json`. Each launch gets its own fresh directory, so this is a plain write.
13. **W&B init**: project, tags, and config snapshot. (Grouping by config identity is no longer automatic — see 5.3.)
14. **SLURM submission**: render the SBATCH template, submit, record job id in W&B config.

### 5.3 Run identity

Config hashing has been removed — it added complexity without payoff for the
local-run workflow we actually use. Runs are identified by a readable,
timestamped name instead of a content hash:

```
<experiment>-<family>-<scale>-s<seed>-<UTC>
e.g. champion-llama3-1_2b-s42-20260524T200332Z
```

It is computed in `resolve_config` and stored as `cfg._derived.run_name` /
`cfg._derived.run_dir`. Every launch gets its own fresh `runs/<run_name>/`
directory, so the resolved-config archive is a plain write (no collision check).

Reproducibility no longer rests on a hash: each run archives its full
`resolved_config.yaml` alongside `git_sha`, `megatron_sha`, `patch_set_hash`,
`dataset_hash`, and `seed` in `launch_metadata.json`. To reproduce a run, replay
from its archived `resolved_config.yaml`.

**Consequences (these supersede earlier sections).** Removing the hash drops
three properties the spec previously assumed:
- No content-addressed deduplication of identical configs.
- Seeds of one config no longer share a run directory or an automatic W&B group
  (each launch is its own dir). Principle 7 (§2) and §8.1's `group = config_hash`
  are not currently provided; if seed-grouping is wanted again it needs an
  explicit group key (e.g. the run name minus seed+timestamp).
- No checkpoint resume keyed on config identity (§7.2, §7.3): a restart with the
  same config produces a *new* `run_dir`, so it will not auto-find a prior
  checkpoint.

`config_diff_from_champion` (§5.3.1) is unaffected — it is computed directly from
the resolved config, not from the hash.

### 5.3.1 Config diff from champion

Raw config hashes are useful for deduplication but opaque to humans. The derived field `config_diff_from_champion` provides a compact string describing what's different from the current champion at the same `base.family` and `base.scale`.

```python
# src/utils/config_diff.py

def diff_from_champion(
    resolved_cfg: DictConfig,
    champion_cfg: DictConfig,
    excluded_paths: set[str] = DEFAULT_EXCLUDED,
) -> str:
    """
    Return a compact diff string like:
      'optim.type=muon_hybrid, optim.muon.lr=2e-3'

    Returns 'champion' if the config matches the champion exactly
    (excluding volatile fields).
    """
    champion = OmegaConf.to_container(champion_cfg, resolve=True)
    current = OmegaConf.to_container(resolved_cfg, resolve=True)
    diffs = []
    for path, value in _walk(current):
        if any(path.startswith(ex) for ex in excluded_paths):
            continue
        champion_value = _get_path(champion, path, default=_MISSING)
        if champion_value != value:
            diffs.append(f"{path}={_compact_repr(value)}")
    return ", ".join(sorted(diffs)) or "champion"
```

This string is logged as a W&B config field on every run. In the W&B UI, adding `config_diff_from_champion` as a visible column in the runs table instantly shows what each run actually changed relative to baseline — instead of scrolling through hundreds of config fields, the researcher sees:

```
run_id      config_diff_from_champion                    eval/hellaswag@20B
alice-1     optim.type=muon_hybrid                       0.432
alice-2     optim.type=muon_hybrid, optim.muon.lr=3e-3   0.438
bob-1       attention.type=latent                        0.429
champion    champion                                     0.415
```

This is the primary view for daily comparison. The raw `config_hash` is for deduplication; `config_diff_from_champion` is for reading.

### 5.4 Capability tagging

Capabilities are string tags in `src/precision/capability.py`:

```python
CAPABILITIES = {
    "bf16", "fp16",
    "fp8",          # H100, H800, B200
    "fp4",          # B200 only
    "nvlink",       # assumes NVLink; matters for TP>1
    "ib_fast",      # fast InfiniBand (matters for DP perf but not numerics)
}
```

An experiment declares `required_capabilities`. A cluster declares `capabilities`. The launcher refuses to submit when the experiment's requirements are not a subset of the cluster's capabilities.

Examples:
- FP4 native training experiment → `required_capabilities: [fp4]` → refuses to submit on h800_cn or a100_de.
- Standard BF16 architecture ablation → `required_capabilities: []` → runs anywhere.
- FP8 precision-comparison experiment → `required_capabilities: [fp8]` → refuses on a100_de.

### 5.5 Megatron Core integration

We use Megatron Core (`mcore`), not the legacy Megatron path. The integration points:

**`src/specs/gpt_layer_specs.py`** — a function `build_spec(experiment_cfg) -> TransformerLayerSpec`. It reads `experiment.attention.type`, `experiment.mlp.type`, `experiment.norm.type`, and constructs the ModuleSpec tree accordingly. Every architecture variant is reachable through this function.

```python
def build_spec(cfg: DictConfig) -> TransformerLayerSpec:
    attn_cls = _ATTENTION_REGISTRY[cfg.experiment.attention.type]
    mlp_cls = _MLP_REGISTRY[cfg.experiment.mlp.type]
    norm_cls = _NORM_REGISTRY[cfg.experiment.norm.type]

    return TransformerLayerSpec(
        self_attention=ModuleSpec(module=attn_cls, params=cfg.experiment.attention),
        mlp=ModuleSpec(module=mlp_cls, params=cfg.experiment.mlp),
        input_norm=ModuleSpec(module=norm_cls, params=cfg.experiment.norm),
        ...
    )
```

**`src/optim/__init__.py`** — a factory:

```python
def get_optimizer(cfg: DictConfig, params, mcore_cfg) -> MegatronOptimizer:
    """Wraps Megatron's get_megatron_optimizer or our custom implementations."""
    if cfg.optim.type == "adamw":
        return get_megatron_optimizer(mcore_cfg, params)
    elif cfg.optim.type == "muon_hybrid":
        return MuonHybridOptimizer(cfg.optim, params)
    ...
```

Custom optimizers implement the `MegatronOptimizer` interface: `step()`, `zero_grad()`, `state_dict()`, `load_state_dict()`, distributed-checkpointing hooks, `reload_model_params()`.

**`src/patches/`** — escape hatch for things ModuleSpec cannot reach. Each patch is a small monkey-patch with a required docstring citing the upstream function and Megatron SHA:

```python
# src/patches/parallel_layers.py
"""
PATCH: parallel_layers
Modifies: megatron.core.transformer.transformer_block.TransformerBlock.forward
Upstream SHA ref: abc123def456 (line ~180)
Rationale: Adds parallel-attention+MLP residual support (PaLM-style).
Required by: experiments tagged family:parallel_layers
"""
from src.patches._registry import register_patch

@register_patch(name="parallel_layers")
def apply():
    import megatron.core.transformer.transformer_block as tb
    tb.TransformerBlock.forward = _patched_forward

def _patched_forward(self, hidden_states, ...):
    ...
```

`apply_patches(names)` applies each registered patch, records `(name, patch_sha)` pairs, and returns `patch_set_hash = blake2s(sorted_patches)`. The registry refuses to apply two patches that target the same upstream function, failing at registration time rather than silently.

---

## 6. Data pipeline

### 6.1 Dataset freeze

The pretraining dataset is tokenized once, to Megatron `.bin`+`.idx` format. The freeze process (`tools/freeze_dataset.py`) produces:

- Pre-tokenized shards.
- A manifest JSON: `{shard_name: sha256, tokenizer_hash: ..., total_tokens: ..., version: "v1"}`.
- A top-level `dataset_hash` = blake2s(manifest_json).

The manifest is copied to every cluster. At job start, the launcher verifies every shard's SHA256 against the manifest. This is ~2 minutes for a 1T-token dataset; do it anyway.

### 6.2 Loader determinism

The loader must be deterministic given `(dataset_hash, seed, global_batch_size_tokens, tokens_consumed)`. Specifically:

- Sample ordering is a deterministic function of `seed`.
- Resumption from a checkpoint skips exactly `tokens_consumed` tokens.
- Packing logic (if any) is deterministic per-sample.

Megatron's indexed dataset gives most of this; verify resumption determinism with a unit test: run 1000 steps, checkpoint, resume, compare loss trajectory against a continuous 2000-step run. Must match to <1e-5 relative.

### 6.3 Cross-site distribution

The dataset is shipped once per version (China ↔ Germany is slow; a fixed dataset means paying this cost once). Subsequent cluster additions replicate from whichever site has it locally.

---

## 7. Training loop

### 7.1 WSD scheduler and checkpoint reuse

The Warmup-Stable-Decay scheduler (MiniCPM's, Hu et al. 2024) partitions training into three phases:

1. **Warmup** (typically 2B tokens): LR linearly increases from 0 to `peak_lr`.
2. **Stable** (typically 80% of remaining tokens): LR held at `peak_lr`.
3. **Decay** (remaining 20%): LR decays linearly from `peak_lr` to `min_lr_ratio * peak_lr`.

`checkpointing.save_stable_stage_final: true` saves a special checkpoint at the end of the stable stage. This enables a key efficiency:

**Decay-only ablations.** Instead of re-running the full training for a decay-stage variant (different data mix, different decay shape), resume from the stable-stage checkpoint and only re-run the decay phase. This is `training_regime/final_wsd_decay_only.yaml`.

This does NOT apply to optimizer or architecture ablations — those affect the stable stage and require full reruns.

### 7.2 Preemption handling

For the bid cluster (`h100_de`, `a100_de`, `b200_de`), the SBATCH template sets `--requeue` and installs a SIGTERM trap that:

1. Triggers a final checkpoint save.
2. Flushes W&B metric buffer.
3. Writes a `PREEMPTED` marker file.

The training loop, on (re)start, checks for an existing checkpoint matching the current `config_hash + seed` and resumes. Every 30 minutes, an async checkpoint is written (non-blocking).

### 7.3 Elastic allocation

The bid cluster returns a variable node count. The launcher accepts a node range (`--nodes 2-8`) and at job start recomputes parallelism from the achieved allocation. The **achieved** parallelism config — not the requested one — is logged to W&B.

This is allowed because parallelism is excluded from `config_hash` (Section 5.3). Two runs of the same config with different TP/DP configurations are expected to produce the same curve up to numerical noise. If they don't, it's a numerics bug worth investigating.

### 7.4 Token-milestone evaluation

At pre-declared token counts (configurable, default: 1B, 2B, 5B, 10B, 20B, 50B, 100B, every 50B thereafter), the training loop pauses, runs the eval harness, and logs:

- `eval/loss`
- `eval/hellaswag_acc`, `eval/piqa_acc`, `eval/arc_e_acc`
- `eval/code_probe`, `eval/math_probe`

The harness is **the same binary across all runs**. Eval implementations are in `src/eval/tasks/` and are hash-stable (their source SHA is logged in W&B).

---

## 8. W&B infrastructure

### 8.1 Hosting

**Self-hosted W&B server** on a small VM in Germany. Reasons:
- Consumer W&B is slow/unreliable from China.
- HPC compute nodes have no internet; offline mode is mandatory there.
- If everyone is on offline mode anyway, self-hosting costs us nothing in ergonomics and gains us reliability.

All training jobs run with `WANDB_MODE=offline`. A post-training hook or login-node cron runs `wandb sync` to push to the self-hosted server.

### 8.2 Entity and projects

One entity: `<company>-research`. Projects organized by *purpose*, not by person:

| Project | Purpose | Who writes | Who reads |
|---|---|---|---|
| `pretrain-ablations-300m` | 300M ablation runs (optional) | Everyone | Everyone |
| `pretrain-ablations-600m` | 600M ablation runs | Everyone | Everyone |
| `pretrain-ablations-1_2b` | 1.2B ablation + scale-check | Everyone | Everyone |
| `pretrain-ablations-2_4b` | 2.4B promotion gate runs | Everyone | Everyone |
| `pretrain-ablations-7b` | HPC anchor runs | Lead | Everyone |
| `pretrain-champion` | Current baseline, rerun at promotion | Lead | Everyone |
| `pretrain-final-1_2b` | Final overtrained 1.2B | Lead | Everyone |
| `pretrain-final-2_4b` | Final overtrained 2.4B | Lead | Everyone |
| `sandbox-<username>` | Personal debug / scratch | One person | Everyone |

**The sandbox projects are mandatory.** Debug runs do not land in shared projects. Runs enter `pretrain-ablations-*` only when they are promotion-candidate-clean (no crashes, evals fire, tags populated). This is enforced socially — the launcher checks `status` and emits a warning if a run with `job_type=sandbox` is being submitted to a shared project.

### 8.3 Run conventions

Every run has:

**`group`** = `config_hash` (16 hex chars). This is load-bearing. Seeds of the same config share a group; W&B's group-aggregation (mean ± std across seeds) is the primary curve.

**`job_type`** ∈ `{ablation, promotion_gate, extrapolation, final, champion_baseline, sandbox}`.

**Tags** (list):
- `person:<username>`
- `family:<optim | attention | mlp | norm | precision | moe | data>` — the ablation family
- `base_family:<llama3 | qwen3 | minicpm | ...>` — the base architecture family
- `scale:<300m | 600m | 1_2b | 2_4b | 7b>`
- `cluster:<h800_cn | h100_de | a100_de | b200_de | hpc_de>`
- `precision:<bf16 | fp8 | fp4>`
- `status:<candidate | promoted | deprecated>`
- `regime:<ablation_20x | ablation_40x | final_200x | ...>`
- `month:<YYYY-MM>` — for monthly aggregation filtering

**Config fields** (logged to W&B's config):
- `git_sha`
- `megatron_sha`
- `patch_set_hash`
- `dataset_hash`
- `config_hash`
- `config_diff_from_champion` — **the primary column for daily comparison**
- `experiment_summary` — one-line description copied from `experiment.description`
- `required_capabilities`
- `achieved_parallelism` (TP, PP, DP actually used)
- `non_embedding_params`, `total_params` (derived)
- `launch_timestamp_utc`

**Metric prefixes:**
- `train/*` — per-step training metrics
- `eval/*` — token-milestone evaluation; format: `eval/<task>@<tokens>` (e.g., `eval/hellaswag@20B`)
- `perf/*` — throughput (tokens/sec/GPU, MFU)
- `system/*` — GPU memory, utilization (optional)

### 8.4 Aggregation tooling

W&B's default UI is oriented around single-run inspection. The workflows we actually need — ladder slopes, seed aggregation, monthly cross-experiment comparison — are better served by scripted tooling on top of the W&B API. Three scripts make up the core aggregation layer.

**`tools/ladder_plot.py`** — slope analysis for a single experiment across scales.

```python
# Usage:
#   python -m tools.ladder_plot experiment=muon_hybrid \
#       metric=eval/hellaswag@20B \
#       base_family=qwen3

def plot_ladder(experiment_name: str, metric: str, base_family: str):
    """
    Pull all runs for <experiment_name> across scales, group by
    (scale, config_hash), average across seeds, plot scale vs metric
    with log-log axes and fitted power law. Champion's slope overlaid
    as reference line. Returns Plotly figure, also saves PNG for
    embedding in monthly report.
    """
```

Output: a slope plot with mean ± std per scale, fitted power law, and the champion reference. This is the primary visualization for "does this method scale."

**`tools/monthly_table.py`** — tabular aggregation across all experiments in a month.

```python
# Usage:
#   python -m tools.monthly_table month=2026-03 project=pretrain-ablations-1_2b

def build_monthly_table(month: str, project: str) -> pd.DataFrame:
    """
    Pull all runs tagged month:<month> from <project>, aggregate:
      - Group by (experiment.name, base_family, base_scale)
      - Average seeds; compute std and count
      - Join with experiment.description for human readability
      - Sort by primary metric delta vs champion
    Return DataFrame; also write CSV to runs/reports/<month>/.
    """
```

Output: a DataFrame with columns `[experiment, family, base_family, scale, seeds, metric_mean, metric_std, delta_vs_champion, config_diff, person]`. Rendered as markdown in the monthly report.

**`tools/gen_monthly_report.py`** — orchestrates the full monthly W&B report.

```python
# Usage:
#   python -m tools.gen_monthly_report month=2026-03
```

Sections produced (see 8.5 for full spec):
1. Champion baseline at 1.2B and 2.4B
2. Candidate runs table (from `monthly_table.py`)
3. Slope plots for top candidates (from `ladder_plot.py`)
4. 7B anchor comparison
5. Rejected candidates with notes pulled from `docs/experiments/*.md`

The report is a W&B report object (persistent, versioned, commentable), linked from the promotion PR.

### 8.5 Monthly review via W&B reports

One W&B report per month. Generated by `tools/gen_monthly_report.py`. Sections:

1. **Champion baseline.** Current `main` config at 1.2B and 2.4B, loss and eval curves.
2. **Candidate runs.** Filtered `status:candidate AND scale:1_2b`, grouped by `config_hash`, sorted by primary eval metric at the 20B-tokens milestone. Champion overlaid. Table uses `config_diff_from_champion` as the primary human-readable column.
3. **Slope analysis.** For top 3 candidates: 600M/1.2B/2.4B points on a log-log plot (from `tools/ladder_plot.py`), fitted power law, champion slope overlaid.
4. **7B anchor.** Previous month's 7B run vs. extrapolated champion trajectory.
5. **Rejected candidates.** Things that didn't promote, with notes pulled from `docs/experiments/<n>.md`. Institutional memory.

Promotion decisions are documented in the report; the winner becomes the new champion, the corresponding `docs/experiments/<n>.md` is updated with the promotion event, and `docs/experiments/champion_history.md` gets a new entry.

### 8.6 Experiment notes (`docs/experiments/`)

W&B captures metrics; YAMLs capture configs; but the *reasoning* — hypothesis, what worked, what didn't, why something was promoted or rejected — lives in a human-written markdown file per experiment.

One file per experiment name, using this template:

```markdown
# Experiment: <name>

**Family**: <optim | attention | mlp | ...>
**Status**: <exploratory | candidate | promoted | deprecated>
**Owner**: <username>
**Created**: <YYYY-MM-DD>

## Hypothesis
<2-4 sentences explaining what you think will happen and why.>

## Method summary
<What the experiment actually does, at a level a teammate could
re-implement from this description.>

## Timeline
- YYYY-MM-DD: <event, e.g., first 600M ablation, result>
- YYYY-MM-DD: <event>

## Runs
- Ablation ladder: [W&B group link]
- 2.4B confirmation: [W&B run link]
- 7B anchor: [W&B run link]

## What worked
<Bullet list of findings that held up.>

## What didn't
<Bullet list of tried-and-rejected variants, with one-sentence reasons.>

## Follow-ups
<Open questions, future variants to try.>

## References
<Links to papers, prior experiments.>
```

Every experiment YAML must have a corresponding `docs/experiments/<name>.md`. CI enforces this. The notes are updated by the experiment's owner over the life of the experiment — not just at promotion time.

`docs/experiments/champion_history.md` is an append-only log of champion promotions:

```markdown
# Champion history

## 2026-03 — champion-v3 (muon_hybrid)
- Promoted from: [PR link]
- Config hash: a3f8b2c1d4e5f6a7
- W&B report: [link]
- Primary gain: +0.8% on eval/hellaswag@20B at 2.4B vs champion-v2
- See: docs/experiments/muon_hybrid.md

## 2026-02 — champion-v2 (latent_attention_v1)
- ...
```

This is the "six months later" view: a new team member can read `champion_history.md` and understand the research trajectory without asking anyone.

---

## 9. Launcher (`launchers/submit.py`)

### 9.1 CLI

```bash
# Single run
python -m launchers.submit \
    base/family=qwen3 \
    base/scale=1_2b \
    experiment=optim/muon_hybrid \
    training_regime=ablation_20x \
    cluster=h800_cn \
    seed=42 \
    wandb.project=sandbox-alice

# Full ladder sweep
python -m launchers.sweep \
    base/family=qwen3 \
    experiment=optim/muon_hybrid \
    ladder=[600m,1_2b,2_4b] \
    seeds_per_scale=[3,2,1] \
    training_regime=ablation_20x \
    cluster=h800_cn \
    wandb.project=pretrain-ablations
```

The sweep invocation expands into N `submit.py` calls with proper job dependencies (small scales first, so early signal on failure).

### 9.2 Behavior

Pseudo-code:

```python
def submit(overrides: list[str]):
    cfg = hydra_compose(overrides)

    # 1. Derive: parallelism, token counts, embedding math
    cfg.parallelism = resolve_parallelism(cfg.base.non_embedding_params, cfg.cluster)
    cfg.training.total_tokens = (
        cfg.training.tokens_per_param * cfg.base.non_embedding_params
    )

    vocab = read_dataset_manifest(cfg.data.path).vocab_size
    hidden = cfg.base.model.hidden_size
    cfg._derived.embedding_params = vocab * hidden
    cfg._derived.lm_head_params = (
        0 if cfg.base.model.tie_embeddings else vocab * hidden
    )
    cfg._derived.total_params = (
        cfg.base.non_embedding_params
        + cfg._derived.embedding_params
        + cfg._derived.lm_head_params
    )

    # 2. Capability check
    missing = set(cfg.experiment.required_capabilities) - set(cfg.cluster.capabilities)
    if missing:
        raise CapabilityMismatch(f"Cluster {cfg.cluster.name} lacks: {missing}")

    # 3. Version pins
    assert_version_matches("transformer_engine", cfg.cluster.transformer_engine_version)
    assert_version_matches("torch", cfg.cluster.pytorch_version)

    # 4. Dataset manifest
    cfg._derived.dataset_hash = verify_dataset_manifest(cfg.data.path)

    # 5. Patches
    cfg._derived.patch_set_hash = apply_patches(cfg.experiment.patches)

    # 6. Git metadata (refuse if dirty and not --allow-dirty)
    cfg._derived.git_sha = git_sha(allow_dirty=cfg.wandb.project.startswith("sandbox"))
    cfg._derived.megatron_sha = submodule_sha("third_party/Megatron-LM")

    # 7. Config hash (excludes seed)
    cfg._derived.config_hash = config_hash(cfg)

    # 8. Config diff from champion (same base_family + base_scale)
    champion_cfg = load_champion_for(cfg.base.family, cfg.base.scale)
    cfg._derived.config_diff_from_champion = diff_from_champion(cfg, champion_cfg)
    cfg._derived.experiment_summary = _first_line(cfg.experiment.description)

    # 9. Archive resolved config (append-only)
    archive_dir = Path("runs") / cfg._derived.config_hash
    archive_dir.mkdir(parents=True, exist_ok=True)
    _write_if_new(archive_dir / "resolved_config.yaml", OmegaConf.to_yaml(cfg))
    _append_launch_metadata(archive_dir / "launch_metadata.json", cfg, seed=cfg.seed)

    # 10. W&B init
    wandb_cfg = {
        "project": cfg.wandb.project,
        "entity": cfg.wandb.entity,
        "group": cfg._derived.config_hash,
        "job_type": cfg.wandb.job_type,
        "tags": build_tags(cfg),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mode": "offline" if cfg.cluster.wandb_offline else "online",
    }

    # 9. Render SBATCH, submit
    sbatch = render_sbatch(cfg.cluster.slurm_template, cfg, wandb_cfg)
    job_id = submit_slurm(sbatch)

    # 11. Render SBATCH, submit
    sbatch = render_sbatch(cfg.cluster.slurm_template, cfg, wandb_cfg)
    job_id = submit_slurm(sbatch)

    # 12. Register queued run in W&B (before job starts)
    register_queued_run(wandb_cfg, job_id)

    return job_id
```

### 9.3 Guardrails

The launcher refuses to submit when:

- Required capabilities not satisfied by cluster.
- Git working tree is dirty and target project is not a sandbox.
- Dataset manifest does not match expected hash.
- TE/CUDA/PyTorch versions don't match cluster pin.
- `training_regime` specifies values that override frozen ladder fields (`global_batch_size_tokens`, `seq_length`) without an explicit `--override-ladder-config` flag.
- Target project is a shared ablation project but `job_type=sandbox`.
- `experiment.description` is empty or missing.
- `docs/experiments/<experiment.name>.md` does not exist.
- Champion config cannot be loaded for the requested `(base_family, base_scale)` pair (required for computing `config_diff_from_champion`).

Each guardrail emits a loud, specific error with a suggested remediation.

---

## 10. Testing

### 10.1 Unit tests

- `test_config_hash.py`: same config → same hash; different seeds → same hash; different cluster → same hash; different experiment → different hash.
- `test_capability_check.py`: capability subset logic is correct.
- `test_wsd_scheduler.py`: schedule shape, stable-stage boundary, decay decay.
- `test_ladder_math.py`: token count derivation across scales.

### 10.2 Integration (smoke runs)

`tests/integration/test_smoke_runs.py` launches a 10-step training run at each scale on a single GPU (with heavily reduced seq length / batch) to verify:
- Config composes.
- Model instantiates.
- Optimizer steps.
- Checkpointing round-trips.
- W&B logs at least one metric.

These run on every PR via CI.

### 10.3 Numerics tests

`tests/numerics/test_patch_neutrality.py`: the champion baseline, with `patches=[]` and with `patches=[<every individual patch>]` where the patch is not activated by the current config, must produce bit-identical (or within-1e-6) loss at 100 steps. Patches must be no-ops when their feature is not in use.

---

## 11. Promotion protocol

### 11.1 Branching

- `main` is the champion.
- Each researcher has a long-lived feature branch: `alice/muon`, `bob/latent-attn`.
- Feature branches rebase on `main` at promotion events.
- Feature branches never merge to `main` without a successful promotion.

### 11.2 Weekly cycle

Each person submits their current best candidate to the weekly 2.4B promotion gate. One person (rotating) reviews outputs, flags regressions, verifies tag hygiene.

### 11.3 Monthly cycle

1. `tools/gen_monthly_report.py` builds the monthly review.
2. Team reviews; winner is selected by primary metric improvement over champion at 2.4B, corroborated by ≥2σ agreement across seeds and a favorable 600M→1.2B→2.4B slope.
3. Winner's PR is code-reviewed by ≥1 teammate.
4. PR merged to `main`; new champion baseline runs kicked off at 1.2B and 2.4B.
5. 7B HPC run scheduled on the new champion for next month's anchor.

### 11.4 Promotion PR template

Every promotion PR includes:

- Config diff vs current champion.
- W&B report link showing the slope and 2.4B comparison.
- Smoke-run CI passing.
- `config_hash` before and after promotion.
- A one-paragraph description of what the method does and why it helps.

### 11.5 End-of-month audit (`tools/validate_ladder.py`)

Runs automatically; flags:
- Runs in shared projects without `patch_set_hash` populated.
- Runs without `capability_tags`.
- Runs whose `config_hash` appears only once in the month (i.e., no seed replication).
- Runs whose `dataset_hash` doesn't match the current dataset manifest.
- Runs with `git_dirty=true` in shared projects.

Flagged runs are deprecated (`status:deprecated` tag added); their numbers don't enter the monthly report.

---

## 12. Implementation order (recommended)

For an implementor starting from zero, the suggested ordering:

1. **Repo skeleton + Megatron submodule + CI smoke test.** Prove the environment builds and a 10-step run works.
2. **Config composition.** `base`, `training_regime`, `cluster` configs; Hydra composition; resolution pipeline up to parallelism derivation.
3. **Config hashing + W&B integration.** Get runs logging to a local/test W&B with correct groups and tags.
4. **WSD scheduler + dataset loader + checkpoint round-trip.** Enough to run a real short training.
5. **Baseline champion.** A working baseline config (plain Adam + baseline architecture) that trains cleanly at 300M and 600M.
6. **Eval harness.** Token-milestone evaluation with a minimal task set.
7. **Optimizer factory + one custom optimizer** (Muon hybrid is a good first test case).
8. **ModuleSpec integration.** A non-baseline attention variant.
9. **Patches registry + one patch.** Prove the escape hatch works.
10. **Launcher guardrails + capability tagging.**
11. **Sweep orchestration.** Ladder submission with seed handling.
12. **Monthly report generator.**
13. **Bid-cluster preemption handling.**

Each step produces a demonstrable artifact; none should exceed ~1 week for one engineer.

---

## 13. Non-goals

Things this spec explicitly does not cover:

- Post-training (SFT, RLHF, DPO). Scope is pretraining only.
- Long-context extension. That's a separate track with different frozen configuration.
- Multimodal. Text-only.
- Serving / inference infrastructure beyond checkpoint conversion to safetensors.
- Data curation. The dataset is fixed.
- Automatic hyperparameter search. Hyperparameters are set by `experiment` configs authored by humans; we don't run Bayesian HPO.

---

## 14. Glossary

- **Ablation**: a single experiment variant compared against the champion baseline.
- **Champion**: the current best config, living on `main`, serving as the reference line for all comparisons.
- **Config hash**: 16-char blake2s digest of the experiment-relevant config; identical hash ⟹ expected identical curve up to seed variance.
- **Ladder**: the set of model sizes used for scaling-slope analysis (300M/600M/1.2B/2.4B/7B).
- **ModuleSpec**: Megatron Core's dependency-injection mechanism for assembling transformer layers.
- **Patch**: a monkey-patch of upstream Megatron code for cases ModuleSpec cannot handle; must be registered and hashed.
- **Promotion**: the event of an ablation beating the champion and becoming the new baseline.
- **Regime**: a training configuration specifying token budget, schedule, and checkpointing (e.g., `ablation_20x`).
- **WSD**: Warmup-Stable-Decay learning rate scheduler (Hu et al., MiniCPM 2024).

---

## 15. References

- Hu, S. et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* arXiv:2404.06395.
- MiniCPM Team (2025). *MiniCPM4: Ultra-Efficient LLMs on End Devices.* arXiv:2506.07900.
- Biderman, S. et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML 2023.
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556.
- NVIDIA Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- NVIDIA TransformerEngine: https://github.com/NVIDIA/TransformerEngine
