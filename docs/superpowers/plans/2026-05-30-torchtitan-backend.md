# torchtitan Training Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second training backend, `backend=torchtitan`, that is **interface-identical from the slm-research side** — same six-axis yaml, same `scripts/train_*.sh`, same config structure — so flipping `--backend torchtitan` is the only change. Under the hood it drives torchtitan's **native** first-party models (`llama3`, `qwen3`, `deepseek_v3`) at slm's small scales with AdamW + bf16 + FSDP2. **Exact model-structure parity with Megatron is a non-goal** — the torchtitan model only has to qualify as the same family; the acceptance gate is "trains and converges healthily," not "matches the Megatron curve."

**Architecture:** Mirror the existing Megatron drive path. A new `--backend` flag on the `scripts/train_*.sh` wrappers injects `backend=<value>` into the resolved config and dispatches to a new `launchers/train_torchtitan.py`, which resolves the same 6-axis Hydra config, generates a torchtitan TOML + CLI overrides from `src/utils/torchtitan_args.py`, and runs `torchrun -m torchtitan.train`. The slm wiring (same-corpus dataloader, slm scale-flavors on torchtitan's native models, WSD scheduler) is injected through torchtitan's `experimental.custom_import` extension hook (`register_train_spec`), never by editing the vendored submodule. Megatron-only config/`patches` knobs that the native torchtitan model can't express are **warn-and-skipped**, so the same experiment yaml runs on either backend.

**Tech Stack:** Python 3.12, Hydra/OmegaConf, torchrun, PyTorch FSDP2/DTensor, torchtitan v0.2.2 (vendored submodule), pytest. Reuses Megatron-LM's `megatron.core.datasets` (already vendored) for the dataloader.

**Design spec:** `docs/superpowers/specs/2026-05-30-torchtitan-backend-design.md`

**Reference files (read before starting):**
- `launchers/train_megatron.py` — the launcher we mirror (`build_torchrun_command`, env/PYTHONPATH setup).
- `launchers/submit.py` — `_parse_overrides`, `resolve_config`, `archive_resolved_config`, `_append_launch_metadata`.
- `src/utils/megatron_args.py` — `build_megatron_args(cfg)`; the mapper we parallel.
- `src/utils/scheduler.py` — `scheduler_args(...)`; the WSD source of truth we must match.
- `src/utils/git_meta.py` — `submodule_sha(...)`.
- `tests/unit/test_train_megatron_command.py`, `tests/unit/test_megatron_args.py` — the test styles we mirror.

**Conventions:**
- Commits: single concise sentence, conventional-commit prefix `feat(torchtitan):` / `docs(torchtitan):` / `test(torchtitan):`. No AI attribution trailer.
- `pytest` unit tests are CPU-only and must pass in CI. The GPU parity run (Task 12) is **not** run in this harness — it is handed to the operator to run on a node and report back.
- The vendored `third_party/torchtitan` submodule is **never edited** (same rule as `third_party/Megatron-LM`, SPEC.md §4.1).

---

## File Structure

**Created:**
- `third_party/torchtitan` — git submodule, pinned to v0.2.2 (`73a0e6979dd10b6b1904098eb3c8f62c18ab87ce`).
- `docs/torchtitan_pin.md` — pin SHA + bump procedure (mirrors `docs/megatron_pin.md`).
- `docs/torchtitan_api_notes.md` — exact torchtitan v0.2.2 extension API recorded by the Task 4 discovery spike; later tasks cite it.
- `src/backends/__init__.py` — `BACKENDS` registry + `select_backend(name)` dispatch.
- `src/utils/torchtitan_args.py` — `build_torchtitan_config(cfg) -> (toml_dict, overrides)`; pure function.
- `src/titan_ext/__init__.py` — slm runtime extension package imported by `torchtitan.train` (registers the llama3 flavor, the Megatron-indexed dataloader, and the WSD scheduler) **without editing vendored code**.
- `src/titan_ext/model_flavor.py` — builds a torchtitan llama3 model-args "flavor" from the resolved slm config.
- `src/titan_ext/dataloader.py` — `slm_megatron_indexed` dataloader wrapping `megatron.core.datasets.GPTDataset`.
- `src/titan_ext/lr_scheduler.py` — WSD lambda matching `src/utils/scheduler.py`.
- `launchers/train_torchtitan.py` — parent launcher (mirror of `train_megatron.py`).
- `tests/unit/test_backend_field.py`, `tests/unit/test_backend_dispatch.py`, `tests/unit/test_torchtitan_args.py`, `tests/unit/test_torchtitan_scheduler.py`, `tests/unit/test_train_torchtitan_command.py`, `tests/unit/test_titan_run_name.py`
- `tests/integration/test_titan_megatron_data_parity.py` — M2 gate (CPU).
- `tests/numerics/test_titan_training_health.py` — M3 functional gate (`@pytest.mark.gpu`, operator-run).

**Modified:**
- `.gitmodules` — second `[submodule]` stanza.
- `configs/launch/config.yaml` — add top-level `backend: megatron`.
- `launchers/submit.py` — stamp `_derived.torchtitan_sha`; add `backend` segment to `run_name` when non-megatron; record `torchtitan_sha` in launch metadata.
- `scripts/train_adam.sh` — parse `--backend`, inject override, dispatch launcher.
- `install_slm_env.sh` / `pyproject.toml` — torchtitan runtime deps (`torchdata`, `tomli`/`tomli-w`, `tiktoken`, `blobfile`, `torchao`).

---

## Task 1: Vendor torchtitan as a pinned submodule

**Files:**
- Create: `third_party/torchtitan` (submodule)
- Modify: `.gitmodules`
- Create: `docs/torchtitan_pin.md`

- [ ] **Step 1: Add the submodule and pin to v0.2.2**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research
git submodule add https://github.com/pytorch/torchtitan.git third_party/torchtitan
cd third_party/torchtitan
git fetch --tags
git checkout 73a0e6979dd10b6b1904098eb3c8f62c18ab87ce   # tag v0.2.2
cd ../..
git submodule status third_party/torchtitan
```
Expected: `git submodule status` prints `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce third_party/torchtitan (v0.2.2)`.

- [ ] **Step 2: Verify the package imports from the checkout**

Run:
```bash
PYTHONPATH=third_party/torchtitan python -c "import torchtitan, torchtitan.train; print(torchtitan.__file__)"
```
Expected: prints a path under `third_party/torchtitan/torchtitan/__init__.py` (install the deps from Task 1 Step 4 first if it raises `ModuleNotFoundError` on a third-party package such as `torchdata`).

- [ ] **Step 3: Write `docs/torchtitan_pin.md`**

```markdown
# torchtitan pin and bump history

torchtitan is included under `third_party/torchtitan` as a submodule. Like
Megatron-LM, the pin is conservative: torchtitan's parallelize/quantization
APIs move fast and silently change numerics.

## Current pin

- Tag / SHA: `v0.2.2` -> `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce`
- Pinned on: 2026-05-30
- Reason: first import. `v0.2.2` is the newest release tag (all torchtitan
  releases are GitHub pre-releases; we treat the newest tag as "last stable"
  per the pin-to-a-tag contract used for Megatron-LM, SPEC.md §4.1).

To initialise on a fresh clone:

    git submodule update --init --recursive

## Bump procedure

1. Branch `torchtitan-bump-<date>`.
2. `cd third_party/torchtitan && git fetch --tags && git checkout <new-tag>`.
3. Re-run the discovery checks in `docs/torchtitan_api_notes.md` — if any
   recorded symbol (train-spec registration, JobConfig key, dataloader
   component API) moved, update `src/titan_ext/` and the notes.
4. `pytest tests/unit -k torchtitan` and `pytest tests/integration/test_titan_megatron_data_parity.py`.
5. Rerun the per-family training-health gate (Task 12); curves must stay healthy.
6. Update this doc (SHA, date, reason, any API changes).

## Bump log

| Date | New SHA | Prior SHA | Reason | API changes |
|---|---|---|---|---|
| 2026-05-30 | `v0.2.2` (`73a0e6979`) | — | first import | n/a |
```

- [ ] **Step 4: Add torchtitan runtime deps to the GPU extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add a `torchtitan` extra (do not touch the `gpu`/`data` extras):
```toml
torchtitan = [
    "torchdata>=0.8",
    "tomli>=2.0; python_version < '3.11'",
    "tomli-w>=1.0",
    "tiktoken>=0.7",
    "blobfile>=2.1",
    "torchao>=0.9",
]
```
Then mirror these into `install_slm_env.sh` next to where Megatron deps are installed (search the script for `Megatron` and add a sibling `uv pip install` line for the `torchtitan` extra). Record in `CHANGELOG.md` per repo policy.

- [ ] **Step 5: Commit**

```bash
git add .gitmodules third_party/torchtitan docs/torchtitan_pin.md pyproject.toml install_slm_env.sh
git commit -m "feat(torchtitan): vendor torchtitan v0.2.2 submodule + pin doc + runtime deps"
```

---

## Task 2: Add the `backend` config field and record `torchtitan_sha`

**Files:**
- Modify: `configs/launch/config.yaml`
- Modify: `launchers/submit.py` (resolve_config ~line 180; `_append_launch_metadata` ~line 95)
- Test: `tests/unit/test_backend_field.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_backend_field.py
"""The `backend` field selects the training backend and is recorded per run."""

from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config


def test_backend_defaults_to_megatron():
    cfg = _parse_overrides(["base/family=llama3", "experiment=optim/adam"])
    assert str(cfg.backend) == "megatron"


def test_backend_override_is_applied():
    cfg = _parse_overrides(
        ["base/family=llama3", "experiment=optim/adam", "backend=torchtitan"]
    )
    assert str(cfg.backend) == "torchtitan"


def test_resolve_stamps_torchtitan_sha():
    cfg = _parse_overrides(["base/family=llama3", "experiment=optim/adam"])
    resolve_config(cfg)
    # Present whether or not the submodule is checked out (falls back to a marker).
    assert "torchtitan_sha" in cfg._derived
    assert isinstance(str(cfg._derived.torchtitan_sha), str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_backend_field.py -v`
Expected: FAIL — `backend` key missing (ConfigAttributeError) and `torchtitan_sha` not stamped.

- [ ] **Step 3: Add the field to the base config**

In `configs/launch/config.yaml`, after the `seed: 42` line add:
```yaml
# Training backend. `megatron` (default) drives third_party/Megatron-LM via
# launchers.train_megatron; `torchtitan` drives third_party/torchtitan via
# launchers.train_torchtitan. Part of the resolved config so it is archived
# and shows up in the per-run name.
backend: megatron
```

- [ ] **Step 4: Stamp `torchtitan_sha` in `resolve_config`**

In `launchers/submit.py`, in the "Git metadata" block (right after the `megatron_sha` try/except, ~line 182), add:
```python
    try:
        cfg._derived.torchtitan_sha = submodule_sha("third_party/torchtitan", cwd=REPO_ROOT)
    except Exception:
        cfg._derived.torchtitan_sha = "unpinned"
```
(`submodule_sha` is already imported from `src.utils.git_meta`.)

- [ ] **Step 5: Record it in launch metadata**

In `_append_launch_metadata` (`launchers/submit.py` ~line 95), add to the `entry` dict next to `"megatron_sha"`:
```python
        "torchtitan_sha": cfg._derived.get("torchtitan_sha"),
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/unit/test_backend_field.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add configs/launch/config.yaml launchers/submit.py tests/unit/test_backend_field.py
git commit -m "feat(torchtitan): add backend config field and record torchtitan_sha per run"
```

---

## Task 3: Put the backend into the per-run name (torchtitan only)

**Files:**
- Modify: `launchers/submit.py` (`resolve_config`, run_name block ~line 200)
- Test: `tests/unit/test_titan_run_name.py`

Keeps Megatron run names byte-identical (so existing `test_run_dir_is_readable_name_with_timestamp` stays green) while disambiguating torchtitan runs in `runs/` and W&B.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_titan_run_name.py
from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config


def test_megatron_run_name_has_no_backend_segment():
    cfg = _parse_overrides(["base/family=llama3", "experiment=champion", "seed=42"])
    resolve_config(cfg)
    assert "-torchtitan-" not in cfg._derived.run_name
    assert cfg._derived.run_name.startswith(
        f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}-s42-"
    )


def test_torchtitan_run_name_has_backend_segment():
    cfg = _parse_overrides(
        ["base/family=llama3", "experiment=champion", "seed=42", "backend=torchtitan"]
    )
    resolve_config(cfg)
    assert cfg._derived.run_name.startswith(
        f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}-torchtitan-s42-"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_titan_run_name.py -v`
Expected: FAIL — torchtitan run_name has no `-torchtitan-` segment.

- [ ] **Step 3: Implement the conditional segment**

In `launchers/submit.py`, replace the `run_name = (...)` assignment in `resolve_config` with:
```python
    backend = str(cfg.get("backend", "megatron"))
    backend_seg = "" if backend == "megatron" else f"-{backend}"
    run_name = (
        f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}{backend_seg}"
        f"-s{cfg.seed}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_titan_run_name.py tests/unit/test_train_megatron_command.py -v`
Expected: PASS (new tests + the existing megatron run-name test still green).

- [ ] **Step 5: Commit**

```bash
git add launchers/submit.py tests/unit/test_titan_run_name.py
git commit -m "feat(torchtitan): tag torchtitan runs with a backend segment in the run name"
```

---

## Task 4: Discovery spike — record torchtitan v0.2.2 extension API

No production code; this task reads the **pinned** source and writes `docs/torchtitan_api_notes.md`, the single source later tasks cite for exact symbol names. This avoids guessing torchtitan internals (its model sizes are registry "flavors", not free-form TOML dims, so a custom 300m model must be **registered**, not configured).

> **Verified 2026-05-30 against the v0.2.2 source (these are confirmed; the spike just transcribes them into the notes doc):**
> - **`torchtitan.protocols.train_spec`**: `TrainSpec` is a dataclass with fields `model_cls`, `model_args: Mapping[str, BaseModelArgs]` (the flavor registry — flavor-name → args), `parallelize_fn`, `pipelining_fn`, `build_optimizers_fn`, `build_lr_schedulers_fn`, `build_dataloader_fn`, `build_tokenizer_fn`, `build_loss_fn`, `build_validator_fn`, `build_metrics_processor_fn`, `state_dict_adapter`.
> - `register_train_spec(name: str, train_spec: TrainSpec) -> None` — **takes two args and RAISES `ValueError` if `name` already registered** (no overwrite). `get_train_spec(name: str) -> TrainSpec`.
> - **`train.py`** selects the spec via `get_train_spec(job_config.model.name)`, and runs `importlib.import_module(job_config.experimental.custom_import)` **before** that lookup → our `--experimental.custom_import src.titan_ext` hook is real and correct.
> - Component build calls: `build_dataloader_fn(dp_world_size, dp_rank, tokenizer, job_config)`; `build_optimizers_fn(model_parts, job_config.optimizer, parallel_dims)`; `build_lr_schedulers_fn(optimizers, job_config.lr_scheduler, job_config.training.steps)`.
> - **`[model]` TOML carries ONLY `name`, `flavor`, asset paths — NOT dims.** Model dimensions come from the `model_args` flavor mapping. So `slm_<scale>` dims must be injected by `src.titan_ext` (which reads them from the resolved slm config via the `SLM_RESOLVED_CONFIG` env var the launcher exports).
> - **`[training]`** has BOTH `local_batch_size` and `global_batch_size`, plus `seq_len`, `steps`, `dataset`, `dataset_path`, `mixed_precision_param`, `max_norm`. **There is no `seed` field under `[training]`** — seed lives elsewhere (verify: likely `[training].seed` is absent; check `[job]`/top-level). Record where seed actually goes.
> - **`[optimizer]`**: `name`, `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `implementation`.
> - **`[lr_scheduler]`**: `warmup_steps`, `decay_ratio`, `decay_type`, `min_lr_factor` — **`decay_ratio` is a fraction of total steps, NOT a step count**, and there is **no `decay_steps`** key. Check whether the native scheduler already implements WSD (warmup → stable → decay over the final `decay_ratio`); if so, a custom lambda is unnecessary.
> - **`[parallelism]`**: `data_parallel_replicate_degree`, `data_parallel_shard_degree`, `tensor_parallel_degree`, `pipeline_parallel_degree`, `context_parallel_degree`, `expert_parallel_degree`.
> - **`[metrics]`**: `enable_wandb`, `enable_tensorboard`, `log_freq`, … — **no project/run-name field** → W&B project + run name come from `WANDB_PROJECT` / `WANDB_NAME` env (Task 11's approach is correct).
> - **`TransformerModelArgs`** (llama3) sizes the FFN via `ffn_dim_multiplier` + `multiple_of`, **not an explicit `ffn_hidden_size`** — Task 8 must solve for those to hit slm's exact FFN width. Confirm whether it exposes a `norm_eps`, `max_seq_len`, and any tied-embedding flag (stock llama3 is **untied**).
> - **`ParallelAwareDataloader(dataset, dp_rank, dp_world_size, **kwargs)`** — `batch_size`/`collate_fn` pass through kwargs to `StatefulDataLoader`.

**Files:**
- Create: `docs/torchtitan_api_notes.md`

- [ ] **Step 1: Locate the extension + config surfaces in the checkout**

Run (read-only exploration of the pinned tree):
```bash
cd /lustre/fast/fast/zqiu/slm-research/third_party/torchtitan
sed -n '1,200p' torchtitan/train.py | grep -nE 'def main|JobConfig|train_spec|build_|config_manager|custom'
ls torchtitan/protocols/        # train_spec.py lives here
sed -n '1,200p' torchtitan/protocols/train_spec.py
grep -rnE 'register_train_spec|get_train_spec|class TrainSpec' torchtitan/ | head
grep -rnE 'class JobConfig|class Model|class Training|class Parallelism|class Optimizer|class LRScheduler|class Metrics' torchtitan/config/ | head -40
ls torchtitan/models/llama3/        # model args / flavor registry
grep -rnE 'flavor|TransformerModelArgs|llama3_configs|register' torchtitan/models/llama3/*.py | head
sed -n '1,120p' torchtitan/components/dataloader.py
grep -rnE 'custom_import|custom_args_module|experimental' torchtitan/config/*.py | head
```

- [ ] **Step 2: Record findings in `docs/torchtitan_api_notes.md`**

Write the file with these sections, filled with the **exact** symbols/keys observed (no paraphrase):
```markdown
# torchtitan v0.2.2 extension API notes (pin 73a0e6979)

## 1. Entry & config
- Entry module + function: `python -m torchtitan.train` -> <main fn>
- JobConfig class + how TOML maps to it: <module path>
- Exact dotted keys we will emit, with types and defaults:
  - [model]: name=..., flavor=..., (and whether raw dims are accepted) ...
  - [training]: seq_len, global_batch_size, local_batch_size, steps, seed, mixed_precision_param, dataset, dataset_path ...
  - [optimizer]: name, lr, eps, beta1/beta2 (exact key names), weight_decay ...
  - [lr_scheduler]: warmup_steps, decay_ratio, decay_type, min_lr_factor (exact names) ...
  - [parallelism]: data_parallel_shard_degree, data_parallel_replicate_degree, tensor_parallel_degree, pipeline_parallel_degree, context_parallel_degree ...
  - [metrics]: enable_wandb, ... ; how project/run-name are set (config vs env WANDB_*) ...
  - [checkpoint]: enable, folder, interval ...

## 2. Runtime extension hooks (NO vendored edits)
- TrainSpec dataclass fields: <list>
- register_train_spec / get_train_spec signatures: <exact>
- How train.py selects a TrainSpec from [model].name: <exact>
- How to inject a *custom import* so our register call runs before TrainSpec
  lookup (e.g. `--experimental.custom_import` / `--model.print-after-load` /
  env hook): <exact flag or mechanism>

## 3. Model flavor / args
- Model-args class for llama3 (e.g. TransformerModelArgs): exact fields
  (dim, n_layers, n_heads, n_kv_heads, ffn_dim_multiplier/ multiple_of, rope_theta,
  norm_eps, vocab_size, max_seq_len, tied/ untied embeddings flag).
- The flavor registry dict and how to add an entry at runtime.

## 4. Dataloader component
- BaseDataLoader / ParallelAwareDataloader API the TrainSpec expects.
- Signature of build_dataloader(...) the TrainSpec calls (dp_rank, dp_world_size,
  tokenizer, job_config, ...). Record EXACT param names.

## 5. LR scheduler component
- How TrainSpec supplies the LR scheduler (LambdaLR builder vs config). Record
  the builder signature so src/titan_ext/lr_scheduler.py can match WSD.
```

- [ ] **Step 3: Commit**

```bash
git add docs/torchtitan_api_notes.md
git commit -m "docs(torchtitan): record v0.2.2 extension API (config keys, train-spec, flavor, dataloader)"
```

---

## Task 5: `src/utils/torchtitan_args.py` — config → (toml, overrides) mapper (bf16 AdamW)

Pure function, no torch import — unit-testable on CPU like `megatron_args`. Emits a TOML dict + override list using the exact keys recorded in `docs/torchtitan_api_notes.md §1`. Covers model dims, training, AdamW optimizer, FSDP2 parallelism, and bf16. (Scheduler is Task 6; dataloader wiring is Task 8.)

**Files:**
- Create: `src/utils/torchtitan_args.py`
- Test: `tests/unit/test_torchtitan_args.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_torchtitan_args.py
from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from src.utils.torchtitan_args import build_torchtitan_config


def _cfg():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "experiment=optim/adam",
            "cluster=h100_de",
            "backend=torchtitan",
        ]
    )
    resolve_config(cfg)
    return cfg


def test_returns_toml_dict_and_override_list():
    toml, overrides = build_torchtitan_config(_cfg())
    assert isinstance(toml, dict)
    assert isinstance(overrides, list)
    assert all(isinstance(s, str) for s in overrides)


def test_model_block_selects_slm_spec_and_flavor():
    # [model] TOML carries only name+flavor; dims live in the registered flavor
    # (asserted in Task 8's build_slm_flavor test), not here.
    toml, _ = build_torchtitan_config(_cfg())
    model = toml["model"]
    assert model["name"] == "slm_llama3"
    assert model["flavor"] == "slm_300m"


def test_training_block_uses_resolved_values():
    cfg = _cfg()
    toml, _ = build_torchtitan_config(cfg)
    training = toml["training"]
    assert training["seq_len"] == int(cfg.base.model.seq_length)
    assert training["global_batch_size"] == int(cfg.training.global_batch_size)
    assert training["seed"] == int(cfg.seed)
    assert training["steps"] == int(cfg.training.total_tokens) // int(cfg.base.model.seq_length)


def test_optimizer_is_adamw_with_betas():
    cfg = _cfg()
    toml, _ = build_torchtitan_config(cfg)
    opt = toml["optimizer"]
    assert opt["name"].lower() == "adamw"
    assert opt["lr"] == float(cfg.optim.get("lr", cfg.optim.get("adam", {}).get("lr")))


def test_parallelism_is_fsdp_only_at_300m():
    cfg = _cfg()
    toml, _ = build_torchtitan_config(cfg)
    par = toml["parallelism"]
    assert par["tensor_parallel_degree"] == int(cfg.parallelism.tp)  # 1 at 300m
    assert par["data_parallel_shard_degree"] == -1  # FSDP over all remaining ranks


def test_rejects_non_adamw_optimizer():
    cfg = _parse_overrides(
        ["base/family=llama3", "experiment=optim/poet", "backend=torchtitan"]
    )
    resolve_config(cfg)
    import pytest

    with pytest.raises(ValueError, match="only supports adamw"):
        build_torchtitan_config(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_torchtitan_args.py -v`
Expected: FAIL — `ModuleNotFoundError: src.utils.torchtitan_args`.

- [ ] **Step 3: Write the mapper**

```python
# src/utils/torchtitan_args.py
"""Translate a resolved slm-research config into a torchtitan JobConfig.

Returns ``(toml_dict, overrides)``:
  * ``toml_dict``  -> serialized to ``<run_dir>/torchtitan.toml`` and passed as
                      ``--job.config_file``.
  * ``overrides``  -> dotted ``--section.key value`` CLI args appended after it.

Pure function: no torch, no torchtitan import. The TOML *key names* below are the
ones recorded in docs/torchtitan_api_notes.md §1 for the v0.2.2 pin; if a bump
moves them, update both files together.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def _adam_lr(optim: DictConfig) -> float:
    if optim.get("lr", None) is not None:
        return float(optim.lr)
    return float(optim.get("adam", {}).get("lr", 1.0e-3))


# slm family -> torchtitan native model name (verified to ship at v0.2.2).
_FAMILY_TO_TITAN = {"llama3": "llama3", "qwen3": "qwen3", "deepseek_v3": "deepseek_v3"}
# Families where the shared dense/GQA dims map cleanly onto torchtitan's args, so
# we register a custom `slm_<scale>` flavor. deepseek_v3 is EXCLUDED: its args
# (MLA ranks + MoE sizing: inter_dim/moe_inter_dim/n_dense_layers/q_lora_rank/...)
# don't follow from slm's dense dims, so M1 uses a NATIVE deepseek flavor as-is.
_SLM_FLAVOR_FAMILIES = {"llama3", "qwen3"}


def _model_block(cfg: DictConfig) -> dict:
    # torchtitan's [model] TOML carries ONLY name + flavor (+ asset paths). The
    # model DIMENSIONS live in the registered model_args flavor, NOT in TOML, so
    # src/titan_ext clones torchtitan's NATIVE model of this family and (for the
    # dense families) adds an `slm_<scale>` flavor from SLM_RESOLVED_CONFIG.
    family = str(cfg.base.family)
    if family not in _FAMILY_TO_TITAN:
        raise ValueError(
            f"torchtitan backend supports families {sorted(_FAMILY_TO_TITAN)}; got {family!r}"
        )
    if family in _SLM_FLAVOR_FAMILIES:
        flavor = f"slm_{cfg.base.scale}"       # slm-registered model_args (size)
    else:
        # deepseek_v3 M1: pick a torchtitan-native flavor (overridable per scale
        # via base.model.titan_flavor); a full deepseek dims-mapper is a follow-on.
        flavor = str(cfg.base.model.get("titan_flavor", "debugmodel"))
    return {
        "name": f"slm_{family}",               # slm-registered clone of torchtitan's native model
        "flavor": flavor,
    }


def _training_block(cfg: DictConfig) -> dict:
    seq_len = int(cfg.base.model.seq_length)
    steps = int(cfg.training.total_tokens) // seq_len
    return {
        "seq_len": seq_len,
        "global_batch_size": int(cfg.training.global_batch_size),
        "steps": steps,
        # NOTE: confirm in Task 4 whether seed is [training].seed or top-level;
        # place it under the section the v0.2.2 JobConfig actually defines.
        "seed": int(cfg.seed),
        "mixed_precision_param": "bfloat16",  # M1 baseline; Float8 is a follow-on
        "max_norm": float(cfg.training.get("clip_grad", 1.0)),
    }


def _optimizer_block(cfg: DictConfig) -> dict:
    optim = cfg.optim
    if str(optim.type) != "adamw":
        raise ValueError(
            f"torchtitan backend only supports adamw in milestone 1; got {optim.type!r}"
        )
    betas = list(optim.get("betas", [0.9, 0.95]))
    return {
        "name": "AdamW",
        "lr": _adam_lr(optim),
        "eps": float(optim.get("eps", 1.0e-8)),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "weight_decay": float(optim.get("weight_decay", 0.1)),
    }


def _parallelism_block(cfg: DictConfig) -> dict:
    par = cfg.parallelism
    return {
        "tensor_parallel_degree": int(par.get("tp", 1)),
        "pipeline_parallel_degree": int(par.get("pp", 1)),
        "context_parallel_degree": 1,
        # -1 => FSDP2 shards over all remaining (world / TP / PP / CP) ranks.
        "data_parallel_shard_degree": -1,
        "data_parallel_replicate_degree": 1,
    }


def _metrics_block(cfg: DictConfig) -> dict:
    return {
        "enable_wandb": not bool(cfg.cluster.get("wandb_offline", False)),
    }


def build_torchtitan_config(cfg: DictConfig) -> tuple[dict, list[str]]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    toml: dict = {
        "model": _model_block(cfg),
        "training": _training_block(cfg),
        "optimizer": _optimizer_block(cfg),
        "parallelism": _parallelism_block(cfg),
        "metrics": _metrics_block(cfg),
    }
    overrides: list[str] = []  # scheduler (Task 6) + dataloader (Task 8) append here
    return toml, overrides
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_torchtitan_args.py -v`
Expected: PASS (6 tests).

> Top-level TOML section names are VERIFIED for v0.2.2: `[model]`, `[training]` (has both `local_batch_size` and `global_batch_size`), `[optimizer]`, `[lr_scheduler]`, `[parallelism]`, `[metrics]`. If a future bump moves a key, update the test asserts and the block functions together — the test encodes the contract.

- [ ] **Step 5: Add `unmapped_megatron_knobs` (warn-and-skip, not error)**

Because the torchtitan model is native, slm's Megatron monkey-patches and a few Megatron-only knobs have no effect. Rather than error (which would break "same yaml, just change the backend"), the launcher logs exactly what it skipped. Add the pure detector to `src/utils/torchtitan_args.py`:
```python
def unmapped_megatron_knobs(cfg: DictConfig) -> list[str]:
    """Human-readable notes for Megatron-only signals torchtitan ignores."""
    notes: list[str] = []
    patches = list(cfg.get("experiment", {}).get("patches", []) or [])
    if patches:
        notes.append(f"experiment.patches {patches} — Megatron monkey-patches, ignored on torchtitan")
    if bool(cfg.base.model.get("use_sandwich_norm", False)):
        notes.append("base.model.use_sandwich_norm — no torchtitan-native equivalent")
    return notes
```
Add a test to `tests/unit/test_torchtitan_args.py`:
```python
def test_unmapped_knobs_flags_megatron_patches():
    from omegaconf import OmegaConf

    from src.utils.torchtitan_args import unmapped_megatron_knobs

    cfg = _cfg()
    OmegaConf.set_struct(cfg, False)  # resolved cfg is struct-locked; allow injecting a field
    cfg.experiment.patches = ["sandwich_norm_apply"]
    notes = unmapped_megatron_knobs(cfg)
    assert any("patches" in n for n in notes)
```
(The launcher wiring — logging each note after `resolve_config` — is added in **Task 7 Step 3**, since `launchers/train_torchtitan.py` doesn't exist until then. This step only adds the pure function + its unit test.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_torchtitan_args.py -v`
Expected: PASS (7 tests).

- [ ] **Step 7: Commit**

```bash
git add src/utils/torchtitan_args.py tests/unit/test_torchtitan_args.py
git commit -m "feat(torchtitan): map resolved config to a torchtitan JobConfig (bf16 AdamW) + warn-skip Megatron-only knobs"
```

---

## Task 6: WSD scheduler mapping for torchtitan

Match `src/utils/scheduler.py` semantics. torchtitan's native `[lr_scheduler]` already expresses warmup → stable → decay (`warmup_steps` + `decay_ratio` + `decay_type` + `min_lr_factor`, where `decay_ratio` is the fraction of total steps spent decaying), so the **primary path is config-only**: map the slm `scheduler` block to those keys. The standalone `wsd_lr_multiplier` lambda + custom builder in `src/titan_ext/lr_scheduler.py` is a **fallback** kept only if Task 4 confirms the native curve can't match slm's WSD exactly. The lambda is unit-tested regardless so it's ready if needed.

**Files:**
- Create: `src/titan_ext/__init__.py` (empty package marker for now)
- Create: `src/titan_ext/lr_scheduler.py`
- Modify: `src/utils/torchtitan_args.py` (add `_lr_scheduler_block`, call it in `build_torchtitan_config`)
- Test: `tests/unit/test_torchtitan_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_torchtitan_scheduler.py
from __future__ import annotations

import math

from src.titan_ext.lr_scheduler import wsd_lr_multiplier
from src.utils.torchtitan_args import lr_scheduler_block


def test_warmup_then_stable_then_decay_to_floor():
    total = 1000
    warmup = 100
    decay = 200  # last 200 steps decay
    floor = 0.1

    # warmup ramp
    assert math.isclose(wsd_lr_multiplier(0, total, warmup, decay, floor), 0.0, abs_tol=1e-6)
    assert math.isclose(wsd_lr_multiplier(warmup, total, warmup, decay, floor), 1.0, abs_tol=1e-6)
    # stable plateau
    assert math.isclose(wsd_lr_multiplier(500, total, warmup, decay, floor), 1.0, abs_tol=1e-6)
    # end of run decays to the floor
    assert math.isclose(wsd_lr_multiplier(total, total, warmup, decay, floor), floor, abs_tol=1e-6)


def test_lr_scheduler_block_maps_to_torchtitan_keys():
    sched = {"type": "wsd", "warmup_fraction": 0.1, "wsd_decay_fraction": 0.2,
             "wsd_decay_style": "cosine", "min_lr_ratio": 0.1}
    block = lr_scheduler_block(sched, total_steps=1000)
    assert block["warmup_steps"] == 100
    assert math.isclose(block["decay_ratio"], 0.2)   # fraction, not a step count
    assert block["decay_type"] == "cosine"
    assert math.isclose(block["min_lr_factor"], 0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_torchtitan_scheduler.py -v`
Expected: FAIL — `src.titan_ext.lr_scheduler` and `lr_scheduler_block` don't exist.

- [ ] **Step 3: Implement the WSD lambda**

```python
# src/titan_ext/__init__.py
"""slm-research runtime extensions for torchtitan (registered at train start)."""
```
```python
# src/titan_ext/lr_scheduler.py
"""Warmup-Stable-Decay LR multiplier, matching src/utils/scheduler.py WSD.

Step-based (torchtitan drives the scheduler per optimizer step). Linear warmup
to 1.0 over `warmup_steps`, flat 1.0 plateau, then a linear decay over the final
`decay_steps` down to `floor`.
"""

from __future__ import annotations


def wsd_lr_multiplier(
    step: int, total_steps: int, warmup_steps: int, decay_steps: int, floor: float
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    decay_start = total_steps - decay_steps
    if step < decay_start:
        return 1.0
    if decay_steps <= 0:
        return 1.0
    progress = min(1.0, (step - decay_start) / decay_steps)
    return 1.0 - (1.0 - floor) * progress
```

- [ ] **Step 4: Add `lr_scheduler_block` to the mapper**

In `src/utils/torchtitan_args.py` add:
```python
def lr_scheduler_block(sched: dict, *, total_steps: int) -> dict:
    """Map an slm `scheduler` block to torchtitan [lr_scheduler] keys.

    torchtitan uses `decay_ratio` (a FRACTION of total steps), `decay_type`, and
    `min_lr_factor` — there is NO `decay_steps`. slm's `wsd_decay_fraction` is
    already a ratio, so it maps straight to `decay_ratio`.
    """
    # slm wsd_decay_style -> torchtitan decay_type (confirm names in Task 4 §1).
    _DECAY_TYPE = {"cosine": "cosine", "linear": "linear", "minus_sqrt": "sqrt",
                   "exponential": "linear"}
    warmup_frac = float(sched.get("warmup_fraction", 0.0) or 0.0)
    block = {
        "warmup_steps": int(round(warmup_frac * total_steps)),
        "min_lr_factor": float(sched.get("min_lr_ratio", 0.0) or 0.0),
    }
    if str(sched.get("type", "")).lower() == "wsd":
        block["decay_ratio"] = float(sched.get("wsd_decay_fraction", 0.0) or 0.0)
        block["decay_type"] = _DECAY_TYPE.get(str(sched.get("wsd_decay_style", "cosine")), "cosine")
    return block
```
Then in `build_torchtitan_config`, after building `training`:
```python
    steps = toml["training"]["steps"]
    toml["lr_scheduler"] = lr_scheduler_block(
        OmegaConf.to_container(cfg.scheduler, resolve=True), total_steps=steps
    )
```

> Primary path: torchtitan's native scheduler consumes these `[lr_scheduler]` keys directly — no custom builder needed, so the slm TrainSpec keeps `base.build_lr_schedulers_fn`. Fallback only: if Task 4 shows the native decay can't match slm's WSD curve, override `build_lr_schedulers_fn` with a `LambdaLR` closing over `wsd_lr_multiplier` (helper in `src/titan_ext/lr_scheduler.py`); convert `decay_ratio` back to `decay_steps = round(decay_ratio * total_steps)` for the lambda.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_torchtitan_scheduler.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/titan_ext/__init__.py src/titan_ext/lr_scheduler.py src/utils/torchtitan_args.py tests/unit/test_torchtitan_scheduler.py
git commit -m "feat(torchtitan): WSD lr-scheduler mapping + step-based multiplier"
```

---

## Task 7: `launchers/train_torchtitan.py` — parent launcher

Mirror `launchers/train_megatron.py`: resolve config, archive, write `<run_dir>/torchtitan.toml`, build the `torchrun -m torchtitan.train` command, set PYTHONPATH to include the vendored torchtitan + Megatron (Megatron is needed for the dataloader in Task 8). `--dry-run` prints the command without launching.

**Files:**
- Create: `launchers/train_torchtitan.py`
- Test: `tests/unit/test_train_torchtitan_command.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_train_torchtitan_command.py
from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from launchers.train_torchtitan import build_torchrun_command


def _cfg():
    cfg = _parse_overrides(
        ["base/family=llama3", "base/scale=300m", "experiment=optim/adam",
         "cluster=h100_de", "backend=torchtitan"]
    )
    resolve_config(cfg)
    return cfg


def test_command_targets_torchtitan_train():
    cfg = _cfg()
    cmd = build_torchrun_command(cfg)
    assert cmd[:3] == ["torchrun", "--nproc_per_node", str(cfg.cluster.gpus_per_node)]
    assert "-m" in cmd and "torchtitan.train" in cmd
    assert "--job.config_file" in cmd
    toml_path = cmd[cmd.index("--job.config_file") + 1]
    assert toml_path.endswith(f"{cfg._derived.run_dir}/torchtitan.toml")


def test_command_registers_slm_extension():
    cfg = _cfg()
    cmd = build_torchrun_command(cfg)
    # The slm extension (flavor + dataloader + scheduler) must be imported by
    # torchtitan before TrainSpec lookup. Exact flag name comes from Task 4 §2.
    assert any("src.titan_ext" in part for part in cmd)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_train_torchtitan_command.py -v`
Expected: FAIL — `launchers.train_torchtitan` does not exist.

- [ ] **Step 3: Implement the launcher**

```python
# launchers/train_torchtitan.py
"""Resolve slm config and launch torchtitan through its native train entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

try:
    import tomli_w
except ModuleNotFoundError:  # pragma: no cover - dep installed on compute nodes
    tomli_w = None

from launchers.submit import REPO_ROOT, _parse_overrides, archive_resolved_config, resolve_config
from launchers.train_megatron import _launch_nnodes
from src.utils.torchtitan_args import build_torchtitan_config, unmapped_megatron_knobs

TORCHTITAN_ROOT = Path(REPO_ROOT) / "third_party" / "torchtitan"
MEGATRON_ROOT = Path(REPO_ROOT) / "third_party" / "Megatron-LM"

# VERIFIED against v0.2.2: train.py runs importlib.import_module(
#   job_config.experimental.custom_import) before get_train_spec lookup, so the
# CLI form below imports src.titan_ext (which registers the slm_llama3 spec).
CUSTOM_IMPORT_FLAG = "--experimental.custom_import"


def _write_toml(cfg) -> Path:
    toml_dict, _ = build_torchtitan_config(cfg)
    run_dir = Path(REPO_ROOT) / cfg._derived.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "torchtitan.toml"
    if tomli_w is None:
        raise RuntimeError("tomli-w is required to emit torchtitan.toml; pip install tomli-w")
    with out.open("wb") as fh:
        tomli_w.dump(toml_dict, fh)
    return out


def build_torchrun_command(cfg) -> list[str]:
    toml_path = Path(REPO_ROOT) / cfg._derived.run_dir / "torchtitan.toml"
    _, overrides = build_torchtitan_config(cfg)
    cmd = [
        "torchrun",
        "--nproc_per_node", str(cfg.cluster.gpus_per_node),
        "--nnodes", str(_launch_nnodes()),
        "--node_rank", str(os.environ.get("NODE_RANK", "0")),
        "--master_addr", str(os.environ.get("MASTER_ADDR", "localhost")),
        "--master_port", str(os.environ.get("MASTER_PORT", "6000")),
        "-m", "torchtitan.train",
        "--job.config_file", os.fspath(toml_path),
        CUSTOM_IMPORT_FLAG, "src.titan_ext",
    ]
    cmd.extend(overrides)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args, leftover = parser.parse_known_args()
    overrides = list(args.overrides) + list(leftover)

    cfg = _parse_overrides(overrides)
    if "backend" not in cfg:
        cfg.backend = "torchtitan"
    resolve_config(cfg)
    for note in unmapped_megatron_knobs(cfg):  # warn-and-skip Megatron-only knobs
        print(f"[torchtitan] skipping {note}")
    archive = archive_resolved_config(cfg)
    _write_toml(cfg)
    cmd = build_torchrun_command(cfg)

    payload = {"run_name": str(cfg._derived.run_name), "archive": os.fspath(archive), "command": cmd}
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [os.fspath(REPO_ROOT), os.fspath(TORCHTITAN_ROOT), os.fspath(MEGATRON_ROOT),
         env.get("PYTHONPATH", "")]
    )
    # src.titan_ext (imported via experimental.custom_import inside torchtitan)
    # reads this to build the slm_<scale> flavor dims that can't ride in TOML.
    env["SLM_RESOLVED_CONFIG"] = os.fspath(
        Path(REPO_ROOT) / cfg._derived.run_dir / "resolved_config.yaml"
    )
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_train_torchtitan_command.py -v`
Expected: PASS (2 tests). `build_torchrun_command` reads the toml path only (it does not require the file to exist), so the test needs no GPU and no torchtitan import.

- [ ] **Step 5: Smoke the dry-run end to end**

Run: `python -m launchers.train_torchtitan base/family=llama3 base/scale=300m experiment=optim/adam cluster=h100_de backend=torchtitan --dry-run`
Expected: prints JSON with `command` targeting `torchtitan.train` and a `torchtitan.toml` written under `runs/<run_name>/`.

- [ ] **Step 6: Commit**

```bash
git add launchers/train_torchtitan.py tests/unit/test_train_torchtitan_command.py
git commit -m "feat(torchtitan): parent launcher emits torchtitan.toml and torchrun command"
```

---

## Task 8: slm TrainSpec — register a native-family flavor at slm's size (no dataloader yet)

Via the `src.titan_ext` custom-import hook, register a **new** TrainSpec named `"slm_<family>"` (for `family ∈ {llama3, qwen3, deepseek_v3}`) cloned from torchtitan's **native** spec of that family, whose `model_args` mapping gains an `slm_<scale>` flavor. The flavor is built by **cloning a native flavor of that family** (`dataclasses.replace` on a template from the native `model_args` registry) and overriding only the dimension fields slm sets. This reuses torchtitan's own per-family model-args class + its FFN/MoE conventions — so the result "qualifies as" that family without us constructing any family-specific args or solving FFN widths. Verified facts that shape this: `register_train_spec(name, spec)` takes **two args** and **raises on a duplicate name** (fresh name required), and `TrainSpec.model_args` is a `Mapping[str, BaseModelArgs]` flavor registry. The native LR scheduler + optimizer are inherited from `base` (Task 6); the dataloader is wired in Task 9. We intentionally do **not** force tied embeddings or an exact FFN width — the slm-side goal is "same family + roughly the right scale," not Megatron-identical internals.

> **Per-family reality (verified at v0.2.2).** `llama3` (`TransformerModelArgs`) and `qwen3` (`Qwen3ModelArgs`) share `dim/n_layers/n_heads/n_kv_heads/vocab_size`, both ship a `debugmodel` flavor, and the `hasattr` filter handles their divergent fields (qwen3 has `hidden_dim`/`head_dim`/`enable_weight_tying`; llama3 doesn't) — so the clone+override path is valid for both. **`deepseek_v3` (`DeepSeekV3ModelArgs`) is different**: it has no `n_kv_heads`, sizes itself via `inter_dim`/`moe_inter_dim`/`n_dense_layers`/MLA ranks (`q_lora_rank`, `kv_lora_rank`, `qk_*_head_dim`, `v_head_dim`)/`moe_args`, and its smallest flavor is `debugmodel` (`dim=256, n_layers=6`). Overriding only the dense dims would yield an incoherent model, so **M1 runs deepseek_v3 on a torchtitan-native flavor as-is** (`_SLM_FLAVOR_FAMILIES` excludes it); a proper deepseek dims-mapper (mapping slm's deepseek scale → `inter_dim`/`moe_*`/MLA ranks) is a named follow-on.

**Files:**
- Create: `src/titan_ext/model_flavor.py`
- Modify: `src/titan_ext/__init__.py` (call the registration on import)
- Test: `tests/unit/test_titan_ext_registration.py`

- [ ] **Step 1: Write the failing test (import-time registration, torchtitan-gated)**

```python
# tests/unit/test_titan_ext_registration.py
from __future__ import annotations

import dataclasses
import importlib

import pytest
from omegaconf import OmegaConf

from launchers.submit import _parse_overrides, resolve_config


def test_build_slm_flavor_overrides_only_template_fields():
    """build_slm_flavor clones a native template and overrides only the dim
    fields the template actually has — so it works across the different
    per-family model-args classes (llama3 vs qwen3 vs deepseek_v3) and silently
    ignores slm fields torchtitan doesn't model (e.g. ffn_hidden_size)."""
    from src.titan_ext.model_flavor import build_slm_flavor

    @dataclasses.dataclass
    class FakeArgs:  # stands in for a native TransformerModelArgs
        dim: int = 4096
        n_layers: int = 32
        n_kv_heads: int = 8
        vocab_size: int = 1000
        # NOTE: no ffn_hidden_size field on purpose

    out = build_slm_flavor(FakeArgs(), {"dim": 1024, "n_layers": 12, "n_kv_heads": 4,
                                        "vocab_size": 128256, "ffn_hidden_size": 2560})
    assert out.dim == 1024 and out.n_layers == 12 and out.n_kv_heads == 4
    assert out.vocab_size == 128256  # ffn_hidden_size was ignored, not a crash


def test_import_with_env_registers_slm_family_spec(tmp_path, monkeypatch):
    pytest.importorskip("torchtitan")  # this test needs the real registry
    cfg = _parse_overrides(
        ["base/family=llama3", "base/scale=300m", "experiment=optim/adam", "backend=torchtitan"]
    )
    resolve_config(cfg)
    resolved = tmp_path / "resolved_config.yaml"
    resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    monkeypatch.setenv("SLM_RESOLVED_CONFIG", str(resolved))

    import src.titan_ext as ext
    importlib.reload(ext)  # re-run registration with the env in place

    from torchtitan.protocols.train_spec import get_train_spec

    spec = get_train_spec("slm_llama3")
    assert "slm_300m" in spec.model_args  # our flavor is in the registry
```

- [ ] **Step 2: Run test to verify it fails (or skips without torchtitan)**

Run: `pytest tests/unit/test_titan_ext_registration.py -v`
Expected: FAIL where torchtitan is importable (`build_slm_flavor` missing); SKIP otherwise.

- [ ] **Step 3: Implement the flavor builder + registration**

```python
# src/titan_ext/model_flavor.py
"""Size an slm flavor by cloning a torchtitan NATIVE model-args template.

We do not construct any family-specific model-args class ourselves. Instead we
take an existing flavor of the same family from the native TrainSpec's model_args
registry (the "template") and dataclasses.replace() the dimension fields slm
sets, ignoring any slm field the template doesn't model (e.g. ffn_hidden_size —
torchtitan sizes the FFN from `dim` natively). This keeps torchtitan's own model
class + FFN/MoE conventions, so the flavor "qualifies as" that family. No
torchtitan import here, so the helper is unit-testable on CPU with a fake args
dataclass.
"""

from __future__ import annotations

import dataclasses


def build_slm_flavor(template, dims: dict):
    """Return a copy of native `template` with slm's dims applied.

    Only fields the template actually has are overridden — across families the
    args classes differ, and slm sets fields (ffn_hidden_size) torchtitan derives
    rather than stores. Per the slm-side goal we honor layer/hidden/head/vocab
    counts (so a "300m" is ~300m) and let everything else follow native defaults.
    """
    overrides = {k: v for k, v in dims.items() if hasattr(template, k)}
    return dataclasses.replace(template, **overrides)


def pick_template(model_args: dict):
    """Pick a deterministic template flavor from a native model_args registry."""
    for key in ("debugmodel", "debug", "1B", "8B"):
        if key in model_args:
            return model_args[key]
    return next(iter(model_args.values()))  # any registered flavor of this family
```
```python
# src/titan_ext/__init__.py  (replace the placeholder body)
"""slm-research runtime extensions for torchtitan.

Imported via torchtitan's experimental.custom_import hook BEFORE TrainSpec
lookup (see launchers/train_torchtitan.py CUSTOM_IMPORT_FLAG). On import it reads
the resolved slm config (SLM_RESOLVED_CONFIG), clones torchtitan's NATIVE spec
for the configured family, adds an `slm_<scale>` flavor, and registers the whole
thing as a new TrainSpec "slm_<family>". Task 9 also swaps in the Megatron-indexed
dataloader.

Importing this package MUST NOT raise when torchtitan is absent (CPU unit-test
env) — registration is guarded. It must also be idempotent: register_train_spec
RAISES on a duplicate name, so we skip if "slm_<family>" already exists.
"""

from __future__ import annotations

import dataclasses
import os

# Keep in sync with src/utils/torchtitan_args (_FAMILY_TO_TITAN, _SLM_FLAVOR_FAMILIES).
_FAMILY_TO_TITAN = {"llama3": "llama3", "qwen3": "qwen3", "deepseek_v3": "deepseek_v3"}
_SLM_FLAVOR_FAMILIES = {"llama3", "qwen3"}  # deepseek_v3 uses a native flavor in M1


def _dims_from(cfg) -> dict:
    m = cfg.base.model
    # Superset of dense/GQA dim fields; build_slm_flavor's hasattr filter keeps
    # only the ones the target family's args class actually has (e.g. llama3 has
    # no `hidden_dim`/`head_dim`; qwen3 has both — so qwen3 even honors slm's FFN).
    return {
        "dim": int(m.hidden_size),
        "n_layers": int(m.num_layers),
        "n_heads": int(m.num_attention_heads),
        "n_kv_heads": int(m.num_query_groups),
        "head_dim": int(m.head_dim),
        "hidden_dim": int(m.ffn_hidden_size),   # qwen3 explicit FFN; llama3/deepseek lack this field
        "vocab_size": int(cfg.base.tokenizer.nominal_vocab_size),
        "rope_theta": float(m.rotary_base),
        "norm_eps": float(m.norm_epsilon),
        "max_seq_len": int(m.seq_length),
    }


def _slm_spec_from(base, cfg):
    """Clone native `base`; for dense families add an slm_<scale> flavor sized
    from `cfg`. deepseek_v3 keeps `base`'s native flavors unchanged (its MLA/MoE
    args don't map from slm's dense dims — see _SLM_FLAVOR_FAMILIES).

    Native LR scheduler / optimizer / parallelize fns are inherited from `base`.
    Task 9 additionally sets build_dataloader_fn here. TrainSpec field names are
    the verified ones (docs/torchtitan_api_notes.md §2).
    """
    from src.titan_ext.model_flavor import build_slm_flavor, pick_template

    model_args = dict(base.model_args)  # copy native flavor mapping
    if str(cfg.base.family) in _SLM_FLAVOR_FAMILIES:
        template = pick_template(model_args)
        model_args[f"slm_{cfg.base.scale}"] = build_slm_flavor(template, _dims_from(cfg))
    return dataclasses.replace(base, model_args=model_args)


def _register() -> None:
    try:
        from torchtitan.protocols.train_spec import get_train_spec, register_train_spec  # Task 4 §2
    except Exception:
        return  # torchtitan not importable (CPU unit-test env): no-op
    if "SLM_RESOLVED_CONFIG" not in os.environ:
        return  # no config available yet (e.g. import before launch sets the env)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.environ["SLM_RESOLVED_CONFIG"])
    titan = _FAMILY_TO_TITAN.get(str(cfg.base.family))
    if titan is None:
        return  # unsupported family on torchtitan: leave registry untouched
    slm_name = f"slm_{cfg.base.family}"
    try:
        get_train_spec(slm_name)
        return  # already registered (idempotent — register_train_spec raises on dup)
    except Exception:
        pass
    base = get_train_spec(titan)
    register_train_spec(slm_name, _slm_spec_from(base, cfg))


_register()
```

> Importing the package is safe at every stage (no raise without env/torchtitan). The slm spec is registered under a fresh name `"slm_<family>"` — required because `register_train_spec` raises on a duplicate, so the stock `"llama3"`/`"qwen3"`/`"deepseek_v3"` specs are never mutated. `_model_block` emits `[model].name = "slm_<family>"` so torchtitan selects it. Task 9 adds `build_dataloader_fn=build_dataloader` to the `dataclasses.replace(...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_titan_ext_registration.py -v`
Expected: PASS where torchtitan is importable; SKIP otherwise.

- [ ] **Step 5: Commit**

```bash
git add src/titan_ext/model_flavor.py src/titan_ext/__init__.py tests/unit/test_titan_ext_registration.py
git commit -m "feat(torchtitan): register slm_<family> TrainSpec by cloning native model + slm-sized flavor"
```

---

## Task 9: Megatron-indexed dataloader component (M2: data parity)

The linchpin. A torchtitan dataloader that wraps Megatron's `GPTDataset`
(`third_party/Megatron-LM/megatron/core/datasets/gpt_dataset.py`) so torchtitan
consumes the **same tokens in the same order** as the Megatron path. Wire it as
the TrainSpec's `build_dataloader`. Parity test asserts the first N global
batches are identical across backends.

**Files:**
- Create: `src/titan_ext/dataloader.py`
- Modify: `src/titan_ext/__init__.py` (`_slm_spec_from` now supplies `build_dataloader`)
- Test: `tests/integration/test_titan_megatron_data_parity.py`

- [ ] **Step 1: Write the failing parity test (CPU, tiny synthetic .bin/.idx)**

```python
# tests/integration/test_titan_megatron_data_parity.py
"""First N global batches from the torchtitan dataloader must byte-match the
Megatron GPTDataset on the same (path, seq_len, gbs, seed)."""

from __future__ import annotations

import pytest

pytest.importorskip("megatron.core.datasets.gpt_dataset")

from src.titan_ext.dataloader import build_megatron_indexed_batches
from tests.integration._indexed_fixtures import make_tiny_indexed_dataset  # helper below


def test_first_batches_match_megatron(tmp_path):
    prefix = make_tiny_indexed_dataset(tmp_path, num_docs=64, doc_len=128, vocab=256, seed=0)
    seq_len, gbs, seed, n = 32, 8, 1234, 4

    mg_batches = build_megatron_indexed_batches(
        path=prefix, seq_len=seq_len, global_batch_size=gbs, seed=seed,
        n_batches=n, source="megatron")
    tt_batches = build_megatron_indexed_batches(
        path=prefix, seq_len=seq_len, global_batch_size=gbs, seed=seed,
        n_batches=n, source="torchtitan")

    assert len(mg_batches) == len(tt_batches) == n
    for a, b in zip(mg_batches, tt_batches, strict=True):
        assert a.tolist() == b.tolist()
```
Also create `tests/integration/_indexed_fixtures.py` with `make_tiny_indexed_dataset(...)` that writes a small `.bin/.idx` pair using `megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder` (look up the builder API in the vendored file; write `num_docs` random documents of `doc_len` tokens and return the `text_document` prefix).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_titan_megatron_data_parity.py -v`
Expected: FAIL — `src.titan_ext.dataloader` missing.

- [ ] **Step 3: Implement the dataloader over `GPTDataset`**

```python
# src/titan_ext/dataloader.py
"""torchtitan dataloader backed by Megatron's GPTDataset.

Both backends build the SAME megatron.core.datasets.GPTDataset (same indexing,
same shuffle seed), so token order is identical by construction. The
`source="torchtitan"` path additionally wraps it in torchtitan's
ParallelAwareDataloader (API per docs/torchtitan_api_notes.md §4); the
`source="megatron"` path iterates the GPTDataset directly for the parity test.
"""

from __future__ import annotations

import numpy as np


def _build_gpt_dataset(path: str, seq_len: int, num_samples: int, seed: int):
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig

    # Build a single-split GPTDataset; field names track the vendored mcore pin.
    config = GPTDatasetConfig(
        random_seed=seed,
        sequence_length=seq_len,
        blend=([path], None),
        split="100,0,0",
        path_to_cache=None,
        tokenizer=None,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    builder = BlendedMegatronDatasetBuilder(
        GPTDataset, [num_samples, 0, 0], lambda: True, config
    )
    train, _, _ = builder.build()
    return train


def build_megatron_indexed_batches(*, path, seq_len, global_batch_size, seed, n_batches, source):
    """Return the first `n_batches` global batches as np arrays of token ids."""
    num_samples = global_batch_size * n_batches
    ds = _build_gpt_dataset(path, seq_len, num_samples, seed)
    batches = []
    for b in range(n_batches):
        rows = [np.asarray(ds[b * global_batch_size + i]["tokens"]) for i in range(global_batch_size)]
        batches.append(np.stack(rows, axis=0))
    return batches


def build_dataloader(*, dp_world_size, dp_rank, tokenizer, job_config):
    """TrainSpec build_dataloader_fn. VERIFIED signature (Task 4 §2):
    (dp_world_size, dp_rank, tokenizer, job_config). `tokenizer` is unused —
    the corpus is pre-tokenized .bin/.idx. Data `path` + `seed` are read from
    SLM_RESOLVED_CONFIG (single source of truth) to avoid depending on where the
    v0.2.2 JobConfig puts seed; seq_len/steps/batch come from job_config."""
    import os

    from omegaconf import OmegaConf
    from torchtitan.components.dataloader import ParallelAwareDataloader  # verified §4

    slm = OmegaConf.load(os.environ["SLM_RESOLVED_CONFIG"])
    path, seed = str(slm.data.path), int(slm.seed)
    seq_len = job_config.training.seq_len
    local_bs = job_config.training.local_batch_size  # per-DP-rank batch size
    num_samples = job_config.training.steps * job_config.training.global_batch_size
    ds = _build_gpt_dataset(path, seq_len, num_samples, seed)
    # ParallelAwareDataloader(dataset, dp_rank, dp_world_size, **kwargs) — verified §4.
    return ParallelAwareDataloader(ds, dp_rank, dp_world_size, batch_size=local_bs)
```

- [ ] **Step 4: Carry the data path/seed into the TOML and wire `build_dataloader`**

In `src/utils/torchtitan_args.py` `_training_block`, add the dataset coordinates the dataloader reads:
```python
        "dataset": "slm_megatron_indexed",
        "dataset_path": str(cfg.data.path),
```
In `src/titan_ext/__init__.py` `_slm_spec_from(base, cfg)`, add `build_dataloader_fn=_dataloader` to the `dataclasses.replace(base, model_args=..., ...)` call, where `from src.titan_ext.dataloader import build_dataloader as _dataloader` is imported inside `_slm_spec_from` (exact field name `build_dataloader_fn` from Task 4 §2). This replaces torchtitan's stock dataloader with the Megatron-indexed one for every `slm_<family>` spec.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/integration/test_titan_megatron_data_parity.py -v`
Expected: PASS — the two batch lists are identical.

- [ ] **Step 6: Commit**

```bash
git add src/titan_ext/dataloader.py src/titan_ext/__init__.py src/utils/torchtitan_args.py tests/integration/test_titan_megatron_data_parity.py tests/integration/_indexed_fixtures.py
git commit -m "feat(torchtitan): Megatron-indexed dataloader with byte-level data parity (M2)"
```

---

## Task 10: `--backend` flag dispatch in the train wrappers

`scripts/train_adam.sh` learns `--backend {megatron,torchtitan}` (default `megatron`): it injects `backend=<value>` into the override list and dispatches to the matching `python -m launchers.train_{megatron,torchtitan}`. Other wrappers reject `--backend torchtitan` with a clear message until their algorithms are ported.

**Files:**
- Modify: `scripts/train_adam.sh`
- Modify: `scripts/train_muon.sh`, `scripts/train_poet.sh`, `scripts/train_ngpt.sh` (reject torchtitan)
- Test: `tests/unit/test_backend_dispatch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_backend_dispatch.py
"""train_adam.sh --backend routes to the right launcher and injects backend=."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(args):
    # SLM_DRYRUN_PRINT=1 makes the wrapper echo the final `python -m ...` line
    # instead of executing it (added in Step 3), so this test needs no GPU/env.
    env = {"SLM_DRYRUN_PRINT": "1", "PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        ["bash", str(REPO / "scripts/train_adam.sh"), *args],
        capture_output=True, text=True, env=env, cwd=REPO,
    )
    return out.stdout + out.stderr


def test_default_backend_is_megatron():
    out = _run(["llama3", "--dry-run"])
    assert "launchers.train_megatron" in out
    assert "backend=torchtitan" not in out


def test_backend_torchtitan_routes_and_injects():
    out = _run(["llama3", "--backend", "torchtitan", "--dry-run"])
    assert "launchers.train_torchtitan" in out
    assert "backend=torchtitan" in out


def test_muon_rejects_torchtitan():
    out = subprocess.run(
        ["bash", str(REPO / "scripts/train_muon.sh"), "llama3", "--backend", "torchtitan"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert out.returncode != 0
    assert "not yet supported on torchtitan" in (out.stdout + out.stderr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_backend_dispatch.py -v`
Expected: FAIL — wrappers don't understand `--backend` and always call `train_megatron`.

- [ ] **Step 3: Add backend parsing to `scripts/train_adam.sh`**

After the existing `ARCH`/`shift` block and before the `python -m launchers.train_megatron` call, insert backend extraction (strip `--backend X` out of `"$@"`):
```bash
BACKEND="megatron"
NEWARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --backend=*) BACKEND="${1#*=}"; shift ;;
    *) NEWARGS+=("$1"); shift ;;
  esac
done
set -- "${NEWARGS[@]}"

case "${BACKEND}" in
  megatron)   LAUNCHER="launchers.train_megatron"; BACKEND_OVERRIDE=() ;;
  torchtitan) LAUNCHER="launchers.train_torchtitan"; BACKEND_OVERRIDE=("backend=torchtitan") ;;
  *) echo "Unknown backend: ${BACKEND}. Use megatron or torchtitan." >&2; exit 2 ;;
esac
```
Then change the launch line from `python -m launchers.train_megatron \` to:
```bash
RUN=(python -m "${LAUNCHER}" \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${BACKEND_OVERRIDE[@]}" \
  "cluster=h100_de" \
  "experiment=optim/adam" \
  "training.global_batch_size=512" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "$@")
if [[ "${SLM_DRYRUN_PRINT:-0}" == "1" ]]; then printf '%s ' "${RUN[@]}"; echo; else "${RUN[@]}"; fi
```

- [ ] **Step 4: Reject torchtitan in the non-adam wrappers**

At the top of `scripts/train_muon.sh`, `scripts/train_poet.sh`, `scripts/train_ngpt.sh` (after `set -euo pipefail`), add:
```bash
for a in "$@"; do
  case "$a" in
    --backend=torchtitan|torchtitan) ;;
  esac
done
if printf '%s\n' "$@" | grep -qx -- "torchtitan"; then :; fi
case " $* " in
  *" --backend torchtitan "*|*" --backend=torchtitan "*)
    echo "This optimizer is not yet supported on torchtitan (milestone 1 is AdamW only)." >&2
    exit 2 ;;
esac
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_backend_dispatch.py tests/unit/test_train_scripts.py -v`
Expected: PASS (new dispatch tests + existing `test_train_scripts.py` still green).

- [ ] **Step 6: Commit**

```bash
git add scripts/train_adam.sh scripts/train_muon.sh scripts/train_poet.sh scripts/train_ngpt.sh tests/unit/test_backend_dispatch.py
git commit -m "feat(torchtitan): --backend flag routes train_adam.sh; other wrappers reject torchtitan"
```

---

## Task 11: W&B / metrics parity

Make torchtitan runs land in the **same** W&B project under the **same** run name as the Megatron path. VERIFIED: torchtitan's `[metrics]` has only `enable_wandb` (no project/run-name field), so project + run name come from the `WANDB_PROJECT` / `WANDB_NAME` env the launcher sets; both backends then aggregate on one dashboard.

**Files:**
- Modify: `launchers/train_torchtitan.py` (set `WANDB_PROJECT` / `WANDB_NAME` / `WANDB_MODE` in `env`)
- Test: `tests/unit/test_torchtitan_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_torchtitan_metrics.py
from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from launchers.train_torchtitan import wandb_env_for


def test_wandb_env_matches_run_identity():
    cfg = _parse_overrides(
        ["base/family=llama3", "experiment=optim/adam", "backend=torchtitan",
         "wandb.project=pretrain-ablations-300m"]
    )
    resolve_config(cfg)
    env = wandb_env_for(cfg)
    assert env["WANDB_PROJECT"] == "pretrain-ablations-300m"
    assert env["WANDB_NAME"] == cfg._derived.run_name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_torchtitan_metrics.py -v`
Expected: FAIL — `wandb_env_for` not defined.

- [ ] **Step 3: Implement `wandb_env_for` and use it in `main`**

In `launchers/train_torchtitan.py`:
```python
def wandb_env_for(cfg) -> dict:
    env = {
        "WANDB_PROJECT": str(cfg.wandb.project),
        "WANDB_NAME": str(cfg._derived.run_name),
    }
    if bool(cfg.cluster.get("wandb_offline", False)):
        env["WANDB_MODE"] = "offline"
    entity = cfg.wandb.get("entity", None)
    if entity:
        env["WANDB_ENTITY"] = str(entity)
    return env
```
In `main`, before `subprocess.run`, merge it: `env.update(wandb_env_for(cfg))`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_torchtitan_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add launchers/train_torchtitan.py src/utils/torchtitan_args.py tests/unit/test_torchtitan_metrics.py
git commit -m "feat(torchtitan): route W&B project/run-name so both backends share dashboards"
```

---

## Task 12: M3 functional training gate (operator-run on GPU)

The acceptance gate is **functional, not Megatron-parity**: for each wired family
(llama3 → qwen3 → deepseek_v3) at a small scale, bf16 + AdamW + WSD, a short run
**trains and the loss decreases healthily** (no NaN/Inf, sane curve shape). This
test is `@pytest.mark.gpu`; it is **not** executed in the dev harness — the plan
gives the runnable commands and the operator runs them on a node and reports.

> Exact-Megatron parity is a non-goal (we use torchtitan's native models), so
> there is no `tie_embeddings=false` matching dance and no Megatron-curve
> tolerance. A side-by-side vs Megatron is welcome as an informal sanity signal
> but gates nothing. Before trusting a curve, glance at the **param count**
> torchtitan logs at startup — it should be in the right ballpark for the scale
> (a wildly-off count means the `slm_<scale>` flavor dims are wrong).

**Files:**
- Create: `tests/numerics/test_titan_training_health.py`
- Create: `docs/superpowers/runbooks/2026-05-30-torchtitan-training-health.md`

- [ ] **Step 1: Write the health-gate test (skips without a loss log)**

```python
# tests/numerics/test_titan_training_health.py
"""M3 gate: a torchtitan run trains and the loss decreases healthily.

Marked gpu + slow. Reads one per-step loss jsonl (path in TT_LOSS_LOG, produced
by the runbook) and asserts: no NaN/Inf, and the final-window mean loss is
meaningfully below the early-window mean (the model is learning)."""

from __future__ import annotations

import json
import math
import os

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.numerics]


def _losses(path):
    return [json.loads(line)["loss"] for line in open(path)]


def test_torchtitan_run_is_healthy():
    log = os.environ.get("TT_LOSS_LOG")
    if not log:
        pytest.skip("set TT_LOSS_LOG to a torchtitan run's per-step loss jsonl")
    losses = _losses(log)
    assert len(losses) >= 40, "need enough steps to judge the curve"
    assert all(math.isfinite(x) for x in losses), "loss has NaN/Inf"
    early = sum(losses[:20]) / 20
    late = sum(losses[-20:]) / 20
    assert late < early - 0.1, f"loss not decreasing: early={early:.3f} late={late:.3f}"
```

- [ ] **Step 2: Write the runbook with the exact commands**

Create `docs/superpowers/runbooks/2026-05-30-torchtitan-training-health.md` with a short run per family on torchtitan (small scale, fixed seed, identical `data=`):
```bash
# llama3 (dense) on torchtitan
codexlog titan_health_llama3 \
  scripts/train_adam.sh llama3 --backend torchtitan base/scale=300m \
    training_regime=ablation_20x seed=7 training.train_iters=200

# qwen3 (dense) on torchtitan
codexlog titan_health_qwen3 \
  scripts/train_adam.sh llama3 --backend torchtitan base/family=qwen3 base/scale=300m \
    training_regime=ablation_20x seed=7 training.train_iters=200

# deepseek_v3 (MoE + MLA) on torchtitan — wire last
codexlog titan_health_dsv3 \
  scripts/train_adam.sh deepseek_v3 --backend torchtitan \
    training_regime=ablation_20x seed=7 training.train_iters=200

# health gate on a run's loss log
TT_LOSS_LOG=<tt_loss.jsonl> pytest -m gpu tests/numerics/test_titan_training_health.py -v
```
Document where torchtitan writes its per-step loss (its `[metrics]` log dir / W&B) and how to extract a `{"loss": ...}` jsonl. (Optional sanity: run the same config on `--backend megatron` and eyeball the two curves side by side — informational only.)

- [ ] **Step 3: Hand off to the operator**

Do **not** run GPU work in this harness. Post the `codexlog` commands and ask the operator to run them and report the curves (and the startup param counts). Tighten the health thresholds in a follow-up once real curves are in hand.

- [ ] **Step 4: Commit (test + runbook only)**

```bash
git add tests/numerics/test_titan_training_health.py docs/superpowers/runbooks/2026-05-30-torchtitan-training-health.md
git commit -m "test(torchtitan): M3 functional training-health gate + per-family runbook"
```

---

## Task 13: Docs, SPEC cross-reference, and CHANGELOG

**Files:**
- Modify: `README.md` (quick-start: mention `--backend torchtitan`)
- Modify: `SPEC.md` (§4 add a "4.4 torchtitan" pin note; §5 add a "backend axis" sentence)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: README quick-start note**

Under the per-optimizer wrapper examples, add:
```markdown
   # Same run on the torchtitan backend (native llama3/qwen3/deepseek_v3 + AdamW; experimental)
   scripts/train_adam.sh llama3 --backend torchtitan base/scale=300m --dry-run
   scripts/train_adam.sh llama3 --backend torchtitan base/family=qwen3 base/scale=300m --dry-run
```

- [ ] **Step 2: SPEC.md cross-reference**

Add a short `### 4.4 torchtitan` subsection pointing at `docs/torchtitan_pin.md` and the design spec, and one sentence in §5 noting `backend ∈ {megatron, torchtitan}` is a first-class field (default megatron) that prefixes the run name and is recorded as `torchtitan_sha`.

- [ ] **Step 3: CHANGELOG entry**

Add an entry summarizing: vendored torchtitan v0.2.2; `--backend torchtitan` driving native llama3/qwen3/deepseek_v3 with AdamW; same-corpus Megatron-indexed dataloader; Megatron-only knobs warn-and-skipped; M3 functional training-health gate pending operator run.

- [ ] **Step 4: Run the full unit suite**

Run: `pytest tests/unit tests/integration -q`
Expected: all green (GPU/numerics tests skip without a GPU).

- [ ] **Step 5: Commit**

```bash
git add README.md SPEC.md CHANGELOG.md
git commit -m "docs(torchtitan): document the torchtitan backend, pin, and backend axis"
```

---

## Definition of done

- From the slm side, the only change between backends is `--backend torchtitan`; the same six-axis yaml runs on both. (Tasks 2–3, 10)
- `third_party/torchtitan` pinned at v0.2.2 (`73a0e6979`); `docs/torchtitan_pin.md` + `.gitmodules` updated. (Task 1)
- `backend` field (default megatron), `torchtitan_sha` recorded, run-name disambiguated. (Tasks 2–3)
- `python -m launchers.train_torchtitan ... --dry-run` emits a valid `torchtitan.toml` + `torchrun -m torchtitan.train` command. (Tasks 5–7)
- `slm_<family>` TrainSpec (cloned from torchtitan's **native** llama3/qwen3/deepseek_v3) + slm-sized flavor + WSD scheduler + Megatron-indexed dataloader, all registered via the import hook with **no edits** to vendored torchtitan; Megatron-only knobs warn-and-skip. (Tasks 6, 8, 9)
- torchtitan reads the same `.bin/.idx` corpus; batches byte-identical to Megatron as the correctness check. (Task 9 — **M2 gate**)
- `scripts/train_adam.sh --backend torchtitan` routes correctly for all three families; non-AdamW wrappers reject torchtitan cleanly. (Task 10)
- torchtitan runs share the W&B project/run-name. (Task 11)
- Each family trains with a healthy, decreasing loss curve (no NaN). (Task 12 — **M3 gate**, operator-run)
- All CPU unit + integration tests green; docs updated. (Task 13)
