# UnifiedLogger — Implementation Spec

> **Goal.** A single logging interface that multiplexes to **W&B**, **TensorBoard**, and **Aim** in parallel, so the training stack is not coupled to any one backend. No Megatron-LM source changes. All attribution (per-user, per-experiment-direction, per-capability) expressed via structured properties and tags.

---

## 1. Context

The codebase launches pretraining jobs via `launchers/launch_pretrain.py`, which composes Hydra configs and submits Slurm jobs (see `SPEC.md §5, §9`). Training itself runs a patched Megatron-LM (`third_party/megatron-lm/`, used as a submodule). Megatron-LM natively supports TensorBoard and W&B; it does **not** support Aim.

Rather than patching Megatron-LM to add Aim support, `UnifiedLogger` lives in `src/utils/unified_logger.py` and is driven from our launcher and training entry point. Megatron-LM's internal `tb_logger` and `wandb_logger` are left alone. Our logger wraps the outer lifecycle and receives metrics either (a) via a thin callback we register in training, or (b) by reading Megatron's TensorBoard event files after the fact.

Primary goal: any one backend can be added, removed, or fail at runtime without affecting training correctness or the other backends.

---

## 2. Requirements

### 2.1 Functional

1. **Multi-backend fan-out.** A single call — `logger.log_metric(name, value, step, context=...)` — writes to all enabled backends. Backends enabled via config (`logging.backends: [wandb, tensorboard, aim]`).
2. **Metric types.** Support scalars, dicts of scalars, histograms, and text. Image/audio/figure support optional (scalars and dicts are required).
3. **Hyperparameter logging.** `logger.log_hparams(cfg: dict)` records the full resolved config exactly once at run start, to every backend.
4. **Tags and properties.** `logger.add_tag(tag: str)` and `logger.set_property(key: str, value: Any)`. Tags are flat strings (e.g. `person:alice`, `capability:fp8`). Properties are structured key/value pairs that appear as filterable/sortable columns in each backend's UI.
5. **Run naming and identity.** The logger accepts a single `run_name`, `config_hash`, `group` (= `config_hash`), and `job_type` at init. These get propagated to each backend using that backend's conventions.
6. **Offline-first.** W&B always runs in `offline` mode on clusters (see `SPEC.md §8.1`). Aim uses its remote-tracking mode (HTTP/gRPC to our self-hosted Aim server) when online, and writes to a local `.aim` repo when offline. TensorBoard is inherently local — no online/offline distinction.
7. **Graceful degradation.** If a backend raises on init or on a log call, it is logged (stderr), disabled for the rest of the run, and does not propagate the exception. Training continues.
8. **Close on exit.** `logger.finish(status: str = "success")` flushes and closes every backend. Must be safe to call multiple times and in atexit handlers.

### 2.2 Non-functional

1. **No Megatron-LM source changes.** Zero modifications under `third_party/megatron-lm/`. If Megatron has to emit metrics into our logger, that happens via a function we install on a hook or via a monkey-patch applied in our launcher, not via editing upstream.
2. **Backend versions pinned.** Pin `wandb`, `aim`, and `tensorboard` (or `tensorboardX`) in our environment file. Document pinned versions in this file when updated.
3. **Zero mandatory network calls on hot path.** `log_metric` must not block on network I/O on any backend. W&B offline is local by definition; Aim must use its async/batched transport to the remote server.
4. **Per-user and per-experiment-direction attribution** must be expressible in all three backends, with the same semantic value, so results are comparable whichever UI a team member opens.

---

## 3. Attribution model (mandatory)

Every run carries the following attribution, regardless of backend. These are the filters team members use to answer *"show me Alice's 600M FP8 ablations from last week."*

| Field | Type | Example values | Purpose |
|---|---|---|---|
| `owner` | string | `alice`, `bob`, `zeju` | Who launched this run |
| `experiment_direction` | string | `muon-hybrid`, `fp8-scaling`, `moe-router-ablation` | Research direction (matches `docs/experiments/<name>.md`) |
| `base_family` | string | `llama3`, `qwen3`, `minicpm` | Model family (SPEC §5) |
| `base_scale` | string | `300m`, `600m`, `1_2b`, `2_4b`, `7b` | Scale rung |
| `config_hash` | string (hex) | `a3b4f2…` | Config identity, stable across seeds |
| `capabilities` | list of strings | `[fp8, moe]` | Capability requirements (SPEC §5) |
| `seed` | int | `42` | Reproducibility seed |
| `status` | string | `queued`, `running`, `promoted`, `failed` | Lifecycle state |
| `is_champion` | bool | `true` / `false` | Is this the current champion for its family+scale |

**These must round-trip through every backend.** The table below shows how each maps.

| Field | W&B | TensorBoard | Aim |
|---|---|---|---|
| `owner` | `tag="person:alice"` + `config.owner="alice"` | in run-dir name + `hparams/owner` text summary | `run['owner']='alice'` + `run.add_tag('person:alice')` |
| `experiment_direction` | `tag="direction:muon-hybrid"` + config field | in run-dir name + hparams | `run['direction']` + `run.add_tag('direction:muon-hybrid')` |
| `base_family`, `base_scale` | config fields + tags | hparams | `run['base_family']`, `run['base_scale']` + tags |
| `config_hash` | W&B `group` | filename prefix | `run['config_hash']` + tag |
| `capabilities` | multiple tags `capability:fp8`, `capability:moe` | hparams text | multiple tags + list property |
| `seed` | config field | hparams | `run['seed']` |
| `status` | tag (mutable) | — (TB has no tag mutation) | tag (mutable) |
| `is_champion` | tag `is_champion` if true | hparams | property + tag |

**Rule.** Anything used for filtering (`owner`, `experiment_direction`, `capabilities`, `status`, `is_champion`) must be in **tags** for W&B and Aim (since tag filtering is the UI primitive). Anything used for sorting/grouping (`base_family`, `base_scale`, `seed`, `config_hash`) must be in **properties/config**. Things used for both go in both.

---

## 4. Public interface

```python
# src/utils/unified_logger.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

@dataclass
class RunIdentity:
    """All identity/attribution for a run, backend-agnostic."""
    run_name: str                          # human-readable, unique per launch
    config_hash: str                       # hex, stable across seeds
    owner: str                             # e.g. 'alice'
    experiment_direction: str              # e.g. 'muon-hybrid'
    base_family: str                       # 'llama3' | 'qwen3' | 'minicpm'
    base_scale: str                        # '300m' | '600m' | ...
    capabilities: list[str]                # ['fp8', 'moe']
    seed: int
    project: str                           # W&B project / Aim experiment
    job_type: str                          # 'pretrain', 'finetune', 'eval'
    is_champion: bool = False
    extra_tags: list[str] = field(default_factory=list)

@dataclass
class LoggerConfig:
    """Which backends to enable and how to reach them."""
    backends: list[str]                    # subset of ['wandb', 'tensorboard', 'aim']
    wandb_entity: str | None = None
    wandb_mode: str = "offline"            # 'offline' | 'online' | 'disabled'
    wandb_dir: Path | None = None          # where offline runs go on disk
    tensorboard_dir: Path | None = None
    aim_repo: str | None = None            # e.g. 'aim://aim.nk-slm.com:43800' or local path
    aim_experiment: str | None = None      # defaults to identity.project

class UnifiedLogger:
    """Fan-out logger. Safe against individual backend failures."""

    def __init__(self, identity: RunIdentity, config: LoggerConfig,
                 resolved_config: dict): ...

    # Required calls
    def log_metric(self, name: str, value: float, step: int,
                   context: dict | None = None) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int,
                    context: dict | None = None) -> None: ...
    def log_hparams(self, hparams: dict) -> None: ...
    def add_tag(self, tag: str) -> None: ...
    def set_property(self, key: str, value: Any) -> None: ...
    def log_text(self, name: str, text: str, step: int | None = None) -> None: ...
    def finish(self, status: str = "success") -> None: ...

    # Optional calls (backends silently skip if unsupported)
    def log_histogram(self, name: str, values: Iterable[float],
                      step: int) -> None: ...
    def log_artifact(self, path: Path, name: str,
                     kind: str = "model") -> None: ...
```

### 4.1 Semantics

- `log_metric` is the hot-path call; must be non-blocking for all backends.
- `log_metrics` is `log_metric` over a dict; backends that support bulk (W&B) use bulk, others loop.
- `context` is Aim's native concept: `{'subset': 'train'}` or `{'subset': 'val'}`. For W&B and TensorBoard, the logger flattens it into the metric name — e.g. `loss` + `context={'subset': 'train'}` becomes `train/loss` in W&B/TB, but stays as `loss` with context `{subset:train}` in Aim. Document this flattening rule clearly.
- `step` is a global iteration counter. All backends must agree on this counter.
- `log_hparams` is called exactly once, at run start. Subsequent calls are a no-op with a warning.
- `finish(status)` must be idempotent and safe in atexit.

---

## 5. Backend adapters

Each backend is a class that conforms to an internal `Backend` protocol. The `UnifiedLogger` holds a dict `{name: Backend}` and fans out to each. All backend calls are wrapped in try/except; on exception, the backend is disabled (`self._backends.pop(name)`) with a single stderr warning.

```python
class _Backend:
    def init(self, identity: RunIdentity, config: LoggerConfig,
             resolved_config: dict) -> None: ...
    def log_metric(self, name: str, value: float, step: int,
                   context: dict | None) -> None: ...
    def log_hparams(self, hparams: dict) -> None: ...
    def add_tag(self, tag: str) -> None: ...
    def set_property(self, key: str, value: Any) -> None: ...
    def log_text(self, name: str, text: str, step: int | None) -> None: ...
    def finish(self, status: str) -> None: ...
```

### 5.1 `WandbBackend`

- `wandb.init(project=identity.project, entity=config.wandb_entity, name=identity.run_name, group=identity.config_hash, job_type=identity.job_type, tags=_build_wandb_tags(identity), config=resolved_config, mode=config.wandb_mode, dir=str(config.wandb_dir))`.
- `_build_wandb_tags(identity)` constructs: `["person:{owner}", "direction:{experiment_direction}", f"family:{base_family}", f"scale:{base_scale}"] + [f"capability:{c}" for c in capabilities] + (["is_champion"] if is_champion else []) + extra_tags`.
- `log_metric`: `wandb.log({_wandb_name(name, context): value}, step=step)`. `_wandb_name` flattens context: `context={'subset': 'train'}` → `train/{name}`.
- `add_tag`: `wandb.run.tags = wandb.run.tags + (tag,)`.
- `set_property`: `wandb.run.summary[key] = value`.
- `log_text`: `wandb.log({name: wandb.Html(text) if len(text) > 200 else text}, step=step)`.
- `finish`: `wandb.finish(exit_code=0 if status == "success" else 1)`.

### 5.2 `TensorBoardBackend`

Uses `torch.utils.tensorboard.SummaryWriter`. Run dir: `{config.tensorboard_dir}/{identity.config_hash}/seed{identity.seed}/{identity.run_name}`.

- `log_metric`: `writer.add_scalar(_tb_name(name, context), value, step)`. Same name flattening rule as W&B.
- `log_hparams`: `writer.add_hparams(_flatten_hparams(hparams), {})`. TB's hparam API is weak — see §7.3.
- `add_tag`: TB has no tag concept. The logger captures tags into a `{run_dir}/tags.txt` file and writes them as a text summary: `writer.add_text('tags', ', '.join(tags))`.
- `set_property`: similar to tags — write `{run_dir}/properties.json` and also `writer.add_text('properties', json.dumps(props, indent=2))`.
- `log_text`: `writer.add_text(name, text, step)`.
- `finish`: `writer.flush(); writer.close()`.

### 5.3 `AimBackend`

- `aim.Run(experiment=identity.project, repo=config.aim_repo)`.
- In `init`, set structured properties on the run:
  ```python
  run['owner'] = identity.owner
  run['direction'] = identity.experiment_direction
  run['base_family'] = identity.base_family
  run['base_scale'] = identity.base_scale
  run['config_hash'] = identity.config_hash
  run['capabilities'] = identity.capabilities
  run['seed'] = identity.seed
  run['is_champion'] = identity.is_champion
  run['job_type'] = identity.job_type
  run.name = identity.run_name
  run.description = resolved_config.get('experiment', {}).get('description', '')
  ```
- Tags: `run.add_tag(t)` for each of the tags built the same way as W&B (`_build_aim_tags` — same function body; tags are semantic, backend-agnostic).
- `log_metric`: `run.track(value, name=name, step=step, context=context or {})`. Aim uses context natively; do **not** flatten the name.
- `log_hparams`: `run['hparams'] = hparams` (Aim supports nested dicts).
- `add_tag`: `run.add_tag(tag)`.
- `set_property`: `run[key] = value`.
- `log_text`: `run.track(aim.Text(text), name=name, step=step or 0)`.
- `finish`: `run.close()` if the current Aim API supports it, else rely on atexit.

---

## 6. Integration with the launcher and Megatron

### 6.1 Launcher side (`launchers/launch_pretrain.py`)

Right after the existing config-hash computation (see SPEC §9, step 10), construct `RunIdentity` and `LoggerConfig` from the resolved Hydra config:

```python
identity = RunIdentity(
    run_name=cfg._derived.run_name,
    config_hash=cfg._derived.config_hash,
    owner=cfg.launch.owner,
    experiment_direction=cfg.experiment.name,
    base_family=cfg.base.family,
    base_scale=cfg.base.scale,
    capabilities=cfg._derived.capabilities,
    seed=cfg.seed,
    project=cfg.wandb.project,
    job_type=cfg.wandb.job_type,
    is_champion=cfg._derived.get("is_champion", False),
    extra_tags=cfg.wandb.get("extra_tags", []),
)

logger_cfg = LoggerConfig(
    backends=cfg.logging.backends,
    wandb_entity=cfg.wandb.entity,
    wandb_mode="offline" if cfg.cluster.wandb_offline else "online",
    wandb_dir=Path(cfg.cluster.wandb_dir),
    tensorboard_dir=Path(cfg.cluster.tensorboard_dir),
    aim_repo=cfg.logging.aim_repo,
)

# Serialize identity + logger_cfg into sbatch env so the training process
# can reconstruct them. Launcher itself does not open log backends.
```

The launcher does **not** construct the `UnifiedLogger`; it only serializes identity and config. The training process opens the logger so that W&B / Aim lifecycle is bound to the training process lifecycle.

### 6.2 Training-process side

In our training entry point (the script we invoke from the Slurm job, which internally calls into Megatron's `pretrain_gpt.main` or equivalent), we:

1. Deserialize `RunIdentity` and `LoggerConfig` from env vars.
2. Construct `UnifiedLogger`.
3. Install a metrics callback on Megatron.

Megatron-LM emits metrics through a few internal functions — notably `training_log()` in `megatron/training/training.py`, which receives an iteration's losses, grad norms, throughput, etc., and internally calls its own TensorBoard and W&B writers. We do **not** modify this function. Two clean ways to hook in, pick the one that fits the current Megatron version:

**Option A — monkey-patch `training_log`.** In our entry point, after importing Megatron but before calling into its main loop:

```python
import megatron.training.training as _mt
_orig_training_log = _mt.training_log

def _patched_training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate,
                          iteration, loss_scale, report_memory_flag, skipped_iter,
                          grad_norm, params_norm, num_zeros_in_grad):
    # call original first, so Megatron's own TB/W&B writers still fire
    out = _orig_training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate,
                             iteration, loss_scale, report_memory_flag, skipped_iter,
                             grad_norm, params_norm, num_zeros_in_grad)
    # now log to our unified logger
    metrics = {f"train/{k}": float(v) for k, v in loss_dict.items()}
    metrics["train/learning_rate"] = learning_rate
    metrics["train/grad_norm"] = grad_norm
    metrics["train/params_norm"] = params_norm
    unified_logger.log_metrics(metrics, step=iteration)
    return out

_mt.training_log = _patched_training_log
```

**Option B — register a post-iteration hook** if Megatron's current version exposes one (some forks do; check the pinned version). Prefer this over monkey-patching if available.

In both cases, **Megatron's own W&B / TensorBoard logging continues to run in parallel**. Our unified logger is additive. If W&B is enabled in both Megatron's config and our logger config, we end up with two W&B runs — so **disable Megatron's built-in W&B and TensorBoard logging** via Megatron's own flags (`--no-wandb`, omit `--tensorboard-dir`) and let `UnifiedLogger` be the sole logger. The monkey-patch shown above still intercepts `training_log`'s metric dict regardless of Megatron's own backends being on or off.

### 6.3 Evaluation hooks

Evaluation loops (perplexity on held-out, downstream probes) log through `unified_logger.log_metrics({...}, step=iteration)` directly from our eval code — which we own, so no hook needed.

---

## 7. Edge cases and pitfalls

### 7.1 Step monotonicity

W&B and TB both complain if `step` goes backward. The logger **does not** reorder or dedupe; it passes step through. Upstream (training code) is responsible for monotonic steps. Document this.

### 7.2 Aim context vs W&B/TB name-flattening

If a caller logs `loss` with `context={'subset': 'train'}` and later `loss` with `context={'subset': 'val'}`:

- In Aim: one metric `loss`, two series distinguished by context.
- In W&B / TB: two metrics, `train/loss` and `val/loss`.

Downstream analysis tools (`tools/monthly_table.py`, `tools/ladder_plot.py`) need to know this asymmetry. Recommendation: **when reading Aim, query `metric=loss, context={subset:X}`. When reading W&B/TB, query `metric=X/loss`.** Wrap this in a helper in `tools/read_runs.py`.

### 7.3 TensorBoard hparams are weak

`SummaryWriter.add_hparams` requires a flat dict of simple scalars and produces a separate sub-run, which is awkward. The backend additionally writes `hparams.json` to the run dir. Downstream tools should read `hparams.json` rather than the TB hparams plugin.

### 7.4 Aim server unavailable at run start

If `aim_repo='aim://aim.nk-slm.com:43800'` and the server is down, Aim's client will raise on `Run(...)`. `AimBackend` catches this, logs a warning, and downgrades to a local `.aim` repo inside `{cluster.aim_dir}/{config_hash}/`. A cron on a login node then syncs local `.aim` repos to the server when it's back up — same pattern as `tools/sync_wandb.py` for W&B.

### 7.5 Backend version drift

Pin:
- `wandb` — current stable series (record pinned version here when implementing).
- `aim` — current stable series (record pinned version here when implementing).
- `tensorboard` — current stable series.

Run a weekly smoke test that initializes all three backends with a dummy run and logs 5 scalars. If a pinned version fails, investigate before bumping.

### 7.6 Duplicate runs on retry

Each backend identifies runs differently:
- W&B uses `id=config_hash + seed + job_id` if we want to resume, or a fresh id for a new attempt.
- Aim creates a new `Run` each time `Run(...)` is called.
- TensorBoard's run identity is its run dir.

Our convention: **on a retry, reuse the same `run_name` but a new timestamp suffix**. The launcher sees preemption / requeue and sets `resume=True` in `LoggerConfig`; the logger passes `resume=True` to W&B and creates a child Aim run linked via `run.add_tag(f'parent:{previous_run_hash}')`.

---

## 8. Config surface

Add to `configs/base/logging.yaml` (new file):

```yaml
# configs/base/logging.yaml
logging:
  backends: [wandb, tensorboard, aim]   # any subset

  aim_repo: aim://aim.nk-slm.com:43800
  aim_local_fallback_dir: ${oc.env:HOME}/aim-local

  # wandb_* and tensorboard_dir are in cluster configs (cluster-specific paths)
```

Cluster configs (`configs/clusters/*.yaml`) keep `wandb_dir` and `tensorboard_dir`:

```yaml
cluster:
  wandb_offline: true
  wandb_dir: /scratch/${oc.env:USER}/wandb
  tensorboard_dir: /scratch/${oc.env:USER}/tb_logs
  aim_dir: /scratch/${oc.env:USER}/aim-local
```

Launch-time override to disable backends for debugging:

```bash
python launchers/launch_pretrain.py \
    experiment=muon_hybrid \
    base=llama3/600m \
    cluster=h800_cn \
    logging.backends='[tensorboard]'   # TB only, skip W&B and Aim
```

---

## 9. Testing

Required tests under `tests/unit/logging/`:

1. **`test_identity_to_tags.py`**: `_build_wandb_tags` and `_build_aim_tags` produce the same set of semantic tags from the same `RunIdentity`. Golden test with a fixed identity.
2. **`test_context_flattening.py`**: `_wandb_name('loss', {'subset': 'train'}) == 'train/loss'`. Same for `_tb_name`. Aim keeps name unchanged.
3. **`test_backend_failure_isolation.py`**: a `FailingBackend` that raises on every call must not prevent other backends from logging. Asserts logger state after failure.
4. **`test_finish_idempotent.py`**: two calls to `finish()` do not error.

Integration test under `tests/integration/`:

5. **`test_unified_logger_smoke.py`**: opens all three backends in a temp dir (W&B offline, Aim local repo, TB), logs 100 scalars over 10 steps, logs hparams, calls `finish`. Asserts the expected files/rows exist in each backend's storage.

---

## 10. Deliverables

A PR that includes:

1. `src/utils/unified_logger.py` — the public `UnifiedLogger`, `RunIdentity`, `LoggerConfig` classes, and the three backend adapters (`WandbBackend`, `TensorBoardBackend`, `AimBackend`).
2. `src/utils/wandb_helpers.py` updated so `build_tags(cfg)` delegates to `_build_wandb_tags(identity)` to avoid two sources of truth.
3. `configs/base/logging.yaml` with the schema in §8.
4. `configs/clusters/*.yaml` updated with `aim_dir` keys.
5. Launcher (`launchers/launch_pretrain.py`) updated to construct `RunIdentity` + `LoggerConfig` and serialize them into the training env, replacing the existing inline `wandb_cfg` dict.
6. Training entry point (`src/train/entrypoint.py` or wherever we invoke Megatron) updated with the monkey-patch / hook from §6.2, Megatron's own W&B and TB logging disabled via its flags.
7. All tests in §9 passing.
8. A short README at `src/utils/unified_logger.md` with a minimal usage example, and a pointer to this spec.

### 10.1 Out of scope for this PR

- Aim server deployment (handled separately — see ops docs).
- Sync cron for offline Aim local repos (follow-up PR, mirror of `tools/sync_wandb.py`).
- Migration of historical W&B runs into Aim.
- Rich media (images, audio, 3D). Scalar, dict, text, histogram only for v1.

---

## 11. Open questions for the implementer

1. **Does our pinned Megatron SHA expose a post-iteration hook?** If yes, use hook instead of monkey-patch (§6.2). Check `megatron/training/training.py` and `megatron/core/training_callback.py` (if it exists in this SHA).
2. **Does Aim's current stable release support `run.close()` explicitly**, or does cleanup rely on `__del__` / atexit? If the former, call it in `finish`; if the latter, document that `finish` is a best-effort flush.
3. **TensorBoard writer path collision on resume.** If a run resumes with the same `config_hash` + `seed`, the TB writer writes into the same directory. This is usually fine (append), but confirm with a test that step values don't collide.
