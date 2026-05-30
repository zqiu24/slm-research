# torchtitan v0.2.2 extension API notes (pin 73a0e6979)

Single source of truth for the exact symbols the slm torchtitan backend depends
on. Verified by reading the **pinned** source under `third_party/torchtitan/`
(tag v0.2.2). If a bump moves any symbol below, update `src/titan_ext/`,
`src/utils/torchtitan_args.py`, and this doc together (see
`docs/torchtitan_pin.md` bump procedure).

> **Top correction vs. the original plan (READ THIS):**
> `register_train_spec(train_spec: TrainSpec)` takes **ONE** argument and reads
> `train_spec.name`. `TrainSpec` has a `name: str` field (first field). So to
> register an slm spec you must set `name="slm_<family>"` on the spec
> (`dataclasses.replace(base, name=..., model_args=...)`) and call
> `register_train_spec(spec)` — NOT `register_train_spec(name, spec)`.

## 1. Entry & config

- **Entry:** `python -m torchtitan.train` → `torchtitan/train.py`, class `Trainer`.
  Run via `torchrun -m torchtitan.train --job.config_file <toml>` + dotted overrides.
- **JobConfig:** `torchtitan/config/job_config.py` (one dataclass per TOML section).
- **TOML → JobConfig:** `--job.config_file <path>` loads the TOML; dotted CLI
  flags (`--training.steps 100`) override individual fields.

Exact dotted keys we emit (section → fields, with v0.2.2 defaults):

- **[job]** (`Job`): `config_file`, `dump_folder="./outputs"`, `description`, `print_args`.
- **[model]** (`Model`): `name="llama3"`, `flavor="debugmodel"`, `hf_assets_path`,
  `converters=[]`, `print_after_load`. **NO raw dims** — model dimensions live in
  the registered `model_args` flavor, not in TOML.
- **[training]** (`Training`): `dataset="c4_test"`, `dataset_path=None`,
  `local_batch_size=8`, `global_batch_size=-1`, `seq_len=2048`, `max_norm=1.0`,
  `steps=10000`, `mixed_precision_param="bfloat16"|"float32"`,
  `mixed_precision_reduce="float32"`, **`seed: int | None = None`**, `deterministic=False`.
  → **seed lives at `[training].seed`** (NOT top-level, NOT `[job]`).
  → has BOTH `local_batch_size` and `global_batch_size`.
- **[optimizer]** (`Optimizer`): `name="AdamW"`, `lr=8e-4`, `beta1=0.9`,
  `beta2=0.95`, `eps=1e-8`, `weight_decay=0.1`,
  `implementation="for-loop"|"foreach"|"fused"`, `early_step_in_backward`.
- **[lr_scheduler]** (`LRScheduler`): `warmup_steps=200`,
  `decay_ratio: float | None = None`, `decay_type="linear"|"sqrt"|"cosine"`
  (default `"linear"`), `min_lr_factor=0.0`. **No `decay_steps` key.**
  → `decay_ratio` is a **FRACTION of total steps** spent decaying (None ⇒ decay
  over all steps after warmup; no stable phase).
- **[parallelism]** (`Parallelism`): `data_parallel_replicate_degree=1`,
  `data_parallel_shard_degree=-1`, `tensor_parallel_degree=1`,
  `pipeline_parallel_degree=1`, `context_parallel_degree=1`,
  `expert_parallel_degree=1`, `expert_tensor_parallel_degree=1`,
  `enable_async_tensor_parallel`, `disable_loss_parallel`, `pipeline_parallel_schedule="1F1B"`.
- **[metrics]** (`Metrics`): `log_freq=10`, `enable_tensorboard=False`,
  `enable_wandb=False`, `save_tb_folder="tb"`, `disable_color_printing`,
  `save_for_all_ranks`. **No project / run-name field** → W&B project + run name
  come from env `WANDB_PROJECT` / `WANDB_NAME` (and `WANDB_MODE`/`WANDB_ENTITY`).
- **[checkpoint]** (`Checkpoint`): `enable=False`, `folder="checkpoint"`,
  `interval=500`, `last_save_model_only`, `export_dtype`, …
- **[experimental]** (`Experimental`): **`custom_import: str = ""`** (a module
  path imported before train-spec lookup) and a separate `custom_args_module`.

## 2. Runtime extension hooks (NO vendored edits)

`TrainSpec` dataclass fields (in order, `torchtitan/protocols/train_spec.py`):
`name`, `model_cls`, `model_args`, `parallelize_fn`, `pipelining_fn`,
`build_optimizers_fn`, `build_lr_schedulers_fn`, `build_dataloader_fn`,
`build_tokenizer_fn`, `build_loss_fn`, `build_metrics_processor_fn` (=None),
`build_validator_fn` (=None), `state_dict_adapter` (=None).

- `model_args: Mapping[str, BaseModelArgs]` — the flavor registry (flavor-name → args).
- Methods: `get_args(self, flavor) -> BaseModelArgs` (returns `model_args[flavor]`),
  `get_loss_fn(self)`.

Exact signatures:
```python
def register_train_spec(train_spec: TrainSpec) -> None:
    if train_spec.name in _train_specs:
        raise ValueError(f"Model {train_spec.name} is already registered.")
    _train_specs[train_spec.name] = train_spec

def get_train_spec(name: str) -> TrainSpec:
    if name not in _train_specs:
        raise ValueError(f"Model {name} is not registered.")
    return _train_specs[name]
```
- `register_train_spec` → **ONE arg**, reads `spec.name`, **raises on duplicate name**.
- `get_train_spec` → raises if name is **not** registered. (Idempotency guard:
  `try: get_train_spec(slm_name); return  # already there  except ValueError: register…`.)

How `train.py` selects + imports (in `Trainer.__init__`, in this order):
```python
if job_config.experimental.custom_import:
    importlib.import_module(job_config.experimental.custom_import)   # runs FIRST
...
self.train_spec = train_spec_module.get_train_spec(job_config.model.name)
self.model_args = train_spec.get_args(job_config.model.flavor)
```
→ Our `--experimental.custom_import src.titan_ext` import runs **before** the
`get_train_spec(model.name)` lookup, so registering `slm_<family>` on import works.

Native llama3 registration (shows the real arg shape):
```python
register_train_spec(TrainSpec(name="llama3", model_cls=Transformer,
    model_args=llama3_configs, parallelize_fn=parallelize_llama, ...))
```

## 3. Model flavor / args

- **llama3** — `TransformerModelArgs` (`models/llama3/model/args.py`):
  `dim, n_layers, n_heads, n_kv_heads(=None), vocab_size(=-1), multiple_of(=256),
  ffn_dim_multiplier(=None), norm_eps, rope_theta, max_seq_len, depth_init,
  use_flex_attn, attn_mask_type, eos_id`. **No `head_dim`, `hidden_dim`, or
  `ffn_hidden_size`** — FFN derived from `dim`×`ffn_dim_multiplier`/`multiple_of`;
  embeddings **untied** (no tie flag). Registry `llama3_configs`:
  `debugmodel, debugmodel_flex_attn, 8B, 70B, 405B`.
- **qwen3** — `Qwen3ModelArgs` (`models/qwen3/model/args.py`): like llama3 PLUS
  explicit `head_dim`, explicit `hidden_dim` (FFN width), `qk_norm`, and
  `enable_weight_tying`. No `ffn_dim_multiplier`/`multiple_of`. Registry
  `qwen3_configs`: `debugmodel, 0.6B, 1.7B, 4B, 8B, 14B, 32B`.
- **deepseek_v3** — `DeepSeekV3ModelArgs` (`models/deepseek_v3/model/args.py`):
  structurally different — `dim, inter_dim, moe_inter_dim, n_layers,
  n_dense_layers, n_heads, moe_args, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
  qk_rope_head_dim, v_head_dim, …`. **NO `n_kv_heads`, NO `head_dim`, NO
  `ffn_dim_multiplier`.** Smallest flavor `debugmodel` (dim=256, n_layers=6).
  Registry `deepseekv3_configs`: `debugmodel, 16B, 236B, 671B`.
  → The shared "clone + override dense dims" path does **not** apply; M1 runs
  deepseek_v3 on a native flavor as-is (`base.model.titan_flavor`).

Add a flavor at runtime: copy `spec.model_args` (a dict), insert
`slm_<scale>` = `dataclasses.replace(template, **overrides)`, and
`dataclasses.replace(base, name="slm_<family>", model_args=<new dict>)`.

## 4. Dataloader component

- `torchtitan/components/dataloader.py`: `BaseDataLoader(Stateful, ABC)`;
  `ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader)`.
- `ParallelAwareDataloader.__init__(self, dataset, dp_rank, dp_world_size,
  batch_size, collate_fn=None)` — `batch_size` required (kwarg accepted).
- Native builder `build_hf_dataloader(dp_world_size, dp_rank, tokenizer,
  job_config, infinite=True) -> ParallelAwareDataloader`.
- `train.py` calls `build_dataloader_fn(dp_world_size=..., dp_rank=...,
  tokenizer=..., job_config=...)` (keyword), so our `build_dataloader(*,
  dp_world_size, dp_rank, tokenizer, job_config)` is signature-compatible.

## 5. LR scheduler component

- `torchtitan/components/lr_scheduler.py`:
  `build_lr_schedulers(optimizers, lr_scheduler_config, training_steps) ->
  LRSchedulersContainer`. `train.py` calls
  `build_lr_schedulers_fn(self.optimizers, job_config.lr_scheduler,
  job_config.training.steps)`.
- It **natively implements WSD**: `linear_warmup_stable_decay(current_step,
  warmup_steps, stable_steps, decay_steps, min_lr_factor, decay_type)` with
  `decay_steps = round(training_steps * decay_ratio)` and
  `stable_steps = training_steps - warmup_steps - decay_steps`. `decay_type ∈
  {linear, sqrt, cosine}`.
  → **Primary path is config-only** (emit the `[lr_scheduler]` keys; keep
  `base.build_lr_schedulers_fn`). The standalone `wsd_lr_multiplier` lambda in
  `src/titan_ext/lr_scheduler.py` is a tested fallback only.
