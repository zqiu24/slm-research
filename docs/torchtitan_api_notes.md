# torchtitan v0.2.2 extension API notes (pin 73a0e6979)

Single source of truth for the exact symbols the slm torchtitan backend depends
on. Verified by reading the **pinned** source under `third_party/torchtitan/`
(tag v0.2.2). If a bump moves any symbol below, update `src/titan_ext/`,
`src/utils/torchtitan_args.py`, and this doc together (see
`docs/torchtitan_pin.md` bump procedure).

> **Two facts that the original plan guessed and we verified — code MUST follow these:**
> 1. `register_train_spec(name: str, train_spec: TrainSpec)` takes **TWO** args.
>    `TrainSpec` has **NO `name` field** (its first field is `model_cls`). The
>    name is the registry key, passed separately. So register an slm spec with
>    `register_train_spec("slm_<family>", dataclasses.replace(base, model_args=...))`.
>    It raises `ValueError` if the name is already registered (hence the
>    idempotency guard below).
> 2. **`seed` lives at `[debug].seed`** (dataclass `Debug`), NOT `[training].seed`
>    and NOT top-level. Emit it under `[debug]`.

## 1. Entry & config

- **Entry:** `python -m torchtitan.train` → `torchtitan/train.py`, class `Trainer`
  (`main(trainer_class)` builds a `ConfigManager`, parses args, runs `Trainer`).
  Launch via `torchrun -m torchtitan.train --job.config_file <toml>` + dotted overrides.
- **JobConfig:** `torchtitan/config/job_config.py` (one dataclass per TOML section);
  parsed by `torchtitan/config/manager.py` `ConfigManager` (uses `tyro` — unknown
  keys are rejected, so emit only real fields).
- **TOML → JobConfig:** `--job.config_file <path>` loads the TOML; dotted CLI
  flags (`--training.steps 100`) override individual fields.

Exact dotted keys we emit (section → fields, with v0.2.2 defaults):

- **[job]** (`Job`): `config_file=None`, `dump_folder="./outputs"`,
  `description="default job"`, `print_config=False`, `custom_config_module=""`.
- **[model]** (`Model`): `name="llama3"`, `flavor="debugmodel"`,
  `hf_assets_path="./tests/assets/tokenizer"`, `tokenizer_path=None`,
  `converters=[]`, `print_after_conversion=False`. **No raw dims** — model
  dimensions live in the registered `model_args` flavor, not in TOML.
- **[training]** (`Training`): `dataset="c4_test"`, `dataset_path=None`,
  `local_batch_size=8`, `global_batch_size=-1`, `seq_len=2048`, `max_norm=1.0`,
  `steps=10000`, `dtype="float32"`, `mixed_precision_param="bfloat16"`,
  `mixed_precision_reduce="float32"`, `gc_freq=50`, `enable_cpu_offload=False`.
  → has BOTH `local_batch_size` and `global_batch_size`. **NO `seed` field here.**
- **[optimizer]** (`Optimizer`): `name="AdamW"`, `lr=8e-4`, `beta1=0.9`,
  `beta2=0.95`, `eps=1e-8`, `weight_decay=0.1`,
  `implementation="for-loop"|"foreach"|"fused"` (default `"fused"`),
  `early_step_in_backward=False`.
- **[lr_scheduler]** (`LRScheduler`): `warmup_steps=200`, `total_steps=None`,
  `decay_ratio: float | None = None`, `decay_type="linear"|"sqrt"|"cosine"`
  (default `"linear"`), `min_lr_factor=0.0`. **No `decay_steps` key.**
  → `decay_ratio` is the **FRACTION of total steps** spent decaying; `None` ⇒
  decay over all steps after warmup (no stable phase).
- **[parallelism]** (`Parallelism`): `data_parallel_replicate_degree=1`,
  `data_parallel_shard_degree=-1`, `tensor_parallel_degree=1`,
  `pipeline_parallel_degree=1`, `context_parallel_degree=1`,
  `expert_parallel_degree=1`, `expert_tensor_parallel_degree=1`,
  `enable_async_tensor_parallel=False`, `disable_loss_parallel=False`,
  `pipeline_parallel_schedule="1F1B"`.
- **[metrics]** (`Metrics`): `log_freq=10`, `enable_tensorboard=False`,
  `enable_wandb=False`, `save_tb_folder="tb"`, `disable_color_printing=False`,
  `save_for_all_ranks=False`. **No project / run-name field** → W&B project + run
  name come from env `WANDB_PROJECT` / `WANDB_NAME` (+ `WANDB_MODE`/`WANDB_ENTITY`).
- **[checkpoint]** (`Checkpoint`): `enable=False`, `folder="checkpoint"`,
  `interval=500`, `initial_load_path`, `initial_load_model_only`, … (40+ fields).
- **[debug]** (`Debug`): **`seed: int | None = None`** (+ `deterministic`, …).
  → **seed lives here.** Emit `[debug].seed`.
- **[experimental]** (`Experimental`): **`custom_import: str = ""`** (module path
  imported before train-spec lookup) and a separate (deprecated) `custom_args_module`.

## 2. Runtime extension hooks (NO vendored edits)

`TrainSpec` dataclass fields, IN ORDER (`torchtitan/protocols/train_spec.py`):
`model_cls`, `model_args`, `parallelize_fn`, `pipelining_fn`,
`build_optimizers_fn`, `build_lr_schedulers_fn`, `build_dataloader_fn`,
`build_tokenizer_fn`, `build_loss_fn`, `build_validator_fn=None`,
`build_metrics_processor_fn=None`, `state_dict_adapter=None`.
**There is NO `name` field.** `model_args: Mapping[str, BaseModelArgs]` is the
flavor registry (flavor-name → args).

Exact signatures (verbatim):
```python
def register_train_spec(name: str, train_spec: TrainSpec) -> None:
    global _extra_train_specs
    if name in _extra_train_specs:
        raise ValueError(f"TrainSpec {name} is already registered.")
    _extra_train_specs[name] = train_spec

def get_train_spec(name: str) -> TrainSpec:
    # user-defined specs win; else dispatch to torchtitan.models.<name>.get_train_spec()
    if name in _extra_train_specs:
        return _extra_train_specs[name]
    if name in _supported_models:
        return import_module(f"torchtitan.models.{name}").get_train_spec()
    elif name in _supported_experiments:
        return import_module(f"torchtitan.experiments.{name}").get_train_spec()
    raise ValueError(f"TrainSpec {name} is not registered.")
```
- `register_train_spec` → **TWO args** (`name`, `train_spec`); **raises on dup name**.
- `get_train_spec("llama3")` returns the native spec (dispatches to the model module).
- Idempotency guard: `try: get_train_spec(slm_name); return  except ValueError: register…`.

How `train.py` (`Trainer.__init__`) selects + imports, in this order:
```python
if job_config.experimental.custom_import:
    importlib.import_module(job_config.experimental.custom_import)   # runs FIRST
...
self.train_spec = train_spec_module.get_train_spec(job_config.model.name)
...
model_args = self.train_spec.model_args[job_config.model.flavor]
```
→ Our `--experimental.custom_import src.titan_ext` import runs **before** the
`get_train_spec(model.name)` lookup, so registering `slm_<family>` on import works.

Native llama3 registration shape (a module-level `get_train_spec()` returning):
```python
TrainSpec(model_cls=Transformer, model_args=llama3_args,
          parallelize_fn=parallelize_llama, pipelining_fn=pipeline_llm,
          build_optimizers_fn=build_optimizers, build_lr_schedulers_fn=build_lr_schedulers,
          build_dataloader_fn=build_text_dataloader, build_tokenizer_fn=build_hf_tokenizer,
          build_loss_fn=build_cross_entropy_loss, build_validator_fn=build_validator,
          state_dict_adapter=Llama3StateDictAdapter)
```

Build-call signatures `train.py` uses (so our overrides match):
```python
build_dataloader_fn(dp_world_size=..., dp_rank=..., tokenizer=..., job_config=...)
build_optimizers_fn(model_parts, job_config.optimizer, parallel_dims)
build_lr_schedulers_fn(optimizers, job_config.lr_scheduler, job_config.training.steps)
```

## 3. Model flavor / args

- **llama3** — `TransformerModelArgs` (`models/llama3/model/args.py`), fields:
  `dim=4096, n_layers=32, n_heads=32, n_kv_heads=None, vocab_size=128256,
  multiple_of=256, ffn_dim_multiplier=None, norm_eps=1e-5, rope_theta=10000,
  rope_scaling_args, max_seq_len=131072, depth_init=True, attn_type="sdpa",
  attn_mask_type="causal", eos_id=0`. **No `head_dim`, `hidden_dim`, or
  `ffn_hidden_size`** — FFN derived from `dim`×`ffn_dim_multiplier`/`multiple_of`;
  embeddings **untied** (no tie flag). Registry var `llama3_args`, keys:
  `debugmodel, debugmodel_flex_attn, debugmodel_varlen_attn, 8B, 8B_flex,
  8B_varlen, 70B, 405B`.
- **qwen3** — `Qwen3ModelArgs` (`models/qwen3/model/args.py`): like llama3 PLUS
  explicit `head_dim=128`, explicit `hidden_dim=3072` (FFN width), `qk_norm=True`,
  `enable_weight_tying=False`, and MoE fields (`moe_enabled`, `moe_inter_dim`,
  `moe_args`). No `ffn_dim_multiplier`/`multiple_of`. Registry var `qwen3_args`,
  keys: `debugmodel, 0.6B, 1.7B, 4B, 8B, 14B, 32B, debugmodel_moe, 30B-A3B,
  235B-A22B`.
- **deepseek_v3** — `DeepSeekV3ModelArgs` (`models/deepseek_v3/model/args.py`):
  structurally different — `dim=2048, inter_dim=10944, moe_inter_dim=1408,
  n_layers=27, n_dense_layers=1, n_heads=16, moe_args, q_lora_rank=0,
  kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
  vocab_size=102400, rope_theta, …`. **NO `n_kv_heads`, NO `head_dim`, NO
  `ffn_dim_multiplier`.** Registry var `deepseekv3_args`, keys:
  `debugmodel, debugmodel_flex_attn, 16B, 236B, 671B`; smallest `debugmodel` is
  `dim=256, inter_dim=1024, moe_inter_dim=256, n_layers=6, n_dense_layers=1,
  n_heads=16`.
  → The shared "clone + override dense dims" path does **not** apply; M1 runs
  deepseek_v3 on a native flavor as-is (`base.model.titan_flavor`, default
  `debugmodel`).

Add a flavor at runtime: copy `spec.model_args` (a dict), insert
`slm_<scale>` = `dataclasses.replace(template, **overrides)` (override only fields
the template `hasattr`), then `dataclasses.replace(base, model_args=<new dict>)`
and `register_train_spec("slm_<family>", <new spec>)`.

## 4. Dataloader component

- `torchtitan/components/dataloader.py`: `BaseDataLoader(Stateful, ABC)`;
  `ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader)`.
- `ParallelAwareDataloader.__init__(self, dataset, dp_rank, dp_world_size, **kwargs)`
  — note order is **`dp_rank` then `dp_world_size`**; `batch_size`/`collate_fn`
  pass through `**kwargs` to `StatefulDataLoader`.
- Native builder `build_text_dataloader(dp_world_size, dp_rank, tokenizer,
  job_config, infinite=True) -> ParallelAwareDataloader`.
- `train.py` calls `build_dataloader_fn(dp_world_size=..., dp_rank=...,
  tokenizer=..., job_config=...)` (keyword), so `build_dataloader(*,
  dp_world_size, dp_rank, tokenizer, job_config)` is signature-compatible.

## 5. LR scheduler component

- `torchtitan/components/lr_scheduler.py`:
  `build_lr_schedulers(optimizers, lr_scheduler_config, training_steps) ->
  LRSchedulersContainer`. `train.py` calls
  `build_lr_schedulers_fn(self.optimizers, job_config.lr_scheduler,
  job_config.training.steps)`.
- It **natively implements WSD** via `linear_warmup_stable_decay(current_step,
  warmup_steps, stable_steps, decay_steps, lr_decay_type, min_lr_factor)` with
  `decay_steps = round(training_steps * decay_ratio)` and
  `stable_steps = training_steps + 1 - warmup_steps - decay_steps`. `decay_type ∈
  {linear, sqrt, cosine}`.
  → **Primary path is config-only** (emit the `[lr_scheduler]` keys; keep
  `base.build_lr_schedulers_fn`). The standalone `wsd_lr_multiplier` lambda in
  `src/titan_ext/lr_scheduler.py` is a tested fallback only.
