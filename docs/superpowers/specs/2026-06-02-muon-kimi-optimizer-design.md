# `muon_kimi` Optimizer Integration — Design

**Date:** 2026-06-02
**Status:** Approved design, ready for implementation plan
**Scope:** Single-GPU dev only (DP=TP=PP=1, no distributed optimizer). Add a
new `muon_kimi` optimizer that uses the **exact** Kimi/Moonlight-style Muon
implementation from the user's GaLore fork
([`/lustre/fast/fast/zqiu/tmp/GaLore/MUON/muon_kimi.py`](/lustre/fast/fast/zqiu/tmp/GaLore/MUON/muon_kimi.py)),
wired into the slm-research Megatron training loop and selectable via an
`experiment=optim/muon_kimi` config. Multi-GPU / tensor-parallel parity is
explicitly **out of scope** (the vendored optimizer has no distributed
collectives).

## Motivation

slm-research already has a Muon path (`optim.type: muon_hybrid`,
[`muon_hybrid.yaml`](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/muon_hybrid.yaml)),
but it routes through Megatron-Core's `TensorParallelMuon` →
`emerging_optimizers`. We want a second, independent variant that runs the
**user's own reference Muon code verbatim** — the Keller-Jordan-derived,
Moonlight/Kimi-style optimizer — so dev runs on the 60m model can be compared
against the emerging_optimizers path and against the user's GaLore results
without any risk of re-implementation drift.

## The source optimizer

[`MUON/muon_kimi.py`](/lustre/fast/fast/zqiu/tmp/GaLore/MUON/muon_kimi.py) (~200
lines, MIT, adapted from `github.com/KellerJordan/Muon`) is a standard
`torch.optim.Optimizer`:

- `zeropower_via_newtonschulz5(G, steps)` — `@torch.compile`'d quintic
  Newton-Schulz (coeffs `3.4445, -4.7750, 2.0315`), bf16 internally.
- `Muon.__init__(lr, wd, muon_params, momentum=0.95, nesterov=True,
  ns_steps=5, adamw_params, adamw_betas=(0.9,0.95), adamw_eps=1e-8)` — takes
  two param lists, sets a per-param `state[p]["use_muon"]` flag at init.
- `adjust_lr_for_muon(lr, shape)` → `lr * 0.2 * sqrt(max(d_out, d_in))` (the
  Moonlight RMS-matching scale).
- `step()`: Muon branch (SGD-momentum + nesterov → NS orthogonalize →
  decoupled WD `p*(1-lr*wd)` → `p -= adjusted_lr * u`) for `use_muon` params;
  an inline AdamW branch for the rest.
- **No `torch.distributed` calls in `step()`** — single-process only. This is
  the hard constraint that fixes the scope at single-GPU.

### GaLore recipe being reproduced

In the GaLore training loop
([`torchrun_main_normalized.py:926-1005`](/lustre/fast/fast/zqiu/tmp/GaLore/torchrun_main_normalized.py))
Muon is applied to the `weight` of every `nn.Linear` under `attn`/`mlp`;
everything else (embeddings, lm_head, norms, biases) goes to the internal
AdamW. The intermediate `lr:0.02` muon param-group is **discarded** by
`Muon.__init__` (it flattens `muon_params + adamw_params` into one group at the
single `defaults["lr"]`), so both sides share one base LR and the muon side is
scaled only by `adjust_lr_for_muon`. Therefore `muon_kimi` exposes a **single
`optim.lr`**, not the separate muon/adam LRs that `muon_hybrid` carries.

## Decision: vendor verbatim + POET-style routing patch

The repo already integrates a non-native optimizer (POET) by **patching**
`get_megatron_optimizer` and routing `--slm-optimizer poet` to a custom builder
([`poet_optimizer_setup.py`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py)).
`muon_kimi` mirrors this exactly. The vendored file is **not edited**.

*Alternative rejected:* port the Kimi math into the existing
`TensorParallelMuon`. Rejected because the user explicitly wants "the kimi from
here," and a reimplementation risks subtle drift from the reference.

### Key enabling finding: per-param state survives the bf16 wrapper

Megatron wraps bf16 optimizers in `Float16OptimizerWithFloat16Params`, which
replaces each model param with an fp32 master copy. Critically it **transfers
the inner optimizer's per-param state** to the master param —
[`optimizer.py:688-689`](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/optimizer/optimizer.py#L688-L689):

```python
if param in self.optimizer.state:
    self.optimizer.state[main_param] = self.optimizer.state.pop(param)
```

So the `state[p]["use_muon"]` flag set in `Muon.__init__` is carried onto the
master params, and `step()` reads `self.state[master_p]["use_muon"]` correctly.
This is why the optimizer can be vendored **verbatim** and simply wrapped.

## Architecture

### Unit 1 — vendored optimizer

`third_party/muon_kimi/muon_kimi.py` — byte-for-byte copy of the GaLore file
(keeps its Keller-Jordan MIT attribution header; add a one-line provenance note
naming the GaLore source path and an MIT `LICENSE` alongside). `__init__.py`
re-exports `Muon`, `zeropower_via_newtonschulz5`.

### Unit 2 — builder

`src/optim/muon_kimi.py`: `get_megatron_muon_kimi_optimizer(config,
model_chunks, *, config_overrides=None, use_gloo_process_groups=True)`:

1. **Guards:** raise on `config.use_distributed_optimizer` and on TP size > 1
   (`get_tensor_model_parallel_world_size() > 1`) and on `config.fp16` — Muon
   has no tensor-parallel Newton-Schulz and no loss-scaler support. (DP > 1
   would technically work since DDP all-reduces grads pre-step, but it is out
   of the chosen single-GPU dev scope and not guarded — documented only.)
2. **Param split** (reusing the classification the native muon path uses,
   [`muon.py:283-302`](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/optimizer/muon.py#L283-L302)):
   `muon_params` = 2-D params with `not is_embedding_or_output_parameter`;
   `adamw_params` = everything else. With `unfuse_qkv/unfuse_fc1` on, each
   attn/mlp sub-weight is its own clean 2-D matrix — matching GaLore's
   per-`nn.Linear` selection.
3. **Construct:** `Muon(lr=config.lr, wd=config.weight_decay,
   muon_params=muon_params, adamw_params=adamw_params,
   momentum=config.muon_momentum, nesterov=config.muon_use_nesterov,
   ns_steps=config.muon_num_ns_steps,
   adamw_betas=(config.adam_beta1, config.adam_beta2),
   adamw_eps=config.adam_eps)`.
4. **Wrap:** `Float16OptimizerWithFloat16Params(muon, config, grad_scaler=None,
   init_state_fn)` when `config.bf16`, else `FP32Optimizer(...)`. Return the
   wrapped optimizer directly (no `ChainedOptimizer` — the single Muon instance
   owns both the muon and AdamW updates internally).
5. **`init_state_fn(opt, config=None)`:** lazily create `momentum_buffer` for
   `use_muon` params and `step/moment1/moment2` for the rest, so torch_dist
   checkpointing has a state to serialize. (Resume is not a dev requirement but
   the function is cheap and matches the native muon path's pattern.)

### Unit 3 — routing patch

`src/patches/muon_kimi_optimizer_setup.py` — register a patch (targets
`megatron.training.training.get_megatron_optimizer_config` /
`get_megatron_optimizer`) that, when `args.slm_optimizer == "muon_kimi"`, sets
`config.slm_optimizer = "muon_kimi"` and reroutes the optimizer-builder call to
Unit 2. (The `muon_*` hyperparameters are already first-class `OptimizerConfig`
fields populated from the standard `--muon-*` args, so unlike POET no manual
attribute copy is needed beyond the routing flag.)

### Unit 4 — CLI / arg plumbing

- Add `"muon_kimi"` to the `--slm-optimizer` choices in
  [`pretrain_gpt_slm.py:27`](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L27).
- Add a `muon_kimi` branch to `_optimizer_args`
  ([`megatron_args.py:188`](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L188))
  emitting: `--optimizer adam`, `--slm-optimizer muon_kimi`, `--muon-momentum`,
  `--muon-num-ns-steps`, `--muon-use-nesterov`, and `--adam-beta1/2`,
  `--adam-eps`. (Routing through `--optimizer adam` keeps Megatron off its
  native `--optimizer muon` dispatch; the patch takes over.)

### Unit 5 — experiment config

`configs/experiments/optim/muon_kimi.yaml` (`# @package _global_`):

```yaml
experiment:
  name: muon_kimi
  family: optim
  patches: [model_unfuse_linears, training_log_eta, wandb_metric_normalize]
optim:
  type: muon_kimi
  lr: 1.0e-3
  weight_decay: 0.1
  muon_momentum: 0.95
  muon_use_nesterov: true
  muon_num_ns_steps: 5
  adam:
    betas: [0.9, 0.95]
    eps: 1.0e-8
base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

(Exact key layout to match how `_optimizer_args` reads betas/eps; finalized in
the plan.)

### Unit 6 — dev launcher

`scripts/train_muon_kimi_dev.sh` — clone of
[`train_muon_dev.sh`](/lustre/fast/fast/zqiu/slm-research/scripts/train_muon_dev.sh)
with `experiment=optim/muon_kimi`; keeps the shared dev conventions
(`wandb.project=slm-zeju-dev`, `training_regime=ablation_40x`,
`tie_embeddings=false`, 1024/128 batch, 60m default scale).

## Hyperparameter mapping

| muon_kimi.yaml            | Megatron arg            | `Muon(...)` param  | value   |
|---------------------------|-------------------------|--------------------|---------|
| `optim.lr`                | `--lr`                  | `lr`               | 1.0e-3  |
| `optim.weight_decay`      | `--weight-decay`        | `wd`               | 0.1     |
| `optim.muon_momentum`     | `--muon-momentum`       | `momentum`         | 0.95    |
| `optim.muon_use_nesterov` | `--muon-use-nesterov`   | `nesterov`         | true    |
| `optim.muon_num_ns_steps` | `--muon-num-ns-steps`   | `ns_steps`         | 5       |
| `optim.adam.betas`        | `--adam-beta1/2`        | `adamw_betas`      | 0.9,0.95|
| `optim.adam.eps`          | `--adam-eps`            | `adamw_eps`        | 1e-8    |

## Verification

- `python -m py_compile` on the three new `.py` files; import the builder under
  the slm_env venv.
- CPU dry-run: `bash scripts/train_muon_kimi_dev.sh llama3 --dry-run ...`
  resolves a command containing `--slm-optimizer muon_kimi`, `--optimizer adam`,
  the 60m dims (18 layers, seq 256), `--muon-use-nesterov`, and
  `--wandb-project slm-zeju-dev`.
- Extend [`tests/unit/test_train_scripts.py`](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_train_scripts.py)
  with a `train_muon_kimi_dev.sh` smoke (asserts `--slm-optimizer` + `muon_kimi`).
- **Actual 1-GPU training run is the user's** (per global GPU policy). The
  builder + dry-run prove wiring; the loss curve is validated on the cluster.

## Risks & mitigations

- **bf16 master-weight / `use_muon` interaction** — resolved (see finding
  above); covered by an import+construct smoke if a CPU stub model is feasible.
- **`@torch.compile` on the NS kernel** — the vendored file compiles
  `zeropower_via_newtonschulz5`; if compile is problematic in the slm runtime,
  it degrades to eager and is the user's to tune. No design change.
- **Param classification mismatch vs GaLore** — GaLore selects by module name
  (`attn`/`mlp` `nn.Linear`); we select by `is_embedding_or_output_parameter` +
  2-D shape. With unfuse on these are equivalent for the llama3 dev model;
  noted as the one place to eyeball during the first run.
