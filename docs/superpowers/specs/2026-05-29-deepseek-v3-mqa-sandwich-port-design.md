# DeepSeek-3Bv2 (MQA + sandwich-norm) — first-party port (Sub-project 1)

**Date:** 2026-05-29
**Status:** Approved design, ready for implementation plan
**Scope:** Make the Huawei `train_poet_huawei.sh` DeepSeek-3Bv2 architecture trainable in
first-party slm-research (Megatron 0.17 + `launchers.train_megatron`), with **no**
dependence on `poet_torch_huawei/` or its vendored Megatron 0.14.

## 1. Motivation & context

`scripts/train_poet_huawei.sh` runs an **isolated** vendored stack: Megatron-core 0.14.1 +
`poet_torch` + `pretrain_gpt_poet.py`, launched with its own bash scripts and `PYTHONPATH`.
The 2026-05-28 vendoring plan chose isolation precisely to avoid 0.14→0.17 API drift and
because first-party Megatron lacks sandwich-norm. We now want the **opposite**: the model
trainable through the normal first-party launcher so we stop depending on the vendored tree.

Two facts shape the work:

1. **It is a different DeepSeek than first-party has.** Huawei DeepSeek-3Bv2 is **MQA +
   sandwich-norm** (`num_query_groups=1`, `kv_channels=384`, `rotary_percent=0.25`,
   `--use-sandwich-norm` with `attn/ffn_post_norm_scale=0.03`). First-party `deepseek_v3`
   is **MLA, no sandwich-norm**. So this is a *new architecture variant*.
2. **Only two features are genuinely missing code in first-party Megatron 0.17:**
   sandwich-norm (small) and a stability/monitor suite (large, observability-only). The rest
   of the architecture maps onto config knobs first-party already exposes.

This spec covers **Sub-project 1**: the trainable architecture + sandwich-norm + recipe.
The **stability/monitoring suite is Sub-project 2** (separate spec) and is out of scope here.

## 2. Decisions (locked)

| Decision | Choice |
| --- | --- |
| Scope | Decompose; this spec = arch + sandwich-norm + recipe. Monitors deferred to Sub-project 2. |
| POET coupling | Architecture only; POET layers on top via `experiment=optim/poet` (the unfuse we built handles MQA qkv). |
| Fidelity | Functional equivalent (same arch/hyperparams against Megatron 0.17 conventions; not bit-for-bit). |
| Config structure | New family `deepseek_v3_mqa` + scale `deepseek_3bv2` (leave MLA `deepseek_v3` untouched). |
| Sandwich-norm wiring | Gated patch `sandwich_norm_apply` listed in experiment `patches:` (no-op unless `use_sandwich_norm`), mirroring `model_unfuse_linears`. |

## 3. Goals / non-goals

**Goals**
- A new family + scale that reproduces the Huawei DeepSeek-3Bv2 architecture under Megatron 0.17.
- A `sandwich_norm_apply` patch giving first-party Megatron the post-norm-before-residual
  behavior with `attn/ffn_post_norm_scale` init scaling.
- `megatron_args` emits every architecture flag the Huawei `MODEL_ARGS` set (reconciled list).
- The model trains end-to-end via `launchers.train_megatron` with `experiment=optim/adam`,
  and composes with `experiment=optim/poet` (incl. unfuse).
- A `scripts/train_deepseek.sh` thin entry + the WSD recipe.

**Non-goals**
- The stability/transformer/attention monitor suite (`--enable-*-monitor`, `--stability-*`) — Sub-project 2.
- Any change to the existing MLA `deepseek_v3` family.
- Any change to POET internals (orthogonal; already first-party).
- Bit-level reproduction of the 0.14 run; no edits to `third_party/Megatron-LM` or `poet_torch_huawei/`.
- MTP correctness validation beyond "Megatron's MTP runs" (it's a config knob first-party already emits).

## 4. Architecture mapping (Huawei MODEL_ARGS → first-party)

Source: `poet_torch_huawei/training_scripts/model_args/DeepSeek-3Bv2-sandwich-mqa-poet.yaml`.
Most flags map onto existing `base.model.*` knobs and `megatron_args` emission. The table
lists the **gaps** that need new config fields or new emission; everything not listed is
already emitted by first-party (`--num-layers`, `--hidden-size`, `--ffn-hidden-size`,
`--num-attention-heads`, `--swiglu`, `--disable-bias-linear`, `--qk-layernorm`, RMSNorm,
`--group-query-attention`, `--num-query-groups`, `--kv-channels`, MoE ffn/shared/dispatcher/
score-function/expert-bias, MTP, `--untie-embeddings-and-output-weights`, bf16).

| Huawei flag | First-party action |
| --- | --- |
| `--rotary-percent 0.25` | `megatron_args` currently hardcodes `1.0` → make config-driven `model.get("rotary_percent", 1.0)`. |
| `--use-sandwich-norm`, `--attn-post-norm-scale 0.03`, `--ffn-post-norm-scale 0.03` | New `base.model.use_sandwich_norm` / `attn_post_norm_scale` / `ffn_post_norm_scale`; emit flags; **sandwich_norm_apply** patch implements behavior. |
| `--moe-router-topk-scaling-factor 2.5`, `--moe-aux-loss-coeff 1e-4`, `--moe-router-bias-update-rate 1e-3`, `--moe-router-dtype fp32`, `--moe-permute-fusion`, `--moe-router-fusion`, `--moe-layer-recompute` | Reconcile MoE emission in `_model_args`; add any not already emitted (driven from `model.moe.*`). |
| `--make-vocab-size-divisible-by 3232` | Confirm/emit (likely a fixed value in `megatron_args` today); expose if it differs. |
| `--manual-gc`, `--manual-gc-interval 10`, `--cross-entropy-fusion-impl native`, `--no-rope-fusion` | Emit as fixed flags for this family (or training defaults); reconcile during planning. |
| `--init-method-std 0.006`, `--embedding-init-method-std 0.006` | `init_method_std: 0.006` in the family; add `embedding_init_method_std` emission if missing. |
| `--mtp-num-layers 1`, `--mtp-loss-scaling-factor 0.3` | **Currently emitted only inside the MLA block** in `_model_args`; must be decoupled so MTP emits for MQA too (with `--enable-experimental`). |
| MoE SequentialMLP (no `--moe-grouped-gemm`) + `--moe-token-dispatcher-type alltoall` | `moe.grouped_gemm: false`, `moe.token_dispatcher_type: alltoall` in the family. (POET needs non-grouped; plain Adam runs may flip `grouped_gemm: true` for speed — left as a knob.) |
| `--transformer-impl transformer_engine` | Family default `transformer_impl: transformer_engine` for plain runs; POET runs override to `local` (existing `poet_unfuse_te_impl` already forces this). |

The exact byte-level reconciliation (every flag emitted vs. the Huawei list) is an explicit
plan task: dump `build_megatron_args` for the new family and diff against the Huawei
`MODEL_ARGS`, adding any missing emission.

## 5. Components

### 5a. New family `configs/base/family/deepseek_v3_mqa.yaml`
MQA + sandwich + MoE-SequentialMLP variant. Key fields: `multi_latent_attention: false`,
`num_query_groups: 1`, `rotary_percent: 0.25`, `qk_norm: true`, `normalization: RMSNorm`,
`norm_epsilon: 1e-6`, `rotary_base: 10000`, `init_method_std: 0.006`, `untie_embeddings: true`,
sandwich fields (`use_sandwich_norm: true`, `attn_post_norm_scale: 0.03`,
`ffn_post_norm_scale: 0.03`), MTP (`mtp_num_layers: 1`, `mtp_loss_scaling_factor: 0.3`),
and `moe:` (64 experts, `router_load_balancing_type: seq_aux_loss`, `router_topk: 6`,
`router_score_function: sigmoid`, `router_enable_expert_bias: true`,
`router_bias_update_rate: 1e-3`, `router_dtype: fp32`, `aux_loss_coeff: 1e-4`,
`router_topk_scaling_factor: 2.5`, `token_dispatcher_type: alltoall`, `grouped_gemm: false`,
`permute_fusion: true`). A matching `docs/experiments/`-style note isn't required (families
have no doc-gate), but a header comment documents provenance.

### 5b. New scale `configs/base/scale/deepseek_3bv2.yaml`
Scale sets `num_layers: 12` (Huawei `--num-layers 12`),
`hidden_size: 1280`, `ffn_hidden_size: 7168`, `num_attention_heads: 16`, `head_dim: 384`
(→ `--kv-channels 384`; note head_dim ≠ hidden/heads, which Megatron allows),
`seq_length: 4096`, `moe.ffn_hidden_size: 896`, `moe.shared_expert_intermediate_size: 1792`,
`moe.layer_freq: "([0]*1+[1]*11)"`, `non_embedding_params` annotation.

### 5c. Sandwich-norm: `src/patches/sandwich_norm_apply.py` + `src/model/sandwich_layer.py`
Modeled on `src/patches/ngpt_apply_spec.py`:
1. **Args + config:** wrap `megatron.training.arguments` to register `--use-sandwich-norm`
   (store_true), `--attn-post-norm-scale` (float, default 1.0), `--ffn-post-norm-scale`
   (float, default 1.0), and stamp them onto the built `TransformerConfig`.
2. **Custom layer** (`src/model/sandwich_layer.py`): subclass first-party Megatron 0.17
   `TransformerLayer`. When `config.use_sandwich_norm`: in `__init__`, build
   `post_self_attn_layernorm` and `post_mlp_layernorm` (RMSNorm via the same norm impl the
   layer uses) and multiply their weights by the respective post-norm scale at init; in
   `forward`, apply the post-norm to the attention output and to the MLP output **before**
   the bias-dropout-residual add (matching Huawei `transformer_layer.py:606-609` /
   `:719-721`). No-op when `use_sandwich_norm` is false.
3. **Spec injection:** wrap `gpt_builder` so the GPT layer spec's `module` is the sandwich
   layer subclass (same hook point `ngpt_apply_spec` uses for `_get_transformer_layer_spec`).
4. **Wiring:** add `sandwich_norm_apply` to the `patches:` list of `optim/adam` (and
   `optim/poet`, `optim/muon_hybrid`) — no-op unless `use_sandwich_norm` is set, exactly like
   `model_unfuse_linears`.

**Risk flagged:** the patched `forward` must track the 0.17 `TransformerLayer.forward`
structure (insertion points for the post-norms); like `ngpt_apply_spec`, it is coupled to the
pinned Megatron and must be re-synced if Megatron is bumped. The plan validates with a
CPU forward test on a toy layer.

### 5d. `megatron_args` plumbing (`src/utils/megatron_args.py`)
- `rotary-percent` → `model.get("rotary_percent", 1.0)`.
- Emit `--use-sandwich-norm` / `--attn-post-norm-scale` / `--ffn-post-norm-scale` when
  `model.use_sandwich_norm`.
- Reconcile MoE emission (§4 table) from `model.moe.*`.
- Add `embedding_init_method_std`, `manual_gc`, `cross_entropy_fusion_impl`, `no_rope_fusion`,
  `make_vocab_size_divisible_by` emission as needed (planning reconciliation).

### 5e. Recipe + launch
- `scripts/train_deepseek.sh` mirroring `scripts/train_adam.sh`: defaults
  `base/family=deepseek_v3_mqa base/scale=deepseek_3bv2 experiment=optim/adam scheduler=wsd`,
  `training.save_enabled=true`, plus the WSD/LR/batch hyperparameters (lr 8.6e-4, min 7e-6,
  warmup 2000, wsd-decay 12000, train-iters 48000, GBS 1024, MBS 4, seq 4096). Exposes the
  ARCH/scale override pattern the other scripts use.
- WSD via the existing `configs/scheduler/wsd.yaml`.

### 5f. Tests (`tests/unit/`)
- **Config composition:** `base/family=deepseek_v3_mqa base/scale=deepseek_3bv2` resolves;
  asserts MQA (`num_query_groups=1`), `kv_channels=384`, sandwich fields, MoE dims.
- **megatron_args emission:** `--group-query-attention`, `--num-query-groups 1`,
  `--kv-channels 384`, `--rotary-percent 0.25`, `--use-sandwich-norm`,
  `--attn-post-norm-scale 0.03`, `--ffn-post-norm-scale 0.03`, and the reconciled MoE knobs.
- **rotary-percent regression:** existing families still emit `1.0` (config-driven default).
- **sandwich_norm patch:** registration (targets arguments + gpt_builder, distinct targets);
  CPU forward test on a toy `TransformerLayer`-like module asserting post-norm is applied to
  the sub-layer output before the residual add and that init scaling multiplies the norm weight.

## 6. Files

**Added:**
- `configs/base/family/deepseek_v3_mqa.yaml`
- `configs/base/scale/deepseek_3bv2.yaml`
- `src/patches/sandwich_norm_apply.py`
- `src/model/sandwich_layer.py`
- `scripts/train_deepseek.sh`
- `tests/unit/test_deepseek_v3_mqa_scale.py`, `tests/unit/test_sandwich_norm.py` (+ assertions in `test_megatron_args.py`)

**Modified:**
- `src/utils/megatron_args.py` (rotary-percent config-driven; sandwich + MoE emission)
- `configs/experiments/optim/{adam,poet,muon_hybrid}.yaml` (add `sandwich_norm_apply` to `patches:`)

**Not touched:** `third_party/Megatron-LM/`, `poet_torch_huawei/`, the existing
`configs/base/family/deepseek_v3.yaml` (MLA).

## 7. Open risks / validation gates

- **head_dim 384 with 16 heads** (query-proj 6144 ≠ hidden 1280) is unusual but valid in
  Megatron (`kv-channels` is independent of `hidden_size/heads`). Validate at model-build.
- **Sandwich-norm forward coupling** to Megatron 0.17 internals (re-sync risk) — CPU forward
  test + GPU smoke gate.
- **SequentialMLP vs grouped-gemm**: family defaults non-grouped (POET-compatible); a plain
  Adam run may flip `grouped_gemm: true`. Both must build.
- **MTP under first-party 0.17** with this config — confirm it constructs (it's an existing
  emitted knob).
- **Knob reconciliation** is the main correctness surface: a plan task diffs emitted args vs.
  the Huawei `MODEL_ARGS` and closes gaps.
- **GPU smoke** (single-GPU, mock data, a few steps) is the end-to-end gate; the user runs it.
