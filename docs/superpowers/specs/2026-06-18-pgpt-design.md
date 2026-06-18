# pgpt — Design (v1)

**Date:** 2026-06-18
**Status:** Approved (brainstorming); pending implementation plan
**Author:** Zeju (with Claude Code)

## 1. Summary

**pgpt = nGPT, minus the per-step weight re-projection, designed to be co-trained
with POET.** The forward architecture is identical to nGPT
([src/model/ngpt/](../../../src/model/ngpt/)); the only changes are on the
*optimization* side: nGPT's per-step projection of every weight matrix onto the
unit hypersphere is removed for the matrices POET trains, and narrowed to a
targeted per-step renorm for the two sphere matrices POET does not touch
(token embedding and lm_head).

pgpt is built as a **full fork** into a parallel `pgpt` namespace. It does not
modify, import, or patch any nGPT code. nGPT remains a standalone experiment;
pgpt is a sibling.

## 2. Motivation and thesis

### 2.1 What nGPT's per-step renorm does

nGPT enforces that every embedding/attention/MLP weight matrix lives on the unit
hypersphere — each row (or column, per role) is projected to unit L2 norm at init
and **after every optimizer step** via
[`normalize_module_matrices`](../../../src/model/ngpt/normalize.py) driven by the
[`ngpt_normalize_step`](../../../src/patches/ngpt_normalize_step.py) patch (a wrap
of Megatron's `train_step`). This serves two purposes: (a) keep hidden states on
the sphere, and (b) keep each matrix well-conditioned.

### 2.2 What POET preserves

POET parametrizes each trained linear weight as

```
W_eff = A · W_base · B
```

where `A` (out×out) and `B` (in×in) are **block-diagonal orthogonal** matrices
(Cayley or matrix-exp of a skew generator) composed with fixed permutations, and
`W_base` is frozen
([third_party/poet_torch/poet_layer.py](../../../third_party/poet_torch/poet_layer.py)).
Consequences:

- **Preserved exactly:** the Frobenius norm `‖W‖_F` and the *full singular-value
  spectrum* of `W` (since `A`, `B` are orthogonal, `W_eff` has the same singular
  values as `W_base`).
- **NOT preserved:** nGPT's actual invariant — *per-row / per-col unit norm*. The
  left factor `A` mixes rows within each output block; the right factor `B` mixes
  columns within each input block; a within-block rotation of size > 1 changes
  individual row/col norms even if they all started at 1. So nGPT's per-step
  projection enforces something **strictly stronger** than POET keeps.

### 2.3 Why the per-step renorm becomes redundant under POET

Two independent reasons:

1. **Activations stay spherical regardless of weight norms.** What actually keeps
   *hidden states* on the unit sphere at runtime is the `justnorm` inside the
   residual blend ([layer.py:123,130](../../../src/model/ngpt/layer.py)) and the
   Q/K hypersphere norm ([attention.py:54](../../../src/model/ngpt/attention.py)).
   These are **activation** operations. POET only rotates **weights**, so it never
   perturbs them.
2. **Conditioning is held by POET's spectral preservation.** The conditioning role
   of the per-step renorm is subsumed by POET preserving each matrix's spectrum
   exactly. A one-shot init normalization sets `W_base`'s spectrum; POET holds it
   there for free.

Therefore, for the matrices POET wraps, the per-step weight projection can be
dropped.

### 2.4 The gap: matrices POET does not wrap

POET **skips the lm_head** (`skip_lm_head=True`,
[poet_layers.py:359-361](../../../src/optim/poet_layers.py)) and never wraps the
token embedding (it is a `VocabParallelEmbedding`, not a `Linear`). Those two
matrices are trained by plain Adam, so POET's spectral guarantee does **not**
apply to them. nGPT keeps them on the sphere via the per-step renorm; pgpt must
decide what to do. **Decision: keep a targeted per-step renorm for exactly these
two matrices** (see §4.4). Justification for not simply dropping them: it is the
most faithful-to-nGPT option and cheap; it preserves nGPT's "embedding and vocab
rows are unit vectors" property where POET cannot.

### 2.5 pgpt is a distinct architecture — NOT `ngpt_poet`

There is a pre-existing experiment
[`configs/experiments/arch/ngpt_poet.yaml`](../../../configs/experiments/arch/ngpt_poet.yaml)
that trains **vanilla nGPT with POET**. It is a different thing and pgpt is not
derived from it:

- `ngpt_poet` keeps the **vanilla nGPT base model** — it retains `ngpt_normalize_step`
  and projects *every* weight matrix back onto the sphere every step. Because that
  patch occupies `train_step`, `ngpt_poet` is **forced to omit `poet_merge_step`**
  and runs POET in the `merge_period=0` no-merge regime (`oft_R` never folds into
  `W_base`).
- **pgpt removes the explicit weight normalization from the model itself.** The
  trained weights are no longer constrained to the unit sphere; pgpt relies on
  POET's orthogonal geometry + the runtime activation `justnorm`s instead. This
  makes pgpt a new base model, not "nGPT-with-POET-bolted-on".

A practical consequence of dropping the per-step all-weight renorm: `train_step` is
free, so pgpt **can include `poet_merge_step`** (merges available) — something
`ngpt_poet` structurally cannot do. pgpt is built as its own fork and does not
reference `ngpt_poet`'s config, patches, or model code.

## 3. Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | How far to remove weight normalization | Drop per-step renorm for POET-wrapped matrices; **keep** the one-shot init normalize |
| 2 | POET coupling | **POET-required** — fail fast at spec-build if `args.poet` is false |
| 3 | Embedding + lm_head | **Targeted per-step renorm** for just those two, via an optimizer post-step hook (not a `train_step` wrap) |
| 4 | Code structure | **Full fork** into `src/model/pgpt/`; zero runtime dependency on nGPT |
| 5 | POET merge regime | Include `poet_merge_step` in the patch set (no collision — `train_step` is free); default the v1 config to `merge_period=0`, with `merge_period>0` available as a one-line config flip (the capability `ngpt_poet` lacks) |

## 4. Architecture

### 4.1 File layout

```
src/model/pgpt/
  __init__.py
  normalize.py        # justnorm, normalize_module_matrices       (copy of ngpt)
  scaling_params.py   # LearnedScaling                            (copy of ngpt)
  block.py            # PGPTBlock — CPU parity oracle             (copy of ngpt)
  layer.py            # PGPTTransformerLayer(TransformerLayer)    (copy of ngpt)
  attention.py        # QKHyperNorm                               (copy of ngpt)
  mlp.py              # PGPTMLP / PGPTMLPBody                      (copy of ngpt)
  output_scaling.py   # attach_sz_scaling                         (copy of ngpt)
src/specs/pgpt_layer_spec.py        # build_pgpt_layer_spec       (mirror of ngpt spec)
src/patches/pgpt_apply_spec.py      # wraps gpt_builder + core_transformer_config_from_args
src/patches/pgpt_optimizer_setup.py # cooperative wrap of setup_model_and_optimizer
configs/experiments/arch/pgpt.yaml
scripts/train_pgpt_dev.sh           # fork of scripts/train_ngpt_dev.sh
tests/unit/test_pgpt_*.py
tests/numerics/test_pgpt_megatron_layer_parity.py
tests/_fixtures/pgpt_reference/     # fork of ngpt_reference (full-fork policy: copy, no import)
```

Classes are renamed `NGPT*` → `PGPT*` where it reads cleanly (`PGPTBlock`,
`PGPTTransformerLayer`, `PGPTMLP`). `QKHyperNorm` and `LearnedScaling` keep their
names (no nGPT-specific semantics in them).

The patched/stamped config fields and the model attributes keep the `ngpt_*` /
`_ngpt_*` names in v1 to minimize the diff against the forked code (e.g.
`config.ngpt_base_scale`, `model._ngpt_sz`, `model._ngpt_norm_role_map`). Renaming
them to `pgpt_*` is a mechanical follow-up if desired; it is not required for
correctness and is explicitly out of scope for v1.

### 4.2 Identical to nGPT (forward path — copied verbatim, math unchanged)

`justnorm`; `LearnedScaling` (eigen-LR parametrization); `_residual_blend`
(hypersphere residual); `QKHyperNorm` (Q/K L2-norm + `sqk`); `PGPTMLP` (`suv`
SwiGLU); `attach_sz_scaling` (per-vocab logit scale `sz`); the layer `forward`;
the spec wiring (`IdentityOp` norms, `IdentityFuncOp` bdas, `QKHyperNorm` in the
q/k_layernorm slots, MLP module); `softmax_scale = √head_dim`; MHA-only
(`num_query_groups == num_attention_heads`); the zero-weight-decay grouping of the
scaling params; full activation recompute; the one-shot init normalize and the
weight-norm role-map registration.

### 4.3 Changes from nGPT (only two)

1. **No per-step renorm patch.** nGPT's
   [`ngpt_normalize_step`](../../../src/patches/ngpt_normalize_step.py) (which
   re-projects *all* matrices every step via a `train_step` wrap) is **not ported.**
2. **The surviving per-step renorm is narrowed** to token embedding + lm_head and
   **moved off `train_step`** onto an optimizer post-step hook (§4.4).

### 4.4 The targeted-renorm mechanism

`pgpt_optimizer_setup` **cooperatively wraps** `megatron.training.training.setup_model_and_optimizer`.
It registers with `targets=()` (no exclusive ownership) and wraps the *current*
binding of the symbol, exactly like the always-on
[`wandb_trainable_params`](../../../src/patches/wandb_trainable_params.py),
[`poet_grad_conditioning`](../../../src/patches/poet_grad_conditioning.py), and
[`grad_conditioning`](../../../src/patches/grad_conditioning.py) patches do. This
composes with those wrappers regardless of apply order and raises no
`PatchConflict`.

After calling the original `setup_model_and_optimizer` (which has already run the
POET-wrapped `get_model` + `get_megatron_optimizer`), the wrapper, on the returned
`(model, optimizer, opt_param_scheduler)`:

- **(a) No-WD grouping** — sets `weight_decay = 0.0` on any optimizer param group
  containing a scaling param (`sqk`, `suv`, `attn_alpha`, `mlp_alpha`, `_ngpt_sz`).
  This is the job nGPT did in
  [`ngpt_optimizer_setup`](../../../src/patches/ngpt_optimizer_setup.py); pgpt does
  it here instead of wrapping `get_megatron_optimizer` (which POET owns).
- **(b) Per-step renorm** — installs the embedding+lm_head re-projection by
  **monkey-patching `optimizer.step`**: call the original `step`, then `justnorm`
  the rows of `word_embeddings.weight` and `output_layer.weight`. Chosen over
  `torch`'s `register_step_post_hook` because it is robust across Megatron's
  optimizer wrappers (`ChainedOptimizer` / `Float16OptimizerWithFloat16Params`
  under POET), whose inner-optimizer identity varies. The renorm reuses the role
  map registered by `pgpt_apply_spec`, filtered to the "rows" entries for the
  embedding and `output_layer`.

Gating: both (a) and (b) are no-ops unless the run is pgpt (checked via the stamped
config/args flag), so the wrapper is inert on non-pgpt runs.

### 4.5 POET-required enforcement

`build_pgpt_layer_spec` (and/or `pgpt_apply_spec`) asserts `args.poet` is set, with
a clear message, so a misconfigured experiment fails fast at submit/spec-build time
rather than silently training a non-POET pgpt whose weights would drift off the
sphere.

## 5. Patch set and Megatron hook points

pgpt experiment patch list (`configs/experiments/arch/pgpt.yaml`):

```yaml
patches:
  - model_unfuse_linears        # shared: unfuse qkv/fc1 at build time (pre-DDP)
  - poet_apply_to_model         # POET: wraps get_model
  - poet_optimizer_setup        # POET: wraps get_megatron_optimizer(_config)
  - poet_merge_step             # POET: wraps train_step
  - pgpt_apply_spec             # NEW: wraps gpt_builder + core_transformer_config_from_args
  - pgpt_optimizer_setup        # NEW: cooperative wrap of setup_model_and_optimizer
  - training_log_eta            # logging
  - wandb_metric_normalize      # logging
```

(plus the POET `--poet-*` CLI args supplied by the dev script / optim config.)

**Intentionally OMITTED POET/arch patches** (each would collide with `pgpt_apply_spec`
on a build-time symbol; each is a no-op under pgpt's config anyway):

- `poet_unfuse_te_impl` — wraps `core_transformer_config_from_args` (collides with
  `pgpt_apply_spec`). Only flips `transformer_engine → local`; a no-op because
  `base.model.transformer_impl` is pinned to `local`.
- `sandwich_norm_apply` — wraps `gpt_builder` (collides with `pgpt_apply_spec`). A
  no-op unless `--use-sandwich-norm`.

**Distributed optimizer must be OFF.** POET's optimizer builder rejects Megatron's
distributed optimizer on its custom paths. The dev script passes
`parallelism.distributed_optimizer=false` last (the cluster config sets it after the
experiment YAML, so the override cannot live in `pgpt.yaml`), mirroring
[`scripts/train_ngpt_dev_poet.sh`](../../../scripts/train_ngpt_dev_poet.sh).

### 5.1 Flag wiring

pgpt reuses nGPT's architecture CLI flags and config fields (per §4.1, to minimize
the fork's diff):

- `experiment.kind: pgpt` triggers a **new** `_pgpt_arch_args(cfg)` helper in
  [`src/utils/megatron_args.py`](../../../src/utils/megatron_args.py) — a mirror of
  the existing `_ngpt_arch_args`, gated on `kind == "pgpt"` — that emits `--ngpt`
  plus the scaling-vector inits (`--ngpt-alpha-init`, `--ngpt-sqk-init`,
  `--ngpt-suv-init`, `--ngpt-sz-init`, `--ngpt-no-warmup`) read from `optim.ngpt.*`.
  This does **not** modify `_ngpt_arch_args`; it adds a sibling and one call line.
- `optim.type: poet` emits `--poet` + the `--poet-*` flags (existing POET branch).
- pgpt's patches therefore see `args.ngpt == True` (architecture gate, reused) and
  `args.poet == True` (POET gate). `pgpt_apply_spec` / `pgpt_optimizer_setup` gate
  their behavior on `args.ngpt`; `pgpt_apply_spec` additionally asserts `args.poet`
  (§4.5). They are only ever *applied* on pgpt runs because the experiment lists
  the `pgpt_*` patches (not the `ngpt_*` ones).

Collision analysis (the registry refuses two patches that declare the same target):

| Megatron symbol | Owner(s) | Collision? |
|-----------------|----------|------------|
| `gpt_builders.gpt_builder` | `pgpt_apply_spec` | Clean — POET uses `get_model` |
| `core_transformer_config_from_args` | `pgpt_apply_spec` | Clean |
| `get_model` | `poet_apply_to_model` | Clean — POET wraps the outer provider, sees pgpt's built linears |
| `get_megatron_optimizer(_config)` | `poet_optimizer_setup` | Clean — pgpt does **not** wrap it |
| `train_step` | `poet_merge_step` | Clean — pgpt has **no** train_step patch |
| `setup_model_and_optimizer` | `pgpt_optimizer_setup` (+ `wandb_trainable_params`, grad-conditioning) | Clean — all register `targets=()` and compose cooperatively |

Both original nGPT↔POET collisions are resolved: `train_step` by removal of the
per-step renorm patch, and `get_megatron_optimizer` by relocating pgpt's optimizer
post-processing to the outer `setup_model_and_optimizer`.

### 5.1 Build/order interaction

- `pgpt_apply_spec` wraps `gpt_builder` (inner); it builds the pgpt layers and runs
  the one-shot init normalize + role-map registration + `sz` attach at post-build.
- `poet_apply_to_model` wraps `get_model` (outer); it applies POET to the
  freshly-built pgpt linears **pre-DDP** so `oft_R` lands in the grad buffer.
  POET therefore sees pgpt's already-normalized linears as its `W_base`.
- With `poet_init_type=normalized` (default), POET re-normalizes `W_base` itself.
  So pgpt's init-normalize is **belt-and-suspenders for the POET-wrapped matrices**
  (POET governs their final base scale) and **load-bearing for embedding + lm_head**.
  pgpt keeps init-normalize for all matrices anyway (mirrors nGPT, harmless, simpler).

## 6. Validation

- **CPU parity (forward):** fork nGPT's parity oracle (`PGPTBlock` + reference
  fixture) and assert pgpt's forward is bit-identical to nGPT's — the forward math
  is unchanged; only the optimizer-time behavior differs.
- **Patch registry test:** assert the full pgpt + POET patch set registers with no
  `PatchConflict`. This is the core integration claim.
- **Renorm unit test:** after a fake `optimizer.step`, embedding and lm_head rows
  are unit-norm, and the per-layer POET-wrapped matrices are left untouched.
- **No-WD test:** scaling params (`sqk/suv/attn_alpha/mlp_alpha/_ngpt_sz`) land in a
  `weight_decay == 0` group after `setup_model_and_optimizer`.
- **POET-required test:** spec-build raises a clear error when `args.poet` is false.
- **CPU tests run locally;** GPU smoke is deferred. A `scripts/train_pgpt_dev.sh`
  (fork of [train_ngpt_dev.sh](../../../scripts/train_ngpt_dev.sh) with
  `experiment=arch/pgpt --optimizer adam --slm-optimizer poet`) and the exact
  command are handed to the user to run.

## 7. Out of scope / future iterations

pgpt is intended to be iterated gradually. The following are explicitly **not** in
v1:

- **Role-aware POET (v2 candidate):** rotate each row-normalized matrix on the input
  side only (and each col-normalized matrix on the output side only) so that
  per-row/col unit norm is preserved *exactly* by the rotation — which would let us
  drop even the targeted embedding/lm_head renorm. Requires per-matrix POET
  side-selection.
- **Standalone (non-POET) pgpt mode.**
- **Renaming `ngpt_*` config/attr fields to `pgpt_*`.**
- **TP>1 / MoE / MLA** (inherited nGPT v1 constraints).

## 8. Risks

- **Embedding/lm_head drift still present between renorm and use within a step** —
  negligible: the renorm runs every step right after the update.
- **Coupling to Megatron internals** (`setup_model_and_optimizer` signature,
  optimizer `.step`): same surface nGPT and the grad-conditioning patches already
  depend on; covered by the patch-registry + renorm tests and the Megatron pin.
- **Init-normalize/POET-init interaction** documented in §5.1; harmless by ordering.
