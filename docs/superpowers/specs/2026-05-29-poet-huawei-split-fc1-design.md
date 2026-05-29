# POET (huawei vendored stack) — `poet_split_fc1` + divisibility hard-error

**Date:** 2026-05-29
**Status:** Approved design, ready for implementation plan
**Branch:** `huawei`
**Scope:** the **vendored** `poet_torch_huawei/` stack ONLY (Megatron-core 0.14
+ `poet_adapter`). NOT the first-party `src/optim` POET path, NOT `third_party/`.

> **Sibling spec — read this.** A separate, same-date design
> [2026-05-29-poet-split-fused-layers-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-05-29-poet-split-fused-layers-design.md)
> covers the *same feature idea* (`poet_split_qkv` / `poet_split_fc1`) for the
> **first-party** slm-research POET stack (`src/optim/poet_layers.py`,
> `replace_linears_with_poet`, runtime monkeypatches, opt-in store-true flags).
> **This spec is the independent vendored-stack counterpart** and deliberately
> differs from it: native edits to the vendored Megatron source, auto-on with
> `--use-poet`, and reuse of the vendored Megatron's *already-present*
> `poet_split_qkv` support. The two specs do not overlap in files and are not in
> conflict; they apply the analogous idea to two distinct POET implementations.

---

## 1. Motivation

POET freezes a linear's base weight `W₀` and trains block-diagonal orthogonal
rotations `W_eff = P_outᵀ · R_out · W₀ · R_in · P_in`, where `R_*` are
block-diagonal Cayley orthogonals and `P_*` are **global random permutations**
over the full local in/out dims (redrawn each merge).

The vendored adapter already splits attention QKV — when `--use-poet` is set,
[arguments.py:1185](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/training/arguments.py#L1185)
forces `poet_split_qkv=True`, and the GPT spec
([gpt_layer_specs.py:398-401](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/models/gpt/gpt_layer_specs.py#L398-L401))
builds separate `linear_q` / `linear_k` / `linear_v`. So Q/K/V each get an
independent POET orbit; a rotation never mixes a Q channel with a K channel.

The SwiGLU MLP gets **no such protection**. `linear_fc1` stays one fused
`ColumnParallelLinear` of shape `(2·ffn, hidden)` — gate stacked on up,
doubled at
[mlp.py:94-95](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L94-L95).
POET wraps it as one matrix with one `oft_R`. Crucially, the **output
permutation `P_out` is a global `randperm` over all `2·ffn` channels**
([adapter.py:122](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L122)),
so each 128-wide rotation block acts on channels the permutation pulls from
random positions across *both* the gate region `[0:ffn]` and the up region
`[ffn:2·ffn]`. The learned rotation therefore **entangles the SwiGLU gate and
up branches** before the `glu()` chunk
([mlp.py:172-176](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L172-L176))
ever separates them. (The block size aligning to `ffn` is irrelevant — the
permutation scatters channels across blocks on purpose.) This is the same class
of fused-gate/up drift seen before in this research line for OFT.

We want a `poet_split_fc1` that, mirroring `poet_split_qkv`, builds the SwiGLU
`linear_fc1` as **two** independent `ColumnParallelLinear`s (gate, up), so POET
gives each its own frozen base slice, its own `oft_R`, and its own
`R_in`/`R_out`/`P_in`/`P_out` — genuinely separate orbits per branch.

Independently, we harden the adapter's **divisibility filter** from a silent
skip into a hard error, so a misconfigured `block_size` crashes loudly instead
of quietly under-wrapping the model.

## 2. Goals / non-goals

**Goals**
- `poet_split_fc1` config flag, auto-forced on whenever `--use-poet` is set
  (exactly mirroring `poet_split_qkv`); no new user-facing flag to remember.
- Under the flag, the SwiGLU `linear_fc1` of **every** `MLP` instance becomes
  two truly separate modules (`linear_fc1_gate`, `linear_fc1_up`), each
  `(ffn, hidden)`, with separate state-dict entries and separate POET orbits.
  This covers the dense FFN, the routed `SequentialMLP` experts, and the shared
  expert — all three are the same `MLP` class.
- Bit-identical model output vs. the fused path at POET-identity init
  (`oft_R=0` ⇒ Cayley(0)=I ⇒ `W_eff = W₀` exactly for each half).
- Reuse the existing POET install / merge / optimizer machinery unchanged — the
  split produces ordinary `ColumnParallelLinear` children that
  `install_poet_in_model` wraps like any other linear.
- The MoE **router stays out of POET** — preserved as a first-class acceptance
  gate (see §7).
- Divisibility mismatch becomes a hard error (`RuntimeError`) in both
  `_try_attach` and `_try_attach_te`.

**Non-goals**
- No TP/CP > 1 support (POET already enforces TP=1; the local dims must stay
  block-divisible).
- No change to GroupedMLP / TEGroupedMLP fused-expert handling — those are not
  `MLP` subclasses, are already excluded by POET, and `--moe-grouped-gemm` is
  off for the target config. `--poet-wrap-moe-experts` + `--moe-grouped-gemm`
  already hard-errors
  ([pretrain_gpt_poet.py:135](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/pretrain_gpt_poet.py#L135)).
- No edits to the first-party `src/optim` POET path or `third_party/`.
- No checkpoint round-trip with a fused-fc1 checkpoint — the dev smoke runs
  `SAVE_CKPT=0`; the `sharded_state_dict` handling (§6) is correctness-for-later.

## 3. Decisions (locked with user)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Split mechanism | **Model-level true split** — two separate `ColumnParallelLinear`s built by the spec, not an adapter-level grouped permutation | User choice; most faithful to `poet_split_qkv`; cleanest graph/checkpoint structure. |
| Activation | **Auto-on with `--use-poet`** (no independent opt-in flag) | Exactly mirrors `poet_split_qkv`; maximally faithful to the existing pattern. Accepts that every POET run now splits fc1 (param/layer counts shift vs. the prior fused smoke). |
| Split scope | **All `MLP` instances** — dense + routed experts + shared expert | Uniform single rule (no `is_expert` special-casing); the entanglement is *most* prevalent in experts (27 of 28 fc1 in the dev smoke are expert/shared). |
| Naming | **`linear_fc1_gate` / `linear_fc1_up`** (semantic, mirrors `linear_q/k/v`) | Searchable, parallels the qkv split. Requires the matcher fix below to avoid a `"gate"` substring collision. |
| `_name_matches` | **Harden to dot-bounded token matching** for all patterns | Required for correctness: keeps `.router.` / `.gate.` excluded while letting `.linear_fc1_gate` through. Verified no intended exclusion regresses. |
| Divisibility filter | **Hard error (`RuntimeError`)**, divisibility branch only | User directive. The `None`/can't-introspect guards stay soft skips (legitimate "not a parallel linear" signal). |
| Exit mechanism | **`RuntimeError` propagating to non-zero torchrun exit** (no explicit `sys.exit`) | Runs in `model_provider` pre-DDP; uncaught error crashes with a full traceback. Consistent with the adapter's existing `n_wrapped==0` and FP8 `RuntimeError`s. Deterministic + identical across ranks ⇒ no broadcast/hang risk. |

## 4. Background: how the vendored stack wraps layers today

`install_poet_in_model`
([adapter.py:905-949](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L905-L949))
walks `named_modules()` and applies three filters in order:

1. **Type** — `isinstance(mod, ColumnParallelLinear | RowParallelLinear)` (or a
   TE parallel-linear class). The **only** hard guarantee keeping the router out
   (the router is a `Router(ABC, MegatronModule)` with a raw `nn.Parameter`,
   [router.py:51](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/moe/router.py#L51),
   not a parallel linear).
2. **Name** — `exclude_modules` leaf substrings
   (`lm_head, output_layer, embedding, word_embeddings, router, gate, mtp`) and
   `exclude_ancestors` path substrings (`local_experts, grouped_mlp,
   te_grouped_mlp, .experts.`), both via
   [_name_matches:500-504](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L500-L504)
   (currently a **raw lowercase substring** test — the latent hazard).
3. **Divisibility** — local out/in must each be `% block_size == 0`
   ([adapter.py:649](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L649),
   [:746](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L746)),
   else **silent** `return False` — the filter we are hardening.

## 5. Change set (7 edits)

### 5.1 Config field
[transformer_config.py:163](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/transformer_config.py#L163)
— add `poet_split_fc1: bool = False` adjacent to `poet_split_qkv`.

### 5.2 Arg auto-on
[arguments.py:1185](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/training/arguments.py#L1185)
— add, next to the existing qkv line:
```python
kw_args['poet_split_fc1'] = bool(getattr(args, 'use_poet', False))
```
No argparse entry is needed (matches `poet_split_qkv`, which has none either —
it is derived purely from `use_poet`).

### 5.3 Submodules dataclass
[MLPSubmodules:43-45](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L43-L45)
— add `linear_fc1_gate` and `linear_fc1_up` fields (keep `linear_fc1` for the
fused path):
```python
linear_fc1: Union[ModuleSpec, type] = None
linear_fc1_gate: Union[ModuleSpec, type] = None
linear_fc1_up: Union[ModuleSpec, type] = None
linear_fc2: Union[ModuleSpec, type] = None
```

### 5.4 Spec emission
[get_mlp_module_spec_for_backend:478-508](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/models/gpt/gpt_layer_specs.py#L478-L508)
and the MoE/shared-expert spec builder — thread a `poet_split_fc1` param
(read via `getattr(config, "poet_split_fc1", False)` at the block-spec call
sites, mirroring how `poet_split_qkv` is read at
[gpt_layer_specs.py:546](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/models/gpt/gpt_layer_specs.py#L546)).
When split, emit gate/up as plain `column_parallel_linear()` and set
`linear_fc1=None`, mirroring the qkv branch:
```python
linear_fc1      = None if split_fc1 else col_or_ln_col()
linear_fc1_gate = backend.column_parallel_linear() if split_fc1 else None
linear_fc1_up   = backend.column_parallel_linear() if split_fc1 else None
```
Note the split path uses plain `column_parallel_linear()` (not the LN-fused
`column_parallel_layer_norm_linear()`), same as split_qkv — POET requires
`--transformer-impl local`, so no fused-LN linear is in play anyway.

The MoE / shared-expert specs route through the same `MLPSubmodules`, so they
inherit the gate/up fields once the builder threads `split_fc1` through. The
implementation plan must locate and patch each spec-builder call that
constructs `MLPSubmodules` (dense, `SequentialMLP` experts, shared expert) so
all three receive the split submodules.

### 5.5 `MLP.__init__` / `MLP.forward`
- **`__init__`** ([mlp.py:94-109](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L94-L109)):
  when `submodules.linear_fc1_gate is not None` (split mode), build
  `self.linear_fc1_gate` and `self.linear_fc1_up`, each at width
  `ffn_hidden_size` **before** the SwiGLU `*= 2` doubling (each half is one
  branch). Set a `self.split_fc1 = True` marker. Skip building
  `self.linear_fc1`. The fused branch is unchanged.
- **`forward`** ([mlp.py:127-194](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L127-L194)):
  when `self.split_fc1`, replace the fused `linear_fc1` + activation path with
  ```python
  gate, _ = self.linear_fc1_gate(hidden_states)
  up, _   = self.linear_fc1_up(hidden_states)
  intermediate_parallel = self.activation_func(gate) * up
  ```
  then continue into the existing `linear_fc2` path. The fused path (incl.
  `bias_swiglu_impl` fusion) stays for `split_fc1=False`. Forgoing the fused
  bias-activation kernel on the split path is acceptable — the target config
  runs `--disable-bias-linear: true`.

### 5.6 `_name_matches` hardening (required, not optional)
[adapter.py:500-504](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L500-L504)
— change the bare-substring test to dot-bounded token matching so a leaf token
matches a path segment, not an arbitrary substring:
```python
def _name_matches(name, patterns):
    if not patterns:
        return False
    hay = "." + name.lower() + "."
    return any(("." + p.lower() + ".") in hay for p in patterns)
```
Without this, `exclude_modules`'s `"gate"` pattern would match
`...linear_fc1_gate` and POET would **silently drop the gate half**, training
only the up half's rotation — the exact substring-collision class of bug seen
previously (`"v_proj" in "qkv_proj.oft_R"`). Verified by simulation:
`.router.` and `.mtp.` and `.embedding.` etc. still match their targets;
`.linear_fc1_gate` / `.linear_fc1_up` do not.

> The existing `exclude_ancestors` pattern `".experts."` already carries dots
> intentionally and must keep matching `mlp.experts.` — the dot-bounded form
> preserves this (`."..experts.."` substring check still hits). The plan must
> verify the ancestor patterns under the new matcher as part of acceptance.

### 5.7 Divisibility hard error
[_try_attach:649-659](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L649-L659)
**and** [_try_attach_te:746-756](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L746-L756)
— replace the `logger.info(...) + return False` with a logged error + raise:
```python
if out_local % block_size != 0 or in_local % block_size != 0:
    msg = (
        f"POET: cannot wrap {module_name} (kind={kind}, out_local={out_local}, "
        f"in_local={in_local}) -- not divisible by block_size={block_size}. "
        f"This layer is type/name eligible but its local dims don't tile into "
        f"{block_size}-blocks. Fix the dims, lower --poet-block-size, or exclude "
        f"this layer via --poet-exclude-modules / --poet-exclude-ancestors."
    )
    logger.error(msg)
    raise RuntimeError(msg)
```
Only the divisibility branch is promoted. The `weight is None`
([:635](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L635))
and `out_local is None`
([:647](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L647))
guards remain soft `return False`.

## 6. Checkpoint / sharded_state_dict

The fused path applies `apply_swiglu_sharded_factory` to split the doubled
`linear_fc1` weight for TP at
[mlp.py:204-208](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/mlp.py#L204-L208)
and [experts.py:953](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/moe/experts.py#L953).
When split, the two halves are already separate plain linears, so the swiglu
factory must be **skipped** for them (each serializes as an ordinary
`ColumnParallelLinear`). This is correctness-for-later: the dev smoke runs
`SAVE_CKPT=0`, so it does not block the acceptance run, but the plan must guard
the factory call on `not self.split_fc1` to avoid mis-sharding if saving is
re-enabled.

## 7. Acceptance gates

Re-run the single-GPU smoke (`bash scripts/train_poet_huawei.sh dev`,
`SAVE_CKPT=0`) on the `poet` node. Expected:

1. **Router stays out (hard constraint).** No wrapped module's qualified name
   ends in `.router` or `.gate`; the shared-expert `gate_weight`
   ([shared_experts.py:61](/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/megatron/core/transformer/moe/shared_experts.py#L61))
   is a raw Parameter and also stays out by type. Verify from the per-layer
   wrap log.
2. **Split took effect.** Every `MLP` exposes `linear_fc1_gate` +
   `linear_fc1_up` and no `linear_fc1`; both halves wrapped by POET.
3. **Wrapped count rises 72 → 100** in the dev config. Today: 16 attn
   (q4+k4+v4+proj4) + 28 fc1 + 28 fc2 = 72. After split, each of the 28 fc1
   becomes a gate+up pair (28→56), fc2 and attn unchanged: 16 + 56 + 28 = 100.
4. **Step-0 loss identical** to a fused-POET run (Cayley(0)=I ⇒ each half is
   exact `W₀`).
5. Merge fires at step 20; 30/30 iterations; 0 NaN / 0 skipped; clean exit.
6. **Divisibility now crashes.** A deliberately bad `--poet-block-size` (e.g.
   one that doesn't divide a wrapped dim) raises `RuntimeError` naming the
   module + dims, instead of silently lowering the wrapped count.

CPU-only unit-style check (no dist): with `poet_split_fc1=True`, a small
`MLP` has `linear_fc1_gate`/`linear_fc1_up` (no `linear_fc1`), each
128-divisible, and `act(gate)*up` equals the fused `glu(linear_fc1(x))` output
at equal initialization.

Per [[feedback_no_local_test_run]] the user runs the smoke and reports; this
work is syntax-checked and parity-reasoned, not executed in-harness.

## 8. Files

**Modified (vendored `poet_torch_huawei/` only):**
- `megatron/core/transformer/transformer_config.py` — `poet_split_fc1` field.
- `megatron/training/arguments.py` — derive `poet_split_fc1` from `use_poet`.
- `megatron/core/transformer/mlp.py` — `MLPSubmodules` fields; `MLP.__init__` /
  `MLP.forward` split path; `sharded_state_dict` swiglu-factory guard.
- `megatron/core/models/gpt/gpt_layer_specs.py` — thread `poet_split_fc1`
  through the MLP/MoE spec builders; emit gate/up linears.
- `megatron/core/transformer/moe/experts.py` — `sharded_state_dict`
  swiglu-factory guard for split routed experts.
- `megatron/core/poet_adapter/adapter.py` — dot-bounded `_name_matches`;
  divisibility hard error in `_try_attach` + `_try_attach_te`.

**Added:**
- New CPU unit test under `poet_torch_huawei/` (or the repo's test dir, per the
  plan) covering the split equivalence + divisibility raise.

**Explicitly NOT touched:**
- First-party `src/optim/` POET (covered by the sibling spec).
- `third_party/`.

## 9. Risks / open notes for the plan

- **Spec-builder surface.** The MoE/shared-expert specs may construct
  `MLPSubmodules` in more than one helper; the plan must enumerate every
  construction site so dense + routed + shared all receive split submodules
  uniformly (a missed site = silent fused fallback for that MLP kind).
- **Param-count baseline shifts.** Because the flag is auto-on, the previously
  validated fused smoke (`oft_R = 3.35%` of trainable, 72 wrapped) is no longer
  the baseline; the new numbers (gate=2's `oft_R` is slightly smaller per layer
  but there are more layers) become the reference. Log both for the record.
- **`_name_matches` blast radius.** The dot-bounded change affects POET's
  exclusion for *all* patterns. Argued safe (it only removes accidental
  substring hits), but it is a behavioral change — the acceptance run's wrap
  inventory must be diffed against the pre-change run to confirm the only delta
  is the intended fc1 split.
