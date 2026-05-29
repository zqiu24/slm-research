# POET split-fused-layers — `poet_split_qkv` / `poet_split_fc1`

**Date:** 2026-05-29
**Status:** Approved design, ready for implementation plan
**Scope:** slm-research general codebase POET integration (NOT the vendored
`poet_torch_huawei/` stack, NOT `third_party/`)

## 1. Motivation

POET freezes a linear's base weight `W` and trains block-diagonal orthogonal
rotations `R_out · W · R_in`. Today the general codebase wraps each **fused**
parallel linear as a single `POETLinear`
([src/optim/poet_layers.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L103)),
so for the fused `linear_qkv` (out = Q+K+V) and the SwiGLU `linear_fc1`
(out = gate+up) the block-diagonal rotation can straddle the boundary between
two logically-distinct projections — a single orthogonal block mixes, e.g., a Q
output channel with a K output channel. The same drift this causes for OFT on
fused gate/up has been observed before in this research line.

We want two independent, opt-in arguments that, under `--poet`, give each
sub-projection its **own** frozen base-weight slice, its **own** `oft_R`, and
its **own** permutations — i.e. genuinely separate POET orbits per projection.

- `poet_split_qkv` — split the fused attention `linear_qkv` into separate
  Q / K / V projections.
- `poet_split_fc1` — split the fused MLP `linear_fc1` into separate gate / up
  projections.

## 2. Goals / non-goals

**Goals**
- Two independent store-true args wired through the existing POET arg path.
- Under each flag, the corresponding fused linear becomes **truly separate
  modules** in the model graph (separate state-dict entries, separate POET
  orbits), with Megatron's attention / MLP forward patched to use them.
- Bit-identical pre-POET model output vs. the unsplit path (at POET-identity
  init), for MQA, GQA, and MHA.
- Reuse the existing POET merge / optimizer / cache machinery unchanged.

**Non-goals**
- No support for tensor-parallel > 1 in the split path (POET already enforces
  TP=1; we assert it).
- No support for gated attention (`attention_output_gate`) in `split_qkv` —
  hard-error if encountered (no target config uses it).
- No MLA support for `split_qkv` — MLA has no fused `linear_qkv`
  ([multi_latent_attention.py](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/transformer/multi_latent_attention.py#L88))
  so the flag is inert there (no error, just nothing to split).
- No edits to `poet_torch_huawei/` or `third_party/` — all Megatron forward
  changes are runtime monkeypatches via the patch registry.

## 3. Decisions (locked)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Split mechanism | **Truly separate modules** + patch attention/MLP forward | User choice; cleaner graph/checkpoint structure than a hidden wrapper. |
| QKV correctness | **General GQA-correct** de-interleave from query-group metadata | Correct for MQA (Huawei DeepSeek), GQA (llama3), MHA; collapses to contiguous for MQA. |
| Indivisible sub-segment | **Hard error** up front | Fail fast with a clear message rather than silently skipping POET on a segment. |
| Attention forward strategy | Patched forward **reassembles `mixed_qkv`** then runs the original (TP=1) view/split/norm/rotary path | Minimizes correctness risk: we never reimplement the attention math. Orbits are still genuinely separate (reassembly happens *after* each separate projection computes its output). |

## 4. Arg surface

Mirror the existing three-stop `--poet-*` plumbing:

1. **argparse** — add to `add_slm_args`
   ([launchers/pretrain_gpt_slm.py:29](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L29)):
   ```
   group.add_argument("--poet-split-qkv", action="store_true")
   group.add_argument("--poet-split-fc1", action="store_true")
   ```
2. **YAML → flag** — in the `kind == "poet"` branch
   ([src/utils/megatron_args.py:220](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L220)),
   append `--poet-split-qkv` / `--poet-split-fc1` when
   `optim.poet.split_qkv` / `optim.poet.split_fc1` are truthy (default false,
   read via `.get(...)`).
3. **config surface** — document both under `optim.poet:` in
   [configs/experiments/optim/poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml),
   defaulting to `false`.

Both flags are independent and only meaningful when `--poet` is set; with
`--poet` off they are ignored.

## 5. Integration ordering

The split surgery runs **inside the existing `poet_apply_to_model` wrapped
`get_model`** ([src/patches/poet_apply_to_model.py:27](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py#L27)),
**before** `replace_linears_with_poet`:

```
model = _orig_get_model(...)
if args.poet:
    if args.poet_split_qkv or args.poet_split_fc1:
        split_fused_linears(model, split_qkv=..., split_fc1=...,
                            block_size=..., block_count=...)   # new
    replace_linears_with_poet(model, ...)                      # existing
```

Because the split produces ordinary `ColumnParallelLinear` children, the
existing walker then POET-wraps each sub-linear exactly like any other linear —
so merge ([src/patches/poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py)),
the POET optimizer partition ([src/optim/poet.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L350)),
and the cache all work with no changes. Doing the split inside the same wrapper
(rather than a separate patch) avoids fragile patch-ordering between two
`get_model` wrappers.

## 6. New module: `src/optim/poet_split.py`

Public entry point:

```
def split_fused_linears(model, *, split_qkv, split_fc1,
                        block_size, block_count) -> int: ...
```

Walks `model.named_modules()`; returns the number of fused linears split.

### 6a. Attention surgery (`split_qkv`)

For each `SelfAttention` instance that owns a fused `linear_qkv`
(`isinstance(linear_qkv, ColumnParallelLinear)` — MLA's `linear_q_proj` etc.
are simply never matched):

1. **Guards** (hard-error with a clear message otherwise):
   - TP size == 1 (`world_size == 1`).
   - `config.attention_output_gate` is false.
2. **Geometry** from the attention module:
   - `hd = self.hidden_size_per_attention_head`
   - `ng = self.num_query_groups_per_partition`
   - `nqhpg = self.num_attention_heads_per_partition // ng` (query heads/group)
   - per-group stride `G = (nqhpg + 2) * hd`; fused out = `ng * G`
   - segment out-dims: `q = ng*nqhpg*hd`, `k = ng*hd`, `v = ng*hd`
3. **De-interleave init.** The fused rows are group-major
   `[g: q(nqhpg·hd), k(hd), v(hd)]` (Megatron layout
   `q1 q2 k1 v1 | …`, [attention.py:1431](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/transformer/attention.py#L1431)).
   Build row-index lists `q_rows`, `k_rows`, `v_rows` by gathering, for every
   group `g`, the q/k/v slices out of `[g*G : (g+1)*G]`. Create
   `linear_q/k/v` as `ColumnParallelLinear` and copy
   `W_seg = W_fused[seg_rows]` (and bias likewise if present).
4. **Forward patch** (per-instance bound method replacing
   `get_query_key_value_tensors`): compute `q_out/k_out/v_out` from the three
   linears, **reassemble** into `mixed_qkv` via a precomputed inverse-interleave
   index buffer (`cat([q_out,k_out,v_out],-1)[..., interleave_index]`), then run
   the original TP=1 post-linear logic (view → split → q/k layernorm) to return
   `(query, key, value)`. Rotary is applied later by `Attention.forward` on the
   returned tensors and is unaffected. The `split_qkv=False` /
   `split_arg_list` return contract of the original method is preserved.
5. Register `linear_q/k/v` on the attention module and `del` the fused
   `linear_qkv` so it no longer appears in the state dict.

### 6b. MLP surgery (`split_fc1`)

For each `MLP` instance that owns a fused `linear_fc1`:

1. **Guards:** require `config.gated_linear_unit` (split only meaningful for the
   2-way gate/up fused layout); TP size == 1.
2. **Contiguous split.** fused out = `2 * ffn`, laid out `[gate | up]`
   ([mlp.py:312](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/transformer/mlp.py#L312)).
   Build `linear_fc1_gate` / `linear_fc1_up` as `ColumnParallelLinear` (out =
   `ffn` each), copy the two contiguous weight halves (and bias halves).
3. **Forward patch** (per-instance method replacing `MLP.forward`): compute
   `gate_out`/`up_out`, `cat([gate_out, up_out], -1)`, then run the **unchanged**
   existing activation / bias-activation-fusion / fc2 path — so gated-SiLU
   fusion still applies.
4. Register the two sub-linears and `del` fused `linear_fc1`.

### 6c. Divisibility hard error

After computing each segment's out-dim, validate against the active divisor
(`block_count` if set else `block_size`) for **both** the segment out-dim and
the shared in-dim. On any failure, raise `ValueError` naming the module path,
the offending segment, its `(in, out)` dims, and the divisor — before training
starts. (Replaces the silent skip the unsplit walker uses for non-divisible
layers, per the locked decision.)

## 7. Edge cases / guards

- **Bias:** carry the fused bias's matching slice into each sub-linear; `None`
  when the fused linear has no bias. (Target DeepSeek configs use
  `disable_bias_linear`, so typically no bias.)
- **MLA:** no `linear_qkv` → `split_qkv` matches nothing; no error.
- **Non-gated MLP:** `split_fc1` requires `gated_linear_unit`; hard-error if a
  user sets it on a non-gated MLP.
- **Model chunks (PP/VPP):** `split_fused_linears` iterates each chunk in the
  `model` list exactly as `poet_apply_to_model` already does.
- **block_count vs block_size:** the divisor precedence matches the existing
  walker (`block_count` takes precedence when set).

## 8. Testing (`tests/unit/`)

- **De-interleave / reassembly equivalence:** build a small `SelfAttention`
  (or a faithful stand-in with the documented row layout) for both MQA (`ng=1`)
  and GQA (`ng>1`, `nqhpg>1`); assert the split-then-reassemble `mixed_qkv`
  equals the fused output, and that the resulting `(query, key, value)` match
  the unsplit path at POET-identity init.
- **FC1 split equivalence:** gate/up split reproduces the fused `linear_fc1`
  output and the gated activation result.
- **Divisibility hard error:** a segment out-dim not divisible by `block_size`
  raises `ValueError` with the segment named.
- **Inertness:** `split_qkv` on an MLA-style module makes no changes and does
  not raise.

Follow the existing CPU-friendly patterns in
[tests/unit/](/lustre/fast/fast/zqiu/slm-research/tests/unit/) (pass plain
`nn.Linear` / minimal stand-ins where Megatron isn't importable).

## 9. Files

**Modified (first-party only):**
- [launchers/pretrain_gpt_slm.py](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py) — two args.
- [src/utils/megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py) — emit the two flags from the poet branch.
- [src/patches/poet_apply_to_model.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py) — call `split_fused_linears` before POET wrapping.
- [configs/experiments/optim/poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml) — `split_qkv` / `split_fc1` config surface (default false).

**Added:**
- [src/optim/poet_split.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_split.py) — split surgery, de-interleave, forward patches, divisibility validation.
- New unit test under [tests/unit/](/lustre/fast/fast/zqiu/slm-research/tests/unit/).

**Explicitly NOT touched:**
- `/lustre/fast/fast/zqiu/slm-research/poet_torch_huawei/` (reference only).
- `/lustre/fast/fast/zqiu/slm-research/third_party/` (Megatron-LM + poet_torch);
  forward changes are runtime monkeypatches via the patch registry.
