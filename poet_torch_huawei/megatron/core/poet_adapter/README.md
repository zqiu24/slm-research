# POET / POET-X integration for Megatron-LM

This folder adapts the **POET** / **POET-X** reparameterized training method
([POET paper](https://arxiv.org/abs/2506.08001),
[POET-X paper](https://arxiv.org/abs/2603.05500),
[code](https://github.com/Sphere-AI-Lab/poet)) to Megatron-LM. POET replaces
each linear weight with

```
W_eff = R_out · W_0 · R_in      (with random block permutations on in / out)
```

where `W_0` is frozen at init, and `R_out`, `R_in` are learnable block-diagonal
orthogonal matrices, parameterized via the Cayley–Neumann transform on
skew-symmetric blocks `oft_R`. Every `merge_interval` steps we absorb `R` into
`W_0` and redraw permutations (the **merge-then-reinitialize** step), which
prevents error accumulation in the Neumann approximation.

## Two variants: `--poet-variant {poet, poetx}`

Both are mathematically equivalent to `poet_torch.core.ops.forward_core`; they
differ only in *how* the forward is implemented:

| Variant | Forward                                                                 | Per-forward memory                        | When to use                                 |
| ------- | ----------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------- |
| `poet`  | Materialize `W_eff = (R_out · W_0 · R_in)[P_in, P_out]` then run linear | O(out · in) for `W_eff` + activations     | Default, simple; fine for small/medium dims |
| `poetx` | **Input-centric**: rotate x by `R_in`, run frozen linear, rotate y by `R_out` | O(tokens · features) only (no `W_eff`) | Recommended for large hidden/ffn (POET-X\_fast) |

POET-X\_fast matches the POET paper's forward bit-for-bit (validated by
`tests/poet/test_poetx_equivalence.py`) while avoiding the O(out · in)
allocation per forward. When `W_0` has shape (7168, 1280) in bf16, that's
~17 MB per linear per forward — POET-X skips it entirely.

Orthogonal to the variant choice, `--poet-mem-efficient` and `--poet-quantize`
select the upstream Triton-backed recomputation / INT8 paths; those currently
only apply when using the upstream `POETLinear` module directly (i.e. they
don't compose with Megatron's parallel linears yet — out of scope for this
adapter).

## Files

| File                                                                    | Role                                                                                              |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `poet_torch/` (repo root)                                               | Vendored upstream implementation (Cayley kernels, Triton ops, POET/QPOET linear layers, AdamW).   |
| `megatron/core/poet_adapter/adapter.py`                                 | Megatron-specific glue: wraps `ColumnParallelLinear` / `RowParallelLinear`, both `poet` and `poetx` forwards, merging. |
| `pretrain_gpt_poet.py`                                                  | Entry point — `pretrain_gpt.py` + POET argparser + model-provider wrap + post-step merge hook.    |
| `training_scripts/model_args/DeepSeek-3Bv2-sandwich-mqa-poet.yaml`      | Model config for the `poet` (W\_eff) variant.                                                     |
| `training_scripts/model_args/DeepSeek-3Bv2-sandwich-mqa-poetx.yaml`     | Model config for the `poetx` (input-centric) variant.                                             |
| `training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poet.sh`             | Launch script for `poet`.                                                                         |
| `training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poetx.sh`            | Launch script for `poetx` (recommended).                                                          |

## How the integration works

1. **Arg registration.** `pretrain_gpt_poet.py` passes
   `poet_adapter.add_poet_args` through Megatron's `extra_args_provider`, so
   `--use-poet`, `--poet-variant`, `--poet-block-size`, `--poet-merge-interval`,
   etc. flow through Megatron's normal argparser.
2. **Model wrap.** `model_provider` calls the stock `pretrain_gpt.model_provider`
   first, then (if `--use-poet`) runs `install_poet_in_model` *before* DDP
   wraps the model. For every eligible `ColumnParallelLinear` /
   `RowParallelLinear` (see exclusions below), the adapter:
   * registers `oft_R` as a trainable `nn.Parameter` submodule (`_poet_state`);
   * registers random `perm_in/out` buffers;
   * sets `weight.requires_grad = False` (base weight is frozen);
   * row-normalizes `W_0` for a well-conditioned starting spectrum;
   * monkey-patches `forward` with either the `poet` or `poetx` path:
     * `poet`: compute `W_eff` on-the-fly, pass to Megatron's
       `_forward_impl` (so TP comms / grad accumulation / async all-reduce are
       untouched).
     * `poetx`: replicate the linear's TP-input comm, apply `x @
       block_diag(R_in)` and the permutation on the features, run the frozen
       linear (`linear_with_frozen_weight` via `_forward_impl`), apply the
       output rotation + permutation, then TP-output comm.
3. **Merge hook.** `pretrain_gpt_poet._install_merge_hook()` monkey-patches
   `megatron.training.training.train_step` so that, after every successful
   optimizer step, we check `step % merge_interval == 0` and, if true,
   absorb `R · W_0 · R` into `W_0`, redraw permutations, and zero `oft_R`.
   Rank-0 inside each DP group does the merge and broadcasts to the rest.

## Excluded modules (default)

POET is *not* applied to:

* `lm_head`, `output_layer`, `embedding`, `word_embeddings`, `router`, `gate`, `mtp`
  (small / output layers);
* anything whose qualified name contains `local_experts`, `grouped_mlp`,
  `te_grouped_mlp`, or `.experts.` (MoE grouped-GEMM experts — their weight
  layout is batched and doesn't match a plain 2-D `W`);
* any layer whose **local** in / out dims aren't divisible by
  `--poet-block-size`.

MoE **shared** experts (`mlp.shared_experts.*`) *are* wrapped.

## Dimension check for `DeepSeek-3Bv2-sandwich-mqa-{poet,poetx}.yaml`

With `--poet-block-size 256`, TP=1, the wrapped linears have:

| Layer               | local shape (out × in) | divisible by 256? |
| ------------------- | ---------------------- | ----------------- |
| attn `linear_qkv`   | 6912 × 1280            | ✓ (27 × 5)        |
| attn `linear_proj`  | 1280 × 6144            | ✓ (5 × 24)        |
| dense MLP `fc1`     | 14336 × 1280 (2·ffn)   | ✓                 |
| dense MLP `fc2`     | 1280 × 7168            | ✓                 |
| shared expert `fc1` | 3584 × 1280 (2·1792)   | ✓                 |
| shared expert `fc2` | 1280 × 1792            | ✓                 |

If you change head/ffn dims, rerun the check or drop `--poet-block-size` to 128.

## Known caveats

* **TP/CP > 1** is disallowed by default because the TP-sharded local dim
  must remain a multiple of `block_size`; the launch scripts warn if you
  set `TP != 1` or `CP != 1`. If your local shards stay divisible you can
  ignore the warning.
  * Additionally, for TP>1 on row-parallel linears, the POET-X path applies
    `R_out` to the TP-reduced full output; per-rank permutations would produce
    inconsistent outputs across ranks. With TP=1 (enforced) this is moot.
* **Bias handling.** The `poet` W\_eff path adds bias linearly after the
  effective-weight matmul, while `poetx` adds bias inside the chain (matches
  the POET paper's `chain_layer_x_pytorch`, so bias is itself rotated by
  `R_out` + `perm_out`). For DeepSeek-3Bv2 (`--disable-bias-linear: true`)
  this difference doesn't matter.
* **Must use `--transformer-impl local`, not `transformer_engine`.**
  TE's `TEColumnParallelLinear` / `TELayerNormColumnParallelLinear` /
  `TERowParallelLinear` do **not** subclass Megatron's
  `ColumnParallelLinear` / `RowParallelLinear` (they inherit from
  `TELinear` → `TransformerEngineBaseModule` instead and are API-compatible
  only by duck typing). POET's `isinstance(mod, ColumnParallelLinear)`
  check therefore fails on every layer and `install_poet_in_model` now
  raises `RuntimeError` with a helpful inventory instead of silently
  training baseline. The provided poet/poetx YAMLs pin
  `--transformer-impl: local`. FlashAttention is still active via
  `--use-flash-attn: true`; FP8 is independent of POET (you can't combine
  them in this adapter — POET paper operates on bf16 plain linears).
* **Memory-efficient / quantized modes** (`--poet-mem-efficient`,
  `--poet-quantize`) require Triton. They also replace the linear layer
  wholesale (they're not yet plumbed through Megatron's parallel linear);
  you'd have to switch to the upstream `POETModel` wrapper for those, which
  is out of scope for this adapter.
* **Checkpoint compatibility.** `oft_R` and the permutation buffers live on
  `module._poet_state`, so they appear in the state dict under the
  `<module>._poet_state.*` keys. Loading a non-POET checkpoint into a POET
  run works (POET state initializes fresh); the reverse needs a
  `--poet-merge-interval -1` dummy run plus a merge-everything pass.

## Quick start

```bash
cd /public/shihan/code/Megatron-LM
# POET-X_fast (recommended, input-centric, memory-efficient):
bash training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poetx.sh

# POET (W_eff materialization):
bash training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poet.sh
```

Override POET hyperparameters with env vars:

```bash
POET_BLOCK_SIZE=128 POET_MERGE_INTERVAL=500 \
  bash training_scripts/train_DeepSeek_3bv3_sandwich_mqa_poetx.sh
```
