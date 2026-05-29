# Patches cookbook

When to reach for `src/patches/`: only when Megatron's ModuleSpec system
cannot express what you need (new residual connections, low-level layer
rewrites, kernel-selection changes). If you can implement a variant as an
`nn.Module` wired up via `build_spec`, do that instead — it's cleaner and
doesn't touch upstream code.

## Writing a patch

1. Create `src/patches/<name>.py`.
2. Docstring must cite the upstream function, Megatron SHA, and the
   rationale:
   ```python
   """
   PATCH: <name>
   Modifies: <fully.qualified.symbol>
   Upstream SHA ref: <megatron-sha> (line ~NNN)
   Rationale: <one paragraph>
   Required by: experiments tagged family:<family>
   """
   ```
3. Register:
   ```python
   from src.patches._registry import register_patch

   @register_patch(name="<name>", targets=("<fully.qualified.symbol>",))
   def apply():
       import <upstream>
       <upstream>.<symbol> = <replacement>
   ```
4. Declare a `tests/numerics/test_<name>_neutral.py` test demonstrating
   that the patch is a no-op when its feature is not activated.
5. List the patch in each experiment that needs it under
   `experiment.patches`.

## Conflict handling

Two patches that declare overlapping `targets` raise `PatchConflict` at
registration time. If you need the combined effect, write a single
combined patch.

## Hashing

`apply_patches(names)` returns `patch_set_hash = blake2s(sorted(name:source_sha))`.
This hash is recorded on every run so the exact monkey-patched state is
reproducible.

## Worked example: sandwich-norm (DeepSeek-3Bv2 MQA)

The first-party DeepSeek-3Bv2 port (`base/family=deepseek_v3_mqa`,
`base/scale=deepseek_3bv2`) is MQA (`num_query_groups=1`, `head_dim=384`) +
MoE + MTP + **sandwich-norm** — a post-norm applied to the attention / MLP
output *before* the residual add, with the norm weight scaled small at init
(`attn_post_norm_scale=ffn_post_norm_scale=0.03`).

- `src/model/sandwich_norm_ops.py` — pure, CPU-testable hook + scale helpers.
- `src/model/sandwich_layer.py` — `SandwichTransformerLayer` injects the
  post-norm via PyTorch forward-hooks on `self_attention` / `mlp`, so no
  Megatron `forward` is copied. No-op unless `config.use_sandwich_norm`.
- `src/patches/sandwich_norm_apply.py` — stamps the config from args and swaps
  the layer class across every `gpt_builder` spec path (dense, MoE
  `.layer_specs`, MTP). Targets `gpt_builders.gpt_builder` and
  `megatron.training.arguments.core_transformer_config_from_args`.

The patch is wired into `optim/adam` and `optim/muon_hybrid` (no-op unless
`base.model.use_sandwich_norm`). It is **deliberately not** in `optim/poet`:
`poet_unfuse_te_impl` already owns the `core_transformer_config_from_args`
target, so listing both would raise `PatchConflict` at registration (see
"Conflict handling"). POET + sandwich-norm is deferred until that overlap is
resolved (e.g. by scoping the config-stamp inside the `gpt_builder` wrapper so
the patch declares only the `gpt_builder` target).
