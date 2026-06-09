# nGPT smoke run

## Goal
Confirm the nGPT variant trains end-to-end on a single H100/H800 node for ~100 steps, with no NaNs, monotonically decreasing loss, and post-step weight normalization actually firing.

## Cluster
`h800_cn`, single node, 8 GPUs. Submitter:
```bash
python -m launchers.submit \
    base/family=llama3 \
    base/scale=600m \
    experiment=arch/ngpt \
    training_regime=ablation_20x \
    cluster=h800_cn \
    seed=0 \
    training.tokens_per_param=1  # ~143-step smoke; NOTE: +training.total_tokens is clobbered by resolve_config (submit.py:153), so cap via tokens_per_param
```

## What to look for
- `[nGPT] applied spec + attached sz + registered weight-norm roles` in rank-0 stdout after model build.
- Training loss strictly decreasing across the first 50 steps; no NaN.
- After ~10 steps, sample a parameter (e.g. `module.transformer.layers[0].self_attention.linear_qkv.weight`) and confirm row-norms are ≈ 1.0.
- Check the W&B run has separate `lr_groups/decay` vs `lr_groups/no_decay` (the no-decay group should contain sz, sqk, suv, attn_alpha, mlp_alpha).

## If it fails
1. **Spec swap didn't fire** — `--ngpt` not propagated to argv. Check the `kind == "ngpt_adamw"` block in `src/utils/megatron_args.py` and that `experiment.patches` lists `ngpt_apply_spec`. The patch wraps `gpt_builders.gpt_builder`; it relies on patches being applied (launcher `_apply_runtime_patches`) **before** `import pretrain_gpt` so the wrapped builder/cfg are the ones bound — this ordering was verified at implementation time but is what the smoke ultimately confirms.
2. **`attach_sz_scaling` AttributeError** — `args.padded_vocab_size` not yet set when `gpt_builder` is patched; move the attach to a later hook (post-pretrain init).
3. **Hypersphere normalization doesn't fire** — `model._ngpt_norm_role_map` empty or under-sized. The assertion inside `_register_ngpt_norm_roles` (see `src/patches/ngpt_apply_spec.py`) should trip first with a count; if not, the Megatron submodule naming changed — extend `_NORM_ROLES_BY_SUFFIX`.
4. **NaN at step 1** — confirm `config.softmax_scale ≈ sqrt(head_dim)` by inspecting `model.module.decoder.layers[0].self_attention.core_attention.softmax_scale` in a debugger. If it's still `1/sqrt(head_dim)`, the wrap of `core_transformer_config_from_args` in `ngpt_apply_spec` didn't fire — verify the patch is in `experiment.patches` and was applied before model build.
5. **alphas missing from optimizer** — log `len([n for n, _ in model.named_parameters() if "attn_alpha" in n or "mlp_alpha" in n])` after build; expect `2 * num_layers`. If zero, `NGPTTransformerLayer.__init__` regressed back to lazy build.

## Promotion
If smoke is green at 100 steps, hand off to a 24-hour 24B-token ablation run on the same cluster.
