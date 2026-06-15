# Adding an architecture family

The architecture axis is `base/family` (mechanisms) × `base/scale`
(dimensions realizing a parameter budget). The frozen axes — dataset,
tokenizer, training regime, scheduler, optimizer experiment — never change
when a family is added. The pinned `third_party/Megatron-LM` is never
edited.

## 1. Family config (`configs/base/family/<name>.yaml`)
Architecture mechanisms only: normalization, activation, positional
encoding, attention variant (GQA dims live in scale; MLA/GDN/mamba blocks
live here with sensible defaults), MoE *recipe* (router type, aux loss —
not sizes), `entrypoint: mamba` if the stack contains Mamba layers.
Copy the closest existing family (`deepseek_v3`, `qwen3_next`,
`nemotron_h`) as a template.

## 2. Mechanism support, in order of preference
1. **Native Megatron CLI args** — check first; this pin auto-generates
   flags from TransformerConfig dataclass fields (arguments.py:1655), so
   most config fields are already reachable. Wire them in
   `src/utils/megatron_args.py::_model_args` behind a family-gated config
   key, with an emission test in `tests/unit/test_megatron_args_families.py`.
2. **ModuleSpec** — custom layer composition via `src/specs/` (see
   `ngpt_layer_spec.py`).
3. **Patch** — last resort, only when neither reaches the call site; follow
   `docs/patches_cookbook.md` (unique upstream target, registry conflicts
   are import-time errors).

## 3. Scale realization (`configs/base/scale/<budget>_<family>.yaml`)
Declares the budget (`non_embedding_params`) and the dimensions realizing
it. Add the formula for any *new* mixer/MLP mechanism to
`src/utils/arch_params.py` (TDD with hand-computed vectors), then size with:

    python tools/size_check.py base/family=<name> base/scale=<scale>

Land within ±2% of the budget and add the pair to `BAKEOFF_PAIRS` in
`tests/unit/test_scale_budget.py`. Never tweak `non_embedding_params` to
match the dims — it drives the token budget and the dataset cache key;
tweak the dims.

## 4. Verify
1. `pytest tests/unit -q` (emission + budget gates).
2. Dry-run: `python -m launchers.train_megatron base/family=<name>
   base/scale=<scale> experiment=optim/adam training_regime=ablation_40x
   scheduler=wsd cluster=dev --dry-run` — inspect the emitted command.
3. Pin guard (after any submodule bump):
   `tests/integration/test_megatron_pin_features.py`.
4. GPU smoke (~30M tokens): launch with `training.tokens_per_param=0.05`,
   keep `training.save_enabled=true` (disabling it breaks the Megatron
   wandb writer). Compare the wandb_trainable_params total against
   `tools/size_check.py` — agree within ~2% (embedding params accounted
   separately) before any real run.
