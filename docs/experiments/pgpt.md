# pgpt — nGPT architecture minus the per-step weight projection, trained with POET

pgpt is the nGPT hypersphere architecture with the **explicit per-step weight
projection removed from the model**, co-trained with POET. It is a distinct base
model — NOT `arch/ngpt_poet`, which keeps vanilla nGPT (and its per-step renorm)
and merely swaps the optimizer.

See the design spec:
[docs/superpowers/specs/2026-06-18-pgpt-design.md](../superpowers/specs/2026-06-18-pgpt-design.md).

## Why drop the per-step renorm
POET parametrizes each trained linear as `A·W_base·B` with block-orthogonal
`A,B`, so it preserves each matrix's singular-value spectrum exactly — the
conditioning role nGPT's per-step projection played. Hidden states stay on the
sphere via the runtime residual-blend and Q/K `justnorm`s (activation ops POET
never touches). The two sphere matrices POET does not wrap (token embedding +
lm_head) keep a targeted per-step renorm installed by `pgpt_optimizer_setup`.

## Mechanism
- `pgpt_apply_spec` swaps the layer spec, stamps the nGPT config fields, runs the
  one-shot init normalize, and registers both the full role map and the
  embedding/lm_head post-step subset.
- `pgpt_optimizer_setup` cooperatively wraps `setup_model_and_optimizer`
  (`targets=()`): zero-WD for the scaling params, and a `optimizer.step` hook that
  re-projects only embedding + lm_head.
- `poet_merge_step` is included (no `train_step` collision, unlike `ngpt_poet`);
  inert at `merge_period=0`, flip `optim.poet.merge_period>0` to enable merges.

> **EXPERIMENTAL — GPU-smoke before trusting the loss.** Confirm the
> `[pgpt] optimizer setup …` log line and the `--ngpt`/`--poet` arg evidence both
> appear, and that POET `ortho_err` stays bounded.

## Run
`scripts/train_pgpt_dev.sh` (60m llama3 backbone, single GPU; POET keeps the
Megatron distributed optimizer off automatically).
