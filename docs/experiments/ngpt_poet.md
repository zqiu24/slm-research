# ngpt_poet — nGPT architecture trained with POET (EXPERIMENTAL)

The nGPT hypersphere architecture (see [ngpt.md](ngpt.md)) trained with POET
(see [poet.md](poet.md)) instead of nGPT's native `ngpt_adamw`: each 2-D linear
is POET-ised (frozen base weight + trained block-orthogonal delta `oft_R`).

> **EXPERIMENTAL — GPU-smoke before trusting the loss.** POET re-wraps the nGPT
> linears (`poet_apply_to_model` runs *after* the nGPT model is built), and the
> interaction between POET's frozen-base + orthogonal-delta parameterisation and
> nGPT's per-step L2 hypersphere projection / weight-norm role map / sqk-suv-sz
> scaling is not yet validated. Orthogonal rotations preserve row norms (so an
> orthogonal delta is in principle compatible with unit-norm rows), but patch
> *ordering* and the weight-norm role map have bitten nGPT before — confirm both
> `[nGPT] applied spec` (or the `--ngpt …` arg evidence) and POET orbit logs
> appear, and that `ortho_err` stays bounded.

## Hypothesis
Whether nGPT's gains survive an orthogonal-delta parameterisation of the
hypersphere weights — i.e. learning rotations of frozen base rows rather than the
rows themselves, while still projecting to the sphere each step.

## Mechanism (composition)
nGPT architecture is keyed on `experiment.kind == 'ngpt'` (see
[ngpt_muon.md](ngpt_muon.md)), so `optim.type=poet` keeps the hypersphere model.

The patch registry forbids two patches on one target, so two POET patches are
intentionally omitted to avoid collisions with the nGPT architecture patches:

- `poet_unfuse_te_impl` (wraps `core_transformer_config_from_args`, collides with
  `ngpt_apply_spec`) — only flips `transformer_engine→local`, a no-op here because
  the config pins `base.model.transformer_impl=local`.
- `poet_merge_step` (wraps `train_step`, collides with `ngpt_normalize_step`) —
  only fires when `merge_period>0`, so POET runs in the **`merge_period=0`
  no-merge regime**: `oft_R` is trained continuously and never folded. If
  periodic merge is later wanted on nGPT, it needs a single combined `train_step`
  patch doing both normalize and merge.

`sandwich_norm_apply` is also dropped (it would collide with `ngpt_apply_spec` on
`gpt_builder`; it is a no-op unless `--use-sandwich-norm`).

## How to run
```bash
scripts/train_ngpt_dev_poet.sh                 # llama3 60m dev, 40x regime
# or
python -m launchers.submit \
    base/family=llama3 base/scale=60m \
    experiment=arch/ngpt_poet \
    scheduler=cosine_poet \
    training_regime=ablation_40x cluster=h100_de seed=42
```
Config: [configs/experiments/arch/ngpt_poet.yaml](../../configs/experiments/arch/ngpt_poet.yaml).
