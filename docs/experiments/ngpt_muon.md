# ngpt_muon — nGPT architecture trained with Muon

The nGPT hypersphere architecture (see [ngpt.md](ngpt.md)) trained with the
vendored Kimi/Moonlight Muon optimizer (see [muon_kimi.md](muon_kimi.md))
instead of nGPT's native `ngpt_adamw`. This is the architecture-vs-optimizer
ablation: it isolates whether nGPT's gains come from the hypersphere
*architecture* or from its *optimizer*.

## Hypothesis
nGPT bundles a normalized-on-the-hypersphere architecture with an AdamW-on-the-
sphere optimizer. Swapping in Muon (Newton-Schulz-orthogonalised, RMS-scaled 2-D
updates + internal AdamW on the rest) keeps the architecture fixed and varies
only the optimizer, so a loss delta vs `arch/ngpt` is attributable to the
optimizer choice.

## Mechanism (composition)
The nGPT *architecture* is keyed on `experiment.kind == 'ngpt'` (not on
`optim.type`): `src/utils/megatron_args.py` emits `--ngpt` and the scaling-vector
inits from `_ngpt_arch_args`, independent of the optimizer branch. So
`optim.type=muon_kimi` keeps the full hypersphere model.

The patch list keeps the architecture patches (`ngpt_apply_spec`,
`ngpt_normalize_step`) but replaces `ngpt_optimizer_setup` with
`muon_kimi_optimizer_setup` — both target `get_megatron_optimizer`, and the
patch registry forbids listing two patches on one target. Dropping
`ngpt_optimizer_setup` removes nGPT's zero-WD bucketing of the scaling vectors,
so the config sets `optim.weight_decay=0.0` to keep every param at zero WD (the
nGPT regime).

## Scope / caveats
- Single GPU (the `muon_kimi` builder rejects TP/PP/distributed-optimizer/fp16).
- LR defaults to the `muon_kimi` baseline (`1e-3`); the nGPT+Muon combo likely
  wants its own sweep — nGPT's `15e-4` is for AdamW-on-sphere, not Muon's
  RMS-scaled update. Use `scripts/sweep_muon_kimi_lr.sh` as a starting point.

## How to run
```bash
scripts/train_ngpt_dev_muon.sh                 # llama3 60m dev, 40x regime
# or
python -m launchers.submit \
    base/family=llama3 base/scale=60m \
    experiment=arch/ngpt_muon \
    training_regime=ablation_40x cluster=h100_de seed=42
```
Config: [configs/experiments/arch/ngpt_muon.yaml](../../configs/experiments/arch/ngpt_muon.yaml).
