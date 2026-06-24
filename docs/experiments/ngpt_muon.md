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
- LR defaults to the `muon_kimi` baseline (`1e-3`), which is cold for this combo;
  it has been swept separately via `scripts/sweep_ngpt_muon_lr.sh` (optimum
  ≈6–8e-3) — see **Results** below.

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

## Results — LR sweep (`scripts/sweep_ngpt_muon_lr.sh`)

llama3-60m / `ablation_40x` (40 tpp, seq 256, gbs 1024) / 8×H100 DDP, final eval
@ iter 9155. Recipe matched to the *tuned dense `muon_kimi`* baseline so the only
difference vs dense muon is the nGPT architecture: `optim.weight_decay=0.1`,
`optim.ngpt.no_warmup=false` (1% warmup), momentum 0.95 + nesterov, ns_steps 5.
Grid brackets dense-muon's optimum (4e-3) and extends past nGPT-adam's (1e-2).

| lr     | run    | val loss   |
|--------|--------|------------|
| 0.002  | lr20   | 3.5209     |
| 0.003  | lr30   | 3.5074     |
| 0.004  | lr40   | 3.4981     |
| 0.005  | lr50   | 3.4926     |
| 0.006  | lr60   | 3.4884     |
| 0.008  | lr80   | **3.4882** ← best |
| 0.010  | lr100  | 3.4928     |
| 0.020  | lr200  | 3.5282     |

Optimum is a **flat basin at lr 6e-3–8e-3** (lr60/lr80 tied within noise → 3.488).
The hypersphere arch wants a hotter lr than dense muon (4e-3) but cooler than
nGPT-adam (1e-2), as predicted.

### The completed optimizer × architecture matrix

|              | adam            | muon                 |
|--------------|-----------------|----------------------|
| dense llama3 | 3.4935 (3e-3)   | **3.4514** (4e-3) ← best overall |
| nGPT         | 3.4583 (1e-2)   | 3.4882 (6–8e-3)      |

(adam = `sweep_adam_lr` / `sweep_ngpt_lr`; dense muon = `muon_kimi.log`; all four
verified from the `/lustre/home/zqiu/log` sweep logs.)

### Finding — nGPT and Muon are anti-synergistic
nGPT+Muon (3.4882) is the **worst of the four combos**, landing essentially back
at the plain dense+adam baseline (3.4935). The two gains do not add — they cancel:

- nGPT arch alone helps adam: **−0.035** (3.4935 → 3.4583)
- Muon alone helps dense: **−0.042** (3.4935 → 3.4514)
- Stacking them: nGPT *hurts* Muon **+0.037** (3.4514 → 3.4882); Muon *hurts*
  nGPT **+0.030** (3.4583 → 3.4882)

Mechanically consistent: Muon already controls update geometry (Newton-Schulz
orthogonalization + RMS scaling) and nGPT constrains weights/reps to the
hypersphere. Two overlapping geometric constraints are redundant/conflicting, not
complementary. **The best single recipe stays dense llama3 + Muon (3.4514).**

### Caveat — leaderboard-matched leg, not reference-nGPT leg
This sweep used `wd=0.1` (matched to `muon_kimi`/leaderboard). Because the
composition drops `ngpt_optimizer_setup`, there is no zero-WD bucketing, so
`wd=0.1` also decays the nGPT scaling vectors (alpha/sqk/suv/sz). The vs-dense-muon
comparison is still apples-to-apples (both `wd=0.1`). The pure nGPT-reference A/B
(`optim.weight_decay=0.0 optim.ngpt.no_warmup=true`) at lr≈6–8e-3 has **not** been
run — that is the one remaining variant if the anti-synergy result warrants
confirmation under nGPT's native zero-WD regime.
