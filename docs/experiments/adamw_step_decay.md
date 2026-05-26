# Experiment: adamw_step_decay

**Family**: optim
**Status**: exploratory
**Owner**: zqiu
**Created**: 2026-05-26

## Hypothesis
The piecewise-constant step-decay LR schedule from the Megatron-poet
DeepSeek-3B reference recipe — peak for 80% of the budget, then 0.316×,
then 0.1× — is a simpler alternative to WSD/cosine on MoE pretraining.
Hypothesis: matches WSD on final loss within seed variance on the
ablation ladder, with a less tuned schedule shape (no separate decay
fraction / decay shape).

## Method summary
AdamW on all parameters with a custom LR schedule installed via patch:
`src/patches/lr_decay_style_step.py` extends Megatron's
`OptimizerParamScheduler.get_lr` with a `step` branch. After linear
warmup, LR is `peak` while `step / lr_decay_steps < 0.8`, drops to
`peak * 0.316` between 0.8 and 0.9, and `peak * 0.1` from 0.9 to 1.0.
Past `lr_decay_steps` it pins at `min_lr` per Megatron's convention.
Ratio and coefficient lists are passed via `--lr-decay-step-ratio` and
`--lr-decay-step-coeff` (registered in `add_slm_args`) and read off
`get_args()` inside the patched scheduler `__init__`.

Configured for the DeepSeek-3B recipe defaults: ratios `[0.8, 0.9]`,
coefficients `[0.316, 0.1]`, peak LR 9e-4. Same Adam hyperparameters as
`configs/experiments/champion.yaml`.

## Timeline
- 2026-05-26: ported from Megatron-poet/megatron/core/optimizer_param_scheduler.py
  alongside the new `deepseek_v3_3b` scale.

## Runs
- Ablation ladder: TBD
- 2.4B confirmation: TBD
- 7B anchor: TBD

## What worked
- N/A (not yet run).

## What didn't
- N/A.

## Follow-ups
- Sweep ratio/coeff pairs on the 600M ladder rung to confirm 0.316/0.1
  are not load-bearing — likely a four-cell grid over `[0.5, 0.316]` ×
  `[0.1, 0.05]` is enough.
- Compare against `final_wsd_decay_only` on a `deepseek_v3_3b` 20× run.

## References
- [Megatron-poet train_DeepSeek_3b.sh](https://github.com/Sphere-AI-Lab/Megatron-poet/blob/main/training_scripts/train_DeepSeek_3b.sh)
  — `--lr-decay-style step --lr-decay-step-ratio 0.8 0.9 --lr-decay-step-coeff 0.316 0.1`
- [Megatron-poet optimizer_param_scheduler.py](https://github.com/Sphere-AI-Lab/Megatron-poet/blob/main/megatron/core/optimizer_param_scheduler.py)
  — reference implementation of the `step` branch.
