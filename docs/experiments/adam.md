# Experiment: adam

**Family**: optim
**Status**: exploratory
**Owner**: zqiu
**Created**: 2026-05-28

## Hypothesis
Plain AdamW on all parameters is the control arm of the optimizer comparison —
the reference point every other optimizer (Muon, POET) must beat at matched
compute. Not a hypothesis to confirm so much as the baseline to beat.

## Method summary
- AdamW on all parameters: `lr=1e-3`, `betas=(0.9, 0.95)`, `eps=1e-8`,
  `weight_decay=0.1`.
- Cosine LR schedule with linear warmup; standard SwiGLU / GQA / RMSNorm
  architecture from the active base family. No behavioral patches (only the
  logging-only ETA patch).
- Currently identical to `configs/experiments/champion.yaml`: the AdamW
  baseline fills the (uncrowned) champion slot, so adam runs report
  `config_diff_from_champion == "champion"`.

## Timeline
- 2026-05-28: split out from the champion baseline into a first-class optim
  experiment (sibling to poet / muon); `train_adam.sh` now selects
  `experiment=optim/adam`.

## Runs
- Ablation ladder: (pending)
- 2.4B confirmation: (pending)
- 7B anchor: (pending)

## What worked
(baseline — control arm)

## What didn't
(baseline — control arm)

## Follow-ups
- Keep the experiment + optim sections in sync with `champion.yaml` until a
  non-AdamW method is promoted into the champion slot.

## References
- Hu et al. 2024 (arXiv:2404.06395) — MiniCPM.
