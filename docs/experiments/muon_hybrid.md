# Experiment: muon_hybrid

**Family**: optim
**Status**: exploratory
**Owner**: zqiu
**Created**: 2026-04-23

## Hypothesis
Muon's orthogonal Newton-Schulz step helps dense weight matrices by
constraining the update to a well-conditioned subspace, but the same
update is unstable (or meaningless) for 1-D parameters. A hybrid — Muon on
2-D linear weights, Adam on embeddings / norms / biases / LM head, with
independent LR schedules per group — should capture the Muon benefit
without the instability.

## Method summary
- Param partitioning: `linear_weights` → Muon (5 Newton-Schulz iterations,
  coefficients `(3.4445, -4.7750, 2.0315)`); `{embedding, norm, bias, lm_head}` → Adam.
- Two parallel LR schedules. Default: Muon `lr=2e-3`, Adam `lr=1e-3`, both
  under the regime's WSD schedule.
- No architecture changes. No required capabilities.

## Timeline
- 2026-04-23: YAML + doc created; implementation pending.

## Runs
- Ablation ladder: (pending)
- 2.4B confirmation: (pending)
- 7B anchor: (pending)

## What worked
(pending first runs)

## What didn't
(pending first runs)

## Follow-ups
- Sweep Muon LR in {1e-3, 2e-3, 4e-3} before finalising.
- Compare `ns_steps=5` vs `ns_steps=3` for compute/quality trade-off.

## References
- Jordan 2024 "Muon" blog post.
- Moonlight paper (arXiv:2502.16982).
