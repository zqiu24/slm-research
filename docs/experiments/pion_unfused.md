# pion_unfused

Unfused variant of `pion` (see `docs/experiments/pion.md`). Identical Pion
optimizer and defaults, but `base.model.unfuse_qkv`/`unfuse_fc1` are set so the
`model_unfuse_linears` patch splits the fused qkv/fc1 weights into separate
projections at model-build time.

- **Effect:** Pion's internal per-head qkv / up-gate split goes inert (the fused
  `linear_qkv`/`linear_fc1` tensors no longer exist); each separate q/k/v/up/gate
  is instead rotated as a plain 2-D matrix (muon-like granularity).
- **Use:** ablation against the fused `pion` default to compare internal-split vs
  whole-projection rotation. Same single-GPU scope and wiring as `pion`.

Run with `scripts/train_pion_dev.sh experiment=optim/pion_unfused`.
