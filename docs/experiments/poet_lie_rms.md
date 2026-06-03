# poet_lie_rms — Lie momentum + W-free RMS scaling (§2 Stage 2)

[`poet_lie`](./poet_lie.md) with Pion's **Stage 2 RMS scaling** turned on
(`optim.poet.lie_rms: true`), per
[docs/rms_normalization_poet_interval1.md](../rms_normalization_poet_interval1.md).

After the element-wise Adam direction `A` (Stage 1), the optimizer scales the
rotation generator by

```
α = rms_c · √(n_blocks·block_size) / (‖A‖_F + ε)
oft_R = lr · α · A
```

so the **per-plane rotation angle** is consistent across matrices of any width
(`√(n_blocks·block_size) = √d`, read off the `oft_R` shape — blocking-invariant).
This is **W-free**: no `W` access, no merge change. Net effect:
`‖oft_R‖_F = lr·rms_c·√d`, independent of the gradient magnitude. `rms_c` is the
single new hyperparameter (the RMS target; Pion uses ~0.2).

Everything else matches `poet_lie` (single-step, `block_count=1`,
`reinit_period=-1`, element-wise `v`, Cayley). Run with
[`scripts/train_poet_lie_rms.sh`](../../scripts/train_poet_lie_rms.sh) or
`experiment=optim/poet_lie_rms`; tune `lr` / `lie_rms_c` (RMS scaling enables
larger LR). Composes with alternating (`optim.poet.lie_alternating=true`).

**Deferred:** the Pion-faithful `‖A·W‖_F` normalization (needs `W`, hence the
merge), low-order Cayley / E2 exp, sharded merge.
