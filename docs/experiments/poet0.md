# poet0 — Single-Step POET

Baseline of the POET-X × Pion pipeline
([docs/poetx_pion_pipeline.md](../poetx_pion_pipeline.md) §1). Same stack as
[`experiment=optim/poet`](./poet.md), with two cadences split apart:

- **`merge_period: 1`** — every step, fold `R(oft_R)` into the base weight and
  reset `oft_R` to identity. The per-step rotation angle stays small.
- **`reinit_period: 400`** — every 400 steps, *also* resample the block
  permutation Ψ and reset Adam momentum. Between boundaries the momentum
  persists in one coherent coordinate frame; the `oft_R` master **value** is
  zeroed on every fold regardless (prevents the just-merged rotation from
  springing back on the next optimizer step).

Everything else matches `optim/poet`: stock Megatron-Adam on `oft_R`
(`use_poet_adam: false`), k=3 Cayley (`parameterization: cayley`), two-sided
rotations, `scale: 0.5`, and qkv/fc1 unfusing. `reinit_period` must be a
multiple of `merge_period` (validated at arg-build time).

Run with [`scripts/train_poet0.sh`](../../scripts/train_poet0.sh) (60m dev
scale by default) or `experiment=optim/poet0` on any launcher.

**Out of scope** (later ablations, layered on this baseline): tangent-space
gradient, scalar-`v` Lie momentum, RMS-α step size, low-order Cayley,
alternating single-sided update, sharded merge.
