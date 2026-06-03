# poet_lie — POET × Pion Lie-algebra momentum (increment 1)

Increment 1 of the POET-X × Pion pipeline
([docs/poetx_pion_pipeline.md](../poetx_pion_pipeline.md) §2–§3, §9 step 1).
Same single-step POET stack as [`poet0`](./poet0.md) — `merge_period=1`,
`block_count=1`, two-sided, Cayley — but the `oft_R` optimizer is swapped from
stock Megatron-Adam to **Pion's Lie-algebra momentum** via
`optim.poet.q_optimizer: lie_algebra`.

Per step (`oft_R` born at identity), on the identity-point tangent gradient:
`m ← β1·m + (1−β1)·g`; `v ← β2·v + (1−β2)·‖·‖²` (`lie_v_mode: scalar`) or
element-wise (`elementwise`); `A = −m/(√v+ε)`; `oft_R ← lr·A`. The merge
exponentiates and folds it into `W`. Momentum **persists** across the fold
(buffers `lie_m`/`lie_v`, never reset); `reinit_period: -1` keeps Ψ fixed so the
momentum stays coordinate-coherent.

Step magnitude = cosine-scheduled `lr · scale` (no RMS-α yet — expect to tune LR
*down* vs poet0; RMS-α is a later increment). Run with
[`scripts/train_poet_lie.sh`](../../scripts/train_poet_lie.sh) or
`experiment=optim/poet_lie`.

**Deferred** (later increments, §9 steps 2–5): RMS-α step scaling, low-order
Cayley, alternating single-sided, exact/block-diagonal tangent gradient, sharded
merge.
