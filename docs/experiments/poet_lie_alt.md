# poet_lie_alt — Lie-algebra momentum + alternating single-sided (§6)

[`poet_lie`](./poet_lie.md) with the **alternating single-sided update** (pipeline
doc [§6](../poetx_pion_pipeline.md), Pion Eq. 8) turned on
(`optim.poet.lie_alternating: true`).

Per step, only **one** rotation side's `oft_R` is written — `oft_R_out` on even
steps, `oft_R_in` on odd — while the Lie-algebra momentum (`lie_m`/`lie_v`)
accumulates on **both** sides every step (paper App. D.1). The inactive side
stays at 0 → identity rotation → no-op fold, so there is no merge change. This is
purely a write-gate in the optimizer (`LieAlgebraMomentum`); `lie_alternate_every`
(default 1) sets how many steps each side is held before flipping.

Everything else matches `poet_lie`: single-step (`merge_period=1`),
`block_count=1`, `reinit_period=-1` (fixed Ψ, persistent momentum), Cayley, and
`scale=0.5`. Run with [`scripts/train_poet_lie_alt.sh`](../../scripts/train_poet_lie_alt.sh)
or `experiment=optim/poet_lie_alt`.

Ablate against `poet_lie` (two-sided every step) for the §9 step-4 comparison:
~half the per-step optimizer write/compute at a small loss cost in Pion's
setting — watch the loss-curve shape and whether single-sided coverage slows
convergence.

**Deferred** (later increments): RMS-α step scaling, low-order Cayley, exact
tangent gradient, sharded merge.
