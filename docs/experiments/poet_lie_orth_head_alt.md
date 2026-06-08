# poet_lie_orth_head_alt

Head-aligned attention on the POETX forward frame with a **permuted multi-block
residual** side (`HeadAlignedPOETXLinear`), on top of the alternating champion
(`lie_ortho` + `lie_alternating`, val/loss 3.5332 head-off).

Hypothesis: head-alignment hurt (−0.014) partly because the legacy layer's residual
side is a single dense **perm-free** block. Giving the residual side a real Ψ +
multiple blocks (the POETX-native shape) + alternating may flip the head penalty.

- **Design:** docs/superpowers/specs/2026-06-08-head-aligned-poetx-permuted-resid-design.md
- **Plan:** docs/superpowers/plans/2026-06-08-head-aligned-poetx-permuted-resid.md
- **Baseline:** alternating champion `1ynrrimu` (head-off, 3.5332).
- **Knob:** `head_resid_block_count` (sweep, default 4).
