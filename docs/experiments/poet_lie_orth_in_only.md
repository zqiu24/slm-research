# poet_lie_orth_in_only

Pure one-sided POET (input side). `InOnlyPOETXLinear` trains **only** `oft_R_in` for
the whole run via `optim.poet.single_step_x_one_sided: in`; `oft_R_out` stays at its
zero init (identity). The frozen side's forward rotation, backward gradient, optimizer
momentum, and merge fold are all short-circuited.

- **Recipe:** the champion `lie_ortho` single-side recipe (`poet_lie_orth_alt_x`) —
  `single_step_x`, `lie_ortho` (c=8, muon, 5 NS, distributed), lr 3e-3, `block_count 1`,
  `merge_period 1`, `scale 0.5` — but the side is **fixed**, not alternating.
- **Why no regression:** `AlternatingPOETXLinear` regressed from *alternating* (stale
  momentum when a side reactivates). With the side fixed, the trained side's momentum
  advances and applies every step; the frozen side never moves `W`.
- **Design:** `docs/superpowers/specs/2026-06-19-poet-one-sided-mode-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-19-poet-one-sided-mode.md`
- **Ablation target:** the both-sides champion (val/loss ≈3.5332) and `poet_lie_orth_alt_x`.
