# poet_lie_orth_alt_x

True single-side alternating POETX (`AlternatingPOETXLinear`) on the champion
`lie_ortho` recipe (head-OFF, lr 3e-3, c=8, distributed). Trains one rotation side
per step (out on even iterations, in on odd), short-circuiting the frozen side's
backward `M`, Cayley build, and weight-fold. Each side's first-moment momentum
advances only on its active steps (true single-side — a different optimizer than the
both-side-momentum `poet_lie_alt`).

- **Design:** `docs/superpowers/specs/2026-06-08-alternating-poetx-single-side-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-08-alternating-poetx-single-side.md`
- **Baseline:** the both-sides champion `dwynpk9y` (val/loss 3.5528).
- **Expectation:** d³-machinery speedup (small at 60m, grows with d); quality is an
  open ablation (single-side per step may help or hurt).
