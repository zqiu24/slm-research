# poet_lie_orth_alt

Integrated alternating POETX (both-momenta) on the champion `lie_ortho` recipe
(head-OFF, lr 3e-3, c=8, distributed). Plain `POETXLinear` (`single_step_x`) +
`lie_alternating`: the optimizer writes one rotation side per step while **both**
first-moment momenta stay fresh (`POETXSingleStepFunction` returns both grads). This
is the integrated path, **not** the regressed true-single-side `poet_lie_orth_alt_x`
(which froze the inactive momentum and regressed to 4.22).

- **Design:** `docs/superpowers/specs/2026-06-08-alternating-poetx-integrated-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-08-alternating-poetx-integrated.md`
- **Target:** the `lie_ortho` + `lie_alternating` champion (`1ynrrimu`, val/loss ≈3.5332).
- **Phase 1:** plain POETX, merge folds both sides (frozen side `oft_R=0` → identity →
  no-op) — reproduces the champion at POETX forward speed, zero new code.
- **Phase 2:** active-only merge fold (skip the frozen side's Cayley) — bit-identical
  fold, expected `perf/step_time_s` drop at merge time.
