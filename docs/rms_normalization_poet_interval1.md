# RMS Normalization for POET Interval-1

Spec for Pion's update applied to the interval-1 POET rewrite, operating on the
**autograd gradient w.r.t. Q** (not a manually constructed skew gradient).

Pion's update is **two stages**: (1) full Adam on the skew gradient → direction `A`;
(2) RMS scaling → rotation magnitude. This doc covers both. **The whole thing runs on
the autograd gradient `∂L/∂Q` with no W needed** (W-free default); Pion's exact
`‖A·W‖_F` scaling is an optional variant that brings W back in but is ≈ identical in
the frozen-spectrum setting. The *direction* (Stage 1) is shown equivalent to Pion's
at interval-1.

Scope assumption: **dense** orthogonal transforms (no block-diagonal structure),
**interval-1** (merge every step), **single side per step** (alternating). Block
structure and the √b variant come later.

---

## 1. The setting

- Trainable parameter: `Q`, skew-symmetric (stored as upper-triangle). The
  rotation is `R = CNP_k(Q)`.
- Frozen/merged weight: `W` (read-only; no gradient, no optimizer state on W).
- We do **not** hand-compute `∂L/∂W` or a manual skew gradient. We take what
  autograd gives: `Γ = ∂L/∂Q`.
- One side active per step (out-side shown; in-side symmetric).

---

## 2. Two stages, not one: Adam direction THEN RMS scaling

**Important structural correction.** Pion does not have a single "RMS norm." It has
**two distinct stages**, and conflating them is a mistake:

1. **Adam step (produces a direction).** Full Adam — first moment + *element-wise*
   second moment + element-wise division — on the skew gradient. This gives an
   un-scaled update *direction* `A`.
2. **RMS scaling (controls the rotation magnitude).** A scalar `α` computed from
   `A` (and W) is applied at the exponential, making the rotational magnitude
   scale-consistent across matrices and unlocking large LR.

Both are present in the final Pion algorithm. Stage 1 handles per-coordinate
adaptivity of the direction; stage 2 handles the overall rotational scale. They
are not alternatives.

### Stage 1 — full Adam on the skew gradient (element-wise v)

Per side, with skew gradient `Γ` (= `∂L/∂Q` in our autograd path):

```
m ← β1 m + (1−β1) Γ                      # first moment (skew)
v ← β2 v + (1−β2) (Γ ⊙ Γ)                # second moment — ELEMENT-WISE square
A ← − m / (√v + ε)                       # direction — ELEMENT-WISE division
```

This is ordinary Adam machinery applied to the skew gradient. Pion's ablation found
the **first-order + second-order Lie-algebra combination is strongest** — so the
second moment is *not* optional and is *not* a per-block scalar. It is the standard
element-wise `⊙` / `√v`, exactly as in AdamW. (An earlier version of this spec
argued for a scalar-per-block `v` on so(n)-isotropy grounds; Pion's empirical result
overrides that — use element-wise.)

### Stage 2 — RMS scaling at the exponential

You normalize the direction's magnitude so rotational scale is consistent across
matrices. **Two choices, and the W-free one is a valid default:**

**(default, W-free) — normalize by the direction:**
```
α = c · √(d_out · d_in) / ( ‖A‖_F + ε )
```
Needs **no W** — just the norm of the Adam direction you already have. This makes the
*rotation angle* scale-consistent across matrices. The whole update (Stage 1 + 2)
then runs on autograd gradients alone, no W anywhere in the optimizer.

**(Pion-faithful) — normalize by the effect on W:**
```
α = c · √(d_out · d_in) / ( ‖A·W‖_F + ε )      # out-side
```
Needs W explicitly. This makes the *weight change* ‖ΔW‖ ≈ ηα‖A·W‖ scale-consistent
across matrices (Pion's actual formula).

**When do they differ?** Only when weight matrices have very different scales. Since
`‖A·W‖_F ≈ ‖A‖_F · ‖W‖_F` (roughly), the two differ by a `‖W‖_F` factor. If all
weights have comparable scale, that factor is ~constant and washes into `c` — the two
are **identical in effect**. They only diverge when weight scales vary a lot across
matrices, where Pion's version compensates (equal ΔW) and the W-free one does not
(equal rotation angle).

**For over-parameterized POET specifically:** the spectrum is frozen in `W0`, so each
effective weight's scale is fixed and known (= `W0`'s spectrum, preserved exactly by
the orthogonal updates). With standard uniform-scale `W0` init, `‖W‖_F` is comparable
across matrices, so the W-free `‖A‖_F` normalization is ≈ Pion's `‖A·W‖_F`. The frozen
spectrum *weakens* the case for needing W here — the thing W compensates for (varying
weight scale) is uniform by construction. **So default to the W-free version**; treat
`‖A·W‖_F` as an ablation if weight-scale variation turns out to matter.

- `c` — RMS target constant (the one new hyperparameter).
- `√(d_out·d_in)` — dimension-consistency constant; converts the Frobenius norm into
  an average per-plane rotation angle so matrices of different widths take comparable
  per-plane steps.
- the denominator (`‖A‖_F` or `‖A·W‖_F`) — magnitude control; caps the per-step
  rotation, preventing divergence at high LR.

---

## 3. Where it sits in the per-step update

Single side, interval-1, dense (out-side shown):

```
1.  Γ   ← ∂L/∂Q                          # autograd; W used internally (see §4)
2.  m   ← β1 m + (1−β1) Γ                 # Stage 1: Adam first moment (skew)
        v   ← β2 v + (1−β2) (Γ ⊙ Γ)       #          element-wise second moment
        A   ← − m / (√v + ε)              #          element-wise division → direction
        (bias-correct m, v by 1−β1^t, 1−β2^t if desired)
3.  α   ← c·√(d_out·d_in) / (‖A‖_F + ε)   # Stage 2: RMS scaling — W-FREE (default)
        #  Pion-faithful alt: ‖A·W‖_F (needs W); ≈ identical if weight scales uniform
4.  R   ← E2(A, α) = I + ηαA + ½(ηαA)²    # second-order exp; small angle ⇒ sufficient
5.  W   ← R W                             # apply + merge ⇒ W plain again next step
```

`E2` is Pion's two-term exponential `exp(X) ≈ I + X + ½X²`. At interval-1 the angle
is small so this is sufficient and fast (errors don't compound — each step starts
from identity). This replaces the CNP series; if you keep CNP instead, use low order
(k=1 or 2) for the same small-angle reason.

Decay-on-active convention under alternation: update `m, v` for a side only on the
steps where that side is active.

---

## 4. Does it need W? (definitive)

**Short answer: no, not with the W-free default.** The whole update — Stage 1 Adam +
Stage 2 RMS scaling by `‖A‖_F` — runs on the autograd gradient `Γ = ∂L/∂Q` alone. Your
instinct is right.

W appears in the optimizer **only if you opt into Pion's exact `‖A·W‖_F`
normalization**, which is a choice, not a requirement (§2 Stage 2).

| stage | needs W? (W-free default) | needs W? (Pion-faithful) |
|---|---|---|
| compute `Γ = ∂L/∂Q` (backward pass) | implicitly (autograd) | implicitly (autograd) |
| Stage 1: Adam on `Γ` (m, v, A) | **no** | **no** |
| Stage 2: RMS scaling | **no** (`‖A‖_F`) | **yes** (`‖A·W‖_F`) |
| apply / merge | writes W | writes W |

In **all** cases W carries **no gradient and no optimizer state** — the memory
property holds regardless. The only question is whether the optimizer *reads* W:

- **W-free default:** autograd uses W internally to make `Γ` (it always did, as in
  original POET); the optimizer code itself never references W. Equivalent in spirit
  to "we just use the autograd gradient."
- **Pion-faithful:** additionally forms `A·W` (one matmul) in Stage 2 to normalize by
  the weight-change rather than the rotation angle.

**Recommendation for over-parameterized POET:** use the W-free default. The frozen
spectrum makes weight scales uniform by construction (§2), so `‖A·W‖_F ≈ ‖A‖_F ×
const` and the two normalizations nearly coincide — the W buys you little. Keep
`‖A·W‖_F` in your back pocket as an ablation if weight-scale variation across
matrices turns out to matter.

---

## 5. Why the *direction* equals Pion's at interval-1 (Stage 1)

This equivalence is about the **direction** (Stage 1) — the skew gradient that feeds
Adam. (Stage 2 scaling is W-free by default, §4; if you opt into Pion's `‖A·W‖_F` it
differs by a weight-scale factor, near-constant in the frozen-spectrum setting.)

Two objects:

- **Pion's**: `P = skew(G_eff Wᵀ)` — gradient w.r.t. the rotation `R`.
- **Ours**: `Γ = ∂L/∂Q` — gradient w.r.t. the parameter `Q`.

Linked by the chain rule through `R = CNP(Q)`:

```
Γ = ∂L/∂Q = (∂L/∂R) · (∂R/∂Q) = P · CNP'(Q)
```

So `Γ` is Pion's `P` passed through the **CNP Jacobian** `∂R/∂Q`. Whether they
match depends entirely on what that Jacobian does.

CNP series: `R = I + 2Q + 2Q² + 2Q³ + Q⁴`. Differentiate:

```
∂R/∂Q = 2I + 2·∂(Q²)/∂Q + 2·∂(Q³)/∂Q + ∂(Q⁴)/∂Q
         └─┬─┘  └──────────── all carry a factor of Q ────────────┘
        constant                  vanish as Q → 0
```

**Interval-1 ⇒ Q is small** (Q is reborn at 0 and takes one small step before
merge+reset; it never accumulates). With Q small:

```
∂R/∂Q ≈ 2I          # all Q-dependent terms are O(Q), negligible
⇒ Γ ≈ 2P            # autograd gradient is just 2× Pion's object
```

A **scalar multiple**. So at interval-1, `Γ ≈ 2P` — our autograd gradient is just
Pion's skew-gradient object scaled by 2. This means the **input direction to Stage 1
Adam is the same** (up to the scalar 2) whether you form it by autograd-on-Q (ours)
or by hand as `skew(G_eff Wᵀ)` (Pion). You reproduce Pion's skew gradient *for free*,
without forming it by hand — because the small-angle CNP map is locally linear (slope
2), so its derivative is the constant 2.

Caveat on the scalar under element-wise Adam: a pure first-moment update would have
the 2 cancel cleanly under any later normalization. With *element-wise* second moment
(§2 Stage 1), the 2 cancels in `m/√v` too (numerator scales by 2, `√v` scales by 2),
so `A` is unchanged by the scalar — good. The equivalence of the *direction* `A`
holds. Stage 2's `α` (W-free `‖A‖_F`) is likewise unaffected by the scalar; only the
optional Pion-faithful `‖A·W‖_F` differs, and only by the near-constant weight-scale
factor (§4).

### Where it breaks (longer interval)

If Q is allowed to grow (interval > 1), the `Q²,Q³` terms in `∂R/∂Q` are no longer
negligible. They are **matrices, not scalars** — they scale different directions
of `P` differently. Then `Γ = P·(2I + stuff)` is `P` *reshaped*, not *rescaled*,
and a reshaping does **not** cancel under normalization. So `Γ̃ ≠ P̃`, and the
autograd-based normalization diverges from Pion's.

Tightest equivalence: **interval-1 AND low CNP order** (k=1 or 2). Higher k adds
Jacobian terms even at small Q, carrying more of the parameterization's fingerprint
into `Γ`. This is the same regime favored elsewhere (small angles ⇒ low k suffices),
so everything points the same way.

---

## 6. Summary

- Pion is **two stages**: (1) full Adam — first moment + **element-wise** second
  moment + element-wise division — on the skew gradient, giving direction `A`; then
  (2) RMS scaling applied at the (second-order) exp. The first+second Lie-algebra
  combination is Pion's strongest ablation result, so the second moment is not
  optional and not a scalar.
- **You do not need W.** Both stages run on the autograd gradient `Γ = ∂L/∂Q`:
  Stage 1 is Adam on `Γ`; Stage 2 (default) normalizes by `‖A‖_F`. W has no gradient
  or optimizer state and the optimizer never reads it. Autograd uses W internally to
  produce `Γ` — exactly as in original POET.
- Pion's *exact* formula normalizes by `‖A·W‖_F` (W explicit) to make the
  weight-change consistent rather than the rotation angle. This is optional and, in
  the frozen-spectrum POET setting (uniform weight scales), ≈ identical to the W-free
  version. Keep it as an ablation only.
- At interval-1 with low CNP order (or the two-term exp), the **direction** `A` is
  equivalent to Pion's, because the CNP Jacobian collapses to `2I` (`Γ ≈ 2P`) and the
  scalar 2 cancels in `m/√v`.
