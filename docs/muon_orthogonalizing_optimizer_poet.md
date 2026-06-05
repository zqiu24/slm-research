# POET Interval-1: Muon-like Orthogonalizing Optimizer

A **separate optimizer variant** for the interval-1 POET rewrite. Instead of the
RMS norm (which preserves the gradient's *relative* per-plane angles and only fixes
their aggregate scale), this variant **orthogonalizes the update direction** so that
**all rotation planes turn by the same angle** — the Muon move.

This is a sibling to the RMS-norm optimizer, not a modification of it. Same pipeline
(autograd Γ → momentum → map-to-rotation → merge); the **only** difference is what
happens to the direction before it becomes a rotation.

Scope: dense single-matrix-per-side first (block-diagonal noted in §6),
**interval-1** (merge every step), **single side per step** (alternating).

---

## 1. The one-line idea

The singular values of a skew matrix **are** its per-plane rotation angles. So:

- **RMS-norm optimizer:** scale all planes by one factor → RMS angle = `c`, but the
  planes keep their *relative* sizes (hot plane still rotates more). Preserves the
  gradient's claim about which planes matter.
- **This optimizer:** force all singular values equal (orthogonalize) → **every plane
  rotates by the same angle `c`**. Discards the gradient's magnitude claim entirely,
  keeps only *which subspace* it identified. This is Muon's bet: trust the direction
  structure, not the per-direction magnitudes.

---

## 2. Why this composes with the pipeline (it does — earlier objections don't apply)

A key distinction from the rejected "replace CNP with Newton-Schulz" idea:

- **Rejected idea:** use NS *as the algebra→group map* (NS instead of CNP/exp). This
  breaks Lie-algebra momentum (NS is not magnitude-respecting, additive-Q no longer
  maps to composed rotation) — see the NS-vs-CNP analysis.
- **This variant:** orthogonalize the *direction* `A`, then **still** map to a
  rotation via CNP/exp. Different and clean:
  - **Momentum intact** — accumulation happens *before* orthogonalization, on the raw
    skew gradient, additively in the algebra as usual. Orthogonalization is applied to
    the *result*, not to the momentum buffer.
  - **Small-angle CNP intact** — orthogonalize first (all σ → 1), then scale by a
    *small* `c` (all angles → `c`, small), then CNP. As long as `c` is small the
    small-angle regime holds. **Order matters:** orthogonalize → scale by small `c` →
    CNP. Skipping the scale would rotate every plane by ~1 radian (way too much).

So the only cost is the orthogonalization compute (§5). No correctness conflict with
momentum or interval-1.

---

## 3. The RMS norm folds in for free

After orthogonalization, all singular values are 1, so for a `d×d` skew direction:

```
‖A_orth‖_F = √(2 · d/2) = √d     (exactly, since every σ_k = 1)
⇒ ‖A_orth‖_F / √d = 1            (the RMS normalizer is automatically 1)
```

So **this optimizer does not compute an RMS norm** — orthogonalization already sets
the per-plane RMS to 1. You scale by `c` directly, and `c` *is* the per-plane angle.
The RMS-norm machinery is replaced by the orthogonalization, not run alongside it.
(Exact under the Löwdin form; the cheap default Muon-NS leaves σ in a band ≈ 1±0.3, so
`c` is then a *nominal* angle — see §5.)

---

## 4. The per-step update

Single side, interval-1, dense (out-side shown; in-side symmetric with right-mult):

```
1.  Γ      ← ∂L/∂Q                        # autograd skew gradient (no manual GWᵀ)
2.  m      ← β1 m + (1−β1) Γ              # Lie-algebra momentum (skew)
           (optional second moment / bias correction — see note)
           A  ← m                          # direction = momentum (skew)
3.  A_orth ← orthogonalize(A)              # planes' σ → ~1; STAYS SKEW (NS preserves skew).
                                          #   default Muon-NS (band ~1±0.3, ~5 steps);
                                          #   exact A·(−A²)^(−1/2) (σ=1, ~20 steps). §5
4.  R      ← CNP_k(c · A_orth)             # all planes rotate by angle c; small c ⇒ low k
           # equivalently two-term exp: R ≈ I + c·A_orth + ½(c·A_orth)²
5.  W      ← R W                           # apply + merge ⇒ W plain next step
6.  swap side for t+1
```

- `c` — the single step-size hyperparameter. After orthogonalization it is **directly
  the per-plane rotation angle**, so it is interpretable (e.g. `c = 0.01` rad/plane).
- No `√d`, no `‖A‖_F` division — orthogonalization replaces them (§3).

**Second moment note.** Unlike the RMS-norm optimizer (which mirrors Pion's full Adam
with element-wise `v`), the second moment is less motivated here: orthogonalization
already discards all magnitude information, so an element-wise `v` that rescales
per-entry magnitudes would be partially undone by the subsequent orthogonalization.
Start **first-moment only** (`A = m`). If you want adaptivity, it belongs *before*
orthogonalization or not at all. This is a genuine design difference from the RMS
optimizer — flagged for ablation.

---

## 5. Orthogonalizing a skew matrix (the one nontrivial op)

`A` is skew. "Orthogonalize" = same eigenplanes, singular values → 1, **result still
skew**. Two methods, both skew-preserving; they differ in *accuracy vs cost*. The
choice is **not** about skewness (the obvious worry below is unfounded) — it is about
how close to `σ = 1` you need to get, and how much compute you spend.

### Newton-Schulz preserves skew (the obvious worry is unfounded)

It is natural to fear that NS returns the polar factor `UVᵀ` — "orthogonal, not skew."
For a **skew** input that does not happen: each NS step is `X ← (a I + b XXᵀ + c (XXᵀ)²) X`,
and for skew `X` the matrix `XXᵀ = −X²` is **symmetric and commutes with `X`**. A skew
matrix times a commuting symmetric matrix is skew, so **every NS iterate stays skew**
(measured: `‖X+Xᵀ‖ ≈ 1e-15` at every step). And the polar factor of a skew matrix is
itself skew — it satisfies `Q² = −I`, which is exactly where skew ∩ orthogonal lives,
and it *equals* the symmetric-orthogonalization result below. So raw NS on the gradient
is a perfectly valid rotation-rate; the only question is which σ it lands on.

### ✓ Default — Muon's quintic NS (fast, approximate): band `≈ 1 ± 0.3`

Run Muon's tuned quintic Newton-Schulz on the per-block skew direction (the existing
`orthogonalize_skew_blocks`). Its coefficients are *designed* to push the singular
values into a band around 1 in ~5 steps and stop — they do **not** converge to exactly
1. So the planes end up *roughly* equal (σ ∈ ~[0.7, 1.1]) and `c` is a **nominal**
angle (realized median ≈ 0.75–1.0·`c`). A cheap `½(X − Xᵀ)` at the end removes the
~1e-15 of float dust. This is the default: cheapest, and a band of roughly-equal
angles is plausibly all the experiment needs.

### ✓ Exact alternative — symmetric / Löwdin orthogonalization `A_orth = A (−A²)^(−1/2)`

When you want *every* plane at *exactly* the same angle (`σ = 1` exactly, so `c` is
the literal angle), use the symmetric (Löwdin) orthogonalization. It sets all singular
values to 1, keeps the eigenplanes, and stays skew. Why it stays skew:

- For skew `A`: `AᵀA = (−A)(A) = −A²`, and `A²` is **symmetric** (`(A²)ᵀ = (Aᵀ)² =
  (−A)² = A²`). So `(−A²)` is symmetric PSD, and `S := (−A²)^(−1/2)` is **symmetric**.
- `S` is a function of `A²`, and `A` commutes with `A²`, so `A` commutes with `S`
  (`SA = AS`).
- Then `(AS)ᵀ = SᵀAᵀ = S(−A) = −SA = −AS`. So `A_orth = AS` is **skew**. ✓

Compute it without leaving the skew structure — run the inverse-square-root NS on the
*symmetric* matrix `−A²`, then a single multiply by `A`:

```
B      = −A²                       # symmetric PSD; reuses the Q² kernel
B_isr  = inverse_sqrt_NS(B)        # inverse-sqrt Newton-Schulz on the SYMMETRIC B
A_orth = A · B_isr                 # skew, by the commuting argument above
```

Cost is the catch: the inverse-sqrt converges slowly on the small singular directions,
so it needs **~15–20 steps (≈4× the quintic)** to reach σ = 1 — and a *very* ill-
conditioned direction still cannot be fully equalized in finite steps. Use it as the
exact-angle ablation against the cheap band.

---

## 6. Block-diagonal note (per-head rotations)

If the rotation is block-diagonal (per-head), orthogonalize **per block**: each `b×b`
block's direction `A_j` is orthogonalized independently (all its `b/2` plane angles → 1),
then scaled by `c`. So every plane in every head rotates by `c`. This is the natural
fit for independent heads — no cross-head coupling, each head's rotation fully
equalized. Batches over the block dimension on existing kernels. The per-block √b that
the RMS optimizer needs is moot here, since orthogonalization sets each block's RMS to 1
regardless of `b`.

---

## 7. RMS-norm optimizer vs this optimizer — the experiment

Both share the pipeline; they differ in one operation on the direction. Run as siblings:

| | direction → rotation | per-plane angles | keeps gradient's relative plane info? | cost |
|---|---|---|---|---|
| **RMS-norm** | scale by `c√d/‖A‖_F` | proportional to gradient | **yes** | free (a norm) |
| **this (Muon-like)** | orthogonalize, scale by `c` | **≈ `c`** (band; exact w/ Löwdin) | **no** | NS iterations |

The experiment answers a clean question: **for rotational updates in over-parameterized
POET, are the gradient's relative per-plane angles signal or noise?**

- RMS-norm wins → relative angles are *informative*; equalizing them throws away signal.
- This wins → relative angles are *misleading* (conditioning artifacts); Muon's bet
  holds for rotations too.

Caveat worth stating: Muon's evidence is for **additive** weight updates
(`W ← W − α·NS(grad)`). This variant applies the same idea to **multiplicative /
rotational** updates (`W ← exp(c·A_orth)·W`). Whether the gradient's relative magnitudes
are trustworthy may differ between weight-increments and rotation-rates — this is not a
known result, which is exactly why it's worth the head-to-head.

---

## 8. Summary

- A **separate optimizer**: no RMS norm; orthogonalize the direction so all rotation
  planes turn by the same angle `c`.
- Composes with the pipeline (momentum + interval-1 small-angle CNP intact) because
  orthogonalization acts on the *direction*, not as the exp map and not on the momentum
  buffer.
- Orthogonalization makes the RMS normalizer automatically 1, so `c` directly = the
  per-plane rotation angle. No `√d`, no `‖A‖_F`.
- Both orthogonalization routes stay skew — NS preserves skew on a skew input (every
  step is skew × commuting-symmetric, verified to ~1e-15). They differ in accuracy vs
  cost: **default** = Muon's quintic NS (~5 steps, lands in a band σ ≈ 1±0.3, so `c` is
  *nominal*); **exact** = symmetric / Löwdin `A(−A²)^(−1/2)` (σ = 1 exactly, `c` is the
  literal angle, ~4× the steps via inverse-sqrt NS on the symmetric `−A²`, then ×`A`).
- First-moment-only by default (second moment is partially undone by orthogonalization).
- Run head-to-head against the RMS-norm optimizer to test whether relative per-plane
  angles are signal or noise for rotational updates.
