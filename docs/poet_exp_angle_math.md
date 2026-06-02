# The Rotation-Angle Scaling for POET, Derived with the Matrix Exponential

**Purpose.** Define the correct step-size invariant for optimizing POET's skew-symmetric generator `Q`, and show that the "Muon principle" translates *exactly* onto the rotation manifold when the orthogonal block is parameterized by the true matrix exponential `G = exp(Q)`. Using `exp` (rather than the Cayley–Neumann parameterization) removes every correction term — the factor-of-2, the `arctan`, the convergence ceiling, the orthogonality-approximation error — and makes the analogy literal.

---

## 1. The parameterization and the space `Q` lives in

We parameterize each orthogonal block as

```
G = exp(Q),    Q skew-symmetric (Qᵀ = −Q),    Q ∈ ℝ^{b×b}.
```

The set of skew-symmetric matrices is the **Lie algebra** `so(b)` — exactly the tangent space to the rotation group `SO(b)` at the identity. `exp` maps this tangent space onto the group: for any skew-symmetric `Q`, `exp(Q)` is exactly orthogonal, since

```
exp(Q)ᵀ exp(Q) = exp(Qᵀ) exp(Q) = exp(−Q) exp(Q) = exp(0) = I.
```

No approximation, for any `Q`. This is the first payoff of `exp`: **orthogonality is exact and unconstrained** (no `‖Q‖ < 1` requirement).

---

## 2. Canonical form: what `Q` "is" geometrically

Every real skew-symmetric `Q` can be block-diagonalized by an orthogonal change of basis `U`:

```
Q = U Σ_θ Uᵀ,    Σ_θ = blockdiag( A(θ₁), A(θ₂), …, A(θ_{⌊b/2⌋}) ),    A(θ) = [[0, −θ], [θ, 0]]
```

(plus a single zero on the diagonal if `b` is odd). The `θ_k ≥ 0` carry the structural content of `Q`. Two equivalent readings:

- The **eigenvalues** of `Q` are `{±iθ_k}` — purely imaginary, in conjugate pairs.
- The **singular values** of `Q` are `{θ_k}`, each appearing twice.

So `Q` is intrinsically a list of `⌊b/2⌋` rotation rates `θ_k`, each acting in its own 2-plane (spanned by the corresponding pair of columns of `U`).

---

## 3. Exponentiating: the `θ_k` are *literally* the rotation angles

Because `exp` commutes with orthogonal conjugation,

```
exp(Q) = U exp(Σ_θ) Uᵀ = U blockdiag( R(θ₁), …, R(θ_{⌊b/2⌋}) ) Uᵀ,
```

and exponentiating one 2×2 block is the textbook 2D rotation:

```
exp( [[0, −θ], [θ, 0]] ) = [[cos θ, −sin θ], [sin θ, cos θ]] = R(θ).
```

This is the clean fact Cayley does **not** give: with `exp`, the canonical angle of the rotation in plane `k` is **exactly `θ_k`** — the singular value of `Q`, with no transformation. (Cayley would give `2·arctan(θ_k)` here; that entire nonlinearity vanishes.)

> **Key identity: the singular values of `Q` ARE the rotation angles of `G = exp(Q)`.**

Everything downstream is simple because of this one identity.

---

## 4. Norms of `Q` in terms of the angles

Every scalar summary of `Q` now translates directly into a statement about the angle spectrum `{θ_k}`.

**Frobenius norm.** `Q` is normal (`QQᵀ = QᵀQ`), so `‖Q‖_F² = Σ |eigenvalue|²`. With eigenvalues `±iθ_k`:

```
‖Q‖_F² = Σ_k (θ_k² + θ_k²) = 2 Σ_k θ_k²    ⟹    ‖Q‖_F = √2 · ‖θ‖₂.
```

So `‖Q‖_F` is, up to `√2`, the **L2 norm of the angle vector** — the root-sum-square total rotation. It is computable directly from the stored parameters with **no eigendecomposition**, yet has an exact geometric meaning.

**Spectral norm.**

```
‖Q‖₂ = max_k θ_k = θ_max,
```

the largest single-plane rotation angle. With `exp` there is **no `θ_max < 1` ceiling** — `exp` converges for any angle, so unlike Cayley this is purely informational, not a stability constraint. (Geometrically, `θ_max > π` only means a plane has rotated past a half-turn, where the angle representation wraps around — a mild, well-understood non-uniqueness, not a divergence.)

---

## 5. From a step on `Q` to the realized rotation — exactly

Suppose the optimizer proposes a step `ΔQ` (skew-symmetric, so it stays in the tangent space `so(b)`). How much does the *rotation* change? On a Lie group, to first order in the step,

```
exp(Q + ΔQ) ≈ exp(Q) · exp( J⁻¹(ΔQ) ),
```

where `J` is the left-trivialized differential of `exp` (the `dexp` operator). The crucial simplification: **evaluate at `Q = 0`** — which is exactly the state right after each merge-then-reinitialize reset — where `J = I`, giving

```
exp(0 + ΔQ) = exp(ΔQ)    (exactly).
```

So at the start of each inner loop (the regime POET actually operates in, since it resets `Q → 0` every `Tm` steps), the step `ΔQ` maps to a rotation whose canonical angles are *exactly* the singular values of `ΔQ`. The generator-to-rotation map is **identity-on-angles** at the reset point, and stays near-identity while `Q` remains small.

No factor-of-2, no `arctan` — those were Cayley artifacts. The quantity you control (singular values of the step) **is** the quantity that happens (rotation angles), with no correction, precisely in the regime POET runs in.

---

## 6. The Muon principle, stated *exactly* on the manifold

**Muon on an ordinary weight**, abstractly: take the gradient `g`, **orthogonalize it** (Newton–Schulz drives all singular values to ≈ 1, flattening the spectrum so no direction dominates), then **rescale** to a target magnitude (RMS). Two operations: flatten the update's singular spectrum, then set its size.

On `Q`, the update `ΔQ = ∂f/∂Q` is itself skew-symmetric, so it has its own angle spectrum `{δθ_k}` = its singular values. Translate Muon's two operations literally:

**(a) Flatten the spectrum.** Replace `ΔQ`'s singular values `{δθ_k}` with a uniform value — i.e. orthogonalize `ΔQ`. Take its polar factor:

```
ΔQ = V Σ_δ Wᵀ  (SVD)    →    orthogonalize:  ΔQ̂ = V Wᵀ,
```

then re-skew-symmetrize `ΔQ̂ ← (ΔQ̂ − ΔQ̂ᵀ)/2` to remain in `so(b)`. (For a skew-symmetric input the orthogonalization already nearly preserves skew-symmetry; the projection is a safety step.) The result rotates **all planes by the same angle** — the rotation-manifold meaning of "no direction dominates." Because singular values of `Q` = rotation angles *exactly* under `exp`, "flatten the singular spectrum of the step" and "rotate every plane equally" are the **same statement**, not analogues.

**(b) Set the size.** Rescale so the **aggregate rotation angle per step** hits a target `θ_step`:

```
ΔQ_final = θ_step · ΔQ̂ / ‖ΔQ̂‖_*  ,
```

with the normalization chosen so the angle-vector of `ΔQ_final` has the target L2 norm. After flattening, all `δθ_k` are equal, so this simply sets that common per-plane angle (scaled by `√(#planes)` if targeting the aggregate).

**Resulting update rule:**

```
Q ← Q − θ_step · skew( orthogonalize( momentum( ∂f/∂Q ) ) )
```

`θ_step` is the single tunable, with a clean physical meaning: **the rotation angle taken per step.** This is the exact analogue of Muon's RMS target — but it is an *angle*, the intrinsic invariant of the manifold, rather than an RMS, which is a flat-space coordinate norm with no meaning on the curved space `so(b)`.

---

## 7. The correspondence, now exact

| Muon (flat weight space) | POET-with-`exp` (rotation manifold `so(b)`) | exact under `exp`? |
|---|---|---|
| object updated: weight `W` | object updated: generator `Q ∈ so(b)` | — |
| update's singular values | step's singular values = **rotation angles `θ_k`** | exact (§3) |
| orthogonalize → flat singular spectrum | flatten `{θ_k}` → equal rotation in every plane | exact |
| rescale to target RMS (per-weight size) | rescale to target **aggregate angle `θ_step`** | exact at `Q = 0` (§5) |
| RMS is the right invariant (flat space) | **angle** is the right invariant (curved space) | — |

The exp assumption makes the math "easy" in §3 and §5 specifically: **singular-value-of-`Q` = rotation-angle, with no `arctan` and no first-order factor**, and the generator→rotation map is exactly identity-on-angles at the reset point where POET lives. Every place Cayley forced a correction (`2·arctan(θ)`, the `γ = 0.5`, the `‖Q‖₂ < 1` ceiling, the Neumann truncation error) becomes trivial or absent.

---

## 8. The one remaining subtlety (NOT fixed by `exp`)

`exp` cleans the *generator → rotation* hop completely. It does **not** clean the second hop — *rotation → effect on the weight* — because `W ← G W (…)` depends on `W` itself. The same rotation angle `θ_step` produces a relative weight change `‖ΔW‖/‖W‖` that depends on the block's energy.

- Muon needs only **one** invariant (RMS) because it has no separate generator and weight — they are the same object.
- POET has **two** objects (generator `Q`, weight `W`) and therefore potentially **two** magnitude notions: the rotation angle, and the induced `‖ΔW‖/‖W‖`. They coincide only if block weight-norms are uniform across blocks.

POET's own **spectrum-preservation** keeps each block's weight spectrum bounded and similar, which *may* keep `‖W‖` uniform enough that the two invariants collapse into one — as in Muon. This is an empirical question and it ties the angle-scaling design to the spectrum question. It is **orthogonal to the exp-vs-Cayley choice**: `exp` simplifies the first hop to exactness but leaves this second-hop question exactly where it was.

---

## Load-bearing identities (summary)

1. `Q` skew-symmetric ⟹ eigenvalues `±iθ_k`, singular values `{θ_k}`.
2. `exp(Q)` rotates canonical plane `k` by **exactly** `θ_k` (no `arctan`, no factor-of-2).
3. `‖Q‖_F = √2 · ‖θ‖₂` (aggregate angle; cheap, no eigendecomposition).
4. `‖Q‖₂ = θ_max` (largest plane angle; informational only — `exp` has no convergence ceiling).
5. `exp(0 + ΔQ) = exp(ΔQ)` exactly ⟹ the reset-point step is correction-free; the step's singular values are the realized rotation angles.

These four-plus-one are what make the Muon analogy *literal* rather than approximate, and they are precisely the statements that Cayley/CNP would burden with correction terms.

---

## Implementation note (separate from the derivation)

The derivation above is cleanest in the `exp` frame and should be reasoned about there. Implementation then has three options, a separate decision:

- **Exact `exp`:** clean and exact, but the *backward* pass (Fréchet derivative of the matrix exponential) is the expensive, numerically delicate part — typically via the augmented `2b×2b` block-matrix trick (`≈ 8× O(b³)` per block) or an eigendecomposition with a Daleckiĭ–Kreĭn divided-difference formula. At small block sizes (`b = 256/512`) the 8× factor on one sub-op may be tolerable if the step is matmul-bound; worth benchmarking before assuming it's prohibitive.
- **Cayley–Neumann (current POET):** cheap polynomial matmuls (kernel-friendly), at the cost of the `arctan` angle nonlinearity, the `‖Q‖₂ < 1` ceiling, the reset coupling, and the orthogonality approximation error.
- **Truncated-`exp` polynomial:** e.g. `G ≈ I + Q + Q²/2 + Q³/6` — a low-degree polynomial (cheap matmuls like CNP) that approximates `exp` rather than Cayley, keeping the clean angle relationship (`φ_k ≈ θ_k`, no factor-of-2), at the cost of only-approximate orthogonality (which CNP already tolerates) and a finite accuracy radius (which the reset already handles). Possibly the best of both worlds; the orthogonality error at matched degree/cost vs. CNP is an empirical question.
