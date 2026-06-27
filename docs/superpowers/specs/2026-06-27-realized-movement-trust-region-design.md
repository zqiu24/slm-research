# POET realized-movement trust region (M1) — design

**Date:** 2026-06-27
**Status:** design, pending review
**Optimizer target:** `LieOrthUpdateRMSMomentum` (`q_optimizer=lie_ortho_update_rms`) — the current champion family
**Relationship to decorrelation:** independent, composable mechanism (runs with `decorrelate_sides` on *or* off)

---

## 1. Motivation — angle is not movement

POET trains two orthogonal rotations on a frozen weight, `W_eff = R_out · W₀ · R_in`. Each
step writes a small skew generator `A` (block-diagonal in the weight's block-contiguous
frame). The current best recipe controls the generator's **angle** via the update-RMS law

```
θ = lr · ρ / RMS(W)            (ρ = 0.30, clamped at max_angle)
```

so every layer gets a near-constant per-step angle (modulo slow `W` drift).

But the quantity that actually matters — how much the weight *moves* — is the realized
first-order step

```
D_act = blockdiag(A) · W       (out side)      D_act = W · blockdiag(A)   (in side)
```

and for a **fixed** angle `‖A‖`, `‖D_act‖` varies a lot with *which* subspace `A` rotates:

- `A` rotates where `W` has large singular values → `‖D_act‖` large (a tiny angle moves the weight a lot)
- `A` rotates a low-energy subspace → `‖D_act‖` small (the same angle barely moves it)

So controlling the angle does **not** control realized movement. update-RMS normalizes by
`RMS(W)` — a single static per-layer scalar — so it fixes the layer-*average* movement but
lets the per-step, per-direction movement float free.

**M1 closes that loop:** fix the realized movement and let the angle float — *fixed
movement, adaptive angle.*

### Why this is over-spend control (the cross-side link)

The cross-side shared direction (`cos(D_out, D_in) ≈ 0.44`, ANALYSIS §17.6) is shared
*because* both sides find it productive to push — and "the direction the gradient most
wants" generically coincides with a **high-gain** direction (small rotations there move `W`
a lot). Under angle-control, each unit of angle on that direction yields outsized movement,
so the two sides collectively over-move it (double-charge × high-gain amplification) — the
over-spend that decorrelation targets by *redirection*.

M1 attacks the same thing by *magnitude*: the moment a step's move lands heavily on a
high-gain (shared) direction, `‖D_act‖` spikes and M1 throttles the angle to keep movement
on budget — taxing exactly the over-spent direction, with **no cross-term ever computed**.
M1 doesn't know which part of `D` is shared vs private; it only knows over-spend shows up as
excess realized movement, and caps it.

### Unifying view — one principle, three knobs

| knob | controls | granularity |
|---|---|---|
| update-RMS angle | `θ ∝ 1/RMS(W)` | layer-average, static |
| renorm-off shrink (§2.15c) | reduce movement on the over-spent direction | per-step, only when decorr on |
| **M1** | `θ ∝ 1/‖A_t(W)‖` (cap) | **per-step, per-direction, dynamic** |

M1 is the dynamic generalization that subsumes the intuition behind both prior knobs:
*control realized movement, not generator angle.*

---

## 2. Mechanism

Per active skew param, per step, define the **fractional realized movement**

```
r = ‖D_act‖_F / ‖W‖_F
```

(`‖W‖_F` is permutation-invariant, so it is read straight from `group["weight"]`; the
numerator is computed in the same block-contiguous frame.) Then apply a one-sided **trust
region** to the generator before it is written:

```
if r > ρ_move:                       # over-spend this step
    f = 1 − λ_move · (1 − ρ_move / r) # partial shrink; λ_move=1 → exact clip to budget
    A ← f · A                         # D is linear in A, so ‖D_act‖ ← f · ‖D_act‖
```

Because `D` is linear in `A`, the scalar `f` computed in weight-space applies directly to the
generator (`buf` slice) — no re-projection. Three modes:

| `move_control_mode` | behavior |
|---|---|
| `off` (default) | no-op; champion path **bit-identical** |
| `clip` | intervene only when `r > ρ_move` (one-sided; the recommended primary) |
| `normalize` | always rescale to `r = ρ_move` (two-sided; aggressive, for ablation) |

`λ_move ∈ [0, 1]` (default 1.0) is a partial-strength knob mirroring the decorrelation `λ`
finding ("full is often too much"): it scales how much of the needed shrink is applied.

**Why clip is the primary, not normalize.** Full `‖D‖`-normalization is a *second* whitening
on top of Muon's per-step Newton–Schulz, flattening the realized-movement spectrum and
risking loss of the gradient's signal about which directions matter (the same "full is too
much" seen at λ=1 decorrelation). Clip only throttles the over-spend tail and leaves normal
steps exactly as update-RMS set them.

### Placement in `step()`

`LieOrthUpdateRMSMomentum.step()` ([poet_lie_orth_update_rms.py:417](../../../src/optim/poet_lie_orth_update_rms.py)):

```
active = self._active_side()
self._lie_m_update(active)
buf, slices = self._skew_update_buffer(...)        # θ-scaled generators
if distributed: all_reduce(buf)
if self.decorrelate_sides: _decorrelate_buf_alternating(buf, slices, active)
+ if self.move_control_mode != "off": _movement_trust_region(buf, slices, active)   # NEW
self._apply_skew_update_buffer(buf, slices)
```

M1 runs **after** decorrelation (so it enforces the realized-movement budget on the *final*
written generator) and is otherwise independent of it. It iterates the active skew params,
reads `W = group["weight"]` (already block-contiguous) and `bsz = group["block_size"]`,
forms `D_act` via the active branch of `side_directions` (one `bmm` for out, one `einsum`
for in), and rescales the `buf` slice. Cost ≈ one matmul the size of `W` per active param —
comparable to the existing orthogonalization, no extra backward, no basis state.

### Interactions

- **decorrelation** — orthogonal; `{decorr off/on} × {M1 off/clip}` is the headline ablation.
- **`side_gamma`** — M1 measures `D` *after* the per-side angle multiplier, so it naturally
  respects side asymmetry; no special handling.
- **`gain`** (learnable scale) — `D` uses the frozen owner `W`; for consistency with
  update-RMS (which multiplies its denom by `|gain|`, [poet_lie_orth_update_rms.py:271](../../../src/optim/poet_lie_orth_update_rms.py)) M1 folds `|gain|` into `‖W‖_F` when present.
- **distributed** — M1 acts on the already-`all_reduce`d `buf`; `W` is replicated, so `f` is
  identical on every rank → no extra sync.

---

## 3. Calibration (Phase 0, diagnostic-only)

`ρ_move` must sit near the current realized-movement distribution so `clip` touches only the
over-spend tail. Before any intervention, run the champion with M1 in a **measure-only** mode
that logs `r = ‖D_act‖_F/‖W‖_F` per side (mean / p50 / p90 / p95 / max), reusing the existing
`last_update_rms_stats` wandb hook ([poet_lie_orth_update_rms.py:307](../../../src/optim/poet_lie_orth_update_rms.py)). Set the
Phase-1 `ρ_move` grid around the measured **p50–p90** so the trust region bites the upper
tail, not the median step.

---

## 4. Experimental design

Headline 2×2 (does magnitude-control *substitute for* or *complement* decorr's redirection?):

| | M1 off | M1 clip @ ρ_move |
|---|---|---|
| **decorr off** | 3.4745 (update-RMS + side_γ, §2.12) | ? |
| **decorr λ0.25** | 3.4686 (record, §2.15c) | ? |

- **Seeds 42 / 43 / 44** against the matched no-M1 base — the −0.003-scale effects we are
  chasing are under the ~0.01 single-seed noise floor, so the seed triple is mandatory, not optional.
- `ρ_move` grid from Phase 0 (≈ 3 values around p50–p90); `clip` primary, one `normalize`
  point as an over-whitening probe; `λ_move=1.0` first, `0.5` if full clips too hard.
- One single-arm script per node, matching the existing `sweep_*` / `train_poet_dev*` layout;
  wrap run commands in `codexlog <name> …`.

---

## 5. Config & plumbing

Three new `optim.poet.*` flags (all gated behind `mode=off` default → champion unchanged):

| flag | type / default | meaning |
|---|---|---|
| `poet_lie_move_control_mode` | `off` \| `clip` \| `normalize` (default `off`) | trust-region mode |
| `poet_lie_move_budget_rho` | float (default 0.0 = unset; **required** when `mode != off`, validated at construction) | the cap `ρ_move` |
| `poet_lie_move_lambda` | float in [0,1] (default 1.0) | partial-shrink strength |

Standard 4-edit POET flag path (megatron_args → config → `poet_optimizer_setup` →
optimizer kwargs). **Gotcha:** the args→config copy in `poet_optimizer_setup.py` is the
silent no-op one — verify the value reaches the optimizer (a past A/B lost many turns to
exactly this). A measure-only banner / one logged `r` stat confirms the flag is live.

---

## 6. Testing (CPU, runnable in-session)

- **clip math** — `r > ρ` ⇒ generator scaled by exactly `f`; `r ≤ ρ` ⇒ untouched; `normalize`
  ⇒ post-`r == ρ`; `λ_move` partial scales correctly.
- **`D_act` correctness** — M1's active-side `D` matches `side_directions`' active branch on
  random skew + weight; and `r` is permutation-invariant (sanity, since we use the contiguous frame directly).
- **off-path identity** — `mode=off` ⇒ `buf` bit-identical to the no-M1 path (champion untouched).
- **distributed replication** — same `buf` in/out across simulated ranks (W replicated ⇒ `f` matches).

---

## 7. Risks & open questions

- **Over-whitening** — double whitening with Muon could flatten useful structure. Mitigated
  by `clip` (one-sided) + `λ_move<1`; `normalize` arm quantifies the downside.
- **Redundancy with update-RMS `ρ`** — M1 might just re-discover a better static angle. The
  2×2 ablation + the per-step `r` variance (Phase 0) distinguish "dynamic per-direction
  control" from "a different constant angle."
- **`ρ_move` sensitivity** — handled by the Phase-0 measurement rather than a guess.
- **Norm choice** — fractional Frobenius `‖D‖_F/‖W‖_F` chosen for scale-invariance and
  consistency with update-RMS; an RMS-ratio variant is a trivial later swap if needed.

---

## 8. Out of scope (YAGNI)

- M2 (trajectory-EMA decorrelation) and M3 (joint-move whitening) — considered, deferred.
- Porting M1 to `LieOrthMomentum` (`poet_lie_orth.py`) — only if a non-update-RMS champion
  needs it; the primary target is the update-RMS family.
- The 3a/3b decorrelation sub-knobs (inactive-source, decoupled-renorm) — a separate track,
  not bundled here.
