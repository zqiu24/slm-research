# Cross-side decorrelation on the update-RMS POET champion

**Date:** 2026-06-26
**Status:** design (approved sections 1–3; finalized for review)

## 1. Summary

Combine the two strongest independent POET findings into one optimizer:

1. **The champion ("lieorthrmsmomentum")** — `q_optimizer=lie_ortho_update_rms`
   (`LieOrthUpdateRMSMomentum`, [src/optim/poet_lie_orth_update_rms.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth_update_rms.py)).
   Self-scaled angle `θ = min(lr·ρ / RMS(W), max∠)`, alternating + Muon-orthogonalized
   Lie direction + Nesterov b1.95. Current best POET (POET_dev.md §2.6/§2.12).
2. **The "split" ("with a scale")** — cross-side decorrelation (`decorrelate_sides`),
   which projects each layer's active in/out generator off the other side's
   weight-space direction so `cos(D_out, D_in) → 0`, scaled by a partial coefficient
   `decorrelate_lambda` (λ). Implemented today **only** in the *other* optimizer
   `LieOrthMomentum` ([src/optim/poet_lie_orth.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py)
   `_decorrelate_buf_alternating`). POET_dev.md §J.3 found **λ=0.5 + symmetric + renorm
   = −0.0070** over its (older, default-init non-Nesterov) baseline; **λ=1.0 catastrophic**.

The two are separate classes; the champion has **no** decorrelation. This work ports the
**alternating, partial-λ** decorrelation into the champion class, behind the **existing**
`poet_lie_ortho_decorrelate*` config keys (all already plumbed), and verifies the
champion path stays bit-identical when the flag is off.

### Target

Stack decorrelation on the **symmetric** update-RMS baseline
`init mup_normalized / mup_alpha 4 / ρ0.30 / side_γ=0 / lr5 / max∠0.024` = **3.4758**
(§2.11, clean attribution — `side_γ` asymmetry deliberately excluded so the measured
delta is decorrelation alone). If the §J.3 −0.0070 carries over → ≈**3.4688**, a new POET
record (current champion 3.4745; nGPT 3.4583; muon_kimi 3.4514).

## 2. Scope

**In scope:** optimizer port, config wiring for the `lie_ortho_update_rms` branch, CPU
tests, a Stage-1 sweep script, and exact handoff commands.

**Out of scope (handed to the operator):** all GPU/training runs. Per repo policy, no GPU
run is launched from this work — the sweep script + commands are the deliverable.

**Explicitly not ported:** the *simultaneous* decorrelation path (`_decorrelate_buf`).
`LieOrthUpdateRMSMomentum` raises unless `alternating=True`, so the simultaneous path is
unreachable and would be dead code.

## 3. Optimizer change — `LieOrthUpdateRMSMomentum`

File: [src/optim/poet_lie_orth_update_rms.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth_update_rms.py)

### 3.1 Constructor

Add 6 args mirroring `LieOrthMomentum.__init__`, with identical validation
(`decorrelate_mode ∈ {in_off_out, out_off_in, symmetric}`):

```
decorrelate_sides: bool = False
decorrelate_mode: str = "in_off_out"
decorrelate_lambda: float = 1.0
decorrelate_renorm: bool = False
decorrelate_cos_threshold: float = 0.0
layer_pairs = None        # -> self._decorr_pairs = list(layer_pairs) if layer_pairs else []
```

Store each as `self.<name>`; store `self._decorr_pairs`. These attribute names **must**
match `LieOrthMomentum` exactly, because the bf16 master-param remap in
[poet.py:835](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L835) keys on
`optimizer.decorrelate_sides` and `optimizer._decorr_pairs` generically — matching names
means that remap works for the champion class with **zero** changes there.

### 3.2 `_decorrelate_buf_alternating`

Port the method from
[poet_lie_orth.py:304](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L304)
**verbatim except one line**: this class's `slices` are 3-tuples `(off, n, p)`
(see [poet_lie_orth_update_rms.py:200](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth_update_rms.py#L200)),
whereas `LieOrthMomentum`'s are 4-tuples `(off, n, p, lr)`. So:

```python
# LieOrthMomentum (source):
off_by_id = {id(p): (off, n) for off, n, p, _lr in slices}
# LieOrthUpdateRMSMomentum (ported):
off_by_id = {id(p): (off, n) for off, n, p in slices}
```

Everything else (the `_decorr_pairs` loop, `side_directions`, `block_diag_skew`,
`orthogonalize_skew_direction(-m_inact)`, the `decorrelate_lambda`/`renorm`/`cos_threshold`
logic, the 0-pairs-matched warning guard) is copied unchanged. Imports already present in
this module: `vec_to_skew`, `skew_to_vec`, `orthogonalize_skew_direction`; add
`from src.diag.poet_coordination_diag import block_diag_skew, side_directions` inside the
method (same lazy-import style as the source).

### 3.3 `step()` ordering

In [step()](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth_update_rms.py#L301),
insert the decorrelation call **after** the all-reduce, **before**
`_apply_skew_update_buffer` — identical placement to
[poet_lie_orth.py:414-421](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L414):

```python
if self.distributed and self._dp_world_size > 1 and buf.numel() > 0:
    dist.all_reduce(buf, group=self.dp_group)
if self.decorrelate_sides:                          # NEW
    self._decorrelate_buf_alternating(buf, slices, active)
self._apply_skew_update_buffer(buf, slices)
```

(No `if self.alternating` branch — this class is always alternating.)

### 3.4 The one semantic subtlety (why the port is still correct)

In `LieOrthMomentum`, `buf` holds the **bare** orthogonalized direction; lr/angle are
applied at scatter (`alpha=lr`). In `LieOrthUpdateRMSMomentum`, `buf` holds the
**angle-scaled** generator `θ·x_orth` (θ already baked in; scatter uses `alpha=1.0`).
Decorrelation is unaffected because:

- The projection coefficient `c = ⟨A_act, g⟩ / ⟨g, g⟩` is **scale-invariant in `g`**
  (the inactive direction's magnitude cancels). `g` is derived from the inactive side's
  direction only, so the fact that the inactive direction is sourced un-angle-scaled from
  `orthogonalize(−lie_m)` does not matter.
- `decorrelate_renorm` preserves the active side's **realized** `‖D‖` — which here
  legitimately includes θ (the post-clamp rotation magnitude). It stays a direction-only
  change, exactly as in the source.
- The inactive side's `lie_m` is maintained **every step for both sides** by
  [`_lie_m_update`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth_update_rms.py#L148)
  (it explicitly `del active` and updates all sides), so the inactive direction is always
  available under alternating — same precondition the source relies on.

This subtlety is asserted by a test (§5: renorm preserves realized `‖D‖`).

## 4. Wiring — `poet.py` `lie_ortho_update_rms` branch

File: [src/optim/poet.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py),
branch at [line 761](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L761).
`model_chunks` is in scope (enclosing fn `get_megatron_poet_lie_momentum_optimizer`).

1. Read the flag near the top of the branch:
   `_lie_ortho_decorrelate = bool(getattr(config, "poet_lie_ortho_decorrelate", False))`.
2. Pass the 6 kwargs to the `LieOrthUpdateRMSMomentum(...)` call, identical to the
   `lie_ortho` branch ([poet.py:751-758](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L751)):
   ```python
   decorrelate_sides=_lie_ortho_decorrelate,
   decorrelate_mode=getattr(config, "poet_lie_ortho_decorrelate_mode", "in_off_out"),
   decorrelate_lambda=getattr(config, "poet_lie_ortho_decorrelate_lambda", 1.0),
   decorrelate_renorm=getattr(config, "poet_lie_ortho_decorrelate_renorm", False),
   decorrelate_cos_threshold=getattr(config, "poet_lie_ortho_decorrelate_cos_threshold", 0.0),
   layer_pairs=_build_decorrelate_pairs(model_chunks) if _lie_ortho_decorrelate else None,
   ```
3. **Decorrelation banner:** the `if _lie_ortho_decorrelate:` warning block currently lives
   inside the `lie_ortho` branch ([poet.py:720-733](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L720)).
   Factor it into a tiny module-level helper `_log_decorrelate_banner(config, logger)` and
   call it from **both** branches. (The sweep script's tripwire greps this exact banner —
   it must print for update-RMS runs too, else a dropped override goes unnoticed.)

**No change needed** to: `poet_optimizer_setup.py` (all 5 keys already copied,
[lines 77-90](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py#L77)),
`megatron_args.py` (CLI already emits them unconditionally,
[lines 660-673](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L660)), or
the bf16 master-param remap ([poet.py:835](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L835), generic).

## 5. Tests

New file `tests/unit/test_poet_lie_orth_update_rms_decorrelate.py`, mirroring the
alternating-decorrelate tests in
[tests/unit/test_poet_lie_orth.py:495-577](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_poet_lie_orth.py#L495)
but instantiating `LieOrthUpdateRMSMomentum` (with a square `W` so `bsz_out==bsz_in`, a
single `(out,in)` pair, `update_rms=0.3`, `max_angle=0.024`):

- `test_decorrelate_off_is_bit_identical` — `decorrelate_sides=False` reproduces the plain
  champion buffer exactly (the bit-identical guard).
- `test_alternating_decorrelate_is_not_a_noop` — on enables, the active write changes.
- `test_alternating_decorrelate_removes_inactive_momentum_overlap` — `cos` to the inactive
  momentum direction drops toward 0.
- `test_alternating_decorrelate_lambda_scales_overlap` — overlap removed is monotone in λ
  (the **"with a scale"** invariant: λ=0 → none, λ=1 → full).
- `test_renorm_preserves_realized_norm` — with `renorm=True`, the active side's realized
  `‖D‖` (θ-inclusive) is unchanged vs. decorrelate-off (the §3.4 subtlety).
- `test_rejects_bad_mode` — bad `decorrelate_mode` raises (validation parity).

Run on CPU: `python -m pytest tests/unit/test_poet_lie_orth_update_rms_decorrelate.py -q`.
Also re-run the existing `test_poet_lie_orth_update_rms.py` and `test_poet_lie_orth.py` to
confirm no regression, plus `python -m py_compile` on the three edited files.

## 6. Sweep script + handoff

New `scripts/sweep_update_rms_decorrelate.sh`, mirroring
[scripts/sweep_alt_decorrelate.sh](/lustre/fast/fast/zqiu/slm-research/scripts/sweep_alt_decorrelate.sh)
but on the symmetric update-RMS baseline. Held recipe (= the 3.4758 baseline):

```
scripts/train_poet_lie_orth_update_rms.sh llama3 \
  scheduler=cosine_poet training_regime=ablation_40x \
  optim.lr=0.005 \
  optim.poet.lie_ortho_update_rms=0.30 \
  optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.0 \
  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.scale=1.0 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true optim.poet.lie_ortho_rms_mode=weight
```

Stage-1 grid (baked-in, sequential — split across GPUs to parallelize):

```
optim.poet.lie_ortho_decorrelate=true
optim.poet.lie_ortho_decorrelate_mode=symmetric
optim.poet.lie_ortho_decorrelate_renorm=true
optim.poet.lie_ortho_decorrelate_cos_threshold=0.0
optim.poet.lie_ortho_decorrelate_lambda ∈ {0.25, 0.50, 0.75}
```

Each run wrapped in `codexlog <name>` so output is teed for readback. Diagnostics:
`SLM_POET_COORD_DIAG=1`, `SLM_POET_COORD_DIAG_INTERVAL=250` to log the `cos(D_out,D_in)`
trajectory. **Tripwire:** each startup must print the
`[POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=symmetric, lambda=<L>, renorm=True, …)`
banner; if λ/renorm/mode don't match the arm, an override was dropped — kill and fix.

Compare each arm against the symmetric baseline **3.4758**. Stage 2 (gated on a Stage-1
λ-pick, not built here): finer λ, `mode ∈ {in_off_out, out_off_in}`, and the
`cos_threshold=0.3` module gate; and, separately, re-stacking the winning λ on the
`side_γ=+0.25` champion (3.4745) to check decorrelation × asymmetry interaction.

## 7. Risks / open items

- **Margin vs. seed noise.** §J.3's −0.0070 is near the 60m/9k seed floor (~0.01–0.02), and
  it was measured on a *different* (default-init, non-Nesterov) baseline. The carryover to
  the update-RMS/Nesterov/init-scaled regime is a genuine empirical question — a 2–3 seed
  confirm is the eventual bar for a record claim. Single-seed Stage 1 gives the yes/no.
- **λ=1.0 excluded by design** (catastrophic in §J.3 via the renorm pathology); the grid
  stays in the safe interior (0.25–0.75).
- **Banner refactor** touches the `lie_ortho` branch (shared helper). Low risk, but the
  existing `lie_ortho` decorrelate path should be re-smoke-tested (CPU test already covers
  instantiation; the banner is log-only).
```
