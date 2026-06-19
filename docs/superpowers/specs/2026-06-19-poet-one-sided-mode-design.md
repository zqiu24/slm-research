# POET one-sided update mode (`in_only` / `out_only`) — design

Date: 2026-06-19

## Goal

Add a new POET training mode that is **the current best POET config, but only ever
updates one rotation side** — choosable between **`in_only`** (train only the
input-side rotation `oft_R_in`) and **`out_only`** (train only the output-side
rotation `oft_R_out`). Everything else is identical to the champion POET config.

This is a research/ablation mode: "what if we rotate only one side of every linear
for the whole run?" It is *not* a speed play (that is the regressed
`single_step_x_alternating` story); fidelity to the champion path matters more than
skipping the frozen side's compute.

## Baseline ("cham poet") being cloned

[`configs/experiments/optim/poet_lie_orth_alt.yaml`](../../../configs/experiments/optim/poet_lie_orth_alt.yaml)
— the integrated alternating POETX champion:

- `single_step_x: true` (plain `POETXLinear` forward-frame layer; feeds **both**
  rotation grads every forward/backward)
- `single_step_x_alternating: false` (NOT the dedicated true-single-side layer)
- `lie_alternating: true`, `lie_alternate_every: 1`
- `q_optimizer: lie_ortho` (Muon NS, `lie_ortho_c: 8`, 5 NS steps, distributed)
- `lr 3e-3`, `block_count 1`, `merge_period 1`, `reinit_period -1`, `scale 0.5`,
  `parameterization cayley`, `train_output_rotation: true`
- `base.model.unfuse_qkv: true`, `unfuse_fc1: true`

In this path the **active rotation side toggles** out/in by iteration, while both Lie
momenta advance every step and both grads are fed by the layer.

## Key insight: one source of truth for "active side"

The active side is a single shared signal —
[`third_party/poet_torch/alt_state.py`](../../../third_party/poet_torch/alt_state.py)
`active_side(alternate_every)` — read by exactly the places that matter for the
integrated path:

- the optimizer's **write** side:
  [`poet_lie_orth.py:99-109`](../../../src/optim/poet_lie_orth.py) `_active_side()`,
  and the buffer write guard `if active is not None and group["side"] != active: continue`
  at [`poet_lie_orth.py:171`](../../../src/optim/poet_lie_orth.py)
- the merge's **fold** side:
  [`poet_merge_step.py:606-612`](../../../src/patches/poet_merge_step.py)
  `pl._fold_active_side(active_side(pl.alternate_every), ...)`

Today `active_side()` toggles `"out"` on even iterations, `"in"` on odd. **The entire
feature is: pin it to a fixed side.** Because the optimizer write and the merge fold
both read this one function, pinning it makes the whole mode self-consistent with no
change to the optimizer or merge code.

When pinned, the frozen side's `oft_R` is never written (stays at its `0` init →
Cayley(0)=Identity) and never folded into the base weight. The result is that, for the
entire run, every linear is rotated on exactly one side. Both Lie momenta still advance
(the integrated `_lie_m_update` advances all skew params regardless of `active` —
[`poet_lie_orth.py:111-139`](../../../src/optim/poet_lie_orth.py)), and the layer still
feeds both grads — so the path is bit-identical to the champion except that the side
never toggles.

### Why not reuse `train_output_rotation` (`requires_grad` freeze)?

There is an existing single-sided ablation: `train_output_rotation: false` →
`--poet-freeze-output-rotation` sets `oft_R_out.requires_grad = False`, dropping it from
the optimizer's param-group split
([`poet_lie_momentum.py:_split_poet_lie_params`](../../../src/optim/poet_lie_momentum.py)).
Reusing this (plus a new `train_input_rotation`) composes **badly** with the alternating
integrated path: the optimizer's `_active_side()` and the merge's `_fold_active_side()`
would still alternate to a side that now has no grad/param, requiring `lie_alternating`
to be turned off and falling back to the non-alternating both-sides path. That is no
longer "everything else the same as cham poet." **Rejected** in favor of pinning
`alt_state`.

## Design (chosen)

### Config surface

One new POET key, plus two ready-to-run experiment YAMLs (repo's
one-YAML-per-variant convention).

- `optim.poet.lie_fixed_side: in | out` — default **unset/`null`** = current
  alternating behavior (no change to existing configs).
- New experiments cloning `poet_lie_orth_alt.yaml` with only the new key changed:
  - `configs/experiments/optim/poet_lie_orth_in_only.yaml` → `lie_fixed_side: in`
  - `configs/experiments/optim/poet_lie_orth_out_only.yaml` → `lie_fixed_side: out`

### Data flow

```
YAML poet.lie_fixed_side: in|out
  └─ megatron_args._optim_args(kind=poet): validate + emit  --poet-lie-fixed-side {in,out}
       └─ argparse (launchers/pretrain_gpt_slm.py): args.poet_lie_fixed_side
            └─ poet_apply_to_model._apply_poet_to_chunk: alt_state.set_fixed_side(args.poet_lie_fixed_side)
                 └─ alt_state.active_side()  ── returns the pinned side ──┐
                      ├─ LieOrthMomentum._active_side()  → writes only pinned side, both momenta advance
                      └─ poet_merge_step  → folds only pinned side; frozen side stays identity forever
```

The optimizer and merge are **unchanged** — they already route through `active_side()`.

### Components

1. **`alt_state.py`** — add a module-level fixed-side override:
   - `_FIXED_SIDE: str | None = None`
   - `set_fixed_side(side: str | None)` — validates `side in {None, "in", "out"}`.
   - `active_side(alternate_every=1)` — `return _FIXED_SIDE if _FIXED_SIDE is not None
     else <existing toggle>`.
   - Set once at apply time. Robust across checkpoint resume (re-derived from config,
     not from iteration parity).

2. **argparse** — [`launchers/pretrain_gpt_slm.py`](../../../launchers/pretrain_gpt_slm.py)
   near the other `--poet-lie-*` flags (~line 88):
   `group.add_argument("--poet-lie-fixed-side", choices=["in", "out"], default=None)`.

3. **`megatron_args.py`** [`_optim_args` poet branch](../../../src/utils/megatron_args.py):
   - Emit `--poet-lie-fixed-side <v>` only when set.
   - Validation (in the poet branch): if `lie_fixed_side` is set, require
     `lie_alternating=true`, `single_step_x=true`, `q_optimizer=lie_ortho`,
     `merge_period=1`, `parameterization=cayley`, and **mutually exclusive** with
     `single_step_x_alternating=true`. (These mirror the champion path's own
     requirements; fail fast otherwise.)

4. **`poet_apply_to_model.py`** [`_apply_poet_to_chunk`](../../../src/patches/poet_apply_to_model.py)
   near where `lie_alternating`/`alternate_every` are read (~line 71): read
   `getattr(args, "poet_lie_fixed_side", None)` and call
   `poet_torch.alt_state.set_fixed_side(...)`. (Idempotent across the per-chunk loop.)

5. **optimizer + merge** — **no change** (already read `alt_state.active_side()`).

### Two new experiment YAMLs

Byte-for-byte clones of `poet_lie_orth_alt.yaml` with:
- `experiment.name` → `poet_lie_orth_in_only` / `poet_lie_orth_out_only`
- updated `experiment.description` (one-sided, fixed side, ablation framing)
- add `lie_fixed_side: in` (resp. `out`) under `optim.poet`
- everything else (patches, lr, c, blocks, merge_period, momenta) identical.

## Testing (CPU only)

All verification here is CPU-runnable; GPU smoke is the user's to run.

- **`tests/unit`** new/extended cases:
  - `alt_state.set_fixed_side("in"|"out")` makes `active_side()` return the pinned
    side for several iterations (overrides the toggle); `set_fixed_side(None)` restores
    the toggle; invalid side raises.
  - `megatron_args`: `lie_fixed_side` emits `--poet-lie-fixed-side`; validation rejects
    it without `lie_alternating`/`single_step_x`/`lie_ortho`, and rejects it together
    with `single_step_x_alternating`.
  - the two new YAMLs parse and produce the expected argv (extend the existing
    config→argv test pattern).
- **`python -m py_compile`** / **`ruff`** on touched files.
- Reset `_FIXED_SIDE` between tests to avoid global-state bleed (fixture/teardown).

## GPU validation (hand off — user runs)

Provide exact commands to launch `poet_lie_orth_in_only` and `poet_lie_orth_out_only`.
Expected: training is stable; the frozen side's `oft_R` norm stays exactly 0; one-sided
val/loss is (as an ablation) somewhat worse than the both-sides champion
(val/loss ≈ 3.5332) — the comparison is the point of the mode.

## Out of scope (YAGNI)

- No speed optimization (skipping the frozen side's forward/backward/momentum) — that is
  the `single_step_x_alternating` path and is explicitly *not* this mode.
- No support for one-sided on the non-`lie_ortho` / non-POETX paths.
- No new merge or optimizer code paths.

## Touch-point summary

| # | File | Change |
|---|------|--------|
| 1 | `third_party/poet_torch/alt_state.py` | `_FIXED_SIDE` + `set_fixed_side()`; honor in `active_side()` |
| 2 | `launchers/pretrain_gpt_slm.py` | `--poet-lie-fixed-side {in,out}` argparse |
| 3 | `src/utils/megatron_args.py` | emit flag + validation in poet branch |
| 4 | `src/patches/poet_apply_to_model.py` | `alt_state.set_fixed_side(args.poet_lie_fixed_side)` |
| 5 | `configs/experiments/optim/poet_lie_orth_in_only.yaml` | new (clone + `lie_fixed_side: in`) |
| 6 | `configs/experiments/optim/poet_lie_orth_out_only.yaml` | new (clone + `lie_fixed_side: out`) |
| 7 | `tests/unit/*` | alt_state override + megatron_args validation/emit + YAML→argv |
| — | optimizer / merge | **no change** (already route through `active_side()`) |
