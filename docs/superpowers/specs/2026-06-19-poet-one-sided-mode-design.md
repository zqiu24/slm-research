# POET pure one-sided update mode (`in_only` / `out_only`) — design

Date: 2026-06-19

## Goal

Add a POET mode that trains **exactly one rotation side for the whole run** —
choosable between **`in_only`** (train only the input-side rotation `oft_R_in`) and
**`out_only`** (train only the output-side rotation `oft_R_out`). It is a *pure*
single-sided update: the frozen side's forward rotation, backward gradient, optimizer
momentum, and merge fold are all short-circuited. Everything else matches the champion
single-side POET recipe.

This is a research/ablation mode: "what if we rotate only one side of every linear,
fixed, for the whole run?"

## Background: the existing pieces this builds on

`poet_lie_orth_alt_x` (config
[`poet_lie_orth_alt_x.yaml`](../../../configs/experiments/optim/poet_lie_orth_alt_x.yaml))
already does a *per-step* pure single-side update:

- [`AlternatingPOETXLinear`](../../../third_party/poet_torch/poetx_layer.py) forward
  computes **only the active side's** rotation; its backward
  (`AlternatingPOETXSingleStepFunction`) **zeros the frozen side's gradient**.
- the `lie_ortho` optimizer with `true_single_side=True` **skips the frozen side's
  momentum** ([`poet_lie_orth.py:117`](../../../src/optim/poet_lie_orth.py)) and writes
  only the active side ([`poet_lie_orth.py:171`](../../../src/optim/poet_lie_orth.py)).
- the merge folds **only the active side**
  ([`poet_merge_step.py:608-612`](../../../src/patches/poet_merge_step.py)).

All three read the active side from one source of truth,
[`alt_state.active_side()`](../../../third_party/poet_torch/alt_state.py), which today
*toggles* `out`/`in` by iteration.

`AlternatingPOETXLinear`'s docstring warns it "regressed quality" — but that was caused
by **alternating** (a side's momentum goes stale while it is idle, and the other side
moves `W` in between), **not** by single-sidedness. **When the side is fixed, that
regression cannot occur**: the trained side's momentum advances and applies every step
(a clean Muon-on-one-side), and the frozen side never moves `W`.

## Design (chosen): dedicated one-sided layer classes + pinned side

### Why dedicated classes (not just pinning the alternating layer)

The active side has to be agreed in three places — layer forward, optimizer write, merge
fold. We make the **layer** self-documenting by baking the side into a dedicated class
(its forward differentiates a fixed side, never reads the iteration toggle). For the
**optimizer and merge** we reuse the existing single-source-of-truth: pin
`alt_state.active_side()` to the fixed side (we deliberately do *not* refactor the merge
to read a per-layer attribute or the optimizer to read side from param groups — that is a
larger change for no behavioral gain). Side comes from one config flag, so the layer's
baked side and the `alt_state` pin are consistent by construction.

### New layer classes — `third_party/poet_torch/poetx_layer.py`

```python
class OneSidedPOETXLinear(POETXLinear):
    """POETX layer that trains ONE FIXED rotation side for the whole run.

    side="in" trains only oft_R_in; side="out" only oft_R_out. The frozen side's
    oft_R stays at its 0 init (identity) -- its forward rotation, backward gradient,
    momentum, and merge fold are all short-circuited. Unlike AlternatingPOETXLinear
    the side never toggles, so its momentum-staleness regression does not apply.

    alternating=True routes the merge driver to the active-only fold; the active
    side is pinned globally via alt_state.set_fixed_side(side) at apply time so the
    optimizer (true_single_side) and merge agree with this layer's fixed forward.
    """

    def __init__(self, *args, side: str, alternate_every: int = 1, **kwargs):
        if side not in ("in", "out"):
            raise ValueError(f"side must be 'in' or 'out', got {side!r}")
        super().__init__(*args, alternating=True, alternate_every=alternate_every, **kwargs)
        self.side = side

    def forward(self, x):
        return AlternatingPOETXSingleStepFunction.apply(
            x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
            self.perm_in_inv, self.perm_out_inv,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out, self.side,
        )


class InOnlyPOETXLinear(OneSidedPOETXLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, side="in", **kwargs)


class OutOnlyPOETXLinear(OneSidedPOETXLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, side="out", **kwargs)
```

Exported from [`third_party/poet_torch/__init__.py`](../../../third_party/poet_torch/__init__.py)
next to `AlternatingPOETXLinear`.

### Config surface

One new POET key carries both "enable" and "side":

- `optim.poet.single_step_x_one_sided: in | out` — default **unset/`null`** = off.
- Two experiment YAMLs cloning `poet_lie_orth_alt_x.yaml` with
  `single_step_x_alternating: false` and `single_step_x_one_sided: in` (resp. `out`):
  - `configs/experiments/optim/poet_lie_orth_in_only.yaml`
  - `configs/experiments/optim/poet_lie_orth_out_only.yaml`

### Data flow

```
YAML poet.single_step_x_one_sided: in|out
  └─ megatron_args._optimizer_args(poet): validate + emit --poet-single-step-x-one-sided {in,out}
       └─ argparse (pretrain_gpt_slm.py): args.poet_single_step_x_one_sided
            └─ poet_apply_to_model._apply_poet_to_chunk:
                 ├─ replace_linears_with_poet(single_step_x_one_sided=side)  → In/OutOnlyPOETXLinear
                 └─ alt_state.set_fixed_side(side)
            └─ poet.py get_megatron_poet_optimizer:
                 true_single_side = single_step_x_alternating OR (single_step_x_one_sided is set)
  alt_state.active_side() == side  ──▶ optimizer write+momentum (true_single_side) & merge fold
```

### Dispatch — `src/optim/poet_layers.py` `replace_linears_with_poet`

New keyword `single_step_x_one_sided: str | None = None`; a branch under `if single_step_x:`
(so the forward-frame weight conversion at `:517` still runs), before the
`single_step_x_alternating` branch:

```python
if single_step_x and single_step_x_one_sided is not None:
    from poet_torch import InOnlyPOETXLinear, OutOnlyPOETXLinear

    _PoetCls = InOnlyPOETXLinear if single_step_x_one_sided == "in" else OutOnlyPOETXLinear
    pl = _PoetCls(
        in_features=in_f, out_features=out_f, bias=has_bias,
        device=child.weight.device, dtype=child.weight.dtype,
        parameterization=parameterization, alternate_every=alternate_every, **block_kwargs,
    )
elif single_step_x and single_step_x_alternating:
    ...  # unchanged
```

### Optimizer — `src/optim/poet.py`

`true_single_side` must be on for the one-sided mode so the frozen momentum is skipped:

```python
true_single_side=(
    getattr(config, "poet_single_step_x_alternating", False)
    or getattr(config, "poet_single_step_x_one_sided", None) is not None
),
```

`alternating` stays `False` (the YAML sets `lie_alternating: false`). The merge already
routes one-sided layers to the active-only fold because `OneSidedPOETXLinear.alternating
== True`.

### Validation — `src/utils/megatron_args.py`

When `single_step_x_one_sided` is set: value in `{in, out}`; require `single_step_x=true`,
`merge_period=1`, `parameterization=cayley`, `q_optimizer=lie_ortho`, `head_aligned_attn=false`;
mutually exclusive with `single_step_x_alternating=true` and `lie_alternating=true`;
incompatible with `group_experts=true` (grouped one-sided not implemented). Emit
`--poet-single-step-x-one-sided <v>` only when set.

## Testing (CPU only)

- **`tests/unit/test_poetx_layer.py`** (or a new `test_one_sided_poetx.py`): construct
  `InOnlyPOETXLinear`/`OutOnlyPOETXLinear`; after a backward, the frozen side's
  `oft_R.grad` is all-zeros and the active side's is non-zero; `pl.side` and
  `pl.alternating is True`; `side` validation raises.
- **`tests/unit/test_poet_layers.py`**: `replace_linears_with_poet(single_step_x=True,
  single_step_x_one_sided="in", extra_linear_types=(nn.Linear,))` yields
  `InOnlyPOETXLinear` leaves (and `"out"` yields `OutOnlyPOETXLinear`).
- **`tests/unit/test_alt_state.py`**: `set_fixed_side` pins `active_side()`; `None`
  restores the toggle; invalid raises (with teardown to avoid global bleed).
- **`tests/unit/test_megatron_args.py`**: emit + validation for
  `single_step_x_one_sided`; the two new YAMLs parse and emit the flag.
- `python -m py_compile` / `ruff` on touched files.

## GPU validation (hand off — user runs)

Provide launch commands for `poet_lie_orth_in_only` / `poet_lie_orth_out_only`. Expected:
stable training at single-side POETX speed; the frozen side's `oft_R` norm stays exactly
0; one-sided val/loss is the ablation signal (compared to the both-sides champion,
val/loss ≈ 3.5332, and to the alternating `poet_lie_orth_alt_x`).

## Out of scope (YAGNI)

- Grouped-expert (`group_experts`) one-sided layers.
- One-sided on non-`lie_ortho` / non-POETX paths.
- Refactoring the merge/optimizer to read the side off the layer/param groups (the
  "full decouple from alt_state" option) — we keep the `alt_state` pin.

## Touch-point summary

| # | File | Change |
|---|------|--------|
| 1 | `third_party/poet_torch/poetx_layer.py` | `OneSidedPOETXLinear` + `InOnlyPOETXLinear`/`OutOnlyPOETXLinear` |
| 2 | `third_party/poet_torch/__init__.py` | export the new classes |
| 3 | `third_party/poet_torch/alt_state.py` | `_FIXED_SIDE` + `set_fixed_side()`; honor in `active_side()` |
| 4 | `src/optim/poet_layers.py` | `single_step_x_one_sided` kwarg + dispatch branch |
| 5 | `src/optim/poet.py` | `true_single_side` also on for one-sided |
| 6 | `launchers/pretrain_gpt_slm.py` | `--poet-single-step-x-one-sided {in,out}` argparse |
| 7 | `src/utils/megatron_args.py` | emit flag + validation |
| 8 | `src/patches/poet_apply_to_model.py` | pass `single_step_x_one_sided`; `alt_state.set_fixed_side(side)` |
| 9 | `configs/experiments/optim/poet_lie_orth_in_only.yaml` + `..._out_only.yaml` | new (clone alt_x) |
| 10 | `docs/experiments/poet_lie_orth_in_only.md` + `..._out_only.md` | new (pre-commit hook) |
| 11 | `tests/unit/*` | layer + dispatch + alt_state + megatron_args |
