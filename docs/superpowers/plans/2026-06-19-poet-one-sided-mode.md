# POET pure one-sided update mode (`in_only` / `out_only`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dedicated one-sided POET layers (`InOnlyPOETXLinear` / `OutOnlyPOETXLinear`) and a `optim.poet.single_step_x_one_sided: in|out` mode that trains exactly one fixed rotation side for the whole run — the frozen side's forward, backward, momentum, and merge fold are all short-circuited.

**Architecture:** A new `OneSidedPOETXLinear(POETXLinear)` bakes the trained side into the layer (forward reuses `AlternatingPOETXSingleStepFunction` with a constant `active`). The optimizer (`true_single_side`) skips the frozen momentum and writes only the active side; the merge folds only the active side. Optimizer and merge learn the fixed side from the existing single source of truth, `alt_state.active_side()`, pinned at apply time. The cloned baseline is `poet_lie_orth_alt_x` (the true-single-side recipe), now regression-free because the side is fixed rather than alternating.

**Tech Stack:** Python, PyTorch, vendored Megatron POET stack (`third_party/poet_torch`), OmegaConf/Hydra, pytest.

## Global Constraints

- **Baseline cloned:** `configs/experiments/optim/poet_lie_orth_alt_x.yaml` — `single_step_x: true`, `single_step_fast: true`, `q_optimizer: lie_ortho`, `lie_ortho_c: 8`, `lie_ortho_method: muon`, `lie_ortho_ns_steps: 5`, `lie_ortho_distributed: true`, `lr: 3.0e-3`, `block_count: 1`, `merge_period: 1`, `reinit_period: -1`, `scale: 0.5`, `parameterization: cayley`, `train_output_rotation: true`, `base.model.unfuse_qkv/unfuse_fc1: true`. The new YAMLs set `single_step_x_alternating: false`, `lie_alternating: false`, and `single_step_x_one_sided: in|out`.
- **New flag is opt-in:** `optim.poet.single_step_x_one_sided` defaults to unset/`null`. No existing config changes behavior. `_FIXED_SIDE` defaults to `None`.
- **No change to the optimizer or merge ALGORITHMS** — only the optimizer's `true_single_side` gate is widened (one line) and the side is pinned via `alt_state`.
- **CPU-only verification here.** GPU runs are handed off (Task 8).
- **Every experiment YAML MUST have a matching `docs/experiments/<name>.md`** (pre-commit hook).
- **Every POET experiment YAML MUST set `lie_ortho_distributed: true`** (`test_poet_experiment_yamls_enable_lie_ortho_distributed`). The clone inherits this.
- **`poet_torch` is vendored** — imports are `from poet_torch import ...` with `third_party` on `PYTHONPATH` (repo conftest handles it; else `export PYTHONPATH=$PWD/third_party:$PWD`).
- **Commit style:** conventional commits, `feat(poet): ...`. No AI attribution trailer.
- **Test runner:** `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest`. For tests importing `launchers.submit` (`test_megatron_args.py`), if that import fails use the `/var/tmp/zqiu/slmcpu312` venv.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `third_party/poet_torch/poetx_layer.py` | one-sided layer classes | Modify: add `OneSidedPOETXLinear` + `InOnlyPOETXLinear`/`OutOnlyPOETXLinear` |
| `third_party/poet_torch/__init__.py` | package exports | Modify: export the three new classes |
| `third_party/poet_torch/alt_state.py` | shared active-side signal | Modify: `_FIXED_SIDE` + `set_fixed_side()` |
| `src/optim/poet_layers.py` | linear→POET replacement | Modify: `single_step_x_one_sided` kwarg + dispatch |
| `src/optim/poet.py` | POET optimizer builder | Modify: widen `true_single_side` gate |
| `launchers/pretrain_gpt_slm.py` | POET argparse | Modify: `--poet-single-step-x-one-sided {in,out}` |
| `src/utils/megatron_args.py` | config → CLI args | Modify: validation + emit |
| `src/patches/poet_apply_to_model.py` | apply POET at build | Modify: pass flag + `alt_state.set_fixed_side` |
| `configs/experiments/optim/poet_lie_orth_in_only.yaml` | in_only experiment | Create |
| `configs/experiments/optim/poet_lie_orth_out_only.yaml` | out_only experiment | Create |
| `docs/experiments/poet_lie_orth_in_only.md` + `..._out_only.md` | experiment docs (hook) | Create |
| `tests/unit/test_one_sided_poetx.py` | layer unit tests | Create |
| `tests/unit/test_alt_state.py`, `test_poet_layers.py`, `test_megatron_args.py`, `test_patch_poet_apply.py` | tests | Modify |

---

## Task 1: one-sided layer classes + exports

**Files:**
- Modify: `third_party/poet_torch/poetx_layer.py` (append after `AlternatingPOETXLinear`, ~`:213`)
- Modify: `third_party/poet_torch/__init__.py` (after `:27`)
- Test: `tests/unit/test_one_sided_poetx.py` (create)

**Interfaces:**
- Consumes: `POETXLinear`, `AlternatingPOETXSingleStepFunction` (existing).
- Produces:
  - `OneSidedPOETXLinear(POETXLinear)` — `__init__(*args, side: str, alternate_every=1, **kwargs)`; sets `self.side`, `self.alternating=True`; `forward(x)` differentiates only `self.side`.
  - `InOnlyPOETXLinear(OneSidedPOETXLinear)` (`side="in"`), `OutOnlyPOETXLinear` (`side="out"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_one_sided_poetx.py`:

```python
"""One-sided POETX layers: train exactly one fixed rotation side; the frozen
side's gradient is shape-correct zeros (never trained)."""

import pytest
import torch
from poet_torch import InOnlyPOETXLinear, OneSidedPOETXLinear, OutOnlyPOETXLinear


def _run_backward(pl):
    with torch.no_grad():
        pl.weight.normal_()
    x = torch.randn(4, pl.in_features, requires_grad=True)
    gy = torch.randn(4, pl.out_features)
    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    (pl(x) * gy).sum().backward()
    return pl


def test_in_only_trains_in_freezes_out():
    pl = _run_backward(InOnlyPOETXLinear(in_features=12, out_features=8, block_count=1))
    assert pl.side == "in"
    assert pl.alternating is True
    assert pl.oft_R_in.grad.abs().sum() > 0
    assert torch.count_nonzero(pl.oft_R_out.grad) == 0


def test_out_only_trains_out_freezes_in():
    pl = _run_backward(OutOnlyPOETXLinear(in_features=12, out_features=8, block_count=1))
    assert pl.side == "out"
    assert pl.oft_R_out.grad.abs().sum() > 0
    assert torch.count_nonzero(pl.oft_R_in.grad) == 0


def test_one_sided_rejects_bad_side():
    with pytest.raises(ValueError, match="side"):
        OneSidedPOETXLinear(in_features=12, out_features=8, block_count=1, side="left")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_one_sided_poetx.py -v`
Expected: FAIL — `ImportError: cannot import name 'InOnlyPOETXLinear' from 'poet_torch'`.

- [ ] **Step 3a: Add the classes**

Append to `third_party/poet_torch/poetx_layer.py` (after `AlternatingPOETXLinear`, ~`:213`):

```python
class OneSidedPOETXLinear(POETXLinear):
    """POETX layer that trains ONE FIXED rotation side for the whole run.

    side="in" trains only oft_R_in; side="out" only oft_R_out. The frozen side's
    oft_R stays at its 0 init (identity) -- its forward rotation, backward
    gradient, momentum, and merge fold are all short-circuited (the forward reuses
    AlternatingPOETXSingleStepFunction with a CONSTANT active side). Unlike
    AlternatingPOETXLinear the side never toggles, so the momentum-staleness
    regression does not apply: the trained side's momentum advances and applies
    every step.

    alternating=True routes the merge driver to the active-only fold; the active
    side is pinned globally via alt_state.set_fixed_side(side) at apply time, so the
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
    """OneSidedPOETXLinear pinned to the input side (trains only oft_R_in)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, side="in", **kwargs)


class OutOnlyPOETXLinear(OneSidedPOETXLinear):
    """OneSidedPOETXLinear pinned to the output side (trains only oft_R_out)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, side="out", **kwargs)
```

- [ ] **Step 3b: Export the classes**

In `third_party/poet_torch/__init__.py`, after the `AlternatingPOETXLinear` export (`:27`):

```python
from .poetx_layer import OneSidedPOETXLinear as OneSidedPOETXLinear
from .poetx_layer import InOnlyPOETXLinear as InOnlyPOETXLinear
from .poetx_layer import OutOnlyPOETXLinear as OutOnlyPOETXLinear
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_one_sided_poetx.py -v`
Expected: all three PASS.

- [ ] **Step 5: Lint + compile**

Run: `ruff check third_party/poet_torch/poetx_layer.py third_party/poet_torch/__init__.py tests/unit/test_one_sided_poetx.py && python -m py_compile third_party/poet_torch/poetx_layer.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/poetx_layer.py third_party/poet_torch/__init__.py tests/unit/test_one_sided_poetx.py
git commit -m "feat(poet): add one-sided POETX layer classes (in/out only)"
```

---

## Task 2: `alt_state` fixed-side override

**Files:**
- Modify: `third_party/poet_torch/alt_state.py`
- Test: `tests/unit/test_alt_state.py`

**Interfaces:**
- Produces: `alt_state.set_fixed_side(side: str | None)` (raises `ValueError` unless `None`/`"in"`/`"out"`); `active_side()` returns the pinned side when set, else the iteration toggle.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_alt_state.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def _reset_fixed_side():
    alt_state.set_fixed_side(None)
    yield
    alt_state.set_fixed_side(None)


def test_fixed_side_pins_active_side_regardless_of_iteration():
    alt_state.set_fixed_side("in")
    for it in (0, 1, 2, 3, 100):
        alt_state.set_iteration(it)
        assert alt_state.active_side(1) == "in"
    alt_state.set_fixed_side("out")
    for it in (0, 1, 2, 3, 100):
        alt_state.set_iteration(it)
        assert alt_state.active_side(1) == "out"


def test_clearing_fixed_side_restores_toggle():
    alt_state.set_fixed_side("in")
    alt_state.set_fixed_side(None)
    for it, expected in [(0, "out"), (1, "in"), (2, "out")]:
        alt_state.set_iteration(it)
        assert alt_state.active_side(1) == expected


def test_set_fixed_side_rejects_invalid():
    with pytest.raises(ValueError, match="fixed_side"):
        alt_state.set_fixed_side("left")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alt_state.py -v`
Expected: the three new tests FAIL (`AttributeError: ... has no attribute 'set_fixed_side'`); existing four PASS.

- [ ] **Step 3: Implement the override**

In `third_party/poet_torch/alt_state.py`, add next to `_ITERATION`:

```python
_ITERATION = 0
_FIXED_SIDE = None  # None = alternate by iteration; "in"/"out" = pin one side


def set_fixed_side(side) -> None:
    """Pin active_side() to one rotation side for the whole run (None = alternate).

    Set once at apply time from optim.poet.single_step_x_one_sided. Read by the
    optimizer write side (true_single_side) and the merge fold side via
    active_side(), so the one-sided POET mode stays self-consistent without
    touching optimizer/merge algorithms.
    """
    global _FIXED_SIDE
    if side not in (None, "in", "out"):
        raise ValueError(f"fixed_side must be None, 'in', or 'out', got {side!r}")
    _FIXED_SIDE = side
```

Change `active_side`:

```python
def active_side(alternate_every: int = 1) -> str:
    if _FIXED_SIDE is not None:
        return _FIXED_SIDE
    every = alternate_every if alternate_every and alternate_every > 0 else 1
    return "out" if (_ITERATION // every) % 2 == 0 else "in"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alt_state.py -v`
Expected: all PASS (4 + 3).

- [ ] **Step 5: Lint + compile**

Run: `ruff check third_party/poet_torch/alt_state.py tests/unit/test_alt_state.py && python -m py_compile third_party/poet_torch/alt_state.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/alt_state.py tests/unit/test_alt_state.py
git commit -m "feat(poet): add alt_state fixed-side override for one-sided POET"
```

---

## Task 3: dispatch in `replace_linears_with_poet`

**Files:**
- Modify: `src/optim/poet_layers.py` (signature `:291-315`; dispatch `:453-498`)
- Test: `tests/unit/test_poet_layers.py`

**Interfaces:**
- Consumes: `InOnlyPOETXLinear`/`OutOnlyPOETXLinear` (Task 1).
- Produces: `replace_linears_with_poet(..., single_step_x_one_sided: str | None = None)` builds `In/OutOnlyPOETXLinear` leaves when `single_step_x and single_step_x_one_sided is not None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_poet_layers.py` (mirrors `test_single_step_x_alternating_builds_alternating_poetx`: a named-child module, `init_type="none"`, inner layer at `.poet_linear`):

```python
def test_replace_with_one_sided_in_builds_in_only_layers():
    import torch.nn as nn
    from poet_torch import InOnlyPOETXLinear, OutOnlyPOETXLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)

    m = M()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_x=True,
        single_step_x_one_sided="in",
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, InOnlyPOETXLinear)
    assert not isinstance(pl, OutOnlyPOETXLinear)
    assert pl.side == "in"


def test_replace_with_one_sided_out_builds_out_only_layers():
    import torch.nn as nn
    from poet_torch import OutOnlyPOETXLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)

    m = M()
    replace_linears_with_poet(
        m,
        block_count=1,
        init_type="none",
        extra_linear_types=(nn.Linear,),
        single_step_x=True,
        single_step_x_one_sided="out",
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, OutOnlyPOETXLinear)
    assert pl.side == "out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -k one_sided -v`
Expected: FAIL — `TypeError: replace_linears_with_poet() got an unexpected keyword argument 'single_step_x_one_sided'`.

- [ ] **Step 3a: Add the keyword**

In `src/optim/poet_layers.py`, in the `replace_linears_with_poet` signature, after `single_step_x_alternating: bool = False,` (`:310`):

```python
    single_step_x_one_sided: str | None = None,
```

- [ ] **Step 3b: Add the dispatch branch**

In the `if cache_mode == "none":` block, BEFORE `if single_step_x and single_step_x_alternating:` (`:454`):

```python
                    if single_step_x and single_step_x_one_sided is not None:
                        from poet_torch import InOnlyPOETXLinear, OutOnlyPOETXLinear

                        _PoetCls = (
                            InOnlyPOETXLinear
                            if single_step_x_one_sided == "in"
                            else OutOnlyPOETXLinear
                        )
                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            alternate_every=alternate_every,
                            **block_kwargs,
                        )
                    elif single_step_x and single_step_x_alternating:
```

(Change the existing `if single_step_x and single_step_x_alternating:` on `:454` to `elif`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -k one_sided -v`
Expected: both PASS.

- [ ] **Step 5: Lint + compile + regression**

Run: `ruff check src/optim/poet_layers.py tests/unit/test_poet_layers.py && python -m py_compile src/optim/poet_layers.py && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -q`
Expected: no lint/compile errors; the full `test_poet_layers.py` suite PASSES (dispatch unchanged for existing modes).

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_layers.py tests/unit/test_poet_layers.py
git commit -m "feat(poet): dispatch one-sided POETX layers in replace_linears_with_poet"
```

---

## Task 4: config flag — argparse + emit + validation

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` (after `:126`)
- Modify: `src/utils/megatron_args.py` (validation ~`:473`, emit ~`:573`)
- Test: `tests/unit/test_megatron_args.py`

**Interfaces:**
- Produces: CLI `--poet-single-step-x-one-sided {in,out}` → `args.poet_single_step_x_one_sided`; `_optimizer_args` emits it iff `optim.poet.single_step_x_one_sided` is set, after validating companions.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_megatron_args.py`:

```python
def _one_sided_cfg(side):
    return _poet_cfg(
        {
            "block_count": 1,
            "merge_period": 1,
            "parameterization": "cayley",
            "q_optimizer": "lie_ortho",
            "single_step_fast": True,
            "single_step_x": True,
            "single_step_x_alternating": False,
            "lie_alternating": False,
            "train_output_rotation": True,
            "single_step_x_one_sided": side,
        }
    )


def test_one_sided_emits_flag_when_valid():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_one_sided_cfg("in"))
    assert "--poet-single-step-x-one-sided" in args
    assert args[args.index("--poet-single-step-x-one-sided") + 1] == "in"


def test_one_sided_omitted_when_unset():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {"block_count": 1, "merge_period": 1, "single_step_x": True, "q_optimizer": "lie_ortho"}
        )
    )
    assert "--poet-single-step-x-one-sided" not in args


def test_one_sided_requires_single_step_x():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    cfg = _one_sided_cfg("in")
    cfg.optim.poet.single_step_x = False
    with pytest.raises(ValueError, match="single_step_x_one_sided"):
        _optimizer_args(cfg)


def test_one_sided_mutually_exclusive_with_alternating():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    cfg = _one_sided_cfg("out")
    cfg.optim.poet.single_step_x_alternating = True
    with pytest.raises(ValueError, match="single_step_x_one_sided"):
        _optimizer_args(cfg)


def test_one_sided_rejects_bad_value():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="single_step_x_one_sided"):
        _optimizer_args(_one_sided_cfg("left"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k one_sided -v`
Expected: FAIL (no flag emitted; no validation raised). Use the `/var/tmp/zqiu/slmcpu312` venv if `launchers.submit` import fails.

- [ ] **Step 3a: Register the argparse flag**

In `launchers/pretrain_gpt_slm.py`, after `--poet-single-step-x-alternating` (`:126`):

```python
    # Pure one-sided POETX: train ONE fixed rotation side for the whole run.
    # "in" = InOnlyPOETXLinear (only oft_R_in), "out" = OutOnlyPOETXLinear (only
    # oft_R_out). Requires --poet-single-step-x; mutually exclusive with the
    # alternating layer. Default (absent) = off.
    group.add_argument("--poet-single-step-x-one-sided", choices=["in", "out"], default=None)
```

- [ ] **Step 3b: Add validation**

In `src/utils/megatron_args.py`, in the `if kind == "poet":` branch, after the `single_step_x_alternating` block (before `# block_count ...`, ~`:473`):

```python
        one_sided = poet.get("single_step_x_one_sided", None)
        if one_sided is not None:
            if one_sided not in ("in", "out"):
                raise ValueError(
                    f"optim.poet.single_step_x_one_sided must be 'in' or 'out', got {one_sided!r}."
                )
            if not poet.get("single_step_x", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided requires single_step_x=true "
                    "(the one-sided layer is a POETX subclass)."
                )
            if merge_period != 1:
                raise ValueError("optim.poet.single_step_x_one_sided requires merge_period=1.")
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError(
                    "optim.poet.single_step_x_one_sided requires parameterization=cayley."
                )
            if poet.get("q_optimizer", "adam") != "lie_ortho":
                raise ValueError(
                    "optim.poet.single_step_x_one_sided requires q_optimizer=lie_ortho."
                )
            if poet.get("head_aligned_attn", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided is incompatible with head_aligned_attn=true."
                )
            if poet.get("single_step_x_alternating", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided is mutually exclusive with "
                    "single_step_x_alternating."
                )
            if poet.get("lie_alternating", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided is mutually exclusive with lie_alternating."
                )
            if poet.get("group_experts", False):
                raise ValueError(
                    "optim.poet.single_step_x_one_sided does not support group_experts=true."
                )
```

- [ ] **Step 3c: Emit the flag**

In `src/utils/megatron_args.py`, after the `--poet-single-step-x-alternating` emit (~`:573`):

```python
        # Pure one-sided POETX layer (train one fixed rotation side for the run).
        if poet.get("single_step_x_one_sided", None) is not None:
            poet_args.append("--poet-single-step-x-one-sided")
            poet_args.append(str(poet.get("single_step_x_one_sided")))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k one_sided -v`
Expected: all five PASS.

- [ ] **Step 5: Lint + compile**

Run: `ruff check src/utils/megatron_args.py launchers/pretrain_gpt_slm.py tests/unit/test_megatron_args.py && python -m py_compile src/utils/megatron_args.py launchers/pretrain_gpt_slm.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/utils/megatron_args.py launchers/pretrain_gpt_slm.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): add --poet-single-step-x-one-sided flag with validation"
```

---

## Task 5: apply plumbing + optimizer `true_single_side`

**Files:**
- Modify: `src/patches/poet_apply_to_model.py` (`_apply_poet_to_chunk`, `:57-98`)
- Modify: `src/optim/poet.py` (`true_single_side=...`, `:626`)
- Test: `tests/unit/test_patch_poet_apply.py`

**Interfaces:**
- Consumes: `alt_state.set_fixed_side` (Task 2); `replace_linears_with_poet(single_step_x_one_sided=...)` (Task 3); `args.poet_single_step_x_one_sided` (Task 4).
- Produces: after `_apply_poet_to_chunk(m, args)`, the side is forwarded to `replace_linears_with_poet` AND `alt_state.active_side()` returns the pinned side. The optimizer's `true_single_side` is on whenever the one-sided flag is set.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_patch_poet_apply.py`:

```python
def test_apply_to_chunk_forwards_one_sided_and_pins_alt_state(monkeypatch):
    from types import SimpleNamespace

    import src.patches.poet_apply_to_model as ap
    from poet_torch import alt_state

    captured = {}

    def _fake_replace(m, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr(ap, "replace_linears_with_poet", _fake_replace)
    alt_state.set_fixed_side(None)

    args = SimpleNamespace(
        poet_block_size=256,
        poet_block_count=1,
        poet_single_step_x=True,
        poet_single_step_x_one_sided="in",
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=None,
    )
    try:
        ap._apply_poet_to_chunk(object(), args)
        assert captured["single_step_x_one_sided"] == "in"
        alt_state.set_iteration(0)  # would be "out" under the toggle
        assert alt_state.active_side(1) == "in"
    finally:
        alt_state.set_fixed_side(None)


def test_apply_to_chunk_leaves_alt_state_unpinned_when_unset(monkeypatch):
    from types import SimpleNamespace

    import src.patches.poet_apply_to_model as ap
    from poet_torch import alt_state

    monkeypatch.setattr(ap, "replace_linears_with_poet", lambda m, **kw: 0)
    alt_state.set_fixed_side(None)

    args = SimpleNamespace(
        poet_block_size=256, poet_block_count=1, hidden_size=64,
        num_attention_heads=4, kv_channels=None,
    )
    ap._apply_poet_to_chunk(object(), args)
    alt_state.set_iteration(0)
    assert alt_state.active_side(1) == "out"  # toggle intact
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_apply.py -k one_sided -v`
Expected: `test_apply_to_chunk_forwards_one_sided_and_pins_alt_state` FAILS (`KeyError: 'single_step_x_one_sided'`).

- [ ] **Step 3a: Plumb the apply patch**

In `src/patches/poet_apply_to_model.py`, in `_apply_poet_to_chunk`, after `single_step_x_alternating = getattr(...)` (`:70`):

```python
    single_step_x_one_sided = getattr(args, "poet_single_step_x_one_sided", None)
```

After `alternate_every = getattr(...)` (`:72`), pin the shared signal:

```python
    # One-sided POET: pin the shared active-side signal so the optimizer write side
    # (true_single_side) and the merge fold side both target the fixed side.
    from poet_torch import alt_state

    alt_state.set_fixed_side(single_step_x_one_sided)
```

Add the kwarg to the `replace_linears_with_poet(...)` call (after `single_step_x_alternating=single_step_x_alternating,`, `:94`):

```python
        single_step_x_one_sided=single_step_x_one_sided,
```

- [ ] **Step 3b: Widen the optimizer `true_single_side` gate**

In `src/optim/poet.py`, change the `LieOrthMomentum(...)` arg (`:626`):

```python
            true_single_side=(
                getattr(config, "poet_single_step_x_alternating", False)
                or getattr(config, "poet_single_step_x_one_sided", None) is not None
            ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_apply.py -k one_sided -v`
Expected: both PASS.

- [ ] **Step 5: Lint + compile**

Run: `ruff check src/patches/poet_apply_to_model.py src/optim/poet.py tests/unit/test_patch_poet_apply.py && python -m py_compile src/patches/poet_apply_to_model.py src/optim/poet.py`
Expected: no errors.
(The `poet.py` `true_single_side` one-liner runs only inside the Megatron optimizer builder — not CPU-unit-testable; it is verified by `py_compile` + code review here and exercised in the GPU smoke, Task 8.)

- [ ] **Step 6: Commit**

```bash
git add src/patches/poet_apply_to_model.py src/optim/poet.py tests/unit/test_patch_poet_apply.py
git commit -m "feat(poet): wire one-sided mode through apply + optimizer true_single_side"
```

---

## Task 6: experiment YAMLs + docs + YAML→argv smoke test

**Files:**
- Create: `configs/experiments/optim/poet_lie_orth_in_only.yaml`, `..._out_only.yaml`
- Create: `docs/experiments/poet_lie_orth_in_only.md`, `..._out_only.md`
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing smoke test**

Add to `tests/unit/test_megatron_args.py`:

```python
def test_in_only_yaml_emits_one_sided_in():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=600m",
            "experiment=optim/poet_lie_orth_in_only",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--poet-single-step-x-one-sided"] == "in"
    assert m["--poet-single-step-x"] is True
    assert "--poet-single-step-x-alternating" not in m


def test_out_only_yaml_emits_one_sided_out():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=600m",
            "experiment=optim/poet_lie_orth_out_only",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    m = _args_to_map(build_megatron_args(cfg))
    assert m["--poet-single-step-x-one-sided"] == "out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "in_only or out_only" -v`
Expected: FAIL — Hydra cannot find the experiment configs.

- [ ] **Step 3a: Create `poet_lie_orth_in_only.yaml`**

`configs/experiments/optim/poet_lie_orth_in_only.yaml`:

```yaml
# @package _global_
# poet_lie_orth_in_only: pure one-sided POETX on the champion lie_ortho recipe
# (head-OFF, lr 3e-3, c=8, distributed). InOnlyPOETXLinear trains ONLY the input-side
# rotation oft_R_in for the whole run; oft_R_out stays at its zero init (identity).
# The frozen side's forward, backward, momentum, and merge fold are short-circuited.
# Same recipe as poet_lie_orth_alt_x but the side is FIXED (single_step_x_one_sided)
# instead of alternating -- so the alternating momentum-staleness regression does not
# apply. See docs/superpowers/specs/2026-06-19-poet-one-sided-mode-design.md
experiment:
  name: poet_lie_orth_in_only
  family: optim
  description: |
    Pure one-sided POET (input side): InOnlyPOETXLinear trains only oft_R_in;
    oft_R_out stays identity. Built on the champion lie_ortho recipe with the side
    fixed (not alternating). Ablation vs the both-sides champion (val/loss ≈3.5332)
    and the alternating poet_lie_orth_alt_x.
  references:
    - "POET"
    - "Muon"
    - "Pion"
  patches:
    - model_unfuse_linears
    - poet_optimizer_setup
    - poet_unfuse_te_impl
    - poet_apply_to_model
    - poet_merge_step
    - training_log_eta
    - wandb_metric_normalize
  required_capabilities: []

optim:
  type: poet
  lr: 3.0e-3
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
  poet:
    block_count: 1
    cache_mode: none
    init_type: normalized
    mup_alpha: 1.0
    merge_period: 1
    reinit_period: -1
    scale: 0.5
    use_poet_adam: false
    parameterization: cayley
    q_optimizer: lie_ortho
    lie_b1: 0.9
    lie_b2: 0.95
    lie_eps: 1.0e-8
    lie_v_mode: elementwise
    lie_ortho_c: 8
    lie_ortho_method: muon
    lie_ortho_ns_steps: 5
    lie_ortho_use_second_moment: false
    lie_ortho_distributed: true
    head_aligned_attn: false
    single_step_fast: true
    single_step_x: true
    single_step_x_alternating: false
    lie_alternating: false
    lie_alternate_every: 1
    train_output_rotation: true
    single_step_x_one_sided: in

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 3b: Create `poet_lie_orth_out_only.yaml`**

Identical to Step 3a except header/`name`/`description` say "output side" / `oft_R_out` / `oft_R_in`, and the last poet key is `single_step_x_one_sided: out`. Full file:

```yaml
# @package _global_
# poet_lie_orth_out_only: pure one-sided POETX on the champion lie_ortho recipe
# (head-OFF, lr 3e-3, c=8, distributed). OutOnlyPOETXLinear trains ONLY the output-side
# rotation oft_R_out for the whole run; oft_R_in stays at its zero init (identity).
# The frozen side's forward, backward, momentum, and merge fold are short-circuited.
# Same recipe as poet_lie_orth_alt_x but the side is FIXED (single_step_x_one_sided)
# instead of alternating -- so the alternating momentum-staleness regression does not
# apply. See docs/superpowers/specs/2026-06-19-poet-one-sided-mode-design.md
experiment:
  name: poet_lie_orth_out_only
  family: optim
  description: |
    Pure one-sided POET (output side): OutOnlyPOETXLinear trains only oft_R_out;
    oft_R_in stays identity. Built on the champion lie_ortho recipe with the side
    fixed (not alternating). Ablation vs the both-sides champion (val/loss ≈3.5332)
    and the alternating poet_lie_orth_alt_x.
  references:
    - "POET"
    - "Muon"
    - "Pion"
  patches:
    - model_unfuse_linears
    - poet_optimizer_setup
    - poet_unfuse_te_impl
    - poet_apply_to_model
    - poet_merge_step
    - training_log_eta
    - wandb_metric_normalize
  required_capabilities: []

optim:
  type: poet
  lr: 3.0e-3
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
  poet:
    block_count: 1
    cache_mode: none
    init_type: normalized
    mup_alpha: 1.0
    merge_period: 1
    reinit_period: -1
    scale: 0.5
    use_poet_adam: false
    parameterization: cayley
    q_optimizer: lie_ortho
    lie_b1: 0.9
    lie_b2: 0.95
    lie_eps: 1.0e-8
    lie_v_mode: elementwise
    lie_ortho_c: 8
    lie_ortho_method: muon
    lie_ortho_ns_steps: 5
    lie_ortho_use_second_moment: false
    lie_ortho_distributed: true
    head_aligned_attn: false
    single_step_fast: true
    single_step_x: true
    single_step_x_alternating: false
    lie_alternating: false
    lie_alternate_every: 1
    train_output_rotation: true
    single_step_x_one_sided: out

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 3c: Create the matching experiment docs (pre-commit hook)**

`docs/experiments/poet_lie_orth_in_only.md`:

```markdown
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
```

`docs/experiments/poet_lie_orth_out_only.md`: same with "input"→"output", `oft_R_in`/`oft_R_out` swapped, `OutOnlyPOETXLinear`, and `single_step_x_one_sided: out`.

- [ ] **Step 4: Run test + YAML-walk guard**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py \
  -k "in_only or out_only or lie_ortho_distributed" -v
```
Expected: the two new tests PASS and `test_poet_experiment_yamls_enable_lie_ortho_distributed` PASSES (clones set `lie_ortho_distributed: true`).

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/optim/poet_lie_orth_in_only.yaml \
        configs/experiments/optim/poet_lie_orth_out_only.yaml \
        docs/experiments/poet_lie_orth_in_only.md \
        docs/experiments/poet_lie_orth_out_only.md \
        tests/unit/test_megatron_args.py
git commit -m "feat(poet): add poet_lie_orth_in_only/out_only one-sided experiments"
```

---

## Task 7: full CPU verification

**Files:** none (verification only).

- [ ] **Step 1: Run all touched test modules**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_one_sided_poetx.py \
  tests/unit/test_alt_state.py \
  tests/unit/test_poet_layers.py \
  tests/unit/test_megatron_args.py \
  tests/unit/test_patch_poet_apply.py -v
```
Expected: all PASS. (If `test_megatron_args.py` fails to import `launchers.submit`, rerun that file with `/var/tmp/zqiu/slmcpu312`. Repo has 2 known pre-existing `launchers.submit` failures unrelated to this change — confirm any failures are those two.)

- [ ] **Step 2: POET regression set (no behavior change for existing modes)**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poetx_layer.py \
  tests/unit/test_alternating_poetx.py \
  tests/unit/test_poet_lie_orth.py \
  tests/unit/test_poet_merge_step.py -q
```
Expected: all PASS (`_FIXED_SIDE` defaults to `None`; dispatch unchanged for existing modes; `true_single_side` widening is inert when the flag is unset).

- [ ] **Step 3: Pre-commit on the full change set**

Run: `pre-commit run --files $(git diff --name-only HEAD~6 HEAD)`
Expected: all hooks pass, including "Every experiment YAML has a matching docs/experiments/<name>.md".

- [ ] **Step 4: Update CHANGELOG**

Append a one-line entry under the current date to `NeckariumAI/zqiu/CHANGELOG.md`: "Added pure one-sided POET mode (`optim.poet.single_step_x_one_sided: in|out`) with `InOnlyPOETXLinear`/`OutOnlyPOETXLinear` + `poet_lie_orth_in_only`/`poet_lie_orth_out_only` experiments." Commit:

```bash
git add NeckariumAI/zqiu/CHANGELOG.md
git commit -m "docs(changelog): pure one-sided POET mode"
```

---

## Task 8: GPU validation handoff (user runs)

**Files:** none. Provide commands; do NOT run GPU jobs.

- [ ] **Step 1: Launch commands**

```bash
# in_only
codexlog poet_in_only python launchers/submit.py \
  base/family=llama3 base/scale=600m \
  experiment=optim/poet_lie_orth_in_only \
  training_regime=ablation_20x cluster=h800_cn

# out_only
codexlog poet_out_only python launchers/submit.py \
  base/family=llama3 base/scale=600m \
  experiment=optim/poet_lie_orth_out_only \
  training_regime=ablation_20x cluster=h800_cn
```
(Match `base/scale`, `training_regime`, `cluster` to the champion sweep being compared against.)

- [ ] **Step 2: What to check**

- Stable training (no NaN/OOM); step time ≈ `poet_lie_orth_alt_x` (single-side POETX speed).
- The frozen side's `oft_R` norm stays exactly 0 the whole run (`oft_R_out` for `in_only`, `oft_R_in` for `out_only`) — via the existing POET diag / `_DUMP_POET_PARAMS` logging.
- Log line `[POET] Lie-orth: ...` present and the optimizer is in `true_single_side` mode (one skew side written per step).
- One-sided val/loss is the ablation signal vs the both-sides champion (≈3.5332) and `poet_lie_orth_alt_x`.

---

## Self-Review

**Spec coverage:**
- one-sided layer classes + exports → Task 1 ✓
- alt_state pin → Task 2 ✓
- dispatch → Task 3 ✓
- config flag (argparse + emit + validation) → Task 4 ✓
- apply plumbing + optimizer `true_single_side` → Task 5 ✓
- two YAMLs + matching docs (hook) → Task 6 ✓
- CPU tests (layer, alt_state, dispatch, megatron_args, apply) → Tasks 1-6 + Task 7 ✓
- GPU handoff → Task 8 ✓
- out-of-scope (grouped experts, non-lie_ortho, full alt_state decouple) → not added ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The only "same with swap" is the 6-line `poet_lie_orth_out_only.md` (its YAML twin is given in full at 6.3b; the swap rule is explicit). ✓

**Type consistency:**
- `OneSidedPOETXLinear(side=...)` / `In/OutOnlyPOETXLinear` consistent across Tasks 1, 3, tests ✓
- `single_step_x_one_sided` consistent as: YAML key ↔ `--poet-single-step-x-one-sided` ↔ `args.poet_single_step_x_one_sided` ↔ `replace_linears_with_poet(single_step_x_one_sided=...)` ↔ `set_fixed_side` value, across Tasks 3-6 ✓
- `set_fixed_side(side)` / `active_side()` consistent across Tasks 2, 5, tests ✓
- inner-layer accessor `POETMegatronLinear.poet_linear` matches existing `test_poet_layers.py` tests ✓
