# POET one-sided update mode (`in_only` / `out_only`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a POET mode that runs the champion `poet_lie_orth_alt` recipe but pins the rotation to one fixed side for the whole run — `in_only` (train only `oft_R_in`) or `out_only` (train only `oft_R_out`).

**Architecture:** The active rotation side is a single shared signal, `poet_torch.alt_state.active_side()`, read by both the optimizer's write side and the merge's fold side. The feature adds a module-level fixed-side override to `alt_state`, plumbed from a new `optim.poet.lie_fixed_side: in|out` config flag. The optimizer and merge are unchanged — pinning the one signal makes the whole mode self-consistent. The frozen side's `oft_R` stays at its `0` init (identity) and is never folded.

**Tech Stack:** Python, PyTorch, vendored Megatron POET stack (`third_party/poet_torch`), OmegaConf/Hydra configs, pytest.

## Global Constraints

- **Baseline cloned verbatim:** `configs/experiments/optim/poet_lie_orth_alt.yaml` — `single_step_x: true`, `single_step_fast: true`, `single_step_x_alternating: false`, `lie_alternating: true`, `lie_alternate_every: 1`, `q_optimizer: lie_ortho`, `lie_ortho_c: 8`, `lie_ortho_method: muon`, `lie_ortho_ns_steps: 5`, `lie_ortho_distributed: true`, `lr: 3.0e-3`, `block_count: 1`, `merge_period: 1`, `reinit_period: -1`, `scale: 0.5`, `parameterization: cayley`, `train_output_rotation: true`, `base.model.unfuse_qkv/unfuse_fc1: true`.
- **New flag is opt-in:** `optim.poet.lie_fixed_side` defaults to unset/`null` = current alternating behavior. No existing config changes behavior.
- **No change to optimizer or merge code** — they already route through `alt_state.active_side()`.
- **CPU-only verification here.** GPU runs are handed off to the user (see Task 6).
- **Every experiment YAML MUST have a matching `docs/experiments/<name>.md`** (enforced by a pre-commit hook).
- **Every POET experiment YAML MUST set `lie_ortho_distributed: true`** (enforced by `test_poet_experiment_yamls_enable_lie_ortho_distributed`). The clone inherits this.
- **Commit style:** conventional commits, e.g. `feat(poet): ...`. No AI attribution trailer.
- **Test runner:** `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest`. For tests importing `launchers.submit` (e.g. `test_megatron_args.py`), if that import fails use the `/var/tmp/zqiu/slmcpu312` venv. `poet_torch` imports need `third_party` on `PYTHONPATH` (repo conftest normally handles this; export `PYTHONPATH=$PWD/third_party:$PWD` if a bare run can't find `poet_torch`).

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `third_party/poet_torch/alt_state.py` | Shared active-side signal | Modify: add `_FIXED_SIDE` + `set_fixed_side()`, honor in `active_side()` |
| `tests/unit/test_alt_state.py` | alt_state unit tests | Modify: fixed-side override cases + teardown |
| `launchers/pretrain_gpt_slm.py` | POET argparse registration | Modify: add `--poet-lie-fixed-side {in,out}` |
| `src/utils/megatron_args.py` | config → Megatron CLI args | Modify: validation + emit for `lie_fixed_side` |
| `tests/unit/test_megatron_args.py` | config→argv unit tests | Modify: emit + validation cases |
| `src/patches/poet_apply_to_model.py` | apply POET at model build | Modify: call `alt_state.set_fixed_side(...)` from `_apply_poet_to_chunk` |
| `tests/unit/test_patch_poet_apply.py` | apply-patch unit tests | Modify: assert fixed side reaches alt_state |
| `configs/experiments/optim/poet_lie_orth_in_only.yaml` | in_only experiment | Create (clone + `lie_fixed_side: in`) |
| `configs/experiments/optim/poet_lie_orth_out_only.yaml` | out_only experiment | Create (clone + `lie_fixed_side: out`) |
| `docs/experiments/poet_lie_orth_in_only.md` | in_only doc (hook) | Create |
| `docs/experiments/poet_lie_orth_out_only.md` | out_only doc (hook) | Create |

---

## Task 1: `alt_state` fixed-side override

**Files:**
- Modify: `third_party/poet_torch/alt_state.py`
- Test: `tests/unit/test_alt_state.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `alt_state.set_fixed_side(side: str | None) -> None` — `side` must be `None`, `"in"`, or `"out"`; raises `ValueError` otherwise.
  - `alt_state.active_side(alternate_every: int = 1) -> str` — returns the pinned side when a fixed side is set, else the existing iteration-toggle.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_alt_state.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def _reset_fixed_side():
    # Global module state must not bleed across tests.
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
Expected: the three new tests FAIL (`AttributeError: module 'poet_torch.alt_state' has no attribute 'set_fixed_side'`). The existing four tests still PASS.

- [ ] **Step 3: Implement the override**

In `third_party/poet_torch/alt_state.py`, add the global next to `_ITERATION` and a setter, and honor it in `active_side()`:

```python
_ITERATION = 0
_FIXED_SIDE = None  # None = alternate by iteration; "in"/"out" = pin one side


def set_fixed_side(side) -> None:
    """Pin active_side() to one rotation side for the whole run (None = alternate).

    Set once at apply time from optim.poet.lie_fixed_side. Read by the optimizer
    write side and the merge fold side via active_side(), so pinning here makes the
    one-sided POET mode self-consistent without touching optimizer/merge code.
    """
    global _FIXED_SIDE
    if side not in (None, "in", "out"):
        raise ValueError(f"fixed_side must be None, 'in', or 'out', got {side!r}")
    _FIXED_SIDE = side
```

Then change `active_side` to honor the pin (keep the existing toggle as the fallback):

```python
def active_side(alternate_every: int = 1) -> str:
    if _FIXED_SIDE is not None:
        return _FIXED_SIDE
    every = alternate_every if alternate_every and alternate_every > 0 else 1
    return "out" if (_ITERATION // every) % 2 == 0 else "in"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alt_state.py -v`
Expected: all tests PASS (4 existing + 3 new).

- [ ] **Step 5: Lint + compile**

Run: `ruff check third_party/poet_torch/alt_state.py tests/unit/test_alt_state.py && python -m py_compile third_party/poet_torch/alt_state.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/alt_state.py tests/unit/test_alt_state.py
git commit -m "feat(poet): add alt_state fixed-side override for one-sided POET"
```

---

## Task 2: config flag — argparse + emit + validation

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py:88` (argparse, near the other `--poet-lie-*` flags)
- Modify: `src/utils/megatron_args.py` (poet branch of `_optimizer_args`: validation block ~`:442` and emit block ~`:540`)
- Test: `tests/unit/test_megatron_args.py`

**Interfaces:**
- Consumes: `alt_state` is unaffected here; this task only produces/validates the CLI flag.
- Produces:
  - CLI flag `--poet-lie-fixed-side {in,out}` (default `None`) → `args.poet_lie_fixed_side`.
  - `_optimizer_args(cfg)` emits `--poet-lie-fixed-side <v>` iff `optim.poet.lie_fixed_side` is set, after validating its companions.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_megatron_args.py` (uses the existing `_poet_cfg` / `_optimizer_args` helpers):

```python
def _fixed_side_cfg(side):
    # Minimal champion-shaped poet cfg that satisfies all lie_fixed_side companions.
    return _poet_cfg(
        {
            "block_count": 1,
            "merge_period": 1,
            "parameterization": "cayley",
            "q_optimizer": "lie_ortho",
            "single_step_fast": True,
            "single_step_x": True,
            "single_step_x_alternating": False,
            "lie_alternating": True,
            "train_output_rotation": True,
            "lie_fixed_side": side,
        }
    )


def test_lie_fixed_side_emits_flag_when_valid():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_fixed_side_cfg("in"))
    assert "--poet-lie-fixed-side" in args
    assert args[args.index("--poet-lie-fixed-side") + 1] == "in"


def test_lie_fixed_side_omitted_when_unset():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "merge_period": 1,
                "parameterization": "cayley",
                "q_optimizer": "lie_ortho",
                "single_step_fast": True,
                "single_step_x": True,
                "lie_alternating": True,
            }
        )
    )
    assert "--poet-lie-fixed-side" not in args


def test_lie_fixed_side_requires_lie_alternating():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    cfg = _fixed_side_cfg("in")
    cfg.optim.poet.lie_alternating = False
    with pytest.raises(ValueError, match="lie_fixed_side"):
        _optimizer_args(cfg)


def test_lie_fixed_side_mutually_exclusive_with_single_step_x_alternating():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    cfg = _fixed_side_cfg("out")
    cfg.optim.poet.single_step_x_alternating = True
    cfg.optim.poet.lie_alternating = False  # alt_x forbids lie_alternating; isolate the fixed_side guard
    with pytest.raises(ValueError, match="lie_fixed_side"):
        _optimizer_args(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k lie_fixed_side -v`
Expected: FAIL — the emit test finds no `--poet-lie-fixed-side`; the validation tests do not raise.
(If the module errors on `from launchers.submit import _parse_overrides`, rerun with the `/var/tmp/zqiu/slmcpu312` venv.)

- [ ] **Step 3a: Register the argparse flag**

In `launchers/pretrain_gpt_slm.py`, immediately after the `--poet-lie-alternate-every` line (~`:89`):

```python
    # Pin the rotation to ONE fixed side for the whole run (one-sided POET):
    # "in" = train only oft_R_in, "out" = train only oft_R_out. Default (absent)
    # = the alternating schedule. Built on the lie_alternating integrated path.
    group.add_argument("--poet-lie-fixed-side", choices=["in", "out"], default=None)
```

- [ ] **Step 3b: Add validation in the poet branch**

In `src/utils/megatron_args.py`, inside the `if kind == "poet":` branch, after the `single_step_x_alternating` validation block (just before the `# block_count ...` comment, ~`:473`):

```python
        lie_fixed_side = poet.get("lie_fixed_side", None)
        if lie_fixed_side is not None:
            if lie_fixed_side not in ("in", "out"):
                raise ValueError(
                    f"optim.poet.lie_fixed_side must be 'in' or 'out', got {lie_fixed_side!r}."
                )
            if not poet.get("lie_alternating", False):
                raise ValueError(
                    "optim.poet.lie_fixed_side requires lie_alternating=true "
                    "(it pins the integrated alternating path to one side)."
                )
            if not poet.get("single_step_x", False):
                raise ValueError(
                    "optim.poet.lie_fixed_side requires single_step_x=true (POETX path)."
                )
            if poet.get("q_optimizer", "adam") != "lie_ortho":
                raise ValueError(
                    "optim.poet.lie_fixed_side requires q_optimizer=lie_ortho."
                )
            if poet.get("single_step_x_alternating", False):
                raise ValueError(
                    "optim.poet.lie_fixed_side is mutually exclusive with "
                    "single_step_x_alternating (pick a pinned side OR the true-single-side "
                    "layer, not both)."
                )
            if merge_period != 1:
                raise ValueError("optim.poet.lie_fixed_side requires merge_period=1.")
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError("optim.poet.lie_fixed_side requires parameterization=cayley.")
```

- [ ] **Step 3c: Emit the flag**

In `src/utils/megatron_args.py`, in the store_true / conditional-emit section after the `--poet-single-step-x-alternating` emit (~`:573`):

```python
        # Pin the rotation to one fixed side for the whole run (one-sided POET).
        if poet.get("lie_fixed_side", None) is not None:
            poet_args.append("--poet-lie-fixed-side")
            poet_args.append(str(poet.get("lie_fixed_side")))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k lie_fixed_side -v`
Expected: all four new tests PASS.

- [ ] **Step 5: Lint + compile**

Run: `ruff check src/utils/megatron_args.py launchers/pretrain_gpt_slm.py tests/unit/test_megatron_args.py && python -m py_compile src/utils/megatron_args.py launchers/pretrain_gpt_slm.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/utils/megatron_args.py launchers/pretrain_gpt_slm.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): add --poet-lie-fixed-side flag with validation and emit"
```

---

## Task 3: apply plumbing — set the pin at model build

**Files:**
- Modify: `src/patches/poet_apply_to_model.py` (`_apply_poet_to_chunk`, ~`:57-98`)
- Test: `tests/unit/test_patch_poet_apply.py`

**Interfaces:**
- Consumes: `alt_state.set_fixed_side` (Task 1); `args.poet_lie_fixed_side` (Task 2).
- Produces: side effect — after `_apply_poet_to_chunk(m, args)` runs, `alt_state.active_side()` returns the pinned side when `args.poet_lie_fixed_side` is set (and is restored to alternating when it is `None`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_patch_poet_apply.py`:

```python
def test_apply_to_chunk_sets_fixed_side(monkeypatch):
    from types import SimpleNamespace

    import src.patches.poet_apply_to_model as ap
    from poet_torch import alt_state

    # Don't actually walk a model: stub the linear-replacement walk.
    monkeypatch.setattr(ap, "replace_linears_with_poet", lambda m, **kw: 0)
    alt_state.set_fixed_side(None)

    args = SimpleNamespace(
        poet_block_size=256,
        poet_block_count=1,
        poet_lie_fixed_side="in",
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=None,
    )
    ap._apply_poet_to_chunk(object(), args)
    try:
        alt_state.set_iteration(0)  # would be "out" under the toggle
        assert alt_state.active_side(1) == "in"
    finally:
        alt_state.set_fixed_side(None)


def test_apply_to_chunk_leaves_alternating_when_unset(monkeypatch):
    from types import SimpleNamespace

    import src.patches.poet_apply_to_model as ap
    from poet_torch import alt_state

    monkeypatch.setattr(ap, "replace_linears_with_poet", lambda m, **kw: 0)
    alt_state.set_fixed_side(None)

    args = SimpleNamespace(
        poet_block_size=256,
        poet_block_count=1,
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=None,
    )
    ap._apply_poet_to_chunk(object(), args)
    alt_state.set_iteration(0)
    assert alt_state.active_side(1) == "out"  # toggle intact (no pin)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_apply.py -k fixed_side -v`
Expected: `test_apply_to_chunk_sets_fixed_side` FAILS (`active_side` returns `"out"`, the pin was never applied).

- [ ] **Step 3: Plumb the pin into `_apply_poet_to_chunk`**

In `src/patches/poet_apply_to_model.py`, inside `_apply_poet_to_chunk`, after the `alternate_every = getattr(...)` line (~`:72`):

```python
    lie_fixed_side = getattr(args, "poet_lie_fixed_side", None)
    # One-sided POET: pin the shared active-side signal so the optimizer write side
    # and the merge fold side both target the fixed side (None = alternate as before).
    from poet_torch import alt_state

    alt_state.set_fixed_side(lie_fixed_side)
```

(No change to the `replace_linears_with_poet(...)` call — the layer is unaffected; only the shared `alt_state` signal is pinned.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_apply.py -k fixed_side -v`
Expected: both tests PASS.

- [ ] **Step 5: Lint + compile**

Run: `ruff check src/patches/poet_apply_to_model.py tests/unit/test_patch_poet_apply.py && python -m py_compile src/patches/poet_apply_to_model.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/patches/poet_apply_to_model.py tests/unit/test_patch_poet_apply.py
git commit -m "feat(poet): pin alt_state fixed side at model build from config"
```

---

## Task 4: experiment YAMLs + docs + YAML→argv smoke test

**Files:**
- Create: `configs/experiments/optim/poet_lie_orth_in_only.yaml`
- Create: `configs/experiments/optim/poet_lie_orth_out_only.yaml`
- Create: `docs/experiments/poet_lie_orth_in_only.md`
- Create: `docs/experiments/poet_lie_orth_out_only.md`
- Test: `tests/unit/test_megatron_args.py`

**Interfaces:**
- Consumes: the `lie_fixed_side` flag plumbing (Tasks 1-3).
- Produces: two runnable experiments `optim/poet_lie_orth_in_only` and `optim/poet_lie_orth_out_only`.

- [ ] **Step 1: Write the failing smoke test**

Add to `tests/unit/test_megatron_args.py`:

```python
def test_in_only_yaml_emits_fixed_side_in():
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
    assert m["--poet-lie-fixed-side"] == "in"
    assert m["--poet-single-step-x"] is True
    assert m["--poet-lie-alternating"] is True


def test_out_only_yaml_emits_fixed_side_out():
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
    assert m["--poet-lie-fixed-side"] == "out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "in_only or out_only" -v`
Expected: FAIL — Hydra cannot find `experiment=optim/poet_lie_orth_in_only` (config missing).
(Use the `/var/tmp/zqiu/slmcpu312` venv if `launchers.submit` import fails.)

- [ ] **Step 3a: Create `poet_lie_orth_in_only.yaml`**

`configs/experiments/optim/poet_lie_orth_in_only.yaml`:

```yaml
# @package _global_
# poet_lie_orth_in_only: the integrated alternating POETX champion (poet_lie_orth_alt)
# with the rotation PINNED to the input side for the whole run. Only oft_R_in is ever
# written/folded; oft_R_out stays at its zero init (identity) forever. Same recipe as
# poet_lie_orth_alt (single_step_x + lie_alternating, lie_ortho c=8, lr 3e-3,
# block_count 1, merge_period 1, scale 0.5) except active_side is pinned via
# lie_fixed_side. Both Lie momenta still advance; forward/backward still feeds both
# grads -- bit-identical to the champion except the side never toggles.
# See docs/superpowers/specs/2026-06-19-poet-one-sided-mode-design.md
experiment:
  name: poet_lie_orth_in_only
  family: optim
  description: |
    One-sided POET (input side): champion poet_lie_orth_alt recipe with the rotation
    pinned to the input side. Only oft_R_in trains; oft_R_out stays identity. Ablation
    of single-side-only rotation vs the both-sides alternating champion (val/loss ≈3.5332).
  references:
    - "POET"
    - "Muon"
    - "Pion"
  patches:
    - model_unfuse_linears
    - poet_optimizer_setup
    - poet_unfuse_te_impl
    - poet_moe_local_rmsnorm
    - poet_apply_to_model
    - poet_merge_step
    - sandwich_norm_apply
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
    lie_alternating: true
    lie_alternate_every: 1
    train_output_rotation: true
    lie_fixed_side: in

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 3b: Create `poet_lie_orth_out_only.yaml`**

Identical to Step 3a except: header/`name`/`description` say "output side", and `lie_fixed_side: out`. Full file:

```yaml
# @package _global_
# poet_lie_orth_out_only: the integrated alternating POETX champion (poet_lie_orth_alt)
# with the rotation PINNED to the output side for the whole run. Only oft_R_out is ever
# written/folded; oft_R_in stays at its zero init (identity) forever. Same recipe as
# poet_lie_orth_alt (single_step_x + lie_alternating, lie_ortho c=8, lr 3e-3,
# block_count 1, merge_period 1, scale 0.5) except active_side is pinned via
# lie_fixed_side. Both Lie momenta still advance; forward/backward still feeds both
# grads -- bit-identical to the champion except the side never toggles.
# See docs/superpowers/specs/2026-06-19-poet-one-sided-mode-design.md
experiment:
  name: poet_lie_orth_out_only
  family: optim
  description: |
    One-sided POET (output side): champion poet_lie_orth_alt recipe with the rotation
    pinned to the output side. Only oft_R_out trains; oft_R_in stays identity. Ablation
    of single-side-only rotation vs the both-sides alternating champion (val/loss ≈3.5332).
  references:
    - "POET"
    - "Muon"
    - "Pion"
  patches:
    - model_unfuse_linears
    - poet_optimizer_setup
    - poet_unfuse_te_impl
    - poet_moe_local_rmsnorm
    - poet_apply_to_model
    - poet_merge_step
    - sandwich_norm_apply
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
    lie_alternating: true
    lie_alternate_every: 1
    train_output_rotation: true
    lie_fixed_side: out

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 3c: Create the matching experiment docs (pre-commit hook)**

`docs/experiments/poet_lie_orth_in_only.md`:

```markdown
# poet_lie_orth_in_only

One-sided POET (input side). The integrated alternating POETX champion
(`poet_lie_orth_alt`) with the rotation **pinned to the input side** for the whole
run via `optim.poet.lie_fixed_side: in`. Only `oft_R_in` is ever written and folded;
`oft_R_out` stays at its zero init (identity) forever.

- **Recipe:** identical to `poet_lie_orth_alt` — `single_step_x` + `lie_alternating`,
  `lie_ortho` (c=8, muon, 5 NS, distributed), lr 3e-3, `block_count 1`,
  `merge_period 1`, `scale 0.5`. The only difference is the pinned side.
- **Mechanism:** `lie_fixed_side` pins the shared `alt_state.active_side()` signal, so
  both the optimizer write side and the merge fold side target `in`. Both Lie momenta
  still advance and forward/backward still feed both grads — bit-identical to the
  champion except the side never toggles.
- **Design:** `docs/superpowers/specs/2026-06-19-poet-one-sided-mode-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-19-poet-one-sided-mode.md`
- **Ablation target:** the both-sides alternating champion (val/loss ≈3.5332).
```

`docs/experiments/poet_lie_orth_out_only.md`: same as above with "input"→"output", `oft_R_in`/`oft_R_out` swapped, and `lie_fixed_side: out`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "in_only or out_only" -v`
Expected: both tests PASS.

- [ ] **Step 5: Confirm the YAML-walk guards still pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "lie_ortho_distributed" -v`
Expected: PASS (the clones set `lie_ortho_distributed: true`).

- [ ] **Step 6: Commit**

```bash
git add configs/experiments/optim/poet_lie_orth_in_only.yaml \
        configs/experiments/optim/poet_lie_orth_out_only.yaml \
        docs/experiments/poet_lie_orth_in_only.md \
        docs/experiments/poet_lie_orth_out_only.md \
        tests/unit/test_megatron_args.py
git commit -m "feat(poet): add poet_lie_orth_in_only/out_only one-sided experiments"
```

---

## Task 5: full CPU verification

**Files:** none (verification only).

- [ ] **Step 1: Run the touched test modules together**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_alt_state.py \
  tests/unit/test_megatron_args.py \
  tests/unit/test_patch_poet_apply.py -v
```
Expected: all PASS. (If `test_megatron_args.py` fails to import `launchers.submit`, rerun that file with the `/var/tmp/zqiu/slmcpu312` venv. The repo has 2 known pre-existing `launchers.submit` failures unrelated to this change — confirm any failures are those two, not new.)

- [ ] **Step 2: Run the broader POET regression set (no behavior change expected)**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_lie_orth.py \
  tests/unit/test_poet_merge_step.py \
  tests/unit/test_alternating_poetx.py \
  tests/unit/test_poetx_layer.py -v
```
Expected: all PASS (optimizer/merge/layer untouched; `_FIXED_SIDE` defaults to `None`).

- [ ] **Step 3: Pre-commit on the full change set**

Run: `pre-commit run --files $(git diff --name-only HEAD~4 HEAD)`
Expected: all hooks pass — in particular "Every experiment YAML has a matching docs/experiments/<name>.md".

- [ ] **Step 4: Update CHANGELOG**

Append a one-line entry under the current date to `NeckariumAI/zqiu/CHANGELOG.md` (per the user's changelog policy): "Added one-sided POET mode (`optim.poet.lie_fixed_side: in|out`) + `poet_lie_orth_in_only`/`poet_lie_orth_out_only` experiments." Commit:

```bash
git add NeckariumAI/zqiu/CHANGELOG.md
git commit -m "docs(changelog): one-sided POET mode"
```

---

## Task 6: GPU validation handoff (user runs)

**Files:** none. Hand the user exact launch commands; do NOT run GPU jobs.

- [ ] **Step 1: Provide the launch commands**

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
(Adjust `base/scale`, `training_regime`, and `cluster` to match the champion sweep being compared against.)

- [ ] **Step 2: What to check in the run**

- Training is stable (no NaN/OOM); step time ≈ the `poet_lie_orth_alt` champion.
- The frozen side's `oft_R` norm stays exactly 0 for the whole run (input side frozen for `out_only`, output side frozen for `in_only`) — confirm via the existing POET diag/`_DUMP_POET_PARAMS` logging.
- One-sided val/loss is the ablation signal — expected to be somewhat worse than the both-sides champion (val/loss ≈3.5332); the comparison is the point of the mode.

---

## Self-Review

**Spec coverage:**
- alt_state override → Task 1 ✓
- config flag (`lie_fixed_side`) argparse + emit + validation → Task 2 ✓
- apply plumbing (`set_fixed_side`) → Task 3 ✓
- optimizer + merge unchanged → no task (explicitly verified in Task 5 Step 2) ✓
- two experiment YAMLs → Task 4 ✓
- matching `docs/experiments/*.md` (hook) → Task 4 Step 3c ✓
- CPU tests (alt_state, megatron_args, apply) → Tasks 1-4 + Task 5 ✓
- GPU handoff → Task 6 ✓
- out-of-scope items (speed, non-lie_ortho paths, new merge/opt code) → none added ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code (the only "same as above" is `poet_lie_orth_out_only.md`, a 6-line doc whose full content is fully specified by the swap rule — and the YAML twin is given in full at 3b). ✓

**Type consistency:** `set_fixed_side(side)` / `active_side(alternate_every)` signatures consistent across Tasks 1, 3, and tests; `--poet-lie-fixed-side` ↔ `args.poet_lie_fixed_side` ↔ `optim.poet.lie_fixed_side` consistent across Tasks 2-4. ✓
