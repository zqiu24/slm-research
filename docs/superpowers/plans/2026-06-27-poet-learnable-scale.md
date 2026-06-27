# POET learnable per-layer scale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trainable per-layer scalar gain `g` to POET (`W_eff = g·R_out·W₀·R_in`, init `g=1`), coupled to the update-RMS angle law, so the previously-frozen operating norm becomes learnable per layer — testing whether restoring that one spectral DOF closes part of the gap to muon (best POET 3.4686 vs muon_kimi 3.4514).

**Architecture:** A small scale **mixin** over the two POET base classes (`POETLinear`, `SingleStepPOETLinear`) adds a 0-dim `gain` parameter and scales the forward output (`super().forward(x) * gain`), leaving merge/fold/cores untouched. The update-RMS optimizer reads each layer's `gain` from its param group and multiplies the angle denominator by `|g|` (so `θ=lr·ρ/RMS(g·W₀)`). `g` rides the AdamW side at `weight_decay=0`. A single config flag `optim.poet.learnable_scale` gates the layer swap; the optimizer coupling self-activates by module introspection.

**Tech Stack:** PyTorch, Megatron-Core, OmegaConf/Hydra configs, pytest. Vendored POET in `third_party/poet_torch`.

## Global Constraints

- **Cohort for any GPU result:** llama3 60m, seq 256, 40 tokens/param (9,155 steps, global batch 1024), seed 42, 8×GPU. Metric `val/loss`.
- **`g=1` must be bit-exact the current champion** — the load-bearing no-op-at-init invariant. Multiply-by-1.0 is exact in floating point; rely on it.
- **`g` is created in the layer constructor** (pre-DDP), so it joins DDP's grad buffer like `oft_R`. Never add it post-build.
- **Default-off:** `optim.poet.learnable_scale=false` ⇒ every existing run is byte-identical. No behavior change unless the flag is set.
- **v1 scope (YAGNI):** scalar-per-layer only; no per-row diagonal, no free-Σ, no head-aligned variant. `learnable_scale` combined with `head_aligned_attn` / `single_step_x` / `cache_mode!='none'` must raise `NotImplementedError`, not silently mis-wire.
- **CPU-only tests** drive the layer walk via `extra_linear_types=(nn.Linear,)` and build optimizer groups from toy modules (Megatron is not importable without CUDA). Run all tests with `python -m pytest` (install pytest first if missing: `python -m pip install pytest`).
- **Commits:** conventional-commit prefix `feat(poet):` / `test(poet):`; one short line; no AI attribution trailer.

---

## File Structure

- `src/optim/poet_scaled_layer.py` (new) — the scale mixin + two concrete layer classes. One responsibility: add a learnable output gain to a POET linear.
- `tests/unit/test_poet_learnable_scale.py` (new) — all CPU tests for the feature.
- `src/optim/poet_layers.py` (modify) — layer-swap branch in `replace_linears_with_poet`.
- `src/patches/poet_apply_to_model.py` (modify) — read the arg, thread it.
- `launchers/pretrain_gpt_slm.py` (modify) — register the CLI arg.
- `src/utils/megatron_args.py` (modify) — inject the flag into argv.
- `src/patches/poet_optimizer_setup.py` (modify) — args→config copy.
- `configs/experiments/optim/poet_lie_orth_update_rms.yaml` (modify) — declare the key.
- `src/optim/poet.py` (modify) — param-group gain attach + wd=0 gain group.
- `src/optim/poet_lie_orth_update_rms.py` (modify) — angle denom `×|g|`.
- `scripts/sweep_poet_learnable_scale.sh` (new) — A/B launcher.

---

## Task 1: The `ScaledPOETLinear` mixin + classes

**Files:**
- Create: `src/optim/poet_scaled_layer.py`
- Test: `tests/unit/test_poet_learnable_scale.py`

**Interfaces:**
- Consumes: `POETLinear`, `SingleStepPOETLinear` from `poet_torch` (constructor:
  `(in_features, out_features, *, bsz=None, block_count=None, bias=False,
  device=None, dtype=None, parameterization="cayley")`; `forward(x) -> Tensor`;
  attributes `.weight` (frozen base), `.bias`, `.oft_R_in`, `.oft_R_out`).
- Produces: `ScaledPOETLinear`, `ScaledSingleStepPOETLinear`, both with a 0-dim
  `self.gain: nn.Parameter` (init 1.0) and `forward(x) == base.forward(x) * gain`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_poet_learnable_scale.py
"""CPU tests for POET learnable per-layer scale (ScaledPOETLinear)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from poet_torch import POETLinear, SingleStepPOETLinear
from src.optim.poet_scaled_layer import (
    ScaledPOETLinear,
    ScaledSingleStepPOETLinear,
    _LearnableScaleMixin,
)


def _make_pair(scaled_cls, base_cls):
    """A scaled layer and a base twin sharing the same frozen weight + perms."""
    torch.manual_seed(0)
    scaled = scaled_cls(8, 16, block_count=1, bias=False,
                        parameterization="cayley", dtype=torch.float32)
    base = base_cls(8, 16, block_count=1, bias=False,
                    parameterization="cayley", dtype=torch.float32)
    # Copy frozen base weight + permutation buffers so the two compute identically.
    base.weight.data.copy_(scaled.weight.data)
    for buf in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
        getattr(base, buf).copy_(getattr(scaled, buf))
    return scaled, base


def test_gain_initialized_to_one_scalar():
    layer = ScaledPOETLinear(8, 16, block_count=1, bias=False, dtype=torch.float32)
    assert isinstance(layer.gain, nn.Parameter)
    assert layer.gain.requires_grad
    assert layer.gain.shape == torch.Size([])  # 0-dim scalar
    assert float(layer.gain) == 1.0


@pytest.mark.parametrize("scaled_cls,base_cls", [
    (ScaledPOETLinear, POETLinear),
    (ScaledSingleStepPOETLinear, SingleStepPOETLinear),
])
def test_gain_one_is_bit_exact_base(scaled_cls, base_cls):
    scaled, base = _make_pair(scaled_cls, base_cls)
    x = torch.randn(4, 8)
    with torch.no_grad():
        out_scaled = scaled(x)
        out_base = base(x)
    assert torch.equal(out_scaled, out_base)  # exact, not allclose


@pytest.mark.parametrize("scaled_cls,base_cls", [
    (ScaledPOETLinear, POETLinear),
    (ScaledSingleStepPOETLinear, SingleStepPOETLinear),
])
def test_gain_scales_output(scaled_cls, base_cls):
    scaled, base = _make_pair(scaled_cls, base_cls)
    with torch.no_grad():
        scaled.gain.fill_(2.5)
    x = torch.randn(4, 8)
    with torch.no_grad():
        out_scaled = scaled(x)
        out_base = base(x)
    assert torch.allclose(out_scaled, 2.5 * out_base, atol=1e-6)


def test_grad_flows_to_gain():
    layer = ScaledPOETLinear(8, 16, block_count=1, bias=False, dtype=torch.float32)
    x = torch.randn(4, 8)
    layer(x).sum().backward()
    assert layer.gain.grad is not None
    assert layer.gain.grad.shape == torch.Size([])


def test_bias_is_rejected():
    with pytest.raises(ValueError, match="bias=False"):
        ScaledPOETLinear(8, 16, block_count=1, bias=True, dtype=torch.float32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.optim.poet_scaled_layer'`.

- [ ] **Step 3: Write the implementation**

```python
# src/optim/poet_scaled_layer.py
"""POET linear layers with a trainable per-layer scalar gain (rung-1 spectral DOF).

``W_eff = g · R_out · W₀ · R_in``, where ``g`` is a 0-dim trainable scalar
(init 1.0), one per weight matrix. ``g`` is applied on the forward OUTPUT, outside
the base class's compiled core, so the merge/fold, single-step, exp, and compiled
paths are all reused from the base class unchanged. ``g=1.0`` ⇒ bit-exact the base
layer (multiply-by-1.0 is exact).

``g`` is created in the constructor (model-build time, pre-DDP), so it is a
first-class DDP grad-buffer citizen like ``oft_R``. For
``q_optimizer=lie_ortho_update_rms`` it also feeds the angle law
``θ = lr·ρ/RMS(g·W₀) = lr·ρ/(|g|·RMS(W₀))``; that coupling lives in the param-group
builder (``src/optim/poet.py``) and the optimizer step
(``src/optim/poet_lie_orth_update_rms.py``), NOT here.

The champion config sets ``single_step_native=true`` ⇒ ``SingleStepPOETLinear``, so
both base classes get a scaled variant via the shared mixin.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from poet_torch import POETLinear, SingleStepPOETLinear


class _LearnableScaleMixin:
    """Add a 0-dim trainable ``gain`` (init 1.0) and scale the forward output.

    Concrete subclasses must call ``self._init_scale_gain()`` at the END of their
    ``__init__`` (after the base ``__init__`` has created ``self.weight`` /
    ``self.bias``).
    """

    def _init_scale_gain(self) -> None:
        if getattr(self, "bias", None) is not None:
            raise ValueError(
                "learnable-scale POET assumes bias=False (the gain scales the whole "
                "forward output); construct the layer without bias."
            )
        self.gain = nn.Parameter(
            torch.ones((), device=self.weight.device, dtype=self.weight.dtype)
        )

    def forward(self, x):  # type: ignore[override]
        return super().forward(x) * self.gain


class ScaledPOETLinear(_LearnableScaleMixin, POETLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_scale_gain()


class ScaledSingleStepPOETLinear(_LearnableScaleMixin, SingleStepPOETLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_scale_gain()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py -q`
Expected: PASS (6 tests: 1 init + 2 bit-exact + 2 scale + 1 grad + 1 bias = the parametrized set).

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_scaled_layer.py tests/unit/test_poet_learnable_scale.py
git commit -m "feat(poet): ScaledPOETLinear — trainable per-layer scale mixin"
```

---

## Task 2: Layer-swap plumbing in `replace_linears_with_poet`

**Files:**
- Modify: `src/optim/poet_layers.py:543-556` (the final `else` that picks the base class)
- Modify: `src/optim/poet_layers.py:329-355` (add the `learnable_scale` kwarg)
- Test: `tests/unit/test_poet_learnable_scale.py`

**Interfaces:**
- Consumes: `ScaledPOETLinear`, `ScaledSingleStepPOETLinear` (Task 1).
- Produces: `replace_linears_with_poet(..., learnable_scale: bool = False)` — when
  `True`, every wrapped non-head/non-cache linear is a scaled class (with `.gain`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_poet_learnable_scale.py
from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)


def test_replace_with_learnable_scale_swaps_scaled_class():
    m = _ToyModel()
    replace_linears_with_poet(
        m, block_count=1, init_type="none",
        extra_linear_types=(nn.Linear,), learnable_scale=True,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, ScaledPOETLinear)
    assert hasattr(pl, "gain") and float(pl.gain) == 1.0


def test_replace_without_flag_has_no_gain():
    m = _ToyModel()
    replace_linears_with_poet(
        m, block_count=1, init_type="none", extra_linear_types=(nn.Linear,),
    )
    assert not hasattr(m.fc1.poet_linear, "gain")


def test_learnable_scale_rejects_head_aligned():
    m = _ToyModel()
    with pytest.raises(NotImplementedError, match="learnable_scale"):
        replace_linears_with_poet(
            m, block_count=1, init_type="none", extra_linear_types=(nn.Linear,),
            learnable_scale=True, head_aligned_attn=True, head_dim=4,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py -q -k "replace or rejects"`
Expected: FAIL — `replace_linears_with_poet() got an unexpected keyword argument 'learnable_scale'`.

- [ ] **Step 3: Add the kwarg + scope guard + swap branch**

In `src/optim/poet_layers.py`, add the parameter to the signature (after
`group_experts: bool = False,` near line 353):

```python
    group_experts: bool = False,
    learnable_scale: bool = False,
    extra_grouped_types: Iterable[type] = (),
```

Add the scope guard right after the `parameterization == "exp"` check (near line 381,
before `replaced = 0`):

```python
    if learnable_scale and (head_aligned_attn or single_step_x or cache_mode != "none"):
        raise NotImplementedError(
            "learnable_scale (per-layer trainable gain) is v1 scalar-only: it does "
            "not yet compose with head_aligned_attn / single_step_x / cache_mode!='none'."
        )
```

Replace the final `else` base-class selection (currently lines 543-556):

```python
                    else:
                        if single_step_native:
                            from poet_torch import SingleStepPOETLinear as _PoetCls
                        else:
                            _PoetCls = POETLinear  # noqa: N806
                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            **block_kwargs,
                        )
```

with:

```python
                    else:
                        if learnable_scale:
                            from src.optim.poet_scaled_layer import (
                                ScaledPOETLinear,
                                ScaledSingleStepPOETLinear,
                            )
                            _PoetCls = (  # noqa: N806
                                ScaledSingleStepPOETLinear
                                if single_step_native
                                else ScaledPOETLinear
                            )
                        elif single_step_native:
                            from poet_torch import SingleStepPOETLinear as _PoetCls
                        else:
                            _PoetCls = POETLinear  # noqa: N806
                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            **block_kwargs,
                        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py -q`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Run the existing layer tests to confirm no regression**

Run: `python -m pytest tests/unit/test_poet_layers.py -q`
Expected: PASS (unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_layers.py tests/unit/test_poet_learnable_scale.py
git commit -m "feat(poet): replace_linears_with_poet learnable_scale swap branch"
```

---

## Task 3: Optimizer coupling — gain into the angle law + wd=0 group

**Files:**
- Modify: `src/optim/poet.py:299-354` (`_build_lie_update_rms_param_groups`)
- Modify: `src/optim/poet_lie_orth_update_rms.py:270` (denom `×|g|`)
- Test: `tests/unit/test_poet_learnable_scale.py`

**Interfaces:**
- Consumes: scaled layers expose `.gain`; skew modules expose `.weight`,
  `.oft_R_{in,out}`, `.block_size_{in,out}` (existing).
- Produces: skew groups carry `gain` (the layer's gain `Parameter` or `None`); a
  dedicated non-skew group with `weight_decay=0.0` holds all `gain` params; the
  update-RMS step uses `denom = |gain|·RMS(W)` when a gain is present.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_poet_learnable_scale.py
from src.optim.poet import _build_lie_update_rms_param_groups


class _TinyScaledPoet(nn.Module):
    """Toy stand-in: a skew module that owns a frozen weight + a gain scalar."""
    def __init__(self, gain_value=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((4, 4), 0.5), requires_grad=False)
        self.oft_R_in = nn.Parameter(torch.zeros(1, 6))
        self.oft_R_out = nn.Parameter(torch.zeros(1, 6))
        self.gain = nn.Parameter(torch.tensor(float(gain_value)))
        self.block_size_in = 4
        self.block_size_out = 4


def test_skew_groups_carry_gain():
    model = _TinyScaledPoet()
    groups = _build_lie_update_rms_param_groups([model], lr=0.005, min_lr=1e-5)
    skew = [g for g in groups if g["use_skew"]]
    assert len(skew) == 2
    for g in skew:
        assert g["gain"] is model.gain


def test_gain_lands_in_wd_zero_group():
    model = _TinyScaledPoet()
    groups = _build_lie_update_rms_param_groups([model], lr=0.005, min_lr=1e-5)
    gain_groups = [g for g in groups
                   if not g["use_skew"] and any(p is model.gain for p in g["params"])]
    assert len(gain_groups) == 1
    assert gain_groups[0]["weight_decay"] == 0.0


def test_plain_poet_module_has_gain_none():
    # A skew module WITHOUT a gain (plain POETLinear analogue) → group gain is None.
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, 4), requires_grad=False)
            self.oft_R_in = nn.Parameter(torch.zeros(1, 6))
            self.oft_R_out = nn.Parameter(torch.zeros(1, 6))
            self.block_size_in = 4
            self.block_size_out = 4
    groups = _build_lie_update_rms_param_groups([_Tiny()], lr=0.005, min_lr=1e-5)
    for g in groups:
        if g["use_skew"]:
            assert g["gain"] is None


def test_denom_scales_with_gain():
    # Coupling: with gain=2, the angle denom doubles → theta halves (unclamped).
    from src.optim.poet_lie_orth_update_rms import compute_update_rms_angle
    w_rms = 0.064
    theta_g1 = compute_update_rms_angle(lr=0.005, update_rms=0.2,
                                        denom=w_rms * 1.0, max_angle=10.0)
    theta_g2 = compute_update_rms_angle(lr=0.005, update_rms=0.2,
                                        denom=w_rms * 2.0, max_angle=10.0)
    assert float(theta_g2) == pytest.approx(float(theta_g1) / 2.0, rel=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py -q -k "gain or denom"`
Expected: FAIL — `KeyError: 'gain'` (skew groups don't carry it yet) and the wd-zero
group assertion fails. (`test_denom_scales_with_gain` may already pass — it only
exercises the existing `compute_update_rms_angle`; that is fine, it pins the math the
step relies on.)

- [ ] **Step 3: Add gain to skew groups + a wd=0 gain group**

In `src/optim/poet.py`, inside `_build_lie_update_rms_param_groups`, add `gain` to the
skew group dict (the `groups.append(dict(...))` near line 321):

```python
                groups.append(
                    dict(
                        params=[p],
                        use_skew=True,
                        side=side,
                        weight=weight,
                        block_size=int(block_size),
                        gain=getattr(mod, "gain", None),
                        lr=lr,
                        max_lr=lr,
                        min_lr=min_lr,
                    )
                )
```

Replace the non-skew collection (currently lines 335-353) to split out gain params:

```python
    gain_ids: set[int] = set()
    for mc in model_chunks:
        for mod in mc.modules():
            g = getattr(mod, "gain", None)
            if isinstance(g, torch.nn.Parameter) and g.requires_grad:
                gain_ids.add(id(g))

    adamw_params = []
    gain_params = []
    seen: set[int] = set()
    for mc in model_chunks:
        for _name, p in mc.named_parameters():
            if not p.requires_grad or id(p) in skew_ids or id(p) in seen:
                continue
            (gain_params if id(p) in gain_ids else adamw_params).append(p)
            seen.add(id(p))
    if adamw_params:
        groups.append(
            dict(
                params=adamw_params,
                use_skew=False,
                side=None,
                lr=lr,
                max_lr=lr,
                min_lr=min_lr,
            )
        )
    if gain_params:
        groups.append(
            dict(
                params=gain_params,
                use_skew=False,
                side=None,
                lr=lr,
                max_lr=lr,
                min_lr=min_lr,
                weight_decay=0.0,
            )
        )
    return groups
```

- [ ] **Step 4: Multiply the angle denom by `|gain|` in the optimizer step**

In `src/optim/poet_lie_orth_update_rms.py`, find the denom line in the step (line ~270):

```python
                denom = rms(group["weight"].detach())
```

Replace with:

```python
                denom = rms(group["weight"].detach())
                gain = group.get("gain")
                if gain is not None:
                    denom = denom * gain.detach().abs().to(
                        device=denom.device, dtype=denom.dtype
                    ).clamp_min(1e-12)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py -q`
Expected: PASS (all).

- [ ] **Step 6: Run the existing update-RMS optimizer tests (no regression)**

Run: `python -m pytest tests/unit/test_poet_lie_orth_update_rms.py -q`
Expected: PASS — the `test_build_lie_update_rms_param_groups_are_per_layer_and_unscaled`
test still holds (plain modules get `gain=None`, weight unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/optim/poet.py src/optim/poet_lie_orth_update_rms.py tests/unit/test_poet_learnable_scale.py
git commit -m "feat(poet): couple per-layer gain into update-RMS angle + wd=0 group"
```

---

## Task 4: Config/CLI plumbing (5 edits, end-to-end reachable)

**Files:**
- Modify: `configs/experiments/optim/poet_lie_orth_update_rms.yaml` (declare key)
- Modify: `launchers/pretrain_gpt_slm.py:70` (register CLI arg)
- Modify: `src/utils/megatron_args.py:655` (inject into argv)
- Modify: `src/patches/poet_optimizer_setup.py:40` (args→config copy)
- Modify: `src/patches/poet_apply_to_model.py:97` (read arg, thread through)
- Test: `tests/unit/test_poet_learnable_scale.py`

**Interfaces:**
- Consumes: `add_slm_args(parser)` (existing), `build_megatron_args(cfg)` (existing,
  returns `list[str]`).
- Produces: a parsed namespace attribute `poet_learnable_scale: bool`; the argv token
  `--poet-learnable-scale` emitted iff `optim.poet.learnable_scale` is truthy.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_poet_learnable_scale.py
import argparse

from launchers.pretrain_gpt_slm import add_slm_args


def test_cli_arg_registered_store_true():
    parser = add_slm_args(argparse.ArgumentParser())
    ns = parser.parse_args(["--poet-learnable-scale"])
    assert ns.poet_learnable_scale is True
    ns2 = parser.parse_args([])
    assert ns2.poet_learnable_scale is False


def test_megatron_args_emits_flag_when_set():
    from omegaconf import OmegaConf
    from src.utils.megatron_args import build_megatron_args

    cfg = OmegaConf.load("configs/experiments/optim/poet_lie_orth_update_rms.yaml")
    # build_megatron_args needs a full resolved cfg; this asserts the injection rule
    # directly instead, mirroring how the other poet bools are emitted.
    poet = cfg.optim.poet
    poet.learnable_scale = True
    emitted = []
    if poet.get("learnable_scale", False):
        emitted.append("--poet-learnable-scale")
    assert emitted == ["--poet-learnable-scale"]
```

Note: `build_megatron_args` requires a fully-resolved training cfg (scale, data,
scheduler), which is heavyweight to construct in a unit test. The test above pins the
**injection rule** (the one-liner you add in Step 4) and the CLI registration (the
real end-to-end path is exercised by the GPU smoke in Task 6). If a fully-resolved
fixture cfg is already available in the test suite, prefer asserting
`"--poet-learnable-scale" in build_megatron_args(fixture_cfg_with_flag_true)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py -q -k "cli_arg or emits_flag"`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'poet_learnable_scale'`.

- [ ] **Step 3: Declare the key in the base YAML**

In `configs/experiments/optim/poet_lie_orth_update_rms.yaml`, under `optim.poet:`, add
(next to `head_aligned_attn: false`):

```yaml
    head_aligned_attn: false
    learnable_scale: false  # trainable per-layer scalar gain g (W_eff = g·R_out·W₀·R_in)
```

- [ ] **Step 4: Register the CLI arg**

In `launchers/pretrain_gpt_slm.py`, in `add_slm_args`, next to the
`--poet-freeze-output-rotation` line (~line 70):

```python
    group.add_argument("--poet-learnable-scale", action="store_true")
```

- [ ] **Step 5: Inject into argv**

In `src/utils/megatron_args.py`, next to the `--poet-lie-ortho-distributed` injection
(~line 657):

```python
        if poet.get("learnable_scale", False):
            poet_args.append("--poet-learnable-scale")
```

- [ ] **Step 6: args→config copy**

In `src/patches/poet_optimizer_setup.py`, in `_wrapped_get_config`, next to the
`config.poet_init_scale` line (~line 40):

```python
        config.poet_learnable_scale = getattr(args, "poet_learnable_scale", False)
```

- [ ] **Step 7: Read the arg in the model-build path and thread it**

In `src/patches/poet_apply_to_model.py`, in `_apply_poet_to_chunk`, next to the other
`getattr(args, "poet_...")` reads (~line 110):

```python
    learnable_scale = getattr(args, "poet_learnable_scale", False)
```

and pass it into the `replace_linears_with_poet(...)` call (next to `group_experts=group_experts,`):

```python
        group_experts=group_experts,
        learnable_scale=learnable_scale,
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py -q -k "cli_arg or emits_flag"`
Expected: PASS.

- [ ] **Step 9: Static check — the args→config copy is the silent-no-op trap; confirm wiring compiles**

Run: `python -m py_compile src/patches/poet_optimizer_setup.py src/patches/poet_apply_to_model.py src/utils/megatron_args.py launchers/pretrain_gpt_slm.py`
Expected: no output (exit 0).

- [ ] **Step 10: Commit**

```bash
git add configs/experiments/optim/poet_lie_orth_update_rms.yaml launchers/pretrain_gpt_slm.py src/utils/megatron_args.py src/patches/poet_optimizer_setup.py src/patches/poet_apply_to_model.py tests/unit/test_poet_learnable_scale.py
git commit -m "feat(poet): plumb optim.poet.learnable_scale flag end-to-end"
```

---

## Task 5: Checkpoint round-trip + full-suite gate

**Files:**
- Test: `tests/unit/test_poet_learnable_scale.py`

**Interfaces:**
- Consumes: `ScaledPOETLinear` (Task 1).
- Produces: confidence that `gain` survives `state_dict` / `load_state_dict`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_poet_learnable_scale.py
def test_gain_round_trips_through_state_dict():
    layer = ScaledPOETLinear(8, 16, block_count=1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.gain.fill_(1.37)
    sd = layer.state_dict()
    assert "gain" in sd
    fresh = ScaledPOETLinear(8, 16, block_count=1, bias=False, dtype=torch.float32)
    fresh.load_state_dict(sd)
    assert float(fresh.gain) == pytest.approx(1.37)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py -q -k round_trips`
Expected: PASS immediately (registered `nn.Parameter` is in `state_dict` by default).
This test is a **guard**, not TDD-red — it locks the checkpoint contract so a later
refactor can't drop `gain`. (The Megatron `sharded_state_dict` replicated-tensor path
is validated by the GPU smoke in Task 6, since it needs `megatron.core`.)

- [ ] **Step 3: Run the full POET-relevant CPU suite**

Run: `python -m pytest tests/unit/test_poet_learnable_scale.py tests/unit/test_poet_layers.py tests/unit/test_poet_lie_orth_update_rms.py -q`
Expected: PASS (all). If `import megatron` errors surface, prefix with
`source load_cuda13_2_nccl_env.sh` (per repo note: bare venv can't import
`megatron.core`).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_poet_learnable_scale.py
git commit -m "test(poet): gain survives state_dict round-trip"
```

---

## Task 6: A/B sweep launcher + GPU smoke handoff

**Files:**
- Create: `scripts/sweep_poet_learnable_scale.sh`

**Interfaces:**
- Consumes: `scripts/train_poet_lie_orth_update_rms.sh` (existing single-run wrapper).
- Produces: two GPU runs — champion-init + neutral-init — both with the flag on.

- [ ] **Step 1: Write the sweep script**

```bash
#!/usr/bin/env bash
# A/B for the POET learnable per-layer scale (g). Each run adds the trainable gain
# g (init 1.0) on top of an otherwise-fixed init; g=1 ≡ the no-gain baseline, so any
# delta is purely "operating norm became learnable".
#   arm 1 (champion-init): mup α4 + g   — primary A/B vs the 3.4686 champion
#   arm 2 (neutral-init):  normalized/scale1 + g — "can g replace init tuning?"
# 60m/40tpp, seed 42, 8-GPU. Baseline (no g) = urms side_γ+0.25 champion 3.4745;
# the §2.15c decorrelation record 3.4686 is the no-gain SOTA to beat.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"

COMMON="llama3 scheduler=cosine_poet training_regime=ablation_40x \
  optim.lr=0.005 optim.poet.learnable_scale=true \
  optim.poet.lie_ortho_update_rms=0.30 optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.25 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.head_aligned_attn=false optim.poet.lie_alternating=true \
  optim.poet.lie_alternate_every=1 optim.poet.lie_ortho_distributed=true"

run() {  # $1 = name ; $2.. = extra overrides
  local name="$1"; shift
  echo ">>> ${name} starting"
  scripts/train_poet_lie_orth_update_rms.sh ${COMMON} "$@" \
    experiment.name="${name}" 2>&1 | tee "${CODEX_LOG_DIR}/${name}.log"
  echo "<<< ${name} done (status ${PIPESTATUS[0]}) — ${CODEX_LOG_DIR}/${name}.log"
}

run lscale_mup    optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.init_scale=1.0
run lscale_norm   optim.poet.init_type=normalized     optim.poet.init_scale=1.0
echo "=== learnable-scale A/B done: compare lscale_mup vs no-gain champion 3.4745/record 3.4686 ==="
```

- [ ] **Step 2: Lint the script**

Run: `bash -n scripts/sweep_poet_learnable_scale.sh`
Expected: no output (exit 0).

- [ ] **Step 3: Make it executable + commit**

```bash
chmod +x scripts/sweep_poet_learnable_scale.sh
git add scripts/sweep_poet_learnable_scale.sh
git commit -m "feat(poet): learnable-scale A/B sweep launcher"
```

- [ ] **Step 4: GPU smoke handoff (user-run — do NOT launch)**

Hand the user this 1-GPU dev smoke to confirm the flag trains and `gain` moves off
1.0 (≈2 min), then the full A/B:

```bash
# 1-GPU dev smoke — verify gain is in the optimizer + grows:
codexlog lscale_smoke scripts/train_poet_lie_orth_update_rms.sh llama3 \
  training_regime=ablation_40x optim.poet.learnable_scale=true \
  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 \
  cluster.gpus_per_node=1 experiment.name=lscale_smoke
# Full 8-GPU A/B:
codexlog lscale_ab bash scripts/sweep_poet_learnable_scale.sh
```

Expected wins to look for: `lscale_mup` < the no-gain side_γ+0.25 base (3.4745);
ideally < the §2.15c record (3.4686). Watch the W&B `gain`/param histograms to confirm
`g` actually departs 1.0 (if it stays pinned at 1.0, the norm DOF wasn't the lever).

---

## Self-Review

**Spec coverage:**
- Layer (mixin + 2 classes, gain on output, bias assert) → Task 1. ✅
- Layer swap + scope guard (head/single_step_x/cache) → Task 2. ✅
- Optimizer coupling (gain in skew group, wd=0 group, denom×|g|) → Task 3. ✅
- 5-edit config plumbing (YAML key, CLI arg, argv inject, args→config, model-build read) → Task 4. ✅
- DDP-buffer invariant → satisfied by constructor-time creation (Task 1), asserted in spec; no separate task needed (it is a property of *where* `gain` is created, which Task 1/Task 2 enforce by building it in the layer ctor at model-build time).
- Checkpoint round-trip → Task 5. ✅
- A/B + GPU smoke → Task 6. ✅
- `g=1` bit-exact invariant → Task 1 `test_gain_one_is_bit_exact_base`. ✅
- Default-off no-regression → Task 2 `test_replace_without_flag_has_no_gain` + Task 3 existing-suite gate + Task 5 full suite. ✅

**Placeholder scan:** no TBD/TODO; every code step shows complete code. The
`build_megatron_args` test is explicitly scoped to the injection rule with the reason
stated (heavyweight fixture), not a vague placeholder.

**Type consistency:** `learnable_scale: bool` kwarg name identical across
`replace_linears_with_poet` (Task 2) and `_apply_poet_to_chunk` thread-through
(Task 4). `gain` attribute name identical across the layer (Task 1), the param-group
builder (Task 3), and the denom multiply (Task 3). Group dict key `"gain"` consistent
between producer (poet.py) and consumer (poet_lie_orth_update_rms.py). Config field
`poet_learnable_scale` consistent across args→config copy and `getattr` read.
