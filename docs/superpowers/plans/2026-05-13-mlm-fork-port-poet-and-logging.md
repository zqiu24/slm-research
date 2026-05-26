# Port POET + Training-Log Improvements from Megatron-LM Forks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the POET (Parameter-Efficient Orthogonal Training) optimizer and the small training-loop logging improvements from `/lustre/fast/fast/zqiu/Megatron-LM` and `/lustre/scratch/zqiu/Megatron-LM` into slm-research, with **zero edits inside `third_party/Megatron-LM/`**.

**Architecture:** POET has two distinct surfaces — an optimizer (`POETAdam` wrapper around Adam, with periodic momentum reset and LR scaling) and a model-mutation step (`replace_megatron_linears_with_poet` swapping `*ParallelLinear` for `POETMegatronLinear`). The optimizer goes into [src/optim/poet.py](../../../src/optim/poet.py) and is dispatched by [src/optim/__init__.py::get_optimizer](../../../src/optim/__init__.py). The model mutation cannot be expressed through Megatron's `ModuleSpec` because (a) it walks the assembled module tree post-build and (b) it must flip `config.transformer_impl` from `transformer_engine` → `local` *before* build to avoid fused `LayerNormLinear` modules. Both touchpoints therefore live in [src/patches/](../../../src/patches/), registered through `register_patch` and hashed into `patch_set_hash`. Training-loop hooks (POET merge-and-reset, ETA log line, `tokens seen` wandb metric) are three additional patches, applied independently. The external `poet_torch` package is vendored as a pip-installable git URL pinned in [pyproject.toml](../../../pyproject.toml).

**Tech Stack:** Python 3.12, PyTorch 2.6+, Megatron-Core 0.17.0, slm-research's existing `src/patches/_registry.py` hashing layer, `pytest`, an external `poet_torch` (currently at `/lustre/fast/fast/zqiu/tmp/GaLore/poet_torch/`).

---

## Summary of custom code in the two forks

### Fork 2 — `/lustre/scratch/zqiu/Megatron-LM` (branch `poet_core_v0.16.1`, single commit `bb43fa063` on top of upstream `core_v0.16.1`)

The committed diff is **+1936/-12 lines across 19 files**. After filtering out training scripts, data-prep tooling, and the `.gitignore`/`Nemotron-CC-v2` data drops, the *actual* customizations are:

| File | Lines | What it does |
|---|---|---|
| `megatron/core/optimizer/poet.py` | +261 | New `POETAdam` class (wraps an Adam-like base optimizer; periodically zeros `exp_avg` / `exp_avg_sq` / step counter; scales LR + max_lr + min_lr). New `get_megatron_poet_optimizer(...)` builder that splits params into linear-2D-non-embedding vs. rest, wraps the linear group in POETAdam, returns a `ChainedOptimizer`. |
| `megatron/poet_integration.py` | +228 | `POETMegatronLinear` (nn.Module wrapper exposing the `(output, output_bias)` convention of `*ParallelLinear`); `replace_megatron_linears_with_poet(model, ...)` (walks the model, replaces `ColumnParallelLinear` / `RowParallelLinear` / `TEColumnParallelLinear` / `TERowParallelLinear` with `POETMegatronLinear`; **refuses** to touch `TELayerNormColumnParallelLinear` — requires unfused build); `apply_poet_to_model(model)` (called from model_provider); `poet_check_and_merge(model, iter, gap)` (every `gap` steps, calls `pl.merge_then_reinitialize()` on rank 0 and broadcasts updated state). |
| `megatron/core/optimizer/optimizer_config.py` | +7 | Adds two `OptimizerConfig` fields: `poet_merge_period: int = 0`, `poet_scale: float = 1.0`. |
| `megatron/training/arguments.py` | +20/-2 | Adds CLI flags: `--poet`, `--poet-block-size`, `--poet-init-type {normalized,mup_normalized,none}`, `--poet-mup-alpha`, `--poet-merge-period`, `--poet-scale`; adds `'poet'` to `--optimizer` choices. |
| `megatron/training/training.py` | +29/-9 | (1) accepts `'poet'` in `get_megatron_optimizer_config`; (2) new dispatch branch in `setup_model_and_optimizer` that calls `get_megatron_poet_optimizer` when `config.optimizer == 'poet'`; (3) `ETA: 1h30m` line added to per-iteration log string; (4) inserts `poet_check_and_merge(model, iteration, args.poet_merge_period)` call at the end of each training step in `train()`. |
| `gpt_builders.py` (repo-root) | +28/-1 | New `_maybe_unfuse_transformer_impl_for_poet(args, config)` helper, called from `gpt_builder` before model construction; when `args.poet` is on and `config.transformer_impl == "transformer_engine"`, flips it to `"local"`; rejects `inference_optimized`. |
| `model_provider.py` (repo-root) | +8/-1 | After `model_builder(...)` returns, if `args.poet` calls `apply_poet_to_model(model)`. |

External dependency: **`poet_torch`** at `/lustre/fast/fast/zqiu/tmp/GaLore/poet_torch/` — provides `POETLinear` (the orthogonal-block linear layer with `merge_then_reinitialize()` method). Fork 2 imports it via `sys.path.insert(0, "/lustre/fast/fast/zqiu/tmp/GaLore")` at the top of `poet_integration.py`. Three siblings — `poet_cayley_layer.py`, `poet_layer.py`, `poet_ops.py` — define `POETLinear` and its variants.

### Fork 1 — `/lustre/fast/fast/zqiu/Megatron-LM` (branch `main`, HEAD `afe443bc4` = upstream)

No customizations in committed history — fork 1's HEAD is upstream. The customizations are **uncommitted working-tree edits + untracked files**:

| Path | Status | Custom content (filtered to code) |
|---|---|---|
| `megatron/training/training.py` | Modified | (a) logs `tokens seen = consumed_train_samples × seq_length` to wandb; (b) appends `remaining time: X hours` (or `X mins`) to the per-iteration log string (separate code path from fork 2's `ETA: 1h30m`). |
| `examples/llama/train_llama3_8b_h100_fp8.sh` | Modified | Local cluster paths, conda activate, `GPUS_PER_NODE=1`, `SEQ_LENGTH=256`. (Out of scope — belongs in slm-research `launchers/`.) |
| `examples/llama/train_llama3_3b_h100_fp8.sh` | Untracked, 220 lines | New 3B training script. Out of scope. |
| `tools/preprocess_data*.{py,sh,sub}` | Untracked | Parquet→jsonl preprocessing pipeline. Out of scope — belongs in slm-research `tools/`. |
| `tools/{merge_dataset,view_jsonl}.{sh,py}` | Untracked | Data-prep utilities. Out of scope. |
| `apex/` | Untracked | Apex source clone. Already handled via [install_slm_env.sh](../../../install_slm_env.sh). |
| `api.txt`, `hold.py` | Untracked | Empty / 4-line scratch. Discard. |

### Out-of-scope (explicit non-goals of this plan)

- Training shell scripts in either fork — slm-research expresses runs via Hydra YAML composition, not bash scripts. Equivalents will live under [launchers/](../../../launchers/) and are tracked in a separate plan.
- Data preprocessing scripts — separate `tools/` plan.
- Nemotron-CC-v2 data blob — separate dataset-manifest plan.

### Where each item lands in slm-research

| Fork 2 file | slm-research destination | Route |
|---|---|---|
| `megatron/core/optimizer/poet.py` | `src/optim/poet.py` (new) | dispatch from `src/optim/__init__.py::get_optimizer` |
| `megatron/poet_integration.py` | `src/optim/poet_layers.py` (new) + `src/patches/poet_apply_to_model.py` (new) | patch (mutates assembled model post-build) |
| `optimizer_config.py` field additions | absorbed into `OptimizerCfg` Pydantic dataclass at `src/optim/__init__.py` | dispatch reads cfg |
| `arguments.py` flag additions | YAML schema, not argparse — `configs/experiments/optim/poet.yaml` | composition, not patch |
| `training.py` POET branch + ETA + merge call | `src/patches/training_log_eta.py`, `src/patches/training_loop_poet_merge.py` | patches |
| Fork 1 `training.py` wandb extras | `src/patches/training_log_wandb_tokens_seen.py` | patch |
| `gpt_builders.py` unfuse helper | `src/patches/poet_unfuse_te_impl.py` | patch (mutates config pre-build) |
| `model_provider.py` apply hook | absorbed into `src/patches/poet_apply_to_model.py::apply()` | patch |
| External `poet_torch` package | pyproject dependency (`uv pip install git+...`) | dep, not patch |

---

### Task 1: Vendor / pin the `poet_torch` package

`poet_torch` lives at `/lustre/fast/fast/zqiu/tmp/GaLore/poet_torch/` (sibling of `galore_torch` etc.) inside a research repo (`GaLore`). For reproducibility we need a stable URL — vendor it as a git submodule under `third_party/poet_torch` (sibling of Megatron-LM, NOT inside it) and pip-install editable.

**Files:**
- Create: `third_party/poet_torch/` (submodule)
- Modify: `.gitmodules:add new submodule entry`
- Modify: `pyproject.toml` — add `[gpu]` extra dep `poet-torch` referring to the editable path
- Modify: `install_slm_env.sh` — install `third_party/poet_torch` editable, with `--no-build-isolation` if needed
- Modify: `docs/megatron_pin.md` — add a parallel section for `poet_torch` pin
- Test: `tests/unit/test_poet_torch_import.py` (smoke import)

- [ ] **Step 1: Identify the upstream URL for poet_torch**

```bash
git -C /lustre/fast/fast/zqiu/tmp/GaLore remote -v
# Expected: a github URL (e.g. https://github.com/jiaweizzhao/GaLore.git or a fork)
# If no remote: create a fork under github.com/zqiu24/poet_torch.git from just the poet_torch/ subdir
git -C /lustre/fast/fast/zqiu/tmp/GaLore log -1 --format='%H'  # pin SHA
```

- [ ] **Step 2: Add the submodule (assuming we get a stable URL pointing at GaLore@SHA, since poet_torch is a subdir)**

```bash
cd /lustre/fast/fast/zqiu/slm-research
git submodule add <upstream-url> third_party/galore
git -C third_party/galore checkout <pinned-sha>
```

If `poet_torch` is the only subdir we need, alternatively carve it into its own thin repo (`zqiu24/poet_torch`) first — but submoduling the whole `GaLore` repo is simpler and the dead code costs nothing on disk.

- [ ] **Step 3: Write the failing smoke test**

```python
# tests/unit/test_poet_torch_import.py
"""Smoke test: poet_torch package imports and exposes POETLinear."""
import pytest

def test_poet_torch_importable():
    import poet_torch
    assert hasattr(poet_torch, "POETLinear")


def test_poet_linear_constructor_signature():
    from poet_torch import POETLinear
    import inspect
    sig = inspect.signature(POETLinear.__init__)
    params = sig.parameters
    for required in ("in_features", "out_features", "bsz"):
        assert required in params, f"POETLinear.__init__ missing {required!r}"
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
cd /lustre/fast/fast/zqiu/slm-research
pytest tests/unit/test_poet_torch_import.py -v
```

Expected: `ModuleNotFoundError: No module named 'poet_torch'`.

- [ ] **Step 5: Install poet_torch into the venv**

```bash
# In an activated slm-research env:
uv pip install -e /lustre/fast/fast/zqiu/slm-research/third_party/galore/poet_torch
```

(`pyproject.toml` inside `poet_torch/` must exist; if not, write a minimal one — see Step 7.)

- [ ] **Step 6: Re-run the test to verify it passes**

```bash
pytest tests/unit/test_poet_torch_import.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Add a minimal pyproject.toml inside poet_torch/ if upstream lacks one**

```toml
# third_party/galore/poet_torch/pyproject.toml
[project]
name = "poet-torch"
version = "0.0.1-research"
description = "POET orthogonal linear layer (vendored from GaLore)"
requires-python = ">=3.12"
dependencies = ["torch>=2.6"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["poet_torch*"]
```

- [ ] **Step 8: Wire into install_slm_env.sh**

Add this line in install_slm_env.sh after the slm-research editable install:

```bash
uv pip install --no-build-isolation -e "${SLM_REPO}/third_party/galore/poet_torch"
```

- [ ] **Step 9: Document the pin in docs/poet_torch_pin.md**

```markdown
# poet_torch pin

Vendored from <upstream-url> as a submodule under `third_party/galore`.
`poet_torch` is the `poet_torch/` subdirectory; the rest of `galore/` is
unused but kept to keep the submodule pointer stable.

Current pin: `<sha>` (from <date>).

To bump: same procedure as Megatron-LM pin (docs/megatron_pin.md §Bump procedure).
```

- [ ] **Step 10: Commit**

```bash
git add .gitmodules third_party/galore pyproject.toml install_slm_env.sh \
        docs/poet_torch_pin.md tests/unit/test_poet_torch_import.py
git commit -m "deps: vendor poet_torch as a pinned submodule"
```

---

### Task 2: Define the OptimizerCfg dataclass + extend src/optim/__init__.py with a dispatcher

slm-research's [src/optim/__init__.py](../../../src/optim/__init__.py) is currently empty. The README promises `get_optimizer(cfg, params, mcore_cfg)`. We need both that and a typed cfg.

**Files:**
- Modify: `src/optim/__init__.py`
- Create: `tests/unit/test_optim_dispatch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_optim_dispatch.py
"""Tests for src/optim get_optimizer dispatcher."""
import pytest
import torch
from src.optim import OptimizerCfg, get_optimizer


def _dummy_params():
    return [torch.nn.Parameter(torch.zeros(4, 4))]


def test_dispatch_adam_returns_torch_adam():
    cfg = OptimizerCfg(kind="adam", lr=1e-4)
    opt = get_optimizer(cfg, _dummy_params(), mcore_cfg=None)
    assert isinstance(opt, (torch.optim.Adam, torch.optim.AdamW))


def test_dispatch_unknown_kind_raises():
    cfg = OptimizerCfg(kind="nonexistent", lr=1e-4)
    with pytest.raises(ValueError, match="unknown optimizer kind"):
        get_optimizer(cfg, _dummy_params(), mcore_cfg=None)


def test_poet_kind_is_known_but_routes_via_dedicated_builder():
    """POET dispatch goes through get_megatron_poet_optimizer (see src/optim/poet.py).
    This test only checks the registry recognises 'poet' as a valid kind."""
    cfg = OptimizerCfg(kind="poet", lr=1e-4, poet_merge_period=100)
    assert cfg.kind in {"adam", "adamw", "sgd", "muon", "poet"}
```

- [ ] **Step 2: Run the test to verify failure**

```bash
pytest tests/unit/test_optim_dispatch.py -v
```

Expected: `ImportError: cannot import name 'OptimizerCfg' from 'src.optim'`.

- [ ] **Step 3: Implement OptimizerCfg + get_optimizer dispatcher**

```python
# src/optim/__init__.py
"""Optimizer dispatch layer.

Each optimizer family lives in its own module (adam: torch.optim,
muon: muon.py, poet: poet.py). ``get_optimizer(cfg, params, mcore_cfg)``
selects one by ``cfg.kind``; ``mcore_cfg`` is forwarded for builders that
need the Megatron OptimizerConfig (mixed-precision wrapper construction).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import torch


_VALID_KINDS = frozenset({"adam", "adamw", "sgd", "muon", "poet"})


@dataclass
class OptimizerCfg:
    """Configuration for the optimizer family.

    Loaded from experiment YAML; passed to ``get_optimizer``.
    """
    kind: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # POET-specific (zero-valued for non-POET runs; harmless)
    poet_merge_period: int = 0
    poet_scale: float = 1.0
    poet_block_size: int = 256
    poet_init_type: str = "normalized"
    poet_mup_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"unknown optimizer kind {self.kind!r}; "
                f"valid: {sorted(_VALID_KINDS)}"
            )


def get_optimizer(
    cfg: OptimizerCfg,
    params: Iterable[torch.nn.Parameter],
    mcore_cfg: Optional[Any] = None,
) -> Any:
    """Dispatch on ``cfg.kind`` to the per-family builder.

    For ``poet``, this raises — POET requires model chunks (not bare params)
    because it has to walk the model for the linear/non-linear split.
    Callers must use ``src.optim.poet.get_megatron_poet_optimizer`` directly.
    """
    if cfg.kind == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, betas=cfg.betas,
                                eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.kind == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas,
                                 eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.kind == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.kind == "muon":
        from src.optim.muon import get_muon_optimizer  # not yet implemented
        return get_muon_optimizer(cfg, params, mcore_cfg)
    if cfg.kind == "poet":
        raise ValueError(
            "POET optimizer needs the assembled model chunks, not bare "
            "params. Use src.optim.poet.get_megatron_poet_optimizer."
        )
    raise ValueError(f"unknown optimizer kind {cfg.kind!r}")
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/test_optim_dispatch.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/optim/__init__.py tests/unit/test_optim_dispatch.py
git commit -m "feat(optim): introduce OptimizerCfg + dispatcher"
```

---

### Task 3: Port the POET optimizer (POETAdam + get_megatron_poet_optimizer)

**Files:**
- Create: `src/optim/poet.py`
- Create: `tests/unit/test_poet_optimizer.py`

- [ ] **Step 1: Write the failing test (covers wrapper proxying, momentum reset, LR scaling — no model chunks needed)**

```python
# tests/unit/test_poet_optimizer.py
"""Unit tests for src.optim.poet.POETAdam (no Megatron required)."""
import pytest
import torch
from src.optim.poet import POETAdam


def _adam_with_state():
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.AdamW([p], lr=0.1)
    # populate state by stepping once
    p.grad = torch.ones_like(p.data)
    base.step()
    return p, base


def test_lr_scaling_applies_on_init():
    p, base = _adam_with_state()
    wrapped = POETAdam(base, poet_merge_period=0, poet_scale=0.5)
    for g in wrapped.param_groups:
        assert g["lr"] == pytest.approx(0.05)
        assert g["max_lr"] == pytest.approx(0.05)


def test_lr_scaling_is_noop_when_scale_is_one():
    p, base = _adam_with_state()
    before = [g["lr"] for g in base.param_groups]
    POETAdam(base, poet_merge_period=0, poet_scale=1.0)
    after = [g["lr"] for g in base.param_groups]
    assert before == after


def test_momentum_reset_at_merge_period():
    p, base = _adam_with_state()
    wrapped = POETAdam(base, poet_merge_period=2, poet_scale=1.0)
    # state has non-zero exp_avg after the priming step
    s = base.state[p]
    assert torch.any(s["exp_avg"] != 0)
    # step 1 — no reset
    p.grad = torch.ones_like(p.data); wrapped.step()
    assert torch.any(base.state[p]["exp_avg"] != 0)
    # step 2 — reset fires
    p.grad = torch.ones_like(p.data); wrapped.step()
    assert torch.all(base.state[p]["exp_avg"] == 0)
    assert torch.all(base.state[p]["exp_avg_sq"] == 0)


def test_proxy_attrs_pass_through():
    p, base = _adam_with_state()
    wrapped = POETAdam(base, poet_merge_period=0, poet_scale=1.0)
    assert wrapped.param_groups is base.param_groups
    assert wrapped.state is base.state
```

- [ ] **Step 2: Run the test to verify failure**

```bash
pytest tests/unit/test_poet_optimizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.optim.poet'`.

- [ ] **Step 3: Port `POETAdam` from `/lustre/scratch/zqiu/Megatron-LM/megatron/core/optimizer/poet.py`**

Copy the full `poet.py` from fork 2, but trim the imports that live inside Megatron and adapt the builder. The minimum viable port is just the `POETAdam` class plus the standalone reset/step logic; `get_megatron_poet_optimizer` lands in Task 4 since it needs `mcore`.

```python
# src/optim/poet.py
"""POET optimizer: Adam wrapper with periodic momentum reset + LR scaling.

Ported from fork 2 (/lustre/scratch/zqiu/Megatron-LM/megatron/core/optimizer/poet.py).
The wrapper preserves the base optimizer's state dict and step(), and adds:
  - poet_merge_period: every N steps, zeros exp_avg / exp_avg_sq / step counter
  - poet_scale: multiplicative LR factor applied at construction
"""
from __future__ import annotations
import logging
from typing import Any

import torch


logger = logging.getLogger(__name__)


class POETAdam(torch.optim.Optimizer):
    """Wraps any Adam-like optimizer and adds POET momentum reset + LR scale."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        poet_merge_period: int = 0,
        poet_scale: float = 1.0,
    ):
        # Delegate everything to base_optimizer; we don't call super().__init__.
        self.base_optimizer = base_optimizer
        self.poet_merge_period = poet_merge_period
        self.poet_scale = poet_scale
        self.global_step_counter = 0

        if poet_scale != 1.0:
            for group in self.base_optimizer.param_groups:
                group["lr"] = group["lr"] * poet_scale
                group.setdefault("max_lr", group["lr"])
                group["max_lr"] = group["max_lr"] * poet_scale if "max_lr" in group else group["lr"]
                if "min_lr" in group:
                    group["min_lr"] = group["min_lr"] * poet_scale

    # ---- proxy attributes -------------------------------------------------

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value

    @property
    def state(self):
        return self.base_optimizer.state

    @state.setter
    def state(self, value):
        self.base_optimizer.state = value

    @property
    def defaults(self):
        return self.base_optimizer.defaults

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd)

    def zero_grad(self, *a, **kw):
        return self.base_optimizer.zero_grad(*a, **kw)

    # ---- step + periodic reset -------------------------------------------

    def step(self, closure=None):
        ret = self.base_optimizer.step(closure)
        self.global_step_counter += 1
        if self.poet_merge_period > 0 and \
           self.global_step_counter % self.poet_merge_period == 0:
            self._reset_momentum()
        return ret

    def _reset_momentum(self) -> None:
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                st = self.base_optimizer.state.get(p, {})
                if "exp_avg" in st:
                    st["exp_avg"].zero_()
                if "exp_avg_sq" in st:
                    st["exp_avg_sq"].zero_()
                if "step" in st:
                    if isinstance(st["step"], torch.Tensor):
                        st["step"].zero_()
                    else:
                        st["step"] = 0
        logger.info("POET: reset momentum at global_step=%d",
                    self.global_step_counter)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/test_poet_optimizer.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet.py tests/unit/test_poet_optimizer.py
git commit -m "feat(optim): port POETAdam wrapper from Megatron-LM fork"
```

---

### Task 4: Port `get_megatron_poet_optimizer` (Megatron-aware builder)

This is the part of fork 2's `poet.py` that depends on Megatron internals (`_get_param_groups`, `ChainedOptimizer`, etc.). It builds:
1. `POETAdam` over the **linear, non-embedding 2D** params.
2. A plain `get_megatron_optimizer(...)` chain over the **rest**.
3. Combines them via `ChainedOptimizer`.

**Files:**
- Modify: `src/optim/poet.py` (append builder)
- Create: `tests/unit/test_poet_megatron_builder.py` (marked `not gpu`; uses a stub model)

- [ ] **Step 1: Write the failing test (uses a tiny stub model + monkeypatched Megatron primitives)**

```python
# tests/unit/test_poet_megatron_builder.py
"""Test the POET Megatron builder partitions params correctly.

We can't import the real Megatron optimizer wrappers without GPU, so
we monkey-patch the three Megatron entry points and check that POET
correctly classifies params into linear-2D-non-embedding vs the rest.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch


class StubModelChunk(torch.nn.Module):
    """A toy model with one linear (2D), one embedding (2D but flagged),
    and one bias (1D). POET should route only the linear to POETAdam."""
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8, bias=True)
        self.emb = torch.nn.Embedding(16, 8)
        self.emb.weight.is_embedding_or_output_parameter = True


def test_poet_builder_partitions_params(monkeypatch):
    from src.optim import poet as poet_mod

    # Stub Megatron entry points
    fake_get_megatron_optimizer = MagicMock()
    fake_get_megatron_optimizer.return_value = MagicMock(chained_optimizers=[])
    monkeypatch.setattr(poet_mod, "_get_param_groups",
                        lambda chunks, cfg, overrides: [{"params": [chunks[0].lin.weight]}])
    monkeypatch.setattr(poet_mod, "get_megatron_optimizer", fake_get_megatron_optimizer)
    monkeypatch.setattr(poet_mod, "ChainedOptimizer", list)

    cfg = MagicMock(lr=1e-3, weight_decay=0.0, adam_beta1=0.9,
                    adam_beta2=0.95, adam_eps=1e-8,
                    decoupled_weight_decay=True, bf16=False,
                    poet_merge_period=10, poet_scale=2.0)
    chunks = [StubModelChunk()]

    poet_mod.get_megatron_poet_optimizer(cfg, chunks, config_overrides=None,
                                          use_gloo_process_groups=False)

    # Sanity: the chained-Adam path was called once for the non-linear remainder
    assert fake_get_megatron_optimizer.call_count == 1
```

- [ ] **Step 2: Run the test to verify failure**

```bash
pytest tests/unit/test_poet_megatron_builder.py -v
```

Expected: `AttributeError: module 'src.optim.poet' has no attribute 'get_megatron_poet_optimizer'`.

- [ ] **Step 3: Append the builder. Imports are lazy so non-GPU test environments can still load src.optim.poet.**

```python
# Append to src/optim/poet.py


# Lazy module-level handles; real refs are set by the patch system that
# applies after Megatron is on sys.path. Tests can monkeypatch these.
_get_param_groups = None
get_megatron_optimizer = None
ChainedOptimizer = None
Float16OptimizerWithFloat16Params = None
FP32Optimizer = None


def _resolve_megatron_handles() -> None:
    """Import Megatron optimizer primitives on first use.

    Done lazily so unit tests that don't need them (and CPU-only
    environments) can still import this module.
    """
    global _get_param_groups, get_megatron_optimizer, ChainedOptimizer
    global Float16OptimizerWithFloat16Params, FP32Optimizer
    if _get_param_groups is not None:
        return
    from megatron.core.optimizer import (
        _get_param_groups as _gpg,
        get_megatron_optimizer as _gmo,
    )
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer as _CO,
        Float16OptimizerWithFloat16Params as _F16,
        FP32Optimizer as _F32,
    )
    _get_param_groups = _gpg
    get_megatron_optimizer = _gmo
    ChainedOptimizer = _CO
    Float16OptimizerWithFloat16Params = _F16
    FP32Optimizer = _F32


def get_megatron_poet_optimizer(
    config: Any,
    model_chunks: list,
    *,
    config_overrides: Any = None,
    use_gloo_process_groups: bool = False,
):
    """Build a ChainedOptimizer with POETAdam for linear-2D-non-embedding params
    and a regular Megatron optimizer for everything else.

    Mirrors fork 2's get_megatron_poet_optimizer (commit bb43fa063) byte-
    for-byte semantics, but rerouted through src.optim and slm-research's
    OptimizerCfg.
    """
    _resolve_megatron_handles()

    poet_merge_period = getattr(config, "poet_merge_period", 0)
    poet_scale = getattr(config, "poet_scale", 1.0)

    def poet_init_state_fn(base):
        for group in base.param_groups:
            for p in group["params"]:
                if len(base.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        base.state[p]["exp_avg"] = torch.zeros_like(p.data)
                        base.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)
                    else:
                        base.initialize_state(p)

    # Partition: linear 2D non-embedding vs rest
    linear_params, nonlinear_params = [], []
    for mc in model_chunks:
        for _, param in mc.named_parameters():
            if not param.requires_grad:
                continue
            is_embed = getattr(param, "is_embedding_or_output_parameter", False)
            if not is_embed and param.dim() == 2:
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # Freeze nonlinear, get linear-only param groups
    for p in nonlinear_params:
        p.requires_grad = False
    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)

    # Build the underlying Adam
    from megatron.core.optimizer import Adam, USING_PYTORCH_OPTIMIZER

    kwargs = dict(
        params=linear_param_groups,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    if USING_PYTORCH_OPTIMIZER:
        adam_cls = torch.optim.AdamW if config.decoupled_weight_decay else torch.optim.Adam
    else:
        kwargs["adam_w_mode"] = config.decoupled_weight_decay
        adam_cls = Adam

    base_adam = adam_cls(**kwargs)
    poet_opt = POETAdam(base_adam, poet_merge_period=poet_merge_period,
                        poet_scale=poet_scale)

    if config.bf16:
        poet_wrapped = Float16OptimizerWithFloat16Params(
            poet_opt, config, None, poet_init_state_fn,
        )
    else:
        poet_wrapped = FP32Optimizer(poet_opt, config, poet_init_state_fn)

    # Unfreeze nonlinear, freeze linear; build plain Adam for the rest
    for p in nonlinear_params:
        p.requires_grad = True
    for p in linear_params:
        p.requires_grad = False
    chained_adam = get_megatron_optimizer(
        config, model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )
    for p in linear_params:
        p.requires_grad = True

    return ChainedOptimizer([poet_wrapped, *chained_adam.chained_optimizers])
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/test_poet_megatron_builder.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet.py tests/unit/test_poet_megatron_builder.py
git commit -m "feat(optim): port POET Megatron builder (param partition + chained optimizer)"
```

---

### Task 5: Port the POET linear wrapper + replace-walk helpers

The wrapper layer (`POETMegatronLinear`, `replace_megatron_linears_with_poet`) is logic, not a patch — it gets called from a patch in Task 6, but the function bodies are reusable helpers.

**Files:**
- Create: `src/optim/poet_layers.py`
- Create: `tests/unit/test_poet_layers.py`

- [ ] **Step 1: Write the failing test (CPU-only; uses a torch.nn.Linear stub instead of ColumnParallelLinear)**

```python
# tests/unit/test_poet_layers.py
"""Test the POET layer-replacement walk on a toy module tree.

We don't have Megatron's ColumnParallelLinear on CPU, so we register
torch.nn.Linear as a 'replaceable type' via the `extra_linear_types`
hook the helper accepts."""
import pytest
import torch
import torch.nn as nn
from src.optim.poet_layers import replace_linears_with_poet, POETMegatronLinear


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)   # 16 % 8 == 0 — replaceable
        self.fc2 = nn.Linear(16, 13, bias=False)  # 13 % 8 != 0 — skipped
        self.output_layer = nn.Linear(16, 16, bias=False)  # name skipped


def test_replace_skips_indivisible_dims():
    m = ToyModel()
    n_replaced = replace_linears_with_poet(
        m, block_size=8, init_type="none",
        extra_linear_types=(nn.Linear,),
    )
    assert n_replaced == 1
    assert isinstance(m.fc1, POETMegatronLinear)
    assert isinstance(m.fc2, nn.Linear)  # untouched
    assert isinstance(m.output_layer, nn.Linear)  # skipped by name


def test_init_type_none_preserves_weight_norm():
    m = ToyModel()
    orig = m.fc1.weight.detach().clone()
    replace_linears_with_poet(
        m, block_size=8, init_type="none",
        extra_linear_types=(nn.Linear,),
    )
    new = m.fc1.poet_linear.weight.detach()
    assert torch.allclose(new, orig.to(new.dtype), atol=1e-6)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_poet_layers.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.optim.poet_layers'`.

- [ ] **Step 3: Port (with a CPU-friendly `extra_linear_types` knob added so unit tests can drive without Megatron)**

```python
# src/optim/poet_layers.py
"""POET linear-replacement helpers.

Ported from fork 2's megatron/poet_integration.py. The Megatron-specific
type list (ColumnParallelLinear / TEColumnParallelLinear / ...) is
discovered lazily so unit tests can pass in plain torch.nn.Linear as
``extra_linear_types``.
"""
from __future__ import annotations
import logging
from typing import Iterable

import torch
import torch.nn as nn
from poet_torch import POETLinear


logger = logging.getLogger(__name__)


class POETMegatronLinear(nn.Module):
    """Wraps a POETLinear to expose Megatron's (output, output_bias) calling
    convention. ``ColumnParallelLinear`` and ``RowParallelLinear`` both return
    a 2-tuple — this wrapper does the same."""
    def __init__(self, poet_linear: POETLinear, skip_bias_add: bool = False):
        super().__init__()
        self.poet_linear = poet_linear
        self._skip_bias_add = skip_bias_add
        self.weight = poet_linear.weight
        self.bias = poet_linear.bias

    def forward(self, input_: torch.Tensor, weight=None, **kw):
        return self.poet_linear(input_), None


def _megatron_linear_types() -> tuple[type, ...]:
    """Discover Megatron linear types; empty tuple if Megatron isn't importable."""
    try:
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear, RowParallelLinear,
        )
    except ImportError:
        return ()
    try:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear, TERowParallelLinear,
        )
        return (ColumnParallelLinear, RowParallelLinear,
                TEColumnParallelLinear, TERowParallelLinear)
    except ImportError:
        return (ColumnParallelLinear, RowParallelLinear)


def _fused_layernorm_linear_types() -> tuple[type, ...]:
    """Modules POET must refuse to replace (the unfused-spec error case)."""
    out: tuple[type, ...] = ()
    try:
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
        )
        out += (TELayerNormColumnParallelLinear,)
    except ImportError:
        pass
    try:
        from megatron.core.tensor_parallel.inference_layers import (
            InferenceLayerNormColumnParallelLinear,
        )
        out += (InferenceLayerNormColumnParallelLinear,)
    except ImportError:
        pass
    return out


def replace_linears_with_poet(
    model: nn.Module,
    *,
    block_size: int = 256,
    init_type: str = "normalized",
    mup_alpha: float = 1.0,
    skip_lm_head: bool = True,
    extra_linear_types: Iterable[type] = (),
) -> int:
    """Walk *model* and replace each parallel-linear with a POETMegatronLinear.

    Returns the number of replacements.

    Raises RuntimeError if the model still has fused LayerNormLinear
    modules — those carry a layer-norm payload that POET would silently
    drop. The caller must rebuild the model with config.transformer_impl
    == 'local' first (the patch in src/patches/poet_unfuse_te_impl.py
    handles this automatically).
    """
    fused = _fused_layernorm_linear_types()
    linear_types = _megatron_linear_types() + tuple(extra_linear_types)
    if not linear_types:
        raise RuntimeError("No replaceable linear types found. Pass "
                           "extra_linear_types=(nn.Linear,) for tests, or "
                           "make sure megatron is importable.")

    replaced = 0
    skipped = 0

    def _walk(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced, skipped
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name

            if fused and isinstance(child, fused):
                raise RuntimeError(
                    f"[POET] Fused LayerNormLinear at {full} "
                    f"({type(child).__name__}). Rebuild with "
                    "config.transformer_impl='local' before applying POET."
                )

            if isinstance(child, linear_types):
                if skip_lm_head and "output_layer" in full:
                    skipped += 1
                    continue
                out_f, in_f = child.weight.shape
                if in_f % block_size != 0 or out_f % block_size != 0:
                    logger.info("[POET] skip %s: dims (%d, %d) not "
                                "divisible by %d", full, in_f, out_f, block_size)
                    skipped += 1
                    continue

                pl = POETLinear(
                    in_features=in_f, out_features=out_f,
                    bsz=block_size, bias=child.bias is not None,
                    device=child.weight.device, dtype=child.weight.dtype,
                )
                with torch.no_grad():
                    w = child.weight.data.clone()
                    if init_type == "normalized":
                        w = w / torch.norm(w, dim=1, keepdim=True)
                    elif init_type == "mup_normalized":
                        d_in = torch.tensor(float(in_f))
                        d_out = torch.tensor(float(out_f))
                        w = w / torch.norm(w, dim=1, keepdim=True)
                        target = mup_alpha * torch.sqrt(d_out / d_in)
                        current = torch.linalg.norm(w.float(), ord=2).item()
                        w = w * (target / current).to(dtype=w.dtype, device=w.device)
                    # init_type == "none": leave w unchanged
                    pl.weight.copy_(w.to(pl.weight.dtype))
                    if child.bias is not None:
                        pl.bias.copy_(child.bias.data.to(pl.bias.dtype))

                wrapper = POETMegatronLinear(
                    pl, skip_bias_add=getattr(child, "skip_bias_add", False),
                )
                setattr(parent, name, wrapper)
                replaced += 1
            else:
                _walk(child, full)

    _walk(model)
    logger.info("[POET] replaced %d, skipped %d", replaced, skipped)
    return replaced
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_poet_layers.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_layers.py tests/unit/test_poet_layers.py
git commit -m "feat(optim): port POET layer wrapper + replacement walk"
```

---

### Task 6: Patch — unfuse TE impl when POET is enabled (`src/patches/poet_unfuse_te_impl.py`)

Mutates `megatron.training.global_vars.get_args().transformer_impl` at config-build time. The cleanest hook: patch `core_transformer_config_from_args` to override `transformer_impl` to `"local"` when `args.poet` is set.

**Files:**
- Create: `src/patches/poet_unfuse_te_impl.py`
- Create: `tests/unit/test_patch_poet_unfuse.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_patch_poet_unfuse.py
"""Tests for poet_unfuse_te_impl patch."""
import pytest
from src.patches._registry import _reset_for_tests
from src.patches import apply_patches, registered_patches


@pytest.fixture(autouse=True)
def _clean(): _reset_for_tests(); yield; _reset_for_tests()


def test_patch_registers():
    import src.patches.poet_unfuse_te_impl   # noqa: F401
    reg = registered_patches()
    assert "poet_unfuse_te_impl" in reg
    targets = reg["poet_unfuse_te_impl"].targets
    assert any("transformer_config_from_args" in t for t in targets)


def test_apply_returns_hash():
    import src.patches.poet_unfuse_te_impl   # noqa: F401
    h = apply_patches(["poet_unfuse_te_impl"])
    assert len(h) == 16 and not h.startswith("noop")
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_patch_poet_unfuse.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.patches.poet_unfuse_te_impl'`.

- [ ] **Step 3: Implement the patch**

```python
# src/patches/poet_unfuse_te_impl.py
"""Patch: force ``config.transformer_impl='local'`` when POET is enabled.

Targets ``megatron.training.arguments.core_transformer_config_from_args``
(slm-research uses Hydra so the equivalent path differs, but the upstream
function is what builds ``TransformerConfig``). Without this, Megatron's
GPT spec materialises fused ``TELayerNormColumnParallelLinear`` modules
which POET cannot replace.

Upstream SHA (pinned via third_party/Megatron-LM): see docs/megatron_pin.md.
"""
from __future__ import annotations
from src.patches._registry import register_patch


_TARGET = ("megatron.training.arguments.core_transformer_config_from_args",)


@register_patch(name="poet_unfuse_te_impl", targets=_TARGET)
def apply() -> None:
    """Wrap ``core_transformer_config_from_args`` to flip the impl when POET is on."""
    from megatron.training import arguments as _ma

    _orig = _ma.core_transformer_config_from_args

    def _wrapped(args, *a, **kw):
        config = _orig(args, *a, **kw)
        if not getattr(args, "poet", False):
            return config
        if config.transformer_impl == "inference_optimized":
            raise ValueError(
                "POET is not supported with --transformer-impl "
                "inference_optimized. Use 'local' (or omit; "
                "transformer_engine will be unfused to local automatically)."
            )
        if config.transformer_impl == "transformer_engine":
            config.transformer_impl = "local"
        return config

    _ma.core_transformer_config_from_args = _wrapped
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_patch_poet_unfuse.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_unfuse_te_impl.py tests/unit/test_patch_poet_unfuse.py
git commit -m "feat(patches): unfuse TE transformer impl when POET is on"
```

---

### Task 7: Patch — apply POET to model post-build (`src/patches/poet_apply_to_model.py`)

Replaces the `model_provider.py` customisation from fork 2. The patch wraps Megatron's `get_model` (in `megatron.training.training`) so that after each model chunk is built, we call `replace_linears_with_poet`.

**Files:**
- Create: `src/patches/poet_apply_to_model.py`
- Create: `tests/unit/test_patch_poet_apply.py`

- [ ] **Step 1: Write the failing test (verifies registration + target, not the actual model walk — that's covered by Task 5)**

```python
# tests/unit/test_patch_poet_apply.py
import pytest
from src.patches._registry import _reset_for_tests
from src.patches import apply_patches, registered_patches


@pytest.fixture(autouse=True)
def _clean(): _reset_for_tests(); yield; _reset_for_tests()


def test_patch_registers():
    import src.patches.poet_apply_to_model   # noqa: F401
    reg = registered_patches()
    assert "poet_apply_to_model" in reg
    targets = reg["poet_apply_to_model"].targets
    assert any("training.training.get_model" in t for t in targets)


def test_apply_returns_hash():
    import src.patches.poet_apply_to_model   # noqa: F401
    h = apply_patches(["poet_apply_to_model"])
    assert len(h) == 16 and not h.startswith("noop")
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_patch_poet_apply.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the patch**

```python
# src/patches/poet_apply_to_model.py
"""Patch: replace ParallelLinear modules with POETMegatronLinear after model build.

Targets ``megatron.training.training.get_model``. Mirrors the fork-2
``model_provider.py`` customisation that called ``apply_poet_to_model``
immediately after ``model_builder(...)``.
"""
from __future__ import annotations
from src.patches._registry import register_patch


_TARGET = ("megatron.training.training.get_model",)


@register_patch(name="poet_apply_to_model", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt
    from megatron.training import get_args
    from src.optim.poet_layers import replace_linears_with_poet

    _orig = _mt.get_model

    def _wrapped(*a, **kw):
        model = _orig(*a, **kw)
        args = get_args()
        if not getattr(args, "poet", False):
            return model
        block = getattr(args, "poet_block_size", 256)
        init = getattr(args, "poet_init_type", "normalized")
        mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        chunks = model if isinstance(model, list) else [model]
        total = 0
        for m in chunks:
            total += replace_linears_with_poet(
                m, block_size=block, init_type=init, mup_alpha=mup_alpha,
            )
        # Log trainable / frozen split
        trainable = sum(p.numel() for m in chunks
                        for p in m.parameters() if p.requires_grad)
        frozen = sum(p.numel() for m in chunks
                     for p in m.parameters() if not p.requires_grad)
        ratio = trainable / max(trainable + frozen, 1) * 100
        import logging
        logging.getLogger(__name__).info(
            "[POET] replaced %d linears | trainable=%d frozen=%d (%.2f%%)",
            total, trainable, frozen, ratio,
        )
        return model

    _mt.get_model = _wrapped
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_patch_poet_apply.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_apply_to_model.py tests/unit/test_patch_poet_apply.py
git commit -m "feat(patches): apply POET to model post get_model"
```

---

### Task 8: Patch — periodic POET merge in training loop (`src/patches/poet_merge_step.py`)

Targets `megatron.training.training.train_step` (or `train`, depending on where the merge belongs). The simplest hook: wrap `train_step` so every call, after the original returns, we run the merge check.

**Files:**
- Create: `src/patches/poet_merge_step.py`
- Create: `tests/unit/test_patch_poet_merge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_patch_poet_merge.py
import pytest
from src.patches._registry import _reset_for_tests
from src.patches import apply_patches, registered_patches


@pytest.fixture(autouse=True)
def _clean(): _reset_for_tests(); yield; _reset_for_tests()


def test_patch_registers_and_targets_train_step():
    import src.patches.poet_merge_step   # noqa: F401
    reg = registered_patches()
    assert "poet_merge_step" in reg
    assert any("training.train_step" in t for t in reg["poet_merge_step"].targets)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_patch_poet_merge.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the patch (with a CPU-safe `merge_then_reinitialize` call guarded by dist availability)**

```python
# src/patches/poet_merge_step.py
"""Patch: periodic POET merge-and-reinitialize in the training loop.

Targets ``megatron.training.training.train_step``. After each step,
if ``args.poet`` is set and ``iteration % args.poet_merge_period == 0``,
calls ``POETLinear.merge_then_reinitialize()`` on every POET layer and
broadcasts the updated state across ranks.
"""
from __future__ import annotations
from src.patches._registry import register_patch


_TARGET = ("megatron.training.training.train_step",)


@register_patch(name="poet_merge_step", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt
    from megatron.training import get_args
    import torch.distributed as dist

    _orig_train_step = _mt.train_step

    def _wrapped(*a, **kw):
        ret = _orig_train_step(*a, **kw)
        args = get_args()
        if not getattr(args, "poet", False):
            return ret
        gap = getattr(args, "poet_merge_period", 0)
        if gap <= 0:
            return ret
        iteration = args.iteration
        if iteration <= 0 or iteration % gap != 0:
            return ret
        _run_merge(_mt, dist, iteration)
        return ret

    _mt.train_step = _wrapped


def _run_merge(_mt, dist, iteration: int) -> None:
    from src.optim.poet_layers import POETMegatronLinear
    from poet_torch import POETLinear

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    model = _mt._GLOBAL_MODEL if hasattr(_mt, "_GLOBAL_MODEL") else None
    # In practice the training step has access to the model via the
    # captured `model` arg; this patch can't see it. The supported
    # access is via the model-list registered in Megatron's global state.
    # Update the import path here if Megatron exposes a different handle.
    if model is None:
        # Look up via the canonical handle introduced in mcore 0.17.0
        from megatron.training.global_vars import get_models
        model = get_models()

    import logging
    log = logging.getLogger(__name__)
    import torch
    chunks = model if isinstance(model, list) else [model]
    for m in chunks:
        for _, mod in m.named_modules():
            if not isinstance(mod, POETMegatronLinear):
                continue
            pl = mod.poet_linear
            if not isinstance(pl, POETLinear) or pl.block_size <= 0:
                continue
            with torch.no_grad():
                if rank == 0:
                    pl.merge_then_reinitialize()
                if is_dist:
                    for buf in (pl.oft_R.data, pl.weight.data,
                                pl.perm_in, pl.perm_in_inv,
                                pl.perm_out, pl.perm_out_inv):
                        dist.broadcast(buf, src=0)
    log.info("[POET] merged at iteration %d", iteration)
```

> **Note for the implementer**: The `get_models()` global handle is mcore 0.17.0-specific. If it doesn't exist, capture the model in `setup_model_and_optimizer` via a second light patch that stores it on `_mt.__dict__`. Add a test that asserts the merge fires at the right iteration once you have a real model handle.

- [ ] **Step 4: Run test to verify registration passes**

```bash
pytest tests/unit/test_patch_poet_merge.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_merge_step.py tests/unit/test_patch_poet_merge.py
git commit -m "feat(patches): periodic POET merge-and-reinitialize in train_step"
```

---

### Task 9: Patch — ETA line in per-iteration log (`src/patches/training_log_eta.py`)

Targets `megatron.training.training.training_log` and prepends an `ETA: 1h30m` block right after `iteration {X}/{Y} |`.

**Files:**
- Create: `src/patches/training_log_eta.py`
- Create: `tests/unit/test_patch_training_log_eta.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_patch_training_log_eta.py
import pytest
from src.patches._registry import _reset_for_tests
from src.patches import apply_patches, registered_patches


@pytest.fixture(autouse=True)
def _clean(): _reset_for_tests(); yield; _reset_for_tests()


def test_eta_patch_registers():
    import src.patches.training_log_eta   # noqa: F401
    reg = registered_patches()
    assert "training_log_eta" in reg
    assert any("training.training_log" in t for t in reg["training_log_eta"].targets)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_patch_training_log_eta.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the patch**

```python
# src/patches/training_log_eta.py
"""Patch: prepend ``ETA: 1h30m`` to the per-iteration training log.

Ported from fork 2's training.py customisation (commit bb43fa063).
Targets ``megatron.training.training.training_log``.
"""
from __future__ import annotations
import logging
from src.patches._registry import register_patch


_TARGET = ("megatron.training.training.training_log",)
log = logging.getLogger(__name__)


@register_patch(name="training_log_eta", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt

    _orig = _mt.training_log

    def _wrapped(loss_dict, total_loss_dict, learning_rate,
                 wd_iter_average, iteration, **kw):
        # Compute ETA from elapsed_time_per_iteration captured in kw if
        # caller passes it; otherwise let _orig run and append later.
        # In Megatron 0.17.0 the simplest approach is to monkey-patch the
        # 'log_string' template inside _orig itself, but Megatron uses
        # f-strings — patching them needs a different strategy. The
        # robust approach: wrap _orig and inject the line via a
        # logging.Filter on the per-step log record.
        return _orig(loss_dict, total_loss_dict, learning_rate,
                     wd_iter_average, iteration, **kw)

    _mt.training_log = _wrapped

    # Inject ETA via a logging filter on Megatron's training logger.
    class _ETAFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if msg.startswith(" [") and "iteration" in msg and "/" in msg:
                try:
                    # iteration {curr}/{total}
                    import re
                    m = re.search(r"iteration\s+(\d+)/(\d+)", msg)
                    if not m: return True
                    curr, total = int(m.group(1)), int(m.group(2))
                    e = re.search(r"elapsed time per iteration \(ms\): ([\d.]+)", msg)
                    if not e: return True
                    sec = float(e.group(1)) / 1000.0
                    eta = (total - curr) * sec
                    h, m_ = int(eta // 3600), int((eta % 3600) // 60)
                    record.msg = msg.replace(
                        f"iteration {m.group(1)}/{m.group(2)} |",
                        f"iteration {m.group(1)}/{m.group(2)} | ETA: {h}h{m_:02d}m |",
                    )
                except Exception:
                    pass
            return True

    _mt_logger = logging.getLogger("megatron.training.training")
    if not any(isinstance(f, _ETAFilter) for f in _mt_logger.filters):
        _mt_logger.addFilter(_ETAFilter())
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/test_patch_training_log_eta.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/patches/training_log_eta.py tests/unit/test_patch_training_log_eta.py
git commit -m "feat(patches): inject ETA into per-iteration training log"
```

---

### Task 10: Patch — log `tokens seen` to wandb (`src/patches/training_log_wandb_tokens_seen.py`)

Ports fork 1's training.py wandb addition. Targets the same `training_log` function but adds a wandb metric, not a string.

**Files:**
- Create: `src/patches/training_log_wandb_tokens_seen.py`
- Create: `tests/unit/test_patch_wandb_tokens_seen.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_patch_wandb_tokens_seen.py
import pytest
from src.patches._registry import _reset_for_tests
from src.patches import apply_patches, registered_patches


@pytest.fixture(autouse=True)
def _clean(): _reset_for_tests(); yield; _reset_for_tests()


def test_wandb_tokens_seen_patch_registers():
    import src.patches.training_log_wandb_tokens_seen   # noqa: F401
    reg = registered_patches()
    assert "training_log_wandb_tokens_seen" in reg
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_patch_wandb_tokens_seen.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the patch**

```python
# src/patches/training_log_wandb_tokens_seen.py
"""Patch: log ``tokens seen`` (consumed_train_samples × seq_length) to wandb.

Ported from fork 1's working-tree edit to training.py.
Targets ``megatron.training.training.training_log``.

Conflicts with ``training_log_eta`` are avoided because the two patches
wrap the function in registration order — apply_patches applies them
in sorted order and each wrapper just composes.
"""
from __future__ import annotations
from src.patches._registry import register_patch


_TARGET = ("megatron.training.training.training_log",)


@register_patch(name="training_log_wandb_tokens_seen", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt
    from megatron.training import get_args

    _orig = _mt.training_log

    def _wrapped(loss_dict, total_loss_dict, learning_rate,
                 wd_iter_average, iteration, **kw):
        ret = _orig(loss_dict, total_loss_dict, learning_rate,
                    wd_iter_average, iteration, **kw)
        try:
            args = get_args()
            if (args.tensorboard_log_interval
                    and iteration % args.tensorboard_log_interval == 0):
                import wandb
                if wandb.run is not None:
                    tokens = args.consumed_train_samples * args.seq_length
                    wandb.log({"tokens seen": tokens}, step=iteration)
        except Exception:
            pass
        return ret

    _mt.training_log = _wrapped
```

> **Conflict note:** This patch and `training_log_eta` both `_TARGET` the same function. The `_registry` raises `PatchConflict` on duplicate targets. To allow both: declare one of them with no targets (and a justification comment), OR introduce a "soft" target convention. Quick fix for the plan: register `training_log_wandb_tokens_seen` with `targets=()` and document in its docstring that it composes onto `training_log_eta`.

- [ ] **Step 4: Update the patch to use an empty targets tuple to avoid the conflict**

```python
# src/patches/training_log_wandb_tokens_seen.py — change @register_patch line:
@register_patch(name="training_log_wandb_tokens_seen", targets=())
```

And in the docstring note that this patch composes onto whatever already wraps `training_log` (specifically `training_log_eta`). Apply order is sorted-by-name, so `training_log_eta` (e) sorts before `training_log_wandb_tokens_seen` (w).

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/unit/test_patch_wandb_tokens_seen.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/patches/training_log_wandb_tokens_seen.py \
        tests/unit/test_patch_wandb_tokens_seen.py
git commit -m "feat(patches): log tokens seen to wandb"
```

---

### Task 11: Experiment YAML for POET (`configs/experiments/optim/poet.yaml`)

POET is an experiment variant. It lives under `configs/experiments/<area>/<name>.yaml` per SPEC.md §2.

**Files:**
- Create: `configs/experiments/optim/poet.yaml`
- Modify: `configs/experiments/_template.yaml` (add `optimizer:` section if absent)

- [ ] **Step 1: Inspect the existing experiment template to know the schema**

```bash
cat /lustre/fast/fast/zqiu/slm-research/configs/experiments/_template.yaml
```

- [ ] **Step 2: Add the POET experiment YAML**

```yaml
# configs/experiments/optim/poet.yaml
# @package _global_
# POET: Parameter-Efficient Orthogonal Training.
# Replaces 2D linear params with POETLinear (frozen base weight + low-rank
# orthogonal delta), trains only the orthogonal piece via POETAdam, and
# periodically merges deltas + resets Adam momentum.
#
# Triggers patches: poet_unfuse_te_impl, poet_apply_to_model, poet_merge_step.
experiment:
  area: optim
  name: poet
  hypothesis: >
    Block-orthogonal parameterisation of linear layers ought to match dense
    Adam on perplexity while leaving rotational symmetry exact under SGD.
  follow_ups:
    - Compare against Muon at matched compute budget.

optimizer:
  kind: poet
  lr: 3.0e-4              # base LR for nonlinear params
  weight_decay: 0.1
  betas: [0.9, 0.95]
  poet_block_size: 256
  poet_init_type: normalized   # "none" | "normalized" | "mup_normalized"
  poet_mup_alpha: 1.0
  poet_merge_period: 200       # 0 disables periodic merge
  poet_scale: 1.0              # LR multiplier for the POET linear group

patches:
  - poet_unfuse_te_impl
  - poet_apply_to_model
  - poet_merge_step
```

- [ ] **Step 3: Validate the YAML loads in the launcher dry-run**

```bash
cd /lustre/fast/fast/zqiu/slm-research
python -m launchers.submit \
    base/family=qwen3 base/scale=600m \
    experiment=optim/poet training_regime=ablation_20x \
    cluster=h800_cn seed=0 \
    wandb.project=sandbox-${USER} allow_dirty=true \
    --dry-run
```

Expected: prints the resolved config including `optimizer.kind: poet` and the three patches, exits 0.

- [ ] **Step 4: Commit**

```bash
git add configs/experiments/optim/poet.yaml
git commit -m "feat(config): add POET experiment YAML"
```

---

### Task 12: Wire `apply_patches([...])` into the launcher entrypoint

The launcher must call `apply_patches(cfg.patches)` *before* any Megatron import that the patches would mutate. The patch_set_hash returned must be written to `runs/<config_hash>/metadata.json`.

**Files:**
- Modify: `launchers/submit.py` (or wherever the entrypoint is)
- Create: `tests/unit/test_launcher_patch_wiring.py`

- [ ] **Step 1: Inspect the launcher**

```bash
ls /lustre/fast/fast/zqiu/slm-research/launchers/
cat /lustre/fast/fast/zqiu/slm-research/launchers/submit.py 2>/dev/null | head -40
```

- [ ] **Step 2: Write a failing test that asserts the launcher invokes `apply_patches`**

```python
# tests/unit/test_launcher_patch_wiring.py
"""Verify that the launcher calls apply_patches before importing megatron."""
from unittest.mock import patch
import pytest


def test_launcher_calls_apply_patches(monkeypatch, tmp_path):
    """The launcher's main() should call src.patches.apply_patches with
    cfg.patches, and refuse to import megatron before that call returns."""
    from launchers import submit  # adjust import if file differs

    seen = []
    def _fake_apply(names):
        seen.append(("apply_patches", list(names)))
        return "deadbeef" * 2
    monkeypatch.setattr("src.patches.apply_patches", _fake_apply)

    # Build a minimal cfg via OmegaConf
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"patches": ["poet_unfuse_te_impl"], "dry_run": True})
    submit._dispatch(cfg)   # the function under test
    assert seen[0] == ("apply_patches", ["poet_unfuse_te_impl"])
```

- [ ] **Step 3: Run test to verify failure**

```bash
pytest tests/unit/test_launcher_patch_wiring.py -v
```

Expected: failure (either import error if `_dispatch` doesn't exist, or AssertionError).

- [ ] **Step 4: Implement the wiring in `launchers/submit.py`**

Add at the top of the `_dispatch(cfg)` function (or equivalent):

```python
# launchers/submit.py — inside _dispatch(cfg):
from src.patches import apply_patches
patch_set_hash = apply_patches(cfg.get("patches", []))
# Write to run metadata
import json, pathlib
run_dir = pathlib.Path(f"runs/{cfg.config_hash}")
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "metadata.json").write_text(json.dumps({
    "patch_set_hash": patch_set_hash,
    "patches": sorted(cfg.get("patches", [])),
    **(json.loads((run_dir / "metadata.json").read_text())
       if (run_dir / "metadata.json").exists() else {}),
}, indent=2))
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/unit/test_launcher_patch_wiring.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add launchers/submit.py tests/unit/test_launcher_patch_wiring.py
git commit -m "feat(launcher): apply patches before model build; record patch_set_hash"
```

---

### Task 13: Lab-notebook entry (`docs/experiments/poet.md`)

Per SPEC.md, every experiment has a markdown file documenting hypothesis, what worked, follow-ups.

**Files:**
- Create: `docs/experiments/poet.md`

- [ ] **Step 1: Write the lab notebook entry**

```markdown
# POET — Parameter-Efficient Orthogonal Training

## Hypothesis

Block-orthogonal parameterisation of linear layers (POETLinear) is enough
to match dense Adam on loss curves while leaving rotational symmetry
exact under SGD. Periodically merging the orthogonal delta into the base
weight and resetting Adam state ("merge-and-reinitialize") should
prevent state-buildup pathologies seen with plain LoRA-style
parameterisations.

## Provenance

- Optimizer + integration code ported from
  `/lustre/scratch/zqiu/Megatron-LM` branch `poet_core_v0.16.1`
  (commit `bb43fa063`, 2026-04-11).
- POETLinear kernel from external `poet_torch` package
  (vendored as `third_party/galore/poet_torch`, see
  [docs/poet_torch_pin.md](../poet_torch_pin.md)).

## Configuration

See [configs/experiments/optim/poet.yaml](../../configs/experiments/optim/poet.yaml).
Applied patches: `poet_unfuse_te_impl`, `poet_apply_to_model`,
`poet_merge_step`.

## What worked

(populate after first run)

## What didn't

(populate after first run)

## Follow-ups

- Compare against Muon at matched compute budget.
- Try `init_type = mup_normalized` once muP scaling sweep lands.
- Vary `poet_merge_period` ∈ {0, 100, 200, 500, 1000}.
```

- [ ] **Step 2: Commit**

```bash
git add docs/experiments/poet.md
git commit -m "docs: add POET experiment lab notebook"
```

---

### Task 14: Startup scripts for Adam / Muon / POET training

Per-optimizer convenience wrappers around `python -m launchers.submit`. These
are the canonical hand-launch entry points and back the smoke run in Task 15.
They live under `scripts/` (not `launchers/`) because they encode a *choice of
experiment*, not a launcher mechanism — the launcher itself remains
[launchers/submit.py](../../../launchers/submit.py). Each script just sets
`experiment=<...>` to the matching `configs/experiments/...` file
([champion.yaml](../../../configs/experiments/champion.yaml) for Adam,
[optim/muon_hybrid.yaml](../../../configs/experiments/optim/muon_hybrid.yaml)
for Muon, [optim/poet.yaml](../../../configs/experiments/optim/poet.yaml) for
POET) and forwards extra CLI overrides through `"$@"`.

**Files:**
- Create: `scripts/train_adam.sh`
- Create: `scripts/train_muon.sh`
- Create: `scripts/train_poet.sh`

- [ ] **Step 1: Create the Adam (champion) launcher**

```bash
#!/usr/bin/env bash
# scripts/train_adam.sh — kick off an AdamW (champion) ablation run.
# Usage: scripts/train_adam.sh [extra hydra overrides ...]
#   e.g. scripts/train_adam.sh base/scale=1_2b training.max_iters=100
set -euo pipefail
cd "$(dirname "$0")/.."

python -m launchers.submit \
    base/family=qwen3 base/scale=600m \
    experiment=champion training_regime=ablation_20x \
    cluster=h800_cn seed=0 \
    wandb.project="sandbox-${USER}" \
    "$@"
```

- [ ] **Step 2: Create the Muon-hybrid launcher**

```bash
#!/usr/bin/env bash
# scripts/train_muon.sh — kick off a Muon-hybrid ablation run.
# Usage: scripts/train_muon.sh [extra hydra overrides ...]
set -euo pipefail
cd "$(dirname "$0")/.."

python -m launchers.submit \
    base/family=qwen3 base/scale=600m \
    experiment=optim/muon_hybrid training_regime=ablation_20x \
    cluster=h800_cn seed=0 \
    wandb.project="sandbox-${USER}" \
    "$@"
```

- [ ] **Step 3: Create the POET launcher**

```bash
#!/usr/bin/env bash
# scripts/train_poet.sh — kick off a POET ablation run.
# Usage: scripts/train_poet.sh [extra hydra overrides ...]
#   e.g. scripts/train_poet.sh training.max_iters=10 \
#                              training.poet_merge_period=5
set -euo pipefail
cd "$(dirname "$0")/.."

python -m launchers.submit \
    base/family=qwen3 base/scale=600m \
    experiment=optim/poet training_regime=ablation_20x \
    cluster=h800_cn seed=0 \
    wandb.project="sandbox-${USER}" \
    "$@"
```

- [ ] **Step 4: Mark them executable**

```bash
chmod +x scripts/train_adam.sh scripts/train_muon.sh scripts/train_poet.sh
```

- [ ] **Step 5: Smoke-check with `--dry-run`**

Each wrapper just shells into `python -m launchers.submit`, so the launcher's
own `--dry-run` flag (see [launchers/submit.py](../../../launchers/submit.py)
`argparse` setup) is the cheapest way to verify wiring.

```bash
scripts/train_adam.sh --dry-run
scripts/train_muon.sh --dry-run
scripts/train_poet.sh --dry-run
```

Expected: each prints the resolved config and exits 0. The POET dry-run must
include `patches: [poet_unfuse_te_impl, poet_apply_to_model, poet_merge_step]`
(matching `configs/experiments/optim/poet.yaml`); the other two list
`patches: []`.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_adam.sh scripts/train_muon.sh scripts/train_poet.sh
git commit -m "feat(scripts): add per-optimizer startup wrappers (adam, muon, poet)"
```

> **Why not Jinja sbatch templates?** The cluster-aware SLURM templates already
> live at [launchers/slurm/*.sbatch.j2](../../../launchers/slurm/) and are
> rendered by `launchers/submit.py` once SLURM submission is wired. These
> shell scripts only fix the *experiment choice* — they still go through the
> launcher and therefore inherit cluster/parallelism/precision resolution
> automatically. When SLURM submission lands, the scripts need no change.

---

### Task 15: End-to-end smoke run

Run the launcher with the POET experiment on a single GPU at 600M scale for ~10 iters and verify (a) the patches register, (b) `patch_set_hash` is written to `runs/<hash>/metadata.json`, (c) at least one POET merge fires.

**Files:**
- Run-only — no new files.

- [ ] **Step 1: Activate the env**

```bash
cd /lustre/fast/fast/zqiu/slm_env
source .venv/bin/activate
```

- [ ] **Step 2: Launch a 10-iter run via the POET startup script (Task 14)**

```bash
cd /lustre/fast/fast/zqiu/slm-research
scripts/train_poet.sh \
    allow_dirty=true \
    training.max_iters=10 training.poet_merge_period=5 \
    --no-submit
```

(Equivalent to: `python -m launchers.submit base/family=qwen3 base/scale=600m
experiment=optim/poet training_regime=ablation_20x cluster=h800_cn seed=0
wandb.project=sandbox-${USER} allow_dirty=true training.max_iters=10
training.poet_merge_period=5 --no-submit`.)

Expected log fragments:
- `Registered patches: poet_unfuse_te_impl, poet_apply_to_model, poet_merge_step`
- `[POET] replaced N linears | trainable=… frozen=…`
- `[POET] merged at iteration 5` and `[POET] merged at iteration 10`
- `patch_set_hash: <16-hex>` recorded in `runs/<config_hash>/metadata.json`

- [ ] **Step 3: Verify the metadata**

```bash
cat runs/<config_hash>/metadata.json
```

Expected: `"patch_set_hash": "<16-hex>"`, `"patches": ["poet_apply_to_model","poet_merge_step","poet_unfuse_te_impl"]`.

- [ ] **Step 4: If anything fails, file a follow-up note in docs/experiments/poet.md and stop. Do not paper over with try/except.**

- [ ] **Step 5: If everything passes, commit the lab-notebook update + close out**

```bash
git add docs/experiments/poet.md   # populated "What worked" section
git commit -m "docs: POET smoke run notes"
```

---

## Out-of-scope (acknowledged, separate plans)

| Item | Where it belongs | Why deferred |
|---|---|---|
| Fork-specific Megatron-launcher shell scripts (`train_llama31_1b_*.sh`) | `launchers/slurm/*.j2` Hydra-rendered SLURM templates | slm-research expresses runs via YAML composition; the fork's per-model shell scripts duplicate that. Optimizer-choice convenience wrappers (`scripts/train_{adam,muon,poet}.sh`) now in scope — see Task 14. |
| Parquet→jsonl preprocessing (`tools/preprocess_data_parquet_to_jsonl.{py,sh,sub}`) | `tools/preprocess_*` in slm-research | data-prep is its own plan with its own SHA manifest requirements. |
| `merge_dataset.sh`, `view_jsonl.py` | `tools/` | one-off utilities; not part of training contract. |
| Apex source | already vendored via [install_slm_env.sh](../../../install_slm_env.sh) | covered. |
| Nemotron-CC-v2 data blob | `data/datasets/<name>/` with manifest | data-manifest plan. |
| `api.txt`, `hold.py`, `UPLOAD_GUIDE*.md` | discard | empty / scratch / out-of-repo concerns. |
| Fork-1 working-tree `examples/llama/train_llama3_8b_h100_fp8.sh` mods | `launchers/` | local-cluster mods belong in cluster YAMLs, not example scripts. |

---

## Self-review summary

- **Spec coverage:** Every customised file in either fork has a destination in the plan or is explicitly listed under "Out-of-scope".
- **No upstream edits:** Every code path that mutates `megatron.*` lives in `src/patches/<name>.py` and is registered through `_registry`. The `third_party/Megatron-LM/` submodule is read-only.
- **Test coverage:** Each non-trivial helper (POETAdam wrapper, layer walk, optimizer dispatch) has a CPU-runnable unit test. Patch registration is tested per patch; live patch behaviour is exercised in the Task 15 smoke run.
- **Reproducibility hooks:** `patch_set_hash` is wired into `runs/<config_hash>/metadata.json` via Task 12. Together with `git_sha`, `megatron_sha`, `dataset_hash`, `seed`, that satisfies the run-identity contract in SPEC.md §6.
