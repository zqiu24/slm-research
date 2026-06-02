# POET × Muon-on-Q — Stage 0 diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two diagnostic instruments needed to run Stage 0 of the POET × Muon-on-Q investigation — a single-batch-overfit harness (Probe 0A) and an `∂f/∂Q` conditioning capture (Probe 0B) — with no changes to POET math or the optimizer core.

**Architecture:** Two pure, CPU-testable utility modules (`src/diag/`) hold the math (skew reconstruction, spectral stats) and the batch-replay logic. Two thin Megatron patches (`src/patches/`) wire them into the training loop, gated by environment variables so they are completely inert on normal runs. Both patches are added to the launcher's `_ALWAYS_ON_PATCHES` (which is deliberately excluded from `patch_set_hash`), so enabling a probe is just prefixing an env var onto the existing dev scripts.

**Tech Stack:** Python 3.12, PyTorch, Megatron-LM (vendored under `third_party/Megatron-LM`), the slm-research patch registry (`src/patches/_registry.py`), Hydra configs, W&B.

**Scope:** This plan covers **only** the Stage 0 diagnostics. Probe −1 (the `merge_period` sweep) needs no code. Everything downstream of the Stage 0 decision table — Muon-on-`oft_R`, the rotation-angle scaling rule, momentum transport, Stage 4 scale-up, and the Appendix A learnable-Σ branch — is **out of scope** and gated behind the human-review stop in [the execution plan](2026-06-02-poet-muon-q-stage0-execution.md).

**Test env:** Run all pytest with the CPU venv that has torch/omegaconf:
`/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest …` (the base `python` lacks torch).

**GPU/training split:** The four CPU-testable units (Tasks 1–3, plus patch-registration tests) are verified here. The actual GPU probe runs are the user's — this plan ends by handing over the exact `codexlog` commands; it does not run training.

---

## How the probes are run (after this plan lands)

All at 60m, DP=1 for 0A (single GPU removes "which rank's batch" ambiguity). Env var on, existing dev script otherwise:

```bash
# Probe -1 — merge_period sweep (NO CODE; runnable today)
codexlog s0_poet400  bash scripts/train_poet_dev.sh
codexlog s0_poet1600 bash scripts/train_poet_dev.sh optim.poet.merge_period=1600
codexlog s0_poet0    bash scripts/train_poet_dev.sh optim.poet.merge_period=0

# Probe 0A — single-batch overfit (after Tasks 3-4)
SLM_OVERFIT_SINGLE_BATCH=1 codexlog s0a_adam    bash scripts/train_adam_dev.sh scheduler=constant
SLM_OVERFIT_SINGLE_BATCH=1 codexlog s0a_poet400 bash scripts/train_poet_dev.sh scheduler=constant
SLM_OVERFIT_SINGLE_BATCH=1 codexlog s0a_poet0   bash scripts/train_poet_dev.sh scheduler=constant optim.poet.merge_period=0

# Probe 0B — conditioning (after Tasks 1-2-5; ONLY if 0A == OPTIMIZATION-LIMITED)
SLM_POET_GRAD_CONDITIONING=1 codexlog s0b_cond  bash scripts/train_poet_dev.sh
```

> **Gate:** Task 5 (the conditioning patch) should not be *run* until Probe 0A returns OPTIMIZATION-LIMITED. It is fine to *build* it eagerly (it is inert), but if you prefer the lazy path, stop after Task 4, run 0A, and only continue to Tasks 1/2/5 if the verdict warrants it.

---

## File Structure

- `src/diag/__init__.py` — new package (empty marker).
- `src/diag/skew_conditioning.py` — pure math: `vec_to_skew()` (upper-tri vector → skew-symmetric block(s)) and `block_spectral_stats()` (singular-value summary). No Megatron/CUDA imports → CPU-testable.
- `src/diag/single_batch.py` — `BatchReplay`: caches the first value it sees and returns it forever. No Megatron imports → CPU-testable.
- `src/patches/overfit_single_batch.py` — env-gated patch wrapping `pretrain_gpt.get_batch` with a `BatchReplay`.
- `src/patches/poet_grad_conditioning.py` — env-gated patch wrapping `get_megatron_optimizer` + the returned optimizer's `.step`, using the two utils to log per-block spectra to W&B.
- `launchers/pretrain_gpt_slm.py` — modify `_ALWAYS_ON_PATCHES` to include both new patches.
- Tests: `tests/unit/test_diag_skew_conditioning.py`, `tests/unit/test_diag_single_batch.py`, `tests/unit/test_patch_overfit_single_batch.py`, `tests/unit/test_patch_poet_grad_conditioning.py`.

---

## Task 1: Spectral-stats util (`block_spectral_stats`)

**Files:**
- Create: `src/diag/__init__.py`
- Create: `src/diag/skew_conditioning.py`
- Test: `tests/unit/test_diag_skew_conditioning.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_diag_skew_conditioning.py
import math

import torch

from src.diag.skew_conditioning import block_spectral_stats


def test_block_spectral_stats_on_known_skew():
    # A 4x4 skew matrix built from two 2x2 rotation generators with angles a,b.
    # Its singular values are {a, a, b, b} (paired, as skew matrices always are).
    a, b = 3.0, 1.0
    q = torch.zeros(1, 4, 4)
    q[0, 0, 1], q[0, 1, 0] = a, -a
    q[0, 2, 3], q[0, 3, 2] = b, -b

    stats = block_spectral_stats(q)

    # one block in -> one row of stats
    assert stats["condition_number"].shape == (1,)
    # sigma_max/sigma_min = a/b
    assert math.isclose(stats["condition_number"][0].item(), a / b, rel_tol=1e-5)
    # stable rank = ||.||_F^2 / sigma_max^2 = (2a^2 + 2b^2) / a^2
    expected_sr = (2 * a**2 + 2 * b**2) / a**2
    assert math.isclose(stats["stable_rank"][0].item(), expected_sr, rel_tol=1e-5)
    # sigma_max / median(sigmas): median of [a,a,b,b] sorted = (a+b)/2
    assert math.isclose(stats["sigma_max_over_median"][0].item(), a / ((a + b) / 2), rel_tol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_skew_conditioning.py::test_block_spectral_stats_on_known_skew -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.diag'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/diag/__init__.py
```

```python
# src/diag/skew_conditioning.py
"""Pure-math diagnostics for POET's per-block ∂f/∂Q conditioning (Probe 0B).

No Megatron / CUDA / poet_torch imports — every function here takes plain
tensors so the math is unit-testable on CPU.
"""

from __future__ import annotations

import torch


def block_spectral_stats(skew: torch.Tensor, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    """Summarize the singular-value spectrum of a batch of (skew-symmetric) blocks.

    Args:
        skew: tensor of shape (num_blocks, b, b). Skew-symmetric inputs have
            *paired* singular values; the stats below are well-defined on the
            full (paired) spectrum and pairing is not removed.
        eps: floor for sigma_min to avoid div-by-zero on rank-deficient blocks.

    Returns dict of shape-(num_blocks,) tensors:
        condition_number   = sigma_max / max(sigma_min, eps)
        stable_rank        = ||.||_F^2 / sigma_max^2
        sigma_max_over_median = sigma_max / median(sigma)
    """
    if skew.dim() == 2:
        skew = skew.unsqueeze(0)
    sv = torch.linalg.svdvals(skew.to(torch.float32))  # (num_blocks, b), descending
    sigma_max = sv[:, 0]
    sigma_min = sv[:, -1].clamp_min(eps)
    fro_sq = (sv * sv).sum(dim=1)
    # quantile(0.5), NOT torch.median: for even-length spectra torch.median
    # returns the lower-middle value, not the arithmetic median the metric wants
    # (e.g. median of [3,3,1,1] is 2.0, not 1.0).
    median = torch.quantile(sv, 0.5, dim=1)
    return {
        "condition_number": sigma_max / sigma_min,
        "stable_rank": fro_sq / (sigma_max * sigma_max),
        "sigma_max_over_median": sigma_max / median.clamp_min(eps),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_skew_conditioning.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/diag/__init__.py src/diag/skew_conditioning.py tests/unit/test_diag_skew_conditioning.py
git commit -m "feat(diag): add block_spectral_stats for POET ∂f/∂Q conditioning"
```

---

## Task 2: Skew reconstruction util (`vec_to_skew`)

The trainable `oft_R` (and its gradient) is stored as the **upper-triangular** vector of each block, with the layout `rows, cols = torch.triu_indices(b, b, 1)` (see [adapter.py](../../../poet_torch_huawei/megatron/core/poet_adapter/adapter.py#L117) and `pytorch_skew_symmetric` in [poet_layer.py](../../../third_party/poet_torch/poet_layer.py#L207)). This util rebuilds the full skew block(s) from that vector — independent of `poet_torch` so it stays CPU-pure.

**Files:**
- Modify: `src/diag/skew_conditioning.py`
- Test: `tests/unit/test_diag_skew_conditioning.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_diag_skew_conditioning.py
from src.diag.skew_conditioning import vec_to_skew


def test_vec_to_skew_is_skew_symmetric_and_matches_layout():
    b = 4
    # b(b-1)/2 = 6 upper-tri entries, two blocks stacked
    vec = torch.arange(1.0, 13.0).reshape(2, 6)
    q = vec_to_skew(vec, b)

    assert q.shape == (2, b, b)
    # skew-symmetry: Q == -Q^T
    assert torch.allclose(q, -q.transpose(-1, -2))
    # diagonal is zero
    assert torch.allclose(torch.diagonal(q, dim1=-2, dim2=-1), torch.zeros(2, b))
    # first upper-tri entry (row 0, col 1) of block 0 is vec[0,0]
    assert q[0, 0, 1].item() == 1.0
    assert q[0, 1, 0].item() == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_skew_conditioning.py::test_vec_to_skew_is_skew_symmetric_and_matches_layout -v`
Expected: FAIL with `ImportError: cannot import name 'vec_to_skew'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/diag/skew_conditioning.py
def vec_to_skew(vec: torch.Tensor, block_size: int) -> torch.Tensor:
    """Map upper-triangular vectors to full skew-symmetric blocks.

    Args:
        vec: shape (num_blocks, b*(b-1)/2), the trainable/grad entries in the
            same order as ``torch.triu_indices(b, b, 1)``.
        block_size: b.

    Returns: (num_blocks, b, b) with Q[..., r, c] = vec, Q[..., c, r] = -vec.
    """
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    b = block_size
    n = vec.shape[0]
    rows, cols = torch.triu_indices(b, b, 1)
    q = torch.zeros(n, b, b, dtype=vec.dtype, device=vec.device)
    q[:, rows, cols] = vec
    q[:, cols, rows] = -vec
    return q
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_skew_conditioning.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/diag/skew_conditioning.py tests/unit/test_diag_skew_conditioning.py
git commit -m "feat(diag): add vec_to_skew reconstruction matching POET triu layout"
```

---

## Task 3: Single-batch replay util (`BatchReplay`)

**Files:**
- Create: `src/diag/single_batch.py`
- Test: `tests/unit/test_diag_single_batch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_diag_single_batch.py
from src.diag.single_batch import BatchReplay


def test_replay_caches_first_and_repeats():
    replay = BatchReplay()
    calls = iter([("batch-A",), ("batch-B",), ("batch-C",)])

    def fake_get_batch():
        return next(calls)

    first = replay(fake_get_batch)
    second = replay(fake_get_batch)
    third = replay(fake_get_batch)

    assert first == ("batch-A",)
    # subsequent calls return the cached first batch; the producer is NOT advanced
    assert second == ("batch-A",)
    assert third == ("batch-A",)
    assert replay.calls == 3
    assert replay.producer_calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_single_batch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.diag.single_batch'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/diag/single_batch.py
"""Single-batch-overfit helper (Probe 0A): cache the first batch, replay forever."""

from __future__ import annotations

from typing import Any, Callable


class BatchReplay:
    """Caches the first value produced and returns it on every later call.

    The wrapped producer (e.g. Megatron's ``get_batch``) is invoked exactly
    once; thereafter the cached value is returned without advancing it, so the
    model sees one fixed minibatch every step.
    """

    def __init__(self) -> None:
        self._cached: Any = None
        self._have: bool = False
        self.calls: int = 0
        self.producer_calls: int = 0

    def __call__(self, producer: Callable[[], Any]) -> Any:
        self.calls += 1
        if not self._have:
            self._cached = producer()
            self.producer_calls += 1
            self._have = True
        return self._cached
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_single_batch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/diag/single_batch.py tests/unit/test_diag_single_batch.py
git commit -m "feat(diag): add BatchReplay single-batch-overfit helper"
```

---

## Task 4: `overfit_single_batch` patch (Probe 0A wiring)

Wraps `pretrain_gpt.get_batch` so each call returns the cached first batch. Env-gated by `SLM_OVERFIT_SINGLE_BATCH=1`; inert (does not wrap) otherwise. `get_batch` is resolved as a module global by `pretrain_gpt.forward_step` at call time, so replacing the module attribute is sufficient. `train_step` is owned by `poet_merge_step`, so this targets a different, free symbol.

**Files:**
- Create: `src/patches/overfit_single_batch.py`
- Modify: `launchers/pretrain_gpt_slm.py:91` (`_ALWAYS_ON_PATCHES`)
- Test: `tests/unit/test_patch_overfit_single_batch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_patch_overfit_single_batch.py
import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SLM_OVERFIT_SINGLE_BATCH", raising=False)
    _reset_for_tests()
    sys.modules.pop("src.patches.overfit_single_batch", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.overfit_single_batch", None)


def test_patch_registers_with_unique_target():
    importlib.import_module("src.patches.overfit_single_batch")
    reg = registered_patches()
    assert "overfit_single_batch" in reg
    assert any("get_batch" in t for t in reg["overfit_single_batch"].targets)


def test_apply_is_inert_without_env(monkeypatch):
    """With the env var unset, apply() must NOT replace get_batch."""
    fake_mod = type(sys)("pretrain_gpt")
    sentinel = object()
    fake_mod.get_batch = sentinel
    monkeypatch.setitem(sys.modules, "pretrain_gpt", fake_mod)

    mod = importlib.import_module("src.patches.overfit_single_batch")
    mod.apply()

    assert fake_mod.get_batch is sentinel  # untouched


def test_apply_wraps_get_batch_when_enabled(monkeypatch):
    monkeypatch.setenv("SLM_OVERFIT_SINGLE_BATCH", "1")
    fake_mod = type(sys)("pretrain_gpt")
    seq = iter([("A",), ("B",)])
    fake_mod.get_batch = lambda *a, **k: next(seq)
    monkeypatch.setitem(sys.modules, "pretrain_gpt", fake_mod)

    mod = importlib.import_module("src.patches.overfit_single_batch")
    mod.apply()

    assert fake_mod.get_batch("iter") == ("A",)
    assert fake_mod.get_batch("iter") == ("A",)  # replayed, not advanced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_overfit_single_batch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.patches.overfit_single_batch'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/patches/overfit_single_batch.py
"""Patch (Probe 0A): single-batch overfit by replaying the first get_batch.

Env-gated: only active when SLM_OVERFIT_SINGLE_BATCH=1. Inert otherwise, so it
is safe in _ALWAYS_ON_PATCHES and never perturbs a normal run. Targets
``pretrain_gpt.get_batch`` (free; train_step is owned by poet_merge_step).
"""

from __future__ import annotations

import logging
import os

from src.patches._registry import register_patch

_TARGET = ("pretrain_gpt.get_batch",)
logger = logging.getLogger(__name__)


@register_patch(name="overfit_single_batch", targets=_TARGET)
def apply() -> None:
    if os.environ.get("SLM_OVERFIT_SINGLE_BATCH") != "1":
        return  # inert unless explicitly enabled

    import pretrain_gpt as mg
    from src.diag.single_batch import BatchReplay

    _orig_get_batch = mg.get_batch
    _replay = BatchReplay()

    def _wrapped(*args, **kwargs):
        return _replay(lambda: _orig_get_batch(*args, **kwargs))

    mg.get_batch = _wrapped
    logger.warning("[OVERFIT] single-batch overfit ENABLED — replaying the first get_batch every step")
```

- [ ] **Step 4: Register it as an always-on (but inert) patch**

In `launchers/pretrain_gpt_slm.py`, change the always-on tuple and its comment:

```python
# Patches applied to EVERY Megatron run regardless of experiment. The logging
# ones are no-ops on the model; the diagnostic ones (overfit_single_batch,
# poet_grad_conditioning) self-disable unless their SLM_* env var is set, so
# they are inert on normal runs and stay out of the experiment patch_set_hash.
_ALWAYS_ON_PATCHES = (
    "wandb_trainable_params",
    "overfit_single_batch",
    "poet_grad_conditioning",
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_overfit_single_batch.py -v`
Expected: PASS (3 tests)

> Note: Task 4 references `poet_grad_conditioning` in `_ALWAYS_ON_PATCHES` before Task 5 creates it. `_apply_runtime_patches` imports each name, so until Task 5 lands, importing the missing module would fail **only if a real run is launched**. The unit tests above do not import the launcher. If you run Tasks strictly in order and launch nothing between 4 and 5, this is fine; otherwise complete Task 5 before any launch.

- [ ] **Step 6: Commit**

```bash
git add src/patches/overfit_single_batch.py launchers/pretrain_gpt_slm.py tests/unit/test_patch_overfit_single_batch.py
git commit -m "feat(diag): add env-gated single-batch overfit patch (Probe 0A)"
```

---

## Task 5: `poet_grad_conditioning` patch (Probe 0B wiring)

Composes on top of `poet_optimizer_setup` by wrapping the **current** `megatron.training.training.get_megatron_optimizer` (which receives `model`, so we can find the `oft_R` params), then wraps the returned optimizer's `.step` to read each tracked param's `main_grad` *before* the update, reconstruct the skew, and log spectral stats to W&B. Env-gated by `SLM_POET_GRAD_CONDITIONING=1`; interval via `SLM_POET_GRAD_CONDITIONING_INTERVAL` (default 2000). Declares a **unique** target label so it never conflicts with `poet_optimizer_setup` (which owns `get_megatron_optimizer`).

**Files:**
- Create: `src/patches/poet_grad_conditioning.py`
- Test: `tests/unit/test_patch_poet_grad_conditioning.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_patch_poet_grad_conditioning.py
import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SLM_POET_GRAD_CONDITIONING", raising=False)
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_grad_conditioning", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_grad_conditioning", None)


def test_patch_registers_with_unique_target():
    importlib.import_module("src.patches.poet_grad_conditioning")
    reg = registered_patches()
    assert "poet_grad_conditioning" in reg
    # unique label, NOT the real get_megatron_optimizer symbol owned by poet_optimizer_setup
    assert all("get_megatron_optimizer" not in t for t in reg["poet_grad_conditioning"].targets)


def test_select_target_params_picks_representative_blocks():
    """Pure selection logic is CPU-testable without Megatron."""
    import torch

    from src.patches.poet_grad_conditioning import select_target_params

    class FakePOET:
        def __init__(self, name, b):
            self.block_size_in = b
            self.block_size_out = b
            # one block's worth of upper-tri entries: b*(b-1)/2
            self.oft_R_in = torch.zeros(1, b * (b - 1) // 2)
            self.oft_R_out = torch.zeros(1, b * (b - 1) // 2)
            self._name = name

    layers = {
        "decoder.layers.0.self_attention.linear_q": FakePOET("q0", 4),
        "decoder.layers.5.self_attention.linear_v": FakePOET("v5", 4),
        "decoder.layers.9.mlp.linear_fc2": FakePOET("down9", 4),
        "decoder.layers.1.mlp.linear_no_match": FakePOET("x1", 4),
    }
    targets = select_target_params(layers.items(), max_targets=8)
    labels = {t["label"] for t in targets}
    # q/v/down projections selected (both R_in and R_out factors), the no-match dropped
    assert any("linear_q" in lbl for lbl in labels)
    assert any("linear_v" in lbl for lbl in labels)
    assert any("linear_fc2" in lbl for lbl in labels)
    assert not any("linear_no_match" in lbl for lbl in labels)
    # each selected layer contributes its R_in and R_out factor
    assert all(t["factor"] in ("R_in", "R_out") for t in targets)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_grad_conditioning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.patches.poet_grad_conditioning'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/patches/poet_grad_conditioning.py
"""Patch (Probe 0B): log per-block ∂f/∂Q conditioning during a POET run.

Env-gated by SLM_POET_GRAD_CONDITIONING=1 (interval via
SLM_POET_GRAD_CONDITIONING_INTERVAL, default 2000). Inert otherwise, so it is
safe in _ALWAYS_ON_PATCHES.

Mechanism: wrap the (possibly poet-routed) ``get_megatron_optimizer`` — which
receives ``model`` — to (a) pick ~8 representative oft_R blocks and (b) wrap the
returned optimizer's ``.step`` so that, every ``interval`` steps, it reads each
block's gradient from ``main_grad`` (the Megatron DDP fp32 buffer; falls back to
``.grad``), reconstructs the skew, and logs spectral stats to W&B BEFORE the
optimizer consumes the grad.
"""

from __future__ import annotations

import logging
import os

from src.patches._registry import register_patch

# Unique label: this patch does NOT own get_megatron_optimizer (poet_optimizer_setup
# does); it composes on top of whatever that symbol currently is.
_TARGET = ("slm.diagnostics.poet_grad_conditioning.optimizer_step",)
logger = logging.getLogger(__name__)

# projection name fragments we care about (HF-ish + Megatron names)
_WANTED = ("linear_q", "q_proj", "linear_v", "v_proj", "linear_fc2", "down_proj", "linear_fc1", "up_proj")


def select_target_params(named_layers, max_targets: int = 8):
    """From (name, poet_layer) pairs, pick representative blocks to probe.

    Returns a list of dicts: {label, factor ('R_in'|'R_out'), param, block_size}.
    A layer contributes a target only if its name matches a wanted projection.
    """
    targets = []
    for name, layer in named_layers:
        if not any(w in name for w in _WANTED):
            continue
        for factor, attr, bsz_attr in (
            ("R_in", "oft_R_in", "block_size_in"),
            ("R_out", "oft_R_out", "block_size_out"),
        ):
            param = getattr(layer, attr, None)
            bsz = getattr(layer, bsz_attr, None)
            if param is None or bsz is None:
                continue
            targets.append({"label": f"{name}.{factor}", "factor": factor, "param": param, "block_size": int(bsz)})
            if len(targets) >= max_targets:
                return targets
    return targets


def _log_conditioning(targets, iteration: int) -> None:
    import torch

    from src.diag.skew_conditioning import block_spectral_stats, vec_to_skew

    try:
        import wandb
    except Exception:  # noqa: BLE001
        wandb = None

    for t in targets:
        param = t["param"]
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        if grad is None:
            logger.warning("[COND] no grad for %s at iter %d", t["label"], iteration)
            continue
        vec = grad.detach().to(torch.float32).reshape(param.shape[0], -1)
        skew = vec_to_skew(vec, t["block_size"])
        stats = block_spectral_stats(skew)
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(
                {
                    f"poet_cond/{t['label']}/condition_number": stats["condition_number"].mean().item(),
                    f"poet_cond/{t['label']}/stable_rank": stats["stable_rank"].mean().item(),
                    f"poet_cond/{t['label']}/sigma_max_over_median": stats["sigma_max_over_median"].mean().item(),
                },
                step=iteration,
            )


def _install_step_hook(optimizer, targets, interval: int) -> None:
    _orig_step = optimizer.step
    state = {"n": 0}

    def _wrapped_step(*args, **kwargs):
        if state["n"] % interval == 0:
            try:
                _log_conditioning(targets, state["n"])
            except Exception:  # noqa: BLE001 — diagnostics must never break training
                logger.exception("[COND] conditioning log failed at step %d", state["n"])
        state["n"] += 1
        return _orig_step(*args, **kwargs)

    optimizer.step = _wrapped_step


@register_patch(name="poet_grad_conditioning", targets=_TARGET)
def apply() -> None:
    if os.environ.get("SLM_POET_GRAD_CONDITIONING") != "1":
        return  # inert unless explicitly enabled

    interval = int(os.environ.get("SLM_POET_GRAD_CONDITIONING_INTERVAL", "2000"))
    from megatron.training import training as _mt

    _orig_get_optimizer = _mt.get_megatron_optimizer

    def _wrapped_get_optimizer(config, model, **kwargs):
        optimizer = _orig_get_optimizer(config, model, **kwargs)
        chunks = model if isinstance(model, list) else [model]
        named_layers = [
            (name, mod)
            for m in chunks
            for name, mod in m.named_modules()
            if hasattr(mod, "oft_R_in") or hasattr(mod, "oft_R_out")
        ]
        targets = select_target_params(named_layers)
        if targets:
            _install_step_hook(optimizer, targets, interval)
            logger.warning("[COND] ∂f/∂Q conditioning ENABLED — probing %d blocks every %d steps", len(targets), interval)
        else:
            logger.warning("[COND] no oft_R layers found; conditioning probe is a no-op")
        return optimizer

    _mt.get_megatron_optimizer = _wrapped_get_optimizer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_grad_conditioning.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_grad_conditioning.py tests/unit/test_patch_poet_grad_conditioning.py
git commit -m "feat(diag): add env-gated ∂f/∂Q conditioning probe patch (Probe 0B)"
```

---

## Task 6: Full suite + lint gate

**Files:** none (verification only).

- [ ] **Step 1: Run the new tests together**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_skew_conditioning.py tests/unit/test_diag_single_batch.py tests/unit/test_patch_overfit_single_batch.py tests/unit/test_patch_poet_grad_conditioning.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 2: Confirm the patch registry still composes (no conflicts)**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patches_registry.py tests/unit/test_launcher_patch_wiring.py -v`
Expected: PASS (the two new always-on patches declare unique targets, so no `PatchConflict`)

- [ ] **Step 3: Lint**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m ruff check src/diag src/patches/overfit_single_batch.py src/patches/poet_grad_conditioning.py tests/unit/test_diag_skew_conditioning.py tests/unit/test_diag_single_batch.py tests/unit/test_patch_overfit_single_batch.py tests/unit/test_patch_poet_grad_conditioning.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit any lint fixes (if needed)**

```bash
git add -A && git commit -m "style(diag): ruff fixes for Stage-0 diagnostics"
```

---

## Post-implementation: running the probes (USER — GPU)

Hand the user the commands from the "How the probes are run" section. The two deliverable reports are authored from the W&B output, not generated by code:
- `stage0a_overfit.md` — the `merge_period` sweep curves + three-arm overfit floors + verdict (OPTIMIZATION / REPRESENTATION / RESET-limited).
- `stage1_conditioning.md` — the `poet_cond/*` condition-number / stable-rank curves + PROCEED/STOP verdict.

**STOP for human review once Stage 0 resolves to a decision-table row.** Do not begin Muon-on-`oft_R` work before that.

---

## Self-Review

**Spec coverage** (against [the execution plan](2026-06-02-poet-muon-q-stage0-execution.md)):
- Probe −1 (merge_period sweep): no code — commands provided. ✓
- Probe 0A (single-batch overfit, 3 arms, reg off): Tasks 3–4 + `scheduler=constant` (exists) + `weight_decay=0`/dropout 0 (defaults). ✓
- Probe 0B (conditioning hook: ~8 blocks across layers×projections, both factors; skew reconstruct; svdvals w/ paired SVs; condition number / stable rank / σ_max/σ_median; W&B): Tasks 1–2–5. ✓
- Read `main_grad` not `.grad`; fire before the optimizer consumes the grad: `_install_step_hook` reads on `.step` entry. ✓
- Deferred items (Muon-on-Q, angle scaling, transport, Appendix A): explicitly out of scope. ✓

**Placeholder scan:** no TBD/TODO; every code step has complete code; every test has real assertions. ✓

**Type/name consistency:** `vec_to_skew(vec, block_size)` and `block_spectral_stats(skew)` defined in Task 1–2 and consumed in Task 5's `_log_conditioning` with matching signatures; `BatchReplay()` callable defined in Task 3, used in Task 4; `select_target_params(named_layers, max_targets)` defined and tested in Task 5; env var names (`SLM_OVERFIT_SINGLE_BATCH`, `SLM_POET_GRAD_CONDITIONING`, `SLM_POET_GRAD_CONDITIONING_INTERVAL`) consistent across patches, run commands, and tests. ✓

**Known integration risks (flagged, not placeholders):**
1. `main_grad` is assumed populated and unzeroed at `optimizer.step` entry. If a given Megatron build zeroes it inside an earlier hook, the probe logs a "no grad" warning rather than wrong numbers — verify the first `poet_cond/*` value is finite on the first real run.
2. `oft_R_in/out` first-dim = number of blocks; the reshape in `_log_conditioning` assumes the grad flattens to `(num_blocks, b*(b-1)/2)`. Confirm against one real layer's shapes on the first run (a 1-line assert can be added if it surprises).
