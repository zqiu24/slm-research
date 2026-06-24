# Pion Optimizer Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Pion optimizer (vendored from `third_party/pion`) into the slm-research Megatron backend as a first-class `optim.type=pion` experiment, with a dev training script and an LR sweep script — mirroring the existing `muon_kimi` integration.

**Architecture:** Pion drives 2-D matrix (non-embedding) weights via the orthogonal-equivalence update `W ← W·exp(A_in) + exp(A_out)·W − W` (truncated matrix exponential); a stock Megatron AdamW drives everything else (embeddings, norms, biases, LM head). The two are combined into a Megatron `ChainedOptimizer`, exactly like POET's matrix+adam split (`src/optim/poet.py`). The Pion algorithm is vendored verbatim (`src/optim/_pion.py`, do-not-edit, like `_kimi_muon.py`); a thin builder (`src/optim/pion.py`) adapts it to slm-research conventions; a patch (`src/patches/pion_optimizer_setup.py`) reroutes Megatron's optimizer-builder to it. Single-GPU dev scope only (no TP/PP/distributed-optimizer/fp16).

**Tech Stack:** Python 3, PyTorch, NVIDIA Megatron-LM (pinned submodule `third_party/Megatron-LM`), Hydra/OmegaConf configs, pytest. Reference upstream: `third_party/pion/megatron-lm/megatron/core/optimizer/pion.py`.

## Global Constraints

- **Single-GPU dev only.** Pion builder MUST raise on `use_distributed_optimizer`, `fp16`, `tensor_model_parallel_world_size > 1`, `pipeline_model_parallel_world_size > 1` (mirror `src/optim/muon_kimi.py:68-79`).
- **No submodule edits.** Do NOT modify `third_party/Megatron-LM` or `third_party/pion`. All Pion CLI args are registered in `launchers/pretrain_gpt_slm.py`; all Pion config fields are set onto the `OptimizerConfig` object at runtime by the patch (the `pion_*` names are NOT declared `OptimizerConfig` dataclass fields, so the stock `get_megatron_optimizer_config` arg→field copy will not carry them — the patch copies them explicitly).
- **No recursion.** The builder calls the *original* `megatron.core.optimizer.get_megatron_optimizer` for the Adam side, resolved lazily (like `src/optim/poet.py:200-239`). The patch only rebinds `megatron.training.training.get_megatron_optimizer`, so the core call does not re-enter the patch.
- **Vendored algorithm is verbatim.** `src/optim/_pion.py` is a copy of the upstream algorithm with ONLY the Megatron-import line replaced by a stdlib shim. Do not alter the math.
- **`optim.type=pion` routes through `--optimizer adam --slm-optimizer pion`** (same as `muon_kimi`: the stock Megatron path builds an `AdamOptimizerConfig`; the patch tags + reroutes).
- **Pion defaults to FUSED qkv/fc1, with unfusing as an OPT-IN.** By default the `pion` experiment keeps qkv/fc1 fused and Pion splits them per-head / up-gate *internally* (Pion-native granularity, like the upstream reference). Unfusing is supported as an option, mirroring `adam.yaml`: `pion.yaml` lists the `model_unfuse_linears` patch (a no-op unless `base.model.unfuse_qkv`/`unfuse_fc1` are set), and a sibling `pion_unfused.yaml` turns it on. The builder routes BOTH paths correctly with **no code change**: when unfused, the separate q/k/v/up/gate weights do not match `is_qkv_fn`/`is_fc1_up_gate_fn` (which test for `linear_qkv.weight` / `linear_fc1.weight` in the param name), so each is rotated as a plain 2-D matrix; when fused, Pion's internal split applies. Unfusing simply makes Pion's internal split inert (the fused tensor name is gone) — the two are mutually exclusive per tensor and never conflict.
- **Reference defaults** (from `third_party/pion/megatron-lm/opt_llama_60M_pion.sh`): `pion_scaling=rms`, `pion_rms=0.2`, `pion_update_side=alternate`, `pion_momentum=transported_ambient_ambient`, `pion_degree=2`, `pion_beta1=0.9`, `pion_beta2=0.95`, adam betas `(0.9, 0.95)`, `adam_eps=1e-8`, `lr=1e-3`, `weight_decay=0.1`.
- **`pion_msign` (the exploration variant) is OUT OF SCOPE.** Only the two core Pion momentum geometries (`lie_lie`, `transported_ambient_ambient`) in `pion.py` are ported.
- **`src/optim/__init__.py` (`OptimizerCfg`/`get_optimizer`) is NOT touched.** That dispatcher serves the torchtitan/direct path; the Megatron path uses `slm_optimizer` and never reaches it (same as `muon_kimi`, which is absent from `_VALID_KINDS`).

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `src/optim/_pion.py` | Vendored Pion algorithm: helper fns + `PionOptimizer` class. Do-not-edit math. | Create |
| `src/optim/pion.py` | `get_megatron_pion_optimizer(config, model_chunks, …)` — builds Pion(matrix)+AdamW(rest) `ChainedOptimizer`; single-GPU guards; lazy Megatron-symbol resolution. | Create |
| `src/patches/pion_optimizer_setup.py` | Patch: tag `config.slm_optimizer="pion"`, copy `pion_*` args→config, reroute builder. Mirrors `muon_kimi_optimizer_setup.py`. | Create |
| `launchers/pretrain_gpt_slm.py` | Add `"pion"` to `--slm-optimizer` choices; register `--pion-*` CLI args. | Modify |
| `src/utils/megatron_args.py` | `_optimizer_args`: add `kind == "pion"` branch emitting the Pion argv. | Modify |
| `configs/experiments/optim/pion.yaml` | `optim.type=pion` experiment config, FUSED default (Pion-native internal split). Lists `model_unfuse_linears` as a no-op-by-default opt-in. | Create |
| `configs/experiments/optim/pion_unfused.yaml` | Unfused Pion variant: identical to `pion.yaml` but sets `base.model.unfuse_qkv/unfuse_fc1=true` so Pion rotates each separate q/k/v/up/gate as a plain matrix. | Create |
| `docs/experiments/pion.md`, `docs/experiments/pion_unfused.md` | Required by the `experiment-doc-exists` pre-commit hook (one doc per experiment `name`). | Create |
| `scripts/train_pion_dev.sh` | Single-GPU dev launcher (`experiment=optim/pion`). Mirrors `train_muon_dev.sh`. | Create |
| `scripts/sweep_pion_lr.sh` | LR sweep over `optim.lr`, all else at Pion defaults. Mirrors `sweep_muon_kimi_lr.sh`. | Create |
| `tests/unit/test_pion_optimizer.py` | CPU unit tests for the vendored `PionOptimizer` (spectrum preservation, determinism, both momentum modes). | Create |
| `tests/unit/test_patch_pion_optimizer_setup.py` | CPU test for the patch (tag + reroute + delegate). Mirrors `test_patch_muon_kimi_optimizer_setup.py`. | Create |
| `tests/unit/test_megatron_args.py` | Add `test_pion_argv_routes_through_adam_and_sets_pion_knobs`. | Modify |
| `tests/unit/test_train_scripts.py` | Add a `train_pion_dev.sh` dry-run/arg-check test. | Modify |

---

## Task 1: Vendor the Pion algorithm (`src/optim/_pion.py`)

**Files:**
- Create: `src/optim/_pion.py`
- Test: `tests/unit/test_pion_optimizer.py`

**Interfaces:**
- Produces: `PionOptimizer(params, lr, betas, weight_decay, degree, split_qkv, is_qkv_fn, qkv_split_shapes, split_fc1_up_gate, is_fc1_up_gate_fn, split_qkv_per_head, qkv_split_granularity, pion_scaling, pion_rms, pion_momentum, pion_update_side, pion_beta1, pion_beta2)` — a `torch.optim.Optimizer` whose `step()` updates only 2-D params in its param groups. Also the module-level helpers `_momentum_mode`, `_configured_update_side`, etc. (used only internally).

The vendored file is an extraction of the upstream `third_party/pion/megatron-lm/megatron/core/optimizer/pion.py`: keep the algorithm (module helpers `_matrix_exp_truncated` … `_apply_biside_update`, and the `PionOptimizer` class — upstream lines 40-757), drop the Megatron builder (upstream lines 759-952), and replace the Megatron import block (upstream lines 1-37) with a stdlib header + a `log_single_rank` shim. The `PionOptimizer` class uses only `torch` + `log_single_rank`; the helpers use only `torch`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pion_optimizer.py
"""CPU unit tests for the vendored Pion optimizer (src/optim/_pion.py)."""

from __future__ import annotations

import torch

from src.optim._pion import PionOptimizer


def _square_param(seed: int = 0, n: int = 16) -> torch.nn.Parameter:
    gen = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(n, n, generator=gen))


def test_pion_step_lie_lie_preserves_spectrum_for_small_step():
    """Pion's orthogonal-equivalence update preserves singular values; with a
    tiny lr the truncated-exp approximation keeps them within 5%."""
    w = _square_param(seed=1)
    sv_before = torch.linalg.svdvals(w.detach().clone())
    opt = PionOptimizer(
        [w], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0, degree=2,
        pion_scaling="rms", pion_rms=0.2,
        pion_momentum="lie_lie", pion_update_side="both",
    )
    gen = torch.Generator().manual_seed(2)
    w.grad = torch.randn(16, 16, generator=gen)
    opt.step()
    assert torch.isfinite(w.detach()).all()
    sv_after = torch.linalg.svdvals(w.detach())
    rel = ((sv_after - sv_before).abs() / (sv_before.abs() + 1e-6)).max()
    assert rel < 0.05, f"singular values drifted by {rel:.4f} (>5%)"


def test_pion_step_changes_weight_and_is_deterministic():
    """Same seed + same grad → identical update (no Date.now/rng leakage)."""
    results = []
    for _ in range(2):
        w = _square_param(seed=3)
        before = w.detach().clone()
        opt = PionOptimizer(
            [w], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0, degree=2,
            pion_scaling="rms", pion_rms=0.2,
            pion_momentum="transported_ambient_ambient", pion_update_side="alternate",
        )
        gen = torch.Generator().manual_seed(4)
        w.grad = torch.randn(16, 16, generator=gen)
        opt.step()
        assert not torch.allclose(w.detach(), before)
        results.append(w.detach().clone())
    assert torch.allclose(results[0], results[1])


def test_pion_skips_non_2d_params():
    """1-D params in a Pion group are left untouched (Pion is matrix-only)."""
    bias = torch.nn.Parameter(torch.randn(16))
    before = bias.detach().clone()
    opt = PionOptimizer(
        [bias], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0,
        pion_momentum="lie_lie", pion_update_side="both",
    )
    bias.grad = torch.randn(16)
    opt.step()
    assert torch.allclose(bias.detach(), before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_pion_optimizer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.optim._pion'`

- [ ] **Step 3: Create the vendored file**

Create `src/optim/_pion.py` by extracting the upstream algorithm. Concretely:

1. Copy `third_party/pion/megatron-lm/megatron/core/optimizer/pion.py` to `src/optim/_pion.py`.
2. Replace the entire header (upstream lines 1-40, i.e. from the `# pyright:` comment through the `logger = logging.getLogger(__name__)` line — everything ABOVE the first helper `def _matrix_exp_truncated`) with the block below.
3. Delete everything from `def _matrix_param_groups(` (upstream line 760) to end of file (the builder + `__all__`). The file must END right after `PionOptimizer.step()`'s `return loss` (upstream line 757).

New header to paste at the top (replaces upstream lines 1-40):

```python
"""Vendored Pion optimizer algorithm (single-process).

Extracted verbatim from
third_party/pion/megatron-lm/megatron/core/optimizer/pion.py (the algorithm
portion: module helpers + the ``PionOptimizer`` class). The Megatron builder
(``get_megatron_pion_optimizer``) is NOT vendored here — integration lives in
src/optim/pion.py. Do not edit the algorithm; only the import header below is
adapted (the original ``log_single_rank`` import is replaced by a stdlib shim).

Pion (Shi, Li, Qiu, Wen, Buchholz, Liu): a spectrum-preserving optimizer via
orthogonal equivalence transformation. Matrix updates use two Lie generators:

    W <- W exp(A_in) + exp(A_out) W - W

with both exponentials approximated by a truncated Taylor expansion.
"""

import csv
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer


logger = logging.getLogger(__name__)


def log_single_rank(logger_, level, message, *args, **kwargs):
    """Stdlib shim for Megatron's ``log_single_rank`` (rank-0-only logging)."""
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        logger_.log(level, message, *args, **kwargs)
```

Everything from `def _matrix_exp_truncated(` through the end of `PionOptimizer.step()` is kept byte-for-byte from upstream (do not retype it — preserve it exactly).

- [ ] **Step 4: Compile + run tests**

Run: `python -m py_compile src/optim/_pion.py && python -m pytest tests/unit/test_pion_optimizer.py -q`
Expected: `py_compile` silent (exit 0); pytest `3 passed`.

- [ ] **Step 5: Lint**

Run: `ruff check src/optim/_pion.py tests/unit/test_pion_optimizer.py`
Expected: no errors (the kept algorithm may carry upstream-style names; if ruff flags unused `Any/Callable/Dict/List/Optional/Tuple` imports that the kept code does not reference, trim ONLY the unused names from the `from typing import …` line — do not touch the algorithm body).

- [ ] **Step 6: Commit**

```bash
git add src/optim/_pion.py tests/unit/test_pion_optimizer.py
git commit -m "feat(pion): vendor Pion optimizer algorithm + CPU unit tests"
```

---

## Task 2: Pion Megatron builder (`src/optim/pion.py`)

**Files:**
- Create: `src/optim/pion.py`

**Interfaces:**
- Consumes: `PionOptimizer` from `src/optim/_pion.py`; lazily-resolved Megatron primitives `_get_param_groups`, `get_megatron_optimizer`, `ChainedOptimizer`, `Float16OptimizerWithFloat16Params`, `FP32Optimizer` (from `megatron.core.optimizer` / `megatron.core.optimizer.optimizer`).
- Produces: `get_megatron_pion_optimizer(config, model_chunks, *, config_overrides=None, use_gloo_process_groups=True) -> MegatronOptimizer` (a `ChainedOptimizer`). Signature MUST match what the patch passes (Task 3) — same kwargs as `get_megatron_muon_kimi_optimizer` in `src/optim/muon_kimi.py:53-59`.
- **Fused/unfused routing (no code branch needed):** the param walk tags `param.is_qkv = True` only when `"linear_qkv.weight" in name` and `param.is_fc1_up_gate = True` only when `"linear_fc1.weight" in name`. With the FUSED architecture (default) those tags fire and Pion splits internally; with the UNFUSED architecture (`base.model.unfuse_qkv/unfuse_fc1=true`) the fused names are absent, the tags never fire, and each separate q/k/v/up/gate is a plain 2-D matrix param routed to Pion's `main`-block update. Both are correct; the builder code is identical.

This task has no standalone CPU unit test (a full build needs a real Megatron model on GPU). It is verified by import-compile here, by the patch test in Task 3 (which monkeypatches the builder), and by the GPU smoke in Task 9. Mirror the upstream builder (`third_party/pion/megatron-lm/megatron/core/optimizer/pion.py:777-949`), adapted per the Global Constraints.

- [ ] **Step 1: Write the file**

Create `src/optim/pion.py`:

```python
"""Builder for the ``pion`` optimizer (Megatron integration).

Wraps the vendored single-process Pion algorithm (``src/optim/_pion.py``) in
Megatron's optimizer machinery as a ``ChainedOptimizer``: Pion drives the 2-D
matrix (non-embedding) weights and a stock Megatron AdamW drives everything else
(embeddings, norms, biases, LM head). Single-GPU dev scope only.

Mirrors the upstream ``get_megatron_pion_optimizer``
(third_party/pion/megatron-lm/megatron/core/optimizer/pion.py) but resolves the
Megatron optimizer primitives lazily from ``megatron.core.optimizer`` — the
UN-patched originals — so the chained-Adam call does NOT recurse back into the
``pion_optimizer_setup`` patch (which only rebinds the names in
``megatron.training.training``). Same no-recursion design as ``src/optim/poet.py``.
Reached via ``src/patches/pion_optimizer_setup.py``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, cast

logger = logging.getLogger(__name__)

# Lazy handles; populated by _resolve_megatron_handles on first build. Kept as
# module globals so unit tests can monkeypatch them without importing Megatron.
_get_param_groups = None
get_megatron_optimizer = None
ChainedOptimizer = None
Float16OptimizerWithFloat16Params = None
FP32Optimizer = None


def _resolve_megatron_handles() -> None:
    """Import Megatron optimizer primitives on first use.

    Resolved from ``megatron.core.optimizer`` (the originals), NOT from
    ``megatron.training.training`` (which the pion_optimizer_setup patch wraps) —
    so the chained-Adam build below does not recurse into the patch.
    """
    global _get_param_groups, get_megatron_optimizer, ChainedOptimizer
    global Float16OptimizerWithFloat16Params, FP32Optimizer
    if _get_param_groups is not None:
        return
    from megatron.core.optimizer import _get_param_groups as _gpg
    from megatron.core.optimizer import get_megatron_optimizer as _gmo
    from megatron.core.optimizer.optimizer import ChainedOptimizer as _Chained
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params as _F16,
    )
    from megatron.core.optimizer.optimizer import FP32Optimizer as _FP32

    _get_param_groups = _gpg
    get_megatron_optimizer = _gmo
    ChainedOptimizer = _Chained
    Float16OptimizerWithFloat16Params = _F16
    FP32Optimizer = _FP32


def _matrix_param_groups(
    model_chunks: List[Any],
    config: Any,
    config_overrides: Optional[dict],
    matrix_params: List[Any],
) -> List[dict]:
    """Restrict Megatron's standard param groups to the Pion matrix params."""
    matrix_param_ids = {id(p) for p in matrix_params}
    groups: List[dict] = []
    for group in _get_param_groups(model_chunks, config, config_overrides):
        params = [p for p in group["params"] if id(p) in matrix_param_ids]
        if params:
            new_group = dict(group)
            new_group["params"] = params
            groups.append(new_group)
    return groups


def get_megatron_pion_optimizer(
    config: Any,
    model_chunks: List[Any],
    *,
    config_overrides: Optional[dict] = None,
    use_gloo_process_groups: bool = True,
) -> Any:
    from megatron.core import parallel_state as mpu

    from src.optim._pion import PionOptimizer

    if config.use_distributed_optimizer:
        raise ValueError(
            "pion does not support the distributed optimizer (single-GPU dev only)."
        )
    if config.fp16:
        raise ValueError("pion does not support fp16; use bf16.")
    if mpu.get_tensor_model_parallel_world_size() > 1:
        raise ValueError("pion does not support tensor parallelism > 1 (single-GPU dev only).")
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        raise ValueError("pion does not support pipeline parallelism > 1 (single-GPU dev only).")

    _resolve_megatron_handles()

    # The Pion experiment routes through --optimizer adam (the stock path builds
    # an AdamOptimizerConfig). Keep config.optimizer == "adam" so the chained-Adam
    # build below takes the standard path.
    config.optimizer = "adam"

    matrix_params: List[Any] = []
    non_matrix_params: List[Any] = []
    qkv_split_shapes: Optional[Tuple[int, int, int]] = None
    split_fc1_up_gate = False

    for model_chunk in model_chunks:
        num_attention_heads = getattr(model_chunk.config, "num_attention_heads", None)
        num_query_groups = getattr(model_chunk.config, "num_query_groups", None)
        kv_channels = getattr(model_chunk.config, "kv_channels", None)
        if (
            num_attention_heads is not None
            and num_query_groups is not None
            and kv_channels is not None
        ):
            qkv_split_shapes = (
                num_attention_heads // num_query_groups * kv_channels,
                kv_channels,
                kv_channels,
            )
        gated_linear_unit = getattr(model_chunk.config, "gated_linear_unit", False)
        split_fc1_up_gate = gated_linear_unit and getattr(config, "pion_split_gate", True)

        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 2 and not getattr(
                param, "is_embedding_or_output_parameter", False
            ):
                setattr(param, "_pion_param_name", name)
                if "linear_qkv.weight" in name:
                    param.is_qkv = True
                if "linear_fc1.weight" in name and split_fc1_up_gate:
                    param.is_fc1_up_gate = True
                matrix_params.append(param)
            else:
                non_matrix_params.append(param)

    # Diagnostic: surface the routing split (mirror muon_kimi.py:101-109).
    logger.info(
        "pion: %d matrix params (2D non-embedding), %d adamw params",
        len(matrix_params),
        len(non_matrix_params),
    )
    if not matrix_params:
        logger.warning("pion: no 2D non-embedding params found — Pion is a no-op (pure AdamW).")

    lr = float(config.lr if config.lr is not None else 1e-4)
    matrix_param_groups = _matrix_param_groups(model_chunks, config, config_overrides, matrix_params)
    if not matrix_param_groups:
        matrix_param_groups = [
            {
                "params": matrix_params,
                "max_lr": lr,
                "min_lr": config.min_lr,
                "wd_mult": 1.0,
                "lr_mult": 1.0,
                "is_expert_parallel": False,
                "default_config": True,
            }
        ]

    degree = getattr(config, "pion_degree", 2)
    pion_scaling = getattr(config, "pion_scaling", "rms")
    pion_rms = getattr(config, "pion_rms", 0.2)
    pion_momentum = getattr(config, "pion_momentum", "none")
    pion_use_second_momentum = getattr(config, "pion_use_second_momentum", None)
    pion_update_side = getattr(config, "pion_update_side", "both")
    pion_qkv_split_granularity = getattr(config, "pion_qkv_split_granularity", None)
    if pion_qkv_split_granularity is None:
        pion_qkv_split_granularity = (
            "head" if getattr(config, "pion_split_qkv_per_head", True) else "qkv"
        )
    pion_exp_map = getattr(config, "pion_exp_map", "exp_truncated")
    adam_eps = getattr(config, "adam_eps", 1e-8)
    pion_beta1 = getattr(config, "pion_beta1", 0.9)
    pion_beta2 = getattr(config, "pion_beta2", 0.999)

    for group in matrix_param_groups:
        group["degree"] = degree
        group["pion_scaling"] = pion_scaling
        group["pion_rms"] = pion_rms
        group["pion_momentum"] = pion_momentum
        group["pion_use_second_momentum"] = pion_use_second_momentum
        group["pion_update_side"] = pion_update_side
        group["pion_qkv_split_granularity"] = pion_qkv_split_granularity
        group["pion_12_momentum"] = getattr(config, "pion_12_momentum", "none")
        group["pion_first_momentum"] = getattr(config, "pion_first_momentum", "none")
        group["pion_second_momentum"] = getattr(config, "pion_second_momentum", "none")
        group["pion_exp_map"] = pion_exp_map
        group["pion_update_csv"] = getattr(config, "pion_update_csv", None)
        group["pion_update_csv_interval"] = getattr(config, "pion_update_csv_interval", 1)
        group["adam_eps"] = adam_eps
        group["pion_beta1"] = pion_beta1
        group["pion_beta2"] = pion_beta2

    pion_optimizer = PionOptimizer(
        matrix_param_groups,
        lr=lr,
        betas=(pion_beta1, pion_beta2),
        weight_decay=config.weight_decay,
        degree=degree,
        split_qkv=getattr(config, "pion_split_qkv", True),
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=qkv_split_shapes,
        split_fc1_up_gate=split_fc1_up_gate,
        is_fc1_up_gate_fn=lambda p: getattr(p, "is_fc1_up_gate", False),
        split_qkv_per_head=getattr(config, "pion_split_qkv_per_head", True),
        qkv_split_granularity=pion_qkv_split_granularity,
        pion_scaling=pion_scaling,
        pion_rms=pion_rms,
        pion_momentum=pion_momentum,
        pion_update_side=pion_update_side,
        pion_beta1=pion_beta1,
        pion_beta2=pion_beta2,
    )

    def pion_init_state_fn(opt, _config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                if len(opt.state[p]) == 0:
                    opt.state[p]["step"] = 0

    if config.bf16:
        wrapped = Float16OptimizerWithFloat16Params(pion_optimizer, config, None, pion_init_state_fn)
    else:
        wrapped = FP32Optimizer(pion_optimizer, config, pion_init_state_fn)

    optimizers: List[Any] = [wrapped]

    # Build the stock Megatron AdamW for the NON-matrix params. Freeze the matrix
    # params first so _get_param_groups (which skips requires_grad=False) hands the
    # standard Adam path only the embeddings/norms/biases/head; unfreeze after.
    for p in matrix_params:
        p.requires_grad = False
    chained_adam = cast(
        Any,
        get_megatron_optimizer(
            config,
            model_chunks,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        ),
    )
    for p in matrix_params:
        p.requires_grad = True

    optimizers += chained_adam.chained_optimizers
    setattr(wrapped, "grad_stats_parallel_group", mpu.get_model_parallel_group())
    setattr(wrapped, "tp_group", mpu.get_tensor_model_parallel_group())
    return ChainedOptimizer(optimizers)


__all__ = ["get_megatron_pion_optimizer"]
```

- [ ] **Step 2: Compile**

Run: `python -m py_compile src/optim/pion.py`
Expected: silent (exit 0).

- [ ] **Step 3: Lint**

Run: `ruff check src/optim/pion.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/optim/pion.py
git commit -m "feat(pion): Megatron builder — Pion(matrix)+AdamW(rest) chained optimizer"
```

---

## Task 3: Optimizer-setup patch (`src/patches/pion_optimizer_setup.py`)

**Files:**
- Create: `src/patches/pion_optimizer_setup.py`
- Test: `tests/unit/test_patch_pion_optimizer_setup.py`

**Interfaces:**
- Consumes: `register_patch` from `src/patches/_registry.py`; the builder `src.optim.pion.get_megatron_pion_optimizer` (Task 2).
- Produces: a registered patch named `"pion_optimizer_setup"` targeting `megatron.training.training.get_megatron_optimizer_config` and `...get_megatron_optimizer`. The wrapped `get_megatron_optimizer_config` tags `config.slm_optimizer="pion"` AND copies the `pion_*` attributes from `args` onto `config` (they are not declared `OptimizerConfig` fields, so the stock arg→field copy skips them).

- [ ] **Step 1: Write the failing test**

Mirror `tests/unit/test_patch_muon_kimi_optimizer_setup.py` (same mocked-Megatron structure). Key additions: assert `pion_*` args get copied onto config.

```python
# tests/unit/test_patch_pion_optimizer_setup.py
"""Tests for the pion optimizer setup patch."""

from __future__ import annotations

import importlib
import sys
import types

from src.patches._registry import _reset_for_tests


def test_pion_optimizer_setup_registers_targets():
    _reset_for_tests()
    sys.modules.pop("src.patches.pion_optimizer_setup", None)

    importlib.import_module("src.patches.pion_optimizer_setup")

    from src.patches import registered_patches

    entry = registered_patches()["pion_optimizer_setup"]
    assert "megatron.training.training.get_megatron_optimizer_config" in entry.targets
    assert "megatron.training.training.get_megatron_optimizer" in entry.targets


def test_pion_setup_tags_config_and_copies_pion_args(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.pion_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.pion_optimizer_setup")

    calls = []
    fake_training = types.SimpleNamespace()

    def original_get_config(args):
        cfg = types.SimpleNamespace(optimizer="adam", lr=1.0e-3)
        return cfg, {"from": "original"}

    def original_get_optimizer(config, model, **kwargs):
        calls.append(("original", config, model, kwargs))
        return "adam-optimizer"

    fake_training.get_megatron_optimizer_config = original_get_config
    fake_training.get_megatron_optimizer = original_get_optimizer

    fake_builder = types.ModuleType("src.optim.pion")

    def fake_pion_builder(config, model_chunks, **kwargs):
        calls.append(("pion", config, model_chunks, kwargs))
        return "pion-optimizer"

    fake_builder.get_megatron_pion_optimizer = fake_pion_builder

    fake_megatron = types.ModuleType("megatron")
    fake_megatron_training_pkg = types.ModuleType("megatron.training")
    fake_megatron_training_pkg.training = fake_training
    fake_megatron.training = fake_megatron_training_pkg
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", fake_megatron_training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", fake_training)
    monkeypatch.setitem(sys.modules, "src.optim.pion", fake_builder)

    patch_mod.apply()

    # --- tag + copy path: slm_optimizer == "pion" ---
    args = types.SimpleNamespace(
        slm_optimizer="pion",
        pion_scaling="rms",
        pion_rms=0.2,
        pion_update_side="alternate",
        pion_momentum="transported_ambient_ambient",
        pion_degree=2,
        pion_beta1=0.9,
        pion_beta2=0.95,
        pion_use_second_momentum=False,
    )
    cfg, overrides = fake_training.get_megatron_optimizer_config(args)
    assert overrides == {"from": "original"}
    assert cfg.slm_optimizer == "pion"
    assert cfg.pion_scaling == "rms"
    assert cfg.pion_update_side == "alternate"
    assert cfg.pion_momentum == "transported_ambient_ambient"
    assert cfg.pion_beta2 == 0.95

    out = fake_training.get_megatron_optimizer(cfg, ["model"], use_gloo_process_groups=False)
    assert out == "pion-optimizer"
    assert calls[-1][0] == "pion"

    # --- delegate-to-original path: slm_optimizer != "pion" ---
    args_other = types.SimpleNamespace(slm_optimizer="adam")
    cfg_other, _ = fake_training.get_megatron_optimizer_config(args_other)
    assert not hasattr(cfg_other, "slm_optimizer") or cfg_other.slm_optimizer != "pion"

    out_other = fake_training.get_megatron_optimizer(cfg_other, ["model"])
    assert out_other == "adam-optimizer"
    assert calls[-1][0] == "original"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_patch_pion_optimizer_setup.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.patches.pion_optimizer_setup'`

- [ ] **Step 3: Write the patch**

Create `src/patches/pion_optimizer_setup.py`:

```python
"""Patch: route slm-research ``pion`` optimizer through Megatron's Adam branch.

slm-research passes ``--optimizer adam --slm-optimizer pion``; this patch tags
the OptimizerConfig, copies the ``pion_*`` CLI knobs onto it (they are not
declared OptimizerConfig dataclass fields, so the stock arg->field copy in
``get_megatron_optimizer_config`` skips them), and reroutes the optimizer-builder
call to ``src.optim.pion.get_megatron_pion_optimizer``. Mirrors
``muon_kimi_optimizer_setup``.
"""

from __future__ import annotations

from src.patches._registry import register_patch

# NOTE: poet_optimizer_setup / muon_kimi_optimizer_setup target these same two
# functions. The patch registry raises PatchConflict if two are registered at
# once, but that never happens in a real run (one experiment's patches load per
# process). Never list pion_optimizer_setup together with another
# *_optimizer_setup patch in experiment.patches.
_TARGET = (
    "megatron.training.training.get_megatron_optimizer_config",
    "megatron.training.training.get_megatron_optimizer",
)

# pion_* args copied from the parsed CLI args onto the OptimizerConfig so the
# builder (src/optim/pion.py) can read them via getattr. Keep in sync with the
# args registered in launchers/pretrain_gpt_slm.py:add_slm_args.
_PION_CONFIG_ATTRS = (
    "pion_scaling",
    "pion_rms",
    "pion_update_side",
    "pion_momentum",
    "pion_degree",
    "pion_beta1",
    "pion_beta2",
    "pion_use_second_momentum",
    "pion_qkv_split_granularity",
    "pion_split_qkv",
    "pion_split_gate",
    "pion_split_qkv_per_head",
    "pion_exp_map",
)


@register_patch(name="pion_optimizer_setup", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt

    _orig_get_config = _mt.get_megatron_optimizer_config
    _orig_get_optimizer = _mt.get_megatron_optimizer

    def _wrapped_get_config(args):
        config, overrides = _orig_get_config(args)
        if getattr(args, "slm_optimizer", "") != "pion":
            return config, overrides
        config.slm_optimizer = "pion"
        for attr in _PION_CONFIG_ATTRS:
            if hasattr(args, attr):
                setattr(config, attr, getattr(args, attr))
        return config, overrides

    def _wrapped_get_optimizer(config, model, **kwargs):
        if getattr(config, "slm_optimizer", "") != "pion":
            return _orig_get_optimizer(config, model, **kwargs)
        from src.optim.pion import get_megatron_pion_optimizer

        return get_megatron_pion_optimizer(
            config,
            model,
            config_overrides=kwargs.get("config_overrides"),
            use_gloo_process_groups=kwargs.get("use_gloo_process_groups", True),
        )

    _mt.get_megatron_optimizer_config = _wrapped_get_config
    _mt.get_megatron_optimizer = _wrapped_get_optimizer
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/test_patch_pion_optimizer_setup.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Compile + lint**

Run: `python -m py_compile src/patches/pion_optimizer_setup.py && ruff check src/patches/pion_optimizer_setup.py tests/unit/test_patch_pion_optimizer_setup.py`
Expected: silent / no errors.

- [ ] **Step 6: Commit**

```bash
git add src/patches/pion_optimizer_setup.py tests/unit/test_patch_pion_optimizer_setup.py
git commit -m "feat(pion): optimizer-setup patch — tag config, copy pion_* args, reroute builder"
```

---

## Task 4: Register Pion CLI args in the launcher (`launchers/pretrain_gpt_slm.py`)

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py:26-30` (add `"pion"` choice) and `:158-160` area (add `--pion-*` args inside `add_slm_args`).

**Interfaces:**
- Produces: CLI flags on the Megatron `args` namespace: `--pion-scaling`, `--pion-rms`, `--pion-update-side`, `--pion-momentum`, `--pion-degree`, `--pion-beta1`, `--pion-beta2`, `--pion-use-second-momentum` (store_true), `--pion-qkv-split-granularity`, plus split toggles. Consumed by Task 3's patch (`_PION_CONFIG_ATTRS`) and Task 5's argv builder.

- [ ] **Step 1: Write the failing test**

Add to a new test file `tests/unit/test_pion_launcher_args.py`:

```python
# tests/unit/test_pion_launcher_args.py
"""The launcher's add_slm_args registers Pion flags + the pion slm-optimizer choice."""

from __future__ import annotations

import argparse

from launchers.pretrain_gpt_slm import add_slm_args


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    return parser.parse_args(["--slm-config-path", "x", *argv])


def test_pion_is_a_valid_slm_optimizer_choice():
    args = _parse(["--slm-optimizer", "pion"])
    assert args.slm_optimizer == "pion"


def test_pion_args_have_reference_defaults():
    args = _parse([])
    assert args.pion_scaling == "rms"
    assert args.pion_rms == 0.2
    assert args.pion_update_side == "both"
    assert args.pion_momentum == "transported_ambient_ambient"
    assert args.pion_degree == 2
    assert args.pion_beta1 == 0.9
    assert args.pion_beta2 == 0.999
    assert args.pion_use_second_momentum is False


def test_pion_args_are_overridable():
    args = _parse([
        "--pion-update-side", "alternate",
        "--pion-momentum", "lie_lie",
        "--pion-beta2", "0.95",
        "--pion-use-second-momentum",
    ])
    assert args.pion_update_side == "alternate"
    assert args.pion_momentum == "lie_lie"
    assert args.pion_beta2 == 0.95
    assert args.pion_use_second_momentum is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_pion_launcher_args.py -q`
Expected: FAIL — `test_pion_is_a_valid_slm_optimizer_choice` raises `SystemExit` (invalid choice `pion`); the others raise `AttributeError: 'Namespace' object has no attribute 'pion_scaling'`.

- [ ] **Step 3: Add `"pion"` to the `--slm-optimizer` choices**

In `launchers/pretrain_gpt_slm.py`, change the `--slm-optimizer` choices (currently `["adamw", "muon", "poet", "ngpt_adamw", "muon_kimi"]`):

```python
    group.add_argument(
        "--slm-optimizer",
        choices=["adamw", "muon", "poet", "ngpt_adamw", "muon_kimi", "pion"],
        default="adamw",
    )
```

- [ ] **Step 4: Register the `--pion-*` args**

In `add_slm_args`, immediately AFTER the `--unfuse-fc1` argument (around line 159) — i.e. still inside `add_slm_args`, before the function returns — insert:

```python
    # --- Pion optimizer (src/optim/pion.py; --slm-optimizer pion). Registered
    # here (not in the Megatron-LM submodule) so Pion lives entirely in
    # slm-research. Copied onto the OptimizerConfig at runtime by the
    # pion_optimizer_setup patch (the pion_* names are not OptimizerConfig
    # fields). Defaults match third_party/pion/.../opt_llama_60M_pion.sh. ---
    group.add_argument("--pion-scaling", type=str, default="rms", choices=["rms", "none"])
    group.add_argument("--pion-rms", type=float, default=0.2)
    group.add_argument(
        "--pion-update-side",
        type=str,
        default="both",
        choices=["both", "alternate", "in", "out"],
    )
    group.add_argument(
        "--pion-momentum",
        type=str,
        default="transported_ambient_ambient",
        choices=["lie_lie", "transported_ambient_ambient"],
    )
    group.add_argument("--pion-degree", type=int, default=2)
    group.add_argument("--pion-beta1", type=float, default=0.9)
    group.add_argument("--pion-beta2", type=float, default=0.999)
    group.add_argument("--pion-use-second-momentum", action="store_true")
    # QKV update granularity: head | qkv | group. None follows the per-head default.
    group.add_argument(
        "--pion-qkv-split-granularity",
        type=str,
        default=None,
        choices=["head", "qkv", "group"],
    )
    group.add_argument("--pion-no-split-qkv", action="store_false", dest="pion_split_qkv", default=True)
    group.add_argument("--pion-no-split-gate", action="store_false", dest="pion_split_gate", default=True)
    group.add_argument(
        "--pion-no-split-qkv-per-head",
        action="store_false",
        dest="pion_split_qkv_per_head",
        default=True,
    )
    group.add_argument("--pion-exp-map", type=str, default="exp_truncated", choices=["exp_truncated", "taylor"])
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/unit/test_pion_launcher_args.py -q`
Expected: `3 passed`.

- [ ] **Step 6: Compile + lint**

Run: `python -m py_compile launchers/pretrain_gpt_slm.py && ruff check launchers/pretrain_gpt_slm.py tests/unit/test_pion_launcher_args.py`
Expected: silent / no errors.

- [ ] **Step 7: Commit**

```bash
git add launchers/pretrain_gpt_slm.py tests/unit/test_pion_launcher_args.py
git commit -m "feat(pion): register --pion-* CLI args + pion slm-optimizer choice"
```

---

## Task 5: Emit Pion argv from config (`src/utils/megatron_args.py`)

**Files:**
- Modify: `src/utils/megatron_args.py` — add a `kind == "pion"` branch in `_optimizer_args` (after the `muon_kimi` branch, around line 370).
- Test: `tests/unit/test_megatron_args.py` (add one test).

**Interfaces:**
- Consumes: `cfg.optim` with `type="pion"` and the Pion sub-knobs. Emits `--optimizer adam --slm-optimizer pion` + the `--pion-*` flags registered in Task 4 + adam betas/eps. (As with `muon_kimi`, `--lr` and `--weight-decay` are emitted by the training/scheduler arg builders, not here.)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_megatron_args.py` (after `test_muon_kimi_argv_routes_through_adam_and_sets_muon_knobs`, ~line 598):

```python
def test_pion_argv_routes_through_adam_and_sets_pion_knobs():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "pion",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "pion_scaling": "rms",
                "pion_rms": 0.2,
                "pion_update_side": "alternate",
                "pion_momentum": "transported_ambient_ambient",
                "pion_degree": 2,
                "pion_beta1": 0.9,
                "pion_beta2": 0.95,
                "pion_use_second_momentum": False,
                "adam": {"betas": [0.9, 0.95], "eps": 1.0e-8},
            }
        }
    )
    args = _optimizer_args(cfg)
    amap = {args[i]: args[i + 1] for i in range(0, len(args) - 1)}
    assert amap["--optimizer"] == "adam"
    assert amap["--slm-optimizer"] == "pion"
    assert amap["--pion-scaling"] == "rms"
    assert amap["--pion-rms"] == "0.2"
    assert amap["--pion-update-side"] == "alternate"
    assert amap["--pion-momentum"] == "transported_ambient_ambient"
    assert amap["--pion-degree"] == "2"
    assert amap["--pion-beta1"] == "0.9"
    assert amap["--pion-beta2"] == "0.95"
    assert amap["--adam-beta1"] == "0.9"
    assert amap["--adam-beta2"] == "0.95"
    assert "--pion-use-second-momentum" not in args


def test_pion_argv_emits_second_momentum_flag_when_enabled():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {"optim": {"type": "pion", "lr": 1e-3, "weight_decay": 0.1,
                   "pion_use_second_momentum": True}}
    )
    assert "--pion-use-second-momentum" in _optimizer_args(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_megatron_args.py -k pion -q`
Expected: FAIL with `ValueError: Unsupported optimizer type 'pion'` (the final `raise` in `_optimizer_args`).

- [ ] **Step 3: Add the `pion` branch**

In `src/utils/megatron_args.py`, insert this branch in `_optimizer_args` immediately after the `muon_kimi` branch's `return _sequence(argv)` (after line 370, before `if kind == "poet":`):

```python
    if kind == "pion":
        adam = optim.get("adam", {})
        betas = adam.get("betas", [0.9, 0.95])
        argv = [
            "--optimizer",
            "adam",
            "--slm-optimizer",
            "pion",
            "--pion-scaling",
            optim.get("pion_scaling", "rms"),
            "--pion-rms",
            optim.get("pion_rms", 0.2),
            "--pion-update-side",
            optim.get("pion_update_side", "both"),
            "--pion-momentum",
            optim.get("pion_momentum", "transported_ambient_ambient"),
            "--pion-degree",
            optim.get("pion_degree", 2),
            "--pion-beta1",
            optim.get("pion_beta1", 0.9),
            "--pion-beta2",
            optim.get("pion_beta2", 0.999),
            "--adam-beta1",
            betas[0],
            "--adam-beta2",
            betas[1],
            "--adam-eps",
            adam.get("eps", 1.0e-8),
        ]
        granularity = optim.get("pion_qkv_split_granularity", None)
        if granularity is not None:
            argv += ["--pion-qkv-split-granularity", granularity]
        if bool(optim.get("pion_use_second_momentum", False)):
            argv.append("--pion-use-second-momentum")
        return _sequence(argv)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/test_megatron_args.py -k pion -q`
Expected: `2 passed`.

- [ ] **Step 5: Run the full megatron_args suite (no regression)**

Run: `python -m pytest tests/unit/test_megatron_args.py -q`
Expected: all pass (note: per the project memory, 3 pre-existing failures in `test_megatron_args.py` may exist unrelated to Pion — confirm the count is unchanged vs. `git stash` baseline if any fail).

- [ ] **Step 6: Compile + lint + commit**

```bash
python -m py_compile src/utils/megatron_args.py
ruff check src/utils/megatron_args.py
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(pion): emit pion argv from optim.type=pion in megatron_args"
```

---

## Task 6: Pion experiment configs (fused default + unfused variant)

**Files:**
- Create: `configs/experiments/optim/pion.yaml` (FUSED default — Pion-native internal split)
- Create: `configs/experiments/optim/pion_unfused.yaml` (UNFUSED variant — opt-in ablation)

**Interfaces:**
- Consumes: the `optim.type=pion` argv branch (Task 5), the `pion_optimizer_setup` patch (Task 3), and the always-no-op-unless-flagged `model_unfuse_linears` patch.
- Produces: two Hydra experiments selectable via `experiment=optim/pion` (fused) and `experiment=optim/pion_unfused` (unfused).

- [ ] **Step 1: Write the fused (default) config**

Create `configs/experiments/optim/pion.yaml`. Note it LISTS `model_unfuse_linears` (a no-op unless `base.model.unfuse_qkv`/`unfuse_fc1` are set — same pattern as `adam.yaml:16-19`), so a fused-default run can opt into unfusing at the CLI without editing the config:

```yaml
# @package _global_
# Pion optimizer (vendored src/optim/_pion.py, routed via --slm-optimizer pion).
# Single-GPU dev only. Pion drives 2-D matrix (non-embedding) weights via the
# orthogonal-equivalence update W <- W exp(A_in) + exp(A_out) W - W (truncated
# matrix exponential); a chained Megatron AdamW drives embeddings, norms, biases,
# and the LM head. Defaults match third_party/pion/.../opt_llama_60M_pion.sh.
#
# FUSED by default: Pion splits qkv per-head and fc1 up/gate INTERNALLY in the
# optimizer (Pion-native granularity). The model_unfuse_linears patch is listed
# but is a NO-OP unless base.model.unfuse_qkv / unfuse_fc1 are set — so you can
# opt into the unfused path on the CLI (base.model.unfuse_qkv=true
# base.model.unfuse_fc1=true) or via experiment=optim/pion_unfused. When unfused,
# Pion's internal split goes inert (the fused linear_qkv/linear_fc1 names are
# gone) and each separate q/k/v/up/gate is rotated as a plain matrix.
experiment:
  name: pion
  family: optim
  description: |
    Pion: a spectrum-preserving optimizer via orthogonal equivalence
    transformation (Shi, Li, Qiu, Wen, Buchholz, Liu). Matrix weights rotated by
    two Lie generators; non-matrix params on AdamW. Single base LR for both
    sides. Single-GPU dev. Fused qkv/fc1 (Pion-native internal split).
  references:
    - "Pion (Sphere-AI-Lab/pion) — transported_ambient_ambient + lie_lie"
  patches:
    - pion_optimizer_setup
    - model_unfuse_linears    # NO-OP unless base.model.unfuse_qkv/unfuse_fc1 set
    - training_log_eta        # prepend "ETA: HhMMm" to the per-iteration log
    - wandb_metric_normalize  # canonicalize W&B metric keys + add tokens_seen / step_time
  required_capabilities: []

optim:
  type: pion
  lr: 1.0e-3
  weight_decay: 0.1
  pion_scaling: rms
  pion_rms: 0.2
  pion_update_side: alternate
  pion_momentum: transported_ambient_ambient
  pion_degree: 2
  pion_beta1: 0.9
  pion_beta2: 0.95
  pion_use_second_momentum: false
  adam:
    betas: [0.9, 0.95]
    eps: 1.0e-8
```

- [ ] **Step 2: Write the unfused variant config**

Create `configs/experiments/optim/pion_unfused.yaml` — identical to `pion.yaml` except `experiment.name`, the description, and the added `base.model` unfuse flags (this turns the `model_unfuse_linears` patch ON, so Pion rotates each separate q/k/v/up/gate as a whole matrix — the muon-like granularity, for ablation against the fused default):

```yaml
# @package _global_
# UNFUSED Pion variant. Identical to optim/pion.yaml but unfuses qkv/fc1 at
# model-build time (model_unfuse_linears patch), so Pion treats each separate
# q/k/v/up/gate projection as a plain 2-D matrix instead of splitting a fused
# weight internally. Use to ablate Pion's internal per-head/up-gate split
# (fused, the default) against whole-projection rotation (unfused).
experiment:
  name: pion_unfused
  family: optim
  description: |
    Pion (unfused): qkv/fc1 unfused at build time so each separate projection is
    rotated as a plain matrix (muon-like granularity). Ablation against the fused
    optim/pion default, which splits qkv per-head and fc1 up/gate internally.
  references:
    - "Pion (Sphere-AI-Lab/pion) — transported_ambient_ambient + lie_lie"
  patches:
    - pion_optimizer_setup
    - model_unfuse_linears    # ACTIVE here: base.model.unfuse_* are set below
    - training_log_eta
    - wandb_metric_normalize
  required_capabilities: []

optim:
  type: pion
  lr: 1.0e-3
  weight_decay: 0.1
  pion_scaling: rms
  pion_rms: 0.2
  pion_update_side: alternate
  pion_momentum: transported_ambient_ambient
  pion_degree: 2
  pion_beta1: 0.9
  pion_beta2: 0.95
  pion_use_second_momentum: false
  adam:
    betas: [0.9, 0.95]
    eps: 1.0e-8

# Unfuse fused qkv/fc1 into separate projections (forward-equivalent). Applied by
# the model_unfuse_linears patch above; turns Pion's internal split inert.
base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 3: Write the matching experiment docs (REQUIRED by the `experiment-doc-exists` pre-commit hook)**

`tools/check_experiment_docs.py` (pre-commit hook `experiment-doc-exists`) fails the commit unless every `configs/experiments/*.yaml` has a `docs/experiments/<experiment.name>.md` AND a non-empty `experiment.description`. Create both docs (mirror `docs/experiments/muon_kimi.md`):

Create `docs/experiments/pion.md`:

```markdown
# pion

Pion optimizer (`src/optim/_pion.py`, vendored from Sphere-AI-Lab/pion) — a
spectrum-preserving optimizer via orthogonal equivalence transformation. Matrix
weights are rotated by two Lie generators `W <- W exp(A_in) + exp(A_out) W - W`
(truncated matrix exponential); a chained Megatron AdamW drives the rest.

- **Scope:** single GPU (DP=TP=PP=1). The builder (`src/optim/pion.py`) raises on
  tensor parallelism, pipeline parallelism, the distributed optimizer, or fp16.
- **Param routing:** 2-D non-embedding/output weights → Pion; embeddings,
  lm_head, norms, biases → chained AdamW (`ChainedOptimizer`, like POET).
- **Fused (default):** qkv split per-head and fc1 up/gate split happen INSIDE the
  optimizer. The `model_unfuse_linears` patch is listed but no-op unless
  `base.model.unfuse_qkv`/`unfuse_fc1` are set — see `pion_unfused` for the
  opt-in unfused variant.
- **LR:** one `optim.lr` for both sides (the Pion side is RMS-scaled internally by
  `pion_rms*sqrt(m*n)`).
- **Defaults:** `pion_momentum=transported_ambient_ambient`,
  `pion_update_side=alternate`, `pion_scaling=rms`, `pion_rms=0.2`,
  `pion_degree=2`, betas `(0.9, 0.95)` — from `opt_llama_60M_pion.sh`.
- **Wiring:** `--optimizer adam --slm-optimizer pion`, rerouted by the
  `pion_optimizer_setup` patch.

Run on the 60m dev model with `scripts/train_pion_dev.sh`; sweep LR with
`scripts/sweep_pion_lr.sh`.
```

Create `docs/experiments/pion_unfused.md`:

```markdown
# pion_unfused

Unfused variant of `pion` (see `docs/experiments/pion.md`). Identical Pion
optimizer and defaults, but `base.model.unfuse_qkv`/`unfuse_fc1` are set so the
`model_unfuse_linears` patch splits the fused qkv/fc1 weights into separate
projections at model-build time.

- **Effect:** Pion's internal per-head qkv / up-gate split goes inert (the fused
  `linear_qkv`/`linear_fc1` tensors no longer exist); each separate q/k/v/up/gate
  is instead rotated as a plain 2-D matrix (muon-like granularity).
- **Use:** ablation against the fused `pion` default to compare internal-split vs
  whole-projection rotation. Same single-GPU scope and wiring as `pion`.

Run with `scripts/train_pion_dev.sh experiment=optim/pion_unfused`.
```

- [ ] **Step 4: Verify both configs resolve + the doc hook passes**

Run:
```bash
python -c "
from omegaconf import OmegaConf
f = OmegaConf.load('configs/experiments/optim/pion.yaml')
u = OmegaConf.load('configs/experiments/optim/pion_unfused.yaml')
print('fused  :', f.optim.type, f.optim.pion_momentum, list(f.experiment.patches))
print('unfused:', u.optim.type, u.experiment.name, bool(u.base.model.unfuse_qkv), bool(u.base.model.unfuse_fc1))
assert 'model_unfuse_linears' in f.experiment.patches
assert u.base.model.unfuse_qkv is True and u.base.model.unfuse_fc1 is True
print('OK')
"
python3 tools/check_experiment_docs.py && echo "DOC HOOK OK"
```
Expected: prints the fused/unfused summaries, `OK`, then `DOC HOOK OK` (exit 0). The fused config must NOT define `base.model.unfuse_qkv` (defaults off); the unfused config must set both flags True.

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/optim/pion.yaml configs/experiments/optim/pion_unfused.yaml \
        docs/experiments/pion.md docs/experiments/pion_unfused.md
git commit -m "feat(pion): add optim/pion (fused) + optim/pion_unfused experiment configs + docs"
```

---

## Task 7: Dev training script (`scripts/train_pion_dev.sh`)

**Files:**
- Create: `scripts/train_pion_dev.sh` (chmod +x)
- Test: `tests/unit/test_train_scripts.py` (add a pion case)

**Interfaces:**
- Consumes: `experiment=optim/pion`, the 60m dev scale, `ablation_40x` regime. Mirrors `scripts/train_muon_dev.sh` exactly (single-GPU dev), differing only in `experiment=optim/pion` and the comment header.
- The script ends with `"$@"`, so the **unfused** path is reachable without a second script: either select the variant config — `scripts/train_pion_dev.sh experiment=optim/pion_unfused` — or override the fused default in place — `scripts/train_pion_dev.sh base.model.unfuse_qkv=true base.model.unfuse_fc1=true`.

- [ ] **Step 1: Write the script**

Create `scripts/train_pion_dev.sh` (copy `scripts/train_muon_dev.sh` verbatim, change only the header comment and the `experiment=` line):

```bash
#!/usr/bin/env bash
set -euo pipefail

# Dev launcher for the vendored Pion optimizer (experiment=optim/pion),
# single-GPU only. Defaults to the tiny 60m scale (configs/base/scale/60m.yaml,
# hidden=512, ~61M non-embedding params) for fast local iteration. Untied
# embeddings are forced on (overridable); any "$@" override still wins.

# torchtitan is AdamW-only in milestone 1; reject --backend torchtitan here so the
# same flag fails fast on this non-AdamW wrapper (see scripts/train_adam.sh).
case " $* " in
  *" --backend torchtitan "*|*" --backend=torchtitan "*)
    echo "This optimizer is not yet supported on torchtitan (milestone 1 is AdamW only)." >&2
    exit 2 ;;
esac

# Auto-source the cluster env loader so the user doesn't have to remember.
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" ]]; then
  shift || true
else
  ARCH="llama3"
fi

case "${ARCH}" in
  llama3)
    FAMILY="llama3"
    DEFAULT_SCALE="60m"            # tiny dev scale; override with base/scale=...
    ;;
  deepseek_v3)
    FAMILY="deepseek_v3"
    DEFAULT_SCALE="deepseek_v3_proxy_small"
    ;;
  *)
    echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2
    exit 2
    ;;
esac

# Inject dev defaults unless overridden on the command line:
#   scale=60m (tiny dev scale) and the 40x-tokens-per-param dev regime.
USER_SET_SCALE="no"
USER_SET_REGIME="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
    training_regime=*) USER_SET_REGIME="yes" ;;
  esac
done

SCALE_ARGS=()
if [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]]; then
  SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
fi

REGIME_ARGS=()
if [[ "${USER_SET_REGIME}" == "no" ]]; then
  REGIME_ARGS=("training_regime=ablation_40x")
fi

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${REGIME_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=optim/pion" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=128" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "base.model.tie_embeddings=false" \
  "wandb.project=slm-zeju-dev" \
  "$@"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/train_pion_dev.sh`
Expected: no output.

- [ ] **Step 3: Add the train-script test**

`_run` (in `tests/unit/test_train_scripts.py:11-25`) calls the script with `--dry-run cluster.nodes=1 cluster.gpus_per_node=1` and `check=True`; the launcher's `--dry-run` prints the resolved command JSON (containing `"command"` and the full argv) instead of launching training, so it runs fine on the CPU dev box (the script still sources `load_cuda13_2_nccl_env.sh`, which succeeds without a GPU). Mirror `test_muon_kimi_dev_script_routes_to_kimi_optimizer` (`:67-72`) exactly:

```python
def test_pion_dev_script_routes_to_pion_optimizer():
    proc = _run("train_pion_dev.sh", "llama3")
    assert '"command"' in proc.stdout
    assert "--slm-optimizer" in proc.stdout
    assert "pion" in proc.stdout
    assert "slm-zeju-dev" in proc.stdout
```

This dry-run transitively exercises `_optimizer_args` (Task 5) and resolves `experiment=optim/pion` (Task 6), so both must be complete before this test passes.

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/unit/test_train_scripts.py -k pion -q`
Expected: `1 passed`.

- [ ] **Step 5: Shellcheck + commit**

```bash
bash -n scripts/train_pion_dev.sh   # syntax check
git add scripts/train_pion_dev.sh tests/unit/test_train_scripts.py
git commit -m "feat(pion): add single-GPU dev training script scripts/train_pion_dev.sh"
```

---

## Task 8: LR sweep script (`scripts/sweep_pion_lr.sh`)

**Files:**
- Create: `scripts/sweep_pion_lr.sh` (chmod +x)

**Interfaces:**
- Consumes: `scripts/train_pion_dev.sh` (Task 7). Varies ONLY `optim.lr`; all Pion knobs stay at the `pion.yaml` defaults. Mirrors `scripts/sweep_muon_kimi_lr.sh` (codexlog tee per the project convention).

- [ ] **Step 1: Write the sweep script**

Create `scripts/sweep_pion_lr.sh`:

```bash
#!/usr/bin/env bash
# Pion LEARNING-RATE sweep — everything else at the optim/pion defaults.
# Run on one node (sequential runs, each grabs the whole node and blocks):
#   bash scripts/sweep_pion_lr.sh
#
# Baseline being tuned: pion (vendored src/optim/_pion.py; Pion on 2-D attn/MLP
# weights, chained AdamW on embeddings/norms/biases/head). ONLY optim.lr changes;
# pion_scaling (rms), pion_rms (0.2), pion_update_side (alternate),
# pion_momentum (transported_ambient_ambient), pion_degree (2), betas, and the
# stock cosine schedule (min_lr 0.1) are all left at the optim/pion defaults.
#
# Launcher = scripts/train_pion_dev.sh, which reproduces the dev cohort exactly:
# experiment=optim/pion, llama3-60m, ablation_40x (40 tpp), seq 256, gbs 1024,
# mbs 128, transformer_impl=local, tie_embeddings=false. Pion uses ONE base
# optim.lr for BOTH the Pion side (scaled internally by pion_rms*sqrt(m*n)) and
# the chained-AdamW side.
#
#   lr      note
#   5e-4    cooler probe
#   1e-3    reference default (opt_llama_60M_pion.sh)
#   2e-3    hotter probe
#   3e-3    hottest probe
# Each run uses experiment.name=pion (distinct run dirs by timestamp).

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOGDIR=/lustre/home/zqiu/log
mkdir -p "$LOGDIR"

codexlog() {
  local name="$1"; shift
  echo ">>> START ${name}  $(date '+%F %T')"
  "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
  echo "<<< END   ${name}  (status ${PIPESTATUS[0]})  $(date '+%F %T')"
}

LRS=(0.0005 0.001 0.002 0.003)

for lr in "${LRS[@]}"; do
  name="pion_lr${lr}"
  echo "### ${name}: lr=${lr} (all else = optim/pion defaults)"
  codexlog "$name" scripts/train_pion_dev.sh \
    optim.lr="$lr" experiment.name="pion"
done

echo "=== pion LR sweep complete (${#LRS[@]} runs) ==="
```

- [ ] **Step 2: Make executable + syntax check**

Run: `chmod +x scripts/sweep_pion_lr.sh && bash -n scripts/sweep_pion_lr.sh`
Expected: no output (valid syntax).

- [ ] **Step 3: Commit**

```bash
git add scripts/sweep_pion_lr.sh
git commit -m "feat(pion): add LR sweep script scripts/sweep_pion_lr.sh"
```

---

## Task 9: GPU smoke validation (handoff — user runs)

**Files:** none (validation only).

This is real GPU compute (single GPU), so per the project's compute policy it is the user's to run. The agent prepares the exact commands; the user runs them and reports back.

- [ ] **Step 1: Full CPU test sweep (agent runs)**

Run:
```bash
python -m pytest tests/unit/test_pion_optimizer.py tests/unit/test_patch_pion_optimizer_setup.py tests/unit/test_pion_launcher_args.py tests/unit/test_megatron_args.py -k "pion or megatron_args" -q
```
Expected: all Pion tests pass (and no new `test_megatron_args.py` failures vs. the pre-existing baseline noted in Task 5).

- [ ] **Step 2: Hand the user the single-GPU smoke command**

Provide:
```bash
# Short single-GPU Pion smoke at the tiny 60m dev scale (override iters down).
codexlog pion_smoke scripts/train_pion_dev.sh \
  training.train_iters=20 training.eval_interval=10 wandb.mode=disabled
```
Acceptance: process reaches the first optimizer step and logs `pion: N matrix params (2D non-embedding), M adamw params` with N>0; loss is finite and decreasing across the 20 iters; no `does not support` guard fires; run completes and writes a run dir under `runs/pion-*`.

- [ ] **Step 3: Hand the user the UNFUSED smoke command (validates the opt-in unfuse path)**

Provide:
```bash
# Same smoke, unfused architecture (Pion rotates each separate q/k/v/up/gate).
codexlog pion_unfused_smoke scripts/train_pion_dev.sh experiment=optim/pion_unfused \
  training.train_iters=20 training.eval_interval=10 wandb.mode=disabled
```
Acceptance: same as Step 2, AND the matrix-param count `N` is LARGER than the fused run (qkv → 3 separate weights, fc1 → 2), confirming the unfuse patch fired and Pion routed the split projections. Loss finite + decreasing.

- [ ] **Step 4: On green, record in the tracker + memory**

After the user confirms the smoke is green, append a Pion entry to the optimizer tracker (`POET_dev.md` / `CHANGELOG.md` per the repo's convention) and add a `pion-optimizer-integration` memory noting branch state, that CPU tests are green, and that the single-GPU smoke passed.

---

## Self-Review

**1. Spec coverage:**
- "clone pion into third_party" → done during planning (`third_party/pion`); the plan vendors from it (Task 1) and references it (Tasks 1-2, 8).
- "mimic adamw or muon to give configs/training scripts" → Task 6 (configs mirror `muon_kimi.yaml`/`adam.yaml`), Task 7 (script mirrors `train_muon_dev.sh`).
- "mimic the muon_dev scripts" → Task 7 is a direct mirror of `scripts/train_muon_dev.sh`.
- "create an lr sweep script" → Task 8 mirrors `scripts/sweep_muon_kimi_lr.sh`.
- "add the possibilities to unfuse" → Task 6 lists `model_unfuse_linears` on the fused default (no-op unless flagged) AND adds `pion_unfused.yaml`; Task 7's `"$@"` makes both paths reachable; Task 2's interface note documents the no-branch dual routing; Task 9 Step 3 smoke-validates the unfused path. Global Constraints updated from "must NOT unfuse" to "fused default, unfuse opt-in."
- "use superpower plan writing first" → this document.
- Implicit requirement to actually RUN: Tasks 1-6 wire the optimizer end-to-end (vendor → builder → patch → args → config), Task 9 validates fused + unfused on GPU.

**2. Placeholder scan:** No TBD/TODO/"add error handling"/"similar to Task N". Every code step has complete code; Task 1's verbatim extraction gives exact upstream line ranges + the exact replacement header; Task 7's test is a verified mirror of `test_muon_kimi_dev_script_routes_to_kimi_optimizer` (`_run` confirmed to use `--dry-run`).

**2b. Hook/gate scan (final-review additions):**
- `experiment-doc-exists` pre-commit hook (`tools/check_experiment_docs.py`) requires `docs/experiments/<name>.md` for every experiment YAML → Task 6 Step 3 creates `pion.md` + `pion_unfused.md`, and Step 5 stages them in the same commit.
- Patch-target collision: `pion_optimizer_setup` and `model_unfuse_linears` target different functions (optimizer-builder vs model-builder), so listing both in one config is safe — `muon_kimi.yaml` already lists `model_unfuse_linears` alongside its optimizer-setup patch. Never list two `*_optimizer_setup` patches together (registry raises `PatchConflict`); `pion.yaml`/`pion_unfused.yaml` list only `pion_optimizer_setup`.
- Arg/attr coverage: every `--pion-*` flag emitted by Task 5 is registered in Task 4; `--adam-*`/`--optimizer` come from Megatron-LM / `add_slm_args`. The builder's `getattr(config, "pion_*", default)` reads are satisfied either by `_PION_CONFIG_ATTRS` copy (Task 3) or by safe defaults (`pion_first/second/12_momentum="none"` → ignored when `pion_momentum` is set; `adam_eps` is a real OptimizerConfig field already copied by the stock path).

**3. Type consistency:**
- Builder name `get_megatron_pion_optimizer` is identical across Task 2 (definition), Task 3 (patch import), and Task 3's test (`fake_pion_builder`).
- Vendored class `PionOptimizer` identical across Task 1 (def) and Task 2 (import).
- Patch name `"pion_optimizer_setup"` identical across Task 3 (register), its test, and both Task 6 configs (`patches:`); `model_unfuse_linears` (an existing patch) is listed in both configs and is no-op-by-default on the fused one.
- `--slm-optimizer pion` value identical across Task 3 (tag check), Task 4 (choice), Task 5 (argv emit).
- `_PION_CONFIG_ATTRS` (Task 3) is a superset of the args registered in Task 4 and emitted in Task 5 — every emitted/registered `pion_*` flag is copied onto the config; extras in the tuple are harmless (`hasattr` guard).
- Builder signature `(config, model_chunks, *, config_overrides=None, use_gloo_process_groups=True)` matches the patch's call in Task 3 and the `muon_kimi` analog.

## Execution Handoff

Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, two-stage review between tasks, fast iteration. CPU tests run automatically; the single-GPU smokes (Task 9 Steps 2-3) are handed to the user.
2. **Inline Execution** — execute tasks in this session via superpowers:executing-plans, batched with review checkpoints.

Tasks 1-8 are CPU-only and fully verifiable in-session; Task 9 is the user-run GPU validation.
