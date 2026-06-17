# POET × MoE: Grouped-GEMM Guard + Throughput Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop POET from silently skipping grouped-GEMM MoE experts (fail fast instead), and add an env-gated per-phase profiler so the real cause of POET's ~4.2 TFLOP/s/GPU throughput on DeepSeek-3Bv2 can be located with evidence before any optimization.

**Architecture:** Two independent changes. (1) A config-time validation in the POET arg-builder that raises when `optim.type=poet` is combined with `moe.grouped_gemm=true` — because grouped experts are batched `weight1/weight2` Parameters that POET's module walk cannot wrap, so they would train as dense Adam weights with zero orbit and no warning. (2) An env-gated timing instrument inside the existing `poet_merge_step` `train_step` wrapper that, for one chosen iteration, attributes GPU time across forward+backward / optimizer (lie_ortho) / merge using CUDA events, with an optional `torch.profiler` drill-down for per-op (expert-GEMM vs Muon-NS vs Cayley) attribution. Pure decision/format helpers are CPU-unit-tested; the CUDA plumbing is verified by a smoke run.

**Tech Stack:** Python 3.12, PyTorch (CUDA events, `torch.profiler`), Megatron-LM (vendored under `third_party/`), OmegaConf/Hydra configs, pytest.

## Global Constraints

- **Test interpreter:** run all pytest with `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest` (Python 3.12.13). The repo requires 3.12; the harness default `python` is 3.10 and lacks omegaconf/torch.
- **CPU-import safety:** new module-level helpers in `src/patches/poet_merge_step.py` MUST NOT import Megatron at module load and MUST be importable on a CPU-only node. The test file `tests/unit/test_poet_merge_step.py` documents this invariant ("module-level helpers must not import megatron"). Import `torch` / `torch.distributed` *inside* functions only.
- **No behavior change when not profiling:** the profiler is a no-op unless `POET_PROFILE_STEP` is set; the existing merge/reset behavior must be byte-for-byte preserved on non-profiled iterations.
- **Guard scope:** the guard fires only when ALL of `optim.type=="poet"`, `moe.enabled==true`, `moe.grouped_gemm==true`. It must NOT fire for the `champion` (adamw) deepseek_v3 path already covered by `test_deepseek_args_include_mla_moe_and_deepseek_router_knobs`, nor for the `deepseek_v3_mqa` POET path (grouped_gemm=false).
- **POET runs at TP=1** (parallelism rules pin tp=1 for POET scales) — no TP-axis concerns in any new code.
- **Commit style:** one short conventional-commit line per task (`fix(...)` / `feat(...)`), anonymous, no AI attribution.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/utils/megatron_args.py` | Translate slm config → Megatron CLI argv; houses POET arg validation | **Modify** — add grouped_gemm guard in the `kind == "poet"` block of `_optimizer_args` |
| `tests/unit/test_megatron_args.py` | Arg-emission unit tests | **Modify** — add guard reject/allow tests |
| `src/patches/poet_merge_step.py` | `train_step` wrapper: active-side seed + periodic merge | **Modify** — add env-gated CUDA-event profiler + torch.profiler drill-down + pure helpers |
| `tests/unit/test_poet_merge_step.py` | CPU tests for the merge-step pure helpers | **Modify** — test the new pure helpers (`_profile_target_iteration`, `_torch_profile_enabled`, `_format_profile`, `_dominant_phase`) |
| `tests/unit/test_patch_poet_merge.py` | CPU tests for the merge-step patch wiring (fake-megatron + `apply()`) | **Modify** — test the rewritten `_wrapped` closure preserves merge behavior with profiling off |

---

## Task 1: Fail-fast guard for POET + grouped-GEMM experts

**Files:**
- Modify: `src/utils/megatron_args.py` (inside `_optimizer_args`, the `if kind == "poet":` block, right after `poet = optim.poet`)
- Test: `tests/unit/test_megatron_args.py`

**Interfaces:**
- Consumes: `_optimizer_args(cfg: DictConfig) -> list[str]` (existing). Reads `cfg.optim` (already) and now also `cfg.base.model.moe` via safe `.get(...)` chaining (mirrors the existing `unfuse_qkv` read at the top of the same block).
- Produces: same `list[str]`; raises `ValueError` (whose message contains the substring `grouped_gemm`) on the rejected combination.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/unit/test_megatron_args.py`:

```python
def _poet_moe_cfg(grouped_gemm: bool):
    """Minimal cfg exercising the POET arg-builder with an MoE model block.

    The poet sub-keys mirror test_poet_argv_includes_cache_mode (the minimal set
    _optimizer_args needs to complete). base.model.moe drives the new guard.
    """
    return OmegaConf.create(
        {
            "base": {"model": {"moe": {"enabled": True, "grouped_gemm": grouped_gemm}}},
            "optim": {
                "type": "poet",
                "lr": 3e-4,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "poet": {
                    "block_size": 256,
                    "cache_mode": "none",
                    "init_type": "normalized",
                    "mup_alpha": 1.0,
                    "merge_period": 1,
                    "scale": 1.0,
                },
            },
        }
    )


def test_poet_rejects_grouped_gemm_experts():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="grouped_gemm"):
        _optimizer_args(_poet_moe_cfg(grouped_gemm=True))


def test_poet_allows_sequential_experts():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_moe_cfg(grouped_gemm=False))
    assert "--poet" in args  # arg build completes, no raise


def test_poet_guard_inert_without_moe():
    # No base.model.moe block at all -> guard must not fire (dense POET path).
    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 3e-4,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "poet": {
                    "block_size": 256,
                    "cache_mode": "none",
                    "init_type": "normalized",
                    "mup_alpha": 1.0,
                    "merge_period": 1,
                    "scale": 1.0,
                },
            }
        }
    )
    assert "--poet" in _optimizer_args(cfg)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py::test_poet_rejects_grouped_gemm_experts -v`
Expected: FAIL — `_optimizer_args` returns argv without raising (the guard does not exist yet), so `pytest.raises(ValueError)` fails with "DID NOT RAISE".

- [ ] **Step 3: Add the guard**

In `src/utils/megatron_args.py`, locate the `if kind == "poet":` block in `_optimizer_args`. It begins:

```python
    if kind == "poet":
        poet = optim.poet
        if poet.get("head_aligned_attn", False) and not bool(
            cfg.get("base", {}).get("model", {}).get("unfuse_qkv", False)
        ):
            raise ValueError(
                "optim.poet.head_aligned_attn requires base.model.unfuse_qkv=true "
                "(head-aligned blocks need unfused q/k/v)."
            )
```

Insert the guard immediately after `poet = optim.poet` and before the `head_aligned_attn` check:

```python
    if kind == "poet":
        poet = optim.poet
        # POET wraps per-module ColumnParallelLinear / RowParallelLinear. Grouped-GEMM
        # experts are batched weight1/weight2 Parameters (GroupedMLP / TEGroupedMLP),
        # NOT linear modules -- POET's module walk silently skips them, so every routed
        # expert would train as a dense Adam weight with zero orbit and no warning. Only
        # grouped_gemm=false (SequentialMLP, per-expert linear modules) lets POET reach
        # the experts. Fail fast rather than silently degrade.
        _moe = cfg.get("base", {}).get("model", {}).get("moe", {}) or {}
        if bool(_moe.get("enabled", False)) and bool(_moe.get("grouped_gemm", False)):
            raise ValueError(
                "[POET] optim.type=poet is incompatible with base.model.moe.grouped_gemm=true: "
                "grouped-GEMM experts are batched parameters POET cannot wrap, so they would "
                "be silently skipped (trained as dense Adam weights, no orbit). Set "
                "base.model.moe.grouped_gemm=false (SequentialMLP) to POET-ise the experts."
            )
        if poet.get("head_aligned_attn", False) and not bool(
            cfg.get("base", {}).get("model", {}).get("unfuse_qkv", False)
        ):
            raise ValueError(
                "optim.poet.head_aligned_attn requires base.model.unfuse_qkv=true "
                "(head-aligned blocks need unfused q/k/v)."
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -v -k "poet"`
Expected: PASS — `test_poet_rejects_grouped_gemm_experts`, `test_poet_allows_sequential_experts`, `test_poet_guard_inert_without_moe`, and the pre-existing poet arg tests all green.

- [ ] **Step 5: Run the full arg-builder suite to confirm no regression**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py tests/unit/test_megatron_args_families.py tests/unit/test_train_scripts.py -v`
Expected: PASS — in particular `test_deepseek_args_include_mla_moe_and_deepseek_router_knobs` (deepseek_v3 + grouped_gemm=true under the `champion`/adamw recipe) stays green because the guard is scoped to `kind == "poet"`.

- [ ] **Step 6: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "$(cat <<'EOF'
fix(megatron_args): fail fast when POET is combined with moe.grouped_gemm=true (grouped experts can't be POET-wrapped and would silently train as dense Adam)
EOF
)"
```

---

## Task 2: Env-gated per-phase CUDA-event profiler

**Files:**
- Modify: `src/patches/poet_merge_step.py` (add module-level imports + pure helpers + CUDA timers; rewrite the `_wrapped` closure inside `apply()`)
- Test: `tests/unit/test_poet_merge_step.py`

**Interfaces:**
- Produces (pure, CPU-safe):
  - `_profile_target_iteration() -> int | None` — parses env `POET_PROFILE_STEP`; returns a positive int or `None` (unset / non-integer / `<= 0`).
  - `_torch_profile_enabled() -> bool` — true iff env `POET_PROFILE_TORCH` is one of `{"1","true","yes"}` (case-insensitive).
  - `_dominant_phase(timings: dict[str, float]) -> str | None` — among leaf keys `forward_backward`/`optimizer`/`merge`, returns the key with the largest value (`train_step_total` excluded as it is the fwd+bwd+optimizer sum); `None` if no leaves present.
  - `_format_profile(timings: dict[str, float]) -> str` — multi-line `[POET-PROFILE]`-prefixed summary in fixed order `train_step_total, forward_backward, optimizer, merge`, ending with the dominant-component line when derivable. `train_step_total` is the train_step wall time (fwd+bwd+optimizer); `merge` runs *after* train_step and is reported on its own line, so it is NOT part of `train_step_total`.
- Produces (GPU plumbing, import torch internally):
  - `_cuda_timer(timings, key, enabled)` — contextmanager; records CUDA-event GPU ms into `timings[key]`; no-op if `enabled` false or CUDA unavailable.
  - `_maybe_wrap_optimizer_step(optimizer, timings, enabled)` — contextmanager; temporarily replaces `optimizer.step` with a CUDA-timed wrapper writing `timings["optimizer"]`, restores on exit; no-op if disabled / `optimizer is None` / no CUDA.
  - `_run_train_step_torch_profiled(orig, args, kwargs, dist)` — runs one `orig(*args, **kwargs)` under `torch.profiler.profile`, prints the top-25 ops table on rank 0, returns `orig`'s result.
  - `_emit_profile(timings, dist)` — prints `_format_profile(timings)` on rank 0 only.
- Consumes: the existing `_merge_decision`, `_run_merge`, `_reset_vanilla_oft_state`, `_seed_active_side` (unchanged).

- [ ] **Step 1: Write the failing tests for the pure helpers**

Add to `tests/unit/test_poet_merge_step.py`:

```python
def test_profile_target_iteration_parses_env(monkeypatch):
    from src.patches.poet_merge_step import _profile_target_iteration

    monkeypatch.delenv("POET_PROFILE_STEP", raising=False)
    assert _profile_target_iteration() is None

    monkeypatch.setenv("POET_PROFILE_STEP", "20")
    assert _profile_target_iteration() == 20

    monkeypatch.setenv("POET_PROFILE_STEP", "0")
    assert _profile_target_iteration() is None  # non-positive -> off

    monkeypatch.setenv("POET_PROFILE_STEP", "notanint")
    assert _profile_target_iteration() is None  # malformed -> off


def test_torch_profile_enabled(monkeypatch):
    from src.patches.poet_merge_step import _torch_profile_enabled

    monkeypatch.delenv("POET_PROFILE_TORCH", raising=False)
    assert _torch_profile_enabled() is False
    monkeypatch.setenv("POET_PROFILE_TORCH", "1")
    assert _torch_profile_enabled() is True
    monkeypatch.setenv("POET_PROFILE_TORCH", "TRUE")
    assert _torch_profile_enabled() is True
    monkeypatch.setenv("POET_PROFILE_TORCH", "0")
    assert _torch_profile_enabled() is False


def test_dominant_phase_picks_largest_leaf():
    from src.patches.poet_merge_step import _dominant_phase

    assert _dominant_phase({}) is None
    # train_step_total is the sum and must be excluded from the leaf comparison.
    timings = {"train_step_total": 100.0, "forward_backward": 70.0, "optimizer": 25.0, "merge": 5.0}
    assert _dominant_phase(timings) == "forward_backward"
    assert _dominant_phase({"optimizer": 9.0, "merge": 40.0}) == "merge"


def test_format_profile_orders_and_labels():
    from src.patches.poet_merge_step import _format_profile

    out = _format_profile(
        {"train_step_total": 100.0, "forward_backward": 70.0, "optimizer": 25.0, "merge": 5.0}
    )
    assert "[POET-PROFILE]" in out
    # fixed order: train_step_total before forward_backward before optimizer before merge
    assert out.index("train_step_total") < out.index("forward_backward") < out.index(
        "optimizer"
    ) < out.index("merge")
    assert "dominant component: forward_backward" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_step.py -v -k "profile or dominant"`
Expected: FAIL — `ImportError: cannot import name '_profile_target_iteration'` (and the other helpers) because they don't exist yet.

- [ ] **Step 3: Add module-level imports and pure helpers**

In `src/patches/poet_merge_step.py`, the current header is:

```python
from __future__ import annotations

import logging

from src.patches._registry import register_patch
```

Replace it with:

```python
from __future__ import annotations

import contextlib
import logging
import os

from src.patches._registry import register_patch
```

Then add these pure helpers near the top of the module (after `logger = logging.getLogger(__name__)`, before `_merge_decision`):

```python
# ----------------------------------------------------------------------------
# Profiling (env-gated; pure helpers are CPU-safe and must NOT import megatron)
# ----------------------------------------------------------------------------

_PROFILE_LEAF_KEYS = ("forward_backward", "optimizer", "merge")
_PROFILE_ORDER = ("train_step_total", *_PROFILE_LEAF_KEYS)


def _profile_target_iteration():
    """Iteration to profile from POET_PROFILE_STEP, or None if unset/invalid/<=0."""
    raw = os.environ.get("POET_PROFILE_STEP")
    if raw is None:
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _torch_profile_enabled() -> bool:
    """True iff POET_PROFILE_TORCH requests the per-op torch.profiler drill-down."""
    return os.environ.get("POET_PROFILE_TORCH", "").strip().lower() in {"1", "true", "yes"}


def _dominant_phase(timings: dict):
    """Largest leaf component (train_step_total excluded as it is the sum); None if none."""
    leaves = {k: timings[k] for k in _PROFILE_LEAF_KEYS if k in timings}
    if not leaves:
        return None
    return max(leaves, key=leaves.get)


def _format_profile(timings: dict) -> str:
    """Fixed-order, [POET-PROFILE]-prefixed per-phase timing summary (ms).

    train_step_total is the train_step wall time (forward_backward + optimizer);
    merge runs AFTER train_step and is reported separately, so it is not part of
    train_step_total. _dominant_phase therefore compares only the three leaves.
    """
    lines = ["[POET-PROFILE] per-phase GPU time (ms):"]
    for k in _PROFILE_ORDER:
        if k in timings:
            lines.append(f"[POET-PROFILE]   {k:<18} {timings[k]:10.2f}")
    dom = _dominant_phase(timings)
    if dom is not None:
        lines.append(f"[POET-PROFILE] dominant component: {dom}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the pure-helper tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_step.py -v -k "profile or dominant"`
Expected: PASS — all four new tests green.

- [ ] **Step 5: Add the GPU timing plumbing (import torch internally)**

Append these functions to `src/patches/poet_merge_step.py` (module level, after the pure helpers; they import torch lazily so the module stays CPU-import-safe):

```python
@contextlib.contextmanager
def _cuda_timer(timings: dict, key: str, enabled: bool):
    """Record CUDA-event-bounded GPU time (ms) for the wrapped block into
    timings[key]. No-op when disabled or CUDA is unavailable."""
    if not enabled:
        yield
        return
    import torch

    if not torch.cuda.is_available():
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        torch.cuda.synchronize()
        timings[key] = timings.get(key, 0.0) + start.elapsed_time(end)


@contextlib.contextmanager
def _maybe_wrap_optimizer_step(optimizer, timings: dict, enabled: bool):
    """Temporarily wrap optimizer.step to record its CUDA time into
    timings['optimizer'] (the lie_ortho / Adam step). Restores the original step
    on exit. No-op when disabled, optimizer is None, or CUDA is unavailable."""
    if not enabled or optimizer is None:
        yield
        return
    import torch

    if not torch.cuda.is_available():
        yield
        return
    orig_step = optimizer.step

    def _timed_step(*a, **k):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = orig_step(*a, **k)
        end.record()
        torch.cuda.synchronize()
        timings["optimizer"] = timings.get("optimizer", 0.0) + start.elapsed_time(end)
        return out

    optimizer.step = _timed_step
    try:
        yield
    finally:
        optimizer.step = orig_step


def _emit_profile(timings: dict, dist) -> None:
    """Print the per-phase summary on rank 0 only."""
    if not timings:
        return
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    if rank == 0:
        print(_format_profile(timings), flush=True)


def _run_train_step_torch_profiled(orig, args, kwargs, dist):
    """Run one train_step under torch.profiler and print the top CUDA ops on
    rank 0. Reveals whether time is in expert GEMMs (SequentialMLP), Muon NS, or
    Cayley fold ops."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        ret = orig(*args, **kwargs)
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    if rank == 0:
        sort_key = (
            "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        )
        print("[POET-PROFILE] torch.profiler top ops:", flush=True)
        print(prof.key_averages().table(sort_by=sort_key, row_limit=25), flush=True)
    return ret
```

- [ ] **Step 6: Wire profiling into the `_wrapped` train_step closure**

In `src/patches/poet_merge_step.py`, replace the entire `_wrapped` closure inside `apply()` (currently lines ~90-121) with:

```python
    def _wrapped(*args, **kwargs):
        opts = get_args()
        if not getattr(opts, "poet", False):
            return _orig_train_step(*args, **kwargs)
        iteration = kwargs.get("iteration")
        if iteration is None and len(args) >= 8:
            iteration = args[7]
        if iteration is None:
            iteration = getattr(opts, "iteration", 0)
        # Seed the active-side signal BEFORE forward so the layer reads this step's side.
        _seed_active_side(iteration)

        # Profiling (POET_PROFILE_STEP=<iter>): attribute one iteration's GPU time
        # across forward+backward / optimizer (lie_ortho) / merge to locate the
        # throughput bottleneck. POET_PROFILE_TORCH=1 swaps in a torch.profiler
        # per-op drill-down for that iteration instead of the coarse phase timers.
        profile = _profile_target_iteration() == int(iteration)
        optimizer = args[3] if len(args) >= 4 else kwargs.get("optimizer")
        timings: dict = {}

        if profile and _torch_profile_enabled():
            ret = _run_train_step_torch_profiled(_orig_train_step, args, kwargs, dist)
        else:
            with _maybe_wrap_optimizer_step(optimizer, timings, profile):
                with _cuda_timer(timings, "train_step_total", profile):
                    ret = _orig_train_step(*args, **kwargs)
            if profile and "train_step_total" in timings and "optimizer" in timings:
                timings["forward_backward"] = max(
                    timings["train_step_total"] - timings["optimizer"], 0.0
                )

        merge_period = getattr(opts, "poet_merge_period", 0)
        reinit_period = getattr(opts, "poet_reinit_period", 0)
        folding, do_reinit = _merge_decision(iteration, merge_period, reinit_period)
        model = args[2] if len(args) >= 3 else kwargs.get("model")
        if folding and model is None:
            logger.warning("[POET] merge step skipped: model not found in train_step args")
        elif folding:
            with _cuda_timer(timings, "merge", profile):
                _run_merge(model, dist, iteration, reinit_perm=do_reinit)
                # Megatron-Adam path (default): reset momentum ONLY when Ψ is
                # resampled (do_reinit); the master VALUE is zeroed every fold
                # inside _reset_vanilla_oft_state regardless.
                if not getattr(opts, "poet_use_poet_adam", False):
                    if optimizer is not None:
                        _reset_vanilla_oft_state(
                            optimizer, model, iteration, reset_moments=do_reinit
                        )

        if profile:
            _emit_profile(timings, dist)
        return ret
```

Note: this preserves all prior behavior on non-profiled iterations — `_cuda_timer`/`_maybe_wrap_optimizer_step` with `enabled=False` are pure pass-throughs, and the merge/reset branch is logically identical to the original (the early `return ret` on `not folding` is replaced by skipping the merge branch and falling through to `return ret`).

- [ ] **Step 7: Add a regression test for the rewritten `_wrapped` closure**

The `_wrapped` closure is the riskiest edit (it changes train_step control flow), and the existing suite tests `_merge_decision`/`_run_merge`/`_reset_vanilla_oft_state` *directly* — never through the wrapper. Add a fake-megatron test (real Megatron is not CPU-importable: it eagerly loads transformer_engine → `OSError: libcublas.so.12`). Add to `tests/unit/test_patch_poet_merge.py`:

```python
def test_wrapped_train_step_preserves_merge_behavior(monkeypatch):
    """With profiling OFF, the rewritten train_step wrapper must call _run_merge
    exactly when _merge_decision folds -- i.e. byte-for-byte the pre-profiler
    behavior. Inject a fake megatron.training carrying get_args + train_step."""
    import sys
    import types

    from src.patches import poet_merge_step as pms

    calls = {"merge": 0, "orig": 0}

    class _Args:
        poet = True
        poet_merge_period = 1
        poet_reinit_period = -1
        poet_use_poet_adam = True  # skip the optimizer-reset branch (no real optimizer)
        iteration = 0

    def _orig_train_step(*a, **k):
        calls["orig"] += 1
        return {"lm loss": 1.0}

    mt = types.SimpleNamespace(train_step=_orig_train_step)
    fake_pkg = types.SimpleNamespace(get_args=lambda: _Args(), training=mt)
    monkeypatch.setitem(sys.modules, "megatron", types.SimpleNamespace(training=fake_pkg))
    monkeypatch.setitem(sys.modules, "megatron.training", fake_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", mt)
    monkeypatch.setattr(
        pms, "_run_merge", lambda *a, **k: calls.__setitem__("merge", calls["merge"] + 1)
    )
    monkeypatch.delenv("POET_PROFILE_STEP", raising=False)

    pms.apply()  # register_patch returns apply unchanged -> body runs, installs _wrapped on mt.train_step

    # train_step(forward_step_func, data_iterator, model, optimizer,
    #            opt_param_scheduler, config, forward_backward_func, iteration)
    out = mt.train_step(None, None, ["model"], None, None, None, None, 5)
    assert out == {"lm loss": 1.0}  # return value passed through unchanged
    assert calls["orig"] == 1
    assert calls["merge"] == 1  # merge_period=1 -> folds at iteration 5

    calls["merge"] = 0
    mt.train_step(None, None, ["model"], None, None, None, None, 0)
    assert calls["merge"] == 0  # iteration 0 never folds
```

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_merge.py::test_wrapped_train_step_preserves_merge_behavior -v`
Expected: PASS — the rewritten wrapper folds at iter 5, not at iter 0, and returns the orig result unchanged. (This is a regression/characterization test: it would also pass against the pre-rewrite closure, which is the point — the rewrite must not change this behavior.)

- [ ] **Step 8: Run the merge-step test suite to confirm no regression**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_step.py tests/unit/test_patch_poet_merge.py -v`
Expected: PASS — new profiler-helper tests and pre-existing merge tests (`_merge_decision`, `_seed_active_side`, etc.) all green.

- [ ] **Step 9: Byte-compile check (catch import/syntax errors without GPU)**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/poet_merge_step.py src/utils/megatron_args.py`
Expected: no output (success).

- [ ] **Step 10: Commit**

```bash
git add src/patches/poet_merge_step.py tests/unit/test_poet_merge_step.py tests/unit/test_patch_poet_merge.py
git commit -m "$(cat <<'EOF'
feat(poet): env-gated per-phase profiler (POET_PROFILE_STEP / POET_PROFILE_TORCH) to attribute train_step GPU time across forward+backward / optimizer / merge
EOF
)"
```

---

## Task 3: Run the profiler and record findings (USER-RUN — GPU/cluster)

> This task runs real compute and is the user's to run per project policy. The agent provides exact commands and records the reported numbers afterward; it does not launch GPU/cluster jobs itself.

**Caveats (read before running):**
- The coarse `forward_backward` figure is the *whole model* — it does NOT isolate the routed-expert `SequentialMLP` loop from attention. To attribute within forward/backward (expert GEMMs vs attention), use the `torch.profiler` drill-down in Step 3; the coarse mode only answers the merge-vs-optimizer-vs-fwd+bwd fork.
- The CUDA-event path *does* correctly capture distributed-`lie_ortho` cost: CUDA events measure GPU-stream time including NCCL collective kernels and the idle gaps where the stream waits on CPU-launched work, so a launch-bound or collective-bound optimizer shows up in the `optimizer` figure. This makes the coarse mode the right *primary* tool on the 8-GPU run.
- `POET_PROFILE_STEP` must reach every rank. Single-node torchrun (the `full` recipe) inherits the parent environment, so the `env VAR=...` prefix propagates; for any srun/multi-node launch, set it in the job environment instead.
- `torch.profiler` traces one full train_step (32 grad-accum micro-batches × 64 experts) — expect real overhead and a large in-memory trace. Run it for ONE iteration only (Step 3), never a whole job.

**Files:**
- Modify (after results come back): `docs/superpowers/plans/2026-06-17-poet-moe-guard-and-profiler.md` (append a "Findings" section)

- [ ] **Step 1: Validate the instrumentation on a 1-GPU dev smoke (cheap)**

Command (single GPU; the dev recipe reaches ~18 iterations, so profile iter 10):

```bash
codexlog deepseek_poet_profile_dev \
  env POET_PROFILE_STEP=10 \
  bash scripts/train_deepseek_poet.sh dev training.log_interval=1
```

Expected on stdout near iteration 10:
```
[POET-PROFILE] per-phase GPU time (ms):
[POET-PROFILE]   train_step_total          ......
[POET-PROFILE]   forward_backward    ......
[POET-PROFILE]   optimizer           ......
[POET-PROFILE]   merge               ......
[POET-PROFILE] dominant component: <forward_backward|optimizer|merge>
```
Purpose: confirms the timers print once, on the right iteration, without crashing. (1-GPU dev does NOT exercise the distributed lie_ortho collectives, so its split is only a sanity check, not the real diagnosis.)

- [ ] **Step 2: Profile the real bottleneck on the 8-GPU full config**

The ~4.2 TFLOP/s symptom is the multi-GPU run (distributed lie_ortho is multi-GPU-only). Profile a steady-state iteration there:

```bash
codexlog deepseek_poet_profile_full \
  env POET_PROFILE_STEP=20 \
  bash scripts/train_deepseek_poet.sh full training.log_interval=1
```

Read the `[POET-PROFILE]` block for iteration 20 (steady state, past warmup). The dominant component answers the fork:
- `forward_backward` dominates → the cost is model compute, i.e. the `SequentialMLP` per-expert loop forced by `grouped_gemm=false` (POET's structural cost). Next lever: expert exclusion or a grouped-compatible POET path.
- `optimizer` dominates → distributed `lie_ortho` Muon-NS over ~4800 `oft_R` tensors with per-tensor collectives. Next lever: reduce the number of POET'd experts (`skip_routed_experts`) or batch the distributed NS.
- `merge` dominates (unlikely; it is 1×/iter) → batched Cayley fold cost; revisit `_build_R_batched`.

- [ ] **Step 3: Per-op attribution (only if `forward_backward` or `optimizer` dominates and the source is ambiguous)**

```bash
codexlog deepseek_poet_profile_torch \
  env POET_PROFILE_STEP=20 POET_PROFILE_TORCH=1 \
  bash scripts/train_deepseek_poet.sh full training.log_interval=1
```

Read the `torch.profiler top ops` table: expert/grouped GEMM rows vs Muon Newton-Schulz matmuls vs Cayley ops directly attribute the time. (Heavier — one iteration only.)

- [ ] **Step 4: Record findings**

Append a `## Findings (YYYY-MM-DD)` section to this plan file with: the per-phase ms from Step 2, the dominant component, the torch top-ops summary if run, and the recommended next optimization. Commit:

```bash
git add docs/superpowers/plans/2026-06-17-poet-moe-guard-and-profiler.md
git commit -m "docs(poet): record DeepSeek-3Bv2 POET profiler findings and recommended next lever"
```

---

## Findings (2026-06-17)

Instrumentation ran on the 8-GPU `full` config (DeepSeek-3Bv2: 12 layers, 64 experts,
top-6, `moe_grouped_gemm=False` → SequentialMLP, TP=PP=EP=1 / DP=8, 32 grad-accum
micro-batches/iter, 2.73B params) and the 1-GPU `dev` sanity check.

**Per-phase profiler (`POET_PROFILE_STEP`):**

| phase | dev (1-GPU, iter 10) | full (8-GPU, iter 20) |
|---|---|---|
| `train_step_total` | 9728.56 ms | 37735.13 ms |
| `forward_backward` | 9414.50 ms (96.8%) | 37547.42 ms (**99.5%**) |
| `optimizer` (lie_ortho) | 314.07 ms (3.2%) | 187.71 ms (**0.5%**) |
| `merge` (separate, post-step) | 1068.78 ms | 1018.33 ms (~2.6% of wall) |
| **dominant** | forward_backward | forward_backward |

**torch.profiler drill-down (`POET_PROFILE_TORCH=1`): did not complete.** The log ends
exactly at the `torch.profiler top ops:` header — building `key_averages().table()`
over a one-iteration trace (32 micro-batches × 64 experts) was too heavy and the
process was killed there, as the Task 3 caveat warned. No per-op table was produced.

**Diagnosis — the original suspicion (distributed `lie_ortho` Muon-NS over ~4800 `oft_R`
tensors) is REFUTED.** The optimizer is 0.5% and merge is ~2.6% of wall on the 8-GPU
run; neither is worth optimizing. The cost is forward+backward.

The Adam baseline (`deepseek_adam_full.log`) localizes it further. It runs the **same**
`grouped_gemm=False` SequentialMLP model, yet:

| run (8-GPU, identical model) | steady-state s/iter | TFLOP/s/GPU |
|---|---|---|
| **Adam** (iter 390–430) | ~24.2 s | **7.5** |
| **POET** (iter 560–580, `deepseek_poet_full.log`) | ~43.2 s | **4.2** |

POET is **~1.79× slower for the same useful FLOPs**, and the profiler localizes the
entire ~19 s/iter gap inside `forward_backward` (optimizer + merge together cannot
account for ~2 s of it). Since the SequentialMLP loop is a cost Adam pays too (at 7.5),
the POET delta is the **per-expert orbit overhead in forward/backward**: applying the
rotation on 64 experts × 12 layers × 2 linears, every one of the 32 micro-batches. The
analytic Megatron TFLOP/s formula does not count POET's extra Cayley/rotation FLOPs, so
that work shows up as depressed throughput rather than higher TFLOP/s.

**Recommended next lever: `skip_routed_experts`** — do NOT POET-ise the 64 routed
experts; keep POET on the dense / attention / shared weights and leave the experts on
Adam. Double win: (1) removes the bulk of POET's forward/backward overhead (the
~19 s/iter delta), and (2) lets the experts use `grouped_gemm=true` again (the Task 1
guard fires only *because* POET tries to wrap them), replacing the slow SequentialMLP
loop with a fused grouped GEMM — should pull throughput toward or past the 7.5 Adam
number. The merge (`_build_R_batched`) and the distributed `lie_ortho` step are NOT
worth touching.

---

## Out of scope (optional future hardening)

- **Defense-in-depth runtime guard.** The Task 1 guard fires at arg-build time, which covers every real launch (all go through `build_megatron_args`). A belt-and-suspenders check could be added later in [poet_apply_to_model.py](src/patches/poet_apply_to_model.py): after the walk, error if the model contains MoE expert modules but zero expert linears were wrapped — catching any future code path that bypasses the arg builder. Not implemented here because the config-time guard is sufficient for all current launch paths.
- **Expert-cost reduction** (`skip_routed_experts`, batched distributed NS, etc.) is deliberately deferred until Task 3's profiling names the real bottleneck — optimizing before then would be guessing.

## Self-Review

**Spec coverage:**
- Fix #1 (fail-fast guard, hard error on `poet + grouped_gemm`) → Task 1. ✓
- Fix #2 (profile first to locate the throughput bottleneck, since amortizing the merge is blocked by `single_step_x`'s `merge_period=1` invariant and is not the likely bottleneck) → Tasks 2 (instrument) + 3 (run & decide). ✓
- Guard must not regress the existing adamw deepseek grouped-GEMM test → Task 1 Step 5 (verified `champion` resolves to `optim.type=adamw`, guard scoped to `kind=="poet"`). ✓
- CPU-import safety of new merge-step helpers → Task 2 helpers import torch only inside functions; pure helpers tested in the CPU file. ✓
- Behavior preservation of the rewritten `_wrapped` closure (the riskiest edit) → Task 2 Step 7 regression test (fake-megatron `apply()`, asserts merge fires iff folding with profiling off). ✓

**Placeholder scan:** every code step contains complete code; commands include expected output; no "TBD"/"add validation"/"handle edge cases". ✓

**Type/name consistency:** helper names (`_profile_target_iteration`, `_torch_profile_enabled`, `_dominant_phase`, `_format_profile`, `_cuda_timer`, `_maybe_wrap_optimizer_step`, `_emit_profile`, `_run_train_step_torch_profiled`) are identical across their definitions (Task 2 Steps 3/5/6), tests (Task 2 Step 1), and the Interfaces blocks. Timing keys (`train_step_total`, `forward_backward`, `optimizer`, `merge`) are consistent across `_PROFILE_ORDER`, `_PROFILE_LEAF_KEYS`, the `_wrapped` writes, and the tests. ✓

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-17-poet-moe-guard-and-profiler.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute Tasks 1 and 2 in this session (CPU-testable end-to-end); Task 3 is handed to you to run on the cluster.

**Which approach?**
