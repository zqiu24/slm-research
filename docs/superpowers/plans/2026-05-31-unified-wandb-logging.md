# Unified W&B Logging Across Backends — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Megatron (`scripts/train_adam.sh`) and torchtitan (`scripts/train_adam_titan.sh`) backends emit the **same canonical W&B metric keys** for the core training-health curves, so one dashboard overlays them; backend-specific extras pass through unchanged.

**Architecture:** One pure first-party module (`src/utils/wandb_metrics.py`) owns the canonical schema and a `normalize(metrics, backend)` function. Two thin interceptors call it at each backend's W&B-log boundary, both guarded so logging never crashes training: a registered Megatron patch wraps `wandb.log` (Megatron's W&B writer *is* the `wandb` module) and additively computes the metrics Megatron doesn't emit to W&B; `src/titan_ext/metrics.py` wraps torchtitan's `WandBLogger.log`. No edits to the vendored submodules.

**Tech Stack:** Python 3.11, pytest, OmegaConf (Hydra-style configs), Megatron-LM + torchtitan (vendored, never edited), the `src/patches/` monkey-patch registry, and torchtitan's `experimental.custom_import` hook (`src/titan_ext`).

**Spec:** [docs/superpowers/specs/2026-05-31-unified-wandb-logging-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-05-31-unified-wandb-logging-design.md)

---

## Background an implementer needs (verified facts)

- **Megatron's W&B writer is the `wandb` module itself.** `megatron/training/global_vars.py:254` does `_GLOBAL_WANDB_WRITER = wandb`, so every `wandb_writer.log({...}, iteration)` inside `training_log` is literally `wandb.log({...}, iteration)`. To rename keys, wrap `wandb.log`.
- **`training_log` signature** (`megatron/training/training.py:1953`): `training_log(loss_dict, total_loss_dict, learning_rate, iteration, loss_scale, report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad, max_attention_logit, pg_collection=None, is_first_iteration=False)`. So **`iteration` is positional index 3**. (The existing `log_grad_norm_extra` patch uses `a[3]` — correct. The `training_log_wandb_tokens_seen` patch uses `args[4]` — a latent bug for this SHA; do **not** copy it.)
- **Our runs do NOT set `--log-timers-to-tensorboard`** (`src/utils/megatron_args.py` sets `--log-throughput` only). Megatron's `iteration-time` and `throughput` W&B scalars are both gated on `log_timers_to_tensorboard`, so today Megatron logs **neither to W&B** (stdout only). Therefore `perf/step_time_s` must be **computed in the hook** on the Megatron side (a perf-counter window), not renamed.
- **Throughput is NOT normalized across backends** (correctness — they are different quantities). Megatron's `throughput` is **TFLOP/s/GPU**; torchtitan's `throughput(tps)` is tokens/sec **normalized by `non_data_parallel_size`** (a per-model-parallel-group rate, `metrics.py:422-423`), whereas a Megatron-computed tokens/sec would be the **global aggregate** — these would overlay as two curves a large constant factor apart. Neither is mapped to a shared key; each backend's native throughput passes through unchanged. `perf/step_time_s` (plain wall-seconds per iteration, parallelism-independent) is the comparable perf metric instead.
- **Native W&B metrics are gated on `iteration % args.tensorboard_log_interval == 0`** (`training.py:2052`, default `1`). The computed Megatron metrics must use the **same** gate (`tensorboard_log_interval or log_interval`, matching `training_log_wandb_tokens_seen.py:42-44`) so they log at the same cadence as the renamed ones, not 10× sparser.
- **torchtitan's `WandBLogger.log(self, metrics, step)`** (`third_party/torchtitan/torchtitan/components/metrics.py:163`) applies an optional `tag/` prefix then calls `self.wandb.log(...)`. Wrapping this method (normalize *before* the prefix) is the clean torchtitan interception point. It sits beside the existing `MetricsProcessor.log` wrapper in `src/titan_ext/metrics.py`.
- **The patch registry** (`src/patches/_registry.py`): `@register_patch(name=..., targets=(...))` registers a module-level `apply()`. Two patches with overlapping `targets` raise `PatchConflict`. `log_grad_norm_extra` owns `targets=("megatron.training.training.training_log",)`. Our patch also wraps `training_log`, so it must register with **`targets=()`** (the same trick `training_log_wandb_tokens_seen` uses) and rely on sorted-by-name apply order (`log_grad_norm_extra` before `wandb_metric_normalize`).
- **Patch modules must be import-safe on CPU**: `import megatron`/`import wandb`/`import torchtitan` only *inside* functions, never at module top. `launchers.submit._register_experiment_patches(cfg)` imports each named patch on the launch node (no GPU) to compute the patch-set hash.
- **Tests** live under `tests/unit/`, import via `from src.… import …`, run with `python -m pytest tests/unit/<file> -v` from the repo root (`testpaths = ["tests"]`). Patch-registration tests use the `_reset_for_tests()` fixture pattern (see `tests/unit/test_patch_training_log_eta.py`).

---

## File Structure

**Create:**
- `src/utils/wandb_metrics.py` — canonical schema constants + `normalize(metrics, backend)`. Pure (stdlib only). Single source of truth.
- `src/patches/wandb_metric_normalize.py` — Megatron interceptor patch (wraps `wandb.log` for renames; wraps `training_log` for computed metrics).
- `tests/unit/test_wandb_metrics.py` — unit tests for `normalize` + schema.
- `tests/unit/test_patch_wandb_metric_normalize.py` — unit tests for the Megatron patch (registration + pure helpers).
- `tests/unit/test_titan_wandb_normalize.py` — unit tests for the torchtitan interceptor helper.
- `tests/unit/test_adam_experiment_wandb_normalize.py` — config-wiring + patch-composition test.

**Modify:**
- `src/titan_ext/metrics.py` — add `_wrap_titan_wandb_log` + `apply_titan_wandb_normalize`.
- `src/titan_ext/__init__.py:91-97` — call `apply_titan_wandb_normalize()` from `_patch_metrics()`.
- `configs/experiments/optim/adam.yaml` — add `wandb_metric_normalize` to `patches:`.
- `configs/experiments/champion.yaml` — add `wandb_metric_normalize` to `patches:` (kept in sync with adam).
- `WANDB_SETUP.md` — document the canonical schema table.
- `CHANGELOG.md` — Unreleased entry.

---

## Task 1: Canonical schema + `normalize()`

**Files:**
- Create: `src/utils/wandb_metrics.py`
- Test: `tests/unit/test_wandb_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_wandb_metrics.py`:

```python
"""Tests for the canonical W&B metric schema + normalize()."""

from src.utils.wandb_metrics import (
    CORE_CANONICAL,
    MEGATRON_TO_CANONICAL,
    TITAN_TO_CANONICAL,
    normalize,
)


def test_megatron_core_keys_map_to_canonical():
    out = normalize(
        {"lm loss": 2.5, "learning-rate": 1e-3, "grad-norm": 1.2}, "megatron"
    )
    assert out == {"train/loss": 2.5, "train/lr": 1e-3, "train/grad_norm": 1.2}


def test_megatron_validation_loss_maps_to_val_loss():
    assert normalize({"lm loss validation": 3.0}, "megatron") == {"val/loss": 3.0}


def test_megatron_unmapped_keys_pass_through():
    out = normalize(
        {"params-norm": 9.0, "num-zeros": 4, "grad-norm-clipped": 1.0}, "megatron"
    )
    assert out == {"params-norm": 9.0, "num-zeros": 4, "grad-norm-clipped": 1.0}


def test_megatron_throughput_is_NOT_remapped_to_tps():
    # Megatron 'throughput' is TFLOP/s/GPU, not tokens/sec — must stay native.
    assert normalize({"throughput": 177.2}, "megatron") == {"throughput": 177.2}


def test_torchtitan_core_keys_map_to_canonical():
    out = normalize(
        {
            "loss_metrics/global_avg_loss": 2.0,
            "loss_metrics/global_max_loss": 2.4,
            "lr": 1e-3,
            "grad_norm": 1.1,
            "n_tokens_seen": 4096,
            "time_metrics/end_to_end(s)": 0.5,
        },
        "torchtitan",
    )
    assert out == {
        "train/loss": 2.0,
        "train/loss_max": 2.4,
        "train/lr": 1e-3,
        "train/grad_norm": 1.1,
        "train/tokens_seen": 4096,
        "perf/step_time_s": 0.5,
    }


def test_torchtitan_validation_loss_maps_to_val_loss():
    assert normalize({"validation_metrics/loss": 3.1}, "torchtitan") == {"val/loss": 3.1}


def test_torchtitan_throughput_tps_is_NOT_normalized():
    # torchtitan tps is normalized by non_data_parallel_size; a Megatron-computed
    # tokens/sec is the global aggregate. Different quantities -> stay native.
    assert normalize({"throughput(tps)": 8000.0}, "torchtitan") == {"throughput(tps)": 8000.0}


def test_torchtitan_unmapped_keys_pass_through():
    out = normalize(
        {"mfu(%)": 30.0, "tflops": 120.0, "memory/max_active(GiB)": 40.0}, "torchtitan"
    )
    assert out == {"mfu(%)": 30.0, "tflops": 120.0, "memory/max_active(GiB)": 40.0}


def test_normalize_is_idempotent_on_canonical_input():
    canonical = {"train/loss": 2.0, "perf/step_time_s": 0.5}
    assert normalize(canonical, "megatron") == canonical
    assert normalize(canonical, "torchtitan") == canonical


def test_normalize_empty_and_none():
    assert normalize({}, "megatron") == {}
    assert normalize(None, "torchtitan") == {}


def test_both_backends_cover_the_core_set():
    # Every CORE_CANONICAL key (except the two computed-only on megatron) must be a
    # target of at least one backend map.
    targets = set(MEGATRON_TO_CANONICAL.values()) | set(TITAN_TO_CANONICAL.values())
    targets |= {"val/loss"}  # produced by the validation-suffix rule
    assert CORE_CANONICAL <= targets
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_wandb_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.wandb_metrics'`.

- [ ] **Step 3: Write the implementation**

Create `src/utils/wandb_metrics.py`:

```python
"""Canonical W&B metric schema + per-backend key normalization.

Single source of truth for the metric KEY names both training backends
("megatron", "torchtitan") emit to Weights & Biases, so one dashboard overlays
comparable curves. Pure module (stdlib only) — importable in any CPU/unit-test
env; no torch / wandb / megatron / torchtitan dependency.

Both backends are normalized at their wandb-log boundary and call
`normalize(metrics, backend)`:
  * megatron   -> src/patches/wandb_metric_normalize.py (wraps wandb.log)
  * torchtitan -> src/titan_ext/metrics.py (wraps WandBLogger.log)

See docs/superpowers/specs/2026-05-31-unified-wandb-logging-design.md.
"""

from __future__ import annotations

# The cross-backend comparison set: keys both backends converge on. (perf/* and
# train/tokens_seen are *computed* in the megatron interceptor, not renamed.)
CORE_CANONICAL = frozenset(
    {
        "train/loss",
        "train/lr",
        "train/grad_norm",
        "train/tokens_seen",
        "perf/step_time_s",
        "val/loss",
    }
)

# Megatron native key -> canonical. Deliberately EXCLUDES:
#   * "throughput"      — Megatron's is TFLOP/s/GPU; throughput is not normalized
#                         across backends (torchtitan's tps uses a different
#                         normalization), so it stays native on both sides.
#   * "iteration-time"  — gated off in our runs (no --log-timers-to-tensorboard);
#                         step time is COMPUTED in the interceptor instead.
# Unlisted keys pass through unchanged.
MEGATRON_TO_CANONICAL = {
    "lm loss": "train/loss",
    "learning-rate": "train/lr",
    "grad-norm": "train/grad_norm",
    "tokens seen": "train/tokens_seen",  # robustness if the legacy patch is enabled
}

# Torchtitan native key -> canonical. Deliberately EXCLUDES "throughput(tps)":
# it is tokens/sec normalized by non_data_parallel_size (a per-model-parallel-
# group rate), not comparable to a global-aggregate tokens/sec, so it stays
# native (like Megatron's TFLOP/s "throughput"). Other unlisted keys (mfu(%),
# tflops, memory/*, time_metrics/data_loading*, validation_metrics/*) pass through.
TITAN_TO_CANONICAL = {
    "loss_metrics/global_avg_loss": "train/loss",
    "loss_metrics/global_max_loss": "train/loss_max",
    "lr": "train/lr",
    "grad_norm": "train/grad_norm",
    "n_tokens_seen": "train/tokens_seen",
    "time_metrics/end_to_end(s)": "perf/step_time_s",
    "validation_metrics/loss": "val/loss",
}


def _canonical_key(key: str, backend: str) -> str:
    if backend == "megatron":
        if key in MEGATRON_TO_CANONICAL:
            return MEGATRON_TO_CANONICAL[key]
        # Megatron logs validation loss as "lm loss validation" (key + suffix).
        if key.startswith("lm loss") and "validation" in key:
            return "val/loss"
        return key
    if backend == "torchtitan":
        return TITAN_TO_CANONICAL.get(key, key)
    return key  # unknown backend: pass through unchanged


def normalize(metrics: dict | None, backend: str) -> dict:
    """Return a new dict with core overlapping keys renamed to the canonical
    schema; unknown / backend-specific keys pass through unchanged.

    Pure and idempotent on already-canonical input. `backend` is "megatron" or
    "torchtitan"; any other value is a pass-through no-op.
    """
    return {_canonical_key(k, backend): v for k, v in (metrics or {}).items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_wandb_metrics.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add src/utils/wandb_metrics.py tests/unit/test_wandb_metrics.py
git commit -m "feat(wandb): canonical metric schema + normalize() (single source of truth)"
```

---

## Task 2: Megatron interceptor patch

**Files:**
- Create: `src/patches/wandb_metric_normalize.py`
- Test: `tests/unit/test_patch_wandb_metric_normalize.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_patch_wandb_metric_normalize.py`:

```python
"""Tests for the wandb_metric_normalize patch (Megatron interceptor)."""

import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.wandb_metric_normalize", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.wandb_metric_normalize", None)


def test_patch_registers_with_empty_targets():
    # Empty targets so it composes with log_grad_norm_extra (which owns
    # training_log) without a PatchConflict.
    importlib.import_module("src.patches.wandb_metric_normalize")
    reg = registered_patches()
    assert "wandb_metric_normalize" in reg
    assert reg["wandb_metric_normalize"].targets == ()


def test_wrap_wandb_log_renames_megatron_keys():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    captured = {}

    def fake_log(data, *args, **kwargs):
        captured["data"] = data
        captured["args"] = args

    wrapped = mod._wrap_wandb_log(fake_log)
    wrapped({"lm loss": 2.5, "grad-norm": 1.0, "num-zeros": 3}, 100)

    assert captured["data"] == {"train/loss": 2.5, "train/grad_norm": 1.0, "num-zeros": 3}
    assert captured["args"] == (100,)  # the positional step is preserved


def test_wrap_wandb_log_passes_non_dict_through_untouched():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    seen = {}

    def fake_log(data, *a, **k):
        seen["data"] = data

    mod._wrap_wandb_log(fake_log)("not-a-dict")
    assert seen["data"] == "not-a-dict"


def test_extra_metrics_first_call_only_tokens():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    metrics, state = mod._extra_metrics(
        consumed_samples=10, seq_length=2048, iteration=10, now=100.0, last=None
    )
    assert metrics == {"train/tokens_seen": 10 * 2048}
    assert state == {"time": 100.0, "tokens": 10 * 2048, "iter": 10}


def test_extra_metrics_second_call_adds_step_time():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    last = {"time": 100.0, "tokens": 20480, "iter": 10}  # 10 samples * 2048
    metrics, state = mod._extra_metrics(
        consumed_samples=20, seq_length=2048, iteration=20, now=110.0, last=last
    )
    # tokens now = 20*2048 = 40960; dt = 10s; steps = 20-10 = 10.
    assert metrics["train/tokens_seen"] == 40960
    assert metrics["perf/step_time_s"] == pytest.approx(1.0)  # 10s / 10 steps
    # Throughput is intentionally NOT emitted (not comparable across backends).
    assert "perf/throughput_tps" not in metrics
    assert state == {"time": 110.0, "tokens": 40960, "iter": 20}


def test_extra_metrics_zero_dt_skips_step_time():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    last = {"time": 100.0, "tokens": 20480, "iter": 10}
    metrics, _ = mod._extra_metrics(
        consumed_samples=20, seq_length=2048, iteration=20, now=100.0, last=last
    )
    assert metrics == {"train/tokens_seen": 40960}  # no step time when dt == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_patch_wandb_metric_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.patches.wandb_metric_normalize'`.

- [ ] **Step 3: Write the implementation**

Create `src/patches/wandb_metric_normalize.py`:

```python
"""Patch: normalize Megatron's W&B metric KEYS to the canonical schema, and add
the canonical metrics Megatron doesn't emit to W&B in our config.

Two orthogonal changes, both guarded so logging never crashes training:

1. **Rename keys.** Megatron's W&B "writer" IS the ``wandb`` module itself
   (``megatron.training.global_vars._GLOBAL_WANDB_WRITER = wandb``), so every
   ``wandb_writer.log({...}, it)`` is ``wandb.log(...)``. We wrap ``wandb.log`` to
   run ``src.utils.wandb_metrics.normalize(d, "megatron")`` on the dict first
   (``lm loss`` -> ``train/loss``, ``learning-rate`` -> ``train/lr``, ...).
   Unmapped keys (grad-norm-clipped, params-norm, ...) pass through.

2. **Add computed metrics.** Our runs don't set --log-timers-to-tensorboard, so
   Megatron logs neither iteration-time nor throughput to W&B. We wrap
   ``training_log`` to additively emit ``train/tokens_seen`` (from
   ``consumed_train_samples * seq_length``) and ``perf/step_time_s`` (a
   perf-counter window), on the W&B-logging rank only, gated on the SAME interval
   as the native metrics (``tensorboard_log_interval`` or ``log_interval``).
   Throughput is intentionally NOT emitted: Megatron's would be a global-aggregate
   tokens/sec while torchtitan's ``throughput(tps)`` is normalized by
   ``non_data_parallel_size`` — not comparable, so each backend keeps its native
   throughput as a passthrough extra.

Registered with ``targets=()`` so it composes with ``log_grad_norm_extra``
(which owns ``training.training_log``). Apply order is sorted-by-name:
``log_grad_norm_extra`` (l) wraps first, ``wandb_metric_normalize`` (w) wraps the
result. Module import is CPU-safe (megatron / wandb imported only inside
``apply()``).
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)


def _extra_metrics(consumed_samples, seq_length, iteration, now, last):
    """Canonical metrics Megatron does not natively log to W&B in our config.

    Returns ``(metrics, new_state)``. ``last`` is the previous emit's state dict
    (or None on the first emit). The first emit carries only the cumulative
    ``train/tokens_seen``; later emits add the per-window ``perf/step_time_s``.
    Throughput is intentionally omitted (see module docstring).
    """
    tokens = int(consumed_samples) * int(seq_length)
    metrics = {"train/tokens_seen": tokens}
    if last is not None:
        dt = float(now) - float(last["time"])
        steps = max(1, int(iteration) - int(last["iter"]))
        if dt > 0:
            metrics["perf/step_time_s"] = dt / steps
    new_state = {"time": float(now), "tokens": tokens, "iter": int(iteration)}
    return metrics, new_state


def _wrap_wandb_log(orig_log):
    """Return a ``wandb.log`` wrapper that canonicalizes Megatron metric keys."""
    from src.utils.wandb_metrics import normalize

    def _log(data, *args, **kwargs):
        try:
            if isinstance(data, dict):
                data = normalize(data, "megatron")
        except Exception:  # logging must never crash training
            pass
        return orig_log(data, *args, **kwargs)

    _log._slm_wandb_normalize = True
    return _log


@register_patch(name="wandb_metric_normalize", targets=())
def apply() -> None:
    import time

    import wandb
    from megatron.training import get_args
    from megatron.training import training as _mt

    # Both the key-rename and the computed metrics ride on one training_log wrap.
    # wandb.log is wrapped LAZILY inside it: apply() runs before Megatron's
    # wandb.init() (deep in pretrain()), and wandb.init() rebinds the module-level
    # wandb.log — so a wrap installed here is silently discarded.
    _orig = _mt.training_log
    if getattr(_orig, "_slm_wandb_extra", False):
        return
    state = {"last": None}

    def _wrapped(*args, **kwargs):
        # (1) (Re)wrap wandb.log at call time (post wandb.init, so it's the stable
        #     function) so Megatron's own wandb_writer.log({'lm loss': ...}) calls
        #     inside _orig get renamed. Re-checked every call -> self-healing.
        if not getattr(wandb.log, "_slm_wandb_normalize", False):
            wandb.log = _wrap_wandb_log(wandb.log)
        ret = _orig(*args, **kwargs)
        try:
            # (2) Add computed metrics. get_wandb_writer() is None on non-logging
            #     ranks -> nothing to do.
            if _mt.get_wandb_writer() is not None:
                opts = get_args()
                iteration = kwargs.get("iteration")
                if iteration is None and len(args) > 3:
                    iteration = args[3]  # training_log(loss_dict, total, lr, iteration, ...)
                # Match the native-metric cadence: training_log gates its wandb
                # block on tensorboard_log_interval (training.py:2052). Mirror the
                # tokens_seen patch's gate so computed metrics aren't 10x sparser.
                interval = int(
                    getattr(opts, "tensorboard_log_interval", None)
                    or getattr(opts, "log_interval", 0)
                    or 0
                )
                if iteration is not None and interval and int(iteration) % interval == 0:
                    metrics, state["last"] = _extra_metrics(
                        getattr(opts, "consumed_train_samples", 0),
                        getattr(opts, "seq_length", 0),
                        int(iteration),
                        time.perf_counter(),
                        state["last"],
                    )
                    # wandb.log is the wrapped (renaming) one; canonical keys pass
                    # through normalize() idempotently.
                    wandb.log(metrics, int(iteration))
        except Exception:  # logging must never crash training
            pass
        return ret

    _wrapped._slm_wandb_extra = True
    _mt.training_log = _wrapped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_patch_wandb_metric_normalize.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/patches/wandb_metric_normalize.py tests/unit/test_patch_wandb_metric_normalize.py
git commit -m "feat(wandb): Megatron interceptor patch (rename keys + computed tokens/throughput)"
```

---

## Task 3: Wire the patch into the experiment configs

**Files:**
- Modify: `configs/experiments/optim/adam.yaml`
- Modify: `configs/experiments/champion.yaml`
- Test: `tests/unit/test_adam_experiment_wandb_normalize.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_adam_experiment_wandb_normalize.py`:

```python
"""The adam/champion experiments must enable wandb_metric_normalize, and it must
co-register with log_grad_norm_extra (both touch training_log) without conflict."""

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.patches._registry import _reset_for_tests

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    for name in (
        "src.patches.log_grad_norm_extra",
        "src.patches.wandb_metric_normalize",
        "src.patches.training_log_eta",
        "src.patches.model_unfuse_linears",
    ):
        sys.modules.pop(name, None)
    yield
    _reset_for_tests()


@pytest.mark.parametrize("rel", ["optim/adam.yaml", "champion.yaml"])
def test_experiment_lists_wandb_normalize(rel):
    cfg = OmegaConf.load(_REPO / "configs" / "experiments" / rel)
    assert "wandb_metric_normalize" in list(cfg.experiment.patches)


def test_wandb_normalize_composes_with_grad_norm_extra():
    # Both wrap training_log; one declares the target, the other targets=().
    # _register_experiment_patches imports + hashes them (no apply, CPU-safe).
    from launchers.submit import _register_experiment_patches
    from src.patches import registered_patches

    cfg = OmegaConf.create(
        {"experiment": {"patches": ["log_grad_norm_extra", "wandb_metric_normalize"]}}
    )
    h = _register_experiment_patches(cfg)
    reg = registered_patches()
    assert "wandb_metric_normalize" in reg and "log_grad_norm_extra" in reg
    assert len(h) == 16 and not h.startswith("noop")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_adam_experiment_wandb_normalize.py -v`
Expected: FAIL — the two `test_experiment_lists_wandb_normalize` cases fail (`wandb_metric_normalize` not yet in the `patches:` lists). The compose test should PASS already.

- [ ] **Step 3: Edit `configs/experiments/optim/adam.yaml`**

In the `patches:` block, after the `log_grad_norm_extra` line, add:

```yaml
    - log_grad_norm_extra     # always-on: log post-clip grad-norm + clip-coeff
    - wandb_metric_normalize  # canonicalize W&B metric keys + add tokens_seen / step_time
```

(The first line already exists — add only the `wandb_metric_normalize` line beneath it.)

- [ ] **Step 4: Edit `configs/experiments/champion.yaml`**

In its `patches:` block, after `log_grad_norm_extra`, add the identical line:

```yaml
    - log_grad_norm_extra     # always-on: log post-clip grad-norm + clip-coeff
    - wandb_metric_normalize  # canonicalize W&B metric keys + add tokens_seen / step_time
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_adam_experiment_wandb_normalize.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add configs/experiments/optim/adam.yaml configs/experiments/champion.yaml tests/unit/test_adam_experiment_wandb_normalize.py
git commit -m "feat(wandb): enable wandb_metric_normalize in adam + champion experiments"
```

---

## Task 4: torchtitan interceptor

**Files:**
- Modify: `src/titan_ext/metrics.py`
- Modify: `src/titan_ext/__init__.py`
- Test: `tests/unit/test_titan_wandb_normalize.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_titan_wandb_normalize.py`:

```python
"""Tests for the torchtitan WandBLogger key-normalization wrapper.

Importing src.titan_ext.metrics is CPU-safe: torchtitan is imported only inside
the apply_* functions, and the package __init__'s _patch_metrics() no-ops when
torchtitan is absent.
"""


def test_wrap_titan_wandb_log_renames_keys_and_preserves_step():
    from src.titan_ext.metrics import _wrap_titan_wandb_log

    captured = {}

    def fake_log(self, metrics, step):
        captured["metrics"] = metrics
        captured["step"] = step

    wrapped = _wrap_titan_wandb_log(fake_log)

    class FakeLogger:
        pass

    wrapped(
        FakeLogger(),
        {"loss_metrics/global_avg_loss": 2.0, "mfu(%)": 30.0, "lr": 1e-3},
        50,
    )
    assert captured["metrics"] == {"train/loss": 2.0, "mfu(%)": 30.0, "train/lr": 1e-3}
    assert captured["step"] == 50


def test_apply_titan_wandb_normalize_noops_without_torchtitan():
    # On a CPU box without torchtitan importable, apply returns False, never raises.
    from src.titan_ext.metrics import apply_titan_wandb_normalize

    assert apply_titan_wandb_normalize() in (True, False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_titan_wandb_normalize.py -v`
Expected: FAIL — `ImportError: cannot import name '_wrap_titan_wandb_log'`.

- [ ] **Step 3: Add the wrapper + apply function to `src/titan_ext/metrics.py`**

Append to the end of `src/titan_ext/metrics.py`:

```python


_WANDB_WRAP_FLAG = "_slm_wandb_normalize_wrapped"


def _wrap_titan_wandb_log(orig_log):
    """Return a ``WandBLogger.log`` wrapper that canonicalizes metric keys.

    Runs ``src.utils.wandb_metrics.normalize(metrics, "torchtitan")`` BEFORE the
    upstream body (which applies any ``tag/`` prefix and calls ``wandb.log``), so
    e.g. ``loss_metrics/global_avg_loss`` -> ``train/loss``. Backend-specific keys
    (mfu(%), tflops, memory/*) pass through. Guarded: a transform failure falls
    back to the original metrics, never crashing training.
    """
    from src.utils.wandb_metrics import normalize

    def _log(self, metrics, step, *args, **kwargs):
        try:
            metrics = normalize(metrics, "torchtitan")
        except Exception:
            pass
        return orig_log(self, metrics, step, *args, **kwargs)

    setattr(_log, _WANDB_WRAP_FLAG, True)
    return _log


def apply_titan_wandb_normalize() -> bool:
    """Monkeypatch ``WandBLogger.log`` to canonicalize metric keys.

    Returns True if the patch is in place (or was already), False if torchtitan
    is not importable (e.g. a CPU unit-test env).
    """
    try:
        import torchtitan.components.metrics as _m
    except Exception:
        return False

    if getattr(_m.WandBLogger.log, _WANDB_WRAP_FLAG, False):
        return True

    _m.WandBLogger.log = _wrap_titan_wandb_log(_m.WandBLogger.log)
    logger.info("[titan_ext] patched WandBLogger.log (canonical metric keys)")
    return True
```

- [ ] **Step 4: Wire it into `src/titan_ext/__init__.py`**

Replace the `_patch_metrics()` function body (currently at `src/titan_ext/__init__.py:91-97`) so it also calls the new function:

```python
def _patch_metrics() -> None:
    # Rank-0-only per-step console line + ETA, and canonical W&B metric keys, to
    # match / align with the Megatron path. Independent of the TrainSpec
    # registration above and of SLM_RESOLVED_CONFIG; each no-ops if torchtitan is
    # absent (CPU unit-test env).
    from src.titan_ext.metrics import (
        apply_titan_metrics_patch,
        apply_titan_wandb_normalize,
    )

    apply_titan_metrics_patch()
    apply_titan_wandb_normalize()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_titan_wandb_normalize.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/titan_ext/metrics.py src/titan_ext/__init__.py tests/unit/test_titan_wandb_normalize.py
git commit -m "feat(wandb): torchtitan interceptor canonicalizes WandBLogger metric keys"
```

---

## Task 5: Documentation + changelog

**Files:**
- Modify: `WANDB_SETUP.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the canonical schema table to `WANDB_SETUP.md`**

Append this section to the end of `WANDB_SETUP.md`:

```markdown
## Canonical metric schema (cross-backend)

Both training backends normalize their core W&B metric keys onto one schema so a
single dashboard overlays Megatron and torchtitan runs. Implemented by
`src/utils/wandb_metrics.py` (`normalize()`), applied at each backend's W&B-log
boundary (`src/patches/wandb_metric_normalize.py` for Megatron;
`src/titan_ext/metrics.py` for torchtitan). Backend-specific extras keep their
native names (passthrough). See
`docs/superpowers/specs/2026-05-31-unified-wandb-logging-design.md`.

| Canonical key | Meaning | Unit | Megatron source | Torchtitan source |
|---|---|---|---|---|
| `train/loss` | training loss (mean) | nats | `lm loss` | `loss_metrics/global_avg_loss` |
| `train/loss_max` | max micro-batch loss | nats | — | `loss_metrics/global_max_loss` |
| `train/lr` | learning rate | — | `learning-rate` | `lr` |
| `train/grad_norm` | raw (pre-clip) grad norm | — | `grad-norm` | `grad_norm` |
| `train/tokens_seen` | cumulative tokens | tokens | computed (consumed_samples × seq_len) | `n_tokens_seen` |
| `perf/step_time_s` | wall-time per iteration | seconds | computed (perf-counter window) | `time_metrics/end_to_end(s)` |
| `val/loss` | validation loss | nats | `lm loss validation` | `validation_metrics/loss` |

**Throughput is intentionally NOT normalized.** Megatron's native `throughput`
is **TFLOP/s/GPU**; torchtitan's `throughput(tps)` is tokens/sec normalized by
`non_data_parallel_size` (a per-model-parallel-group rate). These are different
quantities and would not overlay, so each backend keeps its native throughput key
as a passthrough extra; `perf/step_time_s` is the comparable perf metric.
TensorBoard keys are not normalized (W&B only).
```

- [ ] **Step 2: Add a CHANGELOG entry**

Under the `## Unreleased` heading in `CHANGELOG.md`, add a bullet under the most appropriate `### Added` (or create `### Added — unified W&B logging`):

```markdown
- **Unified W&B metric keys across backends** (`src/utils/wandb_metrics.py`):
  both backends now normalize their core training-health metrics onto one
  canonical schema (`train/loss`, `train/lr`, `train/grad_norm`,
  `train/tokens_seen`, `perf/step_time_s`, `val/loss`) so Megatron and torchtitan
  runs overlay on one dashboard. Applied via a registered Megatron patch
  (`wandb_metric_normalize`, wraps `wandb.log`; also computes tokens_seen /
  step_time, which our runs don't log to W&B) and a torchtitan `WandBLogger.log`
  wrapper in `src/titan_ext/metrics.py`. Throughput is deliberately NOT normalized
  (Megatron's `throughput` is TFLOP/s/GPU; torchtitan's `throughput(tps)` is
  normalized by `non_data_parallel_size` — not comparable), so each backend keeps
  its native throughput as a passthrough. Backend-specific extras pass through
  unchanged; no vendored-submodule edits. Design + plan in
  `docs/superpowers/{specs,plans}/2026-05-31-unified-wandb-logging*.md`.
```

- [ ] **Step 3: Commit**

```bash
git add WANDB_SETUP.md CHANGELOG.md
git commit -m "docs(wandb): document the cross-backend canonical metric schema"
```

---

## Task 6: Full unit-test sweep

- [ ] **Step 1: Run all the new tests together**

Run:
```bash
python -m pytest \
  tests/unit/test_wandb_metrics.py \
  tests/unit/test_patch_wandb_metric_normalize.py \
  tests/unit/test_adam_experiment_wandb_normalize.py \
  tests/unit/test_titan_wandb_normalize.py -v
```
Expected: PASS (all green).

- [ ] **Step 2: Run the broader patch + launcher test suite for regressions**

Run:
```bash
python -m pytest tests/unit -k "patch or launcher or titan or wandb" -v
```
Expected: PASS — in particular the pre-existing `test_patch_training_log_eta`, `test_patch_wandb_tokens_seen`, `test_launcher_patch_wiring`, and `test_patches_registry` still pass (the new `targets=()` patch must not introduce a `PatchConflict`).

---

## Operator validation (run by the user, not in this plan)

These require GPUs / a real run and are **not** executed by the implementing agent (no working training env in the harness):

1. Short Megatron smoke: `bash scripts/train_adam.sh llama3 training.steps=20 training.log_interval=5` and confirm W&B shows `train/loss`, `train/lr`, `train/grad_norm`, `train/tokens_seen`, `perf/step_time_s` (and that `lm loss` / `learning-rate` / `grad-norm` no longer appear as separate curves).
2. Short torchtitan smoke: `bash scripts/train_adam_titan.sh llama3 training.steps=20` and confirm the same canonical keys appear, plus torchtitan extras (`mfu(%)`, `tflops`, `memory/*`, native `throughput(tps)`).
3. Overlay both runs on one W&B dashboard; confirm `train/loss` curves are directly comparable.

---

## Self-Review

**Spec coverage:**
- §1 canonical schema → Task 1 (`CORE_CANONICAL`, maps) + Task 5 table. ✅
- §2 mapping table → Task 1 (`MEGATRON_TO_CANONICAL`, `TITAN_TO_CANONICAL`, validation rule) + tests. ✅
- §3 computed Megatron metrics → Task 2 `_extra_metrics` (tokens_seen always; step_time on later emits; throughput intentionally omitted as non-comparable) + tests. ✅
- §4.1 `src/utils/wandb_metrics.py` → Task 1. ✅
- §4.2 Megatron patch (wrap `wandb.log` + `training_log`, `targets=()`, added to adam/champion) → Tasks 2 + 3. ✅
- §4.3 torchtitan interceptor (wrap `WandBLogger.log`, wired into `__init__`) → Task 4. ✅
- §6 error handling (guarded, never crashes training) → try/except in both wrappers + tests for non-dict / zero-dt. ✅
- §7 testing (golden maps, passthrough, idempotency, throughput guard, registry compose, fake writer/logger) → Tasks 1–4 tests. ✅
- §8 deliverables (module, patch, titan_ext, configs, tests, CHANGELOG, docs) → Tasks 1–5. ✅
- Non-goal "no TB normalization / no cross-fill beyond computed Megatron metrics" respected (only the 3 computed Megatron metrics, justified because they're absent from W&B). ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test shows assertions; commands have expected output. ✅

**Type/name consistency:** `normalize(metrics, backend)`, `_extra_metrics(consumed_samples, seq_length, iteration, now, last) -> (dict, state)`, `_wrap_wandb_log(orig_log)`, `_wrap_titan_wandb_log(orig_log)`, `apply_titan_wandb_normalize()`, guard flags `_slm_wandb_normalize` / `_slm_wandb_extra` / `_WANDB_WRAP_FLAG`, patch name `wandb_metric_normalize`, `CORE_CANONICAL` / `MEGATRON_TO_CANONICAL` / `TITAN_TO_CANONICAL` — all referenced consistently across tasks and tests. ✅
