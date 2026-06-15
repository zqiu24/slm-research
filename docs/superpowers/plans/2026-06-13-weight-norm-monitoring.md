# Weight-Matrix Row/Column Norm Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an always-registered, flag-gated patch that logs row-norm and column-norm summaries (+ per-layer RMS histograms) of a few transformer weight matrices to W&B, so POET / Muon / Adam runs can be overlaid to see how weight norms evolve without weight decay.

**Architecture:** A new patch `src/patches/weight_norm_monitor.py` wraps `megatron.training.training.train_step` as the OUTER wrapper (it sorts after `poet_merge_step`, so for POET the per-step merge has already folded `R` into the base weight when we read it — the raw base weight then equals the effective weight `W_eff`). It is inert unless `--log-weight-norms` is set. Pure helper functions (layer selection, name classification, norm stats, cadence) are factored out for CPU unit testing without a real Megatron import. Three new CLI flags are added to the launcher and plumbed through the YAML→CLI builder.

**Tech Stack:** Python, PyTorch, Megatron-LM (vendored), OmegaConf configs, W&B, pytest. CPU test interpreter: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`.

**Key design facts (from the spec):**
- Spec: [2026-06-13-weight-norm-monitoring-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-13-weight-norm-monitoring-design.md)
- Patch registry applies patches in `sorted(name)` order; later = outer wrapper ([_registry.py:88-106](/lustre/fast/fast/zqiu/slm-research/src/patches/_registry.py#L88-L106)). `weight_norm_monitor` > `poet_merge_step` alphabetically ⇒ outer ⇒ reads post-merge weights.
- `poet_merge_step` owns the `train_step` target; to compose we register `targets=()` (same trick as [poet_grad_conditioning.py:212](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_grad_conditioning.py#L212)).
- For POET, `module.weight` of a `POETMegatronLinear` is aliased to the frozen base (`self.weight = poet_linear.weight`, [poet_layers.py:79](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L79)). Reading it on a merge-boundary step gives `W_eff`. `oft_R_*` are excluded because we read the module's `.weight`, not its parameters, and the `poet_linear` child module name does not match our suffix filter.
- `merge_period=1` (most POET configs) ⇒ base == `W_eff` every step. `merge_period=0` ⇒ base frozen ⇒ warn + skip.

---

## File Structure

- **Create** `src/patches/weight_norm_monitor.py` — the patch: pure helpers + `_log_weight_norms` + `train_step` wrapper + `apply()`.
- **Create** `tests/unit/test_patch_weight_norm_monitor.py` — unit tests for all pure helpers, the logging dict, the wrap/ordering, and registry registration.
- **Modify** `launchers/pretrain_gpt_slm.py` — add 3 CLI flags in `add_slm_args`; add `weight_norm_monitor` to `_ALWAYS_ON_PATCHES`.
- **Modify** `src/utils/megatron_args.py` — emit the 3 flags from `_logging_args`.
- **Modify** `tests/unit/test_megatron_args.py` — assert the flags are emitted.
- **Modify** `CHANGELOG.md` — record the change.

---

## Task 1: CLI flags + YAML→CLI plumbing

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py:145` (end of `add_slm_args`, before `return parser`)
- Modify: `src/utils/megatron_args.py` — add a `_weight_norm_args` helper and call it from `_logging_args`
- Test: `tests/unit/test_megatron_args.py`

> **Why a separate helper (not an inline block in `_logging_args`):** `_logging_args` calls `wandb_run_name(cfg)`, which unconditionally reads `optim.type`, `experiment.name`, `base.family`, `base.scale`, and `optim.lr`. A test that called `_logging_args` with a minimal cfg would raise `KeyError` from `wandb_run_name`, not exercise our flags. A pure `_weight_norm_args(training)` helper is testable in isolation with just a `training` dict — matching the existing "minimal `OmegaConf.create` + call the sub-function directly" pattern used by `test_poet_argv_includes_cache_mode`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_weight_norm_args_emits_flags_when_enabled():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _weight_norm_args

    training = OmegaConf.create(
        {
            "log_weight_norms": True,
            "log_weight_norms_interval": 50,
            "weight_norm_layers": "first,last",
        }
    )
    argv = _weight_norm_args(training)
    assert "--log-weight-norms" in argv
    assert argv[argv.index("--log-weight-norms-interval") + 1] == "50"
    assert argv[argv.index("--weight-norm-layers") + 1] == "first,last"


def test_weight_norm_args_omits_flags_by_default():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _weight_norm_args

    assert _weight_norm_args(OmegaConf.create({})) == []
    # bool false also emits nothing
    assert _weight_norm_args(OmegaConf.create({"log_weight_norms": False})) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py::test_weight_norm_args_emits_flags_when_enabled -v`
Expected: FAIL — `cannot import name '_weight_norm_args'`.

- [ ] **Step 3: Add the `_weight_norm_args` helper and call it from `_logging_args`**

In `src/utils/megatron_args.py`, add this helper just above `def _logging_args(` (currently [line 562](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L562)):

```python
def _weight_norm_args(training: DictConfig) -> list[str]:
    """Emit the weight_norm_monitor flags from the `training` config block."""
    args: list[str] = []
    _maybe_bool(args, "--log-weight-norms", training.get("log_weight_norms", False))
    interval = training.get("log_weight_norms_interval", None)
    if interval is not None:
        _add(args, "--log-weight-norms-interval", interval)
    layers = training.get("weight_norm_layers", None)
    if layers is not None:
        _add(args, "--weight-norm-layers", layers)
    return args
```

Then, inside `_logging_args`, immediately before `return args` (currently [line 614](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L614)), add:

```python
    args.extend(_weight_norm_args(cfg.training))
```

- [ ] **Step 4: Add the argparse flags**

In `launchers/pretrain_gpt_slm.py`, immediately before `return parser` at the end of `add_slm_args` (currently [line 146](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L146)), add:

```python
    # Weight-matrix row/column norm monitoring (weight_norm_monitor patch).
    group.add_argument("--log-weight-norms", action="store_true")
    group.add_argument("--log-weight-norms-interval", type=int, default=100)
    group.add_argument("--weight-norm-layers", type=str, default="first,mid,last")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k weight_norm -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add launchers/pretrain_gpt_slm.py src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(slm): add --log-weight-norms flags and YAML plumbing"
```

---

## Task 2: Pure helpers — layer selection, name classification, cadence

**Files:**
- Create: `src/patches/weight_norm_monitor.py`
- Test: `tests/unit/test_patch_weight_norm_monitor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_patch_weight_norm_monitor.py`:

```python
# tests/unit/test_patch_weight_norm_monitor.py
import sys

import pytest

from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.weight_norm_monitor", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.weight_norm_monitor", None)


def test_parse_layer_selection_keywords_and_indices():
    from src.patches.weight_norm_monitor import parse_layer_selection

    assert parse_layer_selection("first,mid,last", 12) == {0, 6, 11}
    assert parse_layer_selection("0,5,11", 12) == {0, 5, 11}
    assert parse_layer_selection("-1", 12) == {11}  # negative wraps
    assert parse_layer_selection("99", 12) == set()  # out of range dropped
    assert parse_layer_selection(" first , 3 ", 12) == {0, 3}  # whitespace tolerant


def test_classify_linear_matches_fused_and_unfused_types_and_skips_others():
    from src.patches.weight_norm_monitor import classify_linear

    # fused names
    assert classify_linear("decoder.layers.5.self_attention.linear_qkv") == (5, "qkv")
    assert classify_linear("decoder.layers.0.self_attention.linear_proj") == (0, "proj")
    assert classify_linear("module.decoder.layers.3.mlp.linear_fc1") == (3, "fc1")
    assert classify_linear("decoder.layers.7.mlp.linear_fc2") == (7, "fc2")
    # unfused names (--unfuse-qkv / --unfuse-fc1, e.g. head-aligned POET configs)
    assert classify_linear("decoder.layers.2.self_attention.linear_q") == (2, "q")
    assert classify_linear("decoder.layers.2.self_attention.linear_k") == (2, "k")
    assert classify_linear("decoder.layers.2.self_attention.linear_v") == (2, "v")
    assert classify_linear("decoder.layers.4.mlp.linear_fc1_gate") == (4, "fc1_gate")
    assert classify_linear("decoder.layers.4.mlp.linear_fc1_up") == (4, "fc1_up")
    # POET base-weight child module is NOT matched (name doesn't end in a type suffix)
    assert classify_linear("decoder.layers.5.self_attention.linear_qkv.poet_linear") is None
    # embeddings / lm_head / norms not matched
    assert classify_linear("embedding.word_embeddings") is None
    assert classify_linear("output_layer") is None


def test_should_log_cadence_for_adam_and_poet():
    from src.patches.weight_norm_monitor import should_log

    # non-POET: every `interval`
    assert should_log(100, 100, poet=False, merge_period=0) is True
    assert should_log(150, 100, poet=False, merge_period=0) is False
    assert should_log(0, 100, poet=False, merge_period=0) is False
    # POET merge_period=1: base == W_eff every step -> gated only by interval
    assert should_log(50, 50, poet=True, merge_period=1) is True
    # POET merge_period=400: only on merge boundaries
    assert should_log(400, 100, poet=True, merge_period=400) is True
    assert should_log(100, 100, poet=True, merge_period=400) is False
    # POET merge_period=0: frozen base -> never
    assert should_log(100, 100, poet=True, merge_period=0) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_norm_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.patches.weight_norm_monitor'`.

- [ ] **Step 3: Create the module with the pure helpers**

Create `src/patches/weight_norm_monitor.py`:

```python
# src/patches/weight_norm_monitor.py
"""Patch: log row/column norm summaries of a few weight matrices to W&B.

Flag-gated by ``--log-weight-norms`` (interval ``--log-weight-norms-interval``,
default 100; layers ``--weight-norm-layers``, default ``first,mid,last``). Inert
otherwise, so it is safe in ``_ALWAYS_ON_PATCHES``.

Mechanism: wrap ``train_step`` as the OUTER wrapper. ``weight_norm_monitor``
sorts after ``poet_merge_step`` in the registry's sorted apply order, so for a
POET run the per-step merge has already folded ``R`` into the base weight when we
read it -> ``module.weight`` equals the effective weight ``W_eff``. For Adam/Muon
the weight IS the trained parameter. We register ``targets=()`` (like
``poet_grad_conditioning``) so we compose with ``poet_merge_step``'s wrapper of
the same symbol instead of raising ``PatchConflict``.
"""

from __future__ import annotations

import logging
import re

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)

# module-name suffix -> short matrix-type label. Covers both the fused mcore
# names and the unfused variants produced by --unfuse-qkv / --unfuse-fc1 (used by
# some POET configs, e.g. head-aligned attention). A module name ends in exactly
# one of these, so endswith matching is unambiguous.
_SUFFIX_TO_TYPE = {
    "self_attention.linear_qkv": "qkv",
    "self_attention.linear_q": "q",
    "self_attention.linear_k": "k",
    "self_attention.linear_v": "v",
    "self_attention.linear_proj": "proj",
    "mlp.linear_fc1": "fc1",
    "mlp.linear_fc1_gate": "fc1_gate",
    "mlp.linear_fc1_up": "fc1_up",
    "mlp.linear_fc2": "fc2",
}
_LAYER_RE = re.compile(r"(?:^|\.)decoder\.layers\.(\d+)\.")

# warn-once state for the POET merge_period=0 (frozen-base) corner case
_state = {"warned_merge0": False}


def parse_layer_selection(spec, num_layers: int) -> set[int]:
    """Parse a layer spec into a set of layer indices.

    Tokens are comma-separated; each is one of ``first`` (0), ``mid``
    (num_layers // 2), ``last`` (num_layers - 1), or an integer index (negatives
    wrap from the end). Out-of-range integer indices are silently dropped.
    """
    out: set[int] = set()
    for raw in str(spec).split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok == "first":
            out.add(0)
        elif tok == "mid":
            out.add(num_layers // 2)
        elif tok == "last":
            out.add(num_layers - 1)
        else:
            idx = int(tok)
            if idx < 0:
                idx += num_layers
            if 0 <= idx < num_layers:
                out.add(idx)
    return out


def classify_linear(name: str):
    """Map a module name to ``(layer_index, matrix_type)`` or ``None``.

    Matches the transformer linears by suffix — fused (qkv/proj/fc1/fc2) and the
    unfused variants (q/k/v, fc1_gate/fc1_up). The POET base-weight child
    (``...linear_qkv.poet_linear``), embeddings, lm_head and norms do not match.
    """
    m = _LAYER_RE.search(name)
    if not m:
        return None
    for suffix, mtype in _SUFFIX_TO_TYPE.items():
        if name.endswith(suffix):
            return int(m.group(1)), mtype
    return None


def should_log(iteration: int, interval: int, *, poet: bool, merge_period: int) -> bool:
    """Decide whether to log on this step.

    Non-POET: every ``interval`` steps. POET: only on merge-boundary steps
    (``iteration % merge_period == 0``) that are also interval steps, because the
    raw base weight equals ``W_eff`` only right after a merge. POET with
    ``merge_period <= 0`` (frozen base) never logs.
    """
    if iteration <= 0 or interval <= 0:
        return False
    if poet:
        if merge_period <= 0:
            return False
        if iteration % merge_period != 0:
            return False
    return iteration % interval == 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_norm_monitor.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/patches/weight_norm_monitor.py tests/unit/test_patch_weight_norm_monitor.py
git commit -m "feat(slm): weight_norm_monitor pure helpers (selection/classify/cadence)"
```

---

## Task 3: Norm statistics computation

**Files:**
- Modify: `src/patches/weight_norm_monitor.py`
- Test: `tests/unit/test_patch_weight_norm_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_patch_weight_norm_monitor.py`:

```python
def test_compute_matrix_norm_stats_values_and_rms():
    import math

    import torch

    from src.patches.weight_norm_monitor import compute_matrix_norm_stats

    # 2x3 matrix: rows have norms; cols have norms.
    w = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    stats = compute_matrix_norm_stats(w)

    # row norms = [3, 4]; row_rms = norm / sqrt(in_dim=3)
    assert stats["row"]["min"] == pytest.approx(3.0)
    assert stats["row"]["max"] == pytest.approx(4.0)
    assert stats["row"]["mean"] == pytest.approx(3.5)
    assert stats["row_rms"]["max"] == pytest.approx(4.0 / math.sqrt(3))
    # col norms = [3, 4, 0]; col_rms = norm / sqrt(out_dim=2)
    assert stats["col"]["max"] == pytest.approx(4.0)
    assert stats["col"]["min"] == pytest.approx(0.0)
    assert stats["col_rms"]["max"] == pytest.approx(4.0 / math.sqrt(2))
    # raw RMS vectors are returned for histogram pooling
    assert stats["_row_rms_vec"].shape == (2,)
    assert stats["_col_rms_vec"].shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_norm_monitor.py::test_compute_matrix_norm_stats_values_and_rms -v`
Expected: FAIL — `cannot import name 'compute_matrix_norm_stats'`.

- [ ] **Step 3: Implement `compute_matrix_norm_stats`**

Append to `src/patches/weight_norm_monitor.py`:

```python
def _summary(vec) -> dict:
    """mean/std/min/max of a 1-D tensor as Python floats."""
    n = vec.numel()
    return {
        "mean": vec.mean().item(),
        "std": vec.std(unbiased=False).item() if n > 1 else 0.0,
        "min": vec.min().item(),
        "max": vec.max().item(),
    }


def compute_matrix_norm_stats(weight) -> dict:
    """Row/col L2-norm summaries (raw and RMS-normalized) for a 2-D weight.

    For ``weight`` of shape ``(out, in)``:
      * row norms ``r_i = ||W[i, :]||`` (length ``out``); RMS = ``r_i / sqrt(in)``
      * col norms ``c_j = ||W[:, j]||`` (length ``in``);  RMS = ``c_j / sqrt(out)``
    RMS divides out the matrix width so different-shaped matrices are comparable.
    Returns scalar summaries under keys ``row``/``col``/``row_rms``/``col_rms``
    plus the raw RMS vectors (``_row_rms_vec``/``_col_rms_vec``) for histograms.
    """
    import torch

    w = weight.detach().to(torch.float32)
    out_dim, in_dim = w.shape
    row = torch.linalg.vector_norm(w, dim=1)
    col = torch.linalg.vector_norm(w, dim=0)
    row_rms = row / (in_dim ** 0.5)
    col_rms = col / (out_dim ** 0.5)
    return {
        "row": _summary(row),
        "col": _summary(col),
        "row_rms": _summary(row_rms),
        "col_rms": _summary(col_rms),
        "_row_rms_vec": row_rms,
        "_col_rms_vec": col_rms,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_norm_monitor.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/patches/weight_norm_monitor.py tests/unit/test_patch_weight_norm_monitor.py
git commit -m "feat(slm): weight_norm_monitor norm-stat computation"
```

---

## Task 4: Collect target weights + build the W&B payload

**Files:**
- Modify: `src/patches/weight_norm_monitor.py`
- Test: `tests/unit/test_patch_weight_norm_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_patch_weight_norm_monitor.py`:

```python
class _FakeMod:
    def __init__(self, weight):
        self.weight = weight


class _FakeChunk:
    """Minimal stand-in for a model chunk exposing named_modules()."""

    def __init__(self, named):
        self._named = named  # list[(name, module)]

    def named_modules(self):
        return iter(self._named)


def _fake_model():
    import torch

    named = [
        ("decoder.layers.0.self_attention.linear_qkv", _FakeMod(torch.ones(6, 4))),
        ("decoder.layers.0.mlp.linear_fc1", _FakeMod(torch.ones(8, 4) * 2)),
        ("decoder.layers.0.self_attention.linear_qkv.poet_linear", _FakeMod(torch.zeros(6, 4))),
        ("decoder.layers.1.self_attention.linear_qkv", _FakeMod(torch.ones(6, 4) * 3)),
        ("embedding.word_embeddings", _FakeMod(torch.ones(100, 4))),
    ]
    return [_FakeChunk(named)]


def test_collect_target_weights_filters_to_selected_layers_and_types():
    from src.patches.weight_norm_monitor import collect_target_weights

    got = collect_target_weights(_fake_model(), {0})
    labels = {(idx, mtype) for idx, mtype, _w in got}
    # layer 0 qkv + fc1 only; the poet_linear child, layer 1, and embeddings dropped
    assert labels == {(0, "qkv"), (0, "fc1")}


def test_log_weight_norms_emits_scalars_and_per_layer_histograms(monkeypatch):
    import sys
    import types

    captured = {}
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda d, step=None: captured.update(d),
        Histogram=lambda x: ("HIST", len(x)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    opts = types.SimpleNamespace(num_layers=2, weight_norm_layers="0")
    from src.patches.weight_norm_monitor import _log_weight_norms

    _log_weight_norms(_fake_model(), iteration=100, opts=opts)

    # scalar keys for both matrices in layer 0, all four kinds
    assert "weightnorm/L0/qkv/row/mean" in captured
    assert "weightnorm/L0/qkv/col_rms/max" in captured
    assert "weightnorm/L0/fc1/row/mean" in captured
    # per-layer pooled RMS histograms (one row + one col), tagged HIST
    assert captured["weightnorm/L0/row_rms_hist"][0] == "HIST"
    assert captured["weightnorm/L0/col_rms_hist"][0] == "HIST"
    # pooled row histogram length = qkv rows (6) + fc1 rows (8) = 14
    assert captured["weightnorm/L0/row_rms_hist"][1] == 14
    # layer 1 was not selected -> no keys for it
    assert not any(k.startswith("weightnorm/L1/") for k in captured)


def test_log_weight_norms_noop_when_wandb_run_is_none(monkeypatch):
    import sys
    import types

    fake_wandb = types.SimpleNamespace(run=None, log=lambda d, step=None: None)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    opts = types.SimpleNamespace(num_layers=2, weight_norm_layers="0")
    from src.patches.weight_norm_monitor import _log_weight_norms

    # must not raise even though no run is active
    _log_weight_norms(_fake_model(), iteration=100, opts=opts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_norm_monitor.py -k "collect_target or _log_weight" -v`
Expected: FAIL — `cannot import name 'collect_target_weights'`.

- [ ] **Step 3: Implement `collect_target_weights` and `_log_weight_norms`**

Append to `src/patches/weight_norm_monitor.py`:

```python
def collect_target_weights(model, selected_layers: set[int]) -> list:
    """Walk the model and return ``[(layer_index, matrix_type, weight)]`` for the
    qkv/proj/fc1/fc2 linears in ``selected_layers``. Reads each module's
    ``.weight`` (the POET base alias == W_eff post-merge; the trained weight for
    Adam/Muon). Non-2-D or missing weights are skipped.
    """
    chunks = model if isinstance(model, list) else [model]
    out = []
    for chunk in chunks:
        for name, mod in chunk.named_modules():
            cls = classify_linear(name)
            if cls is None:
                continue
            layer_idx, mtype = cls
            if layer_idx not in selected_layers:
                continue
            weight = getattr(mod, "weight", None)
            if weight is None or getattr(weight, "dim", lambda: 0)() != 2:
                continue
            out.append((layer_idx, mtype, weight))
    return out


def _log_weight_norms(model, iteration: int, opts) -> None:
    """Compute and log row/col norm summaries + per-layer RMS histograms.

    Rank-gated implicitly: returns early unless a W&B run is active (Megatron
    initializes ``wandb.run`` only on the logging rank). Never raises — callers
    wrap it defensively, but we also guard here.
    """
    import torch

    try:
        import wandb
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return

    num_layers = int(getattr(opts, "num_layers", 0) or 0)
    if num_layers <= 0:
        return
    spec = getattr(opts, "weight_norm_layers", "first,mid,last")
    selected = parse_layer_selection(spec, num_layers)
    if not selected:
        return
    targets = collect_target_weights(model, selected)
    if not targets:
        return

    payload: dict = {}
    pools: dict = {}  # layer_idx -> {"row": [vec...], "col": [vec...]}
    with torch.no_grad():
        for layer_idx, mtype, weight in targets:
            stats = compute_matrix_norm_stats(weight)
            prefix = f"weightnorm/L{layer_idx}/{mtype}"
            for kind in ("row", "col", "row_rms", "col_rms"):
                for stat_name, value in stats[kind].items():
                    payload[f"{prefix}/{kind}/{stat_name}"] = value
            pool = pools.setdefault(layer_idx, {"row": [], "col": []})
            pool["row"].append(stats["_row_rms_vec"])
            pool["col"].append(stats["_col_rms_vec"])
        for layer_idx, pool in pools.items():
            row_all = torch.cat(pool["row"]).cpu().numpy()
            col_all = torch.cat(pool["col"]).cpu().numpy()
            payload[f"weightnorm/L{layer_idx}/row_rms_hist"] = wandb.Histogram(row_all)
            payload[f"weightnorm/L{layer_idx}/col_rms_hist"] = wandb.Histogram(col_all)

    wandb.log(payload, step=iteration)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_norm_monitor.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/patches/weight_norm_monitor.py tests/unit/test_patch_weight_norm_monitor.py
git commit -m "feat(slm): weight_norm_monitor weight collection + W&B payload"
```

---

## Task 5: train_step wrapper, `apply()`, and registry/always-on wiring

**Files:**
- Modify: `src/patches/weight_norm_monitor.py`
- Modify: `launchers/pretrain_gpt_slm.py:168-173` (`_ALWAYS_ON_PATCHES`)
- Test: `tests/unit/test_patch_weight_norm_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_patch_weight_norm_monitor.py`:

```python
def test_wrapper_logs_after_inner_train_step_with_post_step_weights(monkeypatch):
    """The wrapper must call the inner train_step FIRST (so POET's merge has run)
    and only THEN read weights — i.e. it logs the post-step weight values."""
    import sys
    import types

    import torch

    captured = {}
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda d, step=None: captured.update({"_step": step, **d}),
        Histogram=lambda x: ("HIST", len(x)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    # one selected layer-0 qkv weight; inner step "merges" by scaling it to 5.0
    mod = _FakeMod(torch.ones(2, 2))
    model = [_FakeChunk([("decoder.layers.0.self_attention.linear_qkv", mod)])]

    def orig_train_step(*args, **kwargs):
        mod.weight = torch.ones(2, 2) * 5.0  # simulate the post-step / post-merge value
        return ("loss", "skipped", "grad", "extra")

    opts = types.SimpleNamespace(
        log_weight_norms=True,
        log_weight_norms_interval=10,
        weight_norm_layers="0",
        num_layers=1,
        poet=False,
        poet_merge_period=0,
    )

    from src.patches.weight_norm_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    # train_step positional args: (..., model=args[2], ..., iteration=args[7])
    ret = wrapped(None, None, model, None, None, None, None, 10)

    assert ret == ("loss", "skipped", "grad", "extra")  # pass-through unchanged
    assert captured["_step"] == 10
    # row norm of a [5,5] row = sqrt(50); reads the POST-step weight, not the pre-step ones
    assert captured["weightnorm/L0/qkv/row/max"] == pytest.approx(50.0 ** 0.5)


def test_wrapper_is_noop_when_flag_off(monkeypatch):
    import types

    calls = {"n": 0}

    def orig_train_step(*a, **k):
        calls["n"] += 1
        return "ret"

    opts = types.SimpleNamespace(log_weight_norms=False)
    from src.patches.weight_norm_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    assert wrapped(None, None, None) == "ret"
    assert calls["n"] == 1  # inner still runs; logging skipped


def test_patch_registers_with_empty_targets():
    import importlib

    from src.patches import registered_patches

    importlib.import_module("src.patches.weight_norm_monitor")
    reg = registered_patches()
    assert "weight_norm_monitor" in reg
    # targets=() so it composes with poet_merge_step's train_step wrapper
    assert reg["weight_norm_monitor"].targets == ()


def test_weight_norm_monitor_in_always_on_and_sorts_after_merge():
    from launchers.pretrain_gpt_slm import _ALWAYS_ON_PATCHES

    assert "weight_norm_monitor" in _ALWAYS_ON_PATCHES
    # registry applies in sorted order; outer wrapper must sort AFTER poet_merge_step
    assert "weight_norm_monitor" > "poet_merge_step"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_norm_monitor.py -k "wrapper or registers or always_on" -v`
Expected: FAIL — `cannot import name '_wrapped_train_step_factory'` (and the always-on test fails until Step 4).

- [ ] **Step 3: Implement the wrapper and `apply()`**

Append to `src/patches/weight_norm_monitor.py`:

```python
def _wrapped_train_step_factory(orig_train_step, get_args=None):
    """Build the outer train_step wrapper. ``get_args`` is injectable for testing;
    in production it is imported lazily from Megatron at call time.

    Calls the inner train_step FIRST (running optimizer.step and, for POET, the
    periodic merge), then — on a logging step — reads and logs the post-step
    weights. Wrapped in try/except: diagnostics must never break training.
    """

    def _wrapped(*args, **kwargs):
        ret = orig_train_step(*args, **kwargs)
        try:
            _get_args = get_args
            if _get_args is None:
                from megatron.training import get_args as _get_args  # type: ignore
            opts = _get_args()
            if not getattr(opts, "log_weight_norms", False):
                return ret

            iteration = kwargs.get("iteration")
            if iteration is None and len(args) >= 8:
                iteration = args[7]
            if iteration is None:
                iteration = getattr(opts, "iteration", 0)

            interval = int(getattr(opts, "log_weight_norms_interval", 100))
            poet = bool(getattr(opts, "poet", False))
            merge_period = int(getattr(opts, "poet_merge_period", 0))

            if poet and merge_period <= 0 and not _state["warned_merge0"]:
                logger.warning(
                    "[WNORM] POET with merge_period=0: base weight is frozen; "
                    "weight-norm logging is a no-op (W_eff is not materialized)."
                )
                _state["warned_merge0"] = True

            if not should_log(iteration, interval, poet=poet, merge_period=merge_period):
                return ret

            model = args[2] if len(args) >= 3 else kwargs.get("model")
            if model is None:
                logger.warning("[WNORM] model not found in train_step args; skipping")
                return ret
            _log_weight_norms(model, iteration, opts)
        except Exception:  # diagnostics must never break training
            logger.exception("[WNORM] weight-norm logging failed; continuing")
        return ret

    return _wrapped


@register_patch(name="weight_norm_monitor", targets=())
def apply() -> None:
    from megatron.training import training as _mt

    _mt.train_step = _wrapped_train_step_factory(_mt.train_step)
```

- [ ] **Step 4: Add to `_ALWAYS_ON_PATCHES`**

In `launchers/pretrain_gpt_slm.py`, add `"weight_norm_monitor"` to the `_ALWAYS_ON_PATCHES` tuple ([line 168-173](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L168-L173)):

```python
_ALWAYS_ON_PATCHES = (
    "wandb_trainable_params",
    "overfit_single_batch",
    "poet_grad_conditioning",
    "grad_conditioning",
    "weight_norm_monitor",
)
```

- [ ] **Step 5: Run the full patch test file to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_weight_norm_monitor.py -v`
Expected: PASS (11 passed).

- [ ] **Step 6: Commit**

```bash
git add src/patches/weight_norm_monitor.py launchers/pretrain_gpt_slm.py tests/unit/test_patch_weight_norm_monitor.py
git commit -m "feat(slm): weight_norm_monitor train_step wrapper + always-on wiring"
```

---

## Task 6: Regression check, CHANGELOG, and usage docs

**Files:**
- Modify: `CHANGELOG.md`
- Test: existing suites (registry, launcher wiring, megatron_args)

- [ ] **Step 1: Run the related existing tests to confirm no regressions**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_patch_weight_norm_monitor.py \
  tests/unit/test_megatron_args.py \
  tests/unit/test_patches_registry.py \
  tests/unit/test_launcher_patch_wiring.py \
  tests/unit/test_runtime_patch_resolution.py \
  tests/unit/test_poet_merge_step.py -v
```
Expected: PASS (note: per memory, the repo has 2 pre-existing unrelated failures elsewhere — none in these files). If a failure appears in the files above, fix it before continuing.

- [ ] **Step 2: Lint the new/modified files**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m ruff check src/patches/weight_norm_monitor.py tests/unit/test_patch_weight_norm_monitor.py src/utils/megatron_args.py launchers/pretrain_gpt_slm.py
```
Expected: PASS (no errors). Fix any reported issues.

- [ ] **Step 3: Add a CHANGELOG entry**

Add a bullet under the most recent/unreleased section at the top of `CHANGELOG.md`:

```markdown
- Add `weight_norm_monitor` patch: logs row/column norm summaries (+ per-layer RMS
  histograms) of qkv/proj/fc1/fc2 weights for a few layers to W&B, enabling
  POET vs Muon vs Adam weight-norm comparison. Gated by `training.log_weight_norms`
  (interval `log_weight_norms_interval`, layers `weight_norm_layers`).
```

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): weight_norm_monitor patch"
```

- [ ] **Step 5: Enable in experiment configs (documentation — user applies per run)**

To activate for a comparison run, add to the experiment YAML's `training:` block (do NOT commit blanket-on; it is per-experiment). Example for [poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml), and identically for the muon and adamw configs:

```yaml
training:
  log_weight_norms: true
  log_weight_norms_interval: 100   # for POET, use a multiple of optim.poet.merge_period
  weight_norm_layers: first,mid,last
```

W&B keys to overlay across the three runs:
```
weightnorm/L{i}/{qkv,proj,fc1,fc2}/{row,col,row_rms,col_rms}/{mean,std,min,max}
weightnorm/L{i}/{row,col}_rms_hist
```
Note: runs built with `--unfuse-qkv` / `--unfuse-fc1` emit `q,k,v` instead of `qkv`
and `fc1_gate,fc1_up` instead of `fc1`. For a clean 1:1 overlay, keep the same
fusion setting across the POET/Muon/Adam configs you compare; the RMS scalars and
per-layer histograms still line up regardless.

---

## Task 7: GPU smoke verification (HAND TO USER — do not run)

GPU work is the user's to run. After the CPU tests/lint pass, provide this command for the user to run a short single-GPU POET smoke with a tiny interval so weight-norm keys appear quickly, then confirm `weightnorm/...` series show up in W&B.

```bash
codexlog wnorm_smoke <existing single-GPU POET launch command> \
  training.log_weight_norms=true \
  training.log_weight_norms_interval=5 \
  training.weight_norm_layers=first,mid,last
```

Acceptance: run reaches >5 steps; `weightnorm/L0/qkv/row/mean`, the RMS variants, and `weightnorm/L0/row_rms_hist` appear in the W&B run; loss curve unchanged vs a no-flag run (the patch is read-only). Repeat with the muon and adamw configs to confirm the same keys populate for all three optimizers.

---

## Self-Review

**Spec coverage:**
- "few layers only" → Task 2 `parse_layer_selection` (default first/mid/last), Task 4 filtering. ✓
- raw-base-after-merge for POET → Task 5 outer-wrapper ordering + `should_log` merge gating; Task 4 reads `module.weight`. ✓
- merge_period=0 warn+skip → `should_log` returns False + one-time warning in wrapper. ✓
- scalars (raw + RMS) → Task 3 `compute_matrix_norm_stats`. ✓
- per-layer RMS histograms → Task 4 pooling. ✓
- model-wide / few keys intent (per selected layer) → Task 4 keys. ✓
- Adam/Muon read raw weight → same code path (no merge gating when `poet=False`). ✓
- flags + YAML plumbing → Task 1. ✓
- always-on, inert unless flag → Task 5 + `_ALWAYS_ON_PATCHES`. ✓
- patch-order (outer of poet_merge_step) → Task 5 test asserts sort order + `targets=()`. ✓
- rank/tp note → `_log_weight_norms` gates on `wandb.run` (logging rank only); tp=1 assumption documented in spec.

**Placeholder scan:** none — every step has concrete code/commands.

**Type/name consistency:** `parse_layer_selection`, `classify_linear`, `should_log` (kw-only `poet`/`merge_period`), `compute_matrix_norm_stats` (keys `row`/`col`/`row_rms`/`col_rms`/`_row_rms_vec`/`_col_rms_vec`), `collect_target_weights` → `(layer_idx, mtype, weight)`, `_log_weight_norms(model, iteration, opts)`, `_wrapped_train_step_factory(orig, get_args=None)` — all used consistently across tasks and tests. W&B key strings identical between Task 4 implementation and tests.
