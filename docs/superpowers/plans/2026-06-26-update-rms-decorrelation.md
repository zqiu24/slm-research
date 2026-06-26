# Update-RMS × Cross-Side Decorrelation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the partial-λ cross-side decorrelation ("the split, with a scale") into the update-RMS POET champion optimizer, behind the existing `poet_lie_ortho_decorrelate*` flags, so the champion can run with decorrelation on.

**Architecture:** `LieOrthUpdateRMSMomentum` ([src/optim/poet_lie_orth_update_rms.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth_update_rms.py)) gains the same 6 decorrelate constructor knobs and the same `_decorrelate_buf_alternating` method that already exist on the sibling `LieOrthMomentum` ([src/optim/poet_lie_orth.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py)). Only the *alternating* path is ported (this class is always alternating). The config keys are already fully plumbed, so wiring is one branch in [src/optim/poet.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py) plus a shared log-banner helper.

**Tech Stack:** PyTorch, Megatron-LM optimizer harness, pytest.

## Global Constraints

- **Spec:** [docs/superpowers/specs/2026-06-26-update-rms-decorrelation-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-26-update-rms-decorrelation-design.md).
- **No GPU runs.** This work delivers code + CPU tests + a sweep script + handoff commands. Do **not** launch any training run.
- **Attribute names are load-bearing.** The new optimizer attributes must be named exactly `decorrelate_sides`, `decorrelate_mode`, `decorrelate_lambda`, `decorrelate_renorm`, `decorrelate_cos_threshold`, `_decorr_pairs` — the generic bf16 master-param remap at [src/optim/poet.py:835](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L835) keys on `optimizer.decorrelate_sides` and `optimizer._decorr_pairs`.
- **Only the alternating path is ported.** Do NOT port `_decorrelate_buf` (simultaneous) — `LieOrthUpdateRMSMomentum` raises unless `alternating=True`, so it would be dead code.
- **The one porting diff vs. the source method:** this class's `slices` are 3-tuples `(off, n, p)`, not 4-tuples `(off, n, p, lr)`. The `off_by_id` comprehension changes accordingly. Everything else is copied verbatim.
- **Test env:** run pytest from the repo root with `python -m pytest`. If a test import fails on `megatron.core` (only Task 2's test imports `src.optim.poet`), first `source load_cuda13_2_nccl_env.sh` (no GPU needed). Task 1 and Task 3 tests need no env.
- **Commit style:** conventional-commit `feat(poet):` / `test(poet):` / `docs(poet):`, single short line, no AI attribution.

---

### Task 1: Port decorrelation into `LieOrthUpdateRMSMomentum`

**Files:**
- Create: `tests/unit/test_poet_lie_orth_update_rms_decorrelate.py`
- Modify: `src/optim/poet_lie_orth_update_rms.py` (constructor ~L44-127; add method after `_apply_skew_update_buffer` ~L300; wire `step()` ~L310-315)

**Interfaces:**
- Consumes: `vec_to_skew`, `skew_to_vec` (from `src.diag.skew_conditioning`, already imported in the module); `orthogonalize_skew_direction` (from `src.optim.poet_skew_muon`, already imported); `block_diag_skew`, `side_directions` (from `src.diag.poet_coordination_diag`, lazy-imported inside the new method).
- Produces: `LieOrthUpdateRMSMomentum(..., decorrelate_sides: bool=False, decorrelate_mode: str="in_off_out", decorrelate_lambda: float=1.0, decorrelate_renorm: bool=False, decorrelate_cos_threshold: float=0.0, layer_pairs=None)` with public attrs `decorrelate_sides`, `decorrelate_mode`, `decorrelate_lambda`, `decorrelate_renorm`, `decorrelate_cos_threshold`, `_decorr_pairs`, and method `_decorrelate_buf_alternating(buf, slices, active)`. `layer_pairs` is a list of `(out_param, in_param, weight, bsz_out, bsz_in)` tuples.

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/test_poet_lie_orth_update_rms_decorrelate.py`:

```python
"""Cross-side decorrelation on the update-RMS POET optimizer (alternating path).

Mirrors the alternating-decorrelate tests in test_poet_lie_orth.py, but for
LieOrthUpdateRMSMomentum. Key difference from the lie_ortho version: the buffer here
holds the ANGLE-SCALED generator (theta baked in; scatter uses alpha=1.0), so the
applied generator is `oin.data` directly (no division by lr).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.diag.skew_conditioning import vec_to_skew
from src.optim.poet_lie_orth_update_rms import LieOrthUpdateRMSMomentum
from src.optim.poet_skew_muon import orthogonalize_skew_direction


@pytest.fixture(autouse=True)
def _isolate_state():
    from poet_torch import alt_state

    torch.set_default_dtype(torch.float32)
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)
    yield
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)


def _alt_decorr_dirs(decorrelate, mode="in_off_out", active_iter=1, seed=3, **extra):
    """One alternating update-RMS step (active_iter=1 -> 'in' is written). Returns the
    applied active-in weight-space direction D_in and the inactive-out momentum
    direction D_out_mom, both in the W frame."""
    from poet_torch import alt_state

    from src.diag.poet_coordination_diag import side_directions

    torch.manual_seed(seed)
    b = 12
    ne = b * (b - 1) // 2
    oin = nn.Parameter(torch.zeros(1, ne))
    oin.grad = torch.randn(1, ne)
    oout = nn.Parameter(torch.zeros(1, ne))
    oout.grad = torch.randn(1, ne)
    W = nn.Parameter(torch.randn(b, b), requires_grad=False)
    lr = 0.05
    kw = dict(
        decorrelate_sides=decorrelate,
        decorrelate_mode=mode,
        layer_pairs=[(oout, oin, W, b, b)] if decorrelate else None,
    )
    kw.update(extra)
    opt = LieOrthUpdateRMSMomentum(
        [
            dict(params=[oin], use_skew=True, side="in", weight=W, block_size=b, lr=lr),
            dict(params=[oout], use_skew=True, side="out", weight=W, block_size=b, lr=lr),
        ],
        update_rms=0.3,
        max_angle=1.0,  # avoid the clamp so the projection identity is exact
        ortho_method="muon",
        ortho_ns_steps=5,
        **kw,
    )
    alt_state.set_iteration(active_iter)  # 1 -> active 'in'
    opt.step()
    alt_state.set_iteration(0)
    A_in = vec_to_skew(oin.data, b)  # applied generator (theta baked in; oin started at 0)
    m_out = opt.state[oout]["lie_m"]
    A_out_mom = orthogonalize_skew_direction(vec_to_skew(-m_out, b), method="muon", ns_steps=5)
    d_out_mom, d_in = side_directions(A_out_mom, A_in, W.float())
    return d_in, d_out_mom


def _cos(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def test_off_is_deterministic_and_on_changes_write():
    # decorrelate_sides defaults False -> step() skips the projection entirely. Two off
    # runs are bit-identical; enabling the feature must change the active write.
    off1, _ = _alt_decorr_dirs(decorrelate=False)
    off2, _ = _alt_decorr_dirs(decorrelate=False)
    assert torch.equal(off1, off2)
    on, _ = _alt_decorr_dirs(decorrelate=True, mode="in_off_out")
    assert _cos(off1, on) < 0.999


def test_alternating_decorrelate_removes_inactive_momentum_overlap():
    d_in_base, d_out_mom = _alt_decorr_dirs(decorrelate=False)
    base = abs(_cos(d_in_base, d_out_mom))
    assert base > 0.02, f"baseline inactive-momentum overlap should be non-trivial, got {base}"
    d_in, d_out_mom2 = _alt_decorr_dirs(decorrelate=True, mode="in_off_out")
    assert abs(_cos(d_in, d_out_mom2)) < 1e-3


@pytest.mark.parametrize("lam", [0.25, 0.5, 1.0])
def test_alternating_decorrelate_lambda_scales_overlap(lam):
    # Partial projection leaves a (1-lambda) fraction of the parallel component:
    # <D_in', D_out_mom> = (1-lambda) <D_in, D_out_mom>  (exact, renorm off).
    d_in0, d_mom = _alt_decorr_dirs(decorrelate=False)
    ip0 = (d_in0.flatten() @ d_mom.flatten()).item()
    assert abs(ip0) > 1e-3, f"baseline parallel component should be non-trivial, got {ip0}"
    d_in, d_mom2 = _alt_decorr_dirs(decorrelate=True, mode="in_off_out", decorrelate_lambda=lam)
    ip = (d_in.flatten() @ d_mom2.flatten()).item()
    assert ip == pytest.approx((1.0 - lam) * ip0, rel=2e-3, abs=1e-4)


@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_renorm_preserves_realized_norm(lam):
    # With renorm, the active side's realized ||D|| (theta-inclusive) is restored to its
    # pre-projection value -- only the direction changes. This is the §3.4 subtlety.
    d_in0, _ = _alt_decorr_dirs(decorrelate=False)
    n0 = d_in0.norm().item()
    d_in, _ = _alt_decorr_dirs(
        decorrelate=True, mode="in_off_out", decorrelate_lambda=lam, decorrelate_renorm=True
    )
    assert d_in.norm().item() == pytest.approx(n0, rel=1e-4)


def test_without_renorm_shrinks_movement():
    d_in0, _ = _alt_decorr_dirs(decorrelate=False)
    d_in, _ = _alt_decorr_dirs(decorrelate=True, mode="in_off_out", decorrelate_lambda=1.0)
    assert d_in.norm().item() < 0.999 * d_in0.norm().item()


def test_threshold_gates_module():
    d_in0, d_mom = _alt_decorr_dirs(decorrelate=False)
    cos = abs(_cos(d_in0, d_mom))
    assert cos > 0.02, f"need a non-trivial overlap to gate on, got {cos}"
    skipped, _ = _alt_decorr_dirs(
        decorrelate=True, mode="in_off_out", decorrelate_cos_threshold=cos + 0.1
    )
    assert _cos(d_in0, skipped) > 0.9999, "below-threshold layer must be left untouched"
    fired, _ = _alt_decorr_dirs(
        decorrelate=True, mode="in_off_out", decorrelate_cos_threshold=max(cos - 0.1, 0.0)
    )
    assert _cos(d_in0, fired) < 0.999, "above-threshold layer must be decorrelated"


def test_rejects_bad_mode():
    p = nn.Parameter(torch.zeros(1, 1))
    W = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    with pytest.raises(ValueError, match="decorrelate_mode"):
        LieOrthUpdateRMSMomentum(
            [dict(params=[p], use_skew=True, side="in", weight=W, block_size=2, lr=0.01)],
            decorrelate_sides=True,
            decorrelate_mode="bogus",
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_poet_lie_orth_update_rms_decorrelate.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'decorrelate_sides'` (the constructor doesn't accept the new args yet).

- [ ] **Step 3: Add the constructor knobs + validation + attributes**

In `src/optim/poet_lie_orth_update_rms.py`, add the 6 parameters to `__init__` between `dp_group=None,` and `adamw_betas=(0.9, 0.95),`:

```python
        dp_group=None,
        decorrelate_sides: bool = False,
        decorrelate_mode: str = "in_off_out",
        decorrelate_lambda: float = 1.0,
        decorrelate_renorm: bool = False,
        decorrelate_cos_threshold: float = 0.0,
        layer_pairs=None,
        adamw_betas=(0.9, 0.95),
```

Add a validation check next to the other `if ... raise ValueError` guards near the top of `__init__` (e.g. right after the `rms_mode` checks):

```python
        if decorrelate_mode not in ("in_off_out", "out_off_in", "symmetric"):
            raise ValueError(
                "decorrelate_mode must be 'in_off_out' | 'out_off_in' | 'symmetric', "
                f"got {decorrelate_mode!r}"
            )
```

Store the attributes right after `self.dp_group = dp_group`:

```python
        self.dp_group = dp_group
        # Cross-side decorrelation ("the split, with a scale"): project the active
        # written generator off the inactive side's maintained-momentum direction so
        # cos(D_out, D_in) -> 0. lambda = partial-projection fraction (0=off, 1=full);
        # renorm = restore the active side's realized ||D|| (direction-only change);
        # cos_threshold = module-selective gate. Ported from LieOrthMomentum; alternating
        # path only (this optimizer is always alternating). layer_pairs entries are
        # (out_param, in_param, weight, bsz_out, bsz_in).
        self.decorrelate_sides = bool(decorrelate_sides)
        self.decorrelate_mode = decorrelate_mode
        self.decorrelate_lambda = float(decorrelate_lambda)
        self.decorrelate_renorm = bool(decorrelate_renorm)
        self.decorrelate_cos_threshold = float(decorrelate_cos_threshold)
        self._decorr_pairs = list(layer_pairs) if layer_pairs else []
```

- [ ] **Step 4: Add the `_decorrelate_buf_alternating` method**

In `src/optim/poet_lie_orth_update_rms.py`, add this method immediately after `_apply_skew_update_buffer` (before `step`). It is the source method from `poet_lie_orth.py` with the single 3-tuple `off_by_id` change:

```python
    def _decorrelate_buf_alternating(self, buf, slices, active):
        """Alternating-mode cross-side decorrelation. Only the ACTIVE side is written this
        step (the inactive side's buf slice is zero), so source the inactive side's
        weight-space direction from its MAINTAINED momentum (lie_m) and project the active
        written generator off it: "don't keep pushing along the direction the other side
        just moved." Modifies only the active side. `decorrelate_mode` selects WHICH
        active-side steps are treated: in_off_out -> in-write steps; out_off_in -> out-write
        steps; symmetric -> every step. Mutates buf in place. (Ported from
        LieOrthMomentum; slices here are 3-tuples (off, n, p).)"""
        if not self._decorr_pairs or active is None:
            return
        from src.diag.poet_coordination_diag import block_diag_skew, side_directions

        off_by_id = {id(p): (off, n) for off, n, p in slices}
        eps = 1e-12
        mode = self.decorrelate_mode
        matched = 0
        for out_p, in_p, w, bsz_out, bsz_in in self._decorr_pairs:
            so, si = off_by_id.get(id(out_p)), off_by_id.get(id(in_p))
            if so is None or si is None:
                continue
            matched += 1
            (oo, no), (oi, ni) = so, si
            W = w.detach().to(torch.float32)
            if active == "in":
                if mode == "out_off_in":
                    continue  # would modify the inactive (unwritten) out side -> no-op
                act_off, act_n, act_bsz, act_p = oi, ni, bsz_in, in_p
                inact_p, inact_bsz = out_p, bsz_out
            else:  # active == "out"
                if mode == "in_off_out":
                    continue
                act_off, act_n, act_bsz, act_p = oo, no, bsz_out, out_p
                inact_p, inact_bsz = in_p, bsz_in
            m_inact = self.state[inact_p].get("lie_m")
            if m_inact is None:
                continue
            A_act = vec_to_skew(buf[act_off : act_off + act_n].view(act_p.shape[0], -1), act_bsz)
            A_inact = orthogonalize_skew_direction(
                vec_to_skew(-m_inact.float(), inact_bsz),
                method=self.ortho_method,
                ns_steps=self.ortho_ns_steps,
            )
            if active == "in":
                d_out, d_in = side_directions(A_inact, A_act, W)
                d_act = d_in  # the active (in) side's weight-space direction
                g = block_diag_skew(W.transpose(-2, -1) @ d_out, act_bsz)
            else:
                d_out, d_in = side_directions(A_act, A_inact, W)
                d_act = d_out  # the active (out) side's weight-space direction
                g = block_diag_skew(d_in @ W.transpose(-2, -1), act_bsz)
            if self.decorrelate_cos_threshold > 0.0:
                denom = (d_out.norm() * d_in.norm()).clamp_min(eps)
                if (
                    abs(float((d_out.flatten() @ d_in.flatten()) / denom))
                    < self.decorrelate_cos_threshold
                ):
                    continue
            c = (A_act.flatten() @ g.flatten()) / (g.flatten() @ g.flatten()).clamp_min(eps)
            A_act = A_act - self.decorrelate_lambda * c * g
            if self.decorrelate_renorm:
                if active == "in":
                    _, d_act_new = side_directions(A_inact, A_act, W)
                else:
                    d_act_new, _ = side_directions(A_act, A_inact, W)
                A_act = A_act * (d_act.norm() / d_act_new.norm().clamp_min(eps))
            buf[act_off : act_off + act_n] = skew_to_vec(A_act, act_bsz).reshape(-1)
        if matched == 0 and not getattr(self, "_decorr_warned", False):
            self._decorr_warned = True
            import logging

            logging.getLogger(__name__).warning(
                "[decorrelate/alt] decorrelate_sides=True but 0/%d pairs matched the "
                "optimizer's slices — decorrelation is a NO-OP (param identity mismatch).",
                len(self._decorr_pairs),
            )
```

- [ ] **Step 5: Wire the call into `step()`**

In `src/optim/poet_lie_orth_update_rms.py` `step()`, the existing block is:

```python
        buf, slices = self._skew_update_buffer(self._dp_rank, self._dp_world_size, active)
        if self.distributed and self._dp_world_size > 1 and buf.numel() > 0:
            import torch.distributed as dist

            dist.all_reduce(buf, group=self.dp_group)
        self._apply_skew_update_buffer(buf, slices)
```

Insert the decorrelation call after the all-reduce, before apply:

```python
        buf, slices = self._skew_update_buffer(self._dp_rank, self._dp_world_size, active)
        if self.distributed and self._dp_world_size > 1 and buf.numel() > 0:
            import torch.distributed as dist

            dist.all_reduce(buf, group=self.dp_group)
        if self.decorrelate_sides:
            self._decorrelate_buf_alternating(buf, slices, active)
        self._apply_skew_update_buffer(buf, slices)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/unit/test_poet_lie_orth_update_rms_decorrelate.py -q`
Expected: PASS (all tests, including the 3 `lambda` and 2 `renorm` parametrizations).

- [ ] **Step 7: Run the existing optimizer tests + compile check (no regression)**

Run: `python -m pytest tests/unit/test_poet_lie_orth_update_rms.py tests/unit/test_poet_lie_orth.py -q && python -m py_compile src/optim/poet_lie_orth_update_rms.py`
Expected: PASS / no output from py_compile.

- [ ] **Step 8: Commit**

```bash
git add src/optim/poet_lie_orth_update_rms.py tests/unit/test_poet_lie_orth_update_rms_decorrelate.py
git commit -m "feat(poet): cross-side decorrelation on the update-RMS optimizer (alternating path)"
```

---

### Task 2: Wire the decorrelate flags into the `lie_ortho_update_rms` build branch

**Files:**
- Modify: `src/optim/poet.py` (add module-level `_log_decorrelate_banner`; replace the inline banner in the `lie_ortho` branch ~L720-733; add banner call + 6 kwargs in the `lie_ortho_update_rms` branch ~L761-792)
- Test: `tests/unit/test_poet_decorrelate_banner.py`

**Interfaces:**
- Consumes: `LieOrthUpdateRMSMomentum(..., decorrelate_*, layer_pairs=...)` from Task 1; `_build_decorrelate_pairs(model_chunks)` (already imported in poet.py at L605, from `src.optim.poet_lie_momentum`).
- Produces: module-level `_log_decorrelate_banner(config, logger) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_poet_decorrelate_banner.py`:

```python
"""The CROSS-SIDE DECORRELATION startup banner (the sweep scripts' tripwire)."""

from __future__ import annotations

import logging
import types

from src.optim.poet import _log_decorrelate_banner


def test_banner_reports_lambda_and_mode(caplog):
    cfg = types.SimpleNamespace(
        poet_lie_alternating=True,
        poet_lie_ortho_decorrelate_mode="symmetric",
        poet_lie_ortho_decorrelate_lambda=0.5,
        poet_lie_ortho_decorrelate_renorm=True,
        poet_lie_ortho_decorrelate_cos_threshold=0.0,
    )
    logger = logging.getLogger("test.decorr.banner")
    with caplog.at_level(logging.WARNING, logger="test.decorr.banner"):
        _log_decorrelate_banner(cfg, logger)
    msg = caplog.text
    assert "CROSS-SIDE DECORRELATION ON" in msg
    assert "mode=symmetric" in msg
    assert "lambda=0.5" in msg
    assert "renorm=True" in msg
    assert "alternating=True" in msg
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_poet_decorrelate_banner.py -q`
Expected: FAIL — `ImportError: cannot import name '_log_decorrelate_banner' from 'src.optim.poet'`.
(If it instead fails on `import megatron.core`, run `source load_cuda13_2_nccl_env.sh` first, then re-run.)

- [ ] **Step 3: Add the module-level banner helper**

In `src/optim/poet.py`, add this function at module level (e.g. just above `get_megatron_poet_lie_momentum_optimizer` at ~L586):

```python
def _log_decorrelate_banner(config, logger) -> None:
    """One-line CROSS-SIDE DECORRELATION banner — the sweep scripts grep this at startup
    to confirm the override was not silently dropped. Shared by the lie_ortho and
    lie_ortho_update_rms branches."""
    _alt_on = bool(getattr(config, "poet_lie_alternating", False))
    logger.warning(
        "[POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=%s, lambda=%s, renorm=%s, "
        "cos_threshold=%s, alternating=%s) — projects the active generator off the "
        "other side's direction so cos(D_out,D_in)->0. Alternating: the inactive "
        "direction is sourced from its maintained momentum (cross-step over-spend "
        "control); simultaneous: both sides every step (ANALYSIS §17.6).",
        getattr(config, "poet_lie_ortho_decorrelate_mode", "in_off_out"),
        getattr(config, "poet_lie_ortho_decorrelate_lambda", 1.0),
        getattr(config, "poet_lie_ortho_decorrelate_renorm", False),
        getattr(config, "poet_lie_ortho_decorrelate_cos_threshold", 0.0),
        _alt_on,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/unit/test_poet_decorrelate_banner.py -q`
Expected: PASS.

- [ ] **Step 5: Replace the inline banner in the `lie_ortho` branch with the helper**

In `src/optim/poet.py`, the `lie_ortho` branch currently has (~L720-733):

```python
        if _lie_ortho_decorrelate:
            _alt_on = bool(getattr(config, "poet_lie_alternating", False))
            logger.warning(
                "[POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=%s, lambda=%s, renorm=%s, "
                "cos_threshold=%s, alternating=%s) — projects the active generator off the "
                "other side's direction so cos(D_out,D_in)->0. Alternating: the inactive "
                "direction is sourced from its maintained momentum (cross-step over-spend "
                "control); simultaneous: both sides every step (ANALYSIS §17.6).",
                getattr(config, "poet_lie_ortho_decorrelate_mode", "in_off_out"),
                getattr(config, "poet_lie_ortho_decorrelate_lambda", 1.0),
                getattr(config, "poet_lie_ortho_decorrelate_renorm", False),
                getattr(config, "poet_lie_ortho_decorrelate_cos_threshold", 0.0),
                _alt_on,
            )
```

Replace that whole block with:

```python
        if _lie_ortho_decorrelate:
            _log_decorrelate_banner(config, logger)
```

- [ ] **Step 6: Wire the decorrelate kwargs + banner into the `lie_ortho_update_rms` branch**

In `src/optim/poet.py`, the `elif q_optimizer == "lie_ortho_update_rms":` branch begins by computing `_dp_world/_dp_rank/_dp_group` and logging. Add, right after the existing `logger.info(...)` call in that branch (before `optimizer = LieOrthUpdateRMSMomentum(`):

```python
        _lie_ortho_decorrelate = bool(getattr(config, "poet_lie_ortho_decorrelate", False))
        if _lie_ortho_decorrelate:
            _log_decorrelate_banner(config, logger)
```

Then extend the `LieOrthUpdateRMSMomentum(...)` call — add these kwargs alongside the existing `dp_group=_dp_group,` (before `**shared_kwargs,`):

```python
            dp_group=_dp_group,
            decorrelate_sides=_lie_ortho_decorrelate,
            decorrelate_mode=getattr(config, "poet_lie_ortho_decorrelate_mode", "in_off_out"),
            decorrelate_lambda=getattr(config, "poet_lie_ortho_decorrelate_lambda", 1.0),
            decorrelate_renorm=getattr(config, "poet_lie_ortho_decorrelate_renorm", False),
            decorrelate_cos_threshold=getattr(
                config, "poet_lie_ortho_decorrelate_cos_threshold", 0.0
            ),
            layer_pairs=_build_decorrelate_pairs(model_chunks) if _lie_ortho_decorrelate else None,
            **shared_kwargs,
```

- [ ] **Step 7: Compile + re-run the banner test and the existing optimizer-setup tests**

Run: `python -m py_compile src/optim/poet.py && python -m pytest tests/unit/test_poet_decorrelate_banner.py tests/unit/test_patch_poet_optimizer_setup.py -q`
Expected: no py_compile output; tests PASS.
(If imports fail on `megatron.core`, `source load_cuda13_2_nccl_env.sh` first.)

- [ ] **Step 8: Commit**

```bash
git add src/optim/poet.py tests/unit/test_poet_decorrelate_banner.py
git commit -m "feat(poet): wire decorrelate flags into the update-RMS build branch"
```

---

### Task 3: Stage-1 sweep script + handoff

**Files:**
- Create: `scripts/sweep_update_rms_decorrelate.sh`

**Interfaces:**
- Consumes: `scripts/train_poet_lie_orth_update_rms.sh` (sets `q_optimizer=lie_ortho_update_rms`); the `optim.poet.lie_ortho_decorrelate*` Hydra keys (already plumbed); the `codexlog` wrapper.
- Produces: a runnable sweep over `decorrelate_lambda ∈ {0.25, 0.50, 0.75}` on the symmetric update-RMS baseline (the 3.4758 recipe), for the operator to run.

- [ ] **Step 1: Write the sweep script**

Create `scripts/sweep_update_rms_decorrelate.sh`:

```bash
#!/usr/bin/env bash
# Stage 1: cross-side decorrelation ("the split, with a scale") stacked on the SYMMETRIC
# update-RMS champion (POET_dev.md §2.11/§2.12). Baseline to beat: the symmetric
# mup_normalized a4 / rho0.30 / side_gamma=0 / lr5 run = val 3.4758 (§2.11). If the §J.3
# lambda=0.5 win (-0.0070) carries over -> ~3.4688, a new POET record.
#
# Held recipe = the symmetric baseline; only the decorrelation knobs move. The inactive
# side's direction is sourced from its maintained momentum (lie_m) and the active write is
# projected off it (cross-step over-spend control). mode=symmetric (fires every step),
# renorm=true (direction-only change), all layers (cos_threshold=0). lambda is swept; 1.0
# is deliberately EXCLUDED (catastrophic in §J.3 via the renorm pathology).
#
# Runs the three lambda arms SEQUENTIALLY; background or split across GPUs to parallelize.
#
# TRIPWIRE: each run's startup log must print
#   [POET] Lie-orth CROSS-SIDE DECORRELATION ON (mode=symmetric, lambda=<L>, renorm=True,
#          cos_threshold=0.0, alternating=True)
# If lambda/mode/renorm/alternating do NOT match the arm, an override was silently dropped
# — kill it and fix before trusting the result.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export CODEX_LOG_DIR="${CODEX_LOG_DIR:-/lustre/home/zqiu/log}"
mkdir -p "$CODEX_LOG_DIR"
# Log the cos(D_out,D_in) trajectory so the overlap is visibly driven down during the run.
export SLM_POET_COORD_DIAG=1
export SLM_POET_COORD_DIAG_INTERVAL=250

HELD="scheduler=cosine_poet training_regime=ablation_40x \
  optim.lr=0.005 \
  optim.poet.lie_ortho_update_rms=0.30 \
  optim.poet.lie_ortho_max_angle=0.024 \
  optim.poet.lie_ortho_update_rms_side_gamma=0.0 \
  optim.poet.init_type=mup_normalized optim.poet.mup_alpha=4 optim.poet.scale=1.0 \
  optim.poet.lie_ortho_method=muon optim.poet.head_aligned_attn=false \
  optim.poet.lie_alternating=true optim.poet.lie_alternate_every=1 \
  optim.poet.lie_ortho_nesterov=true optim.poet.lie_b1=0.95 \
  optim.poet.lie_ortho_distributed=true optim.poet.lie_ortho_rms_mode=weight"

for LAM in 0.25 0.50 0.75; do
  TAG="${LAM/./p}"                       # 0.25 -> 0p25
  NAME="urms_decorr_sym_renorm_l${TAG}"
  echo ">>> ${NAME} (lambda=${LAM}) starting"
  codexlog "${NAME}" scripts/train_poet_lie_orth_update_rms.sh llama3 ${HELD} \
    optim.poet.lie_ortho_decorrelate=true \
    optim.poet.lie_ortho_decorrelate_mode=symmetric \
    optim.poet.lie_ortho_decorrelate_renorm=true \
    optim.poet.lie_ortho_decorrelate_lambda="${LAM}" \
    optim.poet.lie_ortho_decorrelate_cos_threshold=0.0 \
    experiment.name="${NAME}"
  echo "<<< ${NAME} done (status $?)"
done
echo "=== update-RMS decorrelation Stage-1 sweep complete: lambda {0.25,0.50,0.75} vs baseline 3.4758 ==="
```

- [ ] **Step 2: Syntax-check the script + make it executable**

Run: `bash -n scripts/sweep_update_rms_decorrelate.sh && chmod +x scripts/sweep_update_rms_decorrelate.sh && echo OK`
Expected: `OK` (no syntax errors).

- [ ] **Step 3: Commit**

```bash
git add scripts/sweep_update_rms_decorrelate.sh
git commit -m "feat(poet): Stage-1 update-RMS decorrelation lambda sweep script"
```

---

### Task 4: Record results scaffold in POET_dev.md + memory note

**Files:**
- Modify: `POET_dev.md` (add a §2.14 stub for the new sweep)

- [ ] **Step 1: Add a §2.14 stub to POET_dev.md**

Append a new subsection after §2.13 (Pion) in `POET_dev.md`. Use the same table style as §2.11/§2.12. Fill the baseline row from §2.11; leave the λ rows blank for the operator to fill:

```markdown
## 2.14 update-RMS × cross-side decorrelation — `lambda` sweep (live, filling)

The "split, with a scale": the §J.3 partial-λ cross-side decorrelation, ported into the
update-RMS champion (`q_optimizer=lie_ortho_update_rms`, alternating path). Stacked on the
**symmetric** baseline (`mup_normalized` α4 / ρ0.30 / **side_γ=0** / lr5 / max∠0.024) so the
measured delta is decorrelation alone. `mode=symmetric`, `renorm=true`, all layers
(`cos_threshold=0`); λ=1.0 excluded (catastrophic in §J.3). **Baseline = 3.4758** (§2.11).
Run: `bash scripts/sweep_update_rms_decorrelate.sh`.

| `decorrelate_lambda` | run dir / W&B | val/loss | Δ vs 3.4758 |
|---|---|---|---|
| 0 (baseline) | §2.11 `mup` ρ0.30 | 3.4758 | — |
| 0.25 | ▶ | | |
| 0.50 | ▶ | | |
| 0.75 | ▶ | | |
```

- [ ] **Step 2: Commit**

```bash
git add POET_dev.md
git commit -m "docs(poet): §2.14 update-RMS decorrelation lambda sweep stub"
```

- [ ] **Step 3: Update auto-memory**

After the sweep lands and is committed, add/refresh a memory file under
`/home/zqiu/.claude/projects/-lustre-fast-fast-zqiu-slm-research/memory/` (type `project`)
recording that decorrelation is now available on the update-RMS optimizer (`q_optimizer=lie_ortho_update_rms` + `poet_lie_ortho_decorrelate=true`), link it to `[[poet-alt-decorrelate-feature]]`, and add the one-line pointer to `MEMORY.md`. (Do this as a closing step, not before the code is committed.)

---

## Self-Review

**1. Spec coverage:**
- Spec §3 (optimizer change: constructor knobs, `_decorrelate_buf_alternating`, `step()` ordering, the θ-baked subtlety) → Task 1 (Steps 3-5; the subtlety is pinned by `test_renorm_preserves_realized_norm`). ✓
- Spec §4 (wiring: pass 6 kwargs, factor the banner helper, no changes to optimizer_setup/megatron_args/remap) → Task 2. ✓
- Spec §5 (tests: off-identical, not-a-noop, removes-overlap, λ-scales, renorm-preserves, bad-mode) → Task 1 Step 1 (all six present; "off-identical" realized as determinism + on-changes-write, since the feature is structurally gated). ✓
- Spec §6 (sweep script on the symmetric baseline, λ∈{0.25,0.5,0.75}, banner tripwire) → Task 3. ✓
- Spec §7 (risks: single-seed margin, λ=1.0 excluded) → reflected in Task 3 script comments + Task 4 stub. ✓

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to" — every code step shows complete code. The §2.14 table's blank λ rows are operator-filled *results*, not implementation placeholders. ✓

**3. Type consistency:** Constructor attr names (`decorrelate_sides/_mode/_lambda/_renorm/_cos_threshold`, `_decorr_pairs`) are identical in Task 1 (definition), the Global Constraints (remap dependency), and Task 2 (kwargs passed). Method name `_decorrelate_buf_alternating(buf, slices, active)` matches between definition (Task 1 Step 4) and call site (Task 1 Step 5). `_log_decorrelate_banner(config, logger)` matches between definition (Task 2 Step 3) and both call sites (Task 2 Steps 5-6). `layer_pairs` tuple shape `(out, in, weight, bsz_out, bsz_in)` is consistent between the test helper, `_build_decorrelate_pairs`, and the method's unpacking. ✓
