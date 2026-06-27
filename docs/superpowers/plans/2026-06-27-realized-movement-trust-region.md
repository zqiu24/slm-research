# Realized-Movement Trust Region (M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-step realized-movement trust region to `LieOrthUpdateRMSMomentum` so the POET rotation targets a fixed *weight movement* (`‖D_act‖/‖W‖ ≤ ρ`) with an adaptive angle, instead of a fixed angle.

**Architecture:** A new optimizer method `_movement_trust_region(buf, slices, active)` runs in `step()` after decorrelation, on the already-`all_reduce`d generator buffer. For each active skew param it forms the realized first-order move `D_act = blockdiag(A)·W` (one `bmm`/`einsum`, reusing the block-contiguous `group["weight"]`), computes `r = ‖D_act‖_F/‖W‖_F`, and rescales the generator: `clip` shrinks only the over-budget tail, `normalize` rescales every step to budget, `measure` logs `r` without changing the write. Three new `optim.poet.*` flags flow through the standard 4-edit plumbing; default `mode=off` leaves the champion path bit-identical.

**Tech Stack:** PyTorch, Megatron-LM optimizer wiring, pytest (CPU-only unit tests).

## Global Constraints

- Target optimizer only: `LieOrthUpdateRMSMomentum` in [src/optim/poet_lie_orth_update_rms.py](../../../src/optim/poet_lie_orth_update_rms.py). Do NOT touch `LieOrthMomentum` (`poet_lie_orth.py`).
- Default behavior unchanged: `move_control_mode="off"` ⇒ `step()` skips the new method ⇒ champion path bit-identical.
- `move_budget_rho > 0` is REQUIRED when `move_control_mode in {"clip","normalize"}`; allowed to be 0 for `"off"`/`"measure"`.
- Flag names (verbatim): optimizer kwargs `move_control_mode` / `move_budget_rho` / `move_lambda`; CLI `--poet-lie-move-control-mode` / `--poet-lie-move-budget-rho` / `--poet-lie-move-lambda`; config attrs `poet_lie_move_control_mode` / `poet_lie_move_budget_rho` / `poet_lie_move_lambda`; YAML keys `optim.poet.lie_move_control_mode` / `lie_move_budget_rho` / `lie_move_lambda`.
- Mode vocabulary (verbatim): `"off" | "measure" | "clip" | "normalize"`.
- Stats keys (verbatim): `poet_move/ratio_mean`, `poet_move/ratio_p50`, `poet_move/ratio_p90`, `poet_move/ratio_p95`, `poet_move/ratio_max`.
- `D_act` is built in the same block-contiguous frame as `side_directions`: out-active `D = bmm(A, W.reshape(nb,bsz,in))`; in-active `D = einsum("orb,rbc->orc", W.reshape(out,nb,bsz), A)`.
- The args→config copy in [poet_optimizer_setup.py](../../../src/patches/poet_optimizer_setup.py) is the historically silent-no-op edit — it MUST be present or the flag is dropped.
- CPU pytest is runnable in-session: `python -m pytest <file> -v`. Wrap any cluster run command in `codexlog <name> …`.

---

### Task 1: Optimizer core — trust-region method + flags

**Files:**
- Modify: `src/optim/poet_lie_orth_update_rms.py` (constructor kwargs/validation/attrs near lines 66–75 & 87–97 & 129–134; new method before `step()` at line 417; call inside `step()` at line 431–433)
- Test: `tests/unit/test_poet_movement_trust_region.py` (create)

**Interfaces:**
- Consumes: existing `self._iter_skew_params()`, `self._active_side()`, `vec_to_skew`, `block_size_from_nelems` (already imported at line 20), `group["weight"]`, `group["block_size"]`, optional `group["gain"]`.
- Produces:
  - kwargs `move_control_mode: str = "off"`, `move_budget_rho: float = 0.0`, `move_lambda: float = 1.0`
  - attrs `self.move_control_mode`, `self.move_budget_rho`, `self.move_lambda`
  - method `_movement_trust_region(self, buf, slices, active) -> None` (mutates `buf` in place; clip/normalize rescale, measure is log-only, off returns early)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_poet_movement_trust_region.py`:

```python
"""Realized-movement trust region (M1) on LieOrthUpdateRMSMomentum.

Mirrors the decorrelate-test harness: one alternating step with active 'in'
(active_iter=1), then re-derive the applied in-side weight-space move
D_in = W @ blockdiag(A_in) and its ratio r = ||D_in||_F / ||W||_F.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.diag.poet_coordination_diag import side_directions
from src.diag.skew_conditioning import vec_to_skew
from src.optim.poet_lie_orth_update_rms import LieOrthUpdateRMSMomentum


@pytest.fixture(autouse=True)
def _isolate_state():
    from poet_torch import alt_state

    torch.set_default_dtype(torch.float32)
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)
    yield
    alt_state.set_iteration(0)
    alt_state.set_fixed_side(None)


def _move_step(mode="off", rho=0.0, lam=1.0, seed=3, active_iter=1, **extra):
    """One alternating update-RMS step (active_iter=1 -> 'in' written). Returns
    (ratio, applied_in_vec, optimizer)."""
    from poet_torch import alt_state

    torch.manual_seed(seed)
    b = 12
    ne = b * (b - 1) // 2
    oin = nn.Parameter(torch.zeros(1, ne))
    oin.grad = torch.randn(1, ne)
    oout = nn.Parameter(torch.zeros(1, ne))
    oout.grad = torch.randn(1, ne)
    W = nn.Parameter(torch.randn(b, b), requires_grad=False)
    opt = LieOrthUpdateRMSMomentum(
        [
            dict(params=[oin], use_skew=True, side="in", weight=W, block_size=b, lr=0.05),
            dict(params=[oout], use_skew=True, side="out", weight=W, block_size=b, lr=0.05),
        ],
        update_rms=0.3,
        max_angle=1.0,  # no clamp, so the trust-region identity is exact
        ortho_method="muon",
        ortho_ns_steps=5,
        move_control_mode=mode,
        move_budget_rho=rho,
        move_lambda=lam,
        **extra,
    )
    alt_state.set_iteration(active_iter)
    opt.step()
    alt_state.set_iteration(0)
    A_in = vec_to_skew(oin.data, b)
    _, d_in = side_directions(torch.zeros(1, b, b), A_in, W.float())
    ratio = (d_in.norm() / W.float().norm()).item()
    return ratio, oin.data.clone(), opt


def test_off_is_bit_identical():
    _, v1, _ = _move_step(mode="off")
    _, v2, _ = _move_step(mode="off")
    assert torch.equal(v1, v2)


def test_clip_noop_when_under_budget():
    r0, v0, _ = _move_step(mode="off")
    r1, v1, _ = _move_step(mode="clip", rho=10.0 * r0)
    assert torch.equal(v0, v1)
    assert r1 == pytest.approx(r0, rel=1e-5)


def test_clip_to_budget_when_over():
    r0, _, _ = _move_step(mode="off")
    rho = 0.5 * r0
    r1, _, _ = _move_step(mode="clip", rho=rho, lam=1.0)
    assert r1 == pytest.approx(rho, rel=2e-3)


def test_clip_lambda_partial():
    r0, _, _ = _move_step(mode="off")
    rho = 0.5 * r0  # rho/r0 = 0.5, f = 1 - 0.5*(1-0.5) = 0.75
    r1, _, _ = _move_step(mode="clip", rho=rho, lam=0.5)
    assert r1 == pytest.approx(0.75 * r0, rel=2e-3)


@pytest.mark.parametrize("mult", [0.5, 2.0])
def test_normalize_hits_budget(mult):
    r0, _, _ = _move_step(mode="off")
    rho = mult * r0
    r1, _, _ = _move_step(mode="normalize", rho=rho)
    assert r1 == pytest.approx(rho, rel=2e-3)


def test_measure_does_not_change_write():
    _, v0, _ = _move_step(mode="off")
    _, v1, opt = _move_step(mode="measure")
    assert torch.equal(v0, v1)
    assert "poet_move/ratio_mean" in opt.last_update_rms_stats


def test_rejects_bad_mode():
    p = nn.Parameter(torch.zeros(1, 1))
    W = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    with pytest.raises(ValueError, match="move_control_mode"):
        LieOrthUpdateRMSMomentum(
            [dict(params=[p], use_skew=True, side="in", weight=W, block_size=2, lr=0.01)],
            move_control_mode="bogus",
        )


def test_requires_rho_when_active():
    p = nn.Parameter(torch.zeros(1, 1))
    W = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    with pytest.raises(ValueError, match="move_budget_rho"):
        LieOrthUpdateRMSMomentum(
            [dict(params=[p], use_skew=True, side="in", weight=W, block_size=2, lr=0.01)],
            move_control_mode="clip",
            move_budget_rho=0.0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_poet_movement_trust_region.py -v`
Expected: FAIL — `LieOrthUpdateRMSMomentum.__init__() got an unexpected keyword argument 'move_control_mode'`.

- [ ] **Step 3: Add constructor kwargs**

In `src/optim/poet_lie_orth_update_rms.py`, add to the `__init__` signature (after `decorrelate_cos_threshold: float = 0.0,` at line 70):

```python
        move_control_mode: str = "off",
        move_budget_rho: float = 0.0,
        move_lambda: float = 1.0,
```

- [ ] **Step 4: Add validation**

In `__init__`, after the `decorrelate_mode` validation block (after line 91):

```python
        if move_control_mode not in ("off", "measure", "clip", "normalize"):
            raise ValueError(
                "move_control_mode must be 'off' | 'measure' | 'clip' | 'normalize', "
                f"got {move_control_mode!r}"
            )
        if move_control_mode in ("clip", "normalize") and not (move_budget_rho > 0):
            raise ValueError(
                "move_budget_rho must be positive when move_control_mode is "
                f"'clip'/'normalize', got {move_budget_rho!r}"
            )
```

- [ ] **Step 5: Store attributes**

In `__init__`, after `self._decorr_pairs = ...` (line 134):

```python
        # Realized-movement trust region (M1): scale the active written generator so its
        # realized first-order move ||D_act||_F / ||W||_F stays within move_budget_rho.
        # 'clip' shrinks only the over-budget tail (move_lambda = partial fraction);
        # 'normalize' rescales every step to the budget; 'measure' logs r without changing
        # the write; 'off' is a no-op. Independent of decorrelation (runs after it).
        self.move_control_mode = move_control_mode
        self.move_budget_rho = float(move_budget_rho)
        self.move_lambda = float(move_lambda)
```

- [ ] **Step 6: Add the `_movement_trust_region` method**

In `src/optim/poet_lie_orth_update_rms.py`, immediately before `def step` (line 417):

```python
    def _movement_trust_region(self, buf, slices, active):
        """Realized-movement trust region (M1). For each active skew param, form the
        realized first-order move D_act = blockdiag(A)·W (out) / W·blockdiag(A) (in) in the
        block-contiguous frame, compute r = ||D_act||_F / ||W||_F, and rescale the written
        generator: 'clip' shrinks only when r > rho (move_lambda partial), 'normalize'
        rescales to r == rho, 'measure' only logs. Runs AFTER decorrelation on the
        already-all_reduced buf (D_act is the FINAL write; W replicated => f identical on
        every rank). Mutates buf in place."""
        if self.move_control_mode == "off" or active is None:
            return
        off_by_id = {id(p): (off, n) for off, n, p in slices}
        eps = 1e-12
        rho = self.move_budget_rho
        ratios = []
        for p, group in self._iter_skew_params():
            if group["side"] != active:
                continue
            slot = off_by_id.get(id(p))
            if slot is None:
                continue
            off, n = slot
            bsz = int(group.get("block_size") or block_size_from_nelems(p.shape[1]))
            W = group["weight"].detach().to(torch.float32)
            out_features, in_features = W.shape
            A = vec_to_skew(buf[off : off + n].view(p.shape[0], -1), bsz)
            nb = A.shape[0]
            if active == "out":
                D = torch.bmm(A, W.reshape(nb, bsz, in_features)).reshape(
                    out_features, in_features
                )
            else:  # active == "in"
                D = torch.einsum(
                    "orb,rbc->orc", W.reshape(out_features, nb, bsz), A
                ).reshape(out_features, in_features)
            w_norm = W.norm()
            gain = group.get("gain")
            if gain is not None:
                w_norm = w_norm * gain.detach().abs().to(
                    device=w_norm.device, dtype=w_norm.dtype
                ).clamp_min(eps)
            ratio = float(D.norm() / w_norm.clamp_min(eps))
            ratios.append(ratio)
            if self.move_control_mode == "measure" or ratio <= eps:
                continue
            if self.move_control_mode == "normalize":
                f = rho / ratio
            else:  # clip
                if ratio <= rho:
                    continue
                f = 1.0 - self.move_lambda * (1.0 - rho / ratio)
            buf[off : off + n] *= f
        self._set_move_stats(ratios)

    def _set_move_stats(self, ratios):
        if not ratios:
            return
        t = torch.tensor(ratios, dtype=torch.float32)
        self.last_update_rms_stats.update(
            {
                "poet_move/ratio_mean": t.mean().detach(),
                "poet_move/ratio_p50": torch.quantile(t, 0.5).detach(),
                "poet_move/ratio_p90": torch.quantile(t, 0.9).detach(),
                "poet_move/ratio_p95": torch.quantile(t, 0.95).detach(),
                "poet_move/ratio_max": t.max().detach(),
            }
        )
```

- [ ] **Step 7: Call it in `step()`**

In `step()`, after the decorrelation block (after line 432 `self._decorrelate_buf_alternating(buf, slices, active)`) and before `self._apply_skew_update_buffer(buf, slices)`:

```python
        if self.move_control_mode != "off":
            self._movement_trust_region(buf, slices, active)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_poet_movement_trust_region.py -v`
Expected: PASS (8 tests, incl. both `test_normalize_hits_budget` params).

- [ ] **Step 9: Run the existing update-RMS + decorrelate tests to confirm no regression**

Run: `python -m pytest tests/unit/test_poet_lie_orth_update_rms_decorrelate.py tests/unit/test_poet_lie_orth.py -v`
Expected: PASS (unchanged — `move_control_mode` defaults to `"off"`).

- [ ] **Step 10: Commit**

```bash
git add src/optim/poet_lie_orth_update_rms.py tests/unit/test_poet_movement_trust_region.py
git commit -m "feat(poet): realized-movement trust region (M1) on update-RMS optimizer"
```

---

### Task 2: Config plumbing (4-edit chain) + emit test

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` (argparse, after line 143)
- Modify: `src/utils/megatron_args.py` (emit, after the decorrelate block at line 683)
- Modify: `src/patches/poet_optimizer_setup.py` (config copy, after line 92)
- Modify: `src/optim/poet.py` (optimizer construction, after line 833)
- Test: `tests/unit/test_megatron_args.py` (add two tests near the other `_poet_cfg` tests)

**Interfaces:**
- Consumes: Task 1's optimizer kwargs `move_control_mode` / `move_budget_rho` / `move_lambda`.
- Produces: CLI flags, config attrs, and the wired constructor call (verbatim names from Global Constraints).

- [ ] **Step 1: Write the failing emit tests**

In `tests/unit/test_megatron_args.py`, after `test_poet_argv_emits_block_count_when_set` (line 263):

```python
def test_poet_argv_emits_move_control_when_set():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"lie_move_control_mode": "clip", "lie_move_budget_rho": 0.02})
    )
    assert "--poet-lie-move-control-mode" in args
    assert args[args.index("--poet-lie-move-control-mode") + 1] == "clip"
    assert "--poet-lie-move-budget-rho" in args
    assert args[args.index("--poet-lie-move-budget-rho") + 1] == "0.02"


def test_poet_argv_omits_move_control_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({}))
    assert "--poet-lie-move-control-mode" not in args
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_megatron_args.py::test_poet_argv_emits_move_control_when_set -v`
Expected: FAIL — `--poet-lie-move-control-mode` not in emitted args.

- [ ] **Step 3: Add the emit in `megatron_args.py`**

In `src/utils/megatron_args.py`, after the decorrelate emit block (after line 683, the `--poet-lie-ortho-decorrelate-renorm` append):

```python
        # Realized-movement trust region (M1): emit only when enabled (default off).
        move_mode = poet.get("lie_move_control_mode", "off")
        if move_mode != "off":
            poet_args.extend(
                [
                    "--poet-lie-move-control-mode",
                    move_mode,
                    "--poet-lie-move-budget-rho",
                    str(float(poet.get("lie_move_budget_rho", 0.0))),
                    "--poet-lie-move-lambda",
                    str(float(poet.get("lie_move_lambda", 1.0))),
                ]
            )
```

- [ ] **Step 4: Run to verify the emit tests pass**

Run: `python -m pytest tests/unit/test_megatron_args.py -k move_control -v`
Expected: PASS (both new tests).

- [ ] **Step 5: Add the argparse flags in the launcher**

In `launchers/pretrain_gpt_slm.py`, after line 143 (`--poet-lie-ortho-decorrelate-cos-threshold`):

```python
    # Realized-movement trust region (M1): fix the realized weight move ||D||/||W|| (cap rho)
    # with an adaptive angle instead of a fixed angle. 'measure' logs r without intervening.
    group.add_argument(
        "--poet-lie-move-control-mode",
        choices=["off", "measure", "clip", "normalize"],
        default="off",
    )
    group.add_argument("--poet-lie-move-budget-rho", type=float, default=0.0)
    group.add_argument("--poet-lie-move-lambda", type=float, default=1.0)
```

- [ ] **Step 6: Add the config copy in `poet_optimizer_setup.py`**

In `src/patches/poet_optimizer_setup.py`, after line 92 (`config.poet_lie_ortho_decorrelate_cos_threshold = ...`):

```python
        config.poet_lie_move_control_mode = getattr(args, "poet_lie_move_control_mode", "off")
        config.poet_lie_move_budget_rho = getattr(args, "poet_lie_move_budget_rho", 0.0)
        config.poet_lie_move_lambda = getattr(args, "poet_lie_move_lambda", 1.0)
```

- [ ] **Step 7: Wire the constructor in `poet.py`**

In `src/optim/poet.py`, inside the `LieOrthUpdateRMSMomentum(...)` call, after the `layer_pairs=...` line (line 834):

```python
            move_control_mode=getattr(config, "poet_lie_move_control_mode", "off"),
            move_budget_rho=getattr(config, "poet_lie_move_budget_rho", 0.0),
            move_lambda=getattr(config, "poet_lie_move_lambda", 1.0),
```

- [ ] **Step 8: Verify the full chain compiles and the arg round-trips**

Run: `python -m py_compile launchers/pretrain_gpt_slm.py src/utils/megatron_args.py src/patches/poet_optimizer_setup.py src/optim/poet.py`
Expected: no output (exit 0).

Run: `python -m pytest tests/unit/test_megatron_args.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 9: Commit**

```bash
git add launchers/pretrain_gpt_slm.py src/utils/megatron_args.py src/patches/poet_optimizer_setup.py src/optim/poet.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): plumb M1 move-control flags (argparse -> config -> optimizer)"
```

---

### Task 3: Calibration + experiment scripts

**Files:**
- Create: `scripts/sweep_move_trust_region.sh`

**Interfaces:**
- Consumes: the YAML keys `optim.poet.lie_move_control_mode` / `lie_move_budget_rho` / `lie_move_lambda` from Task 2; the champion recipe used by the existing `sweep_update_rms_decorrelate_gp25.sh` (mup α4, side_γ+0.25, ρ0.30, lr5, decorrelate λ0.25 renorm=off).
- Produces: a Phase-0 measure run + a Phase-1 grid, one arm per invocation, logs via `codexlog`.

- [ ] **Step 1: Find the champion script to clone the recipe from**

Run: `ls scripts/sweep_update_rms_decorrelate_gp25.sh scripts/sweep_poet_init_*.sh 2>/dev/null`
Expected: at least `scripts/sweep_update_rms_decorrelate_gp25.sh` exists — copy its base flags (init, side_gamma, update_rms, lr, decorrelate settings, GPU/launcher boilerplate). Read it before writing Step 2.

- [ ] **Step 2: Write the sweep script**

Create `scripts/sweep_move_trust_region.sh`, cloning the champion arm from `sweep_update_rms_decorrelate_gp25.sh` and adding the M1 axis. Structure (fill the champion base flags verbatim from that script — do NOT invent them):

```bash
#!/usr/bin/env bash
# M1 realized-movement trust region — Phase 0 (measure) + Phase 1 (2x2 vs decorr).
# Base recipe = the §2.15c decorrelation champion (mup a4, side_gamma +0.25, rho0.30,
# lr5, decorrelate lambda0.25 renorm=off). rho_move grid is set from the Phase-0
# p50-p90 of poet_move/ratio_* (read off wandb after the measure arm finishes).
# Usage: bash scripts/sweep_move_trust_region.sh <arm>
#   arm in: measure | clip_off_rA | clip_off_rB | clip_decorr_rA | clip_decorr_rB
set -euo pipefail

ARM="${1:?usage: sweep_move_trust_region.sh <arm>}"

# --- champion base (COPY verbatim from sweep_update_rms_decorrelate_gp25.sh) ---
# BASE_OVERRIDES="optim.poet.q_optimizer=lie_ortho_update_rms optim.poet.init_type=mup_normalized ..."
# (left as a marker: paste the exact BASE line from the reference script here)

# rho_move grid placeholders RA/RB — set AFTER Phase 0 from p50/p90.
RA="__SET_FROM_PHASE0_P50__"
RB="__SET_FROM_PHASE0_P90__"

case "$ARM" in
  measure)        MOVE="optim.poet.lie_move_control_mode=measure" ; DECORR_LAMBDA=0.25 ;;
  clip_off_rA)    MOVE="optim.poet.lie_move_control_mode=clip optim.poet.lie_move_budget_rho=${RA}" ; DECORR_LAMBDA=0.0 ;;
  clip_off_rB)    MOVE="optim.poet.lie_move_control_mode=clip optim.poet.lie_move_budget_rho=${RB}" ; DECORR_LAMBDA=0.0 ;;
  clip_decorr_rA) MOVE="optim.poet.lie_move_control_mode=clip optim.poet.lie_move_budget_rho=${RA}" ; DECORR_LAMBDA=0.25 ;;
  clip_decorr_rB) MOVE="optim.poet.lie_move_control_mode=clip optim.poet.lie_move_budget_rho=${RB}" ; DECORR_LAMBDA=0.25 ;;
  *) echo "unknown arm: $ARM" >&2 ; exit 1 ;;
esac

# DECORR_LAMBDA=0.0 means run with decorrelation OFF (omit the decorrelate flags);
# DECORR_LAMBDA=0.25 means the §2.15c champion decorrelation ON. Translate that into the
# base overrides exactly as the reference script does (optim.poet.lie_ortho_decorrelate=...).

echo "[sweep_move_trust_region] arm=$ARM move='$MOVE'"
# codexlog mtr_${ARM} <launcher> ${BASE_OVERRIDES} ${MOVE} <decorr overrides> seed=42
```

Leave the `BASE_OVERRIDES` / launcher line as an explicit paste-marker so the implementer copies the exact, current champion command from the reference script rather than a guessed one.

- [ ] **Step 3: Shellcheck / syntax-check the script**

Run: `bash -n scripts/sweep_move_trust_region.sh && echo OK`
Expected: `OK` (no syntax errors).

- [ ] **Step 4: Commit**

```bash
git add scripts/sweep_move_trust_region.sh
git commit -m "feat(poet): M1 trust-region sweep (Phase-0 measure + 2x2 vs decorr)"
```

---

## Execution / hand-off notes (not code)

After the code lands (Tasks 1–2 green on CPU), the remaining steps are the operator's (GPU/cluster — per the user's compute policy, hand off the exact commands, do not run):

1. **Phase 0 — calibrate.** Run the `measure` arm on the champion; read `poet_move/ratio_p50` / `ratio_p90` off wandb; set `RA`/`RB` in the script to those values.
2. **Phase 1 — 2×2.** Run `clip_off_rA/rB` and `clip_decorr_rA/rB`; compare against the no-M1 anchors (3.4745 no-decorr, 3.4686 record).
3. **Seed-confirm.** Best arm at seeds 43 & 44 vs its matched no-M1 base (the −0.003-scale effect is under the ~0.01 single-seed floor).
4. **Optional over-whitening probe.** One `normalize` arm at the winning `rho` to bound the downside.

## Self-Review

- **Spec coverage:** §1 motivation → Task 1 (mechanism). §2 mechanism (modes, placement-after-decorr, side_γ/gain/distributed) → Task 1 Steps 6–7. §3 calibration (`measure` + `r` stats) → Task 1 Steps 6 (`measure` branch + `_set_move_stats`) and the measure test (Step 1). §4 experiments (2×2, seeds, anchors) → Task 3 + hand-off notes. §5 plumbing (3 flags, 4 edits, silent-no-op gotcha) → Task 2. §6 testing (clip math, D correctness, off-identity) → Task 1 tests. §7 risks (over-whitening normalize arm) → hand-off note 4. All covered.
- **Placeholder scan:** the only intentional markers are the `BASE_OVERRIDES` / `RA`/`RB` paste-points in the shell script — deliberately deferred to the live champion command and the Phase-0 measurement, with explicit instructions, not silent TODOs.
- **Type consistency:** `move_control_mode`/`move_budget_rho`/`move_lambda` identical across constructor (Task 1), config getattr and emit (Task 2). Mode strings `off/measure/clip/normalize` identical in validation, method, argparse choices, and emit gate. Stats keys identical between `_set_move_stats` and the measure test. `_movement_trust_region(buf, slices, active)` signature matches its `step()` call site.
