# AlternatingPOETXLinear (true single-side POETX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated POETX linear layer that trains only one rotation side per step (alternating in/out) and short-circuits the frozen side's backward, optimizer, and merge work — for a d³-machinery speedup plus a single-side-vs-both-sides quality ablation.

**Architecture:** A new `AlternatingPOETXLinear` (subclass of `POETXLinear`) reads a per-step **active side** from a shared module-level signal seeded once per training step from Megatron's iteration. Its forward is the unchanged bare GEMM; a new autograd Function computes only the active side's rotation-gradient and returns a shape-correct **zeros** gradient for the frozen side (DDP-bucket-safe). The `LieOrthMomentum` optimizer gains a `true_single_side` mode that skips the frozen side's momentum update and write. The merge folds only the active side. Correctness lands first (Tasks 1–6, 8); the merge speed optimization is Task 7.

**Tech Stack:** Python, PyTorch, Megatron-LM (vendored), Triton (Cayley kernel, not touched here), pytest. POET layer code lives in `third_party/poet_torch/`; optimizer + patches in `src/`.

---

## Conventions

- **Test runner (torch / poet_torch tests):**
  `cd /lustre/fast/fast/zqiu/slm-research && PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest <path> -v`
- **Test runner (launcher / megatron_args tests, no torch):** same but use the launcher CPU venv `/var/tmp/zqiu/slmcpu312/bin/python` if the torch venv lacks `omegaconf`/hydra. Try the torch venv first.
- **Active-side convention (must match everywhere):**
  `active = "out" if (iteration // alternate_every) % 2 == 0 else "in"`.
- **All commits** use a single short conventional-commit line, no AI attribution.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `third_party/poet_torch/alt_state.py` | Create | Module-level current-iteration signal + `active_side()`; the single source of truth read by layer, optimizer, merge. |
| `third_party/poet_torch/poetx_ops.py` | Modify | Add `AlternatingPOETXSingleStepFunction` (single-side backward, zeros for frozen). |
| `third_party/poet_torch/poetx_layer.py` | Modify | Add `AlternatingPOETXLinear` (reads active side, one-sided merge in Task 7). |
| `third_party/poet_torch/__init__.py` | Modify | Export the new Function, layer, and `alt_state`. |
| `src/optim/poet_lie_orth.py` | Modify | `true_single_side` mode: active from `alt_state`, frozen-side momentum skip, generalized write-gate. |
| `src/optim/poet.py` | Modify | Pass `true_single_side` into `LieOrthMomentum`. |
| `src/optim/poet_layers.py` | Modify | Walk builds `AlternatingPOETXLinear` when the flag is set; thread `alternate_every`. |
| `src/patches/poet_optimizer_setup.py` | Modify | Thread `poet_single_step_x_alternating` onto the optimizer config. |
| `src/patches/poet_apply_to_model.py` | Modify | Read the flag + `alternate_every` from args; pass to the walk. |
| `src/patches/poet_merge_step.py` | Modify | Seed `alt_state` per step (Task 6); active-only fold (Task 7). |
| `src/utils/megatron_args.py` | Modify | Validate + emit `--poet-single-step-x-alternating`. |
| `launchers/pretrain_gpt_slm.py` | Modify | Add `--poet-single-step-x-alternating` CLI flag. |
| `configs/experiments/optim/poet_lie_orth_alt_x.yaml` | Create | The experiment config (champion knobs + new flags). |
| `docs/experiments/poet_lie_orth_alt_x.md` | Create | Experiment doc (pre-commit hook requires it). |
| `scripts/train_poet_lie_orth_alt_x.sh` | Create | Launch script. |
| `tests/unit/test_alt_state.py` | Create | `alt_state` unit tests. |
| `tests/unit/test_alternating_poetx.py` | Create | Layer + backward unit tests. |
| `tests/unit/test_poet_lie_orth.py` | Modify | Add `true_single_side` optimizer tests. |
| `tests/unit/test_poet_layers.py` | Modify | Walk-selection test for the new layer. |

**Scope note (refines spec §9):** require `q_optimizer=lie_ortho` only (drop `lie_algebra` — champion is lie_ortho, keeps `LieAlgebraMomentum` untouched). **Refines spec §6:** the frozen-side backward returns a shape-correct **zeros** tensor, not `None`, so Megatron's grad buffer never stalls; the expensive d³ `M` GEMM is still skipped. (A future GPU-validated optimization may switch to `None` to also skip the frozen oft_R grad all-reduce.)

---

## Task 1: Shared active-side state module

**Files:**
- Create: `third_party/poet_torch/alt_state.py`
- Modify: `third_party/poet_torch/__init__.py`
- Test: `tests/unit/test_alt_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_alt_state.py`:

```python
"""The shared active-side signal: one iteration int, read by layer/optimizer/merge."""
from poet_torch import alt_state


def test_default_iteration_is_zero():
    alt_state.set_iteration(0)
    assert alt_state.get_iteration() == 0


def test_active_side_alternates_every_one():
    for it, expected in [(0, "out"), (1, "in"), (2, "out"), (3, "in")]:
        alt_state.set_iteration(it)
        assert alt_state.active_side(1) == expected


def test_active_side_holds_each_side_for_alternate_every():
    # alternate_every=2 -> out,out,in,in,out,out
    expected = ["out", "out", "in", "in", "out", "out"]
    for it, exp in enumerate(expected):
        alt_state.set_iteration(it)
        assert alt_state.active_side(2) == exp


def test_alternate_every_below_one_is_treated_as_one():
    alt_state.set_iteration(1)
    assert alt_state.active_side(0) == "in"
    assert alt_state.active_side(-5) == "in"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alt_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poet_torch.alt_state'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/poet_torch/alt_state.py`:

```python
"""Shared active-side signal for true single-side alternating POETX.

ONE source of truth — the current training iteration — read by the layer's
forward (which side to differentiate), the optimizer's step (which side's
momentum to advance + write), and the merge (which side to fold). Seeded once
per training step from Megatron's iteration (by the poet_merge_step train_step
wrapper) so all three agree within a step and resume keeps correct parity.

active_side convention (matches the optimizer's documented schedule):
    "out" on even cycles, "in" on odd, cycle length = alternate_every.
"""
from __future__ import annotations

_ITERATION = 0


def set_iteration(it: int) -> None:
    global _ITERATION
    _ITERATION = int(it)


def get_iteration() -> int:
    return _ITERATION


def active_side(alternate_every: int = 1) -> str:
    every = alternate_every if alternate_every and alternate_every > 0 else 1
    return "out" if (_ITERATION // every) % 2 == 0 else "in"
```

- [ ] **Step 4: Add the export**

In `third_party/poet_torch/__init__.py`, after line 26 (`from .poetx_layer import POETXLinear as POETXLinear`), add:

```python
from . import alt_state as alt_state
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alt_state.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/alt_state.py third_party/poet_torch/__init__.py tests/unit/test_alt_state.py
git commit -m "feat(poet): shared active-side iteration signal for alternating POETX"
```

---

## Task 2: Single-side backward Function

**Files:**
- Modify: `third_party/poet_torch/poetx_ops.py`
- Modify: `third_party/poet_torch/__init__.py`
- Test: `tests/unit/test_alternating_poetx.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_alternating_poetx.py`:

```python
"""AlternatingPOETXLinear: single-side backward (active side matches both-sides
closed form; frozen side returns shape-correct zeros, never None)."""
import torch
from poet_torch import POETLinear, POETXSingleStepFunction
from poet_torch.poetx_ops import AlternatingPOETXSingleStepFunction


def _forward_frame(pl):
    Wx = pl.weight.index_select(0, pl.perm_out).index_select(1, pl.perm_in)
    bias_eff = None if pl.bias is None else pl.bias.index_select(0, pl.perm_out)
    return Wx, bias_eff


def _both(pl, Wx, bias_eff, x):
    return POETXSingleStepFunction.apply(
        x, pl.oft_R_in, pl.oft_R_out, Wx, bias_eff,
        pl.perm_in_inv, pl.perm_out_inv,
        pl.rows_in, pl.cols_in, pl.rows_out, pl.cols_out,
        pl.block_size_in, pl.block_size_out,
    )


def _alt(pl, Wx, bias_eff, x, active):
    return AlternatingPOETXSingleStepFunction.apply(
        x, pl.oft_R_in, pl.oft_R_out, Wx, bias_eff,
        pl.perm_in_inv, pl.perm_out_inv,
        pl.rows_in, pl.cols_in, pl.rows_out, pl.cols_out,
        pl.block_size_in, pl.block_size_out, active,
    )


def _grads(fn):
    pl = POETLinear(in_features=12, out_features=8, block_count=1, bias=True)
    with torch.no_grad():
        pl.weight.normal_()
        pl.bias.normal_()
    Wx, bias_eff = _forward_frame(pl)
    x = torch.randn(5, 12, requires_grad=True)
    gy = torch.randn(5, 8)
    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    (fn(pl, Wx, bias_eff, x) * gy).sum().backward()
    return pl, x


def test_active_in_matches_both_sides_and_frozen_is_zeros():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    # reference: both-sides grads
    pl_b, _ = _grads(lambda pl, Wx, b, x: _both(pl, Wx, b, x))
    gi_ref, go_ref = pl_b.oft_R_in.grad.clone(), pl_b.oft_R_out.grad.clone()
    # alternating with same seed/weights -> rebuild identical layer
    torch.manual_seed(0)
    pl_a, _ = _grads(lambda pl, Wx, b, x: _alt(pl, Wx, b, x, "in"))
    assert torch.allclose(pl_a.oft_R_in.grad, gi_ref, atol=1e-9)
    # frozen side: shape-correct ZEROS, not None
    assert pl_a.oft_R_out.grad is not None
    assert pl_a.oft_R_out.grad.shape == pl_a.oft_R_out.shape
    assert torch.count_nonzero(pl_a.oft_R_out.grad) == 0


def test_active_out_matches_both_sides_and_frozen_is_zeros():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl_b, _ = _grads(lambda pl, Wx, b, x: _both(pl, Wx, b, x))
    go_ref = pl_b.oft_R_out.grad.clone()
    torch.manual_seed(0)
    pl_a, _ = _grads(lambda pl, Wx, b, x: _alt(pl, Wx, b, x, "out"))
    assert torch.allclose(pl_a.oft_R_out.grad, go_ref, atol=1e-9)
    assert pl_a.oft_R_in.grad is not None
    assert torch.count_nonzero(pl_a.oft_R_in.grad) == 0


def test_grad_x_is_independent_of_active_side():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=1, bias=False)
    with torch.no_grad():
        pl.weight.normal_()
    Wx, bias_eff = _forward_frame(pl)
    gy = torch.randn(5, 8)
    xi = torch.randn(5, 12, requires_grad=True)
    (_alt(pl, Wx, bias_eff, xi, "in") * gy).sum().backward()
    xo = xi.detach().clone().requires_grad_(True)
    (_alt(pl, Wx, bias_eff, xo, "out") * gy).sum().backward()
    assert torch.allclose(xi.grad, xo.grad, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alternating_poetx.py -v`
Expected: FAIL — `ImportError: cannot import name 'AlternatingPOETXSingleStepFunction'`.

- [ ] **Step 3: Write minimal implementation**

In `third_party/poet_torch/poetx_ops.py`, append after the existing `POETXSingleStepFunction` class (after line 67):

```python
class AlternatingPOETXSingleStepFunction(torch.autograd.Function):
    """Single-side POETX backward. Identical bare-GEMM forward as
    POETXSingleStepFunction, but the backward computes ONLY the active side's
    rotation-gradient (skipping the frozen side's d^3 M GEMM) and returns a
    shape-correct ZEROS gradient for the frozen side (so Megatron's grad buffer
    never stalls). `active` is "in" or "out"."""

    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, Wx, bias_eff,
                perm_in_inv, perm_out_inv,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out, active):
        y = x @ Wx.t()
        if bias_eff is not None:
            y = y + bias_eff
        ctx.save_for_backward(x, Wx, bias_eff, perm_in_inv, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        ctx.active = active
        ctx.oft_R_in_shape = tuple(oft_R_in.shape)
        ctx.oft_R_out_shape = tuple(oft_R_out.shape)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, Wx, bias_eff, perm_in_inv, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out = ctx.block_size_in, ctx.block_size_out
        out_f, in_f = Wx.shape
        active = ctx.active

        grad_x = grad_y @ Wx  # PLAIN — always needed (upstream gradient)
        G = x.reshape(-1, in_f).t() @ grad_y.reshape(-1, out_f)  # [in, out]
        if active == "in":
            M_in = _conj(G @ Wx, perm_in_inv)
            grad_oft_R_in = _blockdiag_skew_vec(M_in, bs_in, rows_in, cols_in).to(Wx.dtype)
            grad_oft_R_out = torch.zeros(ctx.oft_R_out_shape, dtype=Wx.dtype, device=Wx.device)
        else:  # "out"
            M_out_nat = Wx @ G
            if bias_eff is not None:
                M_out_nat = M_out_nat + torch.outer(bias_eff, grad_y.reshape(-1, out_f).sum(0))
            M_out = _conj(M_out_nat, perm_out_inv)
            grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out).to(Wx.dtype)
            grad_oft_R_in = torch.zeros(ctx.oft_R_in_shape, dtype=Wx.dtype, device=Wx.device)
        # 14 inputs -> 14 returns: grads for x/oft_R_in/oft_R_out, then 11 None.
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None,
                None, None, None, None, None, None)
```

- [ ] **Step 4: Add the export**

In `third_party/poet_torch/__init__.py`, change line 24 to also export the new Function:

```python
from .poetx_ops import POETXSingleStepFunction as POETXSingleStepFunction
from .poetx_ops import AlternatingPOETXSingleStepFunction as AlternatingPOETXSingleStepFunction
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alternating_poetx.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/poetx_ops.py third_party/poet_torch/__init__.py tests/unit/test_alternating_poetx.py
git commit -m "feat(poet): single-side AlternatingPOETXSingleStepFunction backward"
```

---

## Task 3: `AlternatingPOETXLinear` layer

**Files:**
- Modify: `third_party/poet_torch/poetx_layer.py`
- Modify: `third_party/poet_torch/__init__.py`
- Test: `tests/unit/test_alternating_poetx.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_alternating_poetx.py`:

```python
def test_layer_forward_is_bare_gemm_and_backward_is_single_side():
    from poet_torch import AlternatingPOETXLinear, alt_state

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    layer = AlternatingPOETXLinear(
        in_features=12, out_features=8, block_count=1, bias=True, alternate_every=1
    )
    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
    x = torch.randn(4, 12)
    # forward = bare GEMM on the stored forward-frame weight (R=I at merge_period=1)
    y = layer(x)
    assert (y - (x @ layer.weight.t() + layer.bias)).abs().max().item() == 0.0

    # iteration 1 -> active "in": only oft_R_in gets a nonzero grad, oft_R_out zeros
    alt_state.set_iteration(1)
    layer.oft_R_in.grad = layer.oft_R_out.grad = None
    gy = torch.randn(4, 8)
    (layer(x) * gy).sum().backward()
    assert torch.count_nonzero(layer.oft_R_in.grad) > 0
    assert torch.count_nonzero(layer.oft_R_out.grad) == 0

    # iteration 2 -> active "out": flips
    alt_state.set_iteration(2)
    layer.oft_R_in.grad = layer.oft_R_out.grad = None
    (layer(x) * gy).sum().backward()
    assert torch.count_nonzero(layer.oft_R_out.grad) > 0
    assert torch.count_nonzero(layer.oft_R_in.grad) == 0


def test_layer_is_poetx_subclass():
    from poet_torch import AlternatingPOETXLinear, POETXLinear

    layer = AlternatingPOETXLinear(in_features=8, out_features=16, block_count=1)
    assert isinstance(layer, POETXLinear)  # merge driver isinstance tuple includes it
    assert layer.alternate_every == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alternating_poetx.py -k layer -v`
Expected: FAIL — `ImportError: cannot import name 'AlternatingPOETXLinear'`.

- [ ] **Step 3: Write minimal implementation**

In `third_party/poet_torch/poetx_layer.py`, add the import at the top (after line 18):

```python
from .poetx_ops import AlternatingPOETXSingleStepFunction
```

Then append the subclass after `POETXLinear` (after line 145):

```python
class AlternatingPOETXLinear(POETXLinear):
    """POETX layer that trains ONE rotation side per step (true single-side).

    The active side comes from the shared `alt_state` iteration (seeded once per
    training step), so layer forward, optimizer, and merge all agree. Forward is
    the unchanged bare GEMM; the backward (AlternatingPOETXSingleStepFunction)
    computes only the active side's rotation-gradient and zeros the frozen side.
    """

    def __init__(self, *args, alternate_every: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alternate_every = max(1, int(alternate_every))

    def forward(self, x):
        from .alt_state import active_side

        active = active_side(self.alternate_every)
        return AlternatingPOETXSingleStepFunction.apply(
            x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
            self.perm_in_inv, self.perm_out_inv,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out, active,
        )
```

- [ ] **Step 4: Add the export**

In `third_party/poet_torch/__init__.py`, change line 25 region to:

```python
from .poetx_layer import POETXLinear as POETXLinear
from .poetx_layer import AlternatingPOETXLinear as AlternatingPOETXLinear
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alternating_poetx.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/poetx_layer.py third_party/poet_torch/__init__.py tests/unit/test_alternating_poetx.py
git commit -m "feat(poet): AlternatingPOETXLinear layer (single-side, active from alt_state)"
```

---

## Task 4: `true_single_side` mode in `LieOrthMomentum`

**Files:**
- Modify: `src/optim/poet_lie_orth.py`
- Test: `tests/unit/test_poet_lie_orth.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_poet_lie_orth.py`:

```python
def test_true_single_side_freezes_inactive_momentum(monkeypatch):
    # true_single_side: inactive side's momentum must NOT advance/decay, even with
    # a (zeros) grad present; active side updates as usual. Active comes from alt_state.
    from poet_torch import alt_state

    torch.manual_seed(7)
    b = 8
    ne = b * (b - 1) // 2
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_out = nn.Parameter(torch.zeros(1, ne))
    p_in.grad = torch.randn(1, ne)
    p_out.grad = torch.zeros(1, ne)  # frozen side gets zeros from the layer backward
    opt = LieOrthMomentum(
        [
            dict(params=[p_in], use_skew=True, side="in", lr=0.1),
            dict(params=[p_out], use_skew=True, side="out", lr=0.1),
        ],
        ortho_c=0.05,
        true_single_side=True,
    )
    alt_state.set_iteration(1)  # active "in"
    opt.step()
    assert p_in.data.abs().sum() > 0  # active side written
    assert torch.allclose(p_out.data, torch.zeros_like(p_out))  # inactive not written
    assert "lie_m" in opt.state[p_in]
    # inactive side's momentum buffer must be absent OR all-zero (never advanced)
    assert "lie_m" not in opt.state[p_out] or opt.state[p_out]["lie_m"].abs().sum() == 0


def test_true_single_side_active_flips_with_iteration():
    from poet_torch import alt_state

    torch.manual_seed(8)
    ne = 8 * 7 // 2
    p_in = nn.Parameter(torch.zeros(1, ne))
    p_out = nn.Parameter(torch.zeros(1, ne))
    p_in.grad = torch.zeros(1, ne)
    p_out.grad = torch.randn(1, ne)
    opt = LieOrthMomentum(
        [
            dict(params=[p_in], use_skew=True, side="in", lr=0.1),
            dict(params=[p_out], use_skew=True, side="out", lr=0.1),
        ],
        ortho_c=0.05,
        true_single_side=True,
    )
    alt_state.set_iteration(2)  # active "out"
    opt.step()
    assert p_out.data.abs().sum() > 0
    assert torch.allclose(p_in.data, torch.zeros_like(p_in))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -k true_single_side -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'true_single_side'`.

- [ ] **Step 3: Write minimal implementation**

In `src/optim/poet_lie_orth.py`:

(a) Add the constructor arg. Change the `__init__` signature (after line 36 `alternate_every: int = 1,`) to add:

```python
        alternate_every: int = 1,
        true_single_side: bool = False,
```

and set it after `self._alt_step = 0` (after line 57):

```python
        self._alt_step = 0
        # true_single_side: the dedicated AlternatingPOETXLinear path. Active side
        # comes from poet_torch.alt_state (shared with the layer + merge), and the
        # frozen side's momentum does NOT advance (its grad is zeros from the layer).
        self.true_single_side = bool(true_single_side)
```

(b) Add an active-side helper (insert before `_lie_m_update`, after line 84):

```python
    def _active_side(self):
        if self.true_single_side:
            from poet_torch.alt_state import active_side

            return active_side(self.alternate_every)
        if self.alternating:
            return "out" if (self._alt_step // self.alternate_every) % 2 == 0 else "in"
        return None
```

(c) In `_lie_m_update`, skip the frozen side under true_single_side. Replace the loop head (lines 89-91):

```python
        for group in self.param_groups:
            if not group["use_skew"]:
                continue
            if self.true_single_side and active is not None and group["side"] != active:
                continue  # true single-side: frozen side's momentum must not advance
```

(d) In `_skew_update_buffer`, generalize the write-gate. Replace line 144:

```python
            if active is not None and group["side"] != active:
                continue  # inactive side -> no rotation written this step
```

(e) In `step()`, compute active via the helper. Replace lines 182-184:

```python
        active = self._active_side()
```

- [ ] **Step 4: Run new + existing optimizer tests**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -v`
Expected: PASS — the two new `true_single_side` tests pass AND the existing `test_batched_step_alternating_writes_only_active_side` and all sharded/replicated tests still pass (the write-gate change is equivalent when `active` is `None`, and identical when `alternating` computes a non-None active).

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_lie_orth.py tests/unit/test_poet_lie_orth.py
git commit -m "feat(poet): true_single_side mode in LieOrthMomentum (frozen-side momentum skip)"
```

---

## Task 5: Config / CLI / walk wiring

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py`
- Modify: `src/utils/megatron_args.py`
- Modify: `src/patches/poet_optimizer_setup.py`
- Modify: `src/patches/poet_apply_to_model.py`
- Modify: `src/optim/poet_layers.py`
- Modify: `src/optim/poet.py`
- Test: `tests/unit/test_poet_layers.py`, `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing walk-selection test**

Append to `tests/unit/test_poet_layers.py`:

```python
def test_single_step_x_alternating_uses_alternating_poetx_class():
    import torch.nn as nn
    from poet_torch import AlternatingPOETXLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    replace_linears_with_poet(
        model,
        block_count=1,
        parameterization="cayley",
        single_step_x=True,
        single_step_x_alternating=True,
        alternate_every=2,
    )
    pl = model[0].poet_linear if isinstance(model[0], POETMegatronLinear) else None
    assert isinstance(pl, AlternatingPOETXLinear)
    assert pl.alternate_every == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -k single_step_x_alternating -v`
Expected: FAIL — `TypeError: replace_linears_with_poet() got an unexpected keyword argument 'single_step_x_alternating'`.

- [ ] **Step 3: Thread the params through the walk**

In `src/optim/poet_layers.py`, find the `replace_linears_with_poet` signature (and the inner `_walk`/closure that reads `single_step_x`). Add two new keyword params with defaults to the public function signature:

```python
    single_step_x_alternating: bool = False,
    alternate_every: int = 1,
```

Then in the layer-construction branch, change the `cache_mode == "none"` block (lines 312-327) to:

```python
                if cache_mode == "none":
                    if single_step_x and single_step_x_alternating:
                        from poet_torch import AlternatingPOETXLinear as _PoetCls

                        pl = _PoetCls(
                            in_features=in_f,
                            out_features=out_f,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            alternate_every=alternate_every,
                            **block_kwargs,
                        )
                    else:
                        if single_step_x:
                            from poet_torch import POETXLinear as _PoetCls
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

(The `bake_perms_into_weight()` call at line 346-349 already runs for any `single_step_x`, including the alternating subclass, since `AlternatingPOETXLinear` inherits it.)

- [ ] **Step 4: Run the walk test**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -k single_step_x_alternating -v`
Expected: PASS.

- [ ] **Step 5: Add the CLI flag**

In `launchers/pretrain_gpt_slm.py`, after line 118 (`group.add_argument("--poet-single-step-x", action="store_true")`), add:

```python
    group.add_argument("--poet-single-step-x-alternating", action="store_true")
```

- [ ] **Step 6: Thread the flag onto the apply patch + walk call**

In `src/patches/poet_apply_to_model.py`, in `_apply_poet_to_chunk` after line 73 (`single_step_x = getattr(args, "poet_single_step_x", False)`), add:

```python
        single_step_x_alternating = getattr(args, "poet_single_step_x_alternating", False)
        alternate_every = getattr(args, "poet_lie_alternate_every", 1)
```

and pass them into the `replace_linears_with_poet(...)` call (after `single_step_x=single_step_x,` at line 91):

```python
            single_step_x=single_step_x,
            single_step_x_alternating=single_step_x_alternating,
            alternate_every=alternate_every,
```

- [ ] **Step 7: Thread the flag onto the optimizer config + builder**

In `src/patches/poet_optimizer_setup.py`, after line 51 (`config.poet_lie_alternate_every = ...`), add:

```python
        config.poet_single_step_x_alternating = getattr(args, "poet_single_step_x_alternating", False)
```

In `src/optim/poet.py`, in the `LieOrthMomentum(...)` construction (after line 622 `distributed=_lie_ortho_distributed,`), add:

```python
            true_single_side=getattr(config, "poet_single_step_x_alternating", False),
```

- [ ] **Step 8: Validate + emit the flag (megatron_args)**

In `src/utils/megatron_args.py`, after the `single_step_x` validation block (after line 294), add:

```python
        if poet.get("single_step_x_alternating", False):
            if not poet.get("single_step_x", False):
                raise ValueError(
                    "optim.poet.single_step_x_alternating requires single_step_x=true "
                    "(the alternating layer is a POETX subclass)."
                )
            if merge_period != 1:
                raise ValueError(
                    "optim.poet.single_step_x_alternating requires merge_period=1."
                )
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError(
                    "optim.poet.single_step_x_alternating requires parameterization=cayley."
                )
            if poet.get("q_optimizer", "adam") != "lie_ortho":
                raise ValueError(
                    "optim.poet.single_step_x_alternating requires q_optimizer=lie_ortho."
                )
            if poet.get("head_aligned_attn", False):
                raise ValueError(
                    "optim.poet.single_step_x_alternating is incompatible with "
                    "head_aligned_attn=true (head-aligned uses a different layer)."
                )
```

and emit it next to the other `single_step_x` emit (after line 386 `poet_args.append("--poet-single-step-x")`), inside the same `if`-block region add a new block:

```python
        # store_true: dedicated true-single-side alternating POETX layer.
        if poet.get("single_step_x_alternating", False):
            poet_args.append("--poet-single-step-x-alternating")
```

- [ ] **Step 9: Write the args round-trip + validation test**

Append to `tests/unit/test_megatron_args.py` (mirror the existing single_step_x test style in that file — find an existing `single_step` test to copy the harness; the test below assumes a `build_megatron_args(cfg)`-style helper exists in that file, named `_args_for` or similar — reuse whatever the file already uses):

```python
def test_single_step_x_alternating_emits_flag_and_validates():
    # Reuse this file's existing config-building helper (same one the single_step_x
    # tests use). Build a minimal lie_ortho POETX config with the alternating flag.
    cfg = _poet_lie_ortho_cfg()  # <- existing helper in this test module
    cfg.optim.poet.single_step_x = True
    cfg.optim.poet.single_step_x_alternating = True
    cfg.optim.poet.head_aligned_attn = False
    args = _emit_poet_args(cfg)  # <- existing helper
    assert "--poet-single-step-x-alternating" in args

    # validation: head-aligned on must raise
    cfg.optim.poet.head_aligned_attn = True
    with pytest.raises(ValueError, match="head_aligned_attn"):
        _emit_poet_args(cfg)
```

> If `tests/unit/test_megatron_args.py` does not expose helpers named `_poet_lie_ortho_cfg`/`_emit_poet_args`, copy the exact construction the file's nearest `single_step_x` / `single_step_native` test uses (search the file for `single_step` and clone that fixture verbatim — do not invent a new harness).

- [ ] **Step 10: Run the wiring tests**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py tests/unit/test_megatron_args.py -k "single_step_x_alternating or single_step" -v`
(If the torch venv lacks omegaconf, run the megatron_args test with `/var/tmp/zqiu/slmcpu312/bin/python`.)
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add launchers/pretrain_gpt_slm.py src/utils/megatron_args.py src/patches/poet_optimizer_setup.py src/patches/poet_apply_to_model.py src/optim/poet_layers.py src/optim/poet.py tests/unit/test_poet_layers.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): wire single_step_x_alternating through CLI, args, walk, optimizer"
```

---

## Task 6: Seed the shared iteration each training step

**Files:**
- Modify: `src/patches/poet_merge_step.py`
- Test: `tests/unit/test_poet_merge_step.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Append to (or create) `tests/unit/test_poet_merge_step.py`:

```python
def test_active_side_seeding_helper_sets_alt_state():
    """The merge patch exposes a pure helper that seeds alt_state from an iteration
    (so the layer/optimizer/merge all read the same active side)."""
    from poet_torch import alt_state

    from src.patches.poet_merge_step import _seed_active_side

    _seed_active_side(3)
    assert alt_state.get_iteration() == 3
    assert alt_state.active_side(1) == "in"
    _seed_active_side(4)
    assert alt_state.active_side(1) == "out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_step.py -k active_side_seeding -v`
Expected: FAIL — `ImportError: cannot import name '_seed_active_side'`.

- [ ] **Step 3: Implement the helper + call it at train_step entry**

In `src/patches/poet_merge_step.py`, add a module-level helper (after `_merge_decision`, after line 71):

```python
def _seed_active_side(iteration: int) -> None:
    """Seed the shared active-side signal so the layer forward, optimizer step, and
    merge all read the same side within this training step."""
    from poet_torch.alt_state import set_iteration

    set_iteration(int(iteration) if iteration is not None else 0)
```

Then in `_wrapped` (inside `apply`), seed it BEFORE the original train_step runs. Replace the body from line 82 down to the `ret = _orig_train_step(*args, **kwargs)` so iteration is extracted first:

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
        ret = _orig_train_step(*args, **kwargs)
        merge_period = getattr(opts, "poet_merge_period", 0)
        reinit_period = getattr(opts, "poet_reinit_period", 0)
        folding, do_reinit = _merge_decision(iteration, merge_period, reinit_period)
        if not folding:
            return ret
        model = args[2] if len(args) >= 3 else kwargs.get("model")
        if model is None:
            logger.warning("[POET] merge step skipped: model not found in train_step args")
            return ret
        _run_merge(model, dist, iteration, reinit_perm=do_reinit)
        if not getattr(opts, "poet_use_poet_adam", False):
            optimizer = args[3] if len(args) >= 4 else kwargs.get("optimizer")
            if optimizer is not None:
                _reset_vanilla_oft_state(optimizer, model, iteration, reset_moments=do_reinit)
        return ret
```

- [ ] **Step 4: Run the test**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_step.py -k active_side_seeding -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_merge_step.py tests/unit/test_poet_merge_step.py
git commit -m "feat(poet): seed shared active-side signal at train_step entry"
```

---

## Task 7: Active-only merge fold (the d³ speed prize)

> **Perf-critical + GPU-validate.** This skips the frozen side's Cayley build and fold matmul. Correctness anchor: folding only the active side MUST equal the both-sides fold when the frozen side is identity (its `oft_R` is exactly 0). Until GPU-validated, the layer already merges correctly via the inherited both-sides fold (frozen = identity no-op) — so the feature works without this task; this task only adds the speedup.

**Files:**
- Modify: `third_party/poet_torch/poetx_layer.py`
- Modify: `src/patches/poet_merge_step.py`
- Test: `tests/unit/test_alternating_poetx.py`

- [ ] **Step 1: Write the failing parity test**

Append to `tests/unit/test_alternating_poetx.py`:

```python
def test_active_only_fold_matches_both_sides_when_frozen_is_identity():
    """Folding only the active side == folding both sides when the frozen side's
    oft_R is 0 (identity). fp64 parity."""
    from poet_torch import AlternatingPOETXLinear
    from poet_torch.poet_layer import cayley_batch, pytorch_skew_symmetric

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(5)

    def _make():
        layer = AlternatingPOETXLinear(in_features=12, out_features=8, block_count=1, bias=False)
        with torch.no_grad():
            layer.weight.normal_()
        return layer

    def _cayley(layer):
        qi = pytorch_skew_symmetric(layer.oft_R_in, layer.block_size_in, layer.rows_in, layer.cols_in)
        qo = pytorch_skew_symmetric(layer.oft_R_out, layer.block_size_out, layer.rows_out, layer.cols_out)
        return cayley_batch(qo), cayley_batch(qi)  # (R_out, R_in)

    # active "in": only oft_R_in nonzero, oft_R_out stays identity
    ref, act = _make(), _make()
    with torch.no_grad():
        for layer in (ref, act):
            layer.oft_R_in.normal_(std=1e-2)  # oft_R_out left at 0
    # reference: full both-sides fold
    R_out, R_in = _cayley(ref)
    ref._fold_with_R(R_out, R_in, reinit_perm=False)
    # active-only fold
    act._fold_active_side("in", reinit_perm=False, cayley_fn=cayley_batch)
    assert torch.allclose(act.weight, ref.weight, atol=1e-9), (act.weight - ref.weight).abs().max()
    assert torch.count_nonzero(act.oft_R_in) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alternating_poetx.py -k active_only_fold -v`
Expected: FAIL — `AttributeError: 'AlternatingPOETXLinear' object has no attribute '_fold_active_side'`.

- [ ] **Step 3: Implement the one-sided fold on the layer**

In `third_party/poet_torch/poetx_layer.py`, add to `AlternatingPOETXLinear` (uses the verified `block_diag_lr_matmul_decoupled`; the frozen side passes an identity built cheaply as `R = I` blocks — but we SKIP its Cayley, building only the active side's R):

```python
    @torch.no_grad()
    def _fold_active_side(self, active, reinit_perm: bool = False, cayley_fn=None) -> None:
        """Fold ONLY the active side into W (skip the frozen side's Cayley build).

        The frozen side's oft_R is exactly 0 => R = I, so its fold is a no-op; we
        build identity blocks for it (no Cayley) and reuse the verified round-trip
        fold. Bit-identical to the both-sides fold whenever the frozen side is
        identity, but pays one Cayley + one block-fold instead of two.
        """
        import torch as _torch
        from .poet_layer import pytorch_skew_symmetric

        if cayley_fn is None:

            def cayley_fn(Q):
                return _torch.ops.poet.cayley(Q)[0]

        if active == "in":
            R_in = cayley_fn(
                pytorch_skew_symmetric(self.oft_R_in, self.block_size_in, self.rows_in, self.cols_in)
            )
            R_out = _torch.eye(self.block_size_out, dtype=self.weight.dtype, device=self.weight.device)
            R_out = R_out.unsqueeze(0).expand(self.r_out, -1, -1)
        else:  # "out"
            R_out = cayley_fn(
                pytorch_skew_symmetric(self.oft_R_out, self.block_size_out, self.rows_out, self.cols_out)
            )
            R_in = _torch.eye(self.block_size_in, dtype=self.weight.dtype, device=self.weight.device)
            R_in = R_in.unsqueeze(0).expand(self.r_in, -1, -1)
        self._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)
```

> Note: this first version still does the identity block-matmul for the frozen side (it skips the frozen **Cayley**, which is the largest d³ chunk). A follow-up may add a genuine one-sided matmul that also skips the identity fold-matmul; gate that behind a separate parity test.

- [ ] **Step 4: Run the parity test**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alternating_poetx.py -k active_only_fold -v`
Expected: PASS.

- [ ] **Step 5: Wire active-only fold into the merge driver**

In `src/patches/poet_merge_step.py`, in `_merge_layers` (lines 376-391), route `AlternatingPOETXLinear` layers to the active-only fold. Replace the function body with:

```python
def _merge_layers(pls, reinit_perm: bool, disable_batch: bool) -> None:
    """Fold every layer. AlternatingPOETXLinear layers fold ONLY the active side
    (frozen side is identity); the rest use the batched both-sides fold."""
    from poet_torch import AlternatingPOETXLinear
    from poet_torch.alt_state import active_side
    from megatron.training import get_args

    alt_pls = [pl for pl in pls if isinstance(pl, AlternatingPOETXLinear)]
    rest = [pl for pl in pls if not isinstance(pl, AlternatingPOETXLinear)]

    if alt_pls:
        every = getattr(get_args(), "poet_lie_alternate_every", 1)
        side = active_side(every)
        for pl in alt_pls:
            pl._fold_active_side(side, reinit_perm=reinit_perm)

    if disable_batch:
        for pl in rest:
            pl.merge_then_reinitialize(reinit_perm=reinit_perm)
        return
    cayley_pls = [pl for pl in rest if getattr(pl, "parameterization", "cayley") == "cayley"]
    other_pls = [pl for pl in rest if getattr(pl, "parameterization", "cayley") != "cayley"]
    for pl in other_pls:
        pl.merge_then_reinitialize(reinit_perm=reinit_perm)
    if cayley_pls:
        built = _build_R_batched(cayley_pls)
        for pl in cayley_pls:
            R_out, R_in = built[id(pl)]
            pl._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)
```

- [ ] **Step 6: Write the driver-level parity test**

Append to `tests/unit/test_alternating_poetx.py`:

```python
def test_merge_layers_active_only_matches_both_sides(monkeypatch):
    """Through the real _merge_layers driver, an AlternatingPOETXLinear with only
    the active side stepped folds identically to a both-sides POETX fold."""
    from poet_torch import AlternatingPOETXLinear, POETXLinear, alt_state
    from poet_torch.poet_layer import cayley_batch

    import src.patches.poet_merge_step as ms

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(11)

    # Stub get_args() so _merge_layers can read poet_lie_alternate_every.
    class _A:
        poet_lie_alternate_every = 1

    monkeypatch.setattr("megatron.training.get_args", lambda: _A(), raising=False)
    alt_state.set_iteration(1)  # active "in"

    ref = POETXLinear(in_features=12, out_features=8, block_count=1, bias=False)
    act = AlternatingPOETXLinear(in_features=12, out_features=8, block_count=1, bias=False)
    with torch.no_grad():
        ref.weight.normal_()
        act.weight.copy_(ref.weight)
        for b in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(act, b).copy_(getattr(ref, b))
        ref.oft_R_in.normal_(std=1e-2)        # only IN side stepped
        act.oft_R_in.copy_(ref.oft_R_in)

    # both-sides reference fold (frozen out = identity)
    from src.patches.poet_merge_step import _build_R_batched
    R_out, R_in = _build_R_batched([ref], cayley_fn=cayley_batch)[id(ref)]
    ref._fold_with_R(R_out, R_in, reinit_perm=False)
    # active-only via the driver
    ms._merge_layers([act], reinit_perm=False, disable_batch=False)

    assert torch.allclose(act.weight, ref.weight, atol=1e-9), (act.weight - ref.weight).abs().max()
```

- [ ] **Step 7: Run the merge tests (new + existing POETX merge)**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_alternating_poetx.py tests/unit/test_poetx_layer.py -v`
Expected: PASS — new active-only parity passes AND existing `test_batched_merge_folds_poetx` / `test_run_merge_gate_collects_poetx` still pass.

- [ ] **Step 8: Commit**

```bash
git add third_party/poet_torch/poetx_layer.py src/patches/poet_merge_step.py tests/unit/test_alternating_poetx.py
git commit -m "perf(poet): active-only merge fold for AlternatingPOETXLinear (skip frozen Cayley)"
```

---

## Task 8: Experiment config, doc, and launch script

**Files:**
- Create: `configs/experiments/optim/poet_lie_orth_alt_x.yaml`
- Create: `docs/experiments/poet_lie_orth_alt_x.md`
- Create: `scripts/train_poet_lie_orth_alt_x.sh`
- Test: config-load smoke test (reuse existing experiment-load test harness)

- [ ] **Step 1: Create the experiment config**

Create `configs/experiments/optim/poet_lie_orth_alt_x.yaml` (champion knobs: lr 3e-3, c=8, distributed, head-OFF; plus the POETX-X path and the alternating flag):

```yaml
# @package _global_
# poet_lie_orth_alt_x: AlternatingPOETXLinear (true single-side POETX) on top of the
# champion lie_ortho recipe (head-OFF, lr 3e-3, c=8, distributed). Trains ONE rotation
# side per step (out even / in odd), short-circuiting the frozen side's backward +
# Cayley + fold. See docs/superpowers/specs/2026-06-08-alternating-poetx-single-side-design.md
experiment:
  name: poet_lie_orth_alt_x
  family: optim
  description: |
    True single-side alternating POETX: AlternatingPOETXLinear trains one rotation
    side per step and skips the frozen side's d^3 machinery (backward M, Cayley,
    fold). Each side's first-moment momentum advances only on its active steps.
    Built on the champion lie_ortho recipe with head-alignment OFF. Ablates
    single-side-per-step quality vs the both-sides champion (dwynpk9y).
  references:
    - "POET"
    - "Muon"
    - "Pion"
  patches:
    - model_unfuse_linears
    - poet_optimizer_setup
    - poet_unfuse_te_impl
    - poet_apply_to_model
    - poet_merge_step
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
    single_step_x: true                 # forward-frame POETX path
    single_step_x_alternating: true     # the dedicated true-single-side layer
    lie_alternate_every: 1
    train_output_rotation: true

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 2: Create the experiment doc**

Create `docs/experiments/poet_lie_orth_alt_x.md`:

```markdown
# poet_lie_orth_alt_x

True single-side alternating POETX (`AlternatingPOETXLinear`) on the champion
`lie_ortho` recipe (head-OFF, lr 3e-3, c=8, distributed). Trains one rotation side
per step (out on even iterations, in on odd), short-circuiting the frozen side's
backward `M`, Cayley build, and weight-fold. Each side's first-moment momentum
advances only on its active steps (true single-side — a different optimizer than the
both-side-momentum `poet_lie_alt`).

- **Design:** `docs/superpowers/specs/2026-06-08-alternating-poetx-single-side-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-08-alternating-poetx-single-side.md`
- **Baseline:** the both-sides champion `dwynpk9y` (val/loss 3.5528).
- **Expectation:** d³-machinery speedup (small at 60m, grows with d); quality is an
  open ablation (single-side per step may help or hurt).
```

- [ ] **Step 3: Create the launch script**

Create `scripts/train_poet_lie_orth_alt_x.sh` (clone of `train_poet_lie_orth.sh` with the experiment swapped):

```bash
#!/usr/bin/env bash
set -euo pipefail

# poet_lie_orth_alt_x: AlternatingPOETXLinear (true single-side POETX) on the champion
# lie_ortho recipe. Same harness as train_poet_lie_orth.sh, experiment swapped.

case " $* " in
  *" --backend torchtitan "*|*" --backend=torchtitan "*)
    echo "This optimizer is not yet supported on torchtitan (milestone 1 is AdamW only)." >&2
    exit 2 ;;
esac

SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" ]]; then
  shift || true
else
  ARCH="llama3"
fi

case "${ARCH}" in
  llama3) FAMILY="llama3"; DEFAULT_SCALE="60m" ;;
  deepseek_v3) FAMILY="deepseek_v3"; DEFAULT_SCALE="deepseek_v3_proxy_small" ;;
  *) echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2; exit 2 ;;
esac

USER_SET_SCALE="no"; USER_SET_SEQ="no"; USER_SET_SCHED="no"; USER_SET_REGIME="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
    base.model.seq_length=*) USER_SET_SEQ="yes" ;;
    scheduler=*) USER_SET_SCHED="yes" ;;
    training_regime=*) USER_SET_REGIME="yes" ;;
  esac
done

SCALE_ARGS=(); [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]] && SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
REGIME_ARGS=(); [[ "${USER_SET_REGIME}" == "no" ]] && REGIME_ARGS=("training_regime=ablation_40x")
SEQ_ARGS=(); [[ "${USER_SET_SEQ}" == "no" ]] && SEQ_ARGS=("base.model.seq_length=256")
SCHED_ARGS=(); [[ "${USER_SET_SCHED}" == "no" ]] && SCHED_ARGS=("scheduler=cosine_poet")

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${REGIME_ARGS[@]}" \
  "${SEQ_ARGS[@]}" \
  "${SCHED_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=optim/poet_lie_orth_alt_x" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=128" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "base.model.tie_embeddings=false" \
  "optim.weight_decay=0.1" \
  "wandb.project=slm-zeju-dev" \
  "$@"
```

Make it executable:

```bash
chmod +x scripts/train_poet_lie_orth_alt_x.sh
```

- [ ] **Step 4: Smoke-test config load (dry run, no training)**

Run the launcher in print/validate mode the same way the repo validates an experiment compiles its args (mirror how other experiments are dry-run; e.g. `python -m launchers.train_megatron experiment=optim/poet_lie_orth_alt_x base/scale=60m --cfg job` or the repo's existing `--print-args`/`dry-run` path). Confirm it emits `--poet-single-step-x-alternating` and raises no validation error.

Run (CPU venv that has hydra/omegaconf):
```bash
cd /lustre/fast/fast/zqiu/slm-research && /var/tmp/zqiu/slmcpu312/bin/python -m launchers.train_megatron experiment=optim/poet_lie_orth_alt_x base/scale=60m cluster=h100_de --cfg job 2>&1 | head -40
```
Expected: prints the resolved config with `single_step_x_alternating: true`, no `ValueError`.

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/optim/poet_lie_orth_alt_x.yaml docs/experiments/poet_lie_orth_alt_x.md scripts/train_poet_lie_orth_alt_x.sh
git commit -m "feat(poet): poet_lie_orth_alt_x experiment (config, doc, launch script)"
```

---

## Task 9: Verification handoff (CPU run + GPU commands)

> No new code. Run the full CPU suite, then hand the GPU commands to the user (per repo policy, GPU runs are the user's).

- [ ] **Step 1: Run the full new + adjacent CPU suite**

```bash
cd /lustre/fast/fast/zqiu/slm-research && PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_alt_state.py \
  tests/unit/test_alternating_poetx.py \
  tests/unit/test_poetx_layer.py \
  tests/unit/test_poet_lie_orth.py \
  tests/unit/test_poet_layers.py \
  tests/unit/test_poet_merge_step.py -v
```
Expected: all PASS (no regressions in the existing POETX / lie_orth / merge tests).

- [ ] **Step 2: Run the args test on the launcher venv**

```bash
cd /lustre/fast/fast/zqiu/slm-research && /var/tmp/zqiu/slmcpu312/bin/python -m pytest tests/unit/test_megatron_args.py -k single_step -v
```
Expected: PASS (2 pre-existing failures unrelated to this change may remain per the repo's known-failures note).

- [ ] **Step 3: Hand the GPU runs to the user**

Provide these (do NOT launch):

```bash
# Quality + step-time vs champion (60m / 40tpp, head-off). Single-side alternating:
codexlog poet_lie_orth_alt_x bash scripts/train_poet_lie_orth_alt_x.sh llama3

# Both-sides champion baseline for the head-to-head (already the leaderboard champ):
codexlog poet_lie_orth_nohead bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.head_aligned_attn=false
```

Compare `val/loss` (quality) and `perf/step_time_s` (speed) from each run's
`runs/<dir>/**/wandb-summary.json`. Update [POET_dev.md](/lustre/fast/fast/zqiu/slm-research/POET_dev.md) §2.5 with the alt-x arm.

- [ ] **Step 4: (Optional) Kimi-scale step-time microcheck**

Once 60m parity holds, run the alt-x path at a larger `base/scale` to confirm the d³
saving (Cayley + fold + backward M skip) produces a measurable `perf/step_time_s`
drop vs both-sides at large d. (User-run; provide a `base/scale=<large>` override.)

---

## Self-Review

**Spec coverage:** spec §1 goal → Tasks 1–8; §3 true single-side momentum → Task 4; §5 layer → Task 3; §6 single-side backward (zeros not None, refined) → Task 2; §7 optimizer → Task 4; §8 active-only merge → Task 7; §9 config/CLI/validation → Task 5 + Task 8; §10 verification → Task 9. Active-side single-source-of-truth (spec §4) → Task 1 + Task 6. No uncovered requirement.

**Refinements logged (update the spec to match):** (1) frozen-side backward returns **zeros**, not `None` (DDP-bucket safety; §6); (2) require `q_optimizer=lie_ortho` only (drop `lie_algebra`; §9); (3) champion uses `single_step_native`, not plain `POETLinear` (spec §2 prose).

**Type/name consistency:** `AlternatingPOETXSingleStepFunction` (Task 2) used by `AlternatingPOETXLinear.forward` (Task 3); `active_side(alternate_every)`/`set_iteration(it)` (Task 1) used by layer (Task 3), optimizer `_active_side` (Task 4), and `_seed_active_side` (Task 6); `single_step_x_alternating` + `alternate_every` walk params (Task 5) match the apply-patch call (Task 5) and the YAML keys (Task 8); `true_single_side` ctor arg (Task 4) matches the builder call (Task 5). Consistent.

**Placeholder scan:** one deliberate "reuse the existing fixture" instruction in Task 5 Step 9 (the `test_megatron_args.py` helper names differ by repo state) — flagged explicitly with a fallback (clone the nearest `single_step` test verbatim), not a silent TODO.
