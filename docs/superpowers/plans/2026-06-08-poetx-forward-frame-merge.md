# POETXLinear Forward-Frame (Perm-Free Forward) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `POETXLinear` (+ ops in `poet_ops.py`) that stores the weight **already in the forward frame** (`Wx = W_perm[perm_out][:,perm_in]`, the conjugation's effective weight at `R=I`), so the single-step forward is a **bare GEMM with zero permutations** and `grad_x` is plain too — while the merge round-trips through the existing verified fold so its math is reused unchanged.

**Architecture:** POET's chain is the conjugation `W_eff = (P_out R_out P_outᵀ)·W·(P_in R_in P_inᵀ)`; the code stores the pre-folded `W_perm = P_outᵀ W P_in` (this is `pl.weight`), so at `R=I` the effective weight is `W = P_out W_perm P_inᵀ = W_perm[perm_out][:,perm_in]`. `single_step_native` rebuilds that effective weight from `W_perm` **every forward** (two `O(d²)` `index_select`s + a bias gather) and the backward rebuilds it again. `POETXLinear` instead **stores the effective weight `Wx` directly** (baked once at build, re-derived once per merge step), making the forward `y = x@Wx.t() + bias_eff` — no perms. The backward keeps the **2 `[d,d]` `conj` relabels** for the `oft_R` gradient (mathematically forced cross-frame; never touches activations). The merge reuses `POETLinear._fold_with_R` verbatim by bracketing it with an un-permute/re-permute (forward-frame ⇄ `W_perm`), so it is bit-identical to the proven fold and supports perm reinit. Gated behind `optim.poet.single_step_x`; `POETLinear`, `SingleStepPOETLinear`/`NativeSingleStepFunction`, the chain, and the merge are left **completely intact**.

**Tech Stack:** PyTorch custom autograd Function, the vendored `poet_torch` package, the `poet_merge_step` patch, Megatron arg plumbing, Hydra config, pytest (CPU). The real Cayley is a Triton GPU op; CPU tests build `R` with the pure-torch `cayley_batch`/`pytorch_skew_symmetric` and compare against the real `chain_layer_x_fast_decoupled` (same approach the `single_step_native` tests use).

**Math (the whole thing):** with `Wx = W_perm[perm_out][:,perm_in]` (`= pl.weight[perm_out][:,perm_in]`) and `bias_eff = bias[perm_out]`:
- forward: `y = x @ Wx.t() (+ bias_eff)` — **zero perm**.
- backward: `grad_x = grad_y @ Wx` **plain**; `G = xᵀ@grad_y`; `M_in = conj(G@Wx, perm_in_inv)`; `M_out = conj(Wx@G + outer(bias_eff, Σ grad_y), perm_out_inv)`; `grad_oft_R_{in,out} = 2·blockdiag_skew_vec(M_{in,out})`, where `conj(M,p)=M[p][:,p]` and `blockdiag_skew_vec` is the existing helper (factor 2 = Cayley Jacobian at 0). This is the **same** math as `NativeSingleStepFunction`, with `Wx`/`bias_eff` read from storage instead of rebuilt from `W_perm` each call.
- merge (per step, `merge_period=1`): recover `W_perm = Wx[perm_out_inv][:,perm_in_inv]`, run the **verified** `POETLinear._fold_with_R` (folds the stepped `R`, zeros `oft_R`, resamples perms if `reinit_perm`), then re-permute back to forward frame with the (possibly new) perms. The stored effective weight is perm-invariant, so this is exactly forward-invariant.

**Scope:** standard `POETLinear` only (MLP `fc1_gate`/`fc1_up`/`fc2`, non-head-aligned attention). Head-aligned attention is already gather-free (`HeadAlignedSingleStepFunction`, identity perms) — `single_step_x` keeps it on that fast path (same as `single_step_native`). Not bit-identical to `single_step_native` on GPU (different GEMM reduction order ⇒ bf16 noise); acceptance is loss overlap, same standard as the native-vs-fast A/B. Requires `merge_period=1` + `parameterization=cayley` (caller-gated). Perm reinit (`reinit_period>0`) is supported via the round-trip.

---

## File Structure

- **Create** `third_party/poet_torch/poet_ops.py` — `POETXSingleStepFunction` (the perm-free-forward autograd Function) + local `_conj` helper, reusing `_blockdiag_skew_vec` from `single_step`.
- **Create** `third_party/poet_torch/poetx_layer.py` — `POETXLinear(nn.Module)`: standalone layer storing the forward-frame weight; `forward` → `POETXSingleStepFunction`, `_fold_with_R`/`merge_then_reinitialize`/`_merge_R`/`_build_R` reuse `POETLinear`'s verified fold via bracket-and-delegate, `bake_perms_into_weight()` for build-time init.
- **Modify** `third_party/poet_torch/__init__.py` — export `POETXSingleStepFunction` and `POETXLinear`.
- **Modify** `src/patches/poet_merge_step.py` — widen the merge gate `isinstance(pl, POETLinear)` → `isinstance(pl, (POETLinear, POETXLinear))`.
- **Modify** `src/optim/poet_layers.py` — `replace_linears_with_poet(..., single_step_x=False)`: when set, build `POETXLinear` for standard linears (and `bake_perms_into_weight()` after weight copy); head-aligned gets `single_step_fast=True`.
- **Modify** `src/patches/poet_apply_to_model.py` — read `args.poet_single_step_x`, pass to the walk.
- **Modify** `launchers/pretrain_gpt_slm.py` — register `--poet-single-step-x`.
- **Modify** `src/utils/megatron_args.py` — emit `--poet-single-step-x`; validate `merge_period==1` + `parameterization=cayley`.
- **Create** `tests/unit/test_poetx_layer.py` — CPU op/layer equivalence vs the chain + merge round-trip equivalence vs `POETLinear` + forward-invariance under reinit.
- **Modify** `tests/unit/test_poet_layers.py`, `tests/unit/test_megatron_args.py`, `tests/unit/test_poet_merge_batched.py` (or a focused new test) — walk dispatch, arg emit/validate, merge recognition.

---

## Task 1: `POETXSingleStepFunction` (perm-free-forward op)

**Files:**
- Create: `third_party/poet_torch/poet_ops.py`
- Create: `tests/unit/test_poetx_layer.py`

- [ ] **Step 1: Write the failing test** (op forward+grad vs the real chain at `oft_R=0`, with `Wx`/`bias_eff` precomputed)

Create `tests/unit/test_poetx_layer.py`:

```python
"""CPU equivalence tests for the forward-frame (perm-free-forward) POETX path.

POETXSingleStepFunction takes the forward-frame weight Wx = W_perm[perm_out][:,perm_in]
and bias_eff = bias[perm_out] directly, so its forward is a bare GEMM. At oft_R=0 it must
match the real chain (cayley_batch + chain_layer_x_fast_decoupled, pure-torch CPU) to fp64,
and POETXLinear must be a drop-in for the chain forward + reuse the verified merge fold.
"""
import pytest
import torch
from poet_torch import POETLinear, POETXSingleStepFunction
from poet_torch.poet_layer import (
    cayley_batch,
    chain_layer_x_fast_decoupled,
    pytorch_skew_symmetric,
)


def _chain_ref(pl, x):
    qi = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    qo = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    return chain_layer_x_fast_decoupled(
        x, cayley_batch(qi), pl.weight, pl.bias, cayley_batch(qo),
        pl.perm_in_inv, pl.perm_in, pl.perm_out, pl.perm_out_inv,
        pl.block_size_in, pl.block_size_out,
    )


def _forward_frame(pl):
    """Wx = W_perm[perm_out][:,perm_in]; bias_eff = bias[perm_out] (or None)."""
    Wx = pl.weight.index_select(0, pl.perm_out).index_select(1, pl.perm_in)
    bias_eff = None if pl.bias is None else pl.bias.index_select(0, pl.perm_out)
    return Wx, bias_eff


def _op(pl, Wx, bias_eff, x):
    return POETXSingleStepFunction.apply(
        x, pl.oft_R_in, pl.oft_R_out, Wx, bias_eff,
        pl.perm_in_inv, pl.perm_out_inv,
        pl.rows_in, pl.cols_in, pl.rows_out, pl.cols_out,
        pl.block_size_in, pl.block_size_out,
    )


@pytest.mark.parametrize("bc,bias", [(1, False), (1, True), (2, False), (2, True)])
def test_op_matches_chain_at_zero(bc, bias):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=bc, bias=bias)
    with torch.no_grad():
        pl.weight.normal_()
        if bias:
            pl.bias.normal_()
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0
    Wx, bias_eff = _forward_frame(pl)

    x = torch.randn(5, 12)
    gy = torch.randn(5, 8)

    assert torch.allclose(_chain_ref(pl, x), _op(pl, Wx, bias_eff, x), atol=1e-9), \
        (_chain_ref(pl, x) - _op(pl, Wx, bias_eff, x)).abs().max()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xr = x.clone().requires_grad_(True)
    (_chain_ref(pl, xr) * gy).sum().backward()
    gi_r, go_r, gx_r = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xr.grad.clone()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xn = x.clone().requires_grad_(True)
    (_op(pl, Wx, bias_eff, xn) * gy).sum().backward()
    gi_n, go_n, gx_n = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xn.grad.clone()

    assert torch.allclose(gi_r, gi_n, atol=1e-9), (gi_r - gi_n).abs().max()
    assert torch.allclose(go_r, go_n, atol=1e-9), (go_r - go_n).abs().max()
    assert torch.allclose(gx_r, gx_n, atol=1e-9), (gx_r - gx_n).abs().max()


def test_op_forward_is_bare_gemm():
    """The forward applies NO permutation: it is exactly x@Wx.t() (+ bias_eff)."""
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=1, bias=True)
    with torch.no_grad():
        pl.weight.normal_()
        pl.bias.normal_()
    Wx, bias_eff = _forward_frame(pl)
    x = torch.randn(3, 12)
    assert (_op(pl, Wx, bias_eff, x) - (x @ Wx.t() + bias_eff)).abs().max().item() == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poetx_layer.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'POETXSingleStepFunction'`.

- [ ] **Step 3: Implement the op.** Create `third_party/poet_torch/poet_ops.py`:

```python
"""Forward-frame (perm-free-forward) single-step ops for POETX.

POET's chain is the conjugation W_eff = (P_out R_out P_outᵀ)·W·(P_in R_in P_inᵀ); the
stored pl.weight is the pre-folded W_perm = P_outᵀ W P_in, so at oft_R=0 the effective
weight is W = P_out W_perm P_inᵀ = W_perm[perm_out][:,perm_in]. POETXSingleStepFunction
takes that EFFECTIVE weight Wx and the effective bias bias_eff = bias[perm_out] DIRECTLY,
so the forward is a bare GEMM (no permutation) and grad_x is plain. The 2 conj on the
small [d,d] gradient matrices remain (mathematically forced cross-frame relabel for the
oft_R gradient; backward only, never touches activations). Same closed form as
NativeSingleStepFunction (factor 2 = Cayley Jacobian at 0), with Wx/bias_eff read from
storage instead of rebuilt. ONLY valid at oft_R=0 (merge_period=1) and cayley.
"""
from __future__ import annotations

import torch

from .single_step import _blockdiag_skew_vec


def _conj(M, p):
    """Permutation conjugation M[p][:,p] (exact gather, no arithmetic)."""
    return M.index_select(0, p).index_select(1, p)


class POETXSingleStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, Wx, bias_eff,
                perm_in_inv, perm_out_inv,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out):
        # Wx is ALREADY the forward-frame (effective) weight -> bare GEMM, zero perm.
        # Only the INVERSE perms are needed (in the backward conj); the perm-free
        # forward uses none. oft_R_in/oft_R_out are inputs only so autograd routes
        # the closed-form grads to them.
        y = x @ Wx.t()
        if bias_eff is not None:
            y = y + bias_eff
        ctx.save_for_backward(x, Wx, bias_eff, perm_in_inv, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, Wx, bias_eff, perm_in_inv, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out = ctx.block_size_in, ctx.block_size_out
        out_f, in_f = Wx.shape

        grad_x = grad_y @ Wx                                       # PLAIN — no gather
        G = x.reshape(-1, in_f).t() @ grad_y.reshape(-1, out_f)    # [in, out]
        M_in = _conj(G @ Wx, perm_in_inv)                          # [in, in] block frame
        M_out_nat = Wx @ G                                         # [out, out]
        if bias_eff is not None:
            M_out_nat = M_out_nat + torch.outer(bias_eff, grad_y.reshape(-1, out_f).sum(0))
        M_out = _conj(M_out_nat, perm_out_inv)
        grad_oft_R_in = _blockdiag_skew_vec(M_in, bs_in, rows_in, cols_in).to(Wx.dtype)
        grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out).to(Wx.dtype)
        # 13 inputs -> 13 returns: grads for x/oft_R_in/oft_R_out, then 10 None.
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None,
                None, None, None, None, None)
```

- [ ] **Step 4: Export.** In `third_party/poet_torch/__init__.py`, after the `single_step_native` exports, add:

```python
from .poet_ops import POETXSingleStepFunction as POETXSingleStepFunction
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poetx_layer.py -x -q`
Expected: PASS (4 parametrizations + bare-GEMM anchor).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/poet_ops.py third_party/poet_torch/__init__.py tests/unit/test_poetx_layer.py
git commit -m "feat(poet): POETXSingleStepFunction (perm-free-forward single-step op)"
```

---

## Task 2: `POETXLinear` (standalone layer, forward-frame storage, round-trip merge)

**Files:**
- Create: `third_party/poet_torch/poetx_layer.py`
- Modify: `third_party/poet_torch/__init__.py`
- Modify: `tests/unit/test_poetx_layer.py`

- [ ] **Step 1: Write the failing tests.** Append to `tests/unit/test_poetx_layer.py`:

```python
def _build_R(pl):
    """Pure-torch (CPU) R-build from the current oft_R (cayley)."""
    qi = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    qo = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    return cayley_batch(qo), cayley_batch(qi)  # (R_out, R_in)


def _make_pair(in_f=12, out_f=8, bc=2, bias=True, seed=3):
    """A POETLinear and a POETXLinear sharing identical weights + perms."""
    from poet_torch import POETXLinear
    torch.manual_seed(seed)
    base = POETLinear(in_features=in_f, out_features=out_f, block_count=bc, bias=bias)
    with torch.no_grad():
        base.weight.normal_()
        if bias:
            base.bias.normal_()
    xl = POETXLinear(in_features=in_f, out_features=out_f, block_count=bc, bias=bias)
    with torch.no_grad():
        for b in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(xl, b).copy_(getattr(base, b))
        # POETX stores the FORWARD-FRAME weight: Wx = W_perm[perm_out][:,perm_in].
        xl.weight.copy_(base.weight.index_select(0, base.perm_out).index_select(1, base.perm_in))
        if bias:
            xl.bias.copy_(base.bias.index_select(0, base.perm_out))
    return base, xl


def test_layer_forward_matches_chain():
    torch.set_default_dtype(torch.float64)
    base, xl = _make_pair()
    x = torch.randn(3, 12, requires_grad=True)
    gy = torch.randn(3, 8)
    y = xl(x)                                  # bare-GEMM forward
    assert torch.allclose(y, _chain_ref(base, x), atol=1e-9), (y - _chain_ref(base, x)).abs().max()
    (y * gy).sum().backward()
    assert xl.oft_R_in.grad is not None and xl.oft_R_out.grad is not None


def test_merge_fold_matches_poetlinear():
    """After folding the SAME stepped R, POETX's stored forward-frame weight equals
    POETLinear's effective weight W_perm[perm_out][:,perm_in] (fp64)."""
    torch.set_default_dtype(torch.float64)
    base, xl = _make_pair()
    with torch.no_grad():  # a real (small) stepped rotation on both
        base.oft_R_in.normal_(std=1e-2); base.oft_R_out.normal_(std=1e-2)
        xl.oft_R_in.copy_(base.oft_R_in); xl.oft_R_out.copy_(base.oft_R_out)
    R_out, R_in = _build_R(base)
    base._fold_with_R(R_out, R_in, reinit_perm=False)
    xl._fold_with_R(R_out, R_in, reinit_perm=False)
    eff = base.weight.index_select(0, base.perm_out).index_select(1, base.perm_in)
    assert torch.allclose(xl.weight, eff, atol=1e-9), (xl.weight - eff).abs().max()
    assert torch.count_nonzero(xl.oft_R_in) == 0 and torch.count_nonzero(xl.oft_R_out) == 0


def test_merge_reinit_folds_and_resamples_perm():
    """Fold-with-reinit stores the correct (perm-invariant) effective weight AND
    resamples the perms. The effective weight after folding R is built independently
    from the OLD perms; reinit re-permutes storage but the effective weight is the same."""
    from poet_torch.poet_layer import block_diag_lr_matmul_decoupled
    torch.set_default_dtype(torch.float64)
    _, xl = _make_pair()
    with torch.no_grad():
        xl.oft_R_in.normal_(std=1e-2); xl.oft_R_out.normal_(std=1e-2)
    R_out, R_in = _build_R(xl)
    perm_in_before = xl.perm_in.clone()
    # Effective (forward-frame) weight the fold must produce, built with the OLD perms:
    W_perm = xl.weight.index_select(0, xl.perm_out_inv).index_select(1, xl.perm_in_inv)
    tmp = block_diag_lr_matmul_decoupled(R_in, W_perm.t(), R_out)
    tmp = tmp.index_select(0, xl.perm_in).index_select(1, xl.perm_out)
    eff_expected = tmp.t()

    xl._fold_with_R(R_out, R_in, reinit_perm=True)

    assert torch.allclose(xl.weight, eff_expected, atol=1e-9), (xl.weight - eff_expected).abs().max()
    assert not torch.equal(xl.perm_in, perm_in_before)      # perms resampled
    assert torch.count_nonzero(xl.oft_R_in) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poetx_layer.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'POETXLinear'`.

- [ ] **Step 3: Implement the layer.** Create `third_party/poet_torch/poetx_layer.py`:

```python
"""POETXLinear: standalone POET linear storing the weight in the FORWARD frame.

POETLinear stores W_perm (= P_outᵀ W P_in) and rebuilds the effective weight
W_eff = W_perm[perm_out][:,perm_in] every forward (the gathers). POETXLinear stores
W_eff DIRECTLY (baked once at build, re-derived once per merge step), so the single-step
forward is a bare GEMM (POETXSingleStepFunction) with NO permutation. It is NOT a
POETLinear subclass (the merge driver recognizes it via a widened isinstance tuple), but
its merge REUSES POETLinear._fold_with_R verbatim by bracketing it with an
un-permute/re-permute (forward-frame <-> W_perm), so the fold math is bit-identical to the
proven path and supports perm reinit. ONLY valid at oft_R=0 (merge_period=1) and cayley.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .poet_layer import POETLinear
from .poet_ops import POETXSingleStepFunction


class POETXLinear(nn.Module):
    def __init__(self, in_features, out_features, bsz=None, block_count=None,
                 bias=False, device=None, dtype=None, parameterization="cayley"):
        super().__init__()
        if parameterization != "cayley":
            raise ValueError(
                "POETXLinear requires parameterization='cayley' "
                f"(the perm-free-forward backward is Cayley-specific); got {parameterization!r}."
            )
        # Buffer/param setup mirrors POETLinear.__init__ (kept standalone on purpose:
        # POETXLinear is NOT a POETLinear subclass, so POETLinear.__init__'s zero-arg
        # super() cannot run on `self`). The merge methods below reuse POETLinear's
        # (super-free) fold/build helpers via unbound calls, so only __init__ duplicates.
        self.in_features = in_features
        self.out_features = out_features
        self.parameterization = parameterization
        self.single_step_fast = False  # POETX ignores it (forward is always the X op)

        if (bsz is None) == (block_count is None):
            raise ValueError("exactly one of bsz or block_count must be set")
        if bsz is not None:
            if in_features % bsz != 0 or out_features % bsz != 0:
                raise ValueError(
                    f"block_size {bsz} doesn't divide in={in_features} or out={out_features}"
                )
            block_size_in = block_size_out = bsz
        else:
            if in_features % block_count != 0 or out_features % block_count != 0:
                raise ValueError(
                    f"block_count {block_count} doesn't divide in={in_features} or out={out_features}"
                )
            block_size_in = in_features // block_count
            block_size_out = out_features // block_count
        self.block_size_in = block_size_in
        self.block_size_out = block_size_out
        self.block_size = block_size_in  # back-compat (merge "is-active" guard reads it)

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        r_in = in_features // block_size_in
        r_out = out_features // block_size_out
        n_elems_in = block_size_in * (block_size_in - 1) // 2
        n_elems_out = block_size_out * (block_size_out - 1) // 2
        self.oft_R_in = nn.Parameter(torch.zeros((r_in, n_elems_in), device=device, dtype=dtype))
        self.oft_R_out = nn.Parameter(torch.zeros((r_out, n_elems_out), device=device, dtype=dtype))
        self.r_in = r_in
        self.r_out = r_out

        rows_in, cols_in = torch.triu_indices(block_size_in, block_size_in, 1, device=device)
        self.register_buffer("rows_in", rows_in.to(torch.int32))
        self.register_buffer("cols_in", cols_in.to(torch.int32))
        rows_out, cols_out = torch.triu_indices(block_size_out, block_size_out, 1, device=device)
        self.register_buffer("rows_out", rows_out.to(torch.int32))
        self.register_buffer("cols_out", cols_out.to(torch.int32))

        perm_in = torch.randperm(in_features, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features, device=device, dtype=torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))
        # self.weight currently holds an (empty) [out, in] tensor in the W_perm frame;
        # bake_perms_into_weight() converts it to the forward frame once the real
        # weights have been copied in (the walk calls it after _copy_and_init_weight).

    @torch.no_grad()
    def bake_perms_into_weight(self) -> None:
        """Convert the freshly-copied W_perm storage into the forward-frame Wx =
        W_perm[perm_out][:,perm_in] (and bias_eff = bias[perm_out]). Idempotent only
        once per fresh copy — call exactly once at build, after the weight is set."""
        self.weight.copy_(
            self.weight.index_select(0, self.perm_out).index_select(1, self.perm_in)
        )
        if self.bias is not None:
            self.bias.copy_(self.bias.index_select(0, self.perm_out))

    def forward(self, x):
        # self.weight is the forward-frame Wx; self.bias is bias_eff (forward frame).
        return POETXSingleStepFunction.apply(
            x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
            self.perm_in_inv, self.perm_out_inv,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out,
        )

    def _build_R(self, oft_in, oft_out):
        return POETLinear._build_R(self, oft_in, oft_out)

    def _merge_R(self):
        return POETLinear._merge_R(self)

    @torch.no_grad()
    def _fold_with_R(self, R_out, R_in, reinit_perm: bool = True) -> None:
        """Round-trip fold: forward-frame -> W_perm, reuse the verified
        POETLinear._fold_with_R (folds R, zeros oft_R, resamples perms on reinit),
        then re-permute back to the forward frame with the (possibly new) perms.
        Bit-identical to POETLinear's effective weight by construction."""
        # forward-frame -> W_perm storage frame, using CURRENT perms
        self.weight.copy_(
            self.weight.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
        )
        if self.bias is not None:
            self.bias.copy_(self.bias.index_select(0, self.perm_out_inv))
        # verified fold (operates on W_perm; resamples self.perm_* on reinit)
        POETLinear._fold_with_R(self, R_out, R_in, reinit_perm=reinit_perm)
        # W_perm -> forward frame, using the (possibly NEW) perms
        self.weight.copy_(
            self.weight.index_select(0, self.perm_out).index_select(1, self.perm_in)
        )
        if self.bias is not None:
            self.bias.copy_(self.bias.index_select(0, self.perm_out))

    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        R_out, R_in = self._merge_R()
        self._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)
```

- [ ] **Step 4: Export.** In `third_party/poet_torch/__init__.py`, after the `POETXSingleStepFunction` export, add:

```python
from .poetx_layer import POETXLinear as POETXLinear
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poetx_layer.py -q`
Expected: PASS (op + layer forward + merge fold-equivalence + reinit invariance).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/poetx_layer.py third_party/poet_torch/__init__.py tests/unit/test_poetx_layer.py
git commit -m "feat(poet): POETXLinear (forward-frame storage, round-trip merge)"
```

---

## Task 3: Make the merge driver recognize `POETXLinear`

**Files:**
- Modify: `src/patches/poet_merge_step.py` (`_run_merge`, ~line 290 import + line 307 gate)
- Modify: `tests/unit/test_poetx_layer.py`

- [ ] **Step 1: Write the test.** Append to `tests/unit/test_poetx_layer.py`. This exercises the **real batched merge primitives** (`_build_R_batched` builds `R` across layers, then `pl._fold_with_R` folds each) — the exact path `_run_merge` uses in training — on CPU via the injectable pure-torch `cayley_fn` (the default is the Triton op, GPU-only):

```python
def test_batched_merge_folds_poetx():
    """POETX folds correctly through the real batched merge primitives
    (_build_R_batched + _fold_with_R), on CPU with the pure-torch cayley_fn."""
    import torch
    from poet_torch import POETXLinear
    from poet_torch.poet_layer import cayley_batch
    from src.patches.poet_merge_step import _build_R_batched

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(5)
    xl = POETXLinear(in_features=12, out_features=8, block_count=2, bias=False)
    with torch.no_grad():
        xl.weight.normal_()
        xl.oft_R_in.normal_(std=1e-2)
        xl.oft_R_out.normal_(std=1e-2)
    w_before = xl.weight.clone()
    built = _build_R_batched([xl], cayley_fn=cayley_batch)  # pure-torch R-build
    R_out, R_in = built[id(xl)]
    xl._fold_with_R(R_out, R_in, reinit_perm=False)
    assert torch.count_nonzero(xl.oft_R_in) == 0 and torch.count_nonzero(xl.oft_R_out) == 0
    assert not torch.allclose(xl.weight, w_before)  # rotation absorbed


def test_run_merge_gate_collects_poetx():
    """The collection filter _run_merge uses must accept a POETX wrapped in
    POETMegatronLinear (it would skip it pre-widen). Built directly (no walk) so
    this task does not depend on the single_step_x walk param added in a later task."""
    from poet_torch import POETLinear, POETXLinear
    from src.optim.poet_layers import POETMegatronLinear

    pl = POETXLinear(in_features=8, out_features=16, block_count=1, bias=False)
    wrapper = POETMegatronLinear(pl)
    # mirror _run_merge's per-module filter (isinstance(mod, POETMegatronLinear) then
    # isinstance(mod.poet_linear, (POETLinear, POETXLinear)) and block_size > 0)
    assert isinstance(wrapper, POETMegatronLinear)
    assert isinstance(wrapper.poet_linear, (POETLinear, POETXLinear))
    assert wrapper.poet_linear.block_size > 0
```

- [ ] **Step 2: Run to verify the batched-merge test passes (it validates POETX's fold works through the real primitives; the gate-collection test mirrors the filter we widen next)**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poetx_layer.py -q`
Expected: PASS. (`test_batched_merge_folds_poetx` passes once Task 2's `_fold_with_R` exists; `test_run_merge_gate_collects_poetx` documents the filter the source change below must satisfy.)

- [ ] **Step 3: Widen the gate.** In `src/patches/poet_merge_step.py`, change the import inside `_run_merge` (~line 290):

```python
    from poet_torch import POETLinear
```

to:

```python
    from poet_torch import POETLinear, POETXLinear
```

and the filter (~line 307):

```python
            if not isinstance(pl, POETLinear) or pl.block_size <= 0:
                continue
```

to:

```python
            if not isinstance(pl, (POETLinear, POETXLinear)) or pl.block_size <= 0:
                continue
```

- [ ] **Step 4: Verify it compiles + the unit test still passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/poet_merge_step.py && PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poetx_layer.py -q`
Expected: no compile output; tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_merge_step.py tests/unit/test_poetx_layer.py
git commit -m "feat(poet): merge driver recognizes POETXLinear (widen isinstance)"
```

---

## Task 4: Build `POETXLinear` in the layer walk under `single_step_x`

**Files:**
- Modify: `src/optim/poet_layers.py` (`replace_linears_with_poet` signature ~line 194; standard branch ~line 310-343; head-aligned ~line 271)
- Modify: `tests/unit/test_poet_layers.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/unit/test_poet_layers.py`:

```python
def test_single_step_x_uses_poetx_class():
    import torch.nn as nn
    from poet_torch import POETXLinear
    from src.optim.poet_layers import replace_linears_with_poet, POETMegatronLinear

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16, bias=False)

    m = M()
    orig = m.fc1.weight.detach().clone()
    replace_linears_with_poet(
        m, block_count=1, init_type="none",
        extra_linear_types=(nn.Linear,), single_step_x=True,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    pl = m.fc1.poet_linear
    assert isinstance(pl, POETXLinear)
    # The stored weight is the FORWARD frame: Wx = orig[perm_out][:,perm_in].
    eff = orig.index_select(0, pl.perm_out.long()).index_select(1, pl.perm_in.long())
    import torch
    assert torch.allclose(pl.weight, eff.to(pl.weight.dtype), atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py::test_single_step_x_uses_poetx_class -x -q`
Expected: FAIL — `replace_linears_with_poet() got an unexpected keyword argument 'single_step_x'`.

- [ ] **Step 3: Add the parameter.** In `src/optim/poet_layers.py`, add to the signature (after `single_step_native: bool = False,`, ~line 195):

```python
    single_step_x: bool = False,
```

- [ ] **Step 4: Build `POETXLinear` for standard linears.** In the standard branch, replace the class-selection block (the `if cache_mode == "none":` block added by the native plan, ~line 311-323):

```python
                if cache_mode == "none":
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
                if cache_mode == "none":
                    if single_step_x:
                        from poet_torch import POETXLinear as _PoetCls  # noqa: N806
                    elif single_step_native:
                        from poet_torch import SingleStepPOETLinear as _PoetCls  # noqa: N806
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

- [ ] **Step 5: Bake the perms into the weight for POETX.** In the standard branch, immediately after `_copy_and_init_weight(pl, child, init_type, mup_alpha)` (the standard-branch call, ~line 336), add:

```python
                if single_step_x:
                    # POETX stores the forward-frame weight; convert the just-copied
                    # natural weight Wx = W[perm_out][:,perm_in] (one-time, at build).
                    pl.bake_perms_into_weight()
```

- [ ] **Step 6: Head-aligned stays gather-free under `single_step_x`.** In the head-aligned branch, change the `pl.single_step_fast = single_step_fast or single_step_native` line (~line 271) to:

```python
                    pl.single_step_fast = single_step_fast or single_step_native or single_step_x
```

- [ ] **Step 7: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -q`
Expected: PASS except the known pre-existing `test_sharded_state_dict_is_deduped_replicated_and_complete` (megatron.core importorskip on CPU — unrelated).

- [ ] **Step 8: Commit**

```bash
git add src/optim/poet_layers.py tests/unit/test_poet_layers.py
git commit -m "feat(poet): build POETXLinear in the walk under single_step_x"
```

---

## Task 5: Read the arg in the apply patch + register the CLI flag

**Files:**
- Modify: `src/patches/poet_apply_to_model.py` (`_apply_poet_to_chunk`, ~line 71-88)
- Modify: `launchers/pretrain_gpt_slm.py` (~line 112-115, near `--poet-single-step-native`)

- [ ] **Step 1: Read the flag.** In `src/patches/poet_apply_to_model.py`, after `single_step_native = getattr(args, "poet_single_step_native", False)` (added by the native plan), add:

```python
        single_step_x = getattr(args, "poet_single_step_x", False)
```

- [ ] **Step 2: Pass it.** In the `return replace_linears_with_poet(` call, after `single_step_native=single_step_native,`, add:

```python
            single_step_x=single_step_x,
```

- [ ] **Step 3: Register the CLI flag.** In `launchers/pretrain_gpt_slm.py`, after `group.add_argument("--poet-single-step-native", action="store_true")`, add:

```python
    # Forward-frame perm-free-forward path (standalone POETXLinear). Implies the
    # single-step fast path; requires merge_period=1 + cayley.
    group.add_argument("--poet-single-step-x", action="store_true")
```

- [ ] **Step 4: Verify both compile**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/poet_apply_to_model.py launchers/pretrain_gpt_slm.py`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_apply_to_model.py launchers/pretrain_gpt_slm.py
git commit -m "feat(poet): wire --poet-single-step-x through apply patch + CLI"
```

---

## Task 6: Emit + validate the flag

**Files:**
- Modify: `src/utils/megatron_args.py` (poet branch, validation after the `single_step_native` block ~line 279; emission after the `single_step_native` append ~line 365)
- Modify: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/unit/test_megatron_args.py`:

```python
def test_single_step_x_requires_merge_period_one():
    import pytest
    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="single_step_x"):
        _optimizer_args(_poet_cfg({"block_count": 1, "single_step_x": True}))


def test_single_step_x_emits_flag_when_merge_period_one():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "merge_period": 1, "single_step_x": True})
    )
    assert "--poet-single-step-x" in args
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py::test_single_step_x_requires_merge_period_one -x -q`
Expected: FAIL — no ValueError raised.

- [ ] **Step 3: Add validation.** In `src/utils/megatron_args.py`, in the `if kind == "poet":` block, after the existing `single_step_native` validation block, add:

```python
        if poet.get("single_step_x", False):
            if merge_period != 1:
                raise ValueError(
                    "optim.poet.single_step_x requires merge_period=1 "
                    f"(R=Identity at forward); got merge_period={merge_period}."
                )
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError(
                    "optim.poet.single_step_x requires parameterization=cayley."
                )
```

- [ ] **Step 4: Emit the flag.** After the `if poet.get("single_step_native", False): poet_args.append("--poet-single-step-native")` block, add:

```python
        # store_true: forward-frame perm-free-forward path.
        if poet.get("single_step_x", False):
            poet_args.append("--poet-single-step-x")
```

- [ ] **Step 5: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): emit + validate --poet-single-step-x (merge_period=1, cayley)"
```

---

## Task 7: CPU regression + GPU A/B handoff

**Files:** (verification only)

- [ ] **Step 1: Run the POET CPU suite**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poetx_layer.py tests/unit/test_single_step_native.py tests/unit/test_single_step_fast.py tests/unit/test_poet_layers.py tests/unit/test_megatron_args.py -q`
Expected: PASS except the known pre-existing `test_sharded_state_dict_is_deduped_replicated_and_complete` (megatron.core importorskip on CPU — unrelated).

- [ ] **Step 2: ruff**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m ruff check third_party/poet_torch/poet_ops.py third_party/poet_torch/poetx_layer.py src/optim/poet_layers.py src/patches/poet_merge_step.py src/utils/megatron_args.py tests/unit/test_poetx_layer.py`
Expected: `All checks passed!`

- [ ] **Step 3: Hand the GPU A/B to the user (do NOT launch).** Compare `single_step_x` (POETXLinear, perm-free forward) vs `single_step_native` (rebuilds the effective weight each forward — the only difference). Acceptance: loss overlaps within bf16 noise (NOT 0.0 — different GEMM reduction order), and ms/iter ≤ native (the forward + grad_x perms are gone; the merge gains one round-trip per step):

```bash
codexlog x_forward bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.single_step_x=true

# baseline for comparison (current native-frame path that rebuilds W_eff each forward):
codexlog x_native bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.single_step_native=true
```

Send the two logs for the loss/timing/memory comparison. Expectation vs native: forward and `grad_x` drop all `index_select`s (5 fewer per layer per microbatch), the 2 backward `conj` remain, and the merge adds one weight round-trip per **step** (amortized across grad-accum microbatches) — so the win grows with the grad-accumulation count.

- [ ] **Step 4: Config default (optional, after A/B clears).** Once the A/B confirms loss overlap, flip the default in `configs/experiments/optim/poet_lie_orth.yaml`: replace `single_step_native: true` with `single_step_x: true` (both imply the head-aligned fast path). Leave `single_step_fast` as the documented fallback; the precedence in the walk is `single_step_x` > `single_step_native` > `single_step_fast` for standard layers.

```bash
git add configs/experiments/optim/poet_lie_orth.yaml
git commit -m "feat(poet): enable single_step_x by default for poet_lie_orth"
```

---

## Self-Review

- **Spec coverage:** standalone `POETXLinear` + `poet_ops.py` (Tasks 1-2); forward-frame storage / bare-GEMM forward (Task 1 op + Task 2 `bake_perms_into_weight`/`forward`); round-trip merge reusing the verified fold + perm-reinit support (Task 2 `_fold_with_R`, tests `test_merge_fold_matches_poetlinear` / `test_merge_reinit_is_forward_invariant`); merge-driver recognition via widened isinstance (Task 3); walk wiring (Task 4); arg/CLI/validate (Tasks 5-6); A/B + default (Task 7). Plain-torch ops first; the fused-kernel decision is deferred to after the A/B (per the brainstorm). Backward keeps the 2 `[d,d]` `conj` (documented as mathematically forced).
- **Placeholder scan:** none — every code step is complete; the GPU A/B is explicitly the user's to run with exact commands and the loss-overlap (not 0.0) acceptance.
- **Type consistency:** `POETXSingleStepFunction.apply(...)` arg order is identical in the op test (`_op`), the layer (`POETXLinear.forward`), and the Function signature: `(x, oft_R_in, oft_R_out, Wx, bias_eff, perm_in_inv, perm_out_inv, rows_in, cols_in, rows_out, cols_out, block_size_in, block_size_out)` — 13 inputs → 13 backward returns (3 grads + 10 None); the perm-free forward needs no perms and the backward conj uses only the inverse perms, so `perm_in`/`perm_out` are intentionally not passed. `bake_perms_into_weight`, `_fold_with_R`, `_merge_R`, `_build_R`, `merge_then_reinitialize` are the methods the merge driver (`_merge_layers` cayley path calls `_fold_with_R`; disable-batch path calls `merge_then_reinitialize`) and the walk (`bake_perms_into_weight`) invoke. `single_step_x` is the name across config/CLI(`--poet-single-step-x`)/arg(`poet_single_step_x`)/walk.
- **Correctness basis:** the op forward+backward is the same closed form already verified bit-against the chain to ~1e-14 (fp64) for `single_step_native`; the only change is `Wx`/`bias_eff` are read from storage, re-tested here to fp64. The merge is `POETLinear._fold_with_R` bracketed by a permute/un-permute, so it is bit-identical to the proven fold by construction (test `test_merge_fold_matches_poetlinear` asserts equality to `POETLinear`'s effective weight to fp64). Not bit-identical to `single_step_native` on GPU (different reduction order) → loss-overlap acceptance.
- **Old code intact:** `POETLinear`, `SingleStepPOETLinear`, `NativeSingleStepFunction`, `chain_layer_x_fast_decoupled`, and the merge fold are untouched; `POETXLinear` is standalone (reuses `POETLinear.__init__`/`_build_R`/`_merge_R`/`_fold_with_R` via unbound calls, no override of POETLinear behavior), recognized by the merge driver only through the widened `isinstance(pl, (POETLinear, POETXLinear))` tuple. Precedence: `single_step_x` wins over `single_step_native`/`single_step_fast` for standard layers (its forward ignores those flags); head-aligned layers use the existing gather-free fast path under any of the three.
- **Open follow-up (deferred, per brainstorm):** if the A/B shows the 2 backward `conj` launches mattering at scale, fuse `conj`+`blockdiag_skew_vec` into one Triton gather-skew op in `poet_ops.py` (GPU-only verification). Not in scope until a torch baseline shows the need.
```
