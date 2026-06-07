# POET Single-Step Fast Path (Closed-Form Backward) — Plan 1: standard POETLinear

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the per-token rotation chain in `POETLinear`'s forward/backward by recognizing that, with `merge_period=1`, `oft_R=0` at every forward so `R=Cayley(0)=Identity` — replacing the rotate→matmul→rotate chain with a single (permuted) GEMM forward plus a hand-written closed-form backward for `oft_R`.

**Architecture:** A custom `torch.autograd.Function` (`SingleStepPOETFunction`) whose forward is `u = x[…,perm_in_inv]; v = u@Wᵀ + b; y = v[…,perm_out]` (the identity-rotation chain collapsed) and whose backward, **saving only `x`** (one activation — same memory as a plain linear, the whole point of choosing the custom-backward Option B over a graph-saving fold), computes the `oft_R` gradient in closed form **in the chain's natural (right-multiply) orientation**: with `Gv = grad_y[…,perm_out_inv]` and `A = (x[…,perm_in_inv])ᵀ @ Gv` (the rotation-frame weight gradient), `M_in = A @ W`, `M_out = W @ A + outer(bias, Σₜ Gvₜ)` (the bias rank-1 term), then `grad_oft_R_{out,in} = 2·skewvec_blockdiag(M_{out,in})`. The factor 2 is the Cayley Jacobian at 0. **Verified bit-exact (≤3e-14) against autograd through poet's actual `cayley_batch`+chain** for bc∈{1,2,4}, bias∈{F,T}, random perms: [/tmp/poet_plan_selfcheck3.py](/tmp/poet_plan_selfcheck3.py). (NB: a naive `G Wᵀ` form is WRONG — the chain right-multiplies `R` so the skew sign flips, and it drops the bias term; use the `A`/`W` form below.) Gated behind a new `optim.poet.single_step_fast` flag, legal only when `merge_period==1` and `parameterization=cayley`.

**Tech Stack:** PyTorch custom autograd Function, the vendored `poet_torch` package, Megatron arg plumbing, Hydra config, pytest (CPU).

**Scope note:** This plan changes ONLY `POETLinear` (the MLP `fc1_gate`/`fc1_up`/`fc2`, and attention when `head_aligned_attn=false`). `HeadAlignedPOETLinear` subclasses `POETLinear` but **overrides `forward`**, so it is untouched here and keeps using the old chain (still correct). The head-aligned attention fast path is Plan 2 (`2026-06-07-poet-head-aligned-fast-backward.md`), to be implemented after this one. Since the MLP is ~79% of all POET rotation FLOPs, Plan 1 alone delivers the majority of the speedup even in the head-aligned config.

**Math / convention reference (do not skip):**
- Forward chain being replaced: [poet_layer.py:302-344 `chain_layer_x_fast_decoupled`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L302-L344). Input is gathered by `perm_in_inv`, output gathered by `perm_out` ([PermutationFunction, poet_ops.py:287-297](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_ops.py#L287-L297): `forward → x[...,perm]`, `backward → grad[...,inv_perm]`).
- `R=I` invariant: [poet_merge_step.py:208-216](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L208-L216) zeros `oft_R` (model + master) after every step when `merge_period=1`.
- Cayley Jacobian: [poet_layer.py:214-219 `cayley_batch`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L214-L219) = `I + 2(Q+Q²+Q³) + 2Q⁴`, so `dR/dQ|₀ = 2·dQ` → factor 2.
- skew↔vec ordering is `torch.triu_indices(b,b,1)` (`rows`, `cols` buffers on the layer; [poet_layer.py:597-602](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L597-L602)).
- **Rotation orientation (load-bearing):** the chain applies `R` on the RIGHT (`bmm(x_block, R)`, i.e. `x@R`), [poet_layer.py:337,342](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L337-L342). The closed-form below is derived in exactly this orientation (`M_out = vᵀ@Gv`, `M_in = (uᵀ@Gv)@W`); do not substitute a left-multiply form.

---

## File Structure

- **Create** `third_party/poet_torch/single_step.py` — the `SingleStepPOETFunction` autograd Function + `_blockdiag_skew_vec` helper (pure torch; no `src` import so the package stays standalone).
- **Modify** `third_party/poet_torch/__init__.py` — export `SingleStepPOETFunction`.
- **Modify** `third_party/poet_torch/poet_layer.py` — `POETLinear.forward` dispatches to the fast path when `self.single_step_fast` is set; default attribute `single_step_fast=False`.
- **Modify** `src/optim/poet_layers.py` — `replace_linears_with_poet(..., single_step_fast=False)` sets `pl.single_step_fast` on every wrapped layer.
- **Modify** `src/patches/poet_apply_to_model.py` — read `args.poet_single_step_fast`, pass to the walk.
- **Modify** `launchers/pretrain_gpt_slm.py` — register `--poet-single-step-fast` (store_true).
- **Modify** `src/utils/megatron_args.py` — emit `--poet-single-step-fast`; validate `merge_period==1` and `parameterization=cayley` when set.
- **Create** `tests/unit/test_single_step_fast.py` — CPU equivalence test (fast Function vs real `cayley_batch`+chain at `oft_R=0`) + arg-validation test.

---

## Task 1: The closed-form backward Function

**Files:**
- Create: `third_party/poet_torch/single_step.py`
- Create: `tests/unit/test_single_step_fast.py`

- [ ] **Step 1: Write the failing test** (forward + grad equivalence vs the real chain at `oft_R=0`)

Create `tests/unit/test_single_step_fast.py`:

```python
"""CPU equivalence test for the single-step (R=I) fast path.

At oft_R=0 the real POET chain (cayley_batch + chain_layer_x_fast_decoupled,
both pure-torch and CPU-runnable) must produce the SAME forward output and the
SAME oft_R gradients as SingleStepPOETFunction. We compare against poet's actual
cayley_batch (the Neumann series the Triton kernel implements), so this is a
faithful check of the production math, not a toy reimplementation.
"""
import torch
import pytest

from poet_torch import POETLinear, SingleStepPOETFunction
from poet_torch.poet_layer import (
    cayley_batch,
    pytorch_skew_symmetric,
    chain_layer_x_fast_decoupled,
)


def _reference_chain(pl, x):
    """Forward through the REAL chain with R built from oft_R via cayley_batch."""
    Qin = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    Qout = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    Rin, Rout = cayley_batch(Qin), cayley_batch(Qout)
    return chain_layer_x_fast_decoupled(
        x, Rin, pl.weight, pl.bias, Rout,
        pl.perm_in_inv, pl.perm_in, pl.perm_out, pl.perm_out_inv,
        pl.block_size_in, pl.block_size_out,
    )


def _fast(pl, x):
    return SingleStepPOETFunction.apply(
        x, pl.oft_R_in, pl.oft_R_out, pl.weight, pl.bias,
        pl.perm_in_inv, pl.perm_in, pl.perm_out, pl.perm_out_inv,
        pl.rows_in, pl.cols_in, pl.rows_out, pl.cols_out,
        pl.block_size_in, pl.block_size_out,
    )


@pytest.mark.parametrize("in_f,out_f,bc,bias", [(12, 8, 1, False), (12, 8, 2, False), (16, 16, 4, True)])
def test_fast_matches_chain_at_zero(in_f, out_f, bc, bias):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=in_f, out_features=out_f, block_count=bc, bias=bias)
    with torch.no_grad():
        pl.weight.normal_()
        if bias:
            pl.bias.normal_()
    # oft_R is the deployed-invariant value: 0 (R=I). Keep it 0; both paths read it.
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0

    x = torch.randn(5, in_f, requires_grad=True)
    gy = torch.randn(5, out_f)

    # forward equality
    y_ref = _reference_chain(pl, x)
    y_fast = _fast(pl, x)
    assert torch.allclose(y_ref, y_fast, atol=1e-10), (y_ref - y_fast).abs().max()

    # grad equality (oft_R_in/out and x). Two independent backward passes.
    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    x_ref = x.detach().clone().requires_grad_(True)
    (_reference_chain(pl, x_ref) * gy).sum().backward()
    g_in_ref, g_out_ref = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone()
    gx_ref = x_ref.grad.clone()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    x_fast = x.detach().clone().requires_grad_(True)
    (_fast(pl, x_fast) * gy).sum().backward()
    g_in_fast, g_out_fast = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone()
    gx_fast = x_fast.grad.clone()

    assert torch.allclose(g_in_ref, g_in_fast, atol=1e-9), (g_in_ref - g_in_fast).abs().max()
    assert torch.allclose(g_out_ref, g_out_fast, atol=1e-9), (g_out_ref - g_out_fast).abs().max()
    assert torch.allclose(gx_ref, gx_fast, atol=1e-9), (gx_ref - gx_fast).abs().max()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_fast.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'SingleStepPOETFunction' from 'poet_torch'`.

- [ ] **Step 3: Implement the Function**

Create `third_party/poet_torch/single_step.py`:

```python
"""Single-step (R=Identity) fast path for POET.

When merge_period=1 the rotation generators oft_R are folded into the frozen
base weight and zeroed after EVERY optimizer step, so oft_R=0 at every forward
=> R = Cayley(0) = Identity. The stock chain (chain_layer_x_fast_decoupled) then
computes x@I = x via per-token bmms — pure overhead (~3x the base GEMM FLOPs)
that exists only to produce the gradient of oft_R at R=I.

SingleStepPOETFunction collapses that: the forward is the permuted GEMM the chain
reduces to at R=I (u = x[perm_in_inv]; v = u@W^T + bias; y = v[perm_out]), and the
backward (saving ONLY x) computes the oft_R gradient in closed form, in the
chain's NATURAL right-multiply orientation (with Gv = grad_y[perm_out_inv] and
A = (x[perm_in_inv])^T @ Gv, the rotation-frame weight gradient):

    M_in  = A @ W
    M_out = W @ A + outer(bias, Gv.sum(0))   # bias rank-1 term (0 if no bias)
    grad_oft_R_{out,in} = 2 * blockdiag_skew_vec(M_{out,in})

The factor 2 is the Cayley Jacobian at 0 (cayley_batch(Q) = I + 2Q + O(Q^2)).
WARNING: the chain right-multiplies R (x@R), so a left-multiply 'G@W^T' form has
the WRONG skew sign and omits bias -- use the A/W form above (verified bit-exact
in /tmp/poet_plan_selfcheck3.py). ONLY valid at oft_R=0 (merge_period=1) and
parameterization='cayley'. The caller gates on both. The post-step merge is
unchanged: it still builds the real rotation from the stepped oft_R and folds it
into W.
"""
from __future__ import annotations

import torch


def _blockdiag_skew_vec(full: torch.Tensor, b: int, rows: torch.Tensor,
                        cols: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    """Project a [d,d] matrix onto the per-block strictly-upper-triangular skew
    basis: for each diagonal block M_k take factor*(M_k - M_k^T)[rows,cols].

    Returns (nb, n_elems) matching the oft_R layout / triu_indices(b,b,1) order.
    """
    d = full.shape[0]
    nb = d // b
    # diagonal blocks: view [nb, b, nb, b] then pick blocks[k,:,k,:]
    blocks = full.reshape(nb, b, nb, b)
    idx = torch.arange(nb, device=full.device)
    diag = blocks[idx, :, idx, :]                     # (nb, b, b)
    skew = diag - diag.transpose(-1, -2)              # (nb, b, b)
    return factor * skew[:, rows.long(), cols.long()]  # (nb, n_elems)


class SingleStepPOETFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, weight, bias,
                perm_in_inv, perm_in, perm_out, perm_out_inv,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out):
        # R=I: chain collapses to a permuted GEMM. oft_R_in/oft_R_out are accepted
        # as inputs (their VALUES are unused here — they are 0) purely so autograd
        # routes the closed-form grads to them in backward.
        u = x.index_select(-1, perm_in_inv)
        v = u @ weight.t()
        if bias is not None:
            v = v + bias
        y = v.index_select(-1, perm_out)
        # Save ONLY x (one activation, same memory as a plain linear). u and the
        # rotation-frame weight grad A are recomputed in backward; v is never
        # needed (M_out uses the W@A form + bias term). bias may be None.
        ctx.save_for_backward(x, weight, bias, perm_in_inv, perm_in, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, weight, bias, perm_in_inv, perm_in, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out = ctx.block_size_in, ctx.block_size_out
        out_f, in_f = weight.shape

        Gv = grad_y.index_select(-1, perm_out_inv)        # un-permute output grad
        grad_x = (Gv @ weight).index_select(-1, perm_in)  # un-permute -> grad wrt x

        # Closed form, chain's right-multiply orientation (factor 2):
        #   A = u^T @ Gv  (rotation-frame weight grad);  M_in = A @ W;  M_out = W @ A (+bias)
        u = x.index_select(-1, perm_in_inv)
        Gv2 = Gv.reshape(-1, out_f)
        A = u.reshape(-1, in_f).t() @ Gv2                 # (in, out)
        M_in = A @ weight                                 # (in, in)
        M_out = weight @ A                                # (out, out)
        if bias is not None:
            M_out = M_out + torch.outer(bias, Gv2.sum(0))
        grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out)
        grad_oft_R_in = _blockdiag_skew_vec(M_in, bs_in, rows_in, cols_in)

        grad_oft_R_in = grad_oft_R_in.to(weight.dtype)
        grad_oft_R_out = grad_oft_R_out.to(weight.dtype)
        # 15 forward inputs -> 15 returns: real grads for x/oft_R_in/oft_R_out, then
        # 12 None (weight, bias, perm_in_inv, perm_in, perm_out, perm_out_inv,
        # rows_in, cols_in, rows_out, cols_out, block_size_in, block_size_out).
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None, None,
                None, None, None, None, None, None)
```

- [ ] **Step 4: Export it.** Add to `third_party/poet_torch/__init__.py` (end of file):

```python
from .single_step import SingleStepPOETFunction as SingleStepPOETFunction
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_fast.py -x -q`
Expected: PASS (3 parametrizations).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/single_step.py third_party/poet_torch/__init__.py tests/unit/test_single_step_fast.py
git commit -m "feat(poet): closed-form single-step (R=I) backward Function"
```

---

## Task 2: Dispatch from `POETLinear.forward`

**Files:**
- Modify: `third_party/poet_torch/poet_layer.py` (`POETLinear.forward`, ~line 720; add default attr in `__init__`, ~line 548)

- [ ] **Step 1: Write the failing test** (layer routes through the fast path when flagged, and equals the chain)

Append to `tests/unit/test_single_step_fast.py`:

```python
def test_layer_forward_uses_fast_path_when_flagged():
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=2, bias=False)
    with torch.no_grad():
        pl.weight.normal_()
    pl.single_step_fast = True
    x = torch.randn(3, 12, requires_grad=True)
    gy = torch.randn(3, 8)
    # layer forward (fast) must equal the reference chain at oft_R=0
    y = pl(x)
    assert torch.allclose(y, _reference_chain(pl, x), atol=1e-10)
    (y * gy).sum().backward()
    assert pl.oft_R_in.grad is not None and pl.oft_R_out.grad is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_fast.py::test_layer_forward_uses_fast_path_when_flagged -x -q`
Expected: FAIL — `POETLinear` has no `single_step_fast` attribute / forward ignores it (calls Triton cayley, errors on CPU).

- [ ] **Step 3: Add default attribute in `POETLinear.__init__`.** After `self.mem_efficient_mode = mem_efficient_mode` (~line 548), add:

```python
        # Single-step (R=I) fast path: collapses the identity-rotation chain to a
        # permuted GEMM + closed-form oft_R grad. Set by replace_linears_with_poet
        # when optim.poet.single_step_fast is on (requires merge_period=1, cayley).
        self.single_step_fast = False
```

- [ ] **Step 4: Dispatch in `POETLinear.forward`.** At the very top of `forward` (before the `if self.parameterization == "exp"` branch, ~line 720), add:

```python
        # Single-step fast path (R=I). The cayley guard is defensive: the factor-2
        # closed form is Cayley-specific, and build-time validation already forbids
        # single_step_fast with parameterization='exp' — this just fails safe
        # (falls through to the correct chain) if the flag is ever set directly.
        if getattr(self, "single_step_fast", False) and self.parameterization == "cayley":
            from .single_step import SingleStepPOETFunction
            return SingleStepPOETFunction.apply(
                x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
                self.perm_in_inv, self.perm_in, self.perm_out, self.perm_out_inv,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
                self.block_size_in, self.block_size_out,
            )
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_fast.py -x -q`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/poet_layer.py tests/unit/test_single_step_fast.py
git commit -m "feat(poet): dispatch POETLinear.forward to single-step fast path when flagged"
```

---

## Task 3: Plumb the `single_step_fast` flag through the layer walk

**Files:**
- Modify: `src/optim/poet_layers.py` (`replace_linears_with_poet` signature ~line 179-194; set attr in both wrap branches ~line 273 and ~line 339)
- Modify: `tests/unit/test_poet_layers.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/unit/test_poet_layers.py`:

```python
def test_single_step_fast_flag_set_on_wrapped_layers():
    import torch.nn as nn
    from src.optim.poet_layers import replace_linears_with_poet, POETMegatronLinear

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16, bias=False)

    m = M()
    replace_linears_with_poet(
        m, block_count=1, init_type="none",
        extra_linear_types=(nn.Linear,), single_step_fast=True,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    assert m.fc1.poet_linear.single_step_fast is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py::test_single_step_fast_flag_set_on_wrapped_layers -x -q`
Expected: FAIL — `replace_linears_with_poet() got an unexpected keyword argument 'single_step_fast'`.

- [ ] **Step 3: Add the parameter.** In `src/optim/poet_layers.py`, add to the signature (after `resid_permute: bool = True,`, ~line 193):

```python
    single_step_fast: bool = False,
```

- [ ] **Step 4: Set the attribute on every wrapped layer.** In the head-aligned branch, immediately after `_copy_and_init_weight(pl, child, init_type, mup_alpha)` (~line 269, before `wrapper = POETMegatronLinear(`), add:

```python
                    pl.single_step_fast = single_step_fast
```

And in the standard branch, immediately after `_copy_and_init_weight(pl, child, init_type, mup_alpha)` (~line 334, before `wrapper = POETMegatronLinear(`), add:

```python
                pl.single_step_fast = single_step_fast
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -x -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_layers.py tests/unit/test_poet_layers.py
git commit -m "feat(poet): thread single_step_fast through replace_linears_with_poet"
```

---

## Task 4: Read the arg in the apply patch

**Files:**
- Modify: `src/patches/poet_apply_to_model.py` (`_apply_poet_to_chunk`, ~line 61-86)

- [ ] **Step 1: Read the flag.** In `_apply_poet_to_chunk`, after `resid_permute = not getattr(args, "poet_no_head_resid_perm", False)` (~line 70), add:

```python
        single_step_fast = getattr(args, "poet_single_step_fast", False)
```

- [ ] **Step 2: Pass it to the walk.** In the `return replace_linears_with_poet(` call (~line 74-86), add as a kwarg after `resid_permute=resid_permute,`:

```python
            single_step_fast=single_step_fast,
```

- [ ] **Step 3: Verify import compiles**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/poet_apply_to_model.py`
Expected: no output (success).

- [ ] **Step 4: Commit**

```bash
git add src/patches/poet_apply_to_model.py
git commit -m "feat(poet): pass poet_single_step_fast from apply patch to layer walk"
```

---

## Task 5: Register the CLI flag

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` (~line 106, near `--poet-head-aligned-attn`)

- [ ] **Step 1: Register the argument.** After `group.add_argument("--poet-no-head-resid-perm", action="store_true")` (~line 108), add:

```python
    # Single-step (R=I) fast path: collapse the identity-rotation chain to a
    # permuted GEMM + closed-form oft_R grad. ONLY valid with merge_period=1 and
    # parameterization=cayley (validated in src/utils/megatron_args.py).
    group.add_argument("--poet-single-step-fast", action="store_true")
```

- [ ] **Step 2: Verify it parses**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile launchers/pretrain_gpt_slm.py`
Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add launchers/pretrain_gpt_slm.py
git commit -m "feat(poet): register --poet-single-step-fast CLI flag"
```

---

## Task 6: Emit the flag + validate `merge_period==1` and cayley

**Files:**
- Modify: `src/utils/megatron_args.py` (poet branch, validation near ~line 259-266; emission near ~line 346)
- Modify: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/unit/test_megatron_args.py`. This reuses the file's existing `_poet_cfg(poet_overrides)` helper (lines ~130-148; note it defaults `merge_period=200`) and the `_optimizer_args` entry point used by the sibling poet tests (e.g. `test_poet_argv_emits_lie_ortho_distributed`):

```python
def test_single_step_fast_requires_merge_period_one():
    import pytest
    from src.utils.megatron_args import _optimizer_args

    # _poet_cfg defaults merge_period=200 -> single_step_fast must be rejected.
    with pytest.raises(ValueError, match="single_step_fast"):
        _optimizer_args(_poet_cfg({"block_count": 1, "single_step_fast": True}))


def test_single_step_fast_emits_flag_when_merge_period_one():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "merge_period": 1, "single_step_fast": True})
    )
    assert "--poet-single-step-fast" in args
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py::test_single_step_fast_requires_merge_period_one -x -q`
Expected: FAIL — no ValueError raised (flag not validated yet).

- [ ] **Step 3: Add validation.** In `src/utils/megatron_args.py`, in the `if kind == "poet":` block, after the `reinit_period`/`merge_period` validation (~line 266), add:

```python
        if poet.get("single_step_fast", False):
            if merge_period != 1:
                raise ValueError(
                    "optim.poet.single_step_fast requires merge_period=1 "
                    f"(R=Identity at forward only holds when oft_R is folded+zeroed "
                    f"every step); got merge_period={merge_period}."
                )
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError(
                    "optim.poet.single_step_fast requires parameterization=cayley "
                    "(the factor-2 closed-form grad is the Cayley Jacobian at 0)."
                )
```

- [ ] **Step 4: Emit the flag.** In the same block, after `if poet.get("head_aligned_attn", False): poet_args.append("--poet-head-aligned-attn")` (~line 347), add:

```python
        # store_true: single-step (R=I) fast path (closed-form backward).
        if poet.get("single_step_fast", False):
            poet_args.append("--poet-single-step-fast")
```

- [ ] **Step 5: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -x -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): emit + validate --poet-single-step-fast (merge_period=1, cayley)"
```

---

## Task 7: Config key + docs

**Files:**
- Modify: `configs/experiments/optim/poet_lie_orth.yaml` (add `single_step_fast: false` under `optim.poet`)
- Modify: `docs/experiments/poet_lie_orth.md` (document the flag + the A/B command)

- [ ] **Step 1: Add the config key.** In `configs/experiments/optim/poet_lie_orth.yaml`, under `optim.poet:` (after `head_aligned_attn: true`, ~line 70), add:

```yaml
    single_step_fast: false      # opt-in R=I fast path; requires merge_period=1 + cayley
```

- [ ] **Step 2: Document.** Append to `docs/experiments/poet_lie_orth.md`:

```markdown
## Single-step fast path (`single_step_fast`)

With `merge_period=1` the rotation is folded into `W` and `oft_R` zeroed every
step, so `R=Identity` at every forward. `optim.poet.single_step_fast=true`
collapses the identity-rotation chain to a permuted GEMM and computes the `oft_R`
gradient in closed form (factor-2 Cayley Jacobian), removing ~3x the base-GEMM
rotation FLOPs from the MLP (and non-head-aligned attention). Mathematically
identical training. A/B:

```bash
# baseline (chain)
codexlog lieorth_chain bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true
# fast path
codexlog lieorth_fast bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.single_step_fast=true
```

Compare steady-state `elapsed time per iteration (ms)` (skip first ~20 iters for
compile warmup) and confirm the loss curves overlap.
```

- [ ] **Step 3: Commit**

```bash
git add configs/experiments/optim/poet_lie_orth.yaml docs/experiments/poet_lie_orth.md
git commit -m "docs(poet): add single_step_fast config key + A/B instructions"
```

---

## Task 8: Full CPU regression + handoff to GPU smoke

**Files:** (no code change — verification)

- [ ] **Step 1: Run the POET CPU unit suite**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_fast.py tests/unit/test_poet_layers.py tests/unit/test_megatron_args.py -q`
Expected: all PASS.

- [ ] **Step 2: Hand the GPU parity smoke to the user.** The CPU test validates the math against `cayley_batch`; a GPU run confirms the Triton `cayley` path and DDP `main_grad` integration are identical. Provide this command for the user to run (do NOT launch it):

```bash
# Short head-to-head: 50 steps each, loss curves must overlap, fast must be faster.
codexlog ss_chain bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  training.train_iters=50
codexlog ss_fast bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.single_step_fast=true training.train_iters=50
```

Acceptance: per-step loss matches the chain run within optimizer noise; `elapsed time per iteration (ms)` lower on the MLP-dominated forward.

- [ ] **Step 3: Final commit (if any doc/CHANGELOG updates)**

```bash
git add -A && git commit -m "chore(poet): single-step fast path Plan 1 complete"
```

---

## Self-Review

- **Spec coverage:** Function (T1) → layer dispatch (T2) → walk plumbing (T3) → apply patch (T4) → CLI flag (T5) → emit+validate (T6) → config/docs (T7) → regression+GPU handoff (T8). The `merge_period=1` and `cayley` invariants are validated at build time (T6). `HeadAlignedPOETLinear` deliberately out of scope (its `forward` override is untouched → still correct via the chain; Plan 2 handles it).
- **Type consistency:** `single_step_fast` (bool) is the name used everywhere: layer attr, walk kwarg, arg `poet_single_step_fast`, CLI `--poet-single-step-fast`, config `optim.poet.single_step_fast`. `SingleStepPOETFunction.apply(...)` arg order matches between the test, the layer dispatch (T2), and the Function signature (T1). `_blockdiag_skew_vec(full, b, rows, cols, factor)` returns `(nb, n_elems)` matching `oft_R` layout.
- **Placeholder scan:** none. T6's test uses the real `_optimizer_args` entry point and the existing `_poet_cfg` helper (verified in `tests/unit/test_megatron_args.py`, which defaults `merge_period=200`). All insertion points (poet_layer.py forward ~720 / `__init__` ~548; poet_layers.py ~193/~269/~334; megatron_args.py ~250-266/~347; pretrain_gpt_slm.py ~108) verified against the current repo.
- **Defensive guards:** the forward dispatch also checks `self.parameterization == "cayley"` (the factor-2 form is Cayley-specific) so it fails safe even if the flag is set on an `exp` layer outside the build-time validation.
