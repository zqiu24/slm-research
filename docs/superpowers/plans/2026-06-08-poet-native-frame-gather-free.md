# POET Native-Frame Gather-Free Single-Step Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the per-token permutation gathers in the standard `POETLinear` single-step path by recognizing that at `R=I` the conjugating perms cancel — the forward becomes a pure GEMM on the *permuted* (forward-frame) weight `W_eff`, built with one `O(d²)` relabel instead of five `O(N·d)` activation gathers, and the backward applies the permutation only to small `[d,d]` matrices.

**Architecture:** Add an all-new `NativeSingleStepFunction` (pure-GEMM forward `y = x@W_effᵀ` where `W_eff = W[perm_out][:,perm_in]`; closed-form backward with `grad_x = grad_y@W_eff` **plain** and the `oft_R` grads computed by conjugating the `[d,d]` gradient matrices into the block frame) and an all-new `SingleStepPOETLinear(POETLinear)` whose *only* override is `forward`. Weight **storage stays natural** (un-permuted, exactly as `POETLinear` already stores it) and the **merge is inherited unchanged** (the already-verified batched/replicated merge), so this drops into the existing merge machinery with zero changes to it. Everything is gated behind a new `optim.poet.single_step_native` flag; `POETLinear`, `SingleStepPOETFunction`, the chain, and the merge are left **completely intact**.

**Tech Stack:** PyTorch custom autograd Function, the vendored `poet_torch` package, Megatron arg plumbing, Hydra config, pytest (CPU). The real Cayley is a Triton GPU op; CPU tests build R with the pure-torch `cayley_batch` and compare against the real `chain_layer_x_fast_decoupled`.

**Verified before writing this plan** ([/tmp/poet_native_frame_selfcheck.py](/tmp/poet_native_frame_selfcheck.py)): the exact forward+backward below matches the chain to fp64 machine precision (~1e-14) for `bc∈{1,2}`, ±bias; the identity-perm forward is bit-identical (0.0) to `chain_noperm`.

**Math (the whole thing):** at `oft_R=0` the layer's effective base is `P_out·W·P_in` (the perms cancel the rotation, [poet_layer.py:667](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L667)). With `W_eff = W.index_select(0,perm_out).index_select(1,perm_in)`:
- forward: `y = x @ W_effᵀ (+ bias[perm_out])`
- backward: `grad_x = grad_y @ W_eff`; `G = xᵀ@grad_y`; `M_in = conj(G@W_eff, perm_in_inv)`; `M_out = conj(W_eff@G + outer(bias[perm_out], Σ grad_y), perm_out_inv)`; `grad_oft_R_{in,out} = 2·blockdiag_skew_vec(M_{in,out})`, where `conj(M,p) = M[p][:,p]` and `blockdiag_skew_vec` is the existing helper.

**Scope:** standard `POETLinear` only (the MLP `fc1_gate`/`fc1_up`/`fc2`, and non-head-aligned attention). Head-aligned attention is **already** gather-free (`HeadAlignedSingleStepFunction`, identity perms) — `single_step_native` keeps using it for attention. Not bit-identical to the current `single_step_fast` (different GEMM reduction order ⇒ bf16 noise); acceptance is loss overlap, same standard as the original fast-vs-chain A/B.

---

## File Structure

- **Create** `third_party/poet_torch/single_step_native.py` — `NativeSingleStepFunction` (+ local `_conj` helper, reusing `_blockdiag_skew_vec` from `single_step`) and `SingleStepPOETLinear(POETLinear)` (overrides only `forward`).
- **Modify** `third_party/poet_torch/__init__.py` — export both.
- **Modify** `src/optim/poet_layers.py` — `replace_linears_with_poet(..., single_step_native=False)`: when set, create `SingleStepPOETLinear` for standard linears and set `single_step_fast=True` on head-aligned layers.
- **Modify** `src/patches/poet_apply_to_model.py` — read `args.poet_single_step_native`, pass to the walk.
- **Modify** `launchers/pretrain_gpt_slm.py` — register `--poet-single-step-native`.
- **Modify** `src/utils/megatron_args.py` — emit `--poet-single-step-native`; validate `merge_period==1` + `parameterization=cayley`.
- **Create** `tests/unit/test_single_step_native.py` — CPU equivalence vs the chain + identity-perm bit-identity + layer dispatch.

---

## Task 1: `NativeSingleStepFunction`

**Files:**
- Create: `third_party/poet_torch/single_step_native.py`
- Create: `tests/unit/test_single_step_native.py`

- [ ] **Step 1: Write the failing test** (Function forward+grad vs the real chain at `oft_R=0`)

Create `tests/unit/test_single_step_native.py`:

```python
"""CPU equivalence test for the native-frame (gather-free) single-step path.

At oft_R=0 the native forward (pure GEMM on the forward-frame weight W_eff) and its
closed-form backward must match the real chain (cayley_batch + chain_layer_x_fast_decoupled,
pure-torch CPU). Verified bit-against the chain to fp64 machine precision.
"""
import pytest
import torch
from poet_torch import POETLinear, NativeSingleStepFunction
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


def _native(pl, x):
    return NativeSingleStepFunction.apply(
        x, pl.oft_R_in, pl.oft_R_out, pl.weight, pl.bias,
        pl.perm_in, pl.perm_in_inv, pl.perm_out, pl.perm_out_inv,
        pl.rows_in, pl.cols_in, pl.rows_out, pl.cols_out,
        pl.block_size_in, pl.block_size_out,
    )


@pytest.mark.parametrize("bc,bias", [(1, False), (1, True), (2, False), (2, True)])
def test_native_matches_chain_at_zero(bc, bias):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=bc, bias=bias)
    with torch.no_grad():
        pl.weight.normal_()
        if bias:
            pl.bias.normal_()
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0

    x = torch.randn(5, 12)
    gy = torch.randn(5, 8)

    assert torch.allclose(_chain_ref(pl, x), _native(pl, x), atol=1e-9), \
        (_chain_ref(pl, x) - _native(pl, x)).abs().max()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xr = x.clone().requires_grad_(True)
    (_chain_ref(pl, xr) * gy).sum().backward()
    gi_r, go_r, gx_r = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xr.grad.clone()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xn = x.clone().requires_grad_(True)
    (_native(pl, xn) * gy).sum().backward()
    gi_n, go_n, gx_n = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xn.grad.clone()

    assert torch.allclose(gi_r, gi_n, atol=1e-9), (gi_r - gi_n).abs().max()
    assert torch.allclose(go_r, go_n, atol=1e-9), (go_r - go_n).abs().max()
    assert torch.allclose(gx_r, gx_n, atol=1e-9), (gx_r - gx_n).abs().max()


def test_native_forward_identity_perm_is_bit_identical():
    """With identity perms, the native forward is exactly x@Wᵀ (the bit-identity anchor)."""
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = POETLinear(in_features=12, out_features=8, block_count=1, bias=False)
    with torch.no_grad():
        pl.weight.normal_()
        pl.perm_in.copy_(torch.arange(12, dtype=torch.int32))
        pl.perm_in_inv.copy_(torch.arange(12, dtype=torch.int32))
        pl.perm_out.copy_(torch.arange(8, dtype=torch.int32))
        pl.perm_out_inv.copy_(torch.arange(8, dtype=torch.int32))
    x = torch.randn(3, 12)
    assert (_native(pl, x) - x @ pl.weight.t()).abs().max().item() == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_native.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'NativeSingleStepFunction'`.

- [ ] **Step 3: Implement the Function.** Create `third_party/poet_torch/single_step_native.py`:

```python
"""Native-frame (gather-free) single-step fast path for standard POETLinear.

At oft_R=0 the conjugating perms cancel, so the layer's effective base is P_out·W·P_in
and the forward is a pure GEMM on the forward-frame weight W_eff = W[perm_out][:,perm_in]
(one O(d^2) relabel, vs the chain's five O(N*d) activation gathers). The backward is the
closed form (factor 2 = Cayley Jacobian at 0), with grad_x PLAIN and the oft_R grads
obtained by conjugating the small [d,d] gradient matrices into the block frame.

Storage stays NATURAL (un-permuted W, exactly as POETLinear stores it); the merge is
inherited unchanged. ONLY valid at oft_R=0 (merge_period=1) and parameterization=cayley
(the caller gates on both). Verified bit-against the chain in
/tmp/poet_native_frame_selfcheck.py.
"""
from __future__ import annotations

import torch

from .poet_layer import POETLinear
from .single_step import _blockdiag_skew_vec


def _conj(M, p):
    """Permutation conjugation M[p][:,p] (exact gather, no arithmetic)."""
    return M.index_select(0, p).index_select(1, p)


class NativeSingleStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, weight, bias,
                perm_in, perm_in_inv, perm_out, perm_out_inv,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out):
        # Forward-frame weight: one O(d^2) relabel; then a pure GEMM. oft_R_in/oft_R_out
        # are inputs only so autograd routes the closed-form grads to them.
        W_eff = weight.index_select(0, perm_out).index_select(1, perm_in)
        y = x @ W_eff.t()
        if bias is not None:
            y = y + bias.index_select(0, perm_out)
        ctx.save_for_backward(x, weight, bias, perm_in, perm_in_inv, perm_out, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, weight, bias, perm_in, perm_in_inv, perm_out, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out = ctx.block_size_in, ctx.block_size_out
        out_f, in_f = weight.shape

        W_eff = weight.index_select(0, perm_out).index_select(1, perm_in)
        grad_x = grad_y @ W_eff                                    # PLAIN — no gather
        G = x.reshape(-1, in_f).t() @ grad_y.reshape(-1, out_f)    # [in, out]
        M_in = _conj(G @ W_eff, perm_in_inv)                       # [in, in] block frame
        M_out_nat = W_eff @ G                                      # [out, out]
        if bias is not None:
            b_eff = bias.index_select(0, perm_out)
            M_out_nat = M_out_nat + torch.outer(b_eff, grad_y.reshape(-1, out_f).sum(0))
        M_out = _conj(M_out_nat, perm_out_inv)
        grad_oft_R_in = _blockdiag_skew_vec(M_in, bs_in, rows_in, cols_in).to(weight.dtype)
        grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out).to(weight.dtype)
        # 15 inputs -> 15 returns: grads for x/oft_R_in/oft_R_out, then 12 None.
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None, None,
                None, None, None, None, None, None)


class SingleStepPOETLinear(POETLinear):
    """POETLinear that uses the gather-free native-frame single-step forward.

    Identical to POETLinear in every way (natural weight storage, perm/oft_R buffers,
    the inherited merge_then_reinitialize / _fold_with_R) EXCEPT the forward, which
    routes to NativeSingleStepFunction. Valid only at oft_R=0 (merge_period=1, cayley).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The native backward hard-codes the factor-2 Cayley Jacobian and has NO
        # chain fallback, so refuse exp loudly rather than silently produce wrong
        # grads (build-time validation also forbids it; this guards direct use).
        if self.parameterization != "cayley":
            raise ValueError(
                "SingleStepPOETLinear requires parameterization='cayley'; "
                f"got {self.parameterization!r}."
            )

    def forward(self, x):
        return NativeSingleStepFunction.apply(
            x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
            self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out,
        )
```

- [ ] **Step 4: Export.** Add to `third_party/poet_torch/__init__.py`:

```python
from .single_step_native import NativeSingleStepFunction as NativeSingleStepFunction
from .single_step_native import SingleStepPOETLinear as SingleStepPOETLinear
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_native.py -x -q`
Expected: PASS (4 parametrizations + identity-perm anchor).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/single_step_native.py third_party/poet_torch/__init__.py tests/unit/test_single_step_native.py
git commit -m "feat(poet): native-frame gather-free single-step Function + layer"
```

---

## Task 2: Layer-level dispatch test (the new class through its own forward)

**Files:**
- Modify: `tests/unit/test_single_step_native.py`

- [ ] **Step 1: Write the test.** Append:

```python
def test_layer_forward_matches_chain():
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    from poet_torch import SingleStepPOETLinear

    base = POETLinear(in_features=12, out_features=8, block_count=2, bias=True)
    with torch.no_grad():
        base.weight.normal_()
        base.bias.normal_()
    # Build a SingleStepPOETLinear sharing the same weights/perms as `base`.
    layer = SingleStepPOETLinear(in_features=12, out_features=8, block_count=2, bias=True)
    with torch.no_grad():
        layer.weight.copy_(base.weight)
        layer.bias.copy_(base.bias)
        for b in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(layer, b).copy_(getattr(base, b))

    x = torch.randn(3, 12, requires_grad=True)
    gy = torch.randn(3, 8)
    y = layer(x)                              # native forward
    assert torch.allclose(y, _chain_ref(base, x), atol=1e-9)
    (y * gy).sum().backward()
    assert layer.oft_R_in.grad is not None and layer.oft_R_out.grad is not None
```

- [ ] **Step 2: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_native.py -q`
Expected: PASS (all).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_single_step_native.py
git commit -m "test(poet): SingleStepPOETLinear layer forward matches chain"
```

---

## Task 3: Create the new class in the layer walk

**Files:**
- Modify: `src/optim/poet_layers.py` (`replace_linears_with_poet` signature ~line 194; standard branch ~line 308-340)
- Modify: `tests/unit/test_poet_layers.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/unit/test_poet_layers.py`:

```python
def test_single_step_native_uses_new_class():
    import torch.nn as nn
    from poet_torch import SingleStepPOETLinear
    from src.optim.poet_layers import replace_linears_with_poet, POETMegatronLinear

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16, bias=False)

    m = M()
    replace_linears_with_poet(
        m, block_count=1, init_type="none",
        extra_linear_types=(nn.Linear,), single_step_native=True,
    )
    assert isinstance(m.fc1, POETMegatronLinear)
    assert isinstance(m.fc1.poet_linear, SingleStepPOETLinear)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py::test_single_step_native_uses_new_class -x -q`
Expected: FAIL — `replace_linears_with_poet() got an unexpected keyword argument 'single_step_native'`.

- [ ] **Step 3: Add the parameter.** In `src/optim/poet_layers.py`, add to the signature (after `single_step_fast: bool = False,`, ~line 194):

```python
    single_step_native: bool = False,
```

- [ ] **Step 4: Use the new class for standard linears.** In the standard branch, replace the `POETLinear(...)` construction (the `if cache_mode == "none":` block, ~line 308-317) with a class selection. Change:

```python
                if cache_mode == "none":
                    pl = POETLinear(
                        in_features=in_f,
                        out_features=out_f,
                        bias=has_bias,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                        parameterization=parameterization,
                        **block_kwargs,
                    )
```

to:

```python
                if cache_mode == "none":
                    if single_step_native:
                        from poet_torch import SingleStepPOETLinear as _PoetCls
                    else:
                        _PoetCls = POETLinear
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

- [ ] **Step 5: Ensure head-aligned still goes gather-free under native.** In the head-aligned branch, after `pl.single_step_fast = single_step_fast` (~line 271, from the earlier plan), make `single_step_native` imply the head-aligned fast path too:

```python
                    pl.single_step_fast = single_step_fast or single_step_native
```

- [ ] **Step 6: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/optim/poet_layers.py tests/unit/test_poet_layers.py
git commit -m "feat(poet): build SingleStepPOETLinear in the walk under single_step_native"
```

---

## Task 4: Read the arg in the apply patch + register the CLI flag

**Files:**
- Modify: `src/patches/poet_apply_to_model.py` (`_apply_poet_to_chunk`, ~line 70)
- Modify: `launchers/pretrain_gpt_slm.py` (~line 108, near `--poet-single-step-fast`)

- [ ] **Step 1: Read the flag.** In `src/patches/poet_apply_to_model.py`, after `single_step_fast = getattr(args, "poet_single_step_fast", False)` (added by the earlier plan), add:

```python
        single_step_native = getattr(args, "poet_single_step_native", False)
```

- [ ] **Step 2: Pass it.** In the `return replace_linears_with_poet(` call, after `single_step_fast=single_step_fast,`, add:

```python
            single_step_native=single_step_native,
```

- [ ] **Step 3: Register the CLI flag.** In `launchers/pretrain_gpt_slm.py`, after `group.add_argument("--poet-single-step-fast", action="store_true")`, add:

```python
    # Gather-free native-frame single-step path (standard POETLinear). Implies the
    # single-step fast path; requires merge_period=1 + cayley.
    group.add_argument("--poet-single-step-native", action="store_true")
```

- [ ] **Step 4: Verify both compile**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/poet_apply_to_model.py launchers/pretrain_gpt_slm.py`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_apply_to_model.py launchers/pretrain_gpt_slm.py
git commit -m "feat(poet): wire --poet-single-step-native through apply patch + CLI"
```

---

## Task 5: Emit + validate the flag

**Files:**
- Modify: `src/utils/megatron_args.py` (poet branch, validation ~line 266-275; emission ~line 348)
- Modify: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/unit/test_megatron_args.py`:

```python
def test_single_step_native_requires_merge_period_one():
    import pytest
    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="single_step_native"):
        _optimizer_args(_poet_cfg({"block_count": 1, "single_step_native": True}))


def test_single_step_native_emits_flag_when_merge_period_one():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg({"block_count": 1, "merge_period": 1, "single_step_native": True})
    )
    assert "--poet-single-step-native" in args
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py::test_single_step_native_requires_merge_period_one -x -q`
Expected: FAIL — no ValueError raised.

- [ ] **Step 3: Add validation.** In `src/utils/megatron_args.py`, in the `if kind == "poet":` block, after the existing `single_step_fast` validation, add:

```python
        if poet.get("single_step_native", False):
            if merge_period != 1:
                raise ValueError(
                    "optim.poet.single_step_native requires merge_period=1 "
                    f"(R=Identity at forward); got merge_period={merge_period}."
                )
            if poet.get("parameterization", "cayley") != "cayley":
                raise ValueError(
                    "optim.poet.single_step_native requires parameterization=cayley."
                )
```

- [ ] **Step 4: Emit the flag.** After `if poet.get("single_step_fast", False): poet_args.append("--poet-single-step-fast")`, add:

```python
        # store_true: gather-free native-frame single-step path.
        if poet.get("single_step_native", False):
            poet_args.append("--poet-single-step-native")
```

- [ ] **Step 5: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): emit + validate --poet-single-step-native (merge_period=1, cayley)"
```

---

## Task 6: CPU regression + GPU A/B handoff

**Files:** (verification only)

- [ ] **Step 1: Run the POET CPU suite**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_native.py tests/unit/test_single_step_fast.py tests/unit/test_poet_layers.py tests/unit/test_megatron_args.py -q`
Expected: PASS except the known pre-existing `test_sharded_state_dict_is_deduped_replicated_and_complete` (megatron.core importorskip on CPU — unrelated).

- [ ] **Step 2: ruff**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m ruff check third_party/poet_torch/single_step_native.py src/optim/poet_layers.py src/utils/megatron_args.py tests/unit/test_single_step_native.py`
Expected: `All checks passed!`

- [ ] **Step 3: Hand the GPU A/B to the user (do NOT launch).** Compare the native path vs the current `single_step_fast` (the gathers are the only difference). Acceptance: loss overlaps within bf16 noise (NOT 0.0 — the native forward changes the GEMM reduction order), and ms/iter is lower (the per-token gathers are gone):

```bash
codexlog native_fast bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.single_step_native=true

# baseline for comparison (current gather-based single-step fast path):
codexlog native_baseline bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.single_step_fast=true
```

Send the two logs for the loss/timing/memory comparison.

**Launch-count caveat:** the native path trades 5 `O(N·d)` activation gathers for ~8 `O(d²)` `index_select` launches/layer/microbatch (two `W_eff` relabels — forward + backward — plus the two `_conj` per side). Far less data moved, but *more, smaller* kernel launches. At 60m that could partly offset the gather removal; if the A/B shows the relabels dominating, the follow-up is to **cache `W_eff`** (recompute lazily, invalidate on merge) so it's built once per step rather than per microbatch. (`W_eff` is recomputed rather than `save_for_backward`'d on purpose — saving it would add `O(d²)` activation memory per layer, against the save-only-`x` memory goal.)

- [ ] **Step 4: Config key (optional, after A/B clears).** Once the A/B confirms loss overlap, flip the default in `configs/experiments/optim/poet_lie_orth.yaml`: add `single_step_native: true` under `optim.poet` (and it implies the single-step fast path). Leave `single_step_fast` as the documented fallback.

```bash
git add configs/experiments/optim/poet_lie_orth.yaml
git commit -m "feat(poet): enable single_step_native by default for poet_lie_orth"
```

---

## Self-Review

- **Spec coverage:** gather-free forward + closed-form backward = Task 1 (`NativeSingleStepFunction`); the layer = Task 1 (`SingleStepPOETLinear`, overrides only `forward`); walk wiring = Task 3; arg/CLI/validate = Tasks 4-5; A/B + default = Task 6. Storage stays natural and the merge is inherited unchanged (the verified batched/replicated merge) — no merge task, by design. Head-aligned stays on its already-gather-free path (Task 3 Step 5).
- **Placeholder scan:** none — every code step is complete; the GPU A/B is explicitly the user's to run with exact commands and the loss-overlap (not 0.0) acceptance.
- **Type consistency:** `NativeSingleStepFunction.apply(...)` arg order is identical in the test (`_native`), the layer (`SingleStepPOETLinear.forward`), and the Function signature: `(x, oft_R_in, oft_R_out, weight, bias, perm_in, perm_in_inv, perm_out, perm_out_inv, rows_in, cols_in, rows_out, cols_out, block_size_in, block_size_out)` — 15 inputs → 15 backward returns (3 grads + 12 None). `_conj(M,p)=M[p][:,p]`; `_blockdiag_skew_vec` reused from `single_step` (factor 2 default). `single_step_native` is the name across config/CLI(`--poet-single-step-native`)/arg(`poet_single_step_native`)/walk.
- **Correctness basis:** the exact forward+backward were verified bit-against the real chain to ~1e-14 (fp64) in [/tmp/poet_native_frame_selfcheck.py](/tmp/poet_native_frame_selfcheck.py); identity-perm forward is 0.0. Not bit-identical to `single_step_fast` on GPU (different reduction order) → loss-overlap acceptance, same as the original fast-vs-chain A/B.
- **Old code intact:** `POETLinear`, `SingleStepPOETFunction`, `chain_layer_x_fast_decoupled`, and the merge are untouched; `SingleStepPOETLinear` subclasses `POETLinear` and overrides only `forward` (+ a one-line `__init__` cayley guard), so the verified merge machinery (`_run_merge`/`_build_R_batched`/`_fold_with_R`) applies to it unchanged (`isinstance(pl, POETLinear)` holds; it stores natural `W`, which the inherited merge reads identically to the chain — confirmed: chain effective base `(x[perm_in_inv]@Wᵀ)[perm_out]` equals native `x@W_effᵀ` with `W_eff=W[perm_out][:,perm_in]`).
- **Review fixes applied:** (1) `__init__` fail-fast for non-cayley (the native backward is Cayley-specific, no fallback); (2) launch-count caveat in Task 6 (relabels are more/smaller launches — cache `W_eff` if they offset the gather win). Precedence note: if both `single_step_fast` and `single_step_native` are set, the native class wins for standard layers (its forward ignores `single_step_fast`); harmless. End-to-end `autograd.Function` (15→15 arity, leaf-grad routing) and forward/backward-vs-chain (~1e-14) verified before handoff.
