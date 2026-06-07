# POET Single-Step Fast Path — Plan 2: HeadAlignedPOETLinear (block-local backward)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **PREREQUISITE:** Implement `2026-06-07-poet-single-step-fast-backward.md` (Plan 1) FIRST. This plan reuses Plan 1's `single_step.py`, the `single_step_fast` flag plumbing (already set on head-aligned layers by `replace_linears_with_poet`), the `--poet-single-step-fast` CLI flag, and the `merge_period==1`/cayley validation. Do not re-do that wiring here.

**Goal:** Extend the single-step (R=Identity) fast path to `HeadAlignedPOETLinear` (the q/k/v/proj attention rotations), with the head-structured side's `oft_R` gradient computed **block-locally** — batched `head_dim×head_dim` matmuls over heads — so the head side never forms a full `[d,d]` matrix.

**Architecture:** A second custom Function `HeadAlignedSingleStepFunction`. Forward is a bare GEMM `y = x@Wᵀ + b` (head-aligned layers are permutation-free, so no gathers at all). Backward **saves only `x`** and (chain's right-multiply orientation, factor 2; `Gv = grad_y` since no perms) computes `grad_x = grad_y@W` and `A = xᵀ@Gv`, then: the **head side** (block-diagonal, `head_dim` blocks) gets its skew gradient from batched per-head matmuls producing only the diagonal blocks of `M_out = W@A (+bias)` (head_side="out") or `M_in = A@W` (head_side="in"); the **residual side** (dense single block in every deployed config) reuses Plan 1's `_blockdiag_skew_vec` on the one full `[d,d]` matrix. The `head_side="out"` bias term is added per-head as `outer(bias_k, gsumₖ)` with `gsum = Σₜ Gvₜ`. **Verified bit-exact (≤3e-14) vs autograd through the real `chain_noperm`+`cayley_batch`, both `head_side`, bias∈{F,T}** ([/tmp/poet_plan_selfcheck3.py](/tmp/poet_plan_selfcheck3.py)). (NB: a `G@Wᵀ` form is WRONG — wrong skew sign + drops bias.)

**Why block-local:** for q/k/v (out=H, `head_dim`, `nb=num_heads`), forming the full `[H,H] = G Wᵀ` then slicing is `O(H²·in)`; the diagonal blocks alone are `O(H·head_dim·in)` — a `num_heads`× reduction in the amortized per-step gradient cost on the head side. Negligible at 60m, but load-bearing at the Kimi-1T target where `H` is large.

**Tech Stack:** PyTorch custom autograd Function, vendored `poet_torch`, pytest (CPU).

**Math / convention reference:**
- Head-aligned forward being replaced: [head_aligned_layer.py:57-78 `chain_noperm`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/head_aligned_layer.py#L57-L78) (no permutations) and [`HeadAlignedPOETLinear.forward`, lines 230-250](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/head_aligned_layer.py#L230-L250).
- Side→block mapping: [head_aligned_layer.py:182-189](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/head_aligned_layer.py#L182-L189). `head_side="out"` ⇒ `block_size_out=head_dim` (head side), `block_size_in=resid` (dense). `head_side="in"` ⇒ `block_size_in=head_dim`, `block_size_out=resid`.
- `head_side` and per-side block sizes live on the layer (`self.head_side`, `self.block_size_in/out`, `self.rows_in/cols_in/rows_out/cols_out`).
- Closed form (identity perms, chain's right-multiply orientation, factor 2): `A = xᵀ@grad_y`, `gsum = Σₜ grad_yₜ`; head_side="out" → head `M_out = W@A + outer(bias,gsum)` (blocks over OUT), residual `M_in = A@W`; head_side="in" → head `M_in = A@W` (blocks over IN), residual `M_out = W@A + outer(bias,gsum)`. Verified in [/tmp/poet_plan_selfcheck3.py](/tmp/poet_plan_selfcheck3.py).

---

## File Structure

- **Modify** `third_party/poet_torch/single_step.py` — add `_head_out_skew_vec`, `_head_in_skew_vec` (block-local helpers) and `HeadAlignedSingleStepFunction`.
- **Modify** `third_party/poet_torch/__init__.py` — export `HeadAlignedSingleStepFunction`.
- **Modify** `third_party/poet_torch/head_aligned_layer.py` — `HeadAlignedPOETLinear.forward` dispatches to the fast Function when `self.single_step_fast` is set.
- **Modify** `tests/unit/test_single_step_fast.py` — add head-aligned equivalence tests (both `head_side`).

---

## Task 1: Block-local helpers + the head-aligned Function

**Files:**
- Modify: `third_party/poet_torch/single_step.py`
- Modify: `tests/unit/test_single_step_fast.py`

- [ ] **Step 1: Write the failing test** (fast Function vs the real `chain_noperm` at `oft_R=0`, both head sides). Append to `tests/unit/test_single_step_fast.py`:

```python
from poet_torch import HeadAlignedPOETLinear, HeadAlignedSingleStepFunction
from poet_torch.head_aligned_layer import chain_noperm


def _ha_reference(pl, x):
    Qin = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    Qout = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    Rin, Rout = cayley_batch(Qin), cayley_batch(Qout)
    return chain_noperm(x, Rin, pl.weight, pl.bias, Rout, pl.block_size_in, pl.block_size_out)


def _ha_fast(pl, x):
    return HeadAlignedSingleStepFunction.apply(
        x, pl.oft_R_in, pl.oft_R_out, pl.weight, pl.bias,
        pl.rows_in, pl.cols_in, pl.rows_out, pl.cols_out,
        pl.block_size_in, pl.block_size_out, pl.head_side,
    )


@pytest.mark.parametrize("head_side,in_f,out_f", [("out", 12, 8), ("in", 8, 12)])
def test_head_aligned_fast_matches_chain(head_side, in_f, out_f):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    pl = HeadAlignedPOETLinear(
        in_features=in_f, out_features=out_f,
        head_side=head_side, head_dim=4, resid_block_count=1, bias=False,
    )
    with torch.no_grad():
        pl.weight.normal_()
    assert torch.count_nonzero(pl.oft_R_in) == 0 and torch.count_nonzero(pl.oft_R_out) == 0

    x = torch.randn(5, in_f)
    gy = torch.randn(5, out_f)

    # forward
    assert torch.allclose(_ha_reference(pl, x), _ha_fast(pl, x), atol=1e-10)

    # grads
    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xr = x.clone().requires_grad_(True)
    (_ha_reference(pl, xr) * gy).sum().backward()
    gin_ref, gout_ref, gx_ref = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xr.grad.clone()

    pl.oft_R_in.grad = pl.oft_R_out.grad = None
    xf = x.clone().requires_grad_(True)
    (_ha_fast(pl, xf) * gy).sum().backward()
    gin_f, gout_f, gx_f = pl.oft_R_in.grad.clone(), pl.oft_R_out.grad.clone(), xf.grad.clone()

    assert torch.allclose(gin_ref, gin_f, atol=1e-9), (gin_ref - gin_f).abs().max()
    assert torch.allclose(gout_ref, gout_f, atol=1e-9), (gout_ref - gout_f).abs().max()
    assert torch.allclose(gx_ref, gx_f, atol=1e-9), (gx_ref - gx_f).abs().max()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest "tests/unit/test_single_step_fast.py::test_head_aligned_fast_matches_chain" -x -q`
Expected: FAIL — `ImportError: cannot import name 'HeadAlignedSingleStepFunction'`.

- [ ] **Step 3: Implement helpers + Function.** Append to `third_party/poet_torch/single_step.py`:

```python
def _head_out_skew_vec(A, weight, bias, gsum, head_dim, rows, cols, factor=2.0):
    """Diagonal blocks of M_out = W@A (+ outer(bias,gsum)) over OUT blocks, block-local.

    A = x^T @ Gv  [in,out]; block k = W[k] @ A[:,k] with W[k] [head_dim, in] and
    A[:,k] [in, head_dim]. The bias rank-1 term is added per head as
    outer(bias_k, gsum_k). Never forms the full [out,out]. Returns (num_heads, n_elems).
    """
    out_f, in_f = weight.shape
    nb = out_f // head_dim
    Wb = weight.reshape(nb, head_dim, in_f)                        # (nb, head_dim, in)
    Acol = A.reshape(in_f, nb, head_dim).permute(1, 0, 2)          # (nb, in, head_dim)
    blocks = torch.bmm(Wb, Acol)                                   # (nb, hd, hd) = W_k @ A_:,k
    if bias is not None:
        bb = bias.reshape(nb, head_dim)
        gb = gsum.reshape(nb, head_dim)
        blocks = blocks + bb[:, :, None] * gb[:, None, :]          # + outer(bias_k, gsum_k)
    skew = blocks - blocks.transpose(-1, -2)
    return factor * skew[:, rows.long(), cols.long()]


def _head_in_skew_vec(A, weight, head_dim, rows, cols, factor=2.0):
    """Diagonal blocks of M_in = A @ W over IN blocks (heads), block-local.

    A = x^T @ Gv  [in, out]; block k = A[k] @ W[:,k] with A[k] [head_dim, out] and
    W[:,k] [out, head_dim]. Never forms the full [in,in].  Returns (num_heads, n_elems).
    """
    out_f, in_f = weight.shape
    nb = in_f // head_dim
    Ab = A.reshape(nb, head_dim, out_f)                          # (nb, head_dim, out)
    Wb = weight.reshape(out_f, nb, head_dim).permute(1, 0, 2)    # (nb, out, head_dim)
    blocks = torch.bmm(Ab, Wb)                                   # (nb, head_dim, head_dim)
    skew = blocks - blocks.transpose(-1, -2)
    return factor * skew[:, rows.long(), cols.long()]


class HeadAlignedSingleStepFunction(torch.autograd.Function):
    """Single-step (R=I) fast path for HeadAlignedPOETLinear (permutation-free).

    Forward is a bare GEMM (no gathers). Backward saves ONLY x and (chain's
    right-multiply orientation, factor 2; Gv = grad_y since no perms) builds
    A = x^T@Gv once: head side -> block-local skew grad (batched per-head matmul,
    no full [d,d]); residual (dense, single block) side -> _blockdiag_skew_vec on
    the one full matrix. M_in = A@W, M_out = W@A + outer(bias, Gv.sum(0)).
    """
    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, weight, bias,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out, head_side):
        y = x @ weight.t()
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x, weight, bias, rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        ctx.head_side = head_side
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, weight, bias, rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out, head_side = ctx.block_size_in, ctx.block_size_out, ctx.head_side
        out_f, in_f = weight.shape

        grad_x = grad_y @ weight
        Gv2 = grad_y.reshape(-1, out_f)
        A = x.reshape(-1, in_f).t() @ Gv2                # (in, out) = x^T @ Gv
        gsum = Gv2.sum(0)

        if head_side == "out":
            # head side = OUT (block-diagonal, head_dim=bs_out); residual = IN (dense)
            grad_oft_R_out = _head_out_skew_vec(A, weight, bias, gsum, bs_out, rows_out, cols_out)
            grad_oft_R_in = _blockdiag_skew_vec(A @ weight, bs_in, rows_in, cols_in)
        else:  # head_side == "in"
            # head side = IN (block-diagonal, head_dim=bs_in); residual = OUT (dense)
            grad_oft_R_in = _head_in_skew_vec(A, weight, bs_in, rows_in, cols_in)
            M_out = weight @ A
            if bias is not None:
                M_out = M_out + torch.outer(bias, gsum)
            grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out)

        grad_oft_R_in = grad_oft_R_in.to(weight.dtype)
        grad_oft_R_out = grad_oft_R_out.to(weight.dtype)
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None, None, None, None)
```

- [ ] **Step 4: Export it.** Add to `third_party/poet_torch/__init__.py`:

```python
from .single_step import HeadAlignedSingleStepFunction as HeadAlignedSingleStepFunction
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest "tests/unit/test_single_step_fast.py::test_head_aligned_fast_matches_chain" -x -q`
Expected: PASS (both `head_side`).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/single_step.py third_party/poet_torch/__init__.py tests/unit/test_single_step_fast.py
git commit -m "feat(poet): block-local closed-form backward for head-aligned single-step"
```

---

## Task 2: Dispatch from `HeadAlignedPOETLinear.forward`

**Files:**
- Modify: `third_party/poet_torch/head_aligned_layer.py` (`forward`, ~line 230)
- Modify: `tests/unit/test_single_step_fast.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/unit/test_single_step_fast.py`:

```python
@pytest.mark.parametrize("head_side,in_f,out_f", [("out", 12, 8), ("in", 8, 12)])
def test_head_aligned_layer_uses_fast_path_when_flagged(head_side, in_f, out_f):
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)
    pl = HeadAlignedPOETLinear(
        in_features=in_f, out_features=out_f,
        head_side=head_side, head_dim=4, resid_block_count=1, bias=False,
    )
    with torch.no_grad():
        pl.weight.normal_()
    pl.single_step_fast = True
    x = torch.randn(3, in_f, requires_grad=True)
    y = pl(x)                                   # would call Triton cayley on the slow path -> CPU error
    assert torch.allclose(y, _ha_reference(pl, x), atol=1e-10)
    (y * torch.randn(3, out_f)).sum().backward()
    assert pl.oft_R_in.grad is not None and pl.oft_R_out.grad is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest "tests/unit/test_single_step_fast.py::test_head_aligned_layer_uses_fast_path_when_flagged" -x -q`
Expected: FAIL — `forward` ignores `single_step_fast`, routes to the compiled/eager chain which calls `torch.ops.poet.cayley` (Triton) and errors on CPU.

- [ ] **Step 3: Dispatch.** In `third_party/poet_torch/head_aligned_layer.py`, at the top of `HeadAlignedPOETLinear.forward` (immediately after `def forward(self, x):`, ~line 230), add:

```python
        if getattr(self, "single_step_fast", False):
            from .single_step import HeadAlignedSingleStepFunction
            return HeadAlignedSingleStepFunction.apply(
                x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
                self.block_size_in, self.block_size_out, self.head_side,
            )
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_fast.py -x -q`
Expected: PASS (all single-step tests, standard + head-aligned).

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/head_aligned_layer.py tests/unit/test_single_step_fast.py
git commit -m "feat(poet): dispatch HeadAlignedPOETLinear.forward to single-step fast path"
```

---

## Task 3: Full CPU regression + GPU parity handoff

**Files:** (verification only)

- [ ] **Step 1: Run the POET CPU suite**

Run: `PYTHONPATH=third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_single_step_fast.py tests/unit/test_poet_layers.py tests/unit/test_megatron_args.py -q`
Expected: all PASS.

- [ ] **Step 2: Hand the GPU parity smoke to the user** (do NOT launch). With both plans in, the full head-aligned config (`poet_lie_orth`, attention + MLP) runs entirely on the fast path:

```bash
codexlog ss2_chain bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  training.train_iters=50
codexlog ss2_fast bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 optim.poet.lie_ortho_c=8 optim.poet.lie_ortho_distributed=true \
  optim.poet.single_step_fast=true training.train_iters=50
```

Acceptance: per-step loss within optimizer noise of the chain run; `elapsed time per iteration (ms)` lower than both the chain baseline and the Plan-1-only run.

- [ ] **Step 3: Commit any doc/CHANGELOG updates**

```bash
git add -A && git commit -m "chore(poet): head-aligned single-step fast path Plan 2 complete"
```

---

## Self-Review

- **Spec coverage:** block-local helpers + `HeadAlignedSingleStepFunction` (T1) → layer dispatch (T2) → regression + GPU handoff (T3). Both `head_side` values covered (parametrized tests). Residual (dense) side reuses Plan 1's `_blockdiag_skew_vec`; head side uses the new block-local helpers. The `single_step_fast` flag, CLI, and `merge_period==1`/cayley validation come from Plan 1 (prerequisite) — not duplicated.
- **Type consistency:** `HeadAlignedSingleStepFunction.apply(...)` arg order is identical in `_ha_fast` (test), the layer dispatch (T2), and the Function signature (T1): `(x, oft_R_in, oft_R_out, weight, bias, rows_in, cols_in, rows_out, cols_out, block_size_in, block_size_out, head_side)`. `_head_out_skew_vec`/`_head_in_skew_vec` return `(num_heads, n_elems)` matching `oft_R`; `_blockdiag_skew_vec` (from Plan 1) handles the dense residual side. `head_side` is the layer attribute (`"out"`/`"in"`).
- **Placeholder scan:** none — all code blocks are complete; line-number anchors are approximate ("~line N") but the surrounding text uniquely identifies the insertion point.
- **Cross-plan check:** `replace_linears_with_poet` (Plan 1, Task 3) already sets `pl.single_step_fast` in the head-aligned branch, so head-aligned layers carry the flag before this plan runs; T2's dispatch is the only thing needed to activate it.
