# POET Head-Aligned Attention Rotation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give POET's attention projections a per-head block-diagonal rotation on their head-structured side (head_dim blocks, fixed identity permutation, no cross-head mixing) while the residual side stays a normal POET rotation; opt-in, attention-only.

**Architecture:** A new `HeadAlignedPOETLinear` subclass of `POETLinear` (in `third_party/poet_torch/`) with an asymmetric per-side block spec and identity head-side Ψ; an attention-name routing branch in `replace_linears_with_poet`; a per-block RMS fix in `LieAlgebraMomentum`; opt-in CLI flag + experiment config. The optimizer and merge machinery pick it up unchanged via inheritance (`oft_R_in`/`oft_R_out` names; `isinstance(_, POETLinear)`).

**Tech Stack:** PyTorch, Megatron-Core, vendored `poet_torch`, pytest, OmegaConf/Hydra.

**Design spec:** [2026-06-04-poet-head-aligned-attention-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-04-poet-head-aligned-attention-design.md)

**CPU test interpreter (use everywhere below):** `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`

---

## File Structure

- **Create** `third_party/poet_torch/head_aligned_layer.py` — `HeadAlignedPOETLinear` (constructor + merge override). One responsibility: the head-aligned layer.
- **Modify** `third_party/poet_torch/__init__.py` — export `HeadAlignedPOETLinear`.
- **Modify** `src/optim/poet_lie_momentum.py:162-169` — per-block RMS.
- **Modify** `src/optim/poet_layers.py` — `replace_linears_with_poet` head-aligned routing + a `_init_poet_weight` helper (extracted to stay DRY).
- **Modify** `src/patches/poet_apply_to_model.py:61-78` — read the flag + `head_dim` from args, thread through.
- **Modify** `launchers/pretrain_gpt_slm.py` — `--poet-head-aligned-attn`, `--poet-no-head-resid-perm`.
- **Modify** `src/utils/megatron_args.py` — emit the new flags; guard (requires unfuse).
- **Create** `configs/experiments/optim/poet_lie_head.yaml`, `docs/experiments/poet_lie_head.md`, `scripts/train_poet_lie_head.sh`.
- **Create** `tests/unit/test_head_aligned_poet.py`; **extend** `test_poet_lie_momentum.py`, `test_poet_layers.py`, `test_pretrain_gpt_slm.py`, `test_megatron_args.py`.

---

## Task 1: Per-block RMS in `LieAlgebraMomentum`

**Files:**
- Modify: `src/optim/poet_lie_momentum.py:162-169`
- Test: `tests/unit/test_poet_lie_momentum.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_poet_lie_momentum.py`:

```python
def test_rms_is_per_block_consistent():
    """With RMS on, each block's applied update has Frobenius norm
    rms_c*sqrt(block_size) regardless of that block's gradient magnitude."""
    import torch
    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    torch.manual_seed(0)
    bsz = 8
    n_elems = bsz * (bsz - 1) // 2  # 28
    n_blocks = 4
    p = torch.nn.Parameter(torch.zeros(n_blocks, n_elems, dtype=torch.float64))
    # Block 0 huge gradient, block 1 tiny — old global-alpha would scale them unequally.
    g = torch.randn(n_blocks, n_elems, dtype=torch.float64)
    g[0] *= 100.0
    g[1] *= 0.01
    p.grad = g.clone()

    rms_c = 0.2
    opt = LieAlgebraMomentum(
        [{"params": [p], "use_skew": True, "side": "out", "lr": 1.0}],
        b1=0.0, b2=0.0, eps=1e-12, v_mode="elementwise", rms=True, rms_c=rms_c,
    )
    opt.step()

    target = rms_c * (bsz ** 0.5)  # per-block Frobenius of the (lr=1) update
    per_block = torch.linalg.norm(p.detach(), dim=1)
    assert torch.allclose(per_block, torch.full((n_blocks,), target, dtype=torch.float64), atol=1e-6), per_block


def test_rms_block_count_1_unchanged():
    """At n_blocks==1 the per-block RMS equals the old global formula."""
    import torch
    from src.optim.poet_lie_momentum import LieAlgebraMomentum

    torch.manual_seed(1)
    bsz = 8
    n_elems = bsz * (bsz - 1) // 2
    p = torch.nn.Parameter(torch.zeros(1, n_elems, dtype=torch.float64))
    p.grad = torch.randn(1, n_elems, dtype=torch.float64)
    opt = LieAlgebraMomentum(
        [{"params": [p], "use_skew": True, "side": "out", "lr": 1.0}],
        b1=0.0, b2=0.0, eps=1e-12, v_mode="elementwise", rms=True, rms_c=0.2,
    )
    opt.step()
    assert torch.isclose(torch.linalg.norm(p.detach()), torch.tensor(0.2 * (bsz ** 0.5), dtype=torch.float64), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_momentum.py::test_rms_is_per_block_consistent -v`
Expected: FAIL (per-block norms unequal — block 0 ≫ block 1 under the global α).

- [ ] **Step 3: Apply the per-block RMS edit**

In `src/optim/poet_lie_momentum.py`, replace the RMS block (currently lines 162-169):

```python
                    if self.rms:
                        # Stage 2 (W-free), PER BLOCK: normalize each block's
                        # generator so its per-plane angle is dimension-consistent.
                        # dim_const = sqrt(block_size); block_norm reduces over the
                        # n_elems axis only -> alpha is (n_blocks, 1). Identical to
                        # the old global formula when n_blocks == 1.
                        bsz = block_size_from_nelems(A.shape[1])
                        dim_const = bsz ** 0.5
                        block_norm = torch.linalg.norm(A, dim=1, keepdim=True)
                        alpha = self.rms_c * dim_const / (block_norm + eps)
                        A = A * alpha
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_momentum.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_lie_momentum.py tests/unit/test_poet_lie_momentum.py
git commit -F - <<'EOF'
feat(poet): per-block RMS scaling in LieAlgebraMomentum
EOF
```

---

## Task 2: `HeadAlignedPOETLinear` constructor

**Files:**
- Create: `third_party/poet_torch/head_aligned_layer.py`
- Modify: `third_party/poet_torch/__init__.py`
- Test: `tests/unit/test_head_aligned_poet.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_head_aligned_poet.py`:

```python
"""HeadAlignedPOETLinear: CPU constructor/merge/geometry; GPU forward parity."""
from __future__ import annotations

import pytest
import torch


def test_constructor_out_head_side_shapes():
    from poet_torch import HeadAlignedPOETLinear

    layer = HeadAlignedPOETLinear(
        in_features=512, out_features=512, head_side="out", head_dim=64,
        resid_block_count=1, parameterization="exp", dtype=torch.float64,
    )
    assert layer.block_size_out == 64 and layer.block_size_in == 512
    assert layer.head_count == 8
    assert layer.oft_R_out.shape == (8, 64 * 63 // 2)
    assert layer.oft_R_in.shape == (1, 512 * 511 // 2)
    assert layer.oft_R_in.requires_grad and layer.oft_R_out.requires_grad
    assert layer.weight.requires_grad is False
    # Head side (out) has identity permutation; residual (in) is one block -> also identity here.
    assert torch.equal(layer.perm_out, torch.arange(512, dtype=torch.int32))


def test_constructor_in_head_side_for_output_proj():
    from poet_torch import HeadAlignedPOETLinear

    layer = HeadAlignedPOETLinear(
        in_features=512, out_features=512, head_side="in", head_dim=64,
        resid_block_count=4, parameterization="exp", dtype=torch.float64,
    )
    assert layer.block_size_in == 64 and layer.block_size_out == 128  # 512/4
    assert layer.head_count == 8
    assert torch.equal(layer.perm_in, torch.arange(512, dtype=torch.int32))  # head side identity


def test_constructor_validation():
    from poet_torch import HeadAlignedPOETLinear

    with pytest.raises(ValueError, match="head_side"):
        HeadAlignedPOETLinear(in_features=512, out_features=512, head_side="bogus",
                              head_dim=64, resid_block_count=1)
    with pytest.raises(ValueError, match="exactly one of resid"):
        HeadAlignedPOETLinear(in_features=512, out_features=512, head_side="out", head_dim=64)
    with pytest.raises(ValueError, match="head_dim 48 doesn't divide"):
        HeadAlignedPOETLinear(in_features=512, out_features=512, head_side="out",
                              head_dim=48, resid_block_count=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poet.py::test_constructor_out_head_side_shapes -v`
Expected: FAIL with `ImportError: cannot import name 'HeadAlignedPOETLinear'`.

- [ ] **Step 3: Create the class (constructor + the merge override from Task 3 stubbed)**

Create `third_party/poet_torch/head_aligned_layer.py`:

```python
"""HeadAlignedPOETLinear: a POETLinear whose head-structured side is rotated
per attention head.

One side (the "head side") uses block_size = head_dim with a FIXED identity
permutation (block j is head j; Psi is NEVER resampled), so the rotation is
block-diagonal per head with no cross-head mixing and no permutation. The other
("residual") side is an ordinary POET rotation: block size from
resid_block_size / resid_block_count, permutation resampled at merge unless
resid_permute=False. BOTH sides train.

head_side="out": query/key/value projections (rows = heads).
head_side="in" : attention output projection (cols = heads).

Subclasses POETLinear to reuse _build_R / _merge_R / forward / the fused kernels;
only the constructor (asymmetric per-side block spec + identity head Psi) and
merge_then_reinitialize (resample the residual side only) differ.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .poet_layer import POETLinear, block_diag_lr_matmul_decoupled


class HeadAlignedPOETLinear(POETLinear):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        head_side,
        head_dim,
        resid_block_size=None,
        resid_block_count=None,
        resid_permute=True,
        bias=False,
        device=None,
        dtype=None,
        parameterization="cayley",
        mem_efficient_mode=None,
    ):
        nn.Module.__init__(self)
        if head_side not in ("in", "out"):
            raise ValueError(f"head_side must be 'in' or 'out', got {head_side!r}")
        if (resid_block_size is None) == (resid_block_count is None):
            raise ValueError("exactly one of resid_block_size or resid_block_count must be set")
        if parameterization not in ("cayley", "exp"):
            raise ValueError(f"parameterization must be 'cayley' or 'exp', got {parameterization!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.head_side = head_side
        self.head_dim = head_dim
        self.resid_permute = bool(resid_permute)

        head_features = out_features if head_side == "out" else in_features
        resid_features = in_features if head_side == "out" else out_features
        if head_features % head_dim != 0:
            raise ValueError(f"head_dim {head_dim} doesn't divide the head-side dim {head_features}")
        if resid_block_count is not None:
            if resid_features % resid_block_count != 0:
                raise ValueError(
                    f"resid_block_count {resid_block_count} doesn't divide residual dim {resid_features}"
                )
            resid_bs = resid_features // resid_block_count
        else:
            if resid_features % resid_block_size != 0:
                raise ValueError(
                    f"resid_block_size {resid_block_size} doesn't divide residual dim {resid_features}"
                )
            resid_bs = resid_block_size

        if head_side == "out":
            block_size_out, block_size_in = head_dim, resid_bs
        else:
            block_size_in, block_size_out = head_dim, resid_bs
        self.block_size_in = block_size_in
        self.block_size_out = block_size_out
        self.block_size = block_size_in  # back-compat (merge/"is-active" guards)
        self.head_count = head_features // head_dim

        if mem_efficient_mode is None:
            mem_efficient_mode = (parameterization == "exp") or os.environ.get("POET_MEM_EFFICIENT") == "1"
        self.mem_efficient_mode = mem_efficient_mode
        self.parameterization = parameterization

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        r_in = in_features // block_size_in
        r_out = out_features // block_size_out
        n_elems_in = block_size_in * (block_size_in - 1) // 2
        n_elems_out = block_size_out * (block_size_out - 1) // 2
        self.oft_R_in = nn.Parameter(torch.zeros((r_in, n_elems_in), device=device, dtype=dtype))
        self.oft_R_out = nn.Parameter(torch.zeros((r_out, n_elems_out), device=device, dtype=dtype))
        self.r_in, self.r_out = r_in, r_out

        rows_in, cols_in = torch.triu_indices(block_size_in, block_size_in, 1, device=device)
        self.register_buffer("rows_in", rows_in.to(torch.int32))
        self.register_buffer("cols_in", cols_in.to(torch.int32))
        rows_out, cols_out = torch.triu_indices(block_size_out, block_size_out, 1, device=device)
        self.register_buffer("rows_out", rows_out.to(torch.int32))
        self.register_buffer("cols_out", cols_out.to(torch.int32))

        # Head side: identity Psi (never resampled). Residual side: random Psi
        # unless resid_permute=False (then identity, never resampled).
        out_identity = (head_side == "out") or not self.resid_permute
        in_identity = (head_side == "in") or not self.resid_permute
        perm_out = self._make_perm(out_features, out_identity, device)
        perm_in = self._make_perm(in_features, in_identity, device)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    @staticmethod
    def _make_perm(n, identity, device):
        if identity:
            return torch.arange(n, device=device, dtype=torch.int32)
        return torch.randperm(n, device=device).to(torch.int32)

    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        # Filled in Task 3.
        raise NotImplementedError
```

Then in `third_party/poet_torch/__init__.py` add the export (match the existing import/`__all__` style — append `HeadAlignedPOETLinear`):

```python
from .head_aligned_layer import HeadAlignedPOETLinear  # noqa: F401
```
(If `__all__` is defined in that file, add `"HeadAlignedPOETLinear"` to it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poet.py -v`
Expected: PASS for the three constructor tests.

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/head_aligned_layer.py third_party/poet_torch/__init__.py tests/unit/test_head_aligned_poet.py
git commit -F - <<'EOF'
feat(poet): add HeadAlignedPOETLinear constructor
EOF
```

---

## Task 3: `merge_then_reinitialize` (residual-side-only resample)

**Files:**
- Modify: `third_party/poet_torch/head_aligned_layer.py`
- Test: `tests/unit/test_head_aligned_poet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_head_aligned_poet.py`:

```python
def test_merge_matches_stock_poetlinear_when_state_identical():
    """HeadAligned merge math == stock POETLinear(block_count=head_count) merge
    when both have identical state and reinit_perm=False (exp param, CPU)."""
    from poet_torch import HeadAlignedPOETLinear, POETLinear

    torch.manual_seed(0)
    a = HeadAlignedPOETLinear(
        in_features=512, out_features=512, head_side="out", head_dim=64,
        resid_block_count=8, parameterization="exp", dtype=torch.float64,
    )  # bs_out=64, bs_in=64 == stock block_count=8
    b = POETLinear(in_features=512, out_features=512, block_count=8,
                   parameterization="exp", dtype=torch.float64)
    with torch.no_grad():
        b.weight.copy_(torch.randn_like(b.weight))
        a.weight.copy_(b.weight)
        for name in ("oft_R_in", "oft_R_out"):
            new = torch.randn_like(getattr(a, name)) * 1e-2
            getattr(a, name).copy_(new)
            getattr(b, name).copy_(new)
        for buf in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(b, buf).copy_(getattr(a, buf))
    a.merge_then_reinitialize(reinit_perm=False)
    b.merge_then_reinitialize(reinit_perm=False)
    assert torch.allclose(a.weight, b.weight, atol=1e-10)
    assert torch.count_nonzero(a.oft_R_in) == 0 and torch.count_nonzero(a.oft_R_out) == 0


def test_merge_resamples_only_residual_side():
    """reinit_perm=True resamples the residual perm; the head perm stays identity."""
    from poet_torch import HeadAlignedPOETLinear

    torch.manual_seed(1)
    layer = HeadAlignedPOETLinear(
        in_features=512, out_features=512, head_side="out", head_dim=64,
        resid_block_count=8, parameterization="exp", dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        layer.oft_R_in.normal_(std=1e-2)
        layer.oft_R_out.normal_(std=1e-2)
    perm_in_before = layer.perm_in.clone()
    layer.merge_then_reinitialize(reinit_perm=True)
    # Head side (out) Psi stays identity; residual side (in) Psi changes.
    assert torch.equal(layer.perm_out, torch.arange(512, dtype=torch.int32))
    assert not torch.equal(layer.perm_in, perm_in_before)


def test_merge_resid_permute_false_never_resamples():
    from poet_torch import HeadAlignedPOETLinear

    torch.manual_seed(2)
    layer = HeadAlignedPOETLinear(
        in_features=512, out_features=512, head_side="out", head_dim=64,
        resid_block_count=8, resid_permute=False, parameterization="exp", dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        layer.oft_R_in.normal_(std=1e-2)
    pin, pout = layer.perm_in.clone(), layer.perm_out.clone()
    layer.merge_then_reinitialize(reinit_perm=True)
    assert torch.equal(layer.perm_in, pin) and torch.equal(layer.perm_out, pout)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poet.py -k merge -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement the merge override**

In `third_party/poet_torch/head_aligned_layer.py`, replace the stubbed `merge_then_reinitialize` body with:

```python
    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        R_out, R_in = self._merge_R()
        W = self.weight.detach().clone()
        tmp = block_diag_lr_matmul_decoupled(R_in, W.t(), R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        # Resample ONLY the residual side, and only when reinit_perm & resid_permute.
        # The head side keeps its identity Psi forever. When a side does not
        # resample, re-permute back into the CURRENT layout (stock fold-only path).
        out_resamples = reinit_perm and self.resid_permute and (self.head_side == "in")
        in_resamples = reinit_perm and self.resid_permute and (self.head_side == "out")
        device = self.weight.device

        if out_resamples:
            new_perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
            new_perm_out_inv = torch.argsort(new_perm_out).to(torch.int32)
        else:
            new_perm_out, new_perm_out_inv = self.perm_out, self.perm_out_inv
        if in_resamples:
            new_perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
            new_perm_in_inv = torch.argsort(new_perm_in).to(torch.int32)
        else:
            new_perm_in, new_perm_in_inv = self.perm_in, self.perm_in_inv

        expected = expected.index_select(0, new_perm_out_inv).index_select(1, new_perm_in_inv)
        self.weight.detach().copy_(expected)
        self.perm_out.copy_(new_perm_out)
        self.perm_out_inv.copy_(new_perm_out_inv)
        self.perm_in.copy_(new_perm_in)
        self.perm_in_inv.copy_(new_perm_in_inv)
        self.oft_R_in.zero_()
        self.oft_R_out.zero_()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poet.py -k merge -v`
Expected: PASS (3 merge tests).

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/head_aligned_layer.py tests/unit/test_head_aligned_poet.py
git commit -F - <<'EOF'
feat(poet): HeadAlignedPOETLinear residual-only merge resample
EOF
```

---

## Task 4: Geometry — spectrum preservation + no cross-head mixing

**Files:**
- Test: `tests/unit/test_head_aligned_poet.py`

- [ ] **Step 1: Write the failing/again-passing geometry tests**

Append to `tests/unit/test_head_aligned_poet.py`:

```python
def test_merge_preserves_singular_values():
    """Folding orthogonal (exp) rotations + permutation preserves W's spectrum."""
    from poet_torch import HeadAlignedPOETLinear

    torch.manual_seed(3)
    layer = HeadAlignedPOETLinear(
        in_features=512, out_features=512, head_side="out", head_dim=64,
        resid_block_count=1, parameterization="exp", dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        sv_before = torch.linalg.svdvals(layer.weight.double())
        layer.oft_R_in.normal_(std=1e-2)
        layer.oft_R_out.normal_(std=1e-2)
    layer.merge_then_reinitialize(reinit_perm=False)
    sv_after = torch.linalg.svdvals(layer.weight.double())
    assert torch.allclose(sv_before, sv_after, atol=1e-8)


def test_no_cross_head_mixing():
    """Perturbing head j's out-side block changes only head j's rows of the
    folded weight (residual side held at identity)."""
    from poet_torch import HeadAlignedPOETLinear

    torch.manual_seed(4)
    W0 = torch.randn(512, 512, dtype=torch.float64)

    def merged_weight(perturb_block=None):
        layer = HeadAlignedPOETLinear(
            in_features=512, out_features=512, head_side="out", head_dim=64,
            resid_block_count=1, resid_permute=False, parameterization="exp", dtype=torch.float64,
        )
        with torch.no_grad():
            layer.weight.copy_(W0)
            if perturb_block is not None:
                layer.oft_R_out[perturb_block].normal_(std=1e-1)
            # oft_R_in stays 0 (residual identity).
        layer.merge_then_reinitialize(reinit_perm=False)
        return layer.weight.detach().clone()

    base = merged_weight(None)
    pert = merged_weight(perturb_block=2)
    diff = (pert - base).abs()
    rows = slice(2 * 64, 3 * 64)  # head 2's rows (out side, identity perm)
    assert diff[rows].max() > 1e-6                       # head 2 changed
    mask = torch.ones(512, dtype=torch.bool); mask[rows] = False
    assert diff[mask].max() < 1e-12                      # all other heads untouched
```

- [ ] **Step 2: Run tests**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poet.py -k "singular or mixing" -v`
Expected: PASS (these exercise the already-implemented constructor + merge; they are guard tests, no new product code).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_head_aligned_poet.py
git commit -F - <<'EOF'
test(poet): head-aligned spectrum + no-cross-head-mixing guards
EOF
```

---

## Task 5: Apply-policy routing in `replace_linears_with_poet`

**Files:**
- Modify: `src/optim/poet_layers.py`
- Test: `tests/unit/test_poet_layers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_poet_layers.py`:

```python
def test_head_aligned_routing_and_gqa():
    import torch.nn as nn
    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet
    from poet_torch import HeadAlignedPOETLinear, POETLinear

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = nn.Linear(512, 512, bias=False)      # 8 q heads
            self.linear_k = nn.Linear(512, 256, bias=False)      # 4 kv heads (GQA)
            self.linear_v = nn.Linear(512, 256, bias=False)
            self.linear_proj = nn.Linear(512, 512, bias=False)   # o: head side = in

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention = Attn()
            self.linear_fc1_gate = nn.Linear(512, 1536, bias=False)
            self.linear_fc2 = nn.Linear(1536, 512, bias=False)

    m = Block()
    n = replace_linears_with_poet(
        m, block_count=1, head_aligned_attn=True, head_dim=64,
        extra_linear_types=(nn.Linear,),
    )
    assert n == 6

    def inner(mod):
        assert isinstance(mod, POETMegatronLinear)
        return mod.poet_linear

    q = inner(m.self_attention.linear_q)
    assert isinstance(q, HeadAlignedPOETLinear) and q.head_side == "out" and q.head_count == 8
    k = inner(m.self_attention.linear_k)
    assert isinstance(k, HeadAlignedPOETLinear) and k.head_side == "out" and k.head_count == 4  # GQA
    o = inner(m.self_attention.linear_proj)
    assert isinstance(o, HeadAlignedPOETLinear) and o.head_side == "in" and o.head_count == 8
    # MLP stays stock POETLinear.
    assert isinstance(inner(m.linear_fc1_gate), POETLinear)
    assert not isinstance(inner(m.linear_fc1_gate), HeadAlignedPOETLinear)


def test_head_aligned_requires_unfused_qkv():
    import torch.nn as nn
    from src.optim.poet_layers import replace_linears_with_poet

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = nn.Linear(512, 1024, bias=False)  # still fused

    import pytest
    with pytest.raises(ValueError, match="unfused"):
        replace_linears_with_poet(
            Attn(), block_count=1, head_aligned_attn=True, head_dim=64,
            extra_linear_types=(nn.Linear,),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py::test_head_aligned_routing_and_gqa -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'head_aligned_attn'`.

- [ ] **Step 3: Add the routing + a DRY weight-init helper**

In `src/optim/poet_layers.py`, add after the `_UNFUSED_SEGMENT_NAMES` definition:

```python
# Attention projections that take head-aligned rotation, and which side carries
# the heads. q/k/v rows are heads (out); the output projection's cols are (in).
_HEAD_ALIGNED_SIDES = {
    "linear_q": "out",
    "linear_k": "out",
    "linear_v": "out",
    "linear_proj": "in",
}


def _copy_and_init_weight(pl, child, init_type, mup_alpha):
    """Copy child's weight (+bias) into the POET layer's frozen base, applying
    init_type. Shared by the stock and head-aligned branches."""
    import torch

    out_f, in_f = child.weight.shape
    has_bias = child.bias is not None and child.bias.numel() > 0
    with torch.no_grad():
        w = child.weight.data.clone()
        if init_type == "normalized":
            w = w / torch.norm(w, dim=1, keepdim=True)
        elif init_type == "mup_normalized":
            d_in = torch.tensor(float(in_f))
            d_out = torch.tensor(float(out_f))
            w = w / torch.norm(w, dim=1, keepdim=True)
            target = mup_alpha * torch.sqrt(d_out / d_in)
            current = torch.linalg.norm(w.float(), ord=2).item()
            w = w * (target / current).to(dtype=w.dtype, device=w.device)
        pl.weight.copy_(w.to(pl.weight.dtype))
        if has_bias:
            pl.bias.copy_(child.bias.data.to(pl.bias.dtype))
```

Change the `replace_linears_with_poet` signature to add three keyword params (place them after `freeze_output_rotation`):

```python
    freeze_output_rotation: bool = False,
    head_aligned_attn: bool = False,
    head_dim: int | None = None,
    resid_permute: bool = True,
```

Inside `_walk`, immediately after `if isinstance(child, linear_types):` and the `skip_lm_head` continue, insert the head-aligned branch (before the `out_f, in_f = child.weight.shape` / divisor logic):

```python
                if head_aligned_attn and name == "linear_qkv":
                    raise ValueError(
                        f"[POET] head_aligned_attn requires unfused q/k/v "
                        f"(set base.model.unfuse_qkv=true); found fused {full}"
                    )
                if head_aligned_attn and name in _HEAD_ALIGNED_SIDES:
                    from poet_torch import HeadAlignedPOETLinear

                    if head_dim is None:
                        raise ValueError("[POET] head_aligned_attn requires head_dim")
                    out_f, in_f = child.weight.shape
                    has_bias = child.bias is not None and child.bias.numel() > 0
                    resid_kwargs = (
                        {"resid_block_count": block_count}
                        if block_count is not None
                        else {"resid_block_size": block_size}
                    )
                    pl = HeadAlignedPOETLinear(
                        in_features=in_f,
                        out_features=out_f,
                        head_side=_HEAD_ALIGNED_SIDES[name],
                        head_dim=head_dim,
                        resid_permute=resid_permute,
                        bias=has_bias,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                        parameterization=parameterization,
                        **resid_kwargs,
                    )
                    _copy_and_init_weight(pl, child, init_type, mup_alpha)
                    wrapper = POETMegatronLinear(
                        pl, skip_bias_add=getattr(child, "skip_bias_add", False)
                    )
                    setattr(parent, name, wrapper)
                    replaced += 1
                    continue
```

(Leave the existing stock branch unchanged; optionally replace its inline weight-copy block with `_copy_and_init_weight(pl, child, init_type, mup_alpha)` to remove duplication — same behavior.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -v`
Expected: PASS (existing + the two new tests).

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_layers.py tests/unit/test_poet_layers.py
git commit -F - <<'EOF'
feat(poet): head-aligned attention routing in replace_linears_with_poet
EOF
```

---

## Task 6: Thread the flag + head_dim through the apply patch

**Files:**
- Modify: `src/patches/poet_apply_to_model.py:61-78`

- [ ] **Step 1: Edit `_apply_poet_to_chunk`**

In `src/patches/poet_apply_to_model.py`, extend `_apply_poet_to_chunk` to read the new args and pass them through:

```python
    def _apply_poet_to_chunk(m, args) -> int:
        block = getattr(args, "poet_block_size", 256)
        block_count = getattr(args, "poet_block_count", None)
        init = getattr(args, "poet_init_type", "normalized")
        mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        cache_mode = getattr(args, "poet_cache_mode", "none")
        parameterization = getattr(args, "poet_parameterization", "cayley")
        freeze_output_rotation = getattr(args, "poet_freeze_output_rotation", False)
        head_aligned_attn = getattr(args, "poet_head_aligned_attn", False)
        resid_permute = not getattr(args, "poet_no_head_resid_perm", False)
        head_dim = getattr(args, "kv_channels", None)
        if head_dim is None:
            head_dim = args.hidden_size // args.num_attention_heads
        return replace_linears_with_poet(
            m,
            block_size=block,
            block_count=block_count,
            init_type=init,
            mup_alpha=mup_alpha,
            cache_mode=cache_mode,
            parameterization=parameterization,
            freeze_output_rotation=freeze_output_rotation,
            head_aligned_attn=head_aligned_attn,
            head_dim=head_dim,
            resid_permute=resid_permute,
        )
```

- [ ] **Step 2: Static check**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/poet_apply_to_model.py`
Expected: no output (compiles). (Deep integration is verified by the Task 5 routing tests + the Task 8 dry-run + the GPU run; the nested helper is not directly unit-importable.)

- [ ] **Step 3: Commit**

```bash
git add src/patches/poet_apply_to_model.py
git commit -F - <<'EOF'
feat(poet): thread head-aligned flag + head_dim through apply patch
EOF
```

---

## Task 7: CLI flags + megatron_args emission + guard

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py`
- Modify: `src/utils/megatron_args.py`
- Test: `tests/unit/test_pretrain_gpt_slm.py`, `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_head_aligned_flags():
    import argparse
    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        ["--slm-config-path", "x.yaml", "--poet",
         "--poet-head-aligned-attn", "--poet-no-head-resid-perm"]
    )
    assert args.poet_head_aligned_attn is True
    assert args.poet_no_head_resid_perm is True


def test_add_slm_args_head_aligned_defaults():
    import argparse
    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml", "--poet"])
    assert args.poet_head_aligned_attn is False
    assert args.poet_no_head_resid_perm is False
```

Append to `tests/unit/test_megatron_args.py` (mirror the existing POET emission test style in that file):

```python
def test_poet_head_aligned_args_emitted():
    from omegaconf import OmegaConf
    from src.utils.megatron_args import _optimizer_args  # adjust import to match the file

    optim = OmegaConf.create({
        "type": "poet", "lr": 1e-3, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8,
        "poet": {
            "block_count": 1, "merge_period": 1, "reinit_period": -1, "scale": 0.5,
            "init_type": "normalized", "mup_alpha": 1.0, "cache_mode": "none",
            "parameterization": "cayley", "q_optimizer": "lie_algebra",
            "head_aligned_attn": True, "head_resid_perm": False,
        },
    })
    emitted = list(_optimizer_args(optim))
    assert "--poet-head-aligned-attn" in emitted
    assert "--poet-no-head-resid-perm" in emitted
```
(If the helper that emits POET args has a different name/signature in `megatron_args.py`, call that one; the assertion on the emitted token list is the point.)

- [ ] **Step 2: Run to verify failure**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py -k head_aligned -v`
Expected: FAIL (`unrecognized arguments: --poet-head-aligned-attn`).

- [ ] **Step 3a: Add the CLI flags**

In `launchers/pretrain_gpt_slm.py`, after the `--poet-lie-rms-c` argument, add:

```python
    # Head-aligned attention rotation (opt-in): q/k/v/o rotate their head-
    # structured side per head (block_size=head_dim, identity Psi, no perm);
    # the residual side stays a normal POET rotation. Requires --unfuse-qkv.
    group.add_argument("--poet-head-aligned-attn", action="store_true")
    # Disable the residual side's permutation (off-switch ablation).
    group.add_argument("--poet-no-head-resid-perm", action="store_true")
```

- [ ] **Step 3b: Emit from megatron_args + guard**

In `src/utils/megatron_args.py`, in the `kind == "poet"` branch, after the existing conditional appends (near the `lie_rms` / `lie_alternating` appends), add:

```python
        if poet.get("head_aligned_attn", False):
            poet_args.append("--poet-head-aligned-attn")
        if not poet.get("head_resid_perm", True):
            poet_args.append("--poet-no-head-resid-perm")
```

Add the unfuse guard at the top of the same `poet` branch (after `poet = optim.poet`), using whatever handle the function already has to the resolved config's `base.model.unfuse_qkv` (the function builds the full Megatron arg list, so it has access to the model section; if it does not, place this guard in `_apply_poet_to_chunk` instead, raising the same message):

```python
        if poet.get("head_aligned_attn", False) and not bool(
            cfg.get("base", {}).get("model", {}).get("unfuse_qkv", False)
        ):
            raise ValueError(
                "optim.poet.head_aligned_attn requires base.model.unfuse_qkv=true "
                "(head-aligned blocks need unfused q/k/v)."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py tests/unit/test_megatron_args.py -k "head_aligned or poet" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add launchers/pretrain_gpt_slm.py src/utils/megatron_args.py tests/unit/test_pretrain_gpt_slm.py tests/unit/test_megatron_args.py
git commit -F - <<'EOF'
feat(poet): CLI flags + arg emission for head-aligned attention
EOF
```

---

## Task 8: Experiment config, doc, and training script

**Files:**
- Create: `configs/experiments/optim/poet_lie_head.yaml`, `docs/experiments/poet_lie_head.md`, `scripts/train_poet_lie_head.sh`

- [ ] **Step 1: Create the experiment config**

Create `configs/experiments/optim/poet_lie_head.yaml` (clone of `poet_lie.yaml` + head-aligned):

```yaml
# @package _global_
# poet_lie_head: poet_lie + head-aligned attention rotation.
#
# q/k/v/o rotate their head-structured side per head (block_size=head_dim,
# identity Psi, no permutation, no cross-head mixing); the residual side keeps a
# normal POET rotation (block_count=1 dense in dev). Requires unfused q/k/v.
experiment:
  name: poet_lie_head
  family: optim
  description: |
    POET x Pion: Lie-algebra momentum (q_optimizer=lie_algebra, elementwise v)
    with head-aligned attention rotation (head_aligned_attn=true). Attention
    projections rotate their head side per head_dim block with a fixed identity
    permutation; the residual side is a normal POET rotation. Ablate vs poet_lie.
  references:
    - "POET"
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
  lr: 1.0e-3
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
    q_optimizer: lie_algebra
    lie_b1: 0.9
    lie_b2: 0.95
    lie_eps: 1.0e-8
    lie_v_mode: elementwise
    head_aligned_attn: true
    head_resid_perm: true
    train_output_rotation: true

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 2: Create the experiment doc (required by the pre-commit hook)**

Create `docs/experiments/poet_lie_head.md`:

```markdown
# poet_lie_head

POET × Pion Lie-algebra momentum with **head-aligned attention rotation**.

Attention projections (q/k/v/o) rotate their head-structured side with one
`head_dim`-sized block per head, a fixed identity permutation (block j = head j,
never resampled), and no cross-head mixing. The residual side stays a normal POET
rotation (`block_count=1` dense in dev). Requires unfused q/k/v.

Ablate against `optim/poet_lie` (dense both sides) and `optim/poet_lie_rms`.
```

- [ ] **Step 3: Create the training script**

Create `scripts/train_poet_lie_head.sh` (clone of `scripts/train_poet_lie_rms.sh` with `experiment=optim/poet_lie_head`). Copy that file verbatim and change only the experiment line:

```bash
cp scripts/train_poet_lie_rms.sh scripts/train_poet_lie_head.sh
sed -i 's#experiment=optim/poet_lie_rms#experiment=optim/poet_lie_head#' scripts/train_poet_lie_head.sh
chmod +x scripts/train_poet_lie_head.sh
```

(Also update the leading comment block in the new file to describe head-aligned attention rather than RMS.)

- [ ] **Step 4: Dry-run the script to verify it resolves and emits the flag**

Run:
```bash
PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH \
  python -m launchers.train_megatron experiment=optim/poet_lie_head \
  base/family=llama3 base/scale=60m --print-args 2>&1 | grep -- "--poet-head-aligned-attn"
```
Expected: the line `--poet-head-aligned-attn` appears in the resolved Megatron args. (If `train_megatron` has no `--print-args`, use the repo's standard resolve/dry-run entrypoint as in `train_poet_lie_rms.sh`; the assertion is that resolution succeeds and the flag is emitted.)

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/optim/poet_lie_head.yaml docs/experiments/poet_lie_head.md scripts/train_poet_lie_head.sh
git commit -F - <<'EOF'
feat(poet): poet_lie_head experiment config + script
EOF
```

---

## Task 9: GPU forward parity + both-sides-train (CUDA-gated test)

**Files:**
- Test: `tests/unit/test_head_aligned_poet.py`

- [ ] **Step 1: Add the GPU-gated test**

Append to `tests/unit/test_head_aligned_poet.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernel")
def test_forward_matches_reference_and_both_sides_get_grad():
    """The kernel forward (cayley) matches the pure-PyTorch decoupled reference,
    and a backward populates BOTH oft_R grads."""
    from poet_torch import HeadAlignedPOETLinear
    from tests.unit.test_poet_decoupled import poet_reference_forward

    torch.manual_seed(0)
    layer = HeadAlignedPOETLinear(
        in_features=512, out_features=512, head_side="out", head_dim=64,
        resid_block_count=1, device="cuda", dtype=torch.float32,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        layer.oft_R_in.normal_(std=1e-2)
        layer.oft_R_out.normal_(std=1e-2)

    x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
    y = layer(x)
    y_ref = poet_reference_forward(
        x.cpu(), layer.weight.detach().cpu(),
        layer.oft_R_in.detach().cpu(), layer.oft_R_out.detach().cpu(),
        layer.perm_in.cpu(), layer.perm_in_inv.cpu(),
        layer.perm_out.cpu(), layer.perm_out_inv.cpu(),
        layer.block_size_in, layer.block_size_out,
    )
    assert torch.allclose(y.detach().cpu(), y_ref, atol=1e-4)

    y.sum().backward()
    assert layer.oft_R_in.grad is not None and layer.oft_R_in.grad.abs().sum() > 0
    assert layer.oft_R_out.grad is not None and layer.oft_R_out.grad.abs().sum() > 0
```

- [ ] **Step 2: Verify it is collected and skipped on CPU**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poet.py -k forward_matches -v`
Expected: SKIPPED (no CUDA on the login node). The user runs it on a GPU node.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_head_aligned_poet.py
git commit -F - <<'EOF'
test(poet): GPU forward-parity + both-sides-grad for head-aligned layer
EOF
```

---

## Task 10: Full suite, static checks, and GPU launch command

**Files:** none (verification + handoff)

- [ ] **Step 1: Run the full affected CPU test suite**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_head_aligned_poet.py \
  tests/unit/test_poet_lie_momentum.py \
  tests/unit/test_poet_layers.py \
  tests/unit/test_poet_decoupled.py \
  tests/unit/test_pretrain_gpt_slm.py \
  tests/unit/test_megatron_args.py -v
```
Expected: all PASS (GPU-gated ones SKIPPED).

- [ ] **Step 2: Lint + compile the edited files**

Run:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m ruff check \
  third_party/poet_torch/head_aligned_layer.py src/optim/poet_layers.py \
  src/optim/poet_lie_momentum.py src/patches/poet_apply_to_model.py \
  launchers/pretrain_gpt_slm.py src/utils/megatron_args.py
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile \
  third_party/poet_torch/head_aligned_layer.py src/optim/poet_layers.py \
  src/optim/poet_lie_momentum.py src/patches/poet_apply_to_model.py
```
Expected: clean.

- [ ] **Step 3: GPU run (user launches — do NOT run here)**

```bash
codexlog poet_lie_head scripts/train_poet_lie_head.sh
```
Ablate val loss vs `poet_lie` (dense both sides) and `poet_lie_rms`. Optional residual-perm-off ablation: append `optim.poet.head_resid_perm=false`.

---

## Self-Review

**Spec coverage** (each spec section → task):
- §4 `HeadAlignedPOETLinear` (two-sided, head_dim head side, identity Ψ) → Tasks 2-3.
- §4 spectrum preservation / no cross-head mixing → Task 4.
- §5 apply policy (name→side, GQA, MLP stock, opt-in) → Tasks 5-6.
- §6 optimizer routing (unchanged) + per-block RMS → Task 1 (RMS); routing verified by `oft_R_in/out` names (no code change, exercised at GPU run).
- §7 merge integration → covered by inheritance (`isinstance(_, POETLinear)`); the overridden `merge_then_reinitialize` is Task 3; no `_run_merge` edit needed (noted in plan intro).
- §8 config/flags/experiment/script → Tasks 7-8.
- §9 performance → emergent (block_size=head_dim + identity Ψ); no dedicated task.
- §11 tests → Tasks 1-5, 9; full suite Task 10.

**Placeholder scan:** none — every step has concrete code/commands.

**Type/name consistency:** `head_side`/`head_dim`/`resid_block_count`/`resid_block_size`/`resid_permute`/`head_count` consistent across the class, the apply branch, and the tests; flag `--poet-head-aligned-attn` / `--poet-no-head-resid-perm` and config keys `head_aligned_attn` / `head_resid_perm` consistent across launcher, megatron_args, apply patch, and config. `_HEAD_ALIGNED_SIDES` and `_copy_and_init_weight` referenced only where defined.

**Note for the executor:** `merge_then_reinitialize` is stubbed in Task 2 and implemented in Task 3 — run them in order. If `megatron_args.py`'s POET-arg helper has a different name/`cfg` handle than assumed in Task 7, adapt the call/guard placement; the behavioral assertions (emitted flags; ValueError on fused qkv) are the contract.
