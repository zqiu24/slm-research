# POET `exp` Orthogonalization Parameterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `G = exp(Q)` (exact matrix exponential) as a config-selectable alternative to the current Cayley–Neumann orthogonalization in POET, leaving the Cayley path and default behavior byte-for-byte unchanged.

**Architecture:** A new rotation builder `get_weight_poet_decoupled_exp` computes `R = torch.linalg.matrix_exp(Q)` (fp32/fp64 compute, cast back), mirroring the signature of the existing `get_weight_poet_decoupled`. A single `POETLinear._build_R` dispatch on a new `self.parameterization` attribute routes the forward, the merge, and the ΔW-spec estimator through the same map. The `exp` forward (`forward_core_decoupled_exp`) builds `R` eagerly and reuses the existing parameterization-agnostic chain consumers. A new `optim.poet.parameterization` flag (default `"cayley"`) is threaded through the established `poet_*` config plumbing.

**Tech Stack:** PyTorch (`torch.linalg.matrix_exp`, autograd Fréchet backward), Megatron-LM patches, Hydra/OmegaConf config, pytest. No Triton kernel work; no `poet_ops.py` change.

**Spec:** [`docs/superpowers/specs/2026-06-02-poet-exp-parameterization-design.md`](../specs/2026-06-02-poet-exp-parameterization-design.md)

**Test runner (CPU, all tasks):**
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest <path> -q -p no:cacheprovider
```
(First collection is slow — ~30–90s — because importing torch + triton + poet_torch is heavy. This is normal; do not assume a hang.) Run CPU tests yourself and report real output. Do **not** run any GPU/training job — those are the user's.

**Working-tree note:** [`third_party/poet_torch/poet_layer.py`](../../../third_party/poet_torch/poet_layer.py) and [`configs/experiments/optim/poet.yaml`](../../../configs/experiments/optim/poet.yaml) already carry **unrelated** uncommitted edits (separate Muon-Q work). Layer your edits on top; never revert those. Each commit stages **only** the files that belong to that task.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `third_party/poet_torch/poet_layer.py` | exp builder, `_build_R` dispatch, `parameterization` attr, exp forward, merge/estimator routing | 1–4 |
| `tests/unit/test_poet_exp_parameterization.py` | all CPU unit tests for the exp parameterization | 1–5 |
| `src/optim/poet_layers.py` | thread `parameterization` into `POETLinear` construction; guard exp+cache | 5 |
| `src/patches/poet_apply_to_model.py` | read `args.poet_parameterization`, pass through | 5 |
| `launchers/pretrain_gpt_slm.py` | register `--poet-parameterization` CLI arg | 6 |
| `src/utils/megatron_args.py` | emit `--poet-parameterization` in the poet arg sequence | 6 |
| `configs/experiments/optim/poet.yaml` | expose `optim.poet.parameterization` | 6 |
| `tests/unit/test_megatron_args.py` | assert the flag is emitted | 6 |
| `CHANGELOG.md` | log the feature | 7 |

---

## Task 1: `exp` rotation builder (`get_weight_poet_decoupled_exp`)

**Files:**
- Create: `tests/unit/test_poet_exp_parameterization.py`
- Modify: `third_party/poet_torch/poet_layer.py` (insert after `get_weight_poet_decoupled`, currently ending at line 255)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_poet_exp_parameterization.py`:

```python
"""CPU unit tests for the POET `exp` (matrix-exponential) parameterization.

All tests here are CPU-runnable: `torch.linalg.matrix_exp` is pure PyTorch and
the decoupled "fast" chain is plain ops. GPU/compiled-forward parity is left to
the user's smoke run (the repo guards those with skipif(not cuda)).
"""

from __future__ import annotations

import math

import pytest
import torch

torch.manual_seed(0)


def _triu(bs):
    r, c = torch.triu_indices(bs, bs, 1)
    return r.to(torch.int32), c.to(torch.int32)


def test_exp_builder_is_exactly_orthogonal():
    from poet_torch.poet_layer import get_weight_poet_decoupled_exp

    bs_in, bs_out = 8, 8
    r_in, r_out = 2, 2
    ne_in = bs_in * (bs_in - 1) // 2
    ne_out = bs_out * (bs_out - 1) // 2
    oft_in = torch.randn(r_in, ne_in) * 0.1
    oft_out = torch.randn(r_out, ne_out) * 0.1
    rows_in, cols_in = _triu(bs_in)
    rows_out, cols_out = _triu(bs_out)

    R_out, R_in = get_weight_poet_decoupled_exp(
        oft_in, oft_out, bs_in, bs_out, rows_in, cols_in, rows_out, cols_out
    )
    eye_in = torch.eye(bs_in)
    eye_out = torch.eye(bs_out)
    err_in = (R_in @ R_in.transpose(-2, -1) - eye_in).abs().max().item()
    err_out = (R_out @ R_out.transpose(-2, -1) - eye_out).abs().max().item()
    assert err_in < 1e-5, err_in
    assert err_out < 1e-5, err_out
    # det == +1 (proper rotation, in SO(b) not just O(b))
    assert torch.allclose(torch.linalg.det(R_in.float()), torch.ones(r_in), atol=1e-4)


def test_exp_is_tighter_than_cayley_neumann_at_large_angle():
    """At a non-tiny angle, exp stays exactly orthogonal while the degree-4
    Cayley/Neumann truncation drifts measurably."""
    from poet_torch.poet_layer import (
        cayley_batch,
        get_weight_poet_decoupled_exp,
        pytorch_skew_symmetric,
    )

    bs = 8
    ne = bs * (bs - 1) // 2
    oft = torch.randn(1, ne) * 0.6  # large-ish angles
    rows, cols = _triu(bs)

    R_out, _ = get_weight_poet_decoupled_exp(oft, oft, bs, bs, rows, cols, rows, cols)
    Q = pytorch_skew_symmetric(oft, bs, rows, cols)
    R_cayley = cayley_batch(Q)

    eye = torch.eye(bs)
    exp_err = (R_out @ R_out.transpose(-2, -1) - eye).abs().max().item()
    cay_err = (R_cayley @ R_cayley.transpose(-2, -1) - eye).abs().max().item()
    assert exp_err < 1e-5
    assert cay_err > exp_err * 10  # Cayley truncation is much less orthogonal here


def test_exp_2x2_block_rotates_by_exactly_theta_no_factor_of_two():
    """§3 of the math doc: the canonical angle of exp(Q) is exactly the singular
    value of Q (no factor-of-2, unlike Cayley's 2*arctan(theta))."""
    from poet_torch.poet_layer import get_weight_poet_decoupled_exp

    bs = 2
    theta = 0.7
    oft = torch.tensor([[theta]])  # single upper-tri entry => angle theta
    rows, cols = _triu(bs)
    R_out, _ = get_weight_poet_decoupled_exp(oft, oft, bs, bs, rows, cols, rows, cols)
    R = R_out[0]
    # rotation angle recovered from the orthogonal 2x2 block
    angle = math.acos(float(R[0, 0].clamp(-1.0, 1.0)))
    assert abs(angle - theta) < 1e-5, angle
    # explicitly NOT the Cayley factor 2*arctan(theta)
    assert abs(angle - 2.0 * math.atan(theta)) > 1e-3


def test_exp_builder_gradcheck():
    """Autograd flows correctly through skew-construction + matrix_exp (fp64)."""
    from poet_torch.poet_layer import get_weight_poet_decoupled_exp

    bs = 4
    ne = bs * (bs - 1) // 2
    rows, cols = _triu(bs)
    oft = (torch.randn(1, ne, dtype=torch.float64) * 0.1).requires_grad_(True)

    def f(o):
        R_out, _ = get_weight_poet_decoupled_exp(o, o, bs, bs, rows, cols, rows, cols)
        return R_out

    assert torch.autograd.gradcheck(f, (oft,), atol=1e-5, rtol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider
```
Expected: FAIL — `ImportError: cannot import name 'get_weight_poet_decoupled_exp'`.

- [ ] **Step 3: Implement the builder**

In `third_party/poet_torch/poet_layer.py`, insert immediately after the end of `get_weight_poet_decoupled` (after line 255, before `def torch_bmm`):

```python
def _matrix_exp_skew(Q):
    """exp of a skew batch. matrix_exp is numerically delicate below fp32, so
    compute in fp32 (or keep fp64) then cast back. Autograd flows through the
    cast, so gradients return in the input dtype with no custom backward."""
    compute_dtype = Q.dtype if Q.dtype in (torch.float32, torch.float64) else torch.float32
    R = torch.linalg.matrix_exp(Q.to(compute_dtype))
    return R.to(Q.dtype)


def get_weight_poet_decoupled_exp(oft_R_in, oft_R_out,
                                  block_size_in, block_size_out,
                                  rows_in, cols_in, rows_out, cols_out):
    """Matrix-exponential twin of ``get_weight_poet_decoupled``.

    Builds (R_out, R_in) as the *exact* matrix exponential of the skew
    generators instead of the truncated Cayley/Neumann polynomial. R is exactly
    orthogonal for any Q (no ||Q||<1 ceiling, no truncation error), and the
    singular values of Q are exactly the rotation angles of R. Same signature
    and (R_out, R_in) return ordering as ``get_weight_poet_decoupled`` so it is a
    drop-in for the parameterization dispatch.
    """
    Q_in = pytorch_skew_symmetric(oft_R_in, block_size_in, rows_in, cols_in)
    Q_out = pytorch_skew_symmetric(oft_R_out, block_size_out, rows_out, cols_out)
    R_in = _matrix_exp_skew(Q_in)
    R_out = _matrix_exp_skew(Q_out)
    return R_out, R_in
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/poet_layer.py tests/unit/test_poet_exp_parameterization.py
git commit -m "feat(poet): add exact matrix_exp orthogonalization builder"
```

---

## Task 2: `POETLinear` parameterization arg + `_build_R` dispatch

**Files:**
- Modify: `third_party/poet_torch/poet_layer.py` (`POETLinear.__init__` line 442; add `_build_R`; rewire `_merge_R` line 531)
- Test: `tests/unit/test_poet_exp_parameterization.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_poet_exp_parameterization.py`:

```python
def test_poetlinear_defaults_to_cayley():
    from poet_torch import POETLinear

    pl = POETLinear(in_features=16, out_features=16, bsz=8, device="cpu", dtype=torch.float32)
    assert pl.parameterization == "cayley"


def test_poetlinear_rejects_unknown_parameterization():
    from poet_torch import POETLinear

    with pytest.raises(ValueError):
        POETLinear(in_features=16, out_features=16, bsz=8,
                   parameterization="bogus", device="cpu", dtype=torch.float32)


def test_build_R_exp_is_orthogonal_and_cayley_branch_runs():
    from poet_torch import POETLinear

    pl = POETLinear(in_features=16, out_features=16, bsz=8,
                    parameterization="exp", device="cpu", dtype=torch.float32)
    # seed non-zero rotation so R != I
    with torch.no_grad():
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
    R_out, R_in = pl._build_R(pl.oft_R_in, pl.oft_R_out)
    eye = torch.eye(8)
    assert (R_in @ R_in.transpose(-2, -1) - eye).abs().max().item() < 1e-5
    assert (R_out @ R_out.transpose(-2, -1) - eye).abs().max().item() < 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider -k "parameterization or build_R"
```
Expected: FAIL — `POETLinear.__init__` has no `parameterization` kwarg / no `_build_R`.

- [ ] **Step 3: Implement the arg, validation, attribute, and dispatch**

In `third_party/poet_torch/poet_layer.py`, change the `POETLinear.__init__` signature (line 442-443) from:

```python
    def __init__(self, in_features, out_features, bsz=None, block_count=None,
                 bias=False, device=None, dtype=None, mem_efficient_mode=False):
```
to:
```python
    def __init__(self, in_features, out_features, bsz=None, block_count=None,
                 bias=False, device=None, dtype=None, mem_efficient_mode=False,
                 parameterization="cayley"):
```

Immediately after `self.block_size = block_size_in` (line 471), add:

```python
        if parameterization not in ("cayley", "exp"):
            raise ValueError(
                f"parameterization must be 'cayley' or 'exp', got {parameterization!r}"
            )
        self.parameterization = parameterization
```

Add a `_build_R` method directly above the existing `_merge_R` (before line 531):

```python
    def _build_R(self, oft_in, oft_out):
        """Build (R_out, R_in) from skew params using the configured map.

        Single dispatch point so forward, merge, and the dW-spec estimator all
        use the same orthogonalization.
        """
        if self.parameterization == "exp":
            return get_weight_poet_decoupled_exp(
                oft_in, oft_out, self.block_size_in, self.block_size_out,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            )
        return get_weight_poet_decoupled(
            oft_in, oft_out, self.block_size_in, self.block_size_out,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
        )
```

Rewire `_merge_R` (currently lines 531-537) to delegate:

```python
    def _merge_R(self):
        """Build (R_out, R_in) from the two decoupled skew params (no grad)."""
        return self._build_R(self.oft_R_in, self.oft_R_out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider
```
Expected: all tests so far pass (7 total).

- [ ] **Step 5: Run the existing decoupled CPU tests to confirm no Cayley regression**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_decoupled.py -q -p no:cacheprovider -k "block_diag or skew or coupled_op_matches or reference"
```
Expected: all selected tests pass (the default `parameterization="cayley"` path is unchanged).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/poet_layer.py tests/unit/test_poet_exp_parameterization.py
git commit -m "feat(poet): POETLinear parameterization arg + _build_R dispatch"
```

---

## Task 3: `exp` forward path + `POETLinear.forward` branch

**Files:**
- Modify: `third_party/poet_torch/poet_layer.py` (add `forward_core_decoupled_exp` near `forward_core_decoupled` ~line 388; branch in `POETLinear.forward` line 595)
- Test: `tests/unit/test_poet_exp_parameterization.py`

**Design note:** `forward_core_decoupled` is `@torch.compile(fullgraph=True)` and builds R inside the compiled region; `matrix_exp` (especially its backward) may not survive `fullgraph=True`. The committed `exp` path therefore builds R **eagerly** (matrix_exp is `O(b³)` per block, amortized over the whole microbatch) and reuses the existing parameterization-agnostic chain consumers (`chain_layer_x_fast_decoupled` default; `chain_layer_x_checkpoint_mem_o2_decoupled` for mem-efficient). The Cayley path is untouched. Compiling the chain for the exp path is an optional GPU perf follow-up (Task 8).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_poet_exp_parameterization.py`:

```python
def _poet_forward_reference_exp(pl, x):
    """Pure-PyTorch oracle for the exp forward, independent of the layer code.

    Mirrors chain_layer_x_fast_decoupled's math:
      y = perm_out( bmm_out( ( perm_in(x) @blocks Rin ) @ W^T ) @blocks Rout )
    with R = exp(Q) built from the layer's current skew params.
    """
    from poet_torch.poet_layer import pytorch_skew_symmetric

    Qi = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)
    Qo = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)
    R_in = torch.linalg.matrix_exp(Qi.float()).to(x.dtype)
    R_out = torch.linalg.matrix_exp(Qo.float()).to(x.dtype)

    def apply_blocks(t, R, bs):
        lead = t.shape[:-1]
        n = t.numel() // t.shape[-1]
        r = R.size(0)
        tb = t.reshape(n, r, bs).transpose(0, 1)  # [r, n, bs]
        out = torch.bmm(tb, R).transpose(0, 1).reshape(*lead, r * bs)
        return out

    xin = x.index_select(-1, pl.perm_in_inv.long())
    xin = apply_blocks(xin, R_in, pl.block_size_in)
    y = xin @ pl.weight.t()
    if pl.bias is not None:
        y = y + pl.bias
    y = apply_blocks(y, R_out, pl.block_size_out)
    return y.index_select(-1, pl.perm_out.long())


def test_exp_forward_matches_pure_pytorch_oracle_cpu():
    from poet_torch import POETLinear

    pl = POETLinear(in_features=16, out_features=16, bsz=8,
                    parameterization="exp", device="cpu", dtype=torch.float32,
                    mem_efficient_mode=False)
    with torch.no_grad():
        pl.weight.normal_()
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
    x = torch.randn(4, 16)
    y = pl(x)
    y_ref = _poet_forward_reference_exp(pl, x)
    assert torch.allclose(y, y_ref, atol=1e-4, rtol=1e-3), (y - y_ref).abs().max()


def test_exp_forward_backward_runs_cpu():
    from poet_torch import POETLinear

    pl = POETLinear(in_features=16, out_features=16, bsz=8,
                    parameterization="exp", device="cpu", dtype=torch.float32)
    with torch.no_grad():
        pl.weight.normal_()
    x = torch.randn(4, 16)
    pl(x).sum().backward()
    assert pl.oft_R_in.grad is not None
    assert pl.oft_R_out.grad is not None
    assert torch.isfinite(pl.oft_R_in.grad).all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider -k "forward"
```
Expected: FAIL — `POETLinear.forward` ignores `parameterization`, so it routes through the Cayley op (`torch.ops.poet.cayley`) which on CPU raises (no CUDA Triton), or returns Cayley values that don't match the exp oracle.

- [ ] **Step 3: Implement the exp forward and the branch**

In `third_party/poet_torch/poet_layer.py`, add a new function directly after `forward_core_decoupled` (after line 388, before `forward_core_q8`):

```python
def forward_core_decoupled_exp(
    x: torch.Tensor,
    oft_R_in: torch.Tensor,
    oft_R_out: torch.Tensor,
    block_size_in: int,
    block_size_out: int,
    rows_in: torch.Tensor,
    cols_in: torch.Tensor,
    rows_out: torch.Tensor,
    cols_out: torch.Tensor,
    perm_in: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    mem_efficient_mode: bool = False,
) -> torch.Tensor:
    """Forward for the exact-matrix-exponential parameterization.

    Builds R eagerly via matrix_exp (NOT under torch.compile fullgraph, which
    matrix_exp's backward may not support), then reuses the same
    parameterization-agnostic chain consumers as the Cayley path. The R build is
    O(b^3) per block, amortized over the whole microbatch; the per-token chain is
    unchanged.
    """
    R_out, R_in = get_weight_poet_decoupled_exp(
        oft_R_in, oft_R_out, block_size_in, block_size_out,
        rows_in, cols_in, rows_out, cols_out,
    )
    if mem_efficient_mode:
        y = chain_layer_x_checkpoint_mem_o2_decoupled(
            x, R_in, base_weight, base_bias, R_out,
            perm_in_inv, perm_in, perm_out, perm_out_inv,
            block_size_in, block_size_out,
        )
    else:
        y = chain_layer_x_fast_decoupled(
            x, R_in, base_weight, base_bias, R_out,
            perm_in_inv, perm_in, perm_out, perm_out_inv,
            block_size_in, block_size_out,
        )
    return y
```

Change `POETLinear.forward` (lines 595-603) to branch:

```python
    def forward(self, x):
        if self.parameterization == "exp":
            return forward_core_decoupled_exp(
                x, self.oft_R_in, self.oft_R_out,
                self.block_size_in, self.block_size_out,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
                self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv,
                self.weight, self.bias, self.mem_efficient_mode,
            )
        x = forward_core_decoupled(
            x, self.oft_R_in, self.oft_R_out,
            self.block_size_in, self.block_size_out,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv,
            self.weight, self.bias, self.mem_efficient_mode,
        )
        return x
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider
```
Expected: all tests pass (11 total).

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/poet_layer.py tests/unit/test_poet_exp_parameterization.py
git commit -m "feat(poet): exp forward path + POETLinear.forward dispatch"
```

---

## Task 4: merge + ΔW-spec estimator consistency under `exp`

**Files:**
- Modify: `third_party/poet_torch/poet_layer.py` (`estimate_poet_delta_weff_spec` inner `_R`, lines 1057-1064)
- Test: `tests/unit/test_poet_exp_parameterization.py`

**Note:** `merge_then_reinitialize` (line 562) and `merge_then_reinitialize_working` (line 539) already call `self._merge_R()` → `self._build_R()` (Task 2), so they pick up `exp` automatically. This task verifies that and fixes the one remaining direct caller, `estimate_poet_delta_weff_spec`, which still hardcodes the Cayley builder.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_poet_exp_parameterization.py`:

```python
def test_merge_then_reinitialize_exp_rotates_weight_and_zeros_oft():
    from poet_torch import POETLinear
    from poet_torch.poet_layer import block_diag_lr_matmul_decoupled

    pl = POETLinear(in_features=16, out_features=16, bsz=8,
                    parameterization="exp", device="cpu", dtype=torch.float32)
    with torch.no_grad():
        pl.weight.normal_()
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
        W0 = pl.weight.clone()
        R_out, R_in = pl._build_R(pl.oft_R_in, pl.oft_R_out)
        perm_in0, perm_out0 = pl.perm_in.clone(), pl.perm_out.clone()

    # expected merged weight = transpose( perm( Rin @ W0^T @ Rout ) )
    tmp = block_diag_lr_matmul_decoupled(R_in, W0.t(), R_out)
    tmp = tmp.index_select(0, perm_in0).index_select(1, perm_out0)
    expected = tmp.t()

    pl.merge_then_reinitialize()

    assert torch.allclose(pl.oft_R_in, torch.zeros_like(pl.oft_R_in))
    assert torch.allclose(pl.oft_R_out, torch.zeros_like(pl.oft_R_out))
    # weight changed under a real rotation
    assert not torch.allclose(pl.weight, W0)
    # NOTE: merge_then_reinitialize re-permutes the float weight before storing,
    # so compare the un-re-permuted reconstruction:
    reperm = expected.index_select(0, pl.perm_out_inv.long()).index_select(1, pl.perm_in_inv.long())
    assert torch.allclose(pl.weight, reperm, atol=1e-4, rtol=1e-3), (pl.weight - reperm).abs().max()


def test_delta_weff_spec_uses_layer_parameterization():
    from poet_torch import POETLinear
    from poet_torch.poet_layer import estimate_poet_delta_weff_spec

    pl = POETLinear(in_features=16, out_features=16, bsz=8,
                    parameterization="exp", device="cpu", dtype=torch.float32)
    with torch.no_grad():
        pl.weight.normal_()
        prev_in = torch.zeros_like(pl.oft_R_in)
        prev_out = torch.zeros_like(pl.oft_R_out)
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
    dM, sigma = estimate_poet_delta_weff_spec(pl, prev_in, prev_out)
    assert sigma > 0.0
    assert torch.isfinite(dM).all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider -k "merge or delta_weff"
```
Expected: `test_merge_then_reinitialize_exp...` PASSES already (merge routes through `_build_R`), but `test_delta_weff_spec_uses_layer_parameterization` may pass-by-luck or mismatch because `estimate_poet_delta_weff_spec`'s inner `_R` hardcodes the Cayley builder. To make the requirement explicit and the fix verifiable, also assert exp-vs-cayley divergence (see Step 3 rationale). If both pass without the fix, still apply Step 3 — the estimator MUST use the layer's parameterization for correctness.

- [ ] **Step 3: Route the estimator through the layer's parameterization**

In `third_party/poet_torch/poet_layer.py`, change the inner `_R` closure of `estimate_poet_delta_weff_spec` (lines 1057-1064) from the hardcoded `get_weight_poet_decoupled(...)` call to dispatch on the module's parameterization:

```python
    def _R(oft_in, oft_out):
        oft_in = oft_in.to(device=device, dtype=poet_module.oft_R_in.dtype)
        oft_out = oft_out.to(device=device, dtype=poet_module.oft_R_out.dtype)
        if getattr(poet_module, "parameterization", "cayley") == "exp":
            return get_weight_poet_decoupled_exp(
                oft_in, oft_out,
                poet_module.block_size_in, poet_module.block_size_out,
                poet_module.rows_in, poet_module.cols_in,
                poet_module.rows_out, poet_module.cols_out,
            )
        return get_weight_poet_decoupled(
            oft_in, oft_out,
            poet_module.block_size_in, poet_module.block_size_out,
            poet_module.rows_in, poet_module.cols_in,
            poet_module.rows_out, poet_module.cols_out,
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider
```
Expected: all tests pass (13 total).

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/poet_layer.py tests/unit/test_poet_exp_parameterization.py
git commit -m "feat(poet): route merge & dW-spec estimator through parameterization"
```

---

## Task 5: thread `parameterization` into layer replacement + apply patch

**Files:**
- Modify: `src/optim/poet_layers.py` (`replace_linears_with_poet` lines 110-209)
- Modify: `src/patches/poet_apply_to_model.py` (`_apply_poet_to_chunk` lines 61-73)
- Test: `tests/unit/test_poet_exp_parameterization.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_poet_exp_parameterization.py`:

```python
def test_replace_linears_threads_parameterization():
    import torch.nn as nn
    from src.optim.poet_layers import replace_linears_with_poet

    model = nn.Sequential(nn.Linear(16, 16, bias=False))
    n = replace_linears_with_poet(
        model, block_size=8, init_type="none",
        extra_linear_types=(nn.Linear,), parameterization="exp",
    )
    assert n == 1
    pl = model[0].poet_linear
    assert pl.parameterization == "exp"


def test_replace_linears_defaults_to_cayley():
    import torch.nn as nn
    from src.optim.poet_layers import replace_linears_with_poet

    model = nn.Sequential(nn.Linear(16, 16, bias=False))
    replace_linears_with_poet(
        model, block_size=8, init_type="none", extra_linear_types=(nn.Linear,)
    )
    assert model[0].poet_linear.parameterization == "cayley"


def test_replace_linears_rejects_exp_with_cache():
    import torch.nn as nn
    from src.optim.poet_layers import replace_linears_with_poet

    model = nn.Sequential(nn.Linear(16, 16, bias=False))
    with pytest.raises(ValueError):
        replace_linears_with_poet(
            model, block_size=8, init_type="none", extra_linear_types=(nn.Linear,),
            parameterization="exp", cache_mode="cached_fwd_bwd",
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider -k "replace_linears"
```
Expected: FAIL — `replace_linears_with_poet` has no `parameterization` kwarg.

- [ ] **Step 3: Add the kwarg, the exp+cache guard, and forward it**

In `src/optim/poet_layers.py`, add the kwarg to the `replace_linears_with_poet` signature (after `cache_mode: str = "none",` at line 119):

```python
    cache_mode: str = "none",
    parameterization: str = "cayley",
```

Immediately after the `linear_types` guard block (after line 139, before `replaced = 0`), add the unsupported-combo guard:

```python
    if parameterization == "exp" and cache_mode != "none":
        raise ValueError(
            "parameterization='exp' is not supported with cache_mode != 'none' "
            "(the cached Cayley path is a documented dead-end; use cache_mode='none')."
        )
```

Forward `parameterization` to the plain `POETLinear` construction (the `cache_mode == "none"` branch, lines 192-200). Change:

```python
                if cache_mode == "none":
                    pl = POETLinear(
                        in_features=in_f,
                        out_features=out_f,
                        bias=has_bias,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                        **block_kwargs,
                    )
```
to add `parameterization=parameterization`:
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
(The `CachedPOETLinear` branch is left unchanged — it is unreachable for `exp` because of the guard above, and keeps the `"cayley"` default.)

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py -q -p no:cacheprovider -k "replace_linears"
```
Expected: 3 passed.

- [ ] **Step 5: Wire the apply patch to read the arg**

In `src/patches/poet_apply_to_model.py`, in `_apply_poet_to_chunk` (lines 61-73), read the new arg and pass it through. Change:

```python
    def _apply_poet_to_chunk(m, args) -> int:
        block = getattr(args, "poet_block_size", 256)
        block_count = getattr(args, "poet_block_count", None)
        init = getattr(args, "poet_init_type", "normalized")
        mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        cache_mode = getattr(args, "poet_cache_mode", "none")
        return replace_linears_with_poet(
            m,
            block_size=block,
            block_count=block_count,
            init_type=init,
            mup_alpha=mup_alpha,
            cache_mode=cache_mode,
        )
```
to:
```python
    def _apply_poet_to_chunk(m, args) -> int:
        block = getattr(args, "poet_block_size", 256)
        block_count = getattr(args, "poet_block_count", None)
        init = getattr(args, "poet_init_type", "normalized")
        mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        cache_mode = getattr(args, "poet_cache_mode", "none")
        parameterization = getattr(args, "poet_parameterization", "cayley")
        return replace_linears_with_poet(
            m,
            block_size=block,
            block_count=block_count,
            init_type=init,
            mup_alpha=mup_alpha,
            cache_mode=cache_mode,
            parameterization=parameterization,
        )
```

- [ ] **Step 6: Run the patch-apply tests to confirm no regression**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_apply.py tests/unit/test_poet_layers.py -q -p no:cacheprovider
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/optim/poet_layers.py src/patches/poet_apply_to_model.py tests/unit/test_poet_exp_parameterization.py
git commit -m "feat(poet): thread parameterization through layer replacement + apply patch"
```

---

## Task 6: config flag plumbing (launcher arg + megatron_args + yaml)

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` (after line 53)
- Modify: `src/utils/megatron_args.py` (`_optimizer_args`, poet branch, after line 251)
- Modify: `configs/experiments/optim/poet.yaml` (after line 61)
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_argv_includes_parameterization_when_set():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "poet": {
                    "block_size": 8,
                    "init_type": "none",
                    "mup_alpha": 1.0,
                    "merge_period": 0,
                    "scale": 1.0,
                    "parameterization": "exp",
                },
            }
        }
    )
    args = _optimizer_args(cfg)
    assert "--poet-parameterization" in args
    assert args[args.index("--poet-parameterization") + 1] == "exp"


def test_poet_argv_parameterization_defaults_to_cayley():
    from omegaconf import OmegaConf

    from src.utils.megatron_args import _optimizer_args

    cfg = OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 1.0e-3,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "poet": {
                    "block_size": 8,
                    "init_type": "none",
                    "mup_alpha": 1.0,
                    "merge_period": 0,
                    "scale": 1.0,
                },
            }
        }
    )
    args = _optimizer_args(cfg)
    assert args[args.index("--poet-parameterization") + 1] == "cayley"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -q -p no:cacheprovider -k "parameterization"
```
Expected: FAIL — `--poet-parameterization` not emitted.

- [ ] **Step 3: Emit the flag in `_optimizer_args`**

In `src/utils/megatron_args.py`, in the `kind == "poet"` branch, add the flag to the `poet_args` list (after the `--poet-cache-mode` pair at lines 250-251):

```python
            "--poet-cache-mode",
            poet.get("cache_mode", "none"),
            "--poet-parameterization",
            poet.get("parameterization", "cayley"),
```

- [ ] **Step 4: Register the CLI arg in the launcher**

In `launchers/pretrain_gpt_slm.py`, after the `--poet-cache-mode` argument block (after line 53), add:

```python
    group.add_argument(
        "--poet-parameterization",
        choices=["cayley", "exp"],
        default="cayley",
    )
```

- [ ] **Step 5: Expose the field in the experiment config**

In `configs/experiments/optim/poet.yaml`, after the `use_poet_adam: false` block (after line 61), add:

```yaml
    # Orthogonalization map for the block rotation G = f(Q):
    #   "cayley" (default): truncated Cayley/Neumann polynomial (poet::cayley
    #     Triton op). Cheap matmuls; approximate orthogonality; angle = 2*arctan.
    #   "exp": exact matrix exponential G = exp(Q) (torch.linalg.matrix_exp).
    #     Exactly orthogonal for any Q; angle = singular value of Q (no factor
    #     of 2, no ||Q||<1 ceiling). Heavier backward; cache_mode must be "none".
    parameterization: cayley
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -q -p no:cacheprovider
```
Expected: all pass (existing + 2 new).

- [ ] **Step 7: Commit**

```bash
git add launchers/pretrain_gpt_slm.py src/utils/megatron_args.py configs/experiments/optim/poet.yaml tests/unit/test_megatron_args.py
git commit -m "feat(poet): expose optim.poet.parameterization config flag (cayley|exp)"
```

---

## Task 7: full regression run + CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Run the full POET + args CPU test suite**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_exp_parameterization.py tests/unit/test_poet_decoupled.py tests/unit/test_poet_layers.py tests/unit/test_megatron_args.py tests/unit/test_patch_poet_apply.py -q -p no:cacheprovider
```
Expected: all pass (GPU-only tests in `test_poet_decoupled.py` are skipped via `skipif(not cuda)`; everything else passes). Report the real pass/skip counts.

- [ ] **Step 2: Update CHANGELOG**

Add an entry to the top of the changelog section in `CHANGELOG.md`:

```markdown
- POET: add `optim.poet.parameterization: exp` — exact matrix-exponential
  orthogonalization (`G = exp(Q)`) as a config-selectable alternative to the
  default Cayley/Neumann path. Exact orthogonality, angle = singular value of Q.
```
(Match the surrounding bullet/format style of the existing CHANGELOG.)

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(poet): changelog entry for exp parameterization"
```

---

## Task 8 (OPTIONAL — GPU, user-run): attempt to compile the exp chain

This task is **optional** and requires a GPU, so it is the user's to run, not the agent's. It is a perf optimization, not a correctness requirement — Tasks 1–7 deliver a complete, tested feature.

**Goal:** Recover torch.compile fusion on the exp path's per-token chain (the Cayley path gets this via the `@torch.compile(fullgraph=True)` on `forward_core_decoupled`).

- [ ] **Step 1: Try wrapping the exp forward under fullgraph compile**

On a GPU node, add `@torch.compile(fullgraph=True)` to `forward_core_decoupled_exp` and run a 1-GPU POET smoke with `optim.poet.parameterization=exp`. If `matrix_exp` fwd+bwd compiles cleanly with no graph break, keep the decorator.

- [ ] **Step 2: If matrix_exp breaks fullgraph, split R-build from the chain**

If compile errors on `matrix_exp`, factor the chain (everything after `get_weight_poet_decoupled_exp`) into a separate `@torch.compile(fullgraph=True)` helper that takes `R_in, R_out, x, ...` and leave the `matrix_exp` R-build eager in `forward_core_decoupled_exp`. This compiles the per-token chain while keeping matrix_exp in eager.

- [ ] **Step 3: Verify training-loss parity**

Compare a short `parameterization=exp` run against `parameterization=cayley` for sane loss curves; confirm `R @ Rᵀ ≈ I` holds throughout (it must, by construction).

---

## Self-Review (completed by plan author)

**Spec coverage:** Builder (§Unit 1 → Task 1); dispatch + `parameterization` attr (§Unit 2 → Task 2); forward + compile handling (§Unit 3 → Task 3, Task 8); merge/estimator consistency (→ Task 4); config plumbing table (§Unit 4 → Tasks 5–6); testing section (orthogonality/angle/gradcheck/dispatch/merge → Tasks 1–6); out-of-scope guards (exp+cache → Task 5). All spec sections map to tasks.

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows complete code; every run step shows the command and expected result. Task 8 is explicitly optional and GPU-gated, with concrete steps (not a placeholder).

**Type/name consistency:** `get_weight_poet_decoupled_exp` (Tasks 1,2,3,4), `_matrix_exp_skew` (Task 1), `_build_R` (Tasks 2,3,4), `self.parameterization` (Tasks 2,3,4,5), `parameterization=` kwarg (Tasks 2,5), `forward_core_decoupled_exp` (Task 3), `--poet-parameterization` / `poet_parameterization` / `poet.parameterization` (Tasks 5,6) — names are consistent across tasks and match the existing code read during design.
