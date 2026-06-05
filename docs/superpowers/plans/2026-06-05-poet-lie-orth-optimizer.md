# POET Lie-Orth (Muon-like Orthogonalizing) Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Muon-like *orthogonalizing* sibling to the existing POET Lie-RMS optimizer — one that orthonormalizes the skew update direction so **every rotation plane turns by the same angle `c`** (instead of RMS-scaling, which preserves the gradient's relative per-plane angles), to run the head-to-head experiment in `docs/muon_orthogonalizing_optimizer_poet.md` §7.

**Architecture:** The orthogonalizing variant shares the *entire* Lie-momentum pipeline (param split, side-tagged groups, alternating single-sided update, momentum buffers that persist across the merge, AdamW branch for non-skew params). The **only** difference is the transform applied to the direction before it is written into `oft_R`: the RMS path scales by `c·√b/‖A‖_F`; this path orthonormalizes the per-block `b×b` skew direction (all singular values → 1) then scales by `c`. Therefore it is implemented as a **new mode (`ortho=True`) inside `LieAlgebraMomentum`**, mutually exclusive with `rms=True` — exactly mirroring how `rms` was added — rather than as a duplicate optimizer class. First-moment-only by default (docs §4: a second moment is partially undone by orthonormalization). A new pure-tensor helper `orthonormalize_skew_unit` lives next to the existing `orthogonalize_skew_blocks`.

**Tech Stack:** PyTorch (`torch 2.11`, CPU-testable), Megatron-LM (vendored, not needed for unit tests), OmegaConf/Hydra experiment configs, pytest.

**CPU test runner (use this exact interpreter — base `python` lacks torch/omegaconf):**
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest <path> -v
```
Run all commands from the repo root `/lustre/fast/fast/zqiu/slm-research`.

**Source doc:** `docs/muon_orthogonalizing_optimizer_poet.md`

---

## Background: the plumbing chain (read once before starting)

A POET hyperparameter flows through five layers. Every new knob must be added at each:

1. **Experiment YAML** `configs/experiments/optim/poet_lie_orth.yaml` — sets `optim.poet.lie_ortho*`.
2. **argv builder** `src/utils/megatron_args.py` (`_optimizer_args`, `kind == "poet"`) — translates `optim.poet.lie_ortho*` → `--poet-lie-ortho*` argv.
3. **argparse** `launchers/pretrain_gpt_slm.py` (`add_slm_args`) — defines `--poet-lie-ortho*`.
4. **config copy** `src/patches/poet_optimizer_setup.py` (`_wrapped_get_config`) — copies `args.poet_lie_ortho*` → `config.poet_lie_ortho*`.
5. **builder** `src/optim/poet.py` (`get_megatron_poet_lie_momentum_optimizer`) — reads `config.poet_lie_ortho*`, passes to `LieAlgebraMomentum`.

The optimizer math itself is `src/optim/poet_lie_momentum.py` (the `LieAlgebraMomentum.step` skew branch).

**Key data layout:** `oft_R` params have shape `(n_blocks, n_elems)` where `n_elems = b·(b−1)/2` (the strictly-upper-triangular entries of a `b×b` skew block). Conversions live in `src/diag/skew_conditioning.py`: `vec_to_skew(vec, b) → (n_blocks, b, b)`, `skew_to_vec(skew, b) → (n_blocks, n_elems)`, `block_size_from_nelems(n_elems) → b`.

**Naming used consistently across all tasks** (do not vary):
| Layer | Name |
|---|---|
| optimizer kwargs | `ortho`, `ortho_c`, `ortho_method`, `ortho_ns_steps`, `ortho_use_second_moment` |
| `config.*` / `args.*` | `poet_lie_ortho`, `poet_lie_ortho_c`, `poet_lie_ortho_method`, `poet_lie_ortho_ns_steps`, `poet_lie_ortho_use_second_moment` |
| argparse / argv flags | `--poet-lie-ortho`, `--poet-lie-ortho-c`, `--poet-lie-ortho-method`, `--poet-lie-ortho-ns-steps`, `--poet-lie-ortho-use-second-moment` |
| YAML keys (`optim.poet.`) | `lie_ortho`, `lie_ortho_c`, `lie_ortho_method`, `lie_ortho_ns_steps`, `lie_ortho_use_second_moment` |
| helper fn | `orthonormalize_skew_unit` (in `src/optim/poet_skew_muon.py`) |
| experiment | name `poet_lie_orth`, file `configs/experiments/optim/poet_lie_orth.yaml`, script `scripts/train_poet_lie_orth.sh` |

**The angle convention (state this in code comments — it is the one subtlety):** the realized per-plane rotation angle is `group_lr × ortho_c`, *not* `ortho_c` alone. This is identical to how the shipped RMS optimizer behaves (`angle = group_lr × rms_c`): the optimizer writes `oft_R ← lr · (ortho_c · A_orth)`, and the scheduler decays `group_lr`. So to mirror the RMS best run (`lr=0.003, rms_c=4 ⇒ ~0.012 rad/plane`), set `lr=0.003, lie_ortho_c=4`.

---

## File Structure

| File | Create / Modify | Responsibility |
|---|---|---|
| `src/optim/poet_skew_muon.py` | Modify | Add `orthonormalize_skew_unit` (skew → skew, σ→1). Reuses `orthogonalize_skew_blocks`. |
| `src/optim/poet_lie_momentum.py` | Modify | Add `ortho` mode to `LieAlgebraMomentum` (new kwargs + step-branch). |
| `src/optim/poet.py` | Modify | `get_megatron_poet_lie_momentum_optimizer`: read new `config.poet_lie_ortho*`, pass through, extend log. |
| `launchers/pretrain_gpt_slm.py` | Modify | Add `--poet-lie-ortho*` argparse flags. |
| `src/utils/megatron_args.py` | Modify | Emit `--poet-lie-ortho*` from `optim.poet.lie_ortho*`. |
| `src/patches/poet_optimizer_setup.py` | Modify | Copy `args.poet_lie_ortho*` → `config.poet_lie_ortho*`. |
| `configs/experiments/optim/poet_lie_orth.yaml` | Create | The experiment config (sibling of `poet_lie_rms.yaml`). |
| `docs/experiments/poet_lie_orth.md` | Create | Experiment doc — **required** by the `experiment-doc-exists` pre-commit hook (`tools/check_experiment_docs.py` maps `experiment.name` → `docs/experiments/<name>.md`). Missing it blocks the commit. |
| `scripts/train_poet_lie_orth.sh` | Create | Launch wrapper (copy of `train_poet_lie_rms.sh`, new experiment). |
| `tests/unit/test_poet_lie_orth.py` | Create | Unit tests for the helper + the optimizer mode. |
| `tests/unit/test_pretrain_gpt_slm.py` | Modify | Test the new argparse flags. |
| `tests/unit/test_megatron_args.py` | Modify | Test argv emission + the new YAML. |
| `tests/unit/test_patch_poet_optimizer_setup.py` | Modify | Test the config copy. |
| `tests/unit/test_train_scripts.py` | Modify | Smoke-test the new script. |
| `CHANGELOG.md` | Modify | Log the change. |

---

## Task 1: Orthonormalization helper `orthonormalize_skew_unit`

Maps a batch of `b×b` skew blocks to skew blocks whose nonzero singular values are all 1 — i.e. every rotation plane turns at unit rate (docs §5). Two methods: the doc-recommended `spectral` (`A·(−A²)^{-1/2}`, stays skew exactly) and `ns_reskew` (Muon's polar Newton–Schulz then re-skew, reusing the existing kernel).

**Files:**
- Modify: `src/optim/poet_skew_muon.py` (add the function after `orthogonalize_skew_blocks`, ~line 39)
- Test: `tests/unit/test_poet_lie_orth.py` (Create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_poet_lie_orth.py`:

```python
"""Tests for the POET Lie-Orth (Muon-like orthogonalizing) optimizer:
the orthonormalization helper and the ``ortho`` mode of LieAlgebraMomentum.
See docs/muon_orthogonalizing_optimizer_poet.md."""

import math

import pytest
import torch
import torch.nn as nn

from src.diag.skew_conditioning import block_spectral_stats, skew_to_vec, vec_to_skew
from src.optim.poet_lie_momentum import LieAlgebraMomentum
from src.optim.poet_skew_muon import orthonormalize_skew_unit


def _heavy_tailed_skew(num_blocks, b, seed):
    torch.manual_seed(seed)
    ne = b * (b - 1) // 2
    v = torch.randn(num_blocks, ne)
    v[:, :2] *= 5.0  # heavy-tailed but full-rank
    return vec_to_skew(v, b)


@pytest.mark.parametrize("method", ["spectral", "ns_reskew"])
def test_orthonormalize_makes_singular_values_uniform(method):
    M = _heavy_tailed_skew(2, 8, seed=0)
    cond_in = block_spectral_stats(M)["condition_number"].mean().item()
    X = orthonormalize_skew_unit(M, method=method, ns_steps=12)
    sv = torch.linalg.svdvals(X)  # (num_blocks, b)
    assert cond_in > 5.0  # heavy-tailed input
    # all singular values driven to ~1 (every plane rotates at unit rate)
    assert torch.allclose(sv, torch.ones_like(sv), atol=0.05), sv


@pytest.mark.parametrize("method", ["spectral", "ns_reskew"])
def test_orthonormalize_stays_skew(method):
    M = _heavy_tailed_skew(3, 8, seed=1)
    X = orthonormalize_skew_unit(M, method=method, ns_steps=12)
    assert torch.allclose(X, -X.transpose(-2, -1), atol=1e-5)


def test_orthonormalize_is_odd_in_sign():
    # orthonormalization is an odd function: orth(-A) == -orth(A) (spectral form).
    M = _heavy_tailed_skew(2, 8, seed=2)
    pos = orthonormalize_skew_unit(M, method="spectral", ns_steps=12)
    neg = orthonormalize_skew_unit(-M, method="spectral", ns_steps=12)
    assert torch.allclose(neg, -pos, atol=1e-5)


def test_orthonormalize_unit_generator_2d():
    # a single 2D plane [[0, t], [-t, 0]] -> [[0, 1], [-1, 0]] regardless of t>0.
    t = 3.7
    M = torch.tensor([[[0.0, t], [-t, 0.0]]])
    X = orthonormalize_skew_unit(M, method="spectral", ns_steps=12)
    expected = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    assert torch.allclose(X, expected, atol=1e-4), X


def test_orthonormalize_batches_per_block_independently():
    a = _heavy_tailed_skew(1, 8, seed=3)
    c = _heavy_tailed_skew(1, 8, seed=4)
    stacked = torch.cat([a, c], dim=0)
    out_stacked = orthonormalize_skew_unit(stacked, method="spectral", ns_steps=12)
    out_a = orthonormalize_skew_unit(a, method="spectral", ns_steps=12)
    out_c = orthonormalize_skew_unit(c, method="spectral", ns_steps=12)
    assert torch.allclose(out_stacked[0:1], out_a, atol=1e-6)
    assert torch.allclose(out_stacked[1:2], out_c, atol=1e-6)
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -k orthonormalize -v`
Expected: FAIL with `ImportError: cannot import name 'orthonormalize_skew_unit'`.

- [ ] **Step 3: Implement `orthonormalize_skew_unit`**

In `src/optim/poet_skew_muon.py`, insert this function immediately after `orthogonalize_skew_blocks` (after its `return X` at line 38):

```python
def orthonormalize_skew_unit(
    skew: torch.Tensor,
    method: str = "spectral",
    ns_steps: int = 8,
    eps: float = 1e-7,
    reg: float = 1e-6,
) -> torch.Tensor:
    """Map a batch of skew blocks (num_blocks, b, b) to skew blocks whose nonzero
    singular values are all 1 -- every rotation plane turns at unit rate
    (docs/muon_orthogonalizing_optimizer_poet.md SS5). Result stays in so(b).

    method="spectral" (recommended): A_orth = A (-A^2)^{-1/2}. Because this is an
    ODD function of A it stays skew exactly (no re-skew artifact). (-A^2)^{-1/2}
    is computed by a coupled Newton-Schulz inverse-sqrt iteration on the SPD
    matrix C = -A^2 = A^T A.
    method="ns_reskew": Muon's polar Newton-Schulz on A (orthogonalize_skew_blocks)
    then re-skew. Cheaper / reuses the existing kernel, but the re-skew projection
    nudges the singular values slightly off 1.
    """
    A = skew.float()
    if method == "ns_reskew":
        X = orthogonalize_skew_blocks(A, ns_steps)
        return 0.5 * (X - X.transpose(-2, -1))
    if method != "spectral":
        raise ValueError(f"orthonormalize_skew_unit method must be 'spectral' or "
                         f"'ns_reskew', got {method!r}")
    b = A.shape[-1]
    eye = torch.eye(b, dtype=A.dtype, device=A.device).expand_as(A)
    C = -(A @ A)  # = A^T A, symmetric PSD; eigenvalues are A's squared sing. values
    # Normalize so C's spectrum sits in (0, 1] (convergence needs it in (0, 2)),
    # plus a tiny ridge so an odd-dim null direction stays finite (A kills it anyway).
    s = torch.linalg.matrix_norm(C, ord="fro", dim=(-2, -1), keepdim=True) + eps
    Cn = C / s + reg * eye
    Y, Z = Cn, eye.clone()
    for _ in range(ns_steps):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T  # -> Cn^{1/2}
        Z = T @ Z  # -> Cn^{-1/2}
    inv_sqrt_C = Z / s.sqrt()  # Cn^{-1/2} = sqrt(s) * C^{-1/2}  =>  C^{-1/2} = Z/sqrt(s)
    A_orth = A @ inv_sqrt_C
    return 0.5 * (A_orth - A_orth.transpose(-2, -1))  # clean up float residue
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -k orthonormalize -v`
Expected: 8 passed (4 test functions × parametrizations).

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_skew_muon.py tests/unit/test_poet_lie_orth.py
git commit -m "feat(poet): add orthonormalize_skew_unit (spectral + ns_reskew) for lie-orth optimizer"
```

---

## Task 2: `ortho` mode in `LieAlgebraMomentum`

Add a new mode to the optimizer's skew branch: orthonormalize the direction, scale by `ortho_c`, write to `oft_R`. First-moment-only by default. Mutually exclusive with `rms`.

**Files:**
- Modify: `src/optim/poet_lie_momentum.py` (imports ~line 24; `__init__` ~lines 79-117; `step` skew branch ~lines 161-173)
- Test: `tests/unit/test_poet_lie_orth.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_poet_lie_orth.py`:

```python
def _make_ortho_opt(p, lr, ortho_c, **kw):
    return LieAlgebraMomentum(
        [dict(params=[p], use_skew=True, side="out", lr=lr)],
        b1=0.9, b2=0.95, eps=1e-8,
        ortho=True, ortho_c=ortho_c, ortho_method="spectral",
        ortho_ns_steps=12, **kw,
    )


def test_ortho_all_planes_rotate_by_lr_times_c():
    # p born at 0 (identity rotation); one step -> every plane's angle == lr*ortho_c.
    b, ne, lr, c = 8, 8 * 7 // 2, 0.1, 0.05
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    p.grad[:, :2] *= 5.0  # heavy-tailed grad: RMS path would NOT equalize, ortho does
    _make_ortho_opt(p, lr, c).step()
    sv = torch.linalg.svdvals(vec_to_skew(p.data, b))  # all == lr*c if equalized
    assert torch.allclose(sv, torch.full_like(sv, lr * c), atol=lr * c * 0.06), sv


def test_ortho_and_rms_are_mutually_exclusive():
    p = nn.Parameter(torch.zeros(1, 6))
    with pytest.raises(ValueError, match="mutually exclusive"):
        LieAlgebraMomentum(
            [dict(params=[p], use_skew=True, lr=1e-3)],
            ortho=True, rms=True,
        )


def test_ortho_first_moment_only_differs_from_second_moment():
    # With a wildly uneven per-entry grad, the second-moment (Adam) direction and
    # the first-moment-only direction point differently before orthonormalization,
    # so the written rotations differ.
    b, ne, lr, c = 8, 8 * 7 // 2, 0.1, 0.05
    g = torch.randn(1, ne)
    g[:, 0] *= 50.0
    p1 = nn.Parameter(torch.zeros(1, ne)); p1.grad = g.clone()
    p2 = nn.Parameter(torch.zeros(1, ne)); p2.grad = g.clone()
    _make_ortho_opt(p1, lr, c, ortho_use_second_moment=False).step()
    _make_ortho_opt(p2, lr, c, ortho_use_second_moment=True).step()
    assert not torch.allclose(p1.data, p2.data, atol=1e-4)


def test_ortho_grad_sign_flips_the_update():
    # Orthonormalization is odd, so negating the grad negates the written oft_R.
    b, ne, lr, c = 8, 8 * 7 // 2, 0.1, 0.05
    g = torch.randn(1, ne)
    p_pos = nn.Parameter(torch.zeros(1, ne)); p_pos.grad = g.clone()
    p_neg = nn.Parameter(torch.zeros(1, ne)); p_neg.grad = -g.clone()
    _make_ortho_opt(p_pos, lr, c).step()
    _make_ortho_opt(p_neg, lr, c).step()
    assert torch.allclose(p_pos.data, -p_neg.data, atol=1e-5)


def test_ortho_keeps_valid_skew_vector_shape_and_finite():
    ne = 8 * 7 // 2
    p = nn.Parameter(torch.zeros(3, ne))
    p.grad = torch.randn(3, ne)
    _make_ortho_opt(p, 0.1, 0.05).step()
    assert p.data.shape == (3, ne)
    assert torch.isfinite(p.data).all()
```

- [ ] **Step 2: Run the optimizer tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -k ortho_ -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'ortho'`.

- [ ] **Step 3a: Add the imports**

In `src/optim/poet_lie_momentum.py`, replace the import line 24:

```python
from src.diag.skew_conditioning import block_size_from_nelems
```

with:

```python
from src.diag.skew_conditioning import block_size_from_nelems, skew_to_vec, vec_to_skew
from src.optim.poet_skew_muon import orthonormalize_skew_unit
```

- [ ] **Step 3b: Add the `__init__` kwargs and validation**

In `LieAlgebraMomentum.__init__`, add these parameters to the signature (after `rms_c: float = 0.2,`, before `adamw_betas=(0.9, 0.95),`):

```python
        ortho: bool = False,
        ortho_c: float = 0.01,
        ortho_method: str = "spectral",
        ortho_ns_steps: int = 8,
        ortho_use_second_moment: bool = False,
```

Then, immediately after the existing two lines

```python
        self.rms = bool(rms)
        self.rms_c = float(rms_c)
```

add:

```python
        # Muon-like orthogonalizing mode (docs/muon_orthogonalizing_optimizer_poet.md):
        # orthonormalize the direction so ALL rotation planes turn by the same angle
        # (= lr * ortho_c). Mutually exclusive with rms (both transform the direction).
        self.ortho = bool(ortho)
        if self.ortho and self.rms:
            raise ValueError(
                "lie_ortho and lie_rms are mutually exclusive: both replace the "
                "direction->generator transform."
            )
        if ortho_method not in ("spectral", "ns_reskew"):
            raise ValueError(
                f"ortho_method must be 'spectral' or 'ns_reskew', got {ortho_method!r}"
            )
        self.ortho_c = float(ortho_c)
        self.ortho_method = ortho_method
        self.ortho_ns_steps = int(ortho_ns_steps)
        self.ortho_use_second_moment = bool(ortho_use_second_moment)
```

- [ ] **Step 3c: Add the `ortho` branch in `step`**

In `LieAlgebraMomentum.step`, replace this block (currently lines ~161-173):

```python
                    A = -m / (v.sqrt() + eps)
                    if self.rms:
                        # Stage 2 (W-free), PER BLOCK: normalize each block's
                        # generator so its per-plane angle is dimension-consistent.
                        # dim_const = sqrt(block_size); block_norm reduces over the
                        # n_elems axis only -> alpha is (n_blocks, 1). Identical to
                        # the old global formula when n_blocks == 1.
                        bsz = block_size_from_nelems(A.shape[1])
                        dim_const = bsz**0.5
                        block_norm = torch.linalg.norm(A, dim=1, keepdim=True)
                        alpha = self.rms_c * dim_const / (block_norm + eps)
                        A = A * alpha
                    p.add_(A.to(p.dtype), alpha=lr)  # p born at 0 -> p = lr*(alpha)A
```

with:

```python
                    if self.ortho:
                        # Muon-like: orthonormalize the DIRECTION (per b x b block)
                        # so every rotation plane turns by the same angle. After
                        # orthonormalization all singular values = 1, so the RMS
                        # normalizer is automatically 1 (docs SS3) and ortho_c IS the
                        # per-plane angle. First-moment-only by default: a second
                        # moment is partially undone by orthonormalization (docs SS4).
                        # Realized per-plane angle = lr * ortho_c.
                        A_dir = -m / (v.sqrt() + eps) if self.ortho_use_second_moment else -m
                        bsz = block_size_from_nelems(A_dir.shape[1])
                        M = vec_to_skew(A_dir, bsz)  # (n_blocks, b, b)
                        M = orthonormalize_skew_unit(
                            M, method=self.ortho_method, ns_steps=self.ortho_ns_steps
                        )
                        gen = skew_to_vec(self.ortho_c * M, bsz)  # (n_blocks, n_elems)
                        p.add_(gen.to(p.dtype), alpha=lr)
                    else:
                        A = -m / (v.sqrt() + eps)
                        if self.rms:
                            # Stage 2 (W-free), PER BLOCK: normalize each block's
                            # generator so its per-plane angle is dimension-consistent.
                            # dim_const = sqrt(block_size); block_norm reduces over the
                            # n_elems axis only -> alpha is (n_blocks, 1). Identical to
                            # the old global formula when n_blocks == 1.
                            bsz = block_size_from_nelems(A.shape[1])
                            dim_const = bsz**0.5
                            block_norm = torch.linalg.norm(A, dim=1, keepdim=True)
                            alpha = self.rms_c * dim_const / (block_norm + eps)
                            A = A * alpha
                        p.add_(A.to(p.dtype), alpha=lr)  # p born at 0 -> p = lr*(alpha)A
```

- [ ] **Step 4: Run the optimizer tests + the whole new file to verify pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -v`
Expected: all pass (Task 1 + Task 2 tests).

Also confirm the RMS path is untouched:
Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_momentum.py -v`
Expected: all pass (unchanged behavior).

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_lie_momentum.py tests/unit/test_poet_lie_orth.py
git commit -m "feat(poet): add ortho mode to LieAlgebraMomentum (orthonormalize direction, equal-angle planes)"
```

---

## Task 3: Wire the new knobs into the optimizer builder

`get_megatron_poet_lie_momentum_optimizer` must read `config.poet_lie_ortho*` and pass to `LieAlgebraMomentum`. This builder needs Megatron handles, so it is not CPU-unit-testable; verification is `py_compile` plus the downstream arg-plumbing tests (Tasks 5-6) and the GPU smoke (Task 8).

**Files:**
- Modify: `src/optim/poet.py` (`get_megatron_poet_lie_momentum_optimizer`, the log call ~lines 563-578 and the `LieAlgebraMomentum(...)` call ~lines 585-598)

- [ ] **Step 1: Extend the log line**

In `src/optim/poet.py`, in the `logger.info(...)` call inside `get_megatron_poet_lie_momentum_optimizer`, change the format string ending `"rms=%s, rms_c=%s)"` to `"rms=%s, rms_c=%s, ortho=%s, ortho_c=%s, ortho_method=%s)"` and append three args after the existing `getattr(config, "poet_lie_rms_c", 0.2),` line:

```python
        getattr(config, "poet_lie_ortho", False),
        getattr(config, "poet_lie_ortho_c", 0.01),
        getattr(config, "poet_lie_ortho_method", "spectral"),
```

- [ ] **Step 2: Pass the new kwargs to the optimizer**

In the same function, in the `optimizer = LieAlgebraMomentum(...)` constructor call, add after `rms_c=getattr(config, "poet_lie_rms_c", 0.2),`:

```python
        ortho=getattr(config, "poet_lie_ortho", False),
        ortho_c=getattr(config, "poet_lie_ortho_c", 0.01),
        ortho_method=getattr(config, "poet_lie_ortho_method", "spectral"),
        ortho_ns_steps=getattr(config, "poet_lie_ortho_ns_steps", 8),
        ortho_use_second_moment=getattr(config, "poet_lie_ortho_use_second_moment", False),
```

- [ ] **Step 3: Verify the module still compiles**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/optim/poet.py && echo OK`
Expected: `OK` (no syntax errors).

- [ ] **Step 4: Commit**

```bash
git add src/optim/poet.py
git commit -m "feat(poet): thread lie_ortho config knobs into the Lie-momentum builder"
```

---

## Task 4: argparse flags

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` (`add_slm_args`, after the `--poet-lie-rms-c` line ~91)
- Test: `tests/unit/test_pretrain_gpt_slm.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_lie_ortho_flags():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        [
            "--slm-config-path", "x.yaml",
            "--poet-lie-ortho",
            "--poet-lie-ortho-c", "0.02",
            "--poet-lie-ortho-method", "spectral",
            "--poet-lie-ortho-ns-steps", "10",
            "--poet-lie-ortho-use-second-moment",
        ]
    )
    assert args.poet_lie_ortho is True
    assert args.poet_lie_ortho_c == 0.02
    assert args.poet_lie_ortho_method == "spectral"
    assert args.poet_lie_ortho_ns_steps == 10
    assert args.poet_lie_ortho_use_second_moment is True


def test_lie_ortho_flags_default_off():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml"])
    assert args.poet_lie_ortho is False
    assert args.poet_lie_ortho_c == 0.01
    assert args.poet_lie_ortho_method == "spectral"
    assert args.poet_lie_ortho_ns_steps == 8
    assert args.poet_lie_ortho_use_second_moment is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py -k lie_ortho -v`
Expected: FAIL with `unrecognized arguments: --poet-lie-ortho`.

- [ ] **Step 3: Add the argparse flags**

In `launchers/pretrain_gpt_slm.py`, immediately after the line `group.add_argument("--poet-lie-rms-c", type=float, default=0.2)` (~line 91), insert:

```python
    # Muon-like orthogonalizing variant (docs/muon_orthogonalizing_optimizer_poet.md):
    # orthonormalize the skew direction so every rotation plane turns by the same
    # angle (= lr * ortho_c). Sibling of --poet-lie-rms; the two are exclusive.
    group.add_argument("--poet-lie-ortho", action="store_true")
    group.add_argument("--poet-lie-ortho-c", type=float, default=0.01)
    group.add_argument(
        "--poet-lie-ortho-method", choices=["spectral", "ns_reskew"], default="spectral"
    )
    group.add_argument("--poet-lie-ortho-ns-steps", type=int, default=8)
    group.add_argument("--poet-lie-ortho-use-second-moment", action="store_true")
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py -k lie_ortho -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add launchers/pretrain_gpt_slm.py tests/unit/test_pretrain_gpt_slm.py
git commit -m "feat(poet): add --poet-lie-ortho* argparse flags"
```

---

## Task 5: Emit the flags from the experiment YAML (`megatron_args.py`)

**Files:**
- Modify: `src/utils/megatron_args.py` (`_optimizer_args`, `kind == "poet"`: add to the `poet_args` list ~line 312, and the store_true block ~line 332)
- Test: `tests/unit/test_megatron_args.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py` (the `_poet_cfg` helper and `_optimizer_args` are already imported/defined in this file):

```python
def test_poet_argv_emits_lie_ortho():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "q_optimizer": "lie_algebra",
                "lie_ortho": True,
                "lie_ortho_c": 0.02,
                "lie_ortho_method": "spectral",
                "lie_ortho_ns_steps": 10,
                "lie_ortho_use_second_moment": True,
            }
        )
    )
    assert "--poet-lie-ortho" in args
    assert args[args.index("--poet-lie-ortho-c") + 1] == "0.02"
    assert args[args.index("--poet-lie-ortho-method") + 1] == "spectral"
    assert args[args.index("--poet-lie-ortho-ns-steps") + 1] == "10"
    assert "--poet-lie-ortho-use-second-moment" in args


def test_poet_argv_omits_lie_ortho_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-lie-ortho" not in args
    assert "--poet-lie-ortho-use-second-moment" not in args
    assert args[args.index("--poet-lie-ortho-c") + 1] == "0.01"
    assert args[args.index("--poet-lie-ortho-method") + 1] == "spectral"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k lie_ortho -v`
Expected: FAIL — `--poet-lie-ortho-c` not in args (`ValueError: ... is not in list`).

- [ ] **Step 3a: Add the value-carrying flags to `poet_args`**

In `src/utils/megatron_args.py`, in the `kind == "poet"` block, inside the `poet_args = [ ... ]` list, immediately after the two lines:

```python
            "--poet-lie-rms-c",
            poet.get("lie_rms_c", 0.2),
```

insert:

```python
            "--poet-lie-ortho-c",
            poet.get("lie_ortho_c", 0.01),
            "--poet-lie-ortho-method",
            poet.get("lie_ortho_method", "spectral"),
            "--poet-lie-ortho-ns-steps",
            poet.get("lie_ortho_ns_steps", 8),
```

- [ ] **Step 3b: Add the store_true flags**

In the same block, immediately after:

```python
        # store_true: enable Stage 2 RMS scaling (W-free) for q_optimizer=lie_algebra.
        if poet.get("lie_rms", False):
            poet_args.append("--poet-lie-rms")
```

insert:

```python
        # store_true: Muon-like orthogonalizing variant for q_optimizer=lie_algebra.
        if poet.get("lie_ortho", False):
            poet_args.append("--poet-lie-ortho")
        if poet.get("lie_ortho_use_second_moment", False):
            poet_args.append("--poet-lie-ortho-use-second-moment")
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "lie_ortho or poet" -v`
Expected: new tests pass; existing poet tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): emit --poet-lie-ortho* argv from optim.poet.lie_ortho*"
```

---

## Task 6: Copy the args into the optimizer config (`poet_optimizer_setup.py`)

**Files:**
- Modify: `src/patches/poet_optimizer_setup.py` (`_wrapped_get_config`, after the `config.poet_lie_rms_c = ...` line ~53)
- Test: `tests/unit/test_patch_poet_optimizer_setup.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_patch_poet_optimizer_setup.py` (it already imports `importlib`, `sys`, `types` and defines `_reset_for_tests`):

```python
def test_get_config_copies_lie_ortho_flags(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.poet_optimizer_setup")

    fake_training = types.SimpleNamespace()

    def original_get_config(args):
        cfg = types.SimpleNamespace(optimizer="adam", lr=1e-3)
        return cfg, {}

    def original_get_optimizer(config, model, **kwargs):
        return "adam-optimizer"

    fake_training.get_megatron_optimizer_config = original_get_config
    fake_training.get_megatron_optimizer = original_get_optimizer

    fake_megatron = types.ModuleType("megatron")
    fake_megatron_training_pkg = types.ModuleType("megatron.training")
    fake_megatron_training_pkg.training = fake_training
    fake_megatron.training = fake_megatron_training_pkg
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", fake_megatron_training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", fake_training)

    patch_mod.apply()

    args = types.SimpleNamespace(
        slm_optimizer="poet",
        poet_merge_period=1,
        poet_scale=0.5,
        poet_block_size=256,
        poet_init_type="normalized",
        poet_mup_alpha=1.0,
        poet_lie_ortho=True,
        poet_lie_ortho_c=0.02,
        poet_lie_ortho_method="spectral",
        poet_lie_ortho_ns_steps=10,
        poet_lie_ortho_use_second_moment=True,
    )
    cfg, _ = fake_training.get_megatron_optimizer_config(args)
    assert cfg.poet_lie_ortho is True
    assert cfg.poet_lie_ortho_c == 0.02
    assert cfg.poet_lie_ortho_method == "spectral"
    assert cfg.poet_lie_ortho_ns_steps == 10
    assert cfg.poet_lie_ortho_use_second_moment is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_optimizer_setup.py -k lie_ortho -v`
Expected: FAIL with `AttributeError: 'types.SimpleNamespace' object has no attribute 'poet_lie_ortho'`.

- [ ] **Step 3: Add the config copies**

In `src/patches/poet_optimizer_setup.py`, in `_wrapped_get_config`, immediately after the line `config.poet_lie_rms_c = getattr(args, "poet_lie_rms_c", 0.2)`, insert:

```python
        config.poet_lie_ortho = getattr(args, "poet_lie_ortho", False)
        config.poet_lie_ortho_c = getattr(args, "poet_lie_ortho_c", 0.01)
        config.poet_lie_ortho_method = getattr(args, "poet_lie_ortho_method", "spectral")
        config.poet_lie_ortho_ns_steps = getattr(args, "poet_lie_ortho_ns_steps", 8)
        config.poet_lie_ortho_use_second_moment = getattr(
            args, "poet_lie_ortho_use_second_moment", False
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_optimizer_setup.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_optimizer_setup.py tests/unit/test_patch_poet_optimizer_setup.py
git commit -m "feat(poet): copy poet_lie_ortho* args into the optimizer config"
```

---

## Task 7: Experiment config + launch script

**Files:**
- Create: `configs/experiments/optim/poet_lie_orth.yaml`
- Create: `scripts/train_poet_lie_orth.sh`
- Test: `tests/unit/test_megatron_args.py` (append YAML test), `tests/unit/test_train_scripts.py` (append script smoke test)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_lie_orth_experiment_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet_lie_orth.yaml")
    assert cfg.experiment.name == "poet_lie_orth"
    assert cfg.optim.poet.q_optimizer == "lie_algebra"
    assert cfg.optim.poet.lie_ortho is True
    assert cfg.optim.poet.lie_rms is False
    assert cfg.optim.poet.lie_ortho_method == "spectral"
    assert cfg.optim.poet.lie_ortho_c == 4
```

Append to `tests/unit/test_train_scripts.py`:

```python
def test_poet_lie_orth_script_supports_llama3():
    proc = _run("train_poet_lie_orth.sh", "llama3")
    assert "--poet-q-optimizer" in proc.stdout and "lie_algebra" in proc.stdout
    assert "--poet-lie-ortho" in proc.stdout
    assert "--poet-lie-ortho-c" in proc.stdout
    assert "--poet-lie-ortho-method" in proc.stdout
```

- [ ] **Step 2: Run to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k poet_lie_orth_experiment tests/unit/test_train_scripts.py -k poet_lie_orth -v`
Expected: FAIL — YAML test `FileNotFoundError`; script test `subprocess ... No such file`.

- [ ] **Step 3a: Create the experiment YAML**

Create `configs/experiments/optim/poet_lie_orth.yaml`:

```yaml
# @package _global_
# poet_lie_orth: poet_lie + Muon-like orthogonalizing direction transform (the
# sibling of poet_lie_rms). See docs/muon_orthogonalizing_optimizer_poet.md.
#
# Identical Lie-momentum pipeline as poet_lie_rms (single-step, block_count=1,
# reinit_period=-1, cayley, head-aligned) EXCEPT the direction->generator step:
# instead of RMS scaling, orthonormalize the per-block skew direction so EVERY
# rotation plane turns by the same angle. lie_rms is OFF; lie_ortho is ON (the
# two are mutually exclusive). First-moment-only by default (a second moment is
# partially undone by orthonormalization).
#
# Angle convention: realized per-plane angle = group_lr * lie_ortho_c (same as
# the rms path's lr * rms_c). lr=0.003, lie_ortho_c=4 -> ~0.012 rad/plane, the
# same effective angle as the rms "best" run, for a fair head-to-head.
experiment:
  name: poet_lie_orth
  family: optim
  description: |
    POET x Muon: orthonormalize the Lie-algebra momentum direction so all rotation
    planes turn by the same angle c (discards the gradient's relative per-plane
    magnitudes, keeps only the subspace). Sibling of poet_lie_rms; same single-step
    POET stack (merge_period=1, block_count=1, reinit_period=-1, cayley). Run
    head-to-head vs poet_lie_rms to test whether relative per-plane angles are
    signal or noise for rotational updates (docs SS7).
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
    lie_rms: false             # OFF: ortho replaces the RMS-norm transform
    lie_ortho: true            # Muon-like: equalize per-plane angles
    lie_ortho_c: 4             # base per-plane angle; realized angle = lr * c
    lie_ortho_method: spectral # 'spectral' (A(-A^2)^-1/2, stays skew) | 'ns_reskew'
    lie_ortho_ns_steps: 8
    lie_ortho_use_second_moment: false  # first-moment-only by default (docs SS4)
    head_aligned_attn: true    # rotate q/k/v/o per attention head (requires unfuse_qkv=true)
    train_output_rotation: true

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 3b: Create the launch script**

Create `scripts/train_poet_lie_orth.sh` (copy of `scripts/train_poet_lie_rms.sh` with the header comment and `experiment=` changed). Full content:

```bash
#!/usr/bin/env bash
set -euo pipefail

# poet_lie_orth variant: same harness as train_poet_lie_rms.sh, but uses
# experiment=optim/poet_lie_orth — POET x Muon orthogonalizing optimizer.
# Instead of the RMS-norm transform, it orthonormalizes the Lie-algebra momentum
# direction so EVERY rotation plane turns by the same angle (= lr*lie_ortho_c).
# Single-step (merge_period=1), reinit_period=-1, block_count=1. "$@" override wins.

# torchtitan is AdamW-only in milestone 1; reject --backend torchtitan here so the
# same flag fails fast on this non-AdamW wrapper (see scripts/train_adam.sh).
case " $* " in
  *" --backend torchtitan "*|*" --backend=torchtitan "*)
    echo "This optimizer is not yet supported on torchtitan (milestone 1 is AdamW only)." >&2
    exit 2 ;;
esac

# Auto-source the cluster env loader so the user doesn't have to remember.
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

ARCH="${1:-llama3}"
if [[ "${ARCH}" == "llama3" || "${ARCH}" == "deepseek_v3" ]]; then
  shift || true
else
  ARCH="llama3"
fi

case "${ARCH}" in
  llama3)
    FAMILY="llama3"
    DEFAULT_SCALE="60m"            # tiny dev scale; override with base/scale=...
    ;;
  deepseek_v3)
    FAMILY="deepseek_v3"
    DEFAULT_SCALE="deepseek_v3_proxy_small"
    ;;
  *)
    echo "Unknown architecture: ${ARCH}. Use llama3 or deepseek_v3." >&2
    exit 2
    ;;
esac

# Inject debug defaults unless overridden on the command line.
USER_SET_SCALE="no"
USER_SET_SEQ="no"
USER_SET_SCHED="no"
USER_SET_REGIME="no"
for arg in "$@"; do
  case "${arg}" in
    base/scale=*) USER_SET_SCALE="yes" ;;
    base.model.seq_length=*) USER_SET_SEQ="yes" ;;
    scheduler=*) USER_SET_SCHED="yes" ;;
    training_regime=*) USER_SET_REGIME="yes" ;;
  esac
done

SCALE_ARGS=()
if [[ "${USER_SET_SCALE}" == "no" && -n "${DEFAULT_SCALE}" ]]; then
  SCALE_ARGS=("base/scale=${DEFAULT_SCALE}")
fi

REGIME_ARGS=()
if [[ "${USER_SET_REGIME}" == "no" ]]; then
  REGIME_ARGS=("training_regime=ablation_40x")
fi

SEQ_ARGS=()
if [[ "${USER_SET_SEQ}" == "no" ]]; then
  SEQ_ARGS=("base.model.seq_length=256")
fi

SCHED_ARGS=()
if [[ "${USER_SET_SCHED}" == "no" ]]; then
  SCHED_ARGS=("scheduler=cosine_poet")
fi

python -m launchers.train_megatron \
  "base/family=${FAMILY}" \
  "${SCALE_ARGS[@]}" \
  "${REGIME_ARGS[@]}" \
  "${SEQ_ARGS[@]}" \
  "${SCHED_ARGS[@]}" \
  "cluster=h100_de" \
  "experiment=optim/poet_lie_orth" \
  "training.global_batch_size=1024" \
  "training.micro_batch_size=128" \
  "base.model.transformer_impl=local" \
  "training.save_enabled=true" \
  "base.model.tie_embeddings=false" \
  "optim.weight_decay=0.1" \
  "wandb.project=slm-zeju-dev" \
  "$@"
```

- [ ] **Step 3c: Make the script executable**

Run: `chmod +x scripts/train_poet_lie_orth.sh && echo OK`
Expected: `OK`.

- [ ] **Step 3d: Create the experiment doc (required by the pre-commit hook)**

The `experiment-doc-exists` hook (`tools/check_experiment_docs.py`) fails the commit unless `docs/experiments/poet_lie_orth.md` exists. Create it:

```markdown
# poet_lie_orth — Lie momentum + Muon-like orthogonalizing transform

Sibling of [`poet_lie_rms`](./poet_lie_rms.md). Same single-step POET Lie-momentum
stack (`q_optimizer=lie_algebra`, `merge_period=1`, `block_count=1`,
`reinit_period=-1`, `cayley`, head-aligned), but the direction→generator transform
is **orthonormalization** instead of RMS scaling, per
[docs/muon_orthogonalizing_optimizer_poet.md](../muon_orthogonalizing_optimizer_poet.md).

After the (first-moment) Lie direction `A`, the optimizer orthonormalizes each
`b×b` skew block (all singular values → 1) and scales by `c`, so **every rotation
plane turns by the same angle**:

```
A_orth = A·(−A²)^{-1/2}          # stays skew; all σ = 1 (docs §5)
oft_R  = lr · c · A_orth          # realized per-plane angle = lr · lie_ortho_c
```

This discards the gradient's *relative* per-plane magnitudes (keeps only the
subspace) — Muon's bet, applied to rotational updates. `lie_rms` is OFF (the two
transforms are mutually exclusive). First-moment-only by default (a second moment
is partially undone by orthonormalization, docs §4). `lie_ortho_method` selects
`spectral` (the `A(−A²)^{-1/2}` form, default) or `ns_reskew` (polar Newton–Schulz
then re-skew). Run head-to-head vs `poet_lie_rms` to test whether the gradient's
relative per-plane angles are signal or noise for rotational updates (docs §7).
```

- [ ] **Step 4: Run to verify both pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k poet_lie_orth_experiment tests/unit/test_train_scripts.py -k poet_lie_orth -v`
Expected: 2 passed. (The script test runs `bash scripts/train_poet_lie_orth.sh llama3 --dry-run ...` and inspects the printed argv.)

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/optim/poet_lie_orth.yaml docs/experiments/poet_lie_orth.md scripts/train_poet_lie_orth.sh tests/unit/test_megatron_args.py tests/unit/test_train_scripts.py
git commit -m "feat(poet): add poet_lie_orth experiment config + train script"
```

---

## Task 8: Full unit sweep, CHANGELOG, and the GPU smoke command

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Run the full POET unit test sweep**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_lie_orth.py \
  tests/unit/test_poet_lie_momentum.py \
  tests/unit/test_poet_skew_muon.py \
  tests/unit/test_pretrain_gpt_slm.py \
  tests/unit/test_megatron_args.py \
  tests/unit/test_patch_poet_optimizer_setup.py \
  tests/unit/test_train_scripts.py -v
```
Expected: all pass. If any fail, fix before continuing — do not proceed past a red bar.

- [ ] **Step 2: Update CHANGELOG**

Add an entry at the top of the current section of `CHANGELOG.md` (match the file's existing format):

```markdown
- feat(poet): add the Muon-like orthogonalizing optimizer (`q_optimizer=lie_algebra` + `lie_ortho=true`) as a sibling of the Lie-RMS optimizer. Orthonormalizes the skew update direction (`orthonormalize_skew_unit`, spectral `A(-A^2)^{-1/2}` or `ns_reskew`) so every rotation plane turns by the same angle (`= lr * lie_ortho_c`); first-moment-only by default. New experiment `optim/poet_lie_orth` + `scripts/train_poet_lie_orth.sh` for the head-to-head vs `poet_lie_rms` (docs/muon_orthogonalizing_optimizer_poet.md).
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(poet): changelog for the lie-orth orthogonalizing optimizer"
```

- [ ] **Step 4: GPU smoke test — HAND OFF TO THE USER (do NOT run)**

This needs a GPU + the cluster env; it is the user's to run. Provide this command and stop:

```
codexlog poet_lie_orth_smoke bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 \
  optim.poet.lie_ortho_c=4
```

Expected sanity signals (from earlier POET runs): the `[POET] Lie-momentum: ... ortho=True, ortho_c=4.0, ortho_method=spectral)` log line appears at startup; step-1/2 run without OOM or NaN; loss decreases. The head-to-head experiment then compares this against the existing `poet_lie_rms_best` run:

```
codexlog poet_lie_orth_best bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 \
  optim.poet.lie_ortho_c=4
```

---

## Self-Review: spec coverage

| Doc section | Covered by |
|---|---|
| §1 one-line idea (equalize plane angles) | Task 1 (σ→1) + Task 2 (`ortho_c` = per-plane angle) |
| §2 composes with pipeline (momentum before ortho, small-angle CNP after) | Task 2 step-branch: orthonormalize `−m` (the momentum *result*), scale by small `ortho_c`, then the existing merge applies CNP/exp. Momentum buffers (`lie_m`) untouched. |
| §3 RMS folds in for free (σ=1 ⇒ normalizer 1) | Task 1: `orthonormalize_skew_unit` sets σ→1, no `√b`/`‖A‖_F`; Task 2 scales by `ortho_c` directly. |
| §4 per-step update; first-moment-only default | Task 2: `ortho_use_second_moment=False` default ⇒ `A_dir = −m`; flag enables `−m/(√v+eps)`. |
| §5 orthogonalizing a skew matrix (both methods) | Task 1: `spectral` (`A(−A²)^{-1/2}`) default + `ns_reskew` (reuse `orthogonalize_skew_blocks`). |
| §6 block-diagonal / per-head | Task 1: `vec_to_skew` yields `(n_blocks, b, b)`, all ops batch over `n_blocks`; Task 2 derives `b` per param. `head_aligned_attn=true` in the YAML. |
| §7 RMS vs ortho experiment | Task 7: `poet_lie_orth` config + script, `lr=0.003, c=4` mirrors the rms best run; Task 8 head-to-head commands. |
| §8 summary (separate optimizer, no RMS, c=angle) | Whole plan; `ortho`/`rms` mutual exclusion enforced in Task 2. |

**Placeholder scan:** none — every code step is complete and every command has expected output.

**Type/name consistency:** the naming table at the top is used verbatim in every task. Helper `orthonormalize_skew_unit(skew, method, ns_steps, eps, reg)`; optimizer kwargs `ortho/ortho_c/ortho_method/ortho_ns_steps/ortho_use_second_moment`; config/arg `poet_lie_ortho*`; argv `--poet-lie-ortho*`; YAML `lie_ortho*`. Defaults match across argparse (`0.01/spectral/8/False`), `getattr` fallbacks in Tasks 3/5/6, and the optimizer signature.
