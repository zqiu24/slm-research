# POET Lie-Orth (Muon-like Orthogonalizing) Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a **standalone** Muon-like *orthogonalizing* optimizer — a sibling of the POET Lie-RMS optimizer that orthogonalizes the skew update direction so the rotation planes turn by roughly the **same** angle (instead of RMS-scaling, which preserves the gradient's *relative* per-plane angles), to run the head-to-head experiment in `docs/muon_orthogonalizing_optimizer_poet.md` §7.

**Architecture:** A **new optimizer class `LieOrthMomentum` in its own file `src/optim/poet_lie_orth.py`**, selected by a new `q_optimizer=lie_ortho` value (parallel to `lie_algebra`/`muon`). It runs the same Lie-momentum pipeline as `LieAlgebraMomentum` (first-moment momentum on `oft_R` that persists across the merge, side-tagged groups, alternating single-sided update, AdamW branch for non-skew params) but replaces the direction→generator transform: instead of RMS-scaling, it **orthogonalizes** the per-block `b×b` skew direction then scales by `c`. It is a *standalone class* (not a mode flag on `LieAlgebraMomentum`); the two share only (a) the pure param-split/group helpers `_split_poet_lie_params` / `_build_lie_param_groups`, (b) the Megatron-wrapping builder, which branches on `q_optimizer` to instantiate the right class, and (c) the orthogonalization kernel.

**Two orthogonalization methods, default = Muon's (cheap, approximate):**
- **`muon` (default):** Muon's quintic Newton–Schulz on the direction (reuses the existing `orthogonalize_skew_blocks`), then a `½(X−Xᵀ)` float-dust cleanup. Newton–Schulz **preserves skew** on a skew input (every step is skew × commuting-symmetric, verified to ~1e-15). It democratizes the spectrum into a **band around 1** (cond ≈ 1.5, σ ∈ ~[0.68, 1.13]) in ~5 steps. Cheap; `c` is a *nominal* angle (realized median ≈ 0.75–1.0·c).
- **`spectral` (opt-in):** the exact Löwdin form `A·(−A²)^{-1/2}` — stays skew and drives **all** singular values to exactly 1, so every plane turns by *exactly* the same angle. Needs **~15–20 steps** (≈4× Muon's cost) and still cannot fully equalize a *very* ill-conditioned direction. The exact-angle ablation, not the default.

First-moment-only by default (docs §4: a second moment is partially undone by orthogonalization). The orthogonalization helper `orthogonalize_skew_direction` lives next to the existing `orthogonalize_skew_blocks`.

**Tech Stack:** PyTorch (`torch 2.11`, CPU-testable), Megatron-LM (vendored, not needed for unit tests), OmegaConf/Hydra experiment configs, pytest.

**CPU test runner (use this exact interpreter — base `python` lacks torch/omegaconf):**
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest <path> -v
```
Run all commands from the repo root `/lustre/fast/fast/zqiu/slm-research`.

> **Numbers in this plan are measured, not guessed.** Step counts (5 muon / 20 spectral), tolerances, and band bounds were all verified in the CPU env before writing. NS preserves skew to ~1e-15 every step; Muon's band plateaus by step 5; spectral reaches `max|σ−1| ≤ 0.006` at 20 steps on benign inputs; a single muon step gives median plane-angle ≈ 0.75–1.04·lr·c with cond < 1.6; a single spectral step gives σ = lr·c to 1e-6.

**Source doc:** `docs/muon_orthogonalizing_optimizer_poet.md`

---

## Background: the plumbing chain (read once before starting)

A POET hyperparameter flows through five layers. The new optimizer is **selected** by `q_optimizer=lie_ortho`; its knobs flow through each layer:

1. **Experiment YAML** `configs/experiments/optim/poet_lie_orth.yaml` — sets `optim.poet.q_optimizer=lie_ortho` and `optim.poet.lie_ortho_*`.
2. **argv builder** `src/utils/megatron_args.py` (`_optimizer_args`, `kind == "poet"`) — translates `optim.poet.lie_ortho_*` → `--poet-lie-ortho-*` argv.
3. **argparse** `launchers/pretrain_gpt_slm.py` (`add_slm_args`) — defines `--poet-lie-ortho-*` and adds `lie_ortho` to the `--poet-q-optimizer` choices.
4. **config copy** `src/patches/poet_optimizer_setup.py` (`_wrapped_get_config`) — copies `args.poet_lie_ortho_*` → `config.poet_lie_ortho_*`.
5. **builder** `src/optim/poet.py` (`get_megatron_poet_lie_momentum_optimizer`, reached for both `lie_algebra` and `lie_ortho`) — branches on `config.poet_q_optimizer` to build `LieAlgebraMomentum` or `LieOrthMomentum`.

The optimizer math is the standalone class `src/optim/poet_lie_orth.py` (`LieOrthMomentum.step`).

**Key data layout:** `oft_R` params have shape `(n_blocks, n_elems)` where `n_elems = b·(b−1)/2` (the strictly-upper-triangular entries of a `b×b` skew block). Conversions live in `src/diag/skew_conditioning.py`: `vec_to_skew(vec, b) → (n_blocks, b, b)`, `skew_to_vec(skew, b) → (n_blocks, n_elems)`, `block_size_from_nelems(n_elems) → b`.

**Naming used consistently across all tasks** (do not vary):
| Layer | Name |
|---|---|
| optimizer class | `LieOrthMomentum` (in `src/optim/poet_lie_orth.py`) |
| optimizer kwargs | `ortho_c`, `ortho_method`, `ortho_ns_steps`, `ortho_use_second_moment` (+ shared `b1`,`b2`,`eps`,`v_mode`,`alternating`,`alternate_every`,`adamw_*`) |
| selection | `q_optimizer = lie_ortho` |
| `config.*` / `args.*` | `poet_lie_ortho_c`, `poet_lie_ortho_method`, `poet_lie_ortho_ns_steps`, `poet_lie_ortho_use_second_moment` |
| argparse / argv flags | `--poet-lie-ortho-c`, `--poet-lie-ortho-method`, `--poet-lie-ortho-ns-steps`, `--poet-lie-ortho-use-second-moment` |
| YAML keys (`optim.poet.`) | `q_optimizer: lie_ortho`, `lie_ortho_c`, `lie_ortho_method`, `lie_ortho_ns_steps`, `lie_ortho_use_second_moment` |
| method values | `muon` (default) \| `spectral` |
| helper fn | `orthogonalize_skew_direction` (in `src/optim/poet_skew_muon.py`) |
| experiment | name `poet_lie_orth`, file `configs/experiments/optim/poet_lie_orth.yaml`, script `scripts/train_poet_lie_orth.sh` |

**The angle convention (state this in code comments):** the realized per-plane rotation angle is `group_lr × ortho_c`, *not* `ortho_c` alone — identical to how the RMS optimizer behaves (`angle = group_lr × rms_c`). The optimizer writes `oft_R ← lr · (ortho_c · A_orth)` and the scheduler decays `group_lr`. So `lr=0.003, lie_ortho_c=4` ⇒ ~0.012 rad/plane, mirroring the RMS best run. **Under `method=muon`, `ortho_c` is *nominal*** (the band gives ≈ 0.75–1.0·lr·c, input-dependent); under `method=spectral` it is exact.

**Known limitation to document + monitor:** "every plane the same angle" is approximate on real gradients — a *very* ill-conditioned momentum (cond ≳ several hundred) cannot be fully equalized at any step count (inherent to orthogonalization). **Recommend logging the post-orthogonalization condition number** as a health metric — `block_spectral_stats` / `muon_update_spectral_stats` already exist.

---

## File Structure

| File | Create / Modify | Responsibility |
|---|---|---|
| `src/optim/poet_skew_muon.py` | Modify | Add `orthogonalize_skew_direction(method=muon\|spectral)`. `muon` reuses `orthogonalize_skew_blocks`; `spectral` is the exact inverse-sqrt. |
| `src/optim/poet_lie_orth.py` | **Create** | The standalone `LieOrthMomentum` optimizer class. |
| `src/optim/poet.py` | Modify | Route `q_optimizer=lie_ortho` to the shared lie builder; branch there to construct `LieOrthMomentum`. |
| `launchers/pretrain_gpt_slm.py` | Modify | Add `lie_ortho` to `--poet-q-optimizer` choices; add `--poet-lie-ortho-*` knob flags. |
| `src/utils/megatron_args.py` | Modify | Emit `--poet-lie-ortho-*` from `optim.poet.lie_ortho_*`. |
| `src/patches/poet_optimizer_setup.py` | Modify | Copy `args.poet_lie_ortho_*` → `config.poet_lie_ortho_*`. |
| `configs/experiments/optim/poet_lie_orth.yaml` | Create | The experiment config (`q_optimizer: lie_ortho`). |
| `docs/experiments/poet_lie_orth.md` | Create | Experiment doc — **required** by the `experiment-doc-exists` pre-commit hook (`tools/check_experiment_docs.py` maps `experiment.name` → `docs/experiments/<name>.md`). Missing it blocks the commit. |
| `scripts/train_poet_lie_orth.sh` | Create | Launch wrapper (copy of `train_poet_lie_rms.sh`, new experiment). |
| `tests/unit/test_poet_lie_orth.py` | Create | Unit tests for the helper + the `LieOrthMomentum` optimizer. |
| `tests/unit/test_pretrain_gpt_slm.py` | Modify | Test the new argparse flags + q_optimizer choice. |
| `tests/unit/test_megatron_args.py` | Modify | Test argv emission + the new YAML. |
| `tests/unit/test_patch_poet_optimizer_setup.py` | Modify | Test the config copy. |
| `tests/unit/test_train_scripts.py` | Modify | Smoke-test the new script. |
| `CHANGELOG.md` | Modify | Log the change. |

---

## Task 1: Orthogonalization helper `orthogonalize_skew_direction`

Orthogonalizes a batch of `b×b` skew blocks so the rotation planes turn by ~the same angle (docs §5). Two methods: `muon` (default — Muon's quintic + cleanup, a band around 1) and `spectral` (opt-in — exact `A·(−A²)^{-1/2}`, σ→1).

**Files:**
- Modify: `src/optim/poet_skew_muon.py` (add the function after `orthogonalize_skew_blocks`, ~line 39)
- Test: `tests/unit/test_poet_lie_orth.py` (Create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_poet_lie_orth.py` (helper tests only — the optimizer import is added in Task 2):

```python
"""Tests for the POET Lie-Orth (Muon-like orthogonalizing) optimizer:
the orthogonalization helper and the LieOrthMomentum optimizer.
See docs/muon_orthogonalizing_optimizer_poet.md."""

import pytest
import torch
import torch.nn as nn

from src.diag.skew_conditioning import block_spectral_stats, skew_to_vec, vec_to_skew
from src.optim.poet_skew_muon import orthogonalize_skew_direction


def _benign_skew(num_blocks, b, seed):
    torch.manual_seed(seed)
    return vec_to_skew(torch.randn(num_blocks, b * (b - 1) // 2), b)


@pytest.mark.parametrize("method", ["muon", "spectral"])
def test_orthogonalize_skew_direction_stays_skew(method):
    M = _benign_skew(3, 8, seed=1)
    X = orthogonalize_skew_direction(M, method=method, ns_steps=20)
    assert torch.allclose(X, -X.transpose(-2, -1), atol=1e-5)


@pytest.mark.parametrize("method", ["muon", "spectral"])
def test_orthogonalize_skew_direction_batches_per_block(method):
    a = _benign_skew(1, 8, seed=3)
    c = _benign_skew(1, 8, seed=4)
    out = orthogonalize_skew_direction(torch.cat([a, c], dim=0), method=method, ns_steps=20)
    assert torch.allclose(out[0:1], orthogonalize_skew_direction(a, method=method, ns_steps=20), atol=1e-6)
    assert torch.allclose(out[1:2], orthogonalize_skew_direction(c, method=method, ns_steps=20), atol=1e-6)


def test_muon_method_democratizes_the_spectrum():
    # DEFAULT: Muon's quintic flattens a heavy-tailed spectrum into a BAND around 1
    # (condition number ~ 1.5) in ~5 steps. It does NOT drive sigma to exactly 1.
    M = _benign_skew(2, 8, seed=0)
    cond_in = block_spectral_stats(M)["condition_number"].mean().item()
    X = orthogonalize_skew_direction(M, method="muon", ns_steps=5)
    cond_out = block_spectral_stats(X)["condition_number"].mean().item()
    assert cond_in > 5.0  # non-trivial input
    assert cond_out < 2.0 and cond_out < cond_in / 3.0  # democratized into a band


def test_spectral_method_drives_singular_values_to_one():
    # OPT-IN exact variant: every singular value -> 1 (needs ~15-20 steps).
    M = _benign_skew(2, 8, seed=0)
    sv = torch.linalg.svdvals(orthogonalize_skew_direction(M, method="spectral", ns_steps=20))
    assert torch.allclose(sv, torch.ones_like(sv), atol=0.02), sv


def test_spectral_method_is_odd_and_exact_on_a_2d_plane():
    M = _benign_skew(2, 8, seed=2)
    assert torch.allclose(
        orthogonalize_skew_direction(-M, method="spectral", ns_steps=20),
        -orthogonalize_skew_direction(M, method="spectral", ns_steps=20),
        atol=1e-5,
    )
    t = 3.7  # a single 2D plane [[0,t],[-t,0]] -> the unit generator regardless of t>0
    M2 = torch.tensor([[[0.0, t], [-t, 0.0]]])
    X2 = orthogonalize_skew_direction(M2, method="spectral", ns_steps=20)
    assert torch.allclose(X2, torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]]), atol=1e-4), X2
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -v`
Expected: FAIL with `ImportError: cannot import name 'orthogonalize_skew_direction'`.

- [ ] **Step 3: Implement `orthogonalize_skew_direction`**

In `src/optim/poet_skew_muon.py`, insert this function immediately after `orthogonalize_skew_blocks` (after its `return X` at line 38):

```python
def orthogonalize_skew_direction(
    skew: torch.Tensor,
    method: str = "muon",
    ns_steps: int = 5,
    eps: float = 1e-7,
    reg: float = 1e-6,
) -> torch.Tensor:
    """Orthogonalize a batch of skew blocks (num_blocks, b, b) so the rotation
    planes turn by ~the same angle (docs/muon_orthogonalizing_optimizer_poet.md SS5).
    Result stays in so(b).

    method="muon" (default): Muon's quintic Newton-Schulz on the direction
    (orthogonalize_skew_blocks), then a 1/2(X - X^T) cleanup. NS preserves skew on a
    skew input, so the cleanup only removes ~1e-15 float dust. Democratizes the
    spectrum into a BAND around 1 (sigma ~ [0.7, 1.1]) in ~5 steps -- cheap; a band of
    roughly-equal angles may be all this needs. NOT sigma == 1.
    method="spectral": A_orth = A (-A^2)^{-1/2}, an ODD function of A so it stays
    skew exactly and drives ALL nonzero singular values to 1 (every plane the SAME
    angle). Exact but slow: needs ns_steps >= ~15 (coupled Newton-Schulz inverse-sqrt
    of C = -A^2 = A^T A), and still cannot fully equalize a near-rank-deficient block.
    """
    A = skew.float()
    if method == "muon":
        X = orthogonalize_skew_blocks(A, ns_steps)
        return 0.5 * (X - X.transpose(-2, -1))
    if method != "spectral":
        raise ValueError(f"orthogonalize_skew_direction method must be 'muon' or "
                         f"'spectral', got {method!r}")
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

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -v`
Expected: all pass (stays-skew ×2, batches ×2, muon-democratizes, spectral-σ→1, spectral-odd/2D).

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_skew_muon.py tests/unit/test_poet_lie_orth.py
git commit -m "feat(poet): add orthogonalize_skew_direction (muon band + spectral exact)"
```

---

## Task 2: The standalone `LieOrthMomentum` optimizer

A new optimizer class in its own file. Same Lie-momentum pipeline as `LieAlgebraMomentum` (first-moment momentum on `oft_R`, alternating side, AdamW branch), but the skew update orthogonalizes the direction and scales by `ortho_c`. First-moment-only by default.

**Files:**
- Create: `src/optim/poet_lie_orth.py`
- Test: `tests/unit/test_poet_lie_orth.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_poet_lie_orth.py` (add the optimizer import at the top of the file first — change the import block to also include `from src.optim.poet_lie_orth import LieOrthMomentum`):

```python
def _make_opt(p, lr, ortho_c, method="muon", ns_steps=5, **kw):
    return LieOrthMomentum(
        [dict(params=[p], use_skew=True, side="out", lr=lr)],
        b1=0.9, b2=0.95, eps=1e-8,
        ortho_c=ortho_c, ortho_method=method, ortho_ns_steps=ns_steps, **kw,
    )


def test_muon_equalizes_plane_angles_into_a_band():
    # DEFAULT (muon): one step from identity -> the written oft_R's per-plane angles
    # form a tight band (cond < 2) at ~ lr*ortho_c. Equalized, but not exactly equal.
    torch.manual_seed(0)
    b, ne, lr, c = 8, 8 * 7 // 2, 0.1, 0.05
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    _make_opt(p, lr, c).step()
    R = vec_to_skew(p.data, b)
    sv = torch.linalg.svdvals(R)
    cond = block_spectral_stats(R)["condition_number"].mean().item()
    assert cond < 2.0  # planes roughly equalized
    assert 0.5 * lr * c < sv.median().item() < 1.2 * lr * c  # magnitude ~ lr*c (a band)


def test_spectral_makes_every_plane_angle_equal():
    # OPT-IN exact variant: every plane angle == lr*ortho_c (needs ns_steps ~20).
    torch.manual_seed(0)
    b, ne, lr, c = 8, 8 * 7 // 2, 0.1, 0.05
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    _make_opt(p, lr, c, method="spectral", ns_steps=20).step()
    sv = torch.linalg.svdvals(vec_to_skew(p.data, b))
    assert torch.allclose(sv, torch.full_like(sv, lr * c), atol=lr * c * 0.05), sv


def test_invalid_ortho_method_raises():
    p = nn.Parameter(torch.zeros(1, 6))
    with pytest.raises(ValueError, match="ortho_method"):
        LieOrthMomentum([dict(params=[p], use_skew=True, lr=1e-3)], ortho_method="bogus")


def test_first_moment_only_differs_from_second_moment():
    # With a wildly uneven per-entry grad, the second-moment (Adam) direction and the
    # first-moment-only direction point differently before orthogonalization.
    torch.manual_seed(0)
    ne, lr, c = 8 * 7 // 2, 0.1, 0.05
    g = torch.randn(1, ne)
    g[:, 0] *= 50.0
    p1 = nn.Parameter(torch.zeros(1, ne)); p1.grad = g.clone()
    p2 = nn.Parameter(torch.zeros(1, ne)); p2.grad = g.clone()
    _make_opt(p1, lr, c, ortho_use_second_moment=False).step()
    _make_opt(p2, lr, c, ortho_use_second_moment=True).step()
    assert not torch.allclose(p1.data, p2.data, atol=1e-4)


def test_grad_sign_flips_the_update():
    # Orthogonalization is odd in sign, so negating the grad negates the written oft_R.
    torch.manual_seed(0)
    ne, lr, c = 8 * 7 // 2, 0.1, 0.05
    g = torch.randn(1, ne)
    p_pos = nn.Parameter(torch.zeros(1, ne)); p_pos.grad = g.clone()
    p_neg = nn.Parameter(torch.zeros(1, ne)); p_neg.grad = -g.clone()
    _make_opt(p_pos, lr, c).step()
    _make_opt(p_neg, lr, c).step()
    assert torch.allclose(p_pos.data, -p_neg.data, atol=1e-5)


def test_adamw_branch_steps_non_skew_params():
    # non-oft_R params get the AdamW branch (moved off their initial value).
    w = nn.Parameter(torch.randn(4, 4)); w.grad = torch.randn(4, 4); w0 = w.data.clone()
    LieOrthMomentum([dict(params=[w], use_skew=False, lr=1e-2)], adamw_wd=0.0).step()
    assert not torch.allclose(w.data, w0)


def test_momentum_persists_across_value_reset():
    # lie_m persists across the per-step fold (p zeroed between steps); the second
    # step's direction reflects the accumulated EMA, not a fresh start.
    torch.manual_seed(0)
    ne, lr, c = 8 * 7 // 2, 0.1, 0.05
    p = nn.Parameter(torch.zeros(1, ne)); p.grad = torch.randn(1, ne)
    opt = _make_opt(p, lr, c)
    opt.step()
    assert "lie_m" in opt.state[p] and opt.state[p]["lie_m"].abs().sum() > 0
    p.data.zero_()  # simulate the merge fold
    p.grad = torch.randn(1, ne)
    opt.step()
    assert torch.isfinite(p.data).all()
```

- [ ] **Step 2: Run the optimizer tests to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.optim.poet_lie_orth'`.

- [ ] **Step 3: Create `src/optim/poet_lie_orth.py`**

Create the file with the full content:

```python
"""LieOrthMomentum: Muon-like orthogonalizing optimizer for POET's skew generators
(q_optimizer=lie_ortho). Standalone sibling of
src.optim.poet_lie_momentum.LieAlgebraMomentum.

Same Lie-algebra first-moment momentum on oft_R (one or more param groups tagged
use_skew=True) and the same AdamW branch on everything else, but instead of
RMS-scaling the direction it ORTHOGONALIZES it (orthogonalize_skew_direction) so the
rotation planes turn by ~the same angle. Default method='muon' (Muon's quintic NS, a
band around 1, ~5 steps); method='spectral' is the exact A(-A^2)^{-1/2} variant
(sigma=1, ~20 steps). See docs/muon_orthogonalizing_optimizer_poet.md.

First-moment-only by default: a second moment is partially undone by orthogonalization
(docs SS4). State buffers are named lie_m / lie_v so the merge patch's _zero_moments
cannot reset them -- momentum PERSISTS across the per-step fold. Single-process /
DP-replicated (no sharded distributed optimizer); integration lives in
src/optim/poet.py, which reuses _split_poet_lie_params / _build_lie_param_groups.
"""

from __future__ import annotations

import torch

from src.diag.skew_conditioning import block_size_from_nelems, skew_to_vec, vec_to_skew
from src.optim.poet_skew_muon import orthogonalize_skew_direction


class LieOrthMomentum(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        b1: float = 0.9,
        b2: float = 0.95,
        eps: float = 1e-8,
        v_mode: str = "elementwise",
        alternating: bool = False,
        alternate_every: int = 1,
        ortho_c: float = 0.01,
        ortho_method: str = "muon",
        ortho_ns_steps: int = 5,
        ortho_use_second_moment: bool = False,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if v_mode not in ("scalar", "elementwise"):
            raise ValueError(f"v_mode must be 'scalar' or 'elementwise', got {v_mode!r}")
        if ortho_method not in ("muon", "spectral"):
            raise ValueError(
                f"ortho_method must be 'muon' or 'spectral', got {ortho_method!r}"
            )
        # Alternating single-sided update: write only one side's oft_R per step (out on
        # even, in on odd), accumulating momentum on BOTH sides.
        self.alternating = bool(alternating)
        self.alternate_every = max(1, int(alternate_every))
        self._alt_step = 0
        # Orthogonalizing transform (docs/muon_orthogonalizing_optimizer_poet.md):
        # realized per-plane angle = lr * ortho_c (a band under 'muon', exact under
        # 'spectral'). First-moment-only unless ortho_use_second_moment.
        self.ortho_c = float(ortho_c)
        self.ortho_method = ortho_method
        self.ortho_ns_steps = int(ortho_ns_steps)
        self.ortho_use_second_moment = bool(ortho_use_second_moment)
        defaults = dict(
            lr=0.0,
            use_skew=False,
            side=None,
            b1=b1,
            b2=b2,
            eps=eps,
            v_mode=v_mode,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active = None
        if self.alternating:
            active = "out" if (self._alt_step // self.alternate_every) % 2 == 0 else "in"

        for group in self.param_groups:
            lr = group["lr"]
            if group["use_skew"]:
                side = group["side"]
                b1, b2, eps, v_mode = group["b1"], group["b2"], group["eps"], group["v_mode"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    g = g.float()
                    st = self.state[p]
                    if "lie_m" not in st:
                        st["lie_m"] = torch.zeros_like(g)
                        if v_mode == "scalar":
                            st["lie_v"] = torch.zeros(g.shape[0], 1, dtype=g.dtype, device=g.device)
                        else:
                            st["lie_v"] = torch.zeros_like(g)
                    m, v = st["lie_m"], st["lie_v"]
                    # Momentum accumulates on BOTH sides every step ...
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    if v_mode == "scalar":
                        v.mul_(b2).add_(2.0 * (g * g).sum(dim=-1, keepdim=True), alpha=1 - b2)
                    else:
                        v.mul_(b2).add_(g * g, alpha=1 - b2)
                    # ... but only the ACTIVE side's oft_R is written.
                    if self.alternating and side != active:
                        continue
                    # Orthogonalize the DIRECTION (per b x b block) so the planes turn
                    # by ~the same angle. Scale by ortho_c DIRECTLY (the spectrum is
                    # ~democratized, so no sqrt(d)/||A||, docs SS3). First-moment-only
                    # by default. Realized per-plane angle = lr * ortho_c.
                    A_dir = -m / (v.sqrt() + eps) if self.ortho_use_second_moment else -m
                    bsz = block_size_from_nelems(A_dir.shape[1])
                    X = orthogonalize_skew_direction(
                        vec_to_skew(A_dir, bsz),
                        method=self.ortho_method,
                        ns_steps=self.ortho_ns_steps,
                    )
                    gen = skew_to_vec(self.ortho_c * X, bsz)  # (n_blocks, n_elems)
                    p.add_(gen.to(p.dtype), alpha=lr)
            else:
                beta1, beta2 = group["adamw_betas"]
                aeps, wd = group["adamw_eps"], group["adamw_wd"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["moment1"] = torch.zeros_like(g)
                        st["moment2"] = torch.zeros_like(g)
                    st["step"] += 1
                    m1, m2 = st["moment1"], st["moment2"]
                    m1.lerp_(g, 1 - beta1)
                    m2.lerp_(g.square(), 1 - beta2)
                    update = m1 / (aeps + m2.sqrt())
                    bc1 = 1 - beta1 ** st["step"]
                    bc2 = 1 - beta2 ** st["step"]
                    scale = bc1 / bc2**0.5
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr / scale)
        if self.alternating:
            self._alt_step += 1
        return loss
```

- [ ] **Step 4: Run the optimizer tests to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_orth.py -v`
Expected: all pass (Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_lie_orth.py tests/unit/test_poet_lie_orth.py
git commit -m "feat(poet): add standalone LieOrthMomentum orthogonalizing optimizer"
```

---

## Task 3: Select `LieOrthMomentum` via `q_optimizer=lie_ortho` in the builder

Route `q_optimizer=lie_ortho` to the existing shared lie builder, and branch there to construct `LieOrthMomentum` instead of `LieAlgebraMomentum`. The builder's param split, group build, `init_state_fn`, and Megatron wrapping are unchanged (both classes use `lie_m`/`lie_v` buffers). Not CPU-unit-testable (needs Megatron handles); verification is `py_compile` plus the downstream arg tests (Tasks 5-6) and the GPU smoke (Task 8).

**Files:**
- Modify: `src/optim/poet.py` (dispatch ~line 644; the `optimizer = LieAlgebraMomentum(...)` construction ~lines 585-598)

- [ ] **Step 1: Route `lie_ortho` to the shared builder**

In `src/optim/poet.py`, change the dispatch (currently ~line 644):

```python
    if getattr(config, "poet_q_optimizer", "adam") == "lie_algebra":
        return get_megatron_poet_lie_momentum_optimizer(
```

to:

```python
    if getattr(config, "poet_q_optimizer", "adam") in ("lie_algebra", "lie_ortho"):
        return get_megatron_poet_lie_momentum_optimizer(
```

- [ ] **Step 2: Branch the optimizer construction**

In `get_megatron_poet_lie_momentum_optimizer`, replace the existing construction (currently ~lines 585-598):

```python
    optimizer = LieAlgebraMomentum(
        param_groups,
        b1=getattr(config, "poet_lie_b1", 0.9),
        b2=getattr(config, "poet_lie_b2", 0.95),
        eps=getattr(config, "poet_lie_eps", 1e-8),
        v_mode=getattr(config, "poet_lie_v_mode", "elementwise"),
        alternating=getattr(config, "poet_lie_alternating", False),
        alternate_every=getattr(config, "poet_lie_alternate_every", 1),
        rms=getattr(config, "poet_lie_rms", False),
        rms_c=getattr(config, "poet_lie_rms_c", 0.2),
        adamw_betas=(config.adam_beta1, config.adam_beta2),
        adamw_eps=config.adam_eps,
        adamw_wd=config.weight_decay,
    )
```

with:

```python
    shared_kwargs = dict(
        b1=getattr(config, "poet_lie_b1", 0.9),
        b2=getattr(config, "poet_lie_b2", 0.95),
        eps=getattr(config, "poet_lie_eps", 1e-8),
        v_mode=getattr(config, "poet_lie_v_mode", "elementwise"),
        alternating=getattr(config, "poet_lie_alternating", False),
        alternate_every=getattr(config, "poet_lie_alternate_every", 1),
        adamw_betas=(config.adam_beta1, config.adam_beta2),
        adamw_eps=config.adam_eps,
        adamw_wd=config.weight_decay,
    )
    if getattr(config, "poet_q_optimizer", "lie_algebra") == "lie_ortho":
        from src.optim.poet_lie_orth import LieOrthMomentum

        logger.info(
            "[POET] Lie-orth: method=%s, ortho_c=%s, ns_steps=%s, second_moment=%s",
            getattr(config, "poet_lie_ortho_method", "muon"),
            getattr(config, "poet_lie_ortho_c", 0.01),
            getattr(config, "poet_lie_ortho_ns_steps", 5),
            getattr(config, "poet_lie_ortho_use_second_moment", False),
        )
        optimizer = LieOrthMomentum(
            param_groups,
            ortho_c=getattr(config, "poet_lie_ortho_c", 0.01),
            ortho_method=getattr(config, "poet_lie_ortho_method", "muon"),
            ortho_ns_steps=getattr(config, "poet_lie_ortho_ns_steps", 5),
            ortho_use_second_moment=getattr(config, "poet_lie_ortho_use_second_moment", False),
            **shared_kwargs,
        )
    else:
        optimizer = LieAlgebraMomentum(
            param_groups,
            rms=getattr(config, "poet_lie_rms", False),
            rms_c=getattr(config, "poet_lie_rms_c", 0.2),
            **shared_kwargs,
        )
```

(The `from src.optim.poet_lie_momentum import (LieAlgebraMomentum, _build_lie_param_groups, _split_poet_lie_params)` import already at the top of this function stays; `LieOrthMomentum` is imported lazily in the branch.)

- [ ] **Step 3: Verify the module still compiles**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/optim/poet.py src/optim/poet_lie_orth.py && echo OK`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/optim/poet.py
git commit -m "feat(poet): select LieOrthMomentum via q_optimizer=lie_ortho in the builder"
```

---

## Task 4: argparse (q_optimizer choice + knob flags)

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` (`add_slm_args`: the `--poet-q-optimizer` choices ~line 71, and after `--poet-lie-rms-c` ~line 91)
- Test: `tests/unit/test_pretrain_gpt_slm.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_lie_ortho():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        [
            "--slm-config-path", "x.yaml",
            "--poet-q-optimizer", "lie_ortho",
            "--poet-lie-ortho-c", "0.02",
            "--poet-lie-ortho-method", "spectral",
            "--poet-lie-ortho-ns-steps", "20",
            "--poet-lie-ortho-use-second-moment",
        ]
    )
    assert args.poet_q_optimizer == "lie_ortho"
    assert args.poet_lie_ortho_c == 0.02
    assert args.poet_lie_ortho_method == "spectral"
    assert args.poet_lie_ortho_ns_steps == 20
    assert args.poet_lie_ortho_use_second_moment is True


def test_lie_ortho_knobs_default():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml"])
    assert args.poet_lie_ortho_c == 0.01
    assert args.poet_lie_ortho_method == "muon"
    assert args.poet_lie_ortho_ns_steps == 5
    assert args.poet_lie_ortho_use_second_moment is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py -k lie_ortho -v`
Expected: FAIL — `argument --poet-q-optimizer: invalid choice: 'lie_ortho'`.

- [ ] **Step 3a: Add `lie_ortho` to the q_optimizer choices**

In `launchers/pretrain_gpt_slm.py`, change (~line 70-72):

```python
    group.add_argument(
        "--poet-q-optimizer", choices=["adam", "muon", "lie_algebra"], default="adam"
    )
```

to:

```python
    group.add_argument(
        "--poet-q-optimizer",
        choices=["adam", "muon", "lie_algebra", "lie_ortho"],
        default="adam",
    )
```

- [ ] **Step 3b: Add the knob flags**

Immediately after the line `group.add_argument("--poet-lie-rms-c", type=float, default=0.2)` (~line 91), insert:

```python
    # Muon-like orthogonalizing optimizer (q_optimizer=lie_ortho; see
    # docs/muon_orthogonalizing_optimizer_poet.md). Orthogonalize the skew direction
    # so the planes turn by ~the same angle (= lr*ortho_c). method='muon' (quintic NS,
    # band, ~5 steps) | 'spectral' (exact A(-A^2)^-1/2, sigma=1, needs ~20 steps).
    group.add_argument("--poet-lie-ortho-c", type=float, default=0.01)
    group.add_argument(
        "--poet-lie-ortho-method", choices=["muon", "spectral"], default="muon"
    )
    group.add_argument("--poet-lie-ortho-ns-steps", type=int, default=5)
    group.add_argument("--poet-lie-ortho-use-second-moment", action="store_true")
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_pretrain_gpt_slm.py -k lie_ortho -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add launchers/pretrain_gpt_slm.py tests/unit/test_pretrain_gpt_slm.py
git commit -m "feat(poet): add lie_ortho q_optimizer choice + --poet-lie-ortho-* flags"
```

---

## Task 5: Emit the flags from the experiment YAML (`megatron_args.py`)

The `q_optimizer` value already flows through `--poet-q-optimizer` (existing line `poet.get("q_optimizer", "adam")`). Add the three value-carrying knobs (they are emitted unconditionally, like `--poet-lie-rms-c`; harmless when `q_optimizer != lie_ortho`) and the one store_true.

**Files:**
- Modify: `src/utils/megatron_args.py` (`_optimizer_args`, `kind == "poet"`: the `poet_args` list ~line 312, and the store_true block ~line 332)
- Test: `tests/unit/test_megatron_args.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_argv_emits_lie_ortho_knobs():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "q_optimizer": "lie_ortho",
                "lie_ortho_c": 0.02,
                "lie_ortho_method": "spectral",
                "lie_ortho_ns_steps": 20,
                "lie_ortho_use_second_moment": True,
            }
        )
    )
    assert args[args.index("--poet-q-optimizer") + 1] == "lie_ortho"
    assert args[args.index("--poet-lie-ortho-c") + 1] == "0.02"
    assert args[args.index("--poet-lie-ortho-method") + 1] == "spectral"
    assert args[args.index("--poet-lie-ortho-ns-steps") + 1] == "20"
    assert "--poet-lie-ortho-use-second-moment" in args


def test_poet_argv_lie_ortho_defaults():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-lie-ortho-use-second-moment" not in args
    assert args[args.index("--poet-lie-ortho-c") + 1] == "0.01"
    assert args[args.index("--poet-lie-ortho-method") + 1] == "muon"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k lie_ortho -v`
Expected: FAIL — `--poet-lie-ortho-c` not in args (`ValueError: ... is not in list`).

- [ ] **Step 3a: Add the value-carrying flags to `poet_args`**

In `src/utils/megatron_args.py`, inside the `poet_args = [ ... ]` list, immediately after:

```python
            "--poet-lie-rms-c",
            poet.get("lie_rms_c", 0.2),
```

insert:

```python
            "--poet-lie-ortho-c",
            poet.get("lie_ortho_c", 0.01),
            "--poet-lie-ortho-method",
            poet.get("lie_ortho_method", "muon"),
            "--poet-lie-ortho-ns-steps",
            poet.get("lie_ortho_ns_steps", 5),
```

- [ ] **Step 3b: Add the store_true flag**

Immediately after:

```python
        # store_true: enable Stage 2 RMS scaling (W-free) for q_optimizer=lie_algebra.
        if poet.get("lie_rms", False):
            poet_args.append("--poet-lie-rms")
```

insert:

```python
        # store_true: first-vs-second moment for the lie_ortho optimizer.
        if poet.get("lie_ortho_use_second_moment", False):
            poet_args.append("--poet-lie-ortho-use-second-moment")
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "lie_ortho or poet" -v`
Expected: new tests pass; existing poet tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): emit --poet-lie-ortho-* argv from optim.poet.lie_ortho_*"
```

---

## Task 6: Copy the args into the optimizer config (`poet_optimizer_setup.py`)

**Files:**
- Modify: `src/patches/poet_optimizer_setup.py` (`_wrapped_get_config`, after the `config.poet_lie_rms_c = ...` line ~53)
- Test: `tests/unit/test_patch_poet_optimizer_setup.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_patch_poet_optimizer_setup.py` (it already imports `importlib`, `sys`, `types` and defines `_reset_for_tests`):

```python
def test_get_config_copies_lie_ortho_knobs(monkeypatch):
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
        poet_q_optimizer="lie_ortho",
        poet_lie_ortho_c=0.02,
        poet_lie_ortho_method="spectral",
        poet_lie_ortho_ns_steps=20,
        poet_lie_ortho_use_second_moment=True,
    )
    cfg, _ = fake_training.get_megatron_optimizer_config(args)
    assert cfg.poet_q_optimizer == "lie_ortho"
    assert cfg.poet_lie_ortho_c == 0.02
    assert cfg.poet_lie_ortho_method == "spectral"
    assert cfg.poet_lie_ortho_ns_steps == 20
    assert cfg.poet_lie_ortho_use_second_moment is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_optimizer_setup.py -k lie_ortho -v`
Expected: FAIL with `AttributeError: 'types.SimpleNamespace' object has no attribute 'poet_lie_ortho_c'`.

- [ ] **Step 3: Add the config copies**

In `src/patches/poet_optimizer_setup.py`, in `_wrapped_get_config`, immediately after the line `config.poet_lie_rms_c = getattr(args, "poet_lie_rms_c", 0.2)`, insert:

```python
        config.poet_lie_ortho_c = getattr(args, "poet_lie_ortho_c", 0.01)
        config.poet_lie_ortho_method = getattr(args, "poet_lie_ortho_method", "muon")
        config.poet_lie_ortho_ns_steps = getattr(args, "poet_lie_ortho_ns_steps", 5)
        config.poet_lie_ortho_use_second_moment = getattr(
            args, "poet_lie_ortho_use_second_moment", False
        )
```

(The `config.poet_q_optimizer = getattr(args, "poet_q_optimizer", "adam")` copy already exists, so `lie_ortho` selection is already threaded.)

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_optimizer_setup.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_optimizer_setup.py tests/unit/test_patch_poet_optimizer_setup.py
git commit -m "feat(poet): copy poet_lie_ortho_* args into the optimizer config"
```

---

## Task 7: Experiment config + launch script

**Files:**
- Create: `configs/experiments/optim/poet_lie_orth.yaml`
- Create: `docs/experiments/poet_lie_orth.md`
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
    assert cfg.optim.poet.q_optimizer == "lie_ortho"
    assert cfg.optim.poet.lie_ortho_method == "muon"
    assert cfg.optim.poet.lie_ortho_c == 4
```

Append to `tests/unit/test_train_scripts.py`:

```python
def test_poet_lie_orth_script_supports_llama3():
    proc = _run("train_poet_lie_orth.sh", "llama3")
    assert "--poet-q-optimizer" in proc.stdout and "lie_ortho" in proc.stdout
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
# poet_lie_orth: the standalone Muon-like orthogonalizing optimizer
# (q_optimizer=lie_ortho), sibling of poet_lie_rms. See
# docs/muon_orthogonalizing_optimizer_poet.md.
#
# Same single-step Lie-momentum stack as poet_lie_rms (block_count=1, merge_period=1,
# reinit_period=-1, cayley, head-aligned) but the direction->generator step
# orthogonalizes the per-block skew direction instead of RMS-scaling it, so the
# rotation planes turn by ~the same angle. First-moment-only by default.
#
# Default method 'muon' = Muon's quintic Newton-Schulz (a BAND around 1, ~5 steps,
# cheap). 'spectral' = exact A(-A^2)^-1/2 (sigma=1) but needs lie_ortho_ns_steps~20.
#
# Angle convention: realized per-plane angle = group_lr * lie_ortho_c (like the rms
# path's lr * rms_c). lr=0.003, lie_ortho_c=4 -> ~0.012 rad/plane, matching the rms
# "best" run for a fair head-to-head. Under method=muon, c is NOMINAL (band gives
# ~0.75-1.0x that angle, input-dependent); under spectral it is exact.
experiment:
  name: poet_lie_orth
  family: optim
  description: |
    POET x Muon: a standalone optimizer (q_optimizer=lie_ortho) that orthogonalizes
    the Lie-algebra momentum direction so the rotation planes turn by ~the same angle
    (discards the gradient's relative per-plane magnitudes, keeps only the subspace).
    Default method=muon (Muon's quintic, a band around 1, ~5 steps); method=spectral
    is the exact sigma=1 variant. Same single-step POET stack as poet_lie_rms
    (merge_period=1, block_count=1, reinit_period=-1, cayley). Run head-to-head vs
    poet_lie_rms to test whether relative per-plane angles are signal or noise for
    rotational updates (docs SS7).
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
    q_optimizer: lie_ortho       # selects the standalone LieOrthMomentum optimizer
    lie_b1: 0.9
    lie_b2: 0.95
    lie_eps: 1.0e-8
    lie_v_mode: elementwise
    lie_ortho_c: 4               # nominal per-plane angle; realized angle = lr * c
    lie_ortho_method: muon       # 'muon' (quintic NS band, ~5 steps) | 'spectral' (exact, ns~20)
    lie_ortho_ns_steps: 5        # spectral needs ~20; muon plateaus by ~5
    lie_ortho_use_second_moment: false  # first-moment-only by default (docs SS4)
    head_aligned_attn: true      # rotate q/k/v/o per attention head (requires unfuse_qkv=true)
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
# experiment=optim/poet_lie_orth — the standalone POET x Muon orthogonalizing
# optimizer (q_optimizer=lie_ortho). Instead of the RMS-norm transform, it
# orthogonalizes the Lie-algebra momentum direction so the planes turn by ~the same
# angle (= lr*lie_ortho_c). Single-step, reinit_period=-1, block_count=1. "$@" wins.

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
# poet_lie_orth — standalone Muon-like orthogonalizing optimizer

Sibling of [`poet_lie_rms`](./poet_lie_rms.md): the standalone `LieOrthMomentum`
optimizer (`q_optimizer=lie_ortho`). Same single-step POET Lie-momentum stack
(`merge_period=1`, `block_count=1`, `reinit_period=-1`, `cayley`, head-aligned), but
the direction→generator transform is **orthogonalization** instead of RMS scaling, per
[docs/muon_orthogonalizing_optimizer_poet.md](../muon_orthogonalizing_optimizer_poet.md).

After the (first-moment) Lie direction `A`, the optimizer orthogonalizes each `b×b`
skew block and scales by `c`, so the rotation planes turn by ~the same angle:

```
X     = orthogonalize(A)          # planes' singular values driven toward 1
oft_R = lr · c · X                # realized per-plane angle ~ lr · lie_ortho_c
```

This discards the gradient's *relative* per-plane magnitudes (keeps only the
subspace) — Muon's bet, applied to rotational updates. First-moment-only by default
(a second moment is partially undone by orthogonalization, docs §4).

`lie_ortho_method`:
- **`muon`** (default) — Muon's quintic Newton–Schulz then a `½(X−Xᵀ)` cleanup. NS
  preserves skew on a skew input; it democratizes the spectrum into a **band** around
  1 (cond ≈ 1.5) in ~5 steps. Cheap; `c` is a *nominal* angle (band ≈ 0.7–1.1× target).
- **`spectral`** — exact `A(−A²)^{-1/2}`; drives every singular value to 1 so `c` is
  exactly the angle. Needs `lie_ortho_ns_steps ≈ 20` (≈4× the cost).

Run head-to-head vs `poet_lie_rms` to test whether the gradient's relative per-plane
angles are signal or noise for rotational updates (docs §7) — and `muon` vs `spectral`
to test whether a cheap band is as good as exact equalization.
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
- feat(poet): add the standalone Muon-like orthogonalizing optimizer `LieOrthMomentum` (`q_optimizer=lie_ortho`), sibling of the Lie-RMS optimizer. Orthogonalizes the skew update direction (`orthogonalize_skew_direction`) so the rotation planes turn by ~the same angle (`= lr * lie_ortho_c`); first-moment-only by default. Default `method=muon` (Muon's quintic Newton–Schulz, a band around 1, ~5 steps; NS preserves skew); `method=spectral` is the exact `A(-A^2)^{-1/2}` σ=1 variant (~20 steps). New experiment `optim/poet_lie_orth` + `scripts/train_poet_lie_orth.sh` for the head-to-head vs `poet_lie_rms` (docs/muon_orthogonalizing_optimizer_poet.md).
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(poet): changelog for the standalone lie-orth optimizer"
```

- [ ] **Step 4: GPU smoke test — HAND OFF TO THE USER (do NOT run)**

This needs a GPU + the cluster env; it is the user's to run. Provide this command and stop:

```
codexlog poet_lie_orth_smoke bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 \
  optim.poet.lie_ortho_c=4
```

Expected sanity signals (from earlier POET runs): the `[POET] Lie-orth: method=muon, ortho_c=4.0, ...` log line appears at startup; step-1/2 run without OOM or NaN; loss decreases. The head-to-head then compares this against the existing `poet_lie_rms_best` run. Optional second arm — exact equalization (σ=1, more expensive):

```
codexlog poet_lie_orth_spectral bash scripts/train_poet_lie_orth.sh llama3 \
  optim.lr=0.003 \
  optim.poet.lie_ortho_c=4 \
  optim.poet.lie_ortho_method=spectral \
  optim.poet.lie_ortho_ns_steps=20
```

---

## Self-Review: spec coverage

| Doc section | Covered by |
|---|---|
| §1 one-line idea (equalize plane angles) | Task 1 (democratize spectrum / σ→1) + Task 2 (`ortho_c` = per-plane angle) |
| §2 composes with pipeline (momentum before ortho, small-angle CNP after) | Task 2 `step`: orthogonalize `−m` (the momentum *result*), scale by small `ortho_c`, then the existing merge applies CNP/exp. `lie_m` buffers persist. |
| §3 scale by `c` directly (no √d/‖A‖) | Task 2 scales by `ortho_c` directly. **Note:** exact only under `spectral`; `muon` gives a band, so `c` is nominal (documented). |
| §4 per-step update; first-moment-only default | Task 2: `ortho_use_second_moment=False` default ⇒ `A_dir = −m`; flag enables `−m/(√v+eps)`. |
| §5 orthogonalizing a skew matrix (both methods, NS preserves skew) | Task 1: `muon` (Muon's quintic + cleanup, **default**) + `spectral` (`A(−A²)^{-1/2}`, exact, opt-in). |
| §6 block-diagonal / per-head | Task 1: `vec_to_skew` yields `(n_blocks, b, b)`, all ops batch over `n_blocks`; Task 2 derives `b` per param. `head_aligned_attn=true` in the YAML. |
| §7 RMS vs ortho experiment | Task 7: `poet_lie_orth` config + script, `lr=0.003, c=4` mirrors the rms best run; Task 8 head-to-head commands (incl. muon-vs-spectral arm). |
| §8 summary ("a **separate** optimizer") | **Standalone `LieOrthMomentum` class** in its own file, selected by `q_optimizer=lie_ortho` — literally a separate optimizer, per the doc. |

**Placeholder scan:** none — every code step is complete and every command has expected output.

**Numerics provenance:** step counts, tolerances, and band bounds were all measured in the CPU env (NS preserves skew to ~1e-15; muon plateaus by step 5; spectral `max|σ−1| ≤ 0.006` at 20 steps on benign inputs; muon single-step median angle ≈ 0.75–1.04·lr·c, cond < 1.6; spectral single-step σ = lr·c to 1e-6).

**Type/name consistency:** the naming table is used verbatim in every task. Class `LieOrthMomentum(params, b1, b2, eps, v_mode, alternating, alternate_every, ortho_c, ortho_method, ortho_ns_steps, ortho_use_second_moment, adamw_*)`; helper `orthogonalize_skew_direction(skew, method, ns_steps, eps, reg)`; selection `q_optimizer=lie_ortho`; config/arg `poet_lie_ortho_*`; argv `--poet-lie-ortho-*`; YAML `lie_ortho_*`. Defaults match across argparse (`0.01/muon/5/False`), `getattr` fallbacks in Tasks 3/5/6, and the optimizer signature.
