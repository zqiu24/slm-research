# POET × Pion Lie-Algebra Momentum — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `q_optimizer: lie_algebra` POET backend that replaces ambient-space Adam-on-`oft_R` with Pion's Lie-algebra first+second-moment momentum (persisting across the per-step merge), at the poet0 single-step config.

**Architecture:** A new `LieAlgebraMomentum` optimizer (a SkewMuon-shaped class: skew branch on `oft_R`, AdamW branch on the rest) computes the Pion momentum update on the **identity-point tangent gradient** — at `merge_period=1` the ambient `oft_R.grad` equals the skew tangent gradient to O(angle²), so no new gradient plumbing is needed. The update is computed in **vec-space** (upper-triangular, like SkewMuon's `momentum_buffer`), which is provably identical to the paper's skew-space formula (verified: scalar-v needs a ×2 Frobenius factor). It plugs into the existing merge/DDP stack via a builder cloned from the muon path; `block_count=1` is the CPU correctness oracle.

**Tech Stack:** Python, PyTorch, Megatron-Core optimizer wrappers, OmegaConf/Hydra, pytest.

**Spec:** [docs/superpowers/specs/2026-06-03-poet-lie-momentum-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-03-poet-lie-momentum-design.md)

**Conventions (same as the poet0 plan):**
- Repo root: `/lustre/fast/fast/zqiu/slm-research` (run all commands from here).
- CPU test interpreter: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`.
- **Script dry-run tests (Task 7 only)** need the venv on `PATH` (the script shells out to a bare `python`): prefix with `PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH`.
- Commit style: single short conventional-commit sentence, anonymous (no co-author trailer). Let the pre-commit hook run (no `--no-verify`).

---

## Key math (used in Tasks 1 & 3)

The skew branch, per `oft_R` param (shape `(n_blocks, n_elems)`, `n_elems = b(b−1)/2`), every step. `p` is born at 0 (the merge zeroed it last step), so `p ← lr·A`:

```
g = p.grad                                  # vec-space gradient (n_blocks, n_elems)
m = b1·m + (1-b1)·g                          # lie_m  (n_blocks, n_elems), PERSISTS
scalar-v:      v = b2·v + (1-b2)·2·Σ_k g[:,k]²   # lie_v (n_blocks, 1)   — ×2 = full-matrix ‖·‖_F²
elementwise-v: v = b2·v + (1-b2)·(g⊙g)          # lie_v (n_blocks, n_elems)
A = -m / (sqrt(v) + eps)                     # broadcast for scalar-v
p.add_(A, alpha=lr)                          # lr = scheduled group lr (= optim.lr·scale)
```

Verified (CPU) to match the paper's skew-space `A = -M/(√v+ε)` with `M = vec_to_skew(m)`, `v_scalar = ‖vec_to_skew(g)‖_F²`, to machine precision for both v-modes.

`m`/`v` are named `lie_m`/`lie_v` (NOT `exp_avg`/`exp_avg_sq`) so the merge's [`_zero_moments`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L154) can never clobber them.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/optim/poet_lie_momentum.py` | `LieAlgebraMomentum` optimizer + `_build_lie_param_groups` helper | Create |
| `src/optim/poet.py` | `get_megatron_poet_lie_momentum_optimizer` builder + `q_optimizer=="lie_algebra"` dispatch | Modify |
| `launchers/pretrain_gpt_slm.py` | `--poet-q-optimizer` choice + `--poet-lie-*` args | Modify |
| `src/utils/megatron_args.py` | emit `--poet-lie-*` args | Modify |
| `src/patches/poet_optimizer_setup.py` | thread `poet_lie_*` onto OptimizerConfig | Modify |
| `configs/experiments/optim/poet_lie.yaml` | new experiment (`q_optimizer: lie_algebra`, `reinit_period: -1`) | Create |
| `docs/experiments/poet_lie.md` | required by pre-commit hook | Create |
| `scripts/train_poet_lie.sh` | launcher script | Create |
| `tests/unit/test_poet_lie_momentum.py` | optimizer math/persistence/shape + param-group helper | Create |
| `tests/unit/test_patch_poet_merge.py` | `lie_m`/`lie_v` survive `_reset_vanilla_oft_state` | Modify |
| `tests/unit/test_pretrain_gpt_slm.py` | launcher accepts `--poet-lie-*` | Modify |
| `tests/unit/test_megatron_args.py` | emission of `--poet-lie-*` + experiment yaml | Modify |
| `tests/unit/test_train_scripts.py` | dry-run smoke for `train_poet_lie.sh` | Modify |

---

## Task 1: `LieAlgebraMomentum` optimizer

**Files:**
- Create: `src/optim/poet_lie_momentum.py`
- Test: `tests/unit/test_poet_lie_momentum.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_poet_lie_momentum.py`:

```python
"""Tests for the Lie-algebra momentum optimizer (POET q_optimizer=lie_algebra)."""

import torch
import torch.nn as nn

from src.diag.skew_conditioning import skew_to_vec, vec_to_skew


def _skew_space_reference(g_vec, b, b1, b2, eps, lr, v_mode):
    """Paper-faithful skew-space Pion first/second-moment step from ZERO state,
    returns the expected oft_R update (vec-space) = lr * skew_to_vec(A)."""
    G = vec_to_skew(g_vec, b)                 # (n_blocks, b, b) skew
    M = (1 - b1) * G
    if v_mode == "scalar":
        v = (1 - b2) * (G * G).sum(dim=(-2, -1), keepdim=True)   # ‖G‖_F^2 full matrix
    else:
        v = (1 - b2) * (G * G)
    A = -M / (v.sqrt() + eps)
    return lr * skew_to_vec(A, b)


def _make_opt(p, lr, v_mode):
    from src.optim.poet_lie_momentum import LieAlgebraMomentum
    return LieAlgebraMomentum(
        [dict(params=[p], use_skew=True, lr=lr)],
        b1=0.9, b2=0.95, eps=1e-8, v_mode=v_mode,
    )


def test_first_step_matches_pion_scalar_v():
    torch.manual_seed(0)
    b, ne, lr = 4, 6, 1e-3
    p = nn.Parameter(torch.zeros(1, ne))           # born at identity
    p.grad = torch.randn(1, ne)
    expected = _skew_space_reference(p.grad.clone(), b, 0.9, 0.95, 1e-8, lr, "scalar")
    _make_opt(p, lr, "scalar").step()
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()


def test_first_step_matches_pion_elementwise_v():
    torch.manual_seed(0)
    b, ne, lr = 4, 6, 1e-3
    p = nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    expected = _skew_space_reference(p.grad.clone(), b, 0.9, 0.95, 1e-8, lr, "elementwise")
    _make_opt(p, lr, "elementwise").step()
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()


def test_momentum_persists_across_value_reset():
    # Two steps with p zeroed between (simulating the fold) -> lie_m/lie_v are
    # EMAs that ACCUMULATE across the zeroing (NOT reset).
    torch.manual_seed(1)
    b, ne, lr, b1, b2, eps = 4, 6, 1e-3, 0.9, 0.95, 1e-8
    p = nn.Parameter(torch.zeros(1, ne))
    opt = _make_opt(p, lr, "scalar")
    g1 = torch.randn(1, ne); g2 = torch.randn(1, ne)
    p.grad = g1.clone(); opt.step()
    p.data.zero_(); p.grad = g2.clone(); opt.step()
    # hand-compute the 2nd step from the persisted state
    m = b1 * ((1 - b1) * g1) + (1 - b1) * g2
    v = b2 * ((1 - b2) * 2 * (g1 * g1).sum()) + (1 - b2) * 2 * (g2 * g2).sum()
    expected = lr * (-m / (v.sqrt() + eps))
    assert torch.allclose(p.data, expected, atol=1e-7), (p.data - expected).abs().max()
    st = opt.state[p]
    assert torch.allclose(st["lie_m"], m, atol=1e-7)


def test_v_shapes():
    from src.optim.poet_lie_momentum import LieAlgebraMomentum
    ne = 6
    for v_mode, vshape in (("scalar", (2, 1)), ("elementwise", (2, ne))):
        p = nn.Parameter(torch.zeros(2, ne)); p.grad = torch.randn(2, ne)
        opt = LieAlgebraMomentum([dict(params=[p], use_skew=True, lr=1e-3)],
                                 v_mode=v_mode)
        opt.step()
        assert tuple(opt.state[p]["lie_v"].shape) == vshape


def test_adamw_branch_steps_without_error():
    from src.optim.poet_lie_momentum import LieAlgebraMomentum
    p = nn.Parameter(torch.randn(3, 5)); g = torch.randn(3, 5); p.grad = g.clone()
    before = p.data.clone()
    opt = LieAlgebraMomentum([dict(params=[p], use_skew=False, lr=1e-2)])
    opt.step()
    assert not torch.allclose(p.data, before)   # standard AdamW moved it
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_momentum.py -v
```
Expected: FAIL — `ModuleNotFoundError: src.optim.poet_lie_momentum`.

- [ ] **Step 3: Implement the optimizer**

Create `src/optim/poet_lie_momentum.py`:

```python
"""LieAlgebraMomentum: Pion Lie-algebra first+second-moment momentum on POET's
skew generators (q_optimizer=lie_algebra). Increment 1 of the POET-X x Pion
pipeline (docs/poetx_pion_pipeline.md §2-3): import Pion's Lie-algebra momentum
while keeping POET's block-skew oft_R + merge machinery.

Shaped like src/optim/poet_skew_muon.SkewMuon: skew branch on oft_R (one or more
param groups tagged use_skew=True), AdamW branch on everything else. The skew
update is computed in VEC-space (upper-triangular, like SkewMuon's
momentum_buffer) — provably identical to the paper's skew-space A = -M/(sqrt(v)+eps)
with M = vec_to_skew(m); scalar-v uses ||G||_F^2 = 2*sum(g_vec^2) (full-matrix
Frobenius). At merge_period=1 the ambient oft_R.grad equals the skew tangent
gradient to O(angle^2), so no new gradient plumbing is needed.

State buffers are named lie_m / lie_v (NOT exp_avg/exp_avg_sq) so the merge
patch's _zero_moments cannot reset them — Lie momentum PERSISTS across the
per-step fold. Single-process / DP-replicated (no sharded distributed optimizer),
like the muon path; integration lives in src/optim/poet.py.
"""

from __future__ import annotations

import torch


class LieAlgebraMomentum(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        b1: float = 0.9,
        b2: float = 0.95,
        eps: float = 1e-8,
        v_mode: str = "scalar",
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if v_mode not in ("scalar", "elementwise"):
            raise ValueError(f"v_mode must be 'scalar' or 'elementwise', got {v_mode!r}")
        defaults = dict(
            lr=0.0, use_skew=False, b1=b1, b2=b2, eps=eps, v_mode=v_mode,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps, adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            if group["use_skew"]:
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
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    if v_mode == "scalar":
                        # ||vec_to_skew(g)||_F^2 = 2 * sum(g^2) over the upper-tri vec
                        v.mul_(b2).add_(2.0 * (g * g).sum(dim=-1, keepdim=True), alpha=1 - b2)
                    else:
                        v.mul_(b2).add_(g * g, alpha=1 - b2)
                    A = -m / (v.sqrt() + eps)
                    p.add_(A.to(p.dtype), alpha=lr)   # p born at 0 -> p = lr*A
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
                    scale = bc1 / bc2 ** 0.5
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr / scale)
        return loss
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_momentum.py -v
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add src/optim/poet_lie_momentum.py tests/unit/test_poet_lie_momentum.py && \
git commit -F - <<'EOF'
feat(poet): LieAlgebraMomentum optimizer (Pion Lie-algebra momentum on oft_R, vec-space)
EOF
```

---

## Task 2: `lie_m`/`lie_v` survive the merge reset

The merge calls `_reset_vanilla_oft_state` each step; it zeros the master *value* (wanted) and, on reinit, `exp_avg`/`exp_avg_sq`. Prove it never touches `lie_m`/`lie_v` even on a (worst-case) reinit.

**Files:**
- Test: `tests/unit/test_patch_poet_merge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_patch_poet_merge.py`:

```python
def test_reset_vanilla_oft_state_never_clobbers_lie_buffers():
    import torch
    import torch.nn as nn

    from src.patches.poet_merge_step import _reset_vanilla_oft_state

    model = nn.Module()
    model.oft_R_in = nn.Parameter(torch.ones(2, 6))     # bf16 model tensor
    master = nn.Parameter(torch.full((2, 6), 3.0))       # separate fp32 master
    torch_opt = torch.optim.SGD([master], lr=1e-3)       # any torch optimizer
    # Lie-momentum-style state on the MASTER param:
    torch_opt.state[master] = {
        "lie_m": torch.ones(2, 6),
        "lie_v": torch.ones(2, 1),
    }

    class _FakeInner:
        def __init__(self):
            self.float16_groups = [[model.oft_R_in]]
            self.fp32_from_float16_groups = [[master]]
            self.optimizer = torch_opt

    opt = _FakeInner()
    # Worst case: reset_moments=True (the reinit boundary).
    _reset_vanilla_oft_state(opt, model, iteration=400, reset_moments=True)

    assert torch.count_nonzero(master.data) == 0                     # master value zeroed
    assert torch.count_nonzero(torch_opt.state[master]["lie_m"]) == 12  # lie_m intact
    assert torch.count_nonzero(torch_opt.state[master]["lie_v"]) == 2   # lie_v intact
```

- [ ] **Step 2: Run test to verify it passes immediately (no code change needed — this is a guard)**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_patch_poet_merge.py -k lie_buffers -v
```
Expected: PASS. (`_reset_vanilla_oft_state` only references `exp_avg`/`exp_avg_sq`/`step`; `lie_m`/`lie_v` are invisible to it. This test is a **regression guard** — if someone later renames our buffers to `exp_avg*`, it fails.)

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add tests/unit/test_patch_poet_merge.py && \
git commit -F - <<'EOF'
test(poet): guard that the merge reset never clobbers lie_m/lie_v
EOF
```

---

## Task 3: Param-group helper + builder + `q_optimizer=lie_algebra` dispatch

**Files:**
- Modify: `src/optim/poet_lie_momentum.py` (add `_build_lie_param_groups`)
- Modify: `src/optim/poet.py` (add builder near the muon builder; add dispatch at [L547](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L547))
- Test: `tests/unit/test_poet_lie_momentum.py` (helper test; the full builder needs Megatron → covered by Task 7 dry-run)

- [ ] **Step 1: Write the failing test (param-group helper)**

Append to `tests/unit/test_poet_lie_momentum.py`:

```python
def test_build_lie_param_groups_scales_skew_lr():
    import torch.nn as nn
    from src.optim.poet_lie_momentum import _build_lie_param_groups

    skew = [nn.Parameter(torch.zeros(1, 6))]
    adamw = [nn.Parameter(torch.zeros(4))]
    groups = _build_lie_param_groups(skew, adamw, lr=1e-3, min_lr=1e-5, scale=0.5)

    g_skew = next(g for g in groups if g["use_skew"])
    g_adam = next(g for g in groups if not g["use_skew"])
    assert g_skew["lr"] == 5e-4 and g_skew["max_lr"] == 5e-4 and g_skew["min_lr"] == 5e-6
    assert g_adam["lr"] == 1e-3 and g_adam["max_lr"] == 1e-3 and g_adam["min_lr"] == 1e-5


def test_build_lie_param_groups_drops_empty_sides():
    from src.optim.poet_lie_momentum import _build_lie_param_groups
    assert _build_lie_param_groups([], [], 1e-3, 1e-5, 0.5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_lie_momentum.py -k build_lie_param_groups -v
```
Expected: FAIL — `ImportError: cannot import name '_build_lie_param_groups'`.

- [ ] **Step 3a: Add the `_build_lie_param_groups` helper to `src/optim/poet_lie_momentum.py`**

Insert immediately after the `import torch` line, before `class LieAlgebraMomentum`:

```python
def _build_lie_param_groups(skew_params, adamw_params, lr, min_lr, scale):
    """Two param groups carrying lr/max_lr/min_lr so Megatron's scheduler decays
    group['lr'] (skew side scaled by poet_scale, exactly like the vanilla path
    scales oft_R). Empty sides are dropped."""
    groups = []
    skew_params = list(skew_params)
    adamw_params = list(adamw_params)
    if skew_params:
        groups.append(
            dict(params=skew_params, use_skew=True,
                 lr=lr * scale, max_lr=lr * scale, min_lr=min_lr * scale)
        )
    if adamw_params:
        groups.append(
            dict(params=adamw_params, use_skew=False,
                 lr=lr, max_lr=lr, min_lr=min_lr)
        )
    return groups
```

- [ ] **Step 3b: Add the builder and dispatch in `src/optim/poet.py`**

Immediately after `get_megatron_poet_muon_optimizer` (ends at [L525](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L525)), add:

```python
def get_megatron_poet_lie_momentum_optimizer(
    config,
    model_chunks,
    *,
    config_overrides=None,
    use_gloo_process_groups: bool = True,
):
    """POET Lie-algebra momentum: LieAlgebraMomentum on oft_R, AdamW on the rest,
    wrapped for Megatron. Single-process / DP-replicated (no sharded distributed
    optimizer), like the muon path. Increment 1 of POET-X x Pion."""
    _resolve_megatron_handles()
    from megatron.core import parallel_state as mpu
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )

    from src.optim.poet_lie_momentum import LieAlgebraMomentum, _build_lie_param_groups

    if getattr(config, "use_distributed_optimizer", False):
        raise ValueError("POET Lie-momentum does not support the distributed optimizer (dev only).")
    if getattr(config, "fp16", False):
        raise ValueError("POET Lie-momentum does not support fp16; use bf16.")
    if mpu.get_tensor_model_parallel_world_size() > 1:
        raise ValueError("POET Lie-momentum does not support tensor parallelism > 1.")
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        raise ValueError("POET Lie-momentum does not support pipeline parallelism > 1.")

    skew_params, adamw_params = _split_poet_muon_params(model_chunks)
    scale = getattr(config, "poet_scale", 1.0)
    min_lr = getattr(config, "min_lr", 0.0)
    logger.info(
        "[POET] Lie-momentum: %d skew (oft_R) params, %d adamw params (b1=%s, b2=%s, v_mode=%s, scale=%s)",
        len(skew_params), len(adamw_params),
        getattr(config, "poet_lie_b1", 0.9), getattr(config, "poet_lie_b2", 0.95),
        getattr(config, "poet_lie_v_mode", "scalar"), scale,
    )
    if not skew_params:
        logger.warning("[POET] Lie-momentum: no oft_R params found — skew branch is a no-op.")

    param_groups = _build_lie_param_groups(skew_params, adamw_params, config.lr, min_lr, scale)
    optimizer = LieAlgebraMomentum(
        param_groups,
        b1=getattr(config, "poet_lie_b1", 0.9),
        b2=getattr(config, "poet_lie_b2", 0.95),
        eps=getattr(config, "poet_lie_eps", 1e-8),
        v_mode=getattr(config, "poet_lie_v_mode", "scalar"),
        adamw_betas=(config.adam_beta1, config.adam_beta2),
        adamw_eps=config.adam_eps,
        adamw_wd=config.weight_decay,
    )

    def init_state_fn(opt, _config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state[p]
                if group["use_skew"]:
                    st.setdefault("lie_m", torch.zeros_like(p.data))
                    if group["v_mode"] == "scalar":
                        st.setdefault("lie_v", torch.zeros(p.data.shape[0], 1, dtype=p.data.dtype, device=p.data.device))
                    else:
                        st.setdefault("lie_v", torch.zeros_like(p.data))
                elif "moment1" not in st:
                    st["step"] = 0
                    st["moment1"] = torch.zeros_like(p.data)
                    st["moment2"] = torch.zeros_like(p.data)

    if getattr(config, "bf16", False):
        return Float16OptimizerWithFloat16Params(optimizer, config, None, init_state_fn)
    return FP32Optimizer(optimizer, config, init_state_fn)
```

Then add the dispatch in `get_megatron_poet_optimizer`, immediately before the
existing `muon` branch at [L547](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L547):

```python
    if getattr(config, "poet_q_optimizer", "adam") == "lie_algebra":
        return get_megatron_poet_lie_momentum_optimizer(
            config,
            model_chunks,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        )
```

- [ ] **Step 4: Verify it imports and the helper test still passes**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/optim/poet.py && echo "py_compile OK" && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_lie_momentum.py -v
```
Expected: `py_compile OK`; all `test_poet_lie_momentum.py` tests PASS (7 total).

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add src/optim/poet_lie_momentum.py src/optim/poet.py tests/unit/test_poet_lie_momentum.py && \
git commit -F - <<'EOF'
feat(poet): wire q_optimizer=lie_algebra builder + dispatch (scheduled, scale-aware param groups)
EOF
```

---

## Task 4: Launcher args

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py` ([L70](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L70))
- Test: `tests/unit/test_pretrain_gpt_slm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_lie_algebra_args():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args([
        "--poet", "--poet-q-optimizer", "lie_algebra",
        "--poet-lie-b1", "0.9", "--poet-lie-b2", "0.95",
        "--poet-lie-eps", "1e-8", "--poet-lie-v-mode", "elementwise",
    ])
    assert args.poet_q_optimizer == "lie_algebra"
    assert args.poet_lie_b1 == 0.9 and args.poet_lie_b2 == 0.95
    assert args.poet_lie_eps == 1e-8 and args.poet_lie_v_mode == "elementwise"


def test_add_slm_args_lie_defaults():
    import argparse

    from launchers.pretrain_gpt_slm import add_slm_args

    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--poet"])
    assert args.poet_lie_b1 == 0.9 and args.poet_lie_b2 == 0.95
    assert args.poet_lie_eps == 1e-8 and args.poet_lie_v_mode == "scalar"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_pretrain_gpt_slm.py -k lie -v
```
Expected: FAIL — `invalid choice: 'lie_algebra'` / `unrecognized arguments: --poet-lie-b1`.

- [ ] **Step 3: Add the args**

In `launchers/pretrain_gpt_slm.py`, change the `--poet-q-optimizer` choices
([L70](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L70)) and
add the lie args immediately after the muon args (after L73):

```python
    group.add_argument("--poet-q-optimizer", choices=["adam", "muon", "lie_algebra"], default="adam")
    group.add_argument("--poet-muon-theta", type=float, default=0.1)
    group.add_argument("--poet-muon-ns-steps", type=int, default=5)
    group.add_argument("--poet-muon-momentum", type=float, default=0.95)
    # Lie-algebra momentum (q_optimizer=lie_algebra): Pion first/second-moment
    # momentum on oft_R, accumulated in the Lie algebra (persists across merges).
    group.add_argument("--poet-lie-b1", type=float, default=0.9)
    group.add_argument("--poet-lie-b2", type=float, default=0.95)
    group.add_argument("--poet-lie-eps", type=float, default=1e-8)
    group.add_argument("--poet-lie-v-mode", choices=["scalar", "elementwise"], default="scalar")
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_pretrain_gpt_slm.py -k lie -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add launchers/pretrain_gpt_slm.py tests/unit/test_pretrain_gpt_slm.py && \
git commit -F - <<'EOF'
feat(poet): register --poet-q-optimizer lie_algebra + --poet-lie-* launcher args
EOF
```

---

## Task 5: `megatron_args` emission + `poet_optimizer_setup` threading

**Files:**
- Modify: `src/utils/megatron_args.py` (after the muon args, [~L284](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L284))
- Modify: `src/patches/poet_optimizer_setup.py` ([~L45](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py#L45))
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_argv_emits_lie_args():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({
        "block_size": 256, "q_optimizer": "lie_algebra",
        "lie_b1": 0.9, "lie_b2": 0.95, "lie_eps": 1e-8, "lie_v_mode": "elementwise",
    }))
    assert args[args.index("--poet-q-optimizer") + 1] == "lie_algebra"
    assert args[args.index("--poet-lie-b1") + 1] == "0.9"
    assert args[args.index("--poet-lie-b2") + 1] == "0.95"
    assert args[args.index("--poet-lie-v-mode") + 1] == "elementwise"


def test_poet_argv_lie_args_default_when_unset():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert args[args.index("--poet-lie-v-mode") + 1] == "scalar"
    assert args[args.index("--poet-q-optimizer") + 1] == "adam"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_megatron_args.py -k lie -v
```
Expected: FAIL — `--poet-lie-b1` / `--poet-lie-v-mode` not in args.

- [ ] **Step 3: Emit the args in `megatron_args.py`**

In the `kind == "poet"` branch, immediately after the `--poet-muon-momentum`
pair ([L283-284](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L283)), add:

```python
            "--poet-lie-b1",
            poet.get("lie_b1", 0.9),
            "--poet-lie-b2",
            poet.get("lie_b2", 0.95),
            "--poet-lie-eps",
            poet.get("lie_eps", 1.0e-8),
            "--poet-lie-v-mode",
            poet.get("lie_v_mode", "scalar"),
```

- [ ] **Step 4: Thread the config in `poet_optimizer_setup.py`**

In `_wrapped_get_config`, immediately after the `config.poet_muon_momentum = ...`
line ([L45](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py#L45)), add:

```python
        config.poet_lie_b1 = getattr(args, "poet_lie_b1", 0.9)
        config.poet_lie_b2 = getattr(args, "poet_lie_b2", 0.95)
        config.poet_lie_eps = getattr(args, "poet_lie_eps", 1.0e-8)
        config.poet_lie_v_mode = getattr(args, "poet_lie_v_mode", "scalar")
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_megatron_args.py -k lie -v && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile src/patches/poet_optimizer_setup.py && echo OK
```
Expected: PASS (2 tests) + `OK`.

- [ ] **Step 6: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add src/utils/megatron_args.py src/patches/poet_optimizer_setup.py tests/unit/test_megatron_args.py && \
git commit -F - <<'EOF'
feat(poet): emit + thread --poet-lie-* args (b1/b2/eps/v-mode)
EOF
```

---

## Task 6: Experiment config + doc

**Files:**
- Create: `configs/experiments/optim/poet_lie.yaml`
- Create: `docs/experiments/poet_lie.md`
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_lie_experiment_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/experiments/optim/poet_lie.yaml")
    assert cfg.experiment.name == "poet_lie"
    assert cfg.optim.poet.q_optimizer == "lie_algebra"
    assert cfg.optim.poet.merge_period == 1
    assert cfg.optim.poet.reinit_period == -1
    assert cfg.optim.poet.lie_v_mode == "scalar"
    assert cfg.optim.poet.use_poet_adam is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_megatron_args.py -k poet_lie_experiment -v
```
Expected: FAIL — file does not exist.

- [ ] **Step 3: Create the experiment config**

Create `configs/experiments/optim/poet_lie.yaml`:

```yaml
# @package _global_
# poet_lie: POET x Pion increment 1 — Lie-algebra momentum on oft_R.
#
# Replaces stock Megatron-Adam on oft_R with Pion's Lie-algebra first+second-
# moment momentum (q_optimizer=lie_algebra), accumulated in the Lie algebra and
# PERSISTING across the per-step fold. Single-step regime (merge_period=1) so the
# ambient oft_R.grad equals the skew tangent gradient to O(angle^2); reinit_period=-1
# keeps Psi fixed and momentum coordinate-coherent (never reset). block_count=1 is
# the correctness oracle (one block = full matrix). RMS-alpha, low-order Cayley,
# alternating, and the exact tangent gradient are DEFERRED (later increments).
experiment:
  name: poet_lie
  family: optim
  description: |
    POET x Pion (increment 1): Lie-algebra momentum on POET's block-skew
    generators. q_optimizer=lie_algebra runs Pion's A = -M/(sqrt(v)+eps) update
    (lie_v_mode scalar|elementwise) on the identity-point tangent gradient, with
    momentum persisting across the merge. Same single-step POET stack as poet0
    (merge_period=1, block_count=1, two-sided, cayley); only the oft_R optimizer
    changes. Hypothesis: Lie-algebra momentum moves val loss off the POET-Adam
    baseline toward Muon (pipeline doc §9 step 1).
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
    reinit_period: -1        # never resample Psi / reset momentum (block_count=1)
    scale: 0.5
    use_poet_adam: false
    parameterization: cayley
    q_optimizer: lie_algebra
    lie_b1: 0.9
    lie_b2: 0.95
    lie_eps: 1.0e-8
    lie_v_mode: scalar        # scalar (pipeline doc §3) | elementwise (paper Algorithm 1)
    train_output_rotation: true

base:
  model:
    unfuse_qkv: true
    unfuse_fc1: true
```

- [ ] **Step 4: Create the experiment doc**

Create `docs/experiments/poet_lie.md`:

```markdown
# poet_lie — POET × Pion Lie-algebra momentum (increment 1)

Increment 1 of the POET-X × Pion pipeline
([docs/poetx_pion_pipeline.md](../poetx_pion_pipeline.md) §2–§3, §9 step 1).
Same single-step POET stack as [`poet0`](./poet0.md) — `merge_period=1`,
`block_count=1`, two-sided, Cayley — but the `oft_R` optimizer is swapped from
stock Megatron-Adam to **Pion's Lie-algebra momentum** via
`optim.poet.q_optimizer: lie_algebra`.

Per step (`oft_R` born at identity), on the identity-point tangent gradient:
`m ← β1·m + (1−β1)·g`; `v ← β2·v + (1−β2)·‖·‖²` (`lie_v_mode: scalar`) or
element-wise (`elementwise`); `A = −m/(√v+ε)`; `oft_R ← lr·A`. The merge
exponentiates and folds it into `W`. Momentum **persists** across the fold
(buffers `lie_m`/`lie_v`, never reset); `reinit_period: -1` keeps Ψ fixed so the
momentum stays coordinate-coherent.

Step magnitude = cosine-scheduled `lr · scale` (no RMS-α yet — expect to tune LR
*down* vs poet0; RMS-α is a later increment). Run with
[`scripts/train_poet_lie.sh`](../../scripts/train_poet_lie.sh) or
`experiment=optim/poet_lie`.

**Deferred** (later increments, §9 steps 2–5): RMS-α step scaling, low-order
Cayley, alternating single-sided, exact/block-diagonal tangent gradient, sharded
merge.
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_megatron_args.py -k poet_lie_experiment -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add configs/experiments/optim/poet_lie.yaml docs/experiments/poet_lie.md tests/unit/test_megatron_args.py && \
git commit -F - <<'EOF'
feat(poet): add poet_lie experiment config + doc (q_optimizer=lie_algebra, reinit_period=-1)
EOF
```

---

## Task 7: Training script + end-to-end dry-run

**Files:**
- Create: `scripts/train_poet_lie.sh`
- Test: `tests/unit/test_train_scripts.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_train_scripts.py`:

```python
def test_poet_lie_script_supports_llama3():
    proc = _run("train_poet_lie.sh", "llama3")
    assert "--slm-optimizer" in proc.stdout and "poet" in proc.stdout
    assert "--poet-q-optimizer" in proc.stdout
    assert "lie_algebra" in proc.stdout
    assert "--poet-lie-v-mode" in proc.stdout
    assert "--poet-merge-period" in proc.stdout
    assert "--poet-reinit-period" in proc.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run (note the `PATH` prefix — see Conventions):
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_train_scripts.py -k poet_lie -v
```
Expected: FAIL — `bash: scripts/train_poet_lie.sh: No such file or directory`.

- [ ] **Step 3: Create the script**

Clone the poet0 script and point it at `experiment=optim/poet_lie`:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
sed 's#experiment=optim/poet0#experiment=optim/poet_lie#' \
    scripts/train_poet0.sh > scripts/train_poet_lie.sh && \
chmod +x scripts/train_poet_lie.sh
```

Then replace the header comment block (lines 4–8) of `scripts/train_poet_lie.sh`:

```bash
# poet_lie variant: same harness as train_poet0.sh (tiny 60m dev scale,
# seq_length=256, ablation_40x, cosine_poet, untied embeddings), but uses
# experiment=optim/poet_lie — POET x Pion increment 1: Lie-algebra momentum on
# oft_R (q_optimizer=lie_algebra), single-step (merge_period=1), reinit_period=-1
# (fixed Psi, persistent momentum). Any "$@" override still wins.
```

Verify the only functional difference from poet0 is the experiment:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
diff <(grep -v '^#' scripts/train_poet0.sh) <(grep -v '^#' scripts/train_poet_lie.sh)
```
Expected: a single hunk changing `experiment=optim/poet0` → `experiment=optim/poet_lie`.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && \
PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_train_scripts.py -k poet_lie -v
```
Expected: PASS. (The dry-run resolves `experiment=optim/poet_lie` → emits
`--poet-q-optimizer lie_algebra`, `--poet-lie-*`, `--poet-merge-period 1`,
`--poet-reinit-period -1`.)

- [ ] **Step 5: Commit**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
git add scripts/train_poet_lie.sh tests/unit/test_train_scripts.py && \
git commit -F - <<'EOF'
feat(poet): add train_poet_lie.sh launcher (q_optimizer=lie_algebra)
EOF
```

---

## Final verification (after all tasks)

- [ ] **In-process unit tests (Tasks 1–6):**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_lie_momentum.py \
  tests/unit/test_patch_poet_merge.py \
  tests/unit/test_pretrain_gpt_slm.py \
  tests/unit/test_megatron_args.py -v
```
Expected: all new tests PASS. (Pre-existing reds in `test_megatron_args.py` — the
`--poet-merge-period == "200"` assertion and two `wandb_naming` `-scale` ones —
are unrelated; confirmed red before this work.)

- [ ] **Script dry-run test (Task 7) with venv on PATH:**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
PATH=/lustre/fast/fast/zqiu/slm_env/.venv/bin:$PATH \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_train_scripts.py -k "poet_lie or poet0" -v
```
Expected: `test_poet_lie_script_supports_llama3` + the poet0 script test PASS.

- [ ] **Static checks:**

```bash
cd /lustre/fast/fast/zqiu/slm-research && \
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m py_compile \
  src/optim/poet_lie_momentum.py src/optim/poet.py \
  launchers/pretrain_gpt_slm.py src/utils/megatron_args.py \
  src/patches/poet_optimizer_setup.py && \
ruff check src/optim/poet_lie_momentum.py src/optim/poet.py \
  launchers/pretrain_gpt_slm.py src/utils/megatron_args.py \
  src/patches/poet_optimizer_setup.py
```

- [ ] **GPU run (USER — do not run from the agent):** the 60m dev ablation; ablate
  `lie_v_mode` and tune LR (expect lower than poet0's 1e-3 — no RMS-α yet):

```bash
codexlog poet_lie_scalar bash scripts/train_poet_lie.sh llama3 optim.poet.lie_v_mode=scalar
codexlog poet_lie_elem   bash scripts/train_poet_lie.sh llama3 optim.poet.lie_v_mode=elementwise
```
Watch: no per-step spikes (momentum persists, Ψ fixed), and val loss moving off
the POET-Adam baseline toward Muon (§9 step 1). The skew-side LR decays via the
cosine schedule on the scaled skew param group (verify in the run's LR log).

---

## Self-Review Notes (author)

- **Spec coverage:** §2.1 (dispatch)→T3; §2.2 (identity-point grad)→T1 (uses `oft_R.grad` directly); §2.3 (block_count=1 oracle)→T1 equivalence tests; §3 decisions→T1/T6; §4 algorithm→T1; §5 touch points→T3–T7; §6 LR wiring→T3 (`_build_lie_param_groups` sets max_lr/min_lr, scheduler decays group lr); §7 persistence→T1/T2; §8 deferrals→T6 doc; §9 tests→T1–T7. All covered.
- **vec-space ≡ skew-space** verified on CPU (scalar-v ×2 Frobenius factor) before writing; the T1 reference uses `vec_to_skew`/`skew_to_vec` to build the paper-faithful oracle.
- **Naming consistency:** `LieAlgebraMomentum`, `_build_lie_param_groups`, `lie_m`/`lie_v`, `lie_b1/lie_b2/lie_eps/lie_v_mode`, `--poet-lie-*`, `q_optimizer=lie_algebra` used identically across all tasks.
- **One open risk (GPU-only):** that Megatron's `OptimizerParamScheduler` decays `group["lr"]` for the custom optimizer's groups. Mitigated by setting `max_lr`/`min_lr` on every group (the scheduler's contract). If the skew LR does not decay in the run's LR log, fall back to a fixed `lie_eta` config (spec §6 fallback).
```
