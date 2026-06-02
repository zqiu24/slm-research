# POET Muon-on-Q (Stage 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Step POET's skew generators `oft_R` with a Muon-style spectral update (orthogonalize the per-block skew gradient via Newton–Schulz, rescale to a constant rotation angle) instead of AdamW, selectable by `optim.poet.q_optimizer: adam|muon`, to fix the heavy-tailed `∂f/∂Q` conditioning Probe 0B confirmed.

**Architecture:** A new `SkewMuon` optimizer (hybrid: `oft_R`→skew-Muon, everything else→AdamW, mirroring `src/optim/_kimi_muon.Muon`) built into the POET optimizer path. Per `oft_R` block: momentum → inflate vector to `b×b` skew → batched Newton–Schulz → re-skew → constant-angle scale → deflate → step. No merge-reset (`merge_period=0`); momentum accumulates. Parameterization-agnostic, run as a 2×2 {AdamW-on-Q, SkewMuon-on-Q} × {cayley, exp}.

**Tech Stack:** PyTorch, Megatron-LM (`Float16OptimizerWithFloat16Params`), the slm-research POET optimizer path + patch registry, Hydra config.

**Spec:** [`docs/superpowers/specs/2026-06-02-poet-muon-on-q-stage2-design.md`](../specs/2026-06-02-poet-muon-on-q-stage2-design.md)

**Test runner (CPU, all tasks):**
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest <path> -q -p no:cacheprovider
```
First collection is slow (~30–90s: torch + triton import). Run CPU tests yourself and report real output. Do **not** run GPU/training jobs — those are the user's (handed over in Task 6).

**Git note:** there is uncommitted WIP in the working tree (`poet.yaml`, dev scripts, `poet_cache.py`, `test_poet_exp_parameterization.py`). Stage **only** the files each task lists; never revert WIP. NFS can make commits slow — run them with `run_in_background: true` if they stall, and clear a stale `.git/index.lock` only after confirming no live git process.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/diag/skew_conditioning.py` | add `skew_to_vec` (deflate) + `block_size_from_nelems` | 1 |
| `src/optim/poet_skew_muon.py` | the `SkewMuon` optimizer + batched skew-NS helper | 2 |
| `src/optim/poet.py` | `get_megatron_poet_muon_optimizer` builder + `q_optimizer` branch | 3 |
| `src/patches/poet_optimizer_setup.py` | copy `args.poet_q_optimizer`/`muon_*` → `config` | 4 |
| `src/utils/megatron_args.py` | emit `--poet-q-optimizer` + `--poet-muon-*` | 4 |
| `launchers/pretrain_gpt_slm.py` | register the new args | 4 |
| `configs/experiments/optim/poet.yaml` | expose `q_optimizer` + `muon_*` | 4 |
| `src/diag/rotation_diag.py` | `block_rotation_diagnostics` (‖G−I‖, ‖RRᵀ−I‖) | 5 |
| `src/patches/poet_grad_conditioning.py` | also log the rotation diagnostics | 5 |
| `CHANGELOG.md` | log the feature | 6 |
| Tests | `tests/unit/test_poet_skew_muon.py`, `test_diag_skew_conditioning.py`, `test_poet_megatron_builder.py`, `test_megatron_args.py`, `test_diag_rotation.py` | 1–6 |

---

## Task 1: deflate + block-size helpers (`skew_to_vec`, `block_size_from_nelems`)

`SkewMuon` needs to go skew-matrix → upper-tri vector (the inverse of the existing `vec_to_skew`) and to recover `b` from a stored `oft_R`'s `n_elems`.

**Files:**
- Modify: `src/diag/skew_conditioning.py`
- Test: `tests/unit/test_diag_skew_conditioning.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/test_diag_skew_conditioning.py`:

```python
def test_skew_to_vec_is_inverse_of_vec_to_skew():
    from src.diag.skew_conditioning import skew_to_vec, vec_to_skew

    b = 6
    vec = torch.arange(1.0, 1.0 + 2 * (b * (b - 1) // 2)).reshape(2, b * (b - 1) // 2)
    round_trip = skew_to_vec(vec_to_skew(vec, b), b)
    assert torch.allclose(round_trip, vec)


def test_block_size_from_nelems():
    from src.diag.skew_conditioning import block_size_from_nelems

    for b in (2, 4, 8, 256, 512):
        assert block_size_from_nelems(b * (b - 1) // 2) == b
```

- [ ] **Step 2: Run to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_skew_conditioning.py -q -p no:cacheprovider -k "skew_to_vec or block_size_from"`
Expected: FAIL — `cannot import name 'skew_to_vec'` / `'block_size_from_nelems'`.

- [ ] **Step 3: Implement** — add to `src/diag/skew_conditioning.py` (note: `import math` at top if not present):

```python
import math


def block_size_from_nelems(n_elems: int) -> int:
    """Recover block size b from the strictly-upper-triangular count
    n_elems = b*(b-1)/2  =>  b = (1 + sqrt(1 + 8*n_elems)) / 2."""
    return (1 + math.isqrt(1 + 8 * int(n_elems))) // 2


def skew_to_vec(skew: torch.Tensor, block_size: int) -> torch.Tensor:
    """Inverse of ``vec_to_skew``: extract the strictly-upper-triangular entries
    (same ``triu_indices(b,b,1)`` order POET stores).

    Args:
        skew: (num_blocks, b, b) (or (b, b)).
        block_size: b.
    Returns: (num_blocks, b*(b-1)/2).
    """
    if skew.dim() == 2:
        skew = skew.unsqueeze(0)
    b = block_size
    rows, cols = torch.triu_indices(b, b, 1)
    return skew[:, rows, cols]
```

- [ ] **Step 4: Run to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_skew_conditioning.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/diag/skew_conditioning.py tests/unit/test_diag_skew_conditioning.py
git commit -m "feat(diag): add skew_to_vec deflate + block_size_from_nelems"
```

---

## Task 2: the `SkewMuon` optimizer

The core new piece. A hybrid `torch.optim.Optimizer`: `oft_R` params get the skew-Muon update, all others get AdamW — mirroring [`src/optim/_kimi_muon.Muon`](../../../src/optim/_kimi_muon.py#L48). Uses a **batched fp32 Newton–Schulz** (not the single-2D bf16 `zeropower_via_newtonschulz5`) so it's CPU-testable and precise on skew blocks.

**Files:**
- Create: `src/optim/poet_skew_muon.py`
- Test: `tests/unit/test_poet_skew_muon.py`

- [ ] **Step 1: Write the failing tests** — create `tests/unit/test_poet_skew_muon.py`:

```python
import torch

from src.diag.skew_conditioning import block_spectral_stats, vec_to_skew
from src.optim.poet_skew_muon import SkewMuon, orthogonalize_skew_blocks


def _heavy_tailed_skew_vec(num_blocks, b):
    # one dominant rotation direction + small noise => low stable rank
    ne = b * (b - 1) // 2
    v = torch.randn(num_blocks, ne) * 1e-3
    v[:, 0] = 10.0
    return v


def test_ns_flattens_the_skew_spectrum():
    b, num_blocks = 16, 2
    Q = vec_to_skew(_heavy_tailed_skew_vec(num_blocks, b), b)
    sr_in = block_spectral_stats(Q)["stable_rank"].mean().item()
    X = orthogonalize_skew_blocks(Q.float(), ns_steps=5)
    X = (X - X.transpose(-2, -1)) / 2  # re-skew
    sr_out = block_spectral_stats(X)["stable_rank"].mean().item()
    # heavy-tailed input (~1-2) becomes broadly spread (>= b/4)
    assert sr_in < 3.0
    assert sr_out > b / 4
    assert sr_out > 4 * sr_in


def test_constant_angle_scaling_hits_theta():
    b, ne = 8, 8 * 7 // 2
    p = torch.nn.Parameter(torch.zeros(1, ne))
    p.grad = torch.randn(1, ne)
    opt = SkewMuon(skew_params=[p], adamw_params=[], theta=0.3, ns_steps=5, momentum=0.0)
    opt.step()
    # the realized skew step ||step||_F per block == theta (oft_R moved from 0 by -step)
    from src.diag.skew_conditioning import vec_to_skew as _v
    step_fro = torch.linalg.matrix_norm(_v(-p.data, b), ord="fro", dim=(-2, -1))
    assert torch.allclose(step_fro, torch.full_like(step_fro, 0.3), atol=1e-4)


def test_adamw_branch_steps_non_skew_params():
    w = torch.nn.Parameter(torch.randn(4, 4))
    w.grad = torch.randn(4, 4)
    w0 = w.data.clone()
    opt = SkewMuon(skew_params=[], adamw_params=[w], theta=0.3, adamw_lr=1e-2)
    opt.step()
    assert not torch.allclose(w.data, w0)  # moved
    assert opt.state[w]["use_skew"] is False


def test_skew_param_stays_a_valid_skew_vector():
    b, ne = 8, 8 * 7 // 2
    p = torch.nn.Parameter(torch.randn(3, ne) * 0.1)
    p.grad = torch.randn(3, ne)
    opt = SkewMuon(skew_params=[p], adamw_params=[], theta=0.2, ns_steps=5, momentum=0.95)
    opt.step()
    assert p.data.shape == (3, ne)  # still the (n_blocks, n_elems) skew-vector layout
    assert torch.isfinite(p.data).all()
    assert opt.state[p]["use_skew"] is True
    assert "momentum_buffer" in opt.state[p]
```

- [ ] **Step 2: Run to verify they fail**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_skew_muon.py -q -p no:cacheprovider`
Expected: FAIL — `No module named 'src.optim.poet_skew_muon'`.

- [ ] **Step 3: Implement** — create `src/optim/poet_skew_muon.py`:

```python
"""SkewMuon: Muon-on-Q optimizer for POET's skew generators (Stage 2).

Hybrid optimizer mirroring src/optim/_kimi_muon.Muon: params tagged "skew"
(POET's oft_R, shape (n_blocks, n_elems)) get the skew-Muon update; all other
params get AdamW. The skew update orthogonalizes the *per-block b x b skew
matrix* of the gradient (NOT the raw (n_blocks, n_elems) tensor — standard Muon
on that would mix blocks/entries), then rescales to a constant rotation angle.

Parameterization-agnostic: it only touches oft_R and its grad, so it works for
cayley or exp. Designed for the no-reset regime (merge_period=0): momentum
accumulates over the whole run. Single-process (DP-replicated, no sharding) like
muon_kimi; integration lives in src/optim/poet.py.
"""

from __future__ import annotations

import torch

from src.diag.skew_conditioning import block_size_from_nelems, skew_to_vec, vec_to_skew

# Quintic Newton-Schulz coefficients (same as src/optim/_kimi_muon).
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def orthogonalize_skew_blocks(Q: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """Batched quintic Newton-Schulz over a (num_blocks, b, b) batch (fp32).

    Returns ~orthogonal blocks (singular values driven toward ~uniform). The
    caller re-skew-symmetrizes the result. fp32 + batched (vs _kimi_muon's
    single-2D bf16 zeropower) for CPU-testability and skew precision.
    """
    norm = torch.linalg.matrix_norm(Q, ord="fro", dim=(-2, -1), keepdim=True)
    X = Q / (norm + 1e-7)
    for _ in range(ns_steps):
        A = X @ X.transpose(-2, -1)
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    return X


class SkewMuon(torch.optim.Optimizer):
    def __init__(
        self,
        skew_params=None,
        adamw_params=None,
        theta: float = 0.1,
        ns_steps: int = 5,
        momentum: float = 0.95,
        nesterov: bool = True,
        adamw_lr: float = 1e-3,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        skew_params = list(skew_params) if skew_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        defaults = dict(
            theta=theta,
            ns_steps=ns_steps,
            momentum=momentum,
            nesterov=nesterov,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(skew_params + adamw_params, defaults)
        for p in skew_params:
            assert p.ndim == 2, p.shape  # (n_blocks, n_elems)
            self.state[p]["use_skew"] = True
        for p in adamw_params:
            self.state[p]["use_skew"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # ---- skew-Muon branch (oft_R) ----
            for p in (p for p in group["params"] if self.state[p]["use_skew"]):
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                b = block_size_from_nelems(g.shape[-1])
                Q = vec_to_skew(g.float(), b)
                X = orthogonalize_skew_blocks(Q, group["ns_steps"])
                X = (X - X.transpose(-2, -1)) / 2  # re-skew to stay in so(b)
                fro = torch.linalg.matrix_norm(X, ord="fro", dim=(-2, -1), keepdim=True)
                step_skew = group["theta"] * X / (fro + 1e-8)
                step_vec = skew_to_vec(step_skew, b).to(p.dtype)
                p.add_(step_vec, alpha=-1.0)

            # ---- AdamW branch (everything else) ----
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            lr = group["adamw_lr"]
            wd = group["adamw_wd"]
            for p in (p for p in group["params"] if not self.state[p]["use_skew"]):
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                m1, m2 = state["moment1"], state["moment2"]
                m1.lerp_(g, 1 - beta1)
                m2.lerp_(g.square(), 1 - beta2)
                update = m1 / (eps + m2.sqrt())
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                scale = bc1 / bc2**0.5
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr / scale)

        return loss
```

- [ ] **Step 4: Run to verify they pass**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_skew_muon.py -q -p no:cacheprovider`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_skew_muon.py tests/unit/test_poet_skew_muon.py
git commit -m "feat(poet): add SkewMuon optimizer (inflate->NS->constant-angle->deflate)"
```

---

## Task 3: Megatron builder + `q_optimizer` branch

Route `oft_R`→SkewMuon, rest→AdamW, wrapped in Megatron's mixed-precision optimizer — mirroring [`src/optim/muon_kimi.py`](../../../src/optim/muon_kimi.py). Branch into it from `get_megatron_poet_optimizer` when `q_optimizer=muon`.

**Files:**
- Modify: `src/optim/poet.py` (`get_megatron_poet_optimizer` ~line 464; add `get_megatron_poet_muon_optimizer`)
- Test: `tests/unit/test_poet_skew_muon.py`

- [ ] **Step 1: Write the failing test** — append to `tests/unit/test_poet_skew_muon.py`:

```python
def test_split_skew_vs_adamw_by_name():
    """oft_R params -> skew branch, everything else -> adamw (pure split logic)."""
    from src.optim.poet import _split_poet_muon_params

    class P(torch.nn.Parameter):
        pass

    import torch.nn as nn

    class FakeChunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Module()
            self.layer.oft_R_in = nn.Parameter(torch.zeros(1, 6))
            self.layer.oft_R_out = nn.Parameter(torch.zeros(1, 6))
            self.embedding = nn.Parameter(torch.zeros(8, 8))  # non-oft_R

    chunk = FakeChunk()
    skew, adamw = _split_poet_muon_params([chunk])
    assert len(skew) == 2  # oft_R_in, oft_R_out
    assert len(adamw) == 1  # embedding
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_skew_muon.py -q -p no:cacheprovider -k "split_skew"`
Expected: FAIL — `cannot import name '_split_poet_muon_params'`.

- [ ] **Step 3: Implement** — in `src/optim/poet.py`, add a split helper and the muon builder, and branch in `get_megatron_poet_optimizer`.

First, add near the top of `get_megatron_poet_optimizer` (right after `poet_cache_mode = getattr(config, "poet_cache_mode", "none")`, around line 466):

```python
    if getattr(config, "poet_q_optimizer", "adam") == "muon":
        return get_megatron_poet_muon_optimizer(
            config,
            model_chunks,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        )
```

Then add these two module-level functions (e.g. just above `get_megatron_poet_optimizer`):

```python
def _split_poet_muon_params(model_chunks):
    """oft_R params -> skew (SkewMuon); all other trainable params -> AdamW."""
    skew_params, adamw_params = [], []
    for mc in model_chunks:
        for name, param in mc.named_parameters():
            if not param.requires_grad:
                continue
            (skew_params if "oft_R" in name else adamw_params).append(param)
    return skew_params, adamw_params


def get_megatron_poet_muon_optimizer(
    config,
    model_chunks,
    *,
    config_overrides=None,
    use_gloo_process_groups: bool = True,
):
    """POET Muon-on-Q: SkewMuon on oft_R, AdamW on the rest, wrapped for Megatron.
    Single-process / DP-replicated (no sharded distributed optimizer), like
    muon_kimi. Designed for the no-reset regime (merge_period=0)."""
    _resolve_megatron_handles()
    from megatron.core import parallel_state as mpu
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )

    from src.optim.poet_skew_muon import SkewMuon

    if getattr(config, "use_distributed_optimizer", False):
        raise ValueError("POET Muon-on-Q does not support the distributed optimizer (dev only).")
    if getattr(config, "fp16", False):
        raise ValueError("POET Muon-on-Q does not support fp16; use bf16.")
    if mpu.get_tensor_model_parallel_world_size() > 1:
        raise ValueError("POET Muon-on-Q does not support tensor parallelism > 1.")
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        raise ValueError("POET Muon-on-Q does not support pipeline parallelism > 1.")

    skew_params, adamw_params = _split_poet_muon_params(model_chunks)
    logger.info(
        "[POET] Muon-on-Q: %d skew (oft_R) params, %d adamw params (theta=%s, ns_steps=%s)",
        len(skew_params),
        len(adamw_params),
        getattr(config, "poet_muon_theta", 0.1),
        getattr(config, "poet_muon_ns_steps", 5),
    )
    if not skew_params:
        logger.warning("[POET] Muon-on-Q: no oft_R params found — SkewMuon is a no-op.")

    optimizer = SkewMuon(
        skew_params=skew_params,
        adamw_params=adamw_params,
        theta=getattr(config, "poet_muon_theta", 0.1),
        ns_steps=getattr(config, "poet_muon_ns_steps", 5),
        momentum=getattr(config, "poet_muon_momentum", 0.95),
        nesterov=True,
        adamw_lr=config.lr,
        adamw_betas=(config.adam_beta1, config.adam_beta2),
        adamw_eps=config.adam_eps,
        adamw_wd=config.weight_decay,
    )

    def init_state_fn(opt, _config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state[p]
                if st.get("use_skew", False):
                    st.setdefault("momentum_buffer", torch.zeros_like(p.data))
                elif "moment1" not in st:
                    st["step"] = 0
                    st["moment1"] = torch.zeros_like(p.data)
                    st["moment2"] = torch.zeros_like(p.data)

    if getattr(config, "bf16", False):
        return Float16OptimizerWithFloat16Params(optimizer, config, None, init_state_fn)
    return FP32Optimizer(optimizer, config, init_state_fn)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_skew_muon.py -q -p no:cacheprovider`
Expected: all pass (5 total).

- [ ] **Step 5: Run the existing POET builder tests (no regression to the adam path)**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_megatron_builder.py -q -p no:cacheprovider`
Expected: pass (the `q_optimizer` branch only triggers when `config.poet_q_optimizer=="muon"`; default `"adam"` path unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet.py tests/unit/test_poet_skew_muon.py
git commit -m "feat(poet): Muon-on-Q builder + q_optimizer=muon branch (SkewMuon hybrid)"
```

---

## Task 4: config plumbing (`q_optimizer`, `muon_*`)

Thread `optim.poet.q_optimizer` (+ `muon_theta`/`muon_ns_steps`/`muon_momentum`) to `config.poet_*`, the chain `poet_scale` already uses.

**Files:**
- Modify: `configs/experiments/optim/poet.yaml`
- Modify: `src/utils/megatron_args.py` (poet branch, after the `--poet-parameterization` pair)
- Modify: `launchers/pretrain_gpt_slm.py` (after the poet args)
- Modify: `src/patches/poet_optimizer_setup.py:41` (`_wrapped_get_config`)
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing test** — append to `tests/unit/test_megatron_args.py`:

```python
def test_poet_argv_includes_q_optimizer_and_muon_knobs():
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
                    "q_optimizer": "muon",
                    "muon_theta": 0.2,
                    "muon_ns_steps": 5,
                    "muon_momentum": 0.95,
                },
            }
        }
    )
    args = _optimizer_args(cfg)
    assert args[args.index("--poet-q-optimizer") + 1] == "muon"
    assert args[args.index("--poet-muon-theta") + 1] == 0.2


def test_poet_argv_q_optimizer_defaults_to_adam():
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
                "poet": {"block_size": 8, "init_type": "none", "mup_alpha": 1.0,
                         "merge_period": 0, "scale": 1.0},
            }
        }
    )
    args = _optimizer_args(cfg)
    assert args[args.index("--poet-q-optimizer") + 1] == "adam"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -q -p no:cacheprovider -k "q_optimizer"`
Expected: FAIL — `--poet-q-optimizer` not in args.

- [ ] **Step 3a: Emit args** — in `src/utils/megatron_args.py`, in the `kind == "poet"` branch, append to the `poet_args` list right after the `--poet-parameterization` pair:

```python
            "--poet-q-optimizer",
            poet.get("q_optimizer", "adam"),
            "--poet-muon-theta",
            poet.get("muon_theta", 0.1),
            "--poet-muon-ns-steps",
            poet.get("muon_ns_steps", 5),
            "--poet-muon-momentum",
            poet.get("muon_momentum", 0.95),
```

- [ ] **Step 3b: Register args** — in `launchers/pretrain_gpt_slm.py`, after the `--poet-parameterization` argument block, add:

```python
    group.add_argument("--poet-q-optimizer", choices=["adam", "muon"], default="adam")
    group.add_argument("--poet-muon-theta", type=float, default=0.1)
    group.add_argument("--poet-muon-ns-steps", type=int, default=5)
    group.add_argument("--poet-muon-momentum", type=float, default=0.95)
```

- [ ] **Step 3c: Copy args → config** — in `src/patches/poet_optimizer_setup.py`, in `_wrapped_get_config`, after `config.poet_use_poet_adam = ...` (line 41), add:

```python
        config.poet_q_optimizer = getattr(args, "poet_q_optimizer", "adam")
        config.poet_muon_theta = getattr(args, "poet_muon_theta", 0.1)
        config.poet_muon_ns_steps = getattr(args, "poet_muon_ns_steps", 5)
        config.poet_muon_momentum = getattr(args, "poet_muon_momentum", 0.95)
```

- [ ] **Step 3d: Expose config** — in `configs/experiments/optim/poet.yaml`, under `optim.poet:` (after `parameterization: cayley`), add:

```yaml
    # Optimizer that steps the orthogonal generators oft_R:
    #   "adam" (default): stock Megatron-Adam on oft_R (unchanged).
    #   "muon": SkewMuon — orthogonalize each block's skew gradient (Newton-Schulz)
    #     and rescale to a constant rotation angle muon_theta. Fixes heavy-tailed
    #     df/dQ conditioning (Probe 0B). Use with merge_period=0 (no-reset regime).
    q_optimizer: adam
    muon_theta: 0.1        # target per-step rotation angle (the single Muon tunable)
    muon_ns_steps: 5       # Newton-Schulz iterations
    muon_momentum: 0.95
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py tests/unit/test_patch_poet_optimizer_setup.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/optim/poet.yaml src/utils/megatron_args.py launchers/pretrain_gpt_slm.py src/patches/poet_optimizer_setup.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): plumb optim.poet.q_optimizer + muon_* knobs"
```

---

## Task 5: rotation diagnostics (`‖G−I‖`, `‖RRᵀ−I‖`)

Per the spec: log the realized per-block rotation angle `‖G−I‖_F` (calibration) and the orthogonality drift `‖RRᵀ−I‖_F` (validity gate for the cayley no-reset arms). Reuse the existing R-builders and fold into the conditioning probe.

**Files:**
- Create: `src/diag/rotation_diag.py`
- Modify: `src/patches/poet_grad_conditioning.py` (log the new metrics)
- Test: `tests/unit/test_diag_rotation.py`

- [ ] **Step 1: Write the failing test** — create `tests/unit/test_diag_rotation.py`:

```python
import torch


def test_block_rotation_diagnostics_on_identity_and_rotation():
    from src.diag.rotation_diag import block_rotation_diagnostics

    # R = I (zero skew) -> angle 0, ortho-error 0
    R_eye = torch.eye(4).unsqueeze(0)
    d0 = block_rotation_diagnostics(R_eye)
    assert d0["g_minus_i"][0].item() < 1e-6
    assert d0["ortho_err"][0].item() < 1e-6

    # a real rotation: exp of a skew block -> orthogonal (ortho_err ~ 0), angle > 0
    Q = torch.zeros(1, 4, 4)
    Q[0, 0, 1], Q[0, 1, 0] = 0.5, -0.5
    R = torch.linalg.matrix_exp(Q)
    d1 = block_rotation_diagnostics(R)
    assert d1["ortho_err"][0].item() < 1e-5
    assert d1["g_minus_i"][0].item() > 0.1
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_rotation.py -q -p no:cacheprovider`
Expected: FAIL — `No module named 'src.diag.rotation_diag'`.

- [ ] **Step 3: Implement** — create `src/diag/rotation_diag.py`:

```python
"""Per-block rotation diagnostics for POET Muon-on-Q (Stage 2).

Given a batch of block rotation matrices R = f(Q) ((num_blocks, b, b)), report:
  - g_minus_i  = ||R - I||_F  : realized per-block rotation magnitude (angle proxy;
    the spec's theta-calibration check).
  - ortho_err  = ||R R^T - I||_F : orthogonality-approximation error (validity gate
    for the cayley no-reset arms, where the Cayley-Neumann series can leave its
    convergence regime).
"""

from __future__ import annotations

import torch


def block_rotation_diagnostics(R: torch.Tensor) -> dict[str, torch.Tensor]:
    if R.dim() == 2:
        R = R.unsqueeze(0)
    R = R.to(torch.float32)
    b = R.shape[-1]
    eye = torch.eye(b, device=R.device, dtype=R.dtype)
    g_minus_i = torch.linalg.matrix_norm(R - eye, ord="fro", dim=(-2, -1))
    ortho_err = torch.linalg.matrix_norm(
        R @ R.transpose(-2, -1) - eye, ord="fro", dim=(-2, -1)
    )
    return {"g_minus_i": g_minus_i, "ortho_err": ortho_err}
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_diag_rotation.py -q -p no:cacheprovider`
Expected: pass.

- [ ] **Step 5: Wire into the conditioning probe** — in `src/patches/poet_grad_conditioning.py`, in `_log_conditioning`, after the existing `block_spectral_stats` logging, also build `R` for each target's block and log the rotation diagnostics. Inside the `for t in targets:` loop, after the `wandb.log({...})` for the spectral stats, add:

```python
        # rotation diagnostics: build R = f(Q) for this block's CURRENT oft_R and
        # log realized-angle ||G-I|| + orthogonality-drift ||RR^T-I|| (Stage-2 calibration).
        try:
            from poet_torch.poet_layer import (
                get_weight_poet_decoupled,
                get_weight_poet_decoupled_exp,
                pytorch_skew_symmetric,
            )

            from src.diag.rotation_diag import block_rotation_diagnostics

            oft = param.detach()
            bs = t["block_size"]
            rows, cols = torch.triu_indices(bs, bs, 1, device=oft.device)
            Q = pytorch_skew_symmetric(oft.float(), bs, rows.to(torch.int32), cols.to(torch.int32))
            param_kind = getattr(t.get("layer"), "parameterization", "cayley")
            R = (
                torch.linalg.matrix_exp(Q)
                if param_kind == "exp"
                else torch.ops.poet.cayley(Q)[0]
            )
            rd = block_rotation_diagnostics(R)
            if wandb is not None and getattr(wandb, "run", None) is not None:
                wandb.log(
                    {
                        f"poet_rot/{t['label']}/g_minus_i": rd["g_minus_i"].mean().item(),
                        f"poet_rot/{t['label']}/ortho_err": rd["ortho_err"].mean().item(),
                    },
                    step=iteration,
                )
        except Exception:  # diagnostics must never break training
            logger.exception("[COND] rotation diag failed for %s", t["label"])
```

Also, in `select_target_params`, add `"layer": layer` to each target dict so `t.get("layer")` resolves (find the `targets.append({...})` call and add the `layer` key). Note: on CPU the `torch.ops.poet.cayley` Triton op is unavailable, so this block only runs meaningfully on GPU; on CPU it logs the exception and continues (the rotation_diag helper itself is CPU-tested in isolation above).

- [ ] **Step 6: Run the probe tests (no regression)**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_patch_poet_grad_conditioning.py tests/unit/test_diag_rotation.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/diag/rotation_diag.py src/patches/poet_grad_conditioning.py tests/unit/test_diag_rotation.py
git commit -m "feat(diag): per-block rotation diagnostics (||G-I||, ||RR^T-I||) for Muon-on-Q"
```

---

## Task 6: full suite, CHANGELOG, GPU handover

**Files:** `CHANGELOG.md`

- [ ] **Step 1: Run the full new + adjacent CPU suite**

Run:
```
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_poet_skew_muon.py tests/unit/test_diag_skew_conditioning.py \
  tests/unit/test_diag_rotation.py tests/unit/test_megatron_args.py \
  tests/unit/test_patch_poet_optimizer_setup.py tests/unit/test_poet_megatron_builder.py \
  tests/unit/test_patch_poet_grad_conditioning.py -q -p no:cacheprovider
```
Expected: all pass (report real counts). Then `ruff check` the new files.

- [ ] **Step 2: CHANGELOG**

Add to the top of the `## Unreleased` section in `CHANGELOG.md`:

```markdown
### Added — POET Muon-on-Q (Stage 2): SkewMuon optimizer

- **`oft_R` can now be optimized by a Muon-style spectral update** instead of
  AdamW, via `optim.poet.q_optimizer: adam|muon` (+ `muon_theta`/`muon_ns_steps`/
  `muon_momentum`). `SkewMuon` (`src/optim/poet_skew_muon.py`) inflates each
  block's skew gradient to `b×b`, Newton–Schulz-orthogonalizes it, re-skews, and
  rescales to a constant rotation angle `muon_theta`, then steps `oft_R`; all
  non-`oft_R` params stay AdamW (hybrid, `muon_kimi` pattern). Built into the POET
  optimizer path (`get_megatron_poet_muon_optimizer`), single-process/DP-replicated.
  Motivated by Probe 0B (heavy-tailed `∂f/∂Q`). Per-block `‖G−I‖`/`‖RRᵀ−I‖`
  diagnostics added. Default (`adam`) unchanged. Intended for the no-reset regime
  (`merge_period=0`).
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(poet): changelog for Muon-on-Q Stage 2 (SkewMuon)"
```

- [ ] **Step 4: Hand over the GPU 2×2 (USER runs)**

The 2×2, all no-reset (`merge_period=0`), at 60m dev scale. `POET_MEM_EFFICIENT=1` required for cayley arms (recompute chain). The Muon arms also turn on the rotation/conditioning diagnostics via the cond wrapper. Provide these:

```bash
# --- AdamW-on-Q baselines (q_optimizer defaults to adam) ---
POET_MEM_EFFICIENT=1 codexlog s2_A_cay bash scripts/train_poet_dev.sh \
  optim.poet.merge_period=0
codexlog s2_A_exp bash scripts/train_poet_dev.sh \
  optim.poet.merge_period=0 optim.poet.parameterization=exp

# --- SkewMuon-on-Q arms ---
POET_MEM_EFFICIENT=1 codexlog s2_M_cay bash scripts/train_poet_dev.sh \
  optim.poet.merge_period=0 optim.poet.q_optimizer=muon
codexlog s2_M_exp bash scripts/train_poet_dev.sh \
  optim.poet.merge_period=0 optim.poet.q_optimizer=muon optim.poet.parameterization=exp
```
First-run checks: rank-0 log shows `[POET] Muon-on-Q: N skew (oft_R) params ...` (N>0) for the muon arms; for the cayley arms watch `poet_rot/*/ortho_err` in W&B — if it grows large, that arm has left Cayley's convergence regime and is invalid past that step (exp arms stay ~0). Grid `optim.poet.muon_theta` if the realized `poet_rot/*/g_minus_i` is too small/large vs the AdamW-on-Q baseline.

---

## Self-Review (completed by plan author)

**Spec coverage:** Unit 1 SkewMuon (→ Task 2); Unit 2 hybrid wiring/no-reset (→ Task 3, no reset logic built); Unit 3 config (→ Task 4); Unit 4 CPU tests + `‖G−I‖`/`‖RRᵀ−I‖` (→ Tasks 2,5); deflate/`b`-inversion helpers (→ Task 1); 2×2 cayley/exp run handover (→ Task 6). Reuse of `vec_to_skew`/`block_spectral_stats`/`zeropower` coefficients honored. Out-of-scope (Stage 3 transport, Stage 4, `‖ΔW‖`-matched) not built.

**Placeholder scan:** no TBD/TODO; every code step has complete code; every test has real assertions; the one GPU-only path (Task 5 rotation diag using the Triton cayley op) is explicitly flagged CPU-skipping with the helper CPU-tested in isolation.

**Type/name consistency:** `skew_to_vec`/`block_size_from_nelems` (Task 1) consumed in Task 2; `SkewMuon(skew_params, adamw_params, theta, ns_steps, momentum, nesterov, adamw_lr, adamw_betas, adamw_eps, adamw_wd)` + `orthogonalize_skew_blocks(Q, ns_steps)` (Task 2) consumed in Task 3; `_split_poet_muon_params`/`get_megatron_poet_muon_optimizer` (Task 3); `config.poet_q_optimizer`/`poet_muon_theta`/`poet_muon_ns_steps`/`poet_muon_momentum` consistent across Tasks 3–4; `block_rotation_diagnostics` returns `{g_minus_i, ortho_err}` (Task 5). Consistent.
