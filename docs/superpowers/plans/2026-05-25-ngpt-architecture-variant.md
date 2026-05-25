# nGPT Architecture Variant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the **nGPT** (Normalized Transformer on the Hypersphere, [arXiv:2410.01131](https://arxiv.org/abs/2410.01131)) architecture as a first-class *architecture variant* of slm-research, runnable end-to-end on the existing 600M dense scale. Reference implementation is the NVIDIA repo vendored at [/lustre/fast/fast/zqiu/tmp/ngpt](file:///lustre/fast/fast/zqiu/tmp/ngpt).

**Architecture:** nGPT is integrated through (a) a custom `NGPTTransformerLayer` (Megatron `ModuleSpec`) with custom `NGPTSelfAttention` / `NGPTMLP` submodules that inject learnable hypersphere scaling parameters (`sqk`, `suv`, `attn_alpha`, `mlp_alpha`); (b) a small set of slm-research patches (`ngpt_apply_spec`, `ngpt_normalize_step`, `ngpt_optimizer_setup`) that swap the GPT layer spec, run post-step weight L2-normalization, and zero weight-decay on the scaling params; (c) a custom `output_scaling.py` that attaches an `sz` parameter to the model post-build and post-multiplies logits; (d) a numerical parity oracle against the vendored NVIDIA reference at a tiny CPU-runnable config; (e) a `configs/experiments/arch/ngpt.yaml` plus lab-notebook. Megatron-LM upstream is untouched; integration mirrors the POET pattern (`src/patches/poet_*.py`).

**Tech Stack:** PyTorch, Megatron-LM Core (pinned via `third_party/Megatron-LM`), Hydra/OmegaConf configs, the slm-research patch registry (`src/patches/_registry.py`).

**Reference oracle:** [/lustre/fast/fast/zqiu/tmp/ngpt/model.py](file:///lustre/fast/fast/zqiu/tmp/ngpt/model.py) and [/lustre/fast/fast/zqiu/tmp/ngpt/train.py](file:///lustre/fast/fast/zqiu/tmp/ngpt/train.py). These will be vendored *unmodified* into `tests/_fixtures/ngpt_reference/` solely as a parity oracle — they are never imported from `src/`.

---

## Prerequisites (executor reads first)

1. **All work happens in an isolated worktree. Do not commit to the parent branch until the merge gate in Task 16 has passed.** Use the `superpowers:using-git-worktrees` skill to create the worktree:
   - **Branch off** `poet-cayley-cache` (current active development branch with the latest infra).
   - **New branch name:** `ngpt-arch`.
   - **Worktree path** as chosen by the skill (typically a sibling directory).

   This plan does **not** include a worktree-creation task — do it before Task 1. Every commit produced by Tasks 1–15 lives on the `ngpt-arch` branch *only*. Do **not** rebase, fast-forward, or merge `ngpt-arch` into any parent branch until Task 16's gate explicitly authorizes it.

2. **Testing reality** (matches the project convention):
   - The user runs cluster jobs and reports back. Do not attempt training runs locally.
   - Numerical-parity tests are written to run on **CPU** at a tiny config (2 layers, 64 hidden, vocab 100, seq 32). A subagent without a working CUDA env can still run them.
   - GPU-only paths (flash-attention, bf16 numerics matching cluster runs) are not exercised in this plan beyond the smoke runbook in Task 14.
   - Keep [/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md](file:///lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md) updated after each landed task (per user's standing memory). Stay scoped to this plan; do not open sibling plan files under `docs/superpowers/plans/`.

3. **Merge gate (read this before starting Task 1, executed in Task 16).** The `ngpt-arch` branch is merge-eligible only after **all three** of the following have happened, in this order:
   - **(a)** Every test added in Tasks 2-14 passes on the worktree (`pytest tests/unit/test_ngpt_*.py -v` returns zero failures).
   - **(b)** The full existing test suite still passes on the worktree (`pytest tests/unit/ -v` returns zero failures), confirming no regressions to POET, muon, or launcher infra.
   - **(c)** The user has executed the smoke runbook from Task 15 on the cluster and explicitly reports success (loss decreases, no NaN, weight-norm projection visibly fires).

   If any of (a)/(b)/(c) fails, the failure must be fixed on `ngpt-arch` — do not merge a "fix it on the next PR" branch. The merge itself is performed via the `superpowers:finishing-a-development-branch` skill in Task 16, which presents merge / PR / cleanup options to the user.

4. **Scope cuts for v1 (explicit non-goals):**
   - **TP = 1 only.** Spec builder asserts `cfg.parallelism.tp == 1`. TP-aware sharding of `sqk` / `suv` is left for v2.
   - **PP = 1, no sequence-parallel.**
   - **Dense layers only** — no MoE, no MLA. Asserted in the spec builder.
   - **bf16** (matches reference). No FP8 / FP4.
   - **No KV cache / inference path.** Training only.

   These cuts are conservative; they map to the existing `600m` scale config, which is the natural first deliverable.

---

## File map

| Path | Purpose | Status |
|------|---------|--------|
| `tests/_fixtures/ngpt_reference/model.py` | Verbatim copy of NVIDIA reference, used by parity test only. | **NEW** |
| `tests/_fixtures/ngpt_reference/__init__.py` | Empty; marks fixture dir as importable. | **NEW** |
| `tests/_fixtures/ngpt_reference/NOTICE.md` | License notice + provenance. | **NEW** |
| `src/model/ngpt/__init__.py` | Re-export `NGPTTransformerLayer`, `NGPTSelfAttention`, `NGPTMLP`, helpers. | **NEW** |
| `src/model/ngpt/normalize.py` | `justnorm()`, `normalize_module_matrices()`. | **NEW** |
| `src/model/ngpt/scaling_params.py` | `_LearnedScaling(init_value, init_scaling, shape)` helper. | **NEW** |
| `src/model/ngpt/attention.py` | `NGPTSelfAttention` subclass with `sqk` + Q/K L2-norm + softmax-scale override. | **NEW** |
| `src/model/ngpt/mlp.py` | `NGPTMLP` subclass with `suv` SwiGLU intermediate scaling. | **NEW** |
| `src/model/ngpt/layer.py` | `NGPTTransformerLayer` with hypersphere residual blend (`attn_alpha`, `mlp_alpha`). | **NEW** |
| `src/model/ngpt/output_scaling.py` | `attach_ngpt_output_scaling(model)` — adds `sz` parameter; wraps `output_layer`. | **NEW** |
| `src/specs/ngpt_layer_spec.py` | `build_ngpt_layer_spec(config)` returning `ModuleSpec`. | **NEW** |
| `src/patches/ngpt_apply_spec.py` | Patch `gpt_builders.gpt_builder` (spec swap + post-build `sz` attach + weight-norm role registration) **and** `core_transformer_config_from_args` (stamp softmax_scale + ngpt_* fields onto config). | **NEW** |
| `src/patches/ngpt_normalize_step.py` | Patch `train_step` to call `normalize_module_matrices` post-step. | **NEW** |
| `src/patches/ngpt_optimizer_setup.py` | Zero WD on scaling params; install param-group classifier. | **NEW** |
| `launchers/pretrain_gpt_slm.py` | Add `--ngpt`, `--ngpt-base-scale`, `--ngpt-*-init` args. | MODIFY |
| `src/utils/megatron_args.py` | Emit `--ngpt*` flags when `experiment.kind == "ngpt"`. | MODIFY |
| `configs/experiments/arch/ngpt.yaml` | Experiment YAML. | **NEW** |
| `docs/experiments/ngpt.md` | Lab-notebook entry. | **NEW** |
| `tests/unit/test_ngpt_normalize.py` | CPU tests for `justnorm` and `normalize_module_matrices`. | **NEW** |
| `tests/unit/test_ngpt_scaling_params.py` | CPU tests for `_LearnedScaling`. | **NEW** |
| `tests/unit/test_ngpt_attention.py` | CPU test that `NGPTSelfAttention` produces unit-norm Q/K. | **NEW** |
| `tests/unit/test_ngpt_layer_block_forward.py` | CPU test: one nGPT block vs reference Block, numerical match (fp32). | **NEW** |
| `tests/unit/test_ngpt_full_parity.py` | CPU test: full 2-layer nGPT model vs reference GPT(use_nGPT=1), fp32, single forward + loss. | **NEW** |
| `tests/unit/test_ngpt_patch_registry.py` | Importing patches registers them; no conflicts; hash is deterministic. | **NEW** |
| `tests/unit/test_ngpt_megatron_args.py` | `build_megatron_args` emits `--ngpt*` flags for the nGPT experiment. | **NEW** |
| `tests/unit/test_ngpt_optimizer_groups.py` | `_classify_ngpt_param_groups` puts scaling params in the no-decay group. | **NEW** |
| `docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md` | Cluster smoke runbook. | **NEW** |

---

## Task 1: Vendor the NVIDIA nGPT reference as a parity oracle

**Files:**
- Create: `tests/_fixtures/ngpt_reference/__init__.py`
- Create: `tests/_fixtures/ngpt_reference/model.py`
- Create: `tests/_fixtures/ngpt_reference/NOTICE.md`

- [ ] **Step 1.1: Create the fixture directory and copy the reference model**

```bash
mkdir -p tests/_fixtures/ngpt_reference
cp /lustre/fast/fast/zqiu/tmp/ngpt/model.py tests/_fixtures/ngpt_reference/model.py
touch tests/_fixtures/ngpt_reference/__init__.py
```

- [ ] **Step 1.2: Make the fixture importable without `flash_attn`**

The vendored `model.py` does `from flash_attn import flash_attn_qkvpacked_func, flash_attn_func` at module top. The parity test must run on CPU (no `flash_attn`). Replace that import with a soft fallback that uses `torch.nn.functional.scaled_dot_product_attention` so the file imports under CPU. Edit `tests/_fixtures/ngpt_reference/model.py`:

Replace:
```python
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
```
with:
```python
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func  # noqa: F401
except ImportError:  # CPU / parity-test environment
    flash_attn_qkvpacked_func = None  # type: ignore

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                        causal=True, window_size=(-1, -1),
                        alibi_slopes=None, deterministic=True):
        # q,k,v shape: (B, T, H, D). SDPA expects (B, H, T, D).
        import torch.nn.functional as F
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_, k_, v_, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,
        )
        return out.transpose(1, 2).contiguous()  # back to (B, T, H, D)
```

Do **not** modify anything else in `model.py`. This is the *only* surgery on the vendored reference; the rest is upstream verbatim.

- [ ] **Step 1.3: Write the NOTICE**

Create `tests/_fixtures/ngpt_reference/NOTICE.md`:
```markdown
# nGPT reference fixture

`model.py` is a verbatim copy of `model.py` from
https://github.com/NVIDIA/ngpt (MIT License, Copyright (c) 2024 NVIDIA
CORPORATION & AFFILIATES). One surgical change: the top-level
`flash_attn` import is wrapped in a try/except so the file imports on
CPU, falling back to `torch.nn.functional.scaled_dot_product_attention`.

This file exists **solely as the numerical oracle for the parity test
in tests/unit/test_ngpt_full_parity.py**. It is never imported from
`src/` and never used at training time.
```

- [ ] **Step 1.4: Smoke-import on CPU**

Run: `python -c "from tests._fixtures.ngpt_reference.model import GPT, GPTConfig; m = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100, block_size=32, use_nGPT=1, base_scale=1.0/8.0)); print(sum(p.numel() for p in m.parameters()))"`

Expected: a non-zero parameter count, no `ImportError`.

- [ ] **Step 1.5: Commit**

```bash
git add tests/_fixtures/ngpt_reference/
git commit -m "feat(ngpt): vendor NVIDIA reference as CPU-runnable parity oracle"
```

---

## Task 2: Hypersphere primitives (`justnorm`, weight normalization)

**Files:**
- Create: `src/model/ngpt/__init__.py`
- Create: `src/model/ngpt/normalize.py`
- Create: `tests/unit/test_ngpt_normalize.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/unit/test_ngpt_normalize.py`:
```python
"""CPU tests for nGPT hypersphere normalization primitives."""
import torch
import torch.nn as nn

from src.model.ngpt.normalize import justnorm, normalize_module_matrices


def test_justnorm_unit_norm_last_dim():
    x = torch.randn(3, 5, 7)
    y = justnorm(x)
    assert torch.allclose(y.norm(dim=-1), torch.ones(3, 5), atol=1e-5)


def test_justnorm_unit_norm_explicit_dim():
    x = torch.randn(3, 5, 7)
    y = justnorm(x, dim=1)
    assert torch.allclose(y.norm(dim=1), torch.ones(3, 7), atol=1e-5)


def test_justnorm_preserves_dtype_bf16():
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    y = justnorm(x)
    assert y.dtype == torch.bfloat16
    # cast to fp32 before checking the norm: bf16 mantissa is too short
    assert torch.allclose(
        y.float().norm(dim=-1), torch.ones(2), atol=1e-2
    )


def test_normalize_module_matrices_rows_unit_norm():
    lin = nn.Linear(8, 4, bias=False)
    normalize_module_matrices({lin.weight: "rows"})
    # rows: shape (out, in)=(4,8); each of the 4 rows must be unit-norm.
    assert torch.allclose(lin.weight.data.norm(dim=1), torch.ones(4), atol=1e-5)


def test_normalize_module_matrices_cols_unit_norm():
    lin = nn.Linear(8, 4, bias=False)
    normalize_module_matrices({lin.weight: "cols"})
    # cols: each of the 8 columns must be unit-norm.
    assert torch.allclose(lin.weight.data.norm(dim=0), torch.ones(8), atol=1e-5)


def test_normalize_module_matrices_rejects_bad_role():
    lin = nn.Linear(2, 2, bias=False)
    import pytest
    with pytest.raises(ValueError, match="role"):
        normalize_module_matrices({lin.weight: "diag"})
```

- [ ] **Step 2.2: Run the test to verify it fails**

Run: `pytest tests/unit/test_ngpt_normalize.py -v`
Expected: `ModuleNotFoundError: src.model.ngpt`.

- [ ] **Step 2.3: Implement `normalize.py`**

Create `src/model/ngpt/__init__.py` (empty for now; will grow in later tasks):
```python
"""nGPT (Normalized Transformer on the Hypersphere) primitives.

See arXiv:2410.01131. Reference impl at /lustre/fast/fast/zqiu/tmp/ngpt;
parity oracle vendored under tests/_fixtures/ngpt_reference/.
"""
```

Create `src/model/ngpt/normalize.py`:
```python
"""Hypersphere normalization primitives used by nGPT.

`justnorm(x, dim)` is the per-vector L2 projection onto the unit sphere
(reference: `train.py::justnorm`, `model.py::Block.justnorm`).

`normalize_module_matrices(role_map)` does the offline matrix projection
the reference paper applies (a) once after model init and (b) after
every optimizer step. The caller passes a `{parameter -> role}` mapping
where role is "rows" or "cols", matching the reference's convention:

    role="rows" -> normalize each row to unit norm (used when the row
                   indexes an OUTPUT channel that lives on the sphere,
                   e.g. wte, lm_head, q/k/v, c_fc)
    role="cols" -> normalize each column to unit norm (used when the
                   column indexes an INPUT channel that lives on the
                   sphere, e.g. att_c_proj, mlp_c_proj)
"""
from __future__ import annotations

from collections.abc import Mapping

import torch


def justnorm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize `x` along `dim`. Preserves dtype (computes in fp32)."""
    dtype = x.dtype
    x32 = x.float()
    out = x32 / x32.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return out.to(dtype=dtype)


@torch.no_grad()
def normalize_module_matrices(role_map: Mapping[torch.nn.Parameter, str]) -> None:
    """Project each parameter onto the unit sphere in-place per its role.

    role="rows" -> per-row unit norm  (normalize along dim=1)
    role="cols" -> per-column unit norm (normalize along dim=0)
    """
    for param, role in role_map.items():
        if role == "rows":
            param.data.copy_(justnorm(param.data, dim=1))
        elif role == "cols":
            param.data.copy_(justnorm(param.data, dim=0))
        else:
            raise ValueError(
                f"normalize_module_matrices: unknown role {role!r}; "
                "expected 'rows' or 'cols'"
            )
```

- [ ] **Step 2.4: Re-run tests, expect PASS**

Run: `pytest tests/unit/test_ngpt_normalize.py -v`
Expected: 6 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/model/ngpt/__init__.py src/model/ngpt/normalize.py tests/unit/test_ngpt_normalize.py
git commit -m "feat(ngpt): add justnorm + normalize_module_matrices primitives"
```

---

## Task 3: Learned-scaling parameter helper (`sqk`, `suv`, `alpha`, `sz`)

**Files:**
- Create: `src/model/ngpt/scaling_params.py`
- Create: `tests/unit/test_ngpt_scaling_params.py`

The reference repeats a four-tuple pattern for every scaling parameter:
```python
self.X_init_value = ...      # the target effective scale
self.X_init_scaling = ...    # the storage init magnitude
self.X = nn.Parameter(self.X_init_scaling * torch.ones(shape, dtype=torch.float32))
# Use: X_effective = self.X * (self.X_init_value / self.X_init_scaling)
```

We extract that into one helper so the four call-sites (`sqk`, `suv`, `attn_alpha`, `mlp_alpha`, plus `sz` on the model) stay consistent.

- [ ] **Step 3.1: Write the failing test**

Create `tests/unit/test_ngpt_scaling_params.py`:
```python
"""CPU tests for the _LearnedScaling helper."""
import torch

from src.model.ngpt.scaling_params import LearnedScaling


def test_storage_matches_init_scaling():
    ls = LearnedScaling(shape=(4,), init_value=1.0, init_scaling=0.1)
    assert ls.param.shape == (4,)
    assert ls.param.dtype == torch.float32
    assert torch.allclose(ls.param.data, 0.1 * torch.ones(4))


def test_scaled_value_matches_init_value_at_init():
    ls = LearnedScaling(shape=(4,), init_value=0.05, init_scaling=1.0 / 8.0)
    expected = (1.0 / 8.0) * torch.ones(4) * (0.05 / (1.0 / 8.0))
    assert torch.allclose(ls.scaled_value(), expected)
    # i.e. uniform 0.05
    assert torch.allclose(ls.scaled_value(), 0.05 * torch.ones(4))


def test_scaled_value_scales_with_param_data():
    ls = LearnedScaling(shape=(3,), init_value=2.0, init_scaling=0.5)
    ls.param.data.copy_(torch.tensor([0.5, 1.0, 1.5]))
    # multiplier is init_value/init_scaling = 4.0
    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(ls.scaled_value(), expected)


def test_is_registered_as_nn_module_with_one_param():
    ls = LearnedScaling(shape=(2,), init_value=1.0, init_scaling=1.0)
    params = list(ls.parameters())
    assert len(params) == 1
    assert params[0] is ls.param
```

- [ ] **Step 3.2: Run the test to verify it fails**

Run: `pytest tests/unit/test_ngpt_scaling_params.py -v`
Expected: `ModuleNotFoundError: src.model.ngpt.scaling_params`.

- [ ] **Step 3.3: Implement `scaling_params.py`**

Create `src/model/ngpt/scaling_params.py`:
```python
"""Learned scaling parameter helper for nGPT.

Captures the (init_value, init_scaling, fp32 storage) four-tuple that
the reference uses for sqk, suv, attn_alpha, mlp_alpha, and sz.
Effective value at runtime is `param * (init_value / init_scaling)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LearnedScaling(nn.Module):
    """Learnable scaling vector with separated init_value vs storage scale."""

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        init_value: float,
        init_scaling: float,
    ) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.init_value = float(init_value)
        self.init_scaling = float(init_scaling)
        self.param = nn.Parameter(
            self.init_scaling * torch.ones(shape, dtype=torch.float32)
        )

    def scaled_value(self) -> torch.Tensor:
        return self.param * (self.init_value / self.init_scaling)
```

- [ ] **Step 3.4: Re-run tests, expect PASS**

Run: `pytest tests/unit/test_ngpt_scaling_params.py -v`
Expected: 4 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/model/ngpt/scaling_params.py tests/unit/test_ngpt_scaling_params.py
git commit -m "feat(ngpt): add LearnedScaling helper for sqk/suv/alpha/sz parameters"
```

---

## Task 4: `NGPTSelfAttention` — Q/K hypersphere normalization + `sqk` scaling

**Files:**
- Create: `src/model/ngpt/attention.py`
- Create: `tests/unit/test_ngpt_attention.py`

The reference [model.py:128-136](file:///lustre/fast/fast/zqiu/tmp/ngpt/model.py#L128-L136) does, *after RoPE and before SDPA*:
```python
sqk = (self.sqk * (sqk_init_value/sqk_init_scaling)).view(1, 1, n_head, head_dim)
q = sqk * justnorm(q)
k = sqk * justnorm(k)
softmax_scale = sqrt(head_dim)   # NB: not 1/sqrt(head_dim)
```

Megatron's `SelfAttention.forward` (in [third_party/Megatron-LM/megatron/core/transformer/attention.py](file:///lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/transformer/attention.py)) already supports per-head Q/K normalization through `q_layernorm` / `k_layernorm` submodules — those are applied to the per-head tensors of shape `(s, b, h_per_tp, d_head)`. We can therefore inject the L2-norm-then-`sqk` step through those submodule slots without overriding `forward`. The softmax-scale override is set via `config.softmax_scale` (Megatron passes it into `DotProductAttention`).

- [ ] **Step 4.1: Write the failing test**

Create `tests/unit/test_ngpt_attention.py`:
```python
"""CPU tests for the per-head Q/K hypersphere normalization injected by nGPT.

The full SelfAttention is hard to spin up off-cluster; we test the
QKHyperNorm leaf module directly. That module is what nGPT plugs into
Megatron's q_layernorm / k_layernorm submodule slots.
"""
import torch

from src.model.ngpt.attention import QKHyperNorm


def test_qk_hyper_norm_output_is_sqk_times_unit():
    # Megatron passes per-head tensors of shape (s, b, h_per_tp, d_head)
    s, b, h, d = 4, 2, 3, 8
    qkn = QKHyperNorm(num_heads_per_tp=h, head_dim=d, sqk_init_value=1.0,
                      base_scale=1.0 / 8.0)
    x = torch.randn(s, b, h, d)
    y = qkn(x)
    # y / sqk_per_head should be unit-norm along d
    sqk_eff = qkn.sqk.scaled_value().view(1, 1, h, d)
    unit = (y / sqk_eff)
    assert torch.allclose(unit.norm(dim=-1), torch.ones(s, b, h), atol=1e-5)


def test_qk_hyper_norm_at_init_just_normalizes():
    # init_value=1.0 with uniform sqk => sqk_eff == 1, so y == justnorm(x)
    s, b, h, d = 2, 1, 2, 4
    qkn = QKHyperNorm(num_heads_per_tp=h, head_dim=d, sqk_init_value=1.0,
                      base_scale=1.0 / 8.0)
    x = torch.randn(s, b, h, d)
    y = qkn(x)
    expected = x / x.norm(p=2, dim=-1, keepdim=True)
    assert torch.allclose(y, expected, atol=1e-5)


def test_qk_hyper_norm_param_count_is_head_dim_times_heads():
    qkn = QKHyperNorm(num_heads_per_tp=4, head_dim=16, sqk_init_value=1.0,
                      base_scale=1.0 / 16.0)
    n = sum(p.numel() for p in qkn.parameters())
    # one sqk vector of length n_heads * head_dim
    assert n == 4 * 16
```

- [ ] **Step 4.2: Run the test to verify it fails**

Run: `pytest tests/unit/test_ngpt_attention.py -v`
Expected: `ModuleNotFoundError: src.model.ngpt.attention`.

- [ ] **Step 4.3: Implement `attention.py`**

Create `src/model/ngpt/attention.py`:
```python
"""nGPT Q/K hypersphere normalization.

Plugged into Megatron's SelfAttentionSubmodules.q_layernorm and
.k_layernorm slots so that `SelfAttention.forward` applies it to the
per-head tensors `(s, b, h_per_tp, d_head)` right after the QKV split
(and after RoPE — Megatron applies q/k_layernorm AFTER position
encoding, which matches the reference's ordering).

Output: sqk * justnorm(x), per-head, per-position.

Softmax scale override is set elsewhere (config.softmax_scale =
sqrt(head_dim)) so the attention payload uses the nGPT scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.ngpt.normalize import justnorm
from src.model.ngpt.scaling_params import LearnedScaling


class QKHyperNorm(nn.Module):
    """L2-normalize per-head Q or K and scale by learnable per-channel sqk."""

    def __init__(
        self,
        num_heads_per_tp: int,
        head_dim: int,
        sqk_init_value: float,
        base_scale: float,
    ) -> None:
        super().__init__()
        self.num_heads_per_tp = int(num_heads_per_tp)
        self.head_dim = int(head_dim)
        self.sqk = LearnedScaling(
            shape=(self.num_heads_per_tp * self.head_dim,),
            init_value=sqk_init_value,
            init_scaling=base_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (s, b, h_per_tp, d_head). Normalize along d_head, then
        # multiply by per-channel sqk reshaped to broadcast over (s, b).
        normed = justnorm(x, dim=-1)
        sqk_eff = self.sqk.scaled_value().view(1, 1, self.num_heads_per_tp, self.head_dim).to(normed.dtype)
        return sqk_eff * normed
```

- [ ] **Step 4.4: Re-run tests, expect PASS**

Run: `pytest tests/unit/test_ngpt_attention.py -v`
Expected: 3 passed.

- [ ] **Step 4.5: Commit**

```bash
git add src/model/ngpt/attention.py tests/unit/test_ngpt_attention.py
git commit -m "feat(ngpt): QKHyperNorm submodule (justnorm + sqk) for Megatron q/k_layernorm slot"
```

---

## Task 5: `NGPTMLP` — `suv` SwiGLU intermediate scaling

**Files:**
- Create: `src/model/ngpt/mlp.py`
- Create: `tests/unit/test_ngpt_mlp.py`

The reference [model.py:158-164](file:///lustre/fast/fast/zqiu/tmp/ngpt/model.py#L158-L164):
```python
uv = self.c_fc(hin)
if use_nGPT == 1:
    suv = self.suv * ((suv_init_value/suv_init_scaling) * (n_embd ** 0.5))
    uv = suv * uv
u, v = uv.chunk(2, dim=-1)
x_mlp = u * silu(v)
h_mlp = self.mlp_c_proj(x_mlp)
```

The injection point is between `c_fc` and the SiLU/chunk. Megatron's `MLP` calls `self.linear_fc1`, then `bias_geglu_impl` or `bias_swiglu_impl` (depends on `--swiglu`), then `self.linear_fc2`. We override `MLP.forward` only when nGPT is enabled. For simplicity (and to keep this CPU-testable), we subclass `MLP` and reimplement the forward with the suv injection. SwiGLU split convention: Megatron packs `[gate, up]` into the first half / second half — confirm by inspecting `third_party/Megatron-LM/megatron/core/transformer/mlp.py::MLP.forward` and `megatron/core/jit.py::bias_swiglu_impl`; the test in Step 5.1 covers this against the reference.

- [ ] **Step 5.1: Write the failing test**

Create `tests/unit/test_ngpt_mlp.py`:
```python
"""CPU test for NGPTMLP forward matches the reference c_fc/suv/silu path."""
import torch
import torch.nn as nn

from src.model.ngpt.mlp import NGPTMLPBody


def test_ngpt_mlp_body_matches_reference_uv_silu():
    """NGPTMLPBody encapsulates the c_fc -> suv -> chunk -> silu -> mlp_c_proj path.

    We compare it to a hand-written reference that mirrors the lines from
    /lustre/fast/fast/zqiu/tmp/ngpt/model.py::Block.forward, MLP section.
    """
    torch.manual_seed(0)
    n_embd = 16
    n_inner = 4 * n_embd  # nGPT convention
    base_scale = 1.0 / (n_embd ** 0.5)
    suv_init_value, suv_init_scaling = 1.0, 1.0  # reference defaults

    body = NGPTMLPBody(
        hidden_size=n_embd,
        ffn_hidden_size=n_inner,
        base_scale=base_scale,
        suv_init_value=suv_init_value,
        suv_init_scaling=suv_init_scaling,
        dtype=torch.float32,
    )

    # Reference c_fc has output dim 2*4*n_embd = 2*n_inner
    # and stores [u | v] concatenation along the last dim.
    ref_c_fc = nn.Linear(n_embd, 2 * n_inner, bias=False, dtype=torch.float32)
    ref_proj = nn.Linear(n_inner, n_embd, bias=False, dtype=torch.float32)
    # tie weights so behaviour is comparable
    ref_c_fc.weight.data.copy_(body.linear_fc1.weight.data)
    ref_proj.weight.data.copy_(body.linear_fc2.weight.data)
    # suv starts at suv_init_scaling everywhere -> scaled_value() == 1
    suv = body.suv.scaled_value() * (n_embd ** 0.5)

    x = torch.randn(2, 5, n_embd)
    uv = ref_c_fc(x)
    uv = suv * uv
    u, v = uv.chunk(2, dim=-1)
    ref_out = ref_proj(u * torch.nn.functional.silu(v))

    out = body(x)
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_ngpt_mlp_body_param_count():
    n_embd = 16
    n_inner = 64
    body = NGPTMLPBody(
        hidden_size=n_embd, ffn_hidden_size=n_inner,
        base_scale=1.0 / (n_embd ** 0.5),
        suv_init_value=1.0, suv_init_scaling=1.0, dtype=torch.float32,
    )
    # linear_fc1: 2 * n_inner * n_embd
    # linear_fc2: n_embd * n_inner
    # suv:        2 * n_inner
    expected = (2 * n_inner * n_embd) + (n_embd * n_inner) + (2 * n_inner)
    assert sum(p.numel() for p in body.parameters()) == expected
```

- [ ] **Step 5.2: Run the test to verify it fails**

Run: `pytest tests/unit/test_ngpt_mlp.py -v`
Expected: `ModuleNotFoundError: src.model.ngpt.mlp`.

- [ ] **Step 5.3: Implement `mlp.py`**

Create `src/model/ngpt/mlp.py`:
```python
"""nGPT MLP body: c_fc -> suv * uv -> chunk -> silu(v) * u -> mlp_c_proj.

NGPTMLPBody is a CPU-runnable pure-PyTorch module that matches the
reference's MLP fragment. It is what NGPTTransformerLayer instantiates
when the layer spec wires `mlp=NGPTMLPBody`. We deliberately do NOT
subclass `megatron.core.transformer.mlp.MLP` here because (a) MLP
defaults to two RowParallel/ColParallel linears that pull in TP plumbing
unhelpful at TP=1, and (b) staying pure-PyTorch keeps the parity test
runnable on CPU.

A future v2 that adds TP>1 support will subclass `MLP` and override
`forward` so the column-parallel `linear_fc1`, the suv scaling, and the
row-parallel `linear_fc2` all stay TP-aware.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.ngpt.scaling_params import LearnedScaling


class NGPTMLPBody(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        base_scale: float,
        suv_init_value: float,
        suv_init_scaling: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.ffn_hidden_size = int(ffn_hidden_size)
        self._n_embd_sqrt = float(self.hidden_size) ** 0.5

        # nGPT reference packs c_fc with 2*ffn_hidden_size columns:
        # [u_half | v_half]. Same convention here.
        self.linear_fc1 = nn.Linear(
            self.hidden_size, 2 * self.ffn_hidden_size, bias=False, dtype=dtype
        )
        self.linear_fc2 = nn.Linear(
            self.ffn_hidden_size, self.hidden_size, bias=False, dtype=dtype
        )
        # init: row-normalized with std=base_scale
        nn.init.normal_(self.linear_fc1.weight, mean=0.0, std=base_scale)
        nn.init.normal_(self.linear_fc2.weight, mean=0.0, std=base_scale)

        self.suv = LearnedScaling(
            shape=(2 * self.ffn_hidden_size,),
            init_value=suv_init_value,
            init_scaling=suv_init_scaling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        uv = self.linear_fc1(x)
        # Reference effective suv: param * (init_value/init_scaling) * sqrt(n_embd)
        suv = (self.suv.scaled_value() * self._n_embd_sqrt).to(uv.dtype)
        uv = suv * uv
        u, v = uv.chunk(2, dim=-1)
        return self.linear_fc2(u * F.silu(v))
```

- [ ] **Step 5.4: Re-run tests, expect PASS**

Run: `pytest tests/unit/test_ngpt_mlp.py -v`
Expected: 2 passed.

- [ ] **Step 5.5: Commit**

```bash
git add src/model/ngpt/mlp.py tests/unit/test_ngpt_mlp.py
git commit -m "feat(ngpt): NGPTMLPBody — c_fc + suv-scaled SwiGLU + mlp_c_proj"
```

---

## Task 6: `NGPTTransformerLayer` — hypersphere residual blend

**Files:**
- Create: `src/model/ngpt/layer.py`
- Create: `tests/unit/test_ngpt_layer_block_forward.py`

Reference [model.py:144-178](file:///lustre/fast/fast/zqiu/tmp/ngpt/model.py#L144-L178):
```python
# attn branch
lr = abs(attn_alpha * (attn_alpha_init_value / attn_alpha_init_scaling))
h = justnorm(justnorm(h) + lr * (justnorm(h_att) - justnorm(h)))
# mlp branch (same structure)
lr = abs(mlp_alpha * (mlp_alpha_init_value / mlp_alpha_init_scaling))
h = justnorm(justnorm(h) + lr * (justnorm(h_mlp) - justnorm(h)))
```

Because Megatron's `TransformerLayer.forward` is tightly coupled to its (LayerNorm → attn → BDA → LayerNorm → MLP → BDA) sequence, we **override `forward` wholesale** in `NGPTTransformerLayer`, calling the same `self.self_attention(...)` and `self.mlp(...)` submodules but applying our own residual update.

For the v1 dense/TP=1 path we keep this simple: subclass `TransformerLayer`, ignore `input_layernorm` and `pre_mlp_layernorm` (they will be wired to `IdentityOp` by the spec builder), and call `self.self_attention` / `self.mlp` directly with the hypersphere residual blend in between. The `self_attn_bda` and `mlp_bda` slots are also `IdentityOp` so they are never invoked.

- [ ] **Step 6.1: Write the failing test**

The full `TransformerLayer` requires a Megatron `TransformerConfig` and several context arguments; spinning that up CPU-side is heavy. Instead we test the **standalone NGPTBlock** (the pure-PyTorch sibling that the parity test in Task 11 will use). `NGPTTransformerLayer` will share the same `_residual_blend` static helper as `NGPTBlock` so the residual math has *one* implementation, covered here.

Create `tests/unit/test_ngpt_layer_block_forward.py`:
```python
"""Pure-PyTorch NGPTBlock parity vs the vendored reference Block (use_nGPT=1)."""
import torch

from src.model.ngpt.layer import NGPTBlock
from tests._fixtures.ngpt_reference.model import Block as RefBlock, GPTConfig


def _ref_config(n_embd=64, n_head=4, vocab_size=100):
    return GPTConfig(
        block_size=32, vocab_size=vocab_size,
        n_layer=2, n_head=n_head, n_embd=n_embd,
        base_scale=1.0 / (n_embd ** 0.5),
        use_nGPT=1, dropout=0.0, bias=False,
    )


def test_ngpt_block_matches_reference_at_init():
    torch.manual_seed(123)
    cfg = _ref_config()
    ref = RefBlock(cfg, iblock=0).float()
    ours = NGPTBlock(
        hidden_size=cfg.n_embd, num_heads=cfg.n_head,
        ffn_hidden_size=4 * cfg.n_embd,
        base_scale=cfg.base_scale, dtype=torch.float32,
    )
    # Copy reference weights into ours (same shapes / convention).
    with torch.no_grad():
        ours.query.weight.copy_(ref.query.weight)
        ours.key.weight.copy_(ref.key.weight)
        ours.value.weight.copy_(ref.value.weight)
        ours.att_c_proj.weight.copy_(ref.att_c_proj.weight)
        ours.c_fc.weight.copy_(ref.c_fc.weight)
        ours.mlp_c_proj.weight.copy_(ref.mlp_c_proj.weight)
        ours.sqk.param.copy_(ref.sqk)
        ours.suv.param.copy_(ref.suv)
        ours.attn_alpha.param.copy_(ref.attn_alpha)
        ours.mlp_alpha.param.copy_(ref.mlp_alpha)

    x = torch.randn(1, 8, cfg.n_embd)  # (B, T, C) matches reference
    ours.eval(); ref.eval()
    with torch.no_grad():
        y_ours = ours(x)
        y_ref = ref(x.to(torch.bfloat16)).float()
    # Both blocks now do attention in bf16 internally (see NGPTBlock._attn),
    # so the only remaining gap is upstream SDPA vs flash_attn rounding.
    assert torch.allclose(y_ours, y_ref, atol=2e-3, rtol=2e-3), (
        f"max abs diff = {(y_ours - y_ref).abs().max().item()}"
    )


def test_ngpt_block_residual_is_unit_norm_per_token():
    """After the second hypersphere blend the residual lies on S^{C-1}."""
    cfg = _ref_config(n_embd=32, n_head=4)
    blk = NGPTBlock(
        hidden_size=cfg.n_embd, num_heads=cfg.n_head,
        ffn_hidden_size=4 * cfg.n_embd,
        base_scale=cfg.base_scale, dtype=torch.float32,
    )
    x = torch.randn(2, 4, cfg.n_embd)
    y = blk(x)
    norms = y.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
```

- [ ] **Step 6.2: Run the test to verify it fails**

Run: `pytest tests/unit/test_ngpt_layer_block_forward.py -v`
Expected: `ModuleNotFoundError: src.model.ngpt.layer`.

- [ ] **Step 6.3: Implement `layer.py`**

Create `src/model/ngpt/layer.py`:
```python
"""nGPT transformer block.

This module ships two things:

* `NGPTBlock` — a pure-PyTorch transformer block that mirrors the
  reference's `Block` (with the same attention + MLP + residual-blend
  semantics, including the reference's *internal* bf16 cast for
  attention so parity tests aren't fighting a precision delta). It
  exists so the parity test can run CPU-side without a Megatron model.

* `NGPTTransformerLayer` — subclass of Megatron's `TransformerLayer`
  that overrides `forward` to apply nGPT's residual blend. It expects
  the surrounding spec to wire `input_layernorm` and `pre_mlp_layernorm`
  to `IdentityOp` (no pre-norm in nGPT) and `self_attn_bda` /
  `mlp_bda` to `IdentityFuncOp` (we do the residual ourselves; these
  slots are built but never invoked because we override `forward`).
  Learned scaling parameters `attn_alpha` and `mlp_alpha` are built in
  `__init__` — building them in `forward` would mean they don't exist
  when Megatron's optimizer is constructed (which walks
  `model.named_parameters()` before the first forward).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from megatron.core.transformer.transformer_layer import TransformerLayer

from src.model.ngpt.attention import QKHyperNorm  # noqa: F401  (used via spec)
from src.model.ngpt.mlp import NGPTMLPBody  # noqa: F401  (used via spec)
from src.model.ngpt.normalize import justnorm
from src.model.ngpt.scaling_params import LearnedScaling


def _residual_blend(h: torch.Tensor, h_branch: torch.Tensor,
                    alpha: LearnedScaling) -> torch.Tensor:
    """Hypersphere residual: h <- justnorm(justnorm(h) + |alpha| * (justnorm(h_branch) - justnorm(h)))."""
    lr = torch.abs(alpha.scaled_value()).to(h.dtype)
    a = justnorm(h)
    b = justnorm(h_branch)
    return justnorm(a + lr * (b - a))


def _apply_rope(sinusoidal_pos: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
    """Re-implementation of the reference's apply_rotary_position_embeddings."""
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1] // 2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1] // 2, 2)) * torch.stack((cos, sin), dim=-1)
    return q_rot.reshape(q.shape), k_rot.reshape(k.shape)


def _sinusoidal_embeddings(n_positions: int, dim: int) -> torch.Tensor:
    import math
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    emb = torch.zeros((n_positions, dim))
    emb[:, 0::2] = torch.sin(position * div_term)
    emb[:, 1::2] = torch.cos(position * div_term)
    return emb


class NGPTBlock(nn.Module):
    """CPU-runnable parity-oracle block. Mirrors reference Block(use_nGPT=1).

    Attention is computed in bf16 inside this method to match the
    reference's hardcoded `q.to(bfloat16)` / `k.to(bfloat16)` /
    `v.to(bfloat16)` casts inside `Block.forward` (model.py:136).
    Without that, the parity test would have to swallow a sustained
    precision gap that has nothing to do with whether nGPT is wired
    correctly.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        base_scale: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Same Linear shapes as reference.
        self.query = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.att_c_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.c_fc = nn.Linear(hidden_size, 2 * ffn_hidden_size, bias=False, dtype=dtype)
        self.mlp_c_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False, dtype=dtype)

        # Scaling params (init matches reference defaults).
        self.sqk = LearnedScaling((hidden_size,), init_value=1.0, init_scaling=base_scale)
        self.suv = LearnedScaling((2 * ffn_hidden_size,), init_value=1.0, init_scaling=1.0)
        self.attn_alpha = LearnedScaling((hidden_size,), init_value=0.05, init_scaling=base_scale)
        self.mlp_alpha = LearnedScaling((hidden_size,), init_value=0.05, init_scaling=base_scale)

        self._ffn_hidden_size = ffn_hidden_size
        self._n_embd_sqrt = float(hidden_size) ** 0.5

    def _attn(self, h: torch.Tensor) -> torch.Tensor:
        B, T, C = h.size()
        q = self.query(h).view(B, T, self.num_heads, self.head_dim)
        k = self.key(h).view(B, T, self.num_heads, self.head_dim)
        v = self.value(h).view(B, T, self.num_heads, self.head_dim)

        sinusoidal_pos = _sinusoidal_embeddings(T, self.head_dim).to(q.device)
        q, k = _apply_rope(sinusoidal_pos, q.transpose(1, 2), k.transpose(1, 2))
        q, k = q.transpose(2, 1), k.transpose(2, 1)

        sqk = self.sqk.scaled_value().view(1, 1, self.num_heads, self.head_dim).to(q.dtype)
        q = sqk * justnorm(q)
        k = sqk * justnorm(k)

        softmax_scale = self._n_embd_sqrt / (self.num_heads ** 0.5)  # = sqrt(head_dim)
        # Reference (model.py:136) explicitly casts q/k/v to bf16 inside
        # attention, regardless of outer dtype. Mirror that so parity holds.
        q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out_dtype = q.dtype
        q_bf, k_bf, v_bf = (t.to(torch.bfloat16) for t in (q_, k_, v_))
        attn_bf = torch.nn.functional.scaled_dot_product_attention(
            q_bf, k_bf, v_bf, dropout_p=0.0, is_causal=True, scale=softmax_scale,
        )
        attn = attn_bf.to(out_dtype).transpose(1, 2).contiguous().view(B, T, C)
        return self.att_c_proj(attn)

    def _mlp(self, h: torch.Tensor) -> torch.Tensor:
        uv = self.c_fc(h)
        suv = (self.suv.scaled_value() * self._n_embd_sqrt).to(uv.dtype)
        uv = suv * uv
        u, v = uv.chunk(2, dim=-1)
        return self.mlp_c_proj(u * torch.nn.functional.silu(v))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_att = self._attn(h)
        h = _residual_blend(h, h_att, self.attn_alpha)
        h_mlp = self._mlp(h)
        h = _residual_blend(h, h_mlp, self.mlp_alpha)
        return h


# ---------------------------------------------------------------------------
# Megatron-integrated layer (T=1, dense). Used when the spec wires this in.
# ---------------------------------------------------------------------------

class NGPTTransformerLayer(TransformerLayer):
    """nGPT layer for Megatron. Overrides `forward` to apply hypersphere blend.

    The companion spec builder (`src/specs/ngpt_layer_spec.py`) wires:

      input_layernorm    = IdentityOp
      pre_mlp_layernorm  = IdentityOp
      self_attn_bda      = IdentityFuncOp     # built but never called
      mlp_bda            = IdentityFuncOp     # built but never called
      self_attention.q_layernorm/k_layernorm = QKHyperNorm
      mlp.module         = NGPTMLPBody

    `attn_alpha` and `mlp_alpha` are constructed in `__init__` so they
    are present in `model.named_parameters()` *before* Megatron's
    optimizer is built. Building them lazily in `forward` would leave
    them out of the optimizer entirely — they'd never receive gradients
    and the run would silently train without them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v1 scope assertion — `forward` skips Megatron's recompute path.
        assert getattr(self.config, "recompute_granularity", None) is None, (
            "nGPT v1 does not support --recompute-granularity; override "
            "expects a single-pass forward."
        )

        hidden = int(self.config.hidden_size)
        # These fields are stamped onto the config by `ngpt_apply_spec`'s
        # wrap of `core_transformer_config_from_args`. Falling back to
        # defaults makes layer-only unit-testing easier.
        base_scale = float(getattr(self.config, "ngpt_base_scale", 1.0 / (hidden ** 0.5)))
        alpha_init = float(getattr(self.config, "ngpt_alpha_init", 0.05))
        self.attn_alpha = LearnedScaling((hidden,), init_value=alpha_init, init_scaling=base_scale)
        self.mlp_alpha = LearnedScaling((hidden,), init_value=alpha_init, init_scaling=base_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        inference_params=None,
    ):
        # ---- Attention branch ----
        attn_out_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        # attn_out_with_bias is (output, bias). We use only output; bias is None
        # under --disable-bias-linear which nGPT requires.
        attn_out = attn_out_with_bias[0]
        hidden_states = _residual_blend(hidden_states, attn_out, self.attn_alpha)

        # ---- MLP branch ----
        mlp_out_with_bias = self.mlp(hidden_states)
        mlp_out = mlp_out_with_bias[0] if isinstance(mlp_out_with_bias, tuple) else mlp_out_with_bias
        hidden_states = _residual_blend(hidden_states, mlp_out, self.mlp_alpha)

        # Megatron's layer returns (hidden_states, context). nGPT has no context.
        return hidden_states, context
```

- [ ] **Step 6.4: Re-run tests, expect PASS**

Run: `pytest tests/unit/test_ngpt_layer_block_forward.py -v`
Expected: 2 passed. (`test_ngpt_block_matches_reference_at_init` validates `_residual_blend`, the rotary recipe, and the attn / MLP fragments together against the vendored oracle; `test_ngpt_block_residual_is_unit_norm_per_token` validates that the post-blend output lies on the unit sphere.)

- [ ] **Step 6.5: Commit**

```bash
git add src/model/ngpt/layer.py tests/unit/test_ngpt_layer_block_forward.py
git commit -m "feat(ngpt): NGPTBlock parity oracle + NGPTTransformerLayer Megatron subclass"
```

---

## Task 7: Output `sz` scaling helper

**Files:**
- Create: `src/model/ngpt/output_scaling.py`
- Create: `tests/unit/test_ngpt_output_scaling.py`

Reference: `sz` is a learnable per-vocab vector multiplied into logits ([model.py:283-292](file:///lustre/fast/fast/zqiu/tmp/ngpt/model.py#L283-L292)). We attach it to the model post-build via the `ngpt_apply_spec` patch (Task 9), and wrap `model.output_layer.forward` to multiply.

- [ ] **Step 7.1: Write the failing test**

Create `tests/unit/test_ngpt_output_scaling.py`:
```python
"""CPU tests for the post-build `sz` logit scaling wrapper."""
import torch
import torch.nn as nn

from src.model.ngpt.output_scaling import attach_sz_scaling


class _FakeOutput(nn.Module):
    """Stand-in for Megatron's ColumnParallelLinear output_layer.

    Returns (logits, bias) like the real one does.
    """
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden))

    def forward(self, x):
        return torch.matmul(x, self.weight.T), None


def test_attach_sz_scaling_creates_parameter_with_correct_shape():
    out = _FakeOutput(vocab=17, hidden=4)
    holder = nn.Module()
    holder.output_layer = out
    attach_sz_scaling(holder, vocab_size=17, base_scale=0.5)
    assert hasattr(holder, "_ngpt_sz")
    assert holder._ngpt_sz.param.shape == (17,)
    assert torch.allclose(holder._ngpt_sz.param.data, 0.5 * torch.ones(17))


def test_attach_sz_scaling_multiplies_logits():
    out = _FakeOutput(vocab=5, hidden=3)
    holder = nn.Module()
    holder.output_layer = out
    attach_sz_scaling(holder, vocab_size=5, base_scale=1.0)
    # at init, sz_effective = 1.0 everywhere, so logits unchanged
    x = torch.randn(2, 7, 3)
    logits_init, _ = holder.output_layer(x)
    expected_unscaled = torch.matmul(x, out.weight.T)
    assert torch.allclose(logits_init, expected_unscaled, atol=1e-5)

    # bump sz, re-eval
    holder._ngpt_sz.param.data.fill_(3.0)
    logits_scaled, _ = holder.output_layer(x)
    assert torch.allclose(logits_scaled, 3.0 * expected_unscaled, atol=1e-5)


def test_attach_sz_scaling_is_idempotent():
    out = _FakeOutput(vocab=3, hidden=2)
    holder = nn.Module()
    holder.output_layer = out
    attach_sz_scaling(holder, vocab_size=3, base_scale=1.0)
    first = holder._ngpt_sz
    attach_sz_scaling(holder, vocab_size=3, base_scale=1.0)
    assert holder._ngpt_sz is first
```

- [ ] **Step 7.2: Run the test to verify it fails**

Run: `pytest tests/unit/test_ngpt_output_scaling.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 7.3: Implement `output_scaling.py`**

Create `src/model/ngpt/output_scaling.py`:
```python
"""Attach an nGPT `sz` post-multiplier to a model's output_layer.

The reference (model.py:283-292) does, when use_nGPT=1:

    sz_effective = self.sz * (sz_init_value/sz_init_scaling)
    logits = sz_effective * logits

We replicate that here by monkey-patching the `forward` of the given
holder's `output_layer` attribute (in Megatron that is the GPTModel's
ColumnParallelLinear). The wrapper preserves the upstream return
convention `(logits, bias)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.ngpt.scaling_params import LearnedScaling


def attach_sz_scaling(model: nn.Module, vocab_size: int, base_scale: float) -> None:
    if getattr(model, "_ngpt_sz", None) is not None:
        return  # idempotent
    sz = LearnedScaling(
        shape=(int(vocab_size),), init_value=1.0, init_scaling=float(base_scale),
    )
    sz.to(next(model.parameters()).device)
    # Register as a submodule so checkpoint save/load picks it up.
    model.add_module("_ngpt_sz", sz)

    orig_forward = model.output_layer.forward

    def _wrapped(input_, *args, **kwargs):
        out = orig_forward(input_, *args, **kwargs)
        sz_eff = model._ngpt_sz.scaled_value()
        if isinstance(out, tuple):
            logits, bias = out
            return sz_eff.to(logits.dtype) * logits, bias
        return sz_eff.to(out.dtype) * out

    model.output_layer.forward = _wrapped  # type: ignore[assignment]
```

- [ ] **Step 7.4: Re-run tests, expect PASS**

Run: `pytest tests/unit/test_ngpt_output_scaling.py -v`
Expected: 3 passed.

- [ ] **Step 7.5: Commit**

```bash
git add src/model/ngpt/output_scaling.py tests/unit/test_ngpt_output_scaling.py
git commit -m "feat(ngpt): attach_sz_scaling — post-build sz parameter + output_layer wrapper"
```

---

## Task 8: Spec builder (`build_ngpt_layer_spec`)

**Files:**
- Create: `src/specs/ngpt_layer_spec.py`
- Create: `tests/unit/test_ngpt_layer_spec.py`

Mirrors `get_gpt_layer_local_submodules` but with nGPT-specific wiring.

- [ ] **Step 8.1: Write the failing test**

Create `tests/unit/test_ngpt_layer_spec.py`:
```python
"""CPU tests for the nGPT Megatron spec builder."""
import pytest


def test_build_ngpt_layer_spec_returns_module_spec():
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.identity_op import IdentityOp

    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec
    from src.model.ngpt.layer import NGPTTransformerLayer

    class _Cfg:
        hidden_size = 64
        num_attention_heads = 4
        ffn_hidden_size = 256
        num_query_groups = 4
        ngpt_base_scale = 1.0 / 8.0
        ngpt_sqk_init = 1.0
        ngpt_suv_init = 1.0

    spec = build_ngpt_layer_spec(_Cfg())
    assert isinstance(spec, ModuleSpec)
    assert spec.module is NGPTTransformerLayer
    sub = spec.submodules
    assert sub.input_layernorm is IdentityOp
    assert sub.pre_mlp_layernorm is IdentityOp
    # self_attn_bda / mlp_bda must be no-op-equivalent (IdentityFuncOp).
    from megatron.core.transformer.identity_op import IdentityFuncOp
    assert sub.self_attn_bda is IdentityFuncOp
    assert sub.mlp_bda is IdentityFuncOp


def test_build_ngpt_layer_spec_asserts_tp1():
    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec

    class _CfgTp2:
        hidden_size = 64
        num_attention_heads = 4
        ffn_hidden_size = 256
        num_query_groups = 4
        ngpt_base_scale = 1.0 / 8.0
        ngpt_sqk_init = 1.0
        ngpt_suv_init = 1.0
        tensor_model_parallel_size = 2

    with pytest.raises(AssertionError, match="TP"):
        build_ngpt_layer_spec(_CfgTp2())


def test_build_ngpt_layer_spec_asserts_no_moe():
    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec

    class _CfgMoE:
        hidden_size = 64
        num_attention_heads = 4
        ffn_hidden_size = 256
        num_query_groups = 4
        ngpt_base_scale = 1.0 / 8.0
        ngpt_sqk_init = 1.0
        ngpt_suv_init = 1.0
        num_moe_experts = 4

    with pytest.raises(AssertionError, match="MoE"):
        build_ngpt_layer_spec(_CfgMoE())
```

- [ ] **Step 8.2: Run the test to verify it fails**

Run: `pytest tests/unit/test_ngpt_layer_spec.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 8.3: Implement `ngpt_layer_spec.py`**

Create `src/specs/ngpt_layer_spec.py`:
```python
"""Build the Megatron ModuleSpec for an nGPT transformer layer.

v1 constraints (see plan §Prerequisites): TP=1, PP=1, dense (no MoE,
no MLA). These are checked at spec-build time so a misconfigured
experiment fails fast at submit instead of partway into a job.

The softmax-scale override (nGPT uses sqrt(head_dim), not 1/sqrt
(head_dim)) is *not* handled here. It is stamped onto
`TransformerConfig.softmax_scale` by the `ngpt_apply_spec` patch's
wrap of `core_transformer_config_from_args`; from there Megatron's
`SelfAttention.__init__` forwards it into `DotProductAttention`
(attention.py:324, dot_product_attention.py:84). Keeping the override
in the patch means the unit-tested spec builder stays config-agnostic.
"""
from __future__ import annotations

from functools import partial

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (  # noqa: F401  (kept for parity with upstream local-spec layout)
    TENorm,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from src.model.ngpt.attention import QKHyperNorm
from src.model.ngpt.layer import NGPTTransformerLayer
from src.model.ngpt.mlp import NGPTMLPBody


def _qk_hyper_norm_builder(num_heads: int, head_dim: int, sqk_init: float,
                           base_scale: float):
    def _build(hidden_size, eps=None, **_kwargs):
        # Megatron passes hidden_size==head_dim when constructing q/k_layernorm
        # (it does it from the per-head slice). We don't use the eps param.
        return QKHyperNorm(
            num_heads_per_tp=num_heads, head_dim=head_dim,
            sqk_init_value=sqk_init, base_scale=base_scale,
        )
    return _build


def _ngpt_mlp_module_builder(hidden_size: int, ffn_hidden_size: int,
                             base_scale: float, suv_init: float,
                             dtype):
    def _build(config=None, **_kwargs):
        return NGPTMLPBody(
            hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size,
            base_scale=base_scale, suv_init_value=suv_init,
            suv_init_scaling=1.0, dtype=dtype,
        )
    return _build


def build_ngpt_layer_spec(config) -> ModuleSpec:
    tp = getattr(config, "tensor_model_parallel_size", 1)
    assert tp == 1, (
        f"nGPT v1 requires TP=1, got tensor_model_parallel_size={tp}. "
        "TP>1 is a v2 follow-up (sqk/suv sharding)."
    )
    assert getattr(config, "num_moe_experts", None) in (None, 0), (
        "nGPT v1 does not support MoE."
    )
    assert not getattr(config, "multi_latent_attention", False), (
        "nGPT v1 does not support MLA."
    )

    num_heads = int(config.num_attention_heads)
    head_dim = int(config.hidden_size) // num_heads
    base_scale = float(getattr(config, "ngpt_base_scale", 1.0 / (config.hidden_size ** 0.5)))
    sqk_init = float(getattr(config, "ngpt_sqk_init", 1.0))
    suv_init = float(getattr(config, "ngpt_suv_init", 1.0))

    import torch
    param_dtype = torch.bfloat16 if getattr(config, "bf16", True) else torch.float32

    submodules = TransformerLayerSubmodules(
        input_layernorm=IdentityOp,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=_qk_hyper_norm_builder(num_heads, head_dim, sqk_init, base_scale),
                k_layernorm=_qk_hyper_norm_builder(num_heads, head_dim, sqk_init, base_scale),
            ),
        ),
        self_attn_bda=IdentityFuncOp,
        pre_mlp_layernorm=IdentityOp,
        mlp=ModuleSpec(
            module=_ngpt_mlp_module_builder(
                hidden_size=int(config.hidden_size),
                ffn_hidden_size=int(config.ffn_hidden_size),
                base_scale=base_scale,
                suv_init=suv_init,
                dtype=param_dtype,
            ),
        ),
        mlp_bda=IdentityFuncOp,
    )
    return ModuleSpec(module=NGPTTransformerLayer, submodules=submodules)
```

- [ ] **Step 8.4: Re-run tests, expect PASS**

Run: `pytest tests/unit/test_ngpt_layer_spec.py -v`
Expected: 3 passed.

- [ ] **Step 8.5: Commit**

```bash
git add src/specs/ngpt_layer_spec.py tests/unit/test_ngpt_layer_spec.py
git commit -m "feat(ngpt): spec builder asserts TP=1/no-MoE/no-MLA and wires nGPT submodules"
```

---

## Task 9: Patch — `ngpt_apply_spec` (model-build injection)

**Files:**
- Create: `src/patches/ngpt_apply_spec.py`
- Create: `tests/unit/test_ngpt_patch_registry.py`

Mirrors `poet_apply_to_model` in structure: wrap the function that returns the model, swap the spec, attach `sz`, register weight-norm role map for the post-step patch.

- [ ] **Step 9.1: Write the failing test (registry side only)**

Create `tests/unit/test_ngpt_patch_registry.py`:
```python
"""nGPT patches: registration + hash determinism + conflict-freedom."""
import importlib

import pytest


def _reload_patches(names):
    from src.patches._registry import _reset_for_tests
    _reset_for_tests()
    for n in names:
        importlib.import_module(f"src.patches.{n}")


def test_ngpt_patches_register_without_conflict():
    _reload_patches(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    from src.patches._registry import registered_patches
    reg = registered_patches()
    assert "ngpt_apply_spec" in reg
    assert "ngpt_normalize_step" in reg
    assert "ngpt_optimizer_setup" in reg


def test_ngpt_patch_set_hash_is_deterministic():
    from src.patches._registry import patch_set_hash
    _reload_patches(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    h1 = patch_set_hash(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    _reload_patches(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    h2 = patch_set_hash(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    assert h1 == h2 and len(h1) == 16
```

- [ ] **Step 9.2: Run; expect failure (patches don't exist yet)**

Run: `pytest tests/unit/test_ngpt_patch_registry.py -v`
Expected: `ModuleNotFoundError: No module named 'src.patches.ngpt_apply_spec'`.

- [ ] **Step 9.3: Implement `ngpt_apply_spec.py`**

Create `src/patches/ngpt_apply_spec.py`:
```python
"""Patch: swap GPT layer spec to nGPT, stamp config fields, attach sz.

Targets (all triggered only when args.ngpt is set):

- `megatron.training.arguments.core_transformer_config_from_args` —
  wrap to copy nGPT-specific args onto the returned TransformerConfig
  (`softmax_scale`, `ngpt_base_scale`, `ngpt_alpha_init`,
  `ngpt_sqk_init`, `ngpt_suv_init`). The softmax_scale stamp is what
  flips Megatron's `DotProductAttention` from `1/sqrt(head_dim)` to
  `sqrt(head_dim)` (verified at dot_product_attention.py:84 and
  attention.py:324).

- `gpt_builders.gpt_builder` — wrap so `_get_transformer_layer_spec`
  returns our nGPT spec. After model construction: attach `sz`,
  register the weight-normalization role map, and run the one-shot
  initial L2 projection that the reference does at train.py:411.

Upstream SHA: see docs/megatron_pin.md.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from src.patches._registry import register_patch

_TARGET = (
    "gpt_builders.gpt_builder",
    "megatron.training.arguments.core_transformer_config_from_args",
)
logger = logging.getLogger(__name__)


@register_patch(name="ngpt_apply_spec", targets=_TARGET)
def apply() -> None:
    # ---- Wrap config builder ----
    from megatron.training import arguments as _ma

    _orig_cfg = _ma.core_transformer_config_from_args

    def _wrapped_cfg(args, *a, **kw):
        config = _orig_cfg(args, *a, **kw)
        if not getattr(args, "ngpt", False):
            return config
        # softmax_scale: nGPT uses sqrt(head_dim) instead of 1/sqrt(head_dim).
        head_dim = int(args.hidden_size) // int(args.num_attention_heads)
        config.softmax_scale = math.sqrt(head_dim)
        # ngpt_* fields read by NGPTTransformerLayer.__init__ and the spec.
        hidden = int(args.hidden_size)
        config.ngpt_base_scale = float(
            getattr(args, "ngpt_base_scale", None) or (1.0 / math.sqrt(hidden))
        )
        config.ngpt_alpha_init = float(getattr(args, "ngpt_alpha_init", 0.05))
        config.ngpt_sqk_init = float(getattr(args, "ngpt_sqk_init", 1.0))
        config.ngpt_suv_init = float(getattr(args, "ngpt_suv_init", 1.0))
        config.ngpt = True  # boolean shortcut for downstream checks
        return config

    _ma.core_transformer_config_from_args = _wrapped_cfg

    # ---- Wrap GPT model builder ----
    import gpt_builders as _gb  # third_party/Megatron-LM is on sys.path

    from src.model.ngpt.output_scaling import attach_sz_scaling
    from src.specs.ngpt_layer_spec import build_ngpt_layer_spec

    _orig_builder = _gb.gpt_builder

    def _wrapped_builder(args, *a, **kw):
        if not getattr(args, "ngpt", False):
            return _orig_builder(args, *a, **kw)
        from megatron.core.transformer.transformer_config import TransformerConfig
        original_get_spec = _gb._get_transformer_layer_spec

        def _ngpt_get_spec(use_te: bool, config: TransformerConfig):
            return build_ngpt_layer_spec(config)

        _gb._get_transformer_layer_spec = _ngpt_get_spec
        try:
            model = _orig_builder(args, *a, **kw)
        finally:
            _gb._get_transformer_layer_spec = original_get_spec

        # Post-build hooks: sz scaling + weight-norm role registration.
        chunks = model if isinstance(model, list) else [model]
        for m in chunks:
            attach_sz_scaling(
                m,
                vocab_size=args.padded_vocab_size,
                base_scale=float(getattr(args, "ngpt_base_scale", None) or (1.0 / math.sqrt(args.hidden_size))),
            )
            _register_ngpt_norm_roles(m, expected_layers=int(args.num_layers))
            _normalize_now(m)  # one-shot init normalize (reference train.py:411)
        logger.info("[nGPT] applied spec + attached sz + registered weight-norm roles")
        return model

    _gb.gpt_builder = _wrapped_builder


# ---------------------------------------------------------------------------
# Weight-normalization role map
# ---------------------------------------------------------------------------
#
# `_NORM_ROLES` maps the *trailing* qualified-name segments (after the
# last dot) of every nGPT weight matrix to its normalization role.
# Trailing-segment matching is intentional: the recent CanonicalOFT
# substring-match bug (`"v_proj" in "qkv_proj.oft_R"` → True) showed
# how raw substring matches silently drop or double-count updates.

_NORM_ROLES_BY_SUFFIX: dict[tuple[str, ...], str] = {
    # Embedding row = per-token vector → unit norm along hidden.
    ("embedding", "word_embeddings", "weight"): "rows",
    # LM head row = per-vocab vector → unit norm along hidden.
    ("output_layer", "weight"): "rows",
    # Q/K/V projection rows = per-output-channel vectors.
    ("linear_qkv", "weight"): "rows",
    # Attention output projection columns = per-input-channel vectors.
    ("linear_proj", "weight"): "cols",
    # SwiGLU c_fc rows = per-output-channel vectors (gate+up concat).
    ("linear_fc1", "weight"): "rows",
    # SwiGLU mlp_c_proj columns = per-input-channel vectors.
    ("linear_fc2", "weight"): "cols",
}


def _match_role(name: str) -> str | None:
    parts = name.split(".")
    for suffix, role in _NORM_ROLES_BY_SUFFIX.items():
        if len(parts) >= len(suffix) and tuple(parts[-len(suffix):]) == suffix:
            return role
    return None


def _register_ngpt_norm_roles(model, expected_layers: int) -> None:
    """Build a {param -> 'rows'|'cols'} dict on `model._ngpt_norm_role_map`.

    Tied embeddings make `output_layer.weight` and
    `embedding.word_embeddings.weight` alias the same parameter; both
    roles agree ("rows") so the dict-overwrite is benign.
    """
    role_map: dict[Any, str] = {}
    matched_per_role = {"rows": 0, "cols": 0}
    for name, param in model.named_parameters():
        role = _match_role(name)
        if role is None:
            continue
        role_map[param] = role
        matched_per_role[role] += 1

    # Sanity check: detect future Megatron renames or missed weight matrices.
    # Per-layer matrices: linear_qkv (rows), linear_proj (cols),
    # linear_fc1 (rows), linear_fc2 (cols) = 4 per layer.
    # Plus embedding (rows) and output_layer (rows) — counted once
    # even under tying because Megatron's `named_parameters` deduplicates.
    n_unique_params = len(role_map)
    per_layer = 4
    embedding_plus_head_min = 1  # at least the embedding; output_layer aliases it under tying
    expected_min = expected_layers * per_layer + embedding_plus_head_min
    assert n_unique_params >= expected_min, (
        f"nGPT weight-norm role map matched only {n_unique_params} params; "
        f"expected >= {expected_min} (got rows={matched_per_role['rows']} "
        f"cols={matched_per_role['cols']}). A param-name regression in "
        "Megatron would slip through silently — fix _NORM_ROLES_BY_SUFFIX."
    )
    model._ngpt_norm_role_map = role_map


def _normalize_now(model) -> None:
    from src.model.ngpt.normalize import normalize_module_matrices
    if hasattr(model, "_ngpt_norm_role_map"):
        normalize_module_matrices(model._ngpt_norm_role_map)
```

- [ ] **Step 9.4: Re-run tests (registry-side check works without Megatron import)**

The decorator runs at import time but `import gpt_builders` and Megatron-internal imports happen inside `apply()`. The registry test only triggers the decorator, so it should pass without a working Megatron env, *provided* `apply()` is not invoked. Verify:

Run: `pytest tests/unit/test_ngpt_patch_registry.py::test_ngpt_patches_register_without_conflict -v`

This will fail until the other two patch files exist; that's expected. Move on.

- [ ] **Step 9.5: Commit**

```bash
git add src/patches/ngpt_apply_spec.py
git commit -m "feat(ngpt): patch — swap spec, attach sz, register weight-norm roles in gpt_builder"
```

---

## Task 10: Patch — `ngpt_normalize_step` (post-step weight L2-projection)

**Files:**
- Create: `src/patches/ngpt_normalize_step.py`

Mirrors `poet_merge_step` in structure.

- [ ] **Step 10.1: Implement `ngpt_normalize_step.py`**

Create `src/patches/ngpt_normalize_step.py`:
```python
"""Patch: post-step L2-projection of nGPT weight matrices onto the sphere.

Targets ``megatron.training.training.train_step``. After each step,
calls `normalize_module_matrices(model._ngpt_norm_role_map)` for every
model chunk. This mirrors the reference train.py:500 line where
`normalize_matrices()` is called every iteration.

This is structurally identical to src/patches/poet_merge_step.py; we
keep them as separate patches because (a) the role registries differ
and (b) one experiment may run with nGPT but not POET (and vice versa).
"""
from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.train_step",)
logger = logging.getLogger(__name__)


@register_patch(name="ngpt_normalize_step", targets=_TARGET)
def apply() -> None:
    from megatron.training import get_args
    from megatron.training import training as _mt

    from src.model.ngpt.normalize import normalize_module_matrices

    _orig = _mt.train_step

    def _wrapped(*args, **kwargs):
        ret = _orig(*args, **kwargs)
        opts = get_args()
        if not getattr(opts, "ngpt", False):
            return ret
        model = args[2] if len(args) >= 3 else kwargs.get("model")
        if model is None:
            return ret
        chunks = model if isinstance(model, list) else [model]
        for m in chunks:
            role_map = getattr(m, "_ngpt_norm_role_map", None)
            if role_map:
                normalize_module_matrices(role_map)
        return ret

    _mt.train_step = _wrapped
```

- [ ] **Step 10.2: Commit**

```bash
git add src/patches/ngpt_normalize_step.py
git commit -m "feat(ngpt): patch — post-step weight L2-projection (mirrors reference train.py:500)"
```

---

## Task 11: Patch — `ngpt_optimizer_setup` (zero WD on scaling params, no warmup)

**Files:**
- Create: `src/patches/ngpt_optimizer_setup.py`
- Create: `tests/unit/test_ngpt_optimizer_groups.py`

Reference uses `weight_decay=0.0` and `warmup_iters=0` when `use_nGPT==1` ([train.py:111-114](file:///lustre/fast/fast/zqiu/tmp/ngpt/train.py#L111-L114)). For Megatron we keep the standard AdamW (with global WD set per config) and rely on per-parameter zero-WD groups for the scaling vectors. We also disable the LR warmup via a CLI flag, not via the patch.

- [ ] **Step 11.1: Write the failing test**

Create `tests/unit/test_ngpt_optimizer_groups.py`:
```python
"""CPU tests for the nGPT param-group classifier."""
import torch
import torch.nn as nn

from src.patches.ngpt_optimizer_setup import classify_ngpt_param_groups
from src.model.ngpt.scaling_params import LearnedScaling


def test_scaling_params_get_zero_wd_group():
    m = nn.Module()
    m.linear = nn.Linear(8, 8, bias=False)
    m.attn_alpha = LearnedScaling((8,), init_value=0.05, init_scaling=1.0 / 2.83)
    m.mlp_alpha = LearnedScaling((8,), init_value=0.05, init_scaling=1.0 / 2.83)
    m.sqk = LearnedScaling((8,), init_value=1.0, init_scaling=1.0 / 2.83)
    m.suv = LearnedScaling((8,), init_value=1.0, init_scaling=1.0)
    m._ngpt_sz = LearnedScaling((100,), init_value=1.0, init_scaling=1.0 / 2.83)

    decay, no_decay = classify_ngpt_param_groups(m)
    # linear.weight should be in decay (it is the only matrix param)
    decay_ids = {id(p) for p in decay}
    no_decay_ids = {id(p) for p in no_decay}
    assert id(m.linear.weight) in decay_ids
    for p in (m.attn_alpha.param, m.mlp_alpha.param, m.sqk.param,
              m.suv.param, m._ngpt_sz.param):
        assert id(p) in no_decay_ids, (
            f"scaling param shape {p.shape} should be in the no-decay group"
        )
```

- [ ] **Step 11.2: Run; expect failure**

Run: `pytest tests/unit/test_ngpt_optimizer_groups.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 11.3: Implement `ngpt_optimizer_setup.py`**

Create `src/patches/ngpt_optimizer_setup.py`:
```python
"""Patch: route scaling params into a zero-weight-decay group for nGPT runs.

Targets ``megatron.training.training.get_megatron_optimizer`` — we
intercept the call only when args.ngpt is set, so AdamW gets two param
groups (decay vs no-decay) and the scaling vectors (sqk, suv, alpha*,
sz) never get pulled toward zero.

The companion helper `classify_ngpt_param_groups(model)` is what the
test exercises; the patch itself just delegates.
"""
from __future__ import annotations

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.get_megatron_optimizer",)

_SCALING_NAME_FRAGMENTS = (
    "_ngpt_sz.param",
    ".sqk.param",
    ".suv.param",
    ".attn_alpha.param",
    ".mlp_alpha.param",
)


def classify_ngpt_param_groups(model) -> tuple[list, list]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(frag in name for frag in _SCALING_NAME_FRAGMENTS) or name in _SCALING_NAME_FRAGMENTS:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


@register_patch(name="ngpt_optimizer_setup", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt

    _orig = _mt.get_megatron_optimizer

    def _wrapped(config, model, **kwargs):
        opt = _orig(config, model, **kwargs)
        if not getattr(config, "ngpt", False):
            return opt
        # Walk the optimizer's param groups and move scaling params into a
        # zero-WD group. Megatron may return ChainedOptimizer or
        # Float16OptimizerWithFloat16Params; both expose .param_groups via
        # their inner optimizer. We mutate the inner optimizer in place.
        chunks = model if isinstance(model, list) else [model]
        scaling_param_ids: set[int] = set()
        for m in chunks:
            _, no_decay = classify_ngpt_param_groups(m)
            scaling_param_ids.update(id(p) for p in no_decay)

        inner = getattr(opt, "optimizer", None) or opt
        for group in inner.param_groups:
            if any(id(p) in scaling_param_ids for p in group["params"]):
                group["weight_decay"] = 0.0
        return opt

    _mt.get_megatron_optimizer = _wrapped
```

- [ ] **Step 11.4: Re-run tests**

Run: `pytest tests/unit/test_ngpt_optimizer_groups.py tests/unit/test_ngpt_patch_registry.py -v`

Expected: all pass. The `test_ngpt_patch_registry.py` tests should now succeed too because all three patches are importable.

- [ ] **Step 11.5: Commit**

```bash
git add src/patches/ngpt_optimizer_setup.py tests/unit/test_ngpt_optimizer_groups.py
git commit -m "feat(ngpt): patch — zero weight-decay on sqk/suv/alpha/sz scaling params"
```

---

## Task 12: Launcher + argv plumbing

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py:26-42` (extend the slm-research argument group)
- Modify: `src/utils/megatron_args.py` (emit nGPT flags + force `--no-bias-linear` + override LR warmup)
- Create: `tests/unit/test_ngpt_megatron_args.py`

- [ ] **Step 12.1: Add nGPT CLI args**

Edit `launchers/pretrain_gpt_slm.py`. Inside `add_slm_args`, append these lines just before `return parser`:

```python
    group.add_argument("--ngpt", action="store_true")
    group.add_argument("--ngpt-base-scale", type=float, default=None,
                       help="1/sqrt(hidden_size) by default")
    group.add_argument("--ngpt-alpha-init", type=float, default=0.05)
    group.add_argument("--ngpt-sqk-init", type=float, default=1.0)
    group.add_argument("--ngpt-suv-init", type=float, default=1.0)
    group.add_argument("--ngpt-sz-init", type=float, default=1.0)
    group.add_argument("--ngpt-no-warmup", action="store_true",
                       help="Force LR warmup steps to 0 (matches reference train.py:114)")
```

Also extend the choices on `--slm-optimizer`:
```python
group.add_argument("--slm-optimizer", choices=["adamw", "muon", "poet", "ngpt_adamw"], default="adamw")
```

(`ngpt_adamw` is just `adamw` with the nGPT param-group classifier hooked through `ngpt_optimizer_setup`.)

- [ ] **Step 12.2: Write the failing test for `megatron_args`**

Create `tests/unit/test_ngpt_megatron_args.py`:
```python
"""Tests that the nGPT experiment YAML composes into the right Megatron CLI args."""
import pytest
from omegaconf import OmegaConf

from src.utils.megatron_args import build_megatron_args


def _ngpt_cfg():
    return OmegaConf.create({
        "base": {
            "family": "llama3",
            "scale": "600m",
            "non_embedding_params": 600_000_000,
            "model": {
                "num_layers": 4, "hidden_size": 32, "ffn_hidden_size": 128,
                "num_attention_heads": 4, "num_query_groups": 4, "head_dim": 8,
                "seq_length": 64, "max_position_embeddings": 64,
                "positional_encoding": "rope", "rotary_base": 500000,
                "attention_dropout": 0.0, "hidden_dropout": 0.0,
                "normalization": "RMSNorm", "norm_epsilon": 1e-5,
                "init_method_std": 0.02, "tie_embeddings": True,
                "attention_backend": "flash", "qk_norm": False,
            },
        },
        "training": {
            "tokens_per_param": 20, "global_batch_size_tokens": 4096,
            "seq_length": 64, "micro_batch_size": 1, "log_interval": 1,
            "eval_iters": 1, "eval_interval": 1, "save_interval": 100000,
        },
        "optim": {
            "type": "ngpt_adamw", "lr": 15e-4, "weight_decay": 0.0,
            "betas": [0.9, 0.95], "eps": 1e-8,
            "ngpt": {"alpha_init": 0.05, "sqk_init": 1.0,
                     "suv_init": 1.0, "sz_init": 1.0, "no_warmup": True},
        },
        "parallelism": {"tp": 1, "pp": 1, "sequence_parallel": False},
        "data": {
            "path": "/tmp/x", "tokenizer_type": "GPT2BPETokenizer",
            "tokenizer_model": "/tmp/t", "vocab_size": 100, "name": "x",
            "split": "100,0,0", "no_mmap_bin_files": False,
            "no_create_attention_mask_in_dataloader": False, "num_workers": 0,
        },
        "wandb": {"project": "test"},
        "experiment": {"name": "ngpt", "kind": "ngpt"},
        "seed": 0,
    })


def test_emits_ngpt_flag():
    args = build_megatron_args(_ngpt_cfg())
    assert "--ngpt" in args


def test_emits_ngpt_init_flags():
    args = build_megatron_args(_ngpt_cfg())
    for flag in ("--ngpt-alpha-init", "--ngpt-sqk-init", "--ngpt-suv-init",
                 "--ngpt-sz-init"):
        assert flag in args, f"missing flag: {flag}"


def test_no_warmup_emits_zero_warmup_samples():
    args = build_megatron_args(_ngpt_cfg())
    i = args.index("--lr-warmup-samples")
    assert int(args[i + 1]) == 0


def test_no_warmup_false_keeps_default_warmup_samples():
    cfg = _ngpt_cfg()
    cfg.optim.ngpt.no_warmup = False
    args = build_megatron_args(cfg)
    i = args.index("--lr-warmup-samples")
    assert int(args[i + 1]) > 0


def test_disable_bias_linear_still_present():
    """nGPT relies on disable-bias-linear (default in _model_args). Sanity check."""
    args = build_megatron_args(_ngpt_cfg())
    assert "--disable-bias-linear" in args
```

- [ ] **Step 12.3: Run; expect failures**

Run: `pytest tests/unit/test_ngpt_megatron_args.py -v`
Expected: failures on `--ngpt` not in args, etc.

- [ ] **Step 12.4: Extend `megatron_args.py`**

Edit `src/utils/megatron_args.py`. Add an nGPT block inside `_optimizer_args` AFTER the `if kind == "poet"` branch, before the final `raise ValueError`:

```python
    if kind == "ngpt_adamw":
        ng = optim.get("ngpt", {})
        return _sequence([
            "--optimizer", "adam",
            "--slm-optimizer", "ngpt_adamw",
            "--ngpt",
            "--ngpt-alpha-init", float(ng.get("alpha_init", 0.05)),
            "--ngpt-sqk-init", float(ng.get("sqk_init", 1.0)),
            "--ngpt-suv-init", float(ng.get("suv_init", 1.0)),
            "--ngpt-sz-init", float(ng.get("sz_init", 1.0)),
            "--adam-beta1", optim.betas[0],
            "--adam-beta2", optim.betas[1],
            "--adam-eps", optim.eps,
        ] + (["--ngpt-no-warmup"] if bool(ng.get("no_warmup", True)) else []))
```

In `_training_args`, replace:
```python
_add(args, "--lr-warmup-samples", max(1, (total_tokens // seq_length) // 500))
```
with:
```python
warmup_samples = 0 if bool(cfg.optim.get("ngpt", {}).get("no_warmup", False)) else max(
    1, (total_tokens // seq_length) // 500
)
_add(args, "--lr-warmup-samples", warmup_samples)
```

- [ ] **Step 12.5: Re-run tests, expect PASS**

Run: `pytest tests/unit/test_ngpt_megatron_args.py -v`
Expected: 5 passed.

- [ ] **Step 12.6: Commit**

```bash
git add launchers/pretrain_gpt_slm.py src/utils/megatron_args.py tests/unit/test_ngpt_megatron_args.py
git commit -m "feat(ngpt): launcher CLI + megatron_args emit --ngpt* and zero-warmup"
```

---

## Task 13: Experiment YAML + lab notebook

**Files:**
- Create: `configs/experiments/arch/ngpt.yaml`
- Create: `docs/experiments/ngpt.md`

- [ ] **Step 13.1: Write `configs/experiments/arch/ngpt.yaml`**

Create `configs/experiments/arch/ngpt.yaml`:
```yaml
# @package _global_
# nGPT (Normalized Transformer on the Hypersphere). See arXiv:2410.01131.
#
# Triggers patches: ngpt_apply_spec, ngpt_normalize_step, ngpt_optimizer_setup.
experiment:
  name: ngpt
  family: arch
  kind: ngpt
  description: |
    nGPT — every attention/MLP/embedding matrix lives on the unit
    hypersphere; per-step L2 projection enforces that geometrically.
    Residual blocks use a learnable eigen-LR blend instead of additive
    residuals. Q/K are L2-normalized per head; softmax_scale is
    sqrt(head_dim) (not 1/sqrt(head_dim)). Output logits are
    multiplied by a learnable per-vocab scale `sz`.
  references:
    - "Loshchilov et al. 2024 (arXiv:2410.01131)"
    - "Reference: https://github.com/NVIDIA/ngpt"
  patches:
    - ngpt_apply_spec
    - ngpt_normalize_step
    - ngpt_optimizer_setup
  required_capabilities: []

optim:
  type: ngpt_adamw
  lr: 15.0e-4                # reference train.py:98
  weight_decay: 0.0          # scaling params are also zero-WD by patch
  betas: [0.9, 0.95]
  eps: 1.0e-8
  ngpt:
    alpha_init: 0.05         # attn_alpha / mlp_alpha init_value
    sqk_init: 1.0
    suv_init: 1.0
    sz_init: 1.0
    no_warmup: true          # reference train.py:114

# Disable QK layernorm and dropout — they conflict with nGPT.
base:
  model:
    qk_norm: false
    attention_dropout: 0.0
    hidden_dropout: 0.0
```

- [ ] **Step 13.2: Write `docs/experiments/ngpt.md`**

Create `docs/experiments/ngpt.md`:
```markdown
# nGPT — Normalized Transformer on the Hypersphere

**Reference:** Loshchilov et al. 2024 — [arXiv:2410.01131](https://arxiv.org/abs/2410.01131). NVIDIA reference impl: https://github.com/NVIDIA/ngpt (vendored at [/lustre/fast/fast/zqiu/tmp/ngpt](file:///lustre/fast/fast/zqiu/tmp/ngpt)).

## Hypothesis
nGPT replaces additive residuals with a per-channel eigen-LR blend on S^{C-1}, normalizes Q/K per head, and enforces per-row/column unit norm on every matrix after each optimizer step. The paper claims 4×–10× speedups at 1k–8k context relative to a standard GPT baseline. We want to see whether this transfers to slm-research's 600M dense ablation track with our frozen tokenizer.

## Mechanism (slm-research integration)
* Custom `NGPTTransformerLayer` overrides `forward` to do hypersphere blending; standard Megatron `SelfAttention` + custom `NGPTMLPBody`. Spec: [src/specs/ngpt_layer_spec.py](../../src/specs/ngpt_layer_spec.py).
* `QKHyperNorm` plugs into `q_layernorm`/`k_layernorm` slots; provides the `sqk` scaling.
* `attn_alpha`, `mlp_alpha` (per-channel eigen LR) live on each `NGPTTransformerLayer`; `suv` lives on `NGPTMLPBody`; `sz` is attached to the GPTModel post-build.
* Per-step weight normalization runs via the `ngpt_normalize_step` patch on `train_step`.
* No QK layernorm, no bias on linears, no LR warmup, AdamW weight-decay zero on scaling params.

## v1 scope (this implementation)
* 600M dense, single-node, TP=1, PP=1, bf16.
* CPU parity test against the vendored NVIDIA reference at toy config (2 layers / 64 hidden / vocab 100).

## v2 candidates (not in this PR)
* TP > 1: per-rank sqk/suv sharding.
* MoE flavour (nGPT-MoE).
* MLA (nGPT-MLA) compatibility.
* FP8 / FP4 — paper notes nGPT is less sensitive to low precision than baseline GPT, so this is an interesting cross-axis ablation.

## How to run
```bash
python -m launchers.submit \
    base/family=llama3 \
    base/scale=600m \
    experiment=arch/ngpt \
    training_regime=ablation_20x \
    cluster=h800_cn \
    seed=42
```

## Result log
(populate as runs land)
```

- [ ] **Step 13.3: Update CHANGELOG**

Append to [/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md](file:///lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md) (per user's standing instruction):

```
- 2026-05-25 slm-research: add nGPT architecture variant — custom NGPTTransformerLayer + spec
  + three patches (apply_spec / normalize_step / optimizer_setup) + experiment YAML + parity
  test against vendored NVIDIA reference at toy config. v1 scope: 600M dense, TP=1, bf16.
```

- [ ] **Step 13.4: Commit**

```bash
git add configs/experiments/arch/ngpt.yaml docs/experiments/ngpt.md
git commit -m "feat(ngpt): experiment YAML + lab notebook"
```

---

## Task 14: Full-model parity test vs reference GPT

**Files:**
- Create: `tests/unit/test_ngpt_full_parity.py`

This is the load-bearing oracle test. We assemble a 2-layer toy model out of `NGPTBlock`s + a token embedding + LM head + `sz`, mirroring the reference `GPT(use_nGPT=1)`, and compare forward outputs token-for-token.

- [ ] **Step 14.1: Implement the test**

Create `tests/unit/test_ngpt_full_parity.py`:
```python
"""Full-model parity vs the vendored NVIDIA reference at a toy config.

We assemble our own minimal nGPT model from primitives (token
embedding -> N x NGPTBlock -> lm_head -> sz) so we can run in fp32 on
CPU and bit-for-bit compare to the reference GPT(use_nGPT=1).
"""
import math

import pytest
import torch
import torch.nn as nn

from src.model.ngpt.layer import NGPTBlock
from src.model.ngpt.normalize import normalize_module_matrices
from src.model.ngpt.scaling_params import LearnedScaling
from tests._fixtures.ngpt_reference.model import GPT as RefGPT, GPTConfig


def _build_role_map(model, n_layer):
    role_map = {}
    role_map[model.wte.weight] = "rows"
    role_map[model.lm_head.weight] = "rows"
    for i in range(n_layer):
        b = model.blocks[i]
        role_map[b.query.weight] = "rows"
        role_map[b.key.weight] = "rows"
        role_map[b.value.weight] = "rows"
        role_map[b.att_c_proj.weight] = "cols"
        role_map[b.c_fc.weight] = "rows"
        role_map[b.mlp_c_proj.weight] = "cols"
    return role_map


class _OurNGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd, dtype=torch.float32)
        self.blocks = nn.ModuleList([
            NGPTBlock(
                hidden_size=cfg.n_embd, num_heads=cfg.n_head,
                ffn_hidden_size=4 * cfg.n_embd, base_scale=cfg.base_scale,
                dtype=torch.float32,
            ) for _ in range(cfg.n_layer)
        ])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False, dtype=torch.float32)
        self.sz = LearnedScaling((cfg.vocab_size,), init_value=1.0, init_scaling=cfg.base_scale)
        # Initialize all 2D weights as in reference: normal_(0, base_scale)
        with torch.no_grad():
            nn.init.normal_(self.wte.weight, mean=0.0, std=cfg.base_scale)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=cfg.base_scale)
            for b in self.blocks:
                for lin in (b.query, b.key, b.value, b.att_c_proj, b.c_fc, b.mlp_c_proj):
                    nn.init.normal_(lin.weight, mean=0.0, std=cfg.base_scale)

    def forward(self, idx):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(x)
        sz_eff = self.sz.scaled_value()
        return sz_eff * logits


def _copy_ref_to_ours(ref: RefGPT, ours: _OurNGPT, cfg: GPTConfig):
    with torch.no_grad():
        ours.wte.weight.copy_(ref.transformer.wte.weight.float())
        ours.lm_head.weight.copy_(ref.lm_head.weight.float())
        ours.sz.param.copy_(ref.sz)
        for i in range(cfg.n_layer):
            rb = ref.transformer.h[i]
            ob = ours.blocks[i]
            ob.query.weight.copy_(rb.query.weight.float())
            ob.key.weight.copy_(rb.key.weight.float())
            ob.value.weight.copy_(rb.value.weight.float())
            ob.att_c_proj.weight.copy_(rb.att_c_proj.weight.float())
            ob.c_fc.weight.copy_(rb.c_fc.weight.float())
            ob.mlp_c_proj.weight.copy_(rb.mlp_c_proj.weight.float())
            ob.sqk.param.copy_(rb.sqk)
            ob.suv.param.copy_(rb.suv)
            ob.attn_alpha.param.copy_(rb.attn_alpha)
            ob.mlp_alpha.param.copy_(rb.mlp_alpha)


def test_full_model_logit_parity_at_init():
    torch.manual_seed(7)
    cfg = GPTConfig(
        block_size=16, vocab_size=37,
        n_layer=2, n_head=4, n_embd=32,
        base_scale=1.0 / math.sqrt(32),
        use_nGPT=1, dropout=0.0, bias=False,
    )
    ref = RefGPT(cfg).float()
    ours = _OurNGPT(cfg)
    _copy_ref_to_ours(ref, ours, cfg)

    # Normalize like the reference does at init (train.py:411).
    normalize_module_matrices(_build_role_map(ours, cfg.n_layer))

    # Also re-copy normalized weights back to ref so both are starting in
    # the same state.
    with torch.no_grad():
        ref.transformer.wte.weight.copy_(ours.wte.weight)
        ref.lm_head.weight.copy_(ours.lm_head.weight)
        for i in range(cfg.n_layer):
            rb = ref.transformer.h[i]
            ob = ours.blocks[i]
            rb.query.weight.copy_(ob.query.weight.to(rb.query.weight.dtype))
            rb.key.weight.copy_(ob.key.weight.to(rb.key.weight.dtype))
            rb.value.weight.copy_(ob.value.weight.to(rb.value.weight.dtype))
            rb.att_c_proj.weight.copy_(ob.att_c_proj.weight.to(rb.att_c_proj.weight.dtype))
            rb.c_fc.weight.copy_(ob.c_fc.weight.to(rb.c_fc.weight.dtype))
            rb.mlp_c_proj.weight.copy_(ob.mlp_c_proj.weight.to(rb.mlp_c_proj.weight.dtype))

    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    ours.eval(); ref.eval()
    with torch.no_grad():
        ours_logits = ours(idx)
        # reference returns (logits, loss); request loss by passing targets so it
        # runs the full lm_head path (matches our forward).
        ref_logits, _ = ref(idx, targets=idx)
    assert ours_logits.shape == ref_logits.shape
    # bf16 weights inside ref dominate; loose tolerance but functionally equivalent.
    abs_diff = (ours_logits - ref_logits.float()).abs().max().item()
    rel = (ours_logits - ref_logits.float()).abs().max().item() / max(1e-6, ref_logits.float().abs().max().item())
    assert abs_diff < 5e-2 or rel < 1e-2, (
        f"logit parity failed: max abs diff = {abs_diff}, rel = {rel}"
    )
```

- [ ] **Step 14.2: Run the test**

Run: `pytest tests/unit/test_ngpt_full_parity.py -v`
Expected: PASS. (If the abs/rel tolerance is too tight for bf16-vs-fp32 noise on certain machines, document and bump in the same commit — but do NOT relax until you understand the actual delta first.)

- [ ] **Step 14.3: Commit**

```bash
git add tests/unit/test_ngpt_full_parity.py
git commit -m "test(ngpt): full-model logit parity vs vendored NVIDIA reference at toy config"
```

---

## Task 15: GPU smoke runbook

**Files:**
- Create: `docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md`

- [ ] **Step 15.1: Write the runbook**

Create `docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md`:
````markdown
# nGPT smoke run

## Goal
Confirm the nGPT variant trains end-to-end on a single H100/H800 node for ~100 steps, with no NaNs, monotonically decreasing loss, and post-step weight normalization actually firing.

## Cluster
`h800_cn`, single node, 8 GPUs. Submitter:
```bash
python -m launchers.submit \
    base/family=llama3 \
    base/scale=600m \
    experiment=arch/ngpt \
    training_regime=ablation_20x \
    cluster=h800_cn \
    seed=0 \
    +training.total_tokens=2000000000  # cap at ~100 steps for the smoke
```

## What to look for
- `[nGPT] applied spec + attached sz + registered weight-norm roles` in rank-0 stdout after model build.
- Training loss strictly decreasing across the first 50 steps; no NaN.
- After ~10 steps, sample a parameter (e.g. `module.transformer.layers[0].self_attention.linear_qkv.weight`) and confirm row-norms are ≈ 1.0.
- Check the W&B run has separate `lr_groups/decay` vs `lr_groups/no_decay` (the no-decay group should contain sz, sqk, suv, attn_alpha, mlp_alpha).

## If it fails
1. **Spec swap didn't fire** — `--ngpt` not propagated to argv. Check `_optimizer_args` block in `src/utils/megatron_args.py`.
2. **`attach_sz_scaling` AttributeError** — `args.padded_vocab_size` not yet set when `gpt_builder` is patched; move the attach to a later hook (post-pretrain init).
3. **Hypersphere normalization doesn't fire** — `model._ngpt_norm_role_map` empty or under-sized. The assertion inside `_register_ngpt_norm_roles` (see Task 9) should trip first with a count; if not, the Megatron submodule naming changed — extend `_NORM_ROLES_BY_SUFFIX`.
4. **NaN at step 1** — confirm `config.softmax_scale ≈ sqrt(head_dim)` by inspecting `model.module.decoder.layers[0].self_attention.core_attention.softmax_scale` in a debugger. If it's still `1/sqrt(head_dim)`, the wrap of `core_transformer_config_from_args` in `ngpt_apply_spec` didn't fire — verify the patch is in `experiment.patches` and was applied before model build.
5. **alphas missing from optimizer** — log `len([n for n,_ in model.named_parameters() if "attn_alpha" in n or "mlp_alpha" in n])` after build; expect `2 * num_layers`. If zero, `NGPTTransformerLayer.__init__` regressed back to lazy build.

## Promotion
If smoke is green at 100 steps, hand off to a 24-hour 24B-token ablation run on the same cluster.
````

- [ ] **Step 15.2: Final CHANGELOG bump**

Append to [/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md](file:///lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md):
```
- 2026-05-25 slm-research/ngpt: smoke runbook published; ready for cluster smoke.
```

- [ ] **Step 15.3: Commit**

```bash
git add docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md
git commit -m "docs(ngpt): cluster smoke runbook"
```

---

## Task 16: Merge gate — verify, then decide how to land

This task **does not write any new code**. It enforces the merge gate spelled out in Prerequisite #3. Do not skip steps or relax the criteria.

- [ ] **Step 16.1: Run the nGPT test slice in the worktree**

Run: `pytest tests/unit/test_ngpt_normalize.py tests/unit/test_ngpt_scaling_params.py tests/unit/test_ngpt_attention.py tests/unit/test_ngpt_mlp.py tests/unit/test_ngpt_layer_block_forward.py tests/unit/test_ngpt_output_scaling.py tests/unit/test_ngpt_layer_spec.py tests/unit/test_ngpt_patch_registry.py tests/unit/test_ngpt_optimizer_groups.py tests/unit/test_ngpt_megatron_args.py tests/unit/test_ngpt_full_parity.py -v`

Expected: every test passes, zero failures, zero errors.

If anything fails: fix on `ngpt_arch`, re-run. Do **not** proceed to Step 16.2 with a red bar.

- [ ] **Step 16.2: Run the full existing unit-test suite to catch regressions**

Run: `pytest tests/unit/ -v`

Expected: zero failures across the *entire* `tests/unit/` directory. POET, muon, launcher composition, and patch-registry tests must still pass — the new patches must not collide with existing patches at registration time, and `_register_ngpt_norm_roles`'s assertions must not fire during unrelated tests.

If anything regresses (POET tests fail, launcher composition fails, etc.): fix on `ngpt-arch`. Common causes:
- An nGPT patch shares a target with an existing patch (`core_transformer_config_from_args` is already touched by `poet_unfuse_te_impl` — but per [src/patches/_registry.py](../../../src/patches/_registry.py) the conflict only fires when *both* patches are loaded in the same process, which won't happen for mutually-exclusive experiments. Verify by inspection.)
- An nGPT module unconditionally registers itself at import (it shouldn't — modules under `src/model/ngpt/` are inert at import time; only `src/patches/ngpt_*.py` registers anything).

- [ ] **Step 16.3: Hand the smoke runbook to the user**

Post to the user: "Tasks 1–15 are landed on `ngpt-arch`. The CPU test suite is green (Step 16.1) and there are no regressions (Step 16.2). The runbook at `docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md` is ready for a cluster smoke run. Please run it and report back."

Then **stop**. Do not proceed to Step 16.4 until the user replies with cluster results.

- [ ] **Step 16.4: Verify the user's smoke result**

Required signals from the user:
- Training loss strictly decreasing across the first ~50 steps.
- No NaN in any logged scalar.
- Row-norm check on a sampled QKV weight (`module.transformer.layers[0].self_attention.linear_qkv.weight`) returns ≈ 1.0 after step 10 — confirms `ngpt_normalize_step` is firing.
- W&B param-group breakdown shows non-zero parameter counts in both `decay` and `no_decay` groups, with the scaling params (sz, sqk, suv, attn_alpha, mlp_alpha) in the no-decay group.

If any signal is missing or wrong: fix on `ngpt-arch`, re-run smoke. Loop until all four are green.

- [ ] **Step 16.5: Invoke `superpowers:finishing-a-development-branch` to decide how to land**

The skill presents merge / PR / cleanup options to the user. Do not pre-decide — let the user pick. Whatever they pick:
- Do **not** force-push.
- Do **not** rewrite history on a shared branch (POET is still active on `poet-cayley-cache`).
- Do **not** delete the worktree until the user has confirmed the merge succeeded.

- [ ] **Step 16.6: Final CHANGELOG entry**

After the merge (or PR creation) succeeds, append a single closing line to [/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md](file:///lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md):

```
- 2026-05-25 slm-research/ngpt: v1 landed (TP=1, dense, 600M); see <merge SHA or PR URL>.
```

This is the only commit allowed on the parent branch by this plan. Everything else lives on `ngpt-arch`.

---

## Post-plan: what's deliberately not done here

1. **TP > 1.** The spec builder asserts TP=1. v2 will shard `sqk` per TP rank (split along the head dim) and `suv` per TP rank (split along the 2×ffn dim), and replace `NGPTMLPBody`'s `nn.Linear` with Megatron's `ColumnParallelLinear` / `RowParallelLinear`.
2. **MoE / MLA.** Asserted out in the spec builder.
3. **Inference / KV cache path.** Training only.
4. **Cross-family parity.** The parity test uses the reference's own block construction (GPT-2-shaped, full MHA, RoPE on the head dim). A second test against, say, the Megatron Llama-3 reference is out of scope.

---

## Self-review

**Spec coverage:** Every required nGPT mechanism from the reference is mapped to a task:
- `justnorm` → Task 2
- Q/K hypersphere normalization + `sqk` → Task 4
- SwiGLU `suv` scaling → Task 5
- Hypersphere residual blend + `attn_alpha`/`mlp_alpha` → Task 6
- Output `sz` → Task 7
- Post-step weight L2 projection → Task 10
- Zero-WD for scaling params → Task 11
- No-warmup LR schedule → Task 12 (`_training_args` patch)
- `softmax_scale = sqrt(head_dim)` → Task 9 (stamped onto `TransformerConfig` by `ngpt_apply_spec`'s wrap of `core_transformer_config_from_args`)
- Glue to Megatron's model-build pipeline → Task 9
- Worktree isolation + test-before-merge enforcement → Prerequisites #1, #3 and Task 16

**Placeholder scan:** No `TODO`/`TBD`/"implement later" except the v2 candidates explicitly enumerated in Task 15 and "Post-plan" section, which are *non-goals*, not placeholders.

**Type consistency:** `LearnedScaling`, `NGPTBlock`, `NGPTTransformerLayer`, `QKHyperNorm`, `NGPTMLPBody`, `attach_sz_scaling`, `classify_ngpt_param_groups`, `build_ngpt_layer_spec`, `normalize_module_matrices` — names used in tests match names used in implementations; signatures consistent across Tasks 2-11.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-25-ngpt-architecture-variant.md`.

**Before the first task fires:** create the worktree via `superpowers:using-git-worktrees` (branch `ngpt-arch` off `poet-cayley-cache`). All 16 tasks run inside that worktree. Nothing merges into the parent branch until Task 16's gate (CPU tests green + no regressions + user-confirmed cluster smoke) is fully cleared.

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. The subagents work inside the worktree; the merge gate (Task 16) runs in the main session so I can verify test output before authorizing any merge.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Task 16's smoke-result step (16.4) pauses for the user regardless.

Which approach?

---

# EXECUTION STATUS — 2026-05-25 (RESUME HERE)

**Workspace:** Branched **in place** (`git checkout -b ngpt-arch` off `poet-cayley-cache` @ `7532602`, which is untouched). NOT a separate worktree — committing the WIP for a worktree was blocked by the repo's pre-commit ruff, and the editable install pins `src` to this dir. The branch carries the user's uncommitted WIP (do not commit it; only nGPT files). Subagent-driven execution was used for Tasks 1–15.

## Done — Tasks 1–15 committed on `ngpt-arch`
17 commits, `ac2b25e..0ccf90a`. Tasks 1–15 are the 15 `feat/test/docs(ngpt)` commits `ac2b25e..f08b63e`; the last two (`cf89d9b`, `0ccf90a`) are smoke-driven fixes (below).
- **CPU/GPU test slice (gate 16.1): 32 passed**, 0 failures. Parity tests (`test_ngpt_layer_block_forward`, `test_ngpt_full_parity`) match the vendored NVIDIA reference **bit-for-bit (diff 0.0)** — they run on **CUDA** (flash_attn 2.8.3 is installed → the reference is CUDA-only; the plan's "CPU-only" premise was wrong).
- **Full suite (gate 16.2): 145 passed, 2 failed** — the 2 failures are **pre-existing WIP drift, unrelated to nGPT** and out of scope: `test_launcher_config_composition::test_parse_overrides_loads_defaults_and_data_axis` (asserts cluster `h800_cn`, but WIP changed default to `b200_de`) and `test_poet_megatron_builder::test_poet_builder_partitions_params` (mock cfg lacks `poet_cache_mode`). nGPT added 32 passing tests, **zero regressions**.

## Deviations from plan code already made (committed)
- **Task 8**: dropped the plan's `from ...custom_layers.transformer_engine import TENorm` (path doesn't exist in this Megatron; unused).
- **Task 11**: classifier used leading-dot substring fragments that misclassified the test's top-level params → switched to **trailing-segment matching**.
- **`cf89d9b` (MLP wiring, smoke-found)**: the plan wired `mlp`/`q_layernorm`/`k_layernorm` as builder **closures**, but Megatron's `build_module` returns a `FunctionType` **uninstantiated** (the MLP path does), so `layer.mlp` became a bare function with no params. Added a `config`-constructable `NGPTMLP(NGPTMLPBody)` class in `src/model/ngpt/mlp.py` and wired `mlp=ModuleSpec(module=NGPTMLP)`; added a regression-guard test. (QK-norm closures are fine — `SelfAttention` calls those slots *directly*, not via `build_module`.)
- **`0ccf90a` (forward signature, smoke-found)**: this Megatron's `TransformerLayer.forward(self, *args, **kwargs)` dispatches to `_forward_attention` and passes `rotary_pos_cos_sin` (combined). The plan's hardcoded signature crashed. Added `rotary_pos_cos_sin` + `padding_mask` + `**kwargs` to `NGPTTransformerLayer.forward` and pass `rotary_pos_cos_sin` through to `self.self_attention`.

## Local GPU smoke (Task 16.4) — IN PROGRESS, not yet green
Running locally on this 8-GPU node (the user said "run it yourself"). Confirmed live at runtime: `[nGPT] applied spec + attached sz + registered weight-norm roles` fires on all ranks (patch wraps `gpt_builders.gpt_builder` correctly), `--ngpt*` flags/`lr-warmup=0`/TP=PP=1 all flow through, the weight-norm role-map assertion now **passes** (161 params; the MLP fix worked), and the model builds + reaches the forward pass.

### ⏭️ NEXT STEP (resume here): fix GQA in the k_layernorm sqk
Smoke #5 crashed at the **first forward step** with:
`RuntimeError: The size of tensor a (20) must match the size of tensor b (4) at non-singleton dimension 2` in `src/model/ngpt/attention.py:52 (sqk_eff * normed)`, via `attention.py:1492 key = apply_module(self.k_layernorm)(key)`.
Cause: this 600M llama3 config uses **GQA** (`num_attention_heads=20`, `num_query_groups=4`, `kv_channels=64`). The **key** tensor has 4 heads, but `k_layernorm`'s `QKHyperNorm` was built with 20-head `sqk`. The reference nGPT is plain MHA and didn't handle GQA.
**Fix (in `src/specs/ngpt_layer_spec.py`, ~lines 54–69):** build `q_layernorm` with `num_attention_heads` and `k_layernorm` with `num_query_groups`:
```python
num_heads = int(config.num_attention_heads)
num_kv_groups = int(getattr(config, "num_query_groups", None) or num_heads)
head_dim = int(getattr(config, "kv_channels", None) or (int(config.hidden_size) // num_heads))
...
q_layernorm=_qk_hyper_norm_builder(num_heads, head_dim, sqk_init, base_scale),
k_layernorm=_qk_hyper_norm_builder(num_kv_groups, head_dim, sqk_init, base_scale),
```
(q and k then carry separate sqk vectors of size `20*64` and `4*64` — a reasonable GQA extension of the reference's shared per-head sqk.) Then re-run the smoke. Watch for further issues: numerics/NaN at step 1, and whether the hypersphere blend behaves on Megatron's `(s, b, h)` layout.

### How to re-run the local smoke (fast — index cache `c2512f6…` is warm for this exact config)
```bash
source load_cuda13_2_nccl_env.sh   # load-bearing: cuda13.2 + LD_PRELOAD libcublasLt + cudnn
WANDB_MODE=disabled python -m launchers.train_megatron \
  base/family=llama3 base/scale=600m experiment=arch/ngpt \
  base.model.seq_length=256 training.seq_length=256 \
  training.global_batch_size_tokens=16384 training.micro_batch_size=8 \
  training.tokens_per_param=0.001 \
  training.log_interval=1 training.eval_iters=0 training.eval_interval=100000 \
  allow_dirty=true > /tmp/ngpt_smoke.log 2>&1 &
```
Env facts learned: local entrypoint is `launchers.train_megatron` (not `launchers.submit`, which is SLURM); `allow_dirty=true` is required (git-clean gate); `training.tokens_per_param=0.001` caps to ~36 steps (the plan/runbook's `+training.total_tokens=...` Hydra `+` syntax is **silently ignored** by this launcher's config layer — use `tokens_per_param`, an existing key); the dataset sample-index is built over the **entire** 831M-seq dataset regardless of train-samples (~15–20 min first time, then cached per config-hash). Mock data would skip the index but `_data_args` hardcodes `--data-path` and Megatron forbids both, so it'd need a code edit; also mock data won't show loss-decrease.

## Remaining gate steps after smoke is green
- 16.1/16.2 re-run to confirm the GQA + any further fixes didn't regress the CPU suite.
- Confirm the 4 smoke signals: apply marker (✓), loss↓ + no NaN, sampled `linear_qkv.weight` row-norms ≈1 after ~10 steps (proves `ngpt_normalize_step` fires), W&B/optimizer no-decay group holds the scaling params.
- Then `superpowers:finishing-a-development-branch` (merge/PR/cleanup — user picks). Offer to fold in the 2 pre-existing WIP test fixes if the user wants. External CHANGELOG `/lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md` does NOT exist → bumps skipped.
