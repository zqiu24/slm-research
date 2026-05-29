# POET split-fused-layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two opt-in args `--poet-split-qkv` / `--poet-split-fc1` that, under `--poet`, split a fused attention `linear_qkv` into separate Q/K/V projections and a fused MLP `linear_fc1` into separate gate/up projections — each a genuinely separate module with its own POET orbit.

**Architecture:** A new `src/optim/poet_split.py` performs the split surgery *before* the existing POET wrapper runs: it builds typed copies of the fused linear sliced to each sub-projection (GQA-correct de-interleave for QKV, contiguous halves for FC1), deletes the fused linear, and monkeypatches the owning module's forward (per-instance) to call the separate sub-linears. The split runs inside the existing `poet_apply_to_model` wrapper, so the resulting `ColumnParallelLinear` sub-linears are then POET-wrapped by the unchanged `replace_linears_with_poet` walker. No edits to `third_party/` or `poet_torch_huawei/`.

**Tech Stack:** Python, PyTorch, Megatron-core (pinned in `third_party/Megatron-LM`), pytest. POET enforces TP=1 and `transformer_impl=local`.

**Spec:** [docs/superpowers/specs/2026-05-29-poet-split-fused-layers-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-05-29-poet-split-fused-layers-design.md)

**IMPORTANT — test execution:** Per project convention, **the user runs tests/compute** (no working training env in this harness). After writing each test + implementation, present the exact `pytest` command and **wait for the user to report pass/fail** rather than asserting success yourself. The de-interleave math tests (Task 4) are pure-CPU and the most important to get green.

**Refinements vs spec (within intent, called out for review):**
- Sub-linears are built by typed deep-copy + weight-slice (not fresh `ColumnParallelLinear` construction) because they are immediately POET-wrapped and only their weight/bias/shape/type matter downstream.
- The patched `MLP.forward` computes the gated activation directly (Megatron's non-fused `glu()` path); the split path therefore does **not** use the fused `bias_swiglu_impl` kernel. Numerically identical for SwiGLU; a minor perf trade only on the split path.

---

## Runtime context & risks (review findings)

- **Post-DDP timing (load-bearing).** `megatron.training.training.get_model` wraps the model in `Float16Module` and then `DistributedDataParallel` *before returning* (third_party/Megatron-LM/megatron/training/training.py: `Float16Module` ~L1416, `DDP(...)` ~L1495, `return model` L1512). The `poet_apply_to_model` wrapper — and therefore the split — runs on the **already-wrapped, already-on-CUDA** model. This is the *same* position the existing `replace_linears_with_poet` runs in, so the split inherits its proven contract:
  - Deleting the fused base weight and freezing the split sub-weights is fine: base weights are `requires_grad=False` post-POET and never reduced by DDP (existing POET already orphans every replaced base weight this way).
  - The new `oft_R` params are added after DDP built its grad buffers, so DDP never reduces them — they are synced manually by the existing `_sync_oft_R_grads_across_dp` / `_flush_poet_caches_for_step` in [src/optim/poet.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L214), which iterate live POET layers (the split sub-linears are ordinary POET layers, so they are covered with no extra work).
  - **Consequence for `split_fused_linears`:** it walks `model.named_modules()`, which descends through `DDP → Float16Module → decoder → layers → self_attention/mlp` and reaches the fused linears exactly as the existing walker does. No special unwrapping needed. (Fix applied: the interleave-index buffer is created on `qkv.weight.device` because the model is already on CUDA.)
- **Simplified forwards drop some fused/offload paths.** The patched `MLP.forward` skips `bias_swiglu_impl`, fp8 activation store, and CPU-offload of activations; the patched `get_query_key_value_tensors` skips `offload_qkv_linear` offloading. All numerically equivalent; only perf/memory micro-opts are lost, and only on the split path. Acceptable for a research feature.
- **torch.compile.** Per-instance method patches may interact with `torch.compile`/dynamo (the Huawei adapter raised recompile caps for the same reason). POET runs `transformer_impl=local` and is typically eager; if a compiled path is enabled later, recompile-limit bumps may be needed. Out of scope here; flag for the smoke (Task 9).
- **MoE scope.** `split_fc1` matches every `MLP` whose `linear_fc1` is a real column-parallel linear — i.e. the dense MLP, shared experts, and `SequentialMLP` routed experts. Grouped-GEMM experts (`GroupedMLP`/`TEGroupedMLP`) have no such linear and are untouched (same as POET wrapping). For MoE, `block_size` must divide `moe_ffn_hidden_size` or the Task-4 hard error fires at startup (e.g. `moe_ffn_hidden_size=896` needs `block_size` dividing 896, not 256).
- **`_make_sub_linear` is the top production-risk spot** (deepcopy of a real Megatron linear). Process-group attrs are detached across the copy; validated by the Task 9 GPU smoke.

---

## File structure

- **Modify** [launchers/pretrain_gpt_slm.py](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py) — register two store-true args.
- **Modify** [src/utils/megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py) — emit the two flags from the `poet` branch.
- **Modify** [configs/experiments/optim/poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml) — `split_qkv` / `split_fc1` config surface (default false).
- **Create** [src/optim/poet_split.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_split.py) — pure geometry helpers + surgery + forward patches.
- **Modify** [src/patches/poet_apply_to_model.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py) — call `split_fused_linears` before `replace_linears_with_poet`.
- **Create** [tests/unit/test_poet_split.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_poet_split.py) — geometry, surgery, equivalence, hard-error tests.
- **Modify** [tests/unit/test_pretrain_gpt_slm.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_pretrain_gpt_slm.py) and [tests/unit/test_megatron_args.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_megatron_args.py) — arg-plumbing tests.

---

## Task 1: Register the two CLI args

**Files:**
- Modify: [launchers/pretrain_gpt_slm.py:44-49](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L44)
- Test: [tests/unit/test_pretrain_gpt_slm.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_pretrain_gpt_slm.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_pretrain_gpt_slm.py`:

```python
def test_add_slm_args_accepts_split_flags():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(
        [
            "--slm-config-path",
            "x.yaml",
            "--poet",
            "--poet-split-qkv",
            "--poet-split-fc1",
        ]
    )
    assert args.poet_split_qkv is True
    assert args.poet_split_fc1 is True


def test_split_flags_default_false():
    parser = argparse.ArgumentParser()
    add_slm_args(parser)
    args = parser.parse_args(["--slm-config-path", "x.yaml", "--poet"])
    assert args.poet_split_qkv is False
    assert args.poet_split_fc1 is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_pretrain_gpt_slm.py::test_add_slm_args_accepts_split_flags -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'poet_split_qkv'`.

- [ ] **Step 3: Add the args**

In `add_slm_args`, immediately after the `--poet-cache-mode` argument (currently ending at line 49), add:

```python
    group.add_argument("--poet-split-qkv", action="store_true")
    group.add_argument("--poet-split-fc1", action="store_true")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_pretrain_gpt_slm.py -v`
Expected: PASS (all, including the two new tests). **Wait for the user to confirm.**

- [ ] **Step 5: Commit**

```bash
git add launchers/pretrain_gpt_slm.py tests/unit/test_pretrain_gpt_slm.py
git commit -m "feat(poet): add --poet-split-qkv / --poet-split-fc1 CLI args"
```

---

## Task 2: Emit the flags from the megatron_args poet branch

**Files:**
- Modify: [src/utils/megatron_args.py:228-253](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L228)
- Test: [tests/unit/test_megatron_args.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_megatron_args.py)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_megatron_args.py` (the `_poet_cfg` helper already exists in this file):

```python
def test_poet_argv_omits_split_flags_by_default():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256}))
    assert "--poet-split-qkv" not in args
    assert "--poet-split-fc1" not in args


def test_poet_argv_emits_split_flags_when_set():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(_poet_cfg({"block_size": 256, "split_qkv": True, "split_fc1": True}))
    assert "--poet-split-qkv" in args
    assert "--poet-split-fc1" in args
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_megatron_args.py::test_poet_argv_emits_split_flags_when_set -v`
Expected: FAIL — `--poet-split-qkv` not found in args.

- [ ] **Step 3: Emit the flags**

In the `kind == "poet"` branch, change the `return _sequence([...])` so the list is built then extended with the conditional split flags. Replace:

```python
        return _sequence(
            [
                "--optimizer",
                "adam",
                "--slm-optimizer",
                "poet",
                "--poet",
                *block_args,
                "--poet-init-type",
                poet.init_type,
                "--poet-mup-alpha",
                poet.mup_alpha,
                "--poet-merge-period",
                poet.merge_period,
                "--poet-scale",
                poet.scale,
                "--poet-cache-mode",
                poet.get("cache_mode", "none"),
                "--adam-beta1",
                optim.betas[0],
                "--adam-beta2",
                optim.betas[1],
                "--adam-eps",
                optim.eps,
            ]
        )
```

with:

```python
        return _sequence(
            [
                "--optimizer",
                "adam",
                "--slm-optimizer",
                "poet",
                "--poet",
                *block_args,
                "--poet-init-type",
                poet.init_type,
                "--poet-mup-alpha",
                poet.mup_alpha,
                "--poet-merge-period",
                poet.merge_period,
                "--poet-scale",
                poet.scale,
                "--poet-cache-mode",
                poet.get("cache_mode", "none"),
                "--adam-beta1",
                optim.betas[0],
                "--adam-beta2",
                optim.betas[1],
                "--adam-eps",
                optim.eps,
            ]
            + (["--poet-split-qkv"] if poet.get("split_qkv", False) else [])
            + (["--poet-split-fc1"] if poet.get("split_fc1", False) else [])
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_megatron_args.py -k poet -v`
Expected: PASS (existing poet tests + the two new ones). **Wait for the user to confirm.**

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): plumb split_qkv/split_fc1 from config to megatron args"
```

---

## Task 3: Document the config surface in poet.yaml

**Files:**
- Modify: [configs/experiments/optim/poet.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet.yaml)

- [ ] **Step 1: Add the keys**

In the `optim.poet:` block, after the `scale:` line, add:

```yaml
    # Split fused linears into separate POET orbits (only meaningful under POET).
    # split_qkv: fused attention linear_qkv -> separate Q / K / V projections
    #   (GQA-correct de-interleave; inert for MLA, which has no fused linear_qkv).
    # split_fc1: fused SwiGLU linear_fc1 -> separate gate / up projections.
    # Each produced sub-segment must be divisible by block_size/block_count or
    # training hard-errors at startup.
    split_qkv: false
    split_fc1: false
```

- [ ] **Step 2: Sanity-check the config loads**

Run: `python -m pytest tests/unit/test_megatron_args.py::test_poet_args_use_slm_optimizer_and_keep_megatron_optimizer_adam -v`
Expected: PASS (config still resolves with the new keys). **Wait for the user to confirm.**

- [ ] **Step 3: Commit**

```bash
git add configs/experiments/optim/poet.yaml
git commit -m "docs(poet): document split_qkv/split_fc1 in poet experiment config"
```

---

## Task 4: Pure geometry helpers in `poet_split.py`

**Files:**
- Create: [src/optim/poet_split.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_split.py)
- Test: [tests/unit/test_poet_split.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_poet_split.py)

These are the highest-value, fully-CPU tests: they prove the de-interleave/reassembly math.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_poet_split.py`:

```python
"""Tests for POET fused-layer splitting (geometry + surgery)."""

import torch
import torch.nn as nn

from src.optim import poet_split as ps


def test_segment_out_dims_mqa_and_gqa():
    # MQA: 16 heads, 1 group, head_dim 384.
    assert ps.qkv_segment_out_dims(16, 1, 384) == (16 * 384, 1 * 384)
    # GQA: 8 heads, 2 groups, head_dim 64.
    assert ps.qkv_segment_out_dims(8, 2, 64) == (8 * 64, 2 * 64)


def test_deinterleave_row_indices_gqa_layout():
    # 4 heads, 2 groups (2 q-heads/group), head_dim 1 → trivial indices.
    # Fused per-group layout: [q,q,k,v] → group0 rows 0,1,2,3 ; group1 rows 4,5,6,7
    q, k, v = ps.qkv_deinterleave_row_indices(4, 2, 1)
    assert q.tolist() == [0, 1, 4, 5]
    assert k.tolist() == [2, 6]
    assert v.tolist() == [3, 7]


def test_interleave_index_roundtrips_weight():
    # Random fused weight; de-interleave then re-interleave must reproduce it.
    nah, ng, hd, in_f = 8, 2, 16, 32
    q_out, kv_out = ps.qkv_segment_out_dims(nah, ng, hd)
    total = q_out + 2 * kv_out
    W = torch.randn(total, in_f)
    qr, kr, vr = ps.qkv_deinterleave_row_indices(nah, ng, hd)
    cat = torch.cat([W[qr], W[kr], W[vr]], dim=0)  # de-interleaved
    idx = ps.qkv_interleave_index(nah, ng, hd)
    assert torch.equal(cat.index_select(0, idx), W)


def test_validate_divisible_raises_with_segment_name():
    import pytest

    with pytest.raises(ValueError, match="linear_k"):
        ps.validate_divisible("decoder.layers.0.self_attention", "linear_k",
                              in_f=1280, out_f=384, block_size=256, block_count=None)


def test_validate_divisible_ok():
    # 6144 and 1280 both divisible by 256 → no raise.
    ps.validate_divisible("attn", "linear_q", in_f=1280, out_f=6144,
                          block_size=256, block_count=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_poet_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.optim.poet_split'`.

- [ ] **Step 3: Create the module with the pure helpers**

Create `src/optim/poet_split.py`:

```python
"""Split fused parallel linears into separate POET sub-projections.

Runs *before* ``replace_linears_with_poet`` (inside the ``poet_apply_to_model``
wrapper). Splitting produces ordinary parallel-linear sub-modules, which the
existing POET walker then wraps with one independent orbit each.

Supported under POET's constraints only: TP=1, ``transformer_impl='local'``,
non-gated attention. MLA has no fused ``linear_qkv`` so ``split_qkv`` is inert.

This module is import-safe on CPU (no Megatron import at module load); the
Megatron linear types are discovered lazily.
"""

from __future__ import annotations

import copy
import logging
import types
from collections.abc import Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pure geometry helpers (CPU-testable, no Megatron dependency)
# --------------------------------------------------------------------------


def qkv_segment_out_dims(num_attention_heads: int, num_query_groups: int, head_dim: int):
    """Return ``(q_out, kv_out)``: the output dims of the Q projection and of
    *each* of the K / V projections in the fused ``linear_qkv``."""
    q_out = num_attention_heads * head_dim
    kv_out = num_query_groups * head_dim
    return q_out, kv_out


def qkv_deinterleave_row_indices(
    num_attention_heads: int, num_query_groups: int, head_dim: int
):
    """Row indices into the fused ``linear_qkv`` weight for Q, K, V.

    Megatron lays the fused output out group-major (``q1 q2 k1 v1 | q3 q4 k2 v2 | ...``):
    per group ``g`` the rows are ``[ q-heads (nqhpg*hd) | k (hd) | v (hd) ]``,
    where ``nqhpg = num_attention_heads // num_query_groups``.

    Returns ``(q_rows, k_rows, v_rows)`` as ``torch.long`` tensors.
    """
    ng = num_query_groups
    nqhpg = num_attention_heads // ng
    hd = head_dim
    stride = (nqhpg + 2) * hd
    q_rows: list[int] = []
    k_rows: list[int] = []
    v_rows: list[int] = []
    for g in range(ng):
        base = g * stride
        q_rows.extend(range(base, base + nqhpg * hd))
        k_rows.extend(range(base + nqhpg * hd, base + nqhpg * hd + hd))
        v_rows.extend(range(base + (nqhpg + 1) * hd, base + (nqhpg + 2) * hd))
    return (
        torch.tensor(q_rows, dtype=torch.long),
        torch.tensor(k_rows, dtype=torch.long),
        torch.tensor(v_rows, dtype=torch.long),
    )


def qkv_interleave_index(
    num_attention_heads: int, num_query_groups: int, head_dim: int
) -> torch.Tensor:
    """Index that maps the de-interleaved concat ``[q | k | v]`` (along the last
    dim) back to the fused interleaved ``mixed_qkv`` layout.

    With ``idx = qkv_interleave_index(...)`` and a concatenated tensor
    ``cat`` ordered ``[q_out, k_out, v_out]``,
    ``cat.index_select(-1, idx)`` reproduces the fused output.
    """
    q_rows, k_rows, v_rows = qkv_deinterleave_row_indices(
        num_attention_heads, num_query_groups, head_dim
    )
    len_q, len_k, len_v = q_rows.numel(), k_rows.numel(), v_rows.numel()
    total = len_q + len_k + len_v
    idx = torch.empty(total, dtype=torch.long)
    idx[q_rows] = torch.arange(0, len_q)
    idx[k_rows] = torch.arange(len_q, len_q + len_k)
    idx[v_rows] = torch.arange(len_q + len_k, total)
    return idx


def validate_divisible(
    module_path: str,
    seg_name: str,
    *,
    in_f: int,
    out_f: int,
    block_size: int,
    block_count: int | None,
) -> None:
    """Hard-error if a split sub-segment isn't POET-divisible.

    ``block_count`` (when set) takes precedence over ``block_size``, matching
    the unsplit walker's divisor precedence.
    """
    divisor = block_count if block_count is not None else block_size
    label = "block_count" if block_count is not None else "block_size"
    if in_f % divisor != 0 or out_f % divisor != 0:
        raise ValueError(
            f"[POET split] {module_path}.{seg_name} dims (in={in_f}, out={out_f}) "
            f"not divisible by {label}={divisor}. Choose a compatible "
            f"block_size/block_count, or disable splitting this layer."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_poet_split.py -v`
Expected: PASS (5 tests). **Wait for the user to confirm.**

- [ ] **Step 5: Commit**

```bash
git add src/optim/poet_split.py tests/unit/test_poet_split.py
git commit -m "feat(poet): pure de-interleave geometry helpers for fused-linear split"
```

---

## Task 5: FC1 (gate/up) surgery

**Files:**
- Modify: [src/optim/poet_split.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_split.py)
- Test: [tests/unit/test_poet_split.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_poet_split.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_poet_split.py`:

```python
class _FakeConfig:
    def __init__(self):
        self.gated_linear_unit = True
        self.activation_func = torch.nn.functional.silu
        self.activation_func_clamp_value = None
        self.glu_linear_offset = 0.0


class _FakeMLP(nn.Module):
    """Stand-in mimicking Megatron MLP's attributes used by the split path."""

    def __init__(self, hidden=8, ffn=16):
        super().__init__()
        self.config = _FakeConfig()
        self.linear_fc1 = nn.Linear(hidden, 2 * ffn, bias=False)  # [gate; up]
        self.linear_fc2 = nn.Linear(ffn, hidden, bias=False)


def test_split_fc1_creates_separate_modules_and_matches_fused():
    torch.manual_seed(0)
    m = _FakeMLP(hidden=8, ffn=16)
    x = torch.randn(3, 8)

    # Reference: fused forward (silu(gate) * up) -> fc2.
    fused = m.linear_fc1(x)
    gate_ref, up_ref = torch.chunk(fused, 2, dim=-1)
    ref = m.linear_fc2(torch.nn.functional.silu(gate_ref) * up_ref)

    n = ps.split_fused_linears(
        m, split_qkv=False, split_fc1=True,
        block_size=8, block_count=None, linear_types=(nn.Linear,),
    )
    assert n == 1
    assert hasattr(m, "linear_fc1_gate") and hasattr(m, "linear_fc1_up")
    assert not hasattr(m, "linear_fc1")
    assert isinstance(m.linear_fc1_gate, nn.Linear)

    out, out_bias = m.forward(x)
    assert torch.allclose(out, ref, atol=1e-6)


def test_split_fc1_hard_errors_on_indivisible_segment():
    import pytest

    m = _FakeMLP(hidden=8, ffn=20)  # ffn 20 not divisible by 8
    with pytest.raises(ValueError, match="linear_fc1_gate"):
        ps.split_fused_linears(
            m, split_qkv=False, split_fc1=True,
            block_size=8, block_count=None, linear_types=(nn.Linear,),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_poet_split.py::test_split_fc1_creates_separate_modules_and_matches_fused -v`
Expected: FAIL — `AttributeError: module 'src.optim.poet_split' has no attribute 'split_fused_linears'`.

- [ ] **Step 3: Implement the shared infra + FC1 surgery**

Append to `src/optim/poet_split.py`:

```python
# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _column_linear_types(extra: Iterable[type] = ()) -> tuple[type, ...]:
    """Megatron column-parallel linear types, discovered lazily (empty on CPU),
    unioned with any ``extra`` types (tests pass ``nn.Linear``)."""
    types_: tuple[type, ...] = ()
    try:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        types_ += (ColumnParallelLinear,)
    except Exception:
        pass
    try:
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

        types_ += (TEColumnParallelLinear,)
    except Exception:
        pass
    return types_ + tuple(extra)


def _split_linear_out(module, x):
    """Call a linear that may return a tensor (``nn.Linear``) or a
    ``(output, bias)`` tuple (Megatron / POET). Returns ``(output, bias)``."""
    r = module(x)
    if isinstance(r, tuple):
        return r[0], (r[1] if len(r) > 1 else None)
    return r, None


def _make_sub_linear(src: nn.Module, rows: torch.Tensor) -> nn.Module:
    """Build a typed copy of ``src`` whose weight/bias are the rows ``rows`` of
    ``src``. The copy keeps ``src``'s class and config so the unsplit POET
    walker recognises and wraps it; only weight/bias/shape are sliced.

    The copy's own ``forward`` is never used in production (it is replaced by a
    POET module), but it is correct for ``nn.Linear`` in tests.
    """
    # ProcessGroup-bearing attrs on a real Megatron linear are not
    # deepcopy-able; detach them across the copy and restore on both objects.
    # This is the highest-risk spot vs. the real Megatron build — validated by
    # the Task 9 GPU smoke. (No-op for nn.Linear in CPU tests.)
    _pg_attrs = ("tp_group", "process_group", "pg_collection", "explicit_expert_comm")
    _saved = {}
    for attr in _pg_attrs:
        if attr in getattr(src, "__dict__", {}):
            _saved[attr] = src.__dict__[attr]
            src.__dict__[attr] = None
    try:
        sub = copy.deepcopy(src)
    finally:
        for attr, val in _saved.items():
            src.__dict__[attr] = val
            sub.__dict__[attr] = val
    w = src.weight.data.index_select(0, rows.to(src.weight.device)).clone()
    sub.weight = nn.Parameter(w, requires_grad=src.weight.requires_grad)
    has_bias = getattr(src, "bias", None) is not None and src.bias.numel() > 0
    if has_bias:
        b = src.bias.data.index_select(0, rows.to(src.bias.device)).clone()
        sub.bias = nn.Parameter(b, requires_grad=src.bias.requires_grad)
    else:
        sub.bias = None
    out_f = rows.numel()
    # Best-effort fix of Megatron size bookkeeping (absent on nn.Linear).
    for attr in ("out_features", "output_size", "output_size_per_partition"):
        if hasattr(sub, attr):
            setattr(sub, attr, out_f)
    return sub


# --------------------------------------------------------------------------
# FC1 (gate / up) surgery
# --------------------------------------------------------------------------


def _split_mlp_forward(self, hidden_states, per_token_scale=None, **kwargs):
    """Replacement ``MLP.forward`` calling separate gate/up projections.

    Mirrors Megatron's non-fused gated ``glu()`` path
    (megatron/core/transformer/mlp.py). Does not use the fused
    ``bias_swiglu_impl`` kernel; numerically identical for SwiGLU.
    """
    gate, gate_bias = _split_linear_out(self.linear_fc1_gate, hidden_states)
    up, up_bias = _split_linear_out(self.linear_fc1_up, hidden_states)
    if gate_bias is not None:
        gate = gate + gate_bias
    if up_bias is not None:
        up = up + up_bias
    clamp = getattr(self.config, "activation_func_clamp_value", None)
    if clamp is not None:
        gate = gate.clamp(min=None, max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    offset = getattr(self.config, "glu_linear_offset", 0.0)
    intermediate = self.config.activation_func(gate) * (up + offset)
    if per_token_scale is not None:
        od = intermediate.dtype
        intermediate = (intermediate * per_token_scale.unsqueeze(-1)).to(od)
    out = self.linear_fc2(intermediate)
    if isinstance(out, tuple):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


def _split_one_mlp_fc1(mlp, path, *, block_size, block_count, linear_types) -> bool:
    fc1 = getattr(mlp, "linear_fc1", None)
    if fc1 is None or not isinstance(fc1, linear_types):
        return False
    if not getattr(mlp.config, "gated_linear_unit", False):
        raise ValueError(
            f"[POET split] {path}: --poet-split-fc1 requires a gated (SwiGLU) MLP."
        )
    out_f, in_f = fc1.weight.shape
    if out_f % 2 != 0:
        raise ValueError(f"[POET split] {path}.linear_fc1 out dim {out_f} is not even.")
    ffn = out_f // 2
    validate_divisible(path, "linear_fc1_gate", in_f=in_f, out_f=ffn,
                       block_size=block_size, block_count=block_count)
    validate_divisible(path, "linear_fc1_up", in_f=in_f, out_f=ffn,
                       block_size=block_size, block_count=block_count)
    gate_rows = torch.arange(0, ffn, dtype=torch.long)
    up_rows = torch.arange(ffn, 2 * ffn, dtype=torch.long)
    mlp.linear_fc1_gate = _make_sub_linear(fc1, gate_rows)
    mlp.linear_fc1_up = _make_sub_linear(fc1, up_rows)
    del mlp.linear_fc1
    mlp.forward = types.MethodType(_split_mlp_forward, mlp)
    logger.info("[POET split] %s.linear_fc1 -> gate/up (ffn=%d)", path, ffn)
    return True
```

- [ ] **Step 4: Add the top-level dispatcher (FC1-only for now)**

Append to `src/optim/poet_split.py`:

```python
# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def split_fused_linears(
    model: nn.Module,
    *,
    split_qkv: bool,
    split_fc1: bool,
    block_size: int,
    block_count: int | None,
    linear_types: Iterable[type] | None = None,
) -> int:
    """Split fused linears in-place; returns the number of fused linears split.

    ``linear_types`` overrides the recognised column-parallel linear classes
    (tests pass ``(nn.Linear,)``); defaults to Megatron's column-parallel types.
    """
    types_ = _column_linear_types() if linear_types is None else tuple(linear_types)
    n = 0
    for name, mod in list(model.named_modules()):
        if split_fc1 and hasattr(mod, "linear_fc1"):
            if _split_one_mlp_fc1(
                mod, name or "<root>", block_size=block_size,
                block_count=block_count, linear_types=types_,
            ):
                n += 1
    return n
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_poet_split.py -v`
Expected: PASS (Task 4 tests + the two FC1 tests). **Wait for the user to confirm.**

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_split.py tests/unit/test_poet_split.py
git commit -m "feat(poet): split fused SwiGLU linear_fc1 into separate gate/up modules"
```

---

## Task 6: QKV (Q/K/V) surgery

**Files:**
- Modify: [src/optim/poet_split.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_split.py)
- Test: [tests/unit/test_poet_split.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_poet_split.py)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_poet_split.py`:

```python
class _FakeAttnConfig:
    attention_output_gate = False


class _FakeAttention(nn.Module):
    """Stand-in mimicking Megatron SelfAttention attributes used by the split."""

    def __init__(self, hidden=32, num_heads=8, num_groups=2, head_dim=16):
        super().__init__()
        self.config = _FakeAttnConfig()
        self.world_size = 1
        self.hidden_size_per_attention_head = head_dim
        self.num_query_groups_per_partition = num_groups
        self.num_attention_heads_per_partition = num_heads
        self.q_layernorm = None
        self.k_layernorm = None
        q_out = num_heads * head_dim
        kv_out = num_groups * head_dim
        self.linear_qkv = nn.Linear(hidden, q_out + 2 * kv_out, bias=False)

    # Faithful reference of Megatron's TP=1 / no-gate get_query_key_value_tensors.
    def reference_qkv(self, hidden_states):
        mixed, _ = self.linear_qkv(hidden_states), None
        hd = self.hidden_size_per_attention_head
        ng = self.num_query_groups_per_partition
        nqhpg = self.num_attention_heads_per_partition // ng
        mixed = mixed.view(*mixed.size()[:-1], ng, (nqhpg + 2) * hd)
        query, key, value = torch.split(mixed, [nqhpg * hd, hd, hd], dim=-1)
        query = query.reshape(query.size(0), -1, hd)
        return query, key, value


def test_split_qkv_creates_modules_and_matches_reference():
    torch.manual_seed(0)
    a = _FakeAttention(hidden=32, num_heads=8, num_groups=2, head_dim=16)
    x = torch.randn(5, 32)  # [sq*b flattened-ok, hidden]; here treat as [N, hidden]
    q_ref, k_ref, v_ref = a.reference_qkv(x)

    n = ps.split_fused_linears(
        a, split_qkv=True, split_fc1=False,
        block_size=16, block_count=None, linear_types=(nn.Linear,),
    )
    assert n == 1
    assert hasattr(a, "linear_q") and hasattr(a, "linear_k") and hasattr(a, "linear_v")
    assert not hasattr(a, "linear_qkv")

    q, k, v = a.get_query_key_value_tensors(x)
    assert torch.allclose(q, q_ref, atol=1e-6)
    assert torch.allclose(k, k_ref, atol=1e-6)
    assert torch.allclose(v, v_ref, atol=1e-6)


def test_split_qkv_mqa_contiguous():
    torch.manual_seed(1)
    a = _FakeAttention(hidden=32, num_heads=4, num_groups=1, head_dim=16)
    x = torch.randn(5, 32)
    q_ref, k_ref, v_ref = a.reference_qkv(x)
    ps.split_fused_linears(
        a, split_qkv=True, split_fc1=False,
        block_size=16, block_count=None, linear_types=(nn.Linear,),
    )
    q, k, v = a.get_query_key_value_tensors(x)
    assert torch.allclose(q, q_ref, atol=1e-6)
    assert torch.allclose(k, k_ref, atol=1e-6)


def test_split_qkv_hard_errors_on_indivisible_segment():
    import pytest

    # hidden=64 and q_out=128 both divide 64; kv_out=32 does NOT, so the first
    # failing segment is linear_k. (With hidden==kv_out, no block size can fail
    # K while passing Q, hence hidden=64 here.)
    a = _FakeAttention(hidden=64, num_heads=8, num_groups=2, head_dim=16)
    with pytest.raises(ValueError, match="linear_k"):
        ps.split_fused_linears(
            a, split_qkv=True, split_fc1=False,
            block_size=64, block_count=None, linear_types=(nn.Linear,),
        )


def test_split_qkv_inert_without_linear_qkv():
    m = nn.Module()  # MLA-like: no linear_qkv
    n = ps.split_fused_linears(
        m, split_qkv=True, split_fc1=False,
        block_size=16, block_count=None, linear_types=(nn.Linear,),
    )
    assert n == 0
```

> Note on the reference: this `_FakeAttention.reference_qkv` reshapes `query` to `[N, np, hd]` for a 2-D input. The production patch and the stand-in both follow the same view/split/reshape, so they match. The patch handles the real `[sq, b, hidden]` 3-D shape identically (the leading dims are preserved by `view(*mixed.size()[:-1], ...)`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_poet_split.py::test_split_qkv_creates_modules_and_matches_reference -v`
Expected: FAIL — `split_fused_linears` does not yet handle qkv (no `linear_q` created).

- [ ] **Step 3: Implement the QKV surgery + forward patch**

Append to `src/optim/poet_split.py` (before `split_fused_linears`, or anywhere at module scope):

```python
# --------------------------------------------------------------------------
# QKV (Q / K / V) surgery
# --------------------------------------------------------------------------


def _split_qkv_forward(
    self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True
):
    """Replacement ``SelfAttention.get_query_key_value_tensors`` for TP=1,
    non-gated attention. Calls separate Q/K/V projections, reassembles the
    interleaved ``mixed_qkv`` (spec §6a), then runs Megatron's TP=1 post-linear
    view/split/reshape/layernorm so downstream attention math is bit-identical.
    """
    assert not output_gate, "[POET split] split_qkv does not support gated attention."
    q, _ = _split_linear_out(self.linear_q, hidden_states)
    k, _ = _split_linear_out(self.linear_k, hidden_states)
    v, _ = _split_linear_out(self.linear_v, hidden_states)
    mixed = torch.cat([q, k, v], dim=-1).index_select(-1, self._poet_qkv_interleave_index)

    hd = self.hidden_size_per_attention_head
    ng = self.num_query_groups_per_partition
    nqhpg = self.num_attention_heads_per_partition // ng
    mixed = mixed.view(*mixed.size()[:-1], ng, (nqhpg + 2) * hd)
    split_arg_list = [nqhpg * hd, hd, hd]
    if not split_qkv:
        return mixed, split_arg_list
    query, key, value = torch.split(mixed, split_arg_list, dim=-1)
    query = query.reshape(*query.size()[:-2], -1, hd)
    if self.q_layernorm is not None:
        query = self.q_layernorm(query)
    if self.k_layernorm is not None:
        key = self.k_layernorm(key)
    return query, key, value


def _split_backward_qkv_proj(self):
    """Replacement for SelfAttention._backward_qkv_proj after linear_qkv removal.

    Only relevant for Megatron's delayed-wgrad path (inactive for frozen-base
    POET); guarded so it never errors if invoked.
    """
    for attr in ("linear_q", "linear_k", "linear_v"):
        m = getattr(self, attr, None)
        if m is not None and hasattr(m, "backward_dw"):
            m.backward_dw()


def _split_one_attention_qkv(
    attn, path, *, block_size, block_count, linear_types
) -> bool:
    qkv = getattr(attn, "linear_qkv", None)
    if qkv is None or not isinstance(qkv, linear_types):
        return False
    if getattr(attn, "world_size", 1) != 1:
        raise ValueError(
            f"[POET split] {path}: --poet-split-qkv requires TP=1 (POET already enforces it)."
        )
    if getattr(attn.config, "attention_output_gate", False):
        raise ValueError(
            f"[POET split] {path}: --poet-split-qkv does not support gated attention."
        )

    hd = attn.hidden_size_per_attention_head
    ng = attn.num_query_groups_per_partition
    nah = attn.num_attention_heads_per_partition
    q_out, kv_out = qkv_segment_out_dims(nah, ng, hd)
    out_f, in_f = qkv.weight.shape
    if q_out + 2 * kv_out != out_f:
        raise ValueError(
            f"[POET split] {path}.linear_qkv out dim {out_f} != q+2kv "
            f"({q_out}+2*{kv_out}); unexpected layout (gated attention?)."
        )
    validate_divisible(path, "linear_q", in_f=in_f, out_f=q_out,
                       block_size=block_size, block_count=block_count)
    validate_divisible(path, "linear_k", in_f=in_f, out_f=kv_out,
                       block_size=block_size, block_count=block_count)
    validate_divisible(path, "linear_v", in_f=in_f, out_f=kv_out,
                       block_size=block_size, block_count=block_count)

    q_rows, k_rows, v_rows = qkv_deinterleave_row_indices(nah, ng, hd)
    attn.linear_q = _make_sub_linear(qkv, q_rows)
    attn.linear_k = _make_sub_linear(qkv, k_rows)
    attn.linear_v = _make_sub_linear(qkv, v_rows)
    attn.register_buffer(
        "_poet_qkv_interleave_index",
        # The model is already on its compute device when the split runs (get_model
        # calls .cuda() before returning), so pin the index there to avoid a
        # device mismatch in the forward's index_select.
        qkv_interleave_index(nah, ng, hd).to(qkv.weight.device),
        persistent=False,
    )
    del attn.linear_qkv
    attn.get_query_key_value_tensors = types.MethodType(_split_qkv_forward, attn)
    attn._backward_qkv_proj = types.MethodType(_split_backward_qkv_proj, attn)
    logger.info(
        "[POET split] %s.linear_qkv -> q/k/v (q=%d, kv=%d, groups=%d)",
        path, q_out, kv_out, ng,
    )
    return True
```

- [ ] **Step 4: Wire QKV into the dispatcher**

In `split_fused_linears`, add the qkv branch inside the loop, before the fc1 branch:

```python
    for name, mod in list(model.named_modules()):
        if split_qkv and hasattr(mod, "linear_qkv"):
            if _split_one_attention_qkv(
                mod, name or "<root>", block_size=block_size,
                block_count=block_count, linear_types=types_,
            ):
                n += 1
        if split_fc1 and hasattr(mod, "linear_fc1"):
            if _split_one_mlp_fc1(
                mod, name or "<root>", block_size=block_size,
                block_count=block_count, linear_types=types_,
            ):
                n += 1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_poet_split.py -v`
Expected: PASS (all tasks 4–6 tests). **Wait for the user to confirm.**

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_split.py tests/unit/test_poet_split.py
git commit -m "feat(poet): GQA-correct split of fused linear_qkv into separate q/k/v modules"
```

---

## Task 7: Combined split + independence sanity test

**Files:**
- Test: [tests/unit/test_poet_split.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_poet_split.py)

Confirms both splits compose on one module tree and that the split sub-linears are real, separately-named submodules (the prerequisite for the existing POET walker + merge to treat them as independent orbits).

- [ ] **Step 1: Write the test**

Append to `tests/unit/test_poet_split.py`:

```python
class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = _FakeAttention(hidden=32, num_heads=8, num_groups=2, head_dim=16)
        self.mlp = _FakeMLP(hidden=32, ffn=16)


def test_both_splits_compose_and_register_separate_submodules():
    block = _FakeBlock()
    n = ps.split_fused_linears(
        block, split_qkv=True, split_fc1=True,
        block_size=16, block_count=None, linear_types=(nn.Linear,),
    )
    assert n == 2
    names = dict(block.named_modules())
    assert "self_attention.linear_q" in names
    assert "self_attention.linear_k" in names
    assert "self_attention.linear_v" in names
    assert "mlp.linear_fc1_gate" in names
    assert "mlp.linear_fc1_up" in names
    assert "self_attention.linear_qkv" not in names
    assert "mlp.linear_fc1" not in names
    # The interleave index is a non-persistent buffer (not a trainable param).
    assert "_poet_qkv_interleave_index" not in dict(block.named_parameters())
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_poet_split.py::test_both_splits_compose_and_register_separate_submodules -v`
Expected: PASS. **Wait for the user to confirm.**

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_poet_split.py
git commit -m "test(poet): both fused splits compose into separate named submodules"
```

---

## Task 8: Wire the split into `poet_apply_to_model`

**Files:**
- Modify: [src/patches/poet_apply_to_model.py:27-58](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py#L27)
- Test: [tests/unit/test_patch_poet_apply.py](/lustre/fast/fast/zqiu/slm-research/tests/unit/test_patch_poet_apply.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_patch_poet_apply.py`:

```python
def test_wrapped_get_model_runs_split_before_poet(monkeypatch):
    """The wrapped get_model calls split_fused_linears (with the split flags)
    before replace_linears_with_poet."""
    importlib.import_module("src.patches.poet_apply_to_model")
    import src.optim.poet_layers as pl
    import src.optim.poet_split as ps
    from megatron.training import training as mt  # noqa: F401  (patched target module)

    calls = []

    class _Args:
        poet = True
        poet_block_size = 16
        poet_block_count = None
        poet_init_type = "none"
        poet_mup_alpha = 1.0
        poet_cache_mode = "none"
        poet_split_qkv = True
        poet_split_fc1 = True

    import megatron.training as mtt
    monkeypatch.setattr(mtt, "get_args", lambda: _Args(), raising=False)
    monkeypatch.setattr(
        ps, "split_fused_linears",
        lambda model, **kw: calls.append(("split", kw)) or 0,
    )
    monkeypatch.setattr(
        pl, "replace_linears_with_poet",
        lambda model, **kw: calls.append(("replace", kw)) or 0,
    )

    import src.patches._registry as reg_mod
    # Re-run the apply fn to install the wrapper over a stub get_model.
    import src.patches.poet_apply_to_model as mod
    from megatron.training import training as _mt
    monkeypatch.setattr(_mt, "get_model", lambda *a, **k: object(), raising=False)
    mod.apply()
    _mt.get_model("x")

    assert [c[0] for c in calls] == ["split", "replace"]
    assert calls[0][1]["split_qkv"] is True
    assert calls[0][1]["split_fc1"] is True
```

> If `megatron.training` is not importable in the test env, mark this test with `pytest.importorskip("megatron.training")` at the top of the function. The Task 4–7 tests are the authoritative correctness coverage; this test only checks wiring order.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_patch_poet_apply.py::test_wrapped_get_model_runs_split_before_poet -v`
Expected: FAIL — split not called / wrong order (current wrapper never calls `split_fused_linears`).

- [ ] **Step 3: Wire the split into the wrapper**

In `src/patches/poet_apply_to_model.py`, edit the `_wrapped` closure. Replace the body from the `chunks = ...` line through the `replace_linears_with_poet` loop:

```python
    def _wrapped(*a, **kw):
        model = _orig(*a, **kw)
        args = get_args()
        if not getattr(args, "poet", False):
            return model
        block = getattr(args, "poet_block_size", 256)
        block_count = getattr(args, "poet_block_count", None)
        init = getattr(args, "poet_init_type", "normalized")
        mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        cache_mode = getattr(args, "poet_cache_mode", "none")
        chunks = model if isinstance(model, list) else [model]

        split_qkv = getattr(args, "poet_split_qkv", False)
        split_fc1 = getattr(args, "poet_split_fc1", False)
        if split_qkv or split_fc1:
            from src.optim.poet_split import split_fused_linears

            for m in chunks:
                split_fused_linears(
                    m,
                    split_qkv=split_qkv,
                    split_fc1=split_fc1,
                    block_size=block,
                    block_count=block_count,
                )

        total = 0
        for m in chunks:
            total += replace_linears_with_poet(
                m,
                block_size=block,
                block_count=block_count,
                init_type=init,
                mup_alpha=mup_alpha,
                cache_mode=cache_mode,
            )
```

(Leave the trainable/frozen summary logging and `return model` below unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_patch_poet_apply.py -v`
Expected: PASS (existing registration tests + the new wiring test, or skipped if megatron unavailable). **Wait for the user to confirm.**

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_apply_to_model.py tests/unit/test_patch_poet_apply.py
git commit -m "feat(poet): run fused-linear split before POET wrapping in get_model"
```

---

## Task 9: Full suite + lint, and end-to-end smoke (user-run)

**Files:** none (verification only).

- [ ] **Step 1: Run the unit suite**

Run: `python -m pytest tests/unit/test_poet_split.py tests/unit/test_poet_layers.py tests/unit/test_megatron_args.py tests/unit/test_pretrain_gpt_slm.py tests/unit/test_patch_poet_apply.py -v`
Expected: all PASS. **Wait for the user to confirm.**

- [ ] **Step 2: Lint (pre-commit ruff)**

Run: `pre-commit run --files src/optim/poet_split.py src/patches/poet_apply_to_model.py src/utils/megatron_args.py launchers/pretrain_gpt_slm.py tests/unit/test_poet_split.py`
Expected: ruff / ruff-format PASS (fix any reported issues, re-run). **Wait for the user to confirm.**

- [ ] **Step 3: End-to-end smoke (user runs on a GPU node)**

The split path only fully exercises under a real Megatron build. Suggested smoke using the existing POET launcher with an MQA model (has a fused `linear_qkv`):

```bash
codexlog poet_split_smoke bash scripts/train_poet.sh llama3 \
  base.model.transformer_impl=local \
  experiment=optim/poet \
  optim.poet.split_qkv=true optim.poet.split_fc1=true \
  optim.poet.block_size=128 \
  training.train_iters=5 training.save_enabled=false
```

Expected: model builds, the `[POET split]` and `[POET] replaced N linears` log lines appear (N higher than the unsplit run since q/k/v/gate/up are separate), and 5 steps run without shape errors. Pick a `block_size` that divides every produced segment (else the intended hard error fires at startup — that is correct behavior). **The user runs this and reports back.**

- [ ] **Step 4: Update CHANGELOG**

Add an entry to the repo CHANGELOG describing the new `--poet-split-qkv` / `--poet-split-fc1` args and the `poet_split.py` module.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(poet): changelog for poet_split_qkv / poet_split_fc1"
```

---

## Self-review notes

- **Spec coverage:** args (T1–T3), de-interleave geometry §6a (T4), FC1 surgery §6b (T5), QKV surgery §6a (T6), divisibility hard error §6c (T4 validator + T5/T6 raise paths), integration ordering §5 (T8), inertness for MLA + TP=1/output-gate guards §7 (T6 tests + guards), tests §8 (T4–T7). All covered.
- **Type consistency:** `split_fused_linears(model, *, split_qkv, split_fc1, block_size, block_count, linear_types=None)`, `validate_divisible(path, seg_name, *, in_f, out_f, block_size, block_count)`, `_make_sub_linear(src, rows)`, `_split_linear_out(module, x)`, buffer `_poet_qkv_interleave_index`, submodule names `linear_q/k/v`, `linear_fc1_gate/up` — used identically across tasks.
- **CHANGELOG:** required per project convention (T9 step 4).
