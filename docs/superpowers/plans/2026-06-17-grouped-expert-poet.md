# Grouped-Expert POET Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let POET reparameterize the **grouped** MoE expert weight so the 64 experts run as one batched grouped-GEMM (fast) with per-expert rotations, instead of being forced onto the slow per-expert `SequentialMLP` path (or silently skipped under `grouped_gemm=true`).

**Architecture:** Add a `GroupedPOETLinear` (in `poet_torch`) that holds the experts' **3-D** base weight `W0 ∈ [E, out, in]` frozen plus per-expert block-diagonal rotations `oft_R_in/oft_R_out ∈ [E, n_blocks, n_elems]` trainable. Its forward materializes `W_eff[e] = R_out[e] @ W0[e] @ R_in[e]` for all experts in **one batched Cayley + one batched block-diagonal multiply**, then feeds the model's existing grouped-GEMM unchanged. POET's walk learns to detect the native `GroupedMLP` expert weights and swap them for this module; the periodic merge folds the per-expert rotation back into `W0` batched over the expert axis.

**Tech Stack:** PyTorch, vendored Megatron-LM (`third_party/Megatron-LM`), `poet_torch` (`third_party/poet_torch`), the `grouped_gemm` op used by native `GroupedMLP`, slm-research patch registry (`src/patches`), Hydra configs.

## Global Constraints

- **POET runs on the local transformer impl only** (it cannot wrap fused TE linears). Therefore the grouped path to support is the **native** `GroupedMLP` (`grouped_gemm` package), NOT `TEGroupedMLP`. Confirm in Task 1; if the local+`grouped_gemm=true` build instantiates a different class, adapt the wrap target there before writing downstream tasks.
- **bf16** params (FP8 is incompatible with POET weight-swap).
- **Rotations are block-diagonal**, sized by `block_count` (preferred) or `block_size`; per-expert `oft_R` must live in the DDP grad buffer (apply POET **pre-DDP**, mirroring [poet_apply_to_model.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py)).
- **Numerical parity is the gate:** a grouped wrap of E experts must be bit-comparable (≤ 1e-5 bf16, exact in fp32) to E independent `POETLinear`s on the same weights/rotations. No correctness regression vs the current SequentialMLP POET path.
- **No silent skips:** if an expert weight cannot be wrapped (dims not divisible by the block size), raise — never fall through to an un-rotated expert.
- Follow existing POET patterns: name trainable rotations `*oft_R*` (the optimizer per-group LR glob and the merge filter both key on that substring).

## Design decisions (assumed — confirm before Task 2; we skipped the brainstorming spec)

1. **Per-expert independent rotations** (each expert is its own linear → its own `R_in/R_out`). Cost: `oft_R` param count is E× a single expert's, i.e. the SAME total as today's SequentialMLP POET (one rotation per expert either way). *Alternative considered:* a single rotation shared across all experts — far fewer params but changes the method's semantics; rejected unless you want it.
2. **Natural-frame materialization** (`W_eff` rebuilt each forward via batched Cayley), NOT the forward-frame POETX (`single_step_x`) baking. Rationale: the win over SequentialMLP comes from batching the GEMM, and natural-frame is the simplest correct path; forward-frame grouped POETX is a follow-up once parity + speedup land.
3. **Target native `GroupedMLP` only** in this plan. `TEGroupedMLP` is out of scope (POET is on local impl).

If you reject (1) or (2), stop and revise this plan before implementing — they change Tasks 2–3 materially.

## File Structure

- Create `third_party/poet_torch/poet_torch/grouped_poet_layer.py` — `GroupedPOETLinear` + batched rotation helpers. One responsibility: the batched-over-experts reparam.
- Create `third_party/poet_torch/tests_poet/test_grouped_poet_layer.py` — CPU parity/unit tests for the above.
- Modify `src/optim/poet_layers.py` — detect grouped experts; install `GroupedPOETMegatronLinear` wrapper feeding the grouped GEMM.
- Create `tests/optim/test_grouped_expert_wrap.py` — wrap-walk tests with a fake grouped module.
- Modify `src/patches/poet_merge_step.py` — collect the grouped module; batched per-expert fold.
- Modify `src/patches/poet_apply_to_model.py` — only if the param dump / counters need the new module (likely just logging).
- Modify `src/utils/megatron_args.py` — emit a `--poet-wrap-grouped-experts` flag; allow `moe.grouped_gemm=true` under POET.
- Modify `configs/experiments/optim/poet_lie_orth_alt.yaml` (or a new sibling) — enable grouped experts.
- Create `docs/experiments/grouped_expert_poet.md` — short writeup (the experiment-YAML pre-commit hook requires a matching doc if a new experiment file is added).

---

### Task 1: Characterize the grouped-expert representation (discovery + acceptance harness)

**Files:**
- Create: `third_party/poet_torch/tests_poet/test_grouped_repr_characterization.py`
- Read only: `third_party/Megatron-LM/megatron/core/transformer/moe/experts.py` (`GroupedMLP`)

**Interfaces:**
- Produces (consumed by every later task — fill these in from the asserts once they pass):
  - `GROUPED_CLASS` — the module class instantiated for local impl + `grouped_gemm=true` (expected `GroupedMLP`).
  - `W1_SHAPE`, `W2_SHAPE` — exact `weight1`/`weight2` tensor shapes and axis meaning (e.g. is `weight1` `[in, E*fc1_out]` 2-D concatenated, or `[E, fc1_out, in]` 3-D?). **Everything downstream depends on this.**
  - `FORWARD_CONTRACT` — how the expert GEMM is invoked (the `grouped_gemm`/`gg.ops.gmm` call + the per-expert token `group_list`), and whether the gate/up split is fused in `weight1`.

- [ ] **Step 1: Write a characterization test that builds a tiny native GroupedMLP and asserts its weight layout**

```python
# test_grouped_repr_characterization.py
import torch, pytest
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs megatron/grouped_gemm")

def test_grouped_mlp_weight_layout():
    # Build the SMALLEST native GroupedMLP this Megatron supports (E=2 experts,
    # hidden=8, fc1=4) under local impl. Use the same submodule spec path the
    # model_provider takes when --moe-grouped-gemm is set.
    from megatron.core.transformer.moe.experts import GroupedMLP
    mlp, cfg = _build_tiny_grouped_mlp(num_local_experts=2, hidden=8, ffn=4)  # helper in this file
    # PIN the facts the rest of the plan consumes:
    print("weight1", tuple(mlp.weight1.shape), "weight2", tuple(mlp.weight2.shape))
    assert mlp.weight1.dim() in (2, 3)            # record which
    assert hasattr(mlp, "weight1") and hasattr(mlp, "weight2")
    # Record E, and whether gate+up are concatenated inside weight1 (swiglu).
```

- [ ] **Step 2: Run it; read the printed shapes; write them into the module docstring as the canonical contract**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest third_party/poet_torch/tests_poet/test_grouped_repr_characterization.py -s -v`
Expected: PASS, and the printed `weight1/weight2` shapes recorded. (If CUDA/grouped_gemm is unavailable on this node, run on an H100 node — this is the one task that needs the real Megatron build.)

- [ ] **Step 3: Add a forward-contract assertion**

```python
def test_grouped_mlp_forward_uses_group_list():
    mlp, cfg = _build_tiny_grouped_mlp(2, 8, 4)
    tokens = torch.randn(6, 8, device="cuda", dtype=torch.bfloat16)
    tokens_per_expert = torch.tensor([4, 2])          # group_list
    out = mlp(tokens, tokens_per_expert)              # confirm the call signature
    assert out[0].shape == (6, 8)                      # (output, bias) tuple, like ParallelLinear
```

- [ ] **Step 4: Run and record the forward signature**

Run: `... -m pytest .../test_grouped_repr_characterization.py::test_grouped_mlp_forward_uses_group_list -s -v`
Expected: PASS. Record the exact `(tokens, tokens_per_expert)` call + return type into the contract docstring.

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/tests_poet/test_grouped_repr_characterization.py
git commit -m "test(poet): characterize native GroupedMLP weight + forward contract for grouped POET"
```

---

### Task 2: Batched block-diagonal rotation core (`grouped_oft`)

Pure-torch, CPU-testable. Generalizes POET's per-linear Cayley to a leading expert axis.

**Files:**
- Create: `third_party/poet_torch/poet_torch/grouped_poet_layer.py`
- Test: `third_party/poet_torch/tests_poet/test_grouped_poet_layer.py`

**Interfaces:**
- Consumes: POET's existing `pytorch_skew_symmetric(oft, b, rows, cols)` and `cayley_batch` from [poet_torch.poet_layer](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_torch/poet_layer.py).
- Produces: `grouped_block_R(oft_R: Tensor[E, n_blocks, n_elems], block_size: int) -> Tensor[E, dim, dim]` returning a per-expert block-diagonal orthogonal matrix; `apply_grouped_R(W0: Tensor[E, out, in], R_out: Tensor[E,out,out], R_in: Tensor[E,in,in]) -> Tensor[E, out, in]`.

- [ ] **Step 1: Write the failing parity test (batched == per-expert loop)**

```python
# test_grouped_poet_layer.py
import torch
from poet_torch.grouped_poet_layer import grouped_block_R, apply_grouped_R
from poet_torch.poet_layer import pytorch_skew_symmetric, cayley_batch

def test_grouped_R_matches_per_expert_loop():
    torch.manual_seed(0)
    E, dim, b = 3, 8, 4                       # 2 blocks of 4
    n_blocks, n_elems = dim // b, b * (b - 1) // 2
    oft = torch.randn(E, n_blocks, n_elems, dtype=torch.float64)
    R = grouped_block_R(oft, b)               # [E, dim, dim]
    for e in range(E):                        # reference: one expert at a time
        skew = pytorch_skew_symmetric(oft[e], b, n_blocks, n_elems)
        R_ref = cayley_batch(skew)            # block-diagonal assemble
        assert torch.allclose(R[e], _assemble_block_diag(R_ref), atol=1e-10)
    # orthogonality
    eye = torch.eye(dim, dtype=torch.float64).expand(E, dim, dim)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-8)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest third_party/poet_torch/tests_poet/test_grouped_poet_layer.py::test_grouped_R_matches_per_expert_loop -v`
Expected: FAIL with `ModuleNotFoundError: grouped_poet_layer` / `grouped_block_R` undefined.

- [ ] **Step 3: Implement `grouped_block_R` + `apply_grouped_R`**

```python
# grouped_poet_layer.py
import torch
from poet_torch.poet_layer import pytorch_skew_symmetric

def grouped_block_R(oft_R, block_size):
    """oft_R: [E, n_blocks, n_elems] -> R: [E, dim, dim] block-diagonal orthogonal.
    Cayley acts independently per [b,b] block; flatten (E*n_blocks) into the batch
    so ONE cayley call covers every expert+block, then scatter into block-diagonal."""
    E, n_blocks, _ = oft_R.shape
    b = block_size
    skew = torch.stack([pytorch_skew_symmetric(oft_R[e], b, n_blocks, oft_R.shape[-1]) for e in range(E)])
    skew = skew.reshape(E * n_blocks, b, b)
    R_blocks = torch.ops.poet.cayley(skew)[0] if not skew.requires_grad else _cayley_torch(skew)
    R_blocks = R_blocks.reshape(E, n_blocks, b, b)
    dim = n_blocks * b
    R = oft_R.new_zeros(E, dim, dim)
    for k in range(n_blocks):                       # scatter blocks onto the diagonal
        R[:, k*b:(k+1)*b, k*b:(k+1)*b] = R_blocks[:, k]
    return R

def apply_grouped_R(W0, R_out, R_in):
    """W_eff[e] = R_out[e] @ W0[e] @ R_in[e], batched over experts."""
    return R_out @ W0 @ R_in
```

(`_cayley_torch` = the pure-torch Cayley already in poet_torch; use the Triton op on CUDA, torch fallback on CPU so the test runs CPU-only.)

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest .../test_grouped_poet_layer.py::test_grouped_R_matches_per_expert_loop -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/poet_torch/grouped_poet_layer.py third_party/poet_torch/tests_poet/test_grouped_poet_layer.py
git commit -m "feat(poet): batched block-diagonal rotation core for grouped experts"
```

---

### Task 3: `GroupedPOETLinear` module (frozen 3-D weight + per-expert oft_R + batched forward)

**Files:**
- Modify: `third_party/poet_torch/poet_torch/grouped_poet_layer.py`
- Test: `third_party/poet_torch/tests_poet/test_grouped_poet_layer.py`

**Interfaces:**
- Consumes: `grouped_block_R`, `apply_grouped_R` (Task 2); the W1/W2 layout pinned in Task 1.
- Produces: `class GroupedPOETLinear(nn.Module)` with `weight: Parameter[E,out,in]` (`requires_grad=False`), `oft_R_in/oft_R_out: Parameter[E,n_blocks,n_elems]` (`requires_grad=True`), `block_size: int`, `effective_weight() -> Tensor[E,out,in]`, and `merge_then_reinitialize(reinit_perm: bool)`.

- [ ] **Step 1: Write failing test — effective_weight equals E independent POETLinears**

```python
def test_grouped_poet_matches_independent_poet_linears():
    import torch
    from poet_torch import POETLinear
    from poet_torch.grouped_poet_layer import GroupedPOETLinear
    torch.manual_seed(0)
    E, out, inn, bc = 3, 8, 8, 2
    base = torch.randn(E, out, inn, dtype=torch.float64)
    g = GroupedPOETLinear(num_experts=E, in_features=inn, out_features=out, block_count=bc, dtype=torch.float64)
    g.weight.data.copy_(base)
    # mirror g's rotations into E standalone POETLinears
    refs = []
    for e in range(E):
        pl = POETLinear(in_features=inn, out_features=out, block_count=bc, dtype=torch.float64)
        pl.weight.data.copy_(base[e]); pl.oft_R_in.data.copy_(g.oft_R_in[e]); pl.oft_R_out.data.copy_(g.oft_R_out[e])
        refs.append(pl)
    W = g.effective_weight()                       # [E, out, in]
    for e in range(E):
        assert torch.allclose(W[e], refs[e].effective_weight(), atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest .../test_grouped_poet_layer.py::test_grouped_poet_matches_independent_poet_linears -v`
Expected: FAIL (`GroupedPOETLinear` undefined).

- [ ] **Step 3: Implement `GroupedPOETLinear`**

```python
class GroupedPOETLinear(nn.Module):
    def __init__(self, num_experts, in_features, out_features, block_count, dtype=None, device=None):
        super().__init__()
        self.E, self.in_features, self.out_features = num_experts, in_features, out_features
        self.block_count = block_count
        self.block_size_in = in_features // block_count
        self.block_size_out = out_features // block_count
        if in_features % block_count or out_features % block_count:
            raise ValueError(f"[POET] grouped expert dims ({out_features},{in_features}) not divisible by block_count={block_count}")
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features, dtype=dtype, device=device),
                                   requires_grad=False)
        n_in = self.block_size_in * (self.block_size_in - 1) // 2
        n_out = self.block_size_out * (self.block_size_out - 1) // 2
        self.oft_R_in = nn.Parameter(torch.zeros(num_experts, block_count, n_in, dtype=dtype, device=device))
        self.oft_R_out = nn.Parameter(torch.zeros(num_experts, block_count, n_out, dtype=dtype, device=device))

    def effective_weight(self):
        R_in = grouped_block_R(self.oft_R_in, self.block_size_in)     # [E,in,in]
        R_out = grouped_block_R(self.oft_R_out, self.block_size_out)  # [E,out,out]
        return apply_grouped_R(self.weight, R_out, R_in)

    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm=False):
        self.weight.data.copy_(self.effective_weight())
        self.oft_R_in.zero_(); self.oft_R_out.zero_()
```

(`oft_R` inits to zero → R = identity → `effective_weight == weight` at step 0, matching POETLinear's `init_type='normalized'` convention. Confirm against POETLinear's init in Task 2's reference.)

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest .../test_grouped_poet_layer.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/poet_torch/grouped_poet_layer.py third_party/poet_torch/tests_poet/test_grouped_poet_layer.py
git commit -m "feat(poet): GroupedPOETLinear — per-expert rotations over a batched 3-D weight"
```

---

### Task 4: Wire `GroupedPOETLinear` into the model walk (`replace_linears_with_poet`)

The wrapper must (a) hold the experts' base weight as the frozen 3-D `weight`, (b) expose `effective_weight()` to the grouped GEMM at forward, (c) keep the model's `GroupedMLP` forward otherwise intact.

**Files:**
- Modify: `src/optim/poet_layers.py` (the `_walk` in [replace_linears_with_poet](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L231))
- Test: `tests/optim/test_grouped_expert_wrap.py`

**Interfaces:**
- Consumes: `GroupedPOETLinear` (Task 3); `GROUPED_CLASS`/`W1_SHAPE` (Task 1).
- Produces: a `GroupedPOETMegatronLinear` adapter whose forward rebinds `weight1`/`weight2` to `effective_weight()` and calls the original grouped GEMM — analogous to how [POETMegatronLinear](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L65) wraps a parallel linear.

- [ ] **Step 1: Failing test — walking a fake GroupedMLP wraps its expert weights**

```python
# tests/optim/test_grouped_expert_wrap.py  (CPU; fake module mimics the Task-1 contract)
import torch, torch.nn as nn
from src.optim.poet_layers import replace_linears_with_poet

class _FakeGroupedMLP(nn.Module):                 # mirrors GROUPED_CLASS weight layout from Task 1
    def __init__(self, E=2, out=8, inn=8):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(E, out, inn))   # adjust to Task-1 W1_SHAPE
        self.weight2 = nn.Parameter(torch.randn(E, inn, out))

def test_walk_wraps_grouped_experts():
    m = _FakeGroupedMLP()
    n = replace_linears_with_poet(m, block_count=2, extra_grouped_types=(_FakeGroupedMLP,))
    from poet_torch.grouped_poet_layer import GroupedPOETLinear
    assert any(isinstance(mod, GroupedPOETLinear) for mod in m.modules())
    assert n >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/optim/test_grouped_expert_wrap.py -v`
Expected: FAIL (walk doesn't recognize grouped modules).

- [ ] **Step 3: Add grouped detection to `_walk`**

In `replace_linears_with_poet`, add an `extra_grouped_types` param (mirroring the existing `extra_linear_types`) and resolve the concrete grouped types the same way the linear types are resolved. Inside `_walk`, before the generic recurse, detect a grouped-expert module and convert each `weightN` Parameter into a `GroupedPOETLinear` (base weight copied in, frozen) plus an adapter that rebinds the materialized weight at forward. Raise (don't skip) on non-divisible dims. Keep the existing 2-D path untouched so SequentialMLP POET still works.

```python
# near the top of replace_linears_with_poet, mirroring linear_types:
grouped_types = _megatron_grouped_types() + tuple(extra_grouped_types)  # _megatron_grouped_types lazily returns (GroupedMLP,) like _megatron_linear_types
# inside _walk, alongside the linear_types branch:
if grouped_types and isinstance(child, grouped_types):
    replaced += _wrap_grouped_expert_module(child, block_count, block_size, ...)
    continue
```

(`_wrap_grouped_expert_module` builds one `GroupedPOETLinear` per `weightN`, sets the module to call `effective_weight()` in place of the raw weight. Exact rebinding mirrors `POETMegatronLinear.forward`; the precise hook depends on the Task-1 `FORWARD_CONTRACT`.)

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest tests/optim/test_grouped_expert_wrap.py -v`
Expected: PASS.

- [ ] **Step 5: Run the existing POET unit tests to confirm no regression on the 2-D path**

Run: `... -m pytest third_party/poet_torch/tests_poet -q -k "not cuda"`
Expected: PASS (pre-existing failures, if any, unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_layers.py tests/optim/test_grouped_expert_wrap.py
git commit -m "feat(poet): wrap native GroupedMLP experts with GroupedPOETLinear in the model walk"
```

---

### Task 5: Merge integration — batched per-expert fold

The periodic merge ([poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py)) must collect `GroupedPOETLinear` and fold per expert. `GroupedPOETLinear.merge_then_reinitialize` (Task 3) already does the math; this task makes the merge driver find and call it, and verifies the replicate (no-broadcast) path stays bit-identical across ranks.

**Files:**
- Modify: `src/patches/poet_merge_step.py` (the collection loop in [_run_merge](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L296) and `_merge_layers`)
- Test: `tests/patches/test_grouped_merge.py`

**Interfaces:**
- Consumes: `GroupedPOETLinear.merge_then_reinitialize` (Task 3).
- Produces: merge coverage for grouped modules (no new public symbol).

- [ ] **Step 1: Failing test — merge folds grouped rotation into the 3-D weight and zeros oft_R**

```python
# tests/patches/test_grouped_merge.py
import torch
from poet_torch.grouped_poet_layer import GroupedPOETLinear
from src.patches.poet_merge_step import _merge_layers

def test_grouped_merge_folds_and_zeros():
    torch.manual_seed(0)
    g = GroupedPOETLinear(num_experts=3, in_features=8, out_features=8, block_count=2, dtype=torch.float64)
    g.oft_R_in.data.normal_(std=0.1); g.oft_R_out.data.normal_(std=0.1)
    W_before = g.effective_weight().clone()
    _merge_layers([g], reinit_perm=False, disable_batch=True)
    assert torch.allclose(g.weight, W_before, atol=1e-9)        # folded
    assert g.oft_R_in.abs().max() == 0 and g.oft_R_out.abs().max() == 0  # reset
    assert torch.allclose(g.effective_weight(), W_before, atol=1e-9)     # invariant post-merge
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/patches/test_grouped_merge.py -v`
Expected: FAIL (`_merge_layers` doesn't handle `GroupedPOETLinear`).

- [ ] **Step 3: Extend the collection + fold to include grouped modules**

In `_run_merge`, the module loop currently keeps only `POETMegatronLinear` wrapping `POETLinear|POETXLinear`. Add `GroupedPOETLinear` to the collected `pls`. In `_merge_layers`, route grouped layers to their own `merge_then_reinitialize` (they aren't Cayley-batchable with the 2-D layers since the batch axis is experts, not layers — fold them in their own loop).

```python
# _run_merge collection:
if isinstance(mod, GroupedPOETLinear):
    pls.append(mod); continue
# _merge_layers:
grouped = [pl for pl in pls if isinstance(pl, GroupedPOETLinear)]
for g in grouped:
    g.merge_then_reinitialize(reinit_perm=reinit_perm)
rest = [pl for pl in pls if not isinstance(pl, GroupedPOETLinear)]   # existing path unchanged
```

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest tests/patches/test_grouped_merge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/patches/poet_merge_step.py tests/patches/test_grouped_merge.py
git commit -m "feat(poet): fold grouped-expert rotations in the periodic merge"
```

---

### Task 6: Optimizer / grad-buffer / per-group LR integration

Confirm per-expert `oft_R` lands in the DDP grad buffer (POET applied pre-DDP) and is picked up by the POET optimizer's `*oft_R*` per-group LR glob; confirm the merge's master-value zeroing covers grouped `oft_R` so it can't spring back (same hazard the 2-D path documents).

**Files:**
- Modify: `src/patches/poet_optimizer_setup.py` (only if the `oft_R` glob misses the new param names — likely no change), `src/patches/poet_merge_step.py` (the `_reset_vanilla_oft_state` `oft_R` filter already keys on the name substring — verify it matches `GroupedPOETLinear`'s param names).
- Test: `tests/patches/test_grouped_optimizer_integration.py`

**Interfaces:**
- Consumes: `GroupedPOETLinear` param names containing `oft_R`.
- Produces: none (integration assertions only).

- [ ] **Step 1: Failing test — grouped oft_R params are named so the `oft_R` glob/filter catches them**

```python
def test_grouped_oft_param_names():
    from poet_torch.grouped_poet_layer import GroupedPOETLinear
    g = GroupedPOETLinear(2, 8, 8, block_count=2)
    names = [n for n, _ in g.named_parameters()]
    assert any("oft_R_in" in n for n in names) and any("oft_R_out" in n for n in names)
    # the frozen base must NOT be trainable
    assert not dict(g.named_parameters())["weight"].requires_grad
```

- [ ] **Step 2: Run to verify it passes or fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/patches/test_grouped_optimizer_integration.py -v`
Expected: PASS (this is a guard test; if it fails, rename params to contain `oft_R`).

- [ ] **Step 3: Audit the two filters and add a regression assertion**

Read `_reset_vanilla_oft_state` ([poet_merge_step.py:163](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L163)) and the optimizer per-group glob in [poet_optimizer_setup.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py); add a test asserting a model containing a `GroupedPOETLinear` yields oft_R masters in the reset set. No production change expected — if the globs already match `*oft_R*`, the only artifact is the test.

- [ ] **Step 4: Commit**

```bash
git add tests/patches/test_grouped_optimizer_integration.py src/patches/poet_optimizer_setup.py src/patches/poet_merge_step.py
git commit -m "test(poet): grouped-expert oft_R is buffer/optimizer/merge-reset covered"
```

---

### Task 7: Config + flag plumbing, GPU smoke, throughput A/B

Expose the path end-to-end and prove the win: grouped POET must train (loss parity vs SequentialMLP POET over a few hundred steps) AND be materially faster.

**Files:**
- Modify: `src/utils/megatron_args.py` (emit `--poet-wrap-grouped-experts`; stop forcing `grouped_gemm=false` when POET + this flag are set)
- Modify: `configs/experiments/optim/poet_lie_orth_alt.yaml` OR create `configs/experiments/optim/poet_lie_orth_alt_grouped.yaml` (+ `docs/experiments/...md` for the pre-commit hook)
- Modify: `scripts/train_deepseek_poet.sh` (add a `BLOCK_COUNT`-style toggle, or document `optim.poet.wrap_grouped_experts=true`)
- Test: `tests/utils/test_megatron_args_grouped_poet.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: `--poet-wrap-grouped-experts` CLI flag; `optim.poet.wrap_grouped_experts` config key.

- [ ] **Step 1: Failing arg-build test (CPU)**

```python
# tests/utils/test_megatron_args_grouped_poet.py
from omegaconf import OmegaConf
from src.utils.megatron_args import build_megatron_args   # or the optimizer-args helper
def test_grouped_flag_emitted_and_grouped_gemm_allowed():
    cfg = _poet_deepseek_cfg(); cfg.optim.poet.wrap_grouped_experts = True
    args = build_megatron_args(cfg)
    assert "--poet-wrap-grouped-experts" in args
    assert "--moe-grouped-gemm" in args            # grouped GEMM now ON under POET
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/utils/test_megatron_args_grouped_poet.py -v`
Expected: FAIL.

- [ ] **Step 3: Emit the flag + allow grouped GEMM under POET**

Add the `optim.poet.wrap_grouped_experts` read in `megatron_args.py`, emit `--poet-wrap-grouped-experts`, and gate the `--moe-grouped-gemm` emission on it. Thread the flag into `_apply_poet_to_chunk` ([poet_apply_to_model.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py)) so `replace_linears_with_poet(..., grouped_types=...)` is enabled only when set.

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest tests/utils/test_megatron_args_grouped_poet.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py configs/experiments/optim/ docs/experiments/ scripts/train_deepseek_poet.sh tests/utils/test_megatron_args_grouped_poet.py
git commit -m "feat(poet): plumb --poet-wrap-grouped-experts (grouped GEMM under POET)"
```

- [ ] **Step 6: GPU smoke (hand to user — A100/H100 only)**

Provide the command; do NOT launch (cluster runs are the user's):
```bash
codexlog poet_grouped_smoke bash scripts/train_deepseek_poet.sh dev optim.poet.wrap_grouped_experts=true
```
Acceptance: builds, `[POET] replaced N linears` includes grouped experts, a few steps run with finite loss, no NaN.

- [ ] **Step 7: Throughput + loss A/B (hand to user)**

```bash
# grouped POET vs the current SequentialMLP POET, same recipe, ~300 steps
codexlog poet_grouped_full bash scripts/train_deepseek_poet.sh full optim.poet.wrap_grouped_experts=true
codexlog poet_seq_full     bash scripts/train_deepseek_poet.sh full
```
Acceptance: grouped run's TFLOP/s materially > the 4.2 baseline; lm-loss trajectory within noise of the SequentialMLP POET run over the same steps. Record both in `docs/experiments/grouped_expert_poet.md`.

---

## Self-Review notes (for the implementer)

- **Task 1 is load-bearing.** If `weight1`/`weight2` are 2-D concatenated (`[in, E*out]`) rather than 3-D `[E, out, in]`, reshape to `[E, out, in]` at wrap time in Task 4 and reshape back before the grouped GEMM — the rotation math (Tasks 2–3) is defined on `[E, out, in]` and must not change.
- **Parity is the gate at every layer:** Task 3 proves grouped == E independent POETLinears; Task 7 proves grouped POET ≈ SequentialMLP POET on real loss. If Task 7 loss diverges, suspect the gate/up fusion inside `weight1` (the two halves may need independent block alignment) — bisect by wrapping fc2 only first.
- **Open design choices** (per-expert vs shared rotation; natural vs forward frame) are fixed to per-expert / natural-frame here. Forward-frame grouped POETX (folding perms into the 3-D weight, à la `single_step_x`) is a deliberate follow-up, not in this plan.
- **No silent skips:** non-divisible expert dims raise in `GroupedPOETLinear.__init__` (Task 3) and at the wrap site (Task 4).
