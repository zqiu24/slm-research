# Grouped-POETX over Experts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove POET's per-expert rotation-backward tax on DeepSeek MoE by computing the POETX rotation gradient **block-sparse and batched over the expert axis**, while keeping POET on every routed expert.

**Architecture:** A `GroupedPOETXLinear` module owns `E` per-expert `POETXLinear` sub-instances (verified merge + per-expert `oft_R` params + perms) whose frozen weights alias one contiguous `[E,out,in]` buffer. Forward/backward run through a single `GroupedPOETXFunction` that replaces the `2·E` independent per-expert backward calls with two batched-block `bmm`s (block-sparse `M` over `experts × blocks`). The POET walk detects `SequentialMLP`, installs one grouped module per linear role, and swaps `SequentialMLP.forward`. Merge and optimizer are unchanged — `oft_R` stays as `E` separate 2-D params.

**Tech Stack:** PyTorch (custom `autograd.Function`, `bmm`/`gather`), `poet_torch` (vendored, `third_party/poet_torch`), vendored Megatron-LM (`third_party/Megatron-LM`, core_v0.17.0), slm-research POET patch registry (`src/patches`, `src/optim`), Hydra configs.

Design spec: [2026-06-17-grouped-poetx-over-experts-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-17-grouped-poetx-over-experts-design.md).

## Global Constraints

- **Test interpreter:** run all pytest with `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest` (Python 3.12). The harness default `python` is 3.10 and lacks torch/omegaconf.
- **CPU-only for Tasks 1–6.** Every new numerical unit (helper, Function, module, merge) is pure-torch and tested without CUDA/megatron. Do NOT import megatron at module load in any new file.
- **POETX champion path only.** Forward-frame `single_step_x` + `lie_alternating` both-momenta (`POETXLinear(alternating=True)`), `merge_period=1`, `oft_R≡0` regime. Natural-frame `POETLinear` is out of scope.
- **`oft_R` stays E separate 2-D params** `[n_blocks, n_elems]` — the optimizer ([poet_lie_orth.py:150](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L150)) assumes 2-D and already batches across params; only the frozen weight is stacked 3-D. Param names MUST contain `oft_R`.
- **No silent skips / no silent wrong:** non-divisible dims raise; per-expert bias raises (bias support is a follow-up); fp8/fp4 expert path raises (target bf16).
- **Parity is the gate:** the grouped path must be bit-comparable to `E` independent `POETXLinear`s — forward, `grad_oft_R`, and post-merge weight. fp64/fp32 exact; bf16 ≤ 1e-5.
- **Build is profile-gated.** Before the GPU A/B (Task 7), the grad-accum=1 `torch.profiler` re-run must confirm the expert `M`/GEMM ops dominate forward/backward. CPU Tasks 1–6 can proceed in parallel; do not claim the speedup until measured.
- **Commit style:** one short conventional-commit line per task (`feat(poet):` / `test(poet):`), anonymous, no AI attribution.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `third_party/poet_torch/poet_torch/grouped_poetx_ops.py` | `_grouped_blockdiag_skew_vecs` (block-sparse batched M) + `GroupedPOETXFunction` | **Create** |
| `third_party/poet_torch/poet_torch/grouped_poetx_layer.py` | `GroupedPOETXLinear` (E POETXLinear holders + 3-D weight buffer + batched forward + merge delegation) | **Create** |
| `third_party/poet_torch/tests_poet/test_grouped_poetx.py` | CPU parity: helper, Function, module (fwd/bwd/merge) | **Create** |
| `src/optim/poet_layers.py` | `SequentialMLP` detection + grouped install + `SequentialMLP.forward` swap; `group_experts` param | **Modify** |
| `tests/optim/test_grouped_expert_wrap.py` | walk wraps a fake SequentialMLP; forward parity; 2-D path untouched | **Create** |
| `src/patches/poet_merge_step.py` | collect + fold `GroupedPOETXLinear` in `_run_merge` | **Modify** |
| `tests/patches/test_grouped_poetx_merge.py` | grouped fold correctness (CPU) | **Create** |
| `tests/unit/test_lie_ortho_oft_shape.py` | guard: optimizer assumes 2-D `oft_R` | **Create** |
| `src/utils/megatron_args.py` | emit `--poet-group-experts`; thread `optim.poet.group_experts` | **Modify** |
| `src/patches/poet_apply_to_model.py` | thread `group_experts` into `replace_linears_with_poet` | **Modify** |
| `configs/experiments/optim/poet_lie_orth_alt_grouped.yaml` + `docs/experiments/...md` | enable + document | **Create** |

---

## Task 1: Guard test — optimizer assumes 2-D `oft_R`

Pins the load-bearing decision (keep `oft_R` as E separate 2-D params). No production code — a characterization test so a future stacked-param refactor can't silently break `lie_ortho`.

**Files:**
- Create: `tests/unit/test_lie_ortho_oft_shape.py`

**Interfaces:**
- Consumes: [`LieOrthMomentum`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L27), `block_size_from_nelems`/`vec_to_skew` from `src.diag.skew_conditioning`.
- Produces: none.

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_lie_ortho_oft_shape.py
"""Guard: LieOrthMomentum assumes each oft_R param is 2-D (n_blocks, n_elems) and
batches across params. GroupedPOETXLinear must therefore keep E separate 2-D oft_R
params, never a stacked 3-D param. If this test breaks, the grouped design's
optimizer-compatibility assumption broke with it."""
import torch

from src.optim.poet_lie_orth import LieOrthMomentum


def test_lie_ortho_steps_2d_oft_and_rejects_3d():
    # block_size 4 -> n_elems = 4*3/2 = 6 ; two blocks
    p2d = torch.nn.Parameter(torch.zeros(2, 6))           # (n_blocks, n_elems)
    p2d.grad = torch.randn(2, 6)
    opt = LieOrthMomentum(
        [{"params": [p2d], "use_skew": True, "side": "in", "lr": 1e-2}],
        ortho_c=8, ortho_method="muon", ortho_ns_steps=5,
    )
    opt.step()                                            # 2-D path works
    assert torch.isfinite(p2d).all() and p2d.abs().sum() > 0

    # A stacked 3-D oft_R would mis-read n_elems from the wrong axis -> the optimizer
    # cannot consume it. We assert the 2-D contract here so the grouped module honors it.
    p3d = torch.nn.Parameter(torch.zeros(3, 2, 6))        # (E, n_blocks, n_elems)
    p3d.grad = torch.randn(3, 2, 6)
    opt3 = LieOrthMomentum(
        [{"params": [p3d], "use_skew": True, "side": "in", "lr": 1e-2}],
        ortho_c=8, ortho_method="muon", ortho_ns_steps=5,
    )
    with __import__("pytest").raises(Exception):
        opt3.step()                                       # 3-D oft_R is unsupported
```

- [ ] **Step 2: Run it**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_lie_ortho_oft_shape.py -v`
Expected: PASS. (If the 3-D case does NOT raise, the optimizer silently mis-handles it — investigate before relying on the §8 resolution.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_lie_ortho_oft_shape.py
git commit -m "test(poet): pin LieOrthMomentum 2-D oft_R contract for grouped experts"
```

---

## Task 2: Block-sparse, expert-batched rotation gradient

The core FLOP+launch win. Pure torch, CPU. Computes only the block-diagonal of the conjugated `M`, batched over `(experts × blocks)`.

**Files:**
- Create: `third_party/poet_torch/poet_torch/grouped_poetx_ops.py`
- Test: `third_party/poet_torch/tests_poet/test_grouped_poetx.py`

**Interfaces:**
- Consumes: the per-expert reference `_conj` + `_blockdiag_skew_vec` from [poetx_ops.py](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_ops.py#L24) / [single_step.py](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/single_step.py#L32).
- Produces: `_grouped_blockdiag_skew_vecs(G[E,in,out], Wx[E,out,in], perm_in_inv[E,in], perm_out_inv[E,out], bs_in, bs_out, rows_in, cols_in, rows_out, cols_out) -> (grad_in[E,nb_in,ne_in], grad_out[E,nb_out,ne_out])`.

- [ ] **Step 1: Write the failing parity test**

```python
# third_party/poet_torch/tests_poet/test_grouped_poetx.py
import torch

from poet_torch.poetx_ops import _conj
from poet_torch.single_step import _blockdiag_skew_vec
from poet_torch.grouped_poetx_ops import _grouped_blockdiag_skew_vecs


def _ref_one(G, Wx, pin, pout, bs_in, bs_out, ri, ci, ro, co):
    M_in = _conj(G @ Wx, pin)
    M_out = _conj(Wx @ G, pout)
    return (_blockdiag_skew_vec(M_in, bs_in, ri, ci),
            _blockdiag_skew_vec(M_out, bs_out, ro, co))


def test_grouped_blockdiag_matches_per_expert():
    torch.manual_seed(0)
    E, in_f, out_f, b = 4, 8, 8, 4
    nb_in, nb_out = in_f // b, out_f // b
    ri, ci = torch.triu_indices(b, b, 1)
    ro, co = torch.triu_indices(b, b, 1)
    G = torch.randn(E, in_f, out_f, dtype=torch.float64)
    Wx = torch.randn(E, out_f, in_f, dtype=torch.float64)
    pin = torch.stack([torch.randperm(in_f) for _ in range(E)])
    pout = torch.stack([torch.randperm(out_f) for _ in range(E)])

    g_in, g_out = _grouped_blockdiag_skew_vecs(G, Wx, pin, pout, b, b, ri, ci, ro, co)
    for e in range(E):
        r_in, r_out = _ref_one(G[e], Wx[e], pin[e], pout[e], b, b, ri, ci, ro, co)
        assert torch.allclose(g_in[e], r_in, atol=1e-10)
        assert torch.allclose(g_out[e], r_out, atol=1e-10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest third_party/poet_torch/tests_poet/test_grouped_poetx.py::test_grouped_blockdiag_matches_per_expert -v`
Expected: FAIL — `ModuleNotFoundError: poet_torch.grouped_poetx_ops`.

- [ ] **Step 3: Implement the helper**

```python
# third_party/poet_torch/poet_torch/grouped_poetx_ops.py
"""Block-sparse, expert-batched POETX rotation gradients (forward-frame, oft_R=0).

Replaces the per-expert pair of full [d,d] M GEMMs in POETXSingleStepFunction.backward
with two batched-block bmms: only the block-diagonal of the conjugated M is computed,
batched over (experts x blocks). Bit-identical (same summation order over the contracted
index) to per-expert _blockdiag_skew_vec(_conj(...)). CPU-safe: no megatron, no CUDA-only
ops; pure torch."""
from __future__ import annotations

import torch


def _grouped_blockdiag_skew_vecs(G, Wx, perm_in_inv, perm_out_inv,
                                 bs_in, bs_out, rows_in, cols_in, rows_out, cols_out):
    E, in_f, out_f = G.shape
    nb_in, nb_out = in_f // bs_in, out_f // bs_out
    ri, ci = rows_in.long(), cols_in.long()
    ro, co = rows_out.long(), cols_out.long()

    # ---- M_in: block-diagonal blocks of (G @ Wx)[pin][:, pin] ----
    pin = perm_in_inv.long()                                            # [E, in]
    G_sel = torch.gather(G, 1, pin.unsqueeze(-1).expand(E, in_f, out_f))
    G_sel = G_sel.reshape(E * nb_in, bs_in, out_f)                      # [E*nb, b, out]
    W_sel = torch.gather(Wx, 2, pin.unsqueeze(1).expand(E, out_f, in_f))
    W_sel = (W_sel.reshape(E, out_f, nb_in, bs_in)
                  .permute(0, 2, 1, 3).reshape(E * nb_in, out_f, bs_in))  # [E*nb, out, b]
    M_in = torch.bmm(G_sel, W_sel)                                     # [E*nb, b, b]
    skew_in = M_in - M_in.transpose(-1, -2)
    grad_in = (2.0 * skew_in[:, ri, ci]).reshape(E, nb_in, -1).to(Wx.dtype)

    # ---- M_out: block-diagonal blocks of (Wx @ G)[pout][:, pout] ----
    pout = perm_out_inv.long()                                         # [E, out]
    W2 = torch.gather(Wx, 1, pout.unsqueeze(-1).expand(E, out_f, in_f))
    W2 = W2.reshape(E * nb_out, bs_out, in_f)                          # [E*nb, b, in]
    G2 = torch.gather(G, 2, pout.unsqueeze(1).expand(E, in_f, out_f))
    G2 = (G2.reshape(E, in_f, nb_out, bs_out)
            .permute(0, 2, 1, 3).reshape(E * nb_out, in_f, bs_out))    # [E*nb, in, b]
    M_out = torch.bmm(W2, G2)                                          # [E*nb, b, b]
    skew_out = M_out - M_out.transpose(-1, -2)
    grad_out = (2.0 * skew_out[:, ro, co]).reshape(E, nb_out, -1).to(Wx.dtype)
    return grad_in, grad_out
```

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest third_party/poet_torch/tests_poet/test_grouped_poetx.py::test_grouped_blockdiag_matches_per_expert -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/poet_torch/grouped_poetx_ops.py third_party/poet_torch/tests_poet/test_grouped_poetx.py
git commit -m "feat(poet): block-sparse expert-batched POETX rotation gradient"
```

---

## Task 3: `GroupedPOETXFunction` (batched forward + backward)

Wraps Task 2 in one `autograd.Function` spanning all experts for a single linear role.

**Files:**
- Modify: `third_party/poet_torch/poet_torch/grouped_poetx_ops.py`
- Test: `third_party/poet_torch/tests_poet/test_grouped_poetx.py`

**Interfaces:**
- Consumes: `_grouped_blockdiag_skew_vecs` (Task 2); the per-expert [`POETXSingleStepFunction`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_ops.py#L29) for parity.
- Produces: `GroupedPOETXFunction.apply(concat_x, oft_in[E,nb_in,ne_in], oft_out[E,nb_out,ne_out], Wx[E,out,in], perm_in_inv[E,in], perm_out_inv[E,out], rows_in, cols_in, rows_out, cols_out, bs_in, bs_out, sizes:tuple) -> concat_y`. `oft_in/oft_out` are inputs only (forward ignores their values; backward returns their grads).

- [ ] **Step 1: Write the failing parity test**

```python
def test_grouped_function_matches_per_expert_poetx():
    import torch
    from poet_torch.poetx_ops import POETXSingleStepFunction
    from poet_torch.grouped_poetx_ops import GroupedPOETXFunction

    torch.manual_seed(0)
    E, in_f, out_f, b = 3, 8, 8, 4
    ri, ci = torch.triu_indices(b, b, 1).to(torch.int32)
    ro, co = torch.triu_indices(b, b, 1).to(torch.int32)
    sizes = (2, 3, 4)
    Wx = torch.randn(E, out_f, in_f, dtype=torch.float64)
    pin = torch.stack([torch.randperm(in_f) for _ in range(E)]).to(torch.int32)
    pout = torch.stack([torch.randperm(out_f) for _ in range(E)]).to(torch.int32)

    # per-expert reference
    ref_y, ref_gin, ref_gout, xs = [], [], [], []
    for e in range(E):
        x = torch.randn(sizes[e], in_f, dtype=torch.float64, requires_grad=True)
        oin = torch.zeros(in_f // b, b * (b - 1) // 2, dtype=torch.float64, requires_grad=True)
        oout = torch.zeros(out_f // b, b * (b - 1) // 2, dtype=torch.float64, requires_grad=True)
        y = POETXSingleStepFunction.apply(x, oin, oout, Wx[e], None, pin[e], pout[e],
                                          ri, ci, ro, co, b, b)
        y.sum().backward()
        ref_y.append(y.detach()); ref_gin.append(oin.grad); ref_gout.append(oout.grad)
        xs.append(x.detach())

    # grouped
    cx = torch.cat(xs, 0).requires_grad_(True)
    oin = torch.zeros(E, in_f // b, b * (b - 1) // 2, dtype=torch.float64, requires_grad=True)
    oout = torch.zeros(E, out_f // b, b * (b - 1) // 2, dtype=torch.float64, requires_grad=True)
    gy = GroupedPOETXFunction.apply(cx, oin, oout, Wx, pin, pout, ri, ci, ro, co, b, b, sizes)
    gy.sum().backward()

    assert torch.allclose(gy.detach(), torch.cat(ref_y, 0), atol=1e-10)
    for e in range(E):
        assert torch.allclose(oin.grad[e], ref_gin[e], atol=1e-10)
        assert torch.allclose(oout.grad[e], ref_gout[e], atol=1e-10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest third_party/poet_torch/tests_poet/test_grouped_poetx.py::test_grouped_function_matches_per_expert_poetx -v`
Expected: FAIL — `GroupedPOETXFunction` undefined.

- [ ] **Step 3: Implement the Function (append to `grouped_poetx_ops.py`)**

```python
class GroupedPOETXFunction(torch.autograd.Function):
    """Forward-frame, all-experts POETX. Forward: ragged per-expert bare GEMM. Backward:
    plain grad_x + Adam-equivalent G, then the block-sparse expert-batched rotation grad.
    Bias is unsupported (experts are bias-free); pass bias-free weights only."""

    @staticmethod
    def forward(ctx, concat_x, oft_in, oft_out, Wx,
                perm_in_inv, perm_out_inv, rows_in, cols_in, rows_out, cols_out,
                bs_in, bs_out, sizes):
        E = len(sizes)
        x_list = torch.split(concat_x, list(sizes), dim=0)
        y = torch.cat([x_list[e] @ Wx[e].t() for e in range(E)], dim=0)
        ctx.save_for_backward(concat_x, Wx, perm_in_inv, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.sizes = tuple(sizes)
        ctx.bs_in, ctx.bs_out = bs_in, bs_out
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (concat_x, Wx, perm_in_inv, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        sizes, bs_in, bs_out = ctx.sizes, ctx.bs_in, ctx.bs_out
        E, out_f, in_f = Wx.shape
        x_list = torch.split(concat_x, list(sizes), dim=0)
        gy_list = torch.split(grad_y, list(sizes), dim=0)
        grad_x = torch.cat([gy_list[e] @ Wx[e] for e in range(E)], dim=0)
        G = torch.stack([
            x_list[e].reshape(-1, in_f).t() @ gy_list[e].reshape(-1, out_f)
            for e in range(E)
        ])                                                            # [E, in, out]
        grad_in, grad_out = _grouped_blockdiag_skew_vecs(
            G, Wx, perm_in_inv, perm_out_inv, bs_in, bs_out,
            rows_in, cols_in, rows_out, cols_out)
        # 13 inputs -> 13 returns (grads for concat_x/oft_in/oft_out, then 10 None).
        return (grad_x, grad_in, grad_out,
                None, None, None, None, None, None, None, None, None, None)
```

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest third_party/poet_torch/tests_poet/test_grouped_poetx.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/poet_torch/grouped_poetx_ops.py third_party/poet_torch/tests_poet/test_grouped_poetx.py
git commit -m "feat(poet): GroupedPOETXFunction — batched all-experts POETX forward/backward"
```

---

## Task 4: `GroupedPOETXLinear` module

Owns `E` `POETXLinear` sub-instances (verified merge + per-expert `oft_R` params + perms) aliased to one `[E,out,in]` weight buffer; batched forward via Task 3; merge delegates per expert.

**Files:**
- Create: `third_party/poet_torch/poet_torch/grouped_poetx_layer.py`
- Test: `third_party/poet_torch/tests_poet/test_grouped_poetx.py`

**Interfaces:**
- Consumes: [`POETXLinear`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_layer.py#L22), `GroupedPOETXFunction` (Task 3).
- Produces: `GroupedPOETXLinear(num_experts, in_features, out_features, *, block_count, alternating, alternate_every, device, dtype)` with `.experts: ModuleList[POETXLinear]`, buffer `.weight[E,out,in]`, `.bind_weights()`, `.forward(concat_x, tokens_per_expert)`, `.effective_weight()`, `.merge_then_reinitialize(reinit_perm)`, `._fold_active_side(active, reinit_perm)`, `.alternating: bool`.

- [ ] **Step 1: Write the failing module parity test (forward + backward + merge)**

```python
def test_grouped_module_matches_independent_poetx_linears():
    import torch
    from poet_torch import POETXLinear
    from poet_torch.grouped_poetx_layer import GroupedPOETXLinear

    torch.manual_seed(0)
    E, in_f, out_f, bc = 3, 8, 8, 2          # block_count=2 -> block_size 4
    sizes = (2, 3, 4)

    # Build E reference POETXLinears with known weights + perms.
    refs = []
    for e in range(E):
        pl = POETXLinear(in_features=in_f, out_features=out_f, block_count=bc,
                         bias=False, dtype=torch.float64, alternating=True)
        pl.weight.data.copy_(torch.randn(out_f, in_f, dtype=torch.float64))
        pl.bake_perms_into_weight()
        refs.append(pl)

    g = GroupedPOETXLinear(E, in_f, out_f, block_count=bc, alternating=True,
                           alternate_every=1, dtype=torch.float64)
    # mirror each ref into the grouped module's experts, then bind the buffer.
    for e in range(E):
        g.experts[e].weight.data.copy_(refs[e].weight)
        for buf in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(g.experts[e], buf).copy_(getattr(refs[e], buf))
    g.bind_weights()

    # set nonzero oft_R identically on both sides
    for e in range(E):
        gi = torch.randn_like(g.experts[e].oft_R_in) * 0.1
        go = torch.randn_like(g.experts[e].oft_R_out) * 0.1
        g.experts[e].oft_R_in.data.copy_(gi); g.experts[e].oft_R_out.data.copy_(go)
        refs[e].oft_R_in.data.copy_(gi); refs[e].oft_R_out.data.copy_(go)

    # forward + backward parity
    xs = [torch.randn(sizes[e], in_f, dtype=torch.float64) for e in range(E)]
    ref_y = [refs[e](xs[e].clone().requires_grad_(True)) for e in range(E)]
    cx = torch.cat([x.clone() for x in xs], 0).requires_grad_(True)
    gy = g(cx, torch.tensor(sizes))
    assert torch.allclose(gy, torch.cat([y.detach() for y in ref_y], 0), atol=1e-9)

    gy.sum().backward()
    for e in range(E):
        x = xs[e].clone().requires_grad_(True)
        refs[e].oft_R_in.grad = None; refs[e].oft_R_out.grad = None
        refs[e](x).sum().backward()
        assert torch.allclose(g.experts[e].oft_R_in.grad, refs[e].oft_R_in.grad, atol=1e-9)
        assert torch.allclose(g.experts[e].oft_R_out.grad, refs[e].oft_R_out.grad, atol=1e-9)

    # merge parity (active-only fold; alternating=True)
    from poet_torch.alt_state import active_side
    active = active_side(1)
    w_before = g.weight.clone()
    g._fold_active_side(active, reinit_perm=False)
    for e in range(E):
        refs[e]._fold_active_side(active, reinit_perm=False)
        assert torch.allclose(g.weight[e], refs[e].weight, atol=1e-9)
    assert not torch.allclose(g.weight, w_before)        # something actually folded
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest third_party/poet_torch/tests_poet/test_grouped_poetx.py::test_grouped_module_matches_independent_poetx_linears -v`
Expected: FAIL — `grouped_poetx_layer` undefined.

- [ ] **Step 3: Implement the module**

```python
# third_party/poet_torch/poet_torch/grouped_poetx_layer.py
"""GroupedPOETXLinear: E experts' POETX rotation batched over the expert axis.

Owns E POETXLinear sub-instances (each holding its own 2-D oft_R params + perms + the
verified merge methods); their frozen weights alias rows of one contiguous [E,out,in]
buffer. Forward/backward go through the batched GroupedPOETXFunction (the 99.5% path);
merge delegates to the sub-instances unchanged (the 2.6% path). oft_R stays E separate
2-D params so LieOrthMomentum + the merge driver are untouched."""
from __future__ import annotations

import torch
import torch.nn as nn

from poet_torch import POETXLinear
from poet_torch.grouped_poetx_ops import GroupedPOETXFunction


class GroupedPOETXLinear(nn.Module):
    def __init__(self, num_experts, in_features, out_features, *,
                 block_count, alternating, alternate_every, device=None, dtype=None):
        super().__init__()
        self.E = int(num_experts)
        self.in_features, self.out_features = in_features, out_features
        self.experts = nn.ModuleList([
            POETXLinear(in_features=in_features, out_features=out_features,
                        block_count=block_count, bias=False, device=device, dtype=dtype,
                        parameterization="cayley",
                        alternating=alternating, alternate_every=alternate_every)
            for _ in range(self.E)
        ])
        e0 = self.experts[0]
        self.alternating = bool(alternating)
        self.block_size_in, self.block_size_out = e0.block_size_in, e0.block_size_out
        self.block_size = e0.block_size_in                     # merge "is-active" guard
        self.register_buffer(
            "weight", torch.empty(self.E, out_features, in_features, device=device, dtype=dtype)
        )
        # shared block triu indices (identical block sizes across experts)
        for nm in ("rows_in", "cols_in", "rows_out", "cols_out"):
            self.register_buffer(nm, getattr(e0, nm).clone())

    @torch.no_grad()
    def bind_weights(self):
        """Copy each expert's (baked) forward-frame weight into the buffer and repoint
        the expert weight to the buffer row (single storage). Call once at build, after
        each expert weight is copied + baked."""
        for e, ex in enumerate(self.experts):
            self.weight[e].copy_(ex.weight)
            ex.weight.data = self.weight[e]

    def forward(self, concat_x, tokens_per_expert):
        oft_in = torch.stack([ex.oft_R_in for ex in self.experts])
        oft_out = torch.stack([ex.oft_R_out for ex in self.experts])
        pin = torch.stack([ex.perm_in_inv for ex in self.experts])
        pout = torch.stack([ex.perm_out_inv for ex in self.experts])
        sizes = tuple(int(t) for t in tokens_per_expert.tolist())
        return GroupedPOETXFunction.apply(
            concat_x, oft_in, oft_out, self.weight, pin, pout,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out, sizes)

    def effective_weight(self):
        return self.weight                                    # oft_R==0 -> R==I -> Wx==weight

    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm=True):
        for ex in self.experts:
            ex.merge_then_reinitialize(reinit_perm=reinit_perm)

    @torch.no_grad()
    def _fold_active_side(self, active, reinit_perm=False):
        for ex in self.experts:
            ex._fold_active_side(active, reinit_perm=reinit_perm)
```

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest third_party/poet_torch/tests_poet/test_grouped_poetx.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add third_party/poet_torch/poet_torch/grouped_poetx_layer.py third_party/poet_torch/tests_poet/test_grouped_poetx.py
git commit -m "feat(poet): GroupedPOETXLinear — per-expert POETX over a batched weight buffer"
```

---

## Task 5: Merge-driver integration

The periodic merge ([poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py)) must fold `GroupedPOETXLinear`. Each grouped module's experts are bare `POETXLinear`s (NOT wrapped in `POETMegatronLinear`), so the existing collection loop skips them — we add an explicit grouped pass that delegates to the verified per-expert fold.

**Files:**
- Modify: `src/patches/poet_merge_step.py`
- Test: `tests/patches/test_grouped_poetx_merge.py`

**Interfaces:**
- Consumes: `GroupedPOETXLinear` (Task 4); `active_side` from `poet_torch.alt_state`.
- Produces: `_merge_grouped(grouped_modules, reinit_perm)` (module-level, CPU-safe); a grouped collection + fold inside `_run_merge`.

- [ ] **Step 1: Write the failing test**

```python
# tests/patches/test_grouped_poetx_merge.py
import torch

from poet_torch.grouped_poetx_layer import GroupedPOETXLinear
from src.patches.poet_merge_step import _merge_grouped


def test_merge_grouped_folds_and_zeros_active_side():
    torch.manual_seed(0)
    g = GroupedPOETXLinear(3, 8, 8, block_count=2, alternating=True,
                           alternate_every=1, dtype=torch.float64)
    for e in range(3):
        g.experts[e].weight.data.copy_(torch.randn(8, 8, dtype=torch.float64))
        g.experts[e].bake_perms_into_weight()
    g.bind_weights()
    for ex in g.experts:
        ex.oft_R_in.data.normal_(std=0.1)
        ex.oft_R_out.data.normal_(std=0.1)

    # effective weight via a forward at the current oft_R, captured before merge
    w_before = g.weight.clone()
    _merge_grouped([g], reinit_perm=False)
    # active side folded into weight; folded side's oft_R zeroed; weight changed.
    assert not torch.allclose(g.weight, w_before)
    from poet_torch.alt_state import active_side
    active = active_side(1)
    folded = "oft_R_in" if active == "in" else "oft_R_out"
    for ex in g.experts:
        assert getattr(ex, folded).abs().max() == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/patches/test_grouped_poetx_merge.py -v`
Expected: FAIL — `_merge_grouped` undefined.

- [ ] **Step 3: Add `_merge_grouped` and wire it into `_run_merge`**

Add this module-level helper to `src/patches/poet_merge_step.py` (near `_merge_layers`, ~line 545; CPU-safe, no megatron import):

```python
def _merge_grouped(grouped, reinit_perm: bool) -> None:
    """Fold every GroupedPOETXLinear by delegating to its per-expert POETXLinears
    (verified path). alternating modules fold only the active side."""
    from poet_torch.alt_state import active_side

    for g in grouped:
        if getattr(g, "alternating", False):
            g._fold_active_side(active_side(g.experts[0].alternate_every),
                                reinit_perm=reinit_perm)
        else:
            g.merge_then_reinitialize(reinit_perm=reinit_perm)
```

In `_run_merge` ([poet_merge_step.py:455](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L455)), after building `pls` (the loop at ~line 471), collect grouped modules:

```python
    from poet_torch.grouped_poetx_layer import GroupedPOETXLinear

    grouped = []
    for m in chunks:
        for _, mod in m.named_modules():
            if isinstance(mod, GroupedPOETXLinear):
                grouped.append(mod)
```

In the **replicate** branch (after `_merge_layers(pls, ...)`, ~line 504) and in the **broadcast** branch (rank-0 fold, ~line 527), add the grouped fold. For the replicate path, also sync each grouped expert's perms once (extend the existing `_perms_synced` loop to include `g.experts`):

```python
    # replicate branch, alongside the existing _merge_layers call:
    _merge_grouped(grouped, reinit_perm=False)
    # broadcast branch, inside `if rank == 0:`:
    _merge_grouped(grouped, reinit_perm=reinit_perm)
    # broadcast branch, inside `if is_dist:` after the pls loop — also broadcast each
    # grouped expert's oft_R/weight/perms:
    for g in grouped:
        for ex in g.experts:
            for buf in (ex.oft_R_in.data, ex.oft_R_out.data, ex.weight.data,
                        ex.perm_in, ex.perm_in_inv, ex.perm_out, ex.perm_out_inv):
                dist.broadcast(buf, src=0)
        dist.broadcast(g.weight.data, src=0)
```

(The replicate path needs no broadcast — the fold is a deterministic function of DP-identical state, same rationale as the 2-D layers. Add the grouped experts' perms to the one-time `_perms_synced` sync at ~line 498–502.)

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/patches/test_grouped_poetx_merge.py -v`
Expected: PASS.

- [ ] **Step 5: Run the merge suite to confirm no regression**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_merge_step.py tests/unit/test_patch_poet_merge.py -v`
Expected: PASS (pre-existing `test_run_merge_invalidates_cache_on_cached_poet_linear` Triton "0 active drivers" failure on the CPU dev box is unchanged — see [[poet-moe-guard-profiler-on-main]]).

- [ ] **Step 6: Commit**

```bash
git add src/patches/poet_merge_step.py tests/patches/test_grouped_poetx_merge.py
git commit -m "feat(poet): fold GroupedPOETXLinear experts in the periodic merge"
```

---

## Task 6: Walk integration — detect `SequentialMLP`, install grouped, swap forward

The riskiest task. `replace_linears_with_poet` gains a `group_experts` flag; when set and on the POETX path, it detects `SequentialMLP`, builds one `GroupedPOETXLinear` per expert linear *role*, and replaces `SequentialMLP.forward` with a grouped version. The existing 2-D path is untouched.

**Files:**
- Modify: `src/optim/poet_layers.py` (the `_walk` at [line 231](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L231); signature at [line 179](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L179))
- Test: `tests/optim/test_grouped_expert_wrap.py`

**Interfaces:**
- Consumes: `GroupedPOETXLinear` (Task 4); `_copy_and_init_weight` (existing in `poet_layers.py`); `single_step_x`, `lie_alternating`, `alternate_every`, `block_count` (existing walk args).
- Produces: `replace_linears_with_poet(..., group_experts: bool = False)`; a private `_install_grouped_poetx(seq_mlp, *, block_count, alternating, alternate_every, init_type, mup_alpha) -> int`.

Roles & forward contract (from [SequentialMLP](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/transformer/moe/experts.py#L783)): experts are `local_experts[i]`, each an `MLP` with `linear_fc1`, `linear_fc2` (and unfused gate/up segments under `unfuse_fc1`). The grouped forward mirrors the `num_local_experts>1`, bf16, non-fp8 branch: split `permuted_local_hidden_states` by `tokens_per_expert`, run grouped fc1 → activation → grouped fc2, concat. Raise on per-expert bias and on `config.fp8/fp4`.

- [ ] **Step 1: Write the failing test (fake SequentialMLP, CPU)**

```python
# tests/optim/test_grouped_expert_wrap.py
import torch
import torch.nn as nn

from src.optim.poet_layers import replace_linears_with_poet


class _ColLinear(nn.Module):                      # stands in for ColumnParallelLinear
    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i, dtype=torch.float64))
        self.bias = None
        self.skip_bias_add = True
    def forward(self, x):
        return x @ self.weight.t(), None


class _Expert(nn.Module):
    def __init__(self, h=8, f=8):
        super().__init__()
        self.linear_fc1 = _ColLinear(h, f)
        self.linear_fc2 = _ColLinear(f, h)
    def forward(self, x, probs=None):
        h, _ = self.linear_fc1(x)
        h = torch.relu(h)
        o, _ = self.linear_fc2(h)
        return o, None


class _FakeSequentialMLP(nn.Module):
    """Mirrors the SequentialMLP contract the grouped install targets."""
    def __init__(self, E=3, h=8, f=8):
        super().__init__()
        self.num_local_experts = E
        self.local_experts = nn.ModuleList([_Expert(h, f) for _ in range(E)])
    def forward(self, permuted, tokens_per_expert, probs):
        outs = []
        for ex, t in zip(self.local_experts, torch.split(permuted, tokens_per_expert.tolist())):
            o, _ = ex(t)
            outs.append(o)
        return torch.cat(outs, 0), None


def test_walk_installs_grouped_poetx_and_forward_matches():
    from poet_torch.grouped_poetx_layer import GroupedPOETXLinear

    torch.manual_seed(0)
    m = _FakeSequentialMLP().to(torch.float64)
    ref = _FakeSequentialMLP().to(torch.float64)
    ref.load_state_dict(m.state_dict())

    tokens = torch.randn(9, 8, dtype=torch.float64)
    tpe = torch.tensor([2, 3, 4])
    ref_out, _ = ref(tokens, tpe, None)

    n = replace_linears_with_poet(
        m, block_count=2, single_step_x=True, single_step_fast=True,
        lie_alternating=True, alternate_every=1, group_experts=True,
        extra_grouped_types=(_FakeSequentialMLP,),
    )
    assert n >= 1
    assert any(isinstance(mod, GroupedPOETXLinear) for mod in m.modules())
    # oft_R==0 at init -> grouped forward equals the original expert forward.
    out, _ = m(tokens, tpe, None)
    assert torch.allclose(out, ref_out, atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/optim/test_grouped_expert_wrap.py -v`
Expected: FAIL — `replace_linears_with_poet` has no `group_experts`/`extra_grouped_types`.

- [ ] **Step 3: Add the grouped install + forward swap**

In `replace_linears_with_poet` ([line 179](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L179)) add params `group_experts: bool = False` and `extra_grouped_types: Iterable[type] = ()`. Resolve the grouped types once (mirroring `linear_types`):

```python
    grouped_types = tuple(extra_grouped_types) + _megatron_sequential_mlp_types()
```

where `_megatron_sequential_mlp_types()` lazily returns `(SequentialMLP,)` or `()` if megatron is unavailable (mirror `_megatron_linear_types` at [line 126](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L126)). In `_walk`, before the final `else: _walk(child, full)` ([line 412](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L412)):

```python
            if (group_experts and single_step_x and grouped_types
                    and isinstance(child, grouped_types)):
                replaced += _install_grouped_poetx(
                    child, block_count=block_count,
                    alternating=lie_alternating, alternate_every=alternate_every,
                    init_type=init_type, mup_alpha=mup_alpha,
                )
                continue
```

Add the installer (module-level in `poet_layers.py`):

```python
_EXPERT_ROLE_NAMES = ("linear_fc1", "linear_fc2")  # extend if unfuse adds segment names


def _install_grouped_poetx(seq_mlp, *, block_count, alternating, alternate_every,
                           init_type, mup_alpha):
    """Replace a SequentialMLP's per-expert POETX linears with one GroupedPOETXLinear
    per role, and swap its forward to run the grouped path. Returns #roles grouped."""
    import torch
    from poet_torch.grouped_poetx_layer import GroupedPOETXLinear

    experts = list(seq_mlp.local_experts)
    E = len(experts)
    # Discover POET-targetable roles on expert 0 (linears divisible by block_count).
    roles = []
    for name, child in experts[0].named_children():
        w = getattr(child, "weight", None)
        if w is None or w.dim() != 2:
            continue
        if getattr(child, "bias", None) is not None and child.bias.numel() > 0:
            raise ValueError(f"[POET] grouped experts require bias-free linears; {name} has bias")
        out_f, in_f = w.shape
        if in_f % block_count or out_f % block_count:
            raise ValueError(
                f"[POET] grouped expert role {name} dims ({out_f},{in_f}) not divisible "
                f"by block_count={block_count}")
        roles.append(name)

    grouped_by_role = {}
    for name in roles:
        w0 = getattr(experts[0], name).weight
        out_f, in_f = w0.shape
        g = GroupedPOETXLinear(E, in_f, out_f, block_count=block_count,
                               alternating=alternating, alternate_every=alternate_every,
                               device=w0.device, dtype=w0.dtype)
        for e in range(E):
            _copy_and_init_weight(g.experts[e], getattr(experts[e], name), init_type, mup_alpha)
            g.experts[e].bake_perms_into_weight()
        g.bind_weights()
        grouped_by_role[name] = g
        seq_mlp.add_module(f"grouped_{name}", g)

    seq_mlp._poet_grouped = grouped_by_role
    seq_mlp.forward = _grouped_sequential_forward.__get__(seq_mlp, type(seq_mlp))
    return len(roles)


def _grouped_sequential_forward(self, permuted_local_hidden_states, tokens_per_expert, *rest):
    """Grouped replacement for SequentialMLP.forward (bf16, non-fp8, num_experts>1).
    Mirrors the per-expert fc1 -> activation -> fc2 chain through the grouped modules."""
    import torch

    cfg = getattr(self, "config", None)
    if cfg is not None and (getattr(cfg, "fp8", None) or getattr(cfg, "fp4", None)):
        raise ValueError("[POET] grouped experts do not support fp8/fp4 (target bf16)")
    g1 = self._poet_grouped["linear_fc1"]
    g2 = self._poet_grouped["linear_fc2"]
    h = g1(permuted_local_hidden_states, tokens_per_expert)
    h = self.local_experts[0].activation_func(h) if hasattr(self.local_experts[0], "activation_func") \
        else torch.nn.functional.relu(h)
    out = g2(h, tokens_per_expert)
    return out, None
```

NOTE for the implementer: the activation line is the one place the fake test (`torch.relu`) and the real `MLP` diverge. Before wiring the real model, confirm the real expert `MLP`'s activation (swiglu gate/up handling under `unfuse_fc1`, and how `permuted_probs` is applied) and reproduce it exactly here — the grouped forward must be numerically identical to the stock `SequentialMLP.forward` chain. Wrap the gate/up split per the unfuse role set (extend `_EXPERT_ROLE_NAMES`). This is validated end-to-end by the GPU smoke (Task 7); the CPU test only fixes the module wiring + the bare fc1→act→fc2 shape.

- [ ] **Step 4: Run to verify it passes**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/optim/test_grouped_expert_wrap.py -v`
Expected: PASS.

- [ ] **Step 5: Run the POET walk + layer suites to confirm the 2-D path is untouched**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest third_party/poet_torch/tests_poet -q -k "not cuda" && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/optim -q`
Expected: PASS (pre-existing failures, if any, unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_layers.py tests/optim/test_grouped_expert_wrap.py
git commit -m "feat(poet): install GroupedPOETXLinear over SequentialMLP experts in the walk"
```

---

## Task 7: Config + flag plumbing, profile gate, GPU smoke & A/B (USER-RUN GPU)

Expose `optim.poet.group_experts` end-to-end and prove the win. The GPU/cluster runs are the user's per project policy — the agent makes the CPU changes and hands over exact commands.

**Files:**
- Modify: `src/utils/megatron_args.py` (emit `--poet-group-experts`)
- Modify: `src/patches/poet_apply_to_model.py` (thread `group_experts` into `replace_linears_with_poet`)
- Create: `configs/experiments/optim/poet_lie_orth_alt_grouped.yaml` + `docs/experiments/poet_lie_orth_alt_grouped.md`
- Test: `tests/unit/test_megatron_args_grouped_poetx.py`

**Interfaces:**
- Consumes: all prior tasks; the existing `--poet-*` emission block ([megatron_args.py:479](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L479)).
- Produces: `--poet-group-experts` CLI flag; `optim.poet.group_experts` config key; the arg `group_experts` read in `_apply_poet_to_chunk`.

- [ ] **Step 1: Failing arg-build test (CPU)**

```python
# tests/unit/test_megatron_args_grouped_poetx.py
from omegaconf import OmegaConf

from src.utils.megatron_args import _optimizer_args


def _cfg(group_experts):
    return OmegaConf.create({
        "optim": {"type": "poet", "lr": 3e-3, "weight_decay": 0.1,
                  "betas": [0.9, 0.95], "eps": 1e-8,
                  "poet": {"block_count": 8, "cache_mode": "none",
                           "init_type": "normalized", "mup_alpha": 1.0,
                           "merge_period": 1, "scale": 0.5,
                           "single_step_x": True, "single_step_fast": True,
                           "lie_alternating": True, "group_experts": group_experts}}})


def test_group_experts_flag_emitted_only_when_set():
    assert "--poet-group-experts" in _optimizer_args(_cfg(True))
    assert "--poet-group-experts" not in _optimizer_args(_cfg(False))
```

- [ ] **Step 2: Run to verify it fails**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args_grouped_poetx.py -v`
Expected: FAIL.

- [ ] **Step 3: Emit the flag + thread it through apply**

In `_optimizer_args` ([megatron_args.py:479](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L479)), alongside the other conditional `poet_args.append(...)` store-true flags:

```python
        if poet.get("group_experts", False):
            poet_args.append("--poet-group-experts")
```

Register `--poet-group-experts` as a store-true arg wherever the POET argparse flags are declared (mirror `--poet-lie-alternating`), and read it in `_apply_poet_to_chunk` ([poet_apply_to_model.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py)), passing `group_experts=getattr(args, "poet_group_experts", False)` into `replace_linears_with_poet(...)`.

- [ ] **Step 4: Run to verify it passes + full arg suite**

Run: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args_grouped_poetx.py tests/unit/test_megatron_args.py -v`
Expected: PASS (the existing POET + grouped-gemm guard tests stay green; `group_experts` is orthogonal to `moe.grouped_gemm`).

- [ ] **Step 5: Add the config + doc**

`configs/experiments/optim/poet_lie_orth_alt_grouped.yaml`: copy `poet_lie_orth_alt.yaml`, set `optim.poet.group_experts: true`, name it `poet_lie_orth_alt_grouped`. Add `docs/experiments/poet_lie_orth_alt_grouped.md` (the experiment-YAML pre-commit hook requires a matching doc).

- [ ] **Step 6: Commit**

```bash
git add src/utils/megatron_args.py src/patches/poet_apply_to_model.py configs/experiments/optim/poet_lie_orth_alt_grouped.yaml docs/experiments/poet_lie_orth_alt_grouped.md tests/unit/test_megatron_args_grouped_poetx.py
git commit -m "feat(poet): plumb --poet-group-experts (grouped POETX over MoE experts)"
```

- [ ] **Step 7: Profile gate (USER-RUN — confirm the lever before trusting the speedup)**

Confirm the expert `M`/GEMM ops dominate forward/backward (the prior drill-down was killed; grad-accum=1 keeps the trace small):

```bash
env POET_PROFILE_STEP=20 POET_PROFILE_TORCH=1 \
  bash scripts/train_deepseek_poet.sh full \
  training.global_batch_size=8 training.micro_batch_size=1 training.log_interval=1
```

Read the `torch.profiler top ops` table. Proceed only if expert GEMM / `M` rows dominate.

- [ ] **Step 8: GPU smoke (USER-RUN — 1 GPU, cheap)**

```bash
codexlog poetx_grouped_smoke bash scripts/train_deepseek_poet.sh dev optim.poet.group_experts=true training.log_interval=1
```

Acceptance: builds; `[POET] replaced N` logs the grouped experts; a few steps run with finite loss, no NaN; loss at step ~10 matches the non-grouped `dev` run within fp noise (parity sanity).

- [ ] **Step 9: Throughput + loss A/B (USER-RUN — 8 GPU)**

```bash
codexlog poetx_grouped_full bash scripts/train_deepseek_poet.sh full optim.poet.group_experts=true
codexlog poetx_baseline_full bash scripts/train_deepseek_poet.sh full          # current per-expert POET
```

Acceptance: grouped TFLOP/s materially > the 4.2 baseline; lm-loss trajectory within noise of the per-expert POET run over the same steps. Record both in `docs/experiments/poet_lie_orth_alt_grouped.md`.

---

## Self-Review notes (for the implementer)

- **Parity is the gate at every layer:** Task 2 (block-sparse == full-M reference), Task 3 (Function == E independent POETX), Task 4 (module == E POETXLinears, incl. merge), Task 6 (grouped forward == stock at oft_R=0), Task 7 (real loss A/B). If Task 7 loss diverges from the per-expert POET run, suspect the Task-6 activation/probs reproduction (gate/up split under `unfuse_fc1`) — bisect by grouping `linear_fc2` only first.
- **`oft_R` stays 2-D, E separate params** (Task 1 guards it). The only stacked state is the frozen weight buffer. Forward stacks `oft_R` transiently via `torch.stack`; autograd splits the grad back to the leaves. Optimizer + merge see no change.
- **No silent skips/wrong:** non-divisible dims, per-expert bias, and fp8/fp4 all raise (Tasks 4/6). Don't add fallbacks.
- **Forward is unchanged cost.** The win is entirely the batched block-sparse backward; the merge stays the verified per-expert fold (the cheap 2.6%). Don't "optimize" the merge.
- **Riskiest edit is Task 6's `SequentialMLP.forward` swap** — the CPU test fixes the wiring, but the real activation/probs/`unfuse_fc1` reproduction must be confirmed against the live `MLP` before the GPU smoke.
