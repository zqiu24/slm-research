# HeadAlignedPOETXLinear (permuted multi-block residual) + head × alternating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port head-aligned attention rotation onto the POETX forward-frame layer as a thin `POETXLinear` subclass whose **residual side carries a real POETX permutation + multiple blocks** (the "general" design the current perm-free layer cannot express), then test whether this corrected head-aligned + the alternating champion flips the −0.016 head penalty.

**Architecture:** `HeadAlignedPOETXLinear(POETXLinear)` does its own `__init__` (like the existing `HeadAlignedPOETLinear`) to set **asymmetric** block sizes (`head_dim` on the head side, `hidden/head_resid_block_count` on the residual side) and **asymmetric** permutations (identity on the head side, random on the residual side), then **inherits POETXLinear's forward / backward / merge verbatim** — those are already perm-aware (`perm_in_inv`/`perm_out_inv` in the backward conj) and block-aware (decoupled `block_size_in`/`block_size_out`), and already gained the `alternating` active-only merge in the prerequisite plan. No new autograd Function is needed.

**Tech Stack:** Python, PyTorch, Megatron-LM (vendored), Triton (Cayley), pytest. Layer code in `third_party/poet_torch/`; wiring in `src/`.

**Depends on:** [2026-06-08-alternating-poetx-integrated.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/plans/2026-06-08-alternating-poetx-integrated.md) — `POETXLinear` must already carry `alternating`/`alternate_every` and the active-only merge. **Implement that plan first.** Design spec: [2026-06-08-head-aligned-poetx-permuted-resid-design.md](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-08-head-aligned-poetx-permuted-resid-design.md).

---

## Conventions

- **Test runner:** `cd /lustre/fast/fast/zqiu/slm-research && PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest <path> -v`
- **Args/launcher tests** (need omegaconf/hydra): same runner — the `slm_env` venv has omegaconf; fall back to `/var/tmp/zqiu/slmcpu312/bin/python` only if it does not.
- **fp64 for parity tests** (`torch.set_default_dtype(torch.float64)`); reset to float32 in an autouse fixture so the default doesn't leak into later test files (see `tests/unit/test_poet_lie_orth.py` for the pattern).
- **Run CPU tests yourself; GPU runs are the user's** — provide the exact command and stop.
- **Commits:** single short conventional-commit line, no AI attribution.

**Head-side convention (must match the existing walk):** `head_side="out"` for q/k/v (rows = heads), `head_side="in"` for o (cols = heads). The residual side is the *other* side (= `hidden_size`).

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `third_party/poet_torch/head_aligned_poetx_layer.py` | Create | `HeadAlignedPOETXLinear(POETXLinear)` — asymmetric perms + blocks; inherits POETX compute |
| `third_party/poet_torch/__init__.py` | Modify | Export `HeadAlignedPOETXLinear` |
| `src/optim/poet_layers.py` | Modify | Walk builds `HeadAlignedPOETXLinear` for attention q/k/v/o when `head_aligned_attn` + `single_step_x`; thread `head_resid_block_count` |
| `src/patches/poet_apply_to_model.py` | Modify | Read `head_aligned_attn` + `head_resid_block_count` from args; pass to the walk |
| `src/utils/megatron_args.py` | Modify | Validate + emit `--poet-head-resid-block-count`; allow `head_aligned_attn` + `single_step_x` |
| `launchers/pretrain_gpt_slm.py` | Modify | Add `--poet-head-resid-block-count` CLI flag |
| `configs/experiments/optim/poet_lie_orth_head_alt.yaml` | Create | head-on POETX + alternating experiment config |
| `docs/experiments/poet_lie_orth_head_alt.md` | Create | Experiment doc (pre-commit hook requires it) |
| `scripts/train_poet_lie_orth_head_alt.sh` | Create | Launcher |
| `tests/unit/test_head_aligned_poetx.py` | Create | Layer construction + forward-chain + merge parity tests |
| `tests/unit/test_poet_layers.py` | Modify | Walk-selection test for the new layer |
| `tests/unit/test_megatron_args.py` | Modify | `head_resid_block_count` round-trip + validation |

**Reference for the new `__init__`:** mirror `HeadAlignedPOETLinear.__init__` ([head_aligned_layer.py:132-228](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/head_aligned_layer.py#L132-L228)) for the head/resid block logic, and `POETXLinear.__init__` ([poetx_layer.py:22-104](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_layer.py#L22-L104)) for the forward-frame weight + `bake_perms_into_weight`. The **only** differences from `POETXLinear` are: independent block sizes, identity perm on the head side (random on the residual side), and accepting `head_side`/`head_dim`/`head_resid_block_count`.

---

## Task 1: `HeadAlignedPOETXLinear` construction

**Files:**
- Create: `third_party/poet_torch/head_aligned_poetx_layer.py`
- Modify: `third_party/poet_torch/__init__.py`
- Test: `tests/unit/test_head_aligned_poetx.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_head_aligned_poetx.py`:

```python
"""HeadAlignedPOETXLinear: a POETXLinear with head_dim blocks + identity perm on the
head side, and a real random perm + multiple blocks on the residual side."""
import pytest
import torch


@pytest.fixture(autouse=True)
def _isolate_default_dtype():
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(torch.float32)


def test_qkv_head_side_out_structure():
    from poet_torch import HeadAlignedPOETXLinear, POETXLinear

    # q/k/v: head_side="out". out=heads*head_dim, in=hidden.
    heads, head_dim, hidden = 4, 8, 16
    layer = HeadAlignedPOETXLinear(
        in_features=hidden, out_features=heads * head_dim,
        head_side="out", head_dim=head_dim, head_resid_block_count=2, bias=False,
    )
    assert isinstance(layer, POETXLinear)  # merge driver / isinstance routing
    # head side = out -> block_size_out == head_dim, r_out == heads
    assert layer.block_size_out == head_dim
    assert layer.r_out == heads
    # residual side = in -> split into head_resid_block_count blocks
    assert layer.block_size_in == hidden // 2
    assert layer.r_in == 2
    # head-side perm is identity; residual-side perm is a real permutation
    assert torch.equal(layer.perm_out, torch.arange(heads * head_dim, dtype=torch.int32))
    assert not torch.equal(layer.perm_in, torch.arange(hidden, dtype=torch.int32))
    # perm inverses are consistent
    assert torch.equal(layer.perm_in[layer.perm_in_inv.long()],
                       torch.arange(hidden, dtype=torch.int32))


def test_o_head_side_in_structure():
    from poet_torch import HeadAlignedPOETXLinear

    # o: head_side="in". in=heads*head_dim, out=hidden.
    heads, head_dim, hidden = 4, 8, 16
    layer = HeadAlignedPOETXLinear(
        in_features=heads * head_dim, out_features=hidden,
        head_side="in", head_dim=head_dim, head_resid_block_count=2, bias=False,
    )
    assert layer.block_size_in == head_dim
    assert layer.r_in == heads
    assert layer.block_size_out == hidden // 2
    assert layer.r_out == 2
    # head side = in -> identity perm there; residual side = out -> real perm
    assert torch.equal(layer.perm_in, torch.arange(heads * head_dim, dtype=torch.int32))
    assert not torch.equal(layer.perm_out, torch.arange(hidden, dtype=torch.int32))


def test_invalid_head_side_raises():
    from poet_torch import HeadAlignedPOETXLinear

    with pytest.raises(ValueError, match="head_side"):
        HeadAlignedPOETXLinear(
            in_features=16, out_features=32, head_side="bogus",
            head_dim=8, head_resid_block_count=2,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poetx.py -v`
Expected: FAIL — `ImportError: cannot import name 'HeadAlignedPOETXLinear'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/poet_torch/head_aligned_poetx_layer.py`:

```python
"""HeadAlignedPOETXLinear: head-aligned attention rotation on the POETX forward
frame. A thin POETXLinear subclass that sets ASYMMETRIC block sizes and perms:

  * head side (out for q/k/v, in for o): block = head_dim, perm = IDENTITY
    (block j is always head j — no cross-head mixing).
  * residual side (the hidden_size side): block_count = head_resid_block_count
    (> 1), perm = RANDOM Ψ — a real permuted multi-block rotation, the thing the
    legacy perm-free HeadAlignedPOETLinear cannot express.

All compute (forward / backward / merge, incl. the alternating active-only fold)
is INHERITED from POETXLinear — it is already perm-aware (perm_*_inv in the
backward conj) and block-aware (decoupled block_size_in/out). Only __init__ differs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .poetx_layer import POETXLinear


class HeadAlignedPOETXLinear(POETXLinear):
    def __init__(self, in_features, out_features, *, head_side, head_dim,
                 head_resid_block_count, bias=False, device=None, dtype=None,
                 parameterization="cayley", alternating=False, alternate_every=1):
        nn.Module.__init__(self)
        if head_side not in ("in", "out"):
            raise ValueError(f"head_side must be 'in' or 'out', got {head_side!r}")
        if parameterization != "cayley":
            raise ValueError(
                "HeadAlignedPOETXLinear requires parameterization='cayley' "
                f"(POETX backward is Cayley-specific); got {parameterization!r}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.parameterization = parameterization
        self.single_step_fast = False
        self.head_side = head_side
        self.head_dim = head_dim

        head_features = out_features if head_side == "out" else in_features
        resid_features = in_features if head_side == "out" else out_features
        if head_features % head_dim != 0:
            raise ValueError(f"head_dim {head_dim} doesn't divide head-side dim {head_features}")
        if resid_features % head_resid_block_count != 0:
            raise ValueError(
                f"head_resid_block_count {head_resid_block_count} doesn't divide "
                f"residual dim {resid_features}"
            )
        resid_bs = resid_features // head_resid_block_count
        if head_side == "out":
            block_size_out, block_size_in = head_dim, resid_bs
        else:
            block_size_in, block_size_out = head_dim, resid_bs
        self.block_size_in = block_size_in
        self.block_size_out = block_size_out
        self.block_size = block_size_in  # back-compat (merge "is-active" guard)

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype), requires_grad=False
            )
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

        # Head side: identity perm (block j == head j). Residual side: random Ψ.
        if head_side == "out":
            perm_out = torch.arange(out_features, device=device, dtype=torch.int32)
            perm_in = torch.randperm(in_features, device=device).to(torch.int32)
        else:
            perm_in = torch.arange(in_features, device=device, dtype=torch.int32)
            perm_out = torch.randperm(out_features, device=device).to(torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

        # Alternating active-only merge is inherited from POETXLinear (prereq plan).
        self.alternating = bool(alternating)
        self.alternate_every = max(1, int(alternate_every))
        # self.weight is the (empty) W_perm-frame tensor; bake_perms_into_weight()
        # converts it to the forward frame once the real weight is copied in (the
        # walk calls it after _copy_and_init_weight), exactly as for POETXLinear.
```

- [ ] **Step 4: Add the export**

In `third_party/poet_torch/__init__.py`, after the `AlternatingPOETXLinear` export line, add:

```python
from .head_aligned_poetx_layer import HeadAlignedPOETXLinear as HeadAlignedPOETXLinear
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poetx.py -v`
Expected: PASS (3 passed). If ruff flags `Wx`/`R_*` style names in the test, add `tests/unit/test_head_aligned_poetx.py` to `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml` with `["N802", "N803", "N806"]` (mirror `test_poetx_layer.py`).

- [ ] **Step 6: Commit**

```bash
git add third_party/poet_torch/head_aligned_poetx_layer.py third_party/poet_torch/__init__.py tests/unit/test_head_aligned_poetx.py pyproject.toml
git commit -m "feat(poet): HeadAlignedPOETXLinear (asymmetric perm/blocks, permuted residual)"
```

---

## Task 2: Forward-chain parity (inherited POETX forward is correct for the asymmetric layout)

**Files:**
- Test: `tests/unit/test_head_aligned_poetx.py`

This verifies the inherited `POETXLinear.forward` produces the exact POET chain `y = (P_out R_out P_outᵀ)·W·(P_in R_in P_inᵀ)·x (+ bias)` for the asymmetric perms/blocks, i.e. the construction wires `bake_perms_into_weight` + the backward conj correctly.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_head_aligned_poetx.py`:

```python
def _chain_ref(layer, x):
    # Effective weight from the stored forward-frame weight + current oft_R.
    from poet_torch.poet_layer import cayley_batch, pytorch_skew_symmetric

    Ri = cayley_batch(pytorch_skew_symmetric(layer.oft_R_in, layer.block_size_in,
                                             layer.rows_in, layer.cols_in))
    Ro = cayley_batch(pytorch_skew_symmetric(layer.oft_R_out, layer.block_size_out,
                                             layer.rows_out, layer.cols_out))
    # forward-frame weight Wx -> un-permute to W_perm -> apply block-diag R both sides
    Wx = layer.weight
    W_perm = Wx.index_select(0, layer.perm_out_inv.long()).index_select(1, layer.perm_in_inv.long())
    from poet_torch.poet_layer import block_diag_lr_matmul_decoupled

    W_rot = block_diag_lr_matmul_decoupled(Ro, W_perm, Ri.transpose(-2, -1))
    W_eff = W_rot.index_select(0, layer.perm_out.long()).index_select(1, layer.perm_in.long())
    y = x @ W_eff.t()
    if layer.bias is not None:
        y = y + layer.bias.index_select(0, layer.perm_out.long())  # bias stored in fwd frame
    return y


def test_forward_matches_poet_chain_at_small_rotation():
    from poet_torch import HeadAlignedPOETXLinear

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    layer = HeadAlignedPOETXLinear(
        in_features=16, out_features=32, head_side="out", head_dim=8,
        head_resid_block_count=2, bias=True,
    )
    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
        layer.bake_perms_into_weight()  # walk does this after copying the real weight
        layer.oft_R_in.normal_(std=1e-2)
        layer.oft_R_out.normal_(std=1e-2)
    x = torch.randn(5, 16)
    assert torch.allclose(layer(x), _chain_ref(layer, x), atol=1e-9), \
        (layer(x) - _chain_ref(layer, x)).abs().max()
```

> Note: confirm the exact frame conventions of `block_diag_lr_matmul_decoupled` against `POETXLinear._fold_with_R` ([poetx_layer.py:121-140](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_layer.py#L121-L140)) when writing `_chain_ref`; the reference must match POETX's own fold convention (un-permute → fold → re-permute). If `block_diag_lr_matmul_decoupled`'s argument order differs, align `_chain_ref` to whatever `_fold_with_R` uses so the test asserts the *intended* math, not a guess.

- [ ] **Step 2: Run test to verify it (initially) fails or passes**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poetx.py -k forward_matches -v`
Expected: PASS if the inherited forward + construction are correct. If it FAILS, the failure localizes a construction bug (perm assignment, block size, or bake) — fix the `__init__`, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_head_aligned_poetx.py
git commit -m "test(poet): HeadAlignedPOETXLinear forward matches the POET chain (asymmetric perm/blocks)"
```

---

## Task 3: Merge round-trip parity

**Files:**
- Test: `tests/unit/test_head_aligned_poetx.py`

The inherited `POETXLinear.merge_then_reinitialize` / `_fold_with_R` must fold both sides correctly for the asymmetric layout (and, under `alternating`, the inherited active-only fold). This test pins the both-sides fold; the alternating active-only fold is already covered by the prerequisite plan's tests once `HeadAlignedPOETXLinear` inherits it.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_head_aligned_poetx.py`:

```python
def test_merge_folds_and_zeros_oft_r():
    from poet_torch import HeadAlignedPOETXLinear

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    layer = HeadAlignedPOETXLinear(
        in_features=16, out_features=32, head_side="out", head_dim=8,
        head_resid_block_count=2, bias=False,
    )
    with torch.no_grad():
        layer.weight.normal_()
        layer.bake_perms_into_weight()
        layer.oft_R_in.normal_(std=1e-2)
        layer.oft_R_out.normal_(std=1e-2)
    x = torch.randn(4, 16)
    before = layer(x).detach().clone()        # effective map with R != I
    layer.merge_then_reinitialize(reinit_perm=False)
    # after folding R into W and zeroing oft_R, the bare forward reproduces it
    assert torch.count_nonzero(layer.oft_R_in) == 0
    assert torch.count_nonzero(layer.oft_R_out) == 0
    assert torch.allclose(layer(x), before, atol=1e-9), (layer(x) - before).abs().max()
```

- [ ] **Step 2: Run the merge test**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poetx.py -k merge -v`
Expected: PASS (fold is spectrum-preserving and the merged weight reproduces the pre-merge effective map).

- [ ] **Step 3: Run the whole new test file + adjacent POETX tests**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_head_aligned_poetx.py tests/unit/test_poetx_layer.py -v`
Expected: PASS — new tests pass AND existing POETX tests still pass.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_head_aligned_poetx.py
git commit -m "test(poet): HeadAlignedPOETXLinear merge round-trips (asymmetric perm/blocks)"
```

---

## Task 4: Walk wiring — build `HeadAlignedPOETXLinear` for attention

**Files:**
- Modify: `src/optim/poet_layers.py`
- Modify: `src/patches/poet_apply_to_model.py`
- Test: `tests/unit/test_poet_layers.py`

- [ ] **Step 1: Write the failing walk-selection test**

Append to `tests/unit/test_poet_layers.py`:

```python
def test_head_aligned_poetx_built_for_attention_under_single_step_x():
    import torch.nn as nn
    from poet_torch import HeadAlignedPOETXLinear

    from src.optim.poet_layers import POETMegatronLinear, replace_linears_with_poet

    # q_proj is in the head-aligned set (head_side="out"); fc1 is not.
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 32, bias=False)  # hidden=16 -> heads*head_dim=32

    m = Attn()
    replace_linears_with_poet(
        m, block_count=1, init_type="none", extra_linear_types=(nn.Linear,),
        single_step_x=True, head_aligned_attn=True, head_dim=8,
        head_resid_block_count=2,
    )
    assert isinstance(m.q_proj, POETMegatronLinear)
    pl = m.q_proj.poet_linear
    assert isinstance(pl, HeadAlignedPOETXLinear)
    assert pl.head_side == "out"
    assert pl.block_size_out == 8 and pl.r_in == 2
```

> Mirror the existing head-aligned walk test's module/layer naming so `_HEAD_ALIGNED_SIDES` recognizes `q_proj`. Inspect `_HEAD_ALIGNED_SIDES` ([poet_layers.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py)) and reuse a name it maps to `"out"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -k head_aligned_poetx -v`
Expected: FAIL — `TypeError: replace_linears_with_poet() got an unexpected keyword argument 'head_resid_block_count'` (or builds the legacy `HeadAlignedPOETLinear`).

- [ ] **Step 3: Thread `head_resid_block_count` + branch the head construction**

In `src/optim/poet_layers.py`:

(a) Add `head_resid_block_count: int = 1` to the `replace_linears_with_poet` signature (next to `head_dim`).

(b) In the head-aligned construction branch (the `HeadAlignedPOETLinear` block near [poet_layers.py:251-267](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L251-L267)), build the POETX variant when `single_step_x` is set:

```python
                    head_side = _HEAD_ALIGNED_SIDES[name]
                    if single_step_x:
                        from poet_torch import HeadAlignedPOETXLinear

                        pl = HeadAlignedPOETXLinear(
                            in_features=in_f,
                            out_features=out_f,
                            head_side=head_side,
                            head_dim=head_dim,
                            head_resid_block_count=head_resid_block_count,
                            bias=has_bias,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                            parameterization=parameterization,
                            alternating=(single_step_x and lie_alternating),
                            alternate_every=alternate_every,
                        )
                    else:
                        from poet_torch import HeadAlignedPOETLinear

                        # ... existing legacy HeadAlignedPOETLinear construction ...
```

> `lie_alternating` and `alternate_every` are the walk params the prerequisite plan adds/uses for `POETXLinear(alternating=…)` (confirmed: the integrated plan names the new walk param `lie_alternating` and reuses the existing `alternate_every`). After construction, the existing `_copy_and_init_weight` + (for `single_step_x`) `bake_perms_into_weight()` must run for `HeadAlignedPOETXLinear` exactly as for `POETXLinear` — verify the `single_step_x` post-build `bake_perms_into_weight()` guard ([poet_layers.py:346-349](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L346-L349)) covers the head-aligned POETX branch (it should, since `HeadAlignedPOETXLinear` inherits `bake_perms_into_weight`).

- [ ] **Step 4: Thread the flag through the apply patch**

In `src/patches/poet_apply_to_model.py`, after the `head_dim` read, add:

```python
        head_resid_block_count = getattr(args, "poet_head_resid_block_count", 1)
```

and pass `head_resid_block_count=head_resid_block_count` into the `replace_linears_with_poet(...)` call (next to `head_dim=head_dim`).

- [ ] **Step 5: Run the walk test**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_poet_layers.py -k head_aligned_poetx -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/optim/poet_layers.py src/patches/poet_apply_to_model.py tests/unit/test_poet_layers.py
git commit -m "feat(poet): walk builds HeadAlignedPOETXLinear for attention under single_step_x"
```

---

## Task 5: CLI flag, validation, emit

**Files:**
- Modify: `launchers/pretrain_gpt_slm.py`
- Modify: `src/utils/megatron_args.py`
- Test: `tests/unit/test_megatron_args.py`

- [ ] **Step 1: Write the failing args test**

Append to `tests/unit/test_megatron_args.py`:

```python
def test_head_resid_block_count_emits_flag():
    from src.utils.megatron_args import _optimizer_args

    args = _optimizer_args(
        _poet_cfg(
            {
                "block_count": 1,
                "merge_period": 1,
                "parameterization": "cayley",
                "q_optimizer": "lie_ortho",
                "single_step_x": True,
                "head_aligned_attn": True,
                "head_resid_block_count": 4,
            }
        )
    )
    assert "--poet-head-resid-block-count" in args
    assert args[args.index("--poet-head-resid-block-count") + 1] == "4"


def test_head_resid_block_count_requires_head_aligned():
    import pytest

    from src.utils.megatron_args import _optimizer_args

    with pytest.raises(ValueError, match="head_resid_block_count"):
        _optimizer_args(
            _poet_cfg(
                {
                    "block_count": 1,
                    "merge_period": 1,
                    "parameterization": "cayley",
                    "q_optimizer": "lie_ortho",
                    "single_step_x": True,
                    "head_aligned_attn": False,
                    "head_resid_block_count": 4,
                }
            )
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k head_resid_block_count -v`
Expected: FAIL — flag not emitted.

- [ ] **Step 3: Add the CLI flag**

In `launchers/pretrain_gpt_slm.py`, near the other head-aligned flags (around the `--poet-head-aligned-attn` definition), add:

```python
    group.add_argument("--poet-head-resid-block-count", type=int, default=1)
```

- [ ] **Step 4: Validate + emit in megatron_args**

In `src/utils/megatron_args.py`, in the poet validation block (near the `head_aligned_attn` checks), add:

```python
        head_resid_bc = int(poet.get("head_resid_block_count", 1))
        if head_resid_bc != 1:
            if not poet.get("head_aligned_attn", False):
                raise ValueError(
                    "optim.poet.head_resid_block_count > 1 requires head_aligned_attn=true "
                    "(it controls the residual side of the head-aligned POETX layer)."
                )
            if not poet.get("single_step_x", False):
                raise ValueError(
                    "optim.poet.head_resid_block_count requires single_step_x=true "
                    "(the permuted-residual head layer is a POETX subclass)."
                )
```

and in the poet `poet_args` emit block (near the `--poet-head-aligned-attn` emit), add:

```python
        if int(poet.get("head_resid_block_count", 1)) != 1:
            poet_args.append("--poet-head-resid-block-count")
            poet_args.append(str(int(poet.get("head_resid_block_count", 1))))
```

> Also confirm `head_aligned_attn=true` + `single_step_x=true` is not blocked elsewhere in the poet validation (the legacy head layer historically rode `single_step_native`/`single_step_fast`). If a block exists, relax it to allow the POETX head path. Add a one-line test `head_aligned_attn=true` + `single_step_x=true` emits both flags without raising.

- [ ] **Step 5: Run the args tests**

Run: `PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "head_resid_block_count or head_aligned" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add launchers/pretrain_gpt_slm.py src/utils/megatron_args.py tests/unit/test_megatron_args.py
git commit -m "feat(poet): wire head_resid_block_count CLI/args for HeadAlignedPOETXLinear"
```

---

## Task 6: Experiment config, doc, launcher

**Files:**
- Create: `configs/experiments/optim/poet_lie_orth_head_alt.yaml`
- Create: `docs/experiments/poet_lie_orth_head_alt.md`
- Create: `scripts/train_poet_lie_orth_head_alt.sh`

- [ ] **Step 1: Create the experiment config**

Create `configs/experiments/optim/poet_lie_orth_head_alt.yaml` (champion alternating recipe + head-on POETX + permuted residual). Clone `configs/experiments/optim/poet_lie_orth.yaml`'s `experiment`/`optim` blocks and set:

```yaml
# @package _global_
# poet_lie_orth_head_alt: HeadAlignedPOETXLinear (head-on, POETX, permuted multi-block
# residual) on the alternating champion (lie_ortho + lie_alternating, val 3.5332 head-off).
# Tests whether a properly permuted residual side + alternating flips the head penalty.
experiment:
  name: poet_lie_orth_head_alt
  family: optim
  description: |
    Head-aligned attention on the POETX forward frame with a permuted multi-block
    residual side (HeadAlignedPOETXLinear), on top of the alternating champion.
  references: ["POET", "Muon", "Pion"]
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
  lr: 3.0e-3
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
    q_optimizer: lie_ortho
    lie_b1: 0.9
    lie_b2: 0.95
    lie_eps: 1.0e-8
    lie_v_mode: elementwise
    lie_ortho_c: 8
    lie_ortho_method: muon
    lie_ortho_ns_steps: 5
    lie_ortho_use_second_moment: false
    lie_ortho_distributed: true
    head_aligned_attn: true            # head ON, POETX-native (permuted residual)
    head_resid_block_count: 4          # >1 -> real permuted multi-block residual
    single_step_fast: true
    single_step_x: true                # POETX forward frame (required for the head POETX layer)
    lie_alternating: true              # the champion alternating optimizer
    lie_alternate_every: 1
    train_output_rotation: true

base:
  model:
    unfuse_qkv: true                   # head-aligned needs unfused q/k/v
    unfuse_fc1: true
```

- [ ] **Step 2: Create the experiment doc**

Create `docs/experiments/poet_lie_orth_head_alt.md`:

```markdown
# poet_lie_orth_head_alt

Head-aligned attention on the POETX forward frame with a **permuted multi-block
residual** side (`HeadAlignedPOETXLinear`), on top of the alternating champion
(`lie_ortho` + `lie_alternating`, val/loss 3.5332 head-off).

Hypothesis: head-alignment hurt (−0.014) partly because the legacy layer's residual
side is a single dense **perm-free** block. Giving the residual side a real Ψ +
multiple blocks (the POETX-native shape) + alternating may flip the head penalty.

- **Design:** docs/superpowers/specs/2026-06-08-head-aligned-poetx-permuted-resid-design.md
- **Baseline:** alternating champion `1ynrrimu` (head-off, 3.5332).
- **Knob:** `head_resid_block_count` (sweep, default 4).
```

- [ ] **Step 3: Create the launcher**

Create `scripts/train_poet_lie_orth_head_alt.sh` as a clone of `scripts/train_poet_lie_orth_alt_x.sh` (or `train_poet_lie_orth.sh`) with `experiment=optim/poet_lie_orth_head_alt`. Then `chmod +x scripts/train_poet_lie_orth_head_alt.sh`.

- [ ] **Step 4: Smoke-test config compose (dry-run, CPU)**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m launchers.train_megatron \
  base/family=llama3 base/scale=60m training_regime=ablation_40x base.model.seq_length=256 \
  scheduler=cosine_poet cluster=h100_de experiment=optim/poet_lie_orth_head_alt --dry-run 2>&1 | \
  grep -iE "head-resid-block-count|head-aligned|single-step-x|lie-alternating|error|valueerror" | head
```
Expected: emits `--poet-head-aligned-attn`, `--poet-head-resid-block-count 4`, `--poet-single-step-x`, `--poet-lie-alternating`; no `ValueError`.

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/optim/poet_lie_orth_head_alt.yaml docs/experiments/poet_lie_orth_head_alt.md scripts/train_poet_lie_orth_head_alt.sh
git commit -m "feat(poet): poet_lie_orth_head_alt experiment (head POETX permuted residual + alternating)"
```

---

## Task 7: Verification handoff (CPU suite + GPU commands)

> No new code. Run the CPU suite, then hand the GPU runs to the user.

- [ ] **Step 1: Run the new + adjacent CPU suite**

```bash
cd /lustre/fast/fast/zqiu/slm-research && PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_head_aligned_poetx.py \
  tests/unit/test_poetx_layer.py \
  tests/unit/test_poet_layers.py \
  tests/unit/test_alternating_poetx.py \
  tests/unit/test_poet_merge_step.py -v
```
Expected: all PASS. (The `transformer_engine` `.so` failure in `test_poet_layers::test_sharded_state_dict_*` is a known environmental failure on this box — ignore it.)

- [ ] **Step 2: Run the args tests**

```bash
cd /lustre/fast/fast/zqiu/slm-research && PYTHONPATH=$PWD/third_party /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args.py -k "head" -v
```
Expected: PASS.

- [ ] **Step 3: Hand the GPU runs to the user**

Provide these (do NOT launch). Primary arm + a `head_resid_block_count` sweep:

```bash
# Head ON, POETX permuted residual + alternating (resid_block_count=4):
codexlog poet_lie_orth_head_alt bash scripts/train_poet_lie_orth_head_alt.sh llama3

# Sweep the residual block count (e.g. 2 and num_heads):
codexlog poet_lie_orth_head_alt_bc2 bash scripts/train_poet_lie_orth_head_alt.sh llama3 \
  optim.poet.head_resid_block_count=2
codexlog poet_lie_orth_head_alt_bcH bash scripts/train_poet_lie_orth_head_alt.sh llama3 \
  optim.poet.head_resid_block_count=<num_heads>
```

Compare `val/loss` to the head-off alternating champion (3.5332, `1ynrrimu`). Update [POET_dev.md](/lustre/fast/fast/zqiu/slm-research/POET_dev.md) §2.1 (head row), §2.3, §2.5, §2.6 with the head × alternating result and whether the permuted residual flips the penalty.

- [ ] **Step 4: (Optional) Faithful-resid control**

If the verdict is ambiguous, run `head_resid_block_count=1` (single dense residual, perm redundant) + alternating as a control — this isolates "did the *permuted multi-block residual* help" from "did *alternating* help head."

---

## Self-Review

**Spec coverage:** spec Phase 1 (HeadAlignedPOETXLinear) → Tasks 1–3; walk wiring → Task 4; config/CLI/validation → Task 5; experiment → Task 6; verification → Task 7. Phase 2 sweep → Task 7 Step 3. Dependency on the integrated alternating-POETX plan stated in the header. No uncovered requirement.

**Placeholder scan:** the legacy `HeadAlignedPOETLinear` construction in Task 4 Step 3 is referenced as "existing … construction" rather than repeated — the engineer keeps the current branch verbatim and only adds the `single_step_x` branch above it; this is a *preserve-existing* instruction, not a placeholder. The `<num_heads>` token in Task 7 is a run-time value the user fills per model. All code steps contain real code.

**Type/name consistency:** `HeadAlignedPOETXLinear(in_features, out_features, *, head_side, head_dim, head_resid_block_count, bias, device, dtype, parameterization, alternating, alternate_every)` is used identically in Task 1 (def), Task 4 (walk call), and Task 6 (config keys → CLI → args). `head_resid_block_count` is the single name across layer, walk, apply-patch, CLI (`--poet-head-resid-block-count`), and YAML. `alternating`/`alternate_every` reuse the prerequisite plan's POETXLinear params.

**Open dependency:** Task 4 Step 3 reuses the prerequisite plan's walk params — confirmed names are `lie_alternating` (added by the integrated plan's Task 5) and `alternate_every` (already on the walk), plus POETXLinear's `alternating` attribute. The integrated plan must be implemented first.
