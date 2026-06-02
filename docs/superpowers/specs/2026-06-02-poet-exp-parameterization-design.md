# POET `exp` Orthogonalization Parameterization — Design

**Date:** 2026-06-02
**Status:** Approved design, ready for implementation plan
**Scope:** Parameterization only (the `G = exp(Q)` map). The §6 Muon-style
angle-scaling *optimizer* update rule is explicitly **out of scope** here and
tracked separately.

## Motivation

POET parameterizes each orthogonal block as `G = f(Q)` with `Q` skew-symmetric.
The current `f` is a truncated Cayley–Neumann polynomial (degree-4 in `Q`),
implemented as the `poet::cayley` Triton op. Per
[`docs/poet_exp_angle_math.md`](../../poet_exp_angle_math.md), using the **true
matrix exponential** `G = exp(Q)` instead removes every Cayley correction term:

- orthogonality is **exact** for any `Q` (no `‖Q‖ < 1` ceiling, no truncation error);
- the singular values of `Q` are **exactly** the rotation angles of `G`
  (Cayley gives `2·arctan(θ)`; the factor-of-2 and `arctan` vanish);
- at the reset point `Q = 0` the generator→rotation map is exactly
  identity-on-angles.

This design adds `exp` as a **config-selectable** alternative to `cayley`,
leaving the Cayley path byte-for-byte unchanged and the default behavior
identical to today.

## Decision: exact `torch.linalg.matrix_exp`

Of the three options the math doc lists (exact `exp`, truncated-`exp`
polynomial, custom Triton `exp` kernel), we implement **exact
`torch.linalg.matrix_exp`**:

- Exactly orthogonal `R` for any `Q`.
- PyTorch provides the Fréchet-derivative backward for free — **no custom
  kernel, no change to `poet_ops.py`**.
- Matches §1–§5 of the math doc precisely.

Trade-offs accepted: a more expensive backward (Fréchet derivative, ~O(8·b³)
per block) and a `torch.compile` compatibility question for the
orthogonalization sub-op (see §4 below). Both are acceptable because
R-construction is `O(b³)` per block, amortized over the entire microbatch,
while the per-token cost path is unchanged.

## Architecture

### The swap point

Orthogonalization is cleanly isolated. Today the rotation `R` is built by
`get_weight_poet_decoupled(...)` →
[`torch.ops.poet.cayley(Q)`](../../../third_party/poet_torch/poet_layer.py#L238-L255),
and the resulting `R_in / R_out` blocks are consumed by
parameterization-**agnostic** downstream code
(`chain_layer_x_fast_decoupled`, `block_diag_lr_matmul_decoupled`,
`chain_layer_*_mem_o2_decoupled`). The entire change is: **produce `R` a
different way; leave every consumer untouched.**

### Unit 1 — the `exp` rotation builder

New function in
[`third_party/poet_torch/poet_layer.py`](../../../third_party/poet_torch/poet_layer.py),
a sibling of `get_weight_poet_decoupled`:

```python
def get_weight_poet_decoupled_exp(oft_R_in, oft_R_out,
                                  block_size_in, block_size_out,
                                  rows_in, cols_in, rows_out, cols_out):
    """Decoupled matrix-exponential orthogonalization: R = exp(Q)."""
    Q_in  = pytorch_skew_symmetric(oft_R_in,  block_size_in,  rows_in,  cols_in)
    Q_out = pytorch_skew_symmetric(oft_R_out, block_size_out, rows_out, cols_out)
    R_in  = torch.linalg.matrix_exp(Q_in.float()).to(Q_in.dtype)
    R_out = torch.linalg.matrix_exp(Q_out.float()).to(Q_out.dtype)
    return R_out, R_in   # (R_out, R_in) ordering matches get_weight_poet_decoupled
```

- **fp32 compute, cast back:** `matrix_exp` is numerically delicate and
  unreliable in bf16/fp16; compute in fp32 then cast to the param dtype.
  Autograd flows through the cast and through `matrix_exp`, so gradients return
  in the param dtype with no custom backward.
- Reuses the existing `pytorch_skew_symmetric` skew construction unchanged.
- **What it does:** maps two batches of skew generators to two batches of exactly
  orthogonal blocks. **Depends on:** `pytorch_skew_symmetric`,
  `torch.linalg.matrix_exp`. **Interface:** identical signature/return ordering
  to `get_weight_poet_decoupled`, so it is a drop-in for the dispatch.

### Unit 2 — single dispatch point in `POETLinear`

So that **forward, merge, and the ΔW-spec estimator all use the same map**,
route every R-construction through one helper on
[`POETLinear`](../../../third_party/poet_torch/poet_layer.py#L425):

```python
def _build_R(self, oft_in, oft_out):
    if self.parameterization == "exp":
        return get_weight_poet_decoupled_exp(
            oft_in, oft_out, self.block_size_in, self.block_size_out,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out)
    return get_weight_poet_decoupled(
        oft_in, oft_out, self.block_size_in, self.block_size_out,
        self.rows_in, self.cols_in, self.rows_out, self.cols_out)
```

- `__init__` gains `parameterization: str = "cayley"`, validated against
  `{"cayley", "exp"}`, stored as `self.parameterization`.
- [`_merge_R`](../../../third_party/poet_torch/poet_layer.py#L531) calls
  `self._build_R(self.oft_R_in, self.oft_R_out)`.
- [`estimate_poet_delta_weff_spec`](../../../third_party/poet_torch/poet_layer.py#L1040)
  routes its internal `_R(...)` through the module's parameterization
  (reads `poet_module.parameterization`).

### Unit 3 — forward integration and the `torch.compile` question

The hot forward
[`forward_core_decoupled`](../../../third_party/poet_torch/poet_layer.py#L350-L388)
is `@torch.compile(fullgraph=True)` and builds `R` *inside* the compiled
region. `matrix_exp` — especially its backward — may not survive
`fullgraph=True`.

- **The Cayley path stays exactly as-is.** Zero regression risk to the tuned
  kernel path. `POETLinear.forward` branches:
  `if self.parameterization == "exp": return forward_core_decoupled_exp(...)`
  else the current `forward_core_decoupled(...)` call.
- The `exp` path gets its **own** forward `forward_core_decoupled_exp`.
- **Verification gate (decided in the plan, not assumed now):** first *attempt*
  keeping `forward_core_decoupled_exp` under `@torch.compile(fullgraph=True)`
  calling `get_weight_poet_decoupled_exp`. If `matrix_exp` fwd+bwd compiles
  cleanly, keep it compiled. **Committed fallback** if it does not: build `R`
  **eagerly** (`matrix_exp`, an `O(b³)`-per-block op amortized over the whole
  microbatch) and feed `R_in / R_out` into the existing
  `chain_layer_x_fast_decoupled` (fast, default) / `*_mem_o2_decoupled`
  (mem-efficient) consumers for the token-heavy part. Either way the per-token
  cost path is unchanged.

### Unit 4 — config flag threaded through the existing plumbing

`optim.poet.parameterization: "cayley" | "exp"`, default `"cayley"` so existing
runs are identical. Rides the same chain every other `poet_*` flag uses:

| Layer | File | Change |
|---|---|---|
| Hydra config | [`configs/experiments/optim/poet.yaml`](../../../configs/experiments/optim/poet.yaml#L43-L61) | add `parameterization: cayley` with a comment |
| config→CLI | [`src/utils/megatron_args.py`](../../../src/utils/megatron_args.py#L235-L251) | emit `--poet-parameterization`, `poet.get("parameterization", "cayley")` |
| arg registration | [`launchers/pretrain_gpt_slm.py`](../../../launchers/pretrain_gpt_slm.py#L49-L53) | `add_argument("--poet-parameterization", choices=["cayley","exp"], default="cayley")` |
| model-build patch | [`src/patches/poet_apply_to_model.py`](../../../src/patches/poet_apply_to_model.py#L62-L73) | read `args.poet_parameterization`, pass to `replace_linears_with_poet` |
| replacement helper | [`src/optim/poet_layers.py`](../../../src/optim/poet_layers.py#L110-L200) | new kwarg `parameterization="cayley"` → `POETLinear(parameterization=...)` (both the `none` and cached construction branches) |
| layer | [`third_party/poet_torch/poet_layer.py`](../../../third_party/poet_torch/poet_layer.py#L442) | `__init__` arg, `self.parameterization`, dispatch (Units 1–3) |

## Testing

All CPU-runnable (`matrix_exp` works on CPU fp32; small blocks).

- **Orthogonality (§1):** `R @ Rᵀ ≈ I` to ~1e-6 for random skew `Q`; assert it is
  *tighter* than the Cayley–Neumann truncation at matched `Q`.
- **Angle identity (§3):** single 2×2 block with angle θ →
  `R == [[cosθ, −sinθ], [sinθ, cosθ]]` (no factor-of-2, unlike Cayley's
  `2·arctan θ`).
- **gradcheck** through `_build_R(parameterization="exp")` on a small block
  (fp64 input).
- **Dispatch/plumbing:** `POETLinear(parameterization="exp")` builds orthogonal
  `R`; `parameterization="cayley"` is unchanged; `megatron_args` emits the flag;
  the launcher parses it; default is `cayley`.
- **Merge consistency:** `merge_then_reinitialize` under `exp` rotates `W`
  correctly (compare against an independent `exp`-based reconstruction) and zeros
  `oft_R`.

CPU tests are run locally before claiming completion. No GPU/training runs are
performed as part of this work.

## Out of scope (YAGNI)

- The §6 Muon-style angle-scaling optimizer update (`θ_step` rescaling /
  gradient orthogonalization). Tracked separately.
- Truncated-`exp` polynomial and custom-Triton-`exp`-kernel variants.
- `exp` support for `QPOETLinear` (int8) and `POETLinearNeurips` — the active
  path is the decoupled `POETLinear`.
- `cache_mode != "none"` (a documented dead-end) does not get `exp` support
  unless explicitly requested.

## Risks / open items

- **`torch.compile(fullgraph=True)` + `matrix_exp`** — resolved by the verify-then-
  fallback gate in Unit 3; the fallback (eager R + compiled chain) is fully
  specified, so this cannot block delivery.
- **Backward cost** — Fréchet-derivative backward is heavier than the Cayley
  kernel; acceptable because R-construction is amortized per microbatch. Not a
  correctness risk.
- **Working-tree state** — `poet_layer.py` and `poet.yaml` currently carry
  unrelated uncommitted edits (separate Muon-Q work); implementation edits layer
  on top without reverting them, and only files belonging to this change are
  committed.
