# POET × Pion — Head-Aligned Attention Rotation (design)

**Date:** 2026-06-04
**Status:** Design approved, ready for implementation plan
**Related:**
[poetx_pion_pipeline.md](/lustre/fast/fast/zqiu/slm-research/docs/poetx_pion_pipeline.md),
[lie-momentum design](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-03-poet-lie-momentum-design.md),
[implementation status](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-04-poet-pion-implementation-status.md),
[decoupled-block-count plan](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/plans/2026-05-27-poet-decoupled-block-count.md),
Pion paper App. D.1 (per-head application).

---

## 1. Goal

Give POET's attention projections a **head-aligned rotation on their head-structured
side**: instead of one dense `d×d` rotation that can mix across heads, use a
**block-diagonal rotation with one `head_dim`-sized block per head**, with a
**fixed identity permutation** (block *j* = head *j*) that is **never resampled**.
The other (residual-stream) side keeps a **normal POET rotation** — same
`block_size`/`block_count`/permutation machinery as any other layer.

This realizes Pion's per-head application (App. D.1) inside POET, and rests on a
structural observation: POET's permutation exists to overcome the *sparsity* of an
arbitrary block-diagonal rotation on an **unstructured** matrix (it reshuffles which
coordinates share a block so the full rotation group is reachable over merges). On
the **head-structured** side that sparsity is not a defect — heads are the natural,
semantically-independent units, and mixing across them is neither needed nor
wanted. So on that side the block-diagonal structure *is* the target, not an
approximation of a denser one, and **no permutation is required**. On the residual
side there is no such structure, so if it is blocked it still wants permutation.

The payoff is both modeling (per-head, no cross-head mixing) and efficiency
(block-diagonal `head_dim` rotation instead of dense; perm-free head side). It is an
**opt-in, attention-only** change; MLP and everything else are untouched.

## 2. Background

### 2.1 POET block structure (current)

POET's rotation is block-diagonal: `oft_R_in`/`oft_R_out` are stored as
`(n_blocks, n_elems)` skew vectors, lifted per block to `(n_blocks, b, b)` skew
matrices and exponentiated (Cayley/exp) into block-diagonal `R_in`/`R_out`. The
effective weight is `W_eff = R_out · W · R_inᵀ`. POETLinear already stores
`block_size_in` and `block_size_out` **independently** and has a decoupled forward
[`get_weight_poet_decoupled(oft_R_in, oft_R_out, block_size_in, block_size_out, …)`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L238),
so asymmetric per-side block sizes are an already-supported code path. The stock
**constructor**, however, only takes a single symmetric spec — `bsz` (both sides
equal) or `block_count` (both sides `dim//block_count`)
([poet_layer.py:516-520](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L516)).

Permutation `perm_in/out` is passed into the forward kernel **every call**
([poet_layer.py:23](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L23)) (a gather/scatter), and optionally
**resampled** at merge (`merge_then_reinitialize(reinit_perm=True)`).

### 2.2 The dev model head config

The 60m dev model ([60m.yaml](/lustre/fast/fast/zqiu/slm-research/configs/base/scale/60m.yaml#L34)) is MHA:
`hidden_size=512, num_attention_heads=8, num_query_groups=8, head_dim=64`. So
q/k/v/o are all `512×512`, and head-aligned blocking gives `head_count=8` blocks of
`head_dim=64`. `num_attention_heads`, `num_query_groups`, `head_dim` are all on the
attention module config, so the policy can read them directly. (Design supports GQA
generally; at 60m groups == heads.)

### 2.3 Why a `poet_torch` subclass

The new layer reuses POETLinear's forward (`get_weight_poet_decoupled`), fused
kernels, and fold/merge logic — only the per-side block spec and the head-side
permutation policy change. So it lives in `third_party/poet_torch/` as a
**`POETLinear` subclass**, not in `src/optim/`.

## 3. Design decisions (resolved)

| Decision | Choice | Rationale |
|---|---|---|
| Sides trained | **Both** (`oft_R_in` and `oft_R_out`) | Head-alignment is a per-*side* property; the residual side must still train normally. |
| Head side | q/k/v → `out`; o → `in` | The head-structured axis (rows for q/k/v, cols for o). |
| Head-side block size | `head_dim` (`n_blocks = head_count`) | One rotation block per head; no cross-head mixing. |
| Head-side permutation | **Identity, fixed, never resampled** | Heads are natural units → no sparsity → no permutation; lets the forward skip the gather. |
| Residual side | Normal POET rotation: `block_size`/`block_count`/perm from the run | The residual stream is unstructured; keep full POET expressivity (and perm if blocked). |
| Residual-side permutation | Normal by default; **optional off-switch flag** | Free at `block_count=1`; earns its keep at `block_count>1`; flag lets us ablate. |
| Granularity (GQA) | q,o → `num_attention_heads`; k,v → `num_query_groups` | Head count of each projection's head side. |
| Scope | **Attention-only**, opt-in flag | MLP/everything else unchanged; reversible; A/B-able. |
| Step-size normalization | **Per-block RMS** (prerequisite, folded in) | `head_count>1` breaks the current single global-α RMS; needs per-block α. |

## 4. The new layer: `HeadAlignedPOETLinear`

A `POETLinear` subclass in `third_party/poet_torch/`. Holds the frozen base weight
`W (out, in)` and **two** trainable skew params, both updated every step:

- **Head side** (q/k/v: `out`; o: `in`): `block_size = head_dim`,
  `n_blocks = head_count`. Permutation pinned to **identity** (block *j* occupies
  rows/cols `[j·head_dim : (j+1)·head_dim]`, which are already contiguous per head
  after de-interleave in [unfuse_linears.py](/lustre/fast/fast/zqiu/slm-research/src/model/unfuse_linears.py)). The forward uses a
  **perm-free path** (no gather) on this side.
- **Residual side** (the other one): `block_size = dim // block_count` (or
  `block_size`) — the run's normal spec — with normal permutation (resampled per
  the reinit schedule), unless the off-switch flag disables it.

**Constructor (the one new entry point):** accepts the asymmetric spec — head side
`= head_dim`, residual side `= (block_size | block_count)` — and sets
`block_size_in`/`block_size_out` + the per-side permutation policy accordingly. This
is the only piece the stock POETLinear constructor can't express
([poet_layer.py:526](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L526)).

**forward:** unchanged in spirit — `W_eff = R_out · W · R_inᵀ` via
`get_weight_poet_decoupled` with the two block sizes; the head side passes identity
(skippable) perm, the residual side its normal perm. Orthogonal block rotations
preserve `W`'s singular values, so the spectrum-preservation property holds.

**merge_then_reinitialize:** fold **both** `R_in`/`R_out` into `W` and zero both
`oft_R`; resample **only the residual-side** permutation (and only if enabled); the
head-side Ψ stays identity. (`reinit_perm` thus controls the residual side alone.)

## 5. Per-layer apply policy

New flag `--poet-head-aligned-attn` (config `optim.poet.head_aligned_attn: true`).
When on, the apply patch ([poet_apply_to_model.py:57](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py#L57) /
[poet_layers.py:110](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L110)) routes by post-unfuse layer name:

| Layer | Replacement | head side | head_count |
|---|---|---|---|
| `linear_q` | `HeadAlignedPOETLinear` | `out` | `num_attention_heads` |
| `linear_k`, `linear_v` | `HeadAlignedPOETLinear` | `out` | `num_query_groups` |
| `linear_proj` (o) | `HeadAlignedPOETLinear` | `in` | `num_attention_heads` |
| `linear_fc1_gate/up`, `linear_fc2`, others | **stock `POETLinear`** | — | existing `block_count` |

Head counts and `head_dim` are read from the attention module config. The residual
side of each head-aligned layer uses the run's normal `block_size`/`block_count`.
MLP and all non-attention linears are replaced exactly as today — purely additive.

## 6. Optimizer integration

Each `HeadAlignedPOETLinear` exposes the usual `oft_R_in`/`oft_R_out` names, so
[`_split_poet_lie_params`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L27) buckets them into the
`in`/`out` sides with **no optimizer changes** — attention layers contribute to both
sides exactly like MLP. Alternating flips both sides as before.

**Per-block RMS fix (prerequisite, folded into this work).** The head side has
`head_count>1` blocks, so the current single global-α RMS
([poet_lie_momentum.py:162-169](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L162)) would
lump all heads under one `α` and break the dimension-consistent per-plane angle.
Replace it with a **per-block** computation:

```python
bsz        = block_size_from_nelems(A.shape[1])
dim_const  = bsz ** 0.5                                  # √(block_size), not √d
block_norm = torch.linalg.norm(A, dim=1, keepdim=True)   # (n_blocks, 1)
alpha      = self.rms_c * dim_const / (block_norm + eps) # (n_blocks, 1)
A          = A * alpha
```

Provably identical at `block_count=1` (`n_blocks=1` → per-block == global,
`√(1·b)=√b`), so it does **not** disturb existing runs; it normalizes each head's
(and each residual block's) per-plane angle independently when `n_blocks>1`.

## 7. Merge integration

[`_run_merge`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L235) broadens its `isinstance`
check to also handle the head-aligned inner: call its `merge_then_reinitialize()`
(resampling only the residual perm, gated by the residual-perm flag and
`reinit_period`), and broadcast `(oft_R_in, oft_R_out, weight, residual perm
buffers)` — the head-side Ψ is a fixed identity buffer, broadcast harmlessly or
skipped. [`_reset_vanilla_oft_state`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L153) already keys
off the `"oft_R"` substring in the param name, so the master-value zero (anti
spring-back) works unchanged.

## 8. Config, flags, experiment, script

- `--poet-head-aligned-attn` (store_true) + `--poet-head-resid-perm`/
  `--poet-head-no-resid-perm` (the residual off-switch); threaded through
  [megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py) →
  [poet_optimizer_setup.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py) + the apply patch.
- New experiment `configs/experiments/optim/poet_lie_head.yaml` = `poet_lie` +
  `head_aligned_attn: true`, `block_count: 1` (residual stays dense in dev),
  `merge_period: 1`, `reinit_period: -1`, `lie_v_mode: elementwise`, RMS optional.
- New `scripts/train_poet_lie_head.sh` (60m default) + required
  `docs/experiments/poet_lie_head.md`.

## 9. Performance characteristics

Where the head side gets cheaper (per attention projection):

- **Rotation construction (dominant):** dense head side `O(d³)` Cayley/exp →
  `head_count · O(head_dim³) = O(d³/head_count²)` (≈ 64× at 8 heads).
- **Permutation:** identity Ψ → the subclass's custom head-side forward omits the
  permutation gather entirely (it is a mathematical no-op) and never resamples.
- **Optimizer state:** head-side `oft_R` + `lie_m`/`lie_v` shrink from `O(d²)`
  (dense block) to `O(d · head_dim)`.

Caveat: at `block_count=1` the **residual side stays dense** `O(d³)` and dominates
the layer cost in dev. The full per-layer speedup needs the residual side blocked
(`block_count>1`) too — at which point its permutation is doing real work (or is
ablated via the off-switch). This spec makes the head side cheap and perm-free; the
residual side's efficiency is the existing `block_count` knob.

## 10. Out of scope (deferred)

- Frozen-side / single-sided attention variants (we train both sides).
- Re-blocking MLP by any head-like unit (no natural meaning).
- Independent residual `block_count` per attention projection (uses the global one).
- μP spectral / Newton–Schulz on the generators, second-order exp — separate specs.

## 11. Testing & verification

CPU-testable (no GPU/Megatron runtime), with a plain `nn.Linear` swapped for the
head-aligned layer:

1. **Forward identity:** `oft_R_in = oft_R_out = 0 ⇒ W_eff == W` exactly.
2. **Spectrum preservation:** random `oft_R` ⇒ singular values of `W_eff` == those
   of `W` (orthogonal block rotations).
3. **No cross-head mixing:** perturbing head *j*'s skew block changes only head *j*'s
   rows/cols of `W_eff`; all other heads' entries are bit-identical.
4. **Both sides train:** a gradient on the loss produces nonzero `oft_R_in.grad`
   **and** `oft_R_out.grad`.
5. **Merge fold + perm policy:** `forward(merge(W); oft_R=0) == forward(W; oft_R)` to
   O(η²); both `oft_R` zeroed; head-side Ψ unchanged; residual Ψ resampled only when
   `reinit_period`/flag say so.
6. **Per-block RMS:** identical to current at `block_count=1`; per-block-consistent
   angle at `block_count>1`.
7. **Apply policy / GQA:** q/o use `num_attention_heads`, k/v use
   `num_query_groups`; MLP gets stock `POETLinear`; flag off ⇒ no behavior change.
8. **Arg/flag translation, experiment load, script dry-run, `py_compile`/`ruff`.**

GPU run (user's): 60m dev, `experiment=optim/poet_lie_head`, ablated against
`poet_lie` (dense both sides) and `poet_lie_rms`.
