# Alternating single-side POETX layer (`AlternatingPOETXLinear`)

**Date:** 2026-06-08
**Status:** Design approved, ready for implementation plan
**Related:** [POET_dev.md](/lustre/fast/fast/zqiu/slm-research/POET_dev.md) (best POET = `dwynpk9y`, head-off, val/loss 3.5528); supersedes the *speed* intent of the optimizer-gate [alternating spec](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-03-poet-lie-alternating-design.md); builds on the POETX layer ([poetx_layer.py](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_layer.py), [poetx_ops.py](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_ops.py))

---

## 1. Goal

Build a **dedicated POETX layer that trains only one rotation side per step** and
**actually short-circuits the frozen side's compute** — the Cayley build, the
weight-fold, and the backward rotation-gradient. The active side alternates
(`oft_R_out` on even steps, `oft_R_in` on odd, switchable every
`lie_alternate_every` steps), exactly like the existing convention.

Two outcomes, both wanted:

- **Speed.** Halve the per-step POET-specific (O(d³)) machinery. Small at 60m,
  but a visible step-time win at Kimi scale where d³ is a meaningful fraction of
  the step.
- **Quality ablation.** Does "one rotation per step" help or hurt loss in the new
  champion regime (`q_optimizer=lie_ortho`, head-off, lr 3e-3, c=8)? This is a
  genuine optimizer change (see §3), so it needs its own datapoint.

## 2. Why the existing alternating gives no speedup, and what *can* be saved

The existing `lie_alternating` ([poet_lie_orth.py:144-145](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L144-L145),
[poet_lie_momentum.py:119-161](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L119-L161))
is an **optimizer-side write-gate only**: both gradients are computed, both
momenta updated, the merge folds both sides — only the inactive side's *write* is
skipped. ~0 wall-clock/memory win. (The 2026-06-03 spec says so: "optimizer-local,
no merge / W / gradient changes".)

At the champion's `merge_period=1`, the rotation is **identity at forward time**
(POETX's premise — validated at [megatron_args.py:287-294](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L287-L294)).
So the per-step cost splits into two buckets:

| Bucket | Terms | Scales as | Side-specific? |
|---|---|---|---|
| Dense GEMMs (the floor) | forward `y=x·Wxᵀ`, backward `grad_x=grad_y·Wx`, activation outer-product `G=xᵀ·grad_y` | N·d² | **No** — needed whichever side trains |
| POET machinery | Cayley build (×2 sides), weight-fold half (×2), backward rotation-grad `M_in`/`M_out` (×2) | d³ | **Yes** — one side per active step |

A single-side layer **cannot** touch the dense bucket (so no "2×"), but **can**
drop the frozen side's entire d³ contribution. At 60m (d≈512, N≈32k
tokens/microbatch) the d³ bucket is <10% of the layer; at Kimi (d≈7k) it grows
toward ~⅓, so halving it is a real step-time win. The Cayley build is the single
largest d³ chunk (iterative kernel, several d×d matmuls), so most of the saving is
in the **merge**, not the backward.

## 3. The momentum decision (resolved: true single-side)

Skipping the frozen side's backward means its gradient is never computed, so its
momentum **cannot** advance that step. This is a deliberate, principled choice —
"true single-side":

- Each side's first-moment momentum EMAs **only its own active-step gradients**.
- This is the *only correct* behavior once the frozen gradient is dropped: feeding
  momentum a zero gradient would wrongly decay it.
- It is a **different optimizer** than today's `lie_alternating` (which keeps
  both-side momentum per Pion App. D.1). That difference is the ablation question.

We do **not** build a "momentum-preserving" mode into the new layer (YAGNI). The
existing slow `lie_alternating` already embodies both-side-momentum dynamics and
stays available unchanged as the comparison arm.

## 4. Design decisions (resolved)

| Decision | Choice | Rationale |
|---|---|---|
| Layer | New `AlternatingPOETXLinear`, sibling of `POETXLinear` | One home for active-side state + single-side backward; forward unchanged. |
| Forward | Bare GEMM `y=x·Wxᵀ`, **unchanged** (inherit) | Side-agnostic at `merge_period=1`. |
| Momentum | **True single-side** (frozen side's momentum does not advance) | §3. |
| Active-side source | Global **training iteration**: `"out" if (iter // alternate_every) % 2 == 0 else "in"` | Advances once per optimizer step → correct under grad accumulation (forward runs K×/step, all same side). Unifies layer + optimizer + merge. |
| Side schedule | `out` even / `in` odd, period `lie_alternate_every` | Match existing convention. |
| Fold-active-only | In the **merge driver** ([poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py)) | Keep active-side logic in one place. |
| Selector flag | `optim.poet.single_step_x_alternating: bool` (default false) | Builds on existing `single_step_x` plumbing; one flag = the whole experiment. |
| Granularity | reuse `optim.poet.lie_alternate_every` (default 1) | No new knob. |

**Resume caveat:** active-side derives from the live Megatron `iteration`, so —
unlike the in-memory `_alt_step` counter — a mid-run resume keeps correct parity.
(The optimizer's private `_alt_step` is switched to read this shared iteration so
optimizer, layer, and merge cannot drift.)

## 5. The layer (`AlternatingPOETXLinear`)

In [poetx_layer.py](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_layer.py),
sibling of `POETXLinear`:

- **State:** identical — forward-frame `weight`, separate `oft_R_in`/`oft_R_out`,
  perms, triu row/col index buffers. No new persistent buffers.
- **Forward:** reads the active side from the shared iteration accessor, stamps it
  into a new autograd Function, otherwise the same bare GEMM as
  [poetx_layer.py:106-113](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_layer.py#L106-L113).
- **Build/merge hooks:** recognized by the merge driver's widened `isinstance`
  (the POETX entry already exists at [poet_merge_step.py:307](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L307)).

## 6. Single-side backward

New autograd Function in [poetx_ops.py](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_ops.py),
sibling of `POETXSingleStepFunction` ([poetx_ops.py:48-67](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_ops.py#L48-L67)),
taking `active_side` as a static (non-tensor) arg:

- `grad_x = grad_y·Wx` — always (upstream gradient).
- `active == "in"`  → `G = xᵀ·grad_y`; `M_in = conj(G·Wx, perm_in_inv)` →
  `grad_oft_R_in`; return `None` for `grad_oft_R_out`.
- `active == "out"` → `G = xᵀ·grad_y`; `M_out = conj(Wx·G, perm_out_inv)` →
  `grad_oft_R_out`; return `None` for `grad_oft_R_in`.

`G` stays (the active `M` needs it). The single skipped d³ GEMM is the frozen
side's `M`. The frozen `oft_R`'s `main_grad` therefore stays exactly 0 for the
step.

## 7. Optimizer: true single-side momentum

In the `lie_ortho` / `lie_algebra` alternating path, move the existing `continue`
so the frozen side skips **both** its momentum update and its write (today momentum
updates first, then only the write is skipped —
[poet_lie_momentum.py:150-160](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_momentum.py#L150-L160),
[poet_lie_orth.py:144-145](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth.py#L144-L145)).
With the frozen side producing no gradient (§6), this is the only correct path.

This single-side-momentum behavior is gated on the new layer being active
(`single_step_x_alternating=true`); the existing both-side-momentum
`lie_alternating` path is left untouched for the comparison arm. The side-tagged
param-group machinery (`_split_poet_lie_params`, `_build_lie_param_groups`) already
exists and composes with POETX's separately-named `oft_R_in`/`oft_R_out` params —
no change needed there.

## 8. Merge: fold only the active side

In the merge driver ([poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py),
`_merge_layers` / `_build_R_batched`), for `AlternatingPOETXLinear` layers: read
the same active-side signal and fold **only** the active side — skip the frozen
side's Cayley build and its fold half. Reuse POETX's round-trip fold
([poetx_layer.py:121-140](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poetx_layer.py#L121-L140))
restricted to the active side. This is a provable no-op for the frozen side
(`oft_R`≡0 ⇒ R=I), so it is exact. The champion's `reinit_period=-1` ⇒ no perm
resample, keeping the active-only fold simple.

## 9. Config / CLI / validation

| File | Change |
|---|---|
| [pretrain_gpt_slm.py](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py) | `--poet-single-step-x-alternating` (store_true) |
| [megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py) | emit the flag when `single_step_x_alternating` true; **validate**: requires `merge_period=1`, `parameterization=cayley`, `q_optimizer∈{lie_ortho,lie_algebra}`, `single_step_x` selected, `head_aligned_attn=false` |
| [poet_apply_to_model.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py) / [poet_layers.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py) | select `AlternatingPOETXLinear` in the walk when the flag is set |
| [poet_optimizer_setup.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py) | thread `single_step_x_alternating` to config |
| [poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py) | active-only fold for the new layer |
| `configs/experiments/optim/poet_lie_orth_alt_x.yaml` (new) | clone [poet_lie_orth.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet_lie_orth.yaml) + champion knobs (lr 3e-3, c=8, distributed, head-off) + `single_step_x=true`, `single_step_x_alternating=true` |
| `docs/experiments/poet_lie_orth_alt_x.md` (new) | pre-commit hook requires a matching doc |
| `scripts/train_poet_lie_orth_alt_x.sh` (new) | clone [train_poet_lie_orth.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_poet_lie_orth.sh) → the new experiment |

Config keys under `optim.poet`: `single_step_x_alternating: false` (new),
reusing `single_step_x` and `lie_alternate_every`.

## 10. Testing & verification

**CPU-testable (I run these):**

1. **Alternation OFF ⇒ vanilla POETX.** With `single_step_x_alternating=false` (or
   the new layer driven with both sides active), the new layer is bit-identical to
   `POETXLinear` forward+backward.
2. **Active-side gradient matches.** For each active side, the new backward's `M`
   equals the both-sides closed form for that side; the frozen side's grad is
   `None`.
3. **Grad-accumulation consistency.** Over K microbatches in a step, active side is
   constant and the frozen `main_grad` is exactly 0.
4. **Merge active-only == full merge.** Folding only the active side equals the
   full merge when the other side is identity (numerical parity).
5. **Optimizer true single-side.** Frozen side's momentum is unchanged across an
   inactive step (no decay, no update); active side updates as usual.
6. **Arg/emit/validate/yaml/script** tests mirroring the existing `single_step_x`
   and `lie_alternating` plumbing tests (flag round-trips; validation rejects
   `merge_period≠1`, non-cayley, head-aligned).

**Perf microbench (I run on CPU/GPU as available):** d³-work per step, both-sides
POETX vs `AlternatingPOETXLinear`, swept over d (512 → ~7k) to show the win growing
with scale.

**GPU (user runs):** 60m/40tpp quality run vs champion `dwynpk9y` (is single-side
≥ both-sides?) + step-time delta; later a Kimi-scale step-time check. I supply
exact `codexlog` commands; I do not launch GPU jobs.

## 11. Out of scope

- The dense N·d² GEMM path (unchanged — it's the floor).
- Head-aligned attention (a different layer; champion is head-off anyway).
- Windowed live-rotation alternating (approach B — reverts off `merge_period=1`,
  net slower).
- A momentum-preserving mode in the new layer (existing `lie_alternating` covers
  that arm).
- Resume-exactness of the optimizer beyond switching `_alt_step` to the shared
  iteration.
