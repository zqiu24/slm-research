# POET × Pion — Lie-Algebra Momentum (increment 1)

**Date:** 2026-06-03
**Status:** Design approved, ready for implementation plan
**Related:** [docs/poetx_pion_pipeline.md](/lustre/fast/fast/zqiu/slm-research/docs/poetx_pion_pipeline.md) §2–§3, §9 step 1; Pion paper (arXiv 2605.12492) Algorithm 1 (Lie-Algebra variant); [poet0 spec](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-03-poet0-single-step-design.md)

---

## 1. Goal

Import **Pion's Lie-algebra momentum** (Pion paper §2.4.2 / Algorithm 1; pipeline
doc §2–§3) into the POET single-step stack, as a new optimizer backend selected
by `optim.poet.q_optimizer: lie_algebra`. This is **increment 1** of the
POET-X × Pion pipeline — §9 step 1: "Lie-algebra momentum *alone*". It replaces
the stock Megatron-Adam update on POET's skew generators `oft_R` with a
first-and-second-moment momentum accumulated **in the Lie algebra** (on the
skew-symmetric tangent gradient), persisting across the per-step merge.

It deliberately imports **only** §2 (tangent gradient) + §3 (Lie momentum).
RMS-α step scaling (§4), low-order Cayley (§5), alternating single-sided (§6),
and the exact/block-diagonal tangent gradient (§2 routes 2/3) are **deferred** to
later increments, one knob at a time, per §9 steps 2–5.

The value: it isolates whether momentum *in the Lie algebra* (vs ambient-space
Adam on `oft_R`) moves the loss toward Muon's, before any of the other Pion
machinery is layered on — and it does so reusing the entire tested poet0
merge/DDP stack.

## 2. Background

### 2.1 The Q-optimizer dispatch and the SkewMuon precedent

POET already supports a non-Adam optimizer on the skew generators:
`optim.poet.q_optimizer: muon` routes to
[`get_megatron_poet_muon_optimizer`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L460),
which builds a [`SkewMuon`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_skew_muon.py#L62)
(skew branch on `oft_R`, AdamW branch on everything else) and wraps it in
`Float16OptimizerWithFloat16Params` / `FP32Optimizer`. The dispatch is a single
branch in
[`get_megatron_poet_optimizer`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L547):

```python
if getattr(config, "poet_q_optimizer", "adam") == "muon":
    return get_megatron_poet_muon_optimizer(config, model_chunks, ...)
```

The new `lie_algebra` backend mirrors this exactly: one more branch, one cloned
builder, one new `SkewMuon`-shaped class.

SkewMuon's skew branch is the structural template ([poet_skew_muon.py:104-122](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_skew_muon.py#L104)):
`oft_R` has shape `(n_blocks, n_elems)` (upper-triangular vector per block);
`vec_to_skew(g, b)` lifts the gradient to `(n_blocks, b, b)` skew matrices;
the per-block update is computed there; `skew_to_vec(...)` projects back.
Helpers live in `src.diag.skew_conditioning`: `vec_to_skew`, `skew_to_vec`,
`block_size_from_nelems`.

### 2.2 The tangent-vs-ambient gradient — and why interval-1 makes it free

Pion's Lie momentum is accumulated on the **tangent (skew) gradient**:

```
G_in  = Wᵀ G − Gᵀ W        # din×din skew,   G = ∂f/∂W
G_out = G Wᵀ − W Gᵀ        # dout×dout skew
```

In POET the base weight `W` is **frozen** (`requires_grad=False`,
[poet_layer.py:581](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L581)),
so autograd never produces `∂f/∂W`; it produces `∂f/∂oft_R_in` and
`∂f/∂oft_R_out` — the **ambient** gradient, already routed through the Cayley/exp
chain, and already split into the two sides as separate parameters.

**Identity-point shortcut (the key enabler).** At `merge_period=1`, `oft_R` is
reborn at identity (`oft_R = 0`) every step (the merge zeros it,
[poet_layer.py:717](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L717)).
The Cayley/exp Jacobian at `oft_R = 0` is the identity, so the ambient gradient
`∂f/∂oft_R` that autograd **already computes** equals the skew-projected
tangent gradient to **O(angle²)**. Concretely:

```
G_skew_in  ≈ vec_to_skew(oft_R_in.grad,  b)
G_skew_out ≈ vec_to_skew(oft_R_out.grad, b)
```

POET stores `oft_R_in` and `oft_R_out` as separate params, so we get **both**
sides for free, with **no forward/backward changes**. This is what unblocks §9
step 1 at the poet0 config. (The *exact* two-sided `WᵀG`/`GWᵀ` coupling — valid
at any interval — needs materializing `∂f/∂W`, the memory POET avoids; deferred.)

### 2.3 block_count=1 as a correctness oracle

At `block_count=1`, one block **is** the full `din×din` / `dout×dout` skew, so
the per-block `G_skew` *is* the full-matrix tangent gradient. Combined with
"born-at-identity → fold into W", the Lie-momentum step is a discrete-step
approximation of direct-on-W Pion to O(η²). The CPU test (§9) asserts exactly
this against a hand-written direct-on-W Pion step. Once validated, flipping to
`block_size=256` is a config change with **zero code change** — and restores
POET-X's block-diagonal memory advantage.

## 3. Design decisions (resolved)

| Decision | Choice | Rationale |
|---|---|---|
| Backend selection | `optim.poet.q_optimizer: lie_algebra` | Mirrors the `muon` dispatch; clean ablation axis. |
| Gradient source (increment 1) | Identity-point ambient `oft_R.grad` as tangent grad | Free at interval-1, O(angle²)-faithful (§2.2). Exact tangent grad deferred. |
| Interval | `merge_period=1` (poet0 config) | Regime where the identity-point shortcut is valid. |
| 1st moment | Lie-algebra: `M ← β1·M + (1−β1)·G_skew`, **persists** across merges | Pion §2.4.2; momentum in so(n) at identity needs no transport. |
| 2nd moment | **Both** shapes behind `lie_v_mode: scalar \| elementwise` | scalar-v = pipeline doc §3 (isotropic, default); elementwise = paper Algorithm 1. Ablate. |
| Bias correction | **None** (use `M`, `v` directly) | Matches Pion Algorithm 1 (no bias correction). |
| Step magnitude | Cosine-scheduled `lr · poet_scale` (no RMS-α) | Integrates with `scheduler=cosine_poet` like poet0; RMS-α is §9 step 3. Expect to tune LR down (paper App. D.1: momentum-without-RMS needs smaller LR). |
| Exp map | Unchanged (existing merge's Cayley/exp) | Low-order Cayley is §9 step 2. |
| Sides | Two-sided (`train_output_rotation=true`) | Alternating is §9 step 4. |
| Ψ / reset | `reinit_period=-1` (fixed Ψ, never reset) | Keeps momentum coordinate-coherent; avoids the periodic spikes diagnosed for the Adam path. |

## 4. The optimizer algorithm

New class `LieAlgebraMomentum(torch.optim.Optimizer)` in
`src/optim/poet_lie_momentum.py`, shaped like
[`SkewMuon`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_skew_muon.py#L62)
(`skew_params` = `oft_R`, `adamw_params` = everything else; per-param
`state["use_skew"]` flag; AdamW branch copied verbatim).

**Skew branch (per `oft_R` param, executed every step; `p` is born at 0):**

```python
g = p.grad                              # (n_blocks, n_elems), ambient ≈ tangent at identity
b = block_size_from_nelems(g.shape[-1])
G = vec_to_skew(g.float(), b)           # (n_blocks, b, b) skew

state["lie_m"] = β1·state["lie_m"] + (1-β1)·G          # 1st moment, (n_blocks,b,b)

if lie_v_mode == "scalar":
    sq = (G*G).sum(dim=(-2,-1), keepdim=True)          # ||G_j||_F^2 -> (n_blocks,1,1)
    state["lie_v"] = β2·state["lie_v"] + (1-β2)·sq      # scalar per block
else:  # elementwise
    state["lie_v"] = β2·state["lie_v"] + (1-β2)·(G*G)   # (n_blocks,b,b)

A = -state["lie_m"] / (state["lie_v"].sqrt() + eps)    # normalized skew direction
p.add_(skew_to_vec(A, b).to(p.dtype), alpha=η)         # p was 0 -> p = η·A
```

where `η` = the **cosine-scheduled** group LR (already scaled by `poet_scale`;
see §6). The existing merge then exponentiates `R(p)` and folds it into `W`.

**State buffers** `lie_m`, `lie_v` are deliberately **not** named
`exp_avg`/`exp_avg_sq`, so [`_zero_moments`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py#L154)
can never clobber them even if a future run sets `reinit_period > 0` (at
`reinit_period=-1`, `_zero_moments` is never reached anyway).

The **AdamW branch** (norms, embeddings, output layer) is copied unchanged from
[SkewMuon](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_skew_muon.py#L124).

## 5. Architecture & touch points

| File | Change |
|---|---|
| `src/optim/poet_lie_momentum.py` (new) | `LieAlgebraMomentum` class (skew branch above + AdamW branch from SkewMuon) |
| [src/optim/poet.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L460) | `get_megatron_poet_lie_momentum_optimizer` (cloned from the muon builder); add `q_optimizer=="lie_algebra"` branch at [L547](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L547) |
| [launchers/pretrain_gpt_slm.py](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py#L64) | add `--poet-q-optimizer` choice `lie_algebra`; add `--poet-lie-b1/-b2/-eps`, `--poet-lie-v-mode` |
| [src/utils/megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L277) | emit the new `--poet-lie-*` args in the poet branch |
| [src/patches/poet_optimizer_setup.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py#L35) | thread `poet_lie_b1/b2/eps/v_mode` onto the OptimizerConfig |
| `configs/experiments/optim/poet_lie.yaml` (new) | clone of [poet0.yaml](/lustre/fast/fast/zqiu/slm-research/configs/experiments/optim/poet0.yaml) with `q_optimizer: lie_algebra`, `reinit_period: -1`, lie params |
| `docs/experiments/poet_lie.md` (new) | required by the pre-commit "experiment YAML needs a doc" hook |
| `scripts/train_poet_lie.sh` (new) | clone of [train_poet0.sh](/lustre/fast/fast/zqiu/slm-research/scripts/train_poet0.sh) → `experiment=optim/poet_lie` |

Config keys under `optim.poet`: `q_optimizer: lie_algebra`, `lie_b1: 0.9`,
`lie_b2: 0.95`, `lie_eps: 1.0e-8`, `lie_v_mode: scalar`.

## 6. LR / step-magnitude wiring

Increment 1 has **no RMS-α**; the step magnitude is the cosine-scheduled LR
times `poet_scale`, exactly as poet0 handles `oft_R`'s LR. Unlike SkewMuon
(which uses a fixed `theta`), the Lie builder puts the skew params in a param
group carrying `max_lr = config.lr·poet_scale` / `min_lr = config.min_lr·poet_scale`
so Megatron's `OptimizerParamScheduler` decays `group["lr"]`; the skew branch
reads `group["lr"]` as `η`. (Wiring detail to confirm in the plan: the muon
builder bypasses this — the Lie builder must set the group's `max_lr`/`min_lr`
so the scheduler updates it. Fallback if fiddly: a fixed `lie_eta` constant, at
the cost of no LR decay; not preferred.)

## 7. Persistence & merge interaction (unchanged plumbing)

The merge stack is reused verbatim. Per step at `merge_period=1`,
`reinit_period=-1`:

1. `optimizer.step()` sets `oft_R` (born 0) → `η·A`; `lie_m`/`lie_v` updated, **persist**.
2. `_run_merge(reinit_perm=False)` exponentiates `R(oft_R)`, folds into `W`, zeros the bf16 `oft_R`, keeps Ψ.
3. `_reset_vanilla_oft_state(reset_moments=False)` zeros the fp32 **master value** of `oft_R` (prevents spring-back) and touches **nothing else** — `lie_m`/`lie_v` survive.
4. Next step: `oft_R` born at 0 again; momentum intact.

Constraints inherited from the muon path's builder: dev-only (no distributed
optimizer, no TP/PP > 1, bf16 not fp16) — acceptable for the 60m dev ablation.

## 8. Out of scope (deferred increments, per §9)

- §4 RMS-α step scaling (next; §9 step 3)
- §5 low-order Cayley / 2nd-order exp (§9 step 2)
- §6 alternating single-sided update (§9 step 4)
- §2 exact / block-diagonal tangent gradient (only if interval-1 plateaus above Muon)
- §8 sharded merge (§9 step 5)

## 9. Testing & verification

CPU-testable (no GPU/Megatron runtime):

1. **block_count=1 equivalence oracle.** Construct a single `oft_R`-shaped skew
   param at `block_count=1`; run one `LieAlgebraMomentum` skew step from a known
   gradient; independently compute a hand-written direct-on-W Pion first/second-
   moment step `A = -M/(√v+ε)` on the same skew gradient; assert the produced
   skew update matches to O(η²) (and exactly for the first step, where
   `M=(1-β1)G`, `v` per the chosen mode). Run for both `lie_v_mode` values.
2. **Momentum persistence across merge.** Drive `LieAlgebraMomentum.step` twice
   with the `oft_R` value zeroed between steps (simulating the fold); assert
   `lie_m`/`lie_v` accumulate across the zeroing (are NOT reset). Then call
   `_reset_vanilla_oft_state(reset_moments=False)` and assert `lie_m`/`lie_v`
   are still intact and the master value is zeroed.
3. **scalar vs elementwise v shapes.** Assert `lie_v` has shape `(n_blocks,1,1)`
   in scalar mode and `(n_blocks,b,b)` in elementwise mode, and that scalar-v is
   block-isotropic (equal effective scaling across a block's entries).
4. **Arg translation.** `_optimizer_args` with `q_optimizer=lie_algebra` emits
   `--poet-q-optimizer lie_algebra` and the `--poet-lie-*` flags; experiment
   `poet_lie.yaml` loads with the expected keys.
5. **Script dry-run** (with venv on `PATH`, per the poet0 plan convention):
   `experiment=optim/poet_lie` resolves and emits the lie args.
6. `py_compile` / `ruff` on edited files.

Not run here (user's to launch): the GPU run on the 60m dev scale
(`block_count=1, merge_period=1, reinit_period=-1, q_optimizer=lie_algebra`),
ablating `lie_v_mode` and LR; target = move val loss off the POET-Adam baseline
toward Muon (§9 step 1).
