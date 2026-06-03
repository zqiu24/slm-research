# POET × Pion — Alternating Single-Sided Update (§6)

**Date:** 2026-06-03
**Status:** Design approved, ready for implementation plan
**Related:** [docs/poetx_pion_pipeline.md](/lustre/fast/fast/zqiu/slm-research/docs/poetx_pion_pipeline.md) §6, §9 step 4; Pion paper (arXiv 2605.12492) §2.4.3 Eq. 8, Algorithm 1; builds on [Lie-momentum spec](/lustre/fast/fast/zqiu/slm-research/docs/superpowers/specs/2026-06-03-poet-lie-momentum-design.md)

---

## 1. Goal

Add the **alternating single-sided update** (Pion §2.4.3) to the
`q_optimizer=lie_algebra` path: update **one** side of the rotation per step —
`oft_R_out` on even steps, `oft_R_in` on odd — while **accumulating momentum on
both sides every step**. This is pipeline-doc §9 step 4. RMS-α (§4) is **not** in
this increment (deferred — it's the part that needs `W` access).

Value: ~half the per-step optimizer write work and (Pion App. C) ~half the
update-side compute, at ~0.23% loss cost in Pion's setting. It's also a clean
ablation of "does coverage of all neuron pairs survive single-sided updates" in
the block-stochastic POET setting (pipeline doc §6 caveat).

## 2. Why this is optimizer-local (no merge / W / gradient changes)

Every step, **both** `oft_R_in` and `oft_R_out` are born at 0 (the merge folds
`R(oft_R)` into `W` and zeros `oft_R`,
[poet_layer.py:717](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L717)).
So both side gradients are always evaluated at the identity point regardless of
which side we write. Alternating only changes **which side's `oft_R` gets a
non-zero write**:

- **Active side:** `oft_R = lr·A` (as today).
- **Inactive side:** not written → stays 0 → `R = Cayley(0) = I` → the merge
  folds it as a no-op.

The merge ([poet_merge_step.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_merge_step.py))
already handles `oft_R=0` (identity), so **no merge change**. No `W` access, no
new gradient — purely a write-gate inside `LieAlgebraMomentum`.

The momentum (`lie_m`/`lie_v`) is updated for **both** sides on **every** step
(Pion App. D.1: "maintain momentum accumulation on both sides and alternate only
the parameter updates" — reducing the effective sample size for the moment
estimate would raise variance).

## 3. The per-group side tag survives the master swap

`LieAlgebraMomentum` is wrapped in `Float16OptimizerWithFloat16Params`, which
**moves** optimizer state from the bf16 model param to the fp32 master:
`self.optimizer.state[main_param] = self.optimizer.state.pop(param)`
([optimizer.py:688](/lustre/fast/fast/zqiu/slm-research/third_party/Megatron-LM/megatron/core/optimizer/optimizer.py#L688)),
and replaces each group's `params` with masters while preserving the group's
other keys. So a `side` key placed on each skew param group at construction is
intact at step time. We therefore tag the **group**, not per-param state.

## 4. Design decisions (resolved)

| Decision | Choice | Rationale |
|---|---|---|
| Exposure | `lie_alternating: bool` (default **false**) + new `poet_lie_alt.yaml` | poet_lie stays two-sided (unchanged); flipping one flag is the whole ablation. |
| Granularity | `lie_alternate_every: int` (default **1**) | Paper alternates every step but notes "every few steps" is valid. |
| Side schedule | `out` on even, `in` on odd (Eq. 8: ψ=0 even→out) | Match the paper. |
| Momentum | Accumulate `lie_m`/`lie_v` on **both** sides every step | Paper App. D.1. |
| Group structure | **Always** split skew into two side-tagged groups (`in`, `out`) | When `lie_alternating=false`, both are written every step → identical to increment 1. So alternating is purely a write-gate. |
| Parity source | Optimizer's own `self._alt_step` counter | CPU-testable; no Megatron dependency. Caveat below. |

**Resume caveat:** `self._alt_step` is an in-memory counter, not in the
optimizer `state_dict`; a mid-run resume restarts it at 0, shifting the parity.
Acceptable for the 60m dev ablation; noted, not fixed.

## 5. The optimizer change

### 5.1 Param split + groups

New split in [poet.py](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py)
(beside `_split_poet_muon_params`):

```python
def _split_poet_lie_params(model_chunks):
    """oft_R_in -> in-side, oft_R_out -> out-side, everything else -> adamw."""
    skew_in, skew_out, adamw = [], [], []
    for mc in model_chunks:
        for name, p in mc.named_parameters():
            if not p.requires_grad:
                continue
            if "oft_R_in" in name:
                skew_in.append(p)
            elif "oft_R_out" in name:
                skew_out.append(p)
            else:
                adamw.append(p)
    return skew_in, skew_out, adamw
```

`_build_lie_param_groups` is **refactored** to take `(skew_in, skew_out,
adamw)` and emit up to three groups, each skew group carrying `side`:

```python
def _build_lie_param_groups(skew_in, skew_out, adamw_params, lr, min_lr, scale):
    groups = []
    for side, ps in (("in", skew_in), ("out", skew_out)):
        if ps:
            groups.append(dict(params=list(ps), use_skew=True, side=side,
                               lr=lr * scale, max_lr=lr * scale, min_lr=min_lr * scale))
    if adamw_params:
        groups.append(dict(params=list(adamw_params), use_skew=False, side=None,
                           lr=lr, max_lr=lr, min_lr=min_lr))
    return groups
```

(`side=None` is added to the constructor `defaults` so AdamW/test groups are valid.)

### 5.2 Step gate

`LieAlgebraMomentum.__init__` gains `alternating: bool = False`,
`alternate_every: int = 1`, sets `self._alt_step = 0`, and adds `side=None` to
`defaults`. `step()`:

```python
active = None
if self.alternating:
    active = "out" if (self._alt_step // self.alternate_every) % 2 == 0 else "in"

for group in self.param_groups:
    lr = group["lr"]
    if group["use_skew"]:
        side = group["side"]
        # ... compute g, update lie_m / lie_v for EVERY param (both sides) ...
        A = -m / (v.sqrt() + eps)
        if self.alternating and side != active:
            continue                      # momentum updated; skip the write
        p.add_(A.to(p.dtype), alpha=lr)   # active side (or non-alternating): write
    else:
        # ... AdamW branch unchanged ...

if self.alternating:
    self._alt_step += 1
```

When `alternating=false`, the gate is inert and both side groups are written
every step — byte-identical to increment 1.

## 6. Config / arg plumbing

| File | Change |
|---|---|
| [pretrain_gpt_slm.py](/lustre/fast/fast/zqiu/slm-research/launchers/pretrain_gpt_slm.py) | `--poet-lie-alternating` (store_true) + `--poet-lie-alternate-every` (int, default 1) |
| [megatron_args.py](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py) | emit `--poet-lie-alternating` when `lie_alternating` true; always emit `--poet-lie-alternate-every` |
| [poet_optimizer_setup.py](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py) | thread `poet_lie_alternating`, `poet_lie_alternate_every` |
| [poet.py builder](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py) | use `_split_poet_lie_params`; pass `alternating`/`alternate_every` to `LieAlgebraMomentum` |
| `configs/experiments/optim/poet_lie_alt.yaml` (new) | clone of poet_lie.yaml + `lie_alternating: true` |
| `docs/experiments/poet_lie_alt.md` (new) | pre-commit hook |
| `scripts/train_poet_lie_alt.sh` (new) | clone of train_poet_lie.sh → `experiment=optim/poet_lie_alt` |

Config keys under `optim.poet`: `lie_alternating: false`, `lie_alternate_every: 1`.

## 7. Testing & verification

CPU-testable:

1. **Flips one side with parity.** `alternating=true, alternate_every=1`, two
   side groups, drive 4 steps with both-side grads, zeroing `oft_R` between steps
   (simulating the fold): assert step 0 writes only `out` (in stays 0), step 1
   only `in`, step 2 `out`, step 3 `in`.
2. **Momentum accumulates on the inactive side.** After step 0 (out active),
   assert the in-side group's `lie_m` is non-zero (it saw the in-grad) even
   though `oft_R_in` was not written.
3. **`alternate_every=2`** holds each side for two steps: `out,out,in,in`.
4. **`alternating=false` writes both sides** every step (= increment-1 behavior).
5. **`_build_lie_param_groups`** emits side-tagged in/out groups (lr scaled) +
   adamw group; drops empty sides.
6. **Arg/emit/thread/yaml/script** tests mirroring the increment-1 plan (launcher
   accepts `--poet-lie-alternating`/`-alternate-every`; megatron_args emits them;
   `poet_lie_alt.yaml` loads with `lie_alternating=true`; dry-run emits the flags).

Not run here (user's): the GPU run on the 60m dev scale, ablating
`poet_lie` (two-sided) vs `poet_lie_alt` (alternating), per §9 step 4 — watch the
loss-curve shape and whether single-sided coverage slows convergence.

## 8. Out of scope (deferred)

- §4 RMS-α step scaling (the part needing `W` access — next increment).
- §5 low-order Cayley / 2nd-order exp.
- §2 exact / block-diagonal tangent gradient.
- §8 sharded merge.
