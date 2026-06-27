# POET learnable per-layer scale (`ScaledPOETLinear`) — design

**Date:** 2026-06-27
**Status:** approved (design), pending implementation plan
**Author:** zqiu (with Claude)

## Motivation

POET parameterizes every weight as `W_eff = R_out · W₀ · R_in` with `W₀` frozen and
`R` orthogonal. Orthogonal rotations preserve every singular value, so during
training POET can rotate the singular *vectors* but can **never change the
spectrum** of any weight. The §2.7 weight-norm study (POET_dev.md) is the direct
evidence: at matched wd=0/raw-init, POET's per-element RMS is **flat (1.06–1.08×
over 9k steps)** while muon **grows 3.39×** and reshapes the per-type profile to a
common band. Muon's optimum lives at a norm POET structurally cannot reach. This is
the leading explanation for the standing 60m/40tpp gap: best POET **3.4686** vs
muon_kimi **3.4514** (−0.0172).

The single biggest POET lever ever found was *choosing* the frozen operating norm:
scaling the frozen base (`init_scale` / `mup_alpha`) gave **−0.039** (3.5160 →
3.4766, POET_dev.md §2.5-K / §2.1 init row). Every lever since (update-RMS, side_γ,
decorrelation, max_angle, lr) is sub-noise. That signature — one big win from the
frozen norm, everything else at the noise floor — says the binding constraint is
**expressivity (a frozen norm), not the optimizer.**

This spec implements **rung 1** of the "give POET back its spectral DOF" ladder: a
**learnable per-layer scalar `g`** that turns the previously-frozen operating norm
into a trainable quantity that adapts over training (the way muon's norm does),
while keeping POET parameter-efficient (one scalar per weight matrix). Rung 2
(per-row diagonal) and rung 3 (free Σ) are explicitly out of scope — separate specs
if rung 1 pays off.

## Goal

A new layer that adds one trainable scalar per POET weight matrix:

```
W_eff = g · R_out · W₀ · R_in          # g: 0-dim nn.Parameter, init 1.0, one per layer
```

`g` is **coupled to the rotation-angle law** so it behaves as a *trainable*
`init_scale`: it scales the effective weight AND cools the update-RMS rotation,
reproducing the mechanism behind the `init_scale=4` win — but learnable per layer.

### Non-goals (YAGNI)
- No per-row diagonal (rung 2) or free singular spectrum (rung 3).
- No head-aligned variant (the champion is head-off).
- No new behavior when the flag is off — every existing run is bit-identical.

## Design

### 1. The layer — `ScaledPOETLinear(POETLinear)`

A **subclass** of the vendored
[`POETLinear`](/lustre/fast/fast/zqiu/slm-research/third_party/poet_torch/poet_layer.py#L509)
(new file `src/optim/poet_scaled_layer.py`, so the vendored code stays untouched).
Subclassing — not copying — keeps the merge/fold, single-step fast path, exp path,
and the three forward cores **shared** with `POETLinear`, so the scaled variant can
never silently drift from POET. Two overrides:

- `__init__`: `super().__init__(...)`, then register
  `self.gain = nn.Parameter(torch.ones((), dtype=…, device=…))` — a 0-dim scalar,
  `requires_grad=True`, **one per module** (q/k/v/proj/fc1_gate/fc1_up/fc2 each get
  their own).
- `forward`: `return super().forward(x) * self.gain` — applied **outside** the
  compiled core, so it is core-agnostic (works on the fast / exp / compiled paths
  alike) and autograd flows to `g`.

**Bias.** All llama3 POET-wrapped linears are constructed `bias=False` (Megatron
handles bias separately), so `g · forward(x) = g · W_eff · x` exactly. The layer
asserts `self.bias is None` to fail loudly if that assumption is ever violated
(rather than silently scaling a bias).

**Merge is untouched.** `g` lives on the output and is *never* folded into
`self.weight`. The per-step merge (`merge_period=1`) keeps folding only `R` into
`W₀`, exactly as today.

### 2. Optimizer coupling — `g` feeds the angle law

The update-RMS angle is `θ = min(lr·ρ/RMS(W), max_angle)`, computed in
[`compute_update_rms_angle`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth_update_rms.py#L29)
from each layer's frozen "owner weight". Because `g` is a scalar,
`RMS(g·W₀) = |g|·RMS(W₀)`, so the coupling is one multiply in the denominator:

```
denom = |g.detach()| · RMS(owner_weight)
```

- `.detach()` keeps the angle a function of the **current** `g` without routing
  rotation gradients back into `g` — `g` learns only from the forward-output path.
- This coupling **only bites for `q_optimizer=lie_ortho_update_rms`** (the
  champion) — it is the only optimizer that reads `RMS(W)`. Under the fixed-angle
  `lie_ortho` (`lr·scale·c`) the angle does not read the weight, so `g` is
  automatically a **pure output gain** there (still valid, just decoupled).

Wiring: the per-side skew groups built by
[`_build_lie_update_rms_param_groups`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L299)
already carry `weight` and `block_size`; we add `gain=getattr(mod, "gain", None)`
to each group dict, and the optimizer multiplies the denom by `|gain|` when present
(None ⇒ unchanged, so plain `POETLinear` is unaffected).

### 3. How `g` is optimized

`g` is a dense `requires_grad=True` param that is neither a skew (`oft_R_*`) nor the
frozen weight, so the ChainedOptimizer routes it to the **AdamW side** (same as
embeddings/norms), not the Lie optimizer. Defaults:

- **lr** = dense AdamW lr (champion `5e-3`).
- **weight_decay = 0** — a scale gain must not be decayed toward 0 (that would
  shrink the operating norm and fight what `g` is meant to learn). In
  `_build_lie_update_rms_param_groups` the non-skew params are currently lumped into
  one AdamW group that inherits the dense wd (0.1); we split `gain` params into a
  dedicated `weight_decay=0.0` group. (In the standard non-update-RMS param-group
  path, a 0-dim scalar already lands in Megatron's no-decay bucket, so no change is
  needed there.)
- `g`'s lr is the one knob worth a follow-up sweep (`{0.2×, 1×, 2×}` dense lr) if
  the A/B wins — too hot and the norm thrashes early. v1 ships at dense lr.

### 4. Config plumbing (one new flag)

`optim.poet.learnable_scale` (bool, default `false`) gates whether
[`_apply_poet_to_chunk`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py#L97)
→ `replace_linears_with_poet` instantiates `ScaledPOETLinear` vs `POETLinear`. The
4-edit plumbing path (the args→config copy is the one that silently no-ops if
missed):

1. **CLI arg** — `--poet-learnable-scale` (store_true) in
   [`megatron_args.py`](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L576).
2. **args→config copy** — `config.poet_learnable_scale = getattr(args,
   "poet_learnable_scale", False)` in
   [`poet_optimizer_setup.py`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py#L38).
3. **layer-swap consumer** — read `getattr(args, "poet_learnable_scale", False)` in
   `_apply_poet_to_chunk` and thread `learnable_scale=` through
   `replace_linears_with_poet`
   ([poet_layers.py:329](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L329))
   to pick the class.
4. **optimizer consumer** — the param-group `gain` attach + wd=0 group (§2/§3).

Default-off ⇒ zero change to every existing run.

### 5. Init of `g`

`g ← 1.0`, trainable, **on top of the existing fixed init**. `init_type`,
`init_scale`, `mup_alpha` are untouched and stay fixed scalars baked into `W₀`; `g`
is a learnable multiplier above them. Operating norm = `init_scale · shape-norm ·
g(t)`. At init (`g=1`) the layer is **bit-exact the champion** for whatever fixed
init is configured, so the two GPU arms are just config choices:
- **champion-init arm:** `init_type=mup_normalized mup_alpha=4` + `g` — primary A/B
  vs 3.4686 ("does learnable-norm beat fixed?").
- **neutral-init arm:** `init_type=normalized init_scale=1.0` + `g` — secondary
  ("can `g` replace init tuning?").

## Invariants / interactions

- **`g=1` ≡ current POET, bit-exact** (the load-bearing no-op-at-init invariant).
- **Merge unaffected** — `g` never folds into `W₀`.
- **Coupling scoped to update-RMS** — fixed-angle optimizers see `g` as a pure
  output gain.
- **Checkpoint:** `gain` lives under `poet_linear.*` → serialized by `state_dict`;
  the [`sharded_state_dict`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L124)
  in `POETMegatronLinear` drops only the `weight`/`bias` aliases, so `gain` is
  emitted as a replicated (tp=1) ShardedTensor and round-trips.

## Testing

### CPU (run pre-merge)
1. **No-op invariant:** `g=1.0` ⇒ `ScaledPOETLinear` forward **bit-exact**
   `POETLinear` forward.
2. **Scaling:** `g≠1` ⇒ output `== g · POETLinear(x)` (numeric, all three cores).
3. **Grad:** `loss.backward()` ⇒ `gain.grad` populated, correct sign.
4. **Routing:** `gain` lands in an AdamW group with `weight_decay=0`; `oft_R_*`
   stay in the skew groups.
5. **Coupling:** `g=2` halves `θ` vs `g=1` (denom doubles) — unit test on
   `compute_update_rms_angle` + the group `gain` wiring.
6. **Checkpoint:** `gain` survives `state_dict` / `sharded_state_dict` round-trip.
7. **Swap gating:** `learnable_scale=true` swaps in `ScaledPOETLinear`; `false`
   leaves the model byte-identical to today.

### GPU (user-run, 60m/40tpp, seed 42)
- Champion-init A/B: champion recipe + `learnable_scale=true` (`g=1` init, mup α4)
  vs the 3.4686 baseline.
- Neutral-init arm: `normalized`/scale 1 + `g`.
- If a win: `g`-lr micro-sweep `{0.2×, 1×, 2×}`.

## File-by-file change list

| File | Change |
|---|---|
| `src/optim/poet_scaled_layer.py` (new) | `ScaledPOETLinear(POETLinear)`: `gain` param + scaled forward + bias assert |
| [`src/optim/poet_layers.py`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_layers.py#L329) | `replace_linears_with_poet(..., learnable_scale=False)`: instantiate scaled class when set (reuse `_copy_and_init_weight`) |
| [`src/patches/poet_apply_to_model.py`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_apply_to_model.py#L97) | read `poet_learnable_scale`, thread through |
| [`src/utils/megatron_args.py`](/lustre/fast/fast/zqiu/slm-research/src/utils/megatron_args.py#L576) | `--poet-learnable-scale` CLI arg |
| [`src/patches/poet_optimizer_setup.py`](/lustre/fast/fast/zqiu/slm-research/src/patches/poet_optimizer_setup.py#L38) | args→config copy |
| [`src/optim/poet.py`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet.py#L299) | `_build_lie_update_rms_param_groups`: attach `gain` to skew groups + dedicated wd=0 gain group |
| [`src/optim/poet_lie_orth_update_rms.py`](/lustre/fast/fast/zqiu/slm-research/src/optim/poet_lie_orth_update_rms.py#L29) | angle denom `×|gain|` when group carries one |
| `configs/experiments/optim/poet_lie_orth_update_rms_scaled.yaml` (new) | champion recipe + `learnable_scale=true`; leaves the existing champion config untouched |
| `tests/` | the 7 CPU tests above |
| `scripts/` | A/B + neutral-init sweep launcher |
