# Weight-matrix row/column norm monitoring (POET vs Muon vs Adam)

**Date:** 2026-06-13
**Status:** design approved, pending implementation plan

## Goal

During training, monitor the **row norms** and **column norms** of the 2D Linear
weight matrices so we can compare how they evolve under **POET**, **Muon**, and
**Adam(W)**. POET applies no weight decay, so the central question is whether the
effective weight's row/column norms drift or grow over training relative to the
decoupled-weight-decay optimizers.

Single experiment runs use one optimizer (selected by `--slm-optimizer`), so the
comparison happens by overlaying separate wandb runs that share identical metric
keys.

## What gets measured

For a 2D weight `W` of shape `out × in`:

- **row norms:** `r_i = ‖W[i, :]‖₂`, a vector of length `out`
- **col norms:** `c_j = ‖W[:, j]‖₂`, a vector of length `in`

### Layer selection (a few layers only)

We do **not** measure every layer. We measure a small, fixed set of transformer
blocks — default **first / middle / last** = `{0, L//2, L-1}` — configurable via
`--weight-norm-layers` (comma-separated indices, or the keywords
`first,mid,last`). This is cheaper than all-layers and preserves depth
resolution among the chosen few.

Within each selected layer we measure its transformer linears, matched by name
suffix:

- `qkv` — attention QKV projection (fused)
- `proj` — attention output projection
- `fc1` — MLP up/gate projection (fused)
- `fc2` — MLP down projection

Configs built with `--unfuse-qkv` / `--unfuse-fc1` (e.g. head-aligned POET) expose
the unfused variants instead, which we also match: `q`/`k`/`v` (for qkv) and
`fc1_gate`/`fc1_up` (for fc1). Keys then carry those labels; keep the same fusion
setting across compared runs for a clean 1:1 overlay.

Token embeddings, `lm_head`/output layer, and layernorms are **excluded**.

### Per-optimizer source of `W`

| Optimizer | Matrix read | Cadence correctness |
|---|---|---|
| Adam(W) / Muon (and GaLore/AdamW for free) | `model.named_parameters()` weight, `dim()==2` | the weight *is* the trained param |
| POET `merge_period=1` (most configs) | `poet_linear.weight` (raw base) | `R` is folded into `W` every step → base **== `W_eff`** every step ✔ |
| POET `merge_period=M>1` (e.g. `poet.yaml`=400) | `poet_linear.weight` | base **== `W_eff`** only right after a merge → log only when `iteration % M == 0` |
| POET `merge_period=0` (no-reset probe regime) | — | base weight is **frozen forever** → measuring it is meaningless → **warn + skip** |

Rationale for reading the raw base weight rather than reconstructing
`W_eff = R_out · W0 · R_in`: the periodic merge already folds `R` into the base
weight and resets `oft_R` to identity, so immediately after a merge the raw base
weight equals the effective weight — no per-layer matmul needed. For
`merge_period=1` this holds every step.

POET's `oft_R_in` / `oft_R_out` rotation generators (also 2D tensors in
`named_parameters()`) are **excluded** by a name filter (`oft_R` → skip); only the
base weight is measured.

## Metrics emitted (per selected layer)

Pooling row/col norms across matrices of different widths mixes scales, so we
emit an **RMS-normalized** variant (`r_i/√in`, `c_j/√out`, ≈ per-element std)
alongside the raw norms. RMS divides out the matrix width, making qkv/proj/fc1/fc2
comparable.

- **Scalars, per matrix-type** (`mean/std/min/max` of row- and col-norms, raw + RMS):

  ```
  weightnorm/L{i}/{qkv,proj,fc1,fc2}/row/{mean,std,min,max}
  weightnorm/L{i}/{qkv,proj,fc1,fc2}/col/{mean,std,min,max}
  weightnorm/L{i}/{qkv,proj,fc1,fc2}/row_rms/{mean,std,min,max}
  weightnorm/L{i}/{qkv,proj,fc1,fc2}/col_rms/{mean,std,min,max}
  ```

- **Histograms, per layer** (RMS-normalized, pooled over the layer's 4 matrices —
  lighter than per-matrix histograms):

  ```
  weightnorm/L{i}/row_rms_hist   ->  wandb.Histogram
  weightnorm/L{i}/col_rms_hist   ->  wandb.Histogram
  ```

Keys are identical across all three optimizers, so POET / Muon / Adam runs overlay
directly in the wandb UI.

## Where it hooks in

New patch `src/patches/weight_norm_monitor.py`, mirroring the existing
`src/patches/poet_grad_conditioning.py` and `src/patches/wandb_trainable_params.py`
patterns:

- Wraps `megatron.training.training.setup_model_and_optimizer` to capture the model
  handle.
- Wraps `megatron.training.training.train_step` as the **outer** wrapper: it calls
  the inner `train_step` (which runs `optimizer.step()` and, for POET, the periodic
  merge), then — on a logging step — computes and logs the norms. Reading *after*
  the inner call guarantees POET's base weight already reflects the merge.
- Computes under `torch.no_grad()`, **rank-0 only**, and logs via
  `wandb.log({...}, step=iteration)` using the lazy-rebind-safe pattern (wandb.log
  is rebound by `wandb.init()`, so the wrap must be applied lazily post-init, as in
  `wandb_metric_normalize.py`).

## Config / enablement

Two new flags added to `add_slm_args()` in
`launchers/pretrain_gpt_slm.py`, surfaced through YAML:

- `--log-weight-norms` (bool, default **off**)
- `--log-weight-norms-interval` (int, default **100**)
- `--weight-norm-layers` (str, default `first,mid,last`) — comma list of indices
  and/or the keywords `first` / `mid` / `last`

For POET the effective cadence snaps to multiples of `merge_period` so reads land
on post-merge steps; for `merge_period=1` every interval step is valid.

## Risks / limitations (acknowledged, not blockers)

- **Patch order:** the patch must be registered *after* `poet_merge_step` so its
  `train_step` wrapper is outermost (merge runs before we read). Asserted at apply
  time.
- **`merge_period=0` POET:** raw base is frozen → warn and skip (locked default).
- **tp=1 assumption:** true for POET and these model scales; under `tp>1` weights are
  sharded and the norms would be per-shard. Documented limitation, not handled.
- **POET x-axis density:** with `merge_period=400`, POET logs every 400 steps while
  Adam/Muon log every `interval` (sparser but aligned); for `merge_period=1` they
  coincide.

## Out of scope

- Reconstructing `W_eff` for the `merge_period=0` regime.
- Tensor-parallel (`tp>1`) shard reduction.
- Per-matrix histograms (we pool per layer to keep histogram volume down).
- All-layer logging (we deliberately measure only a few layers).
