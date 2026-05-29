# Experiment: poet

**Family**: optim
**Status**: exploratory
**Owner**: zqiu
**Created**: 2026-05-13

## Hypothesis
Block-orthogonal parameterisation of linear layers (POETLinear) is enough
to match dense Adam on loss curves while leaving rotational symmetry
exact under SGD. Periodically merging the orthogonal delta into the base
weight and resetting Adam state ("merge-and-reinitialize") should
prevent the state-buildup pathologies seen with plain LoRA-style
parameterisations.

## Method summary
- Param partitioning: every 2-D non-embedding linear weight in the
  transformer → `POETMegatronLinear` (frozen base weight + orthogonal
  delta `oft_R`, trained by `POETAdam`); everything else (embeddings,
  norms, biases, LM head) → plain Adam via the standard Megatron path.
- `POETAdam` wraps the underlying Adam(W). Every `poet.merge_period`
  steps it zeros `exp_avg` / `exp_avg_sq` / step counters; an
  optional `poet.scale` LR multiplier is applied to the POET group at
  construction.
- Architecture constraint: `config.transformer_impl='local'` (no fused
  TE LayerNormLinear). The `poet_unfuse_te_impl` patch flips this
  automatically when `args.poet` is set.
- Optional fused-linear splitting (`--poet-split-qkv` / `--poet-split-fc1`,
  config `optim.poet.split_qkv` / `split_fc1`, default off): before POET
  wrapping, split the fused attention `linear_qkv` into separate Q/K/V
  projections (GQA-correct de-interleave; inert for MLA) and/or the fused
  SwiGLU `linear_fc1` into separate gate/up projections, so each sub-projection
  gets its own independent POET orbit. Implemented in
  [`src/optim/poet_split.py`](../../src/optim/poet_split.py). Each produced
  sub-segment must be divisible by `block_size`/`block_count` or training
  hard-errors at startup. Requires TP=1 (already enforced by POET) and
  non-gated attention.

## Patches applied
- [`poet_unfuse_te_impl`](../../src/patches/poet_unfuse_te_impl.py) —
  flip `config.transformer_impl` to `local` when POET is enabled.
- [`poet_apply_to_model`](../../src/patches/poet_apply_to_model.py) —
  replace `*ParallelLinear` modules with `POETMegatronLinear` after
  `get_model` returns. When `--poet-split-qkv` / `--poet-split-fc1` are
  set, runs the fused-linear split
  ([`split_fused_linears`](../../src/optim/poet_split.py)) first, so the
  produced Q/K/V/gate/up sub-linears are each wrapped as their own orbit.
- [`poet_merge_step`](../../src/patches/poet_merge_step.py) — periodic
  `POETLinear.merge_then_reinitialize()` after each `train_step`.

## Provenance
- Optimizer + integration ported from
  `/lustre/scratch/zqiu/Megatron-LM` branch `poet_core_v0.16.1`
  (commit `bb43fa063`, 2026-04-11).
- POETLinear kernel from the local `poet_torch` package
  (vendored at [third_party/poet_torch/](../../third_party/poet_torch/),
  pin tracked in [docs/poet_torch_pin.md](../poet_torch_pin.md)).

## Configuration
See [configs/experiments/optim/poet.yaml](../../configs/experiments/optim/poet.yaml).

## Timeline
- 2026-05-13: ported optimizer + 5 patches + YAML; CPU unit tests
  passing (66/66). End-to-end GPU smoke deferred — the launcher's
  override parser still treats `cluster=h800_cn` as a literal string
  rather than loading the cluster YAML (pre-existing limitation,
  separate plan). To run the smoke once the launcher is wired up:

  ```bash
  cd /lustre/fast/fast/zqiu/slm-research
  python -m launchers.submit \
      base/family=qwen3 base/scale=600m \
      experiment=optim/poet training_regime=ablation_20x \
      cluster=h800_cn seed=0 \
      wandb.project=sandbox-${USER} allow_dirty=true \
      training.max_iters=10 training.poet_merge_period=5 \
      --dry-run
  ```

  Expected: `Registered patches: poet_unfuse_te_impl,
  poet_apply_to_model, poet_merge_step`; `[POET] replaced N linears`;
  `[POET] merged at iteration 5` and `[POET] merged at iteration 10`;
  `patch_set_hash: <16-hex>` in `runs/<config_hash>/metadata.json`.
- 2026-05-29: added fused-linear splitting (`--poet-split-qkv` /
  `--poet-split-fc1`) in [`src/optim/poet_split.py`](../../src/optim/poet_split.py)
  with CPU geometry/surgery unit tests
  ([tests/unit/test_poet_split.py](../../tests/unit/test_poet_split.py), 12
  tests). GPU smoke (llama3 1.2b, 8×B200, both splits on, `block_size=128`,
  mock data): model built with `[POET split]` logs on all 52 layers
  (`linear_qkv → q/k/v q=1536,kv=512,groups=8`; `linear_fc1 → gate/up
  ffn=3840`), `[POET] replaced 364 linears`, 140 train steps with loss
  decreasing 12.06 → 11.80 and 0 nan/skipped iterations.

## Runs
- Ablation ladder: (pending)
- 2.4B confirmation: (pending)
- 7B anchor: (pending)

## What worked
(populate after first run)

## What didn't
(populate after first run)

## Follow-ups
- Compare against `muon_hybrid` at matched compute budget.
- Try `init_type=mup_normalized` once a muP scaling sweep lands.
- Sweep `poet.merge_period` ∈ {0, 100, 200, 500, 1000}.

## References
- POET / GaLore (Zhao et al. 2024).
