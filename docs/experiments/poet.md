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
- Optional fused-linear unfusing (`base.model.unfuse_qkv` / `unfuse_fc1`,
  CLI `--unfuse-qkv` / `--unfuse-fc1`; **defaulted on for the poet
  experiment**, off elsewhere): an *architectural*, optimizer-agnostic
  transform that splits the fused attention `linear_qkv` into separate Q/K/V
  projections (GQA-correct de-interleave; inert for MLA) and/or the fused
  SwiGLU `linear_fc1` into separate gate/up projections. Implemented in
  [`src/model/unfuse_linears.py`](../../src/model/unfuse_linears.py) and applied
  at model-build time by the `model_unfuse_linears` patch (pre-DDP). Under POET,
  each sub-projection gets its own independent orbit; under plain Adam it is an
  equivalent-architecture refactor (forward-preserving, modulo fp). Requires
  TP=1 and non-gated attention; `unfuse_fc1` requires a gated (SwiGLU) MLP. When
  POET is on, each produced sub-segment must be divisible by
  `block_size`/`block_count` or POET hard-errors at startup.

## Patches applied
- [`model_unfuse_linears`](../../src/patches/model_unfuse_linears.py) —
  (optimizer-agnostic) at `model_provider` build time, unfuse the fused
  `linear_qkv` / `linear_fc1` into separate Q/K/V and gate/up projections when
  `--unfuse-qkv` / `--unfuse-fc1` are set
  ([`unfuse_fused_linears`](../../src/model/unfuse_linears.py)). Runs before POET
  wrapping (pre-DDP), so POET picks up the unfused sub-linears.
- [`poet_unfuse_te_impl`](../../src/patches/poet_unfuse_te_impl.py) —
  flip `config.transformer_impl` to `local` when POET is enabled.
- [`poet_apply_to_model`](../../src/patches/poet_apply_to_model.py) —
  replace `*ParallelLinear` modules (fused or unfused) with `POETMegatronLinear`
  after `get_model` returns; an indivisible unfused segment is a hard error.
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

  Expected: `Registered patches: model_unfuse_linears, poet_unfuse_te_impl,
  poet_apply_to_model, poet_merge_step`; `[POET] replaced N linears`;
  `[POET] merged at iteration 5` and `[POET] merged at iteration 10`;
  `patch_set_hash: <16-hex>` in `runs/<config_hash>/metadata.json`.
- 2026-05-29: added fused-linear splitting (`--poet-split-qkv` /
  `--poet-split-fc1`) with CPU geometry/surgery unit tests. GPU smoke
  (llama3 1.2b, 8×B200, both splits on, `block_size=128`, mock data): model
  built with split logs on all 52 layers
  (`linear_qkv → q/k/v q=1536,kv=512,groups=8`; `linear_fc1 → gate/up
  ffn=3840`), `[POET] replaced 364 linears`, 140 train steps with loss
  decreasing 12.06 → 11.80 and 0 nan/skipped iterations.
- 2026-05-29: generalized the split into an optimizer-agnostic **unfuse**
  transform — moved to [`src/model/unfuse_linears.py`](../../src/model/unfuse_linears.py)
  + new `model_unfuse_linears` patch (hooks `model_provider`, pre-DDP); args
  renamed to `--unfuse-qkv` / `--unfuse-fc1` under `base.model.*` (default on
  for the poet experiment). The POET-side divisibility hard error now lives in
  `replace_linears_with_poet`. Tests:
  [tests/unit/test_unfuse_linears.py](../../tests/unit/test_unfuse_linears.py).

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
