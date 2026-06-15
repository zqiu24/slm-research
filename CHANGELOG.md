# Changelog

## Unreleased

### Added — gemma3 (text-only) bake-off family (2026-06-14)

- New `gemma3` family + `600m_gemma3` scale (dense, gpt entrypoint, 599.5M
  non-embedding): Gemma 3's local/global sliding-window interleave (5 sliding :
  1 global via `--window-size`/`--window-attn-skip-freq`), GeGLU (`--quick-geglu`),
  zero-centered RMSNorm (`--apply-layernorm-1p`), QK-norm, and sandwich norm
  (reuses the existing `sandwich_norm_apply` patch).
- `megatron_args`: emit `--quick-geglu`, `--apply-layernorm-1p`, and the
  sliding-window flags. `arch_params`: account GeGLU (gated) and the
  sandwich-norm term. Pin guard + `scripts/train_bakeoff_600m.sh` extended.
- Documented approximations: single RoPE base (1M), no √d embedding scale,
  sigmoid-approx `quick_gelu`. See docs/experiments/arch_bakeoff_600m.md.

### Added — weight-matrix norm monitoring (2026-06-13)

- New `weight_norm_monitor` patch: logs row/column L2-norm summaries (+ per-layer
  RMS histograms) of the qkv/proj/fc1/fc2 weights for a few layers to W&B,
  enabling POET vs Muon vs Adam weight-norm comparison without weight decay. It
  is always registered (`_ALWAYS_ON_PATCHES`) but inert unless
  `training.log_weight_norms` is set (interval `log_weight_norms_interval`,
  layers `weight_norm_layers`). Wraps `train_step` as the OUTER wrapper so for
  POET it reads the post-merge effective weight `W_eff`. Warns once if a POET
  run's interval is not a multiple of `merge_period` (logs land only on merge
  boundaries, so the effective cadence is the LCM — easy to silently get sparse
  or zero logging otherwise).

### Added — architecture-family bake-off infrastructure (2026-06-12)

- `src/utils/arch_params.py` + budget gate (`tests/unit/test_scale_budget.py`):
  families realize a declared non-embedding budget within ±2% (600M bake-off:
  deepseek_v3 592.1M, qwen3_next 594.9M, nemotron_h 604.8M); `tools/size_check.py`
  CLI for sizing new realizations.
- `megatron_args`: GDN (`--experimental-attention-variant gated_delta_net`),
  hybrid-mamba, squared-relu, and rope-conditional emission.
- New `launchers/pretrain_mamba_slm.py` + `base.model.entrypoint` routing
  (MambaModel path for nemotron_h; no MTP there by pin limitation).
- New families `qwen3_next`, `nemotron_h`; scales `600m_{deepseek_v3,
  qwen3_next,nemotron_h}`; `scripts/train_bakeoff_600m.sh`; protocol in
  `docs/experiments/arch_bakeoff_600m.md`; guide in `docs/adding_a_family.md`.
- `scripts/run_bakeoff_600m_full.sh`: chains all four full (24B-token) runs
  sequentially on one node (foreground torchrun blocks per run), tee-logging
  each family codexlog-style and printing a pass/fail summary.
- New `configs/training_regime/fixed_12b.yaml` (12B `total_tokens`); the bake-off
  launcher now defaults to it (`REGIME` env override) instead of `ablation_40x`,
  so all four families train on the EXACT same token count and share one
  GPTDataset cache despite their differing `non_embedding_params`.
- `train_bakeoff_600m.sh` defaults (all overridable via env / trailing override):
  `REGIME=fixed_12b`, `base.model.seq_length=256` (`SEQ_LENGTH`; cheap iteration —
  yields `--train-samples 46,875,000`), `training.micro_batch_size=4`
  (`MICRO_BATCH_SIZE`; null would derive to `min(64,gbs)=64` and OOM at the first
  forward on 80GB H100).
- New optional dense ablation: `deepseek_v3_dense` family + `600m_deepseek_v3_dense`
  scale (MLA + MTP identical to `deepseek_v3`, MoE replaced by a dense SwiGLU FFN
  6912 → 604.3M, active==total). Wired into `train_bakeoff_600m.sh` and the budget
  gate; not in the default 4-family sweep (include via `FAMILIES=...`). Isolates
  the value of sparsity at equal total non-embedding params.

### Added — fixed token budgets (dataset pinning across architectures)

- `training.total_tokens` can now be set explicitly (config or CLI override;
  `"500M"`/`"1B"`-style strings accepted via `parse_token_count`) and takes
  precedence over `tokens_per_param * non_embedding_params` (decay-only
  resume's `decay_tokens` still ranks highest). This pins `--train-samples`
  and hence the GPTDataset cache key, so near-scale architecture ablations
  share one pre-built dataset and identical data amounts. New regimes:
  `training_regime/fixed_{500m,1b,10b,50b,100b}`. Switching the data axis
  (different tokenizer) still rebuilds into its own `runs/_data_cache/<name>`
  at the same fixed budget.

### Added — Megatron-Bridge submodule pin

- Vendored [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
  under `third_party/Megatron-Bridge`, pinned to `v0.4.2`
  (`c810129341a84e58f4cbed3093f70668a088c028`) — the latest stable release
  (2026-05-28; newer `26.04-alpha.rc*` tags are pre-releases). Same submodule +
  pin-doc procedure as Megatron-LM/torchtitan; see `docs/megatron_bridge_pin.md`.

### Added — first-party DeepSeek-3Bv2 (MQA + sandwich-norm) port

- Landed the Megatron-poet `DeepSeek-3Bv2-sandwich-mqa` architecture as a
  first-party config/patch (de-vendored from `poet_torch_huawei/`): new
  `deepseek_v3_mqa` family + `deepseek_3bv2` scale (12L, hidden 1280, dense ffn
  7168, 16 heads, MQA head_dim 384, MoE ffn 896 / shared 1792, MTP, rotary 0.25),
  the `sandwich_norm_apply` patch (`SandwichTransformerLayer` post-norm via
  forward-hooks), config-driven `--rotary-percent`, MTP decoupled from MLA, and
  `scripts/train_deepseek.sh`.
- `sandwich_norm_apply` now composes with POET: it owns only
  `gpt_builders.gpt_builder` and stamps the `TransformerConfig` via a temporary
  wrapper inside the builder, so it no longer collides with `poet_unfuse_te_impl`
  on `core_transformer_config_from_args`. It is enabled under `optim/adam`,
  `optim/muon_hybrid`, and `optim/poet`.
- Arg reconciliation vs the Huawei reference: (1) `--use-distributed-optimizer`
  (+ `--overlap-grad-reduce`/`--overlap-param-gather`) now also emits for the
  default POET path (`use_poet_adam=false`, `q_optimizer=adam`), which runs stock
  Megatron-Adam on `oft_R` and supports the sharded distributed optimizer — the
  Muon-on-Q / Lie / POETAdam paths still keep it off (they reject it); (2) added
  config-driven `--embedding-init-method-std` (set to 0.006 in the family) to match
  the reference exactly. The 4 deferred perf-only Huawei flags (`--no-rope-fusion`,
  `--manual-gc[-interval]`, `--cross-entropy-fusion-impl native`,
  `--make-vocab-size-divisible-by 3232`) remain intentionally absent (functionally inert).
- `train_deepseek.sh` now uses the slm-default Transformer Engine path (dropped
  the `transformer_impl=local` override carried over from the TE-less vendored
  `poet` env). The GPU smoke proved `local` is unsupported in the native stack:
  on `h100_de` (TP=4 + sequence-parallel) torch `WrappedTorchNorm` rejects
  sequence-parallel, and even at TP=1 apex `FusedLayerNorm` rejects RMSNorm. TE
  (TENorm) handles both, matches the reference, and is proven on this node. slm
  always emits `--bf16` (never fp8), so POET's weight-swap stays safe.
- CPU unit tests green (`test_sandwich_norm`, `test_patch_sandwich_norm` incl. the
  POET-patchset compose test, `test_deepseek_v3_mqa_scale`, expanded
  `test_megatron_args`). The 8-GPU `cluster=h100_de` adam + POET GPU smoke is the
  remaining user-run acceptance gate.

### Fixed — muon_kimi end-of-training checkpoint crash (skipped final eval)

- muon_kimi runs crashed at the end of training in the (always-fired) end-of-training
  `save_checkpoint` with `AttributeError: 'bool' object has no attribute 'shape'`:
  Megatron's `sharded_state_dict` runs every per-param optimizer-state value through
  `make_sharded_optimizer_tensor` (tensor-only) and only excludes `step`, but the
  vendored Kimi Muon also stores a per-param `use_muon` bool. The crash (exit code 1)
  aborted the post-training validation, so muon_kimi's val curve stopped at the last
  eval-interval (step 9000) instead of the final step (9155) — making it look "shorter"
  than adam/POET and slightly under-reporting its val (still descending at 9000).
- Fix: `_StripUseMuonShardingMixin` in `src/optim/muon_kimi.py` drops `use_muon` from
  the optimizer state while the sharded checkpoint is built, then restores it (it's
  re-derived from param routing on load, so lossless). Applied to both the bf16
  (`Float16OptimizerWithFloat16Params`) and fp32 (`FP32Optimizer`) wrappers. CPU test
  in `tests/unit/test_muon_kimi.py`.

### Result — POET is now the OUTRIGHT BEST at 60m/40tpp (2026-06-09 grid)

- The lr×scale×c cosine grid found a hotter optimum: **`cos_lr4_s50_c8` (`ghsu7t8y`) =
  val/loss 3.5231** (lr 4e-3, scale 0.5, c8 → eff∠ **0.016**, head-OFF + alternating),
  **beating muon_kimi (3.5321) by −0.009** and the prior POET champion `1ynrrimu`
  (3.5332 @ eff∠ 0.012) by −0.010. POET_dev.md leaderboard (§2.3/§2.5/§2.6) updated.
- Findings: the angle sweet spot is **~0.016, not 0.012** (the old "0.012 ceiling / 0.018
  diverges" was the both-sides head-on recipe; head-off+alternating is stable to 0.018,
  only 0.024 diverged); **dense lr wants 4e-3 ≳ 3e-3** (decoupling-down falsified, monotone
  in sweep G); **min_lr_ratio 0.01 is the floor sweet spot** (0.1 and 0.001 both worse);
  **cosine beats WSD** (WSD df0.2 `lodwi7cw` 3.5699, +0.037). Single seed — pending confirm.

### Added — adam LR sweep

- **`scripts/sweep_adam_lr.sh`** — 3-run learning-rate sweep tuning the adam (AdamW)
  baseline (3.5570 @ lr 1e-3) via `scripts/train_adam_dev.sh`: `optim.lr ∈
  {2e-3, 3e-3, 4e-3}`, everything else at the adam defaults (betas [0.9,0.95], wd 0.1,
  stock cosine min_lr 0.1). Probes hotter LRs (the POET grid found the baseline LR cold).

### Added — muon_kimi LR sweep

- **`scripts/sweep_muon_kimi_lr.sh`** — 5-run learning-rate sweep tuning the muon_kimi
  baseline (best non-POET, 3.5321 @ lr 1e-3) via `scripts/train_muon_dev.sh`:
  `optim.lr ∈ {5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3}`, everything else at the muon_kimi
  defaults (momentum 0.95, nesterov, ns_steps=5, wd 0.1, stock cosine min_lr 0.1).
  Checks whether the baseline LR is too cold (as the POET grid found for POET). Cell
  `mk_lr10` reproduces the baseline.

### Added — POET HP-tuning sweeps (cosine grid + dense-LR decoupling)

- **`scripts/sweep_lie_orth_grid_cosine.sh`** — 16-run grid over the best-POET base
  (head-OFF + alternating + muon + distributed, cosine_poet): `optim.lr ∈ {1,2,3,4}e-3 ×
  optim.poet.scale ∈ {0.25,0.5} × lie_ortho_c ∈ {8,12}`. Same eff∠ (`lr·scale·c`) at
  different `optim.lr` = a dense-LR decoupling probe; 3 cells (eff∠ ≥ 0.016) are kept as
  divergence-boundary probes.
- **`scripts/sweep_lie_orth_decouple.sh`** — 16-run dense-LR DECOUPLING sweep (cosine).
  Holds `lie_ortho_c=8` and the rotation-group LR (`optim.lr·scale` → fixed angle) while
  pushing the AdamW DENSE LR down to 1e-3 (raising `scale` to compensate), crossed with
  `scheduler.min_lr_ratio ∈ {0.01, 0.001}`. Tests whether the champion's 3e-3 dense LR
  (3× the adam-optimal 1e-3) is too hot. 4 dense-LR × 2 angles {0.008,0.012} × 2 floors.
- **`configs/scheduler/wsd_poet.yaml`** — WSD sibling of `cosine_poet` (1% floor,
  `wsd_decay_fraction=0.2`, cosine tail). RETAINED as a usable scheduler, but the planned
  WSD sweep was DROPPED: the completed champion-recipe WSD run (`lodwi7cw`, df 0.2) lost
  to cosine by +0.037 (val 3.5699 vs 3.5332) — holding the angle at the ceiling through
  the stable phase keeps loss high and the 20% tail can't recover; WSD→cosine as df→1, so
  it can't beat cosine here. (`9mvs5hsg`: cosine min_lr 0.1 = 3.5413, so the deep 0.01
  floor is worth +0.008 — hence the min_lr_ratio axis in the decoupling sweep.)

### Added — integrated alternating POETX (both-momenta) on the forward-frame layer

- **Phase 1 — champion config `poet_lie_orth_alt`** (`single_step_x` +
  `lie_alternating`, head-OFF, lr 3e-3, c=8, distributed; `single_step_x_alternating`
  OFF). Runs the POET champion behavior — the optimizer writes ONE rotation side per
  step while BOTH Lie momenta stay fresh (`POETXSingleStepFunction` feeds both grads)
  — on the plain forward-frame `POETXLinear`. The merge folds both sides (the frozen
  side's `oft_R=0` → identity → no-op), reproducing the `lie_ortho`+`lie_alternating`
  champion (val/loss ≈3.5332) at POETX forward speed with zero new layer code. Adds
  `configs/experiments/optim/poet_lie_orth_alt.yaml`, `docs/experiments/poet_lie_orth_alt.md`,
  `scripts/train_poet_lie_orth_alt.sh`. This is the integrated both-momenta path, NOT
  the regressed true-single-side `poet_lie_orth_alt_x`.
- **Phase 2 — active-only merge fold on `POETXLinear(alternating=True)`.** `POETXLinear`
  gains an `alternating`/`alternate_every` flag and hosts `_fold_active_side` (the
  research `AlternatingPOETXLinear` is now a thin subclass). The merge driver
  (`_merge_layers`) routes by the `alternating` flag (not `isinstance`), folding ONLY
  the active side — one Cayley + one block-fold per layer instead of two, skipping the
  frozen identity side. Forward/backward stay both-sides so both momenta stay fed. The
  optimizer's `_active_side` (write) and the merge's fold both read the shared
  `alt_state` iteration, so write side == fold side every step. Bit-identical to the
  both-sides fold at fp64 for both `"in"` and `"out"` active sides.

### Added — standalone Muon-like orthogonalizing optimizer (q_optimizer=lie_ortho)

- DP-sharded orthogonalization for `LieOrthMomentum`
  (`optim.poet.lie_ortho_distributed=true`). Each data-parallel rank
  orthogonalizes only its round-robin slice of `oft_R`; one zero-padded
  `all_reduce(SUM)` of update deltas re-syncs every rank, avoiding same-shape
  constraints. Numerically identical to the replicated path when local gloo is
  available for the 2-rank test; off by default and a no-op at `dp_world=1`.
- **Standalone `LieOrthMomentum` optimizer** (`q_optimizer=lie_ortho`), sibling of
  the Lie-RMS optimizer. Orthogonalizes the skew update direction
  (`orthogonalize_skew_direction`) so the rotation planes turn by ~the same angle
  (`= lr * lie_ortho_c`); first-moment-only by default. Default `method=muon`
  (Muon's quintic Newton–Schulz, a band around 1, ~5 steps; NS preserves skew);
  `method=spectral` is the exact `A(-A²)^{-1/2}` σ=1 variant (~20 steps). New
  experiment `optim/poet_lie_orth` + `scripts/train_poet_lie_orth.sh` for the
  head-to-head vs `poet_lie_rms` (see
  `docs/muon_orthogonalizing_optimizer_poet.md`).

### Added — post-orthogonalization (Muon update) spectrum on both probes

- **Both conditioning probes now log the post-Newton–Schulz spectrum**, not just
  the raw gradient — so the "what the optimizer receives" vs "what a Muon-style
  update applies" contrast is visible on every metric. New CPU-pure helper
  `newton_schulz_orthogonalize` (`src/diag/orthogonalize.py`): the canonical Muon
  quintic NS (matching `muon_hybrid`'s `ns_coeffs`/`ns_steps`), with the rows>cols
  transpose, driving all singular values toward ~1.
  - Generic probe: adds `grad_update/<layer>/{condition_number, stable_rank,
    sigma_max_over_median, effective_rank}` alongside the raw `grad_cond/<layer>/*`.
  - POET probe: `poet_update/<label>/*` expands from just `cond_orthogonalized` to
    the full set (`stable_rank`, `effective_rank`, `sigma_max_over_median`);
    `effective_rank` also added to the raw `poet_cond/<label>/*`. `cond_orthogonalized`
    is kept as-is for dashboard continuity.
  - Faithfulness: the NS is applied to the read gradient (not the momentum buffer)
    and, under TP>1, to the local shard — a diagnostic approximation of the realized
    update, not a bit-identical copy.

### Added — generic weight-gradient conditioning probe (any optimizer)

- **Per-layer weight-gradient spectrum logging for plain runs** (AdamW, Muon —
  no POET), via the new env-gated patch `src/patches/grad_conditioning.py`
  (`SLM_GRAD_CONDITIONING=1`; interval `SLM_GRAD_CONDITIONING_INTERVAL`, which
  falls back to the POET probe's `SLM_POET_GRAD_CONDITIONING_INTERVAL`, then 2000,
  so both probes sample at the same cadence; added to `_ALWAYS_ON_PATCHES`, inert
  otherwise). Picks ~8 representative
  `nn.Linear` weights and, every interval, reads each weight's full accumulated
  `main_grad` *before* the optimizer consumes it and logs
  `grad_cond/<layer>/{condition_number, stable_rank, sigma_max_over_median,
  effective_rank}` to W&B. Optimizer-agnostic, so Adam vs Muon is apples-to-apples
  (the canonical "why Muon" plot). Mirrors `poet_grad_conditioning`'s
  setup-wrapper architecture but probes raw 2D weight grads instead of `oft_R`.
- **Entropy effective rank** added to `block_spectral_stats`
  (`src/diag/skew_conditioning.py`): `effective_rank = exp(-Σ pᵢ log pᵢ)`,
  `pᵢ = σᵢ/Σσ` (Roy–Vetterli). Reused by both the new probe and the POET probe.

### Added — POET Muon-on-Q (Stage 2): SkewMuon optimizer

- **`oft_R` can now be optimized by a Muon-style spectral update** instead of
  AdamW, via `optim.poet.q_optimizer: adam|muon` (+ `muon_theta`/`muon_ns_steps`/
  `muon_momentum`). `SkewMuon` (`src/optim/poet_skew_muon.py`) inflates each
  block's skew gradient to `b×b`, Newton–Schulz-orthogonalizes it, re-skews, and
  rescales to a constant rotation angle `muon_theta`, then steps `oft_R`; all
  non-`oft_R` params stay AdamW (hybrid, `muon_kimi` pattern). Built into the POET
  optimizer path (`get_megatron_poet_muon_optimizer`), single-process/DP-replicated.
  Motivated by Probe 0B (heavy-tailed `∂f/∂Q`). Per-block `‖G−I‖`/`‖RRᵀ−I‖`
  diagnostics added. Default (`adam`) unchanged. Intended for the no-reset regime
  (`merge_period=0`).
- **Update-spectrum diagnostic** `poet_update/<label>/cond_orthogonalized`: the
  condition number of the NS-orthogonalized skew gradient — i.e. SkewMuon's actual
  update spectrum (~1) — logged next to the heavy-tailed raw-grad
  `poet_cond/<label>/condition_number`. The raw-grad probe reads `∂f/∂Q` *before*
  the optimizer, so it can't show Muon's preconditioning; this contrast can. Also
  serves as an `ns_steps` health check. New helper `muon_update_spectral_stats`
  (`src/optim/poet_skew_muon.py`); same `SLM_POET_GRAD_CONDITIONING=1` gate.

### Added — POET single-sided (input-only) rotation ablation

- **POET can now freeze the output-side rotation so only the input rotation is
  trained**, via the new `optim.poet.train_output_rotation` flag (`true` default |
  `false` = input-only) (`configs/experiments/optim/poet.yaml`,
  `launchers/pretrain_gpt_slm.py` `--poet-freeze-output-rotation`,
  `src/utils/megatron_args.py`). When false, `replace_linears_with_poet`
  (`src/optim/poet_layers.py`) sets `oft_R_out.requires_grad_(False)` at wrap time
  (pre-DDP, via `src/patches/poet_apply_to_model.py`); since `oft_R_out` inits to
  zero, `R_out` stays the identity and is excluded from the grad buffer + optimizer
  param groups (Megatron only takes `requires_grad` params). `oft_R_in` continues to
  train. New `scripts/train_poet_dev_full_single.sh` = `train_poet_dev_full.sh`
  (block_count=1 full rotation, merge_period=0) + `train_output_rotation=false`.
  Default behavior is unchanged (both rotations trained).

### Added — POET × Muon-on-Q Stage 0 diagnostics (gate instrumentation)

- **Two env-gated diagnostic probes** for deciding whether the Muon-on-Q line is
  worth pursuing, inert on normal runs. Probe 0A: `SLM_OVERFIT_SINGLE_BATCH=1`
  replays the first batch every step (single-batch overfit;
  `src/patches/overfit_single_batch.py` + `src/diag/single_batch.py`
  `BatchReplay`). Probe 0B: `SLM_POET_GRAD_CONDITIONING=1` logs per-block
  `∂f/∂Q` singular-value conditioning (condition number / stable rank /
  σ_max-over-median) to W&B every `SLM_POET_GRAD_CONDITIONING_INTERVAL` steps
  (`src/patches/poet_grad_conditioning.py` + `src/diag/skew_conditioning.py`
  `vec_to_skew`/`block_spectral_stats`). Both registered in the launcher's
  `_ALWAYS_ON_PATCHES`. No POET math or optimizer changes; the Muon-on-Q
  optimizer itself remains gated behind the Stage 0 human-review decision.

### Added — POET `exp` (matrix-exponential) orthogonalization parameterization

- **POET can now build the block rotation as the exact matrix exponential
  `G = exp(Q)` instead of the truncated Cayley/Neumann polynomial**, selected via
  the new `optim.poet.parameterization` flag (`"cayley"` default | `"exp"`)
  (`configs/experiments/optim/poet.yaml`, `launchers/pretrain_gpt_slm.py`
  `--poet-parameterization`, `src/utils/megatron_args.py`). A new builder
  `get_weight_poet_decoupled_exp` (`third_party/poet_torch/poet_layer.py`) mirrors
  the Cayley builder's signature and computes `R = torch.linalg.matrix_exp(Q)` in
  fp32/fp64 (autograd flows through the cast — no custom backward). `R` is
  **exactly** orthogonal for any `Q` (no `‖Q‖<1` ceiling, no truncation error) and
  the singular values of `Q` are exactly the rotation angles of `R` (angle =
  singular value, not Cayley's `2·arctan`). A single `POETLinear._build_R` dispatch
  on `self.parameterization` routes the forward (`forward_core_decoupled_exp`,
  built eagerly outside `torch.compile`), the merge (`merge_then_reinitialize`),
  and the ΔW-spec estimator (`estimate_poet_delta_weff_spec`) through the same map.
  `parameterization="exp"` requires `cache_mode="none"` (guarded in
  `replace_linears_with_poet`). The Cayley path and all default behavior are
  byte-for-byte unchanged. CPU-unit-tested (orthogonality, no-factor-of-2 angle,
  gradcheck, forward parity vs a pure-PyTorch oracle, merge + estimator
  consistency, config plumbing); GPU compile-fusion of the exp chain is an
  optional follow-up.

### Added — trainable/total param counts in the W&B run config

- **Every Megatron run now logs `trainable_params`, `total_params`,
  `trainable_pct`, and `poet_params` into the W&B run config** (Overview → Config
  table, not a chart) (`src/patches/wandb_trainable_params.py`,
  `src/utils/param_count.py`, `launchers/pretrain_gpt_slm.py`). A new always-on
  patch wraps `setup_model_and_optimizer`, counts params *after* it returns (so
  POET's frozen base weights vs. trainable `oft_R` are reflected), SUM-reduces
  over the model-parallel group (TP×PP; no-op in the current DP-only setup), and
  writes the four fields on the W&B-logging rank via
  `get_wandb_writer().config.update(..., allow_val_change=True)` — mirroring
  Megatron's own post-setup `config.update` for `slurm_job_name`. `poet_params`
  is Σ `numel` over params whose name contains `oft_R` (POET's trainable
  orthogonal generators, incl. decoupled `oft_R_in`/`oft_R_out`); it is counted
  by **name** (independent of `requires_grad`) so it is `0` for non-POET runs
  (adam / muon / ngpt) and isolates POET's delta from the rest of the trainable
  set. The collective runs on every rank (outside the rank gate) to avoid a
  hang; the write is best-effort (falls back to `0` on failure so the fields
  always appear). Applied unconditionally in the per-rank launcher
  (`_resolve_runtime_patch_names`), so it needs no `experiment.patches` entry and
  stays out of the experiment patch-set hash. Expert parallelism (EP>1) is not
  yet aggregated across the expert group — a warning is logged so the number is
  never silently wrong. Pure counting helper `count_local_params` is
  CPU-unit-tested (incl. `poet_params` by name); the all-reduce / W&B write are
  covered by a GPU smoke run.

### Changed — POET uses a 1% min-LR floor

- **POET runs now default to a 1% min-LR floor instead of 10%**
  (`configs/scheduler/cosine_poet.yaml`, `scripts/train_poet.sh`). The global
  cosine scheduler floors at `min_lr_ratio=0.1` (`min_lr = 0.1 × optim.lr`),
  which carried through the `optim.poet.scale` multiplier to land the `oft_R`
  group at a 10% floor (`min_lr=5e-6` vs `max_lr=5e-5`). The reference POET
  recipe (Megatron-poet: `LR 8.6e-4 / MIN_LR 7e-6`) uses ~0.8%. New
  `scheduler=cosine_poet` (`min_lr_ratio=0.01`) is injected as the default by
  `train_poet.sh` (overridable with `scheduler=...`), giving global
  `--min-lr=1e-5` and `oft_R min_lr=5e-7` — a true 1% floor. Non-POET runs are
  unaffected; `cosine.yaml` still defaults to `0.1`. Note `optim.min_lr` is not
  a wired key — `--min-lr` is derived solely from `scheduler.min_lr_ratio ×
  optim.lr`.

### Fixed — POET merge loss spike

- **POET merge-and-reinitialize no longer spikes the loss every
  `poet_merge_period` steps** (`src/patches/poet_merge_step.py`). On Megatron's
  mixed-precision optimizers the merge folds the rotation into the frozen weight
  and zeros the **bf16 model** `oft_R`, but the optimizer steps the **fp32
  master** copy — which was left nonzero, so the next `optimizer.step()` copied
  the stale master back and re-applied the just-merged rotation a second time (a
  huge recurring spike the plain-PyTorch GaLore reference never has).
  `_reset_vanilla_oft_state` now also zeros the master *value* (not just the Adam
  moments), and discovers masters on **both** layouts via
  `_iter_model_master_pairs`: `float16_groups`/`fp32_from_float16_groups`
  (single-GPU `Float16OptimizerWithFloat16Params`) and
  `model_float16_groups`/`shard_fp32_from_float16_groups` (multi-GPU
  `DistributedOptimizer`, where the prior id-based reset silently matched 0
  params). Verified on 8-GPU `h100_de` and single-GPU `dev`: loss descends
  smoothly through every merge.

### Changed — POET run name carries oft_R scale

- **POET run names now include the oft_R LR multiplier and order the POET
  segments as `<block>-lr<lr>-scale<v>`** (`src/utils/wandb_naming.py`):
  `wandb_base_name` emits the block parameterization before the LR and trails it
  with a `-scale<v>` segment (`optim.poet.scale`, `:g`-formatted), i.e.
  `...-<bc|bs>-lr<lr>-scale<v>`, e.g. `poet-llama3-300m-bc4-lr0.001-scale0.05`,
  so POET block/scale sweeps are distinguishable on the W&B dashboard. Only the
  canonical W&B run name changes; the on-disk run-dir name
  (`exp-family-scale-s<seed>-<ts>`) is unchanged.

### Added — unified W&B logging

- **Unified W&B metric keys across backends** (`src/utils/wandb_metrics.py`):
  both backends now normalize their core training-health metrics onto one
  canonical schema (`train/loss`, `train/lr`, `train/grad_norm`,
  `train/tokens_seen`, `perf/step_time_s`, `val/loss`) so Megatron and torchtitan
  runs overlay on one dashboard. Applied via a registered Megatron patch
  (`wandb_metric_normalize`, wraps `wandb.log` **lazily inside the `training_log`
  wrapper** — `wandb.init()` rebinds the module-level `wandb.log`, so wrapping at
  patch-apply time was silently clobbered and left Megatron's native keys
  unrenamed; it also computes tokens_seen / step_time, which our runs don't log to
  W&B) and a torchtitan `WandBLogger.log` wrapper in `src/titan_ext/metrics.py`.
  Throughput is deliberately NOT normalized
  (Megatron's `throughput` is TFLOP/s/GPU; torchtitan's `throughput(tps)` is
  normalized by `non_data_parallel_size` — not comparable), so each backend keeps
  its native throughput as a passthrough. Backend-specific extras pass through
  unchanged; no vendored-submodule edits. Design + plan in
  `docs/superpowers/{specs,plans}/2026-05-31-unified-wandb-logging*.md`.
- **Eval/validation in W&B for both backends.** Added a derived `val/ppl`
  (`exp(min(20, val/loss))`, via `wandb_metrics.with_derived`) so eval perplexity
  shows alongside `val/loss` — Megatron logs val PPL only to TensorBoard and
  torchtitan didn't emit it. Enabled **torchtitan validation**: `torchtitan_args`
  emits a `[validation]` block mirroring the Megatron eval cadence
  (`eval_interval`→`freq`, `eval_iters`→`steps`), and `src/titan_ext` monkeypatches
  torchtitan's validation dataloader (`build_text_validation_dataloader`) to a
  **val split of the same Megatron-indexed `GPTDataset`** (same `data.split`), so
  torchtitan's `val/loss` is computed on the same held-out documents as Megatron's
  eval. The vendored `Validator` hardcodes a C4 raw-text loader with no TrainSpec
  hook, hence the monkeypatch (consistent with the existing `titan_ext` pattern).
  Also patched `BaseValidator.should_validate` to **drop torchtitan's step-1
  eval** (`step == 1 or step % freq == 0` → `step % freq == 0`): evaluating an
  untrained model at step 1 produced a huge first point that distorted the curve
  (Megatron only evals at `eval_interval` boundaries). First eval is now at `freq`.
- **`wandb_metric_normalize` enabled in every Megatron experiment**, not just
  adam/champion: added to `poet`, `arch/ngpt`, `muon_hybrid`, and the experiment
  `_template` (alongside the always-on `training_log_eta`). Previously
  `train_poet`/`train_ngpt`/`train_muon` logged native Megatron keys (`lm loss`,
  `learning-rate`, `grad-norm`) instead of the canonical `train/*` schema. The
  torchtitan side is already unconditional (applied on `titan_ext` import).
  Verified all five experiments' patch lists co-register without a `PatchConflict`
  (e.g. POET's `poet_merge_step` wraps `train_step`, `wandb_metric_normalize`
  wraps `training_log`).
- **Removed the `log_grad_norm_extra` patch** (and its `grad-norm-clipped` /
  `grad-norm-clip-coeff` W&B/TensorBoard scalars) from all experiments
  (adam, champion, poet, ngpt, muon_hybrid, template). It was a POET grad-norm
  debugging aid; the POET issue is resolved, so only the raw grad-norm
  (`train/grad_norm`, from Megatron's native `grad-norm`) is kept.

### Added — torchtitan training backend (M1–M2)

- Vendored **torchtitan v0.2.2** as `third_party/torchtitan` (pinned
  `73a0e6979`); pin + bump procedure in `docs/torchtitan_pin.md`; runtime deps in
  the `torchtitan` extra (`pyproject.toml`) and `install_slm_env.sh`.
- New first-class **`backend ∈ {megatron, torchtitan}`** config field (default
  `megatron`). torchtitan runs get a `-torchtitan-` run-name segment and record
  `torchtitan_sha` in the resolved config + launch metadata; Megatron run names
  stay byte-identical.
- `scripts/train_adam.sh --backend torchtitan` dispatches to the new
  `launchers/train_torchtitan.py`, which resolves the same six-axis config, emits
  `<run_dir>/torchtitan.toml`, and builds a `torchrun -m torchtitan.train`
  command. The non-AdamW wrappers (`train_muon/ngpt/poet`) reject
  `--backend torchtitan` with a clear message (AdamW-only in M1).
- `src/utils/torchtitan_args.py`: pure `cfg → (toml, overrides)` mapper (bf16
  AdamW + FSDP2 + WSD scheduler), with **warn-and-skip** for Megatron-only knobs
  (patches, sandwich-norm) so the same experiment yaml runs on either backend.
- `src/titan_ext/` runtime extension (loaded via torchtitan's
  `experimental.custom_import`, **no edits to the vendored submodule**): registers
  an `slm_<family>` TrainSpec cloned from torchtitan's native llama3/qwen3/
  deepseek_v3, adds an `slm_<scale>` flavor for the dense families, and swaps in a
  **Megatron-`GPTDataset`-backed dataloader** so torchtitan reads the same
  `.bin/.idx` corpus. Byte-level data parity vs the Megatron path is the M2 gate
  (`tests/integration/test_titan_megatron_data_parity.py`).
- W&B project/run-name routed via env so both backends share one dashboard.
- **M3 functional training-health gate** (`tests/numerics/test_titan_training_health.py`,
  `@pytest.mark.gpu`) + operator runbook
  (`docs/superpowers/runbooks/2026-05-30-torchtitan-training-health.md`) —
  per-family curves are pending an operator GPU run.
- API surface recorded in `docs/torchtitan_api_notes.md`; design + plan in
  `docs/superpowers/{specs,plans}/2026-05-30-torchtitan-backend*.md`.
- Emit `[comm].init_timeout_seconds` (default **3600s**, override via
  `cluster.comm_init_timeout_seconds`) so rank 0's cold dataset-index build
  doesn't trip torchtitan's 300s default and crash the other ranks at the
  startup barrier. The torchtitan path always cold-builds (different cache hash
  than Megatron), so the 5-min default was insufficient on real corpora.
- `collate_fn=_collate_megatron_to_titan` reshapes each Megatron `GPTDataset`
  sample-dict into torchtitan's required `(input_dict, labels)` 2-tuple
  (`{"input": tokens}`, pre-shifted `labels`), dropping the unused
  attention_mask/loss_mask/position_ids. Without it the first training step
  crashed in `batch_generator` with `too many values to unpack (expected 2)`.
- **Unified W&B run naming** (`src/utils/wandb_naming.py`): both backends now
  derive the same canonical name (`wandb_base_name`: `<exp>-<family>-<scale>-lr<lr>`
  + optimizer segments) and tag it with a `[megatron]` / `[torchtitan]` prefix via
  `wandb_run_name`. torchtitan's `WANDB_NAME` switched from the timestamped run-dir
  name to this shared name, so the two backends' W&B names differ **only** by the
  prefix and land side-by-side on one dashboard. (Run-*dir* names are unchanged.)
- **Megatron 300m aligned to the torchtitan model build** so a Megatron run
  reproduces the torchtitan training curve. torchtitan builds llama3 from its
  native flavor (ignores `ffn_hidden_size`, derives the FFN from `dim`; untied
  embeddings; depth-scaled init), so `configs/base/scale/300m.yaml` now sets
  `ffn_hidden_size: 2816` (matches torchtitan's `dim`-derived width),
  `tie_embeddings: false`, and a new `titan_init: true` flag. The init scheme has
  no Megatron config knob, so `src/model/titan_init.py` re-initializes the built
  model to torchtitan's exact per-weight recipe (std=1.0 embeddings, fixed-0.02
  fan-in / gate, per-layer depth-scaled output·up·down, `dim**-0.5` LM head) —
  wired in `launchers/pretrain_gpt_slm.py` as a post-build, pre-DDP
  `model_provider` wrapper (after unfuse), gated on the config, TP=PP=1, seeded
  for data-parallel-replica determinism. No-op on the torchtitan backend.
  Covered by `tests/unit/test_titan_init.py`.
- **torchtitan per-step console line now matches the Megatron path**
  (`src/titan_ext/metrics.py`, applied via the `custom_import` hook — no submodule
  edit): the `step: … loss: … grad_norm: … tps: … tflops: … mfu: …` line is
  emitted **only on rank 0** (torchtitan's `init_logger` otherwise duplicates it
  per rank) and gains an **`ETA: 1h30m`** segment derived from
  `training.steps` + just-elapsed wall time — same `HhMMm` format as the Megatron
  `training_log_eta` patch. Runtime-wraps `MetricsProcessor.log`, rebinding the
  metrics module logger to a rank-0/ETA proxy for the call; all upstream metric
  math and wandb/tb logging are unchanged. Helpers covered by
  `tests/unit/test_titan_metrics_patch.py`.
- **Fixed torchtitan data starvation** (`src/titan_ext/dataloader.py`): the
  `ParallelAwareDataloader` was built with no `num_workers`, so it defaulted to
  `0` (synchronous, main-process) and the GPU idled ~98% per step waiting for the
  next batch from the shared Megatron `GPTDataset` (mfu ~1.7%; ~5× slower wall
  clock than the Megatron path at 300m/seq256). Now passes `num_workers`
  (= `data.num_workers`, same value Megatron's `--num-workers` uses), `pin_memory`,
  `prefetch_factor`, and `persistent_workers`, so data loading overlaps compute.
  Worker prefetch does not change sample selection or order (same sampler), so the
  training curve is unaffected. Kwargs builder covered by
  `tests/unit/test_titan_dataloader_collate.py`.
