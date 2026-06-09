# nGPT v1 Validation — Design / Spec

**Date:** 2026-06-09
**Status:** Approved for planning
**Author:** Zeju (with Claude Code)

## Context

nGPT (Normalized Transformer on the Hypersphere, [arXiv:2410.01131](https://arxiv.org/abs/2410.01131)) is **already implemented and merged on `main`** in slm-research. The implementation was built against the NVIDIA reference vendored at [/lustre/fast/fast/zqiu/tmp/ngpt](file:///lustre/fast/fast/zqiu/tmp/ngpt) and consists of:

- Model module [src/model/ngpt/](file:///lustre/fast/fast/zqiu/slm-research/src/model/ngpt/) (`normalize`, `scaling_params`, `attention`, `mlp`, `layer`, `output_scaling`).
- Megatron wiring: [src/specs/ngpt_layer_spec.py](file:///lustre/fast/fast/zqiu/slm-research/src/specs/ngpt_layer_spec.py) + patches `ngpt_apply_spec`, `ngpt_normalize_step`, `ngpt_optimizer_setup`.
- Config [configs/experiments/arch/ngpt.yaml](file:///lustre/fast/fast/zqiu/slm-research/configs/experiments/arch/ngpt.yaml), launcher flags, [scripts/train_ngpt.sh](file:///lustre/fast/fast/zqiu/slm-research/scripts/train_ngpt.sh).
- 11 unit test files including a full-model logit parity test vs the vendored NVIDIA reference.
- Implementation plan [docs/superpowers/plans/2026-05-25-ngpt-architecture-variant.md](file:///lustre/fast/fast/zqiu/slm-research/docs/superpowers/plans/2026-05-25-ngpt-architecture-variant.md), lab notebook [docs/experiments/ngpt.md](file:///lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md).

**What is missing is validation, not implementation.** No training run has ever been logged (the result log in the lab notebook is still empty), and on the current login node the full CPU test suite cannot run end-to-end.

## Problem statement

Prove that the existing nGPT v1 is (a) numerically faithful to the reference, (b) trains end-to-end on the cluster without pathologies, and (c) actually delivers the paper's central claim — a convergence speedup over a matched baseline — at the slm-research 600M dense scale.

## Goals

1. Full nGPT CPU test suite green (all 11 files), including the NVIDIA-reference parity test.
2. **Reference parity (correctness):** prove our nGPT reproduces the NVIDIA reference's computation via a **weight-transfer step-parity** check — transfer reference weights into our model, feed an identical batch, and compare the forward and the post-step weights (after one AdamW step + one weight-normalization). This is the correctness gate, separate from the speedup A/B.
3. nGPT trains end-to-end on `h800_cn` for a short smoke (~100 steps): loss ↓, no NaN, per-step weight normalization fires, optimizer param-group split correct.
4. A real nGPT 600M ablation loss curve, logged.
5. A **matched-baseline A/B** quantifying nGPT's convergence *speedup* vs a standard AdamW baseline at the same scale / data / seed.

## Non-goals (explicit)

- No new nGPT features. v2 items (TP>1 sqk/suv sharding, nGPT-MoE, nGPT-MLA, FP8/FP4) are out of scope.
- No re-implementation. The merged code is the artifact under test; we change it only where validation exposes a defect or an env/portability blocker.
- No hyperparameter sweep. One nGPT recipe (the config-native one) vs one baseline recipe.
- **The AdamW A/B is NOT a correctness check.** It measures speedup only. Correctness is established against the *reference implementation* in Goal 2, before any of our own runs are trusted.

## Why weight-transfer step parity (not data-matched training curves)

The natural first instinct — feed the same tokens to both the reference and our codebase and compare loss curves — does **not** yield a conclusive correctness signal, and a constraint blocks the simplest data port:

- **uint16 blocker:** nanoGPT stores tokens as `uint16` (vocab ≤ 65536); all our tokenizers are larger (llama3 128256, qwen3 151936). Reusing our tokens in the reference needs a `uint16`→`uint32` patch to the vendored reference; using the reference's data needs obtaining + GPT-2-tokenizing OpenWebText. Either is possible but neither is free.
- **Confounded curves:** even with identical tokens, training-from-scratch curves diverge for reasons unrelated to nGPT correctness — RoPE convention (reference: interleaved sinusoidal, base 10000; ours: Megatron RoPE, base 500000), attention kernel (`flash_attn` vs Megatron flash/cuDNN), bf16 param storage vs fp32-master mixed precision, init, and data-sampling order. A curve mismatch would not prove our nGPT is wrong, nor a match prove it right.
- **Decisive alternative:** transferring reference weights + feeding identical inputs removes init/RNG/data-order/precision noise, so any discrepancy is attributable to the implementation. The data content becomes incidental. This is the chosen approach.

## Current ground truth (measured 2026-06-09)

- `pytest tests/unit/test_ngpt_*.py` on this login node (`slm_env` venv): **26 passed, 3 failed, 2 collection errors**.
- All 5 non-passing tests fail for **one** root cause: `import megatron.core` → `import transformer_engine` raises `OSError: ... libtransformer_engine.so: undefined symbol: cublasLtGroupedMatrixLayoutInit_internal, version libcublasLt.so.13`. This node's NVIDIA driver is ancient (reported version 8000); it is a CPU/login node. The TE `.so` in `slm_env` is built against a cublasLt this node does not provide.
- The 5 blocked tests: the 3 in [test_ngpt_layer_spec.py](file:///lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_layer_spec.py) (need `build_ngpt_layer_spec` → Megatron `ModuleSpec`), and the 2 parity tests [test_ngpt_full_parity.py](file:///lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_full_parity.py) + [test_ngpt_layer_block_forward.py](file:///lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_layer_block_forward.py) (import `src.model.ngpt.layer`, which imports Megatron's `TransformerLayer` at module top).
- The 26 passing tests are pure-PyTorch and exercise the core nGPT math (`justnorm`, weight projection, `LearnedScaling`, `QKHyperNorm`, `NGPTMLPBody`, `attach_sz_scaling`, optimizer grouping, megatron-args emission, patch registry).

**Conclusion:** the nGPT logic is sound; the blocker is environmental. Two of the five blocked tests (the `NGPTBlock` parity tests) only need pure PyTorch and can be unblocked by a lazy Megatron import. The other three genuinely need a working Megatron env.

## Architecture under test (recap)

nGPT replaces additive residuals with a per-channel eigen-LR blend on the unit hypersphere, normalizes Q/K per head (with learnable `sqk`), scales the SwiGLU intermediate (`suv`) and logits (`sz`), uses `softmax_scale = sqrt(head_dim)`, and projects every matrix to unit row/column norm after each optimizer step. No QK-layernorm, no bias, no LR warmup, zero weight-decay on scaling params.

## Design — three stages with explicit gates

### Stage 0 — Full CPU test suite green (Claude does this)

**0a. Establish a Megatron-importable env.** Identify or repair an environment where `python -c "import megatron.core"` succeeds, so the 3 layer-spec tests and the full-model parity test can run. Candidates, in order of preference:
   1. A repaired/alternate local venv whose `transformer_engine` matches the node's CUDA libs (CPU-only TE import is acceptable; the tests do not need a GPU).
   2. Running the suite as a short CPU job on a cluster node where `slm_env` loads cleanly.

   The exact working env + command is recorded in the lab notebook so the suite is reproducible.

**0b. Lazy-import hardening (small code change).** Move the `from megatron.core.transformer.transformer_layer import TransformerLayer` import in [src/model/ngpt/layer.py](file:///lustre/fast/fast/zqiu/slm-research/src/model/ngpt/layer.py) out of module top-level and into `NGPTTransformerLayer` (e.g. a module-local lazy import or guarded under `TYPE_CHECKING` + import inside `__init__`). Result: importing `NGPTBlock` (pure PyTorch) no longer pulls in Megatron/TE, so the two `NGPTBlock` parity tests run on any CPU. `NGPTTransformerLayer` (the Megatron path) is unchanged at runtime. Re-run the 26 + 2 = 28 non-layer-spec tests on this node to confirm they pass without TE.

**0c. Config-parity dry-run (CPU, Claude does this).** Use `python -m launchers.submit ... --dry-run` (resolves + archives config, skips SLURM) for both the nGPT arm and the matched baseline arm. Capture the generated Megatron args for each and diff them. Confirm the **only** differences are the intended ones (nGPT spec/patches/optimizer recipe + the baseline's `num_query_groups` override), catching config drift before any GPU time.

**Stage 0 gate:** `pytest tests/unit/test_ngpt_*.py` returns **zero failures** in the chosen env; the broader unit suite shows no nGPT-induced regressions; the dry-run arg diff contains only intended deltas.

### Stage 0.5 — Reference weight-transfer step parity (correctness gate)

The decisive correctness check, in two layers (the existing toy logit-parity test already covers pure-torch forward at init; this strengthens it through a training step and extends it to the Megatron path):

- **0.5a — Full-model forward + one-step parity, pure-torch (CPU, Claude).** Reuse the existing `_OurNGPT` assembly + `_copy_ref_to_ours` weight transfer from [tests/unit/test_ngpt_full_parity.py](file:///lustre/fast/fast/zqiu/slm-research/tests/unit/test_ngpt_full_parity.py). After transferring reference weights and running the init normalization on both sides: (i) assert forward logit/loss parity (existing); (ii) run one identical CE backward + one AdamW step (matched lr/betas/eps, wd=0) + one weight-normalization on **both** the reference (`normalize_matrices`) and ours (`normalize_module_matrices`); (iii) assert post-step parity on sampled tensors (a layer's `query` weight, an `alpha`, `sz`, `wte`) within a documented fp32 tolerance. This validates every production primitive in [src/model/ngpt/](file:///lustre/fast/fast/zqiu/slm-research/src/model/ngpt/) — residual blend, Q/K-norm + `sqk`, `suv`, `sz`, the matrix projection, and the optimizer grouping — deterministically, with no kernel/data noise.

- **0.5b — Single-layer Megatron parity vs reference (1 GPU, User).** The pure-torch oracle is not what trains; the Megatron `NGPTTransformerLayer` + spec is. Build one `NGPTTransformerLayer` from [src/specs/ngpt_layer_spec.py](file:///lustre/fast/fast/zqiu/slm-research/src/specs/ngpt_layer_spec.py) on a single GPU, transfer the corresponding reference `Block` weights into it (handling qkv/fc1 fusion state), feed an identical hidden-state input, and compare its output to the reference `Block` within tolerance. This validates the wiring the pure-torch test cannot: `QKHyperNorm` in the `q_layernorm`/`k_layernorm` slots applied post-RoPE, `softmax_scale = sqrt(head_dim)`, `NGPTMLP`'s `suv` path, and the residual blend inside the Megatron forward. Full-GPTModel-in-a-test infra does not exist in this repo, so a single layer is the tractable decisive unit.

- **0.5c — RoPE convention check (1 GPU, User; sub-part of 0.5b).** The reference uses interleaved sinusoidal RoPE (base 10000); Megatron defaults to a different convention/base. Run 0.5b first with Megatron RoPE configured to match (`rotary_interleaved=True`, `rotary_base=10000`). If parity holds only with RoPE disabled on both sides, that localizes the discrepancy to RoPE — a real finding about how our training differs from the published recipe — recorded as a deviation (it may or may not matter for nGPT's claims, but we must know).

**Stage 0.5 gate:** 0.5a passes within tolerance on CPU; 0.5b single-layer Megatron output matches the reference within tolerance on 1 GPU (with the RoPE configuration documented per 0.5c). Any mismatch is a correctness defect fixed before Stage 1.

### Stage 1 — GPU smoke, ~100 steps (User runs; Claude authors command + checklist)

Run the existing [smoke runbook](file:///lustre/fast/fast/zqiu/slm-research/docs/superpowers/runbooks/2026-05-25-ngpt-smoke.md) on `h800_cn` (1 node, 8 GPU) with a small `+training.total_tokens` cap (~100–500 steps). Confirm, from rank-0 stdout + W&B:

- `[nGPT] applied spec + attached sz + registered weight-norm roles` appears after model build.
- Training loss strictly decreasing across the first ~50 steps; **no NaN/Inf**.
- After ~10 steps, a sampled projected matrix (e.g. `...self_attention.linear_qkv.weight`) has **row-norms ≈ 1.0** — proves the post-step projection fires.
- W&B shows distinct `lr_groups/decay` vs `lr_groups/no_decay`, and the no-decay group contains `sz, sqk, suv, attn_alpha, mlp_alpha` (expected count `≈ 2·num_layers` alphas).
- **Correctness item — embedding tying:** 600M base sets `tie_embeddings: true`, but the nGPT reference unties `wte`/`lm_head` and normalizes them as distinct matrices. Verify the nGPT run's behavior under tying is intended (either the config unties for nGPT, or the weight-norm role map + `sz` handle the tied tensor without double-projection). Resolve before Stage 2.

**Stage 1 gate:** all bullets above observed; user reports success. Failures are triaged with the runbook's "If it fails" table and fixed on a branch before proceeding.

### Stage 2 — nGPT 600M ablation (User runs; Claude authors command + interprets)

Full ablation: `base/family=llama3 base/scale=600m experiment=arch/ngpt training_regime=ablation_20x cluster=h800_cn seed=<S>`. At 600M non-embedding params × 20 tok/param ≈ **12B tokens** → ≈ 2.9M samples at seq 4096 → ≈ 2.9k steps (gbs 1024). Produces the nGPT loss curve; populate the result log in [docs/experiments/ngpt.md](file:///lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md).

**Stage 2 gate:** run completes without divergence; final val loss recorded; curve logged to W&B and the lab notebook.

### Stage 3 — Matched-baseline A/B (User runs; Claude authors commands + computes verdict)

Baseline arm: `experiment=optim/adam` with the architecture matched to nGPT —
`base.model.num_query_groups=${base.model.num_attention_heads}` (force MHA; bias is already disabled repo-wide via `--disable-bias-linear`). Same `base/scale=600m`, same `training_regime=ablation_20x`, **same seed and data** as the Stage 2 nGPT run. Only the method differs.

- **By-design recipe deltas (not confounds — they are each method's published recipe):** nGPT uses lr 15e-4, zero weight-decay, no warmup, hypersphere normalization; the AdamW baseline uses lr 1e-3, weight-decay 0.1, cosine+warmup. We compare each method *with its own recipe* but on a **matched architecture / data / seed**, mirroring the paper.
- **Documented deviation (not chased):** the NVIDIA reference stores parameters in bf16 (its README notes this inflates the reported speedup). slm-research uses its standard mixed precision for **both** arms; this is the honest in-repo comparison. Recorded as a deviation in the notebook.
- **Verdict metrics:** val-loss-vs-tokens for both arms; report (i) loss at equal tokens and (ii) tokens-to-reach a fixed target loss → the speedup factor. `wandb_metric_normalize` is on for both arms, so `tokens_seen` / step-time keys align for an apples-to-apples curve.

**Stage 3 gate:** both arms complete at matched budget; speedup factor (or its absence) computed and written to the lab notebook as the validation verdict.

## Division of labor (per standing user rules)

- **Claude:** Stage 0 in full + Stage 0.5a (env, lazy-import change, CPU full-model step-parity test, run CPU tests + report real output, dry-run config-parity diff). Author the 0.5b/0.5c GPU test scaffold + every cluster command and per-stage paste-back checklist. Update [docs/experiments/ngpt.md](file:///lustre/fast/fast/zqiu/slm-research/docs/experiments/ngpt.md) and [NeckariumAI/zqiu/CHANGELOG.md](file:///lustre/home/zqiu/NeckariumAI/zqiu/CHANGELOG.md) as work lands.
- **User:** all GPU runs — Stage 0.5b/0.5c (single-layer parity, quick 1-GPU) and Stages 1–3 (cluster training). This node has no usable GPU (A100+). Claude hands exact commands and stops; user reports back and Claude interprets. Per policy, Claude never launches GPU/cluster jobs unprompted.

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| No Megatron-importable env can be found locally | Run the 5 Megatron tests as a short CPU job on a cluster node; record env in notebook. |
| Lazy-import refactor changes runtime behavior | Change is import-location only; covered by re-running the layer-spec + parity tests in a working env. |
| `tie_embeddings=true` conflicts with nGPT's untied wte/lm_head normalization | Explicit Stage-1 correctness item; resolve before the ablation. |
| Baseline not truly matched → uninterpretable speedup | Stage-0 dry-run arg diff confirms only intended deltas; matched MHA + same data/seed. |
| Smoke divergence / NaN | Runbook "If it fails" triage table (softmax_scale, spec-swap firing, projection firing, alphas in optimizer). |
| Cross-framework weight map wrong (fusion/naming) → false parity failure | Reuse the proven `_copy_ref_to_ours` map for 0.5a; for 0.5b assert shapes match before copy and verify the unfused qkv/fc1 layout (experiment sets `unfuse_qkv/unfuse_fc1=true`). |
| RoPE convention mismatch masks/causes 0.5b failure | 0.5c isolates it: run with matched (interleaved, base 10000) and with RoPE off; attribute the delta explicitly. |

## Acceptance (overall)

Validation is complete when: Stage 0 gate green (full CPU suite passes, dry-run diff clean), **Stage 0.5 reference parity green (0.5a CPU step-parity within tolerance; 0.5b single-layer Megatron parity within tolerance, RoPE documented)**, Stage 1 smoke reported green, Stage 2 nGPT curve logged, and Stage 3 produces a documented speedup verdict (factor or null result) in the lab notebook. No code changes beyond the Stage-0 lazy-import hardening unless a stage exposes a defect.
