# nGPT — Normalized Transformer on the Hypersphere

**Reference:** Loshchilov et al. 2024 — [arXiv:2410.01131](https://arxiv.org/abs/2410.01131). NVIDIA reference impl: https://github.com/NVIDIA/ngpt (vendored at [/lustre/fast/fast/zqiu/tmp/ngpt](file:///lustre/fast/fast/zqiu/tmp/ngpt)).

## Hypothesis
nGPT replaces additive residuals with a per-channel eigen-LR blend on S^{C-1}, normalizes Q/K per head, and enforces per-row/column unit norm on every matrix after each optimizer step. The paper claims 4×–10× speedups at 1k–8k context relative to a standard GPT baseline. We want to see whether this transfers to slm-research's 600M dense ablation track with our frozen tokenizer.

## Mechanism (slm-research integration)
* Custom `NGPTTransformerLayer` overrides `forward` to do hypersphere blending; standard Megatron `SelfAttention` + custom `NGPTMLPBody`. Spec: [src/specs/ngpt_layer_spec.py](../../src/specs/ngpt_layer_spec.py).
* `QKHyperNorm` plugs into `q_layernorm`/`k_layernorm` slots; provides the `sqk` scaling.
* `attn_alpha`, `mlp_alpha` (per-channel eigen LR) live on each `NGPTTransformerLayer`; `suv` lives on `NGPTMLPBody`; `sz` is attached to the GPTModel post-build.
* Per-step weight normalization runs via the `ngpt_normalize_step` patch on `train_step`.
* No QK layernorm, no bias on linears, no LR warmup, AdamW weight-decay zero on scaling params.

## v1 scope (this implementation)
* 600M dense, single-node, TP=1, PP=1, bf16.
* CPU parity test against the vendored NVIDIA reference at toy config (2 layers / 64 hidden / vocab 100).

## v2 candidates (not in this PR)
* TP > 1: per-rank sqk/suv sharding.
* MoE flavour (nGPT-MoE).
* MLA (nGPT-MLA) compatibility.
* FP8 / FP4 — paper notes nGPT is less sensitive to low precision than baseline GPT, so this is an interesting cross-axis ablation.

## How to run
```bash
python -m launchers.submit \
    base/family=llama3 \
    base/scale=600m \
    experiment=arch/ngpt \
    training_regime=ablation_20x \
    cluster=h800_cn \
    seed=42
```

## How to run the CPU / parity test suite

Two tiers (measured 2026-06-09, `slm_env` venv `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`):

**1. Pure-PyTorch parity oracle — plain CPU, no env setup.** Since the pure-torch
`NGPTBlock` lives in [src/model/ngpt/block.py](../../src/model/ngpt/block.py) (split out of
`layer.py` so it pulls in no Megatron), the parity tests run on any CPU without
transformer_engine:
```bash
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest \
  tests/unit/test_ngpt_layer_block_forward.py tests/unit/test_ngpt_full_parity.py -q
```

**2. Full nGPT suite (incl. the Megatron `ModuleSpec` tests) — needs the cuBLAS fix.**
`test_ngpt_layer_spec.py` does `import megatron.core`, which transitively imports
`transformer_engine`. On the login node TE needs the symbol
`cublasLtGroupedMatrixLayoutInit_internal@libcublasLt.so.13`, exported **only** by the
system `cuda-13.2` lib (the venv-bundled `nvidia-cublas==13.1.0.3` lib does not export it),
and it must be `LD_PRELOAD`-ed to win the soname race against torch's RTLD_GLOBAL load.
[load_cuda13_2_nccl_env.sh](../../load_cuda13_2_nccl_env.sh) encodes exactly this — source it
first (CPU-only; no GPU required to *import*):
```bash
source load_cuda13_2_nccl_env.sh   # sets LD_PRELOAD=/is/software/nvidia/cuda-13.2/lib64/libcublasLt.so.13
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_ngpt_*.py -q
```
Verified result (2026-06-09): **32 passed** for the full `tests/unit/test_ngpt_*.py` suite
(`test_ngpt_layer_spec.py` → 3 passed). The plan's original Step 2.1 attempt
(`LD_LIBRARY_PATH` → venv-bundled cuBLAS) does **not** work: the venv lib lacks the symbol
and `LD_LIBRARY_PATH` does not override torch's already-loaded soname.

## Result log

### Stage 0 gate — GREEN (2026-06-09)

Validation work lives on branch `ngpt-v1-validation` (Tasks 1–3 done CPU-only). The three
Stage 0 gate conditions all hold:

1. **Full nGPT suite green in the working env.** `source load_cuda13_2_nccl_env.sh` +
   `pytest tests/unit/test_ngpt_*.py` → **32 passed, 0 failed** (incl. the 3 Megatron
   `ModuleSpec` tests in `test_ngpt_layer_spec.py`).
2. **Pure-torch parity oracle green on plain CPU (no transformer_engine).** After splitting
   `NGPTBlock` into [src/model/ngpt/block.py](../../src/model/ngpt/block.py),
   `pytest test_ngpt_layer_block_forward.py test_ngpt_full_parity.py` → **3 passed** with no
   `LD_PRELOAD`/cuBLAS setup.
3. **Config-parity: arms matched except by intent.**
   [scripts/ngpt_config_parity.py](../../scripts/ngpt_config_parity.py) diffs the two
   `--dry-run` resolved configs (nGPT `experiment=arch/ngpt` vs matched baseline
   `experiment=optim/adam base.model.num_query_groups=20`, both 600m × `ablation_40x`,
   `cluster=h100_de`, seed 0, untied, `transformer_impl=local`, gbs 1024 / mbs 128) and prints
   **`OK: arms differ only by the intended method/recipe deltas.`** Both arms resolve to
   `total_tokens = 24,000,000,000` and `parallelism.tp = 1`. All architecture keys match:
   `num_attention_heads = num_query_groups = 20` (MHA override took on the baseline),
   `hidden_size 1280`, `ffn_hidden_size 3200`, `num_layers 40`, `seq_length 4096`,
   `tie_embeddings false`, `seed 0`, and identical `data.{tokenizer_model,path,vocab_size,split}`.
   The only diffs are `optim.*`, `experiment.*`, and `_derived.*` (run name / hashes / timestamps).

(Remaining: Stage 0.5a CPU step-parity → Task 5; Stage 0.5b/c 1-GPU Megatron-layer parity →
Task 6; Stage 1 smoke → Task 7; Stages 2–3 600m A/B → Tasks 8–9.)
