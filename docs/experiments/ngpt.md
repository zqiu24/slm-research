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

## Result log
(populate as runs land)
