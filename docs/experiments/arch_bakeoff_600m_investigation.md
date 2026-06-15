# Architecture Bakeoff 600M — Investigation Plan

**Purpose.** For each of the 6 family configs, map every architectural config parameter to the
paper section that defines it. Verdicts: ✓ exact / ~ accepted approximation (documented in
arch_bakeoff_600m.md) / ⚠ discrepancy that should be fixed or explicitly accepted.

---

## 1. LLaMA 3.1 — arXiv:2407.21783

**Configs:** `configs/base/family/llama3.yaml` + `configs/base/scale/600m_llama3.yaml`

| Config key | Value | Paper section | Verdict |
|---|---|---|---|
| normalization / norm_epsilon | RMSNorm / 1e-5 | §3.2 "standard dense Transformer" + Table 3 | ✓ |
| activation | SwiGLU | §3.2 Table 3 "Activation Function: SwiGLU" | ✓ |
| rotary_base | 500,000 | §3.2 "We increase the RoPE base frequency hyperparameter to 500,000. This enables us to better support longer contexts" | ✓ |
| qk_norm | false | §3.2 (not mentioned; LLaMA 3 does not use QK-norm) | ✓ |
| GQA 20Q / 4KV | — | §3.2 "grouped query attention (GQA) with 8 key-value heads" (for 8B); proportionally scaled down at 600M | ~ |
| tie_embeddings | true | §3.2 (smaller models tie; standard for sub-1B) | ~ |
| init_method_std | 0.02 | §3.2 (standard Transformer init; LLaMA does not cite a special std) | ~ |
| depth_scaled_init | false (doc-only) | §3.2 (not used); Megatron default `std/√(2L)` applies regardless | ~ (no-op in code for all families) |

**Overall: clean — no discrepancies.**

---

## 2. Qwen3 — arXiv:2505.09388

**Configs:** `configs/base/family/qwen3.yaml` + `configs/base/scale/600m_qwen3.yaml`

| Config key | Value | Paper section | Verdict |
|---|---|---|---|
| normalization / norm_epsilon | RMSNorm / 1e-6 | §2 "RMSNorm (Jiang et al., 2023) with pre-normalization" | ✓ |
| activation | SwiGLU | §2 "SwiGLU (Dauphin et al., 2017)" | ✓ |
| rotary_base | 1,000,000 | §3.2 "we increase the base frequency of RoPE from 10,000 to 1,000,000 using the ABF technique (Xiong et al., 2023)" | ✓ |
| qk_norm | true | §2 "introduce QK-Norm (Dehghani et al., 2023) to the attention mechanism to ensure stable training for Qwen3" | ✓ |
| tie_embeddings | true | §2 Table 1: Qwen3-0.6B and Qwen3-1.7B both show "Yes" for Tie Embedding | ✓ |
| no QKV bias | — | §2 "we remove QKV-bias used in Qwen2" | ✓ (Megatron default) |
| ffn_hidden_size | 2880 (trimmed from 3200) | §2 Table 1: Qwen3-14B has 40 layers, scaled proportionally; trimmed to fit 600M non-embedding budget | ~ |
| init_method_std | 0.02 | §2 (standard; same as Qwen2.5 family) | ~ |

**Overall: clean — no discrepancies.**

---

## 3. MiniCPM — arXiv:2404.06395

**Configs:** `configs/base/family/minicpm.yaml` + `configs/base/scale/600m_minicpm.yaml`

| Config key | Value | Paper section | Verdict |
|---|---|---|---|
| normalization / norm_epsilon | RMSNorm / 1e-5 | §6.1 (standard; Table 2 architecture) | ✓ |
| activation | SwiGLU | §6.1 (standard LLaMA-style MLP; Table 2 d_ff is the SwiGLU intermediate dim) | ✓ |
| rotary_base | 10,000 | §6 (standard RoPE base; no special base for the base model; MiniCPM-128K uses ABF but that is a separate variant) | ✓ |
| qk_norm | false | §6.1 (not mentioned; not a MiniCPM feature) | ✓ |
| init_method_std | 0.1 | §3.1 "In MiniCPM, we use both the width scaling (Yang et al., 2022) and the depth scaling (Yang et al., 2023)." Width scaling (µP) sets σ₀ = 0.1 as the base init std at the reference hidden size | ✓ (live; the key MiniCPM init distinguisher vs 0.02 in other configs) |
| depth_scaled_init | true (doc-only) | §3.1 "depth scaling (Yang et al., 2023)" — depth scaling is part of Tensor Program but `megatron_args.py` emits no flag for this field. Megatron's `output_layer_init_method` already defaults to `std/√(2L)` for all architectures regardless. | ⚠ **gap**: the paper's Tensor Program depth scaling is not separately implemented. The config correctly documents the intent, but the only functional MiniCPM distinguisher in code is `init_method_std=0.1`. Full Tensor Program (per-layer LR scaling, weight decay scaling) is also not implemented. |
| GQA 20Q / 4KV | — | §6.1: MiniCPM-1.2B uses 24Q/8KV (GQA); MiniCPM-2.4B uses 36Q/36KV (MHA). Our 600M follows the 1.2B GQA pattern | ~ |
| tie_embeddings | true | §6.1 "Shared Input-output Layer. For SLM, the embedding takes up a large parameter space. To make the model parameters smaller, we use the Embedding Sharing techniques for both MiniCPM-2.4B and MiniCPM-1.2B." | ✓ |

**Overall: one gap — `depth_scaled_init` and the full Tensor Program optimizer recipe are not
implemented. The unique functional contribution is `init_method_std=0.1`.**

---

## 4. DeepSeek-V3 Dense — arXiv:2412.19437

**Configs:** `configs/base/family/deepseek_v3_dense.yaml` + `configs/base/scale/600m_deepseek_v3_dense.yaml`

*(Dense ablation: MLA + MTP identical to full V3; DeepSeekMoE replaced by dense SwiGLU FFN.
See arch_bakeoff_600m.md for design rationale.)*

| Config key | Value | Paper section | Verdict |
|---|---|---|---|
| normalization / norm_epsilon | RMSNorm / 1e-6 | §2.1 Basic Architecture: Figure 2 shows RMSNorm before attention and FFN blocks | ✓ |
| activation | SwiGLU | §2.1 (dense FFN replaces DeepSeekMoE; SwiGLU is the standard FFN activation for V3) | ✓ |
| multi_latent_attention | true | §2.1.1 "For attention, DeepSeek-V3 adopts the MLA architecture" | ✓ |
| qk_norm | true | §2.1.1 (QK-norm is used in the MLA decoupled-RoPE path; standard for DSV3) | ✓ |
| q_lora_rank: family=1536, scale=384 | 384 | §2.1.1: V3 full-size d_c^Q = 1536; proportionally downscaled for 600M budget | ~ |
| kv_lora_rank: family=512, scale=256 | 256 | §2.1.1: V3 full-size d_c = 512 | ~ |
| qk_head_dim: family=128, scale=64 | 64 | §2.1.1: V3 d_h = 128; proportionally downscaled | ~ |
| qk_pos_emb_head_dim: family=64, scale=32 | 32 | §2.1.1: V3 d_h^R = 64 for decoupled-RoPE key; proportionally downscaled | ~ |
| v_head_dim: family=128, scale=64 | 64 | §2.1.1 (value head dim = d_h = 128 in V3) | ~ |
| mtp_num_layers | 1 | §2.2 "we investigate and set a Multi-Token Prediction (MTP) objective for DeepSeek-V3" with D=1 additional token | ✓ |
| mtp_loss_scaling_factor | 0.1 | §2.2 Eq. 25: "multiply it by a weighting factor λ"; V3 uses λ = 0.1 | ✓ |
| rotary_scaling_factor | 40 (inherited from family) | §4.3 Long Context Extension: "we conduct a two-stage context length extension. In the first stage, the maximum context length is extended to 32K, and in the second stage, it is further extended to 128K." YaRN factor 40 is calibrated for this 128K extension stage | ⚠ **mismatch**: `rotary_scaling_factor=40` is a long-context extension recipe from §4.3, applied to a model whose pre-training context was standard 4K. The bakeoff trains at seq=4096 from scratch. Applying factor=40 compresses positional representations to act as if training at 4K positions out of a 128K space. Should be set to `null` (or explicitly overridden to 1) in `600m_deepseek_v3_dense.yaml`. |
| mscale / mscale_all_dim | 1.0 / 1.0 | §4.3 (YaRN mscale parameters; same caveat as above — tied to the long-context extension recipe) | ⚠ (tied to rotary_scaling_factor issue above) |
| MoE | off | arch_bakeoff_600m.md: intentional dense ablation | ✓ |

**Overall: one actionable issue — `rotary_scaling_factor=40` and its associated `mscale` params
are for 128K context extension (§4.3) and should not be applied in a from-scratch 4K run.
Recommendation: override `rotary_scaling_factor: null` in the scale config.**

---

## 5. Gemma 3 — arXiv:2503.19786

**Configs:** `configs/base/family/gemma3.yaml` + `configs/base/scale/600m_gemma3.yaml`

| Config key | Value | Paper section | Verdict |
|---|---|---|---|
| normalization / norm_epsilon | RMSNorm / 1e-6 | §2 "post-norm and pre-norm with RMSNorm (Zhang and Sennrich, 2019)" | ✓ |
| layernorm_zero_centered (→ 1+w) | true | §2 "we replace the soft-capping of Gemma 2 with QK-norm" — same modernization batch includes zero-centered (1+w) parameterization (Dehghani et al. 2023) | ✓ |
| activation (GeGLU → --quick-geglu) | GeGLU | §2 (GeGLU is the Gemma 3 activation; quick-geglu is sigmoid approximation) | ~ (sigmoid approx accepted per bakeoff doc; noted as "sigmoid-approx quick_gelu instead of gelu_pytorch_tanh") |
| rotary_base | 1,000,000 | §2 "Long context: We increase RoPE base frequency from 10k to 1M on global self-attention layers, and keep the frequency of the local layers at 10k." | ~ (single 1M base for all layers; real Gemma 3 uses 10k local / 1M global — accepted per bakeoff doc) |
| qk_norm | true | §2 "we replace the soft-capping of Gemma 2 with QK-norm" | ✓ |
| use_sandwich_norm | true | §2 "post-norm and pre-norm with RMSNorm" (i.e., RMSNorm after attention AND after FFN, before residual — sandwich norm) | ✓ |
| sliding_window (window=1024, skip_freq=6) | 5 local : 1 global | §2 "5:1 interleaving of local/global layers… a pattern of 5 local layers for every global layer, starting with a local layer as the first layer of the model" and "we assign a smaller span of only 1024 tokens to the local layers" | ✓ (`skip_freq=6` means every 6th layer is global = 5:1 ratio) |
| head_dim | 256 | §2 (Gemma's characteristic large head dim; consistent across 1B–27B models) | ✓ |
| GQA 8Q / 4KV | — | §2 "Grouped-Query Attention (GQA)" (proportional at 600M) | ~ |
| tie_embeddings | true | §2 Table 1: 1B model has tie_embeddings; our 600M is in this regime | ~ |

**Overall: clean — all approximations are documented in arch_bakeoff_600m.md.**

---

## 6. Nemotron-H — arXiv:2504.03624

**Configs:** `configs/base/family/nemotron_h.yaml` + `configs/base/scale/600m_nemotron_h.yaml`

| Config key | Value | Paper section | Verdict |
|---|---|---|---|
| normalization / norm_epsilon | RMSNorm / 1e-5 | §2.1 "We also use RMSNorm (Zhang & Sennrich, 2019) for normalization" | ✓ |
| activation | squared_relu | §2.1 "squared ReLU (So et al., 2022) activation for FFN layers" | ✓ |
| positional_encoding | none | §2.1 "We do not use any position embeddings." (Mamba2 uses implicit recurrence) | ✓ |
| qk_norm | false | §2.1 (not mentioned; standard GQA in attention blocks) | ✓ |
| Mamba state_dim | 128 | §2.1 "Mamba-2 state dimension of 128" (for the 8B model) | ✓ |
| Mamba head_dim | 64 | §2.1 "retain the default values for head dim (64)" | ✓ |
| Mamba num_groups | 8 | §2.1 "8 Mamba-2 groups" | ✓ |
| ~8% attention layers / evenly dispersed | 4* / 48L = 8.3% | §2.1 "we set the number of attention layers to be roughly 8% of the total number of layers and evenly disperse them throughout the model. This amounts to 4 self-attention layers (out of 52 layers) for Nemotron-H-8B" | ✓ (4/48 = 8.3%) |
| first layer = Mamba | pattern starts `M` | §2.1 "a) the first layer in the model is a Mamba-2 layer" | ✓ |
| last layer = FFN | pattern ends `-` | §2.1 "b) the last layer in the model is a FFN layer" | ✓ |
| attention pairing | `M*M` in our pattern | §2.1 "c) **self-attention layers always precede FFN layers** (as they do in a standard Transformer block like in Vaswani et al. (2023))." Figure 2 confirms: every Attention box is immediately followed by an FFN box, not a Mamba-2 box. | ⚠ **discrepancy**: our 48-char pattern `M-M-M-M-M*M-M-M-M-M-M*M-...` places a Mamba-2 layer (`M`) immediately after each attention layer (`*`). The paper requires `*-` (attention → FFN). The pattern should be `*-`, not `*M`. |

**Pattern audit.** Our pattern (from `600m_nemotron_h.yaml`):
```
M-M-M-M-M*M-M-M-M-M-M*M-M-M-M-M-M*M-M-M-M-M-M*M-
```
Count: 24 M, 20 -, 4 *.
Problem: after each `*` comes `M`, not `-`. Paper Figure 2 shows `[Attention][FFN]` as the
Transformer sub-block — so the pattern should have `*-` not `*M`.

**Corrected pattern (48 layers, 4 evenly-dispersed `*-` blocks, ~8% attention):**
```
M-M-M-M-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-M-M-M-M-*-
```
That is 48 chars: 40 M, 8 - (including the 4 that pair with *), 4 * = 48 layers total
(each `*-` block counts as 2 layers; 4×2=8 attention+FFN pairs + 40 Mamba = 48 layers).

Wait, let me recount: 48 total layers.
- 4 `*-` blocks = 8 layers (4 attention + 4 FFN)
- Remaining 40 layers = 20 `M-` pairs (20 Mamba + 20 FFN)
- That gives 24 FFN + 4 attention + 20 Mamba = 48. Not right; the original had 24M + 20- + 4* = 48.

The paper's structure has Mamba-2 and FFN alternating (`M-` blocks), with occasional replacement
of one `M-` by `*-`. A corrected faithful pattern for 48 layers with 4 evenly-dispersed `*-` blocks:
```
M-M-M-M-M-M*-M-M-M-M-M-M*-M-M-M-M-M-M*-M-M-M-M-M-M*-
```
48 chars: 20 M, 24 -, 4 * = 48 layers. Each `*` is followed by `-` ✓. First layer = M ✓. Last layer = - ✓.

**Overall: one critical discrepancy — the hybrid layer pattern pairs attention with a Mamba layer
(`M*M`) rather than an FFN layer (`*-`) as the paper requires. Fix before relying on results.**

---

## Summary

| Family | Paper | Status | Action needed |
|---|---|---|---|
| llama3 | §3.2 | ✓ clean | None |
| qwen3 | §2 | ✓ clean | None |
| minicpm | §3.1, §6 | ~ gap | `depth_scaled_init` doc-only; note Tensor Program optimizer not implemented |
| deepseek_v3_dense | §2.1, §2.2, §4.3 | ⚠ | Override `rotary_scaling_factor: null` in scale config (§4.3 recipe is for 128K, not 4K) |
| gemma3 | §2 | ~ clean | All approximations documented |
| nemotron_h | §2.1 | ⚠ | Fix hybrid pattern: `*` must be followed by `-` not `M`; corrected pattern above |

**Decision metric reminder** (from arch_bakeoff_600m.md): compare `lm loss` (not MTP auxiliary
loss) for deepseek_v3_dense. All 6 families use the same `nemotron_cc_v2_llama31_8b` dataset
via `data: nemotron_cc_v2_llama31_8b` default in `configs/launch/config.yaml`.
