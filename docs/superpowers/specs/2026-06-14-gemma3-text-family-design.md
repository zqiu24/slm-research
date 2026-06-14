# Add `gemma3` (text-only) as a 5th architecture-family bake-off candidate — Design

**Date:** 2026-06-14
**Status:** Design approved; implementation plan pending.
**Builds on:** `docs/superpowers/plans/2026-06-12-arch-family-bakeoff-600m.md`
(the bake-off infrastructure: `base/family` × `600m_<family>` scale, the
`src/utils/arch_params.py` budget gate, `src/utils/megatron_args.py` emission,
`scripts/train_bakeoff_600m.sh`, `docs/experiments/arch_bakeoff_600m.md`).

## Goal

Add Google **Gemma 3 (text-only)** as a 5th family in the 600M-budget
architecture bake-off, slotting into the exact same swappable-architecture
structure as `deepseek_v3` / `qwen3_next` / `nemotron_h` (with the existing
dense `qwen3` rung as control). The family must realize the **same declared
600M non-embedding budget within ±2%** (the budget drives `--train-samples`
and the shared GPTDataset cache key, so it must match the other families), run
on the existing `gpt` per-rank entrypoint, and be gated by the same TDD flow:
param-budget test, arg-emission tests, CPU dry-run, pin guard, then a
user-launched GPU smoke before any real run.

The pinned `third_party/Megatron-LM` (core_v0.17.0, SHA `9539a12e`) is **never
edited** — everything lands in slm-research configs, `src/`, tests, the
launcher script, and docs.

## Scope decisions (user-confirmed)

- **Faithful local/global sliding-window interleave** (Gemma 3's headline
  feature): 5 sliding-window-attention layers : 1 global-attention layer.
- **`head_dim: 256`** kept as Gemma's signature (large head dim), at the cost
  of slightly fewer layers within the 600M budget.
- **Approximate + document** the mechanisms the pin cannot express natively,
  exactly as `qwen3_next` documented its deviations — no new patches for them.

## Verified facts about the pin (read before arguing with the design)

All confirmed by reading `third_party/Megatron-LM` at the current pin:

- **Sliding-window interleave is native CLI.** `--window-size`
  (`megatron/training/arguments.py:2020`, `tuple_type`) and
  `--window-attn-skip-freq` (`arguments.py:2023`, same `moe_freq_type` parser
  as `--moe-layer-freq`, so it accepts an int or a python-list-expression
  string). Both are in the auto-gen *exclude* list (`arguments.py:1925-1955`,
  "types affect docstring") but have explicit `add_argument` definitions, so
  they reach `TransformerConfig` via the `hasattr` copy loop
  (`arguments.py:1655`). `is_layer_window_attention(window_size,
  window_attn_skip_freq, layer_number)` (`megatron/core/transformer/utils.py:453`,
  **layer_number is 1-indexed**): for an int `N` it returns
  `layer_number % N != 0` (windowed) — so **`--window-attn-skip-freq 6` makes
  layers 6, 12, 18… global and the rest sliding-window = exactly 5 local : 1
  global**. The flash backend honors `window_size`
  (`megatron/core/transformer/attention.py:741-744`).
- **GeGLU is native.** `--quick-geglu` (`arguments.py:2077`, `store_true`) sets
  `gated_linear_unit=True` + `activation_func=quick_gelu`
  (`arguments.py:1676-1679`). NOTE: `quick_gelu` is the *sigmoid* approximation
  (`y·sigmoid(1.702y)`), not Gemma's `gelu_pytorch_tanh` — an accepted
  activation approximation, documented below.
- **QK-norm is native** (`--qk-layernorm`, config field `qk_layernorm`) and
  already wired in `megatron_args.py` behind the repo's `qk_norm` config key.
- **Zero-centered (1+w) RMSNorm is native.** Config field
  `layernorm_zero_centered_gamma` → CLI `--apply-layernorm-1p`.
- **Sandwich norm already exists in the repo as a patch.**
  `src/patches/sandwich_norm_apply.py` swaps in
  `src/model/sandwich_layer.py::SandwichTransformerLayer` (post-attention +
  post-FFN norm before the residual add) across the dense / MoE / MTP spec
  paths. It is a **no-op unless `base.model.use_sandwich_norm: true`** and is
  **already listed in `configs/experiments/optim/adam.yaml:20`** — the
  experiment every bake-off family uses. Precedent:
  `configs/base/family/deepseek_v3_mqa.yaml` sets `use_sandwich_norm: true`.
  Emission of `--use-sandwich-norm` / `--attn-post-norm-scale` /
  `--ffn-post-norm-scale` already lives in `megatron_args.py`.
- **Per-layer RoPE base is NOT supported** — only one scalar `--rotary-base`.
- **Embedding ×√d scaling is NOT natively supported** — only MuP-specific
  `mup_embedding_mult`; no general `scale_embeddings` flag.
- **Attn/final logit softcapping** is hardcoded off in the FA path — and Gemma
  3 dropped softcapping anyway (replaced by QK-norm), so nothing to do.
- `head_dim` is set independently via `--kv-channels`, so `head_dim: 256` with
  a smaller `hidden_size` is expressible.

## Architecture mapping

| Gemma 3 (text) mechanism | Realization in this design | Native? |
|---|---|---|
| 5:1 local-sliding / global attention | `sliding_window` block → `--window-size "(W,0)"` + `--window-attn-skip-freq 6` | ✅ native CLI |
| GeGLU activation | `activation: "GeGLU"` → `--quick-geglu` | ✅ native (sigmoid-approx) |
| QK-norm (RMSNorm on Q,K) | `qk_norm: true` → `--qk-layernorm` | ✅ already wired |
| Zero-centered (1+w) RMSNorm | `layernorm_zero_centered: true` → `--apply-layernorm-1p` | ✅ native |
| Sandwich norm (post-attn / post-FFN) | `use_sandwich_norm: true` → existing patch | ✅ existing patch |
| Large `head_dim: 256` + GQA | scale-file dims via `--kv-channels` / `--num-query-groups` | ✅ native |
| Tied embeddings | `tie_embeddings: true` | ✅ already wired |
| Per-layer RoPE base (10k local / 1M global) | **approx:** single `rotary_base: 1000000` | ⚠️ deviation |
| Embedding ×√d scaling | **omitted** (no native flag; not worth a patch) | ⚠️ deviation |
| Attn/final logit softcapping | none (Gemma 3 dropped it; matches pin default) | ✅ N/A |

### Documented deviations (accepted, recorded in the protocol doc)

1. **Single RoPE base (1 000 000)** instead of per-layer 10k (local) / 1M
   (global). The pin has no per-layer rotary base; 1M (the global-layer value)
   is the chosen single base.
2. **No √d embedding scaling.** No native flag; out of scope per the approve
   decision (not worth a new patch).
3. **`quick_gelu` (sigmoid approx)** stands in for `gelu_pytorch_tanh`. Both
   are gated GELU variants with identical parameter counts; the curve
   difference is minor and noted.

These are family-identity-preserving approximations, analogous to how
`qwen3_next` uses standard (not zero-centered) RMSNorm on its full-attention
layers and omits its per-head output gate. Sandwich norm and the sliding-window
interleave are part of Gemma's identity (like DeepSeek's MTP/MLA), not fairness
breaks.

## Sizing (600M non-embedding, dense)

Gemma 3 text is dense (no MoE), so `active == total`, entrypoint `gpt`.
Indicative starting dimensions (Gemma-3-flavored), **finalized to within ±2%
of 600M by `tools/size_check.py` during implementation** — exactly like every
other scale file; never tweak `non_embedding_params` to match the dims:

| field | starting value | rationale |
|---|---|---|
| `hidden_size` | 1152 | Gemma 3 1B hidden |
| `head_dim` | 256 | Gemma signature (large head dim) |
| `num_attention_heads` | 8 | |
| `num_query_groups` | 4 | GQA 2:1 |
| `ffn_hidden_size` | ~6912 | Gemma's wide GeGLU MLP (≈6× hidden) |
| `num_layers` | ~20 | size-tuned to hit 600M |
| `seq_length` | 4096 | matches the bake-off |
| sliding window `W` | 1024 | Gemma 3 local window |
| `--window-attn-skip-freq` | 6 | 5 sliding : 1 global |

Back-of-envelope with `arch_params` formulas (hidden 1152, ffn 6912, head_dim
256, 8 heads / 4 groups, qk_norm, sandwich norm) gives ≈31M params/layer →
≈19–20 layers for 600M; the implementer picks the exact `num_layers` /
`ffn_hidden_size` so `size_check` reports within ±2%.

## Code changes (all in slm-research; pin untouched)

1. **`src/utils/megatron_args.py`** (`_model_args`):
   - Add a `GeGLU` branch to the activation handler:
     `elif activation == "GeGLU": _add(args, "--quick-geglu")`. Unknown
     activations still raise (the existing `gelu`-raises test stays green).
   - Emit `--apply-layernorm-1p` when `model.layernorm_zero_centered` is true.
   - Emit the sliding-window block when `model.sliding_window.enabled`:
     `--window-size "W,0"` and `--window-attn-skip-freq <skip_freq>`. The pin's
     `tuple_type` (`arguments.py:282`) parses `"1024,0"` → `(1024, 0)`, so the
     emitted value is the bare `"<window>,0"` token (W = `sliding_window.window`,
     causal → right = 0). `skip_freq` is an int (default 6) — the pin's help
     defines integer `N` as a `(N-1):1` SWA:full ratio, so **6 ⇒ 5 sliding : 1
     global**; a list-expression string is also accepted, mirroring
     `linear_attention_freq`. The dry-run step confirms the round-trip.
   - (Sandwich-norm emission already exists; no change.)

2. **`src/utils/arch_params.py`**:
   - Treat `GeGLU` as gated (3·hidden·ffn, same as SwiGLU) in `_mlp_params`.
     Window attention does **not** change parameter counts (it only changes the
     attention mask), so no mixer change.
   - Add a sandwich-norm term: +2·hidden per layer (post-attn + post-FFN norm
     weights) when `model.use_sandwich_norm` is true, so the budget gate stays
     honest. (Contribution is ~0.01% of 600M but accounted for correctness.)

3. **`configs/base/family/gemma3.yaml`** — mechanisms only (no scale dims):
   normalization RMSNorm, `layernorm_zero_centered: true`, `activation: GeGLU`,
   `positional_encoding: rope`, `rotary_base: 1000000`, `qk_norm: true`,
   `use_sandwich_norm: true`, `attention_backend: flash`, a `sliding_window`
   block (`enabled: true`, `window: 1024`, `skip_freq: 6`), tokenizer
   descriptive block, `reference:` Gemma 3 (Google, arXiv:2503.19786).

4. **`configs/base/scale/600m_gemma3.yaml`** — `non_embedding_params:
   600_000_000` + the size-tuned dims above (head_dim 256, GQA, GeGLU ffn).

5. **Tests**:
   - `tests/unit/test_arch_params.py`: a GeGLU-MLP vector (== the SwiGLU 3×
     value) and a sandwich-norm term vector.
   - `tests/unit/test_megatron_args_families.py`: GeGLU→`--quick-geglu`,
     `layernorm_zero_centered`→`--apply-layernorm-1p`, sliding-window emission
     (`--window-size`, `--window-attn-skip-freq`), and that `gelu` still
     raises.
   - `tests/unit/test_scale_budget.py`: add `("gemma3", "600m_gemma3")` to
     `BAKEOFF_PAIRS` (budget ±2% + active≤total gates apply automatically).
   - `tests/integration/test_megatron_pin_features.py`: add `window_size`,
     `window_attn_skip_freq`, `quick_geglu`, `layernorm_zero_centered_gamma`
     to `REQUIRED_FIELDS`.

6. **`scripts/train_bakeoff_600m.sh`** — add `gemma3) SCALE="600m_gemma3" ;;`
   to the family `case` and to the usage/`unknown family` lists.

7. **Docs**:
   - `docs/experiments/arch_bakeoff_600m.md` — add a `gemma3` row to the family
     table (dense, entrypoint gpt) and a deviations note (single RoPE base, no
     √d embedding scale, quick_gelu, sliding-window seq caveat).
   - `CHANGELOG.md` — prepend an entry for the gemma3 family.

## Test / verification flow (mirrors the existing plan)

CPU runner: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest` from
the repo root. Per task: write failing test → implement → green → `ruff` →
full `tests/unit -q` for regressions (baseline: the 25 known pre-existing
failures noted in the bake-off plan's STATUS) → commit (single short
conventional-commit line, no attribution trailer).

Then, in order:
- **Dry-run** (CPU): `python -m launchers.train_megatron base/family=gemma3
  base/scale=600m_gemma3 experiment=optim/adam training_regime=ablation_40x
  scheduler=wsd cluster=dev --dry-run` — assert the emitted command contains
  `--quick-geglu`, `--apply-layernorm-1p`, `--qk-layernorm`,
  `--use-sandwich-norm`, `--window-size`, `--window-attn-skip-freq 6`,
  `--rotary-base 1000000`, no `--num-experts`, no `--mtp-num-layers`, and the
  same `--train-samples 5859375` as the other 600M families.
- **Pin guard** (cluster env): `tests/integration/test_megatron_pin_features.py`
  with the new required fields — must PASS before the smoke.
- **GPU smoke** (user-launched, ask first): ~30M tokens via
  `scripts/train_bakeoff_600m.sh gemma3 cluster=dev training.tokens_per_param=0.05`.
  Pass criteria: finite, falling loss; `wandb_trainable_params` non-embedding
  total ≈ `size_check` total ±2%; logs show the sliding-window build composing
  with the SandwichTransformerLayer swap. Fallbacks if needed (recorded and
  applied identically across families it affects): `attention_backend=auto`
  (flash window dispatch), `transformer_impl=local` (TE spec).
- **Real bake-off run** is the user's call (24B tokens at seq 4096), appended
  to the existing four-family suite.

## Caveats baked into docs / smoke gate

1. **Window vs seq length.** At the script's cheap `SEQ_LENGTH=256` iteration
   default, a 1024 sliding window exceeds the sequence, so every "windowed"
   layer behaves as full attention and the local/global distinction vanishes —
   the real Gemma comparison needs `seq ≥ window` (use seq 4096). Same note the
   Mamba/GDN families already carry.
2. **Composition.** Sandwich-layer swap + window attention + flash backend
   composition is verified at the GPU smoke, with the `attention_backend=auto`
   fallback documented.

## Out of scope

- Gemma 3 **vision / multimodal** (this is text-only).
- Per-layer RoPE base, √d embedding scaling, `gelu_pytorch_tanh` exact
  activation, attn softcapping (the last is absent in Gemma 3 anyway).
- A Gemma-faithful tokenizer (the bake-off tokenizer is manifest-frozen; the
  family's tokenizer block is descriptive only).
- Promotion to 1.2B/2.4B (a separate plan, only after the bake-off verdict).
