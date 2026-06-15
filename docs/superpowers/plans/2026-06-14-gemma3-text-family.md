# Gemma 3 (text-only) Bake-off Family — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Google Gemma 3 (text-only) as a 5th architecture-family candidate in the 600M-budget bake-off, consistent with `deepseek_v3` / `qwen3_next` / `nemotron_h`.

**Architecture:** A dense `gpt`-entrypoint family realizing the same declared 600M non-embedding budget (±2%). Gemma's distinctive mechanisms are wired through native Megatron CLI flags (local/global sliding-window interleave via `--window-size` + `--window-attn-skip-freq`; GeGLU via `--quick-geglu`; zero-centered RMSNorm via `--apply-layernorm-1p`; QK-norm already wired) plus the repo's existing `sandwich_norm_apply` patch (already in the `optim/adam` experiment, no-op unless `use_sandwich_norm: true`). Three mechanisms the pin cannot express are approximated and documented (single RoPE base 1M, no √d embedding scale, sigmoid-approx `quick_gelu`). The pinned `third_party/Megatron-LM` is never edited.

**Tech Stack:** Hydra/OmegaConf config composition, Megatron-LM core_v0.17.0 (pinned submodule), pytest (CPU unit tests via `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`).

**Design spec:** `docs/superpowers/specs/2026-06-14-gemma3-text-family-design.md`
**Builds on:** `docs/superpowers/plans/2026-06-12-arch-family-bakeoff-600m.md`

---

## Verified pin facts (read before arguing with the plan)

All confirmed against `third_party/Megatron-LM` at the current pin (core_v0.17.0):

- `--window-size` (`megatron/training/arguments.py:2020`, `tuple_type` at `arguments.py:282` parses `"1024,0"` → `(1024, 0)`) and `--window-attn-skip-freq` (`arguments.py:2023`, `moe_freq_type` → int or list-expr string). `is_layer_window_attention` (`megatron/core/transformer/utils.py:453`, layer_number 1-indexed): int `N` ⇒ `layer_number % N != 0` is windowed, so **`--window-attn-skip-freq 6` = 5 sliding : 1 global**. Flash backend honors `window_size` (`attention.py:741-744`).
- `--quick-geglu` (`arguments.py:2077`, `store_true`, dest `quick_geglu`) sets `gated_linear_unit=True` + `activation_func=quick_gelu` (`arguments.py:1676-1679`). Sigmoid approx, not `gelu_pytorch_tanh` — accepted, documented.
- `--apply-layernorm-1p` → dest `layernorm_zero_centered_gamma`. Declared via the `TransformerConfig` field's `argparse_meta` metadata (`transformer_config.py:170-172`), surfaced by `ArgumentGroupFactory` (`megatron/training/argument_utils.py:185-186` — field metadata has highest precedence). It is **not** a literal `add_argument` in `arguments.py`, so don't grep for it there.
- `--qk-layernorm` (already emitted by `megatron_args.py` behind the `qk_norm` config key).
- `sandwich_norm_apply` patch (`src/patches/sandwich_norm_apply.py`) swaps in `SandwichTransformerLayer` (post-attn + post-FFN norm); it is **already listed in `configs/experiments/optim/adam.yaml:20`** and is a no-op unless `base.model.use_sandwich_norm: true`. Emission of `--use-sandwich-norm` / `--attn-post-norm-scale` / `--ffn-post-norm-scale` already exists in `megatron_args.py`. Precedent: `configs/base/family/deepseek_v3_mqa.yaml`.

## Execution notes

- CPU test runner: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest` from repo root `/lustre/fast/fast/zqiu/slm-research`.
- Full-suite baseline: `pytest tests/unit -q` has **25 pre-existing failures** (stale POET/nGPT/train_scripts tests; see the bake-off plan's STATUS). The new tests must pass with zero new failures.
- Commits: single short conventional-commit line, no attribution trailer.
- GPU smoke + real runs (Task 6) are the **user's to launch** — ask first; never launch unprompted.

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `src/utils/arch_params.py` | modify | Recognize `GeGLU` (gated, 3×); add sandwich-norm term (+2·h/layer) |
| `tests/unit/test_arch_params.py` | modify | GeGLU-MLP, sandwich-term, unknown-activation vectors |
| `src/utils/megatron_args.py` | modify | Emit `--quick-geglu`, `--apply-layernorm-1p`, `--window-size`/`--window-attn-skip-freq` |
| `tests/unit/test_megatron_args_families.py` | modify | Emission tests for the three above |
| `configs/base/family/gemma3.yaml` | create | Gemma 3 text mechanisms |
| `configs/base/scale/600m_gemma3.yaml` | create | 600M dense realization (head_dim 256, GeGLU) |
| `tests/unit/test_scale_budget.py` | modify | Add `("gemma3","600m_gemma3")` to `BAKEOFF_PAIRS` |
| `tests/integration/test_megatron_pin_features.py` | modify | Add gemma3 fields to `REQUIRED_FIELDS` |
| `scripts/train_bakeoff_600m.sh` | modify | Add `gemma3) SCALE="600m_gemma3"` case |
| `docs/experiments/arch_bakeoff_600m.md` | modify | gemma3 rows + deviation note |
| `CHANGELOG.md` | modify | gemma3 family entry |

---

### Task 1: `arch_params` — GeGLU + sandwich-norm accounting

**Files:**
- Modify: `src/utils/arch_params.py` (`_mlp_params` lines 146-149; `_gpt_total` lines 199-213)
- Test: `tests/unit/test_arch_params.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/unit/test_arch_params.py` (the file already imports `pytest` and defines `_dense_model()` = 2 layers / hidden 8 / ffn 16 / heads 2 / groups 1 / head_dim 4 / SwiGLU; `test_dispatch_dense_gpt` proves it equals 1192):

```python
def test_geglu_mlp_matches_swiglu():
    # GeGLU is gated (gate+up+down) like SwiGLU -> identical 3*h*ffn accounting.
    assert non_embedding_params(_dense_model() | {"activation": "GeGLU"}) == 1192


def test_sandwich_norm_adds_two_norms_per_layer():
    # Sandwich norm adds post-attn + post-mlp norm weights = +2*hidden per layer.
    # _dense_model() has 2 layers, hidden 8 -> +2*8*2 = +32.
    assert non_embedding_params(_dense_model() | {"use_sandwich_norm": True}) == 1192 + 32


def test_unknown_activation_rejected():
    with pytest.raises(ValueError):
        non_embedding_params(_dense_model() | {"activation": "gelu"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_arch_params.py -v -k "geglu or sandwich or unknown_activation"`
Expected: `test_geglu_mlp_matches_swiglu` PASSES (GeGLU already hits the 3× default branch); `test_sandwich_norm_adds_two_norms_per_layer` FAILS (no sandwich term yet); `test_unknown_activation_rejected` FAILS (no raise yet).

- [ ] **Step 3: Recognize GeGLU explicitly + reject unknown activations**

In `src/utils/arch_params.py`, replace the tail of `_mlp_params` (lines 146-149):

```python
    ffn = int(model["ffn_hidden_size"])
    if str(model.get("activation", "SwiGLU")) == "squared_relu":
        return 2 * hidden * ffn
    return 3 * hidden * ffn  # SwiGLU
```

with:

```python
    ffn = int(model["ffn_hidden_size"])
    activation = str(model.get("activation", "SwiGLU"))
    if activation == "squared_relu":
        return 2 * hidden * ffn
    if activation in ("SwiGLU", "GeGLU"):
        return 3 * hidden * ffn  # gated GLU: gate + up + down projections
    raise ValueError(f"Unsupported activation {activation!r} in arch_params")
```

- [ ] **Step 4: Add the sandwich-norm term**

In `src/utils/arch_params.py`, replace the body of `_gpt_total` from `total = 0` through `return total` (lines 199-213):

```python
    total = 0
    for i in range(num_layers):
        total += _mixer_params(model, layer_is_linear=bool(linear_pattern[i]))
        total += _mlp_params(model, active=active, layer_is_moe=bool(moe_pattern[i]))
        total += 2 * hidden  # input_layernorm + pre_mlp_layernorm
    total += hidden  # final norm

    # MTP blocks: one decoder layer (same shape as the last layer) + eh_proj
    # (2h -> h) + enorm + hnorm + the MTP block's final norm.
    for _ in range(int(model.get("mtp_num_layers") or 0)):
        total += _mixer_params(model, layer_is_linear=bool(linear_pattern[-1]))
        total += _mlp_params(model, active=active, layer_is_moe=bool(moe_pattern[-1]))
        total += 2 * hidden
        total += 2 * hidden * hidden + 3 * hidden
    return total
```

with:

```python
    # Sandwich norm (Gemma-style) adds a post-attn + post-mlp norm per layer.
    # The sandwich_norm_apply patch swaps the layer class across dense/MoE/MTP
    # spec paths, so the term applies to MTP blocks too.
    sandwich = 2 * hidden if bool(model.get("use_sandwich_norm", False)) else 0

    total = 0
    for i in range(num_layers):
        total += _mixer_params(model, layer_is_linear=bool(linear_pattern[i]))
        total += _mlp_params(model, active=active, layer_is_moe=bool(moe_pattern[i]))
        total += 2 * hidden  # input_layernorm + pre_mlp_layernorm
        total += sandwich  # post-attn + post-mlp norm weights
    total += hidden  # final norm

    # MTP blocks: one decoder layer (same shape as the last layer) + eh_proj
    # (2h -> h) + enorm + hnorm + the MTP block's final norm.
    for _ in range(int(model.get("mtp_num_layers") or 0)):
        total += _mixer_params(model, layer_is_linear=bool(linear_pattern[-1]))
        total += _mlp_params(model, active=active, layer_is_moe=bool(moe_pattern[-1]))
        total += 2 * hidden
        total += sandwich
        total += 2 * hidden * hidden + 3 * hidden
    return total
```

- [ ] **Step 5: Run tests to verify they pass + full regression**

Run: `cd /lustre/fast/fast/zqiu/slm-research && ruff check src/utils/arch_params.py tests/unit/test_arch_params.py && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_arch_params.py -v`
Expected: ruff clean; all arch_params tests PASS (the original 12+ plus the 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/utils/arch_params.py tests/unit/test_arch_params.py
git commit -m "$(cat <<'EOF'
feat(arch): account GeGLU (gated) and sandwich-norm in param budget
EOF
)"
```

---

### Task 2: `megatron_args` — GeGLU, zero-centered RMSNorm, sliding-window emission

**Files:**
- Modify: `src/utils/megatron_args.py` (`_model_args`)
- Test: `tests/unit/test_megatron_args_families.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/unit/test_megatron_args_families.py` (the file already defines `_cfg(**overrides)`, `_value(args, flag)`, and `test_unknown_activation_raises` using `"gelu"`):

```python
def test_geglu_emits_quick_geglu():
    args = _model_args(_cfg(activation="GeGLU"))
    assert "--quick-geglu" in args
    assert "--swiglu" not in args
    assert "--squared-relu" not in args


def test_layernorm_zero_centered_emits_1p():
    args = _model_args(_cfg(layernorm_zero_centered=True))
    assert "--apply-layernorm-1p" in args


def test_layernorm_zero_centered_default_omits_1p():
    assert "--apply-layernorm-1p" not in _model_args(_cfg())


def test_sliding_window_emission():
    args = _model_args(
        _cfg(sliding_window={"enabled": True, "window": 1024, "skip_freq": 6})
    )
    assert _value(args, "--window-size") == "1024,0"
    assert _value(args, "--window-attn-skip-freq") == "6"


def test_sliding_window_disabled_omits_flags():
    args = _model_args(_cfg())
    assert "--window-size" not in args
    assert "--window-attn-skip-freq" not in args
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args_families.py -v -k "geglu or zero_centered or sliding_window"`
Expected: `test_geglu_emits_quick_geglu` FAILS with `ValueError: Unsupported model.activation 'GeGLU'`; the zero-centered and sliding-window tests FAIL (flags not emitted); `test_layernorm_zero_centered_default_omits_1p` and `test_sliding_window_disabled_omits_flags` PASS.

- [ ] **Step 3: Add the GeGLU branch**

In `src/utils/megatron_args.py`, the activation block currently reads:

```python
    activation = str(model.get("activation", "SwiGLU"))
    if activation == "SwiGLU":
        _add(args, "--swiglu")
    elif activation == "squared_relu":
        _add(args, "--squared-relu")
    else:
        raise ValueError(f"Unsupported model.activation {activation!r}")
```

Insert a `GeGLU` branch before the `else`:

```python
    activation = str(model.get("activation", "SwiGLU"))
    if activation == "SwiGLU":
        _add(args, "--swiglu")
    elif activation == "squared_relu":
        _add(args, "--squared-relu")
    elif activation == "GeGLU":
        # Gemma-style gated GELU. --quick-geglu sets gated_linear_unit + the
        # sigmoid-approx quick_gelu (arguments.py:1676, pin core_v0.17.0); the
        # tanh-approx gelu_pytorch_tanh has no native flag (documented approx).
        _add(args, "--quick-geglu")
    else:
        raise ValueError(f"Unsupported model.activation {activation!r}")
```

- [ ] **Step 4: Emit zero-centered RMSNorm**

In `src/utils/megatron_args.py`, immediately after the two normalization lines:

```python
    _add(args, "--normalization", model.normalization)
    _add(args, "--norm-epsilon", model.norm_epsilon)
```

insert:

```python
    # Zero-centered (1+w) RMSNorm (Gemma). --apply-layernorm-1p maps to the
    # config field layernorm_zero_centered_gamma (transformer_config.py:170
    # argparse_meta; pin core_v0.17.0).
    if bool(model.get("layernorm_zero_centered", False)):
        _add(args, "--apply-layernorm-1p")
```

- [ ] **Step 5: Emit the sliding-window interleave**

In `src/utils/megatron_args.py`, immediately after the qk-norm block:

```python
    if bool(model.get("qk_norm", False)):
        _add(args, "--qk-layernorm")
```

insert:

```python
    # Local/global sliding-window interleave (Gemma 3-style). Native Megatron
    # CLI args (arguments.py:2020/2023, pin core_v0.17.0): --window-size parses
    # "W,0" -> (W,0) via tuple_type; --window-attn-skip-freq takes int N (=
    # (N-1):1 SWA:full, so 6 -> 5 sliding : 1 global) or a list-expr string.
    sliding = model.get("sliding_window", {}) or {}
    if bool(sliding.get("enabled", False)):
        _add(args, "--window-size", f"{int(sliding.get('window', 1024))},0")
        _add(args, "--window-attn-skip-freq", sliding.get("skip_freq", 6))
```

- [ ] **Step 6: Run tests + full regression**

Run: `cd /lustre/fast/fast/zqiu/slm-research && ruff check src/utils/megatron_args.py tests/unit/test_megatron_args_families.py && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args_families.py -v`
Expected: ruff clean; all emission tests PASS (the original set plus the 5 new). `test_unknown_activation_raises` (uses `"gelu"`) still PASSES — GeGLU is distinct from gelu.

- [ ] **Step 7: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args_families.py
git commit -m "$(cat <<'EOF'
feat(args): emit GeGLU, zero-centered RMSNorm, sliding-window flags
EOF
)"
```

---

### Task 3: `gemma3` family + `600m_gemma3` scale + budget gate

**Files:**
- Create: `configs/base/family/gemma3.yaml`
- Create: `configs/base/scale/600m_gemma3.yaml`
- Modify: `tests/unit/test_scale_budget.py` (`BAKEOFF_PAIRS`)

- [ ] **Step 1: Add the pair to `BAKEOFF_PAIRS` and watch it fail**

In `tests/unit/test_scale_budget.py`, the list currently ends with `("nemotron_h", "600m_nemotron_h"),`. Add a gemma3 entry:

```python
BAKEOFF_PAIRS = [
    ("deepseek_v3", "600m_deepseek_v3"),
    ("deepseek_v3_dense", "600m_deepseek_v3_dense"),
    ("qwen3_next", "600m_qwen3_next"),
    ("nemotron_h", "600m_nemotron_h"),
    ("gemma3", "600m_gemma3"),
]
```

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_scale_budget.py -v -k gemma3`
Expected: FAIL with `FileNotFoundError` for `configs/base/family/gemma3.yaml`.

- [ ] **Step 2: Write the family file**

`configs/base/family/gemma3.yaml`:

```yaml
# @package _global_
# Family-level defaults — architectural choices that do not depend on scale.
# Gemma 3 (text-only). See docs/superpowers/specs/2026-06-14-gemma3-text-family-design.md.
base:
  family: gemma3
  family_version: "3_text_2503"
  reference: "Gemma 3 (Google, arXiv:2503.19786) — local/global SWA + GeGLU + sandwich norm"
  model:
    normalization: "RMSNorm"
    norm_epsilon: 1.0e-6
    layernorm_zero_centered: true     # (1+w) RMSNorm -> --apply-layernorm-1p
    activation: "GeGLU"               # -> --quick-geglu (sigmoid-approx GELU)
    positional_encoding: "rope"
    rotary_base: 1000000              # single base (approx; Gemma uses 10k local / 1M global)
    rotary_scaling: null
    qk_norm: true                     # -> --qk-layernorm
    use_sandwich_norm: true           # post-attn + post-ffn norm (sandwich_norm_apply patch)
    attention_dropout: 0.0
    hidden_dropout: 0.0
    init_method_std: 0.02
    depth_scaled_init: false
    attention_backend: "flash"
    sliding_window:
      enabled: true
      window: 1024                    # local sliding window (left context); causal
      skip_freq: 6                    # int N => (N-1):1 => 5 sliding : 1 global
  tokenizer:
    # Descriptive only — the actual tokenizer is fixed by the dataset manifest.
    nominal_name: "gemma-3"
    nominal_vocab_size: 262144
```

- [ ] **Step 3: Write the scale file**

`configs/base/scale/600m_gemma3.yaml` — sized by `arch_params`: total 599,512,192 (−0.08% vs budget), dense (active == total). Dimensions are Gemma-3-flavored (head_dim 256, GQA, wide GeGLU MLP):

```yaml
# @package _global_
# Gemma 3 (text-only) mechanisms realized at the 600M non-embedding budget
# (arch bake-off; docs/experiments/arch_bakeoff_600m.md). Sized by
# tools/size_check.py: total 599,512,192 (-0.08%); dense (active == total).
base:
  scale: "600m_gemma3"
  non_embedding_params: 600_000_000
  model:
    num_layers: 20
    hidden_size: 1152
    ffn_hidden_size: 6624           # wide GeGLU MLP (gated)
    num_attention_heads: 8
    num_query_groups: 4             # GQA 2:1
    head_dim: 256                   # Gemma signature large head dim (--kv-channels)
    seq_length: 4096
    tie_embeddings: true
```

- [ ] **Step 4: Run the budget test + sizing tool**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_scale_budget.py -v -k gemma3
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python tools/size_check.py base/family=gemma3 base/scale=600m_gemma3
```
Expected: 2 PASSED; tool prints `total 599,512,192 (-0.08% vs budget)`, `active 599,512,192 (100.0% of total)`. If outside ±2% (e.g. formulas shifted), adjust `ffn_hidden_size` by ±32 (≈ ±0.37% of budget) and re-run — do **not** change `non_embedding_params`.

- [ ] **Step 5: Dry-run the full launcher path (CPU)**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m launchers.train_megatron base/family=gemma3 base/scale=600m_gemma3 experiment=optim/adam training_regime=ablation_40x scheduler=wsd cluster=dev --dry-run`
Expected: a JSON payload whose `command` contains `--quick-geglu`, `--apply-layernorm-1p`, `--qk-layernorm`, `--use-sandwich-norm`, `--window-size 1024,0`, `--window-attn-skip-freq 6`, `--rotary-base 1000000`, `--group-query-attention`, `--num-query-groups 4`, `--kv-channels 256`, `-m launchers.pretrain_gpt_slm`, and the same `--train-samples 5859375` as the other 600M families. It must NOT contain `--num-experts`, `--mtp-num-layers`, `--experimental-attention-variant`, or `--hybrid-layer-pattern`.

- [ ] **Step 6: Commit**

```bash
git add configs/base/family/gemma3.yaml configs/base/scale/600m_gemma3.yaml tests/unit/test_scale_budget.py
git commit -m "$(cat <<'EOF'
feat(config): gemma3 text-only family + 600m bake-off scale + budget gate
EOF
)"
```

---

### Task 4: Pin guard + bake-off launcher routing

**Files:**
- Modify: `tests/integration/test_megatron_pin_features.py` (`REQUIRED_FIELDS`)
- Modify: `scripts/train_bakeoff_600m.sh`

- [ ] **Step 1: Extend the pin-guard required fields**

In `tests/integration/test_megatron_pin_features.py`, the `REQUIRED_FIELDS` list currently ends with the `"squared_relu"` entry under the nemotron comment. Append a gemma3 block before the closing `]`:

```python
    # Activation (nemotron_h family)
    "squared_relu",
    # Gemma 3 family (sliding-window interleave, GeGLU, zero-centered RMSNorm)
    "window_size",
    "window_attn_skip_freq",
    "quick_geglu",
    "layernorm_zero_centered_gamma",
]
```

(`window_size` / `window_attn_skip_freq` / `quick_geglu` are argparse dests from explicit `add_argument`s; `layernorm_zero_centered_gamma` is the dataclass-field dest behind `--apply-layernorm-1p`. `parse_args` sets all four as attributes.)

- [ ] **Step 2: Add the gemma3 case to the launcher script**

In `scripts/train_bakeoff_600m.sh`, the family `case` currently reads (note `deepseek_v3_dense` was added since the bake-off plan; preserve the column alignment):

```bash
case "$FAMILY" in
  qwen3)             SCALE="600m" ;;            # dense control (existing dev rung)
  deepseek_v3)       SCALE="600m_deepseek_v3" ;;
  deepseek_v3_dense) SCALE="600m_deepseek_v3_dense" ;;  # MLA + MTP, MoE off
  qwen3_next)        SCALE="600m_qwen3_next" ;;
  nemotron_h)        SCALE="600m_nemotron_h" ;;
  *) echo "unknown family: $FAMILY (qwen3|deepseek_v3|deepseek_v3_dense|qwen3_next|nemotron_h)" >&2; exit 1 ;;
esac
```

Add a `gemma3` line and extend the error string:

```bash
case "$FAMILY" in
  qwen3)             SCALE="600m" ;;            # dense control (existing dev rung)
  deepseek_v3)       SCALE="600m_deepseek_v3" ;;
  deepseek_v3_dense) SCALE="600m_deepseek_v3_dense" ;;  # MLA + MTP, MoE off
  qwen3_next)        SCALE="600m_qwen3_next" ;;
  nemotron_h)        SCALE="600m_nemotron_h" ;;
  gemma3)            SCALE="600m_gemma3" ;;
  *) echo "unknown family: $FAMILY (qwen3|deepseek_v3|deepseek_v3_dense|qwen3_next|nemotron_h|gemma3)" >&2; exit 1 ;;
esac
```

Also update the usage comment near the top — change the line

```bash
#   family ∈ {qwen3, deepseek_v3, deepseek_v3_dense, qwen3_next, nemotron_h}
```

to

```bash
#   family ∈ {qwen3, deepseek_v3, deepseek_v3_dense, qwen3_next, nemotron_h, gemma3}
```

- [ ] **Step 3: Dry-run gemma3 through the script (CPU)**

Run: `cd /lustre/fast/fast/zqiu/slm-research && bash scripts/train_bakeoff_600m.sh gemma3 cluster=dev base.model.seq_length=4096 --dry-run | tail -3`
Expected: a JSON payload with run_name `adam-gemma3-600m_gemma3-s42-<ts>`, `-m launchers.pretrain_gpt_slm`, and `--window-attn-skip-freq 6`. (If `python` in the sourced env lacks repo deps, prefix with `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m launchers.train_megatron ...` using the same overrides the script passes.)

- [ ] **Step 4: Run the pin guard on a compute node (gated + ask)**

This needs the cluster env (TransformerEngine dlopens CUDA libs; it cannot load on the login node). Hand the user this command (do not run GPU/cluster jobs unprompted):

```bash
cd /lustre/fast/fast/zqiu/slm-research && source load_cuda13_2_nccl_env.sh && \
  PYTHONPATH=third_party/Megatron-LM /lustre/fast/fast/zqiu/slm_env/.venv/bin/python \
  -m pytest tests/integration/test_megatron_pin_features.py -v
```
Expected: 1 PASSED (the 4 new gemma3 fields present alongside the existing 12). If it SKIPS (transformer_engine cannot load on the current node), it must be run on a compute node before the Task 6 smoke.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_megatron_pin_features.py scripts/train_bakeoff_600m.sh
git commit -m "$(cat <<'EOF'
feat(bakeoff): gemma3 pin-guard fields + launcher case
EOF
)"
```

---

### Task 5: Docs + changelog

**Files:**
- Modify: `docs/experiments/arch_bakeoff_600m.md` (family table, deviations note, results table)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add gemma3 to the family table**

In `docs/experiments/arch_bakeoff_600m.md`, the first family table ends with the `nemotron_h` row. Add a gemma3 row beneath it:

```markdown
| nemotron_h | 600m_nemotron_h | 604.8M | =total | mamba |
| gemma3 | 600m_gemma3 | 599.5M | =total | gpt |
```

- [ ] **Step 2: Add the gemma3 deviations note**

In the same file, append to the **Known asymmetries (accepted)** paragraph:

```markdown
gemma3 is dense (no MoE) and keeps its local/global sliding-window interleave
(5 sliding : 1 global), GeGLU, QK-norm, zero-centered RMSNorm, and sandwich
norm as family identity. Accepted approximations (no native Megatron support):
a single RoPE base (1,000,000) instead of per-layer 10k local / 1M global; no
√d embedding scaling; sigmoid-approx `quick_gelu` instead of `gelu_pytorch_tanh`.
NOTE: at the script's cheap `SEQ_LENGTH=256` iteration default the 1024 window
exceeds the sequence, so local/global collapses to full attention — the real
gemma3 comparison needs `seq >= window` (use seq 4096).
```

- [ ] **Step 3: Add gemma3 to the results table**

In the same file, the **Results** table ends with the `nemotron_h` row. Add:

```markdown
| nemotron_h | | | | |
| gemma3 | | | | |
```

- [ ] **Step 4: Prepend a CHANGELOG entry**

In `CHANGELOG.md`, insert a new section immediately after the `## Unreleased` heading and before the existing `### Added — weight-matrix norm monitoring (2026-06-13)` section:

```markdown
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
```

- [ ] **Step 5: Commit**

```bash
git add docs/experiments/arch_bakeoff_600m.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(bakeoff): gemma3 family rows, deviations note, changelog
EOF
)"
```

---

### Task 6: GPU smoke + handoff (USER-RUN)

No file changes. CPU work is done; everything below runs on GPU and is the **user's to launch** — ask before any GPU smoke; never launch unprompted.

- [ ] **Step 1: Confirm the pin guard passed on a compute node** (Task 4 Step 4; rerun there if it skipped on the login node). Do not proceed without a PASS.

- [ ] **Step 2: Run the gemma3 smoke (~30M tokens)** — logs land in `/lustre/home/zqiu/log/<NAME>.log` via codexlog

```bash
cd /lustre/fast/fast/zqiu/slm-research
codexlog bakeoff-smoke-gemma3 bash scripts/train_bakeoff_600m.sh gemma3 cluster=dev \
  base.model.seq_length=4096 training.tokens_per_param=0.05 training.micro_batch_size=2
```

Smoke pass criteria (check the log):
- run reaches the first logged iterations; loss is finite and falling;
- `wandb_trainable_params` non-embedding total ≈ `tools/size_check.py` total
  **599,512,192** (±2%); embeddings add vocab×hidden on top (tied);
- the build log shows sliding-window attention composing with the
  `SandwichTransformerLayer` swap (`[sandwich] swapped layer class ...`), and
  no `--num-experts` / `--mtp-num-layers` / `--hybrid-layer-pattern`;
- known fallbacks if the smoke fails (record whichever was needed and apply it
  identically to every family it affects in the real run):
  `base.model.attention_backend=auto` (flash window dispatch issue),
  `base.model.transformer_impl=local` (TE spec issue).

- [ ] **Step 3: Launch the real gemma3 bake-off run (24B tokens; only after the smoke passes; user's call on cluster + timing)**

```bash
cd /lustre/fast/fast/zqiu/slm-research
codexlog bakeoff-600m-gemma3 bash scripts/train_bakeoff_600m.sh gemma3 cluster=h100_de
```

Append any fallback override identically to the other families' real-run commands it affects, and record it in `docs/experiments/arch_bakeoff_600m.md` (fairness: same override everywhere).

- [ ] **Step 4: After the run — fill the gemma3 row in the Results table in `docs/experiments/arch_bakeoff_600m.md` and apply the existing decision rule alongside the other families.**

---

## Self-review checklist (run after writing code, before each commit)

- Budget numbers in YAML comments must match what `tools/size_check.py` actually prints — update comments if formulas shifted.
- No edits under `third_party/` — `git status third_party/` must stay clean.
- `pytest tests/unit -q` green modulo the 25 known pre-existing failures before every commit (zero new failures).

## Risks & mitigations

- **Sliding-window + sandwich-layer-swap + flash composition misbehaves at runtime** — the GPU smoke (Task 6) is the gate, with `attention_backend=auto` / `transformer_impl=local` fallbacks. If sliding-window is unusable, fall back to full attention (`sliding_window.enabled: false`) and document it (the family then becomes a GeGLU + sandwich-norm dense variant) — but this loses Gemma's headline feature, so prefer fixing via the fallback backend first.
- **`tuple_type` rejects the `--window-size` value format** — caught at the Task 3 dry-run; `tuple_type` (`arguments.py:282`) strips `()` and splits on `,`, so the bare `"1024,0"` token round-trips. If a future pin changes the parser, adjust the emitted string, not the pin.
- **Pin moved and a flag changed** — the Task 4 pin guard catches a missing arg before any GPU spend; re-run after any submodule bump.
- **Sandwich patch not active** — `use_sandwich_norm: true` is a no-op unless `sandwich_norm_apply` is in `experiment.patches`; the bake-off uses `experiment=optim/adam`, which already lists it (`configs/experiments/optim/adam.yaml:20`). Verify `[sandwich] swapped layer class` appears in the smoke log.
- **Zero-centered RMSNorm assert (ruled out)** — `transformer_config.py:2211` asserts `not layernorm_zero_centered_gamma`, but ONLY inside the `if self.transformer_impl == "inference_optimized":` block, which gemma3 never enters. The `--apply-layernorm-1p` + RMSNorm training path is unaffected.
