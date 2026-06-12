# Architecture-Family Bake-off at the 600M Budget — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model architecture a swappable, budget-matched axis on the frozen dataset + training pipeline, implement three candidate families (DeepSeek-V3, Qwen3-Next-style, Nemotron-H-style) at a 600M non-embedding-parameter budget, and run a controlled bake-off to pick the base architecture for from-scratch pretraining.

**Architecture:** The repo already splits `base/family` (architecture mechanisms) from `base/scale` (dimensions). This plan adds: (1) a CPU-only parameter-accounting module so every family realizes the *same* declared `non_embedding_params` budget within ±2% (the budget drives token count and the dataset cache key, so it must be identical across runs); (2) Megatron-arg emission for the two mechanisms not yet wired (GatedDeltaNet linear attention via the GPT path; Mamba2 hybrids via a new per-rank entrypoint mirroring the GPT one); (3) per-family 600M scale realizations plus a bake-off protocol. The pinned `third_party/Megatron-LM` (core_v0.17.0, SHA `9539a12e`) is **never edited** — everything lands in slm-research configs, `src/`, and launchers. No new monkey-patches are required: GDN and Mamba dims are auto-generated CLI flags from `TransformerConfig` dataclass fields in this pin.

**Tech Stack:** Hydra/OmegaConf config composition, Megatron-LM core_v0.17.0 (pinned submodule), pytest (CPU unit tests via `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python`).

---

## Decisions already made (user-confirmed)

- **Empirical bake-off**, not an analytical pick: all three families implemented and compared head-to-head, plus the existing `qwen3` dense 600M as control.
- **600M-class dev rung** (`ablation_40x` regime → 24B tokens), winner promoted to 1.2B/2.4B afterwards.
- **Published cores only**: DeepSeek-V3 (MLA + sigmoid-routed MoE + MTP, arXiv:2412.19437), Qwen3-Next-style (GatedDeltaNet/full-attention hybrid + MoE, Alibaba Sep 2025), Nemotron-H-style (Mamba2/attention/MLP hybrid, squared-ReLU, arXiv:2504.03624). DeepSeek-V4 preview features (DSA) and unpublished "Qwen 3.7" specifics are explicitly out of scope (the pin's `experimental_attention_variant='dsa'` makes DSA a cheap follow-up ablation later).
- **Budget matching**: families match on **total non-embedding params = 600M** (the repo's ladder unit). Active params differ (MoE families ≈ 240–250M active) and are recorded as a bake-off metric, not equalized.
- **Fairness controls**: identical dataset/tokenizer (manifest-frozen), identical `training_regime=ablation_40x`, `scheduler=wsd`, `experiment=optim/adam`, seed 42, identical MoE *router recipe* across the two MoE families (DeepSeek sigmoid + seq_aux_loss + expert bias) so the comparison isolates the mixer/backbone, identical GBS=1024/seq=4096. Known asymmetry, accepted and documented: hidden size differs per family (1024 vs 1280), so tied-embedding param counts differ; embeddings are outside the budget unit by repo policy (SPEC.md §1.3).

## Verified facts about the pin (read before arguing with the plan)

- `megatron/core/ssm/` contains `mamba_mixer.py`, `mamba_block.py`, `gated_delta_net.py`, `mamba_hybrid_layer_allocation.py`; pattern symbols are `M`(mamba) `G`(gdn) `*`(attention) `-`(MLP) `E`(MoE) (`Symbols` class, `mamba_hybrid_layer_allocation.py:14-24`).
- `gpt_builders.py:49-54`: if `args.experimental_attention_variant` is set, the GPT path builds via `get_transformer_block_with_experimental_attention_variant_spec` (supports GDN + per-layer MoE/dense mixing). `'gated_delta_net'` requires `linear_attention_freq` (`transformer_config.py:1067`).
- Megatron auto-generates CLI flags from `TransformerConfig` dataclass fields (`arguments.py:1655`), so `--experimental-attention-variant`, `--linear-num-key-heads`, `--linear-key-head-dim`, `--linear-num-value-heads`, `--linear-value-head-dim`, `--linear-conv-kernel-dim`, `--mamba-state-dim`, `--mamba-head-dim`, `--mamba-num-groups`, `--hybrid-layer-pattern` all exist without slm-side argparse work. Task 8's pin-guard test asserts this.
- Mamba hybrids build through `MambaModel` (`pretrain_mamba.py` + root-level `mamba_builders.py`), NOT `GPTModel` — hence the new launcher in Task 6. **This path has no MTP support** (stated in `pretrain_gpt.py:93-108` docstring).
- `megatron/training/arguments.py` parses only with the cluster env loaded (TransformerEngine `.so` dlopens CUDA 13 libs) — any test importing Megatron must `source load_cuda13_2_nccl_env.sh` first and lives in `tests/integration/`, not `tests/unit/`.

## Execution notes

- CPU test runner: `/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest` from the repo root `/lustre/fast/fast/zqiu/slm-research`.
- If executing in a git worktree: `configs/data/` is gitignored — copy it from the main checkout first (`cp -r /lustre/fast/fast/zqiu/slm-research/configs/data <worktree>/configs/`).
- Commits: single short conventional-commit line (`feat(...)`, `test(...)`, `docs(...)`), no attribution trailers.
- GPU smoke runs and the real bake-off runs are the **user's to launch** — Task 11 hands over exact commands and stops.

## File structure

| Path | Responsibility |
|---|---|
| `src/utils/arch_params.py` (new) | Pure-python non-embedding param accounting for every family mechanism |
| `tests/unit/test_arch_params.py` (new) | Formula + dispatcher tests with hand-computed vectors |
| `src/utils/megatron_args.py` (modify) | Conditional activation/rotary emission; GDN + hybrid/mamba arg blocks |
| `tests/unit/test_megatron_args_families.py` (new) | Per-family arg-emission tests |
| `launchers/pretrain_mamba_slm.py` (new) | Per-rank entrypoint for Mamba-hybrid families (mirror of GPT twin) |
| `launchers/train_megatron.py` (modify) | Entrypoint module selection from `base.model.entrypoint` |
| `tests/unit/test_entrypoint_selection.py` (new) | Command-builder + import tests for the mamba path |
| `configs/base/family/qwen3_next.yaml` (new) | GDN-hybrid + MoE family mechanisms |
| `configs/base/family/nemotron_h.yaml` (new) | Mamba2-hybrid family mechanisms (`entrypoint: mamba`) |
| `configs/base/scale/600m_deepseek_v3.yaml` (new) | DeepSeek-V3 realization of the 600M budget |
| `configs/base/scale/600m_qwen3_next.yaml` (new) | Qwen3-Next realization of the 600M budget |
| `configs/base/scale/600m_nemotron_h.yaml` (new) | Nemotron-H realization of the 600M budget |
| `tests/unit/test_scale_budget.py` (new) | CI gate: each bake-off scale within ±2% of declared budget |
| `tools/size_check.py` (new) | CLI sizing helper for designing new realizations |
| `tests/integration/test_megatron_pin_features.py` (new) | Pin guard: required Megatron CLI args exist |
| `scripts/train_bakeoff_600m.sh` (new) | One launcher, family as the only argument |
| `docs/experiments/arch_bakeoff_600m.md` (new) | Protocol, decision matrix, results template |
| `docs/adding_a_family.md` (new) | How to add the next architecture family |
| `CHANGELOG.md` (modify) | Entry for the bake-off infrastructure |

---

### Task 1: Parameter accounting module (`src/utils/arch_params.py`)

**Files:**
- Create: `src/utils/arch_params.py`
- Test: `tests/unit/test_arch_params.py`

- [x] **Step 1: Write the failing tests**

```python
"""Unit tests for src/utils/arch_params.py (architecture param accounting).

CPU-only, no torch, no Megatron. All expected values are hand-computed from
the formulas documented in arch_params.py; the runtime ground-truth check is
the wandb_trainable_params log compared during GPU smoke (±2%).
"""

from __future__ import annotations

import pytest

from src.utils.arch_params import (
    active_non_embedding_params,
    attention_params,
    gdn_params,
    mamba_params,
    mla_params,
    moe_layer_params,
    moe_layer_active_params,
    non_embedding_params,
)


def test_attention_gqa():
    # q: 8*2*4=64, kv: 2*8*1*4=64, o: 2*4*8=64
    assert attention_params(hidden=8, num_heads=2, num_groups=1, head_dim=4) == 192


def test_attention_qk_norm_adds_two_head_dims():
    assert attention_params(hidden=8, num_heads=2, num_groups=1, head_dim=4, qk_norm=True) == 200


def test_mla():
    # q: 8*4 + 4 + 4*2*(4+2) = 84; kv: 8*(4+2) + 4 + 4*2*(4+4) = 116; o: 2*4*8 = 64
    assert (
        mla_params(
            hidden=8, num_heads=2, q_lora_rank=4, kv_lora_rank=4,
            qk_head_dim=4, qk_pos_emb_head_dim=2, v_head_dim=4,
        )
        == 264
    )


def test_gdn():
    # qk_dim=8, v_dim=16; in: 8*(16+32+8)=448; conv: 4*32=128; out: 128; small: 2*4+4=12
    assert (
        gdn_params(
            hidden=8, num_key_heads=2, key_head_dim=4,
            num_value_heads=4, value_head_dim=4, conv_kernel_dim=4,
        )
        == 716
    )


def test_mamba():
    # d_inner=16, nheads=4, conv_dim=32; in: 8*(32+16+4)=416; conv: 4*32+32=160;
    # out: 128; small: 3*4+16=28
    assert (
        mamba_params(hidden=8, state_dim=4, head_dim=4, num_groups=2)
        == 732
    )


def test_moe_layer():
    # router: 8*4+4=36; experts: 4*3*8*16=1536; shared: 3*8*16=384
    assert (
        moe_layer_params(
            hidden=8, num_experts=4, moe_ffn=16, shared_ffn=16, expert_bias=True,
        )
        == 1956
    )


def test_moe_layer_active():
    # router: 36; topk experts: 2*3*8*16=768; shared: 384
    assert (
        moe_layer_active_params(
            hidden=8, num_experts=4, topk=2, moe_ffn=16, shared_ffn=16, expert_bias=True,
        )
        == 1188
    )


def _dense_model() -> dict:
    return {
        "num_layers": 2, "hidden_size": 8, "ffn_hidden_size": 16,
        "num_attention_heads": 2, "num_query_groups": 1, "head_dim": 4,
        "activation": "SwiGLU", "qk_norm": False,
    }


def test_dispatch_dense_gpt():
    # per layer: attn 192 + swiglu 3*8*16=384 + 2 norms 16 = 592; x2 + final 8
    assert non_embedding_params(_dense_model()) == 1192


def test_dispatch_dense_with_mtp():
    # MTP block: one decoder layer 592 + eh_proj 2*8*8=128 + enorm 8 + hnorm 8 + final 8
    model = _dense_model() | {"mtp_num_layers": 1}
    assert non_embedding_params(model) == 1192 + 744


def test_dispatch_gdn_moe():
    model = {
        "num_layers": 2, "hidden_size": 8, "ffn_hidden_size": 16,
        "num_attention_heads": 2, "num_query_groups": 1, "head_dim": 4,
        "activation": "SwiGLU", "qk_norm": True,
        "linear_attention_freq": "[1, 0]",
        "gdn": {
            "enabled": True, "num_key_heads": 2, "key_head_dim": 4,
            "num_value_heads": 4, "value_head_dim": 4, "conv_kernel_dim": 4,
        },
        "moe": {
            "enabled": True, "layer_freq": "[1, 1]", "num_experts": 4,
            "ffn_hidden_size": 16, "shared_expert_intermediate_size": 16,
            "router_enable_expert_bias": True, "router_topk": 2,
        },
    }
    # layer0: gdn 716 + moe 1956 + norms 16 = 2688
    # layer1: attn(qk_norm) 200 + moe 1956 + norms 16 = 2172; final 8
    assert non_embedding_params(model) == 4868
    # active: moe -> 1188; layer0 1920 + layer1 1404 + final 8
    assert active_non_embedding_params(model) == 3332


def test_int_layer_freq_rejected():
    # Megatron's int form means different things per field; configs must use
    # explicit list-expression strings.
    model = _dense_model() | {
        "gdn": {
            "enabled": True, "num_key_heads": 2, "key_head_dim": 4,
            "num_value_heads": 4, "value_head_dim": 4, "conv_kernel_dim": 4,
        },
        "linear_attention_freq": 2,
    }
    with pytest.raises(ValueError):
        non_embedding_params(model)


def test_dispatch_hybrid_mamba():
    model = {
        "hidden_size": 8, "ffn_hidden_size": 16,
        "num_attention_heads": 2, "num_query_groups": 1, "head_dim": 4,
        "activation": "squared_relu", "qk_norm": False,
        "hybrid_layer_pattern": "M*-",
        "mamba": {"state_dim": 4, "head_dim": 4, "num_groups": 2},
    }
    # M 732 + attn 192 + relu2 mlp 2*8*16=256 + 3 per-layer norms 24 + final 8
    assert non_embedding_params(model) == 1212
    # no MoE -> active == total
    assert active_non_embedding_params(model) == 1212
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_arch_params.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.arch_params'`

- [x] **Step 3: Write the implementation**

```python
"""Architecture-aware non-embedding parameter accounting.

Computes total and active non-embedding parameter counts for every family
mechanism (dense GQA/MQA attention, MLA, GatedDeltaNet, Mamba2 mixers,
SwiGLU / squared-ReLU MLPs, DeepSeek-style MoE, MTP) straight from the merged
``base.model`` mapping — no torch, no Megatron.

Used by tests/unit/test_scale_budget.py (CI gate: every bake-off scale file
realizes its declared ``base.non_embedding_params`` within ±2%) and
tools/size_check.py (interactive sizing when designing a new realization).

Formulas mirror the pinned Megatron's own FLOPs accounting
(megatron/training/training.py, core_v0.17.0) and the megatron/core module
definitions. They are design-time estimates; the runtime ground truth is the
trainable-param count logged by the wandb_trainable_params patch, checked
during GPU smoke (docs/adding_a_family.md §4).
"""

from __future__ import annotations

from typing import Any


def _eval_pattern(freq: Any, num_layers: int) -> list[int]:
    """Resolve a Megatron layer-frequency python-list-expression string.

    Integer frequencies are rejected on purpose: Megatron gives the int form
    *different* semantics per field (linear_attention_freq N -> every Nth
    layer is SDPA, training.py:539; moe_layer_freq differs again). Explicit
    list-expression strings are unambiguous — use those in configs.
    """
    if isinstance(freq, int):
        raise ValueError(
            f"int layer freq {freq!r} is ambiguous across Megatron fields; "
            "use an explicit list expression string like '([1]*3+[0]*1)*6'"
        )
    pattern = list(eval(str(freq), {}, {}))  # noqa: S307 - trusted repo config
    if len(pattern) != num_layers:
        raise ValueError(
            f"pattern {freq!r} has length {len(pattern)}, expected {num_layers}"
        )
    return [int(x) for x in pattern]


def attention_params(
    *, hidden: int, num_heads: int, num_groups: int, head_dim: int, qk_norm: bool = False
) -> int:
    q = hidden * num_heads * head_dim
    kv = 2 * hidden * num_groups * head_dim
    o = num_heads * head_dim * hidden
    norms = 2 * head_dim if qk_norm else 0
    return q + kv + o + norms


def mla_params(
    *,
    hidden: int,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_head_dim: int,
    qk_pos_emb_head_dim: int,
    v_head_dim: int,
) -> int:
    q = (
        hidden * q_lora_rank
        + q_lora_rank  # q down-proj norm
        + q_lora_rank * num_heads * (qk_head_dim + qk_pos_emb_head_dim)
    )
    kv = (
        hidden * (kv_lora_rank + qk_pos_emb_head_dim)
        + kv_lora_rank  # kv down-proj norm
        + kv_lora_rank * num_heads * (qk_head_dim + v_head_dim)
    )
    o = num_heads * v_head_dim * hidden
    return q + kv + o


def gdn_params(
    *,
    hidden: int,
    num_key_heads: int,
    key_head_dim: int,
    num_value_heads: int,
    value_head_dim: int,
    conv_kernel_dim: int,
) -> int:
    qk_dim = num_key_heads * key_head_dim
    v_dim = num_value_heads * value_head_dim
    in_proj = hidden * (2 * qk_dim + 2 * v_dim + 2 * num_value_heads)  # q,k,v,z,a,b
    conv = conv_kernel_dim * (2 * qk_dim + v_dim)
    out_proj = v_dim * hidden
    small = 2 * num_value_heads + value_head_dim  # A_log, dt_bias, gated-norm weight
    return in_proj + conv + out_proj + small


def mamba_params(
    *,
    hidden: int,
    state_dim: int,
    head_dim: int,
    num_groups: int,
    expand: int = 2,
    conv_kernel_dim: int = 4,
) -> int:
    d_inner = expand * hidden
    nheads = d_inner // head_dim
    conv_dim = d_inner + 2 * num_groups * state_dim
    in_proj = hidden * (2 * d_inner + 2 * num_groups * state_dim + nheads)
    conv = conv_kernel_dim * conv_dim + conv_dim  # weight + bias
    out_proj = d_inner * hidden
    small = 3 * nheads + d_inner  # A_log, D, dt_bias, gated-norm weight
    return in_proj + conv + out_proj + small


def moe_layer_params(
    *, hidden: int, num_experts: int, moe_ffn: int, shared_ffn: int, expert_bias: bool
) -> int:
    router = hidden * num_experts + (num_experts if expert_bias else 0)
    experts = num_experts * 3 * hidden * moe_ffn  # SwiGLU experts
    shared = 3 * hidden * shared_ffn if shared_ffn else 0
    return router + experts + shared


def moe_layer_active_params(
    *, hidden: int, num_experts: int, topk: int, moe_ffn: int, shared_ffn: int, expert_bias: bool
) -> int:
    router = hidden * num_experts + (num_experts if expert_bias else 0)
    experts = topk * 3 * hidden * moe_ffn
    shared = 3 * hidden * shared_ffn if shared_ffn else 0
    return router + experts + shared


def _mlp_params(model: dict, *, active: bool, layer_is_moe: bool) -> int:
    hidden = int(model["hidden_size"])
    moe = model.get("moe") or {}
    if layer_is_moe:
        kwargs = dict(
            hidden=hidden,
            num_experts=int(moe["num_experts"]),
            moe_ffn=int(moe["ffn_hidden_size"]),
            shared_ffn=int(moe.get("shared_expert_intermediate_size") or 0),
            expert_bias=bool(moe.get("router_enable_expert_bias", False)),
        )
        if active:
            return moe_layer_active_params(topk=int(moe["router_topk"]), **kwargs)
        return moe_layer_params(**kwargs)
    ffn = int(model["ffn_hidden_size"])
    if str(model.get("activation", "SwiGLU")) == "squared_relu":
        return 2 * hidden * ffn
    return 3 * hidden * ffn  # SwiGLU


def _mixer_params(model: dict, *, layer_is_linear: bool) -> int:
    hidden = int(model["hidden_size"])
    gdn = model.get("gdn") or {}
    if layer_is_linear:
        return gdn_params(
            hidden=hidden,
            num_key_heads=int(gdn["num_key_heads"]),
            key_head_dim=int(gdn["key_head_dim"]),
            num_value_heads=int(gdn["num_value_heads"]),
            value_head_dim=int(gdn["value_head_dim"]),
            conv_kernel_dim=int(gdn.get("conv_kernel_dim", 4)),
        )
    if bool(model.get("multi_latent_attention", False)):
        return mla_params(
            hidden=hidden,
            num_heads=int(model["num_attention_heads"]),
            q_lora_rank=int(model["q_lora_rank"]),
            kv_lora_rank=int(model["kv_lora_rank"]),
            qk_head_dim=int(model["qk_head_dim"]),
            qk_pos_emb_head_dim=int(model["qk_pos_emb_head_dim"]),
            v_head_dim=int(model["v_head_dim"]),
        )
    return attention_params(
        hidden=hidden,
        num_heads=int(model["num_attention_heads"]),
        num_groups=int(model["num_query_groups"]),
        head_dim=int(model["head_dim"]),
        qk_norm=bool(model.get("qk_norm", False)),
    )


def _gpt_total(model: dict, *, active: bool) -> int:
    hidden = int(model["hidden_size"])
    num_layers = int(model["num_layers"])
    gdn = model.get("gdn") or {}
    linear_pattern = (
        _eval_pattern(model["linear_attention_freq"], num_layers)
        if bool(gdn.get("enabled", False))
        else [0] * num_layers
    )
    moe = model.get("moe") or {}
    moe_pattern = (
        _eval_pattern(moe["layer_freq"], num_layers)
        if bool(moe.get("enabled", False))
        else [0] * num_layers
    )

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


def _hybrid_total(model: dict, *, active: bool) -> int:
    hidden = int(model["hidden_size"])
    mamba = model.get("mamba") or {}
    pattern = str(model["hybrid_layer_pattern"])
    total = 0
    for ch in pattern:
        if ch == "M":
            total += mamba_params(
                hidden=hidden,
                state_dim=int(mamba.get("state_dim", 128)),
                head_dim=int(mamba.get("head_dim", 64)),
                num_groups=int(mamba.get("num_groups", 8)),
            )
        elif ch == "*":
            total += _mixer_params(model, layer_is_linear=False)
        elif ch == "-":
            total += _mlp_params(model, active=active, layer_is_moe=False)
        elif ch == "E":
            total += _mlp_params(model, active=active, layer_is_moe=True)
        else:
            raise ValueError(f"Unknown hybrid pattern symbol {ch!r} in {pattern!r}")
        total += hidden  # per-layer norm in the Mamba stack
    total += hidden  # final norm
    return total


def non_embedding_params(model: dict) -> int:
    """Total non-embedding params for a merged ``base.model`` mapping."""
    if model.get("hybrid_layer_pattern"):
        return _hybrid_total(dict(model), active=False)
    return _gpt_total(dict(model), active=False)


def active_non_embedding_params(model: dict) -> int:
    """Per-token active non-embedding params (MoE experts counted topk-only)."""
    if model.get("hybrid_layer_pattern"):
        return _hybrid_total(dict(model), active=True)
    return _gpt_total(dict(model), active=True)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_arch_params.py -v`
Expected: 12 PASSED

- [x] **Step 5: Run ruff + the full unit suite for regressions**

Run: `cd /lustre/fast/fast/zqiu/slm-research && ruff check src/utils/arch_params.py tests/unit/test_arch_params.py && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit -q`
Expected: ruff clean (an S307/eval warning may need `# noqa: S307`, already in the code); unit suite green except the 2 known pre-existing failures (see memory: launchers.submit tests).

- [x] **Step 6: Commit**

```bash
git add src/utils/arch_params.py tests/unit/test_arch_params.py
git commit -m "$(cat <<'EOF'
feat(arch): param accounting for attention/MLA/GDN/mamba/MoE families
EOF
)"
```

---

### Task 2: Sizing CLI (`tools/size_check.py`)

**Files:**
- Create: `tools/size_check.py`

- [x] **Step 1: Write the tool**

```python
"""Print total/active non-embedding params for a family+scale pair.

Usage (from repo root):
  python tools/size_check.py base/family=deepseek_v3 base/scale=600m_deepseek_v3
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.arch_params import (  # noqa: E402
    active_non_embedding_params,
    non_embedding_params,
)


def main() -> None:
    kv = dict(item.split("=", 1) for item in sys.argv[1:])
    family = kv["base/family"]
    scale = kv["base/scale"]
    fam = OmegaConf.load(REPO_ROOT / "configs/base/family" / f"{family}.yaml")
    sc = OmegaConf.load(REPO_ROOT / "configs/base/scale" / f"{scale}.yaml")
    merged = OmegaConf.merge(fam, sc)
    model = OmegaConf.to_container(merged.base.model, resolve=True)
    budget = int(merged.base.non_embedding_params)
    total = non_embedding_params(model)
    active = active_non_embedding_params(model)
    print(f"family={family} scale={scale}")
    print(f"budget {budget:>15,}")
    print(f"total  {total:>15,}  ({(total - budget) / budget:+.2%} vs budget)")
    print(f"active {active:>15,}  ({active / total:.1%} of total)")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Verify it runs against an existing pair**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python tools/size_check.py base/family=qwen3 base/scale=600m`
Expected: prints `budget 600,000,000` and a `total` within a few percent (the existing dense 600m rung; small drift vs the declared budget is informative, not an error).

- [x] **Step 3: Commit**

```bash
git add tools/size_check.py
git commit -m "$(cat <<'EOF'
feat(tools): size_check CLI for family/scale param budgets
EOF
)"
```

---

### Task 3: Generalize Megatron arg emission (`src/utils/megatron_args.py`)

**Files:**
- Modify: `src/utils/megatron_args.py` (function `_model_args`, lines 33–150)
- Test: `tests/unit/test_megatron_args_families.py`

- [x] **Step 1: Write the failing tests**

```python
"""Arg-emission tests for the new family mechanisms (GDN, hybrid mamba,
non-SwiGLU activation, non-rope positional encoding)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.utils.megatron_args import _model_args


def _cfg(**overrides):
    model = {
        "transformer_impl": None,
        "num_layers": 2,
        "hidden_size": 8,
        "ffn_hidden_size": 16,
        "num_attention_heads": 2,
        "num_query_groups": 2,
        "head_dim": 4,
        "seq_length": 16,
        "positional_encoding": "rope",
        "rotary_base": 10000,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "normalization": "RMSNorm",
        "norm_epsilon": 1.0e-6,
        "init_method_std": 0.02,
        "tie_embeddings": True,
        "activation": "SwiGLU",
    }
    model.update(overrides)
    return OmegaConf.create({"base": {"model": model}})


def _value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_swiglu_default_unchanged():
    args = _model_args(_cfg())
    assert "--swiglu" in args
    assert "--squared-relu" not in args
    assert _value(args, "--rotary-base") == "10000"


def test_squared_relu_replaces_swiglu():
    args = _model_args(_cfg(activation="squared_relu"))
    assert "--squared-relu" in args
    assert "--swiglu" not in args


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        _model_args(_cfg(activation="gelu"))


def test_positional_none_omits_rotary_args():
    args = _model_args(_cfg(positional_encoding="none"))
    assert _value(args, "--position-embedding-type") == "none"
    assert "--rotary-base" not in args
    assert "--rotary-percent" not in args


def test_gdn_emission():
    cfg = _cfg(
        qk_norm=True,
        linear_attention_freq="([1]*1+[0]*1)",
        gdn={
            "enabled": True,
            "num_key_heads": 2,
            "key_head_dim": 4,
            "num_value_heads": 4,
            "value_head_dim": 4,
            "conv_kernel_dim": 4,
        },
    )
    args = _model_args(cfg)
    assert _value(args, "--experimental-attention-variant") == "gated_delta_net"
    assert _value(args, "--linear-attention-freq") == "([1]*1+[0]*1)"
    assert _value(args, "--linear-num-key-heads") == "2"
    assert _value(args, "--linear-key-head-dim") == "4"
    assert _value(args, "--linear-num-value-heads") == "4"
    assert _value(args, "--linear-value-head-dim") == "4"
    assert _value(args, "--linear-conv-kernel-dim") == "4"
    assert "--enable-experimental" in args


def test_hybrid_pattern_and_mamba_dims():
    cfg = _cfg(
        positional_encoding="none",
        num_layers=3,
        hybrid_layer_pattern="M*-",
        mamba={"state_dim": 4, "head_dim": 4, "num_groups": 2},
    )
    args = _model_args(cfg)
    assert _value(args, "--hybrid-layer-pattern") == "M*-"
    assert _value(args, "--mamba-state-dim") == "4"
    assert _value(args, "--mamba-head-dim") == "4"
    assert _value(args, "--mamba-num-groups") == "2"


def test_hybrid_pattern_length_mismatch_raises():
    cfg = _cfg(positional_encoding="none", num_layers=2, hybrid_layer_pattern="M*-")
    with pytest.raises(ValueError):
        _model_args(cfg)


def test_hybrid_rejects_mtp():
    cfg = _cfg(
        positional_encoding="none",
        num_layers=3,
        hybrid_layer_pattern="M*-",
        mtp_num_layers=1,
    )
    with pytest.raises(ValueError):
        _model_args(cfg)
```

- [x] **Step 2: Run tests to verify the new ones fail**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args_families.py -v`
Expected: `test_swiglu_default_unchanged` PASSES (current behavior); all others FAIL.

- [x] **Step 3: Implement in `_model_args`**

In `src/utils/megatron_args.py`, replace the three rotary lines (currently lines 62–64):

```python
    _add(args, "--position-embedding-type", model.positional_encoding)
    if str(model.positional_encoding) == "rope":
        _add(args, "--rotary-base", model.rotary_base)
        _add(args, "--rotary-percent", model.get("rotary_percent", 1.0))
```

Replace `_add(args, "--swiglu")` (currently line 77) with:

```python
    activation = str(model.get("activation", "SwiGLU"))
    if activation == "SwiGLU":
        _add(args, "--swiglu")
    elif activation == "squared_relu":
        _add(args, "--squared-relu")
    else:
        raise ValueError(f"Unsupported model.activation {activation!r}")
```

Insert after the MLA block (after the `--enable-experimental` emission, currently line 120) the GDN and hybrid blocks:

```python
    # GatedDeltaNet linear attention (Qwen3-Next-style hybrids). All flags are
    # native Megatron CLI args auto-generated from TransformerConfig fields
    # (arguments.py:1655, pin core_v0.17.0); routed in gpt_builders.py via
    # args.experimental_attention_variant.
    gdn = model.get("gdn", {}) or {}
    if bool(gdn.get("enabled", False)):
        _add(args, "--experimental-attention-variant", "gated_delta_net")
        _add(args, "--linear-attention-freq", model.linear_attention_freq)
        _add(args, "--linear-num-key-heads", gdn.num_key_heads)
        _add(args, "--linear-key-head-dim", gdn.key_head_dim)
        _add(args, "--linear-num-value-heads", gdn.num_value_heads)
        _add(args, "--linear-value-head-dim", gdn.value_head_dim)
        _add(args, "--linear-conv-kernel-dim", gdn.get("conv_kernel_dim", 4))
        if "--enable-experimental" not in args:
            _add(args, "--enable-experimental")

    # Hybrid Mamba2 layer stacks (Nemotron-H-style). Megatron derives
    # num_layers and is_hybrid_model from the pattern; we validate coherence
    # here so a bad config fails at dry-run, not at rank startup. The mamba
    # path has no MTP support (pretrain_gpt.py docstring, pin core_v0.17.0).
    pattern = model.get("hybrid_layer_pattern", None)
    if pattern is not None:
        pattern = str(pattern)
        if len(pattern) != int(model.num_layers):
            raise ValueError(
                f"hybrid_layer_pattern length {len(pattern)} != num_layers "
                f"{int(model.num_layers)}"
            )
        if model.get("mtp_num_layers", None):
            raise ValueError("MTP is not supported on the mamba/hybrid path")
        _add(args, "--hybrid-layer-pattern", pattern)
        mamba = model.get("mamba", {}) or {}
        _add(args, "--mamba-state-dim", mamba.get("state_dim", 128))
        _add(args, "--mamba-head-dim", mamba.get("head_dim", 64))
        _add(args, "--mamba-num-groups", mamba.get("num_groups", 8))
```

- [x] **Step 4: Run the new tests + full unit suite (regression check on existing emission tests)**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_megatron_args_families.py tests/unit -q`
Expected: new file 9 PASSED; rest of suite unchanged (only the 2 known pre-existing failures).

- [x] **Step 5: Commit**

```bash
git add src/utils/megatron_args.py tests/unit/test_megatron_args_families.py
git commit -m "$(cat <<'EOF'
feat(args): emit GDN, hybrid-mamba, squared-relu and rope-conditional flags
EOF
)"
```

---

### Task 4: DeepSeek-V3 realization of the 600M budget

**Files:**
- Create: `configs/base/scale/600m_deepseek_v3.yaml`
- Test: `tests/unit/test_scale_budget.py`

- [x] **Step 1: Write the failing budget-gate test**

```python
"""CI gate: every bake-off scale file realizes its declared
base.non_embedding_params budget within ±2% (computed by arch_params)."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.arch_params import active_non_embedding_params, non_embedding_params

REPO_ROOT = Path(__file__).resolve().parents[2]

BAKEOFF_PAIRS = [
    ("deepseek_v3", "600m_deepseek_v3"),
]


def _merged_model(family: str, scale: str):
    fam = OmegaConf.load(REPO_ROOT / f"configs/base/family/{family}.yaml")
    sc = OmegaConf.load(REPO_ROOT / f"configs/base/scale/{scale}.yaml")
    merged = OmegaConf.merge(fam, sc)
    model = OmegaConf.to_container(merged.base.model, resolve=True)
    return model, int(merged.base.non_embedding_params)


@pytest.mark.parametrize("family,scale", BAKEOFF_PAIRS)
def test_bakeoff_scale_within_budget(family, scale):
    model, budget = _merged_model(family, scale)
    actual = non_embedding_params(model)
    rel = (actual - budget) / budget
    assert abs(rel) <= 0.02, f"{family}/{scale}: {actual:,} vs {budget:,} ({rel:+.2%})"


@pytest.mark.parametrize("family,scale", BAKEOFF_PAIRS)
def test_active_not_above_total(family, scale):
    model, _ = _merged_model(family, scale)
    assert active_non_embedding_params(model) <= non_embedding_params(model)
```

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_scale_budget.py -v`
Expected: FAIL with `FileNotFoundError` for `600m_deepseek_v3.yaml`.

- [x] **Step 2: Write the scale file**

`configs/base/scale/600m_deepseek_v3.yaml` — sized by arch_params: total 592.1M (−1.32% vs budget), active ≈252M. MLA + 1 dense + 23 MoE layers + 1 MTP block. Overrides the family's V3-proxy MLA ranks and MoE sizes down to this budget; `alltoall` dispatcher (single-node 600M runs; the family default `flex`+DeepEP is a multi-node optimization).

```yaml
# @package _global_
# DeepSeek-V3 mechanisms realized at the 600M non-embedding budget
# (arch bake-off; docs/experiments/arch_bakeoff_600m.md). Sized by
# tools/size_check.py: total 592.1M (-1.3%), active ~252M (MoE topk=4/16).
base:
  scale: "600m_deepseek_v3"
  non_embedding_params: 600_000_000
  model:
    num_layers: 24
    hidden_size: 1024
    ffn_hidden_size: 2816          # dense layer(s) only (layer 0)
    num_attention_heads: 16
    num_query_groups: 16           # MLA path; no GQA grouping
    head_dim: 64
    seq_length: 4096
    tie_embeddings: true
    q_lora_rank: 384
    kv_lora_rank: 256
    qk_head_dim: 64
    qk_pos_emb_head_dim: 32
    v_head_dim: 64
    moe:
      num_experts: 16
      layer_freq: "([0]*1+[1]*23)"
      ffn_hidden_size: 384
      shared_expert_intermediate_size: 768
      router_topk: 4
      router_group_topk: null      # n-group routing off at this expert count
      router_num_groups: null
      token_dispatcher_type: "alltoall"
      enable_deepep: false
```

- [x] **Step 3: Run the budget test + sizing tool**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_scale_budget.py -v && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python tools/size_check.py base/family=deepseek_v3 base/scale=600m_deepseek_v3`
Expected: 2 PASSED; tool prints total 592,091,136 (−1.32%), active 252,352,512. If outside ±2%, adjust `moe.ffn_hidden_size` (each ±32 moves total by ≈ ±2.3M × 24) and re-run — do NOT change `non_embedding_params`.

- [x] **Step 4: Dry-run the full launcher path (CPU)**

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m launchers.train_megatron base/family=deepseek_v3 base/scale=600m_deepseek_v3 experiment=optim/adam training_regime=ablation_40x scheduler=wsd cluster=dev --dry-run`
Expected: JSON payload with a `command` containing `--multi-latent-attention`, `--num-experts 16`, `--mtp-num-layers 1`, `--train-samples 5859375` (= 24e9/4096). If the champion diff step fails on a missing key, fix the scale yaml, not the launcher.

- [x] **Step 5: Commit**

```bash
git add configs/base/scale/600m_deepseek_v3.yaml tests/unit/test_scale_budget.py
git commit -m "$(cat <<'EOF'
feat(config): deepseek_v3 600m bake-off scale + budget gate test
EOF
)"
```

---

### Task 5: Qwen3-Next-style family + 600M realization

**Files:**
- Create: `configs/base/family/qwen3_next.yaml`
- Create: `configs/base/scale/600m_qwen3_next.yaml`
- Modify: `tests/unit/test_scale_budget.py` (add pair)

- [x] **Step 1: Add the pair to `BAKEOFF_PAIRS` and watch it fail**

```python
BAKEOFF_PAIRS = [
    ("deepseek_v3", "600m_deepseek_v3"),
    ("qwen3_next", "600m_qwen3_next"),
]
```

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_scale_budget.py -v`
Expected: deepseek pair PASSES, qwen3_next pair FAILS (`FileNotFoundError`).

- [x] **Step 2: Write the family file**

`configs/base/family/qwen3_next.yaml`. Mechanisms from the published Qwen3-Next release (hybrid GatedDeltaNet : full attention at 3:1, MoE with shared expert, QK-norm, 1M rotary base, partial RoPE 25%). Deliberate deviations, documented here: the MoE *router recipe* mirrors the deepseek_v3 family (sigmoid + seq_aux_loss + expert bias) so the bake-off isolates the mixer stack — a Qwen-faithful router is a follow-up ablation; MTP is omitted (composition of MTP with the experimental-attention spec path is unvalidated in this pin); the full-attention layers use standard Megatron attention (no per-head output gate) and standard RMSNorm (not zero-centered) — both are Qwen3-Next refinements with no native Megatron flag, accepted as approximations and noted in the protocol doc.

```yaml
# @package _global_
# Family-level defaults — architectural choices that do not depend on scale.
base:
  family: qwen3_next
  family_version: "next_2509"
  reference: "Qwen3-Next (Alibaba, Sep 2025) — GatedDeltaNet/attention hybrid + MoE"
  model:
    normalization: "RMSNorm"
    norm_epsilon: 1.0e-6
    activation: "SwiGLU"
    positional_encoding: "rope"
    rotary_base: 1000000
    rotary_scaling: null
    qk_norm: true
    rotary_percent: 0.25           # partial RoPE (published Qwen3-Next)
    attention_dropout: 0.0
    hidden_dropout: 0.0
    init_method_std: 0.02
    depth_scaled_init: false
    attention_backend: "flash"
    gdn:
      enabled: true
      num_key_heads: 8
      key_head_dim: 64
      num_value_heads: 16
      value_head_dim: 64
      conv_kernel_dim: 4
    moe:
      enabled: true
      router_load_balancing_type: "seq_aux_loss"
      token_dispatcher_type: "alltoall"
      enable_deepep: false
      router_pre_softmax: true
      grouped_gemm: true
      aux_loss_coeff: 1.0e-4
      router_group_topk: null
      router_num_groups: null
      router_topk_scaling_factor: 2.5
      router_score_function: "sigmoid"
      router_enable_expert_bias: true
      router_bias_update_rate: 1.0e-3
      router_dtype: "fp32"
      permute_fusion: true
  tokenizer:
    # Descriptive only — the actual tokenizer is fixed by the dataset manifest.
    nominal_name: "qwen3"
    nominal_vocab_size: 151936
```

- [x] **Step 3: Write the scale file**

Sized by arch_params: total 594.9M (−0.84%), active ≈241M. 24 layers, 18 GDN + 6 attention (3:1), MoE on every layer.

```yaml
# @package _global_
# Qwen3-Next mechanisms realized at the 600M non-embedding budget
# (arch bake-off; docs/experiments/arch_bakeoff_600m.md). Sized by
# tools/size_check.py: total 594.9M (-0.8%), active ~241M (MoE topk=4/16).
base:
  scale: "600m_qwen3_next"
  non_embedding_params: 600_000_000
  model:
    num_layers: 24
    hidden_size: 1024
    ffn_hidden_size: 2816          # schema-required; unused (all layers MoE)
    num_attention_heads: 16
    num_query_groups: 4            # GQA 4:1 on the 6 full-attention layers
    head_dim: 64
    seq_length: 4096
    tie_embeddings: true
    linear_attention_freq: "([1]*3+[0]*1)*6"
    moe:
      num_experts: 16
      layer_freq: "([1]*24)"
      ffn_hidden_size: 400
      shared_expert_intermediate_size: 416
      router_topk: 4
```

- [x] **Step 4: Run budget test, sizing tool, dry-run**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_scale_budget.py -v
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python tools/size_check.py base/family=qwen3_next base/scale=600m_qwen3_next
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m launchers.train_megatron base/family=qwen3_next base/scale=600m_qwen3_next experiment=optim/adam training_regime=ablation_40x scheduler=wsd cluster=dev --dry-run
```
Expected: 4 PASSED; total ≈ 594,939,712; dry-run command contains `--experimental-attention-variant gated_delta_net`, `--linear-attention-freq ([1]*3+[0]*1)*6`, `--num-experts 16`, and the same `--train-samples 5859375` as Task 4.

- [x] **Step 5: Commit**

```bash
git add configs/base/family/qwen3_next.yaml configs/base/scale/600m_qwen3_next.yaml tests/unit/test_scale_budget.py
git commit -m "$(cat <<'EOF'
feat(config): qwen3_next family (GDN hybrid + MoE) + 600m bake-off scale
EOF
)"
```

---

### Task 6: Mamba per-rank entrypoint + launcher routing

**Files:**
- Create: `launchers/pretrain_mamba_slm.py`
- Modify: `launchers/train_megatron.py` (function `build_torchrun_command`, lines 31–50)
- Test: `tests/unit/test_entrypoint_selection.py`

- [x] **Step 1: Write the failing tests**

```python
"""Entrypoint routing: base.model.entrypoint selects the per-rank module."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

import launchers.train_megatron as tm


def _cfg(entrypoint=None):
    model = {} if entrypoint is None else {"entrypoint": entrypoint}
    return OmegaConf.create(
        {
            "base": {"model": model},
            "cluster": {"gpus_per_node": 8},
            "_derived": {"run_dir": "runs/test"},
        }
    )


@pytest.fixture(autouse=True)
def _stub_megatron_args(monkeypatch):
    monkeypatch.setattr(tm, "build_megatron_args", lambda cfg: [])


def test_default_routes_to_gpt():
    assert "launchers.pretrain_gpt_slm" in tm.build_torchrun_command(_cfg())


def test_mamba_routes_to_mamba_module():
    assert "launchers.pretrain_mamba_slm" in tm.build_torchrun_command(_cfg("mamba"))


def test_unknown_entrypoint_raises():
    with pytest.raises(ValueError):
        tm.build_torchrun_command(_cfg("titan"))


def test_mamba_launcher_module_imports_without_megatron():
    # Module import must stay CPU-safe (Megatron imports live inside main()).
    import launchers.pretrain_mamba_slm as m

    assert callable(m.main)
```

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_entrypoint_selection.py -v`
Expected: all 4 FAIL (no module / no routing).

- [x] **Step 2: Write `launchers/pretrain_mamba_slm.py`**

```python
"""Per-rank Megatron Mamba/hybrid entrypoint for slm-research.

Mirror of pretrain_gpt_slm.py for families whose layer stack contains Mamba2
layers (``base.model.entrypoint: mamba``, e.g. nemotron_h). Megatron builds
these through MambaModel (pretrain_mamba.py + mamba_builders.py at the pin
root), not GPTModel — the GPT entrypoint cannot express the M/*/- pattern.

Deliberate differences from the GPT twin:
- imports pretrain_mamba / mamba_builder instead of pretrain_gpt / gpt_builder
- no titan_init wrapping (a GPT-path reproduction concern)
- no get_embedding_ranks kwarg (pretrain_mamba.py does not pass one)

NOTE: this path has no MTP support (pretrain_gpt.py docstring, pin
core_v0.17.0); src/utils/megatron_args.py rejects mtp on hybrid configs.
The always-on patches compose unchanged: they target
megatron.training.training symbols shared by both entrypoints (the
pretrain_gpt-targeted ones like overfit_single_batch import cleanly and are
simply never hit on this path).
"""

from __future__ import annotations

import os
import sys
from functools import partial

from launchers.pretrain_gpt_slm import (
    _apply_runtime_patches,
    _combined_extra_args_provider,
    _load_resolved_config,
    _prepend_paths,
)


def main() -> None:
    _prepend_paths()

    config_path = None
    for idx, item in enumerate(sys.argv):
        if item == "--slm-config-path" and idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
            break
    if config_path is None:
        raise RuntimeError("--slm-config-path must be present in torchrun args")

    cfg = _load_resolved_config(config_path)

    if bool(cfg.get("cluster", {}).get("wandb_offline", False)):
        os.environ.setdefault("WANDB_MODE", "offline")

    _apply_runtime_patches(cfg)

    import pretrain_mamba as mm
    from mamba_builders import mamba_builder
    from megatron.core.enums import ModelType
    from megatron.training import inprocess_restart, pretrain, set_startup_timestamps

    set_startup_timestamps(
        program_start=mm._PROGRAM_START_TIME,
        main_entry=mm.time.time(),
    )
    mm.train_valid_test_datasets_provider.is_distributed = True
    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    wrapped_pretrain(
        mm.train_valid_test_datasets_provider,
        partial(mm.model_provider, mamba_builder),
        ModelType.encoder_or_decoder,
        mm.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=_combined_extra_args_provider(
            mm.add_modelopt_args if mm.has_nvidia_modelopt else None
        ),
        store=store,
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Route the entrypoint in `build_torchrun_command`**

In `launchers/train_megatron.py`, add above `build_torchrun_command`:

```python
_ENTRYPOINT_MODULES = {
    "gpt": "launchers.pretrain_gpt_slm",
    "mamba": "launchers.pretrain_mamba_slm",
}
```

and inside `build_torchrun_command`, replace the hardcoded `"launchers.pretrain_gpt_slm"` element:

```python
def build_torchrun_command(cfg) -> list[str]:
    entrypoint = str(cfg.base.model.get("entrypoint", "gpt"))
    if entrypoint not in _ENTRYPOINT_MODULES:
        raise ValueError(
            f"Unknown base.model.entrypoint {entrypoint!r}; "
            f"expected one of {sorted(_ENTRYPOINT_MODULES)}"
        )
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(cfg.cluster.gpus_per_node),
        "--nnodes",
        str(_launch_nnodes()),
        "--node_rank",
        str(os.environ.get("NODE_RANK", "0")),
        "--master_addr",
        str(os.environ.get("MASTER_ADDR", "localhost")),
        "--master_port",
        str(os.environ.get("MASTER_PORT", "6000")),
        "-m",
        _ENTRYPOINT_MODULES[entrypoint],
        "--slm-config-path",
        os.fspath(Path(REPO_ROOT) / cfg._derived.run_dir / "resolved_config.yaml"),
    ]
    cmd.extend(build_megatron_args(cfg))
    return cmd
```

- [x] **Step 4: Cross-check the mirror against the pin, then run tests**

Open `third_party/Megatron-LM/pretrain_mamba.py` `__main__` block (lines ~346–366) and verify the mirror passes the same five positional/keyword args (datasets provider, `partial(model_provider, mamba_builder)`, `ModelType.encoder_or_decoder`, `forward_step`, `args_defaults`/`store`/`extra_args_provider`) — reconcile the mirror if the pin differs from the code above; the pin is the source of truth.

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_entrypoint_selection.py tests/unit/test_train_megatron_command.py -v`
Expected: new 4 PASSED; existing command tests still green (default path unchanged).

- [x] **Step 5: Commit**

```bash
git add launchers/pretrain_mamba_slm.py launchers/train_megatron.py tests/unit/test_entrypoint_selection.py
git commit -m "$(cat <<'EOF'
feat(launchers): mamba per-rank entrypoint + base.model.entrypoint routing
EOF
)"
```

---

### Task 7: Nemotron-H-style family + 600M realization

**Files:**
- Create: `configs/base/family/nemotron_h.yaml`
- Create: `configs/base/scale/600m_nemotron_h.yaml`
- Modify: `tests/unit/test_scale_budget.py` (add pair + pattern-shape test)

- [x] **Step 1: Add the failing budget entry and a pattern-shape test**

In `tests/unit/test_scale_budget.py`:

```python
BAKEOFF_PAIRS = [
    ("deepseek_v3", "600m_deepseek_v3"),
    ("qwen3_next", "600m_qwen3_next"),
    ("nemotron_h", "600m_nemotron_h"),
]


def test_nemotron_pattern_shape():
    model, _ = _merged_model("nemotron_h", "600m_nemotron_h")
    pattern = str(model["hybrid_layer_pattern"])
    assert len(pattern) == int(model["num_layers"]) == 48
    assert pattern.count("M") == 24
    assert pattern.count("-") == 20
    assert pattern.count("*") == 4
    assert set(pattern) <= {"M", "-", "*"}
```

Run: `cd /lustre/fast/fast/zqiu/slm-research && /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_scale_budget.py -v`
Expected: nemotron entries FAIL (`FileNotFoundError`), others PASS.

- [x] **Step 2: Write the family file**

```yaml
# @package _global_
# Family-level defaults — architectural choices that do not depend on scale.
base:
  family: nemotron_h
  family_version: "h_2504"
  reference: "Nemotron-H (NVIDIA, arXiv:2504.03624) — Mamba2/attention/MLP hybrid"
  model:
    entrypoint: "mamba"            # builds via MambaModel (pretrain_mamba_slm)
    normalization: "RMSNorm"
    norm_epsilon: 1.0e-5
    activation: "squared_relu"
    positional_encoding: "none"    # Mamba layers carry position; no RoPE
    rotary_scaling: null
    qk_norm: false
    attention_dropout: 0.0
    hidden_dropout: 0.0
    init_method_std: 0.02
    depth_scaled_init: false
    attention_backend: "flash"
    mamba:
      state_dim: 128
      head_dim: 64
      num_groups: 8
  tokenizer:
    # Descriptive only — the actual tokenizer is fixed by the dataset manifest.
    nominal_name: "llama-3.1"
    nominal_vocab_size: 128256
```

- [x] **Step 3: Write the scale file**

Sized by arch_params: total 604.8M (+0.81%). 48 layers = 24 Mamba2 + 20 squared-ReLU MLP + 4 attention (≈8% attention, Nemotron-H ratio). The pattern string is `"M-M-M-M-M*M-" * 4`; the shape test in Step 1 guards the hand-expanded literal.

```yaml
# @package _global_
# Nemotron-H mechanisms realized at the 600M non-embedding budget
# (arch bake-off; docs/experiments/arch_bakeoff_600m.md). Sized by
# tools/size_check.py: total 604.8M (+0.8%); dense (active == total).
# Pattern = "M-M-M-M-M*M-" * 4 (24 M / 20 - / 4 *).
base:
  scale: "600m_nemotron_h"
  non_embedding_params: 600_000_000
  model:
    num_layers: 48
    hidden_size: 1280
    ffn_hidden_size: 5632
    num_attention_heads: 10
    num_query_groups: 2            # GQA 5:1 on the 4 attention layers
    head_dim: 128
    seq_length: 4096
    tie_embeddings: true
    hybrid_layer_pattern: "M-M-M-M-M*M-M-M-M-M-M*M-M-M-M-M-M*M-M-M-M-M-M*M-"
```

- [x] **Step 4: Run budget + shape tests, sizing tool, dry-run**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -c "p='M-M-M-M-M*M-'*4; print(p, len(p), p.count('M'), p.count('-'), p.count('*'))"
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/unit/test_scale_budget.py -v
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python tools/size_check.py base/family=nemotron_h base/scale=600m_nemotron_h
/lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m launchers.train_megatron base/family=nemotron_h base/scale=600m_nemotron_h experiment=optim/adam training_regime=ablation_40x scheduler=wsd cluster=dev --dry-run
```
Expected: the one-liner prints the exact pattern string for the YAML (48, 24, 20, 4) — if it differs from the literal in Step 3, fix the YAML to match the printed string; all tests PASS; total ≈ 604,840,000; dry-run command contains `-m launchers.pretrain_mamba_slm`, `--hybrid-layer-pattern`, `--squared-relu`, `--position-embedding-type none`, no `--rotary-base`, no `--mtp-num-layers`, same `--train-samples 5859375`.

- [x] **Step 5: Commit**

```bash
git add configs/base/family/nemotron_h.yaml configs/base/scale/600m_nemotron_h.yaml tests/unit/test_scale_budget.py
git commit -m "$(cat <<'EOF'
feat(config): nemotron_h family (mamba hybrid) + 600m bake-off scale
EOF
)"
```

---

### Task 8: Pin-guard integration test

**Files:**
- Create: `tests/integration/test_megatron_pin_features.py`

- [x] **Step 1: Write the test**

```python
"""Pin guard: the bake-off families rely on these Megatron CLI args existing
in third_party/Megatron-LM (core_v0.17.0). Re-run after any submodule bump
(SPEC.md §4.1 step 2).

Requires the cluster env (TransformerEngine's .so dlopens CUDA libs):
  source load_cuda13_2_nccl_env.sh
  PYTHONPATH=third_party/Megatron-LM <venv>/python -m pytest tests/integration/test_megatron_pin_features.py -v
"""

from __future__ import annotations

import sys

import pytest

REQUIRED_FIELDS = [
    # GatedDeltaNet (qwen3_next family)
    "experimental_attention_variant",
    "linear_attention_freq",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    # Hybrid mamba (nemotron_h family)
    "hybrid_layer_pattern",
    "mamba_state_dim",
    "mamba_head_dim",
    "mamba_num_groups",
    # Activation (nemotron_h family)
    "squared_relu",
]


def test_pin_exposes_family_flags():
    pytest.importorskip("transformer_engine")
    from megatron.training.arguments import parse_args

    argv, sys.argv = sys.argv, ["pin_guard"]
    try:
        args = parse_args(ignore_unknown_args=True)
    finally:
        sys.argv = argv
    missing = [f for f in REQUIRED_FIELDS if not hasattr(args, f)]
    assert not missing, f"pin lacks expected args: {missing}"
```

- [x] **Step 2: Run it with the cluster env**

Run: `cd /lustre/fast/fast/zqiu/slm-research && source load_cuda13_2_nccl_env.sh && PYTHONPATH=third_party/Megatron-LM /lustre/fast/fast/zqiu/slm_env/.venv/bin/python -m pytest tests/integration/test_megatron_pin_features.py -v`
Expected: 1 PASSED (or SKIPPED if transformer_engine cannot load on the current node — in that case run it on a compute node before the GPU smoke in Task 11; do not proceed to Task 11 without a PASS).

- [x] **Step 3: Commit**

```bash
git add tests/integration/test_megatron_pin_features.py
git commit -m "$(cat <<'EOF'
test(pin): guard Megatron CLI surface required by bake-off families
EOF
)"
```

---

### Task 9: Bake-off launcher script

**Files:**
- Create: `scripts/train_bakeoff_600m.sh`

- [x] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Architecture-family bake-off at the 600M non-embedding budget
# (docs/experiments/arch_bakeoff_600m.md). One run per family; everything
# except base/family + base/scale is identical across runs.
#
# Usage:
#   bash scripts/train_bakeoff_600m.sh <family> [overrides...]
#   family ∈ {qwen3, deepseek_v3, qwen3_next, nemotron_h}
# Examples:
#   bash scripts/train_bakeoff_600m.sh deepseek_v3 cluster=h100_de
#   bash scripts/train_bakeoff_600m.sh nemotron_h cluster=h100_de training.micro_batch_size=8
SLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SLM_REPO/load_cuda13_2_nccl_env.sh"

FAMILY="${1:?usage: train_bakeoff_600m.sh <family> [overrides...]}"
shift
case "$FAMILY" in
  qwen3)       SCALE="600m" ;;            # dense control (existing dev rung)
  deepseek_v3) SCALE="600m_deepseek_v3" ;;
  qwen3_next)  SCALE="600m_qwen3_next" ;;
  nemotron_h)  SCALE="600m_nemotron_h" ;;
  *) echo "unknown family: $FAMILY (qwen3|deepseek_v3|qwen3_next|nemotron_h)" >&2; exit 1 ;;
esac

python -m launchers.train_megatron \
  "base/family=$FAMILY" \
  "base/scale=$SCALE" \
  "experiment=optim/adam" \
  "training_regime=ablation_40x" \
  "scheduler=wsd" \
  "seed=42" \
  "$@"
```

- [x] **Step 2: Dry-run all four families through the script**

Run:
```bash
cd /lustre/fast/fast/zqiu/slm-research && chmod +x scripts/train_bakeoff_600m.sh
for f in qwen3 deepseek_v3 qwen3_next nemotron_h; do
  bash scripts/train_bakeoff_600m.sh "$f" cluster=dev --dry-run | tail -3
done
```
Expected: four JSON payloads, each with run_name `adam-<family>-<scale>-s42-<ts>`; identical `--train-samples` across all four; nemotron's command uses `launchers.pretrain_mamba_slm`. (If `python` in the sourced env lacks the repo deps, prefix with the venv path as in earlier tasks.)

- [x] **Step 3: Commit**

```bash
git add scripts/train_bakeoff_600m.sh
git commit -m "$(cat <<'EOF'
feat(scripts): train_bakeoff_600m.sh — one launcher per bake-off family
EOF
)"
```

---

### Task 10: Protocol doc, extensibility guide, changelog

**Files:**
- Create: `docs/experiments/arch_bakeoff_600m.md`
- Create: `docs/adding_a_family.md`
- Modify: `CHANGELOG.md` (prepend entry)

- [x] **Step 1: Write `docs/experiments/arch_bakeoff_600m.md`**

```markdown
# Architecture-family bake-off @ 600M (2026-06)

**Question.** Which base architecture should slm-research pretrain from
scratch: DeepSeek-V3 (MLA + MoE + MTP), Qwen3-Next-style (GatedDeltaNet
hybrid + MoE), or Nemotron-H-style (Mamba2 hybrid, dense)? Control:
existing `qwen3` dense 600M rung.

**Design.** One run per family, seed 42; everything else frozen:
`training_regime=ablation_40x` (24B tokens), `scheduler=wsd`,
`experiment=optim/adam`, GBS 1024, seq 4096, dataset
`nemotron_cc_v2_llama31_8b` (manifest-frozen tokenizer). All scale files
declare `non_embedding_params: 600_000_000` → identical `--train-samples`
and a shared GPTDataset cache. The two MoE families share the DeepSeek
router recipe (sigmoid, seq_aux_loss 1e-4, expert bias, topk 4/16) so the
comparison isolates the mixer/backbone.

| family | scale | total non-emb | active | entrypoint |
|---|---|---|---|---|
| qwen3 (control) | 600m | ~600M | =total | gpt |
| deepseek_v3 | 600m_deepseek_v3 | 592.1M | ~252M | gpt |
| qwen3_next | 600m_qwen3_next | 594.9M | ~241M | gpt |
| nemotron_h | 600m_nemotron_h | 604.8M | =total | mamba |

**Known asymmetries (accepted).** Hidden size differs (1280 for
nemotron_h/control vs 1024 for the MoE families) → tied-embedding counts
differ (embeddings sit outside the budget unit per SPEC §1.3). MoE families
have ~2.4x fewer active params per token — recorded, not equalized. LR is
the optim/adam default for every family (per-family LR tuning is a
follow-up sweep, not part of the controlled comparison). DeepSeek keeps its
MTP head (family identity); its `lm loss` is the comparison metric, not the
MTP auxiliary loss. qwen3_next approximates the published model on its
full-attention layers (no per-head output gate, standard rather than
zero-centered RMSNorm — no native Megatron flags for either).

**Launch.**
    bash scripts/train_bakeoff_600m.sh <family> cluster=<cluster>

**Decision metrics, in order.**
1. Validation `lm loss` at 24B tokens (primary; from the shared W&B project,
   runs auto-grouped by config identity).
2. Loss-vs-tokens curve over the final 20% (slope still healthy? crossovers?).
3. Train throughput (tokens/s/GPU, `--log-throughput`) and peak reserved
   memory — the efficiency term that scales to the 1.2B/2.4B promotions.
4. Stability: no loss spikes/divergence, grad-norm sane, (MoE) aux loss and
   router load-balance healthy.
5. Qualitative: Megatron parallelism maturity + Megatron-Bridge export path
   at promotion scale.

**Decision rule.** Best validation loss wins unless within seed noise of the
runner-up (use the champion ladder's seed-variance band; if inside it, rerun
the tied families at seeds 43, 44) — ties break on throughput, then
stability. Record the verdict + W&B links in the Results section below, then
promote the winner: realize `1_2b_<winner>.yaml` with tools/size_check.py
and run the 1.2B gate.

**Results.** _(fill after runs)_

| family | val loss @24B | tok/s/GPU | peak mem | verdict |
|---|---|---|---|---|
| qwen3 (control) | | | | |
| deepseek_v3 | | | | |
| qwen3_next | | | | |
| nemotron_h | | | | |
```

- [x] **Step 2: Write `docs/adding_a_family.md`**

```markdown
# Adding an architecture family

The architecture axis is `base/family` (mechanisms) × `base/scale`
(dimensions realizing a parameter budget). The frozen axes — dataset,
tokenizer, training regime, scheduler, optimizer experiment — never change
when a family is added. The pinned `third_party/Megatron-LM` is never
edited.

## 1. Family config (`configs/base/family/<name>.yaml`)
Architecture mechanisms only: normalization, activation, positional
encoding, attention variant (GQA dims live in scale; MLA/GDN/mamba blocks
live here with sensible defaults), MoE *recipe* (router type, aux loss —
not sizes), `entrypoint: mamba` if the stack contains Mamba layers.
Copy the closest existing family (`deepseek_v3`, `qwen3_next`,
`nemotron_h`) as a template.

## 2. Mechanism support, in order of preference
1. **Native Megatron CLI args** — check first; this pin auto-generates
   flags from TransformerConfig dataclass fields (arguments.py:1655), so
   most config fields are already reachable. Wire them in
   `src/utils/megatron_args.py::_model_args` behind a family-gated config
   key, with an emission test in `tests/unit/test_megatron_args_families.py`.
2. **ModuleSpec** — custom layer composition via `src/specs/` (see
   `ngpt_layer_spec.py`).
3. **Patch** — last resort, only when neither reaches the call site; follow
   `docs/patches_cookbook.md` (unique upstream target, registry conflicts
   are import-time errors).

## 3. Scale realization (`configs/base/scale/<budget>_<family>.yaml`)
Declares the budget (`non_embedding_params`) and the dimensions realizing
it. Add the formula for any *new* mixer/MLP mechanism to
`src/utils/arch_params.py` (TDD with hand-computed vectors), then size with:

    python tools/size_check.py base/family=<name> base/scale=<scale>

Land within ±2% of the budget and add the pair to `BAKEOFF_PAIRS` in
`tests/unit/test_scale_budget.py`. Never tweak `non_embedding_params` to
match the dims — it drives the token budget and the dataset cache key;
tweak the dims.

## 4. Verify
1. `pytest tests/unit -q` (emission + budget gates).
2. Dry-run: `python -m launchers.train_megatron base/family=<name>
   base/scale=<scale> experiment=optim/adam training_regime=ablation_40x
   scheduler=wsd cluster=dev --dry-run` — inspect the emitted command.
3. Pin guard (after any submodule bump):
   `tests/integration/test_megatron_pin_features.py`.
4. GPU smoke (~30M tokens): launch with `training.tokens_per_param=0.05`,
   keep `training.save_enabled=true` (disabling it breaks the Megatron
   wandb writer). Compare the wandb_trainable_params total against
   `tools/size_check.py` — agree within ~2% (embedding params accounted
   separately) before any real run.
```

- [x] **Step 3: Prepend a CHANGELOG entry**

Prepend under the changelog's top heading, matching the file's existing entry format:

```markdown
## 2026-06-12 — Architecture-family bake-off infrastructure
- `src/utils/arch_params.py` + budget gate: families realize a declared
  non-embedding budget within ±2% (600M bake-off: deepseek_v3 592.1M,
  qwen3_next 594.9M, nemotron_h 604.8M).
- `megatron_args`: GDN (experimental-attention-variant), hybrid-mamba,
  squared-relu, rope-conditional emission.
- New `launchers/pretrain_mamba_slm.py` + `base.model.entrypoint` routing
  (MambaModel path for nemotron_h; no MTP there by pin limitation).
- New families `qwen3_next`, `nemotron_h`; scales `600m_{deepseek_v3,
  qwen3_next,nemotron_h}`; `scripts/train_bakeoff_600m.sh`;
  protocol in docs/experiments/arch_bakeoff_600m.md.
```

- [x] **Step 4: Commit**

```bash
git add docs/experiments/arch_bakeoff_600m.md docs/adding_a_family.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(bakeoff): protocol, adding-a-family guide, changelog
EOF
)"
```

---

### Task 11: GPU smoke + bake-off handoff (USER-RUN)

No file changes. CPU work is done; everything below runs on GPU and is the **user's to launch** (ask before any GPU smoke; never launch the 24B-token runs).

- [ ] **Step 1: Confirm pin guard passed on a compute node** (Task 8 Step 2; rerun there if it skipped on the login node).

- [ ] **Step 2: Hand over smoke commands (~30M tokens ≈ 7 steps each)**

```bash
cd /lustre/fast/fast/zqiu/slm-research
codexlog bakeoff-smoke-deepseek  bash scripts/train_bakeoff_600m.sh deepseek_v3 cluster=dev training.tokens_per_param=0.05 training.micro_batch_size=2
codexlog bakeoff-smoke-qwen3next bash scripts/train_bakeoff_600m.sh qwen3_next  cluster=dev training.tokens_per_param=0.05 training.micro_batch_size=2
codexlog bakeoff-smoke-nemotron  bash scripts/train_bakeoff_600m.sh nemotron_h  cluster=dev training.tokens_per_param=0.05 training.micro_batch_size=2
```

Smoke pass criteria (check each log):
- run reaches the first logged iterations and loss is finite and falling;
- wandb_trainable_params total ≈ `tools/size_check.py` total + embedding params (±2%);
- nemotron run shows the MambaModel build (and no MTP/rotary flags);
- known fallbacks if a smoke fails: `base.model.transformer_impl=local`
  (TE spec issue), `base.model.moe.grouped_gemm=false` (grouped-GEMM kernel
  issue), `base.model.attention_backend=auto` (flash-attn dispatch issue) —
  record whichever was needed in the experiment doc, applied identically to
  all families it affects.

- [ ] **Step 3: Hand over the real bake-off commands (24B tokens each; user's call on cluster + timing)**

```bash
codexlog bakeoff-600m-qwen3     bash scripts/train_bakeoff_600m.sh qwen3       cluster=h100_de
codexlog bakeoff-600m-deepseek  bash scripts/train_bakeoff_600m.sh deepseek_v3 cluster=h100_de
codexlog bakeoff-600m-qwen3next bash scripts/train_bakeoff_600m.sh qwen3_next  cluster=h100_de
codexlog bakeoff-600m-nemotron  bash scripts/train_bakeoff_600m.sh nemotron_h  cluster=h100_de
```

- [ ] **Step 4: After runs finish — fill the Results table in `docs/experiments/arch_bakeoff_600m.md`, apply the decision rule, and only then start the winner's `1_2b_<winner>.yaml` promotion (new plan).**

---

## Self-review checklist (run after writing code, before each commit)

- Budget numbers in YAML comments must match what `tools/size_check.py` actually prints — update comments if formulas shifted.
- No edits under `third_party/` — `git status third_party/` must stay clean.
- `pytest tests/unit -q` green (modulo the 2 known pre-existing launcher-test failures) before every commit.

## Risks & mitigations

- **GDN/MoE composition or kernels misbehave at runtime** — the experimental-attention spec path is marked experimental upstream; the smoke (Task 11) is the gate, with the three fallback overrides listed there. If GDN is unusable, the bake-off proceeds with 3 runs and qwen3_next is rescoped (e.g. `G`-pattern via the mamba entrypoint as an alternative wiring).
- **Mamba kernels need extra deps** — the pin vendors Triton ops under `megatron/core/ssm/ops`; if the smoke still demands `causal_conv1d`/`mamba_ssm`, install them into the run env and record versions in the cluster config notes.
- **Megatron-internal arg validation rejects a combination** (e.g. GDN + MTP, pattern + flags) — caught at dry-run/smoke; resolve by config change, never by editing the pin.
- **Logging patches on the mamba path** — `training_log_eta` and `wandb_metric_normalize` target `megatron.training.training` symbols shared by both entrypoints, so bake-off metrics stay comparable; verify `tokens_seen` appears in the nemotron smoke's W&B run before the real launch.
