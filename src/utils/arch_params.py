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
    pattern = list(eval(str(freq), {}, {}))
    if len(pattern) != num_layers:
        raise ValueError(f"pattern {freq!r} has length {len(pattern)}, expected {num_layers}")
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
