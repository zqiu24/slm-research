"""Unfuse fused parallel linears into separate sub-projections.

This is a purely **architectural** transform — it does not depend on POET (or
any optimizer). It replaces a fused attention ``linear_qkv`` with separate
Q / K / V projections, and/or a fused SwiGLU ``linear_fc1`` with separate
gate / up projections, then patches the owning module's forward (per instance)
to call them. The forward output is reconstructed identically, so the model is
mathematically equivalent to the fused one (modulo floating-point reduction
order).

Consumers:
  * POET wraps each sub-projection in its own orthogonal orbit (the main use).
  * Plain Adam (etc.) trains them like any other weight — equivalent to the
    fused model.

Constraints (architectural): TP=1, non-gated attention; the MLP must be gated
(SwiGLU) for ``unfuse_fc1``. MLA has no fused ``linear_qkv`` so ``unfuse_qkv``
is inert there. POET-specific block-size divisibility is enforced separately by
the POET wrap, not here.

This module is import-safe on CPU (no Megatron import at module load); the
Megatron linear types are discovered lazily.
"""

from __future__ import annotations

import copy
import logging
import types
from collections.abc import Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pure geometry helpers (CPU-testable, no Megatron dependency)
# --------------------------------------------------------------------------


def qkv_segment_out_dims(num_attention_heads: int, num_query_groups: int, head_dim: int):
    """Return ``(q_out, kv_out)``: the output dims of the Q projection and of
    *each* of the K / V projections in the fused ``linear_qkv``."""
    q_out = num_attention_heads * head_dim
    kv_out = num_query_groups * head_dim
    return q_out, kv_out


def qkv_deinterleave_row_indices(num_attention_heads: int, num_query_groups: int, head_dim: int):
    """Row indices into the fused ``linear_qkv`` weight for Q, K, V.

    Megatron lays the fused output out group-major (``q1 q2 k1 v1 | q3 q4 k2 v2 | ...``):
    per group ``g`` the rows are ``[ q-heads (nqhpg*hd) | k (hd) | v (hd) ]``,
    where ``nqhpg = num_attention_heads // num_query_groups``.

    Returns ``(q_rows, k_rows, v_rows)`` as ``torch.long`` tensors.
    """
    ng = num_query_groups
    nqhpg = num_attention_heads // ng
    hd = head_dim
    stride = (nqhpg + 2) * hd
    q_rows: list[int] = []
    k_rows: list[int] = []
    v_rows: list[int] = []
    for g in range(ng):
        base = g * stride
        q_rows.extend(range(base, base + nqhpg * hd))
        k_rows.extend(range(base + nqhpg * hd, base + nqhpg * hd + hd))
        v_rows.extend(range(base + (nqhpg + 1) * hd, base + (nqhpg + 2) * hd))
    return (
        torch.tensor(q_rows, dtype=torch.long),
        torch.tensor(k_rows, dtype=torch.long),
        torch.tensor(v_rows, dtype=torch.long),
    )


def qkv_interleave_index(
    num_attention_heads: int, num_query_groups: int, head_dim: int
) -> torch.Tensor:
    """Index that maps the de-interleaved concat ``[q | k | v]`` (along the last
    dim) back to the fused interleaved ``mixed_qkv`` layout.

    With ``idx = qkv_interleave_index(...)`` and a concatenated tensor
    ``cat`` ordered ``[q_out, k_out, v_out]``,
    ``cat.index_select(-1, idx)`` reproduces the fused output.
    """
    q_rows, k_rows, v_rows = qkv_deinterleave_row_indices(
        num_attention_heads, num_query_groups, head_dim
    )
    len_q, len_k, len_v = q_rows.numel(), k_rows.numel(), v_rows.numel()
    total = len_q + len_k + len_v
    idx = torch.empty(total, dtype=torch.long)
    idx[q_rows] = torch.arange(0, len_q)
    idx[k_rows] = torch.arange(len_q, len_q + len_k)
    idx[v_rows] = torch.arange(len_q + len_k, total)
    return idx


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _column_linear_types(extra: Iterable[type] = ()) -> tuple[type, ...]:
    """Megatron column-parallel linear types, discovered lazily (empty on CPU),
    unioned with any ``extra`` types (tests pass ``nn.Linear``)."""
    types_: tuple[type, ...] = ()
    try:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        types_ += (ColumnParallelLinear,)
    except Exception:
        pass
    try:
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

        types_ += (TEColumnParallelLinear,)
    except Exception:
        pass
    return types_ + tuple(extra)


def _linear_out(module, x):
    """Call a linear that may return a tensor (``nn.Linear``) or a
    ``(output, bias)`` tuple (Megatron / POET). Returns ``(output, bias)``."""
    r = module(x)
    if isinstance(r, tuple):
        return r[0], (r[1] if len(r) > 1 else None)
    return r, None


def _make_sub_linear(src: nn.Module, rows: torch.Tensor) -> nn.Module:
    """Build a typed copy of ``src`` whose weight/bias are the rows ``rows`` of
    ``src``. The copy keeps ``src``'s class and config so any downstream walker
    (e.g. POET) recognises it; only weight/bias/shape are sliced.

    The copy's own ``forward`` is used directly under plain training (and is
    correct for ``nn.Linear`` in tests); under POET it is later replaced by a
    POET module.
    """
    # ProcessGroup-bearing attrs on a real Megatron linear are not
    # deepcopy-able; detach them across the copy and restore on both objects.
    # (No-op for nn.Linear in CPU tests.) This is the highest-risk spot vs. the
    # real Megatron build — validated by the GPU smoke.
    _pg_attrs = ("tp_group", "process_group", "pg_collection", "explicit_expert_comm")
    _saved = {}
    for attr in _pg_attrs:
        if attr in getattr(src, "__dict__", {}):
            _saved[attr] = src.__dict__[attr]
            src.__dict__[attr] = None
    try:
        sub = copy.deepcopy(src)
    finally:
        for attr, val in _saved.items():
            src.__dict__[attr] = val
            sub.__dict__[attr] = val
    w = src.weight.data.index_select(0, rows.to(src.weight.device)).clone()
    sub.weight = nn.Parameter(w, requires_grad=src.weight.requires_grad)
    has_bias = getattr(src, "bias", None) is not None and src.bias.numel() > 0
    if has_bias:
        b = src.bias.data.index_select(0, rows.to(src.bias.device)).clone()
        sub.bias = nn.Parameter(b, requires_grad=src.bias.requires_grad)
    else:
        sub.bias = None
    out_f = rows.numel()
    # Best-effort fix of Megatron size bookkeeping (absent on nn.Linear).
    for attr in ("out_features", "output_size", "output_size_per_partition"):
        if hasattr(sub, attr):
            setattr(sub, attr, out_f)
    return sub


# --------------------------------------------------------------------------
# FC1 (gate / up) surgery
# --------------------------------------------------------------------------


def _unfused_mlp_forward(self, hidden_states, per_token_scale=None, **kwargs):
    """Replacement ``MLP.forward`` calling separate gate/up projections.

    Mirrors Megatron's non-fused gated ``glu()`` path
    (megatron/core/transformer/mlp.py). Does not use the fused
    ``bias_swiglu_impl`` kernel; numerically identical for SwiGLU.
    """
    gate, gate_bias = _linear_out(self.linear_fc1_gate, hidden_states)
    up, up_bias = _linear_out(self.linear_fc1_up, hidden_states)
    if gate_bias is not None:
        gate = gate + gate_bias
    if up_bias is not None:
        up = up + up_bias
    clamp = getattr(self.config, "activation_func_clamp_value", None)
    if clamp is not None:
        gate = gate.clamp(min=None, max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    offset = getattr(self.config, "glu_linear_offset", 0.0)
    intermediate = self.config.activation_func(gate) * (up + offset)
    if per_token_scale is not None:
        od = intermediate.dtype
        intermediate = (intermediate * per_token_scale.unsqueeze(-1)).to(od)
    out = self.linear_fc2(intermediate)
    if isinstance(out, tuple):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


def _unfused_shared_expert_forward(self, hidden_states, **kwargs):
    """Replacement ``SharedExpertMLP.forward`` for the unfused gate/up path.

    ``SharedExpertMLP`` (megatron/core/transformer/moe/shared_experts.py)
    returns a BARE tensor (not the ``(output, bias)`` tuple a plain MLP
    returns): the MoE layer's ``postprocess`` does ``output + shared_expert``
    on tensors. Reuse the unfused gate/up body, then drop the bias and apply
    the optional shared-expert sigmoid gate, mirroring the original.
    """
    out, _bias = _unfused_mlp_forward(self, hidden_states, **kwargs)
    if getattr(self, "use_shared_expert_gate", False):
        logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
        out = out * torch.nn.functional.sigmoid(logits)
    return out


def _unfuse_one_mlp_fc1(mlp, path, *, linear_types) -> bool:
    fc1 = getattr(mlp, "linear_fc1", None)
    if fc1 is None or not isinstance(fc1, linear_types):
        return False
    if not getattr(mlp.config, "gated_linear_unit", False):
        raise ValueError(f"[unfuse] {path}: --unfuse-fc1 requires a gated (SwiGLU) MLP.")
    out_f, in_f = fc1.weight.shape
    if out_f % 2 != 0:
        raise ValueError(f"[unfuse] {path}.linear_fc1 out dim {out_f} is not even.")
    ffn = out_f // 2
    gate_rows = torch.arange(0, ffn, dtype=torch.long)
    up_rows = torch.arange(ffn, 2 * ffn, dtype=torch.long)
    mlp.linear_fc1_gate = _make_sub_linear(fc1, gate_rows)
    mlp.linear_fc1_up = _make_sub_linear(fc1, up_rows)
    del mlp.linear_fc1
    # SharedExpertMLP returns a bare tensor (MoE postprocess adds it as a
    # tensor); a plain MLP returns (output, bias). Match the right contract.
    is_shared_expert = hasattr(mlp, "use_shared_expert_gate")
    fwd = _unfused_shared_expert_forward if is_shared_expert else _unfused_mlp_forward
    mlp.forward = types.MethodType(fwd, mlp)
    logger.info(
        "[unfuse] %s.linear_fc1 -> gate/up (ffn=%d%s)",
        path,
        ffn,
        ", shared-expert" if is_shared_expert else "",
    )
    return True


# --------------------------------------------------------------------------
# QKV (Q / K / V) surgery
# --------------------------------------------------------------------------


def _unfused_qkv_forward(
    self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True
):
    """Replacement ``SelfAttention.get_query_key_value_tensors`` for TP=1,
    non-gated attention. Calls separate Q/K/V projections, reassembles the
    interleaved ``mixed_qkv``, then runs Megatron's TP=1 post-linear
    view/split/reshape/layernorm so downstream attention math is bit-identical.
    """
    assert not output_gate, "[unfuse] unfuse_qkv does not support gated attention."
    q, _ = _linear_out(self.linear_q, hidden_states)
    k, _ = _linear_out(self.linear_k, hidden_states)
    v, _ = _linear_out(self.linear_v, hidden_states)
    mixed = torch.cat([q, k, v], dim=-1).index_select(-1, self._unfuse_qkv_interleave_index)

    hd = self.hidden_size_per_attention_head
    ng = self.num_query_groups_per_partition
    nqhpg = self.num_attention_heads_per_partition // ng
    mixed = mixed.view(*mixed.size()[:-1], ng, (nqhpg + 2) * hd)
    split_arg_list = [nqhpg * hd, hd, hd]
    if not split_qkv:
        return mixed, split_arg_list
    query, key, value = torch.split(mixed, split_arg_list, dim=-1)
    query = query.reshape(*query.size()[:-2], -1, hd)
    if self.q_layernorm is not None:
        query = self.q_layernorm(query)
    if self.k_layernorm is not None:
        key = self.k_layernorm(key)
    return query, key, value


def _unfused_backward_qkv_proj(self):
    """Replacement for SelfAttention._backward_qkv_proj after linear_qkv removal.

    Only relevant for Megatron's delayed-wgrad path; guarded so it never errors
    if invoked.
    """
    for attr in ("linear_q", "linear_k", "linear_v"):
        m = getattr(self, attr, None)
        if m is not None and hasattr(m, "backward_dw"):
            m.backward_dw()


def _unfuse_one_attention_qkv(attn, path, *, linear_types) -> bool:
    qkv = getattr(attn, "linear_qkv", None)
    if qkv is None or not isinstance(qkv, linear_types):
        return False
    if getattr(attn, "world_size", 1) != 1:
        raise ValueError(f"[unfuse] {path}: --unfuse-qkv requires TP=1.")
    if getattr(attn.config, "attention_output_gate", False):
        raise ValueError(f"[unfuse] {path}: --unfuse-qkv does not support gated attention.")

    hd = attn.hidden_size_per_attention_head
    ng = attn.num_query_groups_per_partition
    nah = attn.num_attention_heads_per_partition
    q_out, kv_out = qkv_segment_out_dims(nah, ng, hd)
    out_f, in_f = qkv.weight.shape
    if q_out + 2 * kv_out != out_f:
        raise ValueError(
            f"[unfuse] {path}.linear_qkv out dim {out_f} != q+2kv "
            f"({q_out}+2*{kv_out}); unexpected layout (gated attention?)."
        )

    q_rows, k_rows, v_rows = qkv_deinterleave_row_indices(nah, ng, hd)
    attn.linear_q = _make_sub_linear(qkv, q_rows)
    attn.linear_k = _make_sub_linear(qkv, k_rows)
    attn.linear_v = _make_sub_linear(qkv, v_rows)
    attn.register_buffer(
        "_unfuse_qkv_interleave_index",
        # Pin to the weight's device so the forward's index_select matches the
        # model's device after any later .cuda()/.to() (the index moves with the
        # module since it is a registered buffer).
        qkv_interleave_index(nah, ng, hd).to(qkv.weight.device),
        persistent=False,
    )
    del attn.linear_qkv
    attn.get_query_key_value_tensors = types.MethodType(_unfused_qkv_forward, attn)
    attn._backward_qkv_proj = types.MethodType(_unfused_backward_qkv_proj, attn)
    logger.info(
        "[unfuse] %s.linear_qkv -> q/k/v (q=%d, kv=%d, groups=%d)",
        path,
        q_out,
        kv_out,
        ng,
    )
    return True


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def unfuse_fused_linears(
    model: nn.Module,
    *,
    unfuse_qkv: bool,
    unfuse_fc1: bool,
    linear_types: Iterable[type] | None = None,
) -> int:
    """Unfuse fused linears in-place; returns the number of fused linears split.

    ``linear_types`` overrides the recognised column-parallel linear classes
    (tests pass ``(nn.Linear,)``); defaults to Megatron's column-parallel types.
    """
    types_ = _column_linear_types() if linear_types is None else tuple(linear_types)
    n = 0
    for name, mod in list(model.named_modules()):
        if (
            unfuse_qkv
            and hasattr(mod, "linear_qkv")
            and _unfuse_one_attention_qkv(mod, name or "<root>", linear_types=types_)
        ):
            n += 1
        if (
            unfuse_fc1
            and hasattr(mod, "linear_fc1")
            and _unfuse_one_mlp_fc1(mod, name or "<root>", linear_types=types_)
        ):
            n += 1
    return n
