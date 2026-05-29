"""Split fused parallel linears into separate POET sub-projections.

Runs *before* ``replace_linears_with_poet`` (inside the ``poet_apply_to_model``
wrapper). Splitting produces ordinary parallel-linear sub-modules, which the
existing POET walker then wraps with one independent orbit each.

Supported under POET's constraints only: TP=1, ``transformer_impl='local'``,
non-gated attention. MLA has no fused ``linear_qkv`` so ``split_qkv`` is inert.

This module is import-safe on CPU (no Megatron import at module load); the
Megatron linear types are discovered lazily.
"""

from __future__ import annotations

import logging

import torch

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


def validate_divisible(
    module_path: str,
    seg_name: str,
    *,
    in_f: int,
    out_f: int,
    block_size: int,
    block_count: int | None,
) -> None:
    """Hard-error if a split sub-segment isn't POET-divisible.

    ``block_count`` (when set) takes precedence over ``block_size``, matching
    the unsplit walker's divisor precedence.
    """
    divisor = block_count if block_count is not None else block_size
    label = "block_count" if block_count is not None else "block_size"
    if in_f % divisor != 0 or out_f % divisor != 0:
        raise ValueError(
            f"[POET split] {module_path}.{seg_name} dims (in={in_f}, out={out_f}) "
            f"not divisible by {label}={divisor}. Choose a compatible "
            f"block_size/block_count, or disable splitting this layer."
        )
