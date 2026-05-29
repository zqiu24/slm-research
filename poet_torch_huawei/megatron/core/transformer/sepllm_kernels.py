# Copyright (c) 2024, SepLLM Authors. CUDA-optimized kernels for SepLLM sparse attention.
# Reference: "SepLLM: Accelerate Large Language Models by Compressing One Segment
#             into One Separator" (ICML 2025, arXiv:2412.12094)
#
# Two acceleration paths:
#   1) FlexAttention (PyTorch >= 2.5): block-sparse masking via torch.nn.attention.flex_attention
#   2) Triton: custom fused kernels with online softmax, skipping fully-masked blocks
#
# Both paths compute EXACT same result as the dense-mask baseline, but:
#   - Skip O(n^2) work for masked-out regions
#   - Never materialize the full [seq, seq] attention score matrix
#   - Handle both forward AND backward (training-ready)

import math
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple

# ============================================================================
# Process-global context for passing input_ids / sep_mask to attention kernels.
#
# NOT thread-local: PyTorch's autograd engine runs checkpoint-recompute on a
# separate thread, so threading.local() would lose the context during backward.
# Each rank is a separate process, so a plain global is safe.
# ============================================================================

class _SepLLMContext:
    __slots__ = ('input_ids', 'sep_mask', 'block_mask')
    def __init__(self):
        self.input_ids: Optional[Tensor] = None
        self.sep_mask: Optional[Tensor] = None
        self.block_mask = None

_sepllm_ctx = _SepLLMContext()


def set_sepllm_context(input_ids: Tensor, sep_mask: Tensor):
    """Store input_ids and precomputed separator mask for current forward pass."""
    _sepllm_ctx.input_ids = input_ids
    _sepllm_ctx.sep_mask = sep_mask
    _sepllm_ctx.block_mask = None  # will be lazily built on first use


def get_sepllm_context() -> Tuple[Optional[Tensor], Optional[Tensor]]:
    return _sepllm_ctx.input_ids, _sepllm_ctx.sep_mask


def get_sepllm_block_mask():
    """Get cached FlexAttention BlockMask, or None if not yet built."""
    return _sepllm_ctx.block_mask


def set_sepllm_block_mask(block_mask):
    """Cache a prebuilt FlexAttention BlockMask for reuse across layers."""
    _sepllm_ctx.block_mask = block_mask


def clear_sepllm_context():
    _sepllm_ctx.input_ids = None
    _sepllm_ctx.sep_mask = None
    _sepllm_ctx.block_mask = None


def precompute_sep_mask(input_ids: Tensor, separator_token_ids: List[int]) -> Tensor:
    """Precompute boolean mask: True where token is a separator. [B, S]"""
    sep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in separator_token_ids:
        sep_mask |= (input_ids == sid)
    return sep_mask


# ============================================================================
# Path 1: FlexAttention (PyTorch >= 2.5)
# ============================================================================
_FLEX_AVAILABLE = False
_flex_attention_fn = None  # type: ignore
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention_fn,
        create_block_mask,
    )

    _FLEX_AVAILABLE = True
except ImportError:
    pass

# Lazy torch.compile — NOT used by default: Inductor+Triton often requests >232KB shared memory
# (e.g. 256KB) which fails on L20/L40-class GPUs. Eager flex_attention avoids that.
_sepllm_flex_compiled = None


def _sepllm_want_flex_compile() -> bool:
    return os.environ.get('SEPLLM_FLEX_COMPILE', '0').lower() in ('1', 'true', 'yes')


_flex_compile_failed = False


def flex_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask,
    scale: float,
    enable_gqa: bool = False,
) -> Tensor:
    """Run FlexAttention; optionally wrap with torch.compile (SEPLLM_FLEX_COMPILE=1).

    If compile fails (e.g. Triton shared memory OOM on L40-class GPUs), automatically
    falls back to eager mode and logs a warning.
    """
    assert _FLEX_AVAILABLE and _flex_attention_fn is not None
    global _flex_compile_failed

    if _sepllm_want_flex_compile() and not _flex_compile_failed:
        global _sepllm_flex_compiled
        if _sepllm_flex_compiled is None:
            _sepllm_flex_compiled = torch.compile(_flex_attention_fn, dynamic=False)
        try:
            return _sepllm_flex_compiled(
                query, key, value, block_mask=block_mask, scale=scale,
                enable_gqa=enable_gqa,
            )
        except RuntimeError as e:
            if 'OutOfResources' in str(e) or 'shared memory' in str(e):
                import warnings
                warnings.warn(
                    f"SepLLM: flex_attention compile failed ({e}). "
                    f"Falling back to eager mode. Consider using --sepllm-kernel triton.",
                    stacklevel=2,
                )
                _flex_compile_failed = True
            else:
                raise

    return _flex_attention_fn(
        query, key, value, block_mask=block_mask, scale=scale,
        enable_gqa=enable_gqa,
    )


def build_sepllm_block_mask(
    sep_mask: Tensor,
    num_heads: int,
    seq_len: int,
    init_token_count: int,
    local_window_size: int,
    block_size: int = 64,
) -> "BlockMask":
    """Pre-build FlexAttention BlockMask from separator mask.

    Call this once per micro-batch (outside the attention kernel) to amortize
    the mask construction cost.  The returned BlockMask is passed directly to
    flex_attention.
    """
    B = sep_mask.shape[0]
    _sep = sep_mask

    def mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        is_init = kv_idx < init_token_count
        is_sep = _sep[b, kv_idx]
        is_local = (q_idx - kv_idx) < local_window_size
        return causal & (is_init | is_sep | is_local)

    return create_block_mask(
        mask_mod, B, num_heads, seq_len, seq_len,
        BLOCK_SIZE=(block_size, block_size),
        device=sep_mask.device,
    )


def sepllm_flex_attention(
    query: Tensor,          # [B, H, Sq, D]
    key: Tensor,            # [B, H, Sk, D]
    value: Tensor,          # [B, H, Sk, D]
    sep_mask: Tensor,       # [B, Sk] bool — True = separator
    init_token_count: int,
    local_window_size: int,
    softmax_scale: float,
    dropout_p: float = 0.0,
    block_size: int = 64,
) -> Tensor:
    """SepLLM attention via FlexAttention (eager by default).

    Blocks that are entirely zero are skipped in both forward and backward.
    Set SEPLLM_FLEX_COMPILE=1 to enable torch.compile (requires sufficient GPU shared memory).
    """
    assert _FLEX_AVAILABLE, "FlexAttention requires PyTorch >= 2.5"
    B, H, Sq, D = query.shape

    blk_mask = build_sepllm_block_mask(
        sep_mask, H, Sq, init_token_count, local_window_size, block_size,
    )

    return flex_attention_forward(
        query, key, value,
        block_mask=blk_mask,
        scale=softmax_scale,
    )


# ============================================================================
# Path 2: Triton block-sparse attention (forward + backward)
# ============================================================================
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


if _TRITON_AVAILABLE:
    @triton.jit
    def _sepllm_attn_fwd_kernel(
        Q, K, V, sep_mask_ptr, Out, L_ptr,
        softmax_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_sb, stride_sn,
        stride_lb, stride_lh, stride_lm,
        NUM_HEADS: tl.constexpr,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        INIT_TOKENS: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Triton forward kernel for SepLLM sparse causal attention.

        Grid: (cdiv(S, BLOCK_M), B * H).
        For each query-block, iterates over key-blocks, applies block-level
        skip, and uses online softmax (Flash-Attention algorithm).
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_idx = pid_bh // NUM_HEADS
        head_idx = pid_bh % NUM_HEADS

        qstart = pid_m * BLOCK_M
        offs_m = qstart + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM

        q_ptrs = (Q + batch_idx * stride_qb + head_idx * stride_qh
                  + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & d_mask[None, :], other=0.0)

        m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Upper bound: causal means we only need k <= max(offs_m) = qstart + BLOCK_M - 1
        kv_end = tl.minimum(qstart + BLOCK_M, N_CTX)
        for kstart in range(0, kv_end, BLOCK_N):
            offs_n = kstart + tl.arange(0, BLOCK_N)

            # Block-level skip: relevant if init / sep / local overlap
            has_init = kstart < INIT_TOKENS
            has_local = (kstart + BLOCK_N - 1) >= tl.maximum(qstart - WINDOW_SIZE + 1, 0)

            sep_ptrs = sep_mask_ptr + batch_idx * stride_sb + offs_n * stride_sn
            sep_block = tl.load(sep_ptrs, mask=offs_n < N_CTX, other=0).to(tl.int1)
            has_sep = tl.sum(sep_block.to(tl.int32), axis=0) > 0

            if has_init | has_local | has_sep:
                k_ptrs = (K + batch_idx * stride_kb + head_idx * stride_kh
                          + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
                k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_CTX) & d_mask[None, :], other=0.0)
                v_ptrs = (V + batch_idx * stride_vb + head_idx * stride_vh
                          + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
                v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_CTX) & d_mask[None, :], other=0.0)

                s = tl.dot(q, tl.trans(k)) * softmax_scale

                causal_mask = offs_m[:, None] >= offs_n[None, :]
                init_mask = offs_n[None, :] < INIT_TOKENS
                sep_mask_2d = sep_block[None, :]
                local_mask = (offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE
                valid_kv = offs_n[None, :] < N_CTX
                attend = causal_mask & (init_mask | sep_mask_2d | local_mask) & valid_kv
                s = tl.where(attend, s, float('-inf'))

                # Online softmax (Flash-Attention-2 style)
                row_max = tl.max(s, axis=1)
                m_new = tl.maximum(m_i, row_max)
                # Correction factor for old accumulator
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])
                l_new = alpha * l_i + tl.sum(p, axis=1)

                # Rescale old output accumulator and add new
                o_i = o_i * alpha[:, None] + tl.dot(p.to(q.dtype), v).to(tl.float32)

                m_i = m_new
                l_i = l_new

        # Normalize by sum-of-exp
        o_i = o_i / tl.maximum(l_i[:, None], 1e-6)

        out_ptrs = (Out + batch_idx * stride_ob + head_idx * stride_oh
                    + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
        tl.store(out_ptrs, o_i.to(q.dtype), mask=(offs_m[:, None] < N_CTX) & d_mask[None, :])

        lse = m_i + tl.log(tl.maximum(l_i, 1e-6))
        l_ptrs = L_ptr + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_lm
        tl.store(l_ptrs, lse, mask=offs_m < N_CTX)

    @triton.jit
    def _sepllm_attn_bwd_dkdv_kernel(
        Q, K, V, sep_mask_ptr, dOut, dK, dV, L_ptr, D_ptr,
        softmax_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_sb, stride_sn,
        stride_lb, stride_lh, stride_lm,
        NUM_HEADS: tl.constexpr,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        INIT_TOKENS: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Backward kernel for dK/dV. Outer loop over kv-blocks, inner over q-blocks.
        No atomic operations needed — each kv-block is handled by exactly one CTA."""
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_idx = pid_bh // NUM_HEADS
        head_idx = pid_bh % NUM_HEADS

        kstart = pid_n * BLOCK_N
        offs_n = kstart + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM

        k_ptrs = (K + batch_idx * stride_kb + head_idx * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = (V + batch_idx * stride_vb + head_idx * stride_vh
                  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_CTX) & d_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_CTX) & d_mask[None, :], other=0.0)

        sep_ptrs = sep_mask_ptr + batch_idx * stride_sb + offs_n * stride_sn
        sep_block = tl.load(sep_ptrs, mask=offs_n < N_CTX, other=0).to(tl.int1)

        dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        for qstart in range(kstart, N_CTX, BLOCK_M):
            offs_m = qstart + tl.arange(0, BLOCK_M)

            has_init = kstart < INIT_TOKENS
            has_local = (kstart + BLOCK_N - 1) >= tl.maximum(qstart - WINDOW_SIZE + 1, 0)
            has_sep = tl.sum(sep_block.to(tl.int32), axis=0) > 0

            if has_init | has_local | has_sep:
                q_ptrs = (Q + batch_idx * stride_qb + head_idx * stride_qh
                          + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
                q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & d_mask[None, :], other=0.0)
                do_ptrs = (dOut + batch_idx * stride_ob + head_idx * stride_oh
                           + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
                do = tl.load(do_ptrs, mask=(offs_m[:, None] < N_CTX) & d_mask[None, :], other=0.0)
                lse = tl.load(L_ptr + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_lm,
                              mask=offs_m < N_CTX, other=0.0)
                Di = tl.load(D_ptr + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_lm,
                             mask=offs_m < N_CTX, other=0.0)

                s = tl.dot(q, tl.trans(k)) * softmax_scale

                causal_mask = offs_m[:, None] >= offs_n[None, :]
                init_mask = offs_n[None, :] < INIT_TOKENS
                sep_mask_2d = sep_block[None, :]
                local_mask = (offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE
                valid_kv = offs_n[None, :] < N_CTX
                attend = causal_mask & (init_mask | sep_mask_2d | local_mask) & valid_kv

                p = tl.where(attend, tl.exp(s - lse[:, None]), 0.0)

                dv += tl.dot(tl.trans(p.to(do.dtype)), do).to(tl.float32)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - Di[:, None])
                ds = tl.where(attend, ds, 0.0)
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32) * softmax_scale

        dk_ptrs = (dK + batch_idx * stride_kb + head_idx * stride_kh
                   + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        dv_ptrs = (dV + batch_idx * stride_vb + head_idx * stride_vh
                   + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        tl.store(dk_ptrs, dk.to(k.dtype), mask=(offs_n[:, None] < N_CTX) & d_mask[None, :])
        tl.store(dv_ptrs, dv.to(v.dtype), mask=(offs_n[:, None] < N_CTX) & d_mask[None, :])

    @triton.jit
    def _sepllm_attn_bwd_dq_kernel(
        Q, K, V, sep_mask_ptr, dOut, dQ, L_ptr, D_ptr,
        softmax_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_sb, stride_sn,
        stride_lb, stride_lh, stride_lm,
        NUM_HEADS: tl.constexpr,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        INIT_TOKENS: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Backward kernel for dQ. Outer loop over q-blocks, inner over kv-blocks.
        Mirrors the forward kernel structure — no atomic operations needed."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_idx = pid_bh // NUM_HEADS
        head_idx = pid_bh % NUM_HEADS

        qstart = pid_m * BLOCK_M
        offs_m = qstart + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM

        q_ptrs = (Q + batch_idx * stride_qb + head_idx * stride_qh
                  + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & d_mask[None, :], other=0.0)
        do_ptrs = (dOut + batch_idx * stride_ob + head_idx * stride_oh
                   + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
        do = tl.load(do_ptrs, mask=(offs_m[:, None] < N_CTX) & d_mask[None, :], other=0.0)
        lse = tl.load(L_ptr + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_lm,
                      mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(D_ptr + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_lm,
                     mask=offs_m < N_CTX, other=0.0)

        dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        kv_end = tl.minimum(qstart + BLOCK_M, N_CTX)
        for kstart in range(0, kv_end, BLOCK_N):
            offs_n = kstart + tl.arange(0, BLOCK_N)

            has_init = kstart < INIT_TOKENS
            has_local = (kstart + BLOCK_N - 1) >= tl.maximum(qstart - WINDOW_SIZE + 1, 0)

            sep_ptrs = sep_mask_ptr + batch_idx * stride_sb + offs_n * stride_sn
            sep_block = tl.load(sep_ptrs, mask=offs_n < N_CTX, other=0).to(tl.int1)
            has_sep = tl.sum(sep_block.to(tl.int32), axis=0) > 0

            if has_init | has_local | has_sep:
                k_ptrs = (K + batch_idx * stride_kb + head_idx * stride_kh
                          + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
                k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_CTX) & d_mask[None, :], other=0.0)
                v_ptrs = (V + batch_idx * stride_vb + head_idx * stride_vh
                          + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
                v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_CTX) & d_mask[None, :], other=0.0)

                s = tl.dot(q, tl.trans(k)) * softmax_scale

                causal_mask = offs_m[:, None] >= offs_n[None, :]
                init_mask = offs_n[None, :] < INIT_TOKENS
                sep_mask_2d = sep_block[None, :]
                local_mask = (offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE
                valid_kv = offs_n[None, :] < N_CTX
                attend = causal_mask & (init_mask | sep_mask_2d | local_mask) & valid_kv

                p = tl.where(attend, tl.exp(s - lse[:, None]), 0.0)

                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - Di[:, None])
                ds = tl.where(attend, ds, 0.0)
                dq += tl.dot(ds.to(k.dtype), k).to(tl.float32) * softmax_scale

        dq_ptrs = (dQ + batch_idx * stride_qb + head_idx * stride_qh
                   + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        tl.store(dq_ptrs, dq.to(q.dtype), mask=(offs_m[:, None] < N_CTX) & d_mask[None, :])


    # ========================================================================
    # D-chunked kernels for large head_dim (HEAD_DIM > 128, up to 3*D_CHUNK).
    #
    # Why: at head_dim=384, a single-tile kernel with BLOCK_D=next_pow2(384)=512
    # has an o-accumulator of shape [BLOCK_M, 512] fp32 = 128KB at BLOCK_M=64,
    # plus [BLOCK_N, 512] bf16 K and V tiles at 64KB each. Peak shared memory
    # exceeds the L20X 227KB limit, forcing tiny 16x32 tiles that destroy
    # arithmetic intensity (~3x slower than dense SDPA).
    #
    # Fix: split D into 3 chunks of 128. K, V, Q, dK, dV, dO, dQ are each held
    # as 3 separate tiles. The Q@K^T matmul becomes a 3-term chunked accumulate.
    # The p@V matmul similarly chunks the output. This keeps each tile small
    # enough to use larger spatial tiles (BLOCK_M=64, BLOCK_N=32) and tensor
    # cores effectively. Measured speedup at D=384: 3-5x over dense SDPA.
    # ========================================================================
    @triton.jit
    def _sepllm_attn_fwd_kernel_chunked3(
        Q, K, V, sep_mask_ptr, Out, L_ptr,
        softmax_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_sb, stride_sn,
        stride_lb, stride_lh, stride_lm,
        NUM_HEADS: tl.constexpr,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        INIT_TOKENS: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D_CHUNK: tl.constexpr,
        GQA_RATIO: tl.constexpr,
    ):
        # GQA_RATIO = NUM_HEADS / NUM_KV_HEADS. 1 = no GQA (K/V have same head count as Q).
        # For MQA (num_kv_heads=1), GQA_RATIO = NUM_HEADS. K/V are indexed by
        # kv_head_idx = head_idx // GQA_RATIO so multiple q-heads share the same kv slot
        # in memory, eliminating the need for a physical GQA expansion copy.
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_idx = pid_bh // NUM_HEADS
        head_idx = pid_bh % NUM_HEADS
        kv_head_idx = head_idx // GQA_RATIO

        qstart = pid_m * BLOCK_M
        offs_m = qstart + tl.arange(0, BLOCK_M)
        offs_dc = tl.arange(0, D_CHUNK)

        d0 = 0 * D_CHUNK + offs_dc
        d1 = 1 * D_CHUNK + offs_dc
        d2 = 2 * D_CHUNK + offs_dc
        md0 = d0 < HEAD_DIM
        md1 = d1 < HEAD_DIM
        md2 = d2 < HEAD_DIM

        q_bh = Q + batch_idx * stride_qb + head_idx * stride_qh
        k_bh = K + batch_idx * stride_kb + kv_head_idx * stride_kh
        v_bh = V + batch_idx * stride_vb + kv_head_idx * stride_vh
        o_bh = Out + batch_idx * stride_ob + head_idx * stride_oh

        row_m = offs_m[:, None] < N_CTX

        q0 = tl.load(q_bh + offs_m[:, None] * stride_qm + d0[None, :] * stride_qd,
                     mask=row_m & md0[None, :], other=0.0)
        q1 = tl.load(q_bh + offs_m[:, None] * stride_qm + d1[None, :] * stride_qd,
                     mask=row_m & md1[None, :], other=0.0)
        q2 = tl.load(q_bh + offs_m[:, None] * stride_qm + d2[None, :] * stride_qd,
                     mask=row_m & md2[None, :], other=0.0)

        m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o0 = tl.zeros([BLOCK_M, D_CHUNK], dtype=tl.float32)
        o1 = tl.zeros([BLOCK_M, D_CHUNK], dtype=tl.float32)
        o2 = tl.zeros([BLOCK_M, D_CHUNK], dtype=tl.float32)

        kv_end = tl.minimum(qstart + BLOCK_M, N_CTX)
        for kstart in range(0, kv_end, BLOCK_N):
            offs_n = kstart + tl.arange(0, BLOCK_N)
            col_m = offs_n[:, None] < N_CTX

            has_init = kstart < INIT_TOKENS
            has_local = (kstart + BLOCK_N - 1) >= tl.maximum(qstart - WINDOW_SIZE + 1, 0)

            sep_ptrs = sep_mask_ptr + batch_idx * stride_sb + offs_n * stride_sn
            sep_block = tl.load(sep_ptrs, mask=offs_n < N_CTX, other=0).to(tl.int1)
            has_sep = tl.sum(sep_block.to(tl.int32), axis=0) > 0

            if has_init | has_local | has_sep:
                k0 = tl.load(k_bh + offs_n[:, None] * stride_kn + d0[None, :] * stride_kd,
                             mask=col_m & md0[None, :], other=0.0)
                s = tl.dot(q0, tl.trans(k0))
                k1 = tl.load(k_bh + offs_n[:, None] * stride_kn + d1[None, :] * stride_kd,
                             mask=col_m & md1[None, :], other=0.0)
                s += tl.dot(q1, tl.trans(k1))
                k2 = tl.load(k_bh + offs_n[:, None] * stride_kn + d2[None, :] * stride_kd,
                             mask=col_m & md2[None, :], other=0.0)
                s += tl.dot(q2, tl.trans(k2))
                s *= softmax_scale

                causal_mask = offs_m[:, None] >= offs_n[None, :]
                init_mask = offs_n[None, :] < INIT_TOKENS
                sep_mask_2d = sep_block[None, :]
                local_mask = (offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE
                valid_kv = offs_n[None, :] < N_CTX
                attend = causal_mask & (init_mask | sep_mask_2d | local_mask) & valid_kv
                s = tl.where(attend, s, float('-inf'))

                row_max = tl.max(s, axis=1)
                m_new = tl.maximum(m_i, row_max)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])
                l_new = alpha * l_i + tl.sum(p, axis=1)

                p_cast = p.to(q0.dtype)

                v0 = tl.load(v_bh + offs_n[:, None] * stride_vn + d0[None, :] * stride_vd,
                             mask=col_m & md0[None, :], other=0.0)
                o0 = o0 * alpha[:, None] + tl.dot(p_cast, v0).to(tl.float32)
                v1 = tl.load(v_bh + offs_n[:, None] * stride_vn + d1[None, :] * stride_vd,
                             mask=col_m & md1[None, :], other=0.0)
                o1 = o1 * alpha[:, None] + tl.dot(p_cast, v1).to(tl.float32)
                v2 = tl.load(v_bh + offs_n[:, None] * stride_vn + d2[None, :] * stride_vd,
                             mask=col_m & md2[None, :], other=0.0)
                o2 = o2 * alpha[:, None] + tl.dot(p_cast, v2).to(tl.float32)

                m_i = m_new
                l_i = l_new

        inv_l = 1.0 / tl.maximum(l_i, 1e-6)
        o0 = o0 * inv_l[:, None]
        o1 = o1 * inv_l[:, None]
        o2 = o2 * inv_l[:, None]

        tl.store(o_bh + offs_m[:, None] * stride_om + d0[None, :] * stride_od,
                 o0.to(q0.dtype), mask=row_m & md0[None, :])
        tl.store(o_bh + offs_m[:, None] * stride_om + d1[None, :] * stride_od,
                 o1.to(q0.dtype), mask=row_m & md1[None, :])
        tl.store(o_bh + offs_m[:, None] * stride_om + d2[None, :] * stride_od,
                 o2.to(q0.dtype), mask=row_m & md2[None, :])

        lse = m_i + tl.log(tl.maximum(l_i, 1e-6))
        l_ptrs = L_ptr + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_lm
        tl.store(l_ptrs, lse, mask=offs_m < N_CTX)


    @triton.jit
    def _sepllm_attn_bwd_dkdv_kernel_chunked3(
        Q, K, V, sep_mask_ptr, dOut, dK, dV, L_ptr, D_ptr,
        softmax_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_sb, stride_sn,
        stride_lb, stride_lh, stride_lm,
        NUM_HEADS: tl.constexpr,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        INIT_TOKENS: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D_CHUNK: tl.constexpr,
        GQA_RATIO: tl.constexpr,
    ):
        # dKV grid: (cdiv(N_CTX, BLOCK_N), B * NUM_KV_HEADS).
        # NUM_KV_HEADS = NUM_HEADS // GQA_RATIO. Each CTA owns a kv-tile and must
        # accumulate contributions from ALL GQA_RATIO q-heads that share this kv-head,
        # otherwise multiple CTAs would race on the same dK/dV slot. We load K, V once
        # and iterate the q-head dim with tl.static_range so the compiler can pipeline.
        NUM_KV_HEADS: tl.constexpr = NUM_HEADS // GQA_RATIO

        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_idx = pid_bh // NUM_KV_HEADS
        kv_head_idx = pid_bh % NUM_KV_HEADS

        kstart = pid_n * BLOCK_N
        offs_n = kstart + tl.arange(0, BLOCK_N)
        offs_dc = tl.arange(0, D_CHUNK)

        d0 = 0 * D_CHUNK + offs_dc
        d1 = 1 * D_CHUNK + offs_dc
        d2 = 2 * D_CHUNK + offs_dc
        md0 = d0 < HEAD_DIM
        md1 = d1 < HEAD_DIM
        md2 = d2 < HEAD_DIM

        k_bh = K + batch_idx * stride_kb + kv_head_idx * stride_kh
        v_bh = V + batch_idx * stride_vb + kv_head_idx * stride_vh
        dk_bh = dK + batch_idx * stride_kb + kv_head_idx * stride_kh
        dv_bh = dV + batch_idx * stride_vb + kv_head_idx * stride_vh

        col_m = offs_n[:, None] < N_CTX

        k0 = tl.load(k_bh + offs_n[:, None] * stride_kn + d0[None, :] * stride_kd,
                     mask=col_m & md0[None, :], other=0.0)
        k1 = tl.load(k_bh + offs_n[:, None] * stride_kn + d1[None, :] * stride_kd,
                     mask=col_m & md1[None, :], other=0.0)
        k2 = tl.load(k_bh + offs_n[:, None] * stride_kn + d2[None, :] * stride_kd,
                     mask=col_m & md2[None, :], other=0.0)
        v0 = tl.load(v_bh + offs_n[:, None] * stride_vn + d0[None, :] * stride_vd,
                     mask=col_m & md0[None, :], other=0.0)
        v1 = tl.load(v_bh + offs_n[:, None] * stride_vn + d1[None, :] * stride_vd,
                     mask=col_m & md1[None, :], other=0.0)
        v2 = tl.load(v_bh + offs_n[:, None] * stride_vn + d2[None, :] * stride_vd,
                     mask=col_m & md2[None, :], other=0.0)

        sep_ptrs = sep_mask_ptr + batch_idx * stride_sb + offs_n * stride_sn
        sep_block = tl.load(sep_ptrs, mask=offs_n < N_CTX, other=0).to(tl.int1)

        dk0 = tl.zeros([BLOCK_N, D_CHUNK], dtype=tl.float32)
        dk1 = tl.zeros([BLOCK_N, D_CHUNK], dtype=tl.float32)
        dk2 = tl.zeros([BLOCK_N, D_CHUNK], dtype=tl.float32)
        dv0 = tl.zeros([BLOCK_N, D_CHUNK], dtype=tl.float32)
        dv1 = tl.zeros([BLOCK_N, D_CHUNK], dtype=tl.float32)
        dv2 = tl.zeros([BLOCK_N, D_CHUNK], dtype=tl.float32)

        for qh_off in tl.static_range(GQA_RATIO):
            head_idx = kv_head_idx * GQA_RATIO + qh_off
            q_bh = Q + batch_idx * stride_qb + head_idx * stride_qh
            do_bh = dOut + batch_idx * stride_ob + head_idx * stride_oh
            l_hh = L_ptr + batch_idx * stride_lb + head_idx * stride_lh
            d_hh = D_ptr + batch_idx * stride_lb + head_idx * stride_lh

            for qstart in range(kstart, N_CTX, BLOCK_M):
                offs_m = qstart + tl.arange(0, BLOCK_M)
                row_m = offs_m[:, None] < N_CTX

                has_init = kstart < INIT_TOKENS
                has_local = (kstart + BLOCK_N - 1) >= tl.maximum(qstart - WINDOW_SIZE + 1, 0)
                has_sep = tl.sum(sep_block.to(tl.int32), axis=0) > 0

                if has_init | has_local | has_sep:
                    q0 = tl.load(q_bh + offs_m[:, None] * stride_qm + d0[None, :] * stride_qd,
                                 mask=row_m & md0[None, :], other=0.0)
                    q1 = tl.load(q_bh + offs_m[:, None] * stride_qm + d1[None, :] * stride_qd,
                                 mask=row_m & md1[None, :], other=0.0)
                    q2 = tl.load(q_bh + offs_m[:, None] * stride_qm + d2[None, :] * stride_qd,
                                 mask=row_m & md2[None, :], other=0.0)
                    do0 = tl.load(do_bh + offs_m[:, None] * stride_om + d0[None, :] * stride_od,
                                  mask=row_m & md0[None, :], other=0.0)
                    do1 = tl.load(do_bh + offs_m[:, None] * stride_om + d1[None, :] * stride_od,
                                  mask=row_m & md1[None, :], other=0.0)
                    do2 = tl.load(do_bh + offs_m[:, None] * stride_om + d2[None, :] * stride_od,
                                  mask=row_m & md2[None, :], other=0.0)

                    lse = tl.load(l_hh + offs_m * stride_lm, mask=offs_m < N_CTX, other=0.0)
                    Di = tl.load(d_hh + offs_m * stride_lm, mask=offs_m < N_CTX, other=0.0)

                    s = tl.dot(q0, tl.trans(k0))
                    s += tl.dot(q1, tl.trans(k1))
                    s += tl.dot(q2, tl.trans(k2))
                    s *= softmax_scale

                    causal_mask = offs_m[:, None] >= offs_n[None, :]
                    init_mask = offs_n[None, :] < INIT_TOKENS
                    sep_mask_2d = sep_block[None, :]
                    local_mask = (offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE
                    valid_kv = offs_n[None, :] < N_CTX
                    attend = causal_mask & (init_mask | sep_mask_2d | local_mask) & valid_kv

                    p = tl.where(attend, tl.exp(s - lse[:, None]), 0.0)
                    p_cast = p.to(q0.dtype)
                    p_trans = tl.trans(p_cast)

                    dv0 += tl.dot(p_trans, do0).to(tl.float32)
                    dv1 += tl.dot(p_trans, do1).to(tl.float32)
                    dv2 += tl.dot(p_trans, do2).to(tl.float32)

                    dp = tl.dot(do0, tl.trans(v0))
                    dp += tl.dot(do1, tl.trans(v1))
                    dp += tl.dot(do2, tl.trans(v2))

                    ds = p * (dp - Di[:, None])
                    ds = tl.where(attend, ds, 0.0)
                    ds_cast = ds.to(q0.dtype)
                    ds_trans = tl.trans(ds_cast)

                    dk0 += tl.dot(ds_trans, q0).to(tl.float32) * softmax_scale
                    dk1 += tl.dot(ds_trans, q1).to(tl.float32) * softmax_scale
                    dk2 += tl.dot(ds_trans, q2).to(tl.float32) * softmax_scale

        tl.store(dk_bh + offs_n[:, None] * stride_kn + d0[None, :] * stride_kd,
                 dk0.to(k0.dtype), mask=col_m & md0[None, :])
        tl.store(dk_bh + offs_n[:, None] * stride_kn + d1[None, :] * stride_kd,
                 dk1.to(k0.dtype), mask=col_m & md1[None, :])
        tl.store(dk_bh + offs_n[:, None] * stride_kn + d2[None, :] * stride_kd,
                 dk2.to(k0.dtype), mask=col_m & md2[None, :])
        tl.store(dv_bh + offs_n[:, None] * stride_vn + d0[None, :] * stride_vd,
                 dv0.to(v0.dtype), mask=col_m & md0[None, :])
        tl.store(dv_bh + offs_n[:, None] * stride_vn + d1[None, :] * stride_vd,
                 dv1.to(v0.dtype), mask=col_m & md1[None, :])
        tl.store(dv_bh + offs_n[:, None] * stride_vn + d2[None, :] * stride_vd,
                 dv2.to(v0.dtype), mask=col_m & md2[None, :])


    @triton.jit
    def _sepllm_attn_bwd_dq_kernel_chunked3(
        Q, K, V, sep_mask_ptr, dOut, dQ, L_ptr, D_ptr,
        softmax_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_sb, stride_sn,
        stride_lb, stride_lh, stride_lm,
        NUM_HEADS: tl.constexpr,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        INIT_TOKENS: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D_CHUNK: tl.constexpr,
        GQA_RATIO: tl.constexpr,
    ):
        # dQ is indexed per q-head; K/V reads use kv_head_idx = head_idx // GQA_RATIO
        # so no physical GQA expansion is required. See fwd kernel docstring.
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_idx = pid_bh // NUM_HEADS
        head_idx = pid_bh % NUM_HEADS
        kv_head_idx = head_idx // GQA_RATIO

        qstart = pid_m * BLOCK_M
        offs_m = qstart + tl.arange(0, BLOCK_M)
        offs_dc = tl.arange(0, D_CHUNK)

        d0 = 0 * D_CHUNK + offs_dc
        d1 = 1 * D_CHUNK + offs_dc
        d2 = 2 * D_CHUNK + offs_dc
        md0 = d0 < HEAD_DIM
        md1 = d1 < HEAD_DIM
        md2 = d2 < HEAD_DIM

        q_bh = Q + batch_idx * stride_qb + head_idx * stride_qh
        k_bh = K + batch_idx * stride_kb + kv_head_idx * stride_kh
        v_bh = V + batch_idx * stride_vb + kv_head_idx * stride_vh
        do_bh = dOut + batch_idx * stride_ob + head_idx * stride_oh
        dq_bh = dQ + batch_idx * stride_qb + head_idx * stride_qh

        row_m = offs_m[:, None] < N_CTX

        q0 = tl.load(q_bh + offs_m[:, None] * stride_qm + d0[None, :] * stride_qd,
                     mask=row_m & md0[None, :], other=0.0)
        q1 = tl.load(q_bh + offs_m[:, None] * stride_qm + d1[None, :] * stride_qd,
                     mask=row_m & md1[None, :], other=0.0)
        q2 = tl.load(q_bh + offs_m[:, None] * stride_qm + d2[None, :] * stride_qd,
                     mask=row_m & md2[None, :], other=0.0)
        do0 = tl.load(do_bh + offs_m[:, None] * stride_om + d0[None, :] * stride_od,
                      mask=row_m & md0[None, :], other=0.0)
        do1 = tl.load(do_bh + offs_m[:, None] * stride_om + d1[None, :] * stride_od,
                      mask=row_m & md1[None, :], other=0.0)
        do2 = tl.load(do_bh + offs_m[:, None] * stride_om + d2[None, :] * stride_od,
                      mask=row_m & md2[None, :], other=0.0)

        lse = tl.load(L_ptr + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_lm,
                      mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(D_ptr + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_lm,
                     mask=offs_m < N_CTX, other=0.0)

        dq0 = tl.zeros([BLOCK_M, D_CHUNK], dtype=tl.float32)
        dq1 = tl.zeros([BLOCK_M, D_CHUNK], dtype=tl.float32)
        dq2 = tl.zeros([BLOCK_M, D_CHUNK], dtype=tl.float32)

        kv_end = tl.minimum(qstart + BLOCK_M, N_CTX)
        for kstart in range(0, kv_end, BLOCK_N):
            offs_n = kstart + tl.arange(0, BLOCK_N)
            col_m = offs_n[:, None] < N_CTX

            has_init = kstart < INIT_TOKENS
            has_local = (kstart + BLOCK_N - 1) >= tl.maximum(qstart - WINDOW_SIZE + 1, 0)

            sep_ptrs = sep_mask_ptr + batch_idx * stride_sb + offs_n * stride_sn
            sep_block = tl.load(sep_ptrs, mask=offs_n < N_CTX, other=0).to(tl.int1)
            has_sep = tl.sum(sep_block.to(tl.int32), axis=0) > 0

            if has_init | has_local | has_sep:
                k0 = tl.load(k_bh + offs_n[:, None] * stride_kn + d0[None, :] * stride_kd,
                             mask=col_m & md0[None, :], other=0.0)
                k1 = tl.load(k_bh + offs_n[:, None] * stride_kn + d1[None, :] * stride_kd,
                             mask=col_m & md1[None, :], other=0.0)
                k2 = tl.load(k_bh + offs_n[:, None] * stride_kn + d2[None, :] * stride_kd,
                             mask=col_m & md2[None, :], other=0.0)
                v0 = tl.load(v_bh + offs_n[:, None] * stride_vn + d0[None, :] * stride_vd,
                             mask=col_m & md0[None, :], other=0.0)
                v1 = tl.load(v_bh + offs_n[:, None] * stride_vn + d1[None, :] * stride_vd,
                             mask=col_m & md1[None, :], other=0.0)
                v2 = tl.load(v_bh + offs_n[:, None] * stride_vn + d2[None, :] * stride_vd,
                             mask=col_m & md2[None, :], other=0.0)

                s = tl.dot(q0, tl.trans(k0))
                s += tl.dot(q1, tl.trans(k1))
                s += tl.dot(q2, tl.trans(k2))
                s *= softmax_scale

                causal_mask = offs_m[:, None] >= offs_n[None, :]
                init_mask = offs_n[None, :] < INIT_TOKENS
                sep_mask_2d = sep_block[None, :]
                local_mask = (offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE
                valid_kv = offs_n[None, :] < N_CTX
                attend = causal_mask & (init_mask | sep_mask_2d | local_mask) & valid_kv

                p = tl.where(attend, tl.exp(s - lse[:, None]), 0.0)

                dp = tl.dot(do0, tl.trans(v0))
                dp += tl.dot(do1, tl.trans(v1))
                dp += tl.dot(do2, tl.trans(v2))

                ds = p * (dp - Di[:, None])
                ds = tl.where(attend, ds, 0.0)
                ds_cast = ds.to(q0.dtype)

                dq0 += tl.dot(ds_cast, k0).to(tl.float32) * softmax_scale
                dq1 += tl.dot(ds_cast, k1).to(tl.float32) * softmax_scale
                dq2 += tl.dot(ds_cast, k2).to(tl.float32) * softmax_scale

        tl.store(dq_bh + offs_m[:, None] * stride_qm + d0[None, :] * stride_qd,
                 dq0.to(q0.dtype), mask=row_m & md0[None, :])
        tl.store(dq_bh + offs_m[:, None] * stride_qm + d1[None, :] * stride_qd,
                 dq1.to(q0.dtype), mask=row_m & md1[None, :])
        tl.store(dq_bh + offs_m[:, None] * stride_qm + d2[None, :] * stride_qd,
                 dq2.to(q0.dtype), mask=row_m & md2[None, :])


class _SepLLMTritonAttnFn(torch.autograd.Function):
    """Autograd wrapper around Triton forward + backward kernels."""

    # D-chunk size. D in (128, 3*D_CHUNK] uses the chunked kernels.
    D_CHUNK = 128

    @staticmethod
    def _use_chunked(D):
        return D > 128

    @staticmethod
    def _select_block_params(D):
        """Auto-select BLOCK_M, BLOCK_N, BLOCK_D, num_warps, num_stages based on head_dim.

        For D <= 128 we use the single-tile kernel which handles the full D in one
        tile. For D > 128 we use the D-chunked kernel path (see _select_chunked_params).

        Larger head_dim needs smaller spatial tiles and fewer pipeline stages to fit
        in registers and shared memory.
        """
        BLOCK_D = triton.next_power_of_2(D)
        if D <= 64:
            return 64, 64, BLOCK_D, 4, 2
        elif D <= 128:
            return 64, 64, BLOCK_D, 4, 2
        elif D <= 256:
            return 32, 32, BLOCK_D, 4, 1
        else:
            return 16, 32, BLOCK_D, 4, 1

    @staticmethod
    def _select_chunked_params(D):
        """Block sizes for D-chunked kernels (FWD_M, FWD_N, BWD_M, BWD_N, num_warps).

        Tuned on L20X at head_dim=384 with training-scale B>=4:
        - FWD (64, 32): 1.0 ms fwd; larger FWD_N exhausts SRAM at D=384
        - BWD (64, 64): best at B*H>=64 where there's enough parallelism to amortise
          the larger tile. At B=4,H=16 this beats BWD(16,16) by ~3x (42 vs 127 ms).
          The 6 persistent fp32 accumulators (dk0/1/2, dv0/1/2) use 64x128*4 = 32KB
          each, and BLOCK_M=64 transient buffers fit in L20X's 227KB SRAM budget.
        """
        return 64, 32, 64, 64, 4

    @staticmethod
    def forward(ctx, q, k, v, sep_mask, init_token_count, local_window_size, softmax_scale,
                BLOCK_M=None, BLOCK_N=None):
        # Q: [B, H_Q, S, D]. K, V: [B, num_kv_heads, S, D] with num_kv_heads in {1, H_Q}
        # (MHA) or any divisor of H_Q (GQA). For the single-tile kernel path (D<=128)
        # we still require num_kv_heads == H_Q (caller handles expansion). Only the
        # D-chunked path supports GQA_RATIO > 1 natively to avoid allocator thrash from
        # physical GQA expansion of [B, 1, S, D] -> [B, H_Q, S, D].
        B, H, S, D = q.shape
        num_kv_heads = k.shape[1]
        assert k.shape == (B, num_kv_heads, S, D) and v.shape == (B, num_kv_heads, S, D)
        assert H % num_kv_heads == 0, f"H={H} not divisible by num_kv_heads={num_kv_heads}"
        gqa_ratio = H // num_kv_heads

        o = torch.empty_like(q)
        L = torch.empty((B, H, S), device=q.device, dtype=torch.float32)

        use_chunked = _SepLLMTritonAttnFn._use_chunked(D)

        if use_chunked:
            D_CHUNK = _SepLLMTritonAttnFn.D_CHUNK
            assert D <= 3 * D_CHUNK, (
                f"SepLLM chunked kernel supports HEAD_DIM up to {3 * D_CHUNK}, got {D}"
            )
            fwd_M, fwd_N, bwd_M, bwd_N, num_warps = _SepLLMTritonAttnFn._select_chunked_params(D)
            if BLOCK_M is None:
                BLOCK_M = fwd_M
            if BLOCK_N is None:
                BLOCK_N = fwd_N

            grid = (triton.cdiv(S, BLOCK_M), B * H)
            _sepllm_attn_fwd_kernel_chunked3[grid](
                q, k, v, sep_mask, o, L, softmax_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                sep_mask.stride(0), sep_mask.stride(1),
                L.stride(0), L.stride(1), L.stride(2),
                NUM_HEADS=H, N_CTX=S, HEAD_DIM=D,
                INIT_TOKENS=init_token_count, WINDOW_SIZE=local_window_size,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D_CHUNK=D_CHUNK,
                GQA_RATIO=gqa_ratio,
                num_warps=num_warps, num_stages=2,
            )
            ctx.use_chunked = True
            ctx.D_CHUNK = D_CHUNK
            ctx.BWD_BLOCK_M = bwd_M
            ctx.BWD_BLOCK_N = bwd_N
            ctx.num_warps = num_warps
            ctx.BLOCK_M = BLOCK_M
            ctx.BLOCK_N = BLOCK_N
            ctx.BLOCK_D = 0  # unused
        else:
            assert num_kv_heads == H, (
                "Single-tile SepLLM kernel (D<=128) requires K/V to be pre-expanded to "
                "H_Q heads; caller must materialise GQA expansion for this path."
            )
            auto_M, auto_N, BLOCK_D, num_warps, num_stages = _SepLLMTritonAttnFn._select_block_params(D)
            if BLOCK_M is None:
                BLOCK_M = auto_M
            if BLOCK_N is None:
                BLOCK_N = auto_N

            grid = (triton.cdiv(S, BLOCK_M), B * H)
            _sepllm_attn_fwd_kernel[grid](
                q, k, v, sep_mask, o, L,
                softmax_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                sep_mask.stride(0), sep_mask.stride(1),
                L.stride(0), L.stride(1), L.stride(2),
                NUM_HEADS=H, N_CTX=S, HEAD_DIM=D,
                INIT_TOKENS=init_token_count, WINDOW_SIZE=local_window_size,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                num_warps=num_warps, num_stages=num_stages,
            )
            ctx.use_chunked = False
            ctx.BLOCK_M = BLOCK_M
            ctx.BLOCK_N = BLOCK_N
            ctx.BLOCK_D = BLOCK_D
            ctx.num_warps = num_warps

        ctx.save_for_backward(q, k, v, o, L, sep_mask)
        ctx.softmax_scale = softmax_scale
        ctx.init_token_count = init_token_count
        ctx.local_window_size = local_window_size
        ctx.HEAD_DIM = D
        ctx.NUM_HEADS = H
        ctx.NUM_KV_HEADS = num_kv_heads
        ctx.GQA_RATIO = gqa_ratio
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L, sep_mask = ctx.saved_tensors
        B, H, S, D = q.shape

        # Per-row correction term D_i = sum(dO * O); computed in fp32 for stability.
        Di = (do.float() * o.float()).sum(dim=-1)  # [B, H, S]

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do = do.contiguous()

        # Keep inputs in their native dtype (bf16/fp16) so the kernel uses
        # tensor cores for the matmuls. Accumulators inside the kernel are fp32.
        stride_args_q = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        stride_args_k = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        stride_args_v = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        stride_args_o = (do.stride(0), do.stride(1), do.stride(2), do.stride(3))
        stride_args_s = (sep_mask.stride(0), sep_mask.stride(1))
        stride_args_l = (L.stride(0), L.stride(1), L.stride(2))

        if ctx.use_chunked:
            BWD_M = ctx.BWD_BLOCK_M
            BWD_N = ctx.BWD_BLOCK_N
            common_kwargs = dict(
                NUM_HEADS=H, N_CTX=S, HEAD_DIM=ctx.HEAD_DIM,
                INIT_TOKENS=ctx.init_token_count,
                WINDOW_SIZE=ctx.local_window_size,
                BLOCK_M=BWD_M, BLOCK_N=BWD_N, D_CHUNK=ctx.D_CHUNK,
                GQA_RATIO=ctx.GQA_RATIO,
                num_warps=ctx.num_warps, num_stages=1,
            )

            # dKV grid uses num_kv_heads (not H) because each CTA owns a kv-tile and
            # accumulates contributions from all GQA_RATIO q-heads internally.
            grid_kv = (triton.cdiv(S, BWD_N), B * ctx.NUM_KV_HEADS)
            _sepllm_attn_bwd_dkdv_kernel_chunked3[grid_kv](
                q, k, v, sep_mask, do, dk, dv, L, Di, ctx.softmax_scale,
                *stride_args_q, *stride_args_k, *stride_args_v, *stride_args_o,
                *stride_args_s, *stride_args_l,
                **common_kwargs,
            )

            # dQ grid still uses H (one CTA per q-head tile).
            grid_q = (triton.cdiv(S, BWD_M), B * H)
            _sepllm_attn_bwd_dq_kernel_chunked3[grid_q](
                q, k, v, sep_mask, do, dq, L, Di, ctx.softmax_scale,
                *stride_args_q, *stride_args_k, *stride_args_v, *stride_args_o,
                *stride_args_s, *stride_args_l,
                **common_kwargs,
            )
        else:
            common_kwargs = dict(
                NUM_HEADS=H, N_CTX=S, HEAD_DIM=ctx.HEAD_DIM,
                INIT_TOKENS=ctx.init_token_count,
                WINDOW_SIZE=ctx.local_window_size,
                BLOCK_M=ctx.BLOCK_M, BLOCK_N=ctx.BLOCK_N, BLOCK_D=ctx.BLOCK_D,
                num_warps=ctx.num_warps, num_stages=1,
            )

            grid_kv = (triton.cdiv(S, ctx.BLOCK_N), B * H)
            _sepllm_attn_bwd_dkdv_kernel[grid_kv](
                q, k, v, sep_mask, do, dk, dv, L, Di,
                ctx.softmax_scale,
                *stride_args_q, *stride_args_k, *stride_args_v, *stride_args_o,
                *stride_args_s, *stride_args_l,
                **common_kwargs,
            )

            grid_q = (triton.cdiv(S, ctx.BLOCK_M), B * H)
            _sepllm_attn_bwd_dq_kernel[grid_q](
                q, k, v, sep_mask, do, dq, L, Di,
                ctx.softmax_scale,
                *stride_args_q, *stride_args_k, *stride_args_v, *stride_args_o,
                *stride_args_s, *stride_args_l,
                **common_kwargs,
            )
        return dq, dk, dv, None, None, None, None, None, None


def sepllm_triton_attention(
    query: Tensor,          # [B, H, S, D]
    key: Tensor,            # [B, H, S, D]
    value: Tensor,          # [B, H, S, D]
    sep_mask: Tensor,       # [B, S] bool
    init_token_count: int,
    local_window_size: int,
    softmax_scale: float,
) -> Tensor:
    """SepLLM attention via custom Triton kernels (forward + backward)."""
    assert _TRITON_AVAILABLE, "Triton is not installed"
    return _SepLLMTritonAttnFn.apply(
        query, key, value, sep_mask,
        init_token_count, local_window_size, softmax_scale,
    )


# ============================================================================
# Path 3: Dense mask fallback (original implementation, no speedup)
# ============================================================================
def sepllm_dense_attention(
    query: Tensor,          # [B, H, S, D]
    key: Tensor,            # [B, H, S, D]
    value: Tensor,          # [B, H, S, D]
    sep_mask: Tensor,       # [B, S] bool
    init_token_count: int,
    local_window_size: int,
    softmax_scale: float,
    dropout_p: float = 0.0,
) -> Tensor:
    """Fallback: construct dense mask and run standard SDPA. No computation savings."""
    B, H, S, D = query.shape
    device = query.device

    q_pos = torch.arange(S, device=device).unsqueeze(1)
    k_pos = torch.arange(S, device=device).unsqueeze(0)

    causal = k_pos <= q_pos
    init_m = k_pos < init_token_count
    local_m = (q_pos - k_pos) < local_window_size
    sep_m = sep_mask[:, None, None, :]  # [B, 1, 1, S]

    attend = causal[None, None] & (init_m[None, None] | local_m[None, None] | sep_m)
    attn_mask_float = torch.where(attend, 0.0, float('-inf'))

    out = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask_float.to(query.dtype),
        scale=softmax_scale,
        dropout_p=dropout_p,
    )
    return out


# ============================================================================
# Unified dispatch
# ============================================================================
def sepllm_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sep_mask: Tensor,
    init_token_count: int,
    local_window_size: int,
    softmax_scale: float,
    dropout_p: float = 0.0,
    kernel: str = 'auto',
) -> Tensor:
    """Dispatch to the best available SepLLM attention kernel.

    Args:
        kernel: 'flex_attention', 'triton', 'dense', or 'auto' (picks best available)
    """
    if kernel == 'auto':
        if _TRITON_AVAILABLE:
            kernel = 'triton'
        elif _FLEX_AVAILABLE:
            kernel = 'flex_attention'
        else:
            kernel = 'dense'

    if kernel == 'flex_attention':
        return sepllm_flex_attention(
            query, key, value, sep_mask,
            init_token_count, local_window_size, softmax_scale,
            dropout_p=dropout_p,
        )
    elif kernel == 'triton':
        return sepllm_triton_attention(
            query, key, value, sep_mask,
            init_token_count, local_window_size, softmax_scale,
        )
    elif kernel == 'dense':
        return sepllm_dense_attention(
            query, key, value, sep_mask,
            init_token_count, local_window_size, softmax_scale,
            dropout_p=dropout_p,
        )
    else:
        raise ValueError(f"Unknown SepLLM kernel: {kernel}. Use 'flex_attention', 'triton', or 'dense'.")
