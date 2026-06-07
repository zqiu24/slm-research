import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from typing import Optional
from .poet_ops import *

import numpy as np
import math
from tqdm import tqdm
import gc
import os
import sys
import logging

logger = logging.getLogger(__name__)

def permute_x(x, perm, inv_perm):
    return PermutationFunction.apply(x, perm, inv_perm)

def chain_layer_x_checkpoint_mem_o2(x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], Rout: torch.Tensor,
                                    perm_in_inv: torch.Tensor, perm_in: torch.Tensor, perm_out: torch.Tensor, perm_out_inv: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_mem_o2(x, Rin, weight, bias, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv, block_size)

def chain_layer_x_checkpoint_mem_o2_decoupled(x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], Rout: torch.Tensor,
                                              perm_in_inv: torch.Tensor, perm_in: torch.Tensor, perm_out: torch.Tensor, perm_out_inv: torch.Tensor,
                                              block_size_in: int, block_size_out: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_mem_o2_decoupled(x, Rin, weight, bias, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv, block_size_in, block_size_out)

def chain_layer_x_checkpoint(x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], Rout: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint(x, Rin, weight, bias, Rout, block_size)

def chain_layer_x_checkpoint_q8(x: torch.Tensor, Rin: torch.Tensor, W_q: torch.Tensor, W_scales: torch.Tensor, W_zeros: torch.Tensor, group_size: int, b: Optional[torch.Tensor], Rout: torch.Tensor, bsz: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_q8(x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, bsz)

def chain_layer_x_checkpoint_mem_o2_q8(x: torch.Tensor, Rin: torch.Tensor, W_q: torch.Tensor, W_scales: torch.Tensor, W_zeros: torch.Tensor, group_size: int, b: Optional[torch.Tensor], Rout: torch.Tensor, perm_in_inv: torch.Tensor, perm_in: torch.Tensor, perm_out: torch.Tensor, perm_out_inv: torch.Tensor, bsz: int) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_mem_o2_q8(x, Rin, W_q, W_scales, W_zeros, group_size, b, Rout, perm_in_inv, perm_in, perm_out, perm_out_inv, bsz)

def chain_layer_x_checkpoint_4bit(
    x: torch.Tensor,
    Rin: torch.Tensor,
    W_4bit: torch.Tensor,
    # quant_state main components:
    qs_absmax: torch.Tensor,
    qs_code: torch.Tensor,
    qs_blocksize: int,
    qs_quant_type: str,
    qs_shape_0: int,
    qs_shape_1: int,
    # quant_state.state2 components (double-quant):
    qs2_absmax: torch.Tensor,
    qs2_code: torch.Tensor,
    qs2_blocksize: int,
    qs_offset: torch.Tensor,
    b: Optional[torch.Tensor],
    Rout: torch.Tensor,
    bsz: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_4bit(
        x, Rin, W_4bit,
        qs_absmax, qs_code, qs_blocksize, qs_quant_type, qs_shape_0, qs_shape_1,
        qs2_absmax, qs2_code, qs2_blocksize, qs_offset,
        b, Rout, bsz, out_features, in_features
    )

def chain_layer_x_checkpoint_mem_o2_4bit(
    x: torch.Tensor,
    Rin: torch.Tensor,
    W_4bit: torch.Tensor,
    # quant_state main components:
    qs_absmax: torch.Tensor,
    qs_code: torch.Tensor,
    qs_blocksize: int,
    qs_quant_type: str,
    qs_shape_0: int,
    qs_shape_1: int,
    # quant_state.state2 components (double-quant):
    qs2_absmax: torch.Tensor,
    qs2_code: torch.Tensor,
    qs2_blocksize: int,
    qs_offset: torch.Tensor,
    b: Optional[torch.Tensor],
    Rout: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_in: torch.Tensor,
    bsz: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return torch.ops.poet.chain_layer_checkpoint_mem_o2_4bit(
        x, Rin, W_4bit,
        qs_absmax, qs_code, qs_blocksize, qs_quant_type, qs_shape_0, qs_shape_1,
        qs2_absmax, qs2_code, qs2_blocksize, qs_offset,
        b, Rout, perm_in_inv, perm_in, bsz, out_features, in_features
    )


def _quantize_tensor_int8(w, q_group_size=-1, n_bit=8):
    # print(f"Quantizing tensor on device: {w.device}")  # ADD THIS
    org_w_shape = w.shape
    if q_group_size > 0:
        assert w.nelement() % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    # assert torch.isnan(scales).sum() == 0
    # assert torch.isnan(w).sum() == 0

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = w.reshape(org_w_shape).to(torch.uint8)

    return w, scales, zeros


def block_diag_lr_matmul(A_blocks: torch.Tensor, W: torch.Tensor, B_blocks: torch.Tensor) -> torch.Tensor:
    """
    Compute (block_diag(A_blocks) @ W @ block_diag(B_blocks)) without materializing block-diagonal matrices.

    Args:
      A_blocks: (r_m, b, b) block-diagonal factors for the left (M = r_m * b)
      W:        (M, N) matrix to multiply, where M = r_m * b, N = r_n * b
      B_blocks: (r_n, b, b) block-diagonal factors for the right (N = r_n * b)

    Returns:
      Tensor of shape (M, N)
    """
    if A_blocks.ndim != 3 or B_blocks.ndim != 3:
        raise ValueError("A_blocks and B_blocks must be 3D: (r, b, b)")
    r_m, b1, b2 = A_blocks.shape
    r_n, b3, b4 = B_blocks.shape
    if not (b1 == b2 == b3 == b4):
        raise ValueError("All block sizes must match and be square b x b.")
    b = b1
    M = r_m * b
    N = r_n * b
    if W.shape != (M, N):
        raise ValueError(f"W must have shape {(M, N)}, got {tuple(W.shape)}")

    # Ensure device/dtype compatibility (keeps things simple and safe)
    if A_blocks.device != W.device or A_blocks.dtype != W.dtype:
        A_blocks = A_blocks.to(device=W.device, dtype=W.dtype)
    if B_blocks.device != W.device or B_blocks.dtype != W.dtype:
        B_blocks = B_blocks.to(device=W.device, dtype=W.dtype)

    # Reshape W into blocks and apply batched matmuls:
    # W_ = (r_m, r_n, b, b), where W_[i, j] is the (i, j) b x b block of W
    W_blocks = W.view(r_m, b, r_n, b).transpose(1, 2)  # (r_m, r_n, b, b)

    # Left multiply each block-row by corresponding A_blocks[i]
    # Shapes: (r_m, 1, b, b) @ (r_m, r_n, b, b) -> (r_m, r_n, b, b)
    left = torch.matmul(A_blocks.unsqueeze(1), W_blocks)

    # Right multiply each block-col by corresponding B_blocks[j]
    # Shapes: (r_m, r_n, b, b) @ (1, r_n, b, b) -> (r_m, r_n, b, b)
    out_blocks = torch.matmul(left, B_blocks.unsqueeze(0))

    # Fold back to (M, N)
    out = out_blocks.permute(0, 2, 1, 3).contiguous().view(M, N)
    return out


def block_diag_lr_matmul_decoupled(A_blocks: torch.Tensor, M: torch.Tensor, B_blocks: torch.Tensor) -> torch.Tensor:
    """Compute ``block_diag(A_blocks) @ M @ block_diag(B_blocks)`` allowing the
    left and right blocks to have *different* (square) sizes.

    Args:
      A_blocks: (r_m, a, a) left block-diagonal factors  (rows  M = r_m * a)
      M:        (r_m*a, r_n*b) dense matrix
      B_blocks: (r_n, b, b) right block-diagonal factors (cols  N = r_n * b)

    Generalises ``block_diag_lr_matmul`` (which requires a == b). Used by the
    decoupled merge where R_in has block size ``bs_in`` and R_out ``bs_out``.
    """
    if A_blocks.ndim != 3 or B_blocks.ndim != 3:
        raise ValueError("A_blocks and B_blocks must be 3D: (r, b, b)")
    r_m, a, a2 = A_blocks.shape
    r_n, b, b2 = B_blocks.shape
    if a != a2 or b != b2:
        raise ValueError("each block must be square")
    rows_M = r_m * a
    cols_N = r_n * b
    if M.shape != (rows_M, cols_N):
        raise ValueError(f"M must have shape {(rows_M, cols_N)}, got {tuple(M.shape)}")
    if A_blocks.device != M.device or A_blocks.dtype != M.dtype:
        A_blocks = A_blocks.to(device=M.device, dtype=M.dtype)
    if B_blocks.device != M.device or B_blocks.dtype != M.dtype:
        B_blocks = B_blocks.to(device=M.device, dtype=M.dtype)

    # left = block_diag(A) @ M : per left-block, A[i] @ M[i*a:(i+1)*a, :]
    left = torch.bmm(A_blocks, M.view(r_m, a, cols_N)).reshape(rows_M, cols_N)
    # out = left @ block_diag(B) : per right-block, left[:, j*b:(j+1)*b] @ B[j]
    right = left.view(rows_M, r_n, b).transpose(0, 1)  # (r_n, rows_M, b)
    out = torch.bmm(right, B_blocks).transpose(0, 1).reshape(rows_M, cols_N)
    return out


def pytorch_skew_symmetric(vec, block_size, rows, cols):
    batch_size = vec.shape[0]
    matrix = vec.new_zeros(batch_size, block_size, block_size)  # Inherits requires_grad
    matrix[:, rows, cols] = vec
    matrix = matrix - matrix.transpose(-2, -1)
    return matrix

def cayley_batch(Qf):
    Q2f = Qf @ Qf
    Yf = 2.0 * (Qf + Q2f + Q2f @ Qf) + 2.0 * Q2f @ Q2f
    # Yf = 2.0 * (Qf + Q2f) + Q2f @ (2.0 * Qf + Q2f)
    Yf.diagonal(dim1=-2, dim2=-1).add_(1.0)
    return Yf

def get_weight_poet(R, block_size, rows, cols, r_out, r_in):
    # r_left = Rl.size(0)
    # r_right = Rr.size(0)

    # R = torch.cat([Ro, Ri], dim=0).contiguous()
    # Q_skew_cat = skew_symmetric(R, block_size, rows, cols, idx_ul)
    Q_skew_cat = pytorch_skew_symmetric(R, block_size, rows, cols)
    # Q_skew_cat = torch.ops.poet.skew_symmetric(R, block_size, rows, cols, idx_ul)

    # R_cat = CayleyTritonFn.apply(Q_skew_cat)
    R_cat = torch.ops.poet.cayley(Q_skew_cat)[0]
    # R_cat = cayley_batch(Q_skew_cat)
    R_out, R_in = R_cat.split([r_out, r_in], dim=0)

    return R_out, R_in


def get_weight_poet_decoupled(oft_R_in, oft_R_out,
                              block_size_in, block_size_out,
                              rows_in, cols_in, rows_out, cols_out):
    """Decoupled Cayley: build (R_out, R_in) from two independent oft_R tensors.

    Unlike ``get_weight_poet`` (one Cayley call on a concatenated ``oft_R``),
    the in/out sides may use different block sizes, so their skew matrices have
    different tile shapes and cannot share a single batched kernel launch. We
    therefore run two Cayley calls. The kernel handles any block size ``B``, so
    each side just supplies its own ``(r, bs, bs)`` skew batch.

    Returns ``(R_out, R_in)`` to match ``get_weight_poet``'s ordering.
    """
    Q_in = pytorch_skew_symmetric(oft_R_in, block_size_in, rows_in, cols_in)
    Q_out = pytorch_skew_symmetric(oft_R_out, block_size_out, rows_out, cols_out)
    R_in = torch.ops.poet.cayley(Q_in)[0]
    R_out = torch.ops.poet.cayley(Q_out)[0]
    return R_out, R_in


def _matrix_exp_skew(Q):
    """exp of a skew batch. matrix_exp is numerically delicate below fp32, so
    compute in fp32 (or keep fp64) then cast back. Autograd flows through the
    cast, so gradients return in the input dtype with no custom backward."""
    compute_dtype = Q.dtype if Q.dtype in (torch.float32, torch.float64) else torch.float32
    R = torch.linalg.matrix_exp(Q.to(compute_dtype))
    return R.to(Q.dtype)


def get_weight_poet_decoupled_exp(oft_R_in, oft_R_out,
                                  block_size_in, block_size_out,
                                  rows_in, cols_in, rows_out, cols_out):
    """Matrix-exponential twin of ``get_weight_poet_decoupled``.

    Builds (R_out, R_in) as the *exact* matrix exponential of the skew
    generators instead of the truncated Cayley/Neumann polynomial. R is exactly
    orthogonal for any Q (no ||Q||<1 ceiling, no truncation error), and the
    singular values of Q are exactly the rotation angles of R. Same signature
    and (R_out, R_in) return ordering as ``get_weight_poet_decoupled`` so it is a
    drop-in for the parameterization dispatch.
    """
    Q_in = pytorch_skew_symmetric(oft_R_in, block_size_in, rows_in, cols_in)
    Q_out = pytorch_skew_symmetric(oft_R_out, block_size_out, rows_out, cols_out)
    R_in = _matrix_exp_skew(Q_in)
    R_out = _matrix_exp_skew(Q_out)
    return R_out, R_in

def torch_bmm(x, R, block_size):
    Bdims = x.shape[:-1]
    xr = x.view(*Bdims, -1, block_size)
    xr = torch.einsum("...rk,rkc->...rc", xr, R)
    x_rot = xr.contiguous().view(*Bdims, -1)
    return x_rot

def chain_layer_x_pytorch(x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor,
                          bias: Optional[torch.Tensor], Rout: torch.Tensor, block_size: int) -> torch.Tensor:
    x = torch_bmm(x, Rin, block_size)
    y = x @ weight.t()
    if bias is not None:
        y = y + bias
    y = torch_bmm(y, Rout, block_size)
    return y


def chain_layer_x_fast_decoupled(
    x: torch.Tensor, Rin: torch.Tensor, weight: torch.Tensor,
    bias: Optional[torch.Tensor], Rout: torch.Tensor,
    perm_in_inv: torch.Tensor, perm_in: torch.Tensor,
    perm_out: torch.Tensor, perm_out_inv: torch.Tensor,
    bsz_in: int, bsz_out: int,
) -> torch.Tensor:
    """Non-recompute ("fast") twin of
    ``poet::chain_layer_checkpoint_mem_o2_decoupled``.

    Identical math (verified bit-exact on fwd + grad_x/grad_Rin/grad_Rout), but
    written as plain ops so autograd *saves* the cheap block-rotation
    activations instead of recomputing the input rotation + dense matmul in the
    backward. Trades a little activation memory for ~1 fewer forward-equivalent
    per layer per microbatch — a net win whenever memory is not the binding
    constraint (e.g. small/medium dense models). It is also fully traceable, so
    ``torch.compile`` can fuse it with the surrounding Cayley output (the custom
    op is opaque to the compiler). Set ``mem_efficient_mode=True`` to fall back
    to the recompute op for memory-bound runs.
    """
    leading_shape = x.shape[:-1]
    Din = x.shape[-1]
    N = x.numel() // Din
    rin = Rin.size(0)
    rout = Rout.size(0)

    # Permutations via PermutationFunction (gather backward) instead of plain
    # advanced indexing. Both have a gather forward, but plain `x[..., perm]`
    # makes autograd emit a scatter-add backward (indexing_backward_kernel) —
    # the single biggest cost in the POET fwd+bwd (~50-63% of the chain). A
    # permutation has no duplicate indices, so its exact backward is the
    # inverse-perm GATHER, ~3.2x faster (bit-identical). inv(perm_in_inv)=perm_in,
    # inv(perm_out)=perm_out_inv.
    x = PermutationFunction.apply(x, perm_in_inv, perm_in)
    xb_r = x.reshape(N, rin, bsz_in).transpose(0, 1)        # [rin, N, b_in]
    xR = torch.bmm(xb_r, Rin).transpose(0, 1).reshape(N, rin * bsz_in)
    yb_flat = xR @ weight.t()
    if bias is not None:
        yb_flat = yb_flat + bias
    yb_r = yb_flat.view(N, rout, bsz_out).transpose(0, 1)   # [rout, N, b_out]
    y = torch.bmm(yb_r, Rout).transpose(0, 1).reshape(*leading_shape, rout * bsz_out)
    y = PermutationFunction.apply(y, perm_out, perm_out_inv)
    return y


@torch.compile(fullgraph=True)
def forward_core(
    x: torch.Tensor, 
    R: torch.Tensor,
    block_size: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
    perm_in: torch.Tensor, 
    perm_in_inv: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    r_in: int,
    r_out: int,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
    mem_efficient_mode: bool = False,
) -> torch.Tensor:

    R_out, R_in = get_weight_poet(R, block_size, rows, cols, r_out, r_in) 

    # POET-X fast
    # x = permute_x(x, perm_in_inv, perm_in)
    # y = chain_layer_x_pytorch(x, R_in, base_weight, base_bias, R_out, block_size)
    # y = permute_x(y, perm_out, perm_out_inv)
    
    # POET-X mem efficient
    y = chain_layer_x_checkpoint_mem_o2(x, R_in, base_weight, base_bias, R_out, perm_in_inv, perm_in, perm_out, perm_out_inv, block_size)

    return y


def _forward_core_decoupled_eager(
    x: torch.Tensor,
    oft_R_in: torch.Tensor,
    oft_R_out: torch.Tensor,
    block_size_in: int,
    block_size_out: int,
    rows_in: torch.Tensor,
    cols_in: torch.Tensor,
    rows_out: torch.Tensor,
    cols_out: torch.Tensor,
    perm_in: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    mem_efficient_mode: bool = False,
) -> torch.Tensor:
    R_out, R_in = get_weight_poet_decoupled(
        oft_R_in, oft_R_out, block_size_in, block_size_out,
        rows_in, cols_in, rows_out, cols_out,
    )
    if mem_efficient_mode:
        # Memory-bound path: recompute the input rotation + dense matmul in the
        # backward (saves activation memory, costs ~1 extra forward per layer).
        y = chain_layer_x_checkpoint_mem_o2_decoupled(
            x, R_in, base_weight, base_bias, R_out,
            perm_in_inv, perm_in, perm_out, perm_out_inv,
            block_size_in, block_size_out,
        )
    else:
        # Fast (default) path: let autograd save activations — no recompute.
        y = chain_layer_x_fast_decoupled(
            x, R_in, base_weight, base_bias, R_out,
            perm_in_inv, perm_in, perm_out, perm_out_inv,
            block_size_in, block_size_out,
        )
    return y


# Compiled (fused) entry used during TRAINING (grad enabled). Its eval/inference
# graph — built the first time the model runs under torch.no_grad() — trips a
# torch-2.11 Inductor scheduler bug (KeyError 'op6' in compute_ancestors) for
# full-block (block_count=1) configs, where the 1536-dim blocks produce an
# inference graph whose schedule references a buffer an inference-only fusion pass
# already eliminated. Training's graph keeps that buffer (it's a saved activation),
# so it never hits it. POETLinear.forward therefore routes no_grad/eval forwards to
# the eager twin `_forward_core_decoupled_eager` above — identical math, just
# unfused. Repro: /lustre/home/zqiu/tmp/poet_op6_repro2.py
forward_core_decoupled = torch.compile(_forward_core_decoupled_eager, fullgraph=True)


def forward_core_decoupled_exp(
    x: torch.Tensor,
    oft_R_in: torch.Tensor,
    oft_R_out: torch.Tensor,
    block_size_in: int,
    block_size_out: int,
    rows_in: torch.Tensor,
    cols_in: torch.Tensor,
    rows_out: torch.Tensor,
    cols_out: torch.Tensor,
    perm_in: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    mem_efficient_mode: bool = False,
) -> torch.Tensor:
    """Forward for the exact-matrix-exponential parameterization.

    Builds R eagerly via matrix_exp (NOT under torch.compile fullgraph, which
    matrix_exp's backward may not support), then reuses the same
    parameterization-agnostic chain consumers as the Cayley path. The R build is
    O(b^3) per block, amortized over the whole microbatch; the per-token chain is
    unchanged.
    """
    R_out, R_in = get_weight_poet_decoupled_exp(
        oft_R_in, oft_R_out, block_size_in, block_size_out,
        rows_in, cols_in, rows_out, cols_out,
    )
    if mem_efficient_mode:
        y = chain_layer_x_checkpoint_mem_o2_decoupled(
            x, R_in, base_weight, base_bias, R_out,
            perm_in_inv, perm_in, perm_out, perm_out_inv,
            block_size_in, block_size_out,
        )
    else:
        y = chain_layer_x_fast_decoupled(
            x, R_in, base_weight, base_bias, R_out,
            perm_in_inv, perm_in, perm_out, perm_out_inv,
            block_size_in, block_size_out,
        )
    return y


@torch.compile(fullgraph=True)
def forward_core_q8(
    x: torch.Tensor, 
    R: torch.Tensor,
    block_size: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
    perm_in: torch.Tensor, 
    perm_in_inv: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    r_in: int,
    r_out: int,
    W_q: torch.Tensor,
    W_scales: torch.Tensor,
    W_zeros: torch.Tensor,
    group_size: int,
    base_bias: torch.Tensor,
    mem_efficient_mode: bool = False,
) -> torch.Tensor:

    R_out, R_in = get_weight_poet(R, block_size, rows, cols, r_out, r_in) 

    # POET-X fast mode
    # x = permute_x(x, perm_in_inv, perm_in)
    # y = chain_layer_x_checkpoint_q8(x, R_in, W_q, W_scales, W_zeros, group_size, base_bias, R_out, block_size)
    # y = permute_x(y, perm_out, perm_out_inv)

    # POET-X mem efficient mode
    y = chain_layer_x_checkpoint_mem_o2_q8(x, R_in, W_q, W_scales, W_zeros, group_size, base_bias, R_out, perm_in_inv, perm_in, perm_out, perm_out_inv, block_size)

    return y


class POETLinear(nn.Module):
    """Block-orthogonal linear layer.

    Internally stores TWO independent skew parameters, ``oft_R_in`` and
    ``oft_R_out``, with potentially different block sizes ``block_size_in`` /
    ``block_size_out``. Two constructor entry points:

    * ``bsz=int`` (legacy): ``block_size_in == block_size_out == bsz``. The
      block size must divide both ``in_features`` and ``out_features``.
    * ``block_count=int``: both sides get ``block_count`` blocks, so
      ``block_size_in = in_features // block_count`` and
      ``block_size_out = out_features // block_count`` (may differ).

    Exactly one of ``bsz`` / ``block_count`` must be given. The forward always
    routes through the decoupled op, so the two paths share one code path.
    """

    def __init__(self, in_features, out_features, bsz=None, block_count=None,
                 bias=False, device=None, dtype=None, mem_efficient_mode=None,
                 parameterization="cayley"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # The exp path runs eager: matrix_exp's backward does not survive
        # torch.compile(fullgraph=True), so forward_core_decoupled_exp is NOT
        # compiled (unlike the Cayley forward_core_decoupled). Without inductor
        # to fuse/recompute the per-token chain, the fast (non-recompute) chain
        # saves every intermediate and drives peak memory ~3x the theoretical
        # footprint, OOMing the downstream fused cross-entropy (observed at
        # 60m/mbs128 on an 80GB H100, 2026-06-02). So default exp to the
        # recompute chain and leave Cayley on the fast path. Cayley can also hit
        # that activation-peak OOM at large mbs (the fast chain holds every
        # rotation activation, leaving no room for the fp32 CE logits buffer);
        # set POET_MEM_EFFICIENT=1 to force the recompute chain for Cayley too.
        # Passing an explicit bool overrides both this env gate and the default.
        if mem_efficient_mode is None:
            mem_efficient_mode = (parameterization == "exp") or os.environ.get(
                "POET_MEM_EFFICIENT"
            ) == "1"
        self.mem_efficient_mode = mem_efficient_mode

        # Single-step (R=I) fast path: collapses the identity-rotation chain to a
        # permuted GEMM + closed-form oft_R grad. Set by replace_linears_with_poet
        # when optim.poet.single_step_fast is on (requires merge_period=1, cayley).
        self.single_step_fast = False

        if (bsz is None) == (block_count is None):
            raise ValueError("exactly one of bsz or block_count must be set")
        if bsz is not None:
            if in_features % bsz != 0 or out_features % bsz != 0:
                raise ValueError(
                    f"block_size {bsz} doesn't divide in={in_features} or out={out_features}"
                )
            block_size_in = block_size_out = bsz
        else:
            if in_features % block_count != 0 or out_features % block_count != 0:
                raise ValueError(
                    f"block_count {block_count} doesn't divide in={in_features} or out={out_features}"
                )
            block_size_in = in_features // block_count
            block_size_out = out_features // block_count

        self.block_size_in = block_size_in
        self.block_size_out = block_size_out
        # ``block_size`` kept for back-compat (merge/"is-active" guards, logging,
        # diagnostics). It equals the shared block size in the legacy/equal case;
        # in the decoupled case it is the input-side block size — callers that
        # need an exact per-side value must read block_size_in / block_size_out.
        self.block_size = block_size_in

        if parameterization not in ("cayley", "exp"):
            raise ValueError(
                f"parameterization must be 'cayley' or 'exp', got {parameterization!r}"
            )
        self.parameterization = parameterization

        # Basic linear layer parameters
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable skew-params per block — two independent tensors.
        r_in = in_features // block_size_in
        r_out = out_features // block_size_out
        n_elems_in = block_size_in * (block_size_in - 1) // 2
        n_elems_out = block_size_out * (block_size_out - 1) // 2
        self.oft_R_in = nn.Parameter(torch.zeros((r_in, n_elems_in), device=device, dtype=dtype))
        self.oft_R_out = nn.Parameter(torch.zeros((r_out, n_elems_out), device=device, dtype=dtype))
        self.r_in = r_in
        self.r_out = r_out

        rows_in, cols_in = torch.triu_indices(block_size_in, block_size_in, 1, device=device)
        self.register_buffer('rows_in', rows_in.to(torch.int32))
        self.register_buffer('cols_in', cols_in.to(torch.int32))
        rows_out, cols_out = torch.triu_indices(block_size_out, block_size_out, 1, device=device)
        self.register_buffer('rows_out', rows_out.to(torch.int32))
        self.register_buffer('cols_out', cols_out.to(torch.int32))

        perm_in = torch.randperm(in_features, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features, device=device, dtype=torch.int32)
        self.register_buffer('perm_in', perm_in)
        self.register_buffer('perm_out', perm_out)
        self.register_buffer('perm_in_inv', torch.argsort(perm_in).to(torch.int32))
        self.register_buffer('perm_out_inv', torch.argsort(perm_out).to(torch.int32))

    def random_init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.oft_R_in, std=1e-3)
        nn.init.normal_(self.oft_R_out, std=1e-3)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def perform_permutation(self) -> None:
        # Merge the self.linear.weight with permutations to avoid P_in.t() @ W_orig.t() @ P_out in the forward pass
        W = self.weight
        Wp = W.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
        self.weight.detach().copy_(Wp)

    def update_permutation(self):
        """Update the permutation of the indices."""
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device)
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(torch.argsort(perm_in))
        perm_out = torch.randperm(self.out_features, device=device)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(torch.argsort(perm_out))

        self.perform_permutation()

    def _build_R(self, oft_in, oft_out):
        """Build (R_out, R_in) from skew params using the configured map.

        Single dispatch point so forward, merge, and the dW-spec estimator all
        use the same orthogonalization.
        """
        if self.parameterization == "exp":
            return get_weight_poet_decoupled_exp(
                oft_in, oft_out, self.block_size_in, self.block_size_out,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            )
        return get_weight_poet_decoupled(
            oft_in, oft_out, self.block_size_in, self.block_size_out,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
        )

    def _merge_R(self):
        """Build (R_out, R_in) from the two decoupled skew params (no grad)."""
        return self._build_R(self.oft_R_in, self.oft_R_out)

    def merge_then_reinitialize_working(self) -> None:
        # with torch.no_grad():
        R_out, R_in = self._merge_R()

        # y = x @ P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
        # 1) P_in.t() @ W_orig.t() @ P_out
        W = self.weight.detach().clone()
        # W0 = W.detach().clone()
        tmp = W.t()
        # # # 2) R_in @ tmp @ R_out  (decoupled: R_in/R_out may differ in block size)
        tmp = block_diag_lr_matmul_decoupled(R_in, tmp, R_out)
        # 3) P_in @ tmp @ P_out.t()
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        # Transpose back to weight shape
        self.weight.detach().copy_(expected)

        self.oft_R_in.zero_()
        self.oft_R_out.zero_()
        self.update_permutation()

    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        R_out, R_in = self._merge_R()
        self._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)

    @torch.no_grad()
    def _fold_with_R(self, R_out, R_in, reinit_perm: bool = True) -> None:
        """Fold given (already-built) rotations into the frozen weight and zero
        oft_R. Split out of merge_then_reinitialize so a batched orchestrator can
        build R for many layers in one Cayley call, then fold each here."""
        W = self.weight.detach().clone()
        tmp = W.t()
        tmp = block_diag_lr_matmul_decoupled(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        if reinit_perm:
            device = self.weight.device
            perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
            perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
            perm_in_inv = torch.argsort(perm_in).to(torch.int32)
            perm_out_inv = torch.argsort(perm_out).to(torch.int32)
            expected = expected.index_select(0, perm_out_inv).index_select(1, perm_in_inv)
            self.weight.detach().copy_(expected)
            self.perm_in.copy_(perm_in)
            self.perm_in_inv.copy_(perm_in_inv)
            self.perm_out.copy_(perm_out)
            self.perm_out_inv.copy_(perm_out_inv)
        else:
            expected = expected.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
            self.weight.detach().copy_(expected)

        self.oft_R_in.zero_()
        self.oft_R_out.zero_()

    def forward(self, x):
        # Single-step fast path (R=I). The cayley guard is defensive: the factor-2
        # closed form is Cayley-specific, and build-time validation already forbids
        # single_step_fast with parameterization='exp' — this just fails safe
        # (falls through to the correct chain) if the flag is ever set directly.
        if getattr(self, "single_step_fast", False) and self.parameterization == "cayley":
            from .single_step import SingleStepPOETFunction
            return SingleStepPOETFunction.apply(
                x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
                self.perm_in_inv, self.perm_in, self.perm_out, self.perm_out_inv,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
                self.block_size_in, self.block_size_out,
            )
        if self.parameterization == "exp":
            return forward_core_decoupled_exp(
                x, self.oft_R_in, self.oft_R_out,
                self.block_size_in, self.block_size_out,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
                self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv,
                self.weight, self.bias, self.mem_efficient_mode,
            )
        # Training uses the compiled (fused) path; eval/inference (no grad) uses the
        # eager twin. The compiled INFERENCE graph trips a torch-2.11 Inductor bug
        # (KeyError 'op6') for full-block (block_count=1) configs; eager is identical
        # math, just unfused. See _forward_core_decoupled_eager.
        core = forward_core_decoupled if torch.is_grad_enabled() else _forward_core_decoupled_eager
        x = core(
            x, self.oft_R_in, self.oft_R_out,
            self.block_size_in, self.block_size_out,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv,
            self.weight, self.bias, self.mem_efficient_mode,
        )
        return x



class POETLinearNeurips(nn.Module):
    def __init__(self, in_features, out_features, bsz=256, bias=False, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = bsz
        # Basic linear layer parameters
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable skew-params per block
        r_in = in_features // bsz
        r_out = out_features // bsz
        n_elements = bsz * (bsz - 1) // 2
        # Param tensors can be any square; we skew them inside forward
        self.oft_R_out = nn.Parameter(torch.zeros((r_out, n_elements), device=device, dtype=dtype))
        self.oft_R_in = nn.Parameter(torch.zeros((r_in, n_elements), device=device, dtype=dtype))

        rows, cols = torch.triu_indices(bsz, bsz, 1, device=device)
        self.register_buffer('rows', rows.to(torch.int32))
        self.register_buffer('cols', cols.to(torch.int32))

        perm_in = torch.randperm(in_features, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features, device=device, dtype=torch.int32)
        self.register_buffer('perm_in', perm_in)
        self.register_buffer('perm_out', perm_out)
        self.register_buffer('perm_in_inv', torch.argsort(perm_in).to(torch.int32))
        self.register_buffer('perm_out_inv', torch.argsort(perm_out).to(torch.int32))

    def update_permutation(self):
        """Update the permutation of the indices."""
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device)
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(torch.argsort(perm_in))
        perm_out = torch.randperm(self.out_features, device=device)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(torch.argsort(perm_out))

    def merge_then_reinitialize(self) -> None:
        # with torch.no_grad():
        R_out, R_in = get_weight_poet(self.oft_R, self.block_size, self.rows, self.cols, self.r_out, self.r_in)

        # y = x @ P_in @ R_in @ P_in.t() @ W_orig.t() @ P_out @ R_out @ P_out.t()
        # 1) P_in.t() @ W_orig.t() @ P_out
        W = self.weight.detach().clone()
        # W0 = W.detach().clone()
        tmp = W.t()
        # # # 2) R_in @ tmp @ R_out
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        # 3) P_in @ tmp @ P_out.t()
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()
        
        # Transpose back to weight shape
        self.weight.detach().copy_(expected)

        self.oft_R.zero_()
        self.update_permutation()

    def get_cayley_transform_neumann_optimized(self, mode='all', num_neumann_terms=5):
        """
        Ultra-optimized version of get_cayley_transform_neumann.
        """
        R_left = None
        R_right = None
        # self.normalize_parameters(rms_norm=1.0)
            
        # Process left transform if needed
        if mode in ['all', 'left']:
            # Initialize result - use existing identity and expand in-place
            # Q_blocks = SkewSymmetricBatched.apply(self.R_out, self.soft_block_size)
            Q_blocks = pytorch_skew_symmetric(self.oft_R_out, self.block_size, self.rows, self.cols)
            R_left = torch.eye(self.block_size, device=self.oft_R_out.device, dtype=self.oft_R_out.dtype).repeat(self.oft_R_out.shape[0], 1, 1)
            
            # For small matrices, unroll the first few iterations
            if num_neumann_terms > 1:
                # First term (i=1): Add 2*Q
                R_left.add_(Q_blocks, alpha=2.0)
                
                if num_neumann_terms > 2:
                    # Second term (i=2): Add 2*Q^2
                    Q_squared = torch.bmm(Q_blocks, Q_blocks)
                    R_left.add_(Q_squared, alpha=2.0)
                    
                    # Use bmm for remaining iterations
                    Q_power = Q_squared
                    for i in range(3, num_neumann_terms):
                        Q_power = torch.bmm(Q_power, Q_blocks)
                        R_left.add_(Q_power, alpha=2.0)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            # Initialize result - use existing identity and expand in-place
            # Q_blocks = SkewSymmetricBatched.apply(self.R_in, self.soft_block_size)
            Q_blocks = pytorch_skew_symmetric(self.oft_R_in, self.block_size, self.rows, self.cols)
            R_right = torch.eye(self.block_size, device=self.oft_R_in.device, dtype=self.oft_R_in.dtype).repeat(self.oft_R_in.shape[0], 1, 1)
            
            # For small matrices, unroll the first few iterations
            if num_neumann_terms > 1:
                # First term (i=1): Add 2*Q
                R_right.add_(Q_blocks, alpha=2.0)
                
                if num_neumann_terms > 2:
                    # Second term (i=2): Add 2*Q^2
                    Q_squared = torch.bmm(Q_blocks, Q_blocks)
                    R_right.add_(Q_squared, alpha=2.0)
                    
                    # Use bmm for remaining iterations
                    Q_power = Q_squared
                    for i in range(3, num_neumann_terms):
                        Q_power = torch.bmm(Q_power, Q_blocks)
                        R_right.add_(Q_power, alpha=2.0)

        return R_left, R_right

    def forward(self, x):
        R_left, R_right = self.get_cayley_transform_neumann_optimized()

        # y = x @ W_new.t()
        # W_new = P_out @ R_out @ P_out.t() @ W @ P_in @ R_in @ P_in.t()
        # Calculation for Inner = (P_out^T @ W) @ P_in
        temp_W1_kernel = self.weight.index_select(0, self.perm_out_inv)
        Inner = temp_W1_kernel.index_select(1, self.perm_in_inv)

        R_left_bs = torch.block_diag(*R_left)
        R_right_bs = torch.block_diag(*R_right)
        Outer = R_left_bs @ Inner @ R_right_bs

        # Calculation for Final = (P_out @ Outer) @ P_in^T
        temp_Outer_kernel = Outer.index_select(0, self.perm_out)
        transformed_weight = temp_Outer_kernel.index_select(1, self.perm_in)

        return F.linear(x, transformed_weight.squeeze(), self.bias)


class QPOETLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        bsz=256,
        device=None,
        dtype=None,
        num_bits=8,
        group_size=256,
        stochastic_round=True,
        mem_efficient_mode=False,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        int8_weight, scales, zeros = _quantize_tensor_int8(weight.data, q_group_size=group_size)
        torch.cuda.empty_cache()

        self.weight = nn.Parameter(int8_weight, requires_grad=False).to(device) # Only Tensors of floating point and complex dtype can require gradients, using float_gradient to store the gradient
        self.register_buffer('weight_scales', scales.to(device))
        self.register_buffer('weight_zeros', zeros.to(device))
        self.weight_group_size = group_size
        self.weight_saved_data_dtype = int8_weight.dtype
        self.weight_stochastic_round = stochastic_round
        self.weight_num_bits = num_bits

        if not num_bits == 8:
            raise NotImplementedError

        self.bias = nn.Parameter(bias, requires_grad=True).to(device) if bias is not None else None

        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]
        self.block_size = bsz
        self.mem_efficient_mode = mem_efficient_mode

        # Trainable skew-params per block (same as POETLinear)
        r_in = self.in_features // bsz
        r_out = self.out_features // bsz
        n_elements = bsz * (bsz - 1) // 2
        self.oft_R = nn.Parameter(torch.zeros((r_in + r_out, n_elements), device=device, dtype=dtype))
        self.r_in = r_in
        self.r_out = r_out

        rows, cols = torch.triu_indices(bsz, bsz, 1, device=device)
        self.register_buffer("rows", rows.to(torch.int32))
        self.register_buffer("cols", cols.to(torch.int32))

        # same perm buffers as POETLinear (perm acts on features/groups, not int8 groups)
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    @torch.no_grad()
    def _requantize_from_float(self, w_float: torch.Tensor):
        q, scales, zeros = _quantize_tensor_int8(w_float, q_group_size=self.weight_group_size, n_bit=self.weight_num_bits)
        self.weight.detach().copy_(q.to(self.weight.device))
        self.weight_scales.copy_(scales.to(self.weight.device))
        self.weight_zeros.copy_(zeros.to(self.weight.device))

    def _dequantize_to(self, dtype: torch.dtype):
        w = self.weight.to(dtype).reshape(-1, self.weight_group_size)   
        w = (w - self.weight_zeros.to(dtype)) * self.weight_scales.to(dtype)
        return w.reshape(self.weight.shape)

    @torch.no_grad()
    def merge_then_reinitialize(self) -> None:
        # Same math as POETLinear.merge_then_reinitialize, but float compute + requantize
        R_out, R_in = get_weight_poet(self.oft_R, self.block_size, self.rows, self.cols, self.r_out, self.r_in)

        # Step 1-4: Merge adapters (same as before)
        W = self._dequantize_to(dtype=self.oft_R.dtype)

        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        # Step 5: Generate NEW permutation BEFORE quantizing
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        perm_out_inv = torch.argsort(perm_out).to(torch.int32)

        # Step 6: Apply NEW permutation to the float weight
        # This undoes OLD perm and applies NEW perm in one step
        expected = expected.index_select(0, perm_out_inv).index_select(1, perm_in_inv)
        
        # Step 7: Quantize (only once, with correct NEW perm)
        self._requantize_from_float(expected)
        
        # Step 8: Update buffers to match the quantized weight
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(perm_in_inv)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(perm_out_inv)

        self.oft_R.zero_()

    def forward(self, x):
        return forward_core_q8(
            x,
            self.oft_R,
            self.block_size,
            self.rows,
            self.cols,
            self.perm_in,
            self.perm_in_inv,
            self.perm_out,
            self.perm_out_inv,
            self.r_in,
            self.r_out,
            self.weight,
            self.weight_scales,
            self.weight_zeros,
            self.weight_group_size,
            self.bias,
            self.mem_efficient_mode,
        )



def replace_linear_with_poet(module: nn.Module, block_size: int, init_type: str, mup_alpha: float, device=None, dtype=None, 
                        mem_efficient_mode=False, neurips_version=False, v2: bool=False) -> int:
    def _convert(m: nn.Module, v2: bool=False):
        # nonlocal replaced
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and 'lm_head' not in name.lower():
                if block_size and child.in_features % block_size == 0 and child.out_features % block_size == 0:
                    if neurips_version:
                        new_lin = POETLinearNeurips(
                            in_features=child.in_features,
                            out_features=child.out_features,
                            bsz=block_size,
                            bias=(child.bias is not None),
                            device=device,
                            dtype=dtype,
                        )
                    else:
                        new_lin = POETLinear(
                            in_features=child.in_features,
                            out_features=child.out_features,
                            bsz=block_size,
                            bias=(child.bias is not None),
                            device=device,
                            dtype=dtype,
                            mem_efficient_mode=mem_efficient_mode,
                        )
                    with torch.no_grad():
                        if init_type == 'normalized':
                            # [Check 1] Measure Spectral Norm BEFORE normalization
                            # child.weight is the original random initialization
                            # spec_before = torch.linalg.norm(child.weight.data.float(), ord=2).item() / torch.sqrt(torch.tensor(child.weight.data.shape[0]) / torch.tensor(child.weight.data.shape[1]))
                            
                            child.weight.data = child.weight.data / torch.norm(child.weight.data, dim=1, keepdim=True)

                        elif init_type == 'mup_normalized':
                            d_in = torch.tensor(child.weight.data.shape[1])
                            d_out = torch.tensor(child.weight.data.shape[0])

                            # [Check 1] Measure Spectral Norm BEFORE normalization
                            # child.weight is the original random initialization
                            # spec_before = torch.linalg.norm(child.weight.data.float(), ord=2).item() / torch.sqrt(d_out / d_in)
                            
                            normed_weight = child.weight.data / torch.norm(child.weight.data, dim=1, keepdim=True)

                            target_spec = mup_alpha * torch.sqrt(d_out / d_in)
                            current_spec = torch.linalg.norm(normed_weight.float(), ord=2).item()

                            scaling_factor = (target_spec / current_spec).to(dtype=normed_weight.dtype, device=normed_weight.device)
                            final_weight = normed_weight * scaling_factor
                            child.weight.data = final_weight

                        new_lin.weight.copy_(child.weight.detach().to(new_lin.weight.dtype))

                        # [Check 2] Measure Spectral Norm AFTER normalization & copy
                        # This should be much smaller (close to 2.0 for large square matrices)
                        # spec_after = torch.linalg.norm(new_lin.weight.float(), ord=2).item() / torch.sqrt(torch.tensor(new_lin.weight.data.shape[0]) / torch.tensor(new_lin.weight.data.shape[1]))
                        # print(f"Weight Spectral Norm (Before): {spec_before:.4f}, (After): {spec_after:.4f}")

                        if child.bias is not None and new_lin.bias is not None:
                            new_lin.bias.copy_(child.bias.detach().to(new_lin.bias.dtype))
                    setattr(m, name, new_lin)
                else:
                    # skip non-divisible layers
                    raise ValueError(f"Layer {name} has in_features {child.in_features} and out_features {child.out_features}, which are not divisible by {block_size}")
            else:
                _convert(child)
    _convert(module)


def prepare_model_for_int8_training_poet(model, args, target_module):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_int8_training_poet(module, args, target_module)

        if isinstance(module, nn.Linear):
            if not name in target_module: continue

            bias_data = module.bias.data if module.bias is not None else None
            weight = module.weight.data
            if args.init_type == 'normalized':
                weight = weight / torch.norm(weight, dim=1, keepdim=True)

            new_layers = QPOETLinear(
                weight,
                bias_data,
                bsz=args.poet_block_size,
                num_bits=args.weight_bits,
                group_size=args.weight_group_size,
                stochastic_round=args.stochastic_round
            )
            model._modules[name] = new_layers

    return model

def check_and_merge(model: nn.Module, iter_count=0, poet_reset_gap=4):
    if iter_count <= 0 or (iter_count % poet_reset_gap != 0):
        return

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    # with torch.compiler.set_stance("force_eager"):
    with torch.compiler.set_stance("eager_then_compile"):
        for name, module in model.named_modules():
            if isinstance(module, (POETLinear, QPOETLinear)) and module.block_size > 0:
                with torch.no_grad():
                    if rank == 0:
                        # rank 0 does the merge + permutation update
                        module.merge_then_reinitialize()

                    # ensure all ranks get the exact same state
                    if isinstance(module, QPOETLinear):
                        torch.distributed.broadcast(module.oft_R.data, src=0)
                        torch.distributed.broadcast(module.weight_scales, src=0)
                        torch.distributed.broadcast(module.weight_zeros, src=0)
                    else:
                        # POETLinear stores two decoupled skew params.
                        torch.distributed.broadcast(module.oft_R_in.data, src=0)
                        torch.distributed.broadcast(module.oft_R_out.data, src=0)
                    torch.distributed.broadcast(module.weight.data, src=0)
                    if module.bias is not None:
                        torch.distributed.broadcast(module.bias.data, src=0)
                    torch.distributed.broadcast(module.perm_in, src=0)
                    torch.distributed.broadcast(module.perm_in_inv, src=0)
                    torch.distributed.broadcast(module.perm_out, src=0)
                    torch.distributed.broadcast(module.perm_out_inv, src=0)

        if is_dist:
            dist.barrier()


def get_grad_clipping_value(global_step, grad_clipping, warmup_steps, period_T, min_ratio=0.1, max_steps=2000):
    """
    Gradient clipping scheduler that linearly increases from min_ratio * grad_clipping 
    to grad_clipping over warmup_steps, repeating every period_T steps
    
    Args:
        global_step: Current training step
        grad_clipping: Maximum gradient clipping value
        warmup_steps: Number of steps to linearly increase clipping value
        period_T: Period after which the warmup cycle repeats
        min_ratio: Starting ratio of grad_clipping (default: 0.1)
        max_steps: Maximum number of steps to apply gradient clipping
    Returns:
        Current gradient clipping value
    """
    if global_step < period_T:
        return grad_clipping
    
    # Calculate position within the current cycle
    cycle_position = global_step % period_T

    if global_step > max_steps:
        return grad_clipping
    
    if cycle_position >= warmup_steps:
        return grad_clipping
        
    # Linear warmup from min_ratio * grad_clipping to grad_clipping
    warmup_factor = min_ratio + (1.0 - min_ratio) * (cycle_position / max(1, warmup_steps))
    return warmup_factor * grad_clipping


@torch.no_grad()
def estimate_poet_delta_weff_spec(
    poet_module: nn.Module,
    oft_R_in_prev: torch.Tensor,
    oft_R_out_prev: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> float:
    """
    Estimates ||ΔW_eff||_2 where (row-space) W_eff^T = Rin @ W^T @ Rout.
    We ignore permutations (they are orthogonal and constant between merges).

    Takes the two decoupled skew snapshots (``oft_R_in_prev``,
    ``oft_R_out_prev``) and compares to the module's current params.
    """
    device = poet_module.weight.device
    W = poet_module.weight.detach().to(device=device, dtype=compute_dtype)

    def _R(oft_in, oft_out):
        oft_in = oft_in.to(device=device, dtype=poet_module.oft_R_in.dtype)
        oft_out = oft_out.to(device=device, dtype=poet_module.oft_R_out.dtype)
        if getattr(poet_module, "parameterization", "cayley") == "exp":
            return get_weight_poet_decoupled_exp(
                oft_in, oft_out,
                poet_module.block_size_in, poet_module.block_size_out,
                poet_module.rows_in, poet_module.cols_in,
                poet_module.rows_out, poet_module.cols_out,
            )
        return get_weight_poet_decoupled(
            oft_in, oft_out,
            poet_module.block_size_in, poet_module.block_size_out,
            poet_module.rows_in, poet_module.cols_in,
            poet_module.rows_out, poet_module.cols_out,
        )

    R_out_prev, R_in_prev = _R(oft_R_in_prev, oft_R_out_prev)
    R_out_cur, R_in_cur = _R(poet_module.oft_R_in.detach(), poet_module.oft_R_out.detach())
    R_out_prev = R_out_prev.to(dtype=compute_dtype)
    R_in_prev = R_in_prev.to(dtype=compute_dtype)
    R_out_cur = R_out_cur.to(dtype=compute_dtype)
    R_in_cur = R_in_cur.to(dtype=compute_dtype)

    # M = Rin @ W^T @ Rout  (shape: in_features x out_features)
    M_prev = block_diag_lr_matmul_decoupled(R_in_prev, W.t(), R_out_prev)
    # M_prev = M_prev.t()
    M_cur = block_diag_lr_matmul_decoupled(R_in_cur, W.t(), R_out_cur)
    # M_cur = M_cur.t()

    dM = M_cur - M_prev

    # Spectral norm (largest singular value) — SVD-based, can be expensive per-step.
    sigma = torch.linalg.matrix_norm(dM, ord=2).item()

    return dM, float(sigma)