"""Single-step (R=Identity) fast path for POET.

When merge_period=1 the rotation generators oft_R are folded into the frozen
base weight and zeroed after EVERY optimizer step, so oft_R=0 at every forward
=> R = Cayley(0) = Identity. The stock chain (chain_layer_x_fast_decoupled) then
computes x@I = x via per-token bmms — pure overhead (~3x the base GEMM FLOPs)
that exists only to produce the gradient of oft_R at R=I.

SingleStepPOETFunction collapses that: the forward is the permuted GEMM the chain
reduces to at R=I (u = x[perm_in_inv]; v = u@W^T + bias; y = v[perm_out]), and the
backward (saving ONLY x) computes the oft_R gradient in closed form, in the
chain's NATURAL right-multiply orientation (with Gv = grad_y[perm_out_inv] and
A = (x[perm_in_inv])^T @ Gv, the rotation-frame weight gradient):

    M_in  = A @ W
    M_out = W @ A + outer(bias, Gv.sum(0))   # bias rank-1 term (0 if no bias)
    grad_oft_R_{out,in} = 2 * blockdiag_skew_vec(M_{out,in})

The factor 2 is the Cayley Jacobian at 0 (cayley_batch(Q) = I + 2Q + O(Q^2)).
WARNING: the chain right-multiplies R (x@R), so a left-multiply 'G@W^T' form has
the WRONG skew sign and omits bias -- use the A/W form above (verified bit-exact
in /tmp/poet_plan_selfcheck3.py). ONLY valid at oft_R=0 (merge_period=1) and
parameterization='cayley'. The caller gates on both. The post-step merge is
unchanged: it still builds the real rotation from the stepped oft_R and folds it
into W.
"""
from __future__ import annotations

import torch


def _blockdiag_skew_vec(full: torch.Tensor, b: int, rows: torch.Tensor,
                        cols: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    """Project a [d,d] matrix onto the per-block strictly-upper-triangular skew
    basis: for each diagonal block M_k take factor*(M_k - M_k^T)[rows,cols].

    Returns (nb, n_elems) matching the oft_R layout / triu_indices(b,b,1) order.
    """
    d = full.shape[0]
    nb = d // b
    # diagonal blocks: view [nb, b, nb, b] then pick blocks[k,:,k,:]
    blocks = full.reshape(nb, b, nb, b)
    idx = torch.arange(nb, device=full.device)
    diag = blocks[idx, :, idx, :]                     # (nb, b, b)
    skew = diag - diag.transpose(-1, -2)              # (nb, b, b)
    return factor * skew[:, rows.long(), cols.long()]  # (nb, n_elems)


class SingleStepPOETFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, weight, bias,
                perm_in_inv, perm_in, perm_out, perm_out_inv,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out):
        # R=I: chain collapses to a permuted GEMM. oft_R_in/oft_R_out are accepted
        # as inputs (their VALUES are unused here — they are 0) purely so autograd
        # routes the closed-form grads to them in backward.
        u = x.index_select(-1, perm_in_inv)
        v = u @ weight.t()
        if bias is not None:
            v = v + bias
        y = v.index_select(-1, perm_out)
        # Save ONLY x (one activation, same memory as a plain linear). u and the
        # rotation-frame weight grad A are recomputed in backward; v is never
        # needed (M_out uses the W@A form + bias term). bias may be None.
        ctx.save_for_backward(x, weight, bias, perm_in_inv, perm_in, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, weight, bias, perm_in_inv, perm_in, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out = ctx.block_size_in, ctx.block_size_out
        out_f, in_f = weight.shape

        Gv = grad_y.index_select(-1, perm_out_inv)        # un-permute output grad
        grad_x = (Gv @ weight).index_select(-1, perm_in)  # un-permute -> grad wrt x

        # Closed form, chain's right-multiply orientation (factor 2):
        #   A = u^T @ Gv  (rotation-frame weight grad);  M_in = A @ W;  M_out = W @ A (+bias)
        u = x.index_select(-1, perm_in_inv)
        Gv2 = Gv.reshape(-1, out_f)
        A = u.reshape(-1, in_f).t() @ Gv2                 # (in, out)
        M_in = A @ weight                                 # (in, in)
        M_out = weight @ A                                # (out, out)
        if bias is not None:
            M_out = M_out + torch.outer(bias, Gv2.sum(0))
        grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out)
        grad_oft_R_in = _blockdiag_skew_vec(M_in, bs_in, rows_in, cols_in)

        grad_oft_R_in = grad_oft_R_in.to(weight.dtype)
        grad_oft_R_out = grad_oft_R_out.to(weight.dtype)
        # 15 forward inputs -> 15 returns: real grads for x/oft_R_in/oft_R_out, then
        # 12 None (weight, bias, perm_in_inv, perm_in, perm_out, perm_out_inv,
        # rows_in, cols_in, rows_out, cols_out, block_size_in, block_size_out).
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None, None,
                None, None, None, None, None, None)


def _head_out_skew_vec(A, weight, bias, gsum, head_dim, rows, cols, factor=2.0):
    """Diagonal blocks of M_out = W@A (+ outer(bias,gsum)) over OUT blocks, block-local.

    A = x^T @ Gv  [in,out]; block k = W[k] @ A[:,k] with W[k] [head_dim, in] and
    A[:,k] [in, head_dim]. The bias rank-1 term is added per head as
    outer(bias_k, gsum_k). Never forms the full [out,out]. Returns (num_heads, n_elems).
    """
    out_f, in_f = weight.shape
    nb = out_f // head_dim
    w_blocks = weight.reshape(nb, head_dim, in_f)               # (nb, head_dim, in)
    a_cols = A.reshape(in_f, nb, head_dim).permute(1, 0, 2)     # (nb, in, head_dim)
    blocks = torch.bmm(w_blocks, a_cols)                        # (nb, hd, hd) = W_k @ A_:,k
    if bias is not None:
        bb = bias.reshape(nb, head_dim)
        gb = gsum.reshape(nb, head_dim)
        blocks = blocks + bb[:, :, None] * gb[:, None, :]       # + outer(bias_k, gsum_k)
    skew = blocks - blocks.transpose(-1, -2)
    return factor * skew[:, rows.long(), cols.long()]


def _head_in_skew_vec(A, weight, head_dim, rows, cols, factor=2.0):
    """Diagonal blocks of M_in = A @ W over IN blocks (heads), block-local.

    A = x^T @ Gv  [in, out]; block k = A[k] @ W[:,k] with A[k] [head_dim, out] and
    W[:,k] [out, head_dim]. Never forms the full [in,in].  Returns (num_heads, n_elems).
    """
    out_f, in_f = weight.shape
    nb = in_f // head_dim
    a_blocks = A.reshape(nb, head_dim, out_f)                       # (nb, head_dim, out)
    w_cols = weight.reshape(out_f, nb, head_dim).permute(1, 0, 2)   # (nb, out, head_dim)
    blocks = torch.bmm(a_blocks, w_cols)                            # (nb, head_dim, head_dim)
    skew = blocks - blocks.transpose(-1, -2)
    return factor * skew[:, rows.long(), cols.long()]


class HeadAlignedSingleStepFunction(torch.autograd.Function):
    """Single-step (R=I) fast path for HeadAlignedPOETLinear (permutation-free).

    Forward is a bare GEMM (no gathers). Backward saves ONLY x and (chain's
    right-multiply orientation, factor 2; Gv = grad_y since no perms) builds
    A = x^T@Gv once: head side -> block-local skew grad (batched per-head matmul,
    no full [d,d]); residual (dense, single block) side -> _blockdiag_skew_vec on
    the one full matrix. M_in = A@W, M_out = W@A + outer(bias, Gv.sum(0)).
    """

    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, weight, bias,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out, head_side):
        y = x @ weight.t()
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x, weight, bias, rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        ctx.head_side = head_side
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, weight, bias, rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out, head_side = ctx.block_size_in, ctx.block_size_out, ctx.head_side
        out_f, in_f = weight.shape

        grad_x = grad_y @ weight
        Gv2 = grad_y.reshape(-1, out_f)
        A = x.reshape(-1, in_f).t() @ Gv2                # (in, out) = x^T @ Gv
        gsum = Gv2.sum(0)

        if head_side == "out":
            # head side = OUT (block-diagonal, head_dim=bs_out); residual = IN (dense)
            grad_oft_R_out = _head_out_skew_vec(A, weight, bias, gsum, bs_out, rows_out, cols_out)
            grad_oft_R_in = _blockdiag_skew_vec(A @ weight, bs_in, rows_in, cols_in)
        else:  # head_side == "in"
            # head side = IN (block-diagonal, head_dim=bs_in); residual = OUT (dense)
            grad_oft_R_in = _head_in_skew_vec(A, weight, bs_in, rows_in, cols_in)
            M_out = weight @ A
            if bias is not None:
                M_out = M_out + torch.outer(bias, gsum)
            grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out)

        grad_oft_R_in = grad_oft_R_in.to(weight.dtype)
        grad_oft_R_out = grad_oft_R_out.to(weight.dtype)
        # 12 forward inputs -> 12 returns: real grads for x/oft_R_in/oft_R_out, then
        # 9 None (weight, bias, rows_in, cols_in, rows_out, cols_out,
        # block_size_in, block_size_out, head_side).
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None, None, None, None, None)
