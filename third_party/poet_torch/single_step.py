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
