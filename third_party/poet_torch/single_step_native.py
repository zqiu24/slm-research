"""Native-frame (gather-free) single-step fast path for standard POETLinear.

At oft_R=0 the conjugating perms cancel, so the layer's effective base is P_out·W·P_in
and the forward is a pure GEMM on the forward-frame weight W_eff = W[perm_out][:,perm_in]
(one O(d^2) relabel, vs the chain's five O(N*d) activation gathers). The backward is the
closed form (factor 2 = Cayley Jacobian at 0), with grad_x PLAIN and the oft_R grads
obtained by conjugating the small [d,d] gradient matrices into the block frame.

Storage stays NATURAL (un-permuted W, exactly as POETLinear stores it); the merge is
inherited unchanged. ONLY valid at oft_R=0 (merge_period=1) and parameterization=cayley
(the caller gates on both). Verified bit-against the chain in
/tmp/poet_native_frame_selfcheck.py.
"""
from __future__ import annotations

import torch

from .poet_layer import POETLinear
from .single_step import _blockdiag_skew_vec


def _conj(M, p):
    """Permutation conjugation M[p][:,p] (exact gather, no arithmetic)."""
    return M.index_select(0, p).index_select(1, p)


class NativeSingleStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, weight, bias,
                perm_in, perm_in_inv, perm_out, perm_out_inv,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out):
        # Forward-frame weight: one O(d^2) relabel; then a pure GEMM. oft_R_in/oft_R_out
        # are inputs only so autograd routes the closed-form grads to them.
        W_eff = weight.index_select(0, perm_out).index_select(1, perm_in)
        y = x @ W_eff.t()
        if bias is not None:
            y = y + bias.index_select(0, perm_out)
        ctx.save_for_backward(x, weight, bias, perm_in, perm_in_inv, perm_out, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, weight, bias, perm_in, perm_in_inv, perm_out, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out = ctx.block_size_in, ctx.block_size_out
        out_f, in_f = weight.shape

        W_eff = weight.index_select(0, perm_out).index_select(1, perm_in)
        grad_x = grad_y @ W_eff                                    # PLAIN — no gather
        G = x.reshape(-1, in_f).t() @ grad_y.reshape(-1, out_f)    # [in, out]
        M_in = _conj(G @ W_eff, perm_in_inv)                       # [in, in] block frame
        M_out_nat = W_eff @ G                                      # [out, out]
        if bias is not None:
            b_eff = bias.index_select(0, perm_out)
            M_out_nat = M_out_nat + torch.outer(b_eff, grad_y.reshape(-1, out_f).sum(0))
        M_out = _conj(M_out_nat, perm_out_inv)
        grad_oft_R_in = _blockdiag_skew_vec(M_in, bs_in, rows_in, cols_in).to(weight.dtype)
        grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out).to(weight.dtype)
        # 15 inputs -> 15 returns: grads for x/oft_R_in/oft_R_out, then 12 None.
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None, None,
                None, None, None, None, None, None)


class SingleStepPOETLinear(POETLinear):
    """POETLinear that uses the gather-free native-frame single-step forward.

    Identical to POETLinear in every way (natural weight storage, perm/oft_R buffers,
    the inherited merge_then_reinitialize / _fold_with_R) EXCEPT the forward, which
    routes to NativeSingleStepFunction. Valid only at oft_R=0 (merge_period=1, cayley).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The native backward hard-codes the factor-2 Cayley Jacobian and has NO
        # chain fallback, so refuse exp loudly rather than silently produce wrong
        # grads (build-time validation also forbids it; this guards direct use).
        if self.parameterization != "cayley":
            raise ValueError(
                "SingleStepPOETLinear requires parameterization='cayley'; "
                f"got {self.parameterization!r}."
            )

    def forward(self, x):
        return NativeSingleStepFunction.apply(
            x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
            self.perm_in, self.perm_in_inv, self.perm_out, self.perm_out_inv,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out,
        )
