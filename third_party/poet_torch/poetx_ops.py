"""Forward-frame (perm-free-forward) single-step ops for POETX.

POET's chain is the conjugation W_eff = (P_out R_out P_outᵀ)·W·(P_in R_in P_inᵀ); the
stored pl.weight is the pre-folded W_perm = P_outᵀ W P_in, so at oft_R=0 the effective
weight is W = P_out W_perm P_inᵀ = W_perm[perm_out][:,perm_in]. POETXSingleStepFunction
takes that EFFECTIVE weight Wx and the effective bias bias_eff = bias[perm_out] DIRECTLY,
so the forward is a bare GEMM (no permutation) and grad_x is plain. The 2 conj on the
small [d,d] gradient matrices remain (mathematically forced cross-frame relabel for the
oft_R gradient; backward only, never touches activations). Same closed form as
NativeSingleStepFunction (factor 2 = Cayley Jacobian at 0), with Wx/bias_eff read from
storage instead of rebuilt. ONLY valid at oft_R=0 (merge_period=1) and cayley.

(Lives in poetx_ops.py — the name poet_ops.py is already taken by the vendored
Triton chain-layer kernels, which poet_layer.py pulls in via `from .poet_ops import *`;
keeping this standalone avoids touching that module and any namespace bleed.)
"""
from __future__ import annotations

import torch

from .single_step import _blockdiag_skew_vec


def _conj(M, p):
    """Permutation conjugation M[p][:,p] (exact gather, no arithmetic)."""
    return M.index_select(0, p).index_select(1, p)


class POETXSingleStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, Wx, bias_eff,
                perm_in_inv, perm_out_inv,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out):
        # Wx is ALREADY the forward-frame (effective) weight -> bare GEMM, zero perm.
        # Only the INVERSE perms are needed (in the backward conj); the perm-free
        # forward uses none. oft_R_in/oft_R_out are inputs only so autograd routes
        # the closed-form grads to them.
        y = x @ Wx.t()
        if bias_eff is not None:
            y = y + bias_eff
        ctx.save_for_backward(x, Wx, bias_eff, perm_in_inv, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, Wx, bias_eff, perm_in_inv, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out = ctx.block_size_in, ctx.block_size_out
        out_f, in_f = Wx.shape

        grad_x = grad_y @ Wx                                       # PLAIN — no gather
        G = x.reshape(-1, in_f).t() @ grad_y.reshape(-1, out_f)    # [in, out]
        M_in = _conj(G @ Wx, perm_in_inv)                          # [in, in] block frame
        M_out_nat = Wx @ G                                         # [out, out]
        if bias_eff is not None:
            M_out_nat = M_out_nat + torch.outer(bias_eff, grad_y.reshape(-1, out_f).sum(0))
        M_out = _conj(M_out_nat, perm_out_inv)
        grad_oft_R_in = _blockdiag_skew_vec(M_in, bs_in, rows_in, cols_in).to(Wx.dtype)
        grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out).to(Wx.dtype)
        # 13 inputs -> 13 returns: grads for x/oft_R_in/oft_R_out, then 10 None.
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None,
                None, None, None, None, None)


class AlternatingPOETXSingleStepFunction(torch.autograd.Function):
    """Single-side POETX backward. Identical bare-GEMM forward as
    POETXSingleStepFunction, but the backward computes ONLY the active side's
    rotation-gradient (skipping the frozen side's d^3 M GEMM) and returns a
    shape-correct ZEROS gradient for the frozen side (so Megatron's grad buffer
    never stalls). `active` is "in" or "out"."""

    @staticmethod
    def forward(ctx, x, oft_R_in, oft_R_out, Wx, bias_eff,
                perm_in_inv, perm_out_inv,
                rows_in, cols_in, rows_out, cols_out,
                block_size_in, block_size_out, active):
        y = x @ Wx.t()
        if bias_eff is not None:
            y = y + bias_eff
        ctx.save_for_backward(x, Wx, bias_eff, perm_in_inv, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.block_size_in = block_size_in
        ctx.block_size_out = block_size_out
        ctx.active = active
        ctx.oft_R_in_shape = tuple(oft_R_in.shape)
        ctx.oft_R_out_shape = tuple(oft_R_out.shape)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x, Wx, bias_eff, perm_in_inv, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        bs_in, bs_out = ctx.block_size_in, ctx.block_size_out
        out_f, in_f = Wx.shape
        active = ctx.active

        grad_x = grad_y @ Wx  # PLAIN — always needed (upstream gradient)
        G = x.reshape(-1, in_f).t() @ grad_y.reshape(-1, out_f)  # [in, out]
        if active == "in":
            M_in = _conj(G @ Wx, perm_in_inv)
            grad_oft_R_in = _blockdiag_skew_vec(M_in, bs_in, rows_in, cols_in).to(Wx.dtype)
            grad_oft_R_out = torch.zeros(ctx.oft_R_out_shape, dtype=Wx.dtype, device=Wx.device)
        else:  # "out"
            M_out_nat = Wx @ G
            if bias_eff is not None:
                M_out_nat = M_out_nat + torch.outer(bias_eff, grad_y.reshape(-1, out_f).sum(0))
            M_out = _conj(M_out_nat, perm_out_inv)
            grad_oft_R_out = _blockdiag_skew_vec(M_out, bs_out, rows_out, cols_out).to(Wx.dtype)
            grad_oft_R_in = torch.zeros(ctx.oft_R_in_shape, dtype=Wx.dtype, device=Wx.device)
        # 14 inputs -> 14 returns: grads for x/oft_R_in/oft_R_out, then 11 None.
        return (grad_x, grad_oft_R_in, grad_oft_R_out,
                None, None, None, None, None,
                None, None, None, None, None, None)
