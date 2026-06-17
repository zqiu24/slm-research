"""Block-sparse, expert-batched POETX rotation gradients (forward-frame, oft_R=0).

Replaces the per-expert pair of full [d,d] M GEMMs in POETXSingleStepFunction.backward
with two batched-block bmms: only the block-diagonal of the conjugated M is computed,
batched over (experts x blocks). Bit-identical (same summation order over the contracted
index) to per-expert _blockdiag_skew_vec(_conj(...)). CPU-safe: no megatron, no CUDA-only
ops; pure torch."""
from __future__ import annotations

import torch


def _grouped_blockdiag_skew_vecs(G, Wx, perm_in_inv, perm_out_inv,
                                 bs_in, bs_out, rows_in, cols_in, rows_out, cols_out):
    E, in_f, out_f = G.shape
    nb_in, nb_out = in_f // bs_in, out_f // bs_out
    ri, ci = rows_in.long(), cols_in.long()
    ro, co = rows_out.long(), cols_out.long()

    # ---- M_in: block-diagonal blocks of (G @ Wx)[pin][:, pin] ----
    pin = perm_in_inv.long()                                            # [E, in]
    G_sel = torch.gather(G, 1, pin.unsqueeze(-1).expand(E, in_f, out_f))
    G_sel = G_sel.reshape(E * nb_in, bs_in, out_f)                      # [E*nb, b, out]
    W_sel = torch.gather(Wx, 2, pin.unsqueeze(1).expand(E, out_f, in_f))
    W_sel = (W_sel.reshape(E, out_f, nb_in, bs_in)
                  .permute(0, 2, 1, 3).reshape(E * nb_in, out_f, bs_in))  # [E*nb, out, b]
    M_in = torch.bmm(G_sel, W_sel)                                     # [E*nb, b, b]
    skew_in = M_in - M_in.transpose(-1, -2)
    grad_in = (2.0 * skew_in[:, ri, ci]).reshape(E, nb_in, -1).to(Wx.dtype)

    # ---- M_out: block-diagonal blocks of (Wx @ G)[pout][:, pout] ----
    pout = perm_out_inv.long()                                         # [E, out]
    W2 = torch.gather(Wx, 1, pout.unsqueeze(-1).expand(E, out_f, in_f))
    W2 = W2.reshape(E * nb_out, bs_out, in_f)                          # [E*nb, b, in]
    G2 = torch.gather(G, 2, pout.unsqueeze(1).expand(E, in_f, out_f))
    G2 = (G2.reshape(E, in_f, nb_out, bs_out)
            .permute(0, 2, 1, 3).reshape(E * nb_out, in_f, bs_out))    # [E*nb, in, b]
    M_out = torch.bmm(W2, G2)                                          # [E*nb, b, b]
    skew_out = M_out - M_out.transpose(-1, -2)
    grad_out = (2.0 * skew_out[:, ro, co]).reshape(E, nb_out, -1).to(Wx.dtype)
    return grad_in, grad_out


class GroupedPOETXFunction(torch.autograd.Function):
    """Forward-frame, all-experts POETX. Forward: ragged per-expert bare GEMM. Backward:
    plain grad_x + Adam-equivalent G, then the block-sparse expert-batched rotation grad.
    Bias is unsupported (experts are bias-free); pass bias-free weights only."""

    @staticmethod
    def forward(ctx, concat_x, oft_in, oft_out, Wx,
                perm_in_inv, perm_out_inv, rows_in, cols_in, rows_out, cols_out,
                bs_in, bs_out, sizes):
        E = len(sizes)
        x_list = torch.split(concat_x, list(sizes), dim=0)
        y = torch.cat([x_list[e] @ Wx[e].t() for e in range(E)], dim=0)
        ctx.save_for_backward(concat_x, Wx, perm_in_inv, perm_out_inv,
                              rows_in, cols_in, rows_out, cols_out)
        ctx.sizes = tuple(sizes)
        ctx.bs_in, ctx.bs_out = bs_in, bs_out
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (concat_x, Wx, perm_in_inv, perm_out_inv,
         rows_in, cols_in, rows_out, cols_out) = ctx.saved_tensors
        sizes, bs_in, bs_out = ctx.sizes, ctx.bs_in, ctx.bs_out
        E, out_f, in_f = Wx.shape
        x_list = torch.split(concat_x, list(sizes), dim=0)
        gy_list = torch.split(grad_y, list(sizes), dim=0)
        grad_x = torch.cat([gy_list[e] @ Wx[e] for e in range(E)], dim=0)
        G = torch.stack([
            x_list[e].reshape(-1, in_f).t() @ gy_list[e].reshape(-1, out_f)
            for e in range(E)
        ])                                                            # [E, in, out]
        grad_in, grad_out = _grouped_blockdiag_skew_vecs(
            G, Wx, perm_in_inv, perm_out_inv, bs_in, bs_out,
            rows_in, cols_in, rows_out, cols_out)
        # 13 inputs -> 13 returns (grads for concat_x/oft_in/oft_out, then 10 None).
        return (grad_x, grad_in, grad_out,
                None, None, None, None, None, None, None, None, None, None)
