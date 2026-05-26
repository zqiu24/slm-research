"""
POETCayleyLinear – POET layer with direct orthogonal R matrices.

Instead of storing skew-symmetric parameters and computing
Cayley-Neumann in the forward pass, this layer stores R_in and R_out
directly as block-diagonal orthogonal matrices of shape (r, bsz, bsz).

Orthogonality is maintained by the CayleyAdam optimizer which performs
updates on the Stiefel manifold.

Forward:  y = permute_out( (permute_in(x) @ R_in) @ W^T @ R_out )
  – pure block-diagonal matmul, no Cayley transform in the hot path.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block-diagonal matmul helpers  (same as poet_layer.py)
# ---------------------------------------------------------------------------

def torch_bmm(x: torch.Tensor, R: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply block-diagonal R to x:  x @ block_diag(R)."""
    Bdims = x.shape[:-1]
    xr = x.view(*Bdims, -1, block_size)
    xr = torch.einsum("...rk,rkc->...rc", xr, R)
    return xr.contiguous().view(*Bdims, -1)


def block_diag_lr_matmul(
    A_blocks: torch.Tensor, W: torch.Tensor, B_blocks: torch.Tensor,
) -> torch.Tensor:
    """
    Compute  block_diag(A) @ W @ block_diag(B)  without materializing full matrices.

    A_blocks: (r_m, b, b)
    W:        (M, N)  with M = r_m*b, N = r_n*b
    B_blocks: (r_n, b, b)
    """
    r_m, b, _ = A_blocks.shape
    r_n = B_blocks.shape[0]
    M, N = W.shape

    if A_blocks.device != W.device or A_blocks.dtype != W.dtype:
        A_blocks = A_blocks.to(device=W.device, dtype=W.dtype)
    if B_blocks.device != W.device or B_blocks.dtype != W.dtype:
        B_blocks = B_blocks.to(device=W.device, dtype=W.dtype)

    W_blocks = W.view(r_m, b, r_n, b).transpose(1, 2)      # (r_m, r_n, b, b)
    left = torch.matmul(A_blocks.unsqueeze(1), W_blocks)     # (r_m, r_n, b, b)
    out_blocks = torch.matmul(left, B_blocks.unsqueeze(0))   # (r_m, r_n, b, b)
    return out_blocks.permute(0, 2, 1, 3).contiguous().view(M, N)


# ---------------------------------------------------------------------------
# Permutation autograd function (same as poet_layer.py)
# ---------------------------------------------------------------------------

class PermutationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, perm, inv_perm):
        ctx.save_for_backward(inv_perm, perm)
        return x[..., perm]

    @staticmethod
    def backward(ctx, grad_output):
        inv_perm, perm = ctx.saved_tensors
        return grad_output[..., inv_perm], None, None


def permute_x(x, perm, inv_perm):
    return PermutationFunction.apply(x, perm, inv_perm)


# ---------------------------------------------------------------------------
# Compiled forward (no Cayley-Neumann – just block-diagonal matmul)
# ---------------------------------------------------------------------------

@torch.compile(fullgraph=True)
def forward_core_cayley(
    x: torch.Tensor,
    R_in: torch.Tensor,      # (r_in, bsz, bsz) orthogonal blocks
    R_out: torch.Tensor,     # (r_out, bsz, bsz) orthogonal blocks
    block_size: int,
    perm_in: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    # Permute input
    x = permute_x(x, perm_in_inv, perm_in)

    # x @ R_in (block-diagonal)
    x = torch_bmm(x, R_in, block_size)

    # x @ W^T + bias
    y = x @ base_weight.t()
    if base_bias is not None:
        y = y + base_bias

    # y @ R_out (block-diagonal)
    y = torch_bmm(y, R_out, block_size)

    # Un-permute output
    y = permute_x(y, perm_out, perm_out_inv)
    return y


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class POETCayleyLinear(nn.Module):
    """
    POET linear layer with direct orthogonal R matrices.

    Parameters are:
      - weight:  (out_features, in_features) – frozen base weight
      - R_in:    (r_in, bsz, bsz)  – trainable orthogonal blocks (input side)
      - R_out:   (r_out, bsz, bsz) – trainable orthogonal blocks (output side)

    The optimizer (CayleyAdam) is responsible for keeping R_in / R_out orthogonal.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bsz: int = 256,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = bsz

        # Frozen base weight
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        # Orthogonal block-diagonal matrices (trainable, kept on Stiefel manifold by optimizer)
        r_in = in_features // bsz
        r_out = out_features // bsz
        self.r_in = r_in
        self.r_out = r_out

        # Initialize as identity (= no transformation)
        R_in_init = torch.eye(bsz, device=device, dtype=dtype).unsqueeze(0).expand(r_in, -1, -1).clone()
        R_out_init = torch.eye(bsz, device=device, dtype=dtype).unsqueeze(0).expand(r_out, -1, -1).clone()
        self.R_in = nn.Parameter(R_in_init)
        self.R_out = nn.Parameter(R_out_init)

        # Permutation buffers
        perm_in = torch.randperm(in_features, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features, device=device, dtype=torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    # ------------------------------------------------------------------
    # Permutation helpers
    # ------------------------------------------------------------------

    def perform_permutation(self) -> None:
        """Fold current permutation into the weight so forward avoids extra indexing."""
        W = self.weight
        Wp = W.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
        self.weight.detach().copy_(Wp)

    def update_permutation(self):
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(torch.argsort(perm_in).to(torch.int32))
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(torch.argsort(perm_out).to(torch.int32))
        self.perform_permutation()

    # ------------------------------------------------------------------
    # Merge + reinitialize  (same math as POETLinear)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def merge_then_reinitialize(self) -> None:
        R_in = self.R_in.data   # (r_in, bsz, bsz)
        R_out = self.R_out.data  # (r_out, bsz, bsz)

        # W_new^T = R_in @ W^T @ R_out  (block-diagonal on both sides)
        W = self.weight.detach().clone()
        tmp = W.t()
        tmp = block_diag_lr_matmul(R_in, tmp, R_out)

        # Apply permutations
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        # Generate NEW permutation BEFORE writing back
        device = self.weight.device
        perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
        perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
        perm_in_inv = torch.argsort(perm_in).to(torch.int32)
        perm_out_inv = torch.argsort(perm_out).to(torch.int32)

        # Fold new permutation into weight
        expected = expected.index_select(0, perm_out_inv).index_select(1, perm_in_inv)
        self.weight.detach().copy_(expected)

        # Update buffers
        self.perm_in.copy_(perm_in)
        self.perm_in_inv.copy_(perm_in_inv)
        self.perm_out.copy_(perm_out)
        self.perm_out_inv.copy_(perm_out_inv)

        # Reset R to identity
        bsz = self.block_size
        eye = torch.eye(bsz, device=device, dtype=self.R_in.dtype)
        self.R_in.data.copy_(eye.unsqueeze(0).expand_as(self.R_in))
        self.R_out.data.copy_(eye.unsqueeze(0).expand_as(self.R_out))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return forward_core_cayley(
            x,
            self.R_in,
            self.R_out,
            self.block_size,
            self.perm_in,
            self.perm_in_inv,
            self.perm_out,
            self.perm_out_inv,
            self.weight,
            self.bias,
        )


# ---------------------------------------------------------------------------
# Model surgery: replace nn.Linear -> POETCayleyLinear
# ---------------------------------------------------------------------------

def replace_linear_with_poet_cayley(
    module: nn.Module,
    block_size: int,
    init_type: str,
    mup_alpha: float = 1.0,
    device=None,
    dtype=None,
) -> None:
    """
    Recursively replace every nn.Linear (except lm_head) with POETCayleyLinear.
    Same interface as ``replace_linear_with_poet`` in poet_layer.py.
    """

    def _convert(m: nn.Module):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and "lm_head" not in name.lower():
                if block_size and child.in_features % block_size == 0 and child.out_features % block_size == 0:
                    new_lin = POETCayleyLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bsz=block_size,
                        bias=(child.bias is not None),
                        device=device,
                        dtype=dtype,
                    )
                    with torch.no_grad():
                        if init_type == "normalized":
                            child.weight.data = child.weight.data / torch.norm(
                                child.weight.data, dim=1, keepdim=True
                            )
                        elif init_type == "mup_normalized":
                            d_in = torch.tensor(child.weight.data.shape[1])
                            d_out = torch.tensor(child.weight.data.shape[0])
                            normed = child.weight.data / torch.norm(
                                child.weight.data, dim=1, keepdim=True
                            )
                            target_spec = mup_alpha * torch.sqrt(d_out / d_in)
                            current_spec = torch.linalg.norm(normed.float(), ord=2).item()
                            scaling = (target_spec / current_spec).to(
                                dtype=normed.dtype, device=normed.device
                            )
                            child.weight.data = normed * scaling

                        new_lin.weight.copy_(child.weight.detach().to(new_lin.weight.dtype))
                        if child.bias is not None and new_lin.bias is not None:
                            new_lin.bias.copy_(child.bias.detach().to(new_lin.bias.dtype))
                    setattr(m, name, new_lin)
                else:
                    raise ValueError(
                        f"Layer {name} has in_features {child.in_features} and "
                        f"out_features {child.out_features}, not divisible by {block_size}"
                    )
            else:
                _convert(child)

    _convert(module)


# ---------------------------------------------------------------------------
# Merge check  (same interface as check_and_merge in poet_layer.py)
# ---------------------------------------------------------------------------

def check_and_merge_cayley(model: nn.Module, iter_count: int = 0, poet_reset_gap: int = 4):
    if iter_count <= 0 or (iter_count % poet_reset_gap != 0):
        return

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    with torch.compiler.set_stance("eager_then_compile"):
        for name, module in model.named_modules():
            if isinstance(module, POETCayleyLinear) and module.block_size > 0:
                with torch.no_grad():
                    if rank == 0:
                        module.merge_then_reinitialize()

                    # Broadcast to all ranks
                    torch.distributed.broadcast(module.R_in.data, src=0)
                    torch.distributed.broadcast(module.R_out.data, src=0)
                    torch.distributed.broadcast(module.weight.data, src=0)
                    if module.bias is not None:
                        torch.distributed.broadcast(module.bias.data, src=0)
                    torch.distributed.broadcast(module.perm_in, src=0)
                    torch.distributed.broadcast(module.perm_in_inv, src=0)
                    torch.distributed.broadcast(module.perm_out, src=0)
                    torch.distributed.broadcast(module.perm_out_inv, src=0)

    if is_dist:
        dist.barrier()
