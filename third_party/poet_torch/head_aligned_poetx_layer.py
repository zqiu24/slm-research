"""HeadAlignedPOETXLinear: head-aligned attention rotation on the POETX forward
frame. A thin POETXLinear subclass that sets ASYMMETRIC block sizes and perms:

  * head side (out for q/k/v, in for o): block = head_dim, perm = IDENTITY
    (block j is always head j -- no cross-head mixing).
  * residual side (the hidden_size side): block_count = head_resid_block_count
    (> 1), perm = RANDOM Ψ -- a real permuted multi-block rotation, the thing the
    legacy perm-free HeadAlignedPOETLinear cannot express.

All compute (forward / backward / merge, incl. the alternating active-only fold)
is INHERITED from POETXLinear -- it is already perm-aware (perm_*_inv in the
backward conj) and block-aware (decoupled block_size_in/out). Only __init__ differs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .poetx_layer import POETXLinear


class HeadAlignedPOETXLinear(POETXLinear):
    def __init__(self, in_features, out_features, *, head_side, head_dim,
                 head_resid_block_count, bias=False, device=None, dtype=None,
                 parameterization="cayley", alternating=False, alternate_every=1):
        nn.Module.__init__(self)
        if head_side not in ("in", "out"):
            raise ValueError(f"head_side must be 'in' or 'out', got {head_side!r}")
        if parameterization != "cayley":
            raise ValueError(
                "HeadAlignedPOETXLinear requires parameterization='cayley' "
                f"(POETX backward is Cayley-specific); got {parameterization!r}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.parameterization = parameterization
        self.single_step_fast = False
        self.head_side = head_side
        self.head_dim = head_dim

        head_features = out_features if head_side == "out" else in_features
        resid_features = in_features if head_side == "out" else out_features
        if head_features % head_dim != 0:
            raise ValueError(f"head_dim {head_dim} doesn't divide head-side dim {head_features}")
        if resid_features % head_resid_block_count != 0:
            raise ValueError(
                f"head_resid_block_count {head_resid_block_count} doesn't divide "
                f"residual dim {resid_features}"
            )
        resid_bs = resid_features // head_resid_block_count
        if head_side == "out":
            block_size_out, block_size_in = head_dim, resid_bs
        else:
            block_size_in, block_size_out = head_dim, resid_bs
        self.block_size_in = block_size_in
        self.block_size_out = block_size_out
        self.block_size = block_size_in  # back-compat (merge "is-active" guard)

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        r_in = in_features // block_size_in
        r_out = out_features // block_size_out
        n_elems_in = block_size_in * (block_size_in - 1) // 2
        n_elems_out = block_size_out * (block_size_out - 1) // 2
        self.oft_R_in = nn.Parameter(torch.zeros((r_in, n_elems_in), device=device, dtype=dtype))
        self.oft_R_out = nn.Parameter(torch.zeros((r_out, n_elems_out), device=device, dtype=dtype))
        self.r_in, self.r_out = r_in, r_out

        rows_in, cols_in = torch.triu_indices(block_size_in, block_size_in, 1, device=device)
        self.register_buffer("rows_in", rows_in.to(torch.int32))
        self.register_buffer("cols_in", cols_in.to(torch.int32))
        rows_out, cols_out = torch.triu_indices(block_size_out, block_size_out, 1, device=device)
        self.register_buffer("rows_out", rows_out.to(torch.int32))
        self.register_buffer("cols_out", cols_out.to(torch.int32))

        # Head side: identity perm (block j == head j). Residual side: random Ψ.
        if head_side == "out":
            perm_out = torch.arange(out_features, device=device, dtype=torch.int32)
            perm_in = torch.randperm(in_features, device=device).to(torch.int32)
        else:
            perm_in = torch.arange(in_features, device=device, dtype=torch.int32)
            perm_out = torch.randperm(out_features, device=device).to(torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

        # Alternating active-only merge is inherited from POETXLinear (prereq plan).
        self.alternating = bool(alternating)
        self.alternate_every = max(1, int(alternate_every))
        # self.weight is the (empty) W_perm-frame tensor; bake_perms_into_weight()
        # converts it to the forward frame once the real weight is copied in (the
        # walk calls it after _copy_and_init_weight), exactly as for POETXLinear.
