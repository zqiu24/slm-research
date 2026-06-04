"""HeadAlignedPOETLinear: a POETLinear whose head-structured side is rotated
per attention head.

One side (the "head side") uses block_size = head_dim with a FIXED identity
permutation (block j is head j; Psi is NEVER resampled), so the rotation is
block-diagonal per head with no cross-head mixing and no permutation. The other
("residual") side is an ordinary POET rotation: block size from
resid_block_size / resid_block_count, permutation resampled at merge unless
resid_permute=False. BOTH sides train.

head_side="out": query/key/value projections (rows = heads).
head_side="in" : attention output projection (cols = heads).

Subclasses POETLinear to reuse _build_R / _merge_R / forward / the fused kernels;
only the constructor (asymmetric per-side block spec + identity head Psi) and
merge_then_reinitialize (resample the residual side only) differ.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .poet_layer import POETLinear, block_diag_lr_matmul_decoupled


class HeadAlignedPOETLinear(POETLinear):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        head_side,
        head_dim,
        resid_block_size=None,
        resid_block_count=None,
        resid_permute=True,
        bias=False,
        device=None,
        dtype=None,
        parameterization="cayley",
        mem_efficient_mode=None,
    ):
        nn.Module.__init__(self)
        if head_side not in ("in", "out"):
            raise ValueError(f"head_side must be 'in' or 'out', got {head_side!r}")
        if (resid_block_size is None) == (resid_block_count is None):
            raise ValueError("exactly one of resid_block_size or resid_block_count must be set")
        if parameterization not in ("cayley", "exp"):
            raise ValueError(f"parameterization must be 'cayley' or 'exp', got {parameterization!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.head_side = head_side
        self.head_dim = head_dim
        self.resid_permute = bool(resid_permute)

        head_features = out_features if head_side == "out" else in_features
        resid_features = in_features if head_side == "out" else out_features
        if head_features % head_dim != 0:
            raise ValueError(f"head_dim {head_dim} doesn't divide the head-side dim {head_features}")
        if resid_block_count is not None:
            if resid_features % resid_block_count != 0:
                raise ValueError(
                    f"resid_block_count {resid_block_count} doesn't divide residual dim {resid_features}"
                )
            resid_bs = resid_features // resid_block_count
        else:
            if resid_features % resid_block_size != 0:
                raise ValueError(
                    f"resid_block_size {resid_block_size} doesn't divide residual dim {resid_features}"
                )
            resid_bs = resid_block_size

        if head_side == "out":
            block_size_out, block_size_in = head_dim, resid_bs
        else:
            block_size_in, block_size_out = head_dim, resid_bs
        self.block_size_in = block_size_in
        self.block_size_out = block_size_out
        self.block_size = block_size_in  # back-compat (merge/"is-active" guards)
        self.head_count = head_features // head_dim

        if mem_efficient_mode is None:
            mem_efficient_mode = (parameterization == "exp") or os.environ.get("POET_MEM_EFFICIENT") == "1"
        self.mem_efficient_mode = mem_efficient_mode
        self.parameterization = parameterization

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
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

        # Head side: identity Psi (never resampled). Residual side: random Psi
        # unless resid_permute=False (then identity, never resampled).
        out_identity = (head_side == "out") or not self.resid_permute
        in_identity = (head_side == "in") or not self.resid_permute
        perm_out = self._make_perm(out_features, out_identity, device)
        perm_in = self._make_perm(in_features, in_identity, device)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))

    @staticmethod
    def _make_perm(n, identity, device):
        if identity:
            return torch.arange(n, device=device, dtype=torch.int32)
        return torch.randperm(n, device=device).to(torch.int32)

    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        R_out, R_in = self._merge_R()
        W = self.weight.detach().clone()
        tmp = block_diag_lr_matmul_decoupled(R_in, W.t(), R_out)
        tmp = tmp.index_select(0, self.perm_in)
        tmp = tmp.index_select(1, self.perm_out)
        expected = tmp.t()

        # Resample ONLY the residual side, and only when reinit_perm & resid_permute.
        # The head side keeps its identity Psi forever. When a side does not
        # resample, re-permute back into the CURRENT layout (stock fold-only path).
        out_resamples = reinit_perm and self.resid_permute and (self.head_side == "in")
        in_resamples = reinit_perm and self.resid_permute and (self.head_side == "out")
        device = self.weight.device

        if out_resamples:
            new_perm_out = torch.randperm(self.out_features, device=device).to(torch.int32)
            new_perm_out_inv = torch.argsort(new_perm_out).to(torch.int32)
        else:
            new_perm_out, new_perm_out_inv = self.perm_out, self.perm_out_inv
        if in_resamples:
            new_perm_in = torch.randperm(self.in_features, device=device).to(torch.int32)
            new_perm_in_inv = torch.argsort(new_perm_in).to(torch.int32)
        else:
            new_perm_in, new_perm_in_inv = self.perm_in, self.perm_in_inv

        expected = expected.index_select(0, new_perm_out_inv).index_select(1, new_perm_in_inv)
        self.weight.detach().copy_(expected)
        self.perm_out.copy_(new_perm_out)
        self.perm_out_inv.copy_(new_perm_out_inv)
        self.perm_in.copy_(new_perm_in)
        self.perm_in_inv.copy_(new_perm_in_inv)
        self.oft_R_in.zero_()
        self.oft_R_out.zero_()
