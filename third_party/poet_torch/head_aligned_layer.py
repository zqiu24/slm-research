"""HeadAlignedPOETLinear: a POETLinear whose head-structured side is rotated
per attention head.

One side (the "head side") uses block_size = head_dim, so the rotation is
block-diagonal per head with no cross-head mixing. The other ("residual") side
is an ordinary block rotation: block size from resid_block_size /
resid_block_count. BOTH sides train.

head_side="out": query/key/value projections (rows = heads).
head_side="in" : attention output projection (cols = heads).

Permutation-free
----------------
The stock POETLinear conjugates each block rotation by a permutation Ψ
(``permute(x) -> block-rotate -> ... -> permute(y)``) so that, across merge
cycles, a block-diagonal rotation can mix *different* neuron pairs and build up a
richer-than-block-diagonal orthogonal transform. That machinery is dead weight
here:

* The head side is identity by design — block j is *always* head j, never a
  resampled set of features. That is the whole point of "head-aligned": no
  cross-head mixing, ever.
* The residual side, in every deployed config, is a single dense block
  (``block_count=1``). A permutation conjugating one dense block, ``Ψ R Ψᵀ``, is
  just another dense orthogonal matrix — Ψ adds no expressivity and nothing to
  resample.

So neither side needs Ψ. Rather than inherit POETLinear's permute → rotate →
permute chain (two gathers + their scatter-add backwards per layer per
microbatch — the single largest cost in the POET fwd/bwd), this subclass
overrides ``forward`` and ``merge_then_reinitialize`` with permutation-free twins
(``chain_noperm`` / fold without ``index_select``). The math is exactly the stock
POET path specialized to identity permutations, so the two stay consistent and
the fold is still spectrum-preserving.

The ``perm_*`` buffers are still registered as fixed identity (and never
consulted by the compute path) so the distributed-checkpoint state dict and the
merge-step broadcast in ``src/patches/poet_merge_step.py`` keep working unchanged.
``resid_permute`` is accepted for call-site/API parity but is now a no-op: the
layer is always permutation-free.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .poet_layer import (
    POETLinear,
    block_diag_lr_matmul_decoupled,
    get_weight_poet_decoupled,
    get_weight_poet_decoupled_exp,
)


def chain_noperm(x, Rin, weight, bias, Rout, bsz_in, bsz_out):
    """Permutation-free twin of ``chain_layer_x_fast_decoupled``.

    Identical block-rotate → dense matmul → block-rotate chain, but with the two
    ``PermutationFunction.apply`` gathers (input and output) dropped. Plain ops,
    so autograd saves the cheap block-rotation activations and ``torch.compile``
    can fuse it.
    """
    leading_shape = x.shape[:-1]
    Din = x.shape[-1]
    N = x.numel() // Din
    rin = Rin.size(0)
    rout = Rout.size(0)

    xb_r = x.reshape(N, rin, bsz_in).transpose(0, 1)        # [rin, N, b_in]
    xR = torch.bmm(xb_r, Rin).transpose(0, 1).reshape(N, rin * bsz_in)
    yb_flat = xR @ weight.t()
    if bias is not None:
        yb_flat = yb_flat + bias
    yb_r = yb_flat.view(N, rout, bsz_out).transpose(0, 1)   # [rout, N, b_out]
    y = torch.bmm(yb_r, Rout).transpose(0, 1).reshape(*leading_shape, rout * bsz_out)
    return y


def _forward_noperm_eager(
    x,
    oft_R_in,
    oft_R_out,
    block_size_in,
    block_size_out,
    rows_in,
    cols_in,
    rows_out,
    cols_out,
    weight,
    bias,
    mem_efficient_mode,
    use_exp,
):
    """Build the two block rotations then run the permutation-free chain.

    Mirrors POETLinear's ``_forward_core_decoupled_eager`` (same R build, same
    fast/mem-efficient split) minus the permutation arguments.
    """
    if use_exp:
        R_out, R_in = get_weight_poet_decoupled_exp(
            oft_R_in, oft_R_out, block_size_in, block_size_out,
            rows_in, cols_in, rows_out, cols_out,
        )
    else:
        R_out, R_in = get_weight_poet_decoupled(
            oft_R_in, oft_R_out, block_size_in, block_size_out,
            rows_in, cols_in, rows_out, cols_out,
        )
    if mem_efficient_mode:
        # Recompute the chain in the backward (saves activation memory). The exp
        # path and any POET_MEM_EFFICIENT run route here, eagerly.
        from torch.utils.checkpoint import checkpoint

        y = checkpoint(
            chain_noperm, x, R_in, weight, bias, R_out, block_size_in, block_size_out,
            use_reentrant=False,
        )
    else:
        y = chain_noperm(x, R_in, weight, bias, R_out, block_size_in, block_size_out)
    return y


# Compiled (fused) entry used during TRAINING (grad enabled) on the Cayley fast
# path, matching POETLinear.forward_core_decoupled. mem_efficient_mode/use_exp are
# only ever passed False here (those branches run eager), so neither the
# checkpoint nor the matrix_exp branch is traced.
forward_core_noperm = torch.compile(_forward_noperm_eager, fullgraph=True)


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
        # Accepted for call-site/API parity; the layer is always permutation-free
        # (see module docstring), so this no longer gates any permutation.
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

        # Permutation-free: Ψ is fixed identity on BOTH sides and never consulted
        # by forward/merge. The buffers are retained (as identity) only so the
        # checkpoint state dict and the merge-step broadcast stay byte-compatible
        # with the stock POETLinear layout. inverse(identity) == identity.
        perm_in = torch.arange(in_features, device=device, dtype=torch.int32)
        perm_out = torch.arange(out_features, device=device, dtype=torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", perm_in.clone())
        self.register_buffer("perm_out_inv", perm_out.clone())

    def forward(self, x):
        # Single-step fast path (R=I); cayley guard is defensive (see POETLinear.forward).
        if getattr(self, "single_step_fast", False) and self.parameterization == "cayley":
            from .single_step import HeadAlignedSingleStepFunction

            return HeadAlignedSingleStepFunction.apply(
                x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
                self.block_size_in, self.block_size_out, self.head_side,
            )
        use_exp = self.parameterization == "exp"
        # exp builds R via matrix_exp (backward not compile-safe) and the
        # mem-efficient path wraps the chain in checkpoint() — both run eager.
        # The Cayley fast path is compiled during training and falls back to the
        # eager twin for eval (the compiled inference graph trips the same
        # torch-2.11 Inductor 'op6' bug as the base; see POETLinear.forward).
        if use_exp or self.mem_efficient_mode:
            return _forward_noperm_eager(
                x, self.oft_R_in, self.oft_R_out,
                self.block_size_in, self.block_size_out,
                self.rows_in, self.cols_in, self.rows_out, self.cols_out,
                self.weight, self.bias, self.mem_efficient_mode, use_exp,
            )
        core = forward_core_noperm if torch.is_grad_enabled() else _forward_noperm_eager
        return core(
            x, self.oft_R_in, self.oft_R_out,
            self.block_size_in, self.block_size_out,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.weight, self.bias, False, False,
        )

    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        R_out, R_in = self._merge_R()
        self._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)

    @torch.no_grad()
    def _fold_with_R(self, R_out, R_in, reinit_perm: bool = True) -> None:
        # Permutation-free fold (identity Ψ): weight <- (R_in @ Wᵀ @ R_out)ᵀ, then
        # reset generators. reinit_perm is accepted for API parity (no-op here).
        W = self.weight.detach().clone()
        tmp = block_diag_lr_matmul_decoupled(R_in, W.t(), R_out)
        self.weight.detach().copy_(tmp.t())
        self.oft_R_in.zero_()
        self.oft_R_out.zero_()
