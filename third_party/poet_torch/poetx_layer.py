"""POETXLinear: standalone POET linear storing the weight in the FORWARD frame.

POETLinear stores W_perm (= P_outᵀ W P_in) and rebuilds the effective weight
W_eff = W_perm[perm_out][:,perm_in] every forward (the gathers). POETXLinear stores
W_eff DIRECTLY (baked once at build, re-derived once per merge step), so the single-step
forward is a bare GEMM (POETXSingleStepFunction) with NO permutation. It is NOT a
POETLinear subclass (the merge driver recognizes it via a widened isinstance tuple), but
its merge REUSES POETLinear._fold_with_R verbatim by bracketing it with an
un-permute/re-permute (forward-frame <-> W_perm), so the fold math is bit-identical to the
proven path and supports perm reinit. ONLY valid at oft_R=0 (merge_period=1) and cayley.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .poet_layer import POETLinear
from .poetx_ops import POETXSingleStepFunction
from .poetx_ops import AlternatingPOETXSingleStepFunction


class POETXLinear(nn.Module):
    def __init__(self, in_features, out_features, bsz=None, block_count=None,
                 bias=False, device=None, dtype=None, parameterization="cayley",
                 alternating=False, alternate_every=1):
        super().__init__()
        if parameterization != "cayley":
            raise ValueError(
                "POETXLinear requires parameterization='cayley' "
                f"(the perm-free-forward backward is Cayley-specific); got {parameterization!r}."
            )
        # Buffer/param setup mirrors POETLinear.__init__ (kept standalone on purpose:
        # POETXLinear is NOT a POETLinear subclass, so POETLinear.__init__'s zero-arg
        # super() cannot run on `self`). The merge methods below reuse POETLinear's
        # (super-free) fold/build helpers via unbound calls, so only __init__ duplicates.
        self.in_features = in_features
        self.out_features = out_features
        self.parameterization = parameterization
        self.single_step_fast = False  # POETX ignores it (forward is always the X op)
        # Alternating both-momenta merge: when set, the merge driver folds ONLY the
        # active side (the frozen side's oft_R is 0 -> identity -> skip its Cayley).
        # Forward/backward stay both-sides (POETXSingleStepFunction feeds BOTH grads),
        # so both momenta stay fed -- the load-bearing ingredient. alternate_every
        # matches the optimizer + alt_state cadence.
        self.alternating = bool(alternating)
        self.alternate_every = max(1, int(alternate_every))

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
        self.block_size = block_size_in  # back-compat (merge "is-active" guard reads it)

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
        self.r_in = r_in
        self.r_out = r_out

        rows_in, cols_in = torch.triu_indices(block_size_in, block_size_in, 1, device=device)
        self.register_buffer("rows_in", rows_in.to(torch.int32))
        self.register_buffer("cols_in", cols_in.to(torch.int32))
        rows_out, cols_out = torch.triu_indices(block_size_out, block_size_out, 1, device=device)
        self.register_buffer("rows_out", rows_out.to(torch.int32))
        self.register_buffer("cols_out", cols_out.to(torch.int32))

        perm_in = torch.randperm(in_features, device=device, dtype=torch.int32)
        perm_out = torch.randperm(out_features, device=device, dtype=torch.int32)
        self.register_buffer("perm_in", perm_in)
        self.register_buffer("perm_out", perm_out)
        self.register_buffer("perm_in_inv", torch.argsort(perm_in).to(torch.int32))
        self.register_buffer("perm_out_inv", torch.argsort(perm_out).to(torch.int32))
        # self.weight currently holds an (empty) [out, in] tensor in the W_perm frame;
        # bake_perms_into_weight() converts it to the forward frame once the real
        # weights have been copied in (the walk calls it after _copy_and_init_weight).

    @torch.no_grad()
    def bake_perms_into_weight(self) -> None:
        """Convert the freshly-copied W_perm storage into the forward-frame Wx =
        W_perm[perm_out][:,perm_in] (and bias_eff = bias[perm_out]). Idempotent only
        once per fresh copy — call exactly once at build, after the weight is set."""
        self.weight.copy_(
            self.weight.index_select(0, self.perm_out).index_select(1, self.perm_in)
        )
        if self.bias is not None:
            self.bias.copy_(self.bias.index_select(0, self.perm_out))

    def forward(self, x):
        # self.weight is the forward-frame Wx; self.bias is bias_eff (forward frame).
        return POETXSingleStepFunction.apply(
            x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
            self.perm_in_inv, self.perm_out_inv,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out,
        )

    def _build_R(self, oft_in, oft_out):
        return POETLinear._build_R(self, oft_in, oft_out)

    def _merge_R(self):
        return POETLinear._merge_R(self)

    @torch.no_grad()
    def _fold_with_R(self, R_out, R_in, reinit_perm: bool = True) -> None:
        """Round-trip fold: forward-frame -> W_perm, reuse the verified
        POETLinear._fold_with_R (folds R, zeros oft_R, resamples perms on reinit),
        then re-permute back to the forward frame with the (possibly new) perms.
        Bit-identical to POETLinear's effective weight by construction."""
        # forward-frame -> W_perm storage frame, using CURRENT perms
        self.weight.copy_(
            self.weight.index_select(0, self.perm_out_inv).index_select(1, self.perm_in_inv)
        )
        if self.bias is not None:
            self.bias.copy_(self.bias.index_select(0, self.perm_out_inv))
        # verified fold (operates on W_perm; resamples self.perm_* on reinit)
        POETLinear._fold_with_R(self, R_out, R_in, reinit_perm=reinit_perm)
        # W_perm -> forward frame, using the (possibly NEW) perms
        self.weight.copy_(
            self.weight.index_select(0, self.perm_out).index_select(1, self.perm_in)
        )
        if self.bias is not None:
            self.bias.copy_(self.bias.index_select(0, self.perm_out))

    def merge_then_reinitialize(self, reinit_perm: bool = True) -> None:
        R_out, R_in = self._merge_R()
        self._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)

    @torch.no_grad()
    def _fold_active_side(self, active, reinit_perm: bool = False, cayley_fn=None) -> None:
        """Fold ONLY the active side into W (skip the frozen side's Cayley build).

        The frozen side's oft_R is exactly 0 => R = I, so its fold is a no-op; we
        build identity blocks for it (no Cayley) and reuse the verified round-trip
        fold. Bit-identical to the both-sides fold whenever the frozen side is
        identity, but pays one Cayley + one block-fold instead of two.
        """
        import torch as _torch
        from .poet_layer import pytorch_skew_symmetric

        if cayley_fn is None:

            def cayley_fn(Q):
                return _torch.ops.poet.cayley(Q)[0]

        if active == "in":
            R_in = cayley_fn(
                pytorch_skew_symmetric(self.oft_R_in, self.block_size_in, self.rows_in, self.cols_in)
            )
            R_out = _torch.eye(self.block_size_out, dtype=self.weight.dtype, device=self.weight.device)
            R_out = R_out.unsqueeze(0).expand(self.r_out, -1, -1).contiguous()  # bmm needs real strides
        else:  # "out"
            R_out = cayley_fn(
                pytorch_skew_symmetric(self.oft_R_out, self.block_size_out, self.rows_out, self.cols_out)
            )
            R_in = _torch.eye(self.block_size_in, dtype=self.weight.dtype, device=self.weight.device)
            R_in = R_in.unsqueeze(0).expand(self.r_in, -1, -1).contiguous()  # bmm needs real strides
        self._fold_with_R(R_out, R_in, reinit_perm=reinit_perm)


class AlternatingPOETXLinear(POETXLinear):
    """POETX layer that trains ONE rotation side per step (true single-side).

    The active side comes from the shared `alt_state` iteration (seeded once per
    training step), so layer forward, optimizer, and merge all agree. Forward is
    the unchanged bare GEMM; the backward (AlternatingPOETXSingleStepFunction)
    computes only the active side's rotation-gradient and zeros the frozen side.
    `alternating=True` routes the merge driver to the active-only fold (inherited
    from POETXLinear). This is the gated research path — it freezes the inactive
    side's MOMENTUM (true_single_side optimizer), which regressed quality; the
    integrated both-momenta path uses a plain POETXLinear(alternating=True) instead.
    """

    def __init__(self, *args, alternate_every: int = 1, **kwargs):
        super().__init__(*args, alternating=True, alternate_every=alternate_every, **kwargs)

    def forward(self, x):
        from .alt_state import active_side

        active = active_side(self.alternate_every)
        return AlternatingPOETXSingleStepFunction.apply(
            x, self.oft_R_in, self.oft_R_out, self.weight, self.bias,
            self.perm_in_inv, self.perm_out_inv,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out, active,
        )
