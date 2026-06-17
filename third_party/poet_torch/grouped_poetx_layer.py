# third_party/poet_torch/grouped_poetx_layer.py
"""GroupedPOETXLinear: E experts' POETX rotation batched over the expert axis.

Owns E POETXLinear sub-instances (each holding its own 2-D oft_R params + perms + the
verified merge methods); their frozen weights alias rows of one contiguous [E,out,in]
buffer. Forward/backward go through the batched GroupedPOETXFunction (the 99.5% path);
merge delegates to the sub-instances unchanged (the 2.6% path). oft_R stays E separate
2-D params so LieOrthMomentum + the merge driver are untouched."""
from __future__ import annotations

import torch
import torch.nn as nn

from poet_torch import POETXLinear
from poet_torch.grouped_poetx_ops import GroupedPOETXFunction


class GroupedPOETXLinear(nn.Module):
    def __init__(self, num_experts, in_features, out_features, *,
                 block_count, alternating, alternate_every, device=None, dtype=None):
        super().__init__()
        self.E = int(num_experts)
        self.in_features, self.out_features = in_features, out_features
        self.experts = nn.ModuleList([
            POETXLinear(in_features=in_features, out_features=out_features,
                        block_count=block_count, bias=False, device=device, dtype=dtype,
                        parameterization="cayley",
                        alternating=alternating, alternate_every=alternate_every)
            for _ in range(self.E)
        ])
        e0 = self.experts[0]
        self.alternating = bool(alternating)
        self.block_size_in, self.block_size_out = e0.block_size_in, e0.block_size_out
        self.block_size = e0.block_size_in                     # merge "is-active" guard
        self.register_buffer(
            "weight", torch.empty(self.E, out_features, in_features, device=device, dtype=dtype)
        )
        # shared block triu indices (identical block sizes across experts)
        for nm in ("rows_in", "cols_in", "rows_out", "cols_out"):
            self.register_buffer(nm, getattr(e0, nm).clone())

    @torch.no_grad()
    def bind_weights(self):
        """Copy each expert's (baked) forward-frame weight into the buffer and repoint
        the expert weight to the buffer row (single storage). Call once at build, after
        each expert weight is copied + baked."""
        for e, ex in enumerate(self.experts):
            self.weight[e].copy_(ex.weight)
            ex.weight.data = self.weight[e]

    def forward(self, concat_x, tokens_per_expert):
        oft_in = torch.stack([ex.oft_R_in for ex in self.experts])
        oft_out = torch.stack([ex.oft_R_out for ex in self.experts])
        pin = torch.stack([ex.perm_in_inv for ex in self.experts])
        pout = torch.stack([ex.perm_out_inv for ex in self.experts])
        sizes = tuple(int(t) for t in tokens_per_expert.tolist())
        return GroupedPOETXFunction.apply(
            concat_x, oft_in, oft_out, self.weight, pin, pout,
            self.rows_in, self.cols_in, self.rows_out, self.cols_out,
            self.block_size_in, self.block_size_out, sizes)

    def effective_weight(self):
        return self.weight                                    # oft_R==0 -> R==I -> Wx==weight

    @torch.no_grad()
    def merge_then_reinitialize(self, reinit_perm=True):
        for ex in self.experts:
            ex.merge_then_reinitialize(reinit_perm=reinit_perm)

    @torch.no_grad()
    def _fold_active_side(self, active, reinit_perm=False, cayley_fn=None):
        # cayley_fn defaults to the Triton op (GPU); CPU tests inject pure-torch
        # cayley_batch — same pattern as POETXLinear._fold_active_side / _build_R_batched.
        for ex in self.experts:
            ex._fold_active_side(active, reinit_perm=reinit_perm, cayley_fn=cayley_fn)
