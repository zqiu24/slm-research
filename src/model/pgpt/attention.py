"""nGPT Q/K hypersphere normalization.

Plugged into Megatron's SelfAttentionSubmodules.q_layernorm and
.k_layernorm slots, so `SelfAttention.get_query_key_value_tensors`
applies it to the per-head tensors `(s, b, h_per_tp, d_head)` right
after the QKV split and **before** RoPE (RoPE is applied later, back in
`Attention.forward`). This is the same pipeline position as llama3's
`qk_norm`, so it is consistent with the llama3 backbone this bake-off
compares against (adam/muon).

NOTE: this deviates from the NVIDIA reference, which applies
`sqk * justnorm` AFTER RoPE (model.py:124-131). `justnorm` commutes with
RoPE (norm-preserving), so the two orderings are identical at init (sqk
uniform) and diverge only once sqk becomes non-uniform within a RoPE
pair. The llama3-consistent placement is intentional, not a parity bug.

Output: sqk * justnorm(x), per-head, per-position.

Softmax scale override is set elsewhere (config.softmax_scale =
sqrt(head_dim)) so the attention payload uses the nGPT scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.pgpt.normalize import justnorm
from src.model.pgpt.scaling_params import LearnedScaling


class QKHyperNorm(nn.Module):
    """L2-normalize per-head Q or K and scale by learnable per-channel sqk."""

    def __init__(
        self,
        num_heads_per_tp: int,
        head_dim: int,
        sqk_init_value: float,
        base_scale: float,
    ) -> None:
        super().__init__()
        self.num_heads_per_tp = int(num_heads_per_tp)
        self.head_dim = int(head_dim)
        self.sqk = LearnedScaling(
            shape=(self.num_heads_per_tp * self.head_dim,),
            init_value=sqk_init_value,
            init_scaling=base_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (s, b, h_per_tp, d_head). Normalize along d_head, then
        # multiply by per-channel sqk reshaped to broadcast over (s, b).
        normed = justnorm(x, dim=-1)
        sqk_eff = (
            self.sqk.scaled_value()
            .view(1, 1, self.num_heads_per_tp, self.head_dim)
            .to(normed.dtype)
        )
        return sqk_eff * normed
