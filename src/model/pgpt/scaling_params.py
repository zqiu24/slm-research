"""Learned scaling parameter helper for nGPT.

Captures the (init_value, init_scaling, fp32 storage) four-tuple that
the reference uses for sqk, suv, attn_alpha, mlp_alpha, and sz.
Effective value at runtime is `param * (init_value / init_scaling)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedScaling(nn.Module):
    """Learnable scaling vector with separated init_value vs storage scale."""

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        init_value: float,
        init_scaling: float,
    ) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.init_value = float(init_value)
        self.init_scaling = float(init_scaling)
        self.param = nn.Parameter(self.init_scaling * torch.ones(shape, dtype=torch.float32))

    def scaled_value(self) -> torch.Tensor:
        return self.param * (self.init_value / self.init_scaling)
