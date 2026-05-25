"""nGPT MLP body: c_fc -> suv * uv -> chunk -> silu(v) * u -> mlp_c_proj.

NGPTMLPBody is a CPU-runnable pure-PyTorch module that matches the
reference's MLP fragment. It is what NGPTTransformerLayer instantiates
when the layer spec wires `mlp=NGPTMLPBody`. We deliberately do NOT
subclass `megatron.core.transformer.mlp.MLP` here because (a) MLP
defaults to two RowParallel/ColParallel linears that pull in TP plumbing
unhelpful at TP=1, and (b) staying pure-PyTorch keeps the parity test
runnable on CPU.

A future v2 that adds TP>1 support will subclass `MLP` and override
`forward` so the column-parallel `linear_fc1`, the suv scaling, and the
row-parallel `linear_fc2` all stay TP-aware.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.ngpt.scaling_params import LearnedScaling


class NGPTMLPBody(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        base_scale: float,
        suv_init_value: float,
        suv_init_scaling: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.ffn_hidden_size = int(ffn_hidden_size)
        self._n_embd_sqrt = float(self.hidden_size) ** 0.5

        # nGPT reference packs c_fc with 2*ffn_hidden_size columns:
        # [u_half | v_half]. Same convention here.
        self.linear_fc1 = nn.Linear(
            self.hidden_size, 2 * self.ffn_hidden_size, bias=False, dtype=dtype
        )
        self.linear_fc2 = nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False, dtype=dtype)
        # init: row-normalized with std=base_scale
        nn.init.normal_(self.linear_fc1.weight, mean=0.0, std=base_scale)
        nn.init.normal_(self.linear_fc2.weight, mean=0.0, std=base_scale)

        self.suv = LearnedScaling(
            shape=(2 * self.ffn_hidden_size,),
            init_value=suv_init_value,
            init_scaling=suv_init_scaling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        uv = self.linear_fc1(x)
        # Reference effective suv: param * (init_value/init_scaling) * sqrt(n_embd)
        suv = (self.suv.scaled_value() * self._n_embd_sqrt).to(uv.dtype)
        uv = suv * uv
        u, v = uv.chunk(2, dim=-1)
        return self.linear_fc2(u * functional.silu(v))
