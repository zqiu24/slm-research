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


class NGPTMLP(NGPTMLPBody):
    """Megatron-instantiable nGPT MLP (the class the layer spec wires).

    Megatron's ``build_module`` instantiates ``submodules.mlp.module`` only
    when it is a *class*; a plain builder *closure* is a ``types.FunctionType``
    and ``build_module`` returns it uninstantiated, leaving ``layer.mlp`` a
    bare function with no parameters (no ``mlp.linear_fc1/linear_fc2`` weights,
    broken forward). So the spec wires a class, not a closure. Geometry +
    nGPT scaling are read from the (patched) ``TransformerConfig`` — the
    ``ngpt_*`` fields are stamped on by ``ngpt_apply_spec``. Returns
    ``(output, None)`` to match Megatron's MLP ``(output, bias)`` contract;
    ``NGPTTransformerLayer.forward`` unpacks the tuple.
    """

    def __init__(self, config, submodules=None, **kwargs) -> None:
        hidden = int(config.hidden_size)
        dtype = getattr(config, "params_dtype", None)
        if dtype is None:
            dtype = torch.bfloat16 if getattr(config, "bf16", True) else torch.float32
        super().__init__(
            hidden_size=hidden,
            ffn_hidden_size=int(config.ffn_hidden_size),
            base_scale=float(getattr(config, "ngpt_base_scale", 1.0 / (hidden**0.5))),
            suv_init_value=float(getattr(config, "ngpt_suv_init", 1.0)),
            suv_init_scaling=1.0,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor):
        return super().forward(hidden_states), None
