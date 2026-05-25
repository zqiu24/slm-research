"""Hypersphere normalization primitives used by nGPT.

`justnorm(x, dim)` is the per-vector L2 projection onto the unit sphere
(reference: `train.py::justnorm`, `model.py::Block.justnorm`).

`normalize_module_matrices(role_map)` does the offline matrix projection
the reference paper applies (a) once after model init and (b) after
every optimizer step. The caller passes a `{parameter -> role}` mapping
where role is "rows" or "cols", matching the reference's convention:

    role="rows" -> normalize each row to unit norm (used when the row
                   indexes an OUTPUT channel that lives on the sphere,
                   e.g. wte, lm_head, q/k/v, c_fc)
    role="cols" -> normalize each column to unit norm (used when the
                   column indexes an INPUT channel that lives on the
                   sphere, e.g. att_c_proj, mlp_c_proj)
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def justnorm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize `x` along `dim`. Preserves dtype (computes in fp32)."""
    dtype = x.dtype
    x32 = x.float()
    out = x32 / x32.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return out.to(dtype=dtype)


@torch.no_grad()
def normalize_module_matrices(role_map: Mapping[torch.nn.Parameter, str]) -> None:
    """Project each parameter onto the unit sphere in-place per its role.

    role="rows" -> per-row unit norm  (normalize along dim=1)
    role="cols" -> per-column unit norm (normalize along dim=0)
    """
    for param, role in role_map.items():
        if role == "rows":
            param.data.copy_(justnorm(param.data, dim=1))
        elif role == "cols":
            param.data.copy_(justnorm(param.data, dim=0))
        else:
            raise ValueError(
                f"normalize_module_matrices: unknown role {role!r}; " "expected 'rows' or 'cols'"
            )
