"""POET linear-replacement helpers.

Ported from fork 2's ``megatron/poet_integration.py`` (commit bb43fa063).
The Megatron-specific type list (``ColumnParallelLinear`` /
``TEColumnParallelLinear`` / ...) is discovered lazily so unit tests can
pass in plain ``torch.nn.Linear`` via ``extra_linear_types``.

POET requires the model to be built with ``config.transformer_impl='local'``
so that ``TELayerNormColumnParallelLinear`` (fused norm + linear) is not
materialised — the patch in ``src/patches/poet_unfuse_te_impl.py`` enforces
that automatically.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn as nn
from poet_torch import POETLinear

from src.optim import poet_cache as _poet_cache

logger = logging.getLogger(__name__)


class POETMegatronLinear(nn.Module):
    """Wraps a :class:`POETLinear` to match Megatron's parallel-linear
    calling convention.

    ``ColumnParallelLinear`` and ``RowParallelLinear`` both return
    ``(output, output_bias)``. This wrapper preserves that convention so
    callers downstream of the swap don't notice the substitution.
    """

    def __init__(self, poet_linear: POETLinear, skip_bias_add: bool = False):
        super().__init__()
        self.poet_linear = poet_linear
        self._skip_bias_add = skip_bias_add
        # Expose weight / bias for DDP and Megatron introspection.
        self.weight = poet_linear.weight
        self.bias = poet_linear.bias

    def forward(self, input_: torch.Tensor, weight=None, **kw):
        output = self.poet_linear(input_)
        return output, None


def _megatron_linear_types() -> tuple[type, ...]:
    """Discover Megatron linear types; empty tuple if Megatron isn't importable.

    We catch ``Exception`` (not just ``ImportError``) because Megatron's
    top-level import eagerly loads ``transformer_engine``, which raises
    ``OSError: libcublas.so.12`` on CPU-only nodes. Returning an empty tuple
    means the caller falls back to ``extra_linear_types``.
    """
    try:
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
    except Exception:
        return ()
    try:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TERowParallelLinear,
        )

        return (
            ColumnParallelLinear,
            RowParallelLinear,
            TEColumnParallelLinear,
            TERowParallelLinear,
        )
    except Exception:
        return (ColumnParallelLinear, RowParallelLinear)


def _fused_layernorm_linear_types() -> tuple[type, ...]:
    """Modules POET must refuse to replace (the unfused-spec error case)."""
    out: tuple[type, ...] = ()
    try:
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
        )

        out += (TELayerNormColumnParallelLinear,)
    except Exception:
        pass
    try:
        from megatron.core.tensor_parallel.inference_layers import (
            InferenceLayerNormColumnParallelLinear,
        )

        out += (InferenceLayerNormColumnParallelLinear,)
    except Exception:
        pass
    return out


def replace_linears_with_poet(
    model: nn.Module,
    *,
    block_size: int = 256,
    init_type: str = "normalized",
    mup_alpha: float = 1.0,
    skip_lm_head: bool = True,
    extra_linear_types: Iterable[type] = (),
    cache_mode: str = "none",
) -> int:
    """Walk ``model`` and replace each parallel-linear with a
    :class:`POETMegatronLinear`.

    Returns the number of replacements.

    Raises ``RuntimeError`` if the model still has fused LayerNormLinear
    modules — those carry a layer-norm payload that POET would silently
    drop. The caller must rebuild the model with
    ``config.transformer_impl == 'local'`` first; the patch in
    ``src/patches/poet_unfuse_te_impl.py`` does that automatically.
    """
    fused = _fused_layernorm_linear_types()
    linear_types: tuple[type, ...] = _megatron_linear_types() + tuple(extra_linear_types)
    if not linear_types:
        raise RuntimeError(
            "No replaceable linear types found. Pass "
            "extra_linear_types=(nn.Linear,) for tests, or make sure "
            "megatron is importable."
        )

    replaced = 0
    skipped = 0

    def _walk(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced, skipped
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name

            if fused and isinstance(child, fused):
                raise RuntimeError(
                    f"[POET] Fused LayerNormLinear at {full} "
                    f"({type(child).__name__}). Rebuild with "
                    "config.transformer_impl='local' before applying POET."
                )

            if isinstance(child, linear_types):
                if skip_lm_head and "output_layer" in full:
                    skipped += 1
                    continue
                out_f, in_f = child.weight.shape
                if in_f % block_size != 0 or out_f % block_size != 0:
                    logger.info(
                        "[POET] skip %s: dims (%d, %d) not divisible by %d",
                        full,
                        in_f,
                        out_f,
                        block_size,
                    )
                    skipped += 1
                    continue

                if cache_mode == "none":
                    pl = POETLinear(
                        in_features=in_f,
                        out_features=out_f,
                        bsz=block_size,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                else:
                    pl = _poet_cache.CachedPOETLinear(
                        in_features=in_f,
                        out_features=out_f,
                        bsz=block_size,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    _poet_cache.register_poet_layer(pl)
                with torch.no_grad():
                    w = child.weight.data.clone()
                    if init_type == "normalized":
                        w = w / torch.norm(w, dim=1, keepdim=True)
                    elif init_type == "mup_normalized":
                        d_in = torch.tensor(float(in_f))
                        d_out = torch.tensor(float(out_f))
                        w = w / torch.norm(w, dim=1, keepdim=True)
                        target = mup_alpha * torch.sqrt(d_out / d_in)
                        current = torch.linalg.norm(w.float(), ord=2).item()
                        w = w * (target / current).to(dtype=w.dtype, device=w.device)
                    # init_type == "none": leave w unchanged.
                    pl.weight.copy_(w.to(pl.weight.dtype))
                    if child.bias is not None:
                        pl.bias.copy_(child.bias.data.to(pl.bias.dtype))

                wrapper = POETMegatronLinear(
                    pl, skip_bias_add=getattr(child, "skip_bias_add", False)
                )
                setattr(parent, name, wrapper)
                replaced += 1
            else:
                _walk(child, full)

    _walk(model)
    logger.info("[POET] replaced %d, skipped %d", replaced, skipped)
    return replaced
