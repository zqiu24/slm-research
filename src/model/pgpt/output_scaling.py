"""Attach an nGPT `sz` post-multiplier to a model's output_layer.

The reference (model.py:283-292) does, when use_nGPT=1:

    sz_effective = self.sz * (sz_init_value/sz_init_scaling)
    logits = sz_effective * logits

We replicate that here by monkey-patching the `forward` of the given
holder's `output_layer` attribute (in Megatron that is the GPTModel's
ColumnParallelLinear). The wrapper preserves the upstream return
convention `(logits, bias)`.
"""

from __future__ import annotations

import torch.nn as nn

from src.model.pgpt.scaling_params import LearnedScaling


def attach_sz_scaling(model: nn.Module, vocab_size: int, base_scale: float) -> None:
    if getattr(model, "_ngpt_sz", None) is not None:
        return  # idempotent
    sz = LearnedScaling(
        shape=(int(vocab_size),),
        init_value=1.0,
        init_scaling=float(base_scale),
    )
    sz.to(next(model.parameters()).device)
    # Register as a submodule so checkpoint save/load picks it up.
    model.add_module("_ngpt_sz", sz)

    orig_forward = model.output_layer.forward

    def _wrapped(input_, *args, **kwargs):
        out = orig_forward(input_, *args, **kwargs)
        sz_eff = model._ngpt_sz.scaled_value()
        if isinstance(out, tuple):
            logits, bias = out
            return sz_eff.to(logits.dtype) * logits, bias
        return sz_eff.to(out.dtype) * out

    model.output_layer.forward = _wrapped  # type: ignore[assignment]
