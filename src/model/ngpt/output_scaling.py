"""Attach an nGPT `sz` post-multiplier to a model's output_layer.

The reference (model.py:283-292) does, when use_nGPT=1:

    sz_effective = self.sz * (sz_init_value/sz_init_scaling)
    logits = sz_effective * logits

Rather than post-multiplying the full `(s, b, vocab)` logits (which forces
autograd to retain that ~8 GB tensor for `sz`'s gradient and duplicates it),
we fold `sz` into the per-vocab rows of the output weight:

    sz_v * (W @ x)_v == (sz_v * W_v) @ x

This is mathematically identical but the retained tensor for `sz`'s gradient is
the `(vocab, hidden)` weight (~130 MB) instead of the full logits, and the
logits reach Megatron's fused cross-entropy un-duplicated — matching the
adam/muon baseline output memory. `ColumnParallelLinear.forward` already takes
an optional `weight=` argument (used here); the *stored* `output_layer.weight`
is untouched, so the per-step row normalization still applies to it and `sz`
stays a separate learned parameter.
"""

from __future__ import annotations

import torch.nn as nn

from src.model.ngpt.scaling_params import LearnedScaling


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

    def _wrapped(input_, weight=None, **kwargs):
        # Fold sz into the per-vocab rows of the (passed or stored) output
        # weight instead of scaling the full logits — see module docstring.
        base_w = weight if weight is not None else model.output_layer.weight
        sz_eff = model._ngpt_sz.scaled_value().to(base_w.dtype)
        return orig_forward(input_, weight=sz_eff.unsqueeze(1) * base_w, **kwargs)

    model.output_layer.forward = _wrapped  # type: ignore[assignment]
