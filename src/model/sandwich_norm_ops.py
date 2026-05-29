"""Pure helpers for sandwich-norm (no Megatron import — CPU-safe).

Sandwich-norm applies a normalization to a sub-layer's *output* before the
residual add, with the norm weight scaled small at init. We inject it via a
forward-hook on the attention / MLP submodule so no Megatron forward is copied.
"""

from __future__ import annotations

import torch


def make_post_norm_hook(norm):
    """Build a forward-hook that post-norms a submodule's primary output.

    Megatron's attention / MLP modules return ``(output, bias)``; we normalize
    ``output`` and pass ``bias`` (and any further elements) through unchanged. A
    bare-tensor output is also supported (for tests / future modules).
    """

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (norm(output[0]), *tuple(output[1:]))
        return norm(output)

    return hook


def apply_post_norm_scale(norm_module, scale: float) -> None:
    """Multiply a norm module's ``weight`` by ``scale`` in-place (no-op at 1.0).

    Matches the Huawei init: post-norm weights start small (e.g. 0.03) so the
    post-norm contributes near-identity at the start of training.
    """
    if scale != 1.0:
        with torch.no_grad():
            norm_module.weight.mul_(scale)
