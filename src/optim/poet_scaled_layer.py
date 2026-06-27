"""POET linear layers with a trainable per-layer scalar gain (rung-1 spectral DOF).

``W_eff = g · R_out · W₀ · R_in``, where ``g`` is a 0-dim trainable scalar
(init 1.0), one per weight matrix. ``g`` is applied on the forward OUTPUT, outside
the base class's compiled core, so the merge/fold, single-step, exp, and compiled
paths are all reused from the base class unchanged. ``g=1.0`` ⇒ bit-exact the base
layer (multiply-by-1.0 is exact).

``g`` is created in the constructor (model-build time, pre-DDP), so it is a
first-class DDP grad-buffer citizen like ``oft_R``. For
``q_optimizer=lie_ortho_update_rms`` it also feeds the angle law
``θ = lr·rho/RMS(g·W₀) = lr·rho/(|g|·RMS(W₀))``; that coupling lives in the param-group
builder (``src/optim/poet.py``) and the optimizer step
(``src/optim/poet_lie_orth_update_rms.py``), NOT here.

The champion config sets ``single_step_native=true`` ⇒ ``SingleStepPOETLinear``, so
both base classes get a scaled variant via the shared mixin.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from poet_torch import POETLinear, SingleStepPOETLinear


class _LearnableScaleMixin:
    """Add a 0-dim trainable ``gain`` (init 1.0) and scale the forward output.

    Concrete subclasses must call ``self._init_scale_gain()`` at the END of their
    ``__init__`` (after the base ``__init__`` has created ``self.weight`` /
    ``self.bias``).
    """

    def _init_scale_gain(self) -> None:
        if getattr(self, "bias", None) is not None:
            raise ValueError(
                "learnable-scale POET assumes bias=False (the gain scales the whole "
                "forward output); construct the layer without bias."
            )
        self.gain = nn.Parameter(torch.ones((), device=self.weight.device, dtype=self.weight.dtype))

    def forward(self, x):  # type: ignore[override]
        return super().forward(x) * self.gain


class ScaledPOETLinear(_LearnableScaleMixin, POETLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_scale_gain()


class ScaledSingleStepPOETLinear(_LearnableScaleMixin, SingleStepPOETLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_scale_gain()
