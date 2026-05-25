"""Patch: post-step L2-projection of nGPT weight matrices onto the sphere.

Targets ``megatron.training.training.train_step``. After each step,
calls `normalize_module_matrices(model._ngpt_norm_role_map)` for every
model chunk. This mirrors the reference train.py:500 line where
`normalize_matrices()` is called every iteration.

This is structurally identical to src/patches/poet_merge_step.py; we
keep them as separate patches because (a) the role registries differ
and (b) one experiment may run with nGPT but not POET (and vice versa).
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.train_step",)
logger = logging.getLogger(__name__)


@register_patch(name="ngpt_normalize_step", targets=_TARGET)
def apply() -> None:
    from megatron.training import get_args
    from megatron.training import training as _mt

    from src.model.ngpt.normalize import normalize_module_matrices

    _orig = _mt.train_step

    def _wrapped(*args, **kwargs):
        ret = _orig(*args, **kwargs)
        opts = get_args()
        if not getattr(opts, "ngpt", False):
            return ret
        model = args[2] if len(args) >= 3 else kwargs.get("model")
        if model is None:
            return ret
        chunks = model if isinstance(model, list) else [model]
        for m in chunks:
            role_map = getattr(m, "_ngpt_norm_role_map", None)
            if role_map:
                normalize_module_matrices(role_map)
        return ret

    _mt.train_step = _wrapped
