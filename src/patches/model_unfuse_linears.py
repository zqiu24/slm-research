"""Patch: unfuse fused parallel linears at model-build time.

Targets ``pretrain_gpt.model_provider`` (the builder passed to Megatron's
``get_model``). Right after each model chunk is built — and **before** DDP /
Float16Module wrapping — replaces the fused ``linear_qkv`` / ``linear_fc1``
with separate Q/K/V and gate/up projections when ``--unfuse-qkv`` /
``--unfuse-fc1`` are set.

This is an architectural transform independent of the optimizer: it runs for
any experiment that lists this patch. POET (a separate ``get_model`` patch)
then wraps whatever linears exist, so it naturally picks up the unfused ones.
Hooking ``model_provider`` (rather than ``get_model``) avoids a target clash
with ``poet_apply_to_model`` and keeps the unfuse on the unwrapped, pre-DDP
model.
"""

from __future__ import annotations

import logging

from src.patches._registry import register_patch

_TARGET = ("pretrain_gpt.model_provider",)
logger = logging.getLogger(__name__)


@register_patch(name="model_unfuse_linears", targets=_TARGET)
def apply() -> None:
    import pretrain_gpt as _mg
    from megatron.training import get_args

    from src.model.unfuse_linears import unfuse_fused_linears

    _orig = _mg.model_provider

    def _wrapped(*a, **kw):
        model = _orig(*a, **kw)
        args = get_args()
        unfuse_qkv = getattr(args, "unfuse_qkv", False)
        unfuse_fc1 = getattr(args, "unfuse_fc1", False)
        if unfuse_qkv or unfuse_fc1:
            n = unfuse_fused_linears(model, unfuse_qkv=unfuse_qkv, unfuse_fc1=unfuse_fc1)
            logger.info(
                "[unfuse] unfused %d fused linears (qkv=%s, fc1=%s)",
                n,
                unfuse_qkv,
                unfuse_fc1,
            )
        return model

    _mg.model_provider = _wrapped
