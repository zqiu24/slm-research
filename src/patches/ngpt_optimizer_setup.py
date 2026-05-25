"""Patch: route scaling params into a zero-weight-decay group for nGPT runs.

Targets ``megatron.training.training.get_megatron_optimizer`` — we
intercept the call only when args.ngpt is set, so AdamW gets two param
groups (decay vs no-decay) and the scaling vectors (sqk, suv, alpha*,
sz) never get pulled toward zero.

The companion helper `classify_ngpt_param_groups(model)` is what the
test exercises; the patch itself just delegates.
"""

from __future__ import annotations

from src.patches._registry import register_patch

_TARGET = ("megatron.training.training.get_megatron_optimizer",)

# Parent-module names that hold an nGPT scaling vector as `.param`.
# Matched on the trailing name segment (not a raw substring) so a weight
# such as `...suv_proj.weight` can never be misclassified.
_SCALING_MODULE_NAMES = frozenset({"sqk", "suv", "attn_alpha", "mlp_alpha", "_ngpt_sz"})


def classify_ngpt_param_groups(model) -> tuple[list, list]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        if len(parts) >= 2 and parts[-1] == "param" and parts[-2] in _SCALING_MODULE_NAMES:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


@register_patch(name="ngpt_optimizer_setup", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt

    _orig = _mt.get_megatron_optimizer

    def _wrapped(config, model, **kwargs):
        opt = _orig(config, model, **kwargs)
        if not getattr(config, "ngpt", False):
            return opt
        # Walk the optimizer's param groups and move scaling params into a
        # zero-WD group. Megatron may return ChainedOptimizer or
        # Float16OptimizerWithFloat16Params; both expose .param_groups via
        # their inner optimizer. We mutate the inner optimizer in place.
        chunks = model if isinstance(model, list) else [model]
        scaling_param_ids: set[int] = set()
        for m in chunks:
            _, no_decay = classify_ngpt_param_groups(m)
            scaling_param_ids.update(id(p) for p in no_decay)

        inner = getattr(opt, "optimizer", None) or opt
        for group in inner.param_groups:
            if any(id(p) in scaling_param_ids for p in group["params"]):
                group["weight_decay"] = 0.0
        return opt

    _mt.get_megatron_optimizer = _wrapped
