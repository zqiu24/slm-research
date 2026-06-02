"""Patch: route slm-research ``muon_kimi`` optimizer through Megatron's Adam branch.

slm-research passes ``--optimizer adam --slm-optimizer muon_kimi``; this patch
tags the OptimizerConfig and reroutes the optimizer-builder call to
``src.optim.muon_kimi.get_megatron_muon_kimi_optimizer``. Mirrors
``poet_optimizer_setup``.
"""

from __future__ import annotations

from src.patches._registry import register_patch

# NOTE: poet_optimizer_setup targets these same two functions. The patch
# registry raises PatchConflict if both are registered at once, but that never
# happens in a real run (one experiment's patches load per process). Never list
# both poet_optimizer_setup and muon_kimi_optimizer_setup in experiment.patches
# simultaneously.
_TARGET = (
    "megatron.training.training.get_megatron_optimizer_config",
    "megatron.training.training.get_megatron_optimizer",
)


@register_patch(name="muon_kimi_optimizer_setup", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt

    _orig_get_config = _mt.get_megatron_optimizer_config
    _orig_get_optimizer = _mt.get_megatron_optimizer

    def _wrapped_get_config(args):
        config, overrides = _orig_get_config(args)
        if getattr(args, "slm_optimizer", "") != "muon_kimi":
            return config, overrides
        config.slm_optimizer = "muon_kimi"
        return config, overrides

    def _wrapped_get_optimizer(config, model, **kwargs):
        if getattr(config, "slm_optimizer", "") != "muon_kimi":
            return _orig_get_optimizer(config, model, **kwargs)
        from src.optim.muon_kimi import get_megatron_muon_kimi_optimizer

        return get_megatron_muon_kimi_optimizer(
            config,
            model,
            config_overrides=kwargs.get("config_overrides"),
            use_gloo_process_groups=kwargs.get("use_gloo_process_groups", True),
        )

    _mt.get_megatron_optimizer_config = _wrapped_get_config
    _mt.get_megatron_optimizer = _wrapped_get_optimizer
