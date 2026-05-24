"""Patch: route slm-research POET optimizer through Megatron's Adam branch.

Targets:
- megatron.training.training.get_megatron_optimizer_config
- megatron.training.training.get_megatron_optimizer

Megatron-Core 0.17.0 does not parse `--optimizer poet`. slm-research passes
`--optimizer adam --slm-optimizer poet` and this patch attaches the POET
settings to the OptimizerConfig, then routes the optimizer builder call to
`src.optim.poet.get_megatron_poet_optimizer`.
"""

from __future__ import annotations

from src.patches._registry import register_patch

_TARGET = (
    "megatron.training.training.get_megatron_optimizer_config",
    "megatron.training.training.get_megatron_optimizer",
)


@register_patch(name="poet_optimizer_setup", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt

    _orig_get_config = _mt.get_megatron_optimizer_config
    _orig_get_optimizer = _mt.get_megatron_optimizer

    def _wrapped_get_config(args):
        config, overrides = _orig_get_config(args)
        if getattr(args, "slm_optimizer", "") != "poet":
            return config, overrides
        config.slm_optimizer = "poet"
        config.poet_merge_period = getattr(args, "poet_merge_period", 0)
        config.poet_scale = getattr(args, "poet_scale", 1.0)
        config.poet_block_size = getattr(args, "poet_block_size", 256)
        config.poet_init_type = getattr(args, "poet_init_type", "normalized")
        config.poet_mup_alpha = getattr(args, "poet_mup_alpha", 1.0)
        config.poet_cache_mode = getattr(args, "poet_cache_mode", "none")
        return config, overrides

    def _wrapped_get_optimizer(config, model, **kwargs):
        if getattr(config, "slm_optimizer", "") != "poet":
            return _orig_get_optimizer(config, model, **kwargs)
        from src.optim.poet import get_megatron_poet_optimizer

        return get_megatron_poet_optimizer(
            config,
            model,
            config_overrides=kwargs.get("config_overrides"),
            use_gloo_process_groups=kwargs.get("use_gloo_process_groups", True),
        )

    _mt.get_megatron_optimizer_config = _wrapped_get_config
    _mt.get_megatron_optimizer = _wrapped_get_optimizer
