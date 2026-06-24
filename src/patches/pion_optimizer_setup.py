"""Patch: route slm-research ``pion`` optimizer through Megatron's Adam branch.

slm-research passes ``--optimizer adam --slm-optimizer pion``; this patch tags
the OptimizerConfig, copies the ``pion_*`` CLI knobs onto it (they are not
declared OptimizerConfig dataclass fields, so the stock arg->field copy in
``get_megatron_optimizer_config`` skips them), and reroutes the optimizer-builder
call to ``src.optim.pion.get_megatron_pion_optimizer``. Mirrors
``muon_kimi_optimizer_setup``.
"""

from __future__ import annotations

from src.patches._registry import register_patch

# NOTE: poet_optimizer_setup / muon_kimi_optimizer_setup target these same two
# functions. The patch registry raises PatchConflict if two are registered at
# once, but that never happens in a real run (one experiment's patches load per
# process). Never list pion_optimizer_setup together with another
# *_optimizer_setup patch in experiment.patches.
_TARGET = (
    "megatron.training.training.get_megatron_optimizer_config",
    "megatron.training.training.get_megatron_optimizer",
)

# pion_* args copied from the parsed CLI args onto the OptimizerConfig so the
# builder (src/optim/pion.py) can read them via getattr. Keep in sync with the
# args registered in launchers/pretrain_gpt_slm.py:add_slm_args.
_PION_CONFIG_ATTRS = (
    "pion_scaling",
    "pion_rms",
    "pion_update_side",
    "pion_momentum",
    "pion_degree",
    "pion_beta1",
    "pion_beta2",
    "pion_use_second_momentum",
    "pion_qkv_split_granularity",
    "pion_split_qkv",
    "pion_split_gate",
    "pion_split_qkv_per_head",
    "pion_exp_map",
)


@register_patch(name="pion_optimizer_setup", targets=_TARGET)
def apply() -> None:
    from megatron.training import training as _mt

    _orig_get_config = _mt.get_megatron_optimizer_config
    _orig_get_optimizer = _mt.get_megatron_optimizer

    def _wrapped_get_config(args):
        config, overrides = _orig_get_config(args)
        if getattr(args, "slm_optimizer", "") != "pion":
            return config, overrides
        config.slm_optimizer = "pion"
        for attr in _PION_CONFIG_ATTRS:
            if hasattr(args, attr):
                setattr(config, attr, getattr(args, attr))
        return config, overrides

    def _wrapped_get_optimizer(config, model, **kwargs):
        if getattr(config, "slm_optimizer", "") != "pion":
            return _orig_get_optimizer(config, model, **kwargs)
        from src.optim.pion import get_megatron_pion_optimizer

        return get_megatron_pion_optimizer(
            config,
            model,
            config_overrides=kwargs.get("config_overrides"),
            use_gloo_process_groups=kwargs.get("use_gloo_process_groups", True),
        )

    _mt.get_megatron_optimizer_config = _wrapped_get_config
    _mt.get_megatron_optimizer = _wrapped_get_optimizer
