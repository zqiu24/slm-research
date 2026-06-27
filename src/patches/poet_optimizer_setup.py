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
        config.poet_init_scale = getattr(args, "poet_init_scale", 1.0)
        config.poet_learnable_scale = getattr(args, "poet_learnable_scale", False)
        config.poet_cache_mode = getattr(args, "poet_cache_mode", "none")
        config.poet_use_poet_adam = getattr(args, "poet_use_poet_adam", False)
        config.poet_q_optimizer = getattr(args, "poet_q_optimizer", "adam")
        config.poet_muon_theta = getattr(args, "poet_muon_theta", 0.1)
        config.poet_muon_ns_steps = getattr(args, "poet_muon_ns_steps", 5)
        config.poet_muon_momentum = getattr(args, "poet_muon_momentum", 0.95)
        config.poet_lie_b1 = getattr(args, "poet_lie_b1", 0.9)
        config.poet_lie_b2 = getattr(args, "poet_lie_b2", 0.95)
        config.poet_lie_eps = getattr(args, "poet_lie_eps", 1.0e-8)
        config.poet_lie_v_mode = getattr(args, "poet_lie_v_mode", "elementwise")
        config.poet_lie_alternating = getattr(args, "poet_lie_alternating", False)
        config.poet_lie_alternate_every = getattr(args, "poet_lie_alternate_every", 1)
        config.poet_single_step_x_alternating = getattr(
            args, "poet_single_step_x_alternating", False
        )
        config.poet_lie_rms = getattr(args, "poet_lie_rms", False)
        config.poet_lie_rms_c = getattr(args, "poet_lie_rms_c", 0.2)
        config.poet_lie_ortho_c = getattr(args, "poet_lie_ortho_c", 0.01)
        config.poet_lie_ortho_update_rms = getattr(args, "poet_lie_ortho_update_rms", 0.2)
        config.poet_lie_ortho_update_rms_side_gamma = getattr(
            args, "poet_lie_ortho_update_rms_side_gamma", 0.0
        )
        config.poet_lie_ortho_max_angle = getattr(args, "poet_lie_ortho_max_angle", 0.024)
        config.poet_lie_ortho_rms_mode = getattr(args, "poet_lie_ortho_rms_mode", "weight")
        config.poet_lie_ortho_method = getattr(args, "poet_lie_ortho_method", "muon")
        config.poet_lie_ortho_ns_steps = getattr(args, "poet_lie_ortho_ns_steps", 5)
        config.poet_lie_ortho_use_second_moment = getattr(
            args, "poet_lie_ortho_use_second_moment", False
        )
        config.poet_lie_ortho_nesterov = getattr(args, "poet_lie_ortho_nesterov", False)
        config.poet_lie_ortho_distributed = getattr(args, "poet_lie_ortho_distributed", False)
        config.poet_lie_ortho_angle_dim_exp = getattr(args, "poet_lie_ortho_angle_dim_exp", 0.0)
        # b_ref for the per-block angle = hidden_size. The OptimizerConfig has NO hidden_size
        # (it's a model-config field), so copy it from args here, else poet.py reads None and
        # the angle scaling silently no-ops (every p arm == champion).
        config.poet_lie_ortho_angle_dim_ref = getattr(args, "hidden_size", None)
        config.poet_lie_ortho_decorrelate = getattr(args, "poet_lie_ortho_decorrelate", False)
        config.poet_lie_ortho_decorrelate_mode = getattr(
            args, "poet_lie_ortho_decorrelate_mode", "in_off_out"
        )
        # Alternating-path overlap-control knobs — MUST be copied here too, else the args
        # set on the CLI silently never reach the optimizer (the §17.6 silent-no-op trap).
        config.poet_lie_ortho_decorrelate_lambda = getattr(
            args, "poet_lie_ortho_decorrelate_lambda", 1.0
        )
        config.poet_lie_ortho_decorrelate_renorm = getattr(
            args, "poet_lie_ortho_decorrelate_renorm", False
        )
        config.poet_lie_ortho_decorrelate_cos_threshold = getattr(
            args, "poet_lie_ortho_decorrelate_cos_threshold", 0.0
        )
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
