from omegaconf import OmegaConf

from src.utils.megatron_args import _optimizer_args


def _cfg(group_experts):
    return OmegaConf.create(
        {
            "optim": {
                "type": "poet",
                "lr": 3e-3,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "poet": {
                    "block_count": 8,
                    "cache_mode": "none",
                    "init_type": "normalized",
                    "mup_alpha": 1.0,
                    "merge_period": 1,
                    "scale": 0.5,
                    "single_step_x": True,
                    "single_step_fast": True,
                    "lie_alternating": True,
                    "group_experts": group_experts,
                },
            }
        }
    )


def test_group_experts_flag_emitted_only_when_set():
    assert "--poet-group-experts" in _optimizer_args(_cfg(True))
    assert "--poet-group-experts" not in _optimizer_args(_cfg(False))
