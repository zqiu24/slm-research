from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from src.utils.torchtitan_args import build_torchtitan_config


def _cfg():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "experiment=optim/adam",
            "cluster=h100_de",
            "backend=torchtitan",
        ]
    )
    resolve_config(cfg)
    return cfg


def test_returns_toml_dict_and_override_list():
    toml, overrides = build_torchtitan_config(_cfg())
    assert isinstance(toml, dict)
    assert isinstance(overrides, list)
    assert all(isinstance(s, str) for s in overrides)


def test_model_block_selects_slm_spec_and_flavor():
    # [model] TOML carries only name+flavor; dims live in the registered flavor
    # (asserted in Task 8's build_slm_flavor test), not here.
    toml, _ = build_torchtitan_config(_cfg())
    model = toml["model"]
    assert model["name"] == "slm_llama3"
    assert model["flavor"] == "slm_300m"


def test_training_block_uses_resolved_values():
    cfg = _cfg()
    toml, _ = build_torchtitan_config(cfg)
    training = toml["training"]
    assert training["seq_len"] == int(cfg.base.model.seq_length)
    assert training["global_batch_size"] == int(cfg.training.global_batch_size)
    assert training["steps"] == int(cfg.training.total_tokens) // int(cfg.base.model.seq_length)
    # VERIFIED against v0.2.2: seed is [debug].seed, NOT [training].seed
    # (docs/torchtitan_api_notes.md §1).
    assert toml["debug"]["seed"] == int(cfg.seed)


def test_optimizer_is_adamw_with_betas():
    cfg = _cfg()
    toml, _ = build_torchtitan_config(cfg)
    opt = toml["optimizer"]
    assert opt["name"].lower() == "adamw"
    assert opt["lr"] == float(cfg.optim.get("lr", cfg.optim.get("adam", {}).get("lr")))


def test_parallelism_is_fsdp_only_at_300m():
    cfg = _cfg()
    toml, _ = build_torchtitan_config(cfg)
    par = toml["parallelism"]
    assert par["tensor_parallel_degree"] == int(cfg.parallelism.tp)  # 1 at 300m
    assert par["data_parallel_shard_degree"] == -1  # FSDP over all remaining ranks


def test_rejects_non_adamw_optimizer():
    cfg = _parse_overrides(["base/family=llama3", "experiment=optim/poet", "backend=torchtitan"])
    resolve_config(cfg)
    import pytest

    with pytest.raises(ValueError, match="only supports adamw"):
        build_torchtitan_config(cfg)


def test_unmapped_knobs_flags_megatron_patches():
    from omegaconf import OmegaConf

    from src.utils.torchtitan_args import unmapped_megatron_knobs

    cfg = _cfg()
    OmegaConf.set_struct(cfg, False)  # resolved cfg is struct-locked; allow injecting a field
    cfg.experiment.patches = ["sandwich_norm_apply"]
    notes = unmapped_megatron_knobs(cfg)
    assert any("patches" in n for n in notes)
