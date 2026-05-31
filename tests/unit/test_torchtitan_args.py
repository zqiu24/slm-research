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
    # [model] TOML carries name+flavor (+ hf_assets_path); dims live in the
    # registered flavor (asserted in Task 8's build_slm_flavor test), not here.
    cfg = _cfg()
    toml, _ = build_torchtitan_config(cfg)
    model = toml["model"]
    assert model["name"] == "slm_llama3"
    assert model["flavor"] == "slm_300m"
    # hf_assets_path must forward the data manifest's HF tokenizer dir so
    # torchtitan's build_hf_tokenizer can load tokenizer.json — the ./assets/tokenizer
    # default was deprecated in PR #1540 and crashes the run if left unset.
    assert model["hf_assets_path"] == str(cfg.data.tokenizer_model)


def test_training_block_uses_resolved_values():
    cfg = _cfg()
    toml, _ = build_torchtitan_config(cfg)
    training = toml["training"]
    assert training["seq_len"] == int(cfg.base.model.seq_length)
    assert training["global_batch_size"] == int(cfg.training.global_batch_size)
    # optimizer steps = total samples / global batch; total samples (sequences) =
    # total_tokens // seq_len, matching the Megatron path's --train-samples.
    assert training["steps"] == int(cfg.training.total_tokens) // (
        int(cfg.base.model.seq_length) * int(cfg.training.global_batch_size)
    )
    # VERIFIED against v0.2.2: seed is [debug].seed, NOT [training].seed
    # (docs/torchtitan_api_notes.md §1).
    assert toml["debug"]["seed"] == int(cfg.seed)


def test_training_steps_override_is_honored():
    # An explicit training.steps must win over the token-budget derivation, else a
    # short smoke run builds the FULL multi-billion-sample dataset index (the
    # dataloader's num_samples = steps * global_batch_size).
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "experiment=optim/adam",
            "cluster=h100_de",
            "backend=torchtitan",
            "training.steps=20",
        ]
    )
    resolve_config(cfg)
    toml, _ = build_torchtitan_config(cfg)
    assert toml["training"]["steps"] == 20


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


def test_comm_block_raises_init_timeout_above_default():
    # The slm_megatron_indexed dataloader cold-builds its sample/shuffle index on
    # rank 0 while other ranks wait at a barrier; a large-corpus build exceeds
    # torchtitan's default comm.init_timeout_seconds=300 and crashes the run with a
    # barrier timeout before training starts. The builder must emit a [comm] block
    # with a generous init_timeout, overridable via cluster.comm_init_timeout_seconds.
    toml, _ = build_torchtitan_config(_cfg())
    assert toml["comm"]["init_timeout_seconds"] > 300
    assert toml["comm"]["init_timeout_seconds"] == 3600  # default when cluster knob unset


def test_comm_init_timeout_is_overridable_from_cluster():
    from omegaconf import OmegaConf

    cfg = _cfg()
    OmegaConf.set_struct(cfg, False)  # resolved cfg is struct-locked; allow injecting a field
    cfg.cluster.comm_init_timeout_seconds = 7200
    toml, _ = build_torchtitan_config(cfg)
    assert toml["comm"]["init_timeout_seconds"] == 7200


def test_validation_block_mirrors_megatron_eval_cadence():
    # torchtitan eval is enabled by default, mirroring the Megatron path's
    # eval_interval/eval_iters, so it logs validation_metrics/loss (-> val/loss)
    # on the same corpus. The titan_ext monkeypatch supplies the val-split loader.
    cfg = _cfg()
    toml, _ = build_torchtitan_config(cfg)
    val = toml["validation"]
    assert val["enable"] is True
    assert val["freq"] == int(cfg.training.get("eval_interval", 500))
    assert val["steps"] == int(cfg.training.get("eval_iters", 32))
    assert val["seq_len"] == int(cfg.base.model.seq_length)


def test_validation_block_omitted_when_eval_disabled():
    from omegaconf import OmegaConf

    cfg = _cfg()
    OmegaConf.set_struct(cfg, False)  # resolved cfg is struct-locked; allow editing
    cfg.training.eval_interval = 0
    toml, _ = build_torchtitan_config(cfg)
    assert "validation" not in toml


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
