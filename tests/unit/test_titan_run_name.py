from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config


def test_megatron_run_name_has_no_backend_segment():
    cfg = _parse_overrides(["base/family=llama3", "experiment=champion", "seed=42"])
    resolve_config(cfg)
    assert "-torchtitan-" not in cfg._derived.run_name
    assert cfg._derived.run_name.startswith(
        f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}-s42-"
    )


def test_torchtitan_run_name_has_backend_segment():
    cfg = _parse_overrides(
        ["base/family=llama3", "experiment=champion", "seed=42", "backend=torchtitan"]
    )
    resolve_config(cfg)
    assert cfg._derived.run_name.startswith(
        f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}-torchtitan-s42-"
    )
