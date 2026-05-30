from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from launchers.train_torchtitan import build_torchrun_command


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


def test_command_targets_torchtitan_train():
    cfg = _cfg()
    cmd = build_torchrun_command(cfg)
    assert cmd[:3] == ["torchrun", "--nproc_per_node", str(cfg.cluster.gpus_per_node)]
    assert "-m" in cmd and "torchtitan.train" in cmd
    assert "--job.config_file" in cmd
    toml_path = cmd[cmd.index("--job.config_file") + 1]
    assert toml_path.endswith(f"{cfg._derived.run_dir}/torchtitan.toml")


def test_command_registers_slm_extension():
    cfg = _cfg()
    cmd = build_torchrun_command(cfg)
    # The slm extension (flavor + dataloader + scheduler) must be imported by
    # torchtitan before TrainSpec lookup (docs/torchtitan_api_notes.md §2).
    assert any("src.titan_ext" in part for part in cmd)
