"""Tests for parent torchrun command generation."""

from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config
from launchers.train_megatron import build_torchrun_command


def test_build_torchrun_command_targets_slm_entrypoint():
    cfg = _parse_overrides(["base/family=llama3", "experiment=optim/poet"])
    resolve_config(cfg)

    cmd = build_torchrun_command(cfg)

    assert cmd[:3] == ["torchrun", "--nproc_per_node", str(cfg.cluster.gpus_per_node)]
    assert "-m" in cmd
    assert "launchers.pretrain_gpt_slm" in cmd
    assert "--slm-config-path" in cmd
    config_path = cmd[cmd.index("--slm-config-path") + 1]
    assert config_path.endswith(f"{cfg._derived.run_dir}/resolved_config.yaml")
    assert "--slm-optimizer" in cmd
    assert "poet" in cmd


def test_run_dir_is_readable_name_with_timestamp():
    """Runs are identified by a readable, timestamped name (no config hash)."""
    cfg = _parse_overrides(["base/family=llama3", "experiment=champion", "seed=42"])
    resolve_config(cfg)

    run_name = cfg._derived.run_name
    assert run_name.startswith(f"{cfg.experiment.name}-{cfg.base.family}-{cfg.base.scale}-s42-")
    assert cfg._derived.run_dir == f"runs/{run_name}"
    # Trailing compact UTC timestamp: YYYYmmddTHHMMSSZ
    timestamp = run_name.rsplit("-", 1)[-1]
    assert timestamp.endswith("Z") and len(timestamp) == len("20260524T200332Z")


def test_build_torchrun_command_defaults_to_single_node(monkeypatch):
    """A local run with no launch-env vars must rendezvous as a single node.

    cfg.cluster.nodes describes the experiment's intended multi-node scale; if
    torchrun is told --nnodes=<that> on one machine it blocks forever waiting
    for peers that never join.
    """
    for var in ("NNODES", "SLURM_NNODES", "SLURM_JOB_NUM_NODES"):
        monkeypatch.delenv(var, raising=False)
    # Force a multi-node experiment scale; the launch must still default to one
    # node so local runs don't block forever in rendezvous waiting for peers.
    cfg = _parse_overrides(["base/family=llama3", "experiment=champion", "cluster.nodes=6"])
    resolve_config(cfg)
    assert int(cfg.cluster.nodes) == 6  # sanity: override applied

    cmd = build_torchrun_command(cfg)

    i = cmd.index("--nnodes")
    assert cmd[i + 1] == "1"


def test_build_torchrun_command_honors_env_node_count(monkeypatch):
    """When a launcher sets the node count in the env, torchrun must use it."""
    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.setenv("SLURM_NNODES", "4")
    cfg = _parse_overrides(["base/family=llama3", "experiment=champion"])
    resolve_config(cfg)

    cmd = build_torchrun_command(cfg)

    i = cmd.index("--nnodes")
    assert cmd[i + 1] == "4"


def test_decay_only_resolves_total_tokens_from_decay_tokens():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "scheduler=wsd_decay_only",
            "experiment=champion",
            "training_regime=final_wsd_decay_only",
            "cluster=h800_cn",
            "training.decay_tokens=1200000000",
            "training.stable_checkpoint_dir=/tmp/stable_ckpt",
        ]
    )
    resolve_config(cfg)  # must not raise on tokens_per_param: null
    assert int(cfg.training.total_tokens) == 1_200_000_000


def test_explicit_total_tokens_overrides_tokens_per_param():
    # ablation_20x sets tokens_per_param: 20, but an explicit budget must win,
    # including human-friendly "10B"-style strings from the CLI dotlist.
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=60m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
            "training.total_tokens=10B",
        ]
    )
    resolve_config(cfg)
    assert int(cfg.training.total_tokens) == 10_000_000_000


def test_default_path_still_uses_tokens_per_param():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=60m",
            "experiment=champion",
            "training_regime=ablation_20x",
            "cluster=h800_cn",
        ]
    )
    resolve_config(cfg)
    # 20 tokens/param * 60M non-embedding params
    assert int(cfg.training.total_tokens) == 1_200_000_000


def test_decay_tokens_still_beats_explicit_total_tokens():
    cfg = _parse_overrides(
        [
            "base/family=llama3",
            "base/scale=300m",
            "scheduler=wsd_decay_only",
            "experiment=champion",
            "training_regime=final_wsd_decay_only",
            "cluster=h800_cn",
            "training.decay_tokens=1200000000",
            "training.stable_checkpoint_dir=/tmp/stable_ckpt",
            "training.total_tokens=10B",
        ]
    )
    resolve_config(cfg)
    assert int(cfg.training.total_tokens) == 1_200_000_000
