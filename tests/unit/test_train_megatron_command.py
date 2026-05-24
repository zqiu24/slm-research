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
    assert "--slm-config-hash" in cmd
    assert str(cfg._derived.config_hash) in cmd
    assert "--slm-optimizer" in cmd
    assert "poet" in cmd


def test_build_torchrun_command_defaults_to_single_node(monkeypatch):
    """A local run with no launch-env vars must rendezvous as a single node.

    cfg.cluster.nodes describes the experiment's intended multi-node scale; if
    torchrun is told --nnodes=<that> on one machine it blocks forever waiting
    for peers that never join.
    """
    for var in ("NNODES", "SLURM_NNODES", "SLURM_JOB_NUM_NODES"):
        monkeypatch.delenv(var, raising=False)
    cfg = _parse_overrides(["base/family=llama3", "experiment=champion"])
    resolve_config(cfg)
    assert int(cfg.cluster.nodes) > 1  # precondition: default cluster is multi-node

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
