"""Entrypoint routing: base.model.entrypoint selects the per-rank module."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

import launchers.train_megatron as tm


def _cfg(entrypoint=None):
    model = {} if entrypoint is None else {"entrypoint": entrypoint}
    return OmegaConf.create(
        {
            "base": {"model": model},
            "cluster": {"gpus_per_node": 8},
            "_derived": {"run_dir": "runs/test"},
        }
    )


@pytest.fixture(autouse=True)
def _stub_megatron_args(monkeypatch):
    monkeypatch.setattr(tm, "build_megatron_args", lambda cfg: [])


def test_default_routes_to_gpt():
    assert "launchers.pretrain_gpt_slm" in tm.build_torchrun_command(_cfg())


def test_mamba_routes_to_mamba_module():
    assert "launchers.pretrain_mamba_slm" in tm.build_torchrun_command(_cfg("mamba"))


def test_unknown_entrypoint_raises():
    with pytest.raises(ValueError):
        tm.build_torchrun_command(_cfg("titan"))


def test_mamba_launcher_module_imports_without_megatron():
    # Module import must stay CPU-safe (Megatron imports live inside main()).
    import launchers.pretrain_mamba_slm as m

    assert callable(m.main)
