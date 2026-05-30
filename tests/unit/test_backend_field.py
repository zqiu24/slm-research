"""The `backend` field selects the training backend and is recorded per run."""

from __future__ import annotations

from launchers.submit import _parse_overrides, resolve_config


def test_backend_defaults_to_megatron():
    cfg = _parse_overrides(["base/family=llama3", "experiment=optim/adam"])
    assert str(cfg.backend) == "megatron"


def test_backend_override_is_applied():
    cfg = _parse_overrides(["base/family=llama3", "experiment=optim/adam", "backend=torchtitan"])
    assert str(cfg.backend) == "torchtitan"


def test_resolve_stamps_torchtitan_sha():
    cfg = _parse_overrides(["base/family=llama3", "experiment=optim/adam"])
    resolve_config(cfg)
    # Present whether or not the submodule is checked out (falls back to a marker).
    assert "torchtitan_sha" in cfg._derived
    assert isinstance(str(cfg._derived.torchtitan_sha), str)
