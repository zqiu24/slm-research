"""Unit tests for the slm-research Muon adapter."""

from __future__ import annotations

import sys
import types


def test_muon_adapter_lazy_routes_to_megatron_builder(monkeypatch):
    from src.optim import muon as muon_mod

    calls = []

    fake_muon_module = types.SimpleNamespace()

    def fake_get_megatron_muon_optimizer(config, model_chunks, **kwargs):
        calls.append((config, model_chunks, kwargs))
        return "muon-optimizer"

    fake_muon_module.get_megatron_muon_optimizer = fake_get_megatron_muon_optimizer
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer.muon", fake_muon_module)

    cfg = types.SimpleNamespace()
    out = muon_mod.get_megatron_muon_optimizer(
        cfg,
        ["model"],
        config_overrides={"x": 1},
        use_gloo_process_groups=False,
        layer_wise_distributed_optimizer=False,
    )

    assert out == "muon-optimizer"
    assert calls[0][0] is cfg
    assert calls[0][1] == ["model"]
    assert calls[0][2]["config_overrides"] == {"x": 1}
