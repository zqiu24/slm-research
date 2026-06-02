"""Tests for the muon_kimi optimizer setup patch."""

from __future__ import annotations

import importlib
import sys
import types

from src.patches._registry import _reset_for_tests


def test_muon_kimi_optimizer_setup_registers_targets():
    _reset_for_tests()
    sys.modules.pop("src.patches.muon_kimi_optimizer_setup", None)

    importlib.import_module("src.patches.muon_kimi_optimizer_setup")

    from src.patches import registered_patches

    entry = registered_patches()["muon_kimi_optimizer_setup"]
    assert "megatron.training.training.get_megatron_optimizer_config" in entry.targets
    assert "megatron.training.training.get_megatron_optimizer" in entry.targets


def test_muon_kimi_setup_tags_config_when_slm_optimizer_is_muon_kimi(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.muon_kimi_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.muon_kimi_optimizer_setup")

    calls = []

    fake_training = types.SimpleNamespace()

    def original_get_config(args):
        cfg = types.SimpleNamespace(optimizer="adam", lr=1.0e-3)
        return cfg, {"from": "original"}

    def original_get_optimizer(config, model, **kwargs):
        calls.append(("original", config, model, kwargs))
        return "adam-optimizer"

    fake_training.get_megatron_optimizer_config = original_get_config
    fake_training.get_megatron_optimizer = original_get_optimizer

    fake_builder = types.ModuleType("src.optim.muon_kimi")

    def fake_muon_kimi_builder(config, model_chunks, **kwargs):
        calls.append(("muon_kimi", config, model_chunks, kwargs))
        return "muon-kimi-optimizer"

    fake_builder.get_megatron_muon_kimi_optimizer = fake_muon_kimi_builder

    # Mock the full megatron.training package chain so the patch's
    # `from megatron.training import training as _mt` doesn't trigger
    # loading of the real Megatron (which requires CUDA / transformer_engine).
    fake_megatron = types.ModuleType("megatron")
    fake_megatron_training_pkg = types.ModuleType("megatron.training")
    fake_megatron_training_pkg.training = fake_training
    fake_megatron.training = fake_megatron_training_pkg
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", fake_megatron_training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", fake_training)
    monkeypatch.setitem(sys.modules, "src.optim.muon_kimi", fake_builder)

    patch_mod.apply()

    # --- tag path: slm_optimizer == "muon_kimi" ---
    args = types.SimpleNamespace(slm_optimizer="muon_kimi")
    cfg, overrides = fake_training.get_megatron_optimizer_config(args)
    assert overrides == {"from": "original"}
    assert cfg.slm_optimizer == "muon_kimi"

    out = fake_training.get_megatron_optimizer(cfg, ["model"], use_gloo_process_groups=False)
    assert out == "muon-kimi-optimizer"
    assert calls[-1][0] == "muon_kimi"

    # --- delegate-to-original path: slm_optimizer != "muon_kimi" ---
    args_other = types.SimpleNamespace(slm_optimizer="adam")
    cfg_other, _ = fake_training.get_megatron_optimizer_config(args_other)
    assert not hasattr(cfg_other, "slm_optimizer") or cfg_other.slm_optimizer != "muon_kimi"

    out_other = fake_training.get_megatron_optimizer(cfg_other, ["model"])
    assert out_other == "adam-optimizer"
    assert calls[-1][0] == "original"
