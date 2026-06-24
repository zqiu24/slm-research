"""Tests for the pion optimizer setup patch."""

from __future__ import annotations

import importlib
import sys
import types

from src.patches._registry import _reset_for_tests


def test_pion_optimizer_setup_registers_targets():
    _reset_for_tests()
    sys.modules.pop("src.patches.pion_optimizer_setup", None)

    importlib.import_module("src.patches.pion_optimizer_setup")

    from src.patches import registered_patches

    entry = registered_patches()["pion_optimizer_setup"]
    assert "megatron.training.training.get_megatron_optimizer_config" in entry.targets
    assert "megatron.training.training.get_megatron_optimizer" in entry.targets


def test_pion_setup_tags_config_and_copies_pion_args(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.pion_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.pion_optimizer_setup")

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

    fake_builder = types.ModuleType("src.optim.pion")

    def fake_pion_builder(config, model_chunks, **kwargs):
        calls.append(("pion", config, model_chunks, kwargs))
        return "pion-optimizer"

    fake_builder.get_megatron_pion_optimizer = fake_pion_builder

    fake_megatron = types.ModuleType("megatron")
    fake_megatron_training_pkg = types.ModuleType("megatron.training")
    fake_megatron_training_pkg.training = fake_training
    fake_megatron.training = fake_megatron_training_pkg
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", fake_megatron_training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", fake_training)
    monkeypatch.setitem(sys.modules, "src.optim.pion", fake_builder)

    patch_mod.apply()

    # --- tag + copy path: slm_optimizer == "pion" ---
    args = types.SimpleNamespace(
        slm_optimizer="pion",
        pion_scaling="rms",
        pion_rms=0.2,
        pion_update_side="alternate",
        pion_momentum="transported_ambient_ambient",
        pion_degree=2,
        pion_beta1=0.9,
        pion_beta2=0.95,
        pion_use_second_momentum=False,
    )
    cfg, overrides = fake_training.get_megatron_optimizer_config(args)
    assert overrides == {"from": "original"}
    assert cfg.slm_optimizer == "pion"
    assert cfg.pion_scaling == "rms"
    assert cfg.pion_update_side == "alternate"
    assert cfg.pion_momentum == "transported_ambient_ambient"
    assert cfg.pion_beta2 == 0.95

    out = fake_training.get_megatron_optimizer(cfg, ["model"], use_gloo_process_groups=False)
    assert out == "pion-optimizer"
    assert calls[-1][0] == "pion"

    # --- delegate-to-original path: slm_optimizer != "pion" ---
    args_other = types.SimpleNamespace(slm_optimizer="adam")
    cfg_other, _ = fake_training.get_megatron_optimizer_config(args_other)
    assert not hasattr(cfg_other, "slm_optimizer") or cfg_other.slm_optimizer != "pion"

    out_other = fake_training.get_megatron_optimizer(cfg_other, ["model"])
    assert out_other == "adam-optimizer"
    assert calls[-1][0] == "original"
