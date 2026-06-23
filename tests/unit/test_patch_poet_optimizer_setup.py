"""Tests for the POET optimizer setup patch."""

from __future__ import annotations

import importlib
import sys
import types

from src.patches._registry import _reset_for_tests


def test_poet_optimizer_setup_registers_targets():
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_optimizer_setup", None)

    importlib.import_module("src.patches.poet_optimizer_setup")

    from src.patches import registered_patches

    entry = registered_patches()["poet_optimizer_setup"]
    assert "megatron.training.training.get_megatron_optimizer_config" in entry.targets
    assert "megatron.training.training.get_megatron_optimizer" in entry.targets


def test_poet_optimizer_setup_routes_slm_optimizer_to_poet_builder(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.poet_optimizer_setup")

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

    fake_builder = types.SimpleNamespace()

    def fake_poet_builder(config, model_chunks, **kwargs):
        calls.append(("poet", config, model_chunks, kwargs))
        return "poet-optimizer"

    fake_builder.get_megatron_poet_optimizer = fake_poet_builder

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
    monkeypatch.setitem(sys.modules, "src.optim.poet", fake_builder)

    patch_mod.apply()

    args = types.SimpleNamespace(
        slm_optimizer="poet",
        poet_merge_period=200,
        poet_scale=1.5,
        poet_block_size=256,
        poet_init_type="normalized",
        poet_mup_alpha=1.0,
    )
    cfg, overrides = fake_training.get_megatron_optimizer_config(args)
    assert overrides == {"from": "original"}
    assert cfg.slm_optimizer == "poet"
    assert cfg.poet_merge_period == 200
    assert cfg.poet_scale == 1.5

    out = fake_training.get_megatron_optimizer(cfg, ["model"], use_gloo_process_groups=False)
    assert out == "poet-optimizer"
    assert calls[-1][0] == "poet"


def test_get_config_threads_cache_mode(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.poet_optimizer_setup")

    fake_training = types.SimpleNamespace()

    def original_get_config(args):
        cfg = types.SimpleNamespace(optimizer="adam", lr=1e-3)
        return cfg, {}

    def original_get_optimizer(config, model, **kwargs):
        return "adam-optimizer"

    fake_training.get_megatron_optimizer_config = original_get_config
    fake_training.get_megatron_optimizer = original_get_optimizer

    fake_megatron = types.ModuleType("megatron")
    fake_megatron_training_pkg = types.ModuleType("megatron.training")
    fake_megatron_training_pkg.training = fake_training
    fake_megatron.training = fake_megatron_training_pkg
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", fake_megatron_training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", fake_training)

    patch_mod.apply()

    args = types.SimpleNamespace(
        slm_optimizer="poet",
        poet_merge_period=0,
        poet_scale=1.0,
        poet_block_size=256,
        poet_init_type="normalized",
        poet_mup_alpha=1.0,
        poet_cache_mode="cached_fwd_bwd",
    )
    cfg, _ = fake_training.get_megatron_optimizer_config(args)
    assert cfg.poet_cache_mode == "cached_fwd_bwd"


def test_get_config_copies_lie_ortho_knobs(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.poet_optimizer_setup")

    fake_training = types.SimpleNamespace()

    def original_get_config(args):
        cfg = types.SimpleNamespace(optimizer="adam", lr=1e-3)
        return cfg, {}

    def original_get_optimizer(config, model, **kwargs):
        return "adam-optimizer"

    fake_training.get_megatron_optimizer_config = original_get_config
    fake_training.get_megatron_optimizer = original_get_optimizer

    fake_megatron = types.ModuleType("megatron")
    fake_megatron_training_pkg = types.ModuleType("megatron.training")
    fake_megatron_training_pkg.training = fake_training
    fake_megatron.training = fake_megatron_training_pkg
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", fake_megatron_training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", fake_training)

    patch_mod.apply()

    args = types.SimpleNamespace(
        slm_optimizer="poet",
        poet_merge_period=1,
        poet_scale=0.5,
        poet_block_size=256,
        poet_init_type="normalized",
        poet_mup_alpha=1.0,
        poet_q_optimizer="lie_ortho",
        poet_lie_ortho_c=0.02,
        poet_lie_ortho_method="spectral",
        poet_lie_ortho_ns_steps=20,
        poet_lie_ortho_use_second_moment=True,
        poet_lie_ortho_decorrelate=True,
        poet_lie_ortho_decorrelate_mode="symmetric",
        poet_lie_ortho_angle_dim_exp=-0.5,
        hidden_size=512,
    )
    cfg, _ = fake_training.get_megatron_optimizer_config(args)
    assert cfg.poet_q_optimizer == "lie_ortho"
    assert cfg.poet_lie_ortho_c == 0.02
    assert cfg.poet_lie_ortho_method == "spectral"
    assert cfg.poet_lie_ortho_ns_steps == 20
    assert cfg.poet_lie_ortho_use_second_moment is True
    assert cfg.poet_lie_ortho_angle_dim_exp == -0.5
    # Regression guard: the decorrelate flag must reach `config` (poet.py reads it
    # from config, not args) — its absence silently no-op'd the §17.6 A/B.
    assert cfg.poet_lie_ortho_decorrelate is True
    assert cfg.poet_lie_ortho_decorrelate_mode == "symmetric"
    assert cfg.poet_lie_ortho_angle_dim_exp == -0.5
    # b_ref (=hidden_size) MUST land on config, else the angle scaling silently no-ops.
    assert cfg.poet_lie_ortho_angle_dim_ref == 512


def test_get_config_copies_lie_ortho_distributed(monkeypatch):
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_optimizer_setup", None)
    patch_mod = importlib.import_module("src.patches.poet_optimizer_setup")

    fake_training = types.SimpleNamespace()
    fake_training.get_megatron_optimizer_config = lambda args: (
        types.SimpleNamespace(optimizer="adam", lr=1e-3),
        {},
    )
    fake_training.get_megatron_optimizer = lambda config, model, **kwargs: "adam-optimizer"

    fake_megatron = types.ModuleType("megatron")
    fake_megatron_training_pkg = types.ModuleType("megatron.training")
    fake_megatron_training_pkg.training = fake_training
    fake_megatron.training = fake_megatron_training_pkg
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", fake_megatron_training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", fake_training)

    patch_mod.apply()
    args = types.SimpleNamespace(
        slm_optimizer="poet",
        poet_merge_period=1,
        poet_scale=0.5,
        poet_block_size=256,
        poet_init_type="normalized",
        poet_mup_alpha=1.0,
        poet_lie_ortho_distributed=True,
    )
    cfg, _ = fake_training.get_megatron_optimizer_config(args)
    assert cfg.poet_lie_ortho_distributed is True
