# tests/unit/test_patch_weight_delta_monitor.py
import sys

import pytest

from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.weight_delta_monitor", None)
    sys.modules.pop("src.patches.weight_norm_monitor", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.weight_delta_monitor", None)
    sys.modules.pop("src.patches.weight_norm_monitor", None)


class _FakeMod:
    def __init__(self, weight):
        self.weight = weight


class _FakeChunk:
    def __init__(self, named):
        self._named = named

    def named_modules(self):
        return iter(self._named)


def _fake_model():
    import torch

    named = [
        ("decoder.layers.0.self_attention.linear_qkv", _FakeMod(torch.ones(2, 2))),
        ("decoder.layers.0.mlp.linear_fc1", _FakeMod(torch.ones(3, 2) * 2.0)),
        ("decoder.layers.0.self_attention.linear_qkv.poet_linear", _FakeMod(torch.zeros(2, 2))),
        ("decoder.layers.1.self_attention.linear_qkv", _FakeMod(torch.ones(2, 2) * 3.0)),
        ("embedding.word_embeddings", _FakeMod(torch.ones(10, 2))),
    ]
    return [_FakeChunk(named)]


def test_compute_delta_w_stats_zero_delta_is_finite():
    import math

    import torch

    from src.patches.weight_delta_monitor import compute_delta_w_stats

    w = torch.eye(4)
    stats = compute_delta_w_stats(w, w)
    assert stats["fro_abs"] == 0.0
    assert stats["fro_rel"] == 0.0
    assert stats["w_fro_ratio"] == 1.0
    assert stats["cos_to_w"] == 0.0
    assert all(math.isfinite(v) for v in stats.values())


def test_compute_delta_w_stats_rank1_delta_has_rank_one():
    import torch

    from src.patches.weight_delta_monitor import compute_delta_w_stats

    before = torch.zeros(4, 4)
    after = torch.zeros(4, 4)
    after[:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    stats = compute_delta_w_stats(before, after)
    assert stats["stable_rank"] == pytest.approx(1.0)
    assert stats["effective_rank"] == pytest.approx(1.0)
    assert stats["stable_rank_frac"] == pytest.approx(0.25)
    assert stats["effective_rank_frac"] == pytest.approx(0.25)


def test_compute_delta_w_stats_identity_delta_has_full_rank():
    import torch

    from src.patches.weight_delta_monitor import compute_delta_w_stats

    before = torch.zeros(4, 4)
    after = torch.eye(4)
    stats = compute_delta_w_stats(before, after)
    assert stats["stable_rank"] == pytest.approx(4.0)
    assert stats["effective_rank"] == pytest.approx(4.0)
    assert stats["stable_rank_frac"] == pytest.approx(1.0)
    assert stats["effective_rank_frac"] == pytest.approx(1.0)
    assert stats["condition_number"] == pytest.approx(1.0)


def test_compute_delta_w_stats_radial_cosine_sign():
    import torch

    from src.patches.weight_delta_monitor import compute_delta_w_stats

    before = torch.eye(3)
    pos = compute_delta_w_stats(before, before * 1.1)
    neg = compute_delta_w_stats(before, before * 0.9)
    assert pos["cos_to_w"] > 0.0
    assert neg["cos_to_w"] < 0.0


def test_compute_delta_w_stats_caps_large_spectral_probe(monkeypatch):
    import torch

    import src.patches.weight_delta_monitor as wdm

    seen = {}

    def fake_block_spectral_stats(matrix, eps=1e-12):
        seen["shape"] = tuple(matrix.shape)
        one = torch.tensor([1.0])
        return {
            "condition_number": one,
            "stable_rank": one,
            "sigma_max_over_median": one,
            "effective_rank": one,
        }

    monkeypatch.setattr(wdm, "block_spectral_stats", fake_block_spectral_stats)

    before = torch.zeros(6, 4)
    after = torch.ones(6, 4)
    stats = wdm.compute_delta_w_stats(before, after, spectral_max_dim=3)

    assert seen["shape"] == (3, 3)
    assert stats["stable_rank"] == pytest.approx(1.0)
    assert stats["stable_rank_frac"] == pytest.approx(1.0 / 3.0)


def test_delta_w_snapshot_target_weights_filters_selected_layers_and_clones_cpu_float32():
    import torch

    from src.patches.weight_delta_monitor import snapshot_target_weights

    snaps = snapshot_target_weights(_fake_model(), {0}, max_targets=1)
    assert len(snaps) == 1
    snap = snaps[0]
    assert (snap.layer_idx, snap.matrix_type) == (0, "qkv")
    assert snap.module_name == "decoder.layers.0.self_attention.linear_qkv"
    assert snap.shape == (2, 2)
    assert snap.weight.device.type == "cpu"
    assert snap.weight.dtype == torch.float32


def test_log_delta_w_snapshots_emits_scalars_and_means(monkeypatch):
    import sys
    import types

    import torch

    from src.patches.weight_delta_monitor import (
        _log_delta_w_snapshots,
        snapshot_target_weights,
    )

    captured = {}
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda d, step=None: captured.update({"_step": step, **d}),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    model = _fake_model()
    before = snapshot_target_weights(model, {0}, max_targets=1)
    model[0]._named[0][1].weight = torch.ones(2, 2) + torch.eye(2)
    after = snapshot_target_weights(model, {0}, max_targets=1)

    _log_delta_w_snapshots(before, after, iteration=10)

    assert captured["_step"] == 10
    assert captured["deltaw/L0/qkv/fro_abs"] == pytest.approx(2.0**0.5)
    assert captured["deltaw/_mean/fro_abs"] == pytest.approx(2.0**0.5)


def test_delta_w_wrapper_snapshots_before_and_after_inner_train_step(monkeypatch):
    import sys
    import types

    import torch

    captured = {}
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda d, step=None: captured.update({"_step": step, **d}),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    mod = _FakeMod(torch.ones(2, 2))
    model = [_FakeChunk([("decoder.layers.0.self_attention.linear_qkv", mod)])]

    def orig_train_step(*args, **kwargs):
        mod.weight = torch.ones(2, 2) + torch.eye(2)
        return ("loss", "skipped", "grad", "extra")

    opts = types.SimpleNamespace(
        log_delta_w=True,
        log_delta_w_interval=10,
        delta_w_layers="0",
        delta_w_max_targets=0,
        num_layers=1,
        poet=False,
        poet_merge_period=0,
        tensor_model_parallel_size=1,
    )

    from src.patches.weight_delta_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    ret = wrapped(None, None, model, None, None, None, None, 10)

    assert ret == ("loss", "skipped", "grad", "extra")
    assert captured["_step"] == 10
    assert captured["deltaw/L0/qkv/fro_abs"] == pytest.approx(2.0**0.5)
    assert captured["deltaw/L0/qkv/w_fro_before"] == pytest.approx(2.0)


def test_delta_w_wrapper_is_noop_when_flag_off(monkeypatch):
    import types

    calls = {"n": 0}

    def orig_train_step(*args, **kwargs):
        calls["n"] += 1
        return "ret"

    opts = types.SimpleNamespace(log_delta_w=False)
    from src.patches.weight_delta_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    assert wrapped(None, None, None) == "ret"
    assert calls["n"] == 1


def test_poet_merge_period_gates_delta_w_logging(monkeypatch):
    import sys
    import types

    import torch

    captured = {}
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda d, step=None: captured.update({"_step": step, **d}),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    mod = _FakeMod(torch.ones(2, 2))
    model = [_FakeChunk([("decoder.layers.0.self_attention.linear_qkv", mod)])]

    def orig_train_step(*args, **kwargs):
        mod.weight = mod.weight + torch.eye(2)
        return "ret"

    opts = types.SimpleNamespace(
        log_delta_w=True,
        log_delta_w_interval=100,
        delta_w_layers="0",
        delta_w_max_targets=0,
        num_layers=1,
        poet=True,
        poet_merge_period=400,
        tensor_model_parallel_size=1,
    )

    from src.patches.weight_delta_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    assert wrapped(None, None, model, None, None, None, None, 100) == "ret"
    assert captured == {}

    assert wrapped(None, None, model, None, None, None, None, 400) == "ret"
    assert captured["_step"] == 400
    assert captured["deltaw/L0/qkv/fro_abs"] == pytest.approx(2.0**0.5)


def test_delta_w_skips_distributed_optimizer_eval_boundary(monkeypatch, caplog):
    import logging
    import sys
    import types

    import torch

    captured = {}
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda d, step=None: captured.update({"_step": step, **d}),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    mod = _FakeMod(torch.ones(2, 2))
    model = [_FakeChunk([("decoder.layers.0.self_attention.linear_qkv", mod)])]

    def orig_train_step(*args, **kwargs):
        mod.weight = mod.weight + torch.eye(2)
        return "ret"

    opts = types.SimpleNamespace(
        log_delta_w=True,
        log_delta_w_interval=250,
        delta_w_layers="0",
        delta_w_max_targets=0,
        delta_w_spectral_max_dim=128,
        num_layers=1,
        poet=False,
        poet_merge_period=0,
        tensor_model_parallel_size=1,
        use_distributed_optimizer=True,
        overlap_param_gather=True,
        eval_interval=500,
    )

    from src.patches.weight_delta_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    with caplog.at_level(logging.WARNING):
        assert wrapped(None, None, model, None, None, None, None, 500) == "ret"

    assert captured == {}
    assert any("eval-boundary" in r.getMessage() for r in caplog.records)

    assert wrapped(None, None, model, None, None, None, None, 750) == "ret"
    assert captured["_step"] == 750
    assert captured["deltaw/L0/qkv/fro_abs"] == pytest.approx(2.0**0.5)


def test_delta_w_poet_merge_period_zero_warns_and_skips(monkeypatch, caplog):
    import logging
    import sys
    import types

    import torch

    captured = {}
    fake_wandb = types.SimpleNamespace(run=object(), log=lambda d, step=None: captured.update(d))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    mod = _FakeMod(torch.ones(2, 2))
    model = [_FakeChunk([("decoder.layers.0.self_attention.linear_qkv", mod)])]

    def orig_train_step(*args, **kwargs):
        mod.weight = mod.weight + torch.eye(2)
        return "ret"

    opts = types.SimpleNamespace(
        log_delta_w=True,
        log_delta_w_interval=10,
        delta_w_layers="0",
        num_layers=1,
        poet=True,
        poet_merge_period=0,
    )

    from src.patches.weight_delta_monitor import _wrapped_train_step_factory

    wrapped = _wrapped_train_step_factory(orig_train_step, get_args=lambda: opts)
    with caplog.at_level(logging.WARNING):
        wrapped(None, None, model, None, None, None, None, 10)
        wrapped(None, None, model, None, None, None, None, 20)

    assert captured == {}
    warnings = [r for r in caplog.records if "merge_period=0" in r.getMessage()]
    assert len(warnings) == 1


def test_delta_w_patch_registers_with_empty_targets():
    import importlib

    from src.patches import registered_patches

    importlib.import_module("src.patches.weight_delta_monitor")
    reg = registered_patches()
    assert "weight_delta_monitor" in reg
    assert reg["weight_delta_monitor"].targets == ()


def test_delta_w_monitor_in_always_on_and_sorts_after_merge():
    from launchers.pretrain_gpt_slm import _ALWAYS_ON_PATCHES

    assert "weight_delta_monitor" in _ALWAYS_ON_PATCHES
    assert "weight_delta_monitor" > "poet_merge_step"
