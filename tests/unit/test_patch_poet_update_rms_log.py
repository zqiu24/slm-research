"""Tests for the POET update-RMS W&B logging patch."""

from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


def setup_function():
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_update_rms_log", None)


def teardown_function():
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_update_rms_log", None)


def test_patch_registers_with_no_owned_target():
    importlib.import_module("src.patches.poet_update_rms_log")
    entry = registered_patches()["poet_update_rms_log"]
    assert entry.targets == ()


def test_log_update_rms_emits_cached_optimizer_stats(monkeypatch):
    from src.patches.poet_update_rms_log import _log_update_rms

    captured = {}
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda data, step=None: captured.update({"step": step, **data}),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    opt = types.SimpleNamespace(
        last_update_rms_stats={
            "poet_update_rms/theta_mean": torch.tensor(0.015),
            "poet_update_rms/clamp_fraction": torch.tensor(0.25),
        }
    )
    _log_update_rms([opt], iteration=12)
    assert captured["step"] == 12
    assert captured["poet_update_rms/theta_mean"] == pytest.approx(0.015)
    assert captured["poet_update_rms/clamp_fraction"] == pytest.approx(0.25)


def test_install_on_setup_wraps_optimizer_step_and_logs(monkeypatch):
    from src.patches.poet_update_rms_log import _install_on_setup

    captured = []
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda data, step=None: captured.append((step, dict(data))),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    class FakeTorchOpt:
        def __init__(self):
            self.last_update_rms_stats = {"poet_update_rms/theta_max": torch.tensor(0.024)}

    class FakeWrappedOpt:
        def __init__(self):
            self.optimizer = FakeTorchOpt()
            self.steps = 0

        def step(self):
            self.steps += 1

    wrapped_opt = FakeWrappedOpt()

    def orig_setup():
        return "model", wrapped_opt, "scheduler"

    setup = _install_on_setup(orig_setup, interval=2)
    model, opt, sched = setup()
    assert (model, opt, sched) == ("model", wrapped_opt, "scheduler")

    opt.step()
    opt.step()
    opt.step()
    assert wrapped_opt.steps == 3
    assert [step for step, _ in captured] == [0, 2]
    assert captured[0][1]["poet_update_rms/theta_max"] == pytest.approx(0.024)
