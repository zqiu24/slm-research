"""Tests for the lr_decay_style_step patch."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.lr_decay_style_step", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.lr_decay_style_step", None)


def _import_and_apply():
    mod = importlib.import_module("src.patches.lr_decay_style_step")
    mod.apply()
    return mod


def _make_scheduler(
    monkeypatch, *, lr_warmup_steps: int, lr_decay_steps: int, max_lr: float, min_lr: float
):
    """Construct a Megatron OptimizerParamScheduler under cosine style, then
    flip it into step style with our patched attributes set explicitly. This
    sidesteps the get_args() lookup that the patched __init__ performs when
    constructed directly under style='step'.
    """
    _import_and_apply()
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    # Minimal optimizer stub: scheduler.step(0) needs param_groups iteration.
    opt = types.SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0}])

    sched = OptimizerParamScheduler(
        optimizer=opt,
        init_lr=0.0,
        max_lr=max_lr,
        min_lr=min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style="cosine",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1,
        wd_incr_style="constant",
    )
    sched.lr_decay_style = "step"
    sched.lr_decay_step_ratio = [0.8, 0.9]
    sched.lr_decay_step_coeff = [0.316, 0.1]
    return sched


def test_patch_registers_with_scheduler_targets():
    importlib.import_module("src.patches.lr_decay_style_step")
    reg = registered_patches()
    assert "lr_decay_style_step" in reg
    targets = reg["lr_decay_style_step"].targets
    assert any("OptimizerParamScheduler.__init__" in t for t in targets)
    assert any("OptimizerParamScheduler.get_lr" in t for t in targets)


def test_step_decay_curve_matches_reference(monkeypatch):
    sched = _make_scheduler(
        monkeypatch, lr_warmup_steps=0, lr_decay_steps=100, max_lr=1.0, min_lr=0.0
    )
    pg = {}

    # Anchor points based on Megatron-poet's formula:
    #   progress >= 0.9 -> 0.1   ; progress in [0.8, 0.9) -> 0.316 ; else 1.0
    sched.num_steps = 0
    assert sched.get_lr(pg) == pytest.approx(1.0)
    sched.num_steps = 50
    assert sched.get_lr(pg) == pytest.approx(1.0)
    sched.num_steps = 79
    assert sched.get_lr(pg) == pytest.approx(1.0)
    sched.num_steps = 80
    assert sched.get_lr(pg) == pytest.approx(0.316)
    sched.num_steps = 89
    assert sched.get_lr(pg) == pytest.approx(0.316)
    sched.num_steps = 90
    assert sched.get_lr(pg) == pytest.approx(0.1)
    sched.num_steps = 100
    assert sched.get_lr(pg) == pytest.approx(0.1)


def test_step_decay_warmup_is_linear(monkeypatch):
    # 10-step warmup, peak 1.0; partial-warmup step should be linear.
    sched = _make_scheduler(
        monkeypatch, lr_warmup_steps=10, lr_decay_steps=100, max_lr=1.0, min_lr=0.0
    )
    pg = {}
    sched.num_steps = 0
    assert sched.get_lr(pg) == pytest.approx(0.0)
    sched.num_steps = 5
    assert sched.get_lr(pg) == pytest.approx(0.5)
    sched.num_steps = 10
    assert sched.get_lr(pg) == pytest.approx(1.0)
    # First post-warmup step still at peak coeff 1.0
    sched.num_steps = 11
    assert sched.get_lr(pg) == pytest.approx(1.0)


def test_step_decay_past_decay_steps_returns_min_lr(monkeypatch):
    # Past lr_decay_steps the upstream `if self.num_steps > self.lr_decay_steps`
    # branch holds at min_lr regardless of style.
    sched = _make_scheduler(
        monkeypatch, lr_warmup_steps=0, lr_decay_steps=100, max_lr=1.0, min_lr=0.05
    )
    sched.num_steps = 101
    assert sched.get_lr({}) == pytest.approx(0.05)


def test_step_decay_uses_min_lr_floor(monkeypatch):
    # coeff applies to (max_lr - min_lr), then min_lr is added back.
    sched = _make_scheduler(
        monkeypatch, lr_warmup_steps=0, lr_decay_steps=100, max_lr=1.0, min_lr=0.05
    )
    sched.num_steps = 95  # coeff = 0.1
    expected = 0.05 + 0.1 * (1.0 - 0.05)
    assert sched.get_lr({}) == pytest.approx(expected)


def test_cosine_style_is_unchanged_after_patch(monkeypatch):
    # Patch is a routing patch: non-step styles must still hit the upstream
    # implementation and produce the same curve.
    _import_and_apply()
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    opt = types.SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0}])
    sched = OptimizerParamScheduler(
        optimizer=opt,
        init_lr=0.0,
        max_lr=1.0,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=100,
        lr_decay_style="cosine",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=1,
        wd_incr_style="constant",
    )
    # At decay_ratio = 0.5 -> cos(pi/2) = 0 -> coeff = 0.5
    sched.num_steps = 50
    assert sched.get_lr({}) == pytest.approx(0.5)


def test_step_decay_validation_rejects_unsorted_ratio(monkeypatch):
    from src.patches.lr_decay_style_step import _validate

    with pytest.raises(ValueError, match="sorted"):
        _validate([0.9, 0.8], [0.316, 0.1])


def test_step_decay_validation_rejects_mismatched_lengths(monkeypatch):
    from src.patches.lr_decay_style_step import _validate

    with pytest.raises(ValueError, match="same length"):
        _validate([0.8, 0.9], [0.316])


def test_step_decay_validation_rejects_out_of_range_ratio(monkeypatch):
    from src.patches.lr_decay_style_step import _validate

    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        _validate([0.0, 0.5], [1.0, 0.1])
    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        _validate([0.5, 1.0], [0.5, 0.1])


def test_step_decay_validation_rejects_missing_lists(monkeypatch):
    from src.patches.lr_decay_style_step import _validate

    with pytest.raises(ValueError, match="requires both"):
        _validate(None, [0.1])
    with pytest.raises(ValueError, match="requires both"):
        _validate([0.5], None)
