"""Tests for the wandb_metric_normalize patch (Megatron interceptor)."""

import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.wandb_metric_normalize", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.wandb_metric_normalize", None)


def test_patch_registers_with_empty_targets():
    # Empty targets so it composes with log_grad_norm_extra (which owns
    # training_log) without a PatchConflict.
    importlib.import_module("src.patches.wandb_metric_normalize")
    reg = registered_patches()
    assert "wandb_metric_normalize" in reg
    assert reg["wandb_metric_normalize"].targets == ()


def test_wrap_wandb_log_renames_megatron_keys():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    captured = {}

    def fake_log(data, *args, **kwargs):
        captured["data"] = data
        captured["args"] = args

    wrapped = mod._wrap_wandb_log(fake_log)
    wrapped({"lm loss": 2.5, "grad-norm": 1.0, "num-zeros": 3}, 100)

    assert captured["data"] == {"train/loss": 2.5, "train/grad_norm": 1.0, "num-zeros": 3}
    assert captured["args"] == (100,)  # the positional step is preserved


def test_wrap_wandb_log_passes_non_dict_through_untouched():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    seen = {}

    def fake_log(data, *a, **k):
        seen["data"] = data

    mod._wrap_wandb_log(fake_log)("not-a-dict")
    assert seen["data"] == "not-a-dict"


def test_extra_metrics_first_call_only_tokens():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    metrics, state = mod._extra_metrics(
        consumed_samples=10, seq_length=2048, iteration=10, now=100.0, last=None
    )
    assert metrics == {"train/tokens_seen": 10 * 2048}
    assert state == {"time": 100.0, "tokens": 10 * 2048, "iter": 10}


def test_extra_metrics_second_call_adds_step_time():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    last = {"time": 100.0, "tokens": 20480, "iter": 10}  # 10 samples * 2048
    metrics, state = mod._extra_metrics(
        consumed_samples=20, seq_length=2048, iteration=20, now=110.0, last=last
    )
    # tokens now = 20*2048 = 40960; dt = 10s; steps = 20-10 = 10.
    assert metrics["train/tokens_seen"] == 40960
    assert metrics["perf/step_time_s"] == pytest.approx(1.0)  # 10s / 10 steps
    # Throughput is intentionally NOT emitted (not comparable across backends).
    assert "perf/throughput_tps" not in metrics
    assert state == {"time": 110.0, "tokens": 40960, "iter": 20}


def test_extra_metrics_zero_dt_skips_step_time():
    mod = importlib.import_module("src.patches.wandb_metric_normalize")
    last = {"time": 100.0, "tokens": 20480, "iter": 10}
    metrics, _ = mod._extra_metrics(
        consumed_samples=20, seq_length=2048, iteration=20, now=100.0, last=last
    )
    assert metrics == {"train/tokens_seen": 40960}  # no step time when dt == 0
