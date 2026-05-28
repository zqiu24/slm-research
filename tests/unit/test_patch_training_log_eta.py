"""Tests for training_log_eta patch registration and log-line rewriting."""

import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.training_log_eta", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.training_log_eta", None)


def test_eta_patch_registers():
    importlib.import_module("src.patches.training_log_eta")
    reg = registered_patches()
    assert "training_log_eta" in reg
    assert any("print_rank_last" in t for t in reg["training_log_eta"].targets)


# A real per-iteration line as emitted by Megatron's training_log.
_LINE = (
    " [2026-05-28 13:01:07.645545] iteration       50/   45776 |"
    " consumed samples:        25600 |"
    " elapsed time per iteration (ms): 263.6 |"
    " throughput per GPU (TFLOP/s/GPU): 177.2 |"
    " learning rate: 1.092267E-04 | global batch size:   512 |"
    " lm loss: 8.184724E+00 | loss scale: 1.0 | grad norm: 1.128 |"
    " number of skipped iterations:   0 | number of nan iterations:   0 |"
)


def test_rewrite_injects_eta_and_strips_fields():
    mod = importlib.import_module("src.patches.training_log_eta")
    out = mod._rewrite(_LINE)

    # ETA injected right after the iteration field. (45776-50)*0.2636s ~= 3h20m.
    assert "| ETA: 3h20m |" in out
    assert out.index("ETA:") < out.index("consumed samples")

    # Noise fields removed.
    for label in ("learning rate", "global batch size", "loss scale", "grad norm"):
        assert label not in out

    # Substantive fields preserved.
    for label in ("consumed samples", "lm loss", "number of nan iterations"):
        assert label in out


def test_rewrite_ignores_non_iteration_lines():
    mod = importlib.import_module("src.patches.training_log_eta")
    sep = "-" * 40
    assert mod._rewrite(sep) == sep
