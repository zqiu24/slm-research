# tests/unit/test_patch_overfit_single_batch.py
import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SLM_OVERFIT_SINGLE_BATCH", raising=False)
    _reset_for_tests()
    sys.modules.pop("src.patches.overfit_single_batch", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.overfit_single_batch", None)


def test_patch_registers_with_unique_target():
    importlib.import_module("src.patches.overfit_single_batch")
    reg = registered_patches()
    assert "overfit_single_batch" in reg
    assert any("get_batch" in t for t in reg["overfit_single_batch"].targets)


def test_apply_is_inert_without_env(monkeypatch):
    """With the env var unset, apply() must NOT replace get_batch."""
    fake_mod = type(sys)("pretrain_gpt")
    sentinel = object()
    fake_mod.get_batch = sentinel
    monkeypatch.setitem(sys.modules, "pretrain_gpt", fake_mod)

    mod = importlib.import_module("src.patches.overfit_single_batch")
    mod.apply()

    assert fake_mod.get_batch is sentinel  # untouched


def test_apply_wraps_get_batch_when_enabled(monkeypatch):
    monkeypatch.setenv("SLM_OVERFIT_SINGLE_BATCH", "1")
    fake_mod = type(sys)("pretrain_gpt")
    seq = iter([("A",), ("B",)])
    fake_mod.get_batch = lambda *a, **k: next(seq)
    monkeypatch.setitem(sys.modules, "pretrain_gpt", fake_mod)

    mod = importlib.import_module("src.patches.overfit_single_batch")
    mod.apply()

    assert fake_mod.get_batch("iter") == ("A",)
    assert fake_mod.get_batch("iter") == ("A",)  # replayed, not advanced
