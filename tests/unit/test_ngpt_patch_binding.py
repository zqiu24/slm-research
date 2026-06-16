"""Unit test for rebinding stale by-value imports of functions ngpt_apply_spec
wraps (the model_unfuse_linears-before-ngpt ordering bug). Covers both the
gpt_builder (in pretrain_gpt) and core_transformer_config_from_args (in
gpt_builders) bindings via the one generic helper."""

import sys
import types

from src.patches.ngpt_apply_spec import _rebind_if_stale


def _orig():  # sentinel originals
    return "orig"


def _wrapped():
    return "wrapped"


def test_rebinds_module_holding_original():
    fake = types.ModuleType("pretrain_gpt")
    fake.gpt_builder = _orig
    sys.modules["pretrain_gpt"] = fake
    try:
        _rebind_if_stale("pretrain_gpt", "gpt_builder", _orig, _wrapped)
        assert sys.modules["pretrain_gpt"].gpt_builder is _wrapped
    finally:
        del sys.modules["pretrain_gpt"]


def test_rebinds_config_function_in_gpt_builders():
    fake = types.ModuleType("gpt_builders")
    fake.core_transformer_config_from_args = _orig
    sys.modules["gpt_builders"] = fake
    try:
        _rebind_if_stale("gpt_builders", "core_transformer_config_from_args", _orig, _wrapped)
        assert sys.modules["gpt_builders"].core_transformer_config_from_args is _wrapped
    finally:
        del sys.modules["gpt_builders"]


def test_noop_when_module_not_imported():
    original = sys.modules.pop("pretrain_gpt", None)
    try:
        # Must not raise and must not create the module.
        _rebind_if_stale("pretrain_gpt", "gpt_builder", _orig, _wrapped)
        assert "pretrain_gpt" not in sys.modules
    finally:
        if original is not None:
            sys.modules["pretrain_gpt"] = original


def test_does_not_rebind_a_foreign_object():
    other = lambda: "other"  # noqa: E731
    fake = types.ModuleType("pretrain_gpt")
    fake.gpt_builder = other
    sys.modules["pretrain_gpt"] = fake
    try:
        _rebind_if_stale("pretrain_gpt", "gpt_builder", _orig, _wrapped)
        # Only rebinds if it currently holds the captured original.
        assert sys.modules["pretrain_gpt"].gpt_builder is other
    finally:
        del sys.modules["pretrain_gpt"]
