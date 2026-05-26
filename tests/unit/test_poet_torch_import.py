"""Smoke test: poet_torch package imports and exposes POETLinear."""

import inspect


def test_poet_torch_importable():
    import poet_torch

    assert hasattr(poet_torch, "POETLinear")


def test_poet_linear_constructor_signature():
    from poet_torch import POETLinear

    sig = inspect.signature(POETLinear.__init__)
    params = sig.parameters
    for required in ("in_features", "out_features", "bsz"):
        assert required in params, f"POETLinear.__init__ missing {required!r}"


def test_poet_linear_has_merge_then_reinitialize():
    """The merge step (Task 8) calls this method on every POETLinear."""
    from poet_torch import POETLinear

    assert hasattr(POETLinear, "merge_then_reinitialize")
