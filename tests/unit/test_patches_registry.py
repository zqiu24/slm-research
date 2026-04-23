"""Tests for the patch registry (SPEC.md §5.5)."""

from __future__ import annotations

import pytest

from src.patches import (
    PatchConflict,
    UnknownPatch,
    apply_patches,
    register_patch,
)
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def test_apply_empty_returns_sentinel_hash():
    h = apply_patches([])
    assert len(h) == 16
    assert h.startswith("noop")


def test_register_and_apply_single_patch():
    calls: list[str] = []

    @register_patch(name="patch_a", targets=("pkg.mod.fn_a",))
    def _apply():
        calls.append("a")

    h = apply_patches(["patch_a"])
    assert calls == ["a"]
    assert len(h) == 16 and h != "noop" + "0" * 12


def test_duplicate_registration_conflicts():
    @register_patch(name="p")
    def _a():
        pass

    with pytest.raises(PatchConflict):

        @register_patch(name="p")
        def _b():
            pass


def test_target_collision_conflicts():
    @register_patch(name="p1", targets=("pkg.mod.X",))
    def _a():
        pass

    with pytest.raises(PatchConflict, match="pkg.mod.X"):

        @register_patch(name="p2", targets=("pkg.mod.X",))
        def _b():
            pass


def test_unknown_patch_raises_before_side_effects():
    applied: list[str] = []

    @register_patch(name="real")
    def _apply():
        applied.append("real")

    with pytest.raises(UnknownPatch):
        apply_patches(["real", "ghost"])
    assert applied == []  # nothing ran


def test_hash_independent_of_input_order():
    @register_patch(name="x")
    def _x():
        pass

    @register_patch(name="y")
    def _y():
        pass

    h1 = apply_patches(["x", "y"])
    _reset_for_tests()

    @register_patch(name="x")
    def _x2():
        pass

    @register_patch(name="y")
    def _y2():
        pass

    h2 = apply_patches(["y", "x"])
    # The source SHA depends on the module source text, which is identical
    # across both registrations -> hash is order-invariant.
    assert h1 == h2
