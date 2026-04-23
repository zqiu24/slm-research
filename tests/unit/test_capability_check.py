"""Unit tests for capability tagging (SPEC.md §5.4, §10.1)."""

from __future__ import annotations

import pytest

from src.precision.capability import (
    CAPABILITIES,
    CapabilityMismatch,
    assert_compatible,
    check,
)


def test_empty_requirements_always_compatible():
    assert check(set(), {"bf16"}) == set()
    assert check([], []) == set()


def test_subset_is_compatible():
    assert check({"bf16", "fp8"}, {"bf16", "fp8", "fp4"}) == set()


def test_superset_reports_missing():
    assert check({"fp4"}, {"bf16", "fp8"}) == {"fp4"}


def test_assert_raises_on_missing():
    with pytest.raises(CapabilityMismatch, match="fp4"):
        assert_compatible({"fp4"}, {"bf16", "fp8"}, cluster_name="h800_cn")


def test_unknown_capability_is_rejected():
    with pytest.raises(ValueError, match="Unknown capability"):
        check({"fp2"}, {"bf16"})


def test_expected_caps_present():
    assert {"bf16", "fp16", "fp8", "fp4", "nvlink", "ib_fast"} <= CAPABILITIES
