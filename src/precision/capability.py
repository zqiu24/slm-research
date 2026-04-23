"""Capability tags used to gate experiments against clusters.

An experiment declares ``required_capabilities``; a cluster declares
``capabilities``. The launcher refuses to submit when the experiment's
requirements are not a subset of the cluster's. See SPEC.md §5.4.
"""

from __future__ import annotations

CAPABILITIES: frozenset[str] = frozenset(
    {
        "bf16",
        "fp16",
        "fp8",  # H100, H800, B200
        "fp4",  # B200 only
        "nvlink",  # assumes NVLink; matters for TP>1
        "ib_fast",  # fast InfiniBand; DP perf, not numerics
    }
)


class CapabilityMismatch(RuntimeError):
    """Raised when an experiment needs capabilities the cluster lacks."""


def check(required: set[str] | list[str], available: set[str] | list[str]) -> set[str]:
    """Return the set of missing capabilities (empty iff compatible)."""
    required_set = set(required)
    unknown = required_set - CAPABILITIES
    if unknown:
        raise ValueError(f"Unknown capability tag(s): {sorted(unknown)}")
    return required_set - set(available)


def assert_compatible(
    required: set[str] | list[str],
    available: set[str] | list[str],
    *,
    cluster_name: str,
) -> None:
    missing = check(required, available)
    if missing:
        raise CapabilityMismatch(
            f"Cluster {cluster_name!r} lacks required capabilities: {sorted(missing)}. "
            f"Pick a cluster that advertises them, or drop the requirement from the experiment."
        )
