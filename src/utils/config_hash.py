"""Deterministic config hashing.

The invariant (SPEC.md §5.3): two runs with the same ``config_hash`` must
produce the same curve up to seed variance. Any field that violates this
invariant must either be included in the hash, or be excluded *because* it
has been proven numerically neutral (see ``tests/numerics/``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf

    _HAS_OMEGACONF = True
except ImportError:  # allows unit tests to run without omegaconf present
    _HAS_OMEGACONF = False
    DictConfig = Any  # type: ignore[misc,assignment]


EXCLUDED_FROM_CONFIG_HASH: frozenset[str] = frozenset(
    {
        # Volatile / allocation-time fields
        "cluster.nodes",
        "cluster.slurm_partition",
        "cluster.slurm_account",
        # WandB / logging
        "wandb",
        "logging",
        # Checkpoint cadence doesn't affect the model
        "checkpointing.save_every_tokens",
        "checkpointing.keep_last",
        # Seed is intentionally excluded — grouping across seeds depends on this
        "seed",
        # Derived fields populated post-hoc
        "_derived",
        # Parallelism realization: same experiment runs same model across TP/PP configs.
        # Excluded only because tests/numerics/test_patch_neutrality.py verifies it.
        "parallelism",
    }
)


def _delete_path(container: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cur: Any = container
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def _to_container(cfg: Any) -> dict:
    if _HAS_OMEGACONF and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, dict):
        return json.loads(json.dumps(cfg))  # deep copy via JSON round-trip
    raise TypeError(f"Cannot hash config of type {type(cfg)!r}")


def config_hash(
    resolved_cfg: Any,
    *,
    excluded: frozenset[str] = EXCLUDED_FROM_CONFIG_HASH,
) -> str:
    """Return a 16-char blake2s hex digest of the experiment-relevant config."""
    container = _to_container(resolved_cfg)
    for path in excluded:
        _delete_path(container, path)
    serialized = json.dumps(container, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2s(serialized.encode("utf-8"), digest_size=8).hexdigest()
