"""Human-readable diff between a resolved config and the current champion.

Emits a compact ``k1=v1, k2=v2`` string that goes into every W&B run as
``config_diff_from_champion`` — the primary column for daily comparison
(SPEC.md §5.3.1).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf

    _HAS_OMEGACONF = True
except ImportError:
    _HAS_OMEGACONF = False
    DictConfig = Any  # type: ignore[misc,assignment]


_MISSING = object()

DEFAULT_EXCLUDED: frozenset[str] = frozenset(
    {
        "cluster.nodes",
        "cluster.slurm_partition",
        "cluster.slurm_account",
        "wandb",
        "logging",
        "checkpointing.save_every_tokens",
        "checkpointing.keep_last",
        "seed",
        "_derived",
        "parallelism",
    }
)


def _walk(d: dict, prefix: str = "") -> Iterator[tuple[str, Any]]:
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _walk(v, path)
        else:
            yield path, v


def _get_path(d: dict, dotted: str, default: Any = _MISSING) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _compact(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_compact(x) for x in v) + "]"
    return json.dumps(v, default=str)


def _to_container(cfg: Any) -> dict:
    if _HAS_OMEGACONF and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return json.loads(json.dumps(cfg))


def diff_from_champion(
    resolved_cfg: Any,
    champion_cfg: Any,
    *,
    excluded: frozenset[str] = DEFAULT_EXCLUDED,
) -> str:
    """Return a compact diff string (or ``"champion"`` if they match)."""
    current = _to_container(resolved_cfg)
    champion = _to_container(champion_cfg)

    diffs: list[str] = []
    seen: set[str] = set()
    for path, value in _walk(current):
        seen.add(path)
        if any(path == ex or path.startswith(ex + ".") for ex in excluded):
            continue
        champ_val = _get_path(champion, path, default=_MISSING)
        if champ_val is _MISSING or champ_val != value:
            diffs.append(f"{path}={_compact(value)}")

    # Fields present in champion but removed in current.
    for path, _value in _walk(champion):
        if path in seen:
            continue
        if any(path == ex or path.startswith(ex + ".") for ex in excluded):
            continue
        diffs.append(f"{path}=<removed>")

    return ", ".join(sorted(diffs)) or "champion"
