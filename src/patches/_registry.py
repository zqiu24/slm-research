"""Patch registry with conflict detection and hashing.

Every patch in ``src/patches/<name>.py`` must register via ``@register_patch``
and provide a ``apply()`` that mutates upstream Megatron. The registry
refuses to apply two patches that target the same upstream function,
failing at registration time rather than silently (SPEC.md §5.5).

``apply_patches(names)`` applies each patch, records ``(name, patch_sha)``
pairs, and returns ``patch_set_hash = blake2s(sorted_patches)``.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class _PatchEntry:
    name: str
    apply_fn: Callable[[], None]
    targets: tuple[str, ...]  # "pkg.mod.ClassOrFn"
    source_sha: str  # hash of the patch's own source, for reproducibility
    applied: bool = field(default=False)


_REGISTRY: dict[str, _PatchEntry] = {}
_TARGET_OWNERS: dict[str, str] = {}


class PatchConflict(RuntimeError):  # noqa: N818
    pass


class UnknownPatch(KeyError):  # noqa: N818
    pass


def _source_sha(fn: Callable) -> str:
    try:
        src = inspect.getsource(inspect.getmodule(fn))  # type: ignore[arg-type]
    except (OSError, TypeError):
        src = fn.__qualname__
    return hashlib.blake2s(src.encode("utf-8"), digest_size=8).hexdigest()


def register_patch(*, name: str, targets: tuple[str, ...] = ()) -> Callable:
    """Decorator registering a patch's ``apply()`` function.

    ``targets`` is the list of upstream symbols the patch mutates (e.g.
    ``"megatron.core.transformer.transformer_block.TransformerBlock.forward"``).
    Two patches declaring overlapping targets raise ``PatchConflict``.
    """

    def decorator(apply_fn: Callable[[], None]) -> Callable[[], None]:
        if name in _REGISTRY:
            raise PatchConflict(f"Patch {name!r} already registered")
        for target in targets:
            owner = _TARGET_OWNERS.get(target)
            if owner is not None and owner != name:
                raise PatchConflict(f"Patch {name!r} and {owner!r} both target {target!r}")
            _TARGET_OWNERS[target] = name
        _REGISTRY[name] = _PatchEntry(
            name=name,
            apply_fn=apply_fn,
            targets=tuple(targets),
            source_sha=_source_sha(apply_fn),
        )
        return apply_fn

    return decorator


def patch_set_hash(names: list[str] | tuple[str, ...]) -> str:
    """Return the deterministic hash for registered patches without applying them."""
    names = sorted(set(names))
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise UnknownPatch(f"Unknown patches: {unknown}. Registered: {sorted(_REGISTRY)}")
    if not names:
        return "noop" + "0" * 12
    payload = "\n".join(f"{n}:{_REGISTRY[n].source_sha}" for n in names)
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()


def apply_patches(names: list[str] | tuple[str, ...]) -> str:
    """Apply the named patches in sorted order; return the patch-set hash.

    Applying the same patch twice is a no-op. Unknown names raise
    ``UnknownPatch`` before any side effect is performed.
    """
    names = sorted(set(names))
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise UnknownPatch(f"Unknown patches: {unknown}. Registered: {sorted(_REGISTRY)}")

    for name in names:
        entry = _REGISTRY[name]
        if entry.applied:
            continue
        entry.apply_fn()
        entry.applied = True

    return patch_set_hash(names)


def registered_patches() -> dict[str, _PatchEntry]:
    """Return a read-only snapshot of the registry (for introspection / tests)."""
    return dict(_REGISTRY)


def _reset_for_tests() -> None:
    """Clear registry state; only use from tests.

    Also pops cached `src.patches.*` modules from `sys.modules` so a
    subsequent `importlib.import_module(...)` re-executes the file and
    re-runs the `@register_patch` decorators. Without this, the registry
    and the module cache fall out of sync across tests.
    """
    import sys as _sys

    for mod_name in list(_sys.modules):
        if mod_name.startswith("src.patches.") and mod_name not in (
            "src.patches._registry",
            "src.patches",
        ):
            _sys.modules.pop(mod_name, None)
    _REGISTRY.clear()
    _TARGET_OWNERS.clear()
