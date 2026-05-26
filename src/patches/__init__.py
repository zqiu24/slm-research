"""Patch layer — the escape hatch for things ModuleSpec can't reach.

See SPEC.md §5.5. Each patch lives in its own file with a required docstring
citing the upstream function and Megatron SHA; registration and hashing go
through ``_registry``.
"""

from src.patches._registry import (
    PatchConflict,
    UnknownPatch,
    apply_patches,
    patch_set_hash,
    register_patch,
    registered_patches,
)

__all__ = [
    "PatchConflict",
    "UnknownPatch",
    "apply_patches",
    "patch_set_hash",
    "register_patch",
    "registered_patches",
]
