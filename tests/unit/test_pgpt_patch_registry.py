"""pgpt + POET patch set: registers without PatchConflict; hash is deterministic."""

import importlib

_PGPT_POET_PATCHES = [
    "model_unfuse_linears",
    "poet_apply_to_model",
    "poet_optimizer_setup",
    "poet_merge_step",
    "pgpt_apply_spec",
    "pgpt_optimizer_setup",
]


def _reload(names):
    from src.patches._registry import _reset_for_tests

    _reset_for_tests()
    for n in names:
        importlib.import_module(f"src.patches.{n}")


def test_pgpt_poet_patches_register_without_conflict():
    _reload(_PGPT_POET_PATCHES)
    from src.patches._registry import registered_patches

    reg = registered_patches()
    for n in _PGPT_POET_PATCHES:
        assert n in reg, f"{n} failed to register"


def test_pgpt_patch_set_hash_is_deterministic():
    from src.patches._registry import patch_set_hash

    _reload(_PGPT_POET_PATCHES)
    h1 = patch_set_hash(_PGPT_POET_PATCHES)
    _reload(_PGPT_POET_PATCHES)
    h2 = patch_set_hash(_PGPT_POET_PATCHES)
    assert h1 == h2 and len(h1) == 16
