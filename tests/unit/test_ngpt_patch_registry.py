"""nGPT patches: registration + hash determinism + conflict-freedom."""

import importlib


def _reload_patches(names):
    from src.patches._registry import _reset_for_tests

    _reset_for_tests()
    for n in names:
        importlib.import_module(f"src.patches.{n}")


def test_ngpt_patches_register_without_conflict():
    _reload_patches(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    from src.patches._registry import registered_patches

    reg = registered_patches()
    assert "ngpt_apply_spec" in reg
    assert "ngpt_normalize_step" in reg
    assert "ngpt_optimizer_setup" in reg


def test_ngpt_patch_set_hash_is_deterministic():
    from src.patches._registry import patch_set_hash

    _reload_patches(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    h1 = patch_set_hash(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    _reload_patches(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    h2 = patch_set_hash(["ngpt_apply_spec", "ngpt_normalize_step", "ngpt_optimizer_setup"])
    assert h1 == h2 and len(h1) == 16
