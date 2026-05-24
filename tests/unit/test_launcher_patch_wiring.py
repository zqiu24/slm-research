"""Verify launcher imports experiment patches and computes a patch_set_hash."""

import sys

import pytest
from omegaconf import OmegaConf

from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    for name in (
        "src.patches.poet_unfuse_te_impl",
        "src.patches.poet_apply_to_model",
        "src.patches.poet_merge_step",
    ):
        sys.modules.pop(name, None)
    yield
    _reset_for_tests()


def test_register_experiment_patches_with_empty_list_returns_sentinel():
    from launchers.submit import _register_experiment_patches

    cfg = OmegaConf.create({"experiment": {"patches": []}})
    h = _register_experiment_patches(cfg)
    assert len(h) == 16 and h.startswith("noop")


def test_register_experiment_patches_imports_and_registers():
    # The parent launcher only imports patch modules (triggering @register_patch)
    # and hashes them; it does NOT apply them — patches are applied per-rank in
    # launchers.pretrain_gpt_slm. So importing on CPU here is safe.
    from launchers.submit import _register_experiment_patches
    from src.patches import registered_patches

    cfg = OmegaConf.create(
        {"experiment": {"patches": ["poet_unfuse_te_impl", "poet_apply_to_model"]}}
    )
    h = _register_experiment_patches(cfg)

    reg = registered_patches()
    assert "poet_unfuse_te_impl" in reg
    assert "poet_apply_to_model" in reg
    assert len(h) == 16 and not h.startswith("noop")


def test_register_experiment_patches_unknown_name_raises():
    from launchers.submit import _register_experiment_patches

    cfg = OmegaConf.create({"experiment": {"patches": ["this_patch_does_not_exist"]}})
    with pytest.raises(ModuleNotFoundError):
        _register_experiment_patches(cfg)
