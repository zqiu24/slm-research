"""pgpt_apply_spec: post-step role matcher + registration + POET-required guard."""

import importlib
import types

import pytest


def test_require_poet_enforces_poet():
    # spec §4.5 / §6: pgpt build must fail fast when args.poet is unset.
    from src.patches.pgpt_apply_spec import _require_poet

    with pytest.raises(RuntimeError, match="POET-required"):
        _require_poet(types.SimpleNamespace(ngpt=True, poet=False))
    with pytest.raises(RuntimeError):
        _require_poet(types.SimpleNamespace(ngpt=True))  # attr missing -> also raises
    _require_poet(types.SimpleNamespace(ngpt=True, poet=True))  # poet set -> no raise


def test_post_step_role_matches_embedding_and_lm_head():
    from src.patches.pgpt_apply_spec import _match_post_step_role

    assert _match_post_step_role("embedding.word_embeddings.weight") == "rows"
    assert _match_post_step_role("output_layer.weight") == "rows"
    # per-layer POET-wrapped matrices must NOT match (they are not re-projected)
    assert _match_post_step_role("decoder.layers.0.self_attention.linear_qkv.weight") is None
    assert _match_post_step_role("decoder.layers.0.mlp.linear_fc2.weight") is None


def test_pgpt_apply_spec_registers():
    from src.patches._registry import _reset_for_tests, registered_patches

    _reset_for_tests()
    importlib.import_module("src.patches.pgpt_apply_spec")
    assert "pgpt_apply_spec" in registered_patches()
