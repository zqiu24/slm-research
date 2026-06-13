# tests/unit/test_patch_weight_norm_monitor.py
import sys

import pytest

from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    sys.modules.pop("src.patches.weight_norm_monitor", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.weight_norm_monitor", None)


def test_parse_layer_selection_keywords_and_indices():
    from src.patches.weight_norm_monitor import parse_layer_selection

    assert parse_layer_selection("first,mid,last", 12) == {0, 6, 11}
    assert parse_layer_selection("0,5,11", 12) == {0, 5, 11}
    assert parse_layer_selection("-1", 12) == {11}  # negative wraps
    assert parse_layer_selection("99", 12) == set()  # out of range dropped
    assert parse_layer_selection(" first , 3 ", 12) == {0, 3}  # whitespace tolerant


def test_classify_linear_matches_fused_and_unfused_types_and_skips_others():
    from src.patches.weight_norm_monitor import classify_linear

    # fused names
    assert classify_linear("decoder.layers.5.self_attention.linear_qkv") == (5, "qkv")
    assert classify_linear("decoder.layers.0.self_attention.linear_proj") == (0, "proj")
    assert classify_linear("module.decoder.layers.3.mlp.linear_fc1") == (3, "fc1")
    assert classify_linear("decoder.layers.7.mlp.linear_fc2") == (7, "fc2")
    # unfused names (--unfuse-qkv / --unfuse-fc1, e.g. head-aligned POET configs)
    assert classify_linear("decoder.layers.2.self_attention.linear_q") == (2, "q")
    assert classify_linear("decoder.layers.2.self_attention.linear_k") == (2, "k")
    assert classify_linear("decoder.layers.2.self_attention.linear_v") == (2, "v")
    assert classify_linear("decoder.layers.4.mlp.linear_fc1_gate") == (4, "fc1_gate")
    assert classify_linear("decoder.layers.4.mlp.linear_fc1_up") == (4, "fc1_up")
    # POET base-weight child module is NOT matched (name doesn't end in a type suffix)
    assert classify_linear("decoder.layers.5.self_attention.linear_qkv.poet_linear") is None
    # embeddings / lm_head / norms not matched
    assert classify_linear("embedding.word_embeddings") is None
    assert classify_linear("output_layer") is None


def test_should_log_cadence_for_adam_and_poet():
    from src.patches.weight_norm_monitor import should_log

    # non-POET: every `interval`
    assert should_log(100, 100, poet=False, merge_period=0) is True
    assert should_log(150, 100, poet=False, merge_period=0) is False
    assert should_log(0, 100, poet=False, merge_period=0) is False
    # POET merge_period=1: base == W_eff every step -> gated only by interval
    assert should_log(50, 50, poet=True, merge_period=1) is True
    # POET merge_period=400: only on merge boundaries
    assert should_log(400, 100, poet=True, merge_period=400) is True
    assert should_log(100, 100, poet=True, merge_period=400) is False
    # POET merge_period=0: frozen base -> never
    assert should_log(100, 100, poet=True, merge_period=0) is False
