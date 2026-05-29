"""CPU unit tests for the POET adapter's pure-Python helpers.

Run from the vendored stack root so `megatron` resolves to the vendored copy:
    cd poet_torch_huawei && PYTHONPATH=. python -m pytest tests_poet/test_adapter_unit.py -v
adapter.py only imports torch at module level (Megatron imports are deferred
inside functions), so these tests need no CUDA / Megatron build.
"""

import pytest

from megatron.core.poet_adapter.adapter import _name_matches

# The default POET leaf-exclusion list (adapter.install_poet_in_model).
EXCLUDE = ("lm_head", "output_layer", "embedding", "word_embeddings", "router", "gate", "mtp")


@pytest.mark.parametrize(
    "name",
    [
        "module.decoder.layers.1.mlp.router.weight",
        "module.mtp.layers.0.transformer_layer.mlp.router.weight",
        "decoder.layers.0.output_layer",
        "embedding.word_embeddings",
    ],
)
def test_excluded_names_still_match(name):
    assert _name_matches(name, EXCLUDE) is True


@pytest.mark.parametrize(
    "name",
    [
        # The new split halves must NOT be caught by the "gate" pattern.
        "decoder.layers.0.mlp.linear_fc1_gate",
        "decoder.layers.1.mlp.experts.local_experts.3.linear_fc1_gate",
        "decoder.layers.0.mlp.linear_fc1_up",
        "decoder.layers.1.self_attention.linear_q",
    ],
)
def test_fc1_split_halves_not_excluded(name):
    assert _name_matches(name, EXCLUDE) is False


def test_ancestor_dotted_pattern_still_matches():
    # exclude_ancestors carries the dotted ".experts." pattern; it must keep
    # matching real expert paths while not matching shared_experts.
    assert _name_matches("decoder.layers.1.mlp.experts.local_experts.0.linear_fc1", (".experts.",)) is True
    assert _name_matches("decoder.layers.1.mlp.shared_experts.linear_fc1", (".experts.",)) is False
