# tests/unit/test_patch_poet_grad_conditioning.py
import importlib
import sys

import pytest

from src.patches import registered_patches
from src.patches._registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SLM_POET_GRAD_CONDITIONING", raising=False)
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_grad_conditioning", None)
    yield
    _reset_for_tests()
    sys.modules.pop("src.patches.poet_grad_conditioning", None)


def test_patch_registers_with_unique_target():
    importlib.import_module("src.patches.poet_grad_conditioning")
    reg = registered_patches()
    assert "poet_grad_conditioning" in reg
    # unique label, NOT the real get_megatron_optimizer symbol owned by poet_optimizer_setup
    assert all("get_megatron_optimizer" not in t for t in reg["poet_grad_conditioning"].targets)


def test_select_target_params_picks_representative_blocks():
    """Pure selection logic is CPU-testable without Megatron."""
    import torch

    from src.patches.poet_grad_conditioning import select_target_params

    class FakePOET:
        def __init__(self, name, b):
            self.block_size_in = b
            self.block_size_out = b
            # one block's worth of upper-tri entries: b*(b-1)/2
            self.oft_R_in = torch.zeros(1, b * (b - 1) // 2)
            self.oft_R_out = torch.zeros(1, b * (b - 1) // 2)
            self._name = name

    layers = {
        "decoder.layers.0.self_attention.linear_q": FakePOET("q0", 4),
        "decoder.layers.5.self_attention.linear_v": FakePOET("v5", 4),
        "decoder.layers.9.mlp.linear_fc2": FakePOET("down9", 4),
        "decoder.layers.1.mlp.linear_no_match": FakePOET("x1", 4),
    }
    targets = select_target_params(layers.items(), max_targets=8)
    labels = {t["label"] for t in targets}
    # q/v/down projections selected (both R_in and R_out factors), the no-match dropped
    assert any("linear_q" in lbl for lbl in labels)
    assert any("linear_v" in lbl for lbl in labels)
    assert any("linear_fc2" in lbl for lbl in labels)
    assert not any("linear_no_match" in lbl for lbl in labels)
    # each selected layer contributes its R_in and R_out factor
    assert all(t["factor"] in ("R_in", "R_out") for t in targets)
