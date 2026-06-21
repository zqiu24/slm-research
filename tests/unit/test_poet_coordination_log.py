"""Tests for the POET coordination-diagnostic install glue
(src/patches/poet_coordination_log.py): the CPU-testable pieces — target
selection, W_perm un-permutation, and the per-layer collect that bridges the
optimizer state to src.diag.poet_coordination_diag. The Megatron/wandb wiring
(state lookup, step-hook install) mirrors the proven poet_grad_conditioning
pattern and is exercised on real runs.
"""

from types import SimpleNamespace

import pytest
import torch

from src.patches.poet_coordination_log import (
    collect_metrics_for_layer,
    select_target_layers,
    w_perm_frame,
)


@pytest.fixture(autouse=True)
def _isolate_default_dtype():
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(torch.float32)


def _ortho5(skew):
    from src.optim.poet_skew_muon import orthogonalize_skew_direction

    return orthogonalize_skew_direction(skew, method="muon", ns_steps=5)


def test_select_requires_a_wanted_projection_and_both_sides():
    two_sided = SimpleNamespace(oft_R_in=object(), oft_R_out=object())
    one_sided = SimpleNamespace(oft_R_in=object(), oft_R_out=None)
    named = [
        ("decoder.layers.0.self_attention.linear_q", two_sided),
        ("decoder.layers.0.mlp.linear_fc1", two_sided),
        ("decoder.layers.0.self_attention.core_attention", two_sided),  # not wanted
        ("decoder.layers.1.self_attention.linear_q", one_sided),  # missing out side
    ]
    labels = [t["label"] for t in select_target_layers(named)]
    assert "decoder.layers.0.self_attention.linear_q" in labels
    assert "decoder.layers.0.mlp.linear_fc1" in labels
    assert "decoder.layers.0.self_attention.core_attention" not in labels
    assert "decoder.layers.1.self_attention.linear_q" not in labels


def test_select_caps_at_max_targets():
    layer = SimpleNamespace(oft_R_in=object(), oft_R_out=object())
    named = [(f"decoder.layers.{i}.mlp.linear_fc1", layer) for i in range(20)]
    assert len(select_target_layers(named, max_targets=5)) == 5


def test_w_perm_frame_inverts_the_forward_permutation():
    # forward-frame weight -> W_perm frame is weight[perm_out_inv][:, perm_in_inv].
    w = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    perm_out_inv = torch.tensor([2, 0, 1], dtype=torch.int32)
    perm_in_inv = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
    layer = SimpleNamespace(weight=w, perm_out_inv=perm_out_inv, perm_in_inv=perm_in_inv)
    got = w_perm_frame(layer)
    exp = w.index_select(0, perm_out_inv.long()).index_select(1, perm_in_inv.long())
    assert torch.equal(got, exp)


def test_collect_metrics_for_layer_returns_metric_dict():
    b = 4
    ne = b * (b - 1) // 2
    layer = SimpleNamespace(
        weight=torch.randn(2 * b, 2 * b),
        perm_out_inv=torch.arange(2 * b, dtype=torch.int32),
        perm_in_inv=torch.arange(2 * b, dtype=torch.int32),
        block_size_out=b,
        block_size_in=b,
    )
    lie_grad = (
        torch.randn(2, ne),  # lie_m_out
        torch.randn(2, ne),  # grad_out
        torch.randn(2, ne),  # lie_m_in
        torch.randn(2, ne),  # grad_in
    )
    m = collect_metrics_for_layer(layer, lie_grad, _ortho5)
    for k in ("mom_cos_out", "mom_cos_in", "cos_D_out_D_in", "r_joint", "gram_cond"):
        assert k in m
        assert isinstance(m[k], float)
