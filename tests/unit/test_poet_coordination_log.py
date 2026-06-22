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


def test_w_perm_frame_is_the_block_contiguous_weight():
    # ``POETLinear.weight`` is ALREADY in the block-contiguous frame the generators
    # operate in (the forward permutes x/y, never the weight — see
    # chain_layer_x_fast_decoupled / single_step.py). So w_perm_frame must return the
    # weight unchanged (fp32, detached); re-permuting it scrambles the block alignment
    # and was the cause of the run's validate_cos~0 (NOT a DP local-vs-global mismatch).
    w = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    perm_out_inv = torch.tensor([2, 0, 1], dtype=torch.int32)
    perm_in_inv = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
    layer = SimpleNamespace(weight=w, perm_out_inv=perm_out_inv, perm_in_inv=perm_in_inv)
    got = w_perm_frame(layer)
    assert torch.equal(got, w.to(torch.float32))


def test_w_perm_frame_matches_live_poet_single_step_backward():
    # End-to-end frame check against the REAL module (the synthetic algebra test in
    # test_poet_coordination_diag.py never touches POETLinear, so it could not catch a
    # live frame error). Build a single-step POETLinear at R=I, run fwd/bwd, capture the
    # ambient G via the wsplit hook logic, and confirm the wsplit validate quantity
    # block_skew(w_perm^T @ g_perm) aligns with the real oft_R_in.grad (|cos| ~ 1).
    poet_torch = pytest.importorskip("poet_torch")
    from src.diag.poet_coordination_diag import block_diag_skew
    from src.diag.skew_conditioning import vec_to_skew

    torch.manual_seed(0)
    in_f, out_f, bsz = 24, 32, 8
    layer = poet_torch.POETLinear(in_f, out_f, bsz=bsz, bias=False, parameterization="cayley")
    layer.single_step_fast = True  # the run's path: R=I closed-form oft_R grad
    with torch.no_grad():
        layer.weight.copy_(torch.randn(out_f, in_f))  # oft_R_* are zeros -> R=I

    cap = {}
    x = torch.randn(7, in_f)
    gy = torch.randn(7, out_f)

    def _fwd_hook(_m, inp, out):
        xx = inp[0].detach()
        out.register_hook(
            lambda go: cap.__setitem__(
                "G", go.detach().reshape(-1, out_f).t().float() @ xx.reshape(-1, in_f).float()
            )
        )

    h = layer.register_forward_hook(_fwd_hook)
    (layer(x) * gy).sum().backward()
    h.remove()

    b_in = layer.block_size_in
    g_perm = (
        cap["G"]
        .index_select(0, layer.perm_out_inv.long())
        .index_select(1, layer.perm_in_inv.long())
    )
    k_g = block_diag_skew(w_perm_frame(layer).transpose(-2, -1) @ g_perm, b_in)
    k_opt = vec_to_skew(layer.oft_R_in.grad.detach().float(), b_in)
    cos = (k_g.flatten() @ k_opt.flatten()) / (k_g.norm() * k_opt.norm())
    assert abs(cos.item()) > 0.99, cos.item()


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
