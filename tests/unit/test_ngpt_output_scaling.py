"""CPU tests for the post-build `sz` logit scaling wrapper."""

import torch
import torch.nn as nn

from src.model.ngpt.output_scaling import attach_sz_scaling


class _FakeOutput(nn.Module):
    """Stand-in for Megatron's ColumnParallelLinear output_layer.

    Accepts an optional ``weight=`` (like the real one) and returns
    ``(logits, bias)``. When ``weight`` is given it is used in place of
    ``self.weight`` — this is the seam the sz fold relies on.
    """

    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden))

    def forward(self, x, weight=None, **kwargs):
        w = weight if weight is not None else self.weight
        return torch.matmul(x, w.T), None


def test_attach_sz_scaling_creates_parameter_with_correct_shape():
    out = _FakeOutput(vocab=17, hidden=4)
    holder = nn.Module()
    holder.output_layer = out
    attach_sz_scaling(holder, vocab_size=17, base_scale=0.5)
    assert hasattr(holder, "_ngpt_sz")
    assert holder._ngpt_sz.param.shape == (17,)
    assert torch.allclose(holder._ngpt_sz.param.data, 0.5 * torch.ones(17))


def test_attach_sz_scaling_multiplies_logits():
    out = _FakeOutput(vocab=5, hidden=3)
    holder = nn.Module()
    holder.output_layer = out
    attach_sz_scaling(holder, vocab_size=5, base_scale=1.0)
    # at init, sz_effective = 1.0 everywhere, so logits unchanged
    x = torch.randn(2, 7, 3)
    logits_init, _ = holder.output_layer(x)
    expected_unscaled = torch.matmul(x, out.weight.T)
    assert torch.allclose(logits_init, expected_unscaled, atol=1e-5)

    # bump sz, re-eval
    holder._ngpt_sz.param.data.fill_(3.0)
    logits_scaled, _ = holder.output_layer(x)
    assert torch.allclose(logits_scaled, 3.0 * expected_unscaled, atol=1e-5)


def test_attach_sz_scaling_is_idempotent():
    out = _FakeOutput(vocab=3, hidden=2)
    holder = nn.Module()
    holder.output_layer = out
    attach_sz_scaling(holder, vocab_size=3, base_scale=1.0)
    first = holder._ngpt_sz
    attach_sz_scaling(holder, vocab_size=3, base_scale=1.0)
    assert holder._ngpt_sz is first


def test_sz_fold_matches_post_multiply_forward_and_grad():
    """Folding sz into the weight is identical — in both the logit values and
    the sz gradient — to the reference post-multiply `sz * (W @ x)`, for a
    non-uniform sz. This is the correctness guarantee behind the memory fix."""
    torch.manual_seed(0)
    vocab, hidden = 11, 4
    base_scale = 0.5
    x = torch.randn(2, 3, hidden)

    out = _FakeOutput(vocab, hidden)
    holder = nn.Module()
    holder.output_layer = out
    attach_sz_scaling(holder, vocab_size=vocab, base_scale=base_scale)
    holder._ngpt_sz.param.data.copy_(torch.randn(vocab))  # non-uniform sz

    # --- fold path (implementation under test) ---
    logits_fold, _ = holder.output_layer(x)
    holder._ngpt_sz.param.grad = None
    logits_fold.square().sum().backward()
    sz_grad_fold = holder._ngpt_sz.param.grad.clone()

    # --- reference: explicit post-multiply with the same weight + sz param ---
    sz_param = holder._ngpt_sz.param.detach().clone().requires_grad_(True)
    sz_eff = sz_param * (1.0 / base_scale)  # init_value / init_scaling
    logits_ref = sz_eff * torch.matmul(x, out.weight.detach().T)
    logits_ref.square().sum().backward()

    assert torch.allclose(logits_fold, logits_ref, atol=1e-5)
    assert torch.allclose(sz_grad_fold, sz_param.grad, atol=1e-5)
