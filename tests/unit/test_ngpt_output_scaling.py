"""CPU tests for the post-build `sz` logit scaling wrapper."""

import torch
import torch.nn as nn

from src.model.ngpt.output_scaling import attach_sz_scaling


class _FakeOutput(nn.Module):
    """Stand-in for Megatron's ColumnParallelLinear output_layer.

    Returns (logits, bias) like the real one does.
    """

    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden))

    def forward(self, x):
        return torch.matmul(x, self.weight.T), None


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
