"""Tests for sandwich-norm post-norm ops (CPU)."""

import torch
import torch.nn as nn

from src.model.sandwich_norm_ops import apply_post_norm_scale, make_post_norm_hook


def test_post_norm_hook_norms_primary_output_preserves_rest():
    norm = nn.Linear(4, 4, bias=False)  # stand-in "norm"
    hook = make_post_norm_hook(norm)
    x = torch.ones(2, 4)
    new = hook(None, None, (x, None))
    assert torch.allclose(new[0], norm(x))
    assert new[1] is None


def test_post_norm_hook_handles_bare_tensor_output():
    norm = nn.Linear(4, 4, bias=False)
    hook = make_post_norm_hook(norm)
    x = torch.ones(2, 4)
    new = hook(None, None, x)
    assert torch.allclose(new, norm(x))


def test_apply_post_norm_scale_multiplies_weight():
    m = nn.LayerNorm(4)  # weight initialised to ones
    apply_post_norm_scale(m, 0.03)
    assert torch.allclose(m.weight, torch.full((4,), 0.03))


def test_apply_post_norm_scale_noop_at_one():
    m = nn.LayerNorm(4)
    apply_post_norm_scale(m, 1.0)
    assert torch.allclose(m.weight, torch.ones(4))
