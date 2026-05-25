"""CPU tests for the per-head Q/K hypersphere normalization injected by nGPT.

The full SelfAttention is hard to spin up off-cluster; we test the
QKHyperNorm leaf module directly. That module is what nGPT plugs into
Megatron's q_layernorm / k_layernorm submodule slots.
"""

import torch

from src.model.ngpt.attention import QKHyperNorm


def test_qk_hyper_norm_output_is_sqk_times_unit():
    # Megatron passes per-head tensors of shape (s, b, h_per_tp, d_head)
    s, b, h, d = 4, 2, 3, 8
    qkn = QKHyperNorm(num_heads_per_tp=h, head_dim=d, sqk_init_value=1.0, base_scale=1.0 / 8.0)
    x = torch.randn(s, b, h, d)
    y = qkn(x)
    # y / sqk_per_head should be unit-norm along d
    sqk_eff = qkn.sqk.scaled_value().view(1, 1, h, d)
    unit = y / sqk_eff
    assert torch.allclose(unit.norm(dim=-1), torch.ones(s, b, h), atol=1e-5)


def test_qk_hyper_norm_at_init_just_normalizes():
    # init_value=1.0 with uniform sqk => sqk_eff == 1, so y == justnorm(x)
    s, b, h, d = 2, 1, 2, 4
    qkn = QKHyperNorm(num_heads_per_tp=h, head_dim=d, sqk_init_value=1.0, base_scale=1.0 / 8.0)
    x = torch.randn(s, b, h, d)
    y = qkn(x)
    expected = x / x.norm(p=2, dim=-1, keepdim=True)
    assert torch.allclose(y, expected, atol=1e-5)


def test_qk_hyper_norm_param_count_is_head_dim_times_heads():
    qkn = QKHyperNorm(num_heads_per_tp=4, head_dim=16, sqk_init_value=1.0, base_scale=1.0 / 16.0)
    n = sum(p.numel() for p in qkn.parameters())
    # one sqk vector of length n_heads * head_dim
    assert n == 4 * 16
