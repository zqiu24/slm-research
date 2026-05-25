"""CPU test for NGPTMLP forward matches the reference c_fc/suv/silu path."""

import torch
import torch.nn as nn

from src.model.ngpt.mlp import NGPTMLPBody


def test_ngpt_mlp_body_matches_reference_uv_silu():
    """NGPTMLPBody encapsulates the c_fc -> suv -> chunk -> silu -> mlp_c_proj path.

    We compare it to a hand-written reference that mirrors the lines from
    /lustre/fast/fast/zqiu/tmp/ngpt/model.py::Block.forward, MLP section.
    """
    torch.manual_seed(0)
    n_embd = 16
    n_inner = 4 * n_embd  # nGPT convention
    base_scale = 1.0 / (n_embd**0.5)
    suv_init_value, suv_init_scaling = 1.0, 1.0  # reference defaults

    body = NGPTMLPBody(
        hidden_size=n_embd,
        ffn_hidden_size=n_inner,
        base_scale=base_scale,
        suv_init_value=suv_init_value,
        suv_init_scaling=suv_init_scaling,
        dtype=torch.float32,
    )

    # Reference c_fc has output dim 2*4*n_embd = 2*n_inner
    # and stores [u | v] concatenation along the last dim.
    ref_c_fc = nn.Linear(n_embd, 2 * n_inner, bias=False, dtype=torch.float32)
    ref_proj = nn.Linear(n_inner, n_embd, bias=False, dtype=torch.float32)
    # tie weights so behaviour is comparable
    ref_c_fc.weight.data.copy_(body.linear_fc1.weight.data)
    ref_proj.weight.data.copy_(body.linear_fc2.weight.data)
    # suv starts at suv_init_scaling everywhere -> scaled_value() == 1
    suv = body.suv.scaled_value() * (n_embd**0.5)

    x = torch.randn(2, 5, n_embd)
    uv = ref_c_fc(x)
    uv = suv * uv
    u, v = uv.chunk(2, dim=-1)
    ref_out = ref_proj(u * torch.nn.functional.silu(v))

    out = body(x)
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_ngpt_mlp_body_param_count():
    n_embd = 16
    n_inner = 64
    body = NGPTMLPBody(
        hidden_size=n_embd,
        ffn_hidden_size=n_inner,
        base_scale=1.0 / (n_embd**0.5),
        suv_init_value=1.0,
        suv_init_scaling=1.0,
        dtype=torch.float32,
    )
    # linear_fc1: 2 * n_inner * n_embd
    # linear_fc2: n_embd * n_inner
    # suv:        2 * n_inner
    expected = (2 * n_inner * n_embd) + (n_embd * n_inner) + (2 * n_inner)
    assert sum(p.numel() for p in body.parameters()) == expected
