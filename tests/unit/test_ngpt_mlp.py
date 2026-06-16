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


def test_ngpt_mlp_body_unfused_matches_fused():
    """Splitting linear_fc1 into u/v (and slicing suv) is bit-identical to the
    packed forward, given the same weights. This is the fairness invariant:
    fused nGPT == unfused nGPT."""
    torch.manual_seed(1)
    n_embd = 16
    n_inner = 4 * n_embd
    base_scale = 1.0 / (n_embd**0.5)

    fused = NGPTMLPBody(
        hidden_size=n_embd,
        ffn_hidden_size=n_inner,
        base_scale=base_scale,
        suv_init_value=1.0,
        suv_init_scaling=1.0,
        dtype=torch.float32,
        unfuse=False,
    )
    unfused = NGPTMLPBody(
        hidden_size=n_embd,
        ffn_hidden_size=n_inner,
        base_scale=base_scale,
        suv_init_value=1.0,
        suv_init_scaling=1.0,
        dtype=torch.float32,
        unfuse=True,
    )
    # Copy fused weights into the split projections + shared suv/fc2.
    unfused.linear_fc1_u.weight.data.copy_(fused.linear_fc1.weight.data[:n_inner])
    unfused.linear_fc1_v.weight.data.copy_(fused.linear_fc1.weight.data[n_inner:])
    unfused.linear_fc2.weight.data.copy_(fused.linear_fc2.weight.data)
    unfused.suv.param.data.copy_(fused.suv.param.data)

    x = torch.randn(2, 5, n_embd)
    assert torch.allclose(fused(x), unfused(x), atol=1e-6)


def test_ngpt_mlp_body_unfused_param_count_matches_fused():
    n_embd, n_inner = 16, 64
    kw = dict(
        hidden_size=n_embd,
        ffn_hidden_size=n_inner,
        base_scale=1.0 / (n_embd**0.5),
        suv_init_value=1.0,
        suv_init_scaling=1.0,
        dtype=torch.float32,
    )
    fused = NGPTMLPBody(unfuse=False, **kw)
    unfused = NGPTMLPBody(unfuse=True, **kw)
    assert sum(p.numel() for p in fused.parameters()) == sum(
        p.numel() for p in unfused.parameters()
    )
    # Split names exist; packed name does not.
    assert hasattr(unfused, "linear_fc1_u") and hasattr(unfused, "linear_fc1_v")
    assert not hasattr(unfused, "linear_fc1")


def test_ngpt_mlp_reads_unfuse_flag_from_config():
    """NGPTMLP (the Megatron-instantiable subclass) selects split projections
    when config.unfuse_fc1 is set."""
    from src.model.ngpt.mlp import NGPTMLP

    class _Cfg:
        hidden_size = 16
        ffn_hidden_size = 64
        ngpt_base_scale = 1.0 / (16**0.5)
        ngpt_suv_init = 1.0
        bf16 = False
        params_dtype = torch.float32

    fused = NGPTMLP(_Cfg())
    assert hasattr(fused, "linear_fc1") and not hasattr(fused, "linear_fc1_u")

    cfg_unfused = _Cfg()
    cfg_unfused.unfuse_fc1 = True
    unfused = NGPTMLP(cfg_unfused)
    assert hasattr(unfused, "linear_fc1_u") and hasattr(unfused, "linear_fc1_v")
    assert not hasattr(unfused, "linear_fc1")
