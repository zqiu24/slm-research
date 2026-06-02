"""CPU tests for the vendored Kimi Muon optimizer (src/optim/_kimi_muon.py)."""

import os

# The vendored Newton-Schulz fn is @torch.compile'd; disable dynamo so the CPU
# test runs eagerly and deterministically. Must be set before importing torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from src.optim._kimi_muon import Muon, zeropower_via_newtonschulz5


def test_muon_routes_2d_to_muon_and_rest_to_adamw():
    lin = torch.nn.Linear(8, 8, bias=True)
    opt = Muon(lr=1e-2, wd=0.0, muon_params=[lin.weight], adamw_params=[lin.bias])
    assert opt.state[lin.weight]["use_muon"] is True
    assert opt.state[lin.bias]["use_muon"] is False


def test_muon_step_updates_both_param_kinds():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8, bias=True)
    opt = Muon(lr=1e-2, wd=0.0, muon_params=[lin.weight], adamw_params=[lin.bias])
    before_w = lin.weight.detach().clone()
    before_b = lin.bias.detach().clone()
    lin(torch.randn(4, 8)).sum().backward()
    opt.step()
    assert not torch.equal(lin.weight, before_w)
    assert not torch.equal(lin.bias, before_b)


def test_newtonschulz_returns_same_shape():
    g = torch.randn(6, 10)
    out = zeropower_via_newtonschulz5(g, steps=5)
    assert out.shape == g.shape
