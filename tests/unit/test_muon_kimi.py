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


def test_sharded_state_dict_hides_use_muon_from_megatron():
    """Regression: the end-of-training ``save_checkpoint`` crashed with
    ``AttributeError: 'bool' object has no attribute 'shape'`` because Megatron's
    ``sharded_state_dict`` runs every per-param optimizer-state value through a
    tensor-only helper, and the Kimi Muon stores a per-param ``use_muon`` bool.
    The crash also aborted the post-training validation, so the final-step eval
    was never logged. ``_StripUseMuonShardingMixin`` must hide ``use_muon`` from
    the parent serializer and restore it afterwards (tensor state untouched)."""
    from src.optim.muon_kimi import _StripUseMuonShardingMixin

    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8, bias=True)
    opt = Muon(lr=1e-2, wd=0.0, muon_params=[lin.weight], adamw_params=[lin.bias])
    lin(torch.randn(4, 8)).sum().backward()
    opt.step()  # populate state: use_muon + momentum_buffer / moment1 / moment2 / step

    seen = {}

    class _FakeMegatronOptimizer:
        """Stand-in for Float16OptimizerWithFloat16Params: records what per-param
        optimizer state the (tensor-only) serializer would see."""

        def __init__(self, optimizer):
            self.optimizer = optimizer

        def sharded_state_dict(self, *args, **kwargs):
            seen["use_muon_visible"] = any("use_muon" in st for st in self.optimizer.state.values())
            seen["tensor_state_present"] = any(
                torch.is_tensor(v) for st in self.optimizer.state.values() for v in st.values()
            )
            return {"ok": True}

    # Negative control: without the mixin the parent sees use_muon (the crash trigger).
    _FakeMegatronOptimizer(opt).sharded_state_dict()
    assert seen["use_muon_visible"] is True

    wrapped_cls = type("W", (_StripUseMuonShardingMixin, _FakeMegatronOptimizer), {})
    out = wrapped_cls(opt).sharded_state_dict("model_sd", is_loading=False)

    # With the mixin: parent never sees the bool, but real tensor state remains.
    assert out == {"ok": True}
    assert seen["use_muon_visible"] is False
    assert seen["tensor_state_present"] is True

    # use_muon is restored on the live optimizer after serialization.
    assert opt.state[lin.weight]["use_muon"] is True
    assert opt.state[lin.bias]["use_muon"] is False
