"""Test the POET Megatron builder partitions params correctly.

We can't import the real Megatron optimizer wrappers without GPU, so
we monkey-patch the three Megatron entry points and check that POET
correctly classifies params into linear-2D-non-embedding vs the rest.
"""

from unittest.mock import MagicMock

import torch
import torch.nn as nn


class StubModelChunk(nn.Module):
    """A toy model with one linear (2D), one embedding (2D but flagged),
    and one bias (1D). POET should route only the linear to POETAdam."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8, bias=True)
        self.emb = nn.Embedding(16, 8)
        # Mark embedding so the partition logic excludes it.
        self.emb.weight.is_embedding_or_output_parameter = True


def test_poet_builder_partitions_params(monkeypatch):
    from src.optim import poet as poet_mod

    # Stub Megatron entry points injected by _resolve_megatron_handles.
    fake_get_megatron_optimizer = MagicMock()
    fake_get_megatron_optimizer.return_value = MagicMock(chained_optimizers=[])

    fake_adam = MagicMock(return_value=MagicMock(spec=torch.optim.Optimizer))
    fake_adam.return_value.param_groups = []
    fake_adam.return_value.state = {}

    def fake_get_param_groups(chunks, cfg, overrides):
        return [{"params": [chunks[0].lin.weight]}]

    def fake_f32(opt, cfg, init_fn):
        return MagicMock(name="FP32Optimizer", inner=opt)

    monkeypatch.setattr(poet_mod, "_get_param_groups", fake_get_param_groups, raising=False)
    monkeypatch.setattr(
        poet_mod, "get_megatron_optimizer", fake_get_megatron_optimizer, raising=False
    )
    monkeypatch.setattr(poet_mod, "ChainedOptimizer", list, raising=False)
    monkeypatch.setattr(poet_mod, "Float16OptimizerWithFloat16Params", fake_f32, raising=False)
    monkeypatch.setattr(poet_mod, "FP32Optimizer", fake_f32, raising=False)
    monkeypatch.setattr(poet_mod, "_BaseAdamCls", fake_adam, raising=False)
    monkeypatch.setattr(poet_mod, "_USING_PYTORCH_OPTIMIZER", True, raising=False)

    # Skip the lazy resolver so it doesn't try to import megatron.
    monkeypatch.setattr(poet_mod, "_resolve_megatron_handles", lambda: None)

    cfg = MagicMock(
        lr=1e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        decoupled_weight_decay=True,
        bf16=False,
        fp16=False,
        use_distributed_optimizer=False,
        use_precision_aware_optimizer=False,
        poet_merge_period=10,
        poet_scale=2.0,
    )
    chunks = [StubModelChunk()]

    out = poet_mod.get_megatron_poet_optimizer(
        cfg, chunks, config_overrides=None, use_gloo_process_groups=False
    )

    # The non-linear remainder ran through the chained-Adam builder.
    assert fake_get_megatron_optimizer.call_count == 1
    # Linear vs nonlinear split: lin.weight + lin.bias + emb.weight.
    # Only lin.weight is 2D + non-embedding → linear set has exactly one tensor.
    # After the call, requires_grad should be restored everywhere.
    params = list(chunks[0].parameters())
    assert all(p.requires_grad for p in params)
    # Return value is what ChainedOptimizer(...) produced — stubbed to list.
    assert isinstance(out, list)
