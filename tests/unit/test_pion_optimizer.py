"""CPU unit tests for the vendored Pion optimizer (src/optim/_pion.py)."""

from __future__ import annotations

import torch

from src.optim._pion import PionOptimizer


def _square_param(seed: int = 0, n: int = 16) -> torch.nn.Parameter:
    gen = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(n, n, generator=gen))


def test_pion_step_lie_lie_preserves_spectrum_for_small_step():
    """Pion's orthogonal-equivalence update preserves singular values; with a
    tiny lr the truncated-exp approximation keeps them within 5%."""
    w = _square_param(seed=1)
    sv_before = torch.linalg.svdvals(w.detach().clone())
    opt = PionOptimizer(
        [w],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        degree=2,
        pion_scaling="rms",
        pion_rms=0.2,
        pion_momentum="lie_lie",
        pion_update_side="both",
    )
    gen = torch.Generator().manual_seed(2)
    w.grad = torch.randn(16, 16, generator=gen)
    opt.step()
    assert torch.isfinite(w.detach()).all()
    sv_after = torch.linalg.svdvals(w.detach())
    rel = ((sv_after - sv_before).abs() / (sv_before.abs() + 1e-6)).max()
    assert rel < 0.05, f"singular values drifted by {rel:.4f} (>5%)"


def test_pion_step_changes_weight_and_is_deterministic():
    """Same seed + same grad → identical update (no Date.now/rng leakage)."""
    results = []
    for _ in range(2):
        w = _square_param(seed=3)
        before = w.detach().clone()
        opt = PionOptimizer(
            [w],
            lr=1e-2,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            degree=2,
            pion_scaling="rms",
            pion_rms=0.2,
            pion_momentum="transported_ambient_ambient",
            pion_update_side="alternate",
        )
        gen = torch.Generator().manual_seed(4)
        w.grad = torch.randn(16, 16, generator=gen)
        opt.step()
        assert not torch.allclose(w.detach(), before)
        results.append(w.detach().clone())
    assert torch.allclose(results[0], results[1])


def test_pion_skips_non_2d_params():
    """1-D params in a Pion group are left untouched (Pion is matrix-only)."""
    bias = torch.nn.Parameter(torch.randn(16))
    before = bias.detach().clone()
    opt = PionOptimizer(
        [bias],
        lr=1e-2,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        pion_momentum="lie_lie",
        pion_update_side="both",
    )
    bias.grad = torch.randn(16)
    opt.step()
    assert torch.allclose(bias.detach(), before)


def test_sharded_state_dict_strips_block_shaped_pion_state():
    """Regression: the end-of-training ``save_checkpoint`` crashed with
    ``Optimizer shape ((64, 512) does not match model shape ((1536, 512))``
    because Pion keeps per-BLOCK momentum buffers (e.g. a ``(kv_channels, hidden)``
    buffer per QKV head) in ``optimizer.state``, and Megatron's
    ``sharded_state_dict`` asserts every per-param state tensor equals the param
    shape (only ``step`` is excluded). The crash also aborted the post-training
    validation, so the final-step eval was never logged.
    ``_StripPionStateShardingMixin`` must hide the momentum tensors from the parent
    serializer and restore them afterwards, while leaving ``step`` in place."""
    from src.optim.pion import _StripPionStateShardingMixin

    torch.manual_seed(0)
    # QKV weight: num_heads = num_query_groups = 2, kv_channels = 4, hidden = 8 ->
    # weight (24, 8); the per-head split keeps (4, 8) momentum blocks per head.
    qkv = torch.nn.Parameter(torch.randn(24, 8))
    qkv.is_qkv = True
    opt = PionOptimizer(
        [qkv],
        lr=1e-2,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        degree=2,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=(4, 4, 4),
        split_qkv_per_head=True,
        qkv_split_granularity="head",
        pion_scaling="rms",
        pion_rms=0.2,
        pion_momentum="transported_ambient_ambient",
        pion_update_side="alternate",
    )
    qkv.grad = torch.randn(24, 8)
    opt.step()  # populate per-block momentum buffers + step

    # Bug precondition: at least one per-param state tensor mismatches the param shape.
    state = opt.state[qkv]
    mismatched = [
        k for k, v in state.items() if torch.is_tensor(v) and tuple(v.shape) != tuple(qkv.shape)
    ]
    assert mismatched, "expected block-shaped momentum buffers in Pion state"

    seen = {}

    class _FakeMegatronOptimizer:
        """Stand-in for Float16OptimizerWithFloat16Params: records what per-param
        optimizer state the (tensor-only) serializer would see."""

        def __init__(self, optimizer):
            self.optimizer = optimizer

        def sharded_state_dict(self, *args, **kwargs):
            seen["tensor_state_visible"] = any(
                torch.is_tensor(v) for st in self.optimizer.state.values() for v in st.values()
            )
            seen["step_visible"] = all("step" in st for st in self.optimizer.state.values())
            return {"ok": True}

    # Negative control: without the mixin the parent sees the tensors (crash trigger).
    _FakeMegatronOptimizer(opt).sharded_state_dict()
    assert seen["tensor_state_visible"] is True

    wrapped_cls = type("W", (_StripPionStateShardingMixin, _FakeMegatronOptimizer), {})
    out = wrapped_cls(opt).sharded_state_dict("model_sd", is_loading=False)

    # With the mixin: no tensor momentum reaches the parent, but step stays.
    assert out == {"ok": True}
    assert seen["tensor_state_visible"] is False
    assert seen["step_visible"] is True

    # Momentum buffers restored on the live optimizer after serialization.
    assert any(torch.is_tensor(v) for v in opt.state[qkv].values())
    assert "step" in opt.state[qkv]
