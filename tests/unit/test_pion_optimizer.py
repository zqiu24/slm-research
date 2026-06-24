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
