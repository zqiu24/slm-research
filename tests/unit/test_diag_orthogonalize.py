# tests/unit/test_diag_orthogonalize.py
import torch

from src.diag.orthogonalize import newton_schulz_orthogonalize
from src.diag.skew_conditioning import block_spectral_stats


def _heavy_tailed(out_dim, in_dim, sing):
    """A (out_dim x in_dim) matrix with exactly the given singular values."""
    k = len(sing)
    u = torch.linalg.qr(torch.randn(out_dim, k))[0]  # (out, k), orthonormal cols
    v = torch.linalg.qr(torch.randn(in_dim, k))[0]  # (in, k), orthonormal cols
    return u @ torch.diag(torch.tensor(sing)) @ v.T


def test_newton_schulz_flattens_spectrum_square():
    """NS democratizes the spectrum: a heavy-tailed square matrix -> condition ~1.
    Mirrors the skew-NS test's thresholds (the math is the same quintic)."""
    torch.manual_seed(0)
    G = _heavy_tailed(8, 8, [20.0, 18.0, 16.0, 2.0, 1.8, 1.6, 1.4, 1.2])
    s_in = block_spectral_stats(G)
    X = newton_schulz_orthogonalize(G, ns_steps=5)
    s_out = block_spectral_stats(X)
    cond_in = s_in["condition_number"][0].item()
    cond_out = s_out["condition_number"][0].item()
    assert cond_in > 10.0
    assert cond_out < 5.0
    assert cond_out < cond_in / 5.0
    # effective rank rises toward the ambient dim as the spectrum flattens
    assert s_out["effective_rank"][0].item() > s_in["effective_rank"][0].item()


def test_newton_schulz_handles_tall_rectangular_and_preserves_shape():
    """rows > cols must go through the transpose path and still flatten,
    returning a matrix of the ORIGINAL shape."""
    torch.manual_seed(1)
    G = _heavy_tailed(32, 8, [20.0, 18.0, 16.0, 2.0, 1.8, 1.6, 1.4, 1.2])
    X = newton_schulz_orthogonalize(G, ns_steps=5)
    assert X.shape == (32, 8)
    s_out = block_spectral_stats(X)
    assert s_out["condition_number"][0].item() < 5.0  # flattened despite rows>cols


def test_newton_schulz_handles_wide_rectangular():
    """cols > rows path (no transpose) flattens too."""
    torch.manual_seed(2)
    G = _heavy_tailed(8, 32, [20.0, 18.0, 16.0, 2.0, 1.8, 1.6, 1.4, 1.2])
    X = newton_schulz_orthogonalize(G, ns_steps=5)
    assert X.shape == (8, 32)
    assert block_spectral_stats(X)["condition_number"][0].item() < 5.0
