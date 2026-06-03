# src/diag/orthogonalize.py
"""Newton-Schulz orthogonalization for diagnostics (the Muon update transform).

Pure tensor math, no Megatron/CUDA imports — CPU-testable. Used by the gradient
conditioning probes to log the POST-orthogonalization spectrum: NS drives all
singular values toward ~1, so this is what a Muon-style update applies regardless
of how ill-conditioned the raw gradient is.

This mirrors the canonical Muon ``zeropower_via_newtonschulz5`` (and the
``ns_coeffs``/``ns_steps`` defaults in ``configs/experiments/optim/muon_hybrid``):
fp32, Frobenius-normalize, quintic iteration, with the rows>cols transpose so the
inner products stay on the smaller dimension. It is a faithful *approximation* of
Muon's transform for diagnostics — not bit-identical to the realized update, which
orthogonalizes the momentum buffer (not the raw grad) and, under tensor
parallelism, all-gathers the sharded weight first (this helper sees only the
tensor it is handed, e.g. the local shard at TP>1).
"""

from __future__ import annotations

import torch

# Quintic Newton-Schulz coefficients (Jordan 2024; same as the muon_hybrid config
# and src/optim/_kimi_muon / poet_skew_muon).
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def newton_schulz_orthogonalize(G: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization of a 2D matrix.

    Args:
        G: a 2D (out, in) matrix (e.g. a weight gradient).
        ns_steps: number of NS iterations (Muon default 5).

    Returns: a matrix of the SAME shape as ``G`` whose singular values are driven
    toward a uniform band (~1), i.e. the orthogonalized ("whitened") update.
    """
    X = G.to(torch.float32)
    transposed = False
    if X.shape[-2] > X.shape[-1]:
        X = X.transpose(-2, -1)
        transposed = True
    X = X / (torch.linalg.matrix_norm(X, ord="fro", dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(ns_steps):
        A = X @ X.transpose(-2, -1)
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X
