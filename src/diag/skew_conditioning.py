# src/diag/skew_conditioning.py
"""Pure-math diagnostics for POET's per-block ∂f/∂Q conditioning (Probe 0B).

No Megatron / CUDA / poet_torch imports — every function here takes plain
tensors so the math is unit-testable on CPU.
"""

from __future__ import annotations

import math

import torch

_TRIU_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _triu_idx(b: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(b), str(device))
    cached = _TRIU_CACHE.get(key)
    if cached is None:
        idx = torch.triu_indices(b, b, 1, device=device)
        cached = (idx[0].contiguous(), idx[1].contiguous())
        _TRIU_CACHE[key] = cached
    return cached


def block_spectral_stats(skew: torch.Tensor, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    """Summarize the singular-value spectrum of a batch of (skew-symmetric) blocks.

    Args:
        skew: tensor of shape (num_blocks, b, b). Skew-symmetric inputs have
            *paired* singular values; the stats below are well-defined on the
            full (paired) spectrum and pairing is not removed.
        eps: floor for sigma_min to avoid div-by-zero on rank-deficient blocks.

    Returns dict of shape-(num_blocks,) tensors:
        condition_number   = sigma_max / max(sigma_min, eps)
        stable_rank        = ||.||_F^2 / sigma_max^2
        sigma_max_over_median = sigma_max / median(sigma)
        effective_rank     = exp(-sum p_i log p_i), p_i = sigma_i / sum(sigma)
                             (Roy-Vetterli entropy effective rank)
    """
    if skew.dim() == 2:
        skew = skew.unsqueeze(0)
    sv = torch.linalg.svdvals(skew.to(torch.float32))  # (num_blocks, b), descending
    sigma_max = sv[:, 0]
    sigma_min = sv[:, -1].clamp_min(eps)
    fro_sq = (sv * sv).sum(dim=1)
    median = torch.quantile(sv, 0.5, dim=1)
    # entropy effective rank: normalize singular values to a distribution and
    # exponentiate the Shannon entropy. p*log(p+eps) sends the p=0 terms to 0.
    p = sv / sv.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(p * (p + eps).log()).sum(dim=1)
    return {
        "condition_number": sigma_max / sigma_min,
        "stable_rank": fro_sq / (sigma_max * sigma_max),
        "sigma_max_over_median": sigma_max / median.clamp_min(eps),
        "effective_rank": entropy.exp(),
    }


def vec_to_skew(vec: torch.Tensor, block_size: int) -> torch.Tensor:
    """Map upper-triangular vectors to full skew-symmetric blocks.

    Args:
        vec: shape (num_blocks, b*(b-1)/2), the trainable/grad entries in the
            same order as ``torch.triu_indices(b, b, 1)``.
        block_size: b.

    Returns: (num_blocks, b, b) with Q[..., r, c] = vec, Q[..., c, r] = -vec.
    """
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    b = block_size
    n = vec.shape[0]
    rows, cols = _triu_idx(b, vec.device)
    q = torch.zeros(n, b, b, dtype=vec.dtype, device=vec.device)
    q[:, rows, cols] = vec
    q[:, cols, rows] = -vec
    return q


def block_size_from_nelems(n_elems: int) -> int:
    """Recover block size b from the strictly-upper-triangular count
    n_elems = b*(b-1)/2  =>  b = (1 + sqrt(1 + 8*n_elems)) / 2."""
    return (1 + math.isqrt(1 + 8 * int(n_elems))) // 2


def skew_to_vec(skew: torch.Tensor, block_size: int) -> torch.Tensor:
    """Inverse of ``vec_to_skew``: extract the strictly-upper-triangular entries
    (same ``triu_indices(b,b,1)`` order POET stores).

    Args:
        skew: (num_blocks, b, b) (or (b, b)).
        block_size: b.
    Returns: (num_blocks, b*(b-1)/2).
    """
    if skew.dim() == 2:
        skew = skew.unsqueeze(0)
    b = block_size
    rows, cols = _triu_idx(b, skew.device)
    return skew[:, rows, cols]
