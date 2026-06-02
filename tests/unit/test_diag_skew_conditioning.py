# tests/unit/test_diag_skew_conditioning.py
import math

import torch

from src.diag.skew_conditioning import block_spectral_stats


def test_block_spectral_stats_on_known_skew():
    # A 4x4 skew matrix built from two 2x2 rotation generators with angles a,b.
    # Its singular values are {a, a, b, b} (paired, as skew matrices always are).
    a, b = 3.0, 1.0
    q = torch.zeros(1, 4, 4)
    q[0, 0, 1], q[0, 1, 0] = a, -a
    q[0, 2, 3], q[0, 3, 2] = b, -b

    stats = block_spectral_stats(q)

    # one block in -> one row of stats
    assert stats["condition_number"].shape == (1,)
    # sigma_max/sigma_min = a/b
    assert math.isclose(stats["condition_number"][0].item(), a / b, rel_tol=1e-5)
    # stable rank = ||.||_F^2 / sigma_max^2 = (2a^2 + 2b^2) / a^2
    expected_sr = (2 * a**2 + 2 * b**2) / a**2
    assert math.isclose(stats["stable_rank"][0].item(), expected_sr, rel_tol=1e-5)
    # sigma_max / median(sigmas): median of [a,a,b,b] sorted = (a+b)/2
    assert math.isclose(stats["sigma_max_over_median"][0].item(), a / ((a + b) / 2), rel_tol=1e-5)
