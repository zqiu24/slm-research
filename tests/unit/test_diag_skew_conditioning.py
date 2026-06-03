# tests/unit/test_diag_skew_conditioning.py
import math

import torch

from src.diag.skew_conditioning import block_spectral_stats, vec_to_skew


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


def test_block_spectral_stats_effective_rank():
    # entropy effective rank (Roy-Vetterli): erank = exp(-sum p_i log p_i),
    # p_i = sigma_i / sum(sigma). Uniform spectrum -> erank == #singular values.
    a = 2.0
    q_uniform = torch.zeros(1, 4, 4)
    q_uniform[0, 0, 1], q_uniform[0, 1, 0] = a, -a
    q_uniform[0, 2, 3], q_uniform[0, 3, 2] = a, -a  # sigmas {a,a,a,a}
    er_uniform = block_spectral_stats(q_uniform)["effective_rank"][0].item()
    assert math.isclose(er_uniform, 4.0, rel_tol=1e-5)

    # Heavy-tailed {3,3,1,1}: 1 < erank < 4, and matches exp(entropy) exactly.
    a, b = 3.0, 1.0
    q = torch.zeros(1, 4, 4)
    q[0, 0, 1], q[0, 1, 0] = a, -a
    q[0, 2, 3], q[0, 3, 2] = b, -b
    er = block_spectral_stats(q)["effective_rank"][0].item()
    sig = torch.tensor([a, a, b, b])
    p = sig / sig.sum()
    expected = math.exp(-(p * p.log()).sum().item())
    assert math.isclose(er, expected, rel_tol=1e-5)
    assert 1.0 < er < 4.0


def test_vec_to_skew_is_skew_symmetric_and_matches_layout():
    b = 4
    # b(b-1)/2 = 6 upper-tri entries, two blocks stacked
    vec = torch.arange(1.0, 13.0).reshape(2, 6)
    q = vec_to_skew(vec, b)

    assert q.shape == (2, b, b)
    # skew-symmetry: Q == -Q^T
    assert torch.allclose(q, -q.transpose(-1, -2))
    # diagonal is zero
    assert torch.allclose(torch.diagonal(q, dim1=-2, dim2=-1), torch.zeros(2, b))
    # first upper-tri entry (row 0, col 1) of block 0 is vec[0,0]
    assert q[0, 0, 1].item() == 1.0
    assert q[0, 1, 0].item() == -1.0


def test_skew_to_vec_is_inverse_of_vec_to_skew():
    from src.diag.skew_conditioning import skew_to_vec, vec_to_skew

    b = 6
    vec = torch.arange(1.0, 1.0 + 2 * (b * (b - 1) // 2)).reshape(2, b * (b - 1) // 2)
    round_trip = skew_to_vec(vec_to_skew(vec, b), b)
    assert torch.allclose(round_trip, vec)


def test_block_size_from_nelems():
    from src.diag.skew_conditioning import block_size_from_nelems

    for b in (2, 4, 8, 256, 512):
        assert block_size_from_nelems(b * (b - 1) // 2) == b
