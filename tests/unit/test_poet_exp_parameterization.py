"""CPU unit tests for the POET `exp` (matrix-exponential) parameterization.

All tests here are CPU-runnable: `torch.linalg.matrix_exp` is pure PyTorch and
the decoupled "fast" chain is plain ops. GPU/compiled-forward parity is left to
the user's smoke run (the repo guards those with skipif(not cuda)).
"""

from __future__ import annotations

import math

import torch

torch.manual_seed(0)


def _triu(bs):
    r, c = torch.triu_indices(bs, bs, 1)
    return r.to(torch.int32), c.to(torch.int32)


def test_exp_builder_is_exactly_orthogonal():
    from poet_torch.poet_layer import get_weight_poet_decoupled_exp

    bs_in, bs_out = 8, 8
    r_in, r_out = 2, 2
    ne_in = bs_in * (bs_in - 1) // 2
    ne_out = bs_out * (bs_out - 1) // 2
    oft_in = torch.randn(r_in, ne_in) * 0.1
    oft_out = torch.randn(r_out, ne_out) * 0.1
    rows_in, cols_in = _triu(bs_in)
    rows_out, cols_out = _triu(bs_out)

    R_out, R_in = get_weight_poet_decoupled_exp(  # noqa: N806
        oft_in, oft_out, bs_in, bs_out, rows_in, cols_in, rows_out, cols_out
    )
    eye_in = torch.eye(bs_in)
    eye_out = torch.eye(bs_out)
    err_in = (R_in @ R_in.transpose(-2, -1) - eye_in).abs().max().item()
    err_out = (R_out @ R_out.transpose(-2, -1) - eye_out).abs().max().item()
    assert err_in < 1e-5, err_in
    assert err_out < 1e-5, err_out
    # det == +1 (proper rotation, in SO(b) not just O(b))
    assert torch.allclose(torch.linalg.det(R_in.float()), torch.ones(r_in), atol=1e-4)


def test_exp_is_tighter_than_cayley_neumann_at_large_angle():
    """At a non-tiny angle, exp stays exactly orthogonal while the degree-4
    Cayley/Neumann truncation drifts measurably.

    Uses ``cayley_batch`` (the pure-Python degree-4 helper, CPU-runnable) as the
    truncation stand-in — NOT the production ``poet::cayley`` Triton kernel,
    which has no CPU fallback. The point is only that a finite polynomial
    truncation drifts from orthogonality where exp does not.
    """
    from poet_torch.poet_layer import (
        cayley_batch,
        get_weight_poet_decoupled_exp,
        pytorch_skew_symmetric,
    )

    bs = 8
    ne = bs * (bs - 1) // 2
    oft = torch.randn(1, ne) * 0.6  # large-ish angles
    rows, cols = _triu(bs)

    R_out, _ = get_weight_poet_decoupled_exp(oft, oft, bs, bs, rows, cols, rows, cols)  # noqa: N806
    Q = pytorch_skew_symmetric(oft, bs, rows, cols)  # noqa: N806
    R_cayley = cayley_batch(Q)  # noqa: N806

    eye = torch.eye(bs)
    exp_err = (R_out @ R_out.transpose(-2, -1) - eye).abs().max().item()
    cay_err = (R_cayley @ R_cayley.transpose(-2, -1) - eye).abs().max().item()
    assert exp_err < 1e-5
    assert cay_err > exp_err * 10  # Cayley truncation is much less orthogonal here


def test_exp_2x2_block_rotates_by_exactly_theta_no_factor_of_two():
    """§3 of the math doc: the canonical angle of exp(Q) is exactly the singular
    value of Q (no factor-of-2, unlike Cayley's 2*arctan(theta))."""
    from poet_torch.poet_layer import get_weight_poet_decoupled_exp

    bs = 2
    theta = 0.7
    oft = torch.tensor([[theta]])  # single upper-tri entry => angle theta
    rows, cols = _triu(bs)
    R_out, _ = get_weight_poet_decoupled_exp(oft, oft, bs, bs, rows, cols, rows, cols)  # noqa: N806
    R = R_out[0]  # noqa: N806
    # rotation angle recovered from the orthogonal 2x2 block
    angle = math.acos(float(R[0, 0].clamp(-1.0, 1.0)))
    assert abs(angle - theta) < 1e-5, angle
    # explicitly NOT the Cayley factor 2*arctan(theta)
    assert abs(angle - 2.0 * math.atan(theta)) > 1e-3


def test_exp_builder_gradcheck():
    """Autograd flows correctly through skew-construction + matrix_exp (fp64)."""
    from poet_torch.poet_layer import get_weight_poet_decoupled_exp

    bs = 4
    ne = bs * (bs - 1) // 2
    rows, cols = _triu(bs)
    oft = (torch.randn(1, ne, dtype=torch.float64) * 0.1).requires_grad_(True)

    def f(o):
        R_out, _ = get_weight_poet_decoupled_exp(o, o, bs, bs, rows, cols, rows, cols)  # noqa: N806
        return R_out

    assert torch.autograd.gradcheck(f, (oft,), atol=1e-5, rtol=1e-3)
