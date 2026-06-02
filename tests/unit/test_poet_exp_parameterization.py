"""CPU unit tests for the POET `exp` (matrix-exponential) parameterization.

All tests here are CPU-runnable: `torch.linalg.matrix_exp` is pure PyTorch and
the decoupled "fast" chain is plain ops. GPU/compiled-forward parity is left to
the user's smoke run (the repo guards those with skipif(not cuda)).
"""

from __future__ import annotations

import math

import pytest
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


def test_poetlinear_defaults_to_cayley():
    from poet_torch import POETLinear

    pl = POETLinear(in_features=16, out_features=16, bsz=8, device="cpu", dtype=torch.float32)
    assert pl.parameterization == "cayley"


def test_poetlinear_rejects_unknown_parameterization():
    from poet_torch import POETLinear

    with pytest.raises(ValueError):
        POETLinear(
            in_features=16,
            out_features=16,
            bsz=8,
            parameterization="bogus",
            device="cpu",
            dtype=torch.float32,
        )


def test_build_R_exp_is_orthogonal():  # noqa: N802
    from poet_torch import POETLinear

    pl = POETLinear(
        in_features=16,
        out_features=16,
        bsz=8,
        parameterization="exp",
        device="cpu",
        dtype=torch.float32,
    )
    # seed non-zero rotation so R != I
    with torch.no_grad():
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
    R_out, R_in = pl._build_R(pl.oft_R_in, pl.oft_R_out)  # noqa: N806
    eye = torch.eye(8)
    assert (R_in @ R_in.transpose(-2, -1) - eye).abs().max().item() < 1e-5
    assert (R_out @ R_out.transpose(-2, -1) - eye).abs().max().item() < 1e-5


def _poet_forward_reference_exp(pl, x):
    """Pure-PyTorch oracle for the exp forward, independent of the layer code.

    Mirrors chain_layer_x_fast_decoupled's math:
      y = perm_out( bmm_out( ( perm_in(x) @blocks Rin ) @ W^T ) @blocks Rout )
    with R = exp(Q) built from the layer's current skew params.
    """
    from poet_torch.poet_layer import pytorch_skew_symmetric

    Qi = pytorch_skew_symmetric(pl.oft_R_in, pl.block_size_in, pl.rows_in, pl.cols_in)  # noqa: N806
    Qo = pytorch_skew_symmetric(pl.oft_R_out, pl.block_size_out, pl.rows_out, pl.cols_out)  # noqa: N806
    R_in = torch.linalg.matrix_exp(Qi.float()).to(x.dtype)  # noqa: N806
    R_out = torch.linalg.matrix_exp(Qo.float()).to(x.dtype)  # noqa: N806

    def apply_blocks(t, R, bs):  # noqa: N803
        lead = t.shape[:-1]
        n = t.numel() // t.shape[-1]
        r = R.size(0)
        tb = t.reshape(n, r, bs).transpose(0, 1)  # [r, n, bs]
        out = torch.bmm(tb, R).transpose(0, 1).reshape(*lead, r * bs)
        return out

    xin = x.index_select(-1, pl.perm_in_inv.long())
    xin = apply_blocks(xin, R_in, pl.block_size_in)
    y = xin @ pl.weight.t()
    if pl.bias is not None:
        y = y + pl.bias
    y = apply_blocks(y, R_out, pl.block_size_out)
    return y.index_select(-1, pl.perm_out.long())


def test_exp_forward_matches_pure_pytorch_oracle_cpu():
    from poet_torch import POETLinear

    pl = POETLinear(
        in_features=16,
        out_features=16,
        bsz=8,
        parameterization="exp",
        device="cpu",
        dtype=torch.float32,
        mem_efficient_mode=False,
    )
    with torch.no_grad():
        pl.weight.normal_()
        pl.oft_R_in.normal_(std=0.1)
        pl.oft_R_out.normal_(std=0.1)
    x = torch.randn(4, 16)
    y = pl(x)
    y_ref = _poet_forward_reference_exp(pl, x)
    assert torch.allclose(y, y_ref, atol=1e-4, rtol=1e-3), (y - y_ref).abs().max()


def test_exp_forward_backward_runs_cpu():
    from poet_torch import POETLinear

    pl = POETLinear(
        in_features=16,
        out_features=16,
        bsz=8,
        parameterization="exp",
        device="cpu",
        dtype=torch.float32,
    )
    with torch.no_grad():
        pl.weight.normal_()
    x = torch.randn(4, 16)
    pl(x).sum().backward()
    assert pl.oft_R_in.grad is not None
    assert pl.oft_R_out.grad is not None
    assert torch.isfinite(pl.oft_R_in.grad).all()
