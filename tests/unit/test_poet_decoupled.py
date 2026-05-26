"""Decoupled-block-count POET: pure-PyTorch reference + parity tests.

This module is the ground-truth oracle for the decoupled-block-count work
(see docs/superpowers/plans/2026-05-27-poet-decoupled-block-count.md).

`block_count = n` gives both sides `n` blocks but potentially different
block sizes::

    block_size_in  = in_features  / n
    block_size_out = out_features / n

The pure-PyTorch reference mirrors the math computed by upstream
``POETLinear.forward`` / ``chain_layer_checkpoint_mem_o2`` exactly, but
allows ``block_size_in != block_size_out``.

Tests split into:
  * CPU-runnable — reference internal consistency (no Triton, no compile).
  * single-GPU-required — parity against the actual Triton-backed layer
    (guarded by ``torch.cuda.is_available()``).
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Pure-PyTorch reference (CPU-runnable; the oracle for all kernel work).
# ---------------------------------------------------------------------------


def _triu_indices(block_size: int, device=None):
    rows, cols = torch.triu_indices(block_size, block_size, 1, device=device)
    return rows.to(torch.int64), cols.to(torch.int64)


def _skew_symmetric(vec: torch.Tensor, block_size: int, rows, cols) -> torch.Tensor:
    """Build a batch of skew-symmetric matrices from upper-triangular params.

    Mirrors ``poet_torch.poet_layer.pytorch_skew_symmetric``.
    """
    batch = vec.shape[0]
    mat = vec.new_zeros(batch, block_size, block_size)
    mat[:, rows, cols] = vec
    mat = mat - mat.transpose(-2, -1)
    return mat


def cayley_pytorch(oft_R: torch.Tensor, block_size: int) -> torch.Tensor:  # noqa: N803
    """4th-order Cayley/Neumann approximant, matching the Triton ``poet::cayley``
    kernel exactly: ``Y = I + 2Q + 2Q^2 + 2Q^3 + Q^4`` (note the coefficient
    on the Q^4 term is **1**, not 2 — the dead ``cayley_batch`` helper in
    poet_layer.py uses 2 and does NOT match the kernel).
    """
    rows, cols = _triu_indices(block_size, device=oft_R.device)
    Q = _skew_symmetric(oft_R, block_size, rows, cols)  # noqa: N806
    Q2 = Q @ Q  # noqa: N806
    Q3 = Q2 @ Q  # noqa: N806
    Q4 = Q2 @ Q2  # noqa: N806
    eye = torch.eye(block_size, device=oft_R.device, dtype=oft_R.dtype)
    return 2.0 * Q + 2.0 * Q2 + 2.0 * Q3 + Q4 + eye.unsqueeze(0)


def apply_block_diag(x: torch.Tensor, R_blocks: torch.Tensor, block_size: int) -> torch.Tensor:  # noqa: N803
    """Right-multiply ``x`` by ``block_diag(R_blocks)`` without materializing it.

    Mirrors ``poet_torch.poet_layer.torch_bmm``: ``out[..., r, c] =
    sum_k x[..., r, k] R[r, k, c]``.
    """
    lead = x.shape[:-1]
    xr = x.view(*lead, -1, block_size)
    xr = torch.einsum("...rk,rkc->...rc", xr, R_blocks)
    return xr.reshape(*lead, -1)


def poet_reference_forward(
    x: torch.Tensor,
    W: torch.Tensor,  # noqa: N803
    oft_R_in: torch.Tensor,  # noqa: N803
    oft_R_out: torch.Tensor,  # noqa: N803
    perm_in: torch.Tensor,
    perm_in_inv: torch.Tensor,
    perm_out: torch.Tensor,
    perm_out_inv: torch.Tensor,
    block_size_in: int,
    block_size_out: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch decoupled POET forward.

    Mirrors ``chain_layer_checkpoint_mem_o2`` exactly: the stored ``W`` is the
    permutation-folded weight, so the forward applies only ``perm_in_inv`` on
    the input and ``perm_out`` on the output (NO internal re-permute between
    R_in and W or between W and R_out). ``perm_in`` / ``perm_out_inv`` are
    accepted for signature symmetry / backward but unused in the forward.
    """
    R_in = cayley_pytorch(oft_R_in, block_size_in)  # noqa: N806
    R_out = cayley_pytorch(oft_R_out, block_size_out)  # noqa: N806
    x = x.index_select(-1, perm_in_inv.long())
    x = apply_block_diag(x, R_in, block_size_in)
    y = x @ W.t()
    if bias is not None:
        y = y + bias
    y = apply_block_diag(y, R_out, block_size_out)
    y = y.index_select(-1, perm_out.long())
    return y


def _dense_block_diag_forward(x, w, r_in_blocks, r_out_blocks, perm_in_inv, perm_out, bias=None):
    """Independent oracle using explicit ``torch.block_diag`` dense matrices."""
    Rin = torch.block_diag(*r_in_blocks)  # noqa: N806
    Rout = torch.block_diag(*r_out_blocks)  # noqa: N806
    xp = x.index_select(-1, perm_in_inv.long())
    y = (xp @ Rin) @ w.t()
    if bias is not None:
        y = y + bias
    y = y @ Rout
    return y.index_select(-1, perm_out.long())


# ---------------------------------------------------------------------------
# CPU tests — reference internal consistency (no Triton, no torch.compile).
# ---------------------------------------------------------------------------


def test_cayley_pytorch_matches_explicit_polynomial():
    """cayley_pytorch == I + 2Q + 2Q^2 + 2Q^3 + Q^4 for a hand-built Q."""
    torch.manual_seed(0)
    bs = 4
    oft = torch.randn(2, bs * (bs - 1) // 2, dtype=torch.float64) * 0.1
    rows, cols = _triu_indices(bs)
    Q = _skew_symmetric(oft, bs, rows, cols)  # noqa: N806
    expected = (
        torch.eye(bs, dtype=torch.float64) + 2 * Q + 2 * (Q @ Q) + 2 * (Q @ Q @ Q) + (Q @ Q @ Q @ Q)
    )
    got = cayley_pytorch(oft, bs)
    assert torch.allclose(got, expected, atol=1e-12)


def test_cayley_pytorch_near_orthogonal_for_small_params():
    """For small skew params the Cayley approximant is near-orthogonal."""
    torch.manual_seed(1)
    bs = 8
    oft = torch.randn(3, bs * (bs - 1) // 2, dtype=torch.float64) * 1e-3
    R = cayley_pytorch(oft, bs)  # noqa: N806
    eye = torch.eye(bs, dtype=torch.float64).unsqueeze(0)
    err = (R @ R.transpose(-2, -1) - eye).abs().max().item()
    assert err < 1e-5, f"R R^T deviates from I by {err:.2e}"


def test_reference_matches_dense_block_diag_equal_blocks():
    """The bmm-based reference matches the explicit dense block-diag oracle
    when block sizes are equal."""
    torch.manual_seed(2)
    in_f, out_f, bs = 32, 32, 8
    r_in, r_out = in_f // bs, out_f // bs
    n_elems = bs * (bs - 1) // 2
    oft_in = torch.randn(r_in, n_elems, dtype=torch.float64) * 1e-2
    oft_out = torch.randn(r_out, n_elems, dtype=torch.float64) * 1e-2
    W = torch.randn(out_f, in_f, dtype=torch.float64)  # noqa: N806
    perm_in = torch.randperm(in_f)
    perm_out = torch.randperm(out_f)
    perm_in_inv = torch.argsort(perm_in)
    perm_out_inv = torch.argsort(perm_out)
    x = torch.randn(4, in_f, dtype=torch.float64)

    y_ref = poet_reference_forward(
        x, W, oft_in, oft_out, perm_in, perm_in_inv, perm_out, perm_out_inv, bs, bs
    )
    R_in = cayley_pytorch(oft_in, bs)  # noqa: N806
    R_out = cayley_pytorch(oft_out, bs)  # noqa: N806
    y_dense = _dense_block_diag_forward(x, W, R_in, R_out, perm_in_inv, perm_out)
    assert torch.allclose(y_ref, y_dense, atol=1e-10)


def test_reference_matches_dense_block_diag_unequal_blocks():
    """Same, but with decoupled block sizes (block_count semantics)."""
    torch.manual_seed(3)
    in_f, out_f, n = 32, 64, 4
    bs_in, bs_out = in_f // n, out_f // n  # 8, 16
    n_in = bs_in * (bs_in - 1) // 2
    n_out = bs_out * (bs_out - 1) // 2
    oft_in = torch.randn(n, n_in, dtype=torch.float64) * 1e-2
    oft_out = torch.randn(n, n_out, dtype=torch.float64) * 1e-2
    W = torch.randn(out_f, in_f, dtype=torch.float64)  # noqa: N806
    perm_in = torch.randperm(in_f)
    perm_out = torch.randperm(out_f)
    perm_in_inv = torch.argsort(perm_in)
    perm_out_inv = torch.argsort(perm_out)
    x = torch.randn(5, in_f, dtype=torch.float64)

    y_ref = poet_reference_forward(
        x, W, oft_in, oft_out, perm_in, perm_in_inv, perm_out, perm_out_inv, bs_in, bs_out
    )
    R_in = cayley_pytorch(oft_in, bs_in)  # noqa: N806
    R_out = cayley_pytorch(oft_out, bs_out)  # noqa: N806
    y_dense = _dense_block_diag_forward(x, W, R_in, R_out, perm_in_inv, perm_out)
    assert y_ref.shape == (5, out_f)
    assert torch.allclose(y_ref, y_dense, atol=1e-10)


def test_block_count_implies_expected_block_sizes():
    """block_count=n on a (in, out) layer ⇒ bs_in=in/n, bs_out=out/n."""
    in_f, out_f, n = 4096, 11008, 8
    assert in_f % n == 0 and out_f % n == 0
    assert in_f // n == 512
    assert out_f // n == 1376


# ---------------------------------------------------------------------------
# Single-GPU test — reference matches the actual Triton-backed POETLinear.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernel")
def test_reference_matches_poet_linear_when_block_sizes_equal():
    """The pure-PyTorch reference reproduces upstream POETLinear.forward
    (single oft_R) when block sizes are equal, by splitting the layer's
    concatenated oft_R into the (R_out, R_in) halves the kernel uses.
    """
    from poet_torch import POETLinear

    torch.manual_seed(0)
    in_f, out_f, bs = 32, 32, 8
    layer = POETLinear(
        in_features=in_f,
        out_features=out_f,
        bsz=bs,
        bias=False,
        device="cuda",
        dtype=torch.float32,
    )
    layer.random_init_parameters()
    # Upstream get_weight_poet splits R_cat as [r_out, r_in]: the first r_out
    # rows of oft_R drive R_out, the remaining r_in rows drive R_in.
    oft_out = layer.oft_R[: layer.r_out].detach().cpu()
    oft_in = layer.oft_R[layer.r_out :].detach().cpu()
    W = layer.weight.detach().cpu()  # noqa: N806
    perm_in = layer.perm_in.detach().cpu()
    perm_in_inv = layer.perm_in_inv.detach().cpu()
    perm_out = layer.perm_out.detach().cpu()
    perm_out_inv = layer.perm_out_inv.detach().cpu()

    x = torch.randn(4, in_f, device="cuda", dtype=torch.float32)
    y_layer = layer(x).detach().cpu()
    y_ref = poet_reference_forward(
        x.cpu(), W, oft_in, oft_out, perm_in, perm_in_inv, perm_out, perm_out_inv, bs, bs
    )
    assert torch.allclose(
        y_layer, y_ref, atol=1e-4
    ), f"max abs diff {(y_layer - y_ref).abs().max().item():.2e}"
