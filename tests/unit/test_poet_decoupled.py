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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernel")
def test_get_weight_poet_decoupled_matches_fused_when_blocks_equal():
    """get_weight_poet_decoupled (two Cayley calls) must reproduce the fused
    get_weight_poet (one concatenated call) when block sizes are equal.

    The Cayley kernel acts per-block independently, so splitting the batch into
    two launches cannot change the per-block result beyond autotuner-config ULP.
    """
    from poet_torch.poet_layer import get_weight_poet, get_weight_poet_decoupled

    torch.manual_seed(0)
    bs = 16
    r_in, r_out = 1, 2  # in=16, out=32
    n_elems = bs * (bs - 1) // 2
    oft_R = torch.randn(r_out + r_in, n_elems, device="cuda", dtype=torch.float32) * 1e-2  # noqa: N806
    rows, cols = torch.triu_indices(bs, bs, 1, device="cuda")
    rows, cols = rows.to(torch.int32), cols.to(torch.int32)

    R_out_ref, R_in_ref = get_weight_poet(oft_R, bs, rows, cols, r_out, r_in)  # noqa: N806
    # Fused split order is [r_out, r_in]: first r_out rows → R_out.
    oft_out = oft_R[:r_out].contiguous()
    oft_in = oft_R[r_out:].contiguous()
    R_out, R_in = get_weight_poet_decoupled(  # noqa: N806
        oft_in, oft_out, bs, bs, rows, cols, rows, cols
    )
    assert torch.allclose(R_out, R_out_ref, atol=1e-6)
    assert torch.allclose(R_in, R_in_ref, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernel")
def test_get_weight_poet_decoupled_unequal_blocks_matches_reference():
    """Decoupled Cayley with unequal block sizes matches the pure-PyTorch
    cayley_pytorch oracle per side."""
    from poet_torch.poet_layer import get_weight_poet_decoupled

    torch.manual_seed(1)
    bs_in, bs_out = 8, 16
    r_in, r_out = 4, 4  # in=32, out=64, block_count=4
    n_in = bs_in * (bs_in - 1) // 2
    n_out = bs_out * (bs_out - 1) // 2
    oft_in = torch.randn(r_in, n_in, device="cuda", dtype=torch.float32) * 1e-2
    oft_out = torch.randn(r_out, n_out, device="cuda", dtype=torch.float32) * 1e-2
    rows_in, cols_in = torch.triu_indices(bs_in, bs_in, 1, device="cuda")
    rows_out, cols_out = torch.triu_indices(bs_out, bs_out, 1, device="cuda")

    R_out, R_in = get_weight_poet_decoupled(  # noqa: N806
        oft_in,
        oft_out,
        bs_in,
        bs_out,
        rows_in.to(torch.int32),
        cols_in.to(torch.int32),
        rows_out.to(torch.int32),
        cols_out.to(torch.int32),
    )
    R_in_ref = cayley_pytorch(oft_in.cpu(), bs_in)  # noqa: N806
    R_out_ref = cayley_pytorch(oft_out.cpu(), bs_out)  # noqa: N806
    assert R_in.shape == (r_in, bs_in, bs_in)
    assert R_out.shape == (r_out, bs_out, bs_out)
    assert torch.allclose(R_in.cpu(), R_in_ref, atol=1e-5)
    assert torch.allclose(R_out.cpu(), R_out_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Decoupled chain_layer op (poet::chain_layer_checkpoint_mem_o2_decoupled).
# The op is pure-PyTorch (no Triton), so it runs on CPU too; we exercise it on
# CUDA when available (the plan's single-GPU scope) but fall back to CPU.
# ---------------------------------------------------------------------------

_OP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _chain_pytorch_decoupled(x, Rin, W, bias, Rout, perm_in_inv, perm_out, bs_in, bs_out):  # noqa: N803
    """Op-level pure-PyTorch oracle: R blocks supplied directly (no Cayley)."""
    x = x.index_select(-1, perm_in_inv.long())
    x = apply_block_diag(x, Rin, bs_in)
    y = x @ W.t()
    if bias is not None:
        y = y + bias
    y = apply_block_diag(y, Rout, bs_out)
    return y.index_select(-1, perm_out.long())


def _make_op_inputs(in_f, out_f, bs_in, bs_out, device, dtype=torch.float32, seed=0, batch=4):
    import poet_torch.poet_ops  # noqa: F401  (registers torch.ops.poet.* custom ops)

    torch.manual_seed(seed)
    r_in, r_out = in_f // bs_in, out_f // bs_out
    R_in = cayley_pytorch(  # noqa: N806
        torch.randn(r_in, bs_in * (bs_in - 1) // 2, dtype=dtype) * 1e-2, bs_in
    ).to(device)
    R_out = cayley_pytorch(  # noqa: N806
        torch.randn(r_out, bs_out * (bs_out - 1) // 2, dtype=dtype) * 1e-2, bs_out
    ).to(device)
    W = torch.randn(out_f, in_f, device=device, dtype=dtype)  # noqa: N806
    perm_in = torch.randperm(in_f, device=device)
    perm_out = torch.randperm(out_f, device=device)
    perm_in_inv = torch.argsort(perm_in).to(torch.int32)
    perm_out_inv = torch.argsort(perm_out).to(torch.int32)
    perm_in = perm_in.to(torch.int32)
    perm_out = perm_out.to(torch.int32)
    x = torch.randn(batch, in_f, device=device, dtype=dtype)
    return x, R_in, R_out, W, perm_in, perm_in_inv, perm_out, perm_out_inv


def test_decoupled_op_matches_coupled_op_when_blocks_equal():
    """Hard constraint #1: with bsz_in == bsz_out the decoupled op must be
    bit-equivalent to the existing coupled op (same operation sequence)."""
    x, R_in, R_out, W, p_in, p_in_inv, p_out, p_out_inv = _make_op_inputs(  # noqa: N806
        32, 64, 8, 8, _OP_DEVICE, seed=1
    )
    y_old = torch.ops.poet.chain_layer_checkpoint_mem_o2(
        x, R_in, W, None, R_out, p_in_inv, p_in, p_out, p_out_inv, 8
    )
    y_new = torch.ops.poet.chain_layer_checkpoint_mem_o2_decoupled(
        x, R_in, W, None, R_out, p_in_inv, p_in, p_out, p_out_inv, 8, 8
    )
    assert torch.equal(y_old, y_new), f"max abs diff {(y_old - y_new).abs().max().item():.2e}"


def test_decoupled_op_unequal_blocks_matches_reference():
    """Forward parity (unequal block sizes) vs the pure-PyTorch op-level oracle."""
    x, R_in, R_out, W, p_in, p_in_inv, p_out, p_out_inv = _make_op_inputs(  # noqa: N806
        32, 64, 8, 16, _OP_DEVICE, seed=2
    )
    y_op = torch.ops.poet.chain_layer_checkpoint_mem_o2_decoupled(
        x, R_in, W, None, R_out, p_in_inv, p_in, p_out, p_out_inv, 8, 16
    )
    y_ref = _chain_pytorch_decoupled(x, R_in, W, None, R_out, p_in_inv, p_out, 8, 16)
    assert torch.allclose(
        y_op, y_ref, atol=1e-4
    ), f"max abs diff {(y_op - y_ref).abs().max().item():.2e}"


def test_decoupled_op_with_bias_matches_reference():
    """Forward parity with a bias term."""
    x, R_in, R_out, W, p_in, p_in_inv, p_out, p_out_inv = _make_op_inputs(  # noqa: N806
        32, 64, 8, 16, _OP_DEVICE, seed=3
    )
    bias = torch.randn(64, device=_OP_DEVICE)
    y_op = torch.ops.poet.chain_layer_checkpoint_mem_o2_decoupled(
        x, R_in, W, bias, R_out, p_in_inv, p_in, p_out, p_out_inv, 8, 16
    )
    y_ref = _chain_pytorch_decoupled(x, R_in, W, bias, R_out, p_in_inv, p_out, 8, 16)
    assert torch.allclose(y_op, y_ref, atol=1e-4)


def test_decoupled_op_backward_matches_reference():
    """Backward parity (unequal blocks): grads wrt x, R_in, R_out from the op's
    hand-written backward match autograd through the pure-PyTorch oracle."""
    x, R_in, R_out, W, p_in, p_in_inv, p_out, p_out_inv = _make_op_inputs(  # noqa: N806
        32, 64, 8, 16, _OP_DEVICE, seed=4
    )

    def run(fn):
        xx = x.clone().requires_grad_(True)
        ri = R_in.clone().requires_grad_(True)
        ro = R_out.clone().requires_grad_(True)
        y = fn(xx, ri, ro)
        loss = (
            y * torch.arange(1, y.numel() + 1, device=y.device, dtype=y.dtype).reshape_as(y)
        ).sum()
        loss.backward()
        return xx.grad, ri.grad, ro.grad

    gx_op, gri_op, gro_op = run(
        lambda xx, ri, ro: torch.ops.poet.chain_layer_checkpoint_mem_o2_decoupled(
            xx, ri, W, None, ro, p_in_inv, p_in, p_out, p_out_inv, 8, 16
        )
    )
    gx_ref, gri_ref, gro_ref = run(
        lambda xx, ri, ro: _chain_pytorch_decoupled(xx, ri, W, None, ro, p_in_inv, p_out, 8, 16)
    )
    assert torch.allclose(gx_op, gx_ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(gri_op, gri_ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(gro_op, gro_ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# POETLinear refactor: constructor + forward (decoupled storage internally).
# ---------------------------------------------------------------------------


def test_poetlinear_constructor_validation():
    """Exactly one of bsz / block_count; block_count must divide both dims."""
    from poet_torch import POETLinear

    with pytest.raises(ValueError, match="exactly one of bsz or block_count"):
        POETLinear(in_features=32, out_features=32)  # neither
    with pytest.raises(ValueError, match="exactly one of bsz or block_count"):
        POETLinear(in_features=32, out_features=32, bsz=8, block_count=4)  # both
    with pytest.raises(ValueError, match="block_count 7 doesn't divide"):
        POETLinear(in_features=32, out_features=64, block_count=7)
    with pytest.raises(ValueError, match="block_size 7 doesn't divide"):
        POETLinear(in_features=32, out_features=64, bsz=7)


def test_poetlinear_block_count_sets_decoupled_sizes():
    """block_count=4 on (32,64) ⇒ bs_in=8, bs_out=16, r_in=r_out=4; two params."""
    from poet_torch import POETLinear

    layer = POETLinear(in_features=32, out_features=64, block_count=4, dtype=torch.float32)
    assert layer.block_size_in == 8
    assert layer.block_size_out == 16
    assert layer.r_in == 4 and layer.r_out == 4
    assert layer.oft_R_in.shape == (4, 8 * 7 // 2)
    assert layer.oft_R_out.shape == (4, 16 * 15 // 2)
    # Legacy bsz path stores equal sizes.
    legacy = POETLinear(in_features=32, out_features=32, bsz=8, dtype=torch.float32)
    assert legacy.block_size_in == legacy.block_size_out == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernel")
def test_poetlinear_bsz_and_block_count_agree_when_equal():
    """Task 4.5: bsz=8 and block_count=4 on a 32x32 layer build identical
    internal shapes and produce identical forward output given identical
    params/perms (both ⇒ block_size_in == block_size_out == 8)."""
    from poet_torch import POETLinear

    torch.manual_seed(0)
    a = POETLinear(in_features=32, out_features=32, bsz=8, device="cuda", dtype=torch.float32)
    a.random_init_parameters()
    b = POETLinear(
        in_features=32, out_features=32, block_count=4, device="cuda", dtype=torch.float32
    )
    # Copy a's state into b so they're identical.
    b.weight.detach().copy_(a.weight.detach())
    b.oft_R_in.detach().copy_(a.oft_R_in.detach())
    b.oft_R_out.detach().copy_(a.oft_R_out.detach())
    for buf in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
        getattr(b, buf).copy_(getattr(a, buf))

    x = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    ya = a(x)
    yb = b(x)
    assert torch.equal(ya, yb), f"max abs diff {(ya - yb).abs().max().item():.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernel")
def test_poetlinear_decoupled_forward_matches_reference():
    """Task 4.6: a decoupled POETLinear (block_count=4 on 32x64, bs_in=8,
    bs_out=16) matches the pure-PyTorch reference."""
    from poet_torch import POETLinear

    torch.manual_seed(1)
    layer = POETLinear(
        in_features=32, out_features=64, block_count=4, device="cuda", dtype=torch.float32
    )
    layer.random_init_parameters()

    x = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    y_layer = layer(x).detach().cpu()
    y_ref = poet_reference_forward(
        x.cpu(),
        layer.weight.detach().cpu(),
        layer.oft_R_in.detach().cpu(),
        layer.oft_R_out.detach().cpu(),
        layer.perm_in.cpu(),
        layer.perm_in_inv.cpu(),
        layer.perm_out.cpu(),
        layer.perm_out_inv.cpu(),
        layer.block_size_in,
        layer.block_size_out,
    )
    assert torch.allclose(
        y_layer, y_ref, atol=1e-4
    ), f"max abs diff {(y_layer - y_ref).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# merge_then_reinitialize (decoupled): block_diag_lr_matmul_decoupled + parity.
# ---------------------------------------------------------------------------


def test_block_diag_lr_matmul_decoupled_matches_coupled_equal_blocks():
    """The decoupled block-diag L/R matmul reduces to the coupled one when
    a == b (CPU; pure PyTorch)."""
    from poet_torch.poet_layer import block_diag_lr_matmul, block_diag_lr_matmul_decoupled

    torch.manual_seed(0)
    r_m, r_n, b = 3, 4, 5
    A = torch.randn(r_m, b, b, dtype=torch.float64)  # noqa: N806
    B = torch.randn(r_n, b, b, dtype=torch.float64)  # noqa: N806
    M = torch.randn(r_m * b, r_n * b, dtype=torch.float64)  # noqa: N806
    out_coupled = block_diag_lr_matmul(A, M, B)
    out_decoupled = block_diag_lr_matmul_decoupled(A, M, B)
    assert torch.allclose(out_coupled, out_decoupled, atol=1e-12)


def test_block_diag_lr_matmul_decoupled_unequal_matches_dense():
    """Unequal block sizes match an explicit dense block_diag(A) @ M @ block_diag(B)."""
    from poet_torch.poet_layer import block_diag_lr_matmul_decoupled

    torch.manual_seed(1)
    r_m, a, r_n, b = 2, 8, 3, 16
    A = torch.randn(r_m, a, a, dtype=torch.float64)  # noqa: N806
    B = torch.randn(r_n, b, b, dtype=torch.float64)  # noqa: N806
    M = torch.randn(r_m * a, r_n * b, dtype=torch.float64)  # noqa: N806
    out = block_diag_lr_matmul_decoupled(A, M, B)
    dense = torch.block_diag(*A) @ M @ torch.block_diag(*B)
    assert out.shape == (r_m * a, r_n * b)
    assert torch.allclose(out, dense, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernel")
def test_merge_equivalence_bsz_vs_block_count_equal():
    """Task 5.3: with equal block sizes, the bsz and block_count merge paths
    produce identical weight + permutation state across several merge cycles
    (given identical params and identical RNG seeding at merge time)."""
    from poet_torch import POETLinear

    torch.manual_seed(0)
    a = POETLinear(in_features=32, out_features=32, bsz=8, device="cuda", dtype=torch.float32)
    a.random_init_parameters()
    b = POETLinear(
        in_features=32, out_features=32, block_count=4, device="cuda", dtype=torch.float32
    )
    with torch.no_grad():
        b.weight.copy_(a.weight)
        for buf in ("perm_in", "perm_in_inv", "perm_out", "perm_out_inv"):
            getattr(b, buf).copy_(getattr(a, buf))

    for cycle in range(3):
        torch.manual_seed(cycle)
        new_in = torch.randn_like(a.oft_R_in) * 1e-2
        new_out = torch.randn_like(a.oft_R_out) * 1e-2
        with torch.no_grad():
            a.oft_R_in.copy_(new_in)
            a.oft_R_out.copy_(new_out)
            b.oft_R_in.copy_(new_in)
            b.oft_R_out.copy_(new_out)
        torch.manual_seed(1000 + cycle)
        a.merge_then_reinitialize()
        torch.manual_seed(1000 + cycle)
        b.merge_then_reinitialize()
        assert torch.allclose(a.weight, b.weight, atol=1e-5), f"cycle {cycle} weight mismatch"
        assert torch.equal(a.perm_in, b.perm_in)
        assert torch.equal(a.perm_out, b.perm_out)
        # oft_R is zeroed after merge.
        assert torch.count_nonzero(a.oft_R_in) == 0 and torch.count_nonzero(a.oft_R_out) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernel")
def test_merge_preserves_effective_weight():
    """A merge folds the orthogonal delta into the base weight, so the layer's
    forward output is (approximately) unchanged immediately after merging."""
    from poet_torch import POETLinear

    torch.manual_seed(2)
    layer = POETLinear(
        in_features=32, out_features=64, block_count=4, device="cuda", dtype=torch.float32
    )
    layer.random_init_parameters()
    with torch.no_grad():
        layer.oft_R_in.normal_(std=1e-2)
        layer.oft_R_out.normal_(std=1e-2)

    x = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    y_before = layer(x).detach()
    layer.merge_then_reinitialize()
    y_after = layer(x).detach()
    # After merge oft_R==0, so R blocks are identity → forward is pure permuted
    # base weight, which equals the pre-merge effective weight.
    assert torch.allclose(
        y_before, y_after, atol=1e-3
    ), f"max abs diff {(y_before - y_after).abs().max().item():.2e}"
