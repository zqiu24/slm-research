# src/diag/poet_coordination_diag.py
"""Tier-0 diagnostics for POET two-sided (in/out) coordination.

Pure-math arbiters for *why* alternating + fresh-momentum beats simultaneous,
designed to be logged on the live champion run (mostly free / one extra matmul,
no extra backward). Plain tensors only — no Megatron / CUDA / poet_torch — so the
math is unit-testable on CPU. Integration (sampling, reaching W, wandb) lives in
the optimizer-side hook, not here.

Two arbiters:
  * momentum_grad_cosine(lie_m, grad): cos between a side's momentum and its FRESH
    skew-tangent gradient at the current weight. Tests the *staleness* mechanism —
    high (~>0.8) on the champion (both momenta fed every step); drops at reactivation
    when the inactive side's momentum is frozen (true_single_side).
  * direction_overlap(D_out, D_in): cos(D_out, D_in), joint-movement ratio, and the
    2x2 direction-Gram condition number, with D_out = A_out @ W, D_in = W @ A_in in
    weight space. Tests the *gauge-redundancy* mechanism — large |cos| / cond(M)
    means simultaneous wastes its matched-||dW|| budget on the redundant/cancelling
    direction; cos ~ 0 falsifies the redundancy story.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from src.diag.skew_conditioning import vec_to_skew


def momentum_grad_cosine(m: torch.Tensor, g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Frobenius cosine between momentum ``m`` and fresh gradient ``g``.

    Reduces over the WHOLE tensor (scalar output), so a batched
    ``(num_blocks, n_elems)`` skew-vector pair yields one number per side/param.
    Scale-invariant; zero input -> 0 (never NaN). The sqrt(2) skew<->vec norm
    factor cancels in the ratio, so this is identical in vec- or skew-space.
    """
    m = m.flatten().to(torch.float32)
    g = g.flatten().to(torch.float32)
    denom = (m.norm() * g.norm()).clamp_min(eps)
    return torch.dot(m, g) / denom


def side_directions(
    a_out_skew: torch.Tensor, a_in_skew: torch.Tensor, w_perm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order weight-space directions of the two sides, in the W_perm frame.

    POET's generators are block-diagonal only in the un-permuted W_perm frame
    (blocks contiguous), so the caller must pass ``w_perm`` (= forward-frame weight
    index-selected back through perm_out_inv / perm_in_inv) and the per-block skew
    generators. Returns the dense first-order contributions

        D_out = blockdiag(A_out) @ W_perm        (shape out x in)
        D_in  = W_perm @ blockdiag(A_in)         (shape out x in)

    computed without ever materializing the dense block-diagonal (one bmm + one
    einsum). cos(D_out, D_in) is permutation-invariant, so the overlap geometry is
    the same as in the forward frame.

    Args:
        a_out_skew: (r_out, b_out, b_out) out-side block generators.
        a_in_skew:  (r_in, b_in, b_in) in-side block generators.
        w_perm:     (r_out*b_out, r_in*b_in) weight in the W_perm frame.
    """
    w = w_perm.to(torch.float32)
    a_out = a_out_skew.to(torch.float32)
    a_in = a_in_skew.to(torch.float32)
    out_features, in_features = w.shape
    r_out, b_out, _ = a_out.shape
    r_in, b_in, _ = a_in.shape
    # D_out = blockdiag(A_out) @ W : group W rows into r_out contiguous b_out-blocks.
    d_out = torch.bmm(a_out, w.reshape(r_out, b_out, in_features)).reshape(
        out_features, in_features
    )
    # D_in = W @ blockdiag(A_in) : group W cols into r_in contiguous b_in-blocks.
    d_in = torch.einsum("orb,rbc->orc", w.reshape(out_features, r_in, b_in), a_in).reshape(
        out_features, in_features
    )
    return d_out, d_in


def direction_overlap(
    d_out: torch.Tensor, d_in: torch.Tensor, eps: float = 1e-12
) -> dict[str, torch.Tensor]:
    """Overlap geometry of the two sides' weight-space directions.

    Args:
        d_out: D_out = A_out @ W, shape (d_out, d_in).
        d_in:  D_in  = W @ A_in,  shape (d_out, d_in).

    Returns scalar tensors:
        cos       = <D_out, D_in>_F / (||D_out||_F ||D_in||_F)
        r_joint   = ||D_out + D_in||_F^2 / (||D_out||_F^2 + ||D_in||_F^2)
                    (<1 cancellation, =1 orthogonal, >1 reinforcement)
        gram_cond = condition number of M = [[||D_out||^2, <D_out,D_in>],
                                             [<D_out,D_in>, ||D_in||^2]]
                    (large => the two directions span a near-singular / redundant
                    2D subspace, i.e. unit-coefficient simultaneous over-spends it)
    """
    d_out = d_out.to(torch.float32)
    d_in = d_in.to(torch.float32)
    a = (d_out * d_out).sum()  # ||D_out||^2
    b = (d_in * d_in).sum()  # ||D_in||^2
    c = (d_out * d_in).sum()  # <D_out, D_in>

    cos = c / (a.sqrt() * b.sqrt()).clamp_min(eps)
    r_joint = ((d_out + d_in) * (d_out + d_in)).sum() / (a + b).clamp_min(eps)

    # Eigenvalues of the symmetric 2x2 Gram, analytically.
    half_tr = 0.5 * (a + b)
    disc = (0.5 * (a - b)) ** 2 + c * c
    root = disc.clamp_min(0.0).sqrt()
    lam_max = half_tr + root
    lam_min = (half_tr - root).clamp_min(eps)
    gram_cond = lam_max / lam_min

    return {"cos": cos, "r_joint": r_joint, "gram_cond": gram_cond}


def layer_coordination_metrics(
    lie_m_out: torch.Tensor,
    grad_out: torch.Tensor,
    lie_m_in: torch.Tensor,
    grad_in: torch.Tensor,
    w_perm: torch.Tensor,
    *,
    block_size_out: int,
    block_size_in: int,
    orthogonalize_fn: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, float]:
    """Assemble the Tier-0 coordination metrics for one POET layer.

    Pure: takes the two sides' momenta (``lie_m_*``, vec form ``(r, n_elems)``),
    their FRESH skew-tangent gradients (``grad_*``, same shape), the layer weight in
    the W_perm frame, and an injected ``orthogonalize_fn`` (the optimizer's Muon NS,
    so the diag module stays free of optim imports). The realized directions are
    ``A = orthogonalize(-lie_m)`` per side — the common ortho_c*lr scale cancels in
    every returned ratio — so this reflects the geometry the optimizer would write.

    Returns plain floats (wandb-ready):
        mom_cos_out / mom_cos_in  -> cos(momentum, fresh grad), the STALENESS arbiter
        cos_D_out_D_in            -> overlap of the two sides' weight-space directions
        r_joint, gram_cond        -> joint-movement ratio + direction-Gram conditioning
        norm_D_out / norm_D_in    -> relative per-side movement magnitude
    """
    mom_cos_out = momentum_grad_cosine(lie_m_out, grad_out)
    mom_cos_in = momentum_grad_cosine(lie_m_in, grad_in)

    a_out = orthogonalize_fn(vec_to_skew(-lie_m_out.to(torch.float32), block_size_out))
    a_in = orthogonalize_fn(vec_to_skew(-lie_m_in.to(torch.float32), block_size_in))
    d_out, d_in = side_directions(a_out, a_in, w_perm)
    ov = direction_overlap(d_out, d_in)

    return {
        "mom_cos_out": mom_cos_out.item(),
        "mom_cos_in": mom_cos_in.item(),
        "cos_D_out_D_in": ov["cos"].item(),
        "r_joint": ov["r_joint"].item(),
        "gram_cond": ov["gram_cond"].item(),
        "norm_D_out": d_out.norm().item(),
        "norm_D_in": d_in.norm().item(),
    }
