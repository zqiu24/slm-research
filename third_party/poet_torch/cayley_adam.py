"""
Cayley Adam – AdamW on the Stiefel manifold via Cayley retraction.

Reference:
  Jun Li et al., "Efficient Riemannian Optimization on the Stiefel Manifold
  via the Cayley Transform", ICLR 2020.
  https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform

For parameters marked ``stiefel=True`` the optimizer:
  1. Projects the Euclidean gradient to the tangent space of the Stiefel manifold.
  2. Maintains Adam-style first / second moments *in the tangent space*.
  3. Builds a skew-symmetric operator W and retracts with the iterative Cayley map.

For all other parameters it falls back to standard decoupled AdamW.

Distributed sharding (inspired by Muon):
  The expensive Stiefel step (tangent projection + Cayley loop) is sharded across
  GPUs. Each rank only computes the manifold update for 1/world_size of the Stiefel
  parameters, then broadcasts to sync. Moment updates happen on all ranks (cheap).
"""

import math
import random
from typing import Callable, Iterable, Tuple

import torch
import torch.distributed as dist
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Helper: iterative Cayley retraction  (from gutils.py in the reference repo)
# ---------------------------------------------------------------------------

def _cayley_loop(X: torch.Tensor, W: torch.Tensor, tan_vec: torch.Tensor,
                 t: float, num_iters: int = 5) -> torch.Tensor:
    """
    Iterative Cayley retraction on the Stiefel manifold (batched).

    Y^0     = X + t * tan_vec
    Y^{k+1} = X + t * W @ 0.5 * (X + Y^k)

    Args:
        X:        (B, n, n) – current point on the manifold.
        W:        (B, n, n) – skew-symmetric operator.
        tan_vec:  (B, n, n) – tangent vector at X.
        t:        scalar step size.
        num_iters: fixed-point iterations (default 5).
    Returns:
        Y: (B, n, n)
    """
    Y = X + t * tan_vec
    for _ in range(num_iters):
        Y = X + t * torch.bmm(W, 0.5 * (X + Y))
    return Y


def _matrix_norm_one_batched(W: torch.Tensor) -> torch.Tensor:
    """Max absolute column sum (1-norm) per batch element. Returns (B,)."""
    return W.abs().sum(dim=-2).max(dim=-1).values


# ---------------------------------------------------------------------------
# QR re-orthogonalization (safety net, disabled by default)
# ---------------------------------------------------------------------------

def _qr_retraction(X: torch.Tensor) -> torch.Tensor:
    """
    QR-based retraction: given (B, n, p), return (B, n, p) with orthonormal columns.
    """
    orig_dtype = X.dtype
    X_f = X.float()  # QR not implemented for BFloat16
    Q, R = torch.linalg.qr(X_f)
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    Q = Q * signs.unsqueeze(-2)
    return Q.to(orig_dtype)


# ---------------------------------------------------------------------------
# Core Stiefel step (standalone so it can be called per-parameter)
# ---------------------------------------------------------------------------

def _stiefel_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    beta1: float,
    beta2: float,
    beta1_power: float,
    beta2_power: float,
    lr: float,
    eps: float,
    cayley_iters: int,
    correct_bias: bool,
    poet_scale: float = 1.0,
) -> None:
    """In-place Stiefel Adam step for a (B, n, n) orthogonal parameter."""
    X = p.data  # (B, n, n)
    G = grad

    # Update moments (Euclidean space)
    m.mul_(beta1).add_(G, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(G, G, value=1.0 - beta2)

    # Bias correction
    if correct_bias:
        bc1 = 1.0 - beta1_power
        bc2 = 1.0 - beta2_power
        m_hat = m / bc1
        v_hat = v / bc2
    else:
        m_hat = m
        v_hat = v

    # Adam direction -> project to tangent space of St(n,n)
    adam_dir = m_hat / (v_hat.sqrt() + eps)
    XtD = torch.bmm(X.transpose(-1, -2), adam_dir)
    sym_XtD = 0.5 * (XtD + XtD.transpose(-1, -2))
    tangent = adam_dir - torch.bmm(X, sym_XtD)

    # Skew-symmetric operator W
    W = torch.bmm(tangent, X.transpose(-1, -2))
    W = W - W.transpose(-1, -2)

    # Adaptive step size
    max_norm = _matrix_norm_one_batched(W).max().item()
    t = 0.5 * 2.0 / (max_norm + eps)
    alpha = min(t, lr * poet_scale)

    # Cayley retraction (negative alpha for descent, matching reference AdamG)
    tan_vec = torch.bmm(W, X)
    p_new = _cayley_loop(X, W, tan_vec, -alpha, num_iters=cayley_iters)
    p.data.copy_(p_new)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class CayleyAdam(Optimizer):
    """
    AdamW with Cayley retraction for Stiefel-manifold parameters.

    Parameter groups may contain ``stiefel=True``. Those parameters are updated
    on the Stiefel manifold via Cayley retraction; others get standard AdamW.

    Distributed sharding: when ``world_size > 1``, the expensive Stiefel
    computation is split across ranks (like Muon's Newton-Schulz sharding).
    Each rank computes the manifold update for ~1/world_size of the Stiefel
    params, then broadcasts. Moment buffers are updated on all ranks (cheap).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        poet_block_size: int = 256,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            correct_bias=correct_bias, stiefel=False,
            cayley_iters=5, qr_prob=0.0, poet_reset_gap=0,
        )
        super().__init__(params, defaults)
        self.global_step_counter = 0
        self.poet_block_size = poet_block_size

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        # Detect distributed context
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        for group in self.param_groups:
            is_stiefel = group.get("stiefel", False)
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            correct_bias = group.get("correct_bias", True)
            cayley_iters = group.get("cayley_iters", 5)
            qr_prob = group.get("qr_prob", 0.0)
            poet_scale = group.get("poet_scale", 1.0)

            # ----- Stiefel params: sharded across ranks -----
            if is_stiefel:
                stiefel_params = [
                    p for p in group["params"]
                    if p.grad is not None and p.ndim == 3
                ]
                # Non-3D params in a stiefel group fall through to AdamW below
                adamw_params = [
                    p for p in group["params"]
                    if p.grad is not None and p.ndim != 3
                ]

                if stiefel_params:
                    # --- 1. Update moments on ALL ranks (cheap) ---
                    for p in stiefel_params:
                        state = self.state[p]
                        if len(state) == 0:
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)

                        reset_gap = group.get("poet_reset_gap", 0)
                        if reset_gap > 0 and self.global_step_counter > 0 and \
                           self.global_step_counter % reset_gap == 0:
                            state["step"] = 0
                            state["exp_avg"].zero_()
                            state["exp_avg_sq"].zero_()

                        state["step"] += 1

                    # --- 2. Shard the expensive Stiefel step across ranks ---
                    # Sort descending by numel for better load balancing
                    stiefel_params.sort(key=lambda x: x.numel(), reverse=True)

                    total = len(stiefel_params)
                    # Pad so total is divisible by world_size
                    padded = list(stiefel_params)
                    if total % world_size != 0:
                        pad_len = world_size - (total % world_size)
                        padded.extend([stiefel_params[-1]] * pad_len)

                    for base_i in range(0, len(padded), world_size):
                        local_idx = base_i + rank

                        # This rank computes the Stiefel step for its assigned param
                        if local_idx < total:
                            p = padded[local_idx]
                            state = self.state[p]
                            step = state["step"]
                            beta1_power = beta1 ** step
                            beta2_power = beta2 ** step

                            if qr_prob > 0 and random.random() < qr_prob:
                                p.data.copy_(_qr_retraction(p.data))

                            _stiefel_step(
                                p, p.grad,
                                state["exp_avg"], state["exp_avg_sq"],
                                beta1, beta2, beta1_power, beta2_power,
                                lr, eps, cayley_iters, correct_bias,
                                poet_scale=poet_scale,
                            )

                        # --- 3. Broadcast results from owner rank to all ---
                        if world_size > 1:
                            for j in range(world_size):
                                idx = base_i + j
                                if idx < total:
                                    owner_rank = j
                                    p_to_sync = padded[idx]
                                    dist.broadcast(p_to_sync.data, src=owner_rank)
                                    # Also sync moments so all ranks stay consistent
                                    st = self.state[p_to_sync]
                                    dist.broadcast(st["exp_avg"], src=owner_rank)
                                    dist.broadcast(st["exp_avg_sq"], src=owner_rank)

                # Fall-through: non-3D params in stiefel group -> AdamW
                for p in adamw_params:
                    self._adamw_step(p, group, beta1, beta2, eps, lr, wd, correct_bias)

            else:
                # ----- Standard AdamW for non-stiefel groups -----
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self._adamw_step(p, group, beta1, beta2, eps, lr, wd, correct_bias)

        self.global_step_counter += 1
        return loss

    def _adamw_step(self, p, group, beta1, beta2, eps, lr, wd, correct_bias):
        """Standard decoupled AdamW update."""
        grad = p.grad
        if grad is None:
            return
        if grad.is_sparse:
            raise RuntimeError("CayleyAdam does not support sparse gradients.")

        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        reset_gap = group.get("poet_reset_gap", 0)
        if reset_gap > 0 and self.global_step_counter > 0 and \
           self.global_step_counter % reset_gap == 0:
            state["step"] = 0
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()

        state["step"] += 1
        step = state["step"]

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        denom = exp_avg_sq.sqrt().add_(eps)

        step_size = lr
        if correct_bias:
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            step_size = step_size * math.sqrt(bc2) / bc1

        p.add_(exp_avg / denom, alpha=-step_size)

        if wd > 0.0:
            p.add_(p, alpha=-lr * wd)
