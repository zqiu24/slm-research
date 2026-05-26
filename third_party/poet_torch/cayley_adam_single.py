"""
Cayley Adam (single-GPU / no sharding) – AdamW on the Stiefel manifold via Cayley retraction.

This is the non-sharded version kept as a simple baseline for testing.
For the distributed-sharded version, see cayley_adam.py.
"""

import math
import random
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


def _cayley_loop(X, W, tan_vec, t, num_iters=5):
    Y = X + t * tan_vec
    for _ in range(num_iters):
        Y = X + t * torch.bmm(W, 0.5 * (X + Y))
    return Y


def _matrix_norm_one_batched(W):
    return W.abs().sum(dim=-2).max(dim=-1).values


def _qr_retraction(X):
    orig_dtype = X.dtype
    Q, R = torch.linalg.qr(X.float())
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    Q = Q * signs.unsqueeze(-2)
    return Q.to(orig_dtype)


class CayleyAdamSingle(Optimizer):
    """
    CayleyAdam without distributed sharding. Every rank computes the full
    Stiefel step for all parameters. Use this for single-GPU or debugging.
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

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CayleyAdamSingle does not support sparse gradients.")

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

                if is_stiefel and p.ndim == 3:
                    # ----- Stiefel manifold update -----
                    X = p.data
                    G = grad
                    m, v = state["exp_avg"], state["exp_avg_sq"]

                    m.mul_(beta1).add_(G, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(G, G, value=1.0 - beta2)

                    if correct_bias:
                        bc1 = 1.0 - beta1 ** step
                        bc2 = 1.0 - beta2 ** step
                        m_hat = m / bc1
                        v_hat = v / bc2
                    else:
                        m_hat, v_hat = m, v

                    adam_dir = m_hat / (v_hat.sqrt() + eps)

                    # Project to tangent space
                    XtD = torch.bmm(X.transpose(-1, -2), adam_dir)
                    sym_XtD = 0.5 * (XtD + XtD.transpose(-1, -2))
                    tangent = adam_dir - torch.bmm(X, sym_XtD)

                    # Skew-symmetric W
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

                    # Optional QR reorthogonalization
                    if qr_prob > 0 and random.random() < qr_prob:
                        p.data.copy_(_qr_retraction(p.data))

                else:
                    # ----- Standard AdamW -----
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

        self.global_step_counter += 1
        return loss
