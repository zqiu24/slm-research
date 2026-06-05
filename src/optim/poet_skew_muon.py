"""SkewMuon: Muon-on-Q optimizer for POET's skew generators (Stage 2).

Hybrid optimizer mirroring src/optim/_kimi_muon.Muon: params tagged "skew"
(POET's oft_R, shape (n_blocks, n_elems)) get the skew-Muon update; all other
params get AdamW. The skew update orthogonalizes the *per-block b x b skew
matrix* of the gradient (NOT the raw (n_blocks, n_elems) tensor — standard Muon
on that would mix blocks/entries), then rescales to a constant rotation angle.

Parameterization-agnostic: it only touches oft_R and its grad, so it works for
cayley or exp. Designed for the no-reset regime (merge_period=0): momentum
accumulates over the whole run. Single-process (DP-replicated, no sharding) like
muon_kimi; integration lives in src/optim/poet.py.
"""

from __future__ import annotations

import torch

from src.diag.skew_conditioning import block_size_from_nelems, skew_to_vec, vec_to_skew

# Quintic Newton-Schulz coefficients (same as src/optim/_kimi_muon).
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def orthogonalize_skew_blocks(Q: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """Batched quintic Newton-Schulz over a (num_blocks, b, b) batch (fp32).

    Returns ~orthogonal blocks (singular values driven toward ~uniform). The
    caller re-skew-symmetrizes the result. fp32 + batched (vs _kimi_muon's
    single-2D bf16 zeropower) for CPU-testability and skew precision.
    """
    norm = torch.linalg.matrix_norm(Q, ord="fro", dim=(-2, -1), keepdim=True)
    X = Q / (norm + 1e-7)
    for _ in range(ns_steps):
        A = X @ X.transpose(-2, -1)
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    return X


def orthogonalize_skew_direction(
    skew: torch.Tensor,
    method: str = "muon",
    ns_steps: int = 5,
    eps: float = 1e-7,
    reg: float = 1e-6,
) -> torch.Tensor:
    """Orthogonalize a batch of skew blocks (num_blocks, b, b) so the rotation
    planes turn by ~the same angle (docs/muon_orthogonalizing_optimizer_poet.md SS5).
    Result stays in so(b).

    method="muon" (default): Muon's quintic Newton-Schulz on the direction
    (orthogonalize_skew_blocks), then a 1/2(X - X^T) cleanup. NS preserves skew on a
    skew input, so the cleanup only removes ~1e-15 float dust. Democratizes the
    spectrum into a BAND around 1 (sigma ~ [0.7, 1.1]) in ~5 steps -- cheap; a band of
    roughly-equal angles may be all this needs. NOT sigma == 1.
    method="spectral": A_orth = A (-A^2)^{-1/2}, an ODD function of A so it stays
    skew exactly and drives ALL nonzero singular values to 1 (every plane the SAME
    angle). Exact but slow: needs ns_steps >= ~15 (coupled Newton-Schulz inverse-sqrt
    of C = -A^2 = A^T A), and still cannot fully equalize a near-rank-deficient block.
    """
    A = skew.float()
    if method == "muon":
        X = orthogonalize_skew_blocks(A, ns_steps)
        return 0.5 * (X - X.transpose(-2, -1))
    if method != "spectral":
        raise ValueError(
            f"orthogonalize_skew_direction method must be 'muon' or " f"'spectral', got {method!r}"
        )
    b = A.shape[-1]
    eye = torch.eye(b, dtype=A.dtype, device=A.device).expand_as(A)
    C = -(A @ A)  # = A^T A, symmetric PSD; eigenvalues are A's squared sing. values
    # Normalize so C's spectrum sits in (0, 1] (convergence needs it in (0, 2)),
    # plus a tiny ridge so an odd-dim null direction stays finite (A kills it anyway).
    s = torch.linalg.matrix_norm(C, ord="fro", dim=(-2, -1), keepdim=True) + eps
    Cn = C / s + reg * eye
    Y, Z = Cn, eye.clone()
    for _ in range(ns_steps):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T  # -> Cn^{1/2}
        Z = T @ Z  # -> Cn^{-1/2}
    inv_sqrt_C = Z / s.sqrt()  # Cn^{-1/2} = sqrt(s) * C^{-1/2}  =>  C^{-1/2} = Z/sqrt(s)
    A_orth = A @ inv_sqrt_C
    return 0.5 * (A_orth - A_orth.transpose(-2, -1))  # clean up float residue


def muon_update_spectral_stats(skew_grad: torch.Tensor, ns_steps: int = 5) -> dict:
    """Spectral stats of the SkewMuon UPDATE direction for a per-block skew
    gradient ``skew_grad`` ((num_blocks, b, b)): NS-orthogonalize then re-skew to
    so(b) — exactly the transform ``SkewMuon.step`` applies before the constant-
    angle rescale (which is scale-only and leaves the spectrum's *shape*, hence the
    condition number, unchanged) — then ``block_spectral_stats``.

    The condition number should be ~1 (NS flattens the spectrum) even when
    ``skew_grad`` is heavy-tailed. This is what makes Muon's step well-conditioned
    regardless of how ill-conditioned ∂f/∂Q is, and it is the metric the raw-grad
    conditioning probe cannot show (it reads the gradient *before* the optimizer).
    Doubles as an ``ns_steps`` health check: cond drifting well above 1 on real
    gradients means Newton-Schulz is under-iterating.
    """
    from src.diag.skew_conditioning import block_spectral_stats

    X = orthogonalize_skew_blocks(skew_grad.float(), ns_steps)
    X = (X - X.transpose(-2, -1)) / 2  # re-skew to so(b), as in SkewMuon.step
    return block_spectral_stats(X)


class SkewMuon(torch.optim.Optimizer):
    def __init__(
        self,
        skew_params=None,
        adamw_params=None,
        theta: float = 0.1,
        ns_steps: int = 5,
        momentum: float = 0.95,
        nesterov: bool = True,
        adamw_lr: float = 1e-3,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        skew_params = list(skew_params) if skew_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        defaults = dict(
            theta=theta,
            ns_steps=ns_steps,
            momentum=momentum,
            nesterov=nesterov,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(skew_params + adamw_params, defaults)
        for p in skew_params:
            assert p.ndim == 2, p.shape  # (n_blocks, n_elems)
            self.state[p]["use_skew"] = True
        for p in adamw_params:
            self.state[p]["use_skew"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # ---- skew-Muon branch (oft_R) ----
            for p in (p for p in group["params"] if self.state[p]["use_skew"]):
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                b = block_size_from_nelems(g.shape[-1])
                Q = vec_to_skew(g.float(), b)
                X = orthogonalize_skew_blocks(Q, group["ns_steps"])
                X = (X - X.transpose(-2, -1)) / 2  # re-skew to stay in so(b)
                fro = torch.linalg.matrix_norm(X, ord="fro", dim=(-2, -1), keepdim=True)
                step_skew = group["theta"] * X / (fro + 1e-8)
                step_vec = skew_to_vec(step_skew, b).to(p.dtype)
                p.add_(step_vec, alpha=-1.0)

            # ---- AdamW branch (everything else) ----
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            lr = group["adamw_lr"]
            wd = group["adamw_wd"]
            for p in (p for p in group["params"] if not self.state[p]["use_skew"]):
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                m1, m2 = state["moment1"], state["moment2"]
                m1.lerp_(g, 1 - beta1)
                m2.lerp_(g.square(), 1 - beta2)
                update = m1 / (eps + m2.sqrt())
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                scale = bc1 / bc2**0.5
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr / scale)

        return loss
