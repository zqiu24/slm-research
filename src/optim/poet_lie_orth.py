"""LieOrthMomentum: Muon-like orthogonalizing optimizer for POET's skew generators
(q_optimizer=lie_ortho). Standalone sibling of
src.optim.poet_lie_momentum.LieAlgebraMomentum.

Same Lie-algebra first-moment momentum on oft_R (one or more param groups tagged
use_skew=True) and the same AdamW branch on everything else, but instead of
RMS-scaling the direction it ORTHOGONALIZES it (orthogonalize_skew_direction) so the
rotation planes turn by ~the same angle. Default method='muon' (Muon's quintic NS, a
band around 1, ~5 steps); method='spectral' is the exact A(-A^2)^{-1/2} variant
(sigma=1, ~20 steps). See docs/muon_orthogonalizing_optimizer_poet.md.

First-moment-only by default: a second moment is partially undone by orthogonalization
(docs SS4). State buffers are named lie_m / lie_v so the merge patch's _zero_moments
cannot reset them -- momentum PERSISTS across the per-step fold. Single-process /
DP-replicated (no sharded distributed optimizer); integration lives in
src/optim/poet.py, which reuses _split_poet_lie_params / _build_lie_param_groups.
"""

from __future__ import annotations

import torch

from src.diag.skew_conditioning import block_size_from_nelems, skew_to_vec, vec_to_skew
from src.optim.poet_skew_muon import orthogonalize_skew_direction


class LieOrthMomentum(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        b1: float = 0.9,
        b2: float = 0.95,
        eps: float = 1e-8,
        v_mode: str = "elementwise",
        alternating: bool = False,
        alternate_every: int = 1,
        ortho_c: float = 0.01,
        ortho_method: str = "muon",
        ortho_ns_steps: int = 5,
        ortho_use_second_moment: bool = False,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if v_mode not in ("scalar", "elementwise"):
            raise ValueError(f"v_mode must be 'scalar' or 'elementwise', got {v_mode!r}")
        if ortho_method not in ("muon", "spectral"):
            raise ValueError(f"ortho_method must be 'muon' or 'spectral', got {ortho_method!r}")
        # Alternating single-sided update: write only one side's oft_R per step (out on
        # even, in on odd), accumulating momentum on BOTH sides.
        self.alternating = bool(alternating)
        self.alternate_every = max(1, int(alternate_every))
        self._alt_step = 0
        # Orthogonalizing transform (docs/muon_orthogonalizing_optimizer_poet.md):
        # realized per-plane angle = lr * ortho_c (a band under 'muon', exact under
        # 'spectral'). First-moment-only unless ortho_use_second_moment.
        self.ortho_c = float(ortho_c)
        self.ortho_method = ortho_method
        self.ortho_ns_steps = int(ortho_ns_steps)
        self.ortho_use_second_moment = bool(ortho_use_second_moment)
        defaults = dict(
            lr=0.0,
            use_skew=False,
            side=None,
            b1=b1,
            b2=b2,
            eps=eps,
            v_mode=v_mode,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active = None
        if self.alternating:
            active = "out" if (self._alt_step // self.alternate_every) % 2 == 0 else "in"

        for group in self.param_groups:
            lr = group["lr"]
            if group["use_skew"]:
                side = group["side"]
                b1, b2, eps, v_mode = group["b1"], group["b2"], group["eps"], group["v_mode"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    g = g.float()
                    st = self.state[p]
                    if "lie_m" not in st:
                        st["lie_m"] = torch.zeros_like(g)
                        if v_mode == "scalar":
                            st["lie_v"] = torch.zeros(g.shape[0], 1, dtype=g.dtype, device=g.device)
                        else:
                            st["lie_v"] = torch.zeros_like(g)
                    m, v = st["lie_m"], st["lie_v"]
                    # Momentum accumulates on BOTH sides every step ...
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    if v_mode == "scalar":
                        v.mul_(b2).add_(2.0 * (g * g).sum(dim=-1, keepdim=True), alpha=1 - b2)
                    else:
                        v.mul_(b2).add_(g * g, alpha=1 - b2)
                    # ... but only the ACTIVE side's oft_R is written.
                    if self.alternating and side != active:
                        continue
                    # Orthogonalize the DIRECTION (per b x b block) so the planes turn
                    # by ~the same angle. Scale by ortho_c DIRECTLY (the spectrum is
                    # ~democratized, so no sqrt(d)/||A||, docs SS3). First-moment-only
                    # by default. Realized per-plane angle = lr * ortho_c.
                    A_dir = -m / (v.sqrt() + eps) if self.ortho_use_second_moment else -m
                    bsz = block_size_from_nelems(A_dir.shape[1])
                    X = orthogonalize_skew_direction(
                        vec_to_skew(A_dir, bsz),
                        method=self.ortho_method,
                        ns_steps=self.ortho_ns_steps,
                    )
                    gen = skew_to_vec(self.ortho_c * X, bsz)  # (n_blocks, n_elems)
                    p.add_(gen.to(p.dtype), alpha=lr)
            else:
                beta1, beta2 = group["adamw_betas"]
                aeps, wd = group["adamw_eps"], group["adamw_wd"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["moment1"] = torch.zeros_like(g)
                        st["moment2"] = torch.zeros_like(g)
                    st["step"] += 1
                    m1, m2 = st["moment1"], st["moment2"]
                    m1.lerp_(g, 1 - beta1)
                    m2.lerp_(g.square(), 1 - beta2)
                    update = m1 / (aeps + m2.sqrt())
                    bc1 = 1 - beta1 ** st["step"]
                    bc2 = 1 - beta2 ** st["step"]
                    scale = bc1 / bc2**0.5
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr / scale)
        if self.alternating:
            self._alt_step += 1
        return loss
