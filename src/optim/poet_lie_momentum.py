"""LieAlgebraMomentum: Pion Lie-algebra first+second-moment momentum on POET's
skew generators (q_optimizer=lie_algebra). Increment 1 of the POET-X x Pion
pipeline (docs/poetx_pion_pipeline.md §2-3): import Pion's Lie-algebra momentum
while keeping POET's block-skew oft_R + merge machinery.

Shaped like src/optim/poet_skew_muon.SkewMuon: skew branch on oft_R (one or more
param groups tagged use_skew=True), AdamW branch on everything else. The skew
update is computed in VEC-space (upper-triangular, like SkewMuon's
momentum_buffer) — provably identical to the paper's skew-space A = -M/(sqrt(v)+eps)
with M = vec_to_skew(m); scalar-v uses ||G||_F^2 = 2*sum(g_vec^2) (full-matrix
Frobenius). At merge_period=1 the ambient oft_R.grad equals the skew tangent
gradient to O(angle^2), so no new gradient plumbing is needed.

State buffers are named lie_m / lie_v (NOT exp_avg/exp_avg_sq) so the merge
patch's _zero_moments cannot reset them — Lie momentum PERSISTS across the
per-step fold. Single-process / DP-replicated (no sharded distributed optimizer),
like the muon path; integration lives in src/optim/poet.py.
"""

from __future__ import annotations

import torch


def _split_poet_lie_params(model_chunks):
    """oft_R_in -> in-side, oft_R_out -> out-side, everything else -> adamw.

    The in/out split (vs the muon path's lumped oft_R) is what lets the optimizer
    update one side per step for the alternating single-sided update (§6).
    """
    skew_in, skew_out, adamw = [], [], []
    for mc in model_chunks:
        for name, p in mc.named_parameters():
            if not p.requires_grad:
                continue
            if "oft_R_in" in name:
                skew_in.append(p)
            elif "oft_R_out" in name:
                skew_out.append(p)
            else:
                adamw.append(p)
    return skew_in, skew_out, adamw


def _build_lie_param_groups(skew_in, skew_out, adamw_params, lr, min_lr, scale):
    """Side-tagged param groups carrying lr/max_lr/min_lr so Megatron's scheduler
    decays group['lr'] (skew sides scaled by poet_scale, like the vanilla path
    scales oft_R). The two skew groups carry side='in'/'out' for the alternating
    single-sided update (§6); empty sides are dropped."""
    groups = []
    for side, ps in (("in", list(skew_in)), ("out", list(skew_out))):
        if ps:
            groups.append(
                dict(
                    params=ps,
                    use_skew=True,
                    side=side,
                    lr=lr * scale,
                    max_lr=lr * scale,
                    min_lr=min_lr * scale,
                )
            )
    if adamw_params:
        groups.append(
            dict(
                params=list(adamw_params),
                use_skew=False,
                side=None,
                lr=lr,
                max_lr=lr,
                min_lr=min_lr,
            )
        )
    return groups


class LieAlgebraMomentum(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        b1: float = 0.9,
        b2: float = 0.95,
        eps: float = 1e-8,
        v_mode: str = "scalar",
        alternating: bool = False,
        alternate_every: int = 1,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if v_mode not in ("scalar", "elementwise"):
            raise ValueError(f"v_mode must be 'scalar' or 'elementwise', got {v_mode!r}")
        # Alternating single-sided update (§6): write only one side's oft_R per
        # step (out on even, in on odd), accumulating momentum on BOTH sides.
        self.alternating = bool(alternating)
        self.alternate_every = max(1, int(alternate_every))
        self._alt_step = 0
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

        # Active side this step (alternating §6): out on even, in on odd
        # (Eq. 8, ψ=0 even→out). None = write both sides (non-alternating).
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
                    # Momentum accumulates on BOTH sides every step (paper App. D.1) ...
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    if v_mode == "scalar":
                        # ||vec_to_skew(g)||_F^2 = 2 * sum(g^2) over the upper-tri vec
                        v.mul_(b2).add_(2.0 * (g * g).sum(dim=-1, keepdim=True), alpha=1 - b2)
                    else:
                        v.mul_(b2).add_(g * g, alpha=1 - b2)
                    # ... but only the ACTIVE side's oft_R is written (the inactive
                    # side stays 0 -> identity rotation -> no-op fold).
                    if self.alternating and side != active:
                        continue
                    A = -m / (v.sqrt() + eps)
                    p.add_(A.to(p.dtype), alpha=lr)  # p born at 0 -> p = lr*A
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
