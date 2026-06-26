"""LieOrthUpdateRMSMomentum: orthogonalized POET Lie updates with per-layer
update-RMS angle targeting.

This is a standalone sibling of :mod:`src.optim.poet_lie_orth`. It keeps the
successful alternating, Muon-like orthogonalized Lie direction, but replaces the
fixed angle ``lr * lie_ortho_c`` with

    theta = min(lr * update_rms / RMS(weight), max_angle)

for the active side's owner layer. ``theta`` already includes the scheduled
learning rate, so skew updates are scattered with alpha=1.0.
"""

from __future__ import annotations

import math

import torch

from src.diag.skew_conditioning import block_size_from_nelems, skew_to_vec, vec_to_skew
from src.optim.poet_skew_muon import orthogonalize_skew_direction


def rms(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return fp32 root-mean-square, clamped away from zero."""
    return t.float().pow(2).mean().sqrt().clamp_min(eps)


def compute_update_rms_angle(
    *,
    lr: float,
    update_rms: float,
    denom: float | torch.Tensor,
    max_angle: float,
) -> torch.Tensor:
    """Compute ``min(lr * update_rms / denom, max_angle)`` as an fp32 tensor."""
    device = denom.device if isinstance(denom, torch.Tensor) else None
    theta = torch.as_tensor(lr, dtype=torch.float32, device=device)
    denom_t = torch.as_tensor(denom, dtype=torch.float32, device=theta.device).clamp_min(1e-12)
    theta = theta * float(update_rms) / denom_t
    return theta.clamp_max(float(max_angle))


class LieOrthUpdateRMSMomentum(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        b1: float = 0.9,
        b2: float = 0.95,
        eps: float = 1e-8,
        v_mode: str = "elementwise",
        alternating: bool = True,
        alternate_every: int = 1,
        update_rms: float = 0.2,
        max_angle: float = 0.024,
        side_gamma: float = 0.0,
        rms_mode: str = "weight",
        ortho_method: str = "muon",
        ortho_ns_steps: int = 5,
        ortho_use_second_moment: bool = False,
        nesterov: bool = False,
        distributed: bool = False,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        dp_group=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if v_mode not in ("scalar", "elementwise"):
            raise ValueError(f"v_mode must be 'scalar' or 'elementwise', got {v_mode!r}")
        if ortho_method not in ("muon", "spectral"):
            raise ValueError(f"ortho_method must be 'muon' or 'spectral', got {ortho_method!r}")
        if rms_mode not in ("weight", "direction"):
            raise ValueError(f"rms_mode must be 'weight' or 'direction', got {rms_mode!r}")
        if rms_mode == "direction":
            raise ValueError(
                "lie_ortho_update_rms currently supports rms_mode='weight' only; "
                "rms_mode='direction' is reserved for an exact diagnostic implementation."
            )
        if not alternating:
            raise ValueError("LieOrthUpdateRMSMomentum requires alternating=True")
        if update_rms <= 0:
            raise ValueError(f"update_rms must be positive, got {update_rms!r}")
        if max_angle <= 0:
            raise ValueError(f"max_angle must be positive, got {max_angle!r}")

        self.alternating = True
        self.alternate_every = max(1, int(alternate_every))
        self._alt_step = 0
        self.update_rms = float(update_rms)
        self.max_angle = float(max_angle)
        # Per-side angle redistribution exponent (in/out asymmetry). The active
        # side's target rho is scaled by (d_side / sqrt(d_out * d_in)) ** side_gamma,
        # where d_out/d_in are the owner weight's fan-out/fan-in. The geometric-mean
        # reference makes this a PURE redistribution between R_out and R_in
        # (factor_out * factor_in == 1), so the per-layer average angle — and hence
        # the overall update strength set by ``update_rms`` — is unchanged. gamma=0
        # (default) => factor 1 on every side => the symmetric champion. gamma>0
        # rotates the larger-dim side more (e.g. fc1's d_out=4*d_in), gamma<0 less.
        self.side_gamma = float(side_gamma)
        self.rms_mode = rms_mode
        self.ortho_method = ortho_method
        self.ortho_ns_steps = int(ortho_ns_steps)
        self.ortho_use_second_moment = bool(ortho_use_second_moment)
        self.nesterov = bool(nesterov)
        self.distributed = bool(distributed)
        self._dp_world_size = int(dp_world_size)
        self._dp_rank = int(dp_rank)
        self.dp_group = dp_group
        self.last_update_rms_angles: dict[int, torch.Tensor] = {}
        self.last_update_rms_stats: dict[str, torch.Tensor] = {}
        defaults = dict(
            lr=0.0,
            use_skew=False,
            side=None,
            weight=None,
            block_size=None,
            b1=b1,
            b2=b2,
            eps=eps,
            v_mode=v_mode,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)
        self._validate_skew_groups()

    def _validate_skew_groups(self) -> None:
        for group in self.param_groups:
            if not group["use_skew"]:
                continue
            if group.get("side") not in ("in", "out"):
                raise ValueError("lie_ortho_update_rms skew groups must set side='in' or 'out'")
            if len(group["params"]) != 1:
                raise ValueError("lie_ortho_update_rms requires one skew param per group")
            if group.get("weight") is None:
                raise ValueError("lie_ortho_update_rms skew groups must carry owner weight")
            if group.get("block_size") is None:
                raise ValueError("lie_ortho_update_rms skew groups must carry block_size")

    def _active_side(self):
        from poet_torch.alt_state import active_side

        return active_side(self.alternate_every)

    def _lie_m_update(self, active):
        """Update Lie momentum for every skew param, including the inactive side."""
        del active  # Fresh both-side momentum is mandatory for this optimizer.
        for group in self.param_groups:
            if not group["use_skew"]:
                continue
            b1, b2, v_mode = group["b1"], group["b2"], group["v_mode"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                g = g.float()
                st = self.state[p]
                if "lie_m" not in st:
                    st["lie_m"] = torch.zeros_like(g)
                    if self.ortho_use_second_moment:
                        if v_mode == "scalar":
                            st["lie_v"] = torch.zeros(g.shape[0], 1, dtype=g.dtype, device=g.device)
                        else:
                            st["lie_v"] = torch.zeros_like(g)
                st["lie_m"].mul_(b1).add_(g, alpha=1 - b1)
                if self.ortho_use_second_moment:
                    v = st["lie_v"]
                    if v_mode == "scalar":
                        v.mul_(b2).add_(2.0 * (g * g).sum(dim=-1, keepdim=True), alpha=1 - b2)
                    else:
                        v.mul_(b2).add_(g * g, alpha=1 - b2)

    def _side_factor(self, group) -> float:
        """Per-side angle multiplier (d_side / sqrt(d_out*d_in)) ** side_gamma.

        Returns 1.0 when side_gamma == 0 or the owner weight is square, so the
        symmetric (champion) path is bit-for-bit unchanged.
        """
        if self.side_gamma == 0.0:
            return 1.0
        w = group["weight"]
        d_out, d_in = int(w.shape[0]), int(w.shape[1])
        d_side = d_out if group["side"] == "out" else d_in
        d_ref = math.sqrt(max(d_out * d_in, 1))
        return (d_side / d_ref) ** self.side_gamma

    def _iter_skew_params(self):
        for group in self.param_groups:
            if not group["use_skew"]:
                continue
            for p in group["params"]:
                if p.grad is not None:
                    yield p, group

    def _skew_update_buffer(self, dp_rank, dp_world, active):
        items = list(self._iter_skew_params())
        slices, total = [], 0
        for p, _group in items:
            slices.append((total, p.numel(), p))
            total += p.numel()
        self.last_update_rms_angles = {}
        self.last_update_rms_stats = {}
        if total == 0:
            return torch.zeros(0), []

        device = items[0][0].grad.device
        buf = torch.zeros(total, dtype=torch.float32, device=device)
        buckets = {}
        theta_records: list[torch.Tensor] = []
        denom_records: list[torch.Tensor] = []
        clamped_records: list[torch.Tensor] = []
        implied_rho_records: list[torch.Tensor] = []
        for i, (p, group) in enumerate(items):
            if (i % dp_world) != dp_rank:
                continue
            if active is not None and group["side"] != active:
                continue
            st = self.state[p]
            m = st["lie_m"]
            if self.nesterov:
                base = p.grad.float().mul(1.0 - group["b1"]).add_(m, alpha=group["b1"])
            else:
                base = m
            a_dir = (
                -base / (st["lie_v"].sqrt() + group["eps"])
                if self.ortho_use_second_moment
                else -base
            )
            bsz = int(group.get("block_size") or block_size_from_nelems(a_dir.shape[1]))
            buckets.setdefault(bsz, []).append((i, a_dir, group))

        for bsz, bucket in buckets.items():
            a_cat = torch.cat([a for _, a, _ in bucket], dim=0)
            x_orth = orthogonalize_skew_direction(
                vec_to_skew(a_cat, bsz),
                method=self.ortho_method,
                ns_steps=self.ortho_ns_steps,
            )
            row_off = 0
            for i, a_dir, group in bucket:
                nb = a_dir.shape[0]
                p = items[i][0]
                denom = rms(group["weight"].detach())
                eff_update_rms = self.update_rms * self._side_factor(group)
                raw_theta = (
                    torch.as_tensor(group["lr"], dtype=torch.float32, device=denom.device)
                    * eff_update_rms
                    / denom.clamp_min(1e-12)
                )
                theta = compute_update_rms_angle(
                    lr=group["lr"],
                    update_rms=eff_update_rms,
                    denom=denom,
                    max_angle=self.max_angle,
                )
                x_part = x_orth[row_off : row_off + nb]
                gen = skew_to_vec(theta.to(x_part.device) * x_part, bsz)
                off, n = slices[i][0], slices[i][1]
                buf[off : off + n] = gen.reshape(-1)
                self.last_update_rms_angles[id(p)] = theta.detach()
                theta_records.append(theta.detach().float())
                denom_records.append(denom.detach().float())
                clamped_records.append((raw_theta > self.max_angle).detach().float())
                implied_rho_records.append(
                    (theta.detach().float() * denom.detach().float())
                    / max(float(group["lr"]), 1e-12)
                )
                row_off += nb

        self._set_update_rms_stats(
            theta_records, denom_records, clamped_records, implied_rho_records
        )
        return buf, slices

    def _set_update_rms_stats(
        self, theta_records, denom_records, clamped_records, implied_rho_records
    ):
        if not theta_records:
            self.last_update_rms_stats = {}
            return
        theta = torch.stack(theta_records).float()
        denom = torch.stack(denom_records).float()
        clamped = torch.stack(clamped_records).float()
        implied = torch.stack(implied_rho_records).float()
        self.last_update_rms_stats = {
            "poet_update_rms/theta_mean": theta.mean().detach(),
            "poet_update_rms/theta_max": theta.max().detach(),
            "poet_update_rms/theta_p95": torch.quantile(theta, 0.95).detach(),
            "poet_update_rms/weight_rms_mean": denom.mean().detach(),
            "poet_update_rms/clamp_fraction": clamped.mean().detach(),
            "poet_update_rms/implied_rho_mean": implied.mean().detach(),
        }

    def _apply_skew_update_buffer(self, buf, slices):
        for off, n, p in slices:
            p.add_(buf[off : off + n].view_as(p).to(p.dtype), alpha=1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active = self._active_side()
        self._lie_m_update(active)
        buf, slices = self._skew_update_buffer(self._dp_rank, self._dp_world_size, active)
        if self.distributed and self._dp_world_size > 1 and buf.numel() > 0:
            import torch.distributed as dist

            dist.all_reduce(buf, group=self.dp_group)
        self._apply_skew_update_buffer(buf, slices)

        for group in self.param_groups:
            if group["use_skew"]:
                continue
            beta1, beta2 = group["adamw_betas"]
            aeps, wd = group["adamw_eps"], group["adamw_wd"]
            lr = group["lr"]
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

        self._alt_step += 1
        return loss
