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
        true_single_side: bool = False,
        ortho_c: float = 0.01,
        ortho_method: str = "muon",
        ortho_ns_steps: int = 5,
        ortho_use_second_moment: bool = False,
        angle_dim_exp: float = 0.0,
        angle_dim_ref: float | None = None,
        nesterov: bool = False,
        distributed: bool = False,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        dp_group=None,
        decorrelate_sides: bool = False,
        decorrelate_mode: str = "in_off_out",
        decorrelate_lambda: float = 1.0,
        decorrelate_renorm: bool = False,
        decorrelate_cos_threshold: float = 0.0,
        layer_pairs=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if v_mode not in ("scalar", "elementwise"):
            raise ValueError(f"v_mode must be 'scalar' or 'elementwise', got {v_mode!r}")
        if ortho_method not in ("muon", "spectral"):
            raise ValueError(f"ortho_method must be 'muon' or 'spectral', got {ortho_method!r}")
        if decorrelate_mode not in ("in_off_out", "out_off_in", "symmetric"):
            raise ValueError(
                "decorrelate_mode must be 'in_off_out' | 'out_off_in' | 'symmetric', "
                f"got {decorrelate_mode!r}"
            )
        # Alternating single-sided update: write only one side's oft_R per step (out on
        # even, in on odd), accumulating momentum on BOTH sides.
        self.alternating = bool(alternating)
        self.alternate_every = max(1, int(alternate_every))
        self._alt_step = 0
        # true_single_side: the dedicated AlternatingPOETXLinear path. Active side
        # comes from poet_torch.alt_state (shared with the layer + merge), and the
        # frozen side's momentum does NOT advance (its grad is zeros from the layer).
        self.true_single_side = bool(true_single_side)
        # Orthogonalizing transform (docs/muon_orthogonalizing_optimizer_poet.md):
        # realized per-plane angle = lr * ortho_c (a band under 'muon', exact under
        # 'spectral'). First-moment-only unless ortho_use_second_moment.
        self.ortho_c = float(ortho_c)
        self.ortho_method = ortho_method
        self.ortho_ns_steps = int(ortho_ns_steps)
        self.ortho_use_second_moment = bool(ortho_use_second_moment)
        # Per-block dimension-dependent angle: scale each block's per-plane angle by
        # (block_size / angle_dim_ref) ** angle_dim_exp. exp=0 (default) => flat angle =
        # the champion. ref is set by poet.py to hidden_size, so the factor is the block's
        # dimensionless ratio to hidden (scale-stable across model sizes). Lets the
        # larger-dim side (e.g. fc-out 1536) rotate more (exp>0) or less (exp<0) than the
        # smaller side — the in/out asymmetry sweep (POET_dev arm K).
        self.angle_dim_exp = float(angle_dim_exp)
        self.angle_dim_ref = float(angle_dim_ref) if angle_dim_ref else None
        # Nesterov look-ahead (Muon-style): orthogonalize the look-ahead direction
        # (1-b1)*g + b1*m instead of the bare first moment m. lie_m is still the same
        # EMA (m = b1*m + (1-b1)*g), so this matches modern Muon's
        # `update = grad.lerp(momentum, beta)`. Skew/rotation branch only (the AdamW
        # branch is untouched). Composes with ortho_use_second_moment (look-ahead is
        # divided by sqrt(lie_v)) though the default is first-moment-only.
        self.nesterov = bool(nesterov)
        # DP-sharded orthogonalization (off by default = replicated path). When on and
        # dp_world_size > 1, each rank orthogonalizes only its round-robin slice of
        # oft_R, then one all_reduce(SUM) of the zero-padded update deltas re-syncs.
        self.distributed = bool(distributed)
        self._dp_world_size = int(dp_world_size)
        self._dp_rank = int(dp_rank)
        self.dp_group = dp_group
        # Cross-side decorrelation (ANALYSIS §17.6 probe): project each layer's in/out
        # generator off the other's weight-space direction so cos(D_out, D_in) -> 0,
        # isolating the inter-side gauge-redundancy channel from per-side conditioning.
        # Meant for the SIMULTANEOUS config (alternating=False) where both sides write
        # each step; in alternating mode only one side is non-zero per step so it is a
        # near no-op. layer_pairs = [(out_param, in_param, weight, bsz_out, bsz_in), ...].
        self.decorrelate_sides = bool(decorrelate_sides)
        self.decorrelate_mode = decorrelate_mode
        # Alternating-path overlap-control knobs (no effect on the simultaneous path):
        # lambda = partial-projection fraction (1.0 = full decorrelation); renorm =
        # rescale the projected active generator back to its pre-projection ||D|| so the
        # realized weight-space movement is preserved (direction-only change); cos_threshold
        # = module-selective gate, only decorrelate layers with |cos(D_out,D_in)| >= it.
        self.decorrelate_lambda = float(decorrelate_lambda)
        self.decorrelate_renorm = bool(decorrelate_renorm)
        self.decorrelate_cos_threshold = float(decorrelate_cos_threshold)
        self._decorr_pairs = list(layer_pairs) if layer_pairs else []
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

    def _active_side(self):
        # The dedicated true-single-side path AND the integrated both-momenta
        # alternating path both read the SAME shared signal (alt_state, seeded
        # once per training step by the poet_merge_step wrapper) so the optimizer's
        # WRITE side equals the merge's FOLD side within a step. Quality-neutral
        # for the both-sides-merge case; REQUIRED for active-only-merge correctness.
        if self.true_single_side or self.alternating:
            from poet_torch.alt_state import active_side

            return active_side(self.alternate_every)
        return None

    def _lie_m_update(self, active):
        """Phase (a): update lie_m (+ lie_v if used) for ALL skew params. Cheap; run on
        every rank so the momentum buffers stay in sync (grads are DP-identical)."""
        for group in self.param_groups:
            if not group["use_skew"]:
                continue
            if self.true_single_side and active is not None and group["side"] != active:
                continue  # true single-side: frozen side's momentum must not advance
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

    def _iter_skew_params(self):
        """Deterministic, rank-identical ordering of skew params that have a grad."""
        for group in self.param_groups:
            if not group["use_skew"]:
                continue
            for p in group["params"]:
                if p.grad is not None:
                    yield p, group

    def _skew_update_buffer(self, dp_rank, dp_world, active):
        """Phase (b), PURE (reads lie_m/lie_v, no mutation): compute the generator
        gen = ortho_c*orthogonalize(-dir) for the round-robin-OWNED skew params
        (i % dp_world == dp_rank), zeros for the rest, packed into one flat fp32 buffer.
        lr is NOT folded in here -- it is applied at scatter (alpha=lr) so the cast to
        bf16 happens in the same order as the inline path (gen.to(dtype) THEN *lr),
        making the buffer/sharded path bit-identical to the old inline update.
        Returns (flat_buffer, slices=[(offset, numel, param, lr), ...])."""
        items = list(self._iter_skew_params())
        slices, total = [], 0
        for p, group in items:
            slices.append((total, p.numel(), p, group["lr"]))
            total += p.numel()
        if total == 0:
            return torch.zeros(0), []
        device = items[0][0].grad.device
        buf = torch.zeros(total, dtype=torch.float32, device=device)
        buckets = {}
        for i, (p, group) in enumerate(items):
            if (i % dp_world) != dp_rank:
                continue  # not this rank's block -> leave zeros (exact under all_reduce SUM)
            if active is not None and group["side"] != active:
                continue  # inactive side -> no rotation written this step
            st = self.state[p]
            m = st["lie_m"]
            if self.nesterov:
                # Muon-style look-ahead: base = (1-b1)*g + b1*m  (= grad.lerp(m, b1)).
                # p.grad is fp32 (main_grad accumulator) and DP-identical; .mul() makes
                # a fresh tensor so neither p.grad nor lie_m is mutated here.
                base = p.grad.float().mul(1.0 - group["b1"]).add_(m, alpha=group["b1"])
            else:
                base = m
            A_dir = (
                -base / (st["lie_v"].sqrt() + group["eps"])
                if self.ortho_use_second_moment
                else -base
            )
            bsz = block_size_from_nelems(A_dir.shape[1])
            buckets.setdefault(bsz, []).append((i, A_dir))

        for bsz, bucket in buckets.items():
            A_cat = torch.cat([a for _, a in bucket], dim=0)
            X = orthogonalize_skew_direction(
                vec_to_skew(A_cat, bsz),
                method=self.ortho_method,
                ns_steps=self.ortho_ns_steps,
            )
            ang = self.ortho_c
            if self.angle_dim_exp != 0.0 and self.angle_dim_ref:
                ang = ang * (bsz / self.angle_dim_ref) ** self.angle_dim_exp
            gen = skew_to_vec(ang * X, bsz)  # (n_blocks, n_elems) float; lr at scatter
            row_off = 0
            for i, A_dir in bucket:
                nb = A_dir.shape[0]
                off, n = slices[i][0], slices[i][1]
                buf[off : off + n] = gen[row_off : row_off + nb].reshape(-1)
                row_off += nb
        return buf, slices

    def _decorrelate_buf(self, buf, slices):
        """Cross-side Gram-Schmidt on the assembled generators (post all-reduce, so both
        sides are complete on every rank): project each layer's in/out generator off the
        other's weight-space direction, driving cos(D_out, D_in) -> 0 while leaving each
        side's Muon whitening ~intact. Exact identity (in the block-contiguous frame the
        generators live in, with W = POETLinear.weight):

            <D_out, D_in>_F = <block_skew(W^T D_out), A_in>  (and the symmetric W D_in^T form),

        so removing the block_skew(W^T D_out) component from A_in zeros the overlap. One
        extra block-matmul per side; no backward, no W grad. Mutates buf in place."""
        if not self._decorr_pairs:
            return
        from src.diag.poet_coordination_diag import block_diag_skew, side_directions

        off_by_id = {id(p): (off, n) for off, n, p, _lr in slices}
        eps = 1e-12
        sym = self.decorrelate_mode == "symmetric"
        matched = 0
        for out_p, in_p, w, bsz_out, bsz_in in self._decorr_pairs:
            so, si = off_by_id.get(id(out_p)), off_by_id.get(id(in_p))
            if so is None or si is None:
                continue
            matched += 1
            (oo, no), (oi, ni) = so, si
            out_vec = buf[oo : oo + no].view(out_p.shape[0], -1)
            in_vec = buf[oi : oi + ni].view(in_p.shape[0], -1)
            # One side may be all-zeros this step (e.g. alternating) -> nothing to project.
            if float(out_vec.abs().sum()) == 0.0 or float(in_vec.abs().sum()) == 0.0:
                continue
            A_out = vec_to_skew(out_vec, bsz_out)
            A_in = vec_to_skew(in_vec, bsz_in)
            W = w.detach().to(torch.float32)
            d_out, d_in = side_directions(A_out, A_in, W)  # from the ORIGINAL generators
            if self.decorrelate_mode in ("in_off_out", "symmetric"):
                g = block_diag_skew(W.transpose(-2, -1) @ d_out, bsz_in)
                c = (A_in.flatten() @ g.flatten()) / (g.flatten() @ g.flatten()).clamp_min(eps)
                A_in = A_in - (0.5 * c if sym else c) * g
                buf[oi : oi + ni] = skew_to_vec(A_in, bsz_in).reshape(-1)
            if self.decorrelate_mode in ("out_off_in", "symmetric"):
                g = block_diag_skew(d_in @ W.transpose(-2, -1), bsz_out)
                c = (A_out.flatten() @ g.flatten()) / (g.flatten() @ g.flatten()).clamp_min(eps)
                A_out = A_out - (0.5 * c if sym else c) * g
                buf[oo : oo + no] = skew_to_vec(A_out, bsz_out).reshape(-1)
        # Loud guard: decorrelation requested but NO pair's params matched the optimizer's
        # slices (e.g. model-vs-master param identity mismatch) => silent no-op. Warn once
        # rather than letting the A/B run bit-identical to baseline undetected.
        if matched == 0 and not getattr(self, "_decorr_warned", False):
            self._decorr_warned = True
            import logging

            logging.getLogger(__name__).warning(
                "[WSPLIT/decorrelate] decorrelate_sides=True but 0/%d pairs matched the "
                "optimizer's param slices — decorrelation is a NO-OP (param identity mismatch).",
                len(self._decorr_pairs),
            )

    def _decorrelate_buf_alternating(self, buf, slices, active):
        """Alternating-mode cross-side decorrelation. Only the ACTIVE side is written this
        step (the inactive side's buf slice is zero), so the simultaneous projection is a
        no-op. Instead source the inactive side's weight-space direction from its
        MAINTAINED momentum (lie_m) and project the active written generator off it:
        "don't keep pushing along the direction the other side just moved." Modifies only
        the active side. `decorrelate_mode` selects WHICH active-side steps are treated:
        in_off_out -> in-write steps only; out_off_in -> out-write steps only; symmetric
        -> every step (full projection, no 0.5 split since only one side moves). Mutates
        buf in place."""
        if not self._decorr_pairs or active is None:
            return
        from src.diag.poet_coordination_diag import block_diag_skew, side_directions

        off_by_id = {id(p): (off, n) for off, n, p, _lr in slices}
        eps = 1e-12
        mode = self.decorrelate_mode
        matched = 0
        for out_p, in_p, w, bsz_out, bsz_in in self._decorr_pairs:
            so, si = off_by_id.get(id(out_p)), off_by_id.get(id(in_p))
            if so is None or si is None:
                continue
            matched += 1
            (oo, no), (oi, ni) = so, si
            W = w.detach().to(torch.float32)
            if active == "in":
                if mode == "out_off_in":
                    continue  # would modify the inactive (unwritten) out side -> no-op
                act_off, act_n, act_bsz, act_p = oi, ni, bsz_in, in_p
                inact_p, inact_bsz = out_p, bsz_out
            else:  # active == "out"
                if mode == "in_off_out":
                    continue
                act_off, act_n, act_bsz, act_p = oo, no, bsz_out, out_p
                inact_p, inact_bsz = in_p, bsz_in
            m_inact = self.state[inact_p].get("lie_m")
            if m_inact is None:
                continue
            # Active generator from buf; inactive direction = orthogonalize(-m) (the
            # direction the inactive side WOULD write), the same transform the optimizer
            # uses, so the projection reflects real weight-space movement.
            A_act = vec_to_skew(buf[act_off : act_off + act_n].view(act_p.shape[0], -1), act_bsz)
            A_inact = orthogonalize_skew_direction(
                vec_to_skew(-m_inact.float(), inact_bsz),
                method=self.ortho_method,
                ns_steps=self.ortho_ns_steps,
            )
            if active == "in":
                d_out, d_in = side_directions(A_inact, A_act, W)
                d_act = d_in  # the active (in) side's weight-space direction
                # <D_out, D_in>_F = <block_skew(W^T D_out), A_in>: project A_in off it.
                g = block_diag_skew(W.transpose(-2, -1) @ d_out, act_bsz)
            else:
                d_out, d_in = side_directions(A_act, A_inact, W)
                d_act = d_out  # the active (out) side's weight-space direction
                # <D_out, D_in>_F = <block_skew(D_in W^T), A_out>: project A_out off it.
                g = block_diag_skew(d_in @ W.transpose(-2, -1), act_bsz)
            # Module-selective gate: only intervene where the inter-side overlap is large
            # enough (preserve useful shared directions elsewhere).
            if self.decorrelate_cos_threshold > 0.0:
                denom = (d_out.norm() * d_in.norm()).clamp_min(eps)
                if (
                    abs(float((d_out.flatten() @ d_in.flatten()) / denom))
                    < self.decorrelate_cos_threshold
                ):
                    continue
            c = (A_act.flatten() @ g.flatten()) / (g.flatten() @ g.flatten()).clamp_min(eps)
            A_act = A_act - self.decorrelate_lambda * c * g
            if self.decorrelate_renorm:
                # Preserve the active side's realized ||D|| (direction-only change). D is
                # linear in A, so the scalar computed in weight-space applies to A directly.
                if active == "in":
                    _, d_act_new = side_directions(A_inact, A_act, W)
                else:
                    d_act_new, _ = side_directions(A_act, A_inact, W)
                A_act = A_act * (d_act.norm() / d_act_new.norm().clamp_min(eps))
            buf[act_off : act_off + act_n] = skew_to_vec(A_act, act_bsz).reshape(-1)
        if matched == 0 and not getattr(self, "_decorr_warned", False):
            self._decorr_warned = True
            import logging

            logging.getLogger(__name__).warning(
                "[decorrelate/alt] decorrelate_sides=True but 0/%d pairs matched the "
                "optimizer's slices — decorrelation is a NO-OP (param identity mismatch).",
                len(self._decorr_pairs),
            )

    def _apply_skew_update_buffer(self, buf, slices):
        """Phase (d): scatter the (already all-reduced) flat buffer back onto oft_R,
        applying each param's lr. Cast order (gen.to(dtype) then alpha=lr) matches the
        inline path exactly, so the buffer/sharded path is bit-identical to replicated."""
        for off, n, p, lr in slices:
            p.add_(buf[off : off + n].view_as(p).to(p.dtype), alpha=lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active = self._active_side()

        # --- skew branch: momentum (all ranks) -> owned-update buffer -> apply ---
        self._lie_m_update(active)
        buf, slices = self._skew_update_buffer(self._dp_rank, self._dp_world_size, active)
        if self.distributed and self._dp_world_size > 1 and buf.numel() > 0:
            import torch.distributed as dist

            dist.all_reduce(buf, group=self.dp_group)
        # Decorrelate AFTER all-reduce so both sides' generators are complete on every
        # rank (W is DP-replicated -> identical result across ranks, no extra collective).
        if self.decorrelate_sides:
            if self.alternating:
                self._decorrelate_buf_alternating(buf, slices, active)
            else:
                self._decorrelate_buf(buf, slices)
        self._apply_skew_update_buffer(buf, slices)

        # --- AdamW branch (non-skew params): unchanged, replicated ---
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

        if self.alternating:
            self._alt_step += 1
        return loss
