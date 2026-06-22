# src/patches/poet_weight_split_log.py
"""Patch: weight-only staleness split for POET two-sided coordination.

Env-gated by ``SLM_POET_WSPLIT=1`` (interval ``SLM_POET_WSPLIT_INTERVAL``, default
250). Inert otherwise, so it is safe in ``_ALWAYS_ON_PATCHES``.

Question it answers (ANALYSIS.md §17.5 deferred probe): the alternating win is
"per-step fresh re-evaluation" (arm J) — but is that because rotating the out-side
genuinely shifts the in-side's preferred direction (inter-side coupling), or just
because the gradient field is fast-moving on its own? This logs the WEIGHT-only part
of that staleness as an angle-free sensitivity:

    s = ||block_skew(D_out^T G)||_F / ||block_skew(W^T G)||_F,

i.e. how much a unit out-side rotation moves the in-side tangent signal
K_in = block_skew(W^T G) (physical per-step shift = eff∠·s, logged as ``relchange``).
s<<1 => out barely moves in => staleness gradient-field-driven, not inter-side
coupling; s~O(1) => real weight coupling. (A bare cosine at the realized eff∠ is ~1
regardless of coupling, so we report the sensitivity, not the cosine.)

POET freezes W and never materializes the ambient G = dL/dW_eff, so we capture it
with fwd/bwd hooks: stash the layer input x, accumulate G += g_y^T x over the
backward of every micro-batch, then DP all-reduce it (so it matches main_grad, which is
DP-reduced — the local-only G is ~orthogonal to the global grad when the tangent is
near-white). NO extra backward. Self-validating: logs ``validate_cos`` =
cos(block_skew(W_perm^T G_perm), oft_R_in.grad), whose **|value| must be ~1** (it lands
near -1: block_skew(W^T G) = -block_skew(G^T W), the convention POET's backward uses).
|validate_cos| far from 1 => the capture/frame is wrong; do not trust the sensitivity.

Mechanism mirrors ``poet_coordination_log`` (wrap setup_model_and_optimizer, select
two-sided layers, wrap optimizer.step), plus per-layer capture hooks gated to the
logging steps.
"""

from __future__ import annotations

import logging
import os

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)

# Shared flag: capture x / accumulate G only on the steps we log (set by the step
# hook for the NEXT forward). dict so the closures mutate one object.
_capture = {"on": False}


def _install_capture_hook(layer):
    """Register a forward hook that stashes x and an output-grad hook that
    accumulates G = sum_microbatch g_y^T x onto ``layer._coord_G``. Gated by
    ``_capture['on']`` so it is a no-op on non-logging steps."""
    import torch

    def _fwd_hook(module, inp, out):
        # Diagnostics must never crash training: a no-grad forward (eval / init /
        # recompute) yields an output that does not require grad, and register_hook
        # raises on it -- skip those, and wrap everything defensively.
        try:
            if not _capture["on"] or not inp:
                return
            x = inp[0]
            if not torch.is_tensor(x) or not torch.is_tensor(out) or not out.requires_grad:
                return
            x = x.detach()

            def _grad_hook(grad_out):
                try:
                    gy = grad_out.detach()
                    # contract all leading (token/batch) dims -> G_eff = g_y^T x, (out, in).
                    contrib = gy.reshape(-1, gy.shape[-1]).transpose(0, 1).to(
                        torch.float32
                    ) @ x.reshape(-1, x.shape[-1]).to(torch.float32)
                    prev = getattr(module, "_coord_G", None)
                    module._coord_G = contrib if prev is None else prev + contrib
                except Exception:  # never break the backward
                    logger.exception("[WSPLIT] grad capture failed")

            out.register_hook(_grad_hook)
        except Exception:  # never break the forward
            logger.exception("[WSPLIT] forward capture failed")

    layer.register_forward_hook(_fwd_hook)


def _allreduce_coord_G(targets) -> None:  # noqa: N802
    """DP all-reduce(SUM) each layer's captured G so it matches the gradient the
    optimizer actually uses (oft_R.main_grad is DP-reduced). MUST run on every rank
    (collective) before the rank-0-only logging — otherwise the local G is just this
    rank's shard, ~orthogonal to the global grad when the tangent gradient is near-white
    (which is exactly why validate_cos read ~0). The per-layer scale is irrelevant to
    both validate_cos and the sensitivity ratio, so SUM (no /world_size) is fine."""
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu
    except Exception:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        if mpu.get_data_parallel_world_size() <= 1:
            return
        grp = mpu.get_data_parallel_group()
    except Exception:
        return
    # Same targets + same capture schedule on every rank => identical reduce set (safe).
    for t in targets:
        g = getattr(t["layer"], "_coord_G", None)
        if g is not None:
            dist.all_reduce(g, group=grp)


def _g_perm(layer):
    """Un-permute the captured forward-frame G_eff into the W_perm frame (same
    index-select as the weight), or None if not captured this step."""
    g = getattr(layer, "_coord_G", None)
    if g is None:
        return None
    return g.index_select(0, layer.perm_out_inv.long()).index_select(1, layer.perm_in_inv.long())


def _log_weight_split(targets, lookup, iteration: int) -> None:
    import torch

    from src.diag.poet_coordination_diag import (
        block_diag_skew,
        side_directions,
        weight_only_sensitivity,
    )
    from src.diag.skew_conditioning import vec_to_skew
    from src.optim.poet_skew_muon import orthogonalize_skew_direction
    from src.patches.poet_coordination_log import (
        _lie_grad_for_layer,
        _realized_angle,
        w_perm_frame,
    )

    # Collective: ALL ranks must reduce their local captured G to the global gradient
    # BEFORE the rank-0-only logging gate below (else local G ~ orthogonal to main_grad).
    _allreduce_coord_G(targets)

    try:
        import wandb
    except Exception:
        wandb = None
    if wandb is None or getattr(wandb, "run", None) is None:
        return

    def _ortho(skew):
        return orthogonalize_skew_direction(skew, method="muon", ns_steps=5)

    payload: dict = {}
    agg: dict[str, list[float]] = {}
    with torch.no_grad():
        for t in targets:
            layer = t["layer"]
            g_perm = _g_perm(layer)
            lie_grad = _lie_grad_for_layer(layer, lookup)
            if g_perm is None or lie_grad is None:
                continue
            try:
                w_perm = w_perm_frame(layer)
                b_in = int(layer.block_size_in)
                b_out = int(layer.block_size_out)
                lie_m_out, _, lie_m_in, _ = lie_grad
                a_out = _ortho(vec_to_skew(-lie_m_out.to(torch.float32), b_out))
                a_in = _ortho(vec_to_skew(-lie_m_in.to(torch.float32), b_in))
                d_out, _ = side_directions(a_out, a_in, w_perm)
                angle = _realized_angle(layer, lookup)

                # Angle-free sensitivity of the in-signal to an out rotation, and the
                # physical per-step shift (eff∠ * sensitivity).
                sens = weight_only_sensitivity(g_perm, w_perm, d_out, block_size_in=b_in).item()
                relchange = angle * sens

                # Self-check: block_skew(W_perm^T G_perm) must align with oft_R_in.grad
                # (|cos| ~ 1) if the G capture + frame-mapping are correct.
                k_in_g = block_diag_skew(w_perm.transpose(-2, -1) @ g_perm, b_in)
                k_in_opt = vec_to_skew(_in_grad(layer), b_in)
                vcos = _flat_cos(k_in_g, k_in_opt)
            except Exception:
                logger.exception("[WSPLIT] metric failed for %s", t["label"])
                continue
            payload[f"poet_wsplit/{t['label']}/sensitivity"] = sens
            payload[f"poet_wsplit/{t['label']}/relchange"] = relchange
            payload[f"poet_wsplit/{t['label']}/validate_cos"] = vcos
            agg.setdefault("sensitivity", []).append(sens)
            agg.setdefault("relchange", []).append(relchange)
            agg.setdefault("validate_cos", []).append(vcos)

    if not payload:
        return
    for k, vals in agg.items():
        payload[f"poet_wsplit/_mean/{k}"] = sum(vals) / len(vals)
    wandb.log(payload, step=iteration)


def _in_grad(layer):
    """Fresh in-side skew-tangent gradient (vec form) from the live oft_R_in."""
    p = layer.oft_R_in
    g = getattr(p, "main_grad", None)
    if g is None:
        g = p.grad
    import torch

    return g.detach().to(torch.float32).reshape(p.shape[0], -1)


def _flat_cos(a, b, eps: float = 1e-12):
    import torch

    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return (torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(eps)).item()


def _clear(targets) -> None:
    for t in targets:
        if hasattr(t["layer"], "_coord_G"):
            t["layer"]._coord_G = None


def _install_step_hook(optimizer, targets, lookup, interval: int) -> None:
    _orig_step = optimizer.step
    state = {"n": 0}
    _capture["on"] = True  # capture the very first (n=0) logging step

    def _wrapped_step(*args, **kwargs):
        if state["n"] % interval == 0:
            try:
                _log_weight_split(targets, lookup, state["n"])
            except Exception:  # diagnostics must never break training
                logger.exception("[WSPLIT] log failed at step %d", state["n"])
        _clear(targets)
        state["n"] += 1
        _capture["on"] = state["n"] % interval == 0  # arm capture for the next step
        return _orig_step(*args, **kwargs)

    optimizer.step = _wrapped_step


def _install_on_setup(orig_setup, interval):
    def _wrapped_setup(*args, **kwargs):
        from src.patches.poet_coordination_log import (
            _build_lie_state_lookup,
            select_target_layers,
        )

        model, optimizer, opt_param_scheduler = orig_setup(*args, **kwargs)
        chunks = model if isinstance(model, list) else [model]
        named_layers = [
            (name, mod)
            for m in chunks
            for name, mod in m.named_modules()
            if hasattr(mod, "oft_R_in") and hasattr(mod, "oft_R_out")
        ]
        targets = select_target_layers(named_layers)
        if targets:
            for t in targets:
                _install_capture_hook(t["layer"])
            lookup = _build_lie_state_lookup(optimizer)
            _install_step_hook(optimizer, targets, lookup, interval)
            logger.warning(
                "[WSPLIT] weight-only staleness split ENABLED — %d layers every %d steps",
                len(targets),
                interval,
            )
        else:
            logger.warning("[WSPLIT] no two-sided oft_R layers found; weight split is a no-op")
        return model, optimizer, opt_param_scheduler

    return _wrapped_setup


@register_patch(name="poet_weight_split_log", targets=())
def apply() -> None:
    if os.environ.get("SLM_POET_WSPLIT") != "1":
        return  # inert unless explicitly enabled

    interval = int(os.environ.get("SLM_POET_WSPLIT_INTERVAL", "250"))
    from megatron.training import training as _mt

    _mt.setup_model_and_optimizer = _install_on_setup(_mt.setup_model_and_optimizer, interval)
