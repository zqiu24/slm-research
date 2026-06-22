# src/patches/poet_coordination_log.py
"""Patch (Tier-0 POET two-sided coordination diagnostics): log, during a POET
lie_ortho run, the metrics that arbitrate *why* alternating + fresh-momentum
beats simultaneous.

Env-gated by ``SLM_POET_COORD_DIAG=1`` (interval via
``SLM_POET_COORD_DIAG_INTERVAL``, default 250). Inert otherwise, so it is safe in
``_ALWAYS_ON_PATCHES``.

Per sampled two-sided POET layer it logs (see src/diag/poet_coordination_diag.py):

  * ``mom_cos_out`` / ``mom_cos_in`` — cos(side momentum lie_m, fresh skew-tangent
    gradient). The STALENESS arbiter: high on the champion (both momenta fed every
    step); collapses on the reactivated side under a frozen-momentum
    (true_single_side) run, which is the cheapest read of established fact #5.
  * ``cos_D_out_D_in`` / ``r_joint`` / ``gram_cond`` — overlap geometry of the two
    sides' weight-space directions D_out = A_out·W, D_in = W·A_in (A = orthogonalize
    (-lie_m), W in the W_perm frame). The GAUGE-REDUNDANCY arbiter: large |cos| /
    cond means a matched-||dW|| simultaneous step over-spends the redundant
    direction; cos ~ 0 falsifies redundancy.
  * ``norm_D_out`` / ``norm_D_in`` — relative per-side movement magnitude.

Mechanism mirrors ``poet_grad_conditioning``: wrap ``setup_model_and_optimizer``
(composes with whatever routed the optimizer), select representative two-sided
layers, and wrap ``optimizer.step`` so that, every ``interval`` steps, it reads
each side's PRE-step momentum (optimizer state) + FRESH gradient (``main_grad``,
falling back to ``.grad``) and logs to W&B before the optimizer consumes them.
Both momenta are read regardless of which side is written, so the metric is the
same for the alternating, simultaneous, and frozen arms — an apples-to-apples
read across the 3-arm comparison.
"""

from __future__ import annotations

import logging
import os

from src.patches._registry import register_patch

logger = logging.getLogger(__name__)

# projection name fragments we care about (HF-ish + Megatron names), same set the
# conditioning probe uses so the two diagnostics sample comparable sites.
_WANTED = (
    "linear_q",
    "q_proj",
    "linear_v",
    "v_proj",
    "linear_fc2",
    "down_proj",
    "linear_fc1",
    "up_proj",
)


def select_target_layers(named_layers, max_targets: int = 8):
    """From (name, module) pairs, pick representative TWO-SIDED POET layers.

    A layer qualifies only if its name matches a wanted projection AND it carries
    both ``oft_R_in`` and ``oft_R_out`` (the overlap geometry needs both sides).
    Returns a list of {label, layer}.
    """
    targets = []
    for name, layer in named_layers:
        if not any(w in name for w in _WANTED):
            continue
        if getattr(layer, "oft_R_in", None) is None or getattr(layer, "oft_R_out", None) is None:
            continue
        targets.append({"label": name, "layer": layer})
        if len(targets) >= max_targets:
            return targets
    return targets


def w_perm_frame(layer):
    """The block-contiguous weight POET's generators are block-diagonal in (fp32,
    detached). This is ``layer.weight`` UNCHANGED: the forward permutes the
    activations (x by perm_in_inv, y by perm_out), never the weight — so
    ``oft_R_in``/``oft_R_out`` rotate contiguous column/row blocks of ``layer.weight``
    directly (see chain_layer_x_fast_decoupled and single_step.py's M_in = A @ weight).
    An earlier version re-permuted by perm_out_inv/perm_in_inv on the false premise
    that ``layer.weight`` was a "forward frame"; that scrambled the block alignment and
    drove the weight-split ``validate_cos`` to ~0 (misdiagnosed as a DP local-vs-global
    issue). Returning the raw weight makes validate_cos land at ~-1 as derived."""
    import torch

    return layer.weight.detach().to(torch.float32)


def collect_metrics_for_layer(layer, lie_grad, orthogonalize_fn, realized_angle: float = 1.0):
    """Bridge one layer's (lie_m_out, grad_out, lie_m_in, grad_in) snapshot + its
    W_perm weight into the pure Tier-0 metric assembler. Returns a dict of floats.

    realized_angle (= eff∠ = skew group_lr * ortho_c) scales r_cross to the physical
    cross-term fraction; 1.0 leaves it as the unit-generator geometric ratio."""
    from src.diag.poet_coordination_diag import layer_coordination_metrics

    lie_m_out, grad_out, lie_m_in, grad_in = lie_grad
    return layer_coordination_metrics(
        lie_m_out,
        grad_out,
        lie_m_in,
        grad_in,
        w_perm_frame(layer),
        block_size_out=int(layer.block_size_out),
        block_size_in=int(layer.block_size_in),
        orthogonalize_fn=orthogonalize_fn,
        realized_angle=realized_angle,
    )


# --------------------------------------------------------------------------
# Megatron glue (optimizer state lookup, step hook, install). Mirrors the proven
# poet_grad_conditioning wiring; exercised on real runs.
# --------------------------------------------------------------------------


def _build_lie_state_lookup(optimizer):
    """Map id(model oft_R param) -> (inner torch optimizer, fp32 master param).

    Reuses poet_merge_step._iter_model_master_pairs so both the plain Float16 and
    the FP32 (master==model) layouts are covered. The lie_ortho optimizer holds the
    oft_R params, so its state[master]['lie_m'] is reachable through this map.
    """
    from src.patches.poet_merge_step import _iter_model_master_pairs

    inner = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    lookup = {}
    for opt in inner:
        torch_opt = getattr(opt, "optimizer", None)
        if torch_opt is None:
            continue
        for model_p, master_p in _iter_model_master_pairs(opt):
            lookup[id(model_p)] = (torch_opt, master_p)
    return lookup


def _side_snapshot(param, lookup):
    """(lie_m, fresh_grad) for one oft_R side, or None if unavailable."""
    import torch

    torch_opt, master = lookup.get(id(param), (None, None))
    if torch_opt is None:
        return None
    st = torch_opt.state.get(master, {})
    lie_m = st.get("lie_m")
    if lie_m is None:
        return None
    grad = getattr(param, "main_grad", None)
    if grad is None:
        grad = param.grad
    if grad is None:
        return None
    lie_m = lie_m.detach().to(torch.float32).reshape(param.shape[0], -1)
    grad = grad.detach().to(torch.float32).reshape(param.shape[0], -1)
    return lie_m, grad


def _lie_grad_for_layer(layer, lookup):
    """(lie_m_out, grad_out, lie_m_in, grad_in) for a layer, or None if either
    side's momentum/grad is missing (e.g. very first step, or a non-lie optimizer)."""
    out = _side_snapshot(layer.oft_R_out, lookup)
    inn = _side_snapshot(layer.oft_R_in, lookup)
    if out is None or inn is None:
        return None
    lie_m_out, grad_out = out
    lie_m_in, grad_in = inn
    return lie_m_out, grad_out, lie_m_in, grad_in


def _realized_angle(layer, lookup) -> float:
    """eff∠ = (skew group lr) * ortho_c for the layer's lie_ortho optimizer, used to
    scale r_cross to the physical cross-term fraction. Falls back to 1.0 (unit-angle
    geometric ratio) if the optimizer / lr / ortho_c is unavailable."""
    torch_opt, _ = lookup.get(id(getattr(layer, "oft_R_out", None)), (None, None))
    if torch_opt is None:
        return 1.0
    ortho_c = getattr(torch_opt, "ortho_c", None)
    if ortho_c is None:
        return 1.0
    for g in getattr(torch_opt, "param_groups", []):
        if g.get("use_skew") and g.get("lr") is not None:
            return float(g["lr"]) * float(ortho_c)
    return 1.0


def _log_coordination(targets, lookup, iteration: int) -> None:
    import torch

    from src.optim.poet_skew_muon import orthogonalize_skew_direction

    try:
        import wandb
    except Exception:
        wandb = None

    if wandb is None or getattr(wandb, "run", None) is None:
        return

    def _ortho(skew):
        return orthogonalize_skew_direction(skew, method="muon", ns_steps=5)

    payload = {}
    agg: dict[str, list[float]] = {}
    with torch.no_grad():
        for t in targets:
            lie_grad = _lie_grad_for_layer(t["layer"], lookup)
            if lie_grad is None:
                continue
            try:
                angle = _realized_angle(t["layer"], lookup)
                m = collect_metrics_for_layer(t["layer"], lie_grad, _ortho, realized_angle=angle)
            except Exception:  # diagnostics must never break training
                logger.exception("[COORD] metric failed for %s", t["label"])
                continue
            for k, v in m.items():
                payload[f"poet_coord/{t['label']}/{k}"] = v
                agg.setdefault(k, []).append(v)

    if not payload:
        return
    for k, vals in agg.items():
        payload[f"poet_coord/_mean/{k}"] = sum(vals) / len(vals)
    wandb.log(payload, step=iteration)


def _install_step_hook(optimizer, targets, lookup, interval: int) -> None:
    _orig_step = optimizer.step
    state = {"n": 0}

    def _wrapped_step(*args, **kwargs):
        # PRE-step: lie_m is this step's accumulated momentum (before its EMA update)
        # and main_grad is the fresh gradient — exactly the staleness snapshot we want.
        if state["n"] % interval == 0:
            try:
                _log_coordination(targets, lookup, state["n"])
            except Exception:  # diagnostics must never break training
                logger.exception("[COORD] coordination log failed at step %d", state["n"])
        state["n"] += 1
        return _orig_step(*args, **kwargs)

    optimizer.step = _wrapped_step


def _install_coordination_on_setup(orig_setup, interval):
    """Wrap ``setup_model_and_optimizer`` to install the coordination step-hook on
    the fully-built optimizer. Factored out so the wrap logic is unit-testable
    without a real Megatron import."""

    def _wrapped_setup(*args, **kwargs):
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
            lookup = _build_lie_state_lookup(optimizer)
            _install_step_hook(optimizer, targets, lookup, interval)
            logger.warning(
                "[COORD] two-sided coordination diag ENABLED — %d layers every %d steps",
                len(targets),
                interval,
            )
        else:
            logger.warning("[COORD] no two-sided oft_R layers found; coordination diag is a no-op")
        return model, optimizer, opt_param_scheduler

    return _wrapped_setup


@register_patch(name="poet_coordination_log", targets=())
def apply() -> None:
    if os.environ.get("SLM_POET_COORD_DIAG") != "1":
        return  # inert unless explicitly enabled

    interval = int(os.environ.get("SLM_POET_COORD_DIAG_INTERVAL", "250"))
    from megatron.training import training as _mt

    _mt.setup_model_and_optimizer = _install_coordination_on_setup(
        _mt.setup_model_and_optimizer, interval
    )
