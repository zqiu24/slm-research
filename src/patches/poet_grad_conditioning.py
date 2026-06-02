# src/patches/poet_grad_conditioning.py
"""Patch (Probe 0B): log per-block ∂f/∂Q conditioning during a POET run.

Env-gated by SLM_POET_GRAD_CONDITIONING=1 (interval via
SLM_POET_GRAD_CONDITIONING_INTERVAL, default 2000). Inert otherwise, so it is
safe in _ALWAYS_ON_PATCHES.

Mechanism: wrap the (possibly poet-routed) ``get_megatron_optimizer`` — which
receives ``model`` — to (a) pick ~8 representative oft_R blocks and (b) wrap the
returned optimizer's ``.step`` so that, every ``interval`` steps, it reads each
block's gradient from ``main_grad`` (the Megatron DDP fp32 buffer; falls back to
``.grad``), reconstructs the skew, and logs spectral stats to W&B BEFORE the
optimizer consumes the grad.
"""

from __future__ import annotations

import logging
import os

from src.patches._registry import register_patch

# Unique label: this patch does NOT own get_megatron_optimizer (poet_optimizer_setup
# does); it composes on top of whatever that symbol currently is.
_TARGET = ("slm.diagnostics.poet_grad_conditioning.optimizer_step",)
logger = logging.getLogger(__name__)

# projection name fragments we care about (HF-ish + Megatron names)
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


def select_target_params(named_layers, max_targets: int = 8):
    """From (name, poet_layer) pairs, pick representative blocks to probe.

    Returns a list of dicts: {label, factor ('R_in'|'R_out'), param, block_size}.
    A layer contributes a target only if its name matches a wanted projection.
    """
    targets = []
    for name, layer in named_layers:
        if not any(w in name for w in _WANTED):
            continue
        for factor, attr, bsz_attr in (
            ("R_in", "oft_R_in", "block_size_in"),
            ("R_out", "oft_R_out", "block_size_out"),
        ):
            param = getattr(layer, attr, None)
            bsz = getattr(layer, bsz_attr, None)
            if param is None or bsz is None:
                continue
            targets.append(
                {
                    "label": f"{name}.{factor}",
                    "factor": factor,
                    "param": param,
                    "block_size": int(bsz),
                }
            )
            if len(targets) >= max_targets:
                return targets
    return targets


def _log_conditioning(targets, iteration: int) -> None:
    import torch

    from src.diag.skew_conditioning import block_spectral_stats, vec_to_skew

    try:
        import wandb
    except Exception:
        wandb = None

    for t in targets:
        param = t["param"]
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        if grad is None:
            logger.warning("[COND] no grad for %s at iter %d", t["label"], iteration)
            continue
        vec = grad.detach().to(torch.float32).reshape(param.shape[0], -1)
        skew = vec_to_skew(vec, t["block_size"])
        stats = block_spectral_stats(skew)
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(
                {
                    f"poet_cond/{t['label']}/condition_number": stats["condition_number"]
                    .mean()
                    .item(),
                    f"poet_cond/{t['label']}/stable_rank": stats["stable_rank"].mean().item(),
                    f"poet_cond/{t['label']}/sigma_max_over_median": stats["sigma_max_over_median"]
                    .mean()
                    .item(),
                },
                step=iteration,
            )


def _install_step_hook(optimizer, targets, interval: int) -> None:
    _orig_step = optimizer.step
    state = {"n": 0}

    def _wrapped_step(*args, **kwargs):
        if state["n"] % interval == 0:
            try:
                _log_conditioning(targets, state["n"])
            except Exception:  # — diagnostics must never break training
                logger.exception("[COND] conditioning log failed at step %d", state["n"])
        state["n"] += 1
        return _orig_step(*args, **kwargs)

    optimizer.step = _wrapped_step


@register_patch(name="poet_grad_conditioning", targets=_TARGET)
def apply() -> None:
    if os.environ.get("SLM_POET_GRAD_CONDITIONING") != "1":
        return  # inert unless explicitly enabled

    interval = int(os.environ.get("SLM_POET_GRAD_CONDITIONING_INTERVAL", "2000"))
    from megatron.training import training as _mt

    _orig_get_optimizer = _mt.get_megatron_optimizer

    def _wrapped_get_optimizer(config, model, **kwargs):
        optimizer = _orig_get_optimizer(config, model, **kwargs)
        chunks = model if isinstance(model, list) else [model]
        named_layers = [
            (name, mod)
            for m in chunks
            for name, mod in m.named_modules()
            if hasattr(mod, "oft_R_in") or hasattr(mod, "oft_R_out")
        ]
        targets = select_target_params(named_layers)
        if targets:
            _install_step_hook(optimizer, targets, interval)
            logger.warning(
                "[COND] ∂f/∂Q conditioning ENABLED — probing %d blocks every %d steps",
                len(targets),
                interval,
            )
        else:
            logger.warning("[COND] no oft_R layers found; conditioning probe is a no-op")
        return optimizer

    _mt.get_megatron_optimizer = _wrapped_get_optimizer
