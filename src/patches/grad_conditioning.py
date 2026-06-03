# src/patches/grad_conditioning.py
"""Patch: log per-layer weight-gradient conditioning during ANY run.

Optimizer-agnostic counterpart to ``poet_grad_conditioning`` (which probes
POET's ``oft_R`` skew gradients). This one probes the raw 2D weight gradients of
ordinary ``nn.Linear`` layers, so it works on the plain AdamW and Muon baselines
(``scripts/train_adam_dev.sh`` / ``scripts/train_muon_dev.sh``) — no POET in the
picture. The canonical "why Muon" diagnostic: Adam's Linear-weight gradients tend
to be low-(stable/effective)-rank / heavy-tailed; Muon orthogonalizes them.

Env-gated by SLM_GRAD_CONDITIONING=1. Interval via SLM_GRAD_CONDITIONING_INTERVAL,
which falls back to the POET probe's SLM_POET_GRAD_CONDITIONING_INTERVAL (then
2000) so both diagnostics sample at the same cadence by default. Inert otherwise,
so it is safe in _ALWAYS_ON_PATCHES.

Mechanism mirrors poet_grad_conditioning: wrap ``setup_model_and_optimizer``
(composes with the other wrappers of that symbol), pick ~8 representative Linear
weights from ``model``, and wrap the optimizer's ``.step`` so that, every
``interval`` steps, it reads each weight's full accumulated gradient from
``main_grad`` (the Megatron DDP fp32 buffer; falls back to ``.grad``) BEFORE the
optimizer consumes it, runs ``block_spectral_stats``, and logs to W&B.
"""

from __future__ import annotations

import logging
import os

from src.patches._registry import register_patch

# Runtime wrapper with no static target ownership: it monkeypatches
# setup_model_and_optimizer at apply() time and composes with any other wrapper
# of that symbol, so it registers targets=() to avoid a PatchConflict.
logger = logging.getLogger(__name__)

# Linear projection name fragments to probe (HF-ish + Megatron names).
_WANTED = (
    "linear_q",
    "q_proj",
    "linear_k",
    "k_proj",
    "linear_v",
    "v_proj",
    "linear_proj",
    "o_proj",
    "linear_fc1",
    "linear_fc2",
    "up_proj",
    "down_proj",
    "gate_proj",
)


def select_linear_grad_targets(named_modules, max_targets: int = 8):
    """From (name, module) pairs, pick representative Linear weights to probe.

    A module contributes a target only if its name matches a wanted projection
    AND it exposes a 2D ``.weight``. Returns a list of {label, param} dicts.
    """
    targets = []
    for name, mod in named_modules:
        if not any(w in name for w in _WANTED):
            continue
        weight = getattr(mod, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            continue
        targets.append({"label": name, "param": weight})
        if len(targets) >= max_targets:
            break
    return targets


def _log_grad_conditioning(targets, iteration: int) -> None:
    import torch

    from src.diag.skew_conditioning import block_spectral_stats

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
            logger.warning("[GRADCOND] no grad for %s at iter %d", t["label"], iteration)
            continue
        mat = grad.detach().to(torch.float32)
        if mat.dim() != 2:
            mat = mat.reshape(mat.shape[0], -1)
        stats = block_spectral_stats(mat)  # 2D -> auto-unsqueezed to (1, out, in)
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(
                {
                    f"grad_cond/{t['label']}/condition_number": stats["condition_number"][0].item(),
                    f"grad_cond/{t['label']}/stable_rank": stats["stable_rank"][0].item(),
                    f"grad_cond/{t['label']}/sigma_max_over_median": stats["sigma_max_over_median"][
                        0
                    ].item(),
                    f"grad_cond/{t['label']}/effective_rank": stats["effective_rank"][0].item(),
                },
                step=iteration,
            )


def _install_step_hook(optimizer, targets, interval: int) -> None:
    _orig_step = optimizer.step
    state = {"n": 0}

    def _wrapped_step(*args, **kwargs):
        if state["n"] % interval == 0:
            try:
                _log_grad_conditioning(targets, state["n"])
            except Exception:  # diagnostics must never break training
                logger.exception("[GRADCOND] conditioning log failed at step %d", state["n"])
        state["n"] += 1
        return _orig_step(*args, **kwargs)

    optimizer.step = _wrapped_step


def _install_grad_conditioning_on_setup(orig_setup, interval):
    """Wrap ``setup_model_and_optimizer`` to install the conditioning step-hook on
    the fully-built optimizer. Factored out so the wrap logic is unit-testable
    without a real Megatron import."""

    def _wrapped_setup(*args, **kwargs):
        model, optimizer, opt_param_scheduler = orig_setup(*args, **kwargs)
        chunks = model if isinstance(model, list) else [model]
        named_modules = [(name, mod) for m in chunks for name, mod in m.named_modules()]
        targets = select_linear_grad_targets(named_modules)
        if targets:
            _install_step_hook(optimizer, targets, interval)
            logger.warning(
                "[GRADCOND] weight-grad conditioning ENABLED — probing %d layers every %d steps",
                len(targets),
                interval,
            )
        else:
            logger.warning(
                "[GRADCOND] no matching linear layers found; grad conditioning is a no-op"
            )
        return model, optimizer, opt_param_scheduler

    return _wrapped_setup


def _resolve_interval(env) -> int:
    """Logging interval, kept consistent with the POET conditioning probe: falls
    back to ``SLM_POET_GRAD_CONDITIONING_INTERVAL`` (then 2000) so both diagnostics
    sample at the same cadence by default; an explicit
    ``SLM_GRAD_CONDITIONING_INTERVAL`` still overrides."""
    val = env.get("SLM_GRAD_CONDITIONING_INTERVAL")
    if val is None:
        val = env.get("SLM_POET_GRAD_CONDITIONING_INTERVAL", "2000")
    return int(val)


@register_patch(name="grad_conditioning", targets=())
def apply() -> None:
    if os.environ.get("SLM_GRAD_CONDITIONING") != "1":
        return  # inert unless explicitly enabled

    interval = _resolve_interval(os.environ)
    from megatron.training import training as _mt

    _mt.setup_model_and_optimizer = _install_grad_conditioning_on_setup(
        _mt.setup_model_and_optimizer, interval
    )
